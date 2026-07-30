# PERF.md — tracing, spans, GPU timestamps, and the benchmark contract

Owner: Niobe (performance engineering).
Status: the tracing module and the benchmark harness are implemented. **Every GPU-side number
described here is currently unobtainable**, because no shader in this repository has ever
executed on any device (`DESIGN.md` §9.1.2). This document specifies the instrument. The
readings come later.

---

## 1. What a span means for a Vulkan EP

We integrate [`onnx-runtime-tracer`](https://crates.io/crates/onnx-runtime-tracer) — the same
crate the sibling project `onnxruntime-mlx` uses (`rust/src/trace.rs` there) — so that a trace
from this EP and a trace from the host can be loaded into the same viewer and overlaid.

### 1.1 The pin, and why it is deliberate

```toml
onnx-runtime-tracer = { version = "0.1.0-dev.5", default-features = false }
```

Two properties are load-bearing:

* **`default-features = false`** — Chrome Trace JSON only, no `prost`/protobuf. A plugin dylib
  that drags a protobuf runtime into the host process is a liability, and the host already
  speaks Chrome Trace.
* **The pin is to a release with an absolute clock.** `TraceClock` reports absolute
  UNIX-microsecond-domain timestamps derived from the OS clock (`clock_gettime(CLOCK_MONOTONIC)`
  / `QueryPerformanceCounter`), so the origin belongs to the *machine*, not to the library
  instance. This matters here more than anywhere: we are a plugin `cdylib`. We have our own
  copy of every static. There is no process-global epoch we could ever share with the host's own
  tracer. Because both sides independently resolve to the same absolute instant, our spans drop
  onto the host's timeline with **no offset negotiation** and no clock-sync handshake.

  A release that changed the epoch to "first call into the library" would break overlay
  *silently* — the traces would still load, still look plausible, and be wrong by an unknowable
  constant. Do not move this pin without re-verifying that property.

### 1.2 Where MLX and Vulkan diverge

MLX has a lazy graph and unified memory. A span there naturally wraps "build the lazy graph"
and "`mlx_eval` it", and because evaluation is synchronous from the caller's point of view, host
wall time around `eval` is a defensible proxy for compute time. There are no explicit transfers
to instrument, because there is nothing to transfer.

None of that holds here. We have explicit command buffers, an explicit staging path across a
PCIe boundary, and a queue that runs asynchronously with respect to the thread that filled it.
So our span vocabulary is different, and the difference is the whole point:

| Phase | What it wraps | What it actually measures |
|---|---|---|
| `compile` | Shader module creation, pipeline layout + compute pipeline creation, descriptor layout. | Real host CPU work. Once per subgraph, amortised over the session. |
| `prepack` | Weight repacking/quantisation-layout fixups on the host. | Real host CPU work. Once. |
| `upload` | Staging-buffer write + transfer of weights and inputs to device-local memory. | Host time *plus* a real transfer. Counters carry bytes and effective GiB/s. |
| `record` | Filling the command buffer: binds, push constants, barriers, dispatches. | Real host CPU work. Per `ENGINE.md` §6.1 this is record-once/replay-many, so a steady-state inference should show *no* `record` span at all — if it does, something is invalidating the recording. |
| `submit` | `vkQueueSubmit` itself. | **Almost nothing.** See §1.3. |
| `fence_wait` | Waiting for the submission's fence. | An **upper bound** on GPU execution, inflated by queue contention, other clients' work, and driver scheduling. Not kernel time. |
| `readback` | Device→host transfer of outputs. | Host time plus a real transfer. Bytes + GiB/s counters. |

Two further span-adjacent facts we record because they are the ones that mislead people:

* **`RecordPath`** — `first_record` / `replay` / `rerecord`. This is our analogue of MLX's cache
  HIT/MISS/RETRACE. It answers "did this inference reuse the command buffer, or did we rebuild
  it?" A shape key never seen before for a given subgraph is classified `rerecord`, not
  `replay`, even if a recording existed — resolving against a *set* of seen keys rather than a
  single last-key, which is the lesson the MLX tracer learned the hard way about alternating
  shapes.
* **`PartitionStats`** — `island_count`, `largest_island_nodes`, `largest_island_flops`,
  `concentration`, `boundary_bytes_per_inference`, `boundary_time_fraction`, and the `declined`
  histogram keyed by `deny!` reason. `largest_island_flops` is the metric of record
  (`OP_COVERAGE.md` §7.3); `claimed_node_fraction` is a diagnostic, not a target. The summary
  prints a warning when `boundary_time_fraction` exceeds 0.20, because at that point we are
  measuring PCIe, not shaders.

### 1.3 Host wall time around a submit measures almost nothing

`vkQueueSubmit` hands a command buffer to the driver and returns. It does not wait for the GPU.
It usually does not even wait for the GPU to *start*. A timer around it measures driver
bookkeeping and validation-layer overhead — hundreds of nanoseconds to a few microseconds —
regardless of whether the command buffer contains one `Add` or ten minutes of matrix multiply.

This is stated in the code, on `Phase::Submit::caveat()`, where someone reaching for the obvious
wrong measurement will read it:

> `submit` is the host cost of handing the command buffer to the driver. It is NOT GPU time.

`Phase::Submit` is the only phase for which `observes_gpu_work()` returns `false`, and there is a
unit test asserting exactly that, so the property cannot be quietly lost in a refactor.

`fence_wait` is the honest *host-observable* bound, and it is labelled `UPPER BOUND` for the
same reason: it includes everything between submit and signal, most of which may not be ours.

**Kernel time comes from the device or it does not come at all.** That is §3.

### 1.4 Measured device facts (this machine, 2026-07-29)

Read with `python bench/devices.py` (source: `vulkaninfo` from the Vulkan SDK — deliberately not
our own EP, because these are the facts we would use to check the EP's arithmetic and a
self-report cannot check itself).

| | Intel Iris Xe | NVIDIA RTX 4060 Laptop |
|---|---|---|
| device type | integrated | discrete |
| driver | 101.6737 | 591.55 |
| **`timestampPeriod`** | **52.0833 ns/tick** | **1.0 ns/tick** |
| **`timestampValidBits`** (compute qf 0) | **36** | **64** |
| tick wrap period | ~3 579 s (≈1 hour) | ~5.8×10⁵ years |
| `maxComputeSharedMemorySize` | 32 KiB | 48 KiB |
| subgroup size | 32 | 32 |
| transfer class | **UMA** (one DEVICE_LOCAL heap) | **discrete** (8.3 GB device heap + 34 GB host heap) |
| device-local host-visible window | (UMA — all of it) | yes, a BAR window — **not** unified memory |
| `VK_EXT_calibrated_timestamps` | yes | yes |
| `VK_EXT_host_query_reset` | yes | yes |

Four things follow, and all four are traps that produce plausible numbers:

1. **A hardcoded `timestampPeriod = 1.0` under-reports Iris Xe GPU time by 52×.** Not by a
   suspicious amount — by an amount that reads as a triumph.
2. **36 valid bits is not a number anyone would guess**, and the Xe's counter wraps about once
   an hour, so wrap recovery is a routine path rather than a theoretical one. The 4060's 64 bits
   is simultaneously the "never shift by 64" case. Both are exercised by `rust/src/trace.rs`'s
   unit tests.
3. **"Some memory type is DEVICE_LOCAL|HOST_VISIBLE" does not mean unified memory.** The
   discrete 4060 exposes exactly that (the resizable-BAR window). The operational definition
   used in `bench/devices.py` is instead *no heap lacks DEVICE_LOCAL*, i.e. "is there separate
   system memory the GPU must copy from?" — which classifies these two devices correctly.
   Conflating them would justify skipping the staging copy we specifically need to measure.
4. **A tile configuration tuned for 48 KiB is not selectable on the Xe.** So a speedup that does
   not name its tile config is comparing two different kernels. `bench/compare.py` refuses rows
   whose recorded `tile_config` differs, and treats two `None`s as *unknown*, never as *equal*.

Both devices expose `VK_EXT_calibrated_timestamps`, so §3.7's preferred host↔device anchoring
path — with a real `maxDeviation` error bar — is available here today; the bracketing fallback
is for other hardware, not for this machine.

**Also needed from Switch (`engine.rs`):** `DeviceInfo` currently carries name, vendor/device ID,
API version, driver version and kind. For the harness to stop depending on an external SDK tool,
it needs `timestamp_period_ns`, `timestamp_valid_bits` (for the compute queue family we bind),
`max_compute_shared_memory`, `subgroup_size`, and whether every memory heap is DEVICE_LOCAL
(the UMA question). Surfacing them through `epctl --probe-loader --json` would let CI record
device facts without a Vulkan SDK install.

---

## 2. Environment contract

Following `rust/src/logging.rs` and the `ONNXRUNTIME_EP_VULKAN_*` convention:

| Variable | Effect |
|---|---|
| `ONNXRUNTIME_EP_VULKAN_TRACE=<path>` | Enables tracing; writes Chrome Trace JSON to `<path>` at session teardown. Implies `Debug` log level (existing behaviour in `logging.rs`). |
| `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` | Additionally enables GPU timestamp queries. Opt-in and separate, because timestamp writes perturb the command buffer and can split a render/compute pass on tile-based mobile GPUs. |
| `ONNXRUNTIME_EP_VULKAN_VERBOSE=1` | Existing verbosity flag; makes the tracer log its summary and slowest-op tables to the ORT logger as well as to the trace file. |
| `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG=<path>` | Existing (`ops/claim_log.rs`) JSON-Lines claim record. The benchmark harness reads this rather than scraping stderr. |

**Zero cost when disabled.** Unset `ONNXRUNTIME_EP_VULKAN_TRACE` and the tracer resolves once,
via `OnceLock`, to a disabled singleton wrapping `TraceContext::noop()`. Every recording entry
point takes the `is_enabled()` early-out; no allocation, no formatting, no timestamp reads, no
query pools created, no command-buffer instrumentation. The check is a relaxed atomic load
behind an already-resolved `OnceLock` — a predictable branch on a path that is already doing
descriptor writes.

### 2.1 The producer is part of the environment

Mouse established for correctness (`OP_COVERAGE.md` §4.18) that **op coverage is relative to a
producer, not to a model architecture**. Justin's own `onnx-genai-models` builder (`mobius`)
emits, per Qwen3 decoder layer, `ai.onnx::Attention` @ opset 23, `ai.onnx::RMSNormalization` and
`ai.onnx::RotaryEmbedding`. The ORT GenAI builder emits the `com.microsoft` contrib equivalents —
`GroupQueryAttention`, `SimplifiedLayerNormalization`, contrib-domain `RotaryEmbedding`, with
`seqlens_k` indirection and in-place KV-cache aliasing. **`MatMulNBits` is the only op both
toolchains agree on.**

The same is true of a benchmark artefact, and it is easier to be fooled by, because a timing has
no shape to disagree about. If `bench/cases.py` builds a graph with one exporter and the table
says "Qwen3", the number describes that exporter's graph — its op set, its fusions, its cache
layout — not the architecture. Two further consequences specific to us:

* The two graphs **partition differently**. "The EP claimed 40% of the graph" is as much a
  statement about the producer as about the EP; the standard-domain graph may be largely
  claimable while the contrib graph is not, or vice versa.
* The `largest_island_flops` metric (`OP_COVERAGE.md` §5) is computed on a specific graph. An
  island size quoted without its producer is not reproducible.

So the producer is recorded next to device, driver, OS and build flags:

| Field | Where |
|---|---|
| `producers[]` — name, kind, version, digest, opsets, model family | result JSON top level and `environment.producers` |
| `producer_fingerprint` — `name@version#digest` | every case row |

`digest` is a SHA-256 of the builder's own source (for the in-repo op builder) or of its versioned
identity (for an external exporter). It is there for the same reason the driver version is in the
device fingerprint: an edit to the builder changes the graph, and a graph change attributed to the
EP would be wrong in a way that looks entirely reasonable.

**Two structural refusals follow** (see §4.0): a case cannot be *named* after a model family its
producer did not export, and two results built by different producers cannot be compared as a
regression. When a real model case does arrive, it must be added **per producer** —
`qwen3_decoder_mobius` and `qwen3_decoder_ortgenai` are two cases, not one case run two ways.
Mouse notes the `mobius` path avoids `seqlens_k` indirection and KV-cache aliasing entirely, so it
is the likelier of the two to give us a model we can build and iterate on **on this machine**;
that is a reason to represent both, not a reason to assume one.

### 2.2 Portability envelope — what a number has to fit inside to be about the EP

Justin's standing directive: **要时刻注意跨平台通用性** — cross-platform generality, at all times.
A Vulkan EP that is really a desktop-NVIDIA EP has no reason to exist; better-supported vendor
backends already do that job.

The measurement version of that directive is narrow and checkable. **A configuration measured on
this desk is only a statement about the EP if a device the EP admits could select it.** The
admission floor is already decided, in `DESIGN.md` §7.2:

| Floor | Value | Source |
|---|---|---|
| shared memory | **16384 B** (16 KiB) | §7.2 R4 |
| workgroup invocations | **256** | §7.2 R3 |
| subgroup size | **no guarantee at all** — Vulkan 1.1 guarantees `BASIC` in compute and says nothing about the size | §7.2 R5 |

§7.0's rule is that capability shortfalls degrade **op coverage**, not **device availability** — so
a device sitting exactly on that floor is a device we *promised to run on*, and it has 16 KiB, not
32 and not 48.

What this rules out that is easy to get wrong:

* **The Iris Xe is not a mobile proxy for shared memory.** It is our closest available proxy for
  the mobile *memory model* — it is UMA, as Adreno and Mali are — but it has 32 KiB, twice the
  floor. A 32 KiB tile passing on it is not portability evidence.
* **Both local GPUs report `subgroupSize == 32`.** That is precisely the coincidence most likely
  to bake a 32 into a kernel and have it pass every local test. A configuration that depends on a
  subgroup size must record which; the harness never assumes one.
* **A UMA shortcut is not a portability win.** Skipping the staging path is correct on the Iris Xe
  and on Adreno/Mali, and wrong on the 4060 except through the resizable-BAR window, which is not
  unified memory (§1.4). Both paths must exist.

`bench/portability.py` turns this into a verdict on a configuration — `portable`,
`needs-fallback`, or `unknown` — with `quotable_as_ep_behaviour` true only for `portable`.
`needs-fallback` is *not* a failure: measuring a 48 KiB tile on the 4060 is legitimate and useful.
It means the number describes a path a floor device cannot take, so the floor-compliant fallback
must be measured before the number is quoted as the EP's behaviour. **`unknown` is not quotable
either** — a configuration nobody recorded is a configuration nobody can reproduce.

Every result row carries a `portability` verdict and every result file carries `portability_floor`
with the selected device's headroom. `bench.py` prints that headroom on every run (the 4060 is
3.0× the floor, the Iris Xe 2.0×), and `compare.py` banners any table whose rows came from
non-floor or unrecorded configurations. Today **every row is `unknown`**, because the engine does
not report tile shape or workgroup size yet — that is the honest state, and it is printed rather
than assumed away.

**Transfer models are fitted per transfer class and may not be blended.**
`portability.transfer_model_merge_refusal()` refuses to combine a UMA fit with a discrete one: on
a UMA part an upload may be a mapped write with no copy at all, on a discrete part it is a staging
buffer and a bus transfer, and a single affine model fitted across both would land plausibly
*between* the two. Plausible and wrong is the failure mode this document exists to prevent.

**Routed to Switch, since §7.2 is enforced in code he owns:** the harness can only record a
configuration the engine reports. `tile_config`, the workgroup size actually dispatched, the
shared-memory bytes actually requested, and whether the kernel took the UMA or the staging path
would each turn a row from `unknown` into a judgeable verdict. Until then the portability column
is honest but empty.
---

## 3. Requirement: Vulkan timestamp queries (routed to Switch)

`rust/src/vk/**` and `engine.rs` are Switch's. `rust/src/trace.rs` deliberately names no Vulkan
type — it does not import `ash` (see the `layering.rs` lint) — and instead exposes a plain-data
ingest seam. This section specifies exactly what must be produced and handed to it.

### 3.1 The seam

`rust/src/trace.rs` exposes:

```rust
pub struct GpuTimestampCalibration {
    pub timestamp_period_ns: f32,     // VkPhysicalDeviceLimits::timestampPeriod
    pub valid_bits: u32,              // VkQueueFamilyProperties::timestampValidBits
    pub host_anchor_us: u64,          // host clock, tracer's absolute domain
    pub device_anchor_ticks: u64,     // device clock at (approximately) the same instant
    pub anchor_uncertainty_us: u64,   // how wrong the pairing above may be
}

pub struct GpuInterval {
    pub label: &'static str,
    pub begin_ticks: u64,
    pub end_ticks: u64,
    pub node_index: Option<usize>,
    pub flops: Option<u64>,
    pub bytes: Option<u64>,
}

pub struct GpuTimestampReport {
    pub calibration: GpuTimestampCalibration,
    pub queue_family: u32,
    pub intervals: Vec<GpuInterval>,
}

// call after the fence signals:
tracer().record_gpu_intervals(&report);
```

Raw ticks, unconverted. `trace.rs` owns the arithmetic, the wrap recovery, the axis placement,
and the tests for all three. Please do not pre-convert to nanoseconds in `vk/` — that would put
the interesting failure modes in the module that has no tests for them.

### 3.2 Query pool

* One `VkQueryPool` with `queryType = VK_QUERY_TYPE_TIMESTAMP`, `queryCount = 2 * dispatch_count`
  (+2 if you also want a whole-command-buffer pair), created per recorded command buffer
  alongside it in `CommandPool`. Created **only** when `ONNXRUNTIME_EP_VULKAN_TRACE_GPU` is set;
  no pool, no writes, no cost otherwise.
* `vkCmdResetQueryPool` for the full range must be the **first** command recorded in the buffer,
  before any timestamp write. Resetting on the host via `vkResetQueryPool` requires
  `VK_EXT_host_query_reset` / Vulkan 1.2 `hostQueryReset` and is fine where available, but the
  in-command-buffer reset is the portable form and works with the record-once/replay-many model
  in `ENGINE.md` §6.1 — every replay re-resets, so results never come from the previous replay.

### 3.3 Where the writes go

Around each dispatch, inside the single per-subgraph command buffer:

```
vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_NONE,            pool, 2*i)      // before
    ... barriers ... vkCmdBindPipeline ... vkCmdBindDescriptorSets ... vkCmdDispatch ...
vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, pool, 2*i+1) // after
```

* Use `vkCmdWriteTimestamp2` where `VK_KHR_synchronization2` / Vulkan 1.3 is available, else
  `vkCmdWriteTimestamp` with `TOP_OF_PIPE` / `BOTTOM_OF_PIPE`.
* Semantics that matter: the timestamp is written when *all previously submitted commands* have
  reached the given stage. So the "before" write must use the earliest stage (`NONE` /
  `TOP_OF_PIPE`) and the "after" write the stage that the dispatch actually completes
  (`COMPUTE_SHADER`, or `BOTTOM_OF_PIPE` for the coarser answer). Using `BOTTOM_OF_PIPE` for
  *both* yields an interval that includes the preceding dispatch's tail — a systematic
  over-report that looks like a plausible number, which is the worst kind of wrong.
* The interval therefore covers the barrier wait plus the dispatch. That is the number we want:
  a dispatch that spends its life waiting on a barrier is not fast.
* `label` should be the op type (`"MatMulNBits"`, `"Add"`), `node_index` the index within the
  subgraph, and `flops` / `bytes` the same values the partitioner computed — so `trace.rs` can
  derive achieved GFLOP/s and GiB/s without re-deriving the model.

### 3.4 Ticks to nanoseconds

```
ns = ticks * VkPhysicalDeviceLimits::timestampPeriod
```

`timestampPeriod` is the number of nanoseconds per tick and it is **not** 1.0 on real hardware.
That is no longer a caution drawn from the spec — it is measured on the machine this text was
written on (§1.4): the RTX 4060 reports `1.0`, and the **Iris Xe in the same laptop reports
52.0833**. Assuming 1.0 under-reports Intel GPU time by 52×, which would manifest as an
implausibly excellent speedup on exactly the vendor we most want to be honest about. (AMD and
Adreno report similarly large periods.) Report the value in
the calibration struct and let `trace.rs` apply it (it does, and there is a test with a non-1.0
period).

### 3.5 `timestampValidBits`

`VkQueueFamilyProperties::timestampValidBits` for the queue family we submit on:

* **`0` ⇒ that queue family supports no timestamps at all.** Not "supports them badly" — none.
  The correct behaviour is to skip GPU tracing entirely and record why, not to emit zeros.
  `GpuTimestampCalibration::is_usable()` returns `false` and `record_gpu_intervals` emits no
  spans; there is a test for that.
* Otherwise only the low `valid_bits` bits are meaningful; the upper bits are undefined and must
  be masked before any arithmetic. `trace.rs` masks (carefully — never shifting by 64) and
  recovers a **single** wrap when `end < begin`. It returns `None`, rather than a guess, when a
  wrap cannot be recovered unambiguously. Please report the real `valid_bits`, not 64.

Note that a 32-bit-valid timestamp on a 1 ns period wraps in about 4.3 seconds, so wrap recovery
is a routine case, not a theoretical one. Measured here (§1.4): the Iris Xe reports **36** valid
bits, which at 52.0833 ns/tick wraps roughly **once an hour** — short enough that a long session
will hit it — while the RTX 4060 reports **64**, which is the case where a wrap bug would never
show up and would therefore survive to ship. Neither device reports 32. Report the real value.

### 3.6 Reading results back without stalling

* Read **after the fence for that submission has signalled**, in the same place
  `submit_and_wait` already learns the work is done. At that point every query is available.
* Use `vkGetQueryPoolResults` with `VK_QUERY_RESULT_64_BIT`. Because we only call it post-fence,
  `VK_QUERY_RESULT_WAIT_BIT` cannot block — but prefer
  `VK_QUERY_RESULT_WITH_AVAILABILITY_BIT` and check the availability word, so a driver that
  disagrees with us degrades to "no data for this dispatch" instead of a hang. Never call it
  with `WAIT_BIT` on a pool whose submission may still be in flight; that stalls the host on the
  GPU and turns the instrument into the thing it is measuring.
* Copying to a device buffer via `vkCmdCopyQueryPoolResults` is the fully non-stalling form and
  is worth having eventually. It is not needed for v1, because we are already synchronising on
  the fence for correctness.

### 3.7 Host↔device correlation

The device tick domain has an arbitrary origin. To place GPU spans on the same absolute timeline
as host spans we need one anchor pair.

* **Preferred: `VK_EXT_calibrated_timestamps`.** `vkGetCalibratedTimestampsEXT` with the
  `DEVICE` domain plus the host domain
  (`QUERY_PERFORMANCE_COUNTER` on Windows, `CLOCK_MONOTONIC` on Linux — must match the domain
  the tracer's clock uses) returns both in one call and, critically, returns `maxDeviation`.
  Pass `maxDeviation` straight through as `anchor_uncertainty_us`.
* **Fallback where the extension is absent:** submit a command buffer containing only a
  timestamp write, bracketed by two host clock reads; wait; pair the device tick with the
  midpoint of the two host reads and report **half the round-trip** as `anchor_uncertainty_us`.
* Re-anchor at least once per session, and preferably per submission for long sessions: GPU and
  host clocks drift, and a stale anchor slides GPU spans off the host timeline in a way that
  looks like a scheduling anomaly.
* `anchor_uncertainty_us` is not decoration. It is the honest error bar on where a GPU span sits
  relative to a host span, and it is surfaced in the trace so nobody reads a 2 µs gap as
  meaningful when the anchor is ±200 µs.

GPU spans are emitted on a synthetic lane (`0x7600_0000 + queue_family`) rather than the
submitting CPU thread's lane, so that asynchronous GPU work is never drawn as if it happened
inside the host call that submitted it. That misdrawing is precisely the illusion this whole
document exists to prevent.

---

## 4. Benchmark methodology

Harness: `bench/`, documented in `bench/README.md`. The rules, from the charter:

1. **Median plus spread, never a single run.** `stats.py` reports median, MAD, IQR, p05/p95 and
   a robust RSD; a sample with RSD > 10% is marked noisy and is not comparable to anything.
2. **Every number carries its environment.** `environment.py` stamps OS, CPU, ORT version,
   Python, the EP artifact path/profile, whether shaders were compiled or stubbed, the Vulkan
   devices as reported by `epctl --probe-loader`, and every `ONNXRUNTIME_EP_VULKAN_*` variable
   in scope. A number without that record is not a result.
3. **Baseline is the ORT CPU EP on the same machine, same process, same ORT build.** Not a
   remembered number, not another machine, not another framework.
4. **Host latency and GPU kernel time are separate columns.** §1.3.
5. **A case the EP did not claim yields no Vulkan number.** CPU fallback is numerically correct
   and therefore invisible in a wall-clock table — the harness reads the claim log and refuses.
6. **A delta inside the noise is not a regression.** `compare.py` requires a delta to exceed both
   the threshold and twice the worse sample's spread.
7. **Nothing is extrapolated.** Unmeasured is `null`.

### 4.0 What the harness *structurally refuses* to do

These are refusals, not conventions. A convention is followed until the first time somebody is
in a hurry.

| Situation | What happens |
|---|---|
| More than one Vulkan device present and no `--device N` | The run **aborts** (exit 2) and lists the devices with their transfer class. The EP's own scoring prefers discrete, so an unpinned run silently benchmarks the 4060. |
| Host ORT older than 1.28 | `register_ep()` returns false and **no Vulkan column is produced at all**. See §5.1. |
| A case the EP claimed no node of | `speedup_end_to_end` is `null`, the row is ⛔, `--fail-on-unclaimed` makes it fatal. |
| `compare.py` given two files from different devices | **Refuses**, exit 2, prints no table. `--cross-device-study` relabels it as a device study with no verdict. |
| `compare.py` given a file that does not name its device | Same refusal. "We forgot to record it" must not degrade to "assume it is the same". |
| Two rows with different recorded `tile_config` | Marked not-comparable. Two `None`s mean *unknown*, never *equal*. |
| Base and PR under different barrier backends | Warned at the top: `ep.force_legacy_barriers` produces a different program. |
| Delta smaller than 2× the worse sample's robust spread | Reported as "within noise", not as a regression. |
| `--gpu-timestamps` on a queue with `timestampValidBits == 0` | GPU tracing is switched off with a printed reason, because zeros are indistinguishable from instant kernels. |
| Transfer calibration on a multi-device machine without `--device` | Refuses. A UMA and a discrete part do not share an affine model; the emitted Rust literal is stamped with the transfer class and the driver it came from. |
| A case named after a model family its producer did not export | **Cannot be constructed.** `producers.assert_family_label_is_earned()` raises during case construction, before any timing exists to be mislabelled. A synthetic op graph called `qwen3_decoder_layer` is a `ProducerProvenanceError`, not a review comment. |
| A model-family label from an *unversioned* exporter | Same refusal. `can_claim_model_family` requires kind=`model`, a named family **and** a version; an unversioned exporter's graph is not reproducible. |
| `compare.py` given two files built by different producers | **Refuses**, exit 2, prints no table. `--cross-producer-study` relabels it as an exporter study with no verdict. |
| `compare.py` given a file with no recorded producer | Same refusal. "We do not know what built these" is not evidence that the same thing built both. |
| Two rows with different `producer_fingerprint` | Marked `🏭 different producer — different graph, not comparable`; the delta is suppressed. |
| A configuration above the §7.2 admission floor (16 KiB shared / 256 invocations) | Verdict `needs-fallback`, `quotable_as_ep_behaviour = false`, and `compare.py` banners the table. Legitimate to measure; not quotable as the EP's behaviour until the floor-compliant fallback is measured too. |
| A configuration nobody recorded | Verdict `unknown`, also not quotable. Today that is every row. |
| Combining a UMA transfer fit with a discrete one | Refused. The blended affine constants would land plausibly between the two and describe neither part. |

`bench/test_plausible_but_wrong.py` tests each of these against the shape of the wrong answer it
prevents, using the real device values from §1.4 as fixtures.

### 4.1 OQ-12

The open question's bar is **≥1.5× over the device's own ORT CPU EP on a GEMM-anchored subgraph,
with zero numerical failures**. Exactly one case in `cases.py` carries `oq12_anchor=True`
(`matmulnbits_q4_b32_K4096_N4096`), so the bar is defined on a named shape rather than on
whichever case happens to look best. It is unanswered and will stay unanswered until a shader
runs.

### 4.2 Calibrating the partition cost model

`rust/src/ops/partition.rs`'s minimum-viable-subgraph constants (`SAFETY = 3.0`, `node_count ≥ 4`,
the 64 KiB boundary floor) are **provisional placeholders**, set conservatively because the cost
model is crude. `bench/transfer_calibration.py` sweeps a doubling byte staircase and fits the
same affine `fixed_ns + bytes / bytes_per_ns` form as `TransferModel::fit`, then prints a
paste-ready Rust literal. Replace the placeholders per device, behind review, with the device
named in the comment — an integrated UMA GPU and a discrete GPU do not share a model, and a
constant tuned on one and applied to the other is worse than the placeholder.

---

## 5. What is real

**Status changed on 2026-07-30.** Everything below §5.0 was written while no shader in this
repository had ever executed on any device. That is no longer true, and the section is kept
rather than rewritten because the reasoning in it is what earned the right to report §6.

### 5.0 What is true as of 2026-07-30T08:21-07:00

* Shaders execute on both local devices, and the model they compute is **correct**: Phi-3.5 at
  the pinned producer-at-version returns `model_output_equivalence = MATCH` against a CPU-only
  run of the same artifact in the same process, on device 0 and on device 1 (§6).
* Therefore §10.0's gate is open and the metric of record may be reported for this artifact.
* **No GPU kernel time exists on any device.** No `VkQueryPool` is created and no
  `vkCmdWriteTimestamp` is recorded — the hooks are still comments in `rust/src/vk/session.rs`
  and `rust/src/vk/dispatch_integration.rs`. Every number in §6 is host wall time. §3 remains a
  specification.
* Device-backed allocation is off by default, so every tensor is host-staged. §6 is a
  measurement of a **staging-bound** configuration.

The paragraph that follows was written before all of that and is retained verbatim, because a
document that quietly edits its own past is the least trustworthy instrument in the room.

---

From `docs/DESIGN.md` §9.1.2, restated here because this is the document where the temptation
lives:

**No shader in this repository has ever executed on any device.** The host-side test suite is
green and large; it is evidence about partitioning, capability gating, descriptor plumbing and
lint discipline. It is *not* evidence about GPU behaviour, numerical accuracy on device, or
performance.

### 5.1 A worked example of why the gates exist

On 2026-07-29 the harness was run for the first time on real hardware, pinned to the RTX 4060,
with a release build of the EP. The installed onnxruntime was 1.27, which rejects the plugin's
API version 28, so ORT ran every node on its own CPU EP — while the harness's "vulkan" column
was still labelled Vulkan. The raw medians it collected were:

```
matmulnbits_q4_b32_K4096_N4096   vulkan= 1.361 ms    cpu= 2.311 ms
```

That is **1.70×**, which is above the OQ-12 pass bar of 1.5×, on the OQ-12 anchor case, on a
real discrete GPU. It is also entirely fictitious: both columns were the same CPU code, and the
gap is run-to-run noise (both samples were flagged noisy, rsd 38.6% and 62.5%).

Nothing was reported, because the claim gate read the EP's own claim log, saw zero claimed
nodes, marked every row ⛔ and set `speedup_end_to_end` to `null`. A version gate was then added
so the Vulkan column is not produced at all under an ORT the EP cannot load.

This is what "a benchmark without a baseline and a variance number is a rumour" looks like in
practice: the number was available, plausible, on the right case, on the right hardware, and
wrong. **This remains the only quantitative result this project has produced, and it is a
measurement of the CPU.**

### 5.2 Consequently

* This document contains no performance numbers, and none may be added to it, to
  `bench/README.md`, or to any commit message until the corresponding op is green in
  `tests/ops/` **on a real device**. The device *facts* in §1.4 are not performance numbers:
  they are properties of the hardware, read from the driver, and every one of them is needed to
  interpret a timing correctly.
* Everything in §3 is a specification. It has been designed against the Vulkan spec and unit
  tested on synthetic tick values; it has never ingested a tick produced by a GPU. It has now
  been checked against two real `timestampPeriod` and `timestampValidBits` values (§1.4), which
  is a check of the arithmetic, not of the plumbing.
* The Vulkan SDK 1.4.350.0 is installed, all 168 shader variants compile, and `epctl
  --probe-loader` confirms both devices pass the §7.2 gate. The host ORT is now **1.28.0**, so
  the plugin **loads**: `register_execution_provider_library` succeeds, the EP enumerates the
  devices, and its claim predicates run against real graphs. Every op declines, with the honest
  reason — for `Add`: *"is in the op table but not enabled: its compute shader compiles but has
  never executed on a device, so claiming it would be a bet"*. **No shader has executed.** The
  EP being loadable is a real milestone and it is not a measurement.
* **Every "vulkan" column produced so far is the CPU EP.** On the RTX 4060,
  `add_fp32_4096x1024` reported 0.858 ms "vulkan" against 1.247 ms "cpu". That ratio is 1.45×,
  it is two samples of the same CPU code, and the claim gate reported it as ⛔ NOT CLAIMED with
  `speedup_end_to_end: null`. This is the second time the harness has manufactured a plausible
  speedup out of CPU noise (§5.1 was the first, at 1.70×) and the second time the gate caught
  it. The gates are not theoretical.
* **No number here is portable yet, and the harness says so.** Every row's `portability` verdict
  is `unknown` (§2.2), because the engine does not yet report the tile shape, workgroup size or
  memory path that produced it. Both local GPUs sit 2–3× above the §7.2 admission floor, so
  "it worked here" would not have been portability evidence even if a kernel had run.
* **When the first dispatch lands, it will be a smoke test.** A single `Add` on one GPU against
  the CPU EP, at whatever iteration count, is a check that the timing path produces plausible
  non-zero numbers on the expected side of zero. It is not a statement about anything a user
  cares about, and it must not be quoted as one. The first quotable number requires: a claimed
  graph, a named device, a named producer, a named tile config that is either floor-compliant or
  paired with a floor-compliant fallback, a median with spread over a non-noisy sample, and a
  GPU kernel time separate from host latency.

The instrument is built and calibrated. It has not yet been pointed at anything.

---

## 6. The first measurement — Phi-3.5, both devices, 2026-07-30

This is the first performance measurement this project has. **It is not a speedup and it is not
good news.** It is a measurement, with its correctness verdict attached, of a configuration that
is known to be pathological in two specific ways that are named with the numbers.

Reproduce with:

```
$env:VULKAN_SDK="C:\VulkanSDK\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\Bin;$env:PATH"
$env:ONNXRUNTIME_VULKAN_EP_LIB="rust\target\release\onnxruntime_vulkan_ep.dll"
python bench\phi35.py --iters 20 --warmup 10 --repeats 3 --out bench\results\phi35.json
```

### 6.1 What was measured

| | |
|---|---|
| artifact | `Phi-3.5-mini-instruct/cuda-int4-rtn-block-32`, Foundry cache, never committed |
| producer | `onnxruntime-genai/builder.py@Phi-3.5-mini-instruct/cuda-int4-rtn-block-32#8e5005d36bbd` (`com.microsoft` contrib graph) |
| workload | single-token prefill, empty KV cache — `tests/ops/test_phi35.py::_build_phi35_feeds` |
| host ORT | 1.28.0 |
| build | `cargo build --release`, Vulkan SDK 1.4.350.0, all 168 shader variants compiled |
| coverage | **257 claimed of 363 nodes probed** |
| islands | **257** (EP counter `subgraphs_live`) — every claimed node is its own single-node island |
| memory | **staging-bound**: `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` unset, so every tensor is host-staged by construction |
| GPU kernel time | **none** — no `VkQueryPool` exists; all figures are host wall time |
| tile config | not reported by the engine; `portability` verdict is `unknown` (§2.2) |

Note the coverage number: the metric of record for this artifact is now
`(claimed_op_coverage = 257, island_count = 257, largest_island_flops = null)`, gated on `MATCH`.
`largest_island_flops` is null because the EP does not emit it, and it would not currently
discriminate anything if it did — with every island a single node, the largest island is one
`MatMulNBits`.

### 6.2 Correctness verdict — the gate, not a footnote

Measured **in the same process, in the same run, on the same session objects that were then
timed**, because §10.0's `UNMEASURED` is defined as "no CPU-only comparison was performed on this
artifact in this run", and a verdict from an earlier run of an earlier build is exactly that.

| | device 0 — Intel Iris Xe | device 1 — RTX 4060 |
|---|---|---|
| `model_output_equivalence` | **MATCH** | **MATCH** |
| argmax (vk / cpu) | 30751 / 30751 | 30751 / 30751 |
| top-10 overlap | 10/10 | 10/10 |
| max abs diff, logits | 0.0312 (0.24%) | 0.0352 (0.27%) |
| KV outputs, max abs diff | 6.1e-05 and below | 6.1e-05 and below |

This independently reproduces the coordinator's numbers to the last digit, through a different
code path. Per **R9** that raises confidence and not evidence — the falsifier is that the same
harness returns `DIVERGENT` on all-zero logits and on top-k disagreement, which is asserted in
`bench/test_plausible_but_wrong.py`.

### 6.3 The numbers

Median of 20 timed iterations after 10 discarded, three whole-process repeats per device.
**The devices are not compared with each other** — different transfer class, different shared
memory budget, different `timestampPeriod`; `bench/compare.py` refuses it structurally.

#### Device 0 — Intel(R) Iris(R) Xe Graphics, UMA, driver 101.6737, Vulkan 1.4.309

```
vulkan   median 2790.7 ms   MAD 31.9   p05-p95 2719.9-2842.8   rsd 1.7%   n=20
cpu-only median  229.8 ms   MAD 24.3   p05-p95  224.4- 349.0   rsd 14.3%  n=20
run-to-run medians (3 repeats): vulkan 2755.9 / 2790.7 / 2827.8 ms  (spread 1.03x)
                                cpu     204.1 /  229.8 /  251.1 ms  (spread 1.23x)
```

**Vulkan is 2561 ms slower — 12.1x slower — than pure CPU on this artifact.**

#### Device 1 — NVIDIA GeForce RTX 4060 Laptop GPU, discrete, driver 591.55, Vulkan 1.4.325

```
vulkan   median 1465.9 ms   MAD 26.2   p05-p95 1418.4-1590.7   rsd 2.6%   n=20
cpu-only median  185.9 ms   MAD  6.7   p05-p95  177.6- 230.9   rsd 5.4%   n=20
run-to-run medians (3 repeats): vulkan 1410.2 / 1471.5 / 1465.9 ms  (spread 1.04x)
                                cpu     220.4 /  181.5 /  185.9 ms  (spread 1.21x)
```

**Vulkan is 1280 ms slower — 7.9x slower — than pure CPU on this artifact.**

### 6.4 Where the time goes — pricing the 257-island boundary

Attributing the entire host-side difference to the island boundaries:

| device | delta | islands | per island |
|---|---|---|---|
| Intel Iris Xe | +2561 ms | 257 | **≥ 9.96 ms** |
| RTX 4060 | +1280 ms | 257 | **≥ 4.98 ms** |

**This is a lower bound, not an estimate, and the direction is counter-intuitive.** The same
difference also contains whatever the GPU *saved* on the 257 GEMVs it took over. If the Vulkan
`MatMulNBits` is faster than the CPU's, that saving is netted against the boundary cost, so the
true per-crossing cost is **larger** than the figure above. Separating them requires GPU kernel
time, which requires the §3 timestamps, which do not exist. That is recorded as a missing
instrument rather than approximated.

Five milliseconds per island crossing on a discrete GPU is far more than a PCIe copy of a few
KiB of activations. Two hypotheses, and the instrument that separates them:

1. **Per-island submit-and-wait.** Each `Compute` records, submits and blocks on a fence, so the
   graph pays a full round trip 257 times per inference rather than pipelining. Predicts a cost
   roughly independent of tensor size.
2. **Per-island staging round trip.** Each boundary copies host→staging→device and back, with
   an allocation each way. Predicts a cost that scales with tensor bytes, and that is
   *qualitatively different on the two devices* — the Iris Xe is UMA, so the "copy" is a copy
   within one memory pool, while the 4060 crosses PCIe.

The Intel part being **2x more expensive per island than the discrete part while having no bus
to cross** argues against (2) being dominant and for (1), or for a fixed per-submission cost
that the weaker integrated queue pays more of. **Instrument that decides it:** the §3 timestamps
plus `Phase::Submit`/`Phase::Readback` spans in `trace.rs`, which would show submit-to-completion
directly. Until then this is a hypothesis, labelled as one.

**What this tells Mouse.** The cheapest large win available is not a faster `MatMulNBits`; it is
*fewer islands*. Every `SkipSimplifiedLayerNormalization` (128 of them) and
`GroupQueryAttention` (32) that lands on the GPU merges neighbouring islands and removes two
crossings each. At ≥5 ms a crossing on the fast device, merging the graph is worth an order of
magnitude more than any kernel tuning.

### 6.5 What these numbers may not be used for

* **Not a speedup, in either direction.** They are one configuration of one artifact from one
  producer on one desk.
* **Not "what the Vulkan EP does".** The staging-bound label is part of the number.
  `epctl --check-counters --require-device-memory` exits 1 on this configuration.
* **Not a device comparison.** 12.1x and 7.9x are not commensurable: different transfer class,
  different tile selection, different driver, and the CPU baselines were measured in different
  processes minutes apart.
* **Not portability evidence.** Both devices are Windows desktop-class; `portability` is
  `unknown` for every row because the engine does not report the configuration that produced it.

### 6.6 Instruments — for each claim, what goes red if it is false

**R9 (§10.0.1): confidence scales with agreeing instruments; evidence scales only with
falsifying ones.** Each row names the instrument, and the silence set says what it cannot see.

| Claim | Instrument that goes red if false | What it cannot see |
|---|---|---|
| The EP was actually used | `refuse_if_ep_absent` — `EP_NAME in get_providers()`, checked before anything is timed | An EP that loads and does nothing |
| The EP did work | `refuse_if_nothing_claimed` — claimed-node count from the EP's own claim log | Work that is claimed but wrong |
| The answer is correct | `classify_outputs` → `DIVERGENT`; explicit all-zero guard and top-k equality | Errors that preserve argmax *and* the whole top-10 |
| Coverage is 257 nodes | claim log vs `subgraphs_live` — two independent counters | Both being wrong in the same way |
| Islands is 257 | `dispatch_accounting`: `compute_calls == islands x inferences`, integer equality, no tolerance | An island that runs but computes nothing (that is the correctness gate's job) |
| The run is staging-bound | `staging_label` basis `configuration` — the env var is off, so it cannot be otherwise | Nothing; it is a property of the configuration, not an observation |
| The median describes a steady state | `stats.drift` — half-medians and monotone fraction | A run that is steady but steadily wrong |
| The number is reproducible | `--repeats`: three whole-process runs, spread reported alongside | Systematic error common to all repeats |
| The CPU baseline is comparable | `baseline_disagreement` — fires above 2x between workers | Drift smaller than 2x |
| No GPU kernel time is claimed | `timestamp_audit` reports `gpu_kernel_time_status: UNMEASURED` and there is no code path that can produce one | — |

Two of these went red during this session and changed the result:

* `stats.drift` caught the Intel run ramping 724 → 903 → 1447 → 2080 → 2669 ms before flattening
  near 2790. Three warmup iterations produced a median of 2639 ms with a p05 of 646 ms — a
  bimodal sample whose median is an artefact of when measurement stopped. The warmup default is
  now 10. **Note the direction: the device got slower, not faster.** A "take the minimum"
  convention would have reported ~700 ms here, four times too fast, and looked entirely
  reasonable.
* `baseline_disagreement` caught the CPU-only baseline moving 218 ms → 665 ms between the two
  device workers in the first run — same CPU, same artifact, 3x apart, from page-cache pressure
  after loading a 2.2 GB model. Both were real measurements. Neither was comparable to the
  other.

---

## 7. GPU timestamp verdict on both devices

Run: `python bench/timestamp_audit.py`. Exits non-zero on any disagreement.

### 7.1 Inputs to the conversion — VERIFIED on both devices

Two instruments that read the driver by different routes are cross-checked: the EP's own
capability probe (`epctl --probe-loader`, i.e. `rust/src/vk/caps.rs` — the values the conversion
will actually be handed) and `vulkaninfoSDK`.

| | Intel Iris Xe | RTX 4060 | lavapipe (CI) |
|---|---|---|---|
| `timestampPeriod`, EP | 52.0833 | 1.0 | 1.0 |
| `timestampPeriod`, vulkaninfo | 52.0833 | 1.0 | 1.0 |
| `timestampValidBits`, EP | 36 | 64 | 64 |
| `timestampValidBits`, vulkaninfo | 36 | 64 | 64 |
| UMA, EP vs derived | true / true | false / false | — |
| counter wraps every | **3579 s (0.99 h)** | never | never |

**Verdict: agree on both devices, on all three properties.** The EP is reading the right fields
from the right queue family.

### 7.2 The arithmetic — VERIFIED against these values, by unit test

`rust/src/trace.rs` now tests the conversion with the *measured* constants rather than invented
ones:

* `treating_intel_ticks_as_nanoseconds_is_wrong_by_fifty_two_times`
* `an_intel_counter_wrap_does_not_produce_a_negative_or_absurd_duration`
* `undefined_upper_bits_on_a_thirty_six_bit_counter_are_masked_away`
* `a_wrap_is_only_recoverable_when_the_counter_is_narrower_than_a_u64`

### 7.3 The cross-platform hazard, stated precisely

**Neither the RTX 4060 nor CI can falsify this code.** lavapipe and NVIDIA both report
`timestampPeriod = 1.0` and `timestampValidBits = 64`, so on both of them:

* treating ticks as nanoseconds is *correct*, and
* the valid-bit mask is a no-op.

A build that dropped the period scaling and the mask entirely would be green on the discrete GPU
and green in CI, and would under-report every Intel duration by **52x** while looking completely
reasonable — nothing negative, nothing absurd, wrong by a constant. This is the same shape as
the tracer-epoch hazard in §1.1, and it is why *"Intel is the spec-conformance oracle"* is not a
slogan: on this property the Iris Xe is the **only** instrument on this desk, and in CI there is
none at all.

`bench/timestamp_audit.py` reports `period_mistake_detectable_on` and `mask_exercisable_on` for
exactly this reason, and adds a problem — non-zero exit — when neither list has a member. On a
CI runner with only lavapipe, that fires.

Mobile makes it worse, not better: Adreno and Mali report periods in the tens of nanoseconds and
valid bits well under 64, so this is the *common* case in the targets that justify the project,
and the desktop configuration is the outlier.

### 7.4 End-to-end GPU timing — UNMEASURED

Not "passing". Not "probably fine". `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` sets a flag that nothing
consumes: no `VkQueryPool` is created, no `vkCmdWriteTimestamp` is recorded, no
`vkGetQueryPoolResults` is called, and `VulkanTracer::record_gpu_intervals` has never been handed
a tick produced by a GPU. §3 is a specification with verified inputs and verified arithmetic and
no plumbing. `trace.rs` already emits the caveat in the trace itself when the flag is off; it
should be read as applying whether it is off or on, until the query pool exists.

---

## 8. `onnx-runtime-tracer` adoption status

Justin's own crate, already used by the sibling project `onnxruntime-mlx` in `rust/src/trace.rs`.
Status here, stated as three separate facts because they are at three different levels of done:

| | status |
|---|---|
| Dependency adopted and pinned | **done** — `rust/Cargo.toml`: `onnx-runtime-tracer = { version = "0.1.0-dev.5", default-features = false }`, with the pin rationale in §1.1 (absolute UNIX-microsecond clock, so a trace emitted from inside the plugin dylib overlays the host's own trace with no offset negotiation) |
| Module written, structured as MLX's sibling | **done** — `rust/src/trace.rs`, `pub mod trace;` in `lib.rs`, span vocabulary in §1.2, 16 unit tests green |
| Environment wiring | **done** — `ONNXRUNTIME_EP_VULKAN_TRACE=<path>` and `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1`, inert and zero-cost when unset |
| **Called from the execution path** | **NOT DONE** |

**The last row is the one that matters and it is not done.** No span is opened anywhere outside
`trace.rs`'s own tests: `ep.rs`, `factory.rs`, `vk/session.rs` and `vk/dispatch_integration.rs`
contain no call into it. Verified empirically rather than by reading — with both env vars set, a
Phi-3.5 run that executed 257 islands over four inferences produced **no trace file at all**:

```
$env:ONNXRUNTIME_EP_VULKAN_TRACE="...\trace_probe.json"
$env:ONNXRUNTIME_EP_VULKAN_TRACE_GPU="1"
python bench\phi35.py --device 1 --iters 2 --warmup 1 --repeats 1
Test-Path $env:ONNXRUNTIME_EP_VULKAN_TRACE   ->   False
```

That is the falsifier for "the tracer is adopted": if it were wired, the file would exist. It
does not. The remaining work is in files owned by Switch (`vk/**`, `engine.rs`) and Tank
(`ep.rs`, `factory.rs`), so it is a routed requirement and not something this document can close:

* `Phase::Compile` around subgraph compilation (`ep.rs` `Compile`).
* `Phase::Prepack` / `Phase::Upload` around weight prepack and upload.
* `Phase::Record` around command-buffer recording, `Phase::Submit` around the submit itself, and
  `Phase::Readback` around the result copy — noting §1.3: **host wall time around a submit
  measures almost nothing**, which is why `Phase::Submit` has a unit test asserting it observes
  no GPU work.
* `VulkanTracer::record_gpu_intervals` fed from the §3 query pool, which is the only thing that
  turns any of this into GPU time.
