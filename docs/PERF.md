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

> **Superseded 2026-07-30T18:xx by §9.** Three of the bullets below were true when written and
> are false now: the `VkQueryPool` exists, GPU kernel time is measured on both devices, and the
> partitioner is wired into `GetCapability`. They are struck rather than deleted. **§9 is the
> current state; read it before quoting anything in §5 or §6.**

* Shaders execute on both local devices, and the model they compute is **correct**: Phi-3.5 at
  the pinned producer-at-version returns `model_output_equivalence = MATCH` against a CPU-only
  run of the same artifact in the same process, on device 0 and on device 1 (§6).
* Therefore §10.0's gate is open and the metric of record may be reported for this artifact.
* ~~**No GPU kernel time exists on any device.** No `VkQueryPool` is created and no
  `vkCmdWriteTimestamp` is recorded — the hooks are still comments in `rust/src/vk/session.rs`
  and `rust/src/vk/dispatch_integration.rs`. Every number in §6 is host wall time. §3 remains a
  specification.~~ **FALSE as of 2026-07-30 evening.** Switch built the query-pool path; GPU
  kernel time is measured on both devices and the tick→ns conversion is verified end to end on
  the part where it can fail (§9.6). See §9.
* Device-backed allocation is off by default, so every tensor is host-staged. §6 is a
  measurement of a **staging-bound** configuration. **Still true in §9**, and §9.3 shows it is
  now the single largest cost in the run.

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

## 6. The first measurement — Phi-3.5, both devices, 2026-07-30 (**SUPERSEDED by §9**)

> **Superseded 2026-07-30 evening.** Two changes since invalidate every number here: the
> partitioner was wired into `GetCapability` (islands 321 → 33) and the `VkQueryPool` path
> landed. The section is kept for its reasoning and its refusals, which still hold. **Its
> figures may not be quoted — §9 replaces them.** Note also that §6's device *labels* are
> subject to the ordering defect documented in §9.1, and its timings were taken with no record
> of machine load, which §10 shows is worth up to 9.5× on its own. Three independent reasons
> not to quote §6, any one of which would be sufficient.

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

| | `--device 0` — **NVIDIA RTX 4060** | `--device 1` — **Intel Iris Xe** |
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

#### `--device 0` — NVIDIA GeForce RTX 4060 Laptop GPU, discrete, driver 591.55, Vulkan 1.4.325

> **Relabelled 2026-07-30.** This block was published as "Device 0 — Intel Iris Xe". It was
> not. See §11.1. The measurements are unchanged; only the name was wrong.

```
vulkan   median 2790.7 ms   MAD 31.9   p05-p95 2719.9-2842.8   rsd 1.7%   n=20
cpu-only median  229.8 ms   MAD 24.3   p05-p95  224.4- 349.0   rsd 14.3%  n=20
run-to-run medians (3 repeats): vulkan 2755.9 / 2790.7 / 2827.8 ms  (spread 1.03x)
                                cpu     204.1 /  229.8 /  251.1 ms  (spread 1.23x)
```

**Vulkan is 2561 ms slower — 12.1x slower — than pure CPU on this artifact.**

#### `--device 1` — Intel(R) Iris(R) Xe Graphics, UMA, driver 101.6737, Vulkan 1.4.309

> **Relabelled 2026-07-30.** Published as "Device 1 — NVIDIA GeForce RTX 4060". It was not.
> See §11.1.

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
| NVIDIA RTX 4060 (`--device 0`) | +2561 ms | 257 | **≥ 9.96 ms** |
| Intel Iris Xe (`--device 1`) | +1280 ms | 257 | **≥ 4.98 ms** |

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

> **Corrected 2026-07-30 — this paragraph's premise was a mislabel, and the mislabel is what made
> the paragraph interesting.** As published it read: *"The Intel part being 2× more expensive per
> island than the discrete part while having no bus to cross argues against (2) being dominant and
> for (1)."* With the labels right (§11.1) the observation is the reverse: **the discrete part paid
> ~2× per island (9.96 ms) and the UMA part paid ~4.98 ms.** That is what hypothesis (2) predicts —
> the device that crosses PCIe pays more for a staging round trip than the device that does not —
> and it is so unremarkable that it would never have prompted a hypothesis at all.
>
> This is the whole chain: a mislabelled row produced a surprising observation, the surprise
> produced my fixed-per-submission hypothesis, the hypothesis got a `VkQueryPool` built to test it,
> and the query pool then produced a phase table that was itself misread (§11.2). **Two of the
> three links were mine.** The instrument was worth building regardless — it is what closed the
> 52× trap and what now measures upload — but the honest account is that the premise was an
> artifact, not a finding. §11.1 records why `device_identity_check` exists and why it is green on
> everything published after it.
>
> **Retained below unedited** so the reasoning chain can be audited rather than quietly repaired.

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

### 7.4 End-to-end GPU timing — ~~UNMEASURED~~ **MEASURED, and the 52× trap is falsified**

> **Rewritten 2026-07-30 evening.** The paragraph struck below was accurate when written and is
> now false. A stale caveat is the most expensive kind of documentation defect this project has
> produced: it survives every review because it *reads* like caution.

~~Not "passing". Not "probably fine". `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` sets a flag that
nothing consumes: no `VkQueryPool` is created, no `vkCmdWriteTimestamp` is recorded, no
`vkGetQueryPoolResults` is called, and `VulkanTracer::record_gpu_intervals` has never been handed
a tick produced by a GPU.~~

`ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` now produces real device-lane spans on both local devices.
The end-to-end conversion check is in `bench/phases.py::timestamp_conversion_integrality` and is
reachable from `bench/timestamp_audit.py --trace <file>`. Its verdicts on 2026-07-30:

| device | verdict |
|---|---|
| Intel Iris Xe (`timestampPeriod` 52.0833, 36 valid bits) | **PASS, and DECISIVE** |
| NVIDIA RTX 4060 (`timestampPeriod` 1.0, 64 valid bits) | **VACUOUS** — reported as a gap, never as a pass |

The check: `gpu_ns` is a tick count times `timestampPeriod`, and tick counts are integers, so
`gpu_ns ÷ period` must be a whole number. A build that dropped the period scale emits raw ticks,
and ticks ÷ 52.0833 is fractional. This is **only decisive where the period is not 1.0**, which
is exactly why the audit exits non-zero when no such device is present: on NVIDIA, and on the
lavapipe CI runner, the arithmetic is a no-op and the check cannot fail. It is reported as
`decisive: false` and surfaced by `red_flags()` as "NOT DECISIVE", never as a pass.

**Why this matters more now, not less.** GPU kernel time is 12.6% (NVIDIA) / 43.9% (Intel) of
time inside `Compute` (§9.3). A 52× under-report of the Intel GPU column would move the *wall
clock* not at all — the wall clock is dominated by host staging — so it is invisible to every
end-to-end benchmark on the project. Only the integrality check sees it.

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

---

## 9. The phase split — where the time actually goes (2026-07-30, both devices, current `main`)

> ⛔ **THE TIMINGS IN THIS SECTION ARE WITHDRAWN — see §10.** Both traces behind §9 were
> captured on a machine that other agents were compiling on. An instrument built afterwards
> (`phases.contention_signature`) tested them from the inside and found that on the RTX 4060,
> **20 of 33 island slots recorded a ≥ 2× spread in host time across repetitions of identical
> work while the GPU's own clock said that work was constant to within 0.3%** — one slot swung
> 190.93 → 501.56 → 1747.51 ms. The Intel run varied 5.25× across whole inferences. Neither run
> was in a steady state, so every duration, share and ratio below is a mean over conditions that
> were not held constant.
>
> **What survives** is everything that is a count, an integer identity or a structural fact:
> the device-ordering defect and its correction (§9.1), the gate verdicts (§9.2), the
> retirement of the per-island statistic (§9.7), the ratio refusal (§9.8), the
> `largest_island_flops` state (§9.9), the 52× closure (§9.6), and the *qualitative* finding
> that recording dominates and that the bulk of it is `memcpy` (§9.4) — the memcpy share is a
> ratio measured inside each individual record span and is not a between-inference comparison.
> **What does not survive** is every millisecond figure, every phase percentage, and the
> record-scaling and warmup verdicts of §9.5, which are statements about durations.

Everything in §6 predates two changes that invalidate it: `partition.rs` was wired into
`GetCapability` (islands 321 → 33), and the `VkQueryPool` path landed. §6 is kept for its
reasoning; **its numbers are superseded by this section.**

Run record: `bench/results/phi35-2026-07-30-phases.json`. Command:

```powershell
$env:VULKAN_SDK="C:\VulkanSDK\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\Bin;$env:PATH"
$env:ONNXRUNTIME_VULKAN_EP_LIB="$PWD\rust\target\release\onnxruntime_vulkan_ep.dll"
python bench\phi35.py --iters 20 --warmup 10 --repeats 3 --trace-iters 6
```

### 9.0 A hypothesis died, and that is the result

A fixed per-submission cost was proposed and deliberately **not** designed around until an
instrument existed. The instrument now exists, and the hypothesis is dead:

| | share of time inside `Compute` |
|---|---|
| `vulkan.submit` | **0.6%** (NVIDIA) / **16.4%** (Intel) |

Submission is not the cost on the discrete part. The reasoning that produced the hypothesis was
drawn from real data and was about the wrong stage — which is what `DESIGN.md` §10.0.1 R9 means
by *evidence scales only with falsifying instruments*. **It took an instrument, not an argument.**

### 9.1 Two devices, two orderings, one mislabelled table — read this before any number

The first version of this table named the wrong GPU on **every row**. It is documented rather
than quietly corrected, because the trap is still live for every other harness in the repo.

There are two orderings of the same two devices on this machine:

| ordering | who uses it | index 0 | index 1 |
|---|---|---|---|
| `vkEnumeratePhysicalDevices` | `vulkaninfo`, `bench/devices.py::probe()`, `epctl --probe-loader`'s "Device N" | Intel Iris Xe | NVIDIA RTX 4060 |
| **best-first** (`DeviceKind::score`: discrete 4 > integrated 3) | **`ep.device_index`** — `engine.rs::probe_devices` is documented "sorted best-first, so index 0 is the default device" | **NVIDIA RTX 4060** | **Intel Iris Xe** |

A harness that passes `ep.device_index = 0` and then labels the row with `probe()[0]` prints
Intel's name, driver, transfer class and `timestampPeriod` over NVIDIA's numbers. Nothing raises.
Every gate stays green. The result is not noisy — it is *wrong and confident*.

**Instrument:** `bench/devices.py::device_identity_check`. It reads the `timestampPeriod` and
`timestampValidBits` the EP wrote into that row's *own trace* (52.0833/36 = Intel, 1.0/64 =
NVIDIA) and compares them to the device the label claims. Disagreement → `MISLABELLED`, and the
row is relabelled from the trace. No fingerprint → `UNVERIFIED`, and `_describe` prints
`UNIDENTIFIED DEVICE` rather than a plausible wrong name. Ambiguous fingerprint (NVIDIA and
lavapipe are both 1.0/64) → no identification, never a first match.

*This is also a routing note.* `engine.rs:543` documents `DeviceInfo::index` as both "index into
`vkEnumeratePhysicalDevices`" *and* "the value of the `ep.device_index` option". Those are two
different numbers. One doc comment, two orderings — that is where the defect lives.

### 9.2 Gate first (§10.0) — no number below may be read without this

| | NVIDIA RTX 4060 Laptop (`ep.device_index` 0) | Intel Iris Xe (`ep.device_index` 1) |
|---|---|---|
| `model_output_equivalence` | **MATCH** | **MATCH** |
| device identity | MATCH (trace fingerprint agrees with best-first order) | MATCH |
| `EP_NAME in get_providers()` | yes | yes |
| claimed nodes | 321 of 363 probed | 321 of 363 |
| islands (`subgraphs_live`) | 33 | 33 |
| `dispatch_accounting` | **ok** — `compute_calls 1023 == 33 islands × 31 inferences` | **ok** — same |
| `gpu_span_accounting` | **ok** — `sum(subgraph.nodes) 5457 == 5457 GPU spans` | **ok** — same |
| memory configuration | **staging-bound** (`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` unset) | staging-bound |

`compute_failures` is 0 on both. Per §9.1.3 that is an **execution-status counter and never a
correctness signal**; it is recorded and not relied on. The gate is `model_output_equivalence`.

**The two devices are not compared.** Different transfer class (UMA vs discrete), different
shared-memory budget, different `timestampPeriod`. `bench/compare.py` exits 2 on a cross-device
comparison and that refusal stands.

### 9.3 The phase split

Shares are of **time inside `Compute`** — the sum of `vulkan.subgraph` spans. That is the EP's
own view of its execution and is **not** process wall time; ORT's graph execution, the CPU EP's
nodes between islands and session setup are all outside it. These shares may not be restated as
shares of the benchmark's wall clock.

The timed pass runs with tracing **off**; the split comes from a separate instrumented pass, and
`tracing_overhead_ratio` (traced median ÷ untraced median) is measured rather than assumed:
1.0207× on NVIDIA, 0.8659× on Intel. The Intel figure being below 1.0 is not negative overhead —
it means the machine state moved between the two passes, and it is reported for that reason.

**NVIDIA RTX 4060 Laptop** — 48563.24 ms inside `Compute`, 561 subgraph invocations:

| phase | total | share | n | median |
|---|---|---|---|---|
| `vulkan.record` | **33456.17 ms** | **68.9%** | 561 | 50.774 ms |
| ├ of which: host **upload memcpy** | **33042.07 ms** | **98.8% of record** | 561 | — |
| └ of which: command construction | **414.10 ms** | 1.2% of record | 561 | 0.459 ms |
| `vulkan.submit` | 308.18 ms | 0.6% | 561 | 0.452 ms |
| `vulkan.fence_wait` | 13980.26 ms | 28.8% | 561 | 27.776 ms |
| unattributed inside `Compute` | 818.63 ms | 1.7% | — | — |
| **GPU kernels (sum)** | **6110.00 ms** | **12.6%** | 5457 | — |

**Intel Iris Xe** — 72148.67 ms inside `Compute`, 561 subgraph invocations:

| phase | total | share | n | median |
|---|---|---|---|---|
| `vulkan.record` | **23946.22 ms** | **33.2%** | 561 | 25.368 ms |
| ├ of which: host **upload memcpy** | **17231.74 ms** | **72.0% of record** | 561 | — |
| └ of which: command construction | 6714.48 ms | 28.0% of record | 561 | 1.393 ms |
| `vulkan.submit` | 11830.53 ms | 16.4% | 561 | 0.349 ms |
| `vulkan.fence_wait` | 34978.91 ms | 48.5% | 561 | 56.652 ms |
| unattributed inside `Compute` | 1393.01 ms | 1.9% | — | — |
| **GPU kernels (sum)** | **31652.94 ms** | **43.9%** | 5457 | — |

Per-kernel GPU time (summed from the per-span `gpu_ns` float, **not** from the integer-µs `dur`
— several of these kernels run in 2–3 µs, where truncation is a 15–30% error over 5457 spans):

| kernel | n | NVIDIA | Intel |
|---|---|---|---|
| `q_gemv_matmul_nbits_f16` | 2737 | 5990.73 ms | 31432.43 ms |
| `skip_simplified_layer_norm_f16` | 1088 | 97.65 ms | 149.10 ms |
| `ew_binary_mul_f16` | 1088 | 14.39 ms | 48.69 ms |
| `ew_unary_sigmoid_f16` | 544 | 7.22 ms | 22.73 ms |

`unattributed` is reported rather than folded into a neighbouring phase: it is the input-pointer
reads, buffer allocation and descriptor-pool work before recording, plus the readback memcpy and
the writes into ORT's output tensors after the fence. **A phase split whose parts do not sum to
the whole should say so.**

### 9.4 The 68% is not `vkCmd*`. It is `memcpy`. — the finding Switch needs

`vulkan.record` is not one activity. It brackets `vkBeginCommandBuffer` → `vkEndCommandBuffer`,
and **the host memcpy of the island's inputs into staging buffers happens inside it.**

| | NVIDIA | Intel |
|---|---|---|
| upload bytes per full run | **33959.1 MiB** over 561 invocations | same |
| upload memcpy time | 33042.07 ms | 17231.74 ms |
| implied host bandwidth | 1.0037 GiB/s | 1.9245 GiB/s |
| command construction residual | **414.10 ms** | 6714.48 ms |
| readback | 13.7 MiB, 38.45 ms | 13.7 MiB, 38.65 ms |

**~60 MiB is copied into staging on every single `Compute` call**, for a 33-island partition over
31 inferences. The weights are being re-uploaded per invocation. On NVIDIA, optimising the
recording loop can recover at most **414 ms of a 33456 ms phase**; the other 98.8% goes away only
if the data stops being re-copied. This is a memory-residency problem, not a recording-loop
problem, and `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` being unset is why.

**Instrument:** `phases.transfer_totals`. The EP pushes `vulkan.transfer_bytes` and
`vulkan.transfer_gib_s` counters **per transfer**, so `bytes ÷ gib_s` recovers exactly the
duration the EP itself measured — the split is not inferred from span nesting. If upload time
ever exceeded the enclosing `record` span, `phase_containment` goes red.

### 9.5 Does recording scale with island size? **No — with bytes.** Is the warmup decline real? **Yes.**

Both devices return `size_verdict = SCALES_WITH_BYTES_NOT_DISPATCHES`.

| | NVIDIA | Intel |
|---|---|---|
| dispatch count spans | 10× (1 vs 10 dispatches) | 10× |
| steady-state record median, 1 vs 10 dispatches | 13.053 → 50.637 ms (**3.9×**) | 6.657 → 24.880 ms (**3.7×**) |
| mean upload bytes, 1 vs 10 dispatches | 15.93 MB → 64.96 MB (**4.08×**) | same |
| **implied record bandwidth by island size** | **1.1367 vs 1.1947 GiB/s (spread 1.051×)** | **2.229 vs 2.4316 GiB/s (spread 1.091×)** |
| command-construction residual, 1 vs 10 dispatches | 0.2775 → 0.4635 ms (**1.7×**) | 1.468 → 1.391 ms (**1.1×**) |
| Spearman(dispatches, record) | 0.2613 | 0.2533 |
| Spearman(upload bytes, record) | 0.3870 | 0.2863 |

A 10× change in dispatch count moves the record median 3.7–3.9×, which looks like a size effect —
but upload bytes also move 4.08×, and **the implied bandwidth is constant to within 1.05–1.09×**.
The part that genuinely is per-dispatch, command construction, moves only 1.1–1.7× for that same
10× change. **Recording is a byte-throughput cost wearing a size label.** Switch is optimising a
distribution over *bytes*, not over dispatches.

**Caveat that must travel with this:** only **2 distinct island sizes** exist in the current
33-island partition (1 and 10 dispatches). Two points do not establish a functional form. This is
a discrimination between two hypotheses, not a fitted model, and it is stated as such in
`record_scaling.size_confound`. A partition with more size diversity would strengthen it.

**The warmup decline is real** — `decline_verdict = REAL_WARMUP` on both devices, and it survives
the island-identity control, which is the control that matters. "Recording gets faster" could be
an artefact of *which* islands happen to run late. It is not:

* **100% of islands** (33 of 33, both devices) record faster on their last inference than on
  their first. Median last/first: 0.392 (NVIDIA), 0.106 (Intel).
* Per-cycle means fall across cycles that each contain **exactly the same 33 islands in the same
  order** — the cycle period is recovered from the record sequence's own repeat structure,
  deliberately independently of `subgraphs_live`.

But the *shape* differs, and this is the part Switch needs:

| | NVIDIA | Intel |
|---|---|---|
| warmup shape | `ONE_OFF_FIRST_INFERENCE` | `RAMP_OVER_12_INFERENCES` |
| cycles to steady | 2 | 12 |
| cycle means (ms) | 144.9, 41.0, 32.2, 53.0, 52.9, 51.9, … 53.4 | 113.8, 40.6, 28.1, 26.5, 22.3, 101.3, 158.5, 26.4, … 17.4, 13.1 |
| tail spread (last 3 cycles) | 1.0747× — **flattened** | 1.5712× — **NOT flattened** |
| steady-state record median | **50.53 ms** | 24.46 ms — **provisional** |

**On NVIDIA the steady state is established and 50.53 ms is the number to optimise.** On Intel it
is not: the last three cycles still span 1.57×, with two 100–160 ms excursions mid-run. The Intel
steady-state median is reported as provisional and must not be quoted as a steady state. The
answer to "is he optimising a constant or a distribution" is **a distribution, and on Intel one
whose tail has not been established**.

> ⛔ **§9.5 IS WITHDRAWN IN FULL — including the `REAL_WARMUP` verdict, which was wrong for a
> reason I should have seen.** The island-identity control rules out *island mix*: it proves the
> decline is not an artefact of which islands happen to run late. **It does not rule out a
> machine that got quieter as the run went on**, because that confound is also monotone in time
> and hits every island equally — which is precisely the signature the control was built to
> confirm. A control that cannot distinguish the hypothesis from its most likely rival is not a
> control for it.
>
> The two "100–160 ms excursions mid-run" on Intel, flagged above as unexplained, are now
> identified: `contention_signature` puts them at inferences 5 and 6, at 2.65× and 2.68× the run
> median, on a host clock whose device clock did not move proportionally. They are stalls, not
> warmup. And the Intel decline runs 1.42 → 0.51 across the run — a machine progressively
> freeing up looks exactly like that.
>
> Re-measurement on a verified-quiet machine is the only way to separate the two, and it is
> cheap to make decisive: a genuine warmup ramp is **reproducible on demand** and a load
> artefact is not. Two consecutive quiet runs that produce the same `cycles_to_steady` establish
> warmup; two that disagree establish load. **`SCALES_WITH_BYTES_NOT_DISPATCHES` is the one
> verdict in §9.5 with a defensible claim to survive** — it compares implied *bandwidth* between
> two island sizes measured within the same run, and contention inflates both terms of that
> ratio together — but it was computed from group medians that individual 9× excursions can skew,
> so it is downgraded to **provisional** rather than withdrawn.

### 9.6 The 52× trap, closed end to end

See §7.4 for the check. Verdicts on this run: **PASS and DECISIVE on Intel** (period 52.0833),
**VACUOUS on NVIDIA** (period 1.0), surfaced by `red_flags()` as "NOT DECISIVE" rather than as a
pass. `valid_bits_applied` green on both. Switch's conversion path is confirmed correct on the
only device that can falsify it.

**`gpu_containment` had to be rebuilt to say this.** Its first version attributed GPU spans to
submissions by timestamp containment and reported **14 violations** on Intel — on a build that had
just passed the integrality check decisively. The traces carry their own explanation:
`anchor_uncertainty_us` reaches **314618 µs (314 ms)** on Intel and 23907 µs on NVIDIA. A
device-lane span's `ts` is a GPU tick projected onto the host timeline through a single
calibration anchor; deciding which submission a 2 µs kernel belongs to by asking whose interval
contains it is a coin flip at that error. Attribution is now **ordinal** — each `vulkan.subgraph`
dispatches exactly `nodes` kernels and writes their queries in order, so the two lists walk in
lockstep and no clock is consulted. The 14 violations are gone and both devices are clean.

The precondition is asserted, not assumed: `gpu_span_accounting` requires
`sum(subgraph.nodes) == len(gpu_spans) == dispatches_executed` as **integer equality with no
tolerance** — 5457 on both devices. That equality is also the falsifier for a dispatch whose
query was never written, which raises nothing, leaves `compute_failures` at 0, and silently
removes time from the GPU column.

**Device-lane span *positions* are not evidence.** Durations are unaffected by anchor error;
placement is not. Nothing in this document infers overlap or serialisation from the GPU lane.

### 9.7 The per-island statistic is retired, not fixed

`per_island_ms_lower_bound` is gone: `null`, with
`per_island_ms_lower_bound_status: "RETIRED 2026-07-30"` and the reason carried alongside.

It was `(vulkan_median − cpu_median) ÷ island_count`, and it read as a per-island boundary cost.
The phase split shows the delta is dominated by per-inference host work — staging memcpy inside
command-buffer recording — that is **not proportional to island count**. The proof it was never
measuring what its name said: across the change that collapsed islands **321 → 33** and cut Intel
wall time **2954.6 → 807.2 ms**, the statistic went **up**, 8.5 → 16.6 ms. A per-island cost that
rises when islands fall by 10× and the run gets 3.7× faster is not a per-island cost.

The arithmetic survives, renamed and disclaimed, as `delta_over_island_count_ms` with
`delta_over_island_count_is_per_island_cost: False`. **Gating it behind a warning was rejected:
a number printed under a warning gets quoted without the warning.** It is replaced by the
directly measured `vulkan.record` median and by the `phases.record_scaling` breakdown, which
measure the thing the retired statistic was reached for.

### 9.8 No ratio is emitted from an unsteady baseline

`stats.drift` used to print a warning beside the ratio. It now **refuses the ratio**.

| | NVIDIA | Intel |
|---|---|---|
| vulkan median | 3023.472 ms (rsd 6.8%) | 3313.618 ms (rsd 22.3%, **noisy**) |
| cpu median | 3698.736 ms (rsd 36.5%, **noisy**) | 1553.654 ms (rsd 28.7%, **noisy**) |
| cpu drift | **NOT steady** — 2731.4 → 3478.7 ms (1.27×), 63% of steps one-way | steady |
| `vulkan_over_cpu_ratio` | **`null` — REFUSED** | 2.13× |
| run-to-run vulkan spread (3 repeats) | 1.30× | 1.11× |
| run-to-run cpu spread (3 repeats) | **2.14×** | 1.32× |

Refusal text, carried in the record so the absence is explained where the number would have been:

> *no vulkan/cpu ratio is reported: the cpu sample is not steady […]. A ratio inherits the
> instability of its worse operand, and a ratio printed under a warning is quoted without the
> warning. The two absolutes above stand on their own and are unaffected.*

The refusal also fires when steadiness is **untestable** (fewer than 4 samples). Untested is not
steady. The Vulkan absolutes are unaffected and are still reported — the refusal is scoped to the
derived quantity, not used as an excuse to report nothing.

`baseline_disagreement` additionally fires on this run: the two workers' CPU baselines differ by
2.4× (3698.7 vs 1553.7 ms) for the same artifact on the same host. Both were noisy; the NVIDIA
worker's was also drifting. **This is a reason to distrust every CPU-relative figure in this
run**, and the 2.13× Intel ratio should be read with it. It is not a device difference — the CPU
baseline does not depend on which GPU the other pass used.

### 9.9 `largest_island_flops` — the third slot is plumbed and empty. It is not 0.

§10.0's metric of record is a triple. The third slot is still unfilled, but the failure mode
changed: the EP now **emits** the key.

`vulkan.getcapability` carries `largest_island_flops: 0` — and, on the same event,
`island_count: 0`, `concentration: 0.0`, `boundary_bytes_per_inference: 0`, while the EP's own
`subgraphs_live` counter reports **33**. `PartitionStats` is constructed and never populated:
`CoverageReport` computes no FLOP estimate.

**"Not computed" is now wearing the appearance of "zero FLOPs", which is strictly worse than the
key being absent.** `phases.partition_stats` reports `third_slot_state: UNPOPULATED` and
`metric_of_record.largest_island_flops` is `null`, never 0, with the emitted value preserved
separately as `largest_island_flops_emitted_value: 0`. **The third slot may not be quoted.**

Ownership: `partition.rs` and `CoverageReport` are **Mouse's**. Exactly one owner is needed and
that is Mouse; the harness already consumes the key and will report a real value the moment one
is emitted, with no further change here. Routed via `.squad/decisions/inbox/`.

### 9.10 Instruments — what goes red if each number is false

Per R9: *confidence scales with agreeing instruments; evidence scales only with falsifying ones.*

| claim | instrument that goes red |
|---|---|
| the EP actually ran | `EP_NAME in session.get_providers()` **and** non-zero `claimed_nodes`, asserted before timing. Two fabricated speedups on this project (1.70× through an EP that could not load, 1.45× through one that declined everything) are why both halves exist. |
| every island executed on every inference | `dispatch_accounting`: `compute_calls == islands × inferences`, **integer equality, no tolerance**. Caught a subgraph never invoked — which raises nothing and leaves `compute_failures` at 0. |
| every dispatch produced GPU time | `gpu_span_accounting`: `sum(subgraph.nodes) == len(gpu_spans) == dispatches_executed`, integer equality. 5457 on both. |
| the row names the device that ran | `devices.device_identity_check` — trace's own `timestampPeriod`/`validBits` vs the label. Caught the entire table naming the wrong GPU. |
| the phase split sums correctly | `phase_containment` — every phase span lies inside its `vulkan.subgraph` span; `unattributed_in_compute_ms` reported, never folded away. |
| GPU time is not over-scaled | `gpu_containment` — per-submission GPU busy ≤ `submit + fence_wait`, **ordinal attribution**, immune to the 314 ms anchor error. |
| the 52× conversion is applied | `timestamp_conversion_integrality` — `gpu_ns ÷ period` must be a whole integer. **Decisive only where period ≠ 1.0**; reports `VACUOUS`, never "pass", on NVIDIA and lavapipe. `bench/timestamp_audit.py` exits non-zero when no local device can falsify it. |
| valid bits are masked | `valid_bits_applied` — green on both. |
| the trace describes the run that was timed | `trace_matches_counters` — trace span counts vs the EP's own counter file. |
| the ratio describes a steady state | `stats.drift` → `ratio_refusal`. **Refuses**, does not warn. |
| the CPU baseline is trustworthy | `baseline_disagreement` — **fired on this run at 2.4×**. |
| tracing did not distort the measurement | `tracing_overhead_ratio` from a separate untraced timed pass — 1.0207× / 0.8659×, measured not assumed. |
| the number is about the EP and not about staging | `memory_configuration` — reports `staging-bound` and forbids quoting the result as "what the Vulkan EP does". |
| the two devices are not compared | `bench/compare.py` exits **2** on a cross-device comparison. |

**Not falsified here, and named so:** whether the Intel `tracing_overhead_ratio` of 0.8659× is
machine drift rather than negative overhead (it must be — but nothing in the harness rules the
alternative out); and the Intel steady-state record median, which is provisional because the tail
has not flattened.

---

## 10. Every number in this document was taken on an unmeasured machine (2026-07-30 evening)

The coordinator measured the same device, the same build and the same test twice and got answers
9.5× apart:

```
device 0, machine quiet          vulkan.record total    19 460 ms
device 0, six agents compiling   vulkan.record total   184 356 ms      9.5x
```

Nothing in `bench/` could see that. Every guard in this harness was pointed at the *program* —
is the EP loaded (`refuse_if_ep_absent`), did every island actually run (`dispatch_accounting`),
did the output match (`model_output_equivalence`), is the sample drifting (`stats.drift`). None
of them can see a number that is uniformly wrong because the machine was busy the whole time.

`stats.drift` is the near miss and it is worth being precise about why it does not cover this.
Drift detects a baseline **moving**. A machine that is loaded for the entire run produces a
baseline that is *stable* and *wrong* — the best-looking possible output, and the one this
harness would have blessed. That is the same defect class as the two fabricated speedups in
`test_plausible_but_wrong.py`: a plausible number, from a working harness, meaning something
other than what it says.

### 10.0 The measurement is now gated on the machine, not only on the model

`metric_of_record` carries a second gate beside `model_output_equivalence`:

```
machine_quiescence ∈ { QUIET, CONTENDED, UNMEASURED }      default UNMEASURED
```

**No performance number may be quoted beside a non-`QUIET` verdict.** It is a refusal, not a
warning: `phi35.py` prints the refusal *in place of* the medians, the delta and the ratio.
A number printed under a warning gets quoted without the warning — that rule cost this project
`per_island_ms_lower_bound` and the unsteady-baseline ratio, and it applies here unchanged.

What is still printed from a contended run is everything contention cannot corrupt: island
counts, `dispatch_accounting`, `gpu_span_accounting`, timestamp integrality, the valid-bit mask,
device identity. Those are counts and integer identities. Withholding them too would discard the
falsifiers that cost the most to collect.

### 10.1 Two instruments, and they fail differently

Per **R9** (DESIGN.md §10.0.1) the point is not to have two instruments that agree; it is to have
two that could each go red on their own. These share no inputs — one reads a system idle counter,
the other reads a Vulkan query pool.

**(1) Out-of-band: the load survey — `bench/contention.py`.**
A background thread samples the system-wide idle counter once a second for the whole measurement.
Busy CPU-seconds are `cores × wall − idle`; this process's own tree is subtracted; what is left is
`foreign_busy_cores`, the average number of cores other processes kept busy while we measured.
It needs no reference and no calibration, and because it reads a system counter rather than
walking the process table it **cannot miss a short-lived process** — a `rustc` that starts and
exits between two samples is still fully counted.

*Sampling during the run is the point.* A machine quiet at the start and busy at minute 20 is the
failure case, and a before-and-after check would miss it entirely.

Its own falsifiers, because an instrument with no failure mode is an assumption:

| falsifier | goes red if |
|---|---|
| `idle_accounting` | `(busy + idle)` fails to reconstruct `cores × wall` within 5% — the idle counter is not what we think it is |
| `own_cpu_not_exceeding_busy` | our own tree used more CPU than the machine reports as busy — the tree walk is double-counting |
| `monitor_not_perturbing` | the sampler's own CPU exceeds 0.05 cores — the instrument has become part of the load it measures |

Any of them red yields `UNMEASURED`, never `QUIET`.

**(2) Out-of-band: the tachometer — `occupancy_probe`.**
A fixed quantity of single-threaded integer work, timed, best-of-7. It measures the thing that
actually matters — not "is the machine busy" but "can this process get a core" — and it is
sensitive to causes the survey cannot see: thermal throttling, a co-tenant VM, a power-plan
change, CPU affinity. It is *relative*, so it needs a quiet reference persisted in
`bench/results/machine-baseline.json`; with no reference it reports **`VACUOUS`**, never "pass",
the same discipline `timestamp_audit` uses on a device whose `timestampPeriod` is 1.0.

A quiet survey with a slow tachometer resolves to `CONTENDED`. Two instruments disagreeing is not
a tie to be broken in favour of the convenient answer.

**(3) In-band: the trace signature — `phases.contention_signature`.**
This is the one that matters most, because it works on traces captured before any of the above
existed — which is every stored number in this document.

A Vulkan trace already contains two clocks with completely different exposure to host load:

* **host phase spans** (`record`, `submit`, `fence_wait`) are wall-clock intervals on a thread
  that must be scheduled to make progress. Take the core away and they stretch.
* **GPU spans** are differences of the device's own timestamp counter. The GPU does not care how
  many copies of `rustc` are running. Take the core away and they do not move.

**The trace carries its own control.** Submissions repeat in a fixed cycle — island slot *s* on
inference *c* does exactly the same work as island slot *s* on inference *c+1*. So for each slot,
compare the spread of its host record time across repetitions against the spread of *its own* GPU
busy time across the same repetitions:

| host spread on a slot | GPU spread on the same slot | that slot is |
|---|---|---|
| ≥ 2× | < 1.25× | a **host-side stall** |
| ≥ 2× | ≥ 1.25× | doing different work |
| < 2× | anything | steady |

The first cycle is discarded, because a genuine warmup ramp produces host-side excursions too,
for a legitimate reason.

**A defect found in this instrument while building it, because the first version got a real trace
wrong.** The first version normalised each record span by its slot median and then took the
*median across slots* per inference. That is the right shape for a run that is uniformly slow and
exactly wrong for the real case: it reported the RTX 4060 trace `STABLE` while slot 0 was
recording 12.48 / 70.19 / 12.59 ms and slot 5 was recording 301 / 1156 / 374 ms. Stalls hit some
islands and not others, so a median across slots averages away the thing the statistic exists to
find. The statistic is now per-slot; the per-inference figure is kept as a secondary detector for
uniform inflation and is labelled as such.

**Where this control is weak, stated rather than discovered later.** On an **integrated** GPU the
device shares its power and thermal budget with the CPU cores, so heavy CPU load can slow the GPU
itself through DVFS. That cannot manufacture a false `HOST_SIDE_EXCURSIONS` — it pushes the other
way — but it can manufacture a false `WORKLOAD_VARIATION`, so on an integrated part that verdict
means "not established", never "quiet", and it is not marked quotable. The method also requires
the islands to have differing dispatch counts, because the cycle period is *recovered* from the
dispatch-count sequence rather than assumed; an artifact whose islands were all the same size
would report `UNTESTABLE`, which is the honest answer and not a period guessed from
`subgraphs_live`.

### 10.2 Verdict on the stored numbers: §9 is withdrawn

Both §9 traces were re-analysed with the in-band signature. No re-run was needed — traces are
cheap to re-analyse and expensive to collect, which is the second time today that has paid.

| | NVIDIA RTX 4060 (`ep.device_index 0`) | Intel Iris Xe (`ep.device_index 1`) |
|---|---|---|
| verdict | **`HOST_SIDE_EXCURSIONS`** | **`NOT_STEADY`** |
| controllable island slots | 33 | 33 |
| slots stalled host-only | **20 of 33 (61%)** | 0 |
| slots where GPU moved too | 2 | 33 |
| worst slot | slot 31, host **9.39×** vs its own GPU **1.0024×** | — |
| whole-inference spread | 1.40× | **5.25×** (min 0.51, max 2.68 of the run median) |
| quotable | **no** | **no** |

Worst offenders on the 4060, host milliseconds across three repetitions of *identical* work, with
that slot's own GPU time beside it:

```
slot 31 (10 dispatches)   267.51   451.03    48.05     GPU spread 1.0024x
slot 14 (10 dispatches)   190.93   501.56  1747.51     GPU spread 1.0019x
slot 15 (10 dispatches)   271.61   557.05  1646.80     GPU spread 1.0035x
slot  0 ( 1 dispatch )     12.48    70.19    12.59     GPU spread 1.0003x
```

A 9× swing in host time for work the device's own clock says was constant to three decimal
places. That is not a slower measurement of the same thing; it is a measurement of a different
machine.

Intel reaches `NOT_STEADY` rather than `HOST_SIDE_EXCURSIONS` because its GPU time moved as well —
and on an integrated part that is not an exoneration (§10.1). Either way the run was not steady.

**This was foreseeable from §9's own output and I did not act on it.** Every one of these was
already printed: cpu rsd 36.5%, run-to-run cpu spread 2.14×, `baseline_disagreement` firing at
2.4×, an Intel tail that would not flatten, two unexplained 101/158 ms cycle excursions, and a
`tracing_overhead_ratio` of 0.8659× — a traced pass that ran *faster* than the untraced one,
which is impossible from tracing and obvious from load. Six symptoms of one cause, each recorded
and each rationalised individually. The instrument did not tell me anything the data had not
already; it made the alternative explanation impossible to keep ignoring.

### 10.3 Live corroboration: this machine varies 2.65× in single-threaded throughput

While writing this section, a background watcher sampled the machine every 20 seconds. Over the
whole window foreign load ran **7.6–18.8 of 20 cores** (`Code.exe`, `copilot.exe`, `rustc.exe`,
`python.exe`, `MsMpEng.exe`), and the tachometer — the same fixed integer loop each time, nothing
to do with Vulkan, ORT or a trace — took:

```
22:09:44   86.46 ms      22:11:27  102.22 ms      22:17:45   59.96 ms   <- fastest seen
22:10:35   76.62 ms      22:12:35  159.19 ms      22:20:02   63.27 ms
                         22:13:36   83.53 ms      22:22:17   62.36 ms
```

**2.65× on identical work (59.96 → 159.19 ms), and 2.08× of that inside five minutes.** A
benchmark that starts at 22:10 and a benchmark that starts at 22:12 are not measuring the same
computer.

The spread widened as the watcher ran: an earlier draft of this section said 2.08×, from a window
that had not yet seen the quiet end. That is worth stating rather than silently correcting —
**the spread of a machine is a lower bound until sampling stops**, so any single "how noisy is this
box" figure is provisional in the direction of being too small.

### 10.3.1 A defect in my own harness: the evidence did not outlive the next run

`_run_trace_pass` wrote its trace to a deterministic scratch path, `phi35_trace_dev{N}.trace.json`.
So the next run on the same device silently overwrote it. This is not hypothetical — while
stamping the retroactive verdicts of §10.2 into the result artifact I found that **my own
three-iteration smoke test had already destroyed §9's Intel trace**, and that device's verdict had
to be transcribed by hand from the analysis output rather than re-derived from the evidence.

That matters more here than it would elsewhere, because on this project **re-analysing a stored
trace has repeatedly been worth more than a fresh run**: both defects found in the per-slot
signature (§10.1) were caught by re-reading traces already on disk, at zero measurement cost, on a
machine too loud to re-measure on. A trace is ~0.5 MB and a two-device run is forty minutes.

Fixed: when `--out` is given, every pass's trace is copied to `results/traces/{artifact}-dev{N}.trace.json`
and the path is recorded in the artifact as `phase_pass.trace_preserved_at`. Falsifier:
`test_preserved_trace_survives_a_later_run_on_the_same_device` rewrites the scratch file after the
copy and asserts the preserved one still holds the original data. §9's surviving NVIDIA trace has
been moved under that scheme by hand.

This is the same defect class as the rest of §10. A number whose evidence cannot be re-examined
later is a number that has to be believed rather than checked.

### 10.4 The Intel-versus-4060 inversion: I cannot adjudicate it, and here is exactly why

> **Updated 2026-07-30 — see §11.1.** The device labels in the figures below are inverted: the
> 807.2 ms figure is the **NVIDIA** part and 1156.0 ms is the **Intel** part. That does not change
> this section's conclusion, and it adds a third independent reason for it — after correct
> labelling the coordinator's run and §6 **disagree about which device is faster**. The ordering
> is not merely unquotable, it does not reproduce.

The coordinator has been quoting Intel 807.2 ms against the 4060's 1156.0 ms — the integrated part
beating the discrete one — and asked whether it survives. **I am not able to confirm or retract
it, for two independent reasons, and the honest answer is to name both rather than produce a
verdict.**

1. **It is a cross-device comparison, which this project refuses structurally.** `bench/compare.py`
   exits 2 on one. Different transfer class, different shared-memory budget, different
   `timestampPeriod`. That refusal does not become optional because the result is interesting.
2. **Neither figure has a quiescence record**, and figures of that kind now fail the gate by
   default (`UNMEASURED`).

What I *can* report, because it is a statement about reproducibility rather than about which
device is faster: **the ordering is not stable across runs.** In §9's run the 4060 came out ahead
of the Intel part on median wall time; in the coordinator's run the Intel part came out ahead.
Both runs fail the quiescence gate. A quantity that reverses its sign between two runs of the same
build, both of them taken on an unmeasured machine, is not a finding about hardware.

**Recommendation to the coordinator: stop quoting the inversion.** Not because it has been refuted
— it has not — but because it has never been established, and it is currently being carried as
though it had been.

### 10.5 Re-measurement is blocked on a quiet machine, and that is now provable

A full two-device run is ~40 minutes and produces nothing quotable if anything else is compiling
through it. `phi35.py --require-quiet` therefore refuses to *start* on a contended machine —
failing in fifteen seconds is cheaper than failing in forty minutes:

```powershell
python bench\contention.py --seconds 20          # exit 0 = quiet, 2 = contended
python bench\phi35.py --require-quiet --iters 20 --warmup 10 --repeats 3 --trace-iters 6
```

As of 2026-07-30 22:22 the machine had **not been quiet at any sample** during this session —
every one of the watcher's observations returned `CONTENDED`, with `loud=100%` (i.e. every
sub-sample above the loud threshold, not merely the window's average). The coordinator's offer to
hold the other agents idle is the unblocking step; the guard now makes the window **provable**
rather than assumed, and stamps the proof into the result artifact.

Note for Switch: this applies to a before/after on the recording fix with more force than to
anything else here. A 9.5× environmental swing will swamp a 3× improvement in either direction —
it can make a real win look like a regression. The before and the after must both carry a `QUIET`
verdict, and they now can.

### 10.6 The environment record

`bench/environment.py::capture()` now carries a `contention` field, for the same reason it carries
the driver version and the CPU model: it changes the answer. A stored number whose environment
record omits the machine's load cannot be re-checked later, because the single largest influence
on it left no trace. `capture(load_seconds=N)` takes a spot-check; the authoritative record is the
per-pass `machine_quiescence` in the result artifact, which covers the whole measurement rather
than one moment of it.

### 10.7 Instruments — what goes red if each claim in §10 is false

| claim | instrument that goes red |
|---|---|
| the machine was quiet while this number was measured | `contention.Monitor` → `quiescence()`; **refuses**, does not warn. Exercised in `test_contention.py` on synthetic quiet/loud/unavailable windows. |
| …and the load survey's own arithmetic is sound | `idle_accounting`, `own_cpu_not_exceeding_busy` — either red ⇒ `UNMEASURED`, never `QUIET` |
| …and the guard is not itself the load | `monitor_not_perturbing`, measured in **thread CPU time**. Wall time was the first attempt and was confounded with the very contention it guards against: on a saturated machine a 5 ms sample takes 100 ms of wall clock. A falsifier that fires on the condition it must be independent of is not a control. |
| this process could get a core at full speed | `occupancy_probe` against the persisted quiet reference; `VACUOUS` with no reference, never "pass" |
| a *stored* trace was taken on a quiet machine | `phases.contention_signature` — per-slot host spread against that slot's own GPU spread |
| …and that verdict is not just the workload varying | `gpu_range_control` — the same slot's GPU time. If it moved too, the host is exonerated (weakly, on integrated parts) |
| …and the statistic can see a stall that hits only some islands | `test_host_stall_on_one_slot_is_caught_even_when_others_are_clean` — the exact case the first implementation got wrong on real data |
| …and a legitimate warmup ramp is not reported as contention | first cycle discarded; `test_first_inference_is_excluded_as_warmup` |
| a run with too few repetitions is not called quiet | `UNDERPOWERED` / `UNTESTABLE` verdicts. Untested is not quiet, exactly as untested is not steady |
| the two guards are independent | they share no input: a system idle counter and a `VkQueryPool`. On the smoke run they agreed (`CONTENDED` + `HOST_SIDE_EXCURSIONS`) without being able to influence each other |
| a published verdict can still be re-derived from its evidence months later | `test_preserved_trace_survives_a_later_run_on_the_same_device` — rewrites the scratch path after the copy and asserts the preserved trace is unchanged (§10.3.1) |

**Not falsified here, and named so:** the `QUIET_BUSY_CORES = 0.5` threshold is a judgement, not
a measurement — it is stated in `contention.py` beside the constant rather than buried in a
comparison, and no run has yet been taken at the boundary to calibrate it. And
`contention_signature` establishes that the host stalled; it **does not name the cause**. Another
process is the obvious candidate, but a page-fault storm, a driver allocation or a thermal event
would look identical from inside the trace. What it establishes is the thing that governs whether
a stored number may be quoted: the run was not in a steady state.

---

## 11. Two corrections from the coordinator, checked against my own artifacts (2026-07-30 evening)

The coordinator issued two corrections that invalidate numbers broadcast to the whole team: the
device labels are inverted, and the "68% command-buffer recording" is upload. **Both are correct
about the world.** One of them is *not* correct about `bench/`, and the difference matters, because
applying it blindly would have introduced the error it was meant to remove.

Each is recorded below with the check I ran, not the instruction I received. That is the standing
rule on this project and it now cuts toward the coordinator as often as away.

### 11.1 The device labels — right about the world, wrong about `bench/results/`

**The correction.** `enumerate_capable_devices()` sorts best-first (`Reverse(kind.score())`,
discrete before integrated) and `select_device` indexes *that sorted list*, while probes print
unsorted `vkEnumeratePhysicalDevices` order. So:

> `ONNXRUNTIME_EP_VULKAN_DEVICE=0` (and `phi35.py --device 0`) is the **NVIDIA RTX 4060**.
> `ONNXRUNTIME_EP_VULKAN_DEVICE=1` is the **Intel Iris Xe** — which remains the conformance oracle.

**Check 1 — the source.** `instance.rs:536` is `result.sort_by_key(|d| Reverse(d.info.kind.score()))`,
and `select_device` indexes the result. Confirmed.

**Check 2 — was it true when the numbers were taken?** A correction to a *label* is worthless
without a date: if best-first sorting post-dated a run, that run's labels would have been right.
`git log -S'Reverse(d.info.kind.score())'` puts it at **`bb885d9`, 2026-07-29** — before every
measurement in this document. So the mapping held throughout.

**Check 3 — what did my artifacts actually record?** This is where the correction stops applying.
`bench/results/phi35-2026-07-30-phases.json` already carries, per result row:

```json
"device_identity": {
  "ep_device_index": 0,
  "assumed_from_ep_order": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "observed_from_trace":  "NVIDIA GeForce RTX 4060 Laptop GPU",
  "reason": "timestamp fingerprint period=1.0 bits=64 matches exactly one device",
  "verdict": "MATCH", "name_may_be_quoted": true }
```

and the mirror for index 1 at `period=52.0833 bits=36`. **`devices.device_identity_check` exists
precisely for this trap and it is green on every row it covers.** The rows are not labelled from
the index at all — they are labelled from the *trace's own* timestamp fingerprint, and the check
goes red if that disagrees with the ep-order name.

So: **`bench/results/` needed no re-labelling, and re-labelling it would have inverted correct
rows.** I re-labelled `docs/PERF.md` §6 instead, which is prose written before the check existed
and which was wrong.

**Why the fingerprint is the right label and the index is not.** An index is an assertion about a
convention; a `timestampPeriod` of 52.0833 with 36 valid bits is a property of the silicon that
appears in the trace the number came from. The label therefore travels *with the evidence* rather
than beside it. Limit, stated: NVIDIA and lavapipe both report 1.0/64, so the fingerprint
distinguishes Intel from either but cannot distinguish those two from each other — the check
reports that as a non-decisive case rather than a pass, exactly as the 52× audit does.

**What did NOT survive.** §6.4's premise. Published as "the Intel part is 2× more expensive per
island *while having no bus to cross*", it is really "the discrete part is 2× more expensive" —
which is what a staging round trip predicts and is not surprising at all. **That surprise is what
produced my fixed-per-submission hypothesis.** The hypothesis was worth testing and the instrument
built for it closed the 52× trap and now measures upload, so the chain paid for itself — but its
first link was an artifact.

**And it does not restore the inversion story either.** Relabelling cuts opposite ways in the two
runs:

| run | as published | after relabel |
|---|---|---|
| coordinator's | "Intel 807.2 beats NVIDIA 1156.0" | NVIDIA 807.2 beats Intel 1156.0 |
| §6 (this doc) | NVIDIA 1465.9 beats Intel 2790.7 | **Intel 1465.9 beats NVIDIA 2790.7** |

**The two runs disagree about which device is faster, after correct labelling, on the same build.**
So the inversion neither survives nor dissolves — it was never measured. Both runs also fail the
§10 quiescence gate. My §10.4 recommendation is unchanged and now has a second independent reason:
stop quoting the ordering. It is a cross-device comparison (`compare.py` exits 2), taken on
unmeasured machines, that does not reproduce.

### 11.2 The 68% was upload, and my own trace says so

**The correction.** `Phase::Record` opens before `vkBeginCommandBuffer` and closes after
`vkEndCommandBuffer`, and the host staging memcpy runs **inside** that window, reporting through
`record_transfer` into a `ph:"C"` counter that deliberately emits no span. An aggregation over
`ph:"X"` spans is therefore *structurally incapable* of seeing it and silently folds it into
`record`.

**Check — my own stored NVIDIA trace, re-analysed at zero measurement cost:**

```
vulkan.record          54389.02 ms   <- NOT A LEAF
  = 53635.57 ms upload (98.6%) + 753.46 ms actual command construction (1.2% of Compute)
host upload memcpy     7990.4 MiB over 4 inferences = 1997.6 MiB / inference
```

**1997.6 MiB per inference — the same figure Tank obtained from a different instrument.** Per R9
that is agreement, so it raises confidence, not evidence; the thing that would have refuted it is
that the counters and the phase spans are independent records and could have disagreed. They do
not, to four significant figures.

Real command-buffer construction is **1.2% of time in Compute**, consistent with Tank's 1–3% of
wall. The EP re-uploads the entire weight set every inference.

#### 11.2.1 The invariant is the byte count, not the share — and the share does not generalise

Tank reported upload as "95.8–98.4% of the `record` phase in every cell, both devices, both
settings". **That share does not reproduce on my Intel trace, and it should not be expected to.**

| | upload per inference | upload bandwidth | upload share of `record` |
|---|---|---|---|
| NVIDIA RTX 4060 (`--device 0`, discrete) | **1997.6 MiB** | 0.1455 GiB/s | 98.6% |
| Intel Iris Xe (`--device 1`, UMA) | **1997.6 MiB** | 0.4454 GiB/s | 59.9% |

**The byte count is identical to five significant figures across two devices, two traces and two
run lengths (4 and 8 inferences).** That is the finding, and it is the one that transfers: it is a
*count*, so by §10's own rule it survives a contended run, and it independently reproduces Tank's
1997.6 MiB from a different instrument on different silicon.

The *share* is not an invariant and must not be quoted as one. It is
`bytes ÷ bandwidth ÷ record_total`, and the bandwidth is a property of the transfer class — the
UMA part is copying within one memory pool, the discrete part is crossing PCIe. A ratio whose
denominator differs by transfer class will differ by transfer class. **This is also why Tank's own
per-device transfer ceilings differ so widely (~94.8% of wall on the discrete part, ~44.0% on the
UMA part) — my numbers agree with those ceilings and not with the "every cell" share claim.**

Two cautions on the table above, both mine to state:

* The two rates are listed **per device and are not ranked**. `bench/compare.py` refuses
  cross-device performance comparison (exit 2) and that refusal is not suspended because the
  quantity is a bandwidth. The reason both appear here is to show that the *share* is
  class-dependent, which is a statement about the arithmetic, not about which part is better.
* The Intel row comes from a **`CONTENDED`** run, so its 59.9% share is provisional in the
  direction of being too low — contention inflates the command-construction residual in the
  denominator. Its byte count is not provisional. That asymmetry is exactly §10's leaf/duration
  distinction, applied here.

**What Switch should take from this:** the target is the 1997.6 MiB, on both devices. The share
tells you how much of `record` disappears when you fix it, and that answer is device-specific.

**This also re-reads §10's contention finding rather than overturning it.** 9.5× inflation of
`record` under CPU contention was correctly diagnosed as *host CPU work* — it just is not the work
the span is named after. A ~2 GB memcpy is exactly the kind of host work that degrades when six
processes are compiling. The contention guard stands; its explanation improves.

**What I changed so the misreading is not available.** `phases.py` now knows that phases form a
tree:

* `PHASE_CHILDREN` records that `record` contains `upload`; `is_leaf_phase()` is the predicate.
* `host_phase_totals()` marks every entry `is_leaf`, and for a non-leaf emits `child_ms`,
  `leaf_ms` and a caveat. With no transfer data `leaf_ms` is **`None`, not the total** —
  the leaf cost is UNKNOWN, and unknown is not equal to the parent.
* The share table renames the parent's share to `record_INCLUDING_upload` and adds
  `record_excl_upload`, so the honest number is the one closest to hand.
* `describe()` prints `<- NOT A LEAF` on the same line as the total, and the split on the next.
* **`phase_leaf_accounting`** is the falsifier: `UNRESOLVED` when a non-leaf phase's children
  cannot be subtracted, and it says so *more loudly* when that phase is also the largest in the
  run — because that is precisely when "record is the bottleneck" is the natural misreading. It
  is `VACUOUS`, never "pass", on a trace with no non-leaf phase.

One deliberate restraint: `host_phase_totals` does **not** derive the upload total itself. It
consumes `record_scaling`'s interval-containment attribution. A second, weaker attribution of the
same quantity — by transfer direction alone — would have been easy and would have created two
numbers that could disagree. One number that can be checked beats two that agree by luck.

### 11.3 The general rule this produced

Both corrections, and three of the five defects catalogued in this document, are the same shape:
**an artifact that is wired, produces output, and whose output's name misdescribes its content.**
`record` is a real span with a real duration and a caveat string that asserts it is command-buffer
recording. `index` is a real integer that indexes a different list than its reader assumes.
Neither is missing, neither raises, and a census that checks whether a mechanism *exists* passes
both.

> **A name is an assertion about content, and it must have a falsifier like any other assertion.**

Concretely, the two falsifiers this section adds are the pattern: `device_identity_check` tests a
label against evidence carried in the same artifact, and `phase_leaf_accounting` tests whether a
duration measures what its name says. Neither can be satisfied by the mechanism merely existing.

### 11.4 Instruments — what goes red if each claim in §11 is false

| claim | instrument that goes red |
|---|---|
| the device named on a results row is the device that ran it | `devices.device_identity_check` — timestamp fingerprint from the row's own trace vs the ep-order name; `MATCH`/`MISMATCH`, and `name_may_be_quoted` gates the name |
| …and it is decisive on this pair | period 52.0833/36 bits vs 1.0/64. **Non-decisive** between NVIDIA and lavapipe, reported as a gap, not a pass |
| the label mapping held when the numbers were taken | `git log -S'Reverse(d.info.kind.score())'` → `bb885d9`, 2026-07-29, before every run here |
| a phase total measures the activity its name names | `phase_leaf_accounting` — `UNRESOLVED` unless the children are subtracted; louder when the non-leaf phase is the largest |
| …and the leaf residual is not silently the total | `leaf_ms is None` + `test_unsubtractable_children_make_the_total_unquotable_not_approximate` |
| …and a reader quoting one line cannot quote the wrong one | `test_describe_never_prints_records_share_without_the_marker` |
| upload is ~98.6% of `record` | two independent records in one trace — `ph:"C"` transfer counters and `ph:"X"` phase spans — that could have disagreed and do not; and Tank's separate probe, agreeing at 1997.6 MiB/inference |
| upload is **1997.6 MiB/inference on both devices** | two traces, two devices, two run lengths (4 and 8 inferences), agreeing to 5 s.f.; it is a byte *count*, so contention cannot move it. If the EP ever stops re-uploading, this number changes and `alloc_device_authoritative_spans` moves off 0 |
| …but the *share* of `record` is not an invariant | 98.6% discrete vs 59.9% UMA, in my own traces. Any restatement of "upload is ~98% of record" as device-independent is refuted by the Intel row |
| the contention finding survives the re-attribution | it is unaffected: `contention_signature` compares host spread against GPU spread per slot and never depended on what the host time was spent *on* |

**Not falsified, and named so:** §6's absolute numbers are re-labelled, not re-measured, and they
remain under §10's withdrawal for contention. Re-labelling a withdrawn number does not un-withdraw
it. The ordering question in §11.1 stays open until two quiet-machine runs agree.


---

## 12. Admissibility: whether a stored number may be quoted at all (2026-07-30 night)

**Standing directive from Justin, this session:** 「要确保我们性能是非常高 一致向高性能推进」 — performance
is now a continuous, first-class goal. That raises the stakes on everything in this document,
because a permanent push for speed is a permanent incentive to quote the most favourable number
available. This section is the counterweight I own: **the gate that decides whether a performance
claim is admissible at all.**

It does not slow the push down. A slow honest number is admissible. Admissibility is about
provenance, not speed.

### 12.1 A double-count that a merge would have introduced, silently

`origin/main` landed three new spans in `trace.rs`: `vulkan.desc_alloc`, `vulkan.pipeline_lookup`
and `vulkan.cmd_upload`. They are real `ph:"X"` spans and they are emitted **inside**
`vulkan.record`.

Every phase total in `bench/phases.py` was, until this session, a sum over sibling spans. Summing
these three alongside `record` counts the same microseconds twice: the host total inflates by
roughly 2× and **every share derived from it** moves with it. Nothing in the trace raises; nothing
in the test suite failed; the resulting table would have looked entirely ordinary.

This is the same defect class as `record`-is-68%, one level down. §11 established that a phase
whose children are invisible to the aggregation must not be reported as a leaf. This is the
converse: **a child that becomes visible must be removed from the sibling sum on the same day it
appears.**

The guard is `phases.phase_nesting`. It derives parenthood **from timestamp containment**, and
separately reads the `host/sub-record:` prefix the EP puts on the child's caveat, and goes red when
the two disagree **in either direction**:

| direction | what it catches |
|---|---|
| declared nested, not contained | the caveat is wrong, or a span escaped its parent |
| contained, not declared | **a new sub-phase landed and every sibling sum since is double-counting** |

The second row is the operational one, and it is why containment is the primary source. R11: *a
measurement's name is not its definition.* Timestamps do not know what a phase is called; a rename
in `trace.rs` cannot disable this check.

`phases.sibling_phases` then takes the **union** of the static `SUB_RECORD_PHASES` table and the
trace's own declaration. The union rather than either alone, and the asymmetry is deliberate: the
table catches a child whose caveat is missing, the declaration catches a child added after the
table was written. Each source covers the other's blind spot.

Nested spans are still reported — under `nested_phases_ms`, as a breakdown *of* `record` — and are
excluded from every share.

### 12.2 Two upload accountings, and neither one alone is evidence

`cmd_upload` measures the host staging memcpy as a wall-clock interval. The transfer counters
measure the same memcpy as bytes, inverted through a rate into a duration. Adding both
double-counts it; picking one silently discards a free cross-check.

`phases.upload_accounting` prefers the span, uses the counters to corroborate it, and goes red when
they disagree by more than 25%. When only one is present it reports **`VACUOUS`, explicitly not a
pass** — one instrument cannot falsify itself.

The precedent is Tank's: `alloc_device_upload_bytes` read **0** on a run where `cmd_upload` was
**15.2 seconds**. Two accountings of the same upload, one of them blind, and nothing went red.

### 12.3 R11 — and why our decomposition check could not fire

The rule was paid for. The coordinator's phase table closed at **99.0%**:

```
68.3 + 16.3 + 14.1 + 0.3 = 99.0%
```

and it was wrong, because a 2 GB memcpy was missing — and it was missing *inside a row*, not
between them. Both sides of that identity were sums over the same tracer's spans. The parts and
the whole moved together. **An identity whose two sides come from the same source is a falsifier
that cannot fire, no matter how badly the rows are named.**

`phases.decomposition_identity` therefore reports two closures and labels them differently:

| closure | whole comes from | strength |
|---|---|---|
| `internal_closure` | `sum(vulkan.subgraph)` — the same tracer | **`WEAK`**, with `why_weak` naming the 99.0% failure inline |
| `external_closure` | the harness's own `perf_counter` | **`CAN FIRE`** |

The weakness is written into the artifact, next to the number, so it travels with it.

With no independent whole the verdict is **`UNCHECKABLE`** and `ok` is **false** — a decomposition
that nothing could contradict is not publishable. The independent whole is now threaded through:
`bench/stats.Sample.loop_wall_ms` records the wall time of the whole warmup+timed loop from
`perf_counter`, a clock that knows nothing about phases, and `_run_trace_pass` passes it into
`analyse()`.

On the stored dev0 trace the check reads **`CLOSES`** — trace-side time inside Compute is 74.1% of
the harness's wall time, the remainder being ORT graph execution, session setup and the CPU EP's
nodes between islands. A synthetic 2× inflation reads **`EXCEEDS_WALL`**. The falsifier fires.

**Per the strengthened §10.0: the wall-clock ratio leads. The decomposition may accompany it, never
replace it, and is publishable only with this identity check attached.**

### 12.4 `bench/admissible.py` — re-checking a number after its process exited

Every other guard in `bench/` runs at measurement time, inside the process that produced the
number. A JSON file in `bench/results/` carries none of them. It gets read by a human, pasted into
a status report, or differenced against another file, hours later. **Nothing re-checks it.**

All three fabricated results on this project came through that gap:

- **1.70×** through an EP that could not load — ORT *printed* the error and did not raise
- **1.45×** through an EP that loaded and declined every node
- a **"GQA speedup"** obtained by differencing two runs whose CPU baselines were **18× apart**

All three are visible in the stored artifact, if something looks. `bench/admissible.py` looks. It
grades every file in `bench/results/` on five gates, and **exits non-zero** when an inadmissible
artifact is present:

| gate | refuses when | the defect it is named after |
|---|---|---|
| `ep_loaded` | EP absent from `get_providers()`, or claimed count zero/absent | the 1.70× and 1.45× |
| `model_output_equivalence` | not `MATCH` | §10.0's gated triple; `UNMEASURED` is the default |
| `device_identity` | absent, or not `MATCH`/`NON_DECISIVE` | the inverted device labels (§11.1) |
| `machine_quiescence` | absent, or not `QUIET` | the 9.5× contention inflation (§10) |
| `measurement_validity` | absent, or failed | the harness's own self-check |

**Absence of a check is a refusal, not a default green.** That is the single design decision the
module turns on, and it is tested directly: removing any one field from a good record must produce
a refusal.

`WITHDRAWN` is its own grade and is **not** a failure — an artifact whose author has already
retracted it is the system working, and re-flagging it forever teaches people to ignore the output.

### 12.5 The cross-artifact check: `baseline_comparability`

The GQA defect does not live in either file. Both are individually unremarkable. It lives **between
them**, and no per-file gate can see it.

A speedup claim is a ratio of ratios, and its denominator is the CPU EP — which **no change to a
Vulkan EP can affect.** So:

> **Two runs may only be differenced if their CPU baselines agree.**

| run | vulkan | **cpu** | vk/cpu |
|---|---|---|---|
| `pre-gqa-dev0` | 3363.9 ms | **6226.8 ms** | 0.54× |
| `post-gqa-dev0` | 618.6 ms | **345.2 ms** | 1.79× |

**The CPU baseline moved 18.0× across a Vulkan-only change.** Read naively — Vulkan before against
Vulkan after — this is a **5.44× speedup**, and it is the most impressive number this project has
ever produced. Normalised to each run's own baseline, the Vulkan side got **3.3× worse**.

**Both readings are inadmissible.** The 6226.8 ms CPU figure is itself anomalous; every other run
in `bench/results/` shows 185–350 ms. Something was wrong with the machine, and the only honest
statement is that these two files cannot support any claim in either direction.

`post-gqa-dev1` (230.7 ms Vulkan vs 254.0 ms CPU) would read as **this project's first win** —
1.10× faster than CPU. It fails four of five gates and **must not be quoted.** That one is worth
stating plainly: the guard's first real act was to refuse a result we wanted.

### 12.6 Audit of `bench/results/` as it stands

```
WITHDRAWN     phi35-2026-07-30-phases.json   (withdrawn by me in §10, taken under contention)
INADMISSIBLE  phi35.json                     no device_identity / quiescence / validity
INADMISSIBLE  pre-gqa-dev0.json              same, and see §12.5
INADMISSIBLE  post-gqa-dev0.json             same, and see §12.5
INADMISSIBLE  post-gqa-dev1.json             same -- would have read as the first win
NOT_A_RESULT  timestamp-audit.json           not a timing artifact; not graded against timing gates
BASELINE_MOVED  pre-gqa-dev0 vs post-gqa-dev0  CPU baseline 18.0x apart
```

`NOT_A_RESULT` matters as much as the refusals. Grading a device-capability report against timing
gates would produce a false red, and **a false red costs a falsifier its authority as surely as a
false green does.**

**There is currently no admissible end-to-end performance number in this repository.** That is the
correct state of the record, and it is not a regression — the numbers were always this weak; only
the reporting has caught up. Re-measurement is blocked on a quiet machine, which has not existed
at any point today.

### 12.7 The instrument that goes red — this section's claims

| claim | instrument that goes red if it is false |
|---|---|
| nested sub-record spans are not double-counted | `phase_nesting` (containment vs caveat, red both directions) + `test_nested_span_is_not_summed_as_a_sibling` asserts the host total is 1.02 ms, not 1.82 ms |
| a new sub-phase cannot silently join the sibling sum | `phase_nesting` "contained but not declared" branch; `sibling_phases` unions table with declaration |
| the two upload accountings agree | `upload_accounting` — `DISAGREE` beyond 25%, `VACUOUS` (never pass) with one instrument |
| the decomposition closes against something that could contradict it | `decomposition_identity.external_closure` vs `Sample.loop_wall_ms`; `UNCHECKABLE` is `ok=False` |
| the internal closure is not evidence | it is labelled `WEAK` **in the artifact**, with the 99.0% failure quoted in `why_weak` |
| a stored number's provenance is intact | `bench/admissible.py`, exit 1 |
| the GQA files cannot support a speedup | `baseline_comparability` — `BASELINE_MOVED`, ratio 18.0 |
| a non-timing artifact is not falsely refused | `test_a_non_timing_artifact_is_not_graded_against_timing_gates` |

163 tests in `bench/` pass.

### 12.8 What I owe, and what I am blocked on

- **Blocked on a quiet machine** (unchanged from §10): re-measurement of both devices, and the
  warmup-decline discriminator (two quiet runs agreeing on `cycles_to_steady`).
- **For Switch:** verify the weight cache **on bytes, not wall time**. `device_upload_bytes` across
  a 1/2/3 inference sweep currently reads 1997.60 / 3995.19 / 5992.79 MiB — exactly linear. **If it
  stops being linear, the cache works.** Bytes are integers, deterministic, and immune to the 9.5×
  contention swing; a wall-clock before/after on this machine is not measurable today.
- **For Switch:** `rust/src/trace.rs:216` still attaches *"host: command-buffer recording; amortised
  across replays"* to the `record` span. §11 established that span is 98.6% upload. **The false
  caveat ships inside every trace we produce.** His file, not mine.
- **`largest_island_flops`** (§10.0's third slot) remains unemitted; my harness consumes it the day
  a `PartitionStats` event carries it.

---

## 13. The gate was wrong, the EP was not — and the first end-to-end decomposition (2026-07-31)

**Status of this section:** the `phase_containment` diagnosis, the structural findings and the
re-ranked levers below are final. **The end-to-end wall-clock numbers are still withheld** — see
§13.5, which explains why, with three independent instruments agreeing.

### 13.0 `phase_containment` — the phases over-reported, and the over-reporting was mine

For three days, on both devices, every phase share in this project was withheld on:

```
[ RED ] phase_containment: 0 phase spans outside any subgraph,
        1 subgraphs whose phases exceed their own duration.
        The attribution of phases to islands is not sound and no phase share
        below may be read.
```

**Which side is wrong: the phases over-report. The defect is in `bench/phases.py`, not in
`rust/src/trace.rs`.** `analyse()` computed `siblings = attribute(subs, sibling_phases(...))`
correctly for every total and every share — and then passed the *unfiltered* `attributed` list to
`phase_containment`. So the check added `record` to the `cmd_upload` that lives **inside**
`record`, and asked the subgraph to contain the same microseconds twice.

On the one-island Phi-3.5 graph the arithmetic is not subtle:

| span | µs (NVIDIA, first inference) |
|---|---|
| `vulkan.subgraph` (the parent) | 13 668 719 |
| `record` | 8 317 740 |
| `cmd_upload` (**inside** `record`) | 7 969 026 |
| `desc_alloc` + `pipeline_lookup` (**inside** `record`) | 92 331 |
| `fence_wait` + `submit` | 284 714 |
| **naive sum of all of them** | **16 663 811** ← 122% of the parent |
| **sibling-only sum** | **8 602 454** ← 63% of the parent, comfortable |

**Why it only appeared now.** It needed two things to coincide: sub-record spans becoming real
`ph:"X"` events (Switch, `692e7d0`), and `cmd_upload` becoming ~97% of `record` — which is what
the one-time 1997.977 MiB weight upload does on the first inference of a *single-island* graph.
Before the graph fused, `record` was spread over 257 islands and no single subgraph's children
were large enough to push the naive sum past its parent. **The partitioner succeeding is what
made my checker fail.**

### 13.0.1 Switch's spans are sound, and I checked before saying so

R13 says a result that confirms a prediction deserves more scrutiny than one that contradicts it.
I predicted double-counting and found double-counting, so:

| checked, on the live trace | result |
|---|---|
| every sub-record span timestamp-contained by a `vulkan.record` span | **2828 / 2828** |
| any `record` whose children sum past it | **0** (worst ratio 0.969) |
| any two sibling phases overlapping | **0 of 11 adjacent pairs** |
| `nested_in` span arg present and correct | on **all 2840** spans (`cmd_upload`/`desc_alloc`/`pipeline_lookup` → `record`; `record`/`submit`/`fence_wait` → `none`) |
| sibling-only sum inside its subgraph | worst ratio **0.911** |

Not one of those is a statement about `record`'s *name*. All five are timestamp geometry. **The
instrument that was wrong was the one deriving a rule from the geometry, not the one recording
it.**

### 13.0.2 What replaced it, and the third terminal state

`phase_containment` now checks **two tiers against their own parents** — siblings against their
subgraph, sub-record children against their own `record` span — and returns one of R13's three
states:

- **PASS** — both tiers close.
- **FAIL(condition)** — a real containment violation. Still fires: the stored 2026-07-30 Intel
  trace has a `record` span outside every subgraph and is `FAIL` under the new rule too, so this
  was a repair and not a mute button.
- **ERROR(instrument)** — a span declaring itself nested was handed in as a sibling. The check
  **issues no verdict**: `red` is `False`, the counts are `None`, and `red_flags` prefixes it
  `ERROR(instrument) — NOT a detection`. This arm exists because it is precisely the mistake that
  just happened, and a guard that can name its own misuse is worth more than one that reports it
  as a discovery.

Parenthood now has three sources unioned — the static table, the `host/sub-record:` caveat, and
the machine-readable `nested_in` arg whose Rust `match` is exhaustive with no `_` arm. Union and
not vote: **being wrongly excluded from a sum costs a line in a report; being wrongly included
double-counts the largest cost in the run.**

Not one synthetic trace in `bench/test_phases.py` contained a sub-record span. That is the whole
reason 163 tests passed over a checker that was wrong on every real trace we owned. There are now
nine that do.

### 13.1 Weight residency changed what the word "inference" means

Per-inference upload, measured from the trace's own transfer counters:

```
inference 1 : 1997.977 MiB      <- the weights, once
inference 2+: 0.387 MiB each    <- the feeds
                                   5162x step, same span name
```

Residency landed and it is not a small win. But it means **the first `Compute` call and every
later one are now different workloads**, and averaging them reports a fixed one-time transfer as
if it were a per-inference cost. Over a four-inference trace the single cold upload is 91% of the
whole `record` total. That is the *record-is-68%* mistake for the third time on this project,
wearing residency as its new disguise — the mean of two populations, published under the name of
one of them.

`phases.steady_state_split()` now separates them, deriving which invocations are cold from the
trace's own cycle structure rather than from a warmup flag, and **reports the cold inference
rather than discarding it** — a user pays it once and it is the right place to read model-load
cost. Fewer than three cycles returns `INSUFFICIENT`: two warm samples cannot show that they
agree with each other.

### 13.2 The in-band contention control had switched itself off — because we succeeded

`contention_signature` is the falsifier that answers "was the machine busy?" *from the trace
alone*, by comparing each island slot's host spread against its own GPU spread. It refused any
trace whose cycle period was 1 — and **the cycle period is the island count.** The moment the
partitioner started fusing Phi-3.5 into a single island, the in-band control reported
`UNTESTABLE` forever, on exactly the configuration this project is trying to reach.

Nothing in the statistic needs two slots: slot 0 still yields one host series and one GPU series
over the same repeated work, which is the entire method. What is genuinely lost at period 1 is
the ability to see a stall hit some islands and not others, so `stalled_slot_fraction` degenerates
to 0.0 or 1.0 and is now annotated `single_slot: true`.

Its refusal text also named the wrong condition — it printed `cycles=15; need >= 4` when 15 ≥ 4
and the failing clause was `period < 2`. **Quote the failure text, never the failure count** (R13)
cuts both ways: text that names a condition which did not fail is worse than a count.

### 13.3 One claim must not carry two constants

`phi35.baseline_disagreement` used `factor = 2.0`. `admissible.baseline_comparability` used
`tol = 0.25`. The same claim — *the CPU-EP control did not move* — had two answers eight-fold
apart, and the looser one is the one that runs during a measurement.

The 2026-07-31 run walked straight into the gap: its CPU baseline read **291.8 ms** during the
NVIDIA pass and **228.7 ms** during the Intel pass three minutes later. **1.276×** — silently fine
by the harness, `BASELINE_MOVED` by the audit. Both files now import one constant,
`admissible.BASELINE_TOL`.

That 27.6% movement is itself evidence, and it is the third independent instrument in §13.5.

### 13.4 Where the time actually goes — the first decomposition of a graph that executes

`model_output_equivalence = MATCH` with a real execution frame behind it; one fused island of
**353 of 363 nodes**; `dispatch accounting ok` (5295 == 5295 == 5295); `phase_containment` **PASS**
on both devices. Warm inferences only, cold excluded.

**Shares are printed. The milliseconds they are shares *of* are not** — see §13.5.

| share of time inside `Compute`, warm | NVIDIA RTX 4060 | Intel Iris Xe |
|---|---|---|
| `fence_wait` | 68.7% | 97.9% |
| ├ **GPU actually busy** | **67.7%** | **97.7%** |
| └ fence-wait GPU **idle** | **0.99%** | **0.18%** |
| `record` (incl. children) | 25.1% | 1.2% |
| ├ `record` leaf — real `vkCmd*` construction | 23.5% | 1.0% |
| ├ `desc_alloc` | 1.27% | 0.16% |
| ├ `pipeline_lookup` | 0.18% | 0.01% |
| └ `cmd_upload` (the feeds) | 0.14% | 0.03% |
| `submit` | 0.27% | 0.01% |
| unattributed inside `Compute` | 5.97% | 0.90% |

`gpu_busy` is **not** a sibling of the host phases — it overlaps `submit` + `fence_wait`. It is
printed against the same denominator so the two can be compared, never so they can be added.

**Why these proportions survive a contended machine when the milliseconds do not.** Contention
inflates *host* work; it does not touch the device timestamp counter. So contention can only push
the `record` share **up** and the `gpu_busy` share **down**. Every conclusion below is of the form
"the GPU share is large", and contention is the wrong sign to manufacture it. A quiet machine can
only make this picture *more* GPU-dominated, not less.

### 13.4.1 The lever ranking has changed, and one item is now dead

The previous ranking — residency, net-benefit declines, fence-wait GPU idle, kernels — was
derived from a phase decomposition taken while the model was running on **CPU fallback**. It
described a run in which this EP did not execute. Re-ranked against a run in which it does:

| # | lever | evidence | owner | was |
|---|---|---|---|---|
| **1** | **`q_gemv_matmul_nbits_f16` — one kernel, see §13.4.2** | GPU is busy **67.7%** (NVIDIA) and **97.7%** (Intel) of time inside `Compute`, and **95.11% / 98.28%** of that GPU time is a single kernel. Everything else in the EP sums to 4.9% / 1.7%. | Switch / Mouse | rank 4 |
| **2** | **Command-buffer reuse — stop re-recording every inference** | `record` is paid on **every** `Compute` call (15 spans, 15 inferences, 353 `desc_alloc` + 353 `pipeline_lookup` each time). 23.5% of NVIDIA in-`Compute` time is host `vkCmd*` construction the GPU waits through. **NVIDIA-only: 1.0% on Intel.** | Switch | not ranked |
| **3** | **The 5.97% unattributed inside `Compute` (NVIDIA)** | Time inside `vulkan.subgraph` that **no phase span covers** — input pointer reads, buffer/descriptor work before recording, output tensor writes after the fence. It is larger than every sub-record phase combined and it is currently *invisible*. Needs spans before it can be ranked properly. | Switch (spans), me (analysis) | not ranked |
| **4** | **Device-backed allocation** | `memory configuration: staging-bound`; `alloc_device_authoritative_spans` is `UNOBSERVABLE` (R12) until `offer_shared_device` is called — still `UNINVOKED`, red in Tank's census. Cannot be measured, so cannot be ranked higher than a hypothesis. | Tank / Switch | rank 1 (as residency) |
| ~~5~~ | ~~fence-wait GPU idle~~ | **Effectively dead: 0.99% NVIDIA, 0.18% Intel.** The fence wait is almost entirely the GPU working. Submission is not the problem. | — | **was rank 3** |
| ✅ | persistent weight residency | **LANDED.** 1997.977 → 0.387 MiB per inference. | Tank / Switch | was rank 1 |
| ↓ | net-benefit declines / island boundary | One island of 353 of 363 nodes. `retain_viable` is wired and there is very little boundary left to price. | Mouse | was rank 2 |

**The one-line version for the team: we are now GPU-bound, and on the Intel part we are*
***only*** *GPU-bound. Kernel work is the top lever for the first time in this project's history.**

Two caveats I will not let travel without the ranking. First, `record` at 23.5% of NVIDIA in-`Compute`
time is a *host* cost measured on a contended machine, so its true share is at or below 23.5% —
which only strengthens the case for putting kernels first. Second, the Intel figure is not a
compliment: 97.7% GPU-busy with a per-inference time far above the CPU EP's means the kernels are
slow, not that the pipeline is efficient.

### 13.4.2 Lever 1 is not "kernels" — it is one kernel

"Kernel efficiency" is too coarse to hand to Switch or Mouse. The per-dispatch GPU spans name the
kernel, so the ranking can be sharpened to a single work item. Summed device time across the whole
15-inference run:

| kernel | NVIDIA share | Intel share | dispatches/run | mean NVIDIA | mean Intel |
|---|---|---|---|---|---|
| `q_gemv_matmul_nbits_f16` | **95.11%** | **98.28%** | 2415 (161/inference) | 253.4 us | 3425.9 us |
| `gqa_f16` | 2.67% | 0.91% | 480 | 35.8 us | 159.2 us |
| `skip_simplified_layer_norm_f16` | 1.63% | 0.56% | 960 | 10.9 us | 48.9 us |
| `ew_binary_mul_f16` | 0.40% | 0.15% | 960 | 2.7 us | 13.6 us |
| `ew_unary_sigmoid_f16` | 0.19% | 0.09% | 480 | 2.6 us | 16.6 us |

Everything that is not `q_gemv_matmul_nbits_f16` sums to **4.9%** of GPU time on NVIDIA and **1.7%**
on Intel. By Amdahl, perfecting every other kernel in the EP to zero cost buys at most 4.9% and 1.7%
respectively. **Lever 1 is `q_gemv_matmul_nbits_f16`, and no other kernel is worth an hour of anyone's
time until it moves.**

The same table prices the two devices against each other on identical work: the same 161 dispatches
per inference cost 13.5x more on the Iris Xe. That ratio is a property of the kernel meeting a very
different memory subsystem, and it is the reason the Intel part is the more informative optimisation
target even though it is the slower one — it is the device on which this kernel's inefficiency is
least hidden.

I am deliberately *not* proposing a fix here. Naming the hot kernel is a measurement result and it is
mine; choosing between subgroup reductions, a different tiling, cooperative-matrix paths or a
dequant-to-f16 staging pass is Switch's and Mouse's call, and I will price whichever they pick.

### 13.4.3 The device-clock GPU-busy figure — and the device that refused to produce one

Host wall-clock is unusable on this machine (S13.5). The device timestamp counter is not: contention
inflates *host* work and cannot touch the GPU's own clock. So there is one figure this run can
legitimately produce. `phases.gpu_steady_tail()` computes it, and it is a falsifier before it is a
statistic — it demands a suffix of at least 5 inferences holding within 2% RSD, and reports
`NO_STEADY_TAIL` rather than a median if no such suffix exists.

| device | verdict | per-inference GPU busy | RSD | discarded | tail n |
|---|---|---|---|---|---|
| NVIDIA RTX 4060 | **STEADY** | **40.201 ms** | **0.033%** | 5 | 10 |
| Intel Iris Xe | **NO_STEADY_TAIL** | *withheld* | 4.6% over the whole run | - | - |

The NVIDIA series is `48.85, 48.91, 48.87, 48.88, 47.82, | 40.19, 40.22, ...` — a step down at
inference 6 that persists, and then ten inferences that agree to three parts in ten thousand. **An
0.033% RSD measured on a machine my own survey calls CONTENDED is itself the evidence that this clock
is the right instrument**: no host-side quantity in this run is anywhere near that stable.

The Intel column is the more important one. The Iris Xe never settled — it wandered between 542 and
629 ms for all fifteen inferences. An integrated GPU shares package power, cooling, and DRAM with the
loaded CPU cores, so the **amount of work completed per timestamp tick** is not contention-immune.
The timestamp tick itself is a different instrument: on this Gen12 part the reported 52.0833 ns is a
19.2 MHz reference timer, not the variable GT execution clock. The durations are therefore valid but
noisy observations of changing GPU performance, not incorrectly-scaled time. So the steady-tail
check still refuses, and Intel has no quotable per-inference point estimate from this run. See
§13.4.4 for the source check and falsifier.

**This is not an end-to-end number and it is not a substitute for one.** It is the summed duration of
one inference's dispatches. It excludes host recording, submission, readback and all of ORT's own
work, and it cannot be compared to a CPU EP latency. It is quotable for exactly one purpose: ranking
where GPU time goes.

### 13.4.4 Fact-check: the Intel timestamp is valid; the workload rate is unstable

Checked 2026-08-01. Ratings use the Fact Checker states, and every figure names the observation that
would falsify it.

| item | rating | finding | falsifier |
|---|---|---|---|
| Intel CPU and iGPU share a constrained package | ✅ **Verified** | This machine is an i7-13800H package. Integrated CPU/GPU power-management literature measures coupled CPU/GPU DVFS and thermal allocation; Intel's iGPU also shares system DRAM. Heavy CPU load can therefore lower achieved GPU frequency and consume memory bandwidth. Sources: Dev et al., *Implications of Integrated CPU-GPU Processors on Thermal and Power Management Techniques* (2018); Linux i915 documentation, “Unified Memory Access” (accessed 2026-08-01). | A controlled CPU-load sweep holding GPU work fixed where GT frequency, package power allocation, DRAM counters, and kernel duration remain invariant within instrument error. We do not yet have this observation. |
| `timestampPeriod = 52.0833 ns/tick`, 36 valid bits | ✅ **Verified for this device/driver** | Recorded by `vulkaninfo` in the 2026-07-29 device facts. `52.0833 ns` is `1 / 19.2 MHz`, matching Intel's Gen11+ timestamp path derived from the platform crystal/reference clock rather than the variable render clock. Source: Linux `intel_gt_clock_utils.c` (source revision checked 2026-08-01). | Re-querying the same physical device after a power-state transition returns a materially different period or a calibrated host/device interval shows the tick rate changing with GT frequency. |
| `timestampPeriod = 1.0 ns`, 64 valid bits on NVIDIA and lavapipe | ✅ **Verified for the recorded devices** | These are device-query observations, not family-wide constants. | A fresh device query for the same device/driver reports another value. No literature claim can override that direct query. |
| CPU load makes Intel timestamp durations “wrong” | ❌ **Contradicted** | Vulkan defines `timestampPeriod` as nanoseconds per timestamp increment. `VK_EXT_calibrated_timestamps` requires device timestamps to remain monotonic across power-management events; Intel's implementation derives the timer from a reference crystal. CPU load changes execution rate, not the unit conversion. The 542–629 ms spread is real performance variation. Sources: Vulkan Queries chapter and `VK_EXT_calibrated_timestamps` proposal (Khronos source checked 2026-08-01); Intel i915 clock source above. | A simultaneous calibrated-timestamp experiment shows the device counter's slope against QPC changing with CPU/GT power state beyond `maxDeviation`. That would be `ERROR(instrument)`, not a slow inference. |
| `NO_STEADY_TAIL` is the correct report | ✅ **Verified** | No suffix of at least five samples met the 2% RSD gate. Per R13 the terminal state is failure of the stability condition, not a fabricated point estimate. | Recomputing the recorded series finds a qualifying suffix, or a new controlled run produces one. Quote `NO_STEADY_TAIL`; do not convert the number of rejected samples into a detection. |

Primary timestamp sources:

- [Khronos Vulkan-Docs, `chapters/queries.adoc`](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/chapters/queries.adoc):
  timestamps monotonically track command execution and are converted by `timestampPeriod`; accessed
  2026-08-01.
- [Khronos, `VK_EXT_calibrated_timestamps` proposal](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_calibrated_timestamps.adoc),
  revision
  `8e076076` (2026-04-29): power-management events must not reset the extension's device timer;
  applications may recalibrate for cross-domain oscillator drift.
- [Linux i915, `intel_gt_clock_utils.c`](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/i915/gt/intel_gt_clock_utils.c),
  revision checked 2026-08-01:
  Gen11+ reads the timestamp reference from `RPM_CONFIG0`/`TIMESTAMP_OVERRIDE`; the source options
  include 19.2 MHz, and the driver separately tracks the GT execution clock.
- [Linux i915 documentation](https://docs.kernel.org/gpu/i915.html), accessed 2026-08-01:
  Intel integrated GPUs use Unified Memory Access.
- [Dev et al., integrated CPU-GPU thermal/power management](https://arxiv.org/pdf/1808.09651),
  2018, accessed 2026-08-01.

### 13.4.5 Fact-check: 13.5x is not predicted by these memory systems

The local Intel assumption is not a generic “Iris Xe” entry. The host reports an Intel i7-13800H
(96 EU Iris Xe, up to 1.5 GHz) and eight SK Hynix `H58G78BK7BX114` LPDDR5 packages configured at
5200 MT/s. A 128-bit LPDDR5-5200 interface has a theoretical peak of
`5200e6 transfers/s × 16 bytes = 83.2 GB/s`. This is the appropriate ceiling for this machine;
DDR4-3200 Iris Xe systems would instead have only 51.2 GB/s.

| item | rating | figure (sources checked 2026-08-01) | falsifier |
|---|---|---|---|
| RTX 4060 Laptop memory bandwidth | ✅ **Verified** | **256 GB/s**: 8 GB GDDR6, 128-bit, 16 Gb/s. Public RTX 4060 Laptop specifications: NVIDIA product family plus TechPowerUp/Notebookcheck device specification databases. | The laptop's VBIOS/memory telemetry reports a memory data rate other than 16 Gb/s or a non-128-bit bus. |
| Local Iris Xe memory bandwidth ceiling | ✅ **Verified as a configured-system calculation** | **83.2 GB/s** for the observed LPDDR5-5200, 128-bit configuration. The memory part is SK Hynix LPDDR5; the configured rate came from `Win32_PhysicalMemory`, not a benchmark. Intel lists LPDDR5 support for the i7-13800H. | Firmware telemetry shows fewer than 128 active data bits or a configured rate other than 5200 MT/s. A bandwidth benchmark cannot falsify the *theoretical* bus ceiling; it can only measure efficiency below it. |
| Bandwidth ratio, NVIDIA / local Iris Xe | ✅ **Verified arithmetic** | **3.08x** (`256 / 83.2`). For a DDR4-3200 Iris Xe the corresponding ceiling ratio would be **5.00x**, which is why an undifferentiated “Iris Xe bandwidth” number is invalid. | Either input specification is falsified as above. |
| FP16 peak, RTX 4060 Laptop | ⚠️ **Unverified as an OEM operating point** | Public databases give **11.61 TFLOP/s** at the reference boost; laptop TGP and boost vary by OEM. This kernel accumulates fp32, so the number is not its roofline. | Sustained clock telemetry plus the Ada CUDA-core issue rate yields a different peak for this laptop. We have no uncontended clock capture. |
| FP16 peak, local Iris Xe | ⚠️ **Unverified as a sustained figure** | Architectural peak is approximately **4.61 TFLOP/s** for 96 EU at 1.5 GHz; fp32 peak is approximately 2.30 TFLOP/s. Both assume max dynamic frequency and are not sustained guarantees. | GT frequency telemetry or an instruction-throughput counter shows the device cannot reach the assumed clock/issue rate. |
| Measured kernel ratio | ✅ **Verified from Niobe's timestamp trace** | **13.52x** (`3425.9 / 253.4`), rounded to 13.5x, for identical 161-dispatch inference work. | Re-summing the 2,415 per-device `q_gemv_matmul_nbits_f16` spans from the raw trace produces different means, or timestamp calibration fails. |
| Ratio expected from a well-tuned int4 GEMV | ⚠️ **Unverified range: about 3–5x** | Batch-1 GEMV streams the weight matrix, so the 3.08x local theoretical bandwidth ratio is the first-order prediction. Allowing unequal attainable bandwidth and UMA interference can widen it, but public specifications do **not** predict 12–15x. | The same known-good kernel, with DRAM byte counters, sustains a 12–15x device ratio while both devices are at comparable fractions of their own bandwidth ceilings. We currently lack those counters, so 3–5x must remain a design expectation, not a measured constant. |

Specification sources, all accessed 2026-08-01:

- [NVIDIA GeForce laptop comparison](https://www.nvidia.com/en-us/geforce/laptops/compare/)
  for the product family; [TechPowerUp RTX 4060 Mobile](https://www.techpowerup.com/gpu-specs/geforce-rtx-4060-mobile.c3946)
  and [Notebookcheck RTX 4060 Laptop](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-Laptop-GPU-Benchmarks-and-Specs.675692.0.html)
  for the 128-bit, 16 Gb/s, 256 GB/s configuration and reference-clock FLOP calculation.
- [Intel 13th-generation Core i7 product family](https://www.intel.com/content/www/us/en/ark/products/series/230486/13th-generation-intel-core-i7-processors.html)
  for the i7-13800H's 96 EU graphics, 1.5 GHz maximum graphics frequency, and memory support.
- [SK Hynix LPDDR5 product family](https://product.skhynix.com/products/dram/lpddr/lpddr5.go)
  for the observed `H58G78BK7BX114` memory technology. The 5200 MT/s configured rate and eight
  16-bit devices are local CIM observations; their product gives the 128-bit bus used in the
  83.2 GB/s calculation.

**Decision:** measured 13.52x divided by the 3.08x bus ratio leaves a **4.39x residual**.
The residual combines kernel portability loss with Intel's shared-power/shared-DRAM conditions; it is
not proof that exactly 4.39x is recoverable. It is enough to reject “hardware fact, do nothing.”
Switch should spend the day on the portability path, but use counters or a matched reference kernel
to separate shader design from CPU/DRAM contention.

Relevant alleged Intel pathologies:

| mechanism | rating | relevance to this kernel | falsifier |
|---|---|---|---|
| 32 KiB Intel vs 48 KiB NVIDIA workgroup shared-memory limit | ✅ **Verified limit; ❌ contradicted as this kernel's cause** | `q_gemv.comp` declares only 256 fp32 values = **1 KiB**. A shader declaring 48 KiB would fail pipeline creation on this Intel device rather than silently become this 13.5x result. | SPIR-V reflection shows more than 1 KiB Workgroup storage in the executed variant, or a 1/16/32 KiB sweep changes duration sharply. |
| `maxComputeWorkGroupInvocations` | ✅ **Verified; noise here** | Both recorded local devices report **1024**; this kernel uses 128 threads for K=3072 and 256 for K=8192. No device limit clips it. | The raw run metadata names a different limit or specialization constant. |
| hard-coded subgroup width | ✅ **Verified hazard generally; ❌ absent here** | Intel compilers may select SIMD8/16/32, and lavapipe reports subgroup size 8. The current shader uses no subgroup operation and sizes its tree from `gl_WorkGroupSize.x`, so it does not bake 32. | SPIR-V disassembly contains subgroup instructions or a literal-32 lane mapping that affects reduction correctness. |
| UMA/device-local system RAM | ✅ **Verified and plausibly material** | Intel's DEVICE_LOCAL heap is system memory. GPU weight reads compete with CPU traffic, and CPU package load can reduce GT headroom. This is the listed mechanism most capable of adding a multi-x penalty under the observed foreign CPU load, but **2–3x is not yet measured**. | A quiet CPU/load sweep with DRAM and GT counters leaves kernel time unchanged, or device-local bandwidth remains a fixed fraction of 83.2 GB/s under load. |

### 13.4.6 Fact-check: current subgroup-free int4 GEMV structure

llama.cpp source was checked at `master` on 2026-08-01; the relevant shader-base revision is
[`d0061be8`](https://github.com/ggml-org/llama.cpp/commit/d0061be838809230db7a4edf62bc9a098025ba98)
(2026-02-18). Primary files:
[`mul_mat_vec.comp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp),
[`mul_mat_vec_base.glsl`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl),
[`vulkan-shaders-gen.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp),
and
[`ggml-vulkan.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp).

| item | rating | finding | falsifier |
|---|---|---|---|
| llama.cpp requires subgroup operations for `mul_mat_vec` | ❌ **Contradicted** | It builds three reduction variants: full shared-memory tree, subgroup+shared hybrid, and subgroup-only/no-shared. Runtime selection uses `device->subgroup_arithmetic`; without it, `SHADER_REDUCTION_MODE_SHMEM` is selected. | Current source or generated SPIR-V lacks the shared-memory variant, or runtime pipeline creation selects subgroup SPIR-V when `subgroup_arithmetic` is false. |
| llama.cpp does more than change the reduction | ✅ **Verified** | Its quant path dequantises in registers, uses packed/vector loads and `vec4` dot products, manually unrolls the K loop, and keeps `NUM_ROWS × NUM_COLS` accumulators per thread so one workgroup can reuse work across multiple outputs. | Source inspection of the selected quant variant shows scalar element-at-a-time loads and only one accumulator, or capture identifies another shader path. |
| A known-good subgroup-free structure exists | ✅ **Verified structurally** | llama.cpp's fallback is precisely that structure: packed/vector loads, register dequantisation, several register accumulators, then a shared-memory tree. Our kernel already has register dequantisation and a correct shared tree, but only one output accumulator and scalar activation loads. | The fallback fails correctness on a no-subgroup implementation, or an A/B build shows it was never selected. Performance remains a separate claim. |
| The no-subgroup constraint explains most of 13.5x | ⚠️ **Unverified; unlikely by itself** | Our 128/256-thread tree pays 7/8 barriers. A subgroup hybrid removes most of them, so a fast path should help, especially on Intel, but the kernel still streams/dequantises thousands of weights. Literature/source inspection does not justify assigning a 2–3x gain to reduction alone. | A capability-gated subgroup variant, with identical loads and arithmetic, reduces Intel time by 2–3x while leaving NVIDIA near unchanged. |

A lavapipe-safe fast path does **not** bake subgroup size 32:

1. Keep the current shared-tree SPIR-V as the mandatory baseline.
2. Query subgroup arithmetic support for compute; optionally query subgroup-size control.
3. Build a hybrid variant using `subgroupAdd`, have lane 0 of each subgroup write one partial, then
   reduce `gl_NumSubgroups` partials. Use `gl_SubgroupInvocationID`, never `lane & 31`.
4. Select subgroup-only reduction only when the workgroup is one subgroup; otherwise select hybrid.
5. Select the baseline when arithmetic support is absent or the driver/CPU implementation is not
   on the validated capability list. Lavapipe's subgroup size 8 then remains correct in either path.

The higher-priority structural experiment is independent of subgroups: vectorise packed loads and
compute multiple output columns per workgroup/thread with multiple register accumulators, while
retaining the existing shared-tree fallback. That is the closest source-backed explanation for why
our scalar one-output kernel leaves much more than the 3.08x hardware bandwidth ratio on Intel.

## 13.5 What remains non-quotable, and why I am not overriding my own gate

**The end-to-end wall-clock figure and the Vulkan/CPU ratio are WITHHELD on both devices.** Under
Morpheus's S10.0 disclosure obligation I state plainly that they are *withheld, not omitted*: the
samples exist, 20 per device per arm, they are in `bench/results/phi35-2026-07-31.json`, and they are
marked non-quotable there. I can see them. They may not be published, including inside this team.

Three instruments refuse them, and they are independent of one another:

1. **The out-of-band load survey** — CONTENDED on both passes: 5.98 and 5.32 foreign cores of load
   (VS Code, other agents' `copilot.exe`, `msedgewebview2.exe`, `GlobalSecureAccessClient.exe`).
   This instrument does not read the trace.
2. **The in-band trace signature** — `contention_signature` reports `HOST_SIDE_EXCURSIONS` on both
   devices, derived purely from the shape of the per-inference host spans. This instrument does not
   read the process table. It corroborates (1) from an entirely separate source, which is the only
   reason either is worth anything under R9.
3. **The CPU-baseline control** — the CPU EP baseline moved **291.8 ms to 228.7 ms, a factor of
   1.276**, between the NVIDIA pass and the Intel pass of the same afternoon, on the same model, same
   machine, same harness. The only thing that changed was the machine's load. A 1.28x drift in the
   *reference* arm is fatal to a ratio: it is 28% of a speedup claim available for free.

I polled for a quiet window for roughly two and a half hours (`bench/_scratch/hunt_quiet.ps1`) and
never got one; this box carries a persistent ~5-6 core foreign load. I ran the benchmark anyway to
collect everything that *is* robust, which is what S13.4.2 and S13.4.3 are made of.

**Per R13, every previously published wall-clock figure for this EP is withdrawn** — including the
3.1x and 3.7x quoted earlier today, which were taken during CPU fallback and therefore measured the
CPU EP against itself with extra bookkeeping. They were never measurements of this EP. Nothing in
this document may be read as superseding them with a better number; they are withdrawn, and their
replacement does not exist yet.

### 13.5.1 The number I most wanted to publish is the one I am most suspicious of

The withheld NVIDIA sample is faster than the CPU EP. Morpheus predicted, and I predicted, that this
EP would be *slower* than CPU at this stage — staging-bound memory, no device-backed allocation,
command buffers rebuilt every inference. A result that contradicts a prediction is cheap to believe
and I still do not believe this one, because R13 cuts the other way too: the reason it is not
admissible has nothing to do with which direction it points. The Intel sample, which *confirms* the
slow prediction, gets **more** scrutiny under R13, not less — and it fails the same three checks.

Both are refused by the same instruments for the same reason. If I let the NVIDIA one through because
it is exciting and held the Intel one because it is disappointing, the gate would be a publicity
department. It refused a 5.44x "speedup" last week that was really an 18x baseline shift; it refuses
this too.

### 13.5.2 What it would take to lift the withholding

Nothing in the EP has to change. This is purely an instrument-conditions problem:

- A machine with no foreign load for the ~6 minutes both passes take, or
- a dedicated benchmark host, or
- a scheduling window where the other agents and the editor are down.

The run is otherwise fully interlocked and has been for a day: `model_output_equivalence = MATCH`
with a real execution frame behind it, `dispatch accounting ok` (31 == 1 island x 31 inferences),
`gpu_span_accounting` 5295 == 5295 == 5295, `phase_containment` **PASS** on both devices since the
fix in S13.0, 20+20 samples per device. **The moment this box is quiet, the first admissible
end-to-end number in this project's history falls out of a single command with no further work:**

```powershell
cd bench; python phi35.py --device 0 --require-quiet
cd bench; python phi35.py --device 1 --require-quiet
```


## 14. The barrier fix, measured as an A/B — and a floor on what a settled tail may claim (2026-08-01)

Three things happened on `main` before this section: Mouse's partitioner fused Phi-3.5 into **one**
island of 355 claimed nodes, Switch replaced **147,618 per-buffer `VkBufferMemoryBarrier` structs
per inference** with a single global `VkMemoryBarrier`, and Tank landed a runtime WARN that speaks
through ORT's own logging sink. All three are visible in what follows, and the third one broke my
harness before it could measure the first two.

### 14.0 What the machine was, and what is therefore withheld

`CONTENDED` on every run below — 4.7 to 11.4 foreign busy cores, sourced (by name, from the survey's
own `top_foreign`) to the other agents' `copilot.exe`, `Code.exe` and Defender, not to Justin's
project, which had indeed finished. **The gate refused, I did not override it, and the end-to-end
wall clock and the Vulkan/CPU ratio remain withheld on both devices** for the third session running.
The samples exist in the JSON records under `vulkan`/`cpu` and are marked non-quotable there.

What follows is entirely **device-clock and paired-difference** work, which is the class of result
this project has established survives a loud machine:

- the device timestamp counter cannot see host load at all (§13);
- Switch's own hog experiment put `gpu_steady_tail` under foreign GPU load and it either landed
  within **0.08%** of solo or refused outright;
- and an A/B whose two arms are alternated inside one sitting cancels a load level that neither arm
  chose.

### 14.1 NVIDIA RTX 4060, device clock, current `main`

| quantity | value | basis |
|---|---|---|
| GPU busy per inference | **13.3432 ms** | `gpu_steady_tail` STEADY, n=43, coverage 100%, RSD 1.72% |
| corroborating run | 13.3468 ms | n=40, coverage 87%, RSD 1.88% |
| `model_output_equivalence` | **MATCH** | same run, same process, 355 of 363 nodes claimed, 1 island |
| dispatch spans | 16,330 == 16,330 == 16,330 | `gpu_span_accounting`, so ordinal attribution is exact |
| driver | NVIDIA 591.55, Vulkan 1.4.325, timestampPeriod 1 ns, validBits 64 | |

**What this is comparable to, and what it is not.** It is comparable to the same figure taken on the
same box, with the same harness and the same artifact, minutes apart — which is exactly what §14.2
does, and that is why §14.2 exists. It is **not** comparable to my published **40.201 ms/inference**
of 2026-07-31. Between those two numbers lie Switch's GEMV kernel rewrite, Mouse's partitioner
fusion (161 islands to 1), persistent weight residency and the barrier fix. Dividing them yields
"3.0x" and attributes four people's work plus an unknown to whichever one is being discussed. **The
40.201 ms figure is retired as a baseline, not beaten by one.**

Intel Iris Xe on the same `main`: **withheld, `NO_STEADY_TAIL`** — "no suffix of >= 5 inferences
holds GPU busy time within 2% RSD. The device never settled." The per-inference series wanders
53.4-91.3 ms across 46 inferences with no flat region. Same refusal as 2026-07-31 and for the same
structural reason: an iGPU shares its power budget with the loaded CPU cores, so on *this* device
the device clock is not contention-immune and the loud machine reaches it. Two devices differing in
kind, not degree. **They are not compared** — `bench/compare.py`'s cross-device refusal stands.

### 14.2 The barrier fix, A/B, interleaved — `bench/results/probe_barrier_ab.py`

Switch's prediction had a falsifier attached, which is why it was worth testing: `vulkan.record`
cost **14.414 ms of host time against 12.156 ms of device-clock busy**, 96.4% of it in no named
span, so removing 147,618 barrier structs built on the host should move **host** time sharply and
**device** time barely. If device time moved a lot, something else changed.

The naive test — run the old DLL, run the new one, subtract — has a confound this project has been
burned by: host time is precisely what machine load moves, this box is not quiet, and the load is
not constant across two runs minutes apart. So the two DLLs are **alternated A B A B in one
sitting**, each run carrying its own load survey. Same harness, same machine, same artifact; the
DLL is the only difference (`git diff 42deaba..1cd0b55 -- rust/` touches `vk/barrier.rs`,
`vk/session.rs` and Tank's host-side logging — **no shader changes**, so a device-side movement
would have to be the barrier).

| run | foreign cores | `record` host median | GPU busy tail | n | coverage |
|---|---|---|---|---|---|
| post (fix) | 7.64 | **3.780 ms** | 12.1833 ms | 38 | 88% |
| pre | 5.06 | **16.412 ms** | 13.3463 ms | 43 | 100% |
| post (fix) | 8.12 | **3.687 ms** | 13.3432 ms | 43 | 100% |
| pre | 8.31 | **20.344 ms** | *refused* | 5 | 12% |

**Host: confirmed, and not by load.** `record` host median falls **4.3x to 5.5x** (leaf-only, with
`desc_alloc`/`pipeline_lookup`/`cmd_upload` subtracted: 810.0 ms to 139.0 ms over 43 recordings,
**5.8x**). The load ordering is scrambled across the arms — the second post run was measured under
*more* foreign load (8.12 cores) than the first pre run (5.06) and was still 4.3x cheaper. Load
cannot produce that ordering.

**Device: unchanged, which is the half of the prediction that could have failed.** The matched pair
— both n=43, both 100% coverage, taken back to back — reads **13.3463 ms pre vs 13.3432 ms post, a
difference of 0.02%.** The remaining post run reads 12.1833 ms, so the run-to-run spread within the
post arm alone (±9%) is larger than anything between the arms. **The fix moved host time and did not
move device time.** Nothing else changed, and we do not have to go looking for what did.

**An arithmetic corroboration I did not expect to get.** Switch derived **~94 ns per barrier struct**
(13.9 ms unnamed / 147,618). The host `record` delta measured here is 16.412 - 3.780 = 12.63 ms
median, and leaf-only 18.84 - 3.23 = 15.6 ms, over the same 147,618 structs: **86 ns to 106 ns per
struct**, straddling his figure. Two decompositions from different instruments agreeing to within
13% on a per-struct cost is the strongest form this project has for "the named mechanism is the one
that was paying."

And the part of his work worth repeating more than the fix: he was confident the cost was
`env::var_os` in the per-dispatch loop, **benchmarked it first — 0.232 µs/call, 0.083 ms/inference,
wrong by 170x** — and left it alone.

### 14.3 RULING — a settled tail is not automatically a quotable one

Morpheus asked whether `gpu_steady_tail` needs a minimum-n floor, on the observation that his
`contended` row **passed at n=8 while sitting 2.1% above solo**. **It does, and the floor is on
coverage more than on n.** Implemented in `phases.gpu_steady_tail`; the new verdict is
`MARGINAL_TAIL`.

The defect is precise: **the 2% RSD bar constrains the tail's internal spread, not its agreement
with the device's true steady rate.** Five samples from a locally flat stretch of a wandering series
clear it exactly as easily as a settled device does. Specimens, all real, all from one day:

| tail | n | discarded | coverage | median | its own run's warm mean | error |
|---|---|---|---|---|---|---|
| pre-fix A/B run 2 | 5 | 38 | 12% | 37.562 ms | 26.412 ms | **+42%** |
| pre-fix long run | 7 | 39 | 15% | 20.055 ms | 15.03 ms | **+33%** |
| Switch `contended` | 8 | 38 | 17% | 11.7697 ms | (solo 11.526) | +2.1% |
| every good tail | 38-43 | 0-6 | 87-100% | 13.343-13.347 ms | 13.35-13.42 ms | **<0.6%** |

The separation is clean and it is not on `n`: it is on **how much of the series the tail keeps**. A
genuine warmup ramp is a short *prefix* (5-6 of 46 here); a device that never settled produces a
short flat *suffix* (38-39 of 46 discarded). So the floors are `n >= 8` **and** `coverage >= 50%`,
and a suffix that clears the RSD bar but fails either is `MARGINAL_TAIL` — **not a slower number, no
number**. Its median is kept under `withheld_median_ms` so that no later reader and no aggregation
can pick it up as one. Consumers gate on `verdict == "STEADY"` and were already correct for this.

This deliberately makes the instrument refuse more often, and its first act was to refuse two of my
own runs from this session — including the one that appeared to show the barrier fix improving GPU
time by 33%, which was the exciting reading and was false. Re-analysed under the floor, the
pre/post GPU figures agree to 0.02%, which is the finding.

### 14.4 The harness died on Tank's WARN, and that is an `ERROR(instrument)` (R12/R13)

`bench/phi35.py` ran `subprocess.run(..., text=True)` on the worker. **ORT's default logging sink on
Windows writes UTF-16LE, our own narrow lines share the same handle, and Tank's new WARN goes
through ORT's sink** — so `subprocess`'s reader thread raised

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa7 in position 1006
```

and then the error path raised on top of it: `proc.stderr.strip()` with `proc.stderr` at `None`.
**A harness that dies on its own error path cannot report the error it found.**

This is R12, textbook: two instruments each correct about a different world — Tank's channel is
UTF-16LE, mine assumed UTF-8 — and the collision surfaced as a crash rather than a wrong number,
which is the good version of this failure. The repair therefore borrows **Tank's `decode_both`**
(`rust/tools/probe_broken_commitment.py`), which reads the same bytes four ways and carries his
written record of what each naive version got wrong. A second decoder in `bench/` would be a second
dialect for one channel, which is what Link refused to create for the verdict vocabulary; if his
module is not importable the tail is `ERROR(instrument=stream_decoder)` rather than a private
fallback decode. Under R13 the capture failure is recorded in `instrument_errors`, never in
`refusals`: an instrument error is the harness not having looked, a refusal is a condition the
harness found, and no aggregation may count one as the other.

### 14.5 Named gap, not fixed today: the `MATCH` in `bench/` carries no execution frame

`bench/admissible.py` refuses every artifact in §14 for two reasons, and the second is the
interesting one:

```
x machine_quiescence: machine_quiescence is CONTENDED.
x model_output_equivalence: MATCH but there is no model_output_equivalence_record, so the
  verdict carries no executed_by frame.
```

`bench/phi35.py` does its own in-process CPU comparison and records `MATCH` as a bare string. The EP
side (`rust/src/counters.rs::EQUIVALENCE_RECORD_KEY`) and Trinity's `tests/ops/_verdict.py` both
carry the full record with its `executed_by` frame; the bench harness does not yet consume that
vocabulary, so its verdict is `UNATTRIBUTED` by my own gate's reckoning. **That is correct
behaviour and it is my gap to close**, in `bench/`, next session. It is named here rather than
half-wired now, because a mechanism that is present and not reaching the call graph is the R10
specimen and I would rather have the hole visible than covered.

**Scope note added to `admissible.py`'s report at the same time:** that gate governs the
**wall-clock** record — `vulkan`/`cpu` medians, their delta and their ratio. It does not reach the
device-clock figures in the same file. Everything published in §14 is device-clock or a paired
host-phase difference, and every wall-clock number in those files remains withheld.

---

## 15. My steady-tail gate cannot see a bias, and §14 was published without the instrument that would have caught it (2026-08-01, later the same day)

### 15.0 The retraction, and it is against me

Six hours ago I ruled on a question the coordinator put to me — whether `gpu_steady_tail()` needed
a minimum-n floor — on the stated premise that the gate *"under foreign GPU load either lands
within 0.08% of solo or refuses outright."* **That premise was withdrawn. It was assembled from
the two runs that agreed with it.**

Switch's `bench/results/probe_gputenancy.py` reproduces the opposite in seconds from committed
artifacts, on any machine, with no GPU present:

| specimen | what the gate said | truth | error | **RSD** |
|---|---|---|---|---|
| `soloA`, sole tenant | STEADY 11.525 ms | 11.525 ms | — | 0.8098% |
| `contended3` truncated to 20 / 28 / 34 inferences | **STEADY 126.647 ms** | ~11.5 ms | **10.99x** | **0.79-0.91%** |
| `base_b`, board pinned at idle clock | **STEADY 246.735 ms** | ~11.5 ms | **21.4x** | **0.1163%** |

**In both failures the wrong number carried a better RSD than the right one.** That is not a
coincidence to be tuned away, it is the mechanism. `gpu_steady_tail` is a **variance test over a
suffix, and a variance test cannot see a bias**: a uniformly wrong series is a *perfectly steady*
one, so a run that is entirely wrong receives the gate's **most confident possible verdict**. This
board idles at 210 MHz against a 3105 MHz boost; **a low clock does not raise RSD, it lowers it.**

### 15.1 Why my floor was necessary and is nowhere near sufficient — and why I am not allowed to present it as the fix

The `n >= 8` / `coverage >= 50%` floor of §14.3 does real work and I am keeping it. It catches a
**wandering** device — a short flat excursion inside a series that never settled — and it caught
two such runs of my own within the hour of writing it. It caught a third today (§15.3).

It does **nothing whatever** about the failure above, and the reason is worth stating plainly
because the instinct is to reach for it: **more samples make a biased series more confident, not
less.** Every one of the 10.99x specimens would sail through an n floor; `base_b` produced its
21.4x-wrong figure at n=46 with **zero** discarded and the best RSD in the entire artifact set.
Raising a threshold on the tail's internal spread admits *more* of this failure, not less.

Under DESIGN.md **R9 rule 5** this is an *anti-correlated falsifier*: a check whose confidence
measure is computed from the same series as the quantity it certifies, and which moves the same
way as the reader's confidence when the quantity is wrong. **An identity whose two sides come from
the same source is a falsifier that cannot fire.** R9's remedy is not a tighter threshold; it is
**a different instrument**. `gpu_steady_tail` is accordingly demoted **from a gate to a
precondition**, and a device-clock claim is `UNMEASURED` until a second quantity, from outside the
series, records the state of the device.

### 15.2 What is now mandatory in `bench/` — `bench/device_state.py`

Every tail returned by `phases.gpu_steady_tail` is now **born `certification: UNCERTIFIED`**.
`phases.analyse(..., device_state=...)` is the only thing that can lift it, and only on the
evidence of a tenancy verdict and an SM-clock record taken **over the same window** by Switch's
`bench/results/probe_gpustate.py` — imported, not re-implemented, for the same reason I imported
Tank's `decode_both` rather than writing a third decoder for one channel.

**Absence of the check is a refusal, not a default green.** There is no code path that turns a
missing companion into a pass; `phi35.py` runs the sampler around its traced pass and threads the
record through, and a report with no companion says `device_state: ABSENT` and quotes nothing.

Five terminal verdicts, and only one releases a number:

| verdict | when | quotable |
|---|---|---|
| `QUOTABLE` | tail STEADY **and** sole tenant **and** peak SM >= 50% of board max | **yes**, with the companion attached |
| `WITHHELD` | foreign GPU work, or the board never left its idle clock | no — this is a **detection** |
| `UNCERTIFIED` | no companion record, or the tail produced no number | no — and **not** a detection |
| `UNOBSERVABLE` | the instrument's frame does not contain this device (R12) | no — and **not** a detection |
| `ERROR` | the companion itself failed (R13) | no — and **never** a finding about contention |
| `UNCERTIFIED(partial_companion)` | **added §16** — tenancy observed, clock axis has no producer (`TENANCY_ONLY`) | no — and **not** a detection |

*(§16 splits the record into a tenancy axis and a clock axis. The four verdicts above are what a
record with **both** axes earns; the fifth is what a record with only tenancy earns, and it never
releases a number. A detection on the tenancy axis alone still stands: `FOREIGN_GPU_WORK` without
a clock record is `WITHHELD`, because no clock reading would unobserve foreign work.)*

**Why the peak clock and not the median.** The median SM clock does not discriminate: the
correctly-clocked `after_coldboard` run (11.5243 ms, right) and the pathological
`baseline_certified` run (246.72 ms, 21.4x wrong) *both* have a median sample of 210 MHz, because
`nvidia-smi` samples at 4 Hz across the whole command including the host phases where the board is
legitimately idle. The **peak** separates them by an order of magnitude and not by a hair: every
correct run reached 2280-2490 MHz (73-80% of the 3105 MHz board maximum), and both wrong ones
never left 210 MHz (6.8%). `clock_ramp_x` separates identically — 11.79x versus 1.0x.

**What this companion cannot see, written into every record it emits** (R9's silence clause; a
caveat that lives in a document does not travel with the number):

- It samples at 4 Hz over the whole command, so it characterises the **regime** a run sat in and
  cannot resolve a single inference. A boost that happened while our kernel was not submitting
  would satisfy the peak-clock floor. The tenancy check covers the case that motivates this, but
  it is a real gap and it is named rather than covered.
- It reads the driver's own account. A clock the driver misreports is invisible to it.
- `nvidia-smi` is NVIDIA-only. On any other vendor this is `UNOBSERVABLE`, which is **not**
  `SOLE_TENANT` and **not** a pass. *(§16: on Windows the WDDM counters now fill the **tenancy**
  axis on any adapter, including Intel. They fill no part of the clock axis, so an Intel figure
  moves from `UNOBSERVABLE` to `UNCERTIFIED(partial_companion)` — better evidence, same
  non-quotability. §16.3 rules that this is permanent on this hardware.)*
- It says nothing about host contention. That is `bench/contention.py`'s subject and it gates a
  different quantity.

### 15.3 Re-measured on `main` (`b04347c`), with the companion attached

Four runs. **One is quotable.** That ratio is the point of the exercise, not a disappointment.

| run | companion | peak SM | tail | median | n | coverage | RSD | **certification** |
|---|---|---|---|---|---|---|---|---|
| dev0 #1 | SOLE_TENANT | 2010/3105 (65%) | STEADY | **12.1847 ms** | 41 | 82% | 1.496% | **QUOTABLE** |
| dev0 #2 | SOLE_TENANT | 2010/3105 (65%) | MARGINAL_TAIL | *(12.187 withheld)* | 13 | 26% | **0.086%** | UNCERTIFIED |
| dev0 #3 | SOLE_TENANT | 2010/3105 (65%) | NO_STEADY_TAIL | — | — | — | — | UNCERTIFIED |
| dev1 (Intel Iris Xe) | **UNOBSERVABLE** → **TENANCY_ONLY** (§16) | — *(no producer)* | NO_STEADY_TAIL | — | — | — | — | UNCERTIFIED *(partial_companion from §16)* |

**The one number: 12.1847 ms of GPU-busy time per inference on the RTX 4060 Laptop GPU**, driver
NVIDIA 591.55, Vulkan 1.4.325, `timestampPeriod` 1 ns, `validBits` 64, 355 of 363 nodes claimed in
one island, `model_output_equivalence` MATCH, sole tenant over 51 samples across 20.7 s.

**Note dev0 #2 carefully, because it argues against me.** Its withheld median is 12.187 ms — it
agrees with the certified run to four significant figures — and it carries an RSD of **0.086%,
the lowest of any run I have ever taken on this device**, on a tail covering **26%** of the
series. **A refusal is not a claim that the number was wrong.** It is a statement that this run
could not establish it, and the fact that the refused number happens to be right here is exactly
why the refusal must not be conditioned on how plausible the number looks.

**What 12.1847 ms is comparable to: at present, nothing.** It is the first device-clock figure
this harness has produced that carries a companion. It becomes comparable to the next figure taken
the same way, on this device, with a certification attached.

Specifically what it is **not** comparable to:

- **Not to Switch's 11.525 ms.** Different branch, different harness invocation, different
  dispatch count. Laying them side by side is the cross-setup error the coordinator has now made
  four times this week and I am not making it a fifth.
- **Not to my own 13.3432 ms of §14.1**, taken this morning on the same device and the same build
  — because *that* one has no companion. It is not wrong; it is unestablished.
- **Not to 40.201 ms as a before/after pair.** See §15.4.

**The Intel part is doubly unquotable and the two reasons are different.** Its tail is
`NO_STEADY_TAIL` — a *detection*, the device did not settle. Its companion is `UNOBSERVABLE` —
**not** a detection; `nvidia-smi` is installed on this host and exits 6 for board index 1, because
an Intel Iris Xe is simply not in its frame. R12: a counter whose event cannot occur in its frame
reports `UNOBSERVABLE`, never `SOLE_TENANT`. Classifying that as `ERROR(instrument)` would have
filed a permanent property of the device as a transient fault of the harness, and my first cut did
exactly that until the Intel run showed me the difference. **There is at present no device-state
companion for the Intel part, and therefore no route by which any Intel device-clock figure can
become quotable.** That is a hole in the instrument set and I am naming it rather than granting an
exemption to the device that cannot be checked.

### 15.4 Why 40.201 ms survives — and the part of it that does not

The coordinator asked me to say why the figure survived, and a figure that survives its
instrument's retraction should carry the reason.

**It survives as a regime.** The two clock regimes on this board are **21x apart and do not
overlap**: an idle-clock run of this workload lands at ~246 ms, a boost-clock run at ~11-13 ms.
40.201 ms is nowhere near 246 ms. Its *regime* is therefore recoverable from its magnitude alone,
without the record that was never taken, and it cannot be an idle-clock artifact.

**It does not survive as the "before" half of a before/after pair.** Switch withdrew his own
12.183 ms baseline on precisely this ground — certified 11.589/11.525/11.524 ms measured against a
number taken with no tenancy verdict and no clock record — with the line I am adopting against my
own figure: ***"it is probably sound at ~1.05x, and 'probably sound' is not the standard."***
40.201 ms was taken with no companion. I may not pair it with 12.1847 ms and call the difference a
result, and I am not going to, having spent §14 warning about exactly this.

### 15.5 A correction to my own use of minima, which touches every bound in this document

Switch caught this in his own writing and it is in mine too. I had been treating
minimum-over-inferences as a **lower** bound on uncontended cost. **It is an upper bound:**
`observed = true + delay` with `delay >= 0`, so `min(observed) >= true`.

The consequence is not cosmetic. **Two upper bounds do not bound a difference from below.**
"`record` <= 14.414 ms before" and "`record` <= 2.704 ms after" does not by itself prove an
improvement, let alone a 5.33x one. Every "at most" in §9 and §13 that I used to argue a floor
under a *difference* is hereby withdrawn as an argument; the individual bounds stand as bounds.

### 15.6 How the barrier result should be stated — the count is the claim, the timing is the estimate

What makes the direction of Switch's barrier fix certain is **not a clock**. It is a **count**:

> **147,618 `VkBufferMemoryBarrier` structs and 354 heap allocations per inference before; 354
> barrier structs after.** 355 kernels, 417 intermediate buffers, one barrier per buffer per
> dispatch, replaced by a single global `VkMemoryBarrier`.

**Counts do not care whether the box is busy, what clock the board is at, or who else is on it.**
No tenancy verdict is required to know that 147,618 is more than 354, and no contention gate can
withhold it. The direction is **certain**.

The timings — my §14.2 A/B's 4.3-5.5x on host `record`, Switch's 5.33x — are **estimates**, and
they are estimates taken without a device-state companion. They are consistent with the count and
they are not proof of its size. **That is the shape the barrier result is published in from here:
the count is the claim, the timing is the estimate**, and §14.2 should be read with §15.5 in mind.

The half of §14.2 that does *not* need re-qualifying is the **negative** result, because it is a
statement about a quantity that did *not* move: on the matched A/B pair the device clock read
13.3463 ms pre and 13.3432 ms post, **0.02%** apart, while host `record` fell 4.3-5.5x. Switch's
prediction had a falsifier and did not trip it. A null result across two builds measured minutes
apart on one machine is far more robust to an uncertified clock than a ratio is — if the board had
been in the wrong regime, it was in the wrong regime for both halves.

## 16. Intel has no clock producer, and a tenancy-only record is not a companion (2026-08-01, later)

The companion of §15.2 made a device-clock figure quotable only with a **tenancy verdict and a
clock record** over the statistic's own window. Its only producer is `nvidia-smi`, so on the Iris
Xe the record is `UNOBSERVABLE` and **no Intel device-clock figure has ever been quotable** — which
is where the open question lives, because the 4.39× of the 13.52× Intel/NVIDIA kernel gap that
memory bandwidth does not explain (§13.4.5) is a claim about Intel.

Windows exposes two counters that are not locked to a GPU vendor:
`\GPU Engine(*)\Utilization Percentage` and `\GPU Engine(*)\Running Time`, produced by the **WDDM
scheduler**. This section says what they can and cannot witness, what verdict a record from them
earns, and whether an Intel device-clock figure can ever be certified on this hardware.

### 16.1 What the Windows counters witness on Intel — stated as a capability

`bench/win_gpu_counters.py`, measured by `bench/results/probe_wingpu.py` on the Iris Xe with a real
Phi-3.5 pass (8 inferences, `MATCH`, 71.7 s window, 46 enumerations, worst blind gap 7.1 s):

| question | answer | evidence |
|---|---|---|
| Does it see **our own** submissions on the Intel adapter? | **yes** | 1.4292 s of engine time on `engtype_3d`, LUID `0x00010aa0`, attributed to our worker PID |
| Does it see **other processes** on that adapter, per PID? | **yes** | `Code.exe` (pid 30232) held it 0.1296 s, 0.18% of the window |
| Does our work leak onto the **other** adapter's record? | **no** | 0 s on the NVIDIA LUID, and 0 s on each of the three other live LUIDs |
| Does it report a **clock**? | **no** | no `GPU *` counter set carries MHz; no `root\wmi` class here does either |
| Does it corroborate `nvidia-smi` where both exist? | **yes** | same window on the RTX 4060: `SOLE_TENANT` from both instruments, `agree: true` |

**The negative control is the part that makes the first row mean anything.** Our worker holds
`\GPU Engine` *instances* on **both** adapters — a process that opens a device on each gets an
instance on each — so "our PID appears in the counters" is worth nothing. What was measured is that
**engine time accrues only on the adapter we ran on**: 1.4292 s on Intel and 0 s on NVIDIA when the
EP ran on the Iris Xe; 2.6165 s on NVIDIA and 0 s on Intel when it ran on the 4060. The LUID join
goes through a registry description string, which is exactly the kind of join that is silently
wrong, and this is what checks it.

So the capability, stated so it can be falsified rather than believed:

> **On this box the WDDM counters witness per-process GPU engine occupancy on the Intel adapter,
> including ours, and they witness no clock at all. That is a tenancy instrument for any WDDM
> adapter, and it is not half of the clock instrument — it is none of it.**

Three second-order facts the record carries because they change what the number means:

- **Compute is scheduled on the `3D` engine node.** Neither adapter exposes an `engtype_Compute`
  instance; all of our GPU time appears under `3d`. An `engtype`-based filter looking for compute
  would have found nothing on either device.
- **PID 4 (`System`) accrues Copy-engine time** — 2.03 s on the NVIDIA arm — doing paging on behalf
  of whoever faulted. It is neither ours nor a stranger's, and counting it foreign would make the
  detector fire on every run. It is reported in its own class.
- **On a hybrid laptop the panel hangs off the iGPU**, so the compositor is on the Intel adapter
  permanently. That gets its own verdict, `FOREIGN_GPU_WORK(display)`, because a condition that can
  never be cleared should say so rather than look like bad luck.

### 16.2 The verdict a tenancy-without-clock record gets, and why it is not a pass

A record with tenancy and no clock is **half of obligation 8**, and the temptation it creates is
the one amendment 2 was written against: *absence is never a waiver*, and a partial record that
certified would be a worse loophole than an empty one **because it looks like diligence**.

Tank's five-state discipline applies — *bypassed*, *all-rejected* and *unobservable* were three
different things sharing one `0` — so this gets its own name at both levels:

| level | state | meaning |
|---|---|---|
| companion record | **`TENANCY_ONLY`** | tenancy observed over the window; clock axis has no producer |
| certification | **`UNCERTIFIED(partial_companion)`** | not `QUOTABLE`, not `UNCERTIFIED`, not `WITHHELD` |

**Why it can never be a pass, in one line: the failure the clock record exists to catch is a
sole-tenant failure.** `base_b` was **verified sole tenant** and **21.4× wrong** — 246.735 ms —
with the project's second-best RSD, because the board never left 210 MHz. A tenancy-only companion
attached to that run says `SOLE_TENANT` and is *correct*, and the figure is still wrong by 21.4×.
There is a test that says so: `test_the_21_4x_wrong_run_would_pass_a_tenancy_only_companion_and_must_not`.

**And engine `Running Time` cannot be pressed into the clock role either.** It is a *duration*: at
a lower clock the same kernel occupies the engine **longer**, so it moves the same way as the
GPU-busy figure it would be certifying. It is a second copy of the quantity under certification
taken through a different API — **not a second quantity from outside the series**, which is what
§10.0.1 R9 amendment 5 requires. Feeding it in would reproduce the same-source falsifier one level
up, which is how 246.735 ms got into the record in the first place.

#### 16.2.1 The asymmetry that makes a half companion safe to have at all

R9 amendment 5's question is not *is the check sound* but **which way does it move when its subject
is wrong**. A tenancy-only record moves one way only:

> **`FOREIGN_GPU_WORK` without a clock record is still a detection** — no clock reading would
> unobserve the foreign work, so the figure is `WITHHELD`. **`SOLE_TENANT` without a clock record
> is not a pass.** This instrument may subtract confidence and may never add it.

That asymmetry is implemented, not merely stated: `device_state.compose` routes an observed
condition to `FOREIGN_GPU_WORK` regardless of the clock axis, and routes a clean tenancy axis with
no clock to `TENANCY_ONLY`. So the Windows producer *is* usable on Intel today — for refusing
figures, which is the half of the job that was never being done there at all.

#### 16.2.2 Three bugs, one shape: when this instrument is wrong, it reads clean

Every failure met while building it produced a **cleaner** record, not a noisier one. This is the
anti-correlated shape R9 amendment 5 names, and it is why none of the three was fixed by tightening
a threshold:

1. **The LUID join went through the wrong device ordering.** `ONNXRUNTIME_EP_VULKAN_DEVICE` indexes
   the EP's best-first order (0 = NVIDIA, 1 = Intel); `vulkaninfo` enumerates the other way
   (0 = Intel, 1 = NVIDIA) — a trap `bench/devices.py` documents and I walked into anyway. The
   sampler watched an adapter the workload never touched and reported a clean `SOLE_TENANT`.
2. **Ancestry resolution cost 21.2 s per round**, walking `psutil` parents for all 204 instances on
   the adapter. A 62 s window got **three** samples, and reported clean about a run it had not
   watched.
3. **PDH caches its instance list per process.** A sampler that opened its query before a job
   started never saw that job — including our own worker, invisible for a whole 60 s run. Clean
   again.

Fixed at the source (EP-order join, cached ancestry for PIDs that actually did work, forced
`PdhEnumObjects` refresh), and backstopped by two interlocks that make the clean reading
*unavailable* rather than merely unlikely:

- **`UNOBSERVABLE(self_not_witnessed)`** — a window with a declared owner in which **our own work
  never appeared on the sampled adapter** is not a tenancy record about our run. Not a detection,
  not a pass. A record must carry positive evidence that it watched the right device.
- **A blind-gap limit** — a sampler that went dark for more than 10× its interval inside the window
  is `ERROR(instrument)`. It fired for real on the first NVIDIA corroboration run (12.6 s gap, four
  samplers each re-enumerating independently) and refused the record; the fix was one shared
  enumeration, not a looser limit.

### 16.3 RULING — an Intel device-clock figure cannot be certified on this hardware, and that is permanent

Following Link's ruling that lavapipe's device-state record is `none_structural` — *permanent
rather than pending, because there is no subject to measure* — the Intel case needs its own
classification, and it is a different one:

> **Intel clock axis: `none_available` — permanent on this machine, and not for want of looking.
> There is no subject-side problem (the Iris Xe has a real clock that really varies); there is no
> producer for it, and the producers that exist are structurally the wrong kind of quantity.**

The search, so the ruling can be reopened by anyone who finds what I did not:

| candidate | outcome |
|---|---|
| `\GPU Engine`, `\GPU Adapter Memory`, `\GPU Local/Non Local Adapter Memory`, `\GPU Process Memory` | enumerated; **no MHz counter in any of them** |
| `root\wmi` GPU/graphics classes | none exist on this host |
| `Win32_VideoController` | name, driver version, RAM, refresh rate — **no core clock** |
| `nvidia-smi` | NVIDIA-only; exits 6 on the Intel board (`UNOBSERVABLE`, §15.2) |
| Vulkan itself | no clock query in core or in any extension we admit |
| engine `Running Time` as a proxy | **inadmissible** — a duration, same-source, see §16.2 |

**So: no.** On this hardware, an Intel device-clock figure is `UNCERTIFIED(partial_companion)` at
best, forever, and `phases.gpu_steady_tail` will keep refusing to release one. Reopening it needs a
*producer*, not a better analysis: a vendor telemetry service, an elevated driver interface, or an
Intel-supplied tool that reports GT frequency. That is a Link question (platform enablement), not a
Niobe one, and it is not on M0's path.

**What this tells Switch about the 4.39× residual — which is the actionable half of the ruling.**
The residual must be attacked with **counts and shapes**, not with clocks: dispatch counts, bytes
moved, occupancy, instruction mix, barrier counts — the quantities §10.0.4 prefers anyway, because
they are invariant under load, clock state and tenancy. A clock-based Intel argument cannot be
certified no matter how carefully it is measured, so it should not be *taken*. This is the same
conclusion §10.0.4 reached from the other direction: *ask at drafting time whether there is a count
that answers this.* On Intel there is now no alternative to asking.

**And the direction this cuts is the uncomfortable one.** Morpheus's amendment 2 noted that the
iGPU shares its power budget with loaded CPU cores, so it is *more* exposed to the clock failure
than the discrete board — and it is the device where we can never see it. The Iris Xe's
`NO_STEADY_TAIL` refusals (§13.4.3, §14.1) are consistent with a wandering clock and are **not
evidence of one**; they are the tail gate refusing, not the clock gate reporting.

### 16.4 The record stays portable where no producer exists

Morpheus's amendment 1 requires the companion be **a record, not a tool**. Windows performance
counters are as locked to Windows as `nvidia-smi` is to NVIDIA, so adding a second Windows-shaped
producer risks the artifact becoming Windows-shaped. The corollary he did not have to write is now
implemented:

> **The record is two independent axes — tenancy and clock — each with its own producer, its own
> verdict and its own silence set; and it is emitted in full on every platform, with `NO_PRODUCER`
> in the axes nothing can fill.**

`device_state.compose(tenancy, clock)` builds it, `device_state.from_nvidia_record` recasts the
NVIDIA-era record into the same shape (both axes, one producer), and a host with neither producer
emits the same keys with `NO_PRODUCER` and a reason. A missing key is indistinguishable from a key
nobody thought to write; a `NO_PRODUCER` axis is a statement. `test_the_record_is_emitted_in_full_where_no_producer_exists_at_all`
holds that line, and the Windows-only tests skip rather than fail off-Windows.

The composite verdict is derived, never borrowed:

| tenancy axis | clock axis | composite |
|---|---|---|
| `SOLE_TENANT` | a clock series | `SOLE_TENANT` (full companion) |
| `SOLE_TENANT` | `UNOBSERVABLE` / `NO_PRODUCER` | **`TENANCY_ONLY`** |
| `FOREIGN_GPU_WORK` | anything | `FOREIGN_GPU_WORK` (detection survives) |
| `UNOBSERVABLE(*)` | anything | `UNOBSERVABLE` |
| `ERROR(instrument)` | anything | `ERROR(instrument)` |

### 16.5 Corroboration, since one was available for once

On the RTX 4060, where both producers exist, they were run over the same window and both said
`SOLE_TENANT` (`agree: true`, `bench/results/wingpu-nvidia-dev0.json`) while the board peaked at
2010 of 3105 MHz. That is obligation 7's shape — two independently-authored instruments on one
question, with the agreement **in the artifact** rather than in someone's memory of both — and it
is now recorded automatically in the `corroboration` block of every NVIDIA companion record.

Note what it does and does not license. It is evidence about the **tenancy** axis only. Neither
instrument says anything about the other's clock reading, and the agreement of two tenancy
instruments does not make a tenancy-only record on Intel any closer to quotable.

### 16.6 Instruments — what goes red if §16 is false

| claim | instrument | red when |
|---|---|---|
| the counters witness our work on Intel | `probe_wingpu.py` `capability.our_work_seen_on_target` | our engine time on the target adapter is 0 |
| the LUID join is real | `capability.negative_control_holds` | our engine time is non-zero on another adapter |
| a half companion never certifies | `test_a_half_companion_never_certifies` | `UNCERTIFIED(partial_companion)` becomes quotable |
| the 21.4× run is not rescued by tenancy | `test_the_21_4x_wrong_run_would_pass_a_tenancy_only_companion_and_must_not` | that figure returns |
| a half companion can still refuse | `test_a_half_companion_may_still_refuse` | a detection is downgraded by a missing clock axis |
| the record exists without producers | `test_the_record_is_emitted_in_full_where_no_producer_exists_at_all` | an axis key goes missing instead of `NO_PRODUCER` |
| the sampler watched the window | `win_gpu_counters` blind-gap limit | a >10× interval gap is reported as tenancy |
| the sampler watched the right device | `UNOBSERVABLE(self_not_witnessed)` | a clean record is emitted for an adapter we were not seen on |

---

## 16. The census is good news about one failure mode and contains a counterexample to itself (2026-08-01 evening)

Switch's `bench/results/probe_frames.py` (merged at `c6cc0f3`) reconciled his ~70% kernel-time
spread against my `STEADY` verdicts and returned `SAME_FRAME_ORDERED_SELECTION`: same series,
different selection. My tail is a **suffix** of the per-inference GPU-busy series; his spread was
the **whole** of it. **No conflict was ever constructible.** His probe calls `phases.gpu_steady_tail`
and `phases.attribute_gpu_ordinally` directly rather than reimplementing my frame, which is why
this is a reconciliation and not a third opinion.

I reproduced the 28-trace census off the committed artifact rather than accepting the summary:
**9 traces carry whole-series RSD >= 30% and not one publishes a tail figure; the 12 that publish
top out at 10.36%.** Disjoint, with a gap from 10.36% to 34.39%.

### 16.1 The line worth sitting with

In that table, `trace_gemv_contended_dev0.json` — **129.51% whole-series RSD, a 10.56x spread, the
most disturbed run of all 28** — has a tail RSD of **0.1067%. That is the third tightest of the
twenty-eight.** Tighter than `baseline_certified` at 0.1163%. Tighter than `solo`. Tighter than
`warmup`, a run that stayed within 1.01x of itself end to end.

It did not publish, because it graded `MARGINAL_TAIL` at 17.39% coverage and **`MARGINAL_TAIL`
withholds its median**. Had it published, **we would have certified our dirtiest run on the
strength of the tightest-looking number this project has ever produced.**

That withholding now has `bench/test_marginal_tail_withholds.py` behind it, carrying this artifact
in its docstring, because it looks like ordinary conservatism and is not. **The tightness of a
short tail is evidence about the length of the window, not the state of the device.** Any series
that moves contains a short stretch that does not, and the more disturbed the series the flatter
its flattest stretch will be. **Selecting the flattest 17% of a 10x-spread run and reporting its
RSD is a search, and the RSD of a search result is not the RSD of a measurement.**

### 16.2 Is the boundary in the right place? Two answers, and the second is the one that matters

**The dispersion answer: no hole is demonstrable, and no clean bill of health is either.**

`bench/results/probe_tail_boundary.py` computes Switch's same-ordinal-across-inferences RSD for
**all 28** traces (he had it for two) and cross-tabulates it against every tail verdict. At a 5%
threshold, 7 publishing traces flag; at any threshold inside the census's only gap
(10.48%–39.60%), zero do. **I have published the whole sensitivity table rather than a threshold,
because I set it at 5%, got 7, and my instinct was to move it until the number was 0.** That is
choosing the answer, and it is the same instinct that cost us the day.

But the signal is **not independent of the thing it is checking**: Spearman **0.903** against
whole-series RSD, r=0.964 on logs, median ratio 1.128, and both series gap in the same place. That
is mechanical — **a disturbance that scales a whole submission moves every dispatch inside it
together**, so the *k*-th dispatch's spread across inferences reproduces the inference sums'
spread. Same-ordinal RSD is a *re-selection* of the quantity it was meant to audit. *(Consequence
for Switch's load guard: it will fire on roughly what whole-series RSD already fires on. Not
useless — being per-dispatch it can localise where the sum cannot — but "tail and guard agree" is
close to a tautology and must not be reported as independent confirmation.)*

**The answer that matters: yes, there is a run that grades `STEADY` and should not have, and it is
in this census.**

`trace_gemv_baseline_certified_dev0.json`:

| measure | value | rank among 28 |
|---|---|---|
| whole-series RSD | **0.12%** | **cleanest** |
| tail RSD | **0.1163%** | 4th tightest |
| same-ordinal RSD | **0.36%** | **cleanest** |
| tail verdict | `STEADY`, n=46, **coverage 100%**, zero discarded | — |
| its paired figure | **246.72 ms** against a true ~11.5 ms | **21.4x wrong** |

**The steadiest trace in the census on all three dispersion measures, wrong by a factor of
twenty-one.** No boundary placement reaches it — it publishes at n=46 and 100% coverage, so no `n`
floor, no coverage floor and no RSD bar can touch it. **This is not a misplaced boundary. It is the
bias blindness, and every dispersion measure we own certifies it.** The only instrument in the set
that refuses it is `bench/device_state.py`, peak SM 210 MHz against a 3105 MHz board maximum,
because its evidence comes from **outside the series**.

So the census's headline — *"the instrument refuses precisely under the condition that would have
made it wrong"* — is true **of dispersion** and contains its own counterexample for bias. Both
belong in the record.

### 16.3 How thin the certified base is

Of the 28 traces, **12 publish a tail figure. Of those 12 my certification gate admits 2**
(`soloA`, `after_coldboard`), **withholds 1** (`baseline_certified`), and leaves **9 `UNCERTIFIED`
for want of any companion at all.** The disjointness of §16 is a statement about dispersion; the
certified base beneath it is two traces. That is the honest size of what we currently know.

### 16.4 Nothing is withdrawn, and the reason is a correction rather than a defence

The coordinator asked whether the **12.1847 ms/inference** NVIDIA figure needed retracting, on the
belief that it came from `baseline_certified`. **It did not.** Two artifacts were conflated:

- **12.1847 ms** is from `bench/results/phi35-certified-dev0.json` — `kind: phi35`, the
  **Phi-3.5-mini** model, 355 nodes in one island, `model_output_equivalence MATCH`, companion
  `SOLE_TENANT`, peak SM **2010 of 3105 MHz**, certification **`QUOTABLE`** (§15.3).
- **`baseline_certified`** is a **gemv microbenchmark** trace whose figure is **246.7195 ms** — the
  idle-clock run, `WITHHELD` by the companion.

Different model, different probe, different number. The standing caveat on 12.1847 ms is unchanged
and is not this one: **it is comparable to nothing yet.**


---

## 17.0 The harness's import surface — a name collision, and the screen that now covers it

`bench/test_marginal_tail_withholds.py` and `ci/test_lane_checks.py` each passed alone and produced
three failures together. The diagnosis I was handed was an unrestored `sys.path.insert`; the actual
mechanism was a **module name collision**, and the distinction matters because the prescribed fix
would have left the defect live.

The failure text, which is the evidence (R13 — quote the text, never the count):

```
AttributeError: module 'device_state' has no attribute 'certifies_comparison'
AttributeError: module 'device_state' has no attribute 'lavapipe_note'
```

`AttributeError`, not `ImportError`: the module was found and it was the wrong one. `import`
consults `sys.modules` before `sys.path`. Two files were named `device_state.py` —
`bench/device_state.py` (this document's mandatory companion, §15.2) and `ci/device_state.py` —
so whichever imported first bound the name process-wide. Restoring `sys.path` to a byte-identical
copy of the original list does **not** fix it; that was measured, not assumed.

**Fix.** `bench/device_state.py` is now **`bench/device_companion.py`**. Every reference in §15 and
§16 to `bench/device_state.py` means this file; the rename changed the `sys.modules` key and
nothing else. **No verdict, threshold, artifact or number in this document moves**: the eight import
sites bind it as `import device_companion as device_state`, `BOOST_FLOOR` is unchanged, and the certification
tests that grade the committed artifacts (`bench/test_device_state.py`, 15 tests, built on Switch's
`gpustate_*.json`) pass unchanged against the renamed module.

**Screen.** `bench/test_import_isolation.py` — no two flat-imported modules under `bench/`,
`bench/results/`, `ci/`, `tests/ops/`, `rust/tools/` may share a base name, and no library module
under `bench/` may leave `sys.path` mutated. Both carry a positive control that feeds them a planted
violation, because an always-green screen and an absent screen are the same artifact. The leak half
found three real violations on its first run — `bench/cases.py`, `bench/island_attribution.py` and
`bench/transfer_calibration.py` all left `tests/ops` permanently at the front of `sys.path`; one of
the three inserted it and never used it. All three fixed.

**Why this is in a performance document.** A harness whose modules silently resolve to each other's
files is an instrument whose identity is not determined by its own source. That is the same class as
§15's central finding — an instrument that cannot see the thing it is being read for — one level
down, in the import graph rather than in the statistics. And per §10.0 the failure of an instrument
is `ERROR(instrument)`, never a detection: the first cut of the leak screen purged `sys.modules` and
hard-faulted the interpreter, which is the screen breaking, not a defect found, and it is recorded
that way.

Verified as one pytest invocation, which is also the only arrangement in which this class is
visible at all: `pytest bench/ ci/ tests/ops/` → **402 passed, 321 skipped, 0 failed**. Under
per-directory runs each step passes with the defect fully present.
