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
| `upload` | Staging-buffer write + transfer of weights and inputs to device-local memory. | Host time *plus* a real transfer. Counters carry bytes and effective GiB/s. **Summary-only: no `vulkan.upload` span is emitted** — the engine folds this in through `record_transfer`. It happens *inside* the `record` bracket. |
| `record` | Filling the command buffer: binds, push constants, barriers, dispatches. | Real host CPU work. Per `ENGINE.md` §6.1 this is record-once/replay-many, so a steady-state inference should show *no* `record` span at all — if it does, something is invalidating the recording. **This engine does not implement that model** (see `RecordPath` below), so `record` is paid on every `Compute` call. |
| `submit` | `vkQueueSubmit` itself. | **Almost nothing.** See §1.3. |
| `fence_wait` | Waiting for the submission's fence. | An **upper bound** on GPU execution, inflated by queue contention, other clients' work, and driver scheduling. Not kernel time. |
| `readback` | Device→host transfer of outputs. | Host time plus a real transfer. Bytes + GiB/s counters. **Summary-only: no `vulkan.readback` span is emitted**, and it is **not** inside `record` — production reads outputs back after the record bracket closes, after `vkQueueSubmit` and after the fence wait, because the outputs do not exist until the GPU has drained. Corrected 2026-08-09 (issue #88); `Phase::Readback::nested_in()` previously claimed `record` as a parent. |

Two further span-adjacent facts we record because they are the ones that mislead people:

* **`RecordPath`** — `first_record` / `replay` / `rerecord` / `recorded_again`. This is our
  analogue of MLX's cache HIT/MISS/RETRACE. It answers "did this inference reuse the command
  buffer, or did we rebuild it?" A shape key never seen before for a given subgraph is classified
  `rerecord`, not `replay`, even if a recording existed — resolving against a *set* of seen keys
  rather than a single last-key, which is the lesson the MLX tracer learned the hard way about
  alternating shapes. **Wired 2026-08-09 (issue #88) and the answer is uncomfortable:** this engine
  holds no cached command buffer, so every `Compute` after the first for a given subgraph is
  `recorded_again` — a fourth path, distinct from `rerecord` because no shape key changed — and
  `replay` is **zero by construction**, not zero because replays are rare. The counters publish
  `record_path_state`, which reads `UNOBSERVED` rather than `0` before anything has been recorded.
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

Shares are of **time inside the EP's engine dispatch** — the sum of `vulkan.subgraph` spans.

**Re-headed 2026-08-09 (issue #88).** These tables previously said "time inside `Compute`", and
that denominator was wrong in a specific, mechanical way: `vulkan.subgraph` brackets
`vk::session::dispatch_ort`, which the `Compute` callback reaches only *after* its null, liveness
and binding checks, and which is never entered at all when one of them refuses. The callback body
around it — ORT context reads, those checks, status construction, and the post-return
broken-commitment disclosure — was outside every span in this table and outside every share
computed from it. It is now measured by `vulkan.ort_compute_callback` (§9.11), and the numbers in
this section are shares of the **dispatch**, not of `Compute`.

The re-heading changes no measured value. **The runs below were captured before
`vulkan.ort_compute_callback` existed, so the outer residual for them is not merely unknown — it
is unmeasurable, and no figure for it is stated anywhere in this document.** What changed is the
name of the denominator, which is what the row said it was a share of.

It is also **not** process wall time; ORT's graph execution, the CPU EP's nodes between islands
and session setup are all outside it. These shares may not be restated as shares of `Compute`, nor
as shares of the benchmark's wall clock.

The timed pass runs with tracing **off**; the split comes from a separate instrumented pass, and
`tracing_overhead_ratio` (traced median ÷ untraced median) is measured rather than assumed:
1.0207× on NVIDIA, 0.8659× on Intel. The Intel figure being below 1.0 is not negative overhead —
it means the machine state moved between the two passes, and it is reported for that reason.

**NVIDIA RTX 4060 Laptop** — 48563.24 ms inside the **engine dispatch**, 561 subgraph invocations:

| phase | total | share of dispatch | n | median |
|---|---|---|---|---|
| `vulkan.record` | **33456.17 ms** | **68.9%** | 561 | 50.774 ms |
| ├ of which: host **upload memcpy** | **33042.07 ms** | **98.8% of record** | 561 | — |
| └ of which: command construction | **414.10 ms** | 1.2% of record | 561 | 0.459 ms |
| `vulkan.submit` | 308.18 ms | 0.6% | 561 | 0.452 ms |
| `vulkan.fence_wait` | 13980.26 ms | 28.8% | 561 | 27.776 ms |
| **inner residual** (dispatch minus its top-level phases) | 818.63 ms | 1.7% | — | — |
| **GPU kernels (sum)** | **6110.00 ms** | **12.6%** | 5457 | — |

**Intel Iris Xe** — 72148.67 ms inside the **engine dispatch**, 561 subgraph invocations:

| phase | total | share of dispatch | n | median |
|---|---|---|---|---|
| `vulkan.record` | **23946.22 ms** | **33.2%** | 561 | 25.368 ms |
| ├ of which: host **upload memcpy** | **17231.74 ms** | **72.0% of record** | 561 | — |
| └ of which: command construction | 6714.48 ms | 28.0% of record | 561 | 1.393 ms |
| `vulkan.submit` | 11830.53 ms | 16.4% | 561 | 0.349 ms |
| `vulkan.fence_wait` | 34978.91 ms | 48.5% | 561 | 56.652 ms |
| **inner residual** (dispatch minus its top-level phases) | 1393.01 ms | 1.9% | — | — |
| **GPU kernels (sum)** | **31652.94 ms** | **43.9%** | 5457 | — |

Per-kernel GPU time (summed from the per-span `gpu_ns` float, **not** from the integer-µs `dur`
— several of these kernels run in 2–3 µs, where truncation is a 15–30% error over 5457 spans):

| kernel | n | NVIDIA | Intel |
|---|---|---|---|
| `q_gemv_matmul_nbits_f16` | 2737 | 5990.73 ms | 31432.43 ms |
| `skip_simplified_layer_norm_f16` | 1088 | 97.65 ms | 149.10 ms |
| `ew_binary_mul_f16` | 1088 | 14.39 ms | 48.69 ms |
| `ew_unary_sigmoid_f16` | 544 | 7.22 ms | 22.73 ms |

The row formerly headed `unattributed inside Compute` is the **inner residual** and is reported
rather than folded into a neighbouring phase: it is the input-pointer reads, buffer allocation and
descriptor-pool work before recording, plus the readback memcpy and the writes into ORT's output
tensors after the fence. **A phase split whose parts do not sum to the whole should say so.** It
is *not* the outer residual issue #88 asks for, and the two may never be added — see §9.11.

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

### 9.11 Two residuals, two levels, and why they may never be added (issue #88)

Landed 2026-08-09T04:33:34.895-07:00. **This section describes an instrument. It states no speed
figure and no host-cost percentage**, because no run captured with `vulkan.ort_compute_callback`
in the binary exists yet; every number in §9.3 predates it.

The EP now emits three strictly-nested tiers, each self-identifying by `cat` and by a `tier` span
arg so that no analyser has to infer a level from a name:

| tier | span | `cat` | what it brackets |
|---|---|---|---|
| callback | `vulkan.ort_compute_callback` | `ep.compute_call` | the whole `OrtNodeComputeInfo::Compute` callback body, from just inside the `extern "C"` entry point to the moment the status is returned to ORT. Carries `outcome` ∈ `ok` / `failed` / `unresolved`. |
| dispatch | `vulkan.subgraph` | `ep` | `vk::session::dispatch_ort` only. |
| phase | `vulkan.record`, `vulkan.submit`, `vulkan.fence_wait`, … | `ep.phase` | one phase inside a dispatch. |

From which there are **two** residuals. They are residuals of **different intervals against
different denominators**:

* **outer** = `Σ vulkan.ort_compute_callback − Σ vulkan.subgraph`, over the callback total. What
  the callback body costs *around* the dispatch.
* **inner** = `Σ vulkan.subgraph − Σ (top-level `ep.phase` spans)`, over the dispatch total. What
  the dispatch costs *around* its own phases. This is the row §9.3 used to call "unattributed".

**They are not two halves of one quantity and their sum names nothing.** The first attempt at this
work shipped an analyser that subtracted the phase spans from the callback total and published the
result under the outer residual's definition; that number is `outer + inner`. On a
1000 µs callback / 900 µs dispatch / 700 µs phase case it reads **300 µs where the outer residual
is 100 µs**. `bench/test_compute_attribution.py` pins exactly that case and asserts that no value
equal to 300 appears anywhere in the artifact, under any key.

`bench/phases.py::two_level_attribution` computes both or neither, and has three terminal states:
`PASS`, `VACUOUS` (the trace carries no callback span at all — it predates the instrument, which is
**not** evidence that the callback body is free) and `REFUSED`. It refuses on: an `ep.phase` span
whose name this module cannot name; a missing/negative `dur`; a missing or wrong `boundary`/`tier`
arg; overlapping or nested callback spans; a dispatch that escapes every callback; a phase that
escapes every dispatch; a dispatch total exceeding the callback total; a phase total exceeding the
dispatch total; and an `outcome` of `unresolved`. On a refusal **every percentage key is absent
from the artifact rather than zero**, so a refusal cannot be formatted as `0.0%`. A callback whose
outcome is `failed` is disclosed and never dropped: excluding it would shrink the denominator by
exactly the calls most likely to be slow.

**Where the timer sits, precisely.** The guard opens in `ep::compute` — the `extern "C"` entry
point — immediately after the compute-info pointer is resolved, and is dropped explicitly after
the post-return broken-commitment disclosure. It therefore covers the FFI status guard, all of
`compute_impl` and the disclosure. It **excludes** the branch taken when ORT hands us a null
`OrtNodeComputeInfo`, because there is nothing to attribute in that case and no subgraph identity
to attribute it to. The span is scope-based; **no claim is made about a number of return paths.**

**Portability.** The tracing/counters instrumentation is device-independent and requires nothing
above Vulkan 1.1: the callback boundary is host-side `Instant` arithmetic on the same microsecond
axis every other host span uses, and the four new resource counters are incremented from the
existing call sites. No new Vulkan feature, extension or limit is required or queried. The
authoritative, portable proof that the counters only move on success is the host-free seam
mutation battery in `counters.rs` (real `None`/`Some` values through the seam plus ten held-out
mis-wired reimplementations). A best-effort live-device corroboration in `vk/pipeline.rs::tests`
additionally attempts to exhaust a real `max_sets(1)` descriptor pool, but that attempt is *not*
portable in the same sense: Vulkan 1.1 §14.2.3 permits, but does not require, an implementation to
fail such an allocation, and Linux lavapipe (Mesa 26.1.3) legitimately accepts it while Windows
lavapipe refuses it. When exhaustion is not observed, the test reports a loud inconclusive result
instead of asserting spec-unguaranteed behaviour.

**Limitations, stated.** (1) No run with this instrument in the binary has been captured, so no
outer residual has been measured on any device and none is quoted. (2) The outer residual is a
*wall-clock* residual on the calling thread; it says where time went, not why. (3) A trace captured
before this landed is `VACUOUS`, and that is a different fact from an outer residual of zero. (4)
The four resource counters (`descriptor_pools_created`, `descriptor_sets_written`,
`command_buffers_allocated`, `queue_submits_completed`) count **successes only** and are recorded
as `uncensused` in `ci/census_surface_map.json` — no wiring-census mechanism reads them yet.

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


---

## 18.0 What foreign GPU work does to the tail — a second hole, larger than the first

§15 established that `gpu_steady_tail` cannot see a **bias**: `baseline_certified` is the cleanest
trace in the census on every dispersion measure and is 21.4x wrong. This section establishes a
second and more ordinary failure mode, and disposes of a proposed in-band remedy.

### 18.1 `PER_DISPATCH` is not a foreign-work signature

The proposal was that `run_disturbance.localise`'s `PER_DISPATCH` character — dispatches disagreeing
more than the inference totals do — is what foreign GPU work produces, and would therefore be the
first in-band signal sensitive to something dispersion is normally blind to. Two traces supported
it. Nine device-0 traces carry both a localisation and a committed tenancy companion, and over all
nine, sorted by `explained_by_level`:

| trace | tenancy | `explained_by_level` |
|---|---|---|
| `contended3` | **FOREIGN_GPU_WORK** | -0.2265 |
| `base_b` | SOLE_TENANT | -0.1039 |
| `baseline_certified` | SOLE_TENANT | -0.0119 |
| `after_coldboard` | SOLE_TENANT | +0.5227 |
| `soloA` | SOLE_TENANT | +0.5413 |
| `ab_p1_r1` | SOLE_TENANT | +0.6804 |
| `ab_p0_r1` | **FOREIGN_GPU_WORK** | +0.8408 |
| `contended` | **FOREIGN_GPU_WORK** | +0.8638 |
| `ab_p1_long` | SOLE_TENANT | +0.8895 |

The witnessed-foreign traces occupy the extreme bottom **and** near the extreme top; `contended`, at
`foreign_sample_fraction = 1.0`, is more submission-level than four of six sole-tenant traces. The
two false positives are the two idle-clock 21.4x specimens, so a guard on this signal would have
fired on the runs whose defect was clock, not tenancy.

**No cut works and none is chosen.** Every cut from -0.50 to +1.00 is swept in
`bench/results/tenancy_signature.json`; the best achievable is Youden's J = +0.333 (7 of 9). The
classes interleave, so this is a signal that does not carry the distinction rather than a boundary
in the wrong place. Published in full, tuned nowhere — the same discipline as the 5%-flag sweep.

Qualification, against my own conclusion: n = 9 with 3 foreign is small, and the decomposition does
separate two real mechanical conditions. What is falsified is the mapping *low `explained_by_level`
-> foreign GPU work*.

### 18.2 The level moves; the verdict does not follow

Against a sole-tenant, boosted-clock reference of **11.5248 ms** (`soloA` and `after_coldboard`, the
only two that published — a withheld `MARGINAL_TAIL` median is never a denominator):

| trace | tenancy | tail verdict | level | vs reference |
|---|---|---|---|---|
| `contended` | FOREIGN_GPU_WORK | `MARGINAL_TAIL` (withheld) | 11.7697 ms | 1.021x |
| `ab_p0_r1` | FOREIGN_GPU_WORK | `MARGINAL_TAIL` (withheld) | 20.6159 ms | 1.789x |
| `contended3` | FOREIGN_GPU_WORK | `NO_STEADY_TAIL` | — | — |

So the tail is **not** level-blind to contention the way it is level-blind to uniform bias.
But the verdict is computed from dispersion, and a sustained, steady foreign load is steady:

```
contended3 truncated to 20   STEADY  126.6465 ms  RSD 0.9103%
contended3 truncated to 28   STEADY  126.6465 ms  RSD 0.8035%
contended3 truncated to 34   STEADY  126.6758 ms  RSD 0.7915%
contended3 full (62)         NO_STEADY_TAIL
```

**The full-length refusal is a property of how long the run was, not of the instrument's
sensitivity.** The ruling:

> **`gpu_steady_tail` detects foreign work only through its non-stationarity, never through its
> magnitude. A steady foreign load is indistinguishable from a slower GPU.**

This is strictly larger than the bias hole. Uniform bias needed a board pinned at idle clock — rare,
and recoverable from magnitude because the two clock regimes are 21x apart and do not overlap. A
sustained foreign tenant is ordinary and produces a wrong level at any magnitude, so no regime
argument rescues it.

### 18.3 The companion is load-bearing for both holes

All four `contended3` readings above — including the three publishing a confident STEADY at 10.99x
wrong with sub-1% RSD — are refused by `device_companion.certify` as `WITHHELD`, on
`foreign_sample_fraction = 1.0`. That evidence comes from **outside** the series and does not care
how steady the series looks. Obligation 8's companion is therefore not corroboration; on this
specimen it is the only thing between us and a certified 126 ms, and it now covers two distinct
failure modes rather than one.

One improvement, attributed correctly: `contended` was previously a confident `STEADY` at n=8
sitting 2.1% high; it now grades `MARGINAL_TAIL` and withholds, which is the n>=8 / coverage>=50%
floor working. It is simultaneously the clearest demonstration that the floor is **necessary and
not sufficient** — `contended3` truncated satisfies every floor and publishes 10.99x wrong.

### 18.4 Provenance, and two instrument errors of my own

No new measurement was taken and no binary was exercised, so no figure here has a binary frame;
saying so is better than a rebuild that implies otherwise. Everything is recomputed from committed
artifacts through code at this commit — Switch's `localise` and `per_inference_kernel_us` from
`bench/run_disturbance.py`, my `gpu_steady_tail` and `certify` from `bench/phases.py` and
`bench/device_companion.py`. Nothing reimplemented; the probe's value is that both sides are the
same code that produced the figures being compared. Reproduced levels match the previously published
11.5252 / 11.7697 / 126.647 / 126.676 exactly.

Two `ERROR(instrument)` events, recorded rather than quietly corrected because both survived a first
reading of the output:

1. **A 1000x units error.** `gpu_steady_tail` takes `busy_us` and converts internally; I pre-divided.
   **Every verdict, RSD, ratio and classification is scale-invariant and did not move**, which is
   exactly why it was invisible. A units error that changes no verdict is the kind that gets
   published. Caught by checking `soloA` against its independently published 11.525 ms.
2. **A reference built partly from withheld medians**, giving 15.5159 ms — a number that exists
   nowhere. Using a withheld median as a denominator publishes it by the back door, against §16's
   own rule. The reference now names its sources and a test asserts no withheld tail is among them.

Neither is a detection. Locked by `bench/test_tenancy_signature.py` (6 tests).

---

## 19. The first real-node run, and what island fragmentation actually costs

> **Superseded on the current build (2026-08-02, §21).** Switch has since landed GQA: the graph
> fuses to **1 island, 355 of 363 claimed**. The 33-island configuration this section measures no
> longer exists on `main`, so §19.5's fragmentation cost is a record of what fragmentation *cost*,
> not a description of the shipping build. The A/B method — slope, not quotient, both arms on one
> binary — stands, and §19.4's idle-clock specimen is unaffected and is now load-bearing (§20.2).

Frame, per R12's fourth generalisation: `main` at `0baf660`, merged into `squad/niobe` and rebuilt.
DLL `E00C7F8B64B3907A…` before, **`47F668336A7BF6A9…`** after. Every figure in this section was
produced by that binary. Model artifact and harness unchanged.

### 19.1 The certified figure was refused, and the refusal is the result

The measurement ran (`--device 0 --iters 40 --warmup 10 --repeats 2 --trace-iters 3`, ~17 min) and
the gate withheld it. Quoting the failure text rather than a count, per R13:

> ⛔ REFUSED: phi-3.5 timing withheld — machine was CONTENDED during measurement. timed pass:
> out-of-band load survey says CONTENDED — other processes kept **7.73 cores** busy on average
> (threshold 0.5); **100%** of samples exceeded 1.0 foreign busy cores (threshold 10%).
> withheld: vulkan median, cpu median, their delta and ratio.

There is no new device-clock figure. **`12.1847 ms` remains the last quotable one**, at its stated
context length — **zero context, one token** (`present.*` shaped `[1,32,1,96]`). Against Switch's
clock-free roofline that is 67% of the 8.18 ms zero-context floor, headroom 1.49×. The roofline is
not a constant (8.22 ms at zero context, 20.80 ms at 8192), so that comparison is only valid at the
context it was taken at, exactly as a timing is only valid at its device state.

What *is* valid from the refused run, because none of it is a timing: `gpu_span_accounting` ok
(4522 = 4522 = 4522), `trace_matches_counters` ok (462 = 462), `model_output_equivalence` MATCH,
dispatch accounting ok (`compute_calls` 1683 = 33 islands × 51 inferences). One RED:
**`phase_containment` FAIL** — one sub-record span falls outside every `vulkan.record` span, so
**no phase share may be read from this run**. That is new, unexplained, and open.

### 19.2 The idle-clock specimen, and it is ours

This run is the cleanest demonstration the project has produced of why the device-state companion is
required rather than diagnostic, and unlike the earlier ones it is self-produced on the current binary.

| what it says | value |
|---|---|
| device-clock series, `gpu_steady_tail` | **STEADY**, n=13 |
| its median | **245.9149 ms** |
| its RSD | **0.0717%** — tighter than every trace in the 28-trace census |
| tenancy verdict | **SOLE_TENANT**, `foreign_sample_fraction` **0.0** |
| SM clock | **210.0 MHz min, median, mean *and max*** of 160 samples, against 3105 MHz boost |
| `clock_at_max_pct` | **6.8%** |
| ratio to the last quotable figure | **20.18×** |

**Both of the things that sound like a pass, pass.** The tail is STEADY and the board is a sole
tenant. The single field that refuses is the clock record, and `certify()` returns `quotable: false`
on it alone. Every earlier specimen of this pathology involved foreign GPU work, so a tenancy check
could plausibly have caught them; **this one has none at all.** The clock record is the only
instrument in the set with a claim on it.

Note also the corroboration: 20.18× sits inside the idle-clock band Switch identified as 21× and
non-overlapping with the boost regime, so the regime is recoverable from magnitude alone — the same
property that rescued `12.1847 ms`. Two independent instruments agree on the diagnosis.

The causal chain is worth writing down because it is not the obvious one: **host contention showed
up on the device axis as an idle clock, not as GPU contention.** 7.73 foreign cores starved
submission, GPU utilisation sat at 0% median / 3.4% mean, and the board never ramped. A host-side
problem produced a device-side symptom that the tenancy verdict is structurally unable to see.

Locked by `bench/test_island_boundary_cost.py::TestIdleClockSpecimen`.

### 19.3 Fusion did fragment, and it is the declined GroupQueryAttention

Confirmed mechanically **on one binary**, not inferred across builds. `evidence/proof_attempts.jsonl`
carries `com.microsoft::GroupQueryAttention/…/past_key+past_value+cos_cache+sin_cache` as
`DIVERGENT` (worst_rel 16.7264). Handing that key to `ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN` restores
**355 of 363 nodes in 1 island** — exactly the historical record — against **323 in 33 islands** as
shipped. 32 layers × 1 GQA = 32 cuts in a chain = 33 segments. `islands_offered` =
`viable_islands_retained` = 33 and `net_benefit_gate_bypasses` = 0, so nothing was declined at island
level; the fragmentation is entirely upstream of the viability gate.

### 19.4 My first answer was wrong, and its shape is R11

I divided `session_staging_upload_bytes` by the inference count in each of two committed records and
reported that fragmentation had *reduced* staging traffic from 78.43 to 43.59 MiB/inference — a 1.78×
improvement. **It is an artifact.** That counter is cumulative and is dominated by a one-time
~2185 MiB weight upload, and the two records had different iteration counts (28 and 51). Dividing one
fixed cost by two denominators manufactures a ratio of 51/28 = 1.82× out of nothing. I measured 1.78×.
**The giveaway was that the "improvement" equalled the iteration ratio**, and I nearly published it
as a finding with a convincing story attached.

The fix is a **slope, not a quotient**: run each configuration at two iteration counts and difference
them, so the fixed session cost cancels and the recovered constant becomes a *check* rather than an
output. It agrees between arms to 0.034%, which it must, because the weights are the same weights.

### 19.5 The cost, per inference, both arms on one binary

| currency | 1 island | 33 islands | direction |
|---|---|---|---|
| host round-trips | **2** | **66** | **33× WORSE** |
| staging bytes | 0.8170 MiB | 1.5670 MiB | **1.92× WORSE** |
| device allocations | 551.0 | 517.0 | 0.94× — *not worse* |
| dispatches | 355.0 | 323.0 | 0.91× — *not worse* |
| device high-water | 2190.7 MiB | 2184.9 MiB | 0.997× — *not worse* |
| fixed weight upload | 2185.46 MiB | 2184.71 MiB | agree, 0.034% |

The marginal traffic decomposes exactly: **+393,208 B out and +393,216 B back**, against
**393,216 B** of KV tensors (64 × `[1,32,1,96]` f16). Fragmentation's entire byte cost is *one extra
host round-trip of the KV tensors*. The readback side is exact and the upload side carries an 8-byte
residual, recorded rather than rounded away.

That this closes is why it is trustworthy rather than suspicious: the 393,216 B comes from the ONNX
output shapes, not from any counter in the records, and the fused arm's readback slope lands on the
model's declared outputs (logits 64,128 + KV 393,216 = **457,344 B**) to the byte. An identity whose
two sides come from the same source cannot fire; these two sides do not.

### 19.6 Ruling

**786,424 B/inference is 0.0376% of the weight read.** In bytes, fragmentation is negligible against
the roofline, and **Switch's shape holds: a large count of small things is not a large thing.** The
allocator-round-trip and descriptor-rebinding hypotheses are **falsified**, not unmeasured — those
counters went slightly *down*.

What remains genuinely unpriced is **32 extra submissions, fence waits and pipeline drains per
inference**. That is a count, and counts do not care whether the box is busy; its *time* cost is a
timing and cannot be taken on a contended box. I attempted a host-minus-GPU bound from the trace and
it came out at 10.48 s/inference — vacuous, and additionally computed under an idle clock that
inflates the GPU term 20×. It is reported as vacuous rather than quoted. And per Switch's correction,
min-over-inferences is an **upper** bound, so two of them would not have bounded the difference from
below in any case.

**The cost that is in neither column, and is probably the large one, is that the 32 declined
GroupQueryAttention nodes now execute on CPU** (`executed_by` = 120 CPU / 99 Vulkan). That is an
execution-location cost, not a boundary cost, it is not in this table, and it is the next thing to
measure. The A/B harness for it now exists and is counts-valid; its timing arm needs a quiet box.

Caveat on that arm: the 1-island configuration runs a form whose proof verdict is `DIVERGENT`, so it
is valid for counters and **not** for correctness.

Reproduce: `python bench/results/probe_island_boundary_cost.py`. Locked by
`bench/test_island_boundary_cost.py` (13 tests).

---

## 20. Standing policy: this box is contended, and that is now the baseline

Frame: `main` at `c1522e2`, merged into `squad/niobe` and rebuilt. The DLL hash is **unchanged**
at `47F668336A7BF6A9…` — the merge carried no Rust change — so §19's figures remain in frame.

The GPU is shared with another team for the foreseeable future. Everything below follows from that
one fact, and it is written as policy rather than as this session's circumstance because the
circumstance is not going to end.

### 20.1 A wall-clock figure is `STEADY_UNCERTIFIED` by default

7.73 foreign busy cores against a 0.5 threshold is the **normal state of this machine**, not an
anomaly. Consequences:

- **`STEADY_UNCERTIFIED` is the expected verdict** for a wall-clock figure on this hardware. The
  companion refusing is the instrument working. It is not a blocked task and it does not need
  escalating.
- **No plan may contain the step "take the measurement when the box settles."** That step never
  completes. A plan that depends on it is not a plan.
- A quotable figure requires the companion **and** a device-state record showing the board was at
  boost. That combination is now rare and **must not be waited for** — see §20.4 for how to catch
  one opportunistically instead.
- **`12.1847 ms` — zero context, one token — remains the only quotable figure this project holds,
  and is likely to remain so.** It is not withdrawn: R13 withdrew the wall-clock *ratios*, taken
  during CPU fallback, which compared the CPU EP against itself. Different instrument, different
  quantity, its own companion attached.

### 20.2 The idle-clock specimen is the argument for obligation 8, not an illustration of it

§19.2 records it; this is why it is load-bearing. On the current binary, sole tenant, **zero**
foreign GPU work:

| | |
|---|---|
| device-clock tail | **STEADY**, median **245.9149 ms**, RSD **0.0717%** — tighter than all 28 census traces |
| tenancy | **SOLE_TENANT**, `foreign_sample_fraction` 0.0 |
| SM clock | **210.0 MHz across min, median, mean *and max*** of 160 samples, boost 3105 MHz |
| truth | **20.18× wrong** |

**Both instruments that sound like a pass, pass.** Only the SM-clock record refuses.

The mechanism generalises to exactly the regime we are now permanently in: **host contention
arrives on the device axis as an idle clock, not as GPU contention.** Foreign cores starve
submission, the board sees too little work to ramp, and it sits at idle — which is *perfectly
steady*, so it yields the gate's most confident possible verdict. Every future reader of a timing
on this project is in that regime. A low clock does not raise RSD; it lowers it.

### 20.3 Performance work runs on counts, and the ceiling is now first-class

Counts do not care whether the box is busy. §10.0.4 already prefers the count and Morpheus added
*prefer the bound you can sign*; on this machine those stop being preferences and become the only
instruments that work.

Switch's byte model needs no clock and produced more usable direction in one round than every
timing attempt of this session. It is therefore promoted from probe output to **`bench/ceiling.py`**,
with the three things the certification apparatus already has:

- **a frame** — model, device, spec source, byte-model provenance, and the **DLL hash**, because
  for a test result the frame is the binary that ran it;
- **an extent** — the contexts at which the bound may be quoted;
- **a refusal** — `UNOBSERVABLE`, never a number, and pointedly never `0`.

**The extent is `[0]`, and the reason is not the one expected.** The obvious refusal would be a
context nobody ran. The real one is that **`GroupQueryAttention` is declined on this build and
executes on CPU** — proof verdict `DIVERGENT`, all 32 instances, which is why the graph carries 33
islands and 323 of 363 claimed nodes (§19.3). **GQA is the op that reads the KV cache.** So on this
build the KV-cache bytes are not GPU DRAM traffic at all, and `island_bytes_phi35.json` charges them
to the GPU roofline anyway — 48 MiB at past_len 128, rising to 3072 MiB at 8192 where they become
60.5% of the modelled stream. Against this build that is a bound on a machine we are not running.

R12: a term whose event cannot occur in the frame reports `UNOBSERVABLE`. **Not `0`** — `0` would
claim the traffic is free, and it is not free, it is elsewhere. The refusal still discloses the
number it declined to publish, so it is auditable rather than merely opaque.

The consequence cuts the right way:

> **At `past_len = 0` the KV term is exactly zero, the question does not arise, and the bound is
> admissible. That is the only context this project has ever run, and `12.1847 ms` — zero context,
> one token — is the only quotable figure we hold. The one admissible bound and the one quotable
> figure sit at the same context.**

That is why the comparison survives, and it should be read as the reason rather than as luck.

`Ceiling.compare()` refuses a comparison whose contexts differ, and a quotable one carries
`past_sequence_length` forward in its own output. The roofline is **not a constant** — 8.22 ms at
zero context against 20.80 ms at 8192 — so a fraction-of-roofline quoted without a context is the
same defect as a timing quoted without its device state.

**One number moved and it is not a result.** Switch quotes 67.1% of roofline against the
weights-and-scales stream (8.179 ms). `ceiling.py` quotes **67.4%** against the by-context total,
which also carries the 9.52 MiB of intermediates (8.218 ms). Same figure, two floors differing by
0.5%; both are correct about their own quantity and neither supersedes the other.

Falsifier for the extent: a staging measurement at `past_len > 0` showing KV-scale bytes crossing
to the device. We run at zero context, so no such measurement exists. If GQA is ever claimed, the
extent widens — and that is a code change, not a judgement call.

### 20.4 The clock record keeps running when nothing is being certified

**`bench/clock_log.py`** samples tenancy and SM clock continuously and appends to a JSONL, reusing
`probe_gpustate`'s sampler rather than adding a second dialect for the same channel. `window()`
assembles a past interval into the shape `device_companion.certify()` already consumes.

This inverts the workflow. The old one — notice the machine is quiet, start a run, sample alongside
it — has a step that never completes. The new one records continuously, so when a run happens to
land in a genuinely quiet minute **the companion for it already exists and the figure can be
certified retrospectively.**

Two refusals it must keep: a window with fewer than 40 usable samples is `UNOBSERVABLE`, because
**an unrecorded window is not a quiet one**; and a retrospective window declares in its own silence
set that it is weaker than an in-run companion — it is sampled over wall time and cannot know which
samples overlapped a submission, so **it may not be used to upgrade a figure an in-run companion
refused.**

### 20.5 Slopes, not quotients — and a screen rather than a rule

§19.4 records the error: a cumulative counter dominated by a one-time ~2185 MiB weight upload,
divided by two different iteration counts, 51/28 = 1.82, published as a 1.78× improvement.

**On a permanently contended box this becomes the most likely error in the repository**, because run
lengths will now vary with whatever else is on the machine — so the denominator will differ between
any two records almost by default, and the artifact will appear without anyone doing anything
unusual.

A rule would decay. `bench/test_ceiling.py::TestCumulativeCounterScreen` is a text-decidable screen
over every file under `bench/`, with a positive control built from the exact line that produced the
false finding and a negative control asserting it does not fire on the correct two-point
construction. It currently reports no offenders.

Reproduce: `python bench/ceiling.py`. Locked by `bench/test_ceiling.py` (21 tests).


## 21. GQA is claimed, so the refusal is discharged — and what was underneath it (2026-08-02)

Frame, per R12's fourth generalisation — for a test result the frame is the binary that ran it.
`main` at `7c9d1b7`, merged into `squad/niobe` and rebuilt. DLL **`47F668336A7BF6A9…` before,
`3A9115417CD1A780…` after**; the binary changed this round, unlike §20's. Every figure below was
produced by that binary. Counters only: nothing in this section is a timing and nothing in it cares
whether the box was busy — which is just as well, because it was.

### 21.0 The claim status, read off the build rather than off the message

The coordinator reported GQA fixed. Read off my own build instead, from a counters-only run
(`--iters 4 --warmup 1 --repeats 1 --no-phases`):

| witness | reading |
|---|---|
| `subgraphs_live` | **1** |
| claimed nodes | **355 of 363** |
| `model_output_equivalence` | **MATCH** |
| session claim log | `com.microsoft::GroupQueryAttention x32 proven` |

Record: `bench/results/phi35-7c9d1b7-dev0.json`. The timing in that run was withheld — CONTENDED,
5.63 foreign busy cores — and none is used here.

### 21.1 The instrument was right about a build nobody was running

Before changing anything I ran `bench/ceiling.py` against the rebuilt tree. It reported:

> `dll  3a9115417cd1a780` … `ADMISSIBLE  [0]` … `because  GroupQueryAttention is declined on this
> build and executes on CPU`

Both halves of that output are in the same paragraph and they contradict each other. The DLL hash
is the *new* binary; the claim status came from `phi35-0baf660-dev0.json`, a record from the
previous one. The frame travelled in the report and was never checked against the thing it framed.
Artifact: `bench/_scratch/ceiling_stale_record_artifact.txt`.

That is the failure the coordinator asked me to avoid — *"the condition should read off the build,
not off my message"* — and it was already present in my own module. `Ceiling.load()` now hashes the
DLL and raises `CeilingError` if the claim record did not come from it. Records that name their
binary only by size and mtime are refused too, so `bench/environment.py` now writes
`environment.build.sha256`.

**Teeth.** A refusal that has just been satisfied is when it is most likely to become decoration,
so `test_ceiling.py::TestTheRefusalStillHasTeeth` carries a positive control: a synthetic *declined*
record that is **in frame** — same binary, 33 islands — must still collapse the extent to `[0]`
with the structural reason. It does. Plus a negative control, an out-of-frame record (refused), a
record with no `sha256` (refused), a missing record (`ERROR(instrument)`, raised rather than
silently yielding an empty extent), and an unhashable DLL (raised). Discharging the condition once
did not wire it open.

### 21.2 `0` was the wrong answer, and the token that replaced it had to be measured

Discharging the structural refusal exposed a second condition that had been masked underneath it:
**the KV byte term was never earned.** `island_bytes_phi35.json` computes it analytically, and at
past_len 8192 it is 60.5% of the modelled stream. Switch had to earn the equivalent claim for
weights *separately* — amplification 1.000000, two non-tautological factors measured apart. Nobody
had done that for KV, and a bound whose dominant term is an assumption is not a bound I can sign.

So it was measured: `bench/results/probe_kv_bytes_earned.py`. Prediction from the model's declared
input shapes and from no counter in the record — `32 × 2 × 32 × 96 × 2 = 393,216 B` per past token.
Cumulative counters differenced across two iteration counts (5 and 25) to cancel the one-time
~2185 MiB weight upload, then the slopes differenced across past_len to cancel every per-inference
term that does not scale with context. **Slope of slopes.** Three context points, not two, so
linearity is a finding rather than an assumption.

| past_len | upload / inference | readback / inference | modelled KV |
|---|---|---|---|
| 0 | 399,376 B | 457,344 B | 0 B |
| 128 | **399,376 B** | 50,788,992 B | 50,331,648 B |
| 512 | **399,376 B** | 201,783,936 B | 201,326,592 B |

| segment | upload B/past-token | readback B/past-token | predicted | readback ratio |
|---|---|---|---|---|
| 0 → 128 | 0.0 | **393,216.0** | 393,216 | **1.000000** |
| 128 → 512 | 0.0 | **393,216.0** | 393,216 | **1.000000** |

Linearity spread between the two segments: **0.000000**.

Two axes, two different answers, and they must not be averaged. My first version of the probe did
average them and printed `residency factor 0.0`, which reads as a refutation of a term that is in
fact confirmed to the byte. Corrected before the record was committed:

- **READBACK — MEASURED.** The present KV cache is copied device→host in full every inference,
  `(past_len + 1) × 393,216 B`. The modelled magnitude is exact on both segments.
- **UPLOAD — UNOBSERVABLE.** Identical at past_len 0, 128 *and* 512. The past KV cache does not
  reach the device by the staging path these counters watch. It reaches it somehow — the answers
  move with past_len — so the counter is blind to the path, and **its silence is not evidence that
  the read side is free.** R12: `UNOBSERVABLE`, never `0`.

**The falsifier for "past_len is wired" is an artifact it produced** (R10). Had the feeds been
ignored, every number above would be a measurement of nothing: at past_len 0, `present.0.key` is
`[1,32,1,96]` and argmax is 30751; at past_len 128 it is `[1,32,129,96]` and argmax is 8521. Both
moved.

Still not earned, and the record says so in a field rather than in a comment: the read-side
residency path, and the **DRAM amplification factor** — how many times the device reads each KV
byte per inference. Staging traffic is host↔device transfer, not kernel DRAM reads.

### 21.3 Two extents, because there are two questions

Earning the KV magnitude turned up a term the roofline never modelled at all — the present KV
crossing the link every inference. So "is the DRAM bound admissible here?" and "is the DRAM bound
the **floor** here?" now have different answers, and `ceiling.py` answers them separately.

Which term binds is decided **without measuring this machine's link**, because measuring it is a
timing and timings on this box are `STEADY_UNCERTIFIED` by standing policy (§20). Instead the
crossover is published — the link speed at which transfer time equals DRAM time — and compared
against two *stated* constants: the slowest PCIe configuration ever shipped (x1 gen1, 0.25 GB/s)
and the fastest consumer link ever shipped (5.0 x16, 63 GB/s). Neither is tuned; the crossover is
published at every context so a reader with a measured link can redo the call themselves.

| past_len | DRAM floor | transfer / inference | crossover | verdict |
|---|---|---|---|---|
| 0 | 8.2180 ms | 0.82 MiB *(measured)* | **0.10 GB/s** | **DRAM binds — this is the floor** |
| 128 | 8.4146 ms | 48.82 MiB *(measured)* | 6.08 GB/s | undecided |
| 512 | 9.0045 ms | 192.82 MiB *(measured)* | 22.45 GB/s | undecided |
| 2048 | 11.3638 ms | 768.82 MiB *(modelled)* | 70.94 GB/s | **transfer-bound** |
| 4096 | 14.5095 ms | 1536.82 MiB *(modelled)* | 111.06 GB/s | **transfer-bound** |
| 8192 | 20.8009 ms | 3072.82 MiB *(modelled)* | 154.90 GB/s | **transfer-bound** |

    extent()          -> [0, 128, 512, 2048, 4096, 8192]   the DRAM bound describes this build
    binding_extent()  -> [0]                               and is the floor of the inference

At 128 and 512 the crossover falls between the two constants and the honest answer is *undecided* —
it needs a measured link bandwidth. That is the threshold-episode discipline again: I did not move
a constant until the count of awkward contexts reached zero. The sensitivity is the table above.

**At past_len ≥ 2048 the crossover exceeds any link that exists, so the inference is bound by
host↔device KV transfer and the DRAM roofline is nowhere near the floor there.** That is a
direction for work, not a figure, and it is conditional on the readback law continuing past 512 —
the verdicts at 2048+ carry that caveat in the record rather than in a footnote.

### 21.4 The pairing, stated per context rather than assumed to travel

My line from §20 was that *the one admissible bound and the one quotable figure sit at the same
context — that is why the 67% comparison survives, not luck.* It stops being automatic the moment
the extent widens, and the extent has now widened.

- The DRAM bound is **admissible at 128…8192, where this project holds no quotable figure at all.**
  The bound is admissible in regimes where the comparison is not.
- It is the **floor only at past_len 0**, which is the one context we have ever run and the context
  of the one quotable figure.

So the sentence survives with its noun changed: the one *binding* bound and the one quotable figure
sit at the same context. That was true for a structural reason and is now true for a measured one.
`compare()` enforces it — a comparison at 8192 still returns a number, because the DRAM bound does
describe this build, but carries `floor_is_binding: False` and says a percentage read there would
be a percentage of the wrong roofline.

**`12.1847 ms` — zero context, one token — remains the only quotable figure, at 67.4% of the
binding roofline, headroom 1.48×.** Unchanged by this round, which is what should happen: the
binary changed, the bound at past_len 0 did not, because the KV term at past_len 0 is exactly zero
and the weight term was already earned.

Reproduce: `python bench/ceiling.py`, `python bench/results/probe_kv_bytes_earned.py`.
Locked by `bench/test_ceiling.py` (36 tests).
## 22. The anchor was never measured — the amplification becomes falsifiable (2026-08-03)

Every bandwidth argument this project has made rests on one number: **each weight byte is read
exactly once per token, amplification 1.000000**. It was published by
`bench/results/probe_island_bytes.py` as a block of five values, and its docstring made a careful,
correct argument for why `1.0` is a result and not an identity — the product `blobs x blob_bytes`
*is* the weight tensor by definition, but two factors in it are contingent: **loads per blob** and
**blobs per workgroup**. It said both had been established by a SPIR-V def-use walk.

**All five values were literals. There was no walk in the tree.** The reasoning happened once, in
a head, and its conclusion was transcribed as a constant. A kernel change that made the packed path
re-read every blob eight times would have left the printed `1.000000` untouched. This is R9's third
generalisation — a criterion is discharged by an observable that changes when the claim is false —
and R13's dangling-reference class: it was not broken, it *resolved anyway*, which is exactly why it
survived being quoted as the anchor.

### The instrument

`bench/spirv_simt.py` is a SPIR-V parser and a lockstep SIMT interpreter: it executes a compiled
module over a whole dispatch grid with numpy lane masks, structured-CFG recursion and per-lane phi
capture, and records every address a chosen binding is loaded from, at 32-bit-word granularity,
along with which workgroup named it. `bench/results/probe_weight_reread.py` drives it.

Three bindings make the reading a reading:

* **The module is located by content digest**, reimplementing `registry.rs::shader_digest_for`, and
  matched against `evidence/proof_ledger.jsonl`. The walk is of the compiled kernel a run
  dispatched, not of a shader source string re-compiled for the occasion.
* **The denominator comes from the graph** — the sum of ONNX initializer sizes over the 161
  `MatMulNBits` B inputs (the external-data `length` field), not a restated constant.
* **The detector has a demonstrated positive state.** Three controls, all seen:
  `tail_tile_N_not_divisible_by_cols` at **1.107692** on the shipped module at N=130;
  `deliberately_rereading_variant` at **2.000000** on `q_gemv.comp` with the packed chunk loop
  doubled (whose unmodified rebuild reproduces the ledger digest exactly, so the control differs
  from the shipped kernel by one edit and nothing else); and `unpacked_path_changes_the_width_not_
  the_bytes`. If no control fires the probe publishes `amplification: null` and `UNWITNESSED`
  rather than a number.

### The reading

**The number did not move.** 116,324,352 InB load instructions at a measured width of 16 B name
1,861,189,632 bytes against 1,861,189,632 bytes of initializer: **amplification 1.000000**, max
loads naming one 4-byte word **1**, words named by more than one workgroup **0**, coverage
**1.000000**. The original reasoning was right. The tree simply held nothing that could have told
us if it had been wrong.

Three things were not obvious from the argument:

* The compiled module has **four** `InB` load sites, not one. "One load per blob" is a property of
  *specialization* — which sites the spec constants make reachable — not of the shader text.
* The unpacked path names **the same bytes** (amplification 1.0) and differs only in width (16 B →
  4 B) and instruction count (×4). An amplification-only detector is blind to it; the width had to
  be measured separately, off the SPIR-V result type.
* The interpreter's own first bug was this same defect shape: masked-off lanes parked on scatter
  index 0, where numpy's last-write-wins ate a live store, producing a GEMV wrong in exactly one
  column per tile and plausible everywhere else. `test_interpreter_reproduces_the_gemv` is the
  negative control that caught it, and it holds bit-exact against a numpy reference.

### The rule: specification, measurement, model

The generalisation is not "do not hardcode". It is that this file **mixed two kinds of number
without saying which was which**, and once mixed, a transcribed conclusion is indistinguishable
from an observation. `probe_island_bytes.py` now carries a `PROVENANCE` table, emitted into its
JSON record, classifying every published quantity:

| Class | Meaning | In this file |
|---|---|---|
| **SPECIFICATION** | A fact about a part, published by its maker. Legitimately a literal — deriving it would be pretending to measure a datasheet. | `PEAK_BYTES_PER_S` (128-bit GDDR6 @ 16 Gbps) |
| **MEASUREMENT** | A fact about *this* graph, *this* module, *this* run. Must be derived here, every time. A literal is the defect. | `SPEC_PART`, `WEIGHT_STREAM_BYTES`, `LAYERS/HIDDEN/FFN/VOCAB/KV_HEADS/HEAD_DIM/FP16`, `weight_reread_amplification` |
| **MODEL** | An analytic construction: neither published nor observed. Legitimately code, never quotable as a measurement. | `intermediate_breakdown_bytes`, `kv_bytes()` |

The subtlety worth keeping is in the first two rows. A spec sheet peak **is** a fact about hardware
— but it is a fact about a *named part*, and the name is a separate claim of a different class. So
`SPEC_PART` is classified a measurement and is now read off the run's `device_identity.observed_
from_trace` — the device whose timestamp fingerprint appears in the row's own trace. Reading it off
`device_index` would have attributed `phi35-certified-dev0.json` to vulkaninfo index 0, which on
this box is an Intel iGPU. The peak and the run agree; that agreement is now checked rather than
assumed, and the probe says so in its output.

`intermediate_breakdown_bytes` is the one that most needed a label. Its multipliers are now node
counts read off the graph (161 MatMulNBits, 64 SkipSimplifiedLayerNorm, 64 Mul, 32 Sigmoid, 32 GQA),
but the *rule* — every tensor crossing a dispatch boundary is written once and read once — is an
assumption nothing observes. It is the size of the fusion prize, not a reading of it, and an
unlabelled model reads exactly like a measurement. That is how the amplification survived: it was a
model's conclusion wearing a measurement's clothes.

**Nothing in this section reads a clock or a hardware counter**, so none of it falls inside the
`a52024f`..`4d47362` window Mouse found the ABI-insertion defect in. One disclosure: no Phi-3.5 run
has ever recorded `pipeline_variants`, so the specialization constants driving the walk are derived
host-side from the graph through the `ops::quant` mirrors (locked by
`test_dispatch_geometry_mirrors_the_host`) rather than witnessed on a run.
`bench/results/gemv_kernel_identity-dev0.json` corroborates `packed=1` for bits=4/block=32 at a
different dtype — corroboration, not a witness.

The downstream consequences are unchanged because the number is unchanged; the roofline's
conclusions are not touched here. What changed is that the anchor can now go wrong.

Reproduce: `python bench/results/probe_weight_reread.py`, then
`python bench/results/probe_island_bytes.py`.
Locked by `bench/test_weight_reread.py` (15 tests).

## 23. The lever every grouped model pays for, and a ledger with no derivation (2026-08-03)
Two items, one shader change, one retraction. Neither needed a clock.

### 23.1 `Nq/Nkv > 1` writes `present` G times

`gqa_f16.comp` indexes `present_key`/`present_value` by `kv_h = h / G`. The value written is a
function of `(b, s_local, kv_h)` only — `k_new`/`v_new` come from `k_base`/`v_base`, and RoPE uses
`tok_pos = past_len + s_local`. **`h` appears nowhere in the written value.** So all G query heads
of a KV group wrote the same half-words, with complementary masks and an `Or` last, which is
bit-identical and therefore invisible to every output comparison in the tree. It is not a
correctness defect. It is G x the KV write traffic on every grouped model.

Phi-3.5 is `Nq == Nkv == 32`, i.e. `G = 1`. **This is the first performance defect the project has
found that our only end-to-end model is structurally incapable of showing.**

The fix is one predicate: `kv_write_leader = (h % group_size) == 0u`, gating step 3's `present`
write; `copy_leader` (which already had the guard) now reuses it. Coverage: within a group
`h in [kv_h*G, (kv_h+1)*G)` exactly one h satisfies `h % G == 0`, namely `h = kv_h*G`, and it maps
back to the same `kv_h` — so every word that had G writers has exactly one, and no word has zero.
The masked-write safety argument **moves from redundancy to disjointness**: for even D the base
`(b*Nkv + kv_h)*present_len*D + tok_pos*D` and the extent D are both even, so the surviving writer
owns both halves of every word it touches. The atomic path is kept for odd D, where two different
`(kv_h, tok_pos)` rows — different leaders — can share a word.

### 23.2 The measurement, and why the arena is load-bearing

Instrument: `bench/spirv_simt.py`, extended this round to trace **stores** as well as loads
(`run_traced(d, load_binding, store_binding) -> (LoadTrace|None, StoreTrace|None)`; `run()` still
delegates, so existing callers are untouched). `probe_kv_write_redundancy.py` compiles the
BASELINE from `git show main:rust/shaders/glsl/gqa_f16.comp` and the FIXED kernel from the
worktree with build.rs's own glslc flags, runs both over 10 arms, and **compares all three output
buffers bit-exact before printing any byte figure**.

| case | G | writers/word | K write bytes | x |
|---|---|---|---|---|
| phi35-like S1 arena | 1 | 1 -> 1 | 2048 -> 2048 | 1.00 |
| gqa4 S1 arena | 4 | 4 -> 1 | 2048 -> 512 | **4.00** |
| gqa4 S4 growing | 4 | 4 -> 1 | 11264 -> 5120 | 2.20 |
| gqa8 S2 growing | 8 | 8 -> 1 | 10240 -> 3072 | 3.33 |
| llama3-8b-decode arena (32/8/128) | 4 | 4 -> 1 | 32768 -> 8192 | **4.00** |
| llama3-8b-decode growing | 4 | 4 -> 1 | 98304 -> 73728 | 1.33 |
| phi35-decode arena (32/32/96) | 1 | 1 -> 1 | 24576 -> 24576 | 1.00 |

The result that is not the headline: **the growing convention hides most of the lever.** Under
growing, the mandatory past-region relocation already had exactly one writer, so it dilutes the
new-token dedup — 1.33x at `past_len 8`, falling toward 1.0 as past grows. Under the arena the
past copy does not exist, the new-token write is 100% of `present` traffic, and the reduction is
exactly G. **L3 (arena) and L4 (dedup) compose; they are not independent levers**, and quoting
4.00x without the arena in the same sentence overstates it by up to 3x.

Scope that must travel with the 4.00x: this is node-level. No grouped model is run end-to-end here
(an 8 GB board will not hold Llama-3 8B), and the byte figures are **words named by store
instructions**, not DRAM transactions — calling them DRAM is `probe_roofline.py`'s cache argument,
an argument, not this measurement.

Trinity's warning landed on me too: `tests/ops/test_gqa.py` had G=4 all along. Before asserting
something is untested, check.

### 23.3 A sixth instrument defect: the interpreter's GLSL.std.450 table was wrong

Extending the interpreter hit `unsupported GLSL.std.450 instruction 42`. Adding 42 would have been
the small fix. Checking the table instead found **four wrong entries**: it said
`30:Fma, 37:FMax, 40:FClamp, 43:FMin`; the real numbering is `30:Log2, 37:FMin, 40:FMax,
43:FClamp, 50:Fma`.

Verified by histogramming the opcode numbers our own parser extracts and matching them against
`spirv-dis`'s names on `gqa_f16.spv` (`27->Exp x2, 42->SMax x1, 58->PackHalf2x16 x3,
62->UnpackHalf2x16 x9`, exact counts), then corroborated across every `.spv` in the tree: 40
appears only in `ew_unary_relu`/`mish`/`softplus`, 43 only in `hardsigmoid`/`hardswish`, 37 only in
`celu`, 45 in `gather`, 32 in the layer norms. Nothing caught it because the interpreter's sole
correctness control is a quantised GEMV that calls none of the four.

**Correction (same day, second pass): I said a `relu` would have been silently miscomputed. That
is false, and I had not checked the thing that decides it.** The discriminator is the **operand
count**. A wrong name taking *more* operands than the real one indexes past the end of the operand
list and raises; a wrong name taking *fewer* silently drops the extra and returns a plausible
float. Executed under both tables rather than reasoned about:

| ext-inst | real | old table said | verdict |
|---|---|---|---|
| 37 | `FMin` | `FMax` (2 ops) | **SILENT** — a minimum returned as a maximum |
| 40 | `FMax` | `FClamp` (3 ops) | **LOUD** — `IndexError` on the first invocation |
| 43 | `FClamp` | `FMin` (2 ops) | **SILENT** — the upper bound dropped entirely |
| 50 | `Fma` | absent | **LOUD** — `unsupported GLSL.std.450 instruction 50` |

So the silent set is `{37, 43}` and the silently-miscomputed kernels are exactly **`celu`,
`hardsigmoid` (f16/f32), `hardswish` (f16/f32)** — five kernels, none of them `relu`. The general
lesson is unchanged; the vivid example was wrong, and it was wrong in the direction that made the
defect sound worse than it was.

**What it could *not* have touched, which is the question that matters to standing results.**
`q_gemv_matmul_nbits_f16` issues only `PackHalf2x16`/`UnpackHalf2x16`; `q_gemv_matmul_nbits_f32`
issues no ext-inst at all; `gqa_f16` issues `27/42/58/62`. **None of them is in either set, so
§22's weight-amplification measurement and §23.2's write traffic are both untouched** — and 42
(`SMax`) was *missing* rather than wrong, which is why extending the interpreter errored loudly
instead of returning a number. Reproduce: `python bench/results/probe_glsl450_blast_radius.py`.

The repair is not just the table. `bench/test_kv_write_redundancy.py` now compiles a kernel that
calls `min`/`max`/`clamp`/`fma` through glslc — so the opcode numbers come from the toolchain
rather than from anybody's memory — executes it, and compares against numpy. That is the
correctness control the four opcodes never had.

### 23.4 The KV lever ledger, re-derived — and three numbers retracted

`docs/DESIGN.md:3699` and `bench/results/kv-int8-budget-prediction.md:107` quote **2.21x / 3.17x /
4.06x**. Those figures exist nowhere in this tree as an artifact — only as mentions.
`bench/results/probe_kv_lever_ledger.py` is a **generator, not a table**: it re-derives every
figure from artifacts at run time and classes each one per §22's SPECIFICATION / MEASUREMENT /
MODEL rule.

| id | lever | axis | class | ratio | status |
|---|---|---|---|---|---|
| L1 | KV cache stays device-resident across `run()` | LINK | MEASUREMENT | **n/a** | LANDED |
| L2 | past-copy fused into the attention loop | DRAM | MODEL | 1.50x | LANDED |
| L3 | KV arena (`present`/`past` one allocation) | DRAM + FOOTPRINT | MODEL | 2.00x | AVAILABLE |
| L4 | one `present` writer per KV group | DRAM (write) | MEASUREMENT | 4.00x at G=4 arena | THIS ROUND |
| L5 | int8 / int4 KV storage | FOOTPRINT | MODEL | 1.377–1.412x / 1.724–1.761x | NOT BUILT |

**L1 has no ratio at all** and this is the point, not an omission: its after-term is zero
(393,216 -> 0 bytes per past token over the link). An elimination has no multiplier, and writing
one requires inventing a denominator.

**LINK bytes, DRAM bytes and FOOTPRINT bytes are three different quantities.** A ratio on one does
not compose with a ratio on another. The KV cache is 60.5% of the stream at ctx 8192 and 0% of the
weights, so a 2x/4x saving on the KV *term* is 1.4x/1.8x of the total.

The reconstruction attempt is in the generator's output. `2.21` sits 0.21 from the naive KV-term
fp16/int8 ratio 2.0; `4.06` sits 0.06 from the naive KV-term fp16/int4 ratio 4.0; **`3.17` fits
nothing**. The hypothesis — labelled as a hypothesis, and nothing depends on it — is that the old
ledger quoted KV-term savings as whole-system savings: an **axis error, not an arithmetic one**,
which is why no amount of re-deriving on the stream or the footprint could land on them. All three
are **RETRACTED rather than corrected**: a number whose derivation cannot be found is not repaired
by finding a derivation that lands nearby. The artifacts support **1.377–1.412x (int8)** and
**1.724–1.761x (int4)** on the footprint axis, and those were written into
`kv-int8-budget-prediction.md` before the first int8 run.

### 23.5 The proof ledger caught my own shader edit

Editing `gqa_f16.comp` moved its digest (`4f8ea70a1a80b290` -> `ae376245998decd6`), which demoted
the GQA proof-ledger entry to `SUBJECT-CHANGED`, and the EP **declined every GQA node** until it
was re-proven. This is the mechanism working. Repair is two steps and the second is easy to miss:
`gen_proof_ledger.py --model evidence/cases/group_query_attention_f16.onnx --reprove --append`,
then **rebuild** — `proof_ledger.jsonl` is `include_str!`d, so the on-disk file only takes effect at
the next build. Any shader edit needs both.

### 23.6 It caught me twice, and the second time cost a rollback

The first commit of this work was merged, verified, found to regress, and backed out. **The
`--check` in my own worktree was green** — I had re-proved the entry against *my* build — but
after the merge the entry described a kernel the merged binary did not contain, and the only
symptom was a silent `SUBJECT-CHANGED` decline. Measured on the merged tree, with
`probe_phi35_claim_reading.py`:

| | main without it | with the stale entry | after re-proving |
|---|---|---|---|
| `claimed_nodes` | 355 | **323** | 355 |
| `unproven_declines` | 5 | **37** | 5 |
| `islands_offered` | 1 | **33** | 1 |

**One declined form shatters a 355-node island into 33.** That is the proof economics §21 measured,
arriving as a bill. Note what the two instruments each told me: `--check` said *one entry
disagrees*; only the claim-reading probe said *it costs 32 nodes and 32 islands*. A subject
mismatch is a fact about the ledger; the claim reading is the fact about the user, and they are
not substitutes.

The hole was that **nothing gated the ledger against the build**. `--check` is run by hand;
`tests/ops/test_proof_ledger.py` checks the artifact's internal consistency and its behaviour
against planted controls, but never calls `check_against_build`. That gate now exists in
`bench/test_kv_write_redundancy.py`, because the discipline it enforces belongs to shader editing.

**Re-prove against a single-form case model, never against Phi-3.5.** The prove pass sets
`session.disable_cpu_ep_fallback`, which requires the EP to claim **every node in the graph**.
Phi-3.5 contains three ops this EP registers no handler for at all — `Shape`, `ReduceSum` and `If`
— so the session can never build, `--reprove` or not, and the failure is identical whether or not
any form is withheld. It is not that an unprovable form blocks the others: **Phi-3.5 has never
been a valid proof subject and cannot become one while `If` is unhandled.** The tool says so in
its own comment — *"every evidence case is a single-form model"* — and the entry proves a **form**,
which does not care which graph exercised it. `evidence/cases/group_query_attention_f16.onnx`
re-proves the GQA form in about ten seconds, `worst_rel 7.29e-04`, `claimed_nodes 1`,
`dispatches_executed 1`.

What the diagnostic does not say is which of those it is. `UNATTRIBUTED` reports the ORT refusal
and lists the *offered* forms, which reads as though the offered-but-unproven ones are the
blocker; the census that would settle it (`no_key: 3`) is already computed one function earlier and
is not shown. That is a diagnostic gap in `gen_proof_ledger.py`, which is Mouse's, and it is filed
rather than fixed here.

Reproduce: `python bench/results/probe_kv_write_redundancy.py`, then
`python bench/results/probe_kv_lever_ledger.py`, then
`python bench/results/probe_glsl450_blast_radius.py`.
Locked by `bench/test_kv_write_redundancy.py` (24 tests).

## 24. The paired ratio does not rescue a timing on this box — and the arm that got *faster* (2026-08-03 night)

**Verdict first, because it is the deliverable: a paired, finely interleaved A/B ratio is NOT a
sound way to time this EP on this machine.** All three runs return
`PAIRING_FAILS`, and they fail for three independent reasons, each of which is on its own
sufficient. This section is the fourth "this cannot be measured here" this project has accepted,
and unlike the previous three it is accompanied by the instrument that establishes it, so the
refusal is falsifiable rather than asserted.

The proposal under test was the standard answer to a noisy shared box, and it is a good one:
contention corrupts an *absolute* number but should cancel out of a *ratio* taken under shared
conditions, so run our EP and a reference back to back, alternating, in one process, on the same
inputs, and publish the ratio with its dispersion. §20 forbids *waiting* for a quiet box; it does
not forbid *measuring* under a busy one. I set out to build that and spent the run attacking it
instead, because it does not survive its own controls.

Artifact: `bench/results/probe_paired_ratio.py`. Records:
`paired_ratio_dev0.json` (NVIDIA, EP vs CPU EP, 12x6 = 72 pairs/phase),
`paired_ratio_resident_dev0.json` (NVIDIA, EP vs *itself* with device-resident KV, 72 pairs/phase),
`paired_ratio_dev1.json` (Intel, EP vs CPU EP, 8x5 = 40 pairs/phase).
All three on DLL `2fb929da1179eb55`, `claimed_nodes 355`, `islands_offered 1`,
`compute_failures 0`, `device_losses 0`, argmax `AGREE` between the arms on every step compared.

### 24.1 The design, and the three things it had to prove about itself

Six phases per run, in order: `solo_vk` (arm A alone, before the other session exists),
`paired` (alternating), `blocked` (each arm's steps in a block, same sessions), `cpuload`
(paired, with N-1 spinning host workers), `gpuload` (paired, with a second Phi-3.5 decoding on the
same board), `solo_ref` (arm B alone). `solo`/`blocked`/`paired` price the **apparatus itself**;
`cpuload`/`gpuload` are the **injections** that test the common-mode claim. The unit of pairing is
one decode step, matched by `(sweep, step)` so the ratio is never pooled across `past_len` — a
decode step is atomic, so **there is no finer interleaving available on this design**, and that
fact turns out to matter more than anything else here.

Three outcomes are defined, and the third is deliberately not a pass:

| outcome | meaning |
|---|---|
| `PAIRING_HOLDS` | the injection moved at least one arm's level and the ratio did not move |
| `PAIRING_FAILS(not_common_mode)` | the ratio moved: the disturbance was not shared |
| `VACUOUS(injection_not_witnessed)` | neither arm moved, so nothing was tested |

`VACUOUS` is the guard that matters. An instrument that reports success when nothing was injected
would have certified every contended run this project has ever taken.

### 24.2 Failure 1 — the apparatus perturbs the two arms unequally, before any foreign load exists

| run | vk apparatus cost | ref apparatus cost | asymmetry |
|---|---|---|---|
| NVIDIA, vs CPU EP | x2.459 | x1.496 | **1.644x** |
| NVIDIA, vs resident KV | x0.987 | x0.464 | **2.125x** |
| Intel, vs CPU EP | x0.968 | x0.858 | 1.128x |

Read the middle row: applying the pairing apparatus more than *halved* the reference arm's step
time. A ratio published from the paired phase of that run carries a factor of **2.1x that belongs
to the instrument**, not to the EP. The three rows also disagree with one another in direction,
which is the same conclusion by a second route: **the apparatus cost is not a stable quantity on
this box**, so it cannot be divided out. Its own confound is stated here rather than buried — the
`solo` phases sit at the ends of the run, so these figures are entangled with warm-up at the start
and board state at the end; they are an upper bound on instrument error, not a clean measurement
of it.

### 24.3 Failure 2 — host contention is not common-mode, and it flatters us

| run | injection | vk lift | ref lift | ratio-of-ratios (95% CI) | verdict |
|---|---|---|---|---|---|
| NVIDIA, vs CPU EP | 19 spinners | x3.32 | x5.94 | **0.560** (0.498-0.629) | FAILS |
| NVIDIA, vs resident KV | 19 spinners | x8.67 | x2.78 | **3.122** (2.777-3.524) | FAILS |
| Intel, vs CPU EP | 19 spinners | x1.06 | x1.66 | **0.638** | FAILS |

Every CI excludes 1.0 by a wide margin. Against the CPU EP the ratio improves ~1.8x **purely
because the box got busy** — the CPU arm is hurt far more than the GPU arm, so a number taken
during a loud hour would look better than the same number taken during a quiet one, in the
direction that flatters this EP. That is the precise failure a paired design is adopted to
prevent.

The middle row is a real finding hiding inside a methodological failure: under host load the
**host-KV lane inflates 8.67x while the device-resident lane inflates only 2.78x**. The lane that
pays the 393,216 B/past-token round trip through host memory every step is the lane that is
sensitive to a busy host, and it is sensitive by a factor of three. That is the KV round trip's
signature showing up on an axis nobody was pointing at it.

### 24.4 Failure 3 — foreign GPU work made our arm *faster*

On the NVIDIA cross-device run, injecting a second Phi-3.5 decode onto the same board produced
`vk_lift_x = 0.771`. **Our arm sped up by a quarter while a competitor was running on the same
GPU.** The mechanism is §20.2's, running in the opposite direction from the naive expectation, and
the clock series attributed to each arm's own executing intervals says it outright:

| phase (NVIDIA) | board clock, phase-wide (cross-device run) | attributed to the vk arm's own steps (same-device run) |
|---|---|---|
| `solo_vk` | 1740 MHz (56% of the 3105 MHz boost ceiling) | 2010 MHz |
| `paired` | **825 MHz** (27%) | 2010 MHz |
| `blocked` | 210 MHz | 1328 MHz |
| `cpuload` | 420 MHz | 795 MHz |
| `gpuload` | **2475 MHz** (80%) | **2460 MHz** |
| `solo_ref` | 210 MHz | 210 MHz (attributed to the ref arm) |

The two columns are two different runs and two different reductions, given side by side because
they disagree in a way that is itself the point: a phase-wide median is dominated by whichever arm
holds the wall clock longest — on the cross-device pair that is a factor of five — so `blocked`
reads 210 MHz phase-wide while the Vulkan arm's own dispatches were seeing 1328 MHz. Only the
attributed column answers the §20.2 question, which is about the board *while our dispatches are in
flight*. Both columns agree on the two extremes: interleaving idles the board, a co-tenant holds it
near boost.

Interleaving with a ~300 ms CPU step leaves the GPU idle for most of every pair, and the board
downclocks into it. A co-tenant holds the board *up*. **The interleaving granularity that would
make a foreign episode land on both arms is the same granularity that manufactures an
own-asymmetry on the device axis**, and since a decode step is atomic there is no granularity
available that does one without the other. This is the deepest of the three failures: it is not a
tuning problem, it is a property of pairing a GPU arm against a slower arm on a boost-clocked
board.

### 24.5 Failure 4, the quiet one — the pairing barely buys anything

§10.3 measured this machine at **2.65x** in single-threaded throughput, and that is the number
that decides whether a paired design is enough. Measured from the runs' own data:

| run | variance reduction from pairing | pairs for a +-5% CI | pairs actually taken |
|---|---|---|---|
| NVIDIA, vs CPU EP | 1.44x | **351** | 72 (giving +-11.4%) |
| NVIDIA, vs resident KV | 1.30x | 55 | 72 (giving +-4.4%) |
| Intel, vs CPU EP | 1.35x | 342 | 40 (giving +-15.3%) |

A paired design is adopted for a 5x-and-up variance reduction. It delivers **1.3-1.4x** here,
because the disturbance the two arms share is a small part of what either arm's dispersion
actually is. Even had the common-mode claim held, the cross-device form would have needed ~350
pairs for a claim of the precision anyone would want to quote.

### 24.6 What the reference is, and why there is no better one today

Stated in the verdict and not in a comment, as required: **the cross-device ratio is a ratio of
this EP on this GPU against ORT's CPU EP on this CPU.** It is an end-to-end system ratio that
confounds the EP with the device, and it may not be quoted as evidence about kernel quality.
A second GPU EP would have been the truer comparison and **is not available on this machine
today**: `onnxruntime-directml` publishes no wheel past ORT 1.24.4, this process runs 1.28.0, and
the Vulkan EP loads through the 1.28 plugin-EP ABI, so the two cannot be co-resident. That is
recorded in every record's `reference.second_gpu_ep`.

The same-device form was built to escape the confound: arm B is **the same EP, same session, same
binary and same board**, differing only in whether the KV round trip is paid. That ratio is a
ratio of the shipping lane against the resident lane and prices one lever. It escapes the *device*
confound and still fails the *contention* test, which is the finding.

### 24.7 The number I would have published, and why it is not a number

The same-device paired ratio, on the NVIDIA board, decode only, `past_len` 4..9, is not one value.
It is this, and its variation is the result:

| box state during the pair | host-KV / resident-KV | dispersion (exp sd) |
|---|---|---|
| blocked | 1.081 | 1.33x |
| paired, box as found | **1.185** (+-4.4%) | 1.20x |
| + foreign GPU tenant | 1.480 | 1.22x |
| + 19 host spinners | 3.701 | 1.62x |

**A 3.4x swing driven entirely by what else was running.** Quoting `1.185` as "the KV round trip
costs 19%" would be exactly the error the Fact Checker diagnosed in my own long-lived claims — a
pre-formed number re-quoted rather than re-derived — because the same instrument produces `3.70`
under a condition this box reaches routinely (its mean foreign load during these runs was
**5.8-7.9 busy cores of 20**, loud on 92.5-99.7% of samples). If a figure from this work is ever
re-quoted, it must be this table and not a row of it.

### 24.8 Intel is permanently uncertifiable for this question

Every phase of the Intel run returns `clock_producer: NO_PRODUCER` (§16.3). The confound that
turned out to *be* the story on NVIDIA — the board's clock moving with the interleaving pattern —
is **unobservable in principle** on that device. An Intel paired ratio is therefore
`UNCERTIFIED(partial_companion)` permanently: the tenancy half can be recorded, the clock half
cannot, and a half-companion is not a pass. The Intel `paired` figure of 0.191 is in the record and
is not quotable.

### 24.9 Provenance of every figure above (§22)

| figure | class |
|---|---|
| `sm_max_mhz` (3105 MHz boost ceiling) | SPECIFICATION |
| step times, ratios, `sm_mhz`, device name, apparatus costs, lifts, variance reduction | MEASUREMENT |
| `BYTES_PER_PAST_TOKEN` = 393,216 | MODEL (32 layers x 2 x 32 heads x 96 dim x 2 B) |
| `pairs_for_5pct_ci`, `unpaired_runs_for_5pct_ci` | MODEL (log-normal, from this run's own sd) |
| `PAIRING_TOLERANCE` = 0.10, `INJECTION_MIN_LIFT` = 0.15 | MODEL (judgement, stated beside itself) |

The 2.65x of §10.3 is carried forward as MEASUREMENT from that section, not re-derived here.

### 24.10 A disclosure about the injections

The injected load is our own descendant process, so `probe_gpustate`'s ancestry classifier calls it
"ours" and the board still reads `SOLE_TENANT` **while a second Phi-3.5 is decoding on it**. The
witness for an injection is therefore the launch plus the utilisation and clock series, never the
tenancy verdict. This is written into each record's `injection.witness` so that a later reader
cannot mistake the tenancy line for evidence that the injection did not happen.

### 24.11 What this does not establish

It establishes that a paired interleaved ratio against ORT's CPU EP, and against this EP's own
device-resident lane, both fail their common-mode controls on this machine, in three named ways,
with the numbers above. It establishes nothing else. Specifically:

- **Nothing about prefill.** Every step measured is a single-token decode. Prefill has a different
  arithmetic intensity and was not touched.
- **Nothing about any other model.** Phi-3.5 mini int4 only, whose Nq/Nkv = 1.00 is the degenerate
  case of the grouped-attention axis.
- **Nothing about long context.** `past_len` 4..9. The shipping lane OOMs at past 4096 on the 8 GB
  discrete card, so at long context the honest comparison is not "slower", it is "does not run".
- **Nothing about a quiet machine.** These are ratios measured under contention and they are not
  predictions of what either arm would do alone. That is not a caveat on the result; §24.3 and
  §24.4 are the demonstration that the box state *is* an argument to the number.
- **Nothing about kernel quality.** No two-GPU-EP comparison was possible, so nothing here
  compares this EP's kernels against anyone's.
- **Nothing about how fast this EP is.** The original question — "is this a high-performance EP" —
  is *still unanswered*, and this section is the argument that it cannot be answered on this box by
  this method. What it would take is stated below.

### 24.12 What it would take

1. **A second machine, or a second board in this one, that is not shared.** The cheapest sound
   answer. Not available.
2. **A second GPU EP at a compatible ORT version**, which removes the device confound and lets the
   two arms share the board's clock state. Blocked on the DirectML wheel matrix, and it would still
   have to pass the §24.2 apparatus test.
3. **A locked clock.** `nvidia-smi --lock-gpu-clocks` would remove the §24.4 mechanism outright and
   would make the interleaving-idle problem disappear. It requires privileges this process does not
   have and it would perturb the co-tenant, which §20 treats as a standing occupant with equal
   claim on the box.
4. **A counting instrument instead of a clock.** This is the project's existing answer and it
   remains the right one: bytes, slopes and bounds survived scrutiny that killed several timings,
   and §22's weight amplification of exactly 1.000000 is worth more than any of the ratios above.

Reproduce: `python bench/results/probe_paired_ratio.py --device 0`,
`python bench/results/probe_paired_ratio.py --device 0 --arm-b resident`,
`python bench/results/probe_paired_ratio.py --device 1 --sweeps 8 --steps 5`.
Requires `ONNXRUNTIME_VULKAN_EP_LIB` — without it `counters_abi.py --check` PASSes having read
nothing. Locked by `bench/test_paired_ratio.py` (19 tests), which includes the positive control
(a common-mode disturbance must yield `PAIRING_HOLDS`), the negative control (an arm-specific one
must yield `PAIRING_FAILS`), and the vacuity control (an injection that moves nothing must not be
reported as a pass).

### 24.13 Independent confirmation of Switch's GLSL.std.450 correction

Asked to confirm rather than take on trust, and confirmed against the shipped SPIR-V rather than
the GLSL source. `spirv-dis` on the release build:
`q_gemv_matmul_nbits_f16.spv` issues **eleven ext-inst calls, all of them `UnpackHalf2x16` (62) or
`PackHalf2x16` (58)**; `q_gemv_matmul_nbits_f32.spv` imports `GLSL.std.450` and issues **no
ext-inst at all**. Neither opcode is in the affected set `{30, 37, 40, 43, 50}`, and neither is in
the silent subset `{37, 43}`. **The weight read amplification of exactly 1.000000 is untouched and
needs no re-run.**



## 25. The amplification the anchor was hiding: `M` passes over the weights (2026-08-06)

§22 made the weight-read amplification falsifiable and read it at **1.000000**. That reading was
correct and it was also *narrow*, in a way the section did not say: it was taken at `M = 1`. The
probe walked one dispatch of a decode-shaped GEMV, and a decode-shaped GEMV reads each weight byte
exactly once because there is only one activation row to read it for.

Prefill is not decode. `q_gemv.comp` mapped one workgroup to one `(row, column-tile)` pair, so a
prefill of `M` rows dispatched `M` times as many workgroups and **every one of them re-read the
whole packed column tile it was assigned**. The amplification at `M` was `M`. Nothing in the tree
said so, because nothing in the tree had ever asked the probe for `M > 1`.

Issue #7. This section is the reading, before and after.

### 25.1 The defect, measured before it was fixed

The first thing the extended probe did was walk the *unmodified* kernel at `M > 1`, so the number
being fixed is a measurement and not a motivation:

| M | amplification, pre-change | what that is |
|---|---|---|
| 1 | 1.000000 | §22's anchor, unchanged |
| 2 | 2.000000 | every weight byte read twice |
| 4 | 4.000000 | every weight byte read four times |
| 5 | 5.000000 | every weight byte read five times |

Denominator as in §22: **1,861,189,632** bytes of int4 initializer over Phi-3.5's 161 MatMulNBits
nodes, summed off the graph's external-data `length` fields, not restated.

### 25.2 The change

One specialisation constant, `QB_ROWS` at id 6, and a second arm in `main()`. A workgroup now owns
a `QB_ROWS x QB_COLS` tile of the output: it loads a packed weight blob once and applies it to all
`QB_ROWS` activation rows before moving on. Weight amplification falls from `M` to `ceil(M/QB_ROWS)`.

Four properties are worth naming because each of them is what makes this portable rather than a
tuning win on one card:

* **It costs no shared memory.** The reduction is *sequential over rows*, reusing the one `red[]`
  array the decode path already had. Shared-memory usage is byte-for-byte what it was, so no device
  that can run the decode kernel can fail to run the tiled one. The row tile is deliberately not a
  device-limit question.
* **It needs no new feature, extension or capability.** Vulkan 1.1 core, unchanged.
* **`QB_ROWS == 1` is a specialisation-constant branch holding the verbatim pre-change code.** That
  makes "decode is bit-identical" a property of the *source text* rather than an argument about
  floating-point associativity. At `M = 1` the host selects `rows = 1`, both arms bind the same
  constants, and the same SPIR-V produces the same pipeline.
* **It is bounded, and the bound is checked twice.** `QB_ROWS * QB_COLS <= 32` is the accumulator
  register budget. The host refuses with `EpError::Internal` rather than dispatching an illegal
  tile, and the shader's first statement is a spec-constant guard that returns before any
  `barrier()`. Both operands are specialisation constants, so it folds away in every real pipeline
  and costs nothing.

The bound is not decorative. `rows=4, cols=16` indexes `acc[r*QB_COLS+c]` up to 63 against a
32-element array, and the SPIR-V interpreter caught it as an out-of-bounds *before* either guard
existed. The host cannot select that tile — but "unreachable" is not the same claim as
"fail-closed", and `test_an_illegal_tile_computes_nothing_rather_than_overrunning` is the control
that makes the difference visible.

### 25.3 The reading, after

Same probe, same denominator, same module-by-content-digest binding, same four positive controls
(the fourth, `row_tile_removes_the_M_fold_weight_reread`, is new and fires):

| M | pre-change | post-change | predicted `ceil(M/rows)` | weight bytes no longer named |
|---|---|---|---|---|
| 1 | 1.000000 | **1.000000** | 1 | — (decode is untouched) |
| 2 | 2.000000 | **1.000000** | 1 | 1,861,189,632 |
| 4 | 4.000000 | **2.000000** | 2 | 3,722,379,264 |
| 5 | 5.000000 | **3.000000** | 3 | 3,722,379,264 |

Every one of Phi-3.5's five distinct MatMulNBits shapes selects `(cols=16, rows=2)` and every one
reaches coverage 1.000000 in both arms. `M = 5` landing on 3 rather than 2.5 is the tail tile being
counted honestly: `ceil(5/2) = 3`, and the third tile does a full pass to compute one row.

**The `M = 4` row is where the tile is currently leaving something.** `gemv_tile` picks `rows = 2`,
not 4, and the reason is arithmetic rather than caution: when `cols | N` the byte model's weight
term depends only on `rows` and its activation term only on `rows/cols`, so `(16, 2)` and `(8, 4)`
name *exactly the same bytes* and the strict-improvement rule keeps the first. `(16, 4)` would be
strictly better and is refused by `cols * rows = 64 > 32`. Raising the accumulator budget is the
obvious next lever and is not taken here.

### 25.4 What it is worth on a clock

A byte the shader does not name is not automatically a microsecond saved, so this was measured
rather than inferred. `bench/results/ab_row_tile.py` alternates the two arms inside one process on
one pinned device — `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS=1` against `=4`, five interleaved repeats,
50 timed iterations after 20 warmup each — on the **NVIDIA RTX A1000** (`vk 1.4.303`, driver
`573.44`, discrete). Shape is the OQ-12 anchor's `K=N=4096`, q4, block 32, fp32.

| M | untiled (ms) | tiled (ms) | speedup, median of paired ratios | [min, max] |
|---|---|---|---|---|
| **1** | 0.248 | 0.229 | **0.994x** | [0.944, 1.151] |
| 2 | 0.323 | 0.290 | 1.162x | [1.082, 1.169] |
| 4 | 0.526 | 0.381 | 1.367x | [1.237, 1.512] |
| 5 | 0.674 | 0.482 | 1.344x | [1.285, 1.429] |
| 8 | 0.938 | 0.582 | **1.618x** | [1.537, 1.704] |

**The `M = 1` row is not a result, it is the control.** Both arms bind identical specialisation
constants and identical SPIR-V there, so its measured 0.994x with a spread of [0.944, 1.151] *is*
this harness's noise floor, and every other row has to be read against it. The `M >= 2` speedups
all sit above the control's maximum, so they are not the harness.

Two disclosures about how that table was obtained, because the first version of it was wrong:

* **A fixed arm order is a systematic bias, not noise.** Running untiled-then-tiled every repeat
  produced a **0.905x** "slowdown" at `M = 1` — on a shape where the two arms are the same
  pipeline, so the only thing it could have been measuring was the order. The GPU clock and the
  page cache drift monotonically inside a repeat and whichever arm always runs second inherits it.
  The arms are now alternated per repeat and the control comes back to 0.994x.
* **An unpinned device is not a device.** The first run did not pin and reported `device None`; on
  this two-GPU box `bench.py::select_device` exists precisely because an unattributed result is not
  a slightly worse result. The A/B now refuses to run rather than report an unnamed device, and the
  device name is in the JSON.

Against the CPU EP at the same shape (`bench/results/prefill_{tiled,untiled}.json`, both pinned to
the A1000), the tile moves prefill from roughly parity into a real win: at `M = 8`, 0.98x untiled
becomes 1.79x tiled; at `M = 4`, 1.49x becomes 2.47x.

The wall-clock gain is smaller than the traffic gain, and the reason is stated rather than glossed:
the tiled arm currently issues **32-bit scalar B loads** where the decode arm issues 128-bit ones
(`tiled_load_widths_bytes: [4]` against `untiled_load_widths_bytes: [16]` in the probe output). It
names the same bytes in four times as many instructions. That is a register-pressure decision, it
is visible in the instrument, and it is the second obvious next lever.

### 25.5 Whether it is still right

Traffic is only interesting if the answer is unchanged, so the equivalence was re-established at
three levels.

**The proof ledger.** The shader edit moved the digest, `gen_proof_ledger.py --check` failed closed
with `FAIL(LEDGER_DOES_NOT_DESCRIBE_THE_BUILD)` naming exactly the six MatMulNBits entries, and all
six were re-proven on the A1000. 133/133 entries live, all MATCH.

**The model's own weights.** `bench/results/probe_real_matmulnbits_rows.py` lifts real MatMulNBits
nodes out of the resolved Phi-3.5 file — real packed int4, real scales, real zero-points, real fp16,
real K and N — into one-node graphs and runs each on both providers at `M ∈ {1,2,3,4,5,8}`. Twelve
nodes covering every distinct form present, **72/72 match**:

| M | match | max abs err | max rel err, significant elements | max rel err, all elements |
|---|---|---|---|---|
| 1 | 12/12 | 1.953e-03 | 9.234e-04 | 6.792e-03 |
| 2 | 12/12 | 3.906e-03 | 9.461e-04 | 1.881e-02 |
| 3 | 12/12 | 3.906e-03 | 9.766e-04 | 3.883e-02 |
| 4 | 12/12 | 7.812e-03 | 9.699e-04 | 2.093e-01 |
| 5 | 12/12 | 3.906e-03 | 9.756e-04 | 5.649e-02 |
| 8 | 12/12 | 1.562e-02 | 9.756e-04 | 1.364e-01 |

The `rel(all)` column looks alarming and is not. It is a **cancellation meter, not an accuracy
meter**: where the CPU result lands near zero a one-ulp difference reads as a huge relative one, and
the number of chances to land near zero grows linearly with `M`. The two columns that are actually
about accuracy say so plainly — `rel(significant)`, restricted to elements at least a tenth of the
output RMS, is **flat at ~9.5e-04 from `M = 1` to `M = 8`**, and every `max abs` value is an exact
power of two, i.e. one to two fp16 ULPs at that magnitude. The tiled arm is neither better nor
worse than the decode arm; it is the same arithmetic in a different order.

**The one honest limitation.** `rust/modelrunner` still reports
`UNSUPPORTED(reason=reference_run_unsupported)` for this model: its GroupQueryAttention nodes reject
*that tool's own generated* inputs on the **CPU reference arm** (`seqlens_k[0] = 7 is out of range
[0, 1)`) because the model's inputs are interdependent and the runner's generic input generator does
not know that. That is a limit of `rust/modelrunner`'s input generation specifically, it has nothing
to do with MatMulNBits or with this change, and it does **not** mean there is no whole-model CPU
reference available at all: `bench/phi35.py`'s `_run_device` already builds one, on real
hand-constructed feeds (`tests/ops/test_phi35.py`'s `_build_phi35_feeds()`, not generated ones), by
opening the same artifact a second time with `providers=["CPUExecutionProvider"]` and running it
through `classify_outputs` as the §10.0 gate *before* anything is timed — a Vulkan run that
disagrees with that CPU run is refused, never silently reported. Nothing here is a new end-to-end
logits claim beyond what §10.0 already established, and nothing here should be read as one. What is
claimed in this section is the operator, on the model's own bytes, plus the graph-wide traffic
reading — which is the narrowest thing that is still about the real model.

### 25.6 The fallback, and why it is the same mechanism as the control

`ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS` clamps the largest selectable row tile. Setting it to `1`
restores the pre-issue-#7 geometry exactly — `rows = 1` is the seed of the tile search and the
shader's `QB_ROWS == 1u` arm is the verbatim old kernel — without a rebuild. It is deliberately the
*same* knob the A/B above uses, for the reason `gemv_packed`'s override already established: an arm
you cannot switch in-place is an arm you cannot measure honestly, and an arm you cannot measure is
one you will not be able to diagnose in the field either. Values are clamped to `[1, 4]` rather
than trusted, and an unparseable value means "the default" rather than "refuse to run".

Reproduce: `python bench/results/probe_weight_reread.py`;
`python bench/results/ab_row_tile.py`; `python bench/results/probe_real_matmulnbits_rows.py`.
Locked by `bench/test_weight_reread.py` (26 tests), `tests/ops/test_matmulnbits.py` (77 tests),
`rust/tests/validation_control.rs` (5 tests), and the `ops::quant` unit tests.

---

## 26. The real model, end to end — and the kernel nobody had timed (2026-08-06/07)

Every performance section before this one measured a **part**: a weight-read count, an operator
lifted out of the graph, an island boundary, a synthetic chain. §25 closed with a limitation stated
plainly — `rust/modelrunner` reports `UNSUPPORTED` for Phi-3.5, so there was no whole-model
reference and no whole-model timing on this branch.

Issue #56 is that gap. This section is a **real-model** harness: the exact Foundry Phi-3.5 int4
file this repository's own tooling resolves, a second real ONNX model for broad-EP overhead, an
`M` sweep, decode with a genuinely non-empty KV cache, three arms with an identical-pipeline null
control, output verification on every compared arm, and per-kernel GPU attribution.

It is also a correction. The lever PR #53 named next for itself — widening `q_gemv`'s 32-bit
scalar `B` loads — targets a **minority** of the time. The measurement says the largest single
consumer of device time in Phi-3.5 is `gqa_f16`, and the reason is one line of GLSL.

> **Dependency.** This section's branch is stacked on PR #53 (`squad/7-tile-matmulnbits-prefill`).
> The `vulkan_tiled` arm below *is* PR #53's row tile; `vulkan_untiled` is its kill switch. Nothing
> here re-litigates §25 — it measures the whole model that §25 could only measure an operator of.

### 26.1 What is under test, exactly

**Models.** Both resolved by repository tooling, both hashed, neither a hard-coded path:

| model | resolver | sha256 | bytes |
|---|---|---|---|
| `phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx` | `foundry_discovery` (no cache-version literal) | `3dbdd4b5…04dac3f` | 26,180,848 |
| ↳ its external weights `…onnx.data` | — | `9ce390a7…0e92217` | **2,291,238,912** |
| `mobilenetv2-12.onnx` | repo model cache, checksum-pinned | `c0c3f76d…0432ad5` | 13,964,571 |

The `.data` row is not padding. The `.onnx` file is 26 MB of graph; **2.29 GB of int4 weights live
outside it**, and a provenance record that hashed only the graph would have certified a file that
contains almost none of the model. `real_model.external_data_provenance` parses the graph's
`external_data` locations, hashes each blob, and reports `missing: True` rather than skipping —
three tests in `bench/test_real_model.py` lock that.

MobileNetV2 is the second model rather than a second transformer, and the reason is stated rather
than glossed: it is the only other real ONNX file in this repository's pinned cache, the Hugging
Face cache on this box is empty, and downloading a second quantised transformer was not necessary
to answer the question. What it buys is a **negative control on scope** — a graph with no
`MatMulNBits` and no `GroupQueryAttention`, where both of this branch's changes must show *nothing*.

**Arms.** Three, fresh session each, order alternated per repeat:

| arm | providers | environment | role |
|---|---|---|---|
| `vulkan_tiled` | Vulkan, CPU | `GEMV_MAX_ROWS=4` | subject |
| `vulkan_untiled` | Vulkan, CPU | `GEMV_MAX_ROWS=1` | kill switch (§25's A/B) |
| `cpu` | CPU only | — | baseline |

**Device.** `NVIDIA RTX A1000`, driver 573.44, discrete, subgroup 32, 48 KB shared memory, device
index pinned. **There is exactly one Vulkan device on this box** — no second GPU, no software ICD —
and the artifact records that absence explicitly in `second_device` rather than leaving it to
inference. This is **not** the RTX 4060 that most of this repository's canonical proof-ledger
evidence was taken on, and no number here should be compared with one from that machine.

**Device identity, and what these artifacts actually record.** Every §26 artifact was measured
before `main` grew stable device identity (#54, `bench/devices.py` `uuid`/`luid`/`pci`), so their
`environment.device` records `index`, `name` and `driver` and **nothing stronger**: on a box with
two identical cards those three would not say which card ran. That is a real limit of *these*
readings and is stated rather than papered over. `bench/results/probe_real_model_latency.py` now
records `uuid`/`luid`/`pci` alongside them, so the limit ends with the next run of the instrument;
no measurement in §26 changed, and none was re-taken, to add the field.

**Environment.** ORT 1.28.0, Python 3.12.10, Windows 11 10.0.26200. Stock power plan, no affinity
mask, no clock lock — so §20 applies and wall clock is `STEADY_UNCERTIFIED` by default. Arms are
interleaved *because* the box cannot be assumed quiet.

**Which EP built which artifact.** §26 does **not** rest on one build. Seven committed artifacts
carry five *distinct* `ep_library_sha256` values, and no section may be read as though a single
binary produced all of them. The full map, so no hash is implied and none is hidden:

| `ep_library_sha256` | artifacts | sections that read them |
|---|---|---|
| `586e9513cdbe4334…` | `real_model_latency_before_gqa.json`, `real_model_diagnostics_before_gqa.json` | §26.3 (the whole baseline table), §26.4's bandwidth differential, §26.6's `before` column |
| `2684bb6fea730bc8…` | `real_model_gqa_local_size.json` | §26.4 (**every** per-kernel row), §26.5 (the sweep and its 36 bitwise comparisons) |
| `2c080583a1e295bf…` | `real_model_latency.json`, `real_model_diagnostics.json` | §26.6's `after` column and its dispatch/island/fallback counts |
| `752cebcfa00cfcfe…` | `real_model_latency_postmerge.json` | §26.9 |
| `7f050805ef817991…` | `real_model_latency_on_main.json` | §26.10 |

The five differ because this repository's Windows `.dll` is **not byte-reproducible across forced
rebuilds** (recorded in `.squad/decisions.md`), so what can be claimed across them is *source*
identity, never binary identity — §26.10 makes that argument in full. In particular §26.4's
per-kernel table and §26.3's latency table were taken on **different builds**, which is why §26.4
is quoted as a *share* of its own run's `total_us` and never differenced against §26.3's wall
clock.

**Method.** 3 repeats × 5 timed iterations, 2 warmups discarded per session. **The published
millisecond figure is `latency.median_ms` — the median of the 15 *pooled* timed iterations**, not
the median of the three per-repeat medians. Those are a different number and the artifact records
them separately, in `per_repeat_median_ms`; they are what the *paired ratio* statistics
(`row_tile_speedup`, `vulkan_vs_cpu_tiled`, `noise_floor`) are built from, and they are what
§26.3's disjointness table quotes. (An earlier draft of this paragraph called the published figure
a median of per-repeat medians. It is not: at `M = 1` prefill tiled the pooled median is 27.52 ms
and the median of `[27.83, 28.80, 27.31]` is 27.83.) Session build time and first-run time are
recorded **separately** and never folded into
the median: at `M = 128` the first run costs 3,254 ms against a steady-state 1,407 ms, and a
harness that averaged those together would be reporting a number no steady-state user ever sees.

**Feeds.** `input_ids [1, M]`, `attention_mask [1, past+M]`, and 64 real `past_key_values.*`
tensors of `[1, 32, past, 96]` fp16 at `past ∈ {128, 512, 1024}`. Decode is measured with a
**non-empty cache**; `past = 0` is kept only as the empty-cache control.

### 26.2 Correctness first — and the two instrument bugs caught before publication

Every arm's outputs are compared against the CPU EP's before any timing is reported. That gate
found two defects **in this harness**, both before any number was published:

1. A naive relative-tolerance gate called *every* Vulkan arm `DIVERGENT` over a one-fp16-ULP
   difference. A relative error over logits is a cancellation meter, not an accuracy meter.
2. The first fix was an aggregate OR across elements — and a **planted 7.0 error walked straight
   through it**. `test_activation_gate_is_elementwise_not_an_aggregate_or` is the control that
   now keeps it elementwise.

The gates were then *calibrated* rather than loosened. The obvious calibration was unavailable:
the CPU EP is **bit-identical to itself** across `intra_op_num_threads ∈ {1,2,4}`, so there is no
reference-side reorder noise to measure a budget against. The budgets are therefore argued from
fp16 numerics and made falsifiable:

* **Logits** (`bench/real_model.py::classify_logits`, Phi-3.5's output 0) — five clauses, **all**
  required: argmax equal, top-`PHI35_TOP_K = 10` identical, reference not all-zero,
  `max|Δ| ≤ PHI35_LOGIT_SCALE_FRACTION = 0.05 × max|reference|`
  (against a `sqrt(3072·32)·2^-11 ≈ 0.15` theoretical fp16 accumulation envelope, so 3× tighter),
  **and** `max |Δp| ≤ PHI35_MAX_PROB_DELTA = 0.02` on the induced softmax distribution. **Relative
  error is recorded and not gated on** — `max_rel` rides along in the artifact under
  `max_rel_note: "relative error over logits is a cancellation meter, not an accuracy meter;
  reported, not gated on"`, and it reads 374.07 at `M = 1` while the case passes. There is **no**
  RMS-restricted relative clause in this gate; `KV_SIGNAL_FRACTION = 0.1` restricts a *reported*
  figure in the activation gate below and gates nothing. The observed reading at `M = 1`
  is `max_abs 0.0625` against a budget of `0.654`, with `max_prob_delta 3.1e-04`.
* **KV activations** (`classify_activation`, Phi-3.5's 64 `present.*` outputs) — a three-band
  tolerance, elementwise, and **not bitwise**: clean below
  `floor = KV_ULP_BUDGET(16)·ε_fp16(2⁻¹⁰)·max|ref| + KV_REL_TOL(0.05)·|ref|`,
  marginal between `floor` and `KV_GROSS_MULTIPLE = 8` floors for at most
  `KV_MARGINAL_FRACTION = 1e-4` of elements, any element above 8 floors ⇒ `DIVERGENT` on its own.
  The floor is a **sum, not an OR**, which is what the planted-7.0 control in defect 2 above
  proves.
* **MobileNetV2** (`classify_tensor`, output 0) — an elementwise **combined absolute-plus-relative
  tolerance**, `|Δ| ≤ MOBILENET_ATOL(1e-4) + MOBILENET_RTOL(1e-2)·|ref|` for **every** element,
  **and** every row's argmax agreeing, **and** the reference not all-zero. It is not a top-5 gate
  and never was; the tolerance is the same one `rust/modelrunner` justifies for this exact model
  in `bench/results/rust-model-runner/mobilenetv2-12.json`, borrowed rather than re-invented, and
  applied elementwise rather than in that tool's aggregate form — strictly stricter, so a `MATCH`
  here implies a `MATCH` there.
* **The `M = 1` null control only** (`bitwise_identical`) — genuinely byte-for-byte, over all 65
  output tensors of the two Vulkan arms. It is the one comparison in §26 where bitwise is a
  legitimate demand, because the two arms are the same pipeline by construction.

Each gate's own clause list is written into the artifact next to its verdict, at
`models[].equivalence[…].arms[…].primary.gate` and `.worst_secondary.gate`, so the prose above is
checkable against the record rather than against memory —
`bench/test_perf_claims.py::test_the_documented_gates_are_the_gates_the_code_applies` reads the
constants out of `bench/real_model.py` and fails if this list drifts from them.

**Result: 18/18 cases `PASS`, 54/54 arm verdicts `MATCH`** (3 arms × 18 cases,
`models[].equivalence` in `real_model_latency_before_gqa.json` and `real_model_latency.json`) —
both before and after the change in §26.4.

**Of those 54 arm verdicts, 18 are the reference checked against itself and 36 are independent.**
The `cpu` arm *is* the reference, so its record carries `"self": true` and the note *"the reference
arm is compared against itself by construction; recorded so the table has no hole"*. Its `MATCH` is
a schema artefact, not evidence. The evidence is the **36 independent comparisons** — 18 cases ×
`{vulkan_tiled, vulkan_untiled}` vs the CPU reference — and no claim in this section may be
supported by counting the self-checks. Verified by reading the field:
`sum(1 for arm in case.arms if arm.self)` is 18 in each of the two matrix artifacts, 3 in each of
§26.9's and §26.10's.

**That 36 is not §26.5's 36.** The coincidence is unlucky and is called out rather than left to
trip someone: §26.5's 36 are *bitwise* comparisons between local-size arms of one build, from
`real_model_gqa_local_size.json`; the 36 here are *budgeted* comparisons between EP arms and the
CPU reference, from the latency matrices. Different artifacts, different populations, different
gates. The counts this section supports are therefore, in full: **18 cases**, **54 arm verdicts**,
**36 independent comparisons**, **18 reference self-checks**.

These are budgeted verdicts, not bitwise ones — with one exception the artifacts *do* record
bitwise: each `M = 1` null-control case also carries `null_control_bitwise`, which compares the two
Vulkan arms' **65 output tensors byte-for-byte** and reads `identical: true` in every case of every
run (6 cases in each matrix artifact, 2 in each of §26.9's and §26.10's). That is a comparison
between two arms of one build, not between builds.

### 26.3 The baseline, and the honest headline

**Conventions, stated once and enforced by test.** The three millisecond columns are each arm's
`latency.median_ms` from `real_model_latency_before_gqa.json`. The two ratio columns are the
artifact's **own paired per-repeat-median ratios** — `row_tile_speedup.median` (untiled ÷ tiled,
above 1 means the row tile helped) and `vulkan_vs_cpu_tiled.median` (CPU ÷ tiled, above 1 means
the EP wins). They are *paired*: repeat `i` against repeat `i`, then the median of the three. That
is a different statistic from a ratio of the two published medians, and §26.6 — which compares two
separate sessions and therefore has no pairing available — uses the other one and says so.

The two ratio columns are **adjacent and unrelated**, and one of them was published as the other in
an earlier draft of this table (decode `past = 512`: the `row-tile` cell read `0.42×`, which is the
`vk/cpu` cell; the artifact's `row_tile_speedup.median` there is `1.0153`, ratios `1.120 / 0.925 /
1.015`). `bench/test_perf_claims.py::test_no_ratio_cell_is_its_neighbours_value` now fails on
exactly that substitution, for every row, in both directions.

| case | Vulkan tiled | Vulkan untiled | CPU EP | row-tile | vk/cpu | tokens/s |
|---|---|---|---|---|---|---|
| prefill M=1 | 27.52 | 28.12 | 87.87 | 0.994× | 3.06× | 36.3 |
| prefill M=2 | 38.19 | 49.67 | 220.33 | 1.323× | 5.80× | 52.4 |
| prefill M=4 | 60.72 | 88.83 | 234.97 | 1.447× | 3.89× | 65.9 |
| prefill M=8 | 108.39 | 165.02 | 313.38 | 1.506× | 2.88× | 73.8 |
| prefill M=16 | 210.19 | 325.33 | 358.66 | 1.557× | 1.74× | 76.1 |
| prefill M=32 | 462.03 | 704.44 | 541.85 | 1.520× | 1.18× | 69.3 |
| prefill M=64 | 1094.28 | 1588.54 | 843.05 | 1.448× | **0.79×** | 58.5 |
| prefill M=128 | 3004.55 | 4016.45 | 1535.79 | 1.336× | **0.51×** | 42.6 |
| decode past=0 | 29.94 | 27.65 | 87.57 | 0.935× | 2.97× | 33.4 |
| decode past=128 | 80.53 | 83.27 | 114.29 | 1.032× | 1.43× | 12.4 |
| decode past=512 | 328.59 | 336.25 | 138.15 | 1.015× | 0.42× | 3.0 |
| decode past=1024 | 642.62 | 628.90 | 184.30 | 0.970× | **0.28×** | 1.6 |

The headline nobody had written down: **the EP wins at narrow prefill and loses badly at wide
prefill and at real decode.** §25's operator-level win is real and it does not survive contact with
the whole model at `M ≥ 64`. Token throughput *falls* as `M` grows past 16 — the opposite of what
batching is for.

The noise floor is the `M = 1` null control, where the tiled and untiled arms bind identical SPIR-V
under identical specialisation and must be the same pipeline. Quoted as the **min–max over the
three per-repeat ratios** in the artifact's own `row_tile_speedup` orientation (untiled ÷ tiled),
from `real_model_latency_before_gqa.json` → `models[0].noise_floor.ratios` = `[0.99411, 0.926348,
1.264193]`: **0.926 – 1.264**, median 0.994. Row-tile ratios anywhere inside that interval — which
is every decode row, including `past = 512`'s 1.015 — are therefore **not readings**; §20's
`STEADY_UNCERTIFIED` applies to them.
MobileNetV2's floor in this same run is worse still, `[0.988282, 5.605136, 0.939699]`: a **5.6×**
contaminated repeat, which is exactly why the ratios are reported rather than averaged away. (The
3.1× outlier is `real_model_latency.json`'s MobileNetV2 floor — the *after* run of §26.6, not this
one; an earlier draft of this paragraph quoted it here.)

#### The null control is worse than "noisy": in one run the two arms do not overlap at all

A ratio range understates it, so here are the per-repeat medians themselves. `M = 1`, phi-3.5,
`per_repeat_median_ms`, tiled against untiled, in each of the four runs. **This table's ratio is
the arm-asymmetry orientation, tiled ÷ untiled** — the reciprocal of the `row_tile_speedup` field
quoted above — because the question here is *how far apart two arms of the same computation got*,
and above 1 reads "the tiled arm was slower". Both orientations are stated because both appear in
this document, and neither may be silently swapped for the other:

| run | tiled (ms) | untiled (ms) | per-repeat ratios (tiled ÷ untiled) | median | mean | ranges overlap? |
|---|---|---|---|---|---|---|
| §26.3 before | 27.83, 28.80, 27.31 | 27.66, 26.68, 34.53 | 1.006, 1.080, 0.791 | 1.006 | 0.959 | yes |
| §26.6 after | 29.19, 27.29, 27.02 | 29.04, 32.74, 29.42 | 1.005, 0.834, 0.918 | 0.918 | 0.919 | yes |
| §26.9 postmerge | 27.43, 28.01, 26.88 | 27.41, 26.01, 29.26 | 1.001, 1.077, 0.919 | 1.001 | 0.999 | yes |
| **§26.10 proposed head** | **31.32, 31.02, 28.51** | **26.83, 25.35, 27.46** | **1.167, 1.224, 1.038** | **1.167** | 1.143 | **no** |

In the last row the two arms' per-repeat spans are **disjoint** — `[28.51, 31.32]` against
`[25.35, 27.46]`, a 1.05 ms gap between the closest pair — with tiled slower in **every** repeat
by **3.8% to 22.4%**: **median 1.167 (mean 1.143)**, i.e. **approximately 17%**. The median and the
mean are separately labelled deliberately. An earlier draft published the *mean*, 1.143, under the
word "median" and then rounded it to "~14%"; the median of `[1.16739, 1.22372, 1.03830]` is
`1.16739`, and `bench/test_perf_claims.py::test_the_null_control_median_is_a_median` now recomputes
both from the artifact and fails if either label moves onto the other's number.

These two arms produce **byte-identical outputs from an identical pipeline**: at `M = 1`, `gemv_tile_with` returns `(base_cols, 1)` before
it ever reads `max_rows` (`rust/src/ops/quant.rs:491`), so `GEMV_MAX_ROWS = 4` and `= 1` specialise
the same way and dispatch the same grid — and the same artifact's `null_control_bitwise` records
all **65 output tensors identical** in that run. A systematic ~17% separation between two arms that
are the same computation is therefore **an asymmetry of the harness or the machine, not of the
code**: arm order within a repeat, session rebuild, thermal and residency drift, and a shared box
are all live and none is controlled for.

**What that limits.** Read the null control as the *width of this harness's arm-to-arm asymmetry*,
not as a symmetric ±noise band. Two consequences, both binding on everything above:

* **A ratio inside the null control's width is not a reading**, in either direction, whatever its
  sign. Taken over all twelve per-repeat ratios of all four runs that width is **0.791 – 1.224** in
  the tiled ÷ untiled orientation and **0.817 – 1.264** in the `row_tile_speedup` orientation the
  §26.3 table above uses. (An earlier draft quoted "0.79 – 1.26", which is one endpoint from each
  orientation — a band that exists in neither.) That covers every decode row and the MobileNetV2
  scope control.
* **A run can be internally consistent and still wrong.** Non-overlapping spans do not make a
  1.17× separation real; §26.10's `M = 1` row is the proof, since there the "effect" is known to be
  zero by construction. No conclusion in §26 rests on any single run's `M = 1` row, and none may.

This is *not* a claim that the wide-prefill result is noise: §26.6's `M = 128` reads 2.135× where
the floor is 1.26 wide at worst, and it reproduces at 1407.40 / 1403.98 / 1410.09 ms across three
separate builds (§26.10). The floor bounds what may be *read*, and 2.135× clears it by a margin
that 1.057× does not.

### 26.4 Where the time actually goes

ORT's own profiler cannot answer this: it attributes every Vulkan node to the **single fused node**
it hands the EP, so its op table shows one row. The attribution below is from the EP's own GPU
timestamp queries (`ONNXRUNTIME_EP_VULKAN_TRACE` + `..._TRACE_GPU`), aggregated by kernel name.

**Exactly one committed artifact carries that aggregate**, and every row below is read off it:
`bench/results/real_model_gqa_local_size.json` → `timing[].points[local_size == 1].by_kernel_us`,
keys `vulkan.gpu.gqa_f16` and `vulkan.gpu.q_gemv_matmul_nbits_f16` against that point's `total_us`
(which is the sum of its own `by_kernel_us`, checked). `local_size = 1` is the pre-change geometry,
so those points *are* the before state. `real_model_diagnostics.json` is **not** the source and
contains no per-kernel GPU field — it carries ORT's node table, counters, fallback and dispatch
records only.

| case | GPU total | `gqa_f16` | `q_gemv_matmul_nbits_f16` |
|---|---|---|---|
| prefill M=1 | 20.03 ms | 1.48 (7.4%) | 17.46 (87.2%) |
| prefill M=8 | 96.04 | 20.96 (21.8%) | 73.62 (76.7%) |
| prefill M=32 | 433.29 | 154.59 (35.7%) | 276.46 (63.8%) |
| prefill M=128 | 2927.34 | **1891.19 (64.6%)** | 1029.21 (35.2%) |
| decode past=512 | 154.12 | 135.56 (88.0%) | 17.46 (11.3%) |
| decode past=1024 | 287.18 | **268.57 (93.5%)** | 17.52 (6.1%) |

Two facts fall out, and both were surprises:

* **`q_gemv` decode time is flat at ~17.5 ms at every cache length.** All of decode's growth is
  GQA. The quantised GEMM is not the decode problem; it is not even a large part of it.
* **Weight streaming is a minority cost at width.** The tiled and untiled arms differ *only* in how
  many passes they make over the same 2.291 GB of packed weights — `M` passes untiled against
  `ceil(M/4)` tiled — so differencing them gives a marginal streaming bandwidth. Eight prefill `M`
  points yield **seven** differential points: `M = 1` yields none, because both arms make exactly
  one pass there and Δpasses is zero. Over those seven (`M = 2 … 128`, from
  `real_model_latency_before_gqa.json`) the readings are **199.7, 244.5, 242.8, 238.8, 226.8,
  222.5, 217.4 GB/s** — range **200–245**, median **227**, and above the ~192 GB/s spec sheet at
  every point, which is what L2 reuse looks like. `M = 2` is the low end and the noisiest point
  (one differenced pass, no averaging); the six points from `M = 4` up sit in 217–245. One full
  weight pass costs 9.4–11.5 ms (median 10.1). At `M = 128` the tiled arm makes 32 passes, so
  streaming is ~323 ms of the 3004.55 ms tiled time — **~11%**.

So PR #53's self-named next lever — widening `q_gemv`'s 32-bit scalar `B` loads — would be
optimising a tenth of the wide-prefill cost. The data said to go elsewhere, so this branch did.

### 26.5 The finding: one lane per subgroup

`gqa_f16.comp` declared:

```glsl
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
```

**The hardware's unit of scheduling is the subgroup, not the invocation.** A workgroup of one
invocation does not occupy one lane; it occupies a whole subgroup with one lane enabled. On this
device the subgroup is 32 wide, so **31 of every 32 lanes were masked off for the entire kernel** —
consistent with the independent arithmetic that GQA decode reads only ~12.6 MB of K+V per layer yet
takes 8.37 ms/layer, roughly 158× off memory-bandwidth peak.

The fix is to make the size specialisation constant 0 and dispatch `ceil(total/local)` workgroups.
`bench/results/probe_gqa_local_size.py` sweeps it on the real graph. `gqa_f16`'s own GPU
milliseconds per inference, one process per point:

| invocations (`B·Nq·S`) | 1 | 2 | 4 | 8 | 16 | 32 | 64 | rule picks |
|---|---|---|---|---|---|---|---|---|
| 32 — prefill M=1 | **1.48** | 1.76 | 1.69 | 1.70 | 1.89 | 1.97 | 1.81 | 1 ✔ |
| 32 — decode past=512 | **135.56** | 139.72 | 141.11 | 146.84 | 166.96 | 209.19 | 203.80 | 1 ✔ |
| 32 — decode past=1024 | **268.57** | 273.77 | 278.69 | 291.81 | 322.72 | 392.42 | 390.53 | 1 ✔ |
| 256 — prefill M=8 | 20.96 | 16.58 | 11.10 | 5.92 | 5.30 | **5.10** | 6.09 | 8 |
| 1024 — prefill M=32 | 154.59 | 90.78 | 60.57 | 49.43 | 35.37 | **25.48** | 25.69 | 32 ✔ |
| 4096 — prefill M=128 | 1891.19 | 1002.68 | 549.56 | 315.98 | 229.62 | 201.71 | **194.27** | 64 ✔ |

Up to **9.7×** on the kernel. The curve is not monotone in the size — it is monotone in the
*ratio* of invocations to workgroups, which is why the rule is a function of the work available
(`gqa_local_size`: largest power of two ≤ 64 leaving ≥ 32 workgroups) and not a constant.

The rule is best or tied-best at five of six points. At `M = 8` it picks 8 (5.92 ms) over the
sweep's best 32 (5.10 ms) — 16% on the smallest absolute number in the table. Buying that 0.8 ms
means lowering `GQA_MIN_GROUPS` to 8, which moves the 32-invocation rows from size 1 to size 4 and
costs 135.56 → 141.11 ms and 268.57 → 278.69 ms on the two decode rows. **Decode is where a
generation loop lives, so the trade is declined and the 16% is left on the table deliberately.**

**Equivalence.** The packing changes scheduling, not arithmetic — a claim about the source text, so
it is verified as one. `probe_gqa_local_size.py` compares each **non-reference** size's whole-model
outputs byte-for-byte against the `local = 1` reference in the same case; it does not compare the
reference with itself, and a self-comparison would be the one comparison that cannot fail. So the
arithmetic of the sweep is: **6 cases × 7 sizes = 42 measured points**, of which **6 cases × 6
non-reference sizes = 36 cross-arm comparisons**, and
`real_model_gqa_local_size.json` → `equivalence[].comparisons[].verdict` records **36 of 36
`BITWISE-IDENTICAL`**, 65 output tensors per comparison. No tolerance is involved and none would be
appropriate.

Do not confuse that 36 with §26.2's equivalence gate, which is a different artifact, a different
schema and — unluckily — the same number: `real_model_latency*.json` → `models[].equivalence`
records **18 cases**, each with a `PASS`/`FAIL` gate, and **54 arm verdicts** (3 arms × 18 cases),
of which **18 are the reference arm compared with itself** (`self: true`) and **36 are independent
comparisons**, those under the recorded budgets rather than bitwise. Two different 36s, from two
different files.

### 26.6 The result

Same harness, same box, same method, same day. `before` is §26.3's table; `after` is a full re-run.

**Convention, stated once and enforced by test — this table is *ratios of medians*, throughout.**
`before` and `after` are two separate sessions, so no repeat of one pairs with any repeat of the
other and the paired statistic §26.3 uses does not exist here. Every ratio below is therefore
computed the one way that does: `gain` is `before.vulkan_tiled.latency.median_ms ÷
after.vulkan_tiled.latency.median_ms`, and each `vk/cpu` cell is that run's
`cpu.latency.median_ms ÷ vulkan_tiled.latency.median_ms`. **The `vk/cpu before` cells are therefore
not copies of §26.3's** — §26.3 publishes the artifact's paired `vulkan_vs_cpu_tiled.median`, which
is a different statistic and differs by up to 2% at the noisier points (`M = 64`: **0.77** here,
`0.787` there). An earlier draft of this table took its `before` cells from §26.3 and computed its
`after` cells as ratios of medians, mixing the two conventions inside one column;
`bench/test_perf_claims.py::test_26_6_is_ratios_of_medians_throughout` recomputes every cell from
the two artifacts and fails on either convention leaking into the other.

| case | before | after | gain | vk/cpu before → after | tokens/s |
|---|---|---|---|---|---|
| prefill M=1 (null control) | 27.52 | 27.29 | 1.008× | 3.19 → 3.32 | 36.6 |
| prefill M=2 | 38.19 | 33.67 | 1.134× | 5.77 → 6.67 | 59.4 |
| prefill M=4 | 60.72 | 54.40 | 1.116× | 3.87 → 4.41 | 73.5 |
| prefill M=8 | 108.39 | 94.13 | 1.151× | 2.89 → 3.15 | 85.0 |
| prefill M=16 | 210.19 | 181.02 | 1.161× | 1.71 → 1.98 | 88.4 |
| prefill M=32 | 462.03 | 345.98 | 1.335× | 1.17 → 1.55 | 92.5 |
| prefill M=64 | 1094.28 | 691.26 | **1.583×** | 0.77 → **1.18** | 92.6 |
| prefill M=128 | 3004.55 | 1407.40 | **2.135×** | 0.51 → **1.05** | 90.9 |
| decode past=0 | 29.94 | 29.92 | 1.001× | 2.92 → 3.01 | 33.4 |
| decode past=128 | 80.53 | 88.05 | 0.915× | 1.42 → 1.29 | 11.4 |
| decode past=512 | 328.59 | 331.16 | 0.992× | 0.42 → 0.43 | 3.0 |
| decode past=1024 | 642.62 | 608.21 | 1.057× | 0.29 → 0.31 | 1.6 |
| mobilenet N=1…32 | 8.12–271.03 | 8.19–269.75 | 0.992–1.010× | unchanged | 118–139 |

Read this table with three cautions on the face of it:

* **The decode rows are not readings.** At 32 invocations the rule returns `local = 1`, so decode's
  dispatch is *literally the same geometry, the same grid and the same pipeline* as before. The
  0.915–1.057× spread is the box, and it sits inside the `M = 1` null control's own width
  (0.834–1.005 in this run, tiled ÷ untiled; 0.791–1.224 taken across all four runs, §26.3).
  Reporting it as an effect would be reporting §20's noise as a result.
* **MobileNetV2 is the scope control and it did what a control should**: 0.992–1.010×, a graph with
  no GQA node showing nothing. Had it moved, the change would not have been what this section says
  it is.
* **The `M = 1` prefill row is the null control** and stayed at 1.008× *here*. It does not stay
  there in every run — §26.10's two arms sit at a median 1.167 (mean 1.143) apart between two arms
  that are the same computation — which is why §26.3 quotes the control's width rather than any
  single run's value.

What *is* a reading: **wide prefill.** `M = 128` is 2.14× faster and the EP crosses from losing to
the CPU EP (0.51×) to beating it (1.05×); `M = 64` crosses from 0.77× to 1.18×. Peak prefill
throughput rises from 76 tok/s to **92.6 tok/s**, and — the part that matters for "utilise the
device" — throughput now *stays* near its peak from `M = 16` to `M = 128` instead of collapsing.
Dispersion is tight where the claim is largest: at `M = 128`, `rsd = 0.008`, p05 1398.59, p95
1427.42 over 15 samples.

Per-kernel, at the sizes the rule picks:

| case | GPU total before → after | `gqa_f16` before → after |
|---|---|---|
| prefill M=8 | 96.04 → 79.02 ms | 20.96 → 5.92 (21.8% → 7.5%) |
| prefill M=32 | 433.29 → 313.84 | 154.59 → 25.48 (35.7% → 8.1%) |
| prefill M=128 | 2927.34 → 1347.23 | 1891.19 → 194.27 (64.6% → **14.4%**) |

At `M = 128` the graph's GPU time is now 85% `q_gemv` again — which is where PR #53's own next
levers become the right thing to do, and were not before.

**Device utilisation, grounded.** Islands stay at **1** (no fragmentation), dispatches at 355 per
inference, CPU fallback at 24 node executions on Vulkan against 1,377 on the CPU EP — all unchanged
by this edit, which is the point: the change moved lane occupancy, not graph partitioning.

### 26.7 Limitations, stated rather than implied

* **Decode is still bad, and this change does not fix it.** At `past = 1024` the EP is 0.31× the
  CPU EP and `gqa_f16` holds 93.5% of GPU time. There is no packing to do at 32 invocations. The
  next lever is parallelism over the **KV sequence** inside a workgroup with a reduction — which is
  a shared-memory and barrier change, i.e. a portability question of exactly the kind §25 declined
  to open without evidence. It should be opened now; it has evidence.
* **The harness feeds KV from host numpy.** At `past = 1024` the wall is 608 ms against 287 ms of
  GPU, so ~320 ms is host-side round trip — roughly 805 MB staged at ~2.3 GB/s. A real generation
  loop would use IO binding and keep the cache device-resident. The decode wall numbers here are
  therefore an **upper bound** on a real loop's, and the GPU-time column is the fairer comparison.
* **One device.** RTX A1000 only; there is no second GPU and no software ICD on this box. The
  *rule* is a function of the invocation count (a property of the model), but the best *size* is a
  property of the machine — which is why `ONNXRUNTIME_EP_VULKAN_GQA_LOCAL_SIZE` exists.
* **GPU time is a timestamp-query total, not an occupancy counter.** It says the kernel finished
  sooner. The subgroup-occupancy explanation is consistent with it and with the 158×-off-peak
  arithmetic, but this harness cannot read a hardware occupancy counter and does not claim to.
* **`rust/modelrunner` still reports `UNSUPPORTED` for Phi-3.5** (its GQA nodes reject generated
  inputs on the CPU *reference* arm). §25's limitation is unchanged; what this section adds is a
  harness that supplies interdependent inputs itself, which is why it can compare whole-model
  outputs where the runner cannot.
* **The bandwidth figure is differential, not instrumented.** It comes from Δtime ÷ Δ(weight passes
  × 2.291 GB) between two arms, so it inherits both arms' noise and assumes the arms differ only in
  weight passes — which is true by construction of the `QB_ROWS` specialisation, but is an argument
  about the source, not a counter reading.
* **The null control is an arm-asymmetry width, not a symmetric noise band.** In §26.10's run the
  two arms' per-repeat spans are disjoint at `M = 1` — where the effect is zero by construction —
  with a **median ratio of 1.167 (mean 1.143)**, tiled ÷ untiled. Any row whose arm ratio sits
  inside 0.791 – 1.224 (tiled ÷ untiled) or the reciprocal 0.817 – 1.264 (the `row_tile_speedup`
  orientation §26.3's table uses) is not a reading, in either direction; see §26.3. This is the
  limitation that governs every decode row.
* **Equivalence counts include the reference checked against itself.** The `cpu` arm carries
  `"self": true`, so 54 arm verdicts are 36 independent comparisons plus 18 self-checks (and 9 are
  6 plus 3). Verdict totals in this section always name which they mean.
* **No reading here identifies its device beyond index, name and driver.** Every §26 artifact was
  taken before `main` grew stable device identity (#54: `uuid`/`luid`/`pci`), so on a box with two
  identical cards these artifacts could not say which one ran. This box has exactly one Vulkan
  device and records that absence explicitly, which is what closes the gap *for these readings* —
  not the identity field, which arrives with the next run of the instrument.
* **No §26 number was re-measured on the head that is now proposed.** §26.6/§26.9/§26.10 were taken
  at `024027d`; two `main` merges (#54, #62) have moved `rust/src/` since. No shader moved
  (`git diff 024027d..HEAD -- rust/shaders` is empty) and the GQA dispatch rule is untouched, so
  the kernel readings stand on identical SPIR-V — but #54 changes device *selection*, and a future
  reading on a multi-device box could legitimately differ.

### 26.8 Reproduce

```
python bench/results/probe_real_model_latency.py                 # the matrix -> real_model_latency.json
python bench/results/probe_real_model_latency.py --diagnose      # -> real_model_diagnostics.json
python bench/results/probe_gqa_local_size.py                     # -> real_model_gqa_local_size.json
```

Artifacts: `bench/results/real_model_latency.json`,
`real_model_latency_before_gqa.json` (§26.3's baseline, kept so the before column is a file and not
a memory), `real_model_diagnostics.json`, `real_model_diagnostics_before_gqa.json`,
`real_model_gqa_local_size.json`, `real_model_latency_postmerge.json` (§26.9) and
`real_model_latency_on_main.json` (§26.10).

**Two test modules lock this section, and they lock different things.**

`bench/test_real_model.py` (**97** GPU-free tests) locks the *harness*: provenance, feeds, arm
isolation, the statistics helpers and every equivalence gate, including the planted-error controls.
One of those is a scar. `--diagnose` inherited the timed pass's default `--out` and
overwrote a completed thirteen-minute matrix with a profiling record of a different schema;
`test_the_two_passes_do_not_default_to_the_same_file` is the control, and the matrix was re-run
rather than reconstructed. Thirty-five of them arrived with issue #78 and are about *identity*
rather than timing: they drive the shipped `resolve_model` on a pinned model, and they include a
cross-reader agreement arm proving `rust/tools/model_provenance.verify_file` and
`bench/pinned_bytes.check_pinned_bytes` cannot disagree where their remits overlap. The identity
authority itself is locked separately by `bench/test_pinned_bytes.py` (**266** GPU-free tests)
and `bench/test_path_screen.py` (**76**), each written against a named mutant of the production
module rather than against its own copy of the rule — see `DESIGN.md` §9.1.5.

`bench/test_perf_claims.py` (**24** GPU-free tests) locks *this document against those artifacts*.
It exists because the first review of this section found published numbers that were internally
plausible and wrong: a ratio copied from its neighbouring column, a mean labelled a median, a
by-kernel row in no artifact, a citation to a file that does not carry the field, and a gate
described as bitwise that is a three-band tolerance. Each of those five failure modes now has a
test that fails on it, and the tests re-derive from `bench/results/*.json` rather than restating a
constant.

The GQA dispatch rule is locked by **ten** `ops::attention` unit tests, all of them added by this
change: `gqa_decode_stays_at_one_invocation_per_workgroup`,
`gqa_local_size_matches_the_measured_sweep`, `gqa_local_size_never_exceeds_the_portable_cap`,
`gqa_local_size_override_of_zero_is_clamped_not_dispatched`,
`gqa_local_size_override_of_one_restores_the_original_geometry`,
`gqa_dispatch_grid_covers_every_invocation`, `gqa_local_size_never_decreases_as_work_grows`,
`gqa_local_size_of_zero_work_is_still_dispatchable`,
`translate_gqa_prefill_packs_the_workgroup_and_declares_it` and
`translate_gqa_decode_declares_the_unpacked_size`. The count is derived, not asserted:
`bench/test_perf_claims.py::test_this_section_names_every_gqa_dispatch_test_that_exists` parses
`rust/src/ops/attention.rs` and fails if a test is added, removed or renamed without this list
moving with it.

### 26.9 The reading survived the merge, which is the only reason it is quoted

Everything above was measured on this branch's first base. The branch then merged PR #53's
advanced head (`8f12b32`, which carries `bb09871` and `5cd4087`), and a merge that touches no
kernel is exactly the kind of change a benchmark is assumed to be indifferent to — an assumption
worth one run rather than one sentence. Three points were re-measured on the merged build, same
protocol, artifact `bench/results/real_model_latency_postmerge.json`:

| case | §26.6 (ms) | after the merge (ms) | untiled, after (ms) | CPU EP, after (ms) |
|---|---|---|---|---|
| prefill M=1 | 27.29 | 27.08 | 27.73 | 91.95 |
| prefill M=128 | 1407.40 | 1403.98 | 2351.75 | 1488.31 |
| decode past=1024 | 608.21 | 627.50 | 630.46 | 190.28 |

All three cases are `PASS` on the equivalence gate, with `MATCH` on all 9 arm verdicts (3 arms ×
3 cases) — **6 independent comparisons and 3 reference self-checks** (`self: true` on the `cpu`
arm), under §26.2's budgets rather than bitwise, plus `null_control_bitwise` on the 2 `M = 1` cases
(65 outputs, `identical: true`). The two prefill
points move by 0.2% and 0.8%; the decode point moves 3.2%, which is inside the host-round-trip
variance §26.7 already describes and is not a reading about the kernel — decode's dispatch geometry
is unchanged by this change and by the merge alike. The merge did not move the result.

### 26.10 Re-measured a third time, on the head that is actually proposed

§26.9's build no longer exists as a proposable tree. PR #53 landed on `main` as the **squash**
`ca61252`, not as `8f12b32`, so the branch this section belongs to was rebuilt as a single commit
on top of `main` and then merged `main` twice more (`5113a0a`, then `3e38ae3`). None of those
commits touch `rust/`: at the head measured here, `git diff --stat ed73a4a HEAD -- rust` was empty
and the whole delta was five `ci/` and `tests/` files from #61.

**This section was measured at `024027d`, and the head now proposed is not that tree.** Two
later `main` merges — #54 (stable-identity Vulkan device selection) and #62 (the landing
simulator) — moved `rust/src/` under `vk/instance.rs`, `vk/device.rs`, `factory.rs`, `registry.rs`
and others, and this branch's own revision moved eight rustdoc lines in `ops/attention.rs`. What is
*not* moved is what these numbers are about: `git diff 024027d..HEAD -- rust/shaders` is **empty**,
so every SPIR-V module in the table below is byte-identical to the one that produced it, and the
GQA dispatch geometry is unchanged (`gqa_local_size` and `GQA_MIN_GROUPS` are untouched, and the
36 comparisons of §26.5 are read off an artifact, not re-derived). Nothing in §26 was re-measured
on the merged head, and nothing here claims it was: #54 changes which device the EP *selects* and
how it *identifies* it, which is a real reason a future reading could differ, and is recorded as a
limitation rather than dismissed.

**The exact property that holds is source identity, not binary identity, and the artifacts say so.**
Each of the three runs records the sha256 of the `.dll` it loaded, and all three differ:
`2c080583…` (§26.6), `752cebcf…` (§26.9), `7f050805…` (here). That is expected rather than
alarming — `.squad/decisions.md` records that this repository's Windows `.dll` is not byte-
reproducible across forced rebuilds, which is why `ci/check_artifact_frame.py`'s `subject_moved`
arm is off by default on Windows — but it means "the same EP" can only be claimed of the **source**:
identical `rust/` tree, three separate compilations. So the honest expectation was not "no change
because it is the same binary" but "no change *if* a recompilation of identical source is
behaviourally equivalent" — which is the thing worth measuring.

`cargo build --release` from the proposed head, then the same three points, same protocol, artifact
`bench/results/real_model_latency_on_main.json`:

| case | §26.6 (ms) | §26.9 (ms) | on the proposed head (ms) | untiled (ms) | CPU EP (ms) |
|---|---|---|---|---|---|
| prefill M=1 (null control) | 27.29 | 27.08 | 29.83 | 26.10 | 85.69 |
| prefill M=128 | 1407.40 | 1403.98 | 1410.09 | 2363.04 | 1531.89 |
| decode past=1024 | 608.21 | 627.50 | 652.18 | 654.13 | 185.21 |

What reproduces across the three builds, stated as narrowly as the artifacts support:

* **Timing, at the one point this change is about.** `M = 128` reads 1407.40 / 1403.98 / 1410.09 ms
  — each within 0.25% of §26.6 and a total spread of 0.44% (6.11 ms) — and the advantage over the
  untiled arm reproduces at 1.681× / 1.675× / 1.676×.
* **Outputs, under the recorded budgets.** 3/3 cases `PASS` and 9/9 arm verdicts `MATCH` in each
  run's `models[].equivalence` — and, counted honestly, that is **6 independent comparisons plus 3
  reference self-checks** per run, because the `cpu` arm is the reference and its record carries
  `"self": true`. Those six are the §26.2 logit and activation budgets, **not** a claim that the
  three builds agree byte-for-byte with each other: no artifact here compares one build's output
  bytes with another's. Each run *does* carry bitwise evidence *within* itself —
  `null_control_bitwise` on its 2 `M = 1` cases, 65 output tensors, `identical: true` — as does
  §26.5's 36-comparison local-size sweep; both compare arms of one build.

Read together with the §26.3 null-control analysis, that is the ceiling on what these three runs
can show: **the M = 128 timing reproduces, and no output claim between builds is available at all.**

The two rows that move more are the two rows §26.7 already declines to read. The `M = 1` null
control is the sharper of the two and is understated by calling it a sign inversion: in **this**
run the tiled arm is slower in **every** repeat and the two arms' per-repeat spans do not overlap —
`[28.51, 31.32]` against `[25.35, 27.46]` ms, a **median ratio of 1.167 (mean 1.143)** over
per-repeat ratios `[1.16739, 1.22372, 1.03830]`, tiled ÷ untiled, i.e. a per-repeat range of
1.038 – 1.224. Both arms specialise identically at `M = 1` and this run's own
`null_control_bitwise` records their 65 outputs as identical, so a systematic ~17% separation is
a property of the harness or the box, never of the code. §26.3 quantifies it and bounds what may be
read from it. Decode moves 4% for the same host-round-trip reason as §26.9. Neither is a kernel
reading, and neither is claimed as one.

Reproduce:

```
python bench/results/probe_real_model_latency.py --device 0 \
  --models phi-3.5-mini-instruct-cuda-int4-rtn-block-32 \
  --prefill-m 1,128 --decode-past 1024 --out bench/results/real_model_latency_on_main.json
```

The history reconciliation itself is a hazard rather than a result, and is written down where it
can be found — and *opened* — by the next person who stacks on an unmerged PR: `.squad/decisions.md`,
entry **"2026-08-07: the stacked-branch squash ledger hazard"**. That file is tracked, so the
citation resolves on any checkout of this branch. (An earlier draft of this paragraph cited
`.squad/decisions/inbox/niobe-stacked-branch-squash-ledger-hazard.md`; `.squad/decisions/inbox/` is
`.gitignore`d — `.gitignore:52` — so that path exists in no tree on any ref and could not be read
by a reviewer. The analysis below is the record, and it now lives at a committed path.)

The short version is that a squash-merged dependency carries the *pre-change* state of every file
you edited on top of it, and is *younger* than your commits, so `ci/check_ledger_census.py`'s
reversed topological walk reads it as an undeclared backward witness move — and a backward
transition can never be declared, because
`tests/ops/test_proof_ledger.py::test_every_declared_transition_lands_on_the_digest_this_build_computes`
correctly requires every declared transition to end at what the build computes. Rebuilding the
stack on `main` is the only fix; merging `main` is not, because CI screens `refs/pull/N/merge`,
which has the same shape.
