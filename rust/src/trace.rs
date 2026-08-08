//! Env-gated observability tracing for the Vulkan EP — the sibling of
//! `onnxruntime-mlx/rust/src/trace.rs`, adapted to an explicit, asynchronous GPU API.
//!
//! Owned by Niobe (performance). Nothing in this module may change what the EP computes; it
//! records what the EP *did* and how long each part of it took, and it is off unless asked for.
//!
//! # What is the same as the MLX EP, and why
//!
//! Both EPs export **Chrome Trace JSON** through Justin's `onnx-runtime-tracer` crate, with
//! `default-features = false` (no prost/Perfetto proto). Both keep a process-wide tracer
//! singleton, gate every entry point on one atomic load, accumulate a human-readable session
//! summary, and write the trace on EP teardown. Both are pinned to a tracer release whose clock
//! is **absolute** (microseconds on the OS monotonic clock, origin = the machine, not the
//! library): a plugin cdylib links its own copy of the tracer with its own statics and can never
//! share a process-global epoch with the host, so an absolute origin is the only thing that lets
//! a span emitted in here be laid over a host trace — or over an MLX EP trace on another
//! machine-hour — with no offset to negotiate. See the pin comment in `rust/Cargo.toml`.
//!
//! # What is different, and why it has to be
//!
//! MLX has a lazy graph and unified memory: one fused subgraph becomes one `mlx_eval`, that call
//! is *synchronous*, and its CPU wall time genuinely is the GPU-inclusive time of the whole
//! subgraph. None of that is true here.
//!
//! Vulkan is explicit and asynchronous. We record a command buffer, we submit it, and the GPU
//! runs it *later*, on its own clock, while our thread is somewhere else. Consequently:
//!
//! * **`vkQueueSubmit` returns before any shader has run.** A host-side wall time around the
//!   submit call measures the driver's bookkeeping — typically single-digit microseconds — and
//!   says nothing whatsoever about the work. [`Phase::Submit`] is labelled `host-only` for
//!   exactly this reason, and the summary prints that label next to the number. If you ever see
//!   a "GPU time" in this project that came from a host clock around a submit, it is wrong.
//! * **`vkWaitForFences` is not GPU time either.** It is *queue latency + GPU execution + any
//!   time the GPU spent on someone else's work*, minus whatever of the execution overlapped our
//!   own recording. It is a legitimate and important number — it is what the user waits for —
//!   but it is an upper bound on kernel time, never the kernel time. [`Phase::FenceWait`].
//! * **Transfers are explicit and they are ours to count.** Staging upload and readback are
//!   separate spans with byte counts ([`Phase::Upload`], [`Phase::Readback`]), because on a
//!   discrete GPU they are frequently the whole story and a benchmark that hides them is
//!   marketing (charter, and `DESIGN.md` §9.2).
//! * **Recording is amortised, not per-inference.** `ENGINE.md` §6.1 records once and replays
//!   the same `VkCommandBuffer`, so the first inference pays [`Phase::Record`] and the rest do
//!   not. [`RecordPath`] distinguishes the two, and a shape-key change that forces a
//!   re-record is called out as [`RecordPath::Rerecord`] — the Vulkan analogue of MLX's
//!   compile-cache `RETRACE`.
//! * **Real GPU time comes from the device's own clock or not at all.** See below.
//!
//! # The span vocabulary
//!
//! This table is **load-bearing**: `bench/phases.py` and `bench/cuda_profile.py` cite it as the
//! declaration of what the artifact contains, and `bench/trace_vocabulary.py` parses this module
//! so a phase that exists here and not there (or the reverse) is a test failure rather than a
//! silently dropped row. A span added to this module and not to this table is a defect.
//!
//! **Structural spans** (`cat == "ep"`) bracket regions. They are never summed, never enter a
//! sibling total, and are not [`Phase`]s:
//!
//! | Span | cat | Clock | What it means |
//! |---|---|---|---|
//! | `vulkan.compute_call` | `ep` | host | The instrumented success-path region opened inside `compute_impl`; absent when a call early-outs before `compute_impl` runs. Buckets every EP span emitted on that path — not all of ORT's `Compute` entry. Contains `vulkan.subgraph`. |
//! | `vulkan.subgraph` | `ep` | host | One fused subgraph's dispatch region, opened inside `dispatch_ort`. |
//!
//! **Phases** (`cat == "ep.phase"`) are the summable vocabulary; each carries `nested_in`:
//!
//! | Span | cat | Clock | What it means |
//! |---|---|---|---|
//! | `vulkan.compile` | `ep.phase` | host | `Compile`: plan build, pipeline/SPIR-V creation, descriptor layout. Once per subgraph. |
//! | `vulkan.prepack` | `ep.phase` | host | Weight prepack + upload of block-quantised initializers. Once per `PackKey`. |
//! | `vulkan.record` | `ep.phase` | host | The `Compute` recording bracket. **Despite the name, dominated by the staging upload it contains (~96-98% on Phi-3.5), not by command recording (1-3%).** See `Phase::Record::caveat`. |
//! | `vulkan.upload` | `ep.phase` | host | Host→device staging copy of inference inputs. Carries `bytes`. **Nested in `record`.** |
//! | `vulkan.cmd_upload` | `ep.phase` | host | The command-buffer-side bracket around the same memcpy as `upload`. **Nested in `record`; overlaps `upload` — take the larger, never the sum.** |
//! | `vulkan.desc_alloc` | `ep.phase` | host | Descriptor-set allocation, once per dispatch while recording. **Nested in `record`.** |
//! | `vulkan.pipeline_lookup` | `ep.phase` | host | Pipeline cache lookup, once per dispatch while recording. **Nested in `record`.** |
//! | `vulkan.submit` | `ep.phase` | host | **`vkQueueSubmit` only.** Host bookkeeping. Measures no GPU work. |
//! | `vulkan.fence_wait` | `ep.phase` | host | CPU blocked on the fence. Upper bound on GPU time, not GPU time. |
//! | `vulkan.readback` | `ep.phase` | host | Device→host copy of outputs. Carries `bytes`. **Nested in `record`.** |
//!
//! **Other events:**
//!
//! | Event | cat | ph | What it means |
//! |---|---|---|---|
//! | `vulkan.gpu.*` | `gpu` | `X` | GPU execution, from `VkQueryPool` timestamp queries only. Emitted on a separate device lane. |
//! | `vulkan.path[FIRST_RECORD\|REPLAY\|RERECORD]` | `ep.path` | `i` | Instant: did this call reuse the command buffer? Carries `path`, `nodes`, `subgraph`, `shape_key`. |
//! | `vulkan.transfer_bytes`, `vulkan.transfer_gib_s` | `counter` | `C` | Explicit host↔device transfer volume and rate. |
//! | `vulkan.island_count`, `vulkan.largest_island_flops`, `vulkan.concentration`, `vulkan.boundary_bytes` | `counter` | `C` | Partition shape, once per session. |
//! | `vulkan.getcapability` | `ep.claim` | `i` | What the EP claimed at partitioning time. |
//! | `vulkan.session_summary` | `summary` | `i` | End-of-session fold. |
//!
//! # GPU timing
//!
//! There is exactly one honest source of GPU-side time here: **`VK_QUERY_TYPE_TIMESTAMP` queries
//! written into the command buffer by the engine.** This module cannot produce them — it must not
//! name Vulkan at all — so it defines the shape of the answer instead: the engine hands back a
//! [`GpuTimestampReport`] and [`VulkanTracer::record_gpu_intervals`] converts device ticks to the
//! shared microsecond axis and emits the spans. What the engine has to provide, precisely, is in
//! `docs/PERF.md` §3 (the requirement routed to Switch).
//!
//! # Cost when disabled
//!
//! Unset env → [`TraceContext::noop`], and every entry point is a relaxed atomic load and an
//! early return. No clock reads, no allocations, no formatting. The engine's call sites take a
//! `None` guard and do nothing.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use onnx_runtime_tracer::{Args, MemoryCollector, SpanGuard, TraceContext};
use std::sync::Arc;

use crate::engine::NodeDesc;

// -------------------------------------------------------------------------------------------
// Environment
// -------------------------------------------------------------------------------------------

/// Set to a filesystem path to enable tracing; the Chrome Trace JSON is written there on EP
/// teardown. Unset → tracing disabled (near-zero cost).
///
/// This is the **same** variable `logging.rs` already documents as "trace file path; implies
/// `Debug`" — one variable, one meaning, so a developer who turns on the trace also gets the log
/// records that explain it.
pub const ENV_TRACE: &str = crate::logging::ENV_TRACE;

/// Set to `1` to print the end-of-run session summary to stderr even when JSON tracing is off.
/// Reuses the crate's existing verbosity switch rather than inventing a second one.
pub const ENV_VERBOSE: &str = crate::logging::ENV_VERBOSE;

/// Set to `1` to ask the engine to write `VkQueryPool` timestamps around dispatches.
///
/// Opt-in and separate from [`ENV_TRACE`] because timestamp queries are not free: they are
/// pipeline-stage writes inside the command buffer, they force a query-pool reset each recording,
/// and on tile-based mobile GPUs a mid-pass timestamp can split a render/compute pass. A
/// steady-state latency number should be measured **without** this, and the kernel attribution
/// measured in a separate run **with** it. The engine reads this; this module only publishes the
/// name and the meaning so both sides agree.
pub const ENV_GPU_TIMESTAMPS: &str = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU";

/// Trace arg key: numeric ONNX Runtime graph node id.
pub const ARG_NODE_ID: &str = "node_id";
/// Trace arg key: ONNX node name.
pub const ARG_NODE: &str = "node";
/// Trace arg key: non-default ONNX operator domain.
pub const ARG_DOMAIN: &str = "domain";
/// Trace arg key: the device a span's work ran on.
pub const ARG_DEVICE: &str = "device";
/// Trace arg key: bytes moved or produced.
pub const ARG_BYTES: &str = "bytes";
/// Trace arg key: floating-point operation estimate.
pub const ARG_FLOPS: &str = "flops";
/// Trace arg key: selected shader variant.
pub const ARG_VARIANT: &str = "kernel_variant";

// --- Structural span names ------------------------------------------------------------------
//
// These are `cat == "ep"` *structural* spans, not [`Phase`]s. They bracket regions; they are
// never summed into a sibling total and they are not part of the phase tree. They are named as
// constants because three Python modules match on these strings and a name that is a prefix of
// another name is a defect, not a style question:
//
// `record_path()` emits **instants** whose names used to be `vulkan.compute[REPLAY]` and friends.
// Every subgraph matcher in `bench/` matched with `startswith`, so a compute-call bracket span
// named `vulkan.compute` would have been captured by the same matcher as those instants and the
// reduction would have silently mixed a span vocabulary with an instant vocabulary. The instants
// are now `vulkan.path[...]`. `vulkan.record_path[...]` was tried first and rejected by
// `no_trace_name_is_a_prefix_of_another`, because `vulkan.record` is a phase and prefixes it —
// which is the point of having the invariant as a test rather than as a naming habit.

/// The instrumented success-path region inside `compute_impl` — **not** ORT's literal `Compute`
/// entry, which additionally covers the null check, `this_info`, the `guard_ffi_status` wrapper
/// and `disclose_broken_commitment`. See [`VulkanTracer::compute_region`].
pub const SPAN_COMPUTE_CALL: &str = "vulkan.compute_call";
/// One fused subgraph's dispatch region, opened inside `dispatch_ort`. See
/// [`VulkanTracer::subgraph_region`].
pub const SPAN_SUBGRAPH: &str = "vulkan.subgraph";
/// Name prefix of the `cat == "ep.path"` **instants** emitted by
/// [`VulkanTracer::record_path`] — `vulkan.path[FIRST_RECORD|REPLAY|RERECORD]`.
///
/// Deliberately shares no prefix with [`SPAN_COMPUTE_CALL`] or [`SPAN_SUBGRAPH`].
pub const INSTANT_RECORD_PATH: &str = "vulkan.path";

/// The `device` arg value for host-side spans. Deliberately not `"gpu"`: a host span that claims
/// to be GPU work is how a trace starts lying.
const DEVICE_HOST: &str = "host";
/// The `device` arg value for spans whose timestamps came from the device's own clock.
const DEVICE_GPU: &str = "vulkan-gpu";

/// The synthetic thread-lane id GPU spans are emitted on.
///
/// GPU work does not belong to the OS thread that submitted it — placing it on that thread's lane
/// would draw kernel execution inside whatever the CPU happened to be doing, which is precisely
/// the confusion this module exists to prevent. A dedicated lane per queue keeps the two
/// timelines visually separate and lets overlap be *seen*.
const GPU_LANE_BASE: u64 = 0x7600_0000;

/// The synthetic trace lane that GPU work from `queue_family` is drawn on.
fn gpu_lane(queue_family: u32) -> u64 {
    GPU_LANE_BASE + u64::from(queue_family)
}

// -------------------------------------------------------------------------------------------
// Phases
// -------------------------------------------------------------------------------------------

/// A timing phase of the Vulkan EP's work, and — importantly — which clock can see it.
///
/// The `host-only` marking is not decoration. `Submit` and `FenceWait` are the two places where
/// a reader is most likely to mistake a host number for a GPU number, so the marking is carried
/// all the way into the printed summary.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Phase {
    /// `Compile`: plan build, pipeline creation, descriptor-set layout. Once per subgraph.
    Compile,
    /// Weight prepack (CPU repack) plus the upload of the packed bytes. Once per `PackKey`.
    Prepack,
    /// Host→device staging copy of this inference's inputs.
    Upload,
    /// The `Compute` recording bracket. The name is historical: this span's host wall time is
    /// **dominated by the staging upload nested inside it** (measured 96-98% of the phase on
    /// Phi-3.5), while actual `vkBeginCommandBuffer..vkEndCommandBuffer` recording is 1-3% of wall
    /// (87-229 ms). R11: a measurement's name is not its definition — subtract `upload`+`readback`
    /// before attributing anything here to recording.
    Record,
    /// Sub-phase of `Record`: `vkCreateDescriptorPool` + `vkAllocateDescriptorSets` +
    /// `vkUpdateDescriptorSets` per dispatch. Nested inside a `Record` span; emitted
    /// per-kernel so the breakdown is visible in the Chrome Trace.
    DescAlloc,
    /// Sub-phase of `Record`: `PipelineCache::get_or_create` per dispatch — a hashmap
    /// lookup on a hit, `vkCreateComputePipelines` on the first call for that shader.
    PipelineLookup,
    /// Sub-phase of `Record`: CPU `memcpy` into staging + `vkCmdCopyBuffer` for all
    /// inputs.  Emitted once per `Compute` call. Isolates the upload cost from
    /// everything else inside the Record span so we know how much is transfer vs. API.
    CmdUpload,
    /// `vkQueueSubmit` and nothing else. **Measures no GPU work whatsoever.**
    Submit,
    /// `vkWaitForFences`. Queue latency + GPU execution, an *upper bound* on kernel time.
    FenceWait,
    /// Device→host copy of this inference's outputs.
    Readback,
}

impl Phase {
    /// Stable lowercase tag used in span names, counters and the summary.
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Compile => "compile",
            Phase::Prepack => "prepack",
            Phase::Upload => "upload",
            Phase::Record => "record",
            Phase::DescAlloc => "desc_alloc",
            Phase::PipelineLookup => "pipeline_lookup",
            Phase::CmdUpload => "cmd_upload",
            Phase::Submit => "submit",
            Phase::FenceWait => "fence_wait",
            Phase::Readback => "readback",
        }
    }

    /// One line explaining what this phase's *host* wall time does and does not contain.
    ///
    /// Emitted as a span arg and printed in the summary, because the single most common way to
    /// deceive yourself about a GPU is to read a host clock and call it kernel time.
    pub fn caveat(self) -> &'static str {
        match self {
            Phase::Compile => {
                "host: pipeline/descriptor creation; may hit the driver's shader compiler"
            }
            Phase::Prepack => "host: CPU repack + staging upload of weights; once per PackKey",
            Phase::Upload => {
                "host: staging copy; on a discrete GPU this is PCIe time and users pay it. \
                 NESTED INSIDE `record` — already counted there, do not add to the sibling total"
            }
            Phase::Record => {
                "host: the whole vkBeginCommandBuffer..vkEndCommandBuffer bracket. It CONTAINS \
                 `upload`/`cmd_upload` (the staging memcpy), `readback`, `desc_alloc` and \
                 `pipeline_lookup`, so it is an INCLUSIVE interval and its name describes its \
                 bracket, not its content (R11). The split is regime-dependent and must be read \
                 from the child rows of THIS run, never from a remembered ratio: with a cold \
                 weight cache `cmd_upload` dominates it (measured 1148 of 1185 ms on Phi-3.5's \
                 first Compute), and with a warm cache the children collapse to ~1-2 ms and the \
                 UNNAMED RESIDUAL — the vkCmd* calls themselves — is ~90% of it. The summary \
                 prints that residual as its own row, but CUMULATIVELY over all calls, so it \
                 mixes the two regimes; the per-call split is only in the trace spans"
            }
            Phase::DescAlloc => {
                "host/sub-record: vkCreateDescriptorPool + vkAllocateDescriptorSets + \
                 vkUpdateDescriptorSets per dispatch; emitted once per kernel per Compute call. \
                 NESTED INSIDE `record` — already counted there, do not add to the sibling total"
            }
            Phase::PipelineLookup => {
                "host/sub-record: PipelineCache::get_or_create — hashmap hit or \
                 vkCreateComputePipelines on first encounter; emitted once per kernel per Compute \
                 call. NESTED INSIDE `record` — already counted there, do not add to the sibling \
                 total"
            }
            Phase::CmdUpload => {
                "host/sub-record: CPU memcpy into staging + vkCmdCopyBuffer for all inputs; \
                 emitted once per Compute call; isolates transfer cost from API overhead. NESTED \
                 INSIDE `record` — already counted there, and it OVERLAPS `upload`, which brackets \
                 the same memcpy: never add the two nested rows together"
            }
            Phase::Submit => {
                "HOST-ONLY: vkQueueSubmit returns before any shader runs — this is NOT GPU time"
            }
            Phase::FenceWait => {
                "host: queue latency + GPU execution — an UPPER BOUND on kernel time, not kernel time"
            }
            Phase::Readback => {
                "host: device->host copy; counts toward end-to-end latency. NESTED INSIDE \
                 `record` — already counted there, do not add to the sibling total"
            }
        }
    }

    /// The phase whose wall time already contains this one, if any.
    ///
    /// # Why this is structural and not a sentence
    ///
    /// `caveat()` has always been emitted as a span arg, so the exclusion mechanism was wired,
    /// invoked, and in every artifact — and `Record`'s caveat said "amortised across replays"
    /// while 96% of its time was a staging memcpy that emits no span of its own. Prose in an
    /// artifact is only read by humans who already suspect something. An aggregator that sums
    /// `ph:"X"` spans by name cannot read prose, and summing `record` with `upload` double-counts
    /// while summing `record` alone attributes a child's cost to its parent's name. Both are one
    /// field away from being impossible.
    ///
    /// Emitted as the `nested_in` span arg. A phase with `nested_in == Some(p)` must never be
    /// added to a total that also contains `p`.
    pub fn nested_in(self) -> Option<Phase> {
        match self {
            // `vk::session` opens Phase::Record before vkBeginCommandBuffer and drops it after
            // vkEndCommandBuffer; the input staging loop and the output readback both run inside
            // that bracket. See session.rs (Record guard) — this is a fact about the call graph,
            // not a policy, and it must be re-checked if that guard moves.
            Phase::Upload | Phase::Readback => Some(Phase::Record),
            // Switch's per-dispatch sub-phases, added in `692e7d0`. They are documented in their
            // own caveats as "sub-record" and they are opened inside the Record guard.
            Phase::DescAlloc | Phase::PipelineLookup | Phase::CmdUpload => Some(Phase::Record),
            // EXHAUSTIVE ON PURPOSE — do not add a `_` arm. A catch-all here classifies every
            // future phase as a top-level sibling by default, which means a new sub-phase gets
            // silently added into SIBLING TOTAL and double-counts its parent. That is how three
            // phases arrived this session: they merged cleanly and would have been summed.
            // Make the compiler ask.
            Phase::Compile | Phase::Prepack | Phase::Record | Phase::Submit | Phase::FenceWait => {
                None
            }
        }
    }

    /// Phases that are top-level: their wall times may be summed.
    pub fn is_sibling(self) -> bool {
        self.nested_in().is_none()
    }

    /// Whether a host clock around this phase can see any GPU execution at all.
    ///
    /// `false` for [`Phase::Submit`]: the call is asynchronous, so its wall time is driver
    /// bookkeeping. Everything else either is host work by definition or (for
    /// [`Phase::FenceWait`]) contains GPU execution without being able to isolate it.
    /// Sub-phases (`DescAlloc`, `PipelineLookup`) are host-only and return `true` by default.
    pub fn observes_gpu_work(self) -> bool {
        !matches!(self, Phase::Submit)
    }

    /// Every phase, in reporting order.
    pub const ALL: [Phase; 10] = [
        Phase::Compile,
        Phase::Prepack,
        Phase::Upload,
        Phase::Record,
        Phase::DescAlloc,
        Phase::PipelineLookup,
        Phase::CmdUpload,
        Phase::Submit,
        Phase::FenceWait,
        Phase::Readback,
    ];
}

/// Which recording path one `Compute` call took — the Vulkan analogue of MLX's compile-cache
/// state, over `ENGINE.md` §6.1's record-once / replay-many model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RecordPath {
    /// First recording of this subgraph's command buffer.
    FirstRecord,
    /// Replayed the cached `VkCommandBuffer` — the steady-state path.
    Replay,
    /// Re-recorded because the input shape key changed. A benchmark that reports a median over
    /// runs where this fired is measuring the recording path, not the steady state.
    Rerecord,
}

impl RecordPath {
    pub fn as_str(self) -> &'static str {
        match self {
            RecordPath::FirstRecord => "FIRST_RECORD",
            RecordPath::Replay => "REPLAY",
            RecordPath::Rerecord => "RERECORD",
        }
    }
}

/// Direction of an explicit host↔device transfer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Transfer {
    /// Host → device (staging upload).
    Upload,
    /// Device → host (readback).
    Readback,
}

impl Transfer {
    pub fn as_str(self) -> &'static str {
        match self {
            Transfer::Upload => "upload",
            Transfer::Readback => "readback",
        }
    }

    fn phase(self) -> Phase {
        match self {
            Transfer::Upload => Phase::Upload,
            Transfer::Readback => Phase::Readback,
        }
    }
}

// -------------------------------------------------------------------------------------------
// Partition metrics — Mouse's contract (OP_COVERAGE.md §7.3), reported, never re-derived
// -------------------------------------------------------------------------------------------

/// The claiming view of one `GetCapability` call, in the vocabulary
/// `rust/src/ops/partition.rs` already defines.
///
/// `claimed_node_fraction` is **not** in this struct as a headline: `node_coverage` is carried
/// because it is a useful diagnostic, but `largest_island_flops` is the metric of record
/// (`OP_COVERAGE.md` §7.3, ratified in `decisions.md`). A change that raises coverage while
/// lowering `largest_island_flops` is a regression and this struct is shaped so that shows.
#[derive(Clone, Debug, Default)]
pub struct PartitionStats {
    pub total_nodes: u64,
    pub claimed_nodes: u64,
    pub island_count: u64,
    pub largest_island_nodes: u64,
    pub largest_island_flops: u64,
    /// `largest_island_flops ÷ total claimed FLOPs` — `CoverageReport::concentration`.
    pub concentration: f64,
    pub boundary_bytes_per_inference: u64,
    /// Transfer + sync time ÷ total. Above ~0.20, coverage work is being wasted and the fix is
    /// partitioning, not another op (`OP_COVERAGE.md` §7.3).
    pub boundary_time_fraction: f64,
    /// `(qualified_op, count, deny! reason)` for every declined node — Mouse's auto-generated
    /// backlog.
    pub declined: Vec<(String, u64, String)>,
}

// -------------------------------------------------------------------------------------------
// GPU timestamps — the shape of what the engine must hand back
// -------------------------------------------------------------------------------------------

/// Everything needed to put a device tick on the shared microsecond axis.
///
/// Produced by the engine (`rust/src/vk/**`), consumed here. This module never names Vulkan, so
/// this struct *is* the interface: the field docs are the requirement.
#[derive(Clone, Copy, Debug)]
pub struct GpuTimestampCalibration {
    /// `VkPhysicalDeviceLimits::timestampPeriod` — nanoseconds per tick. A `f32` because that is
    /// what Vulkan reports. It is **not** always 1.0: NVIDIA reports 1.0, several AMD and Adreno
    /// parts report values from ~20 to ~83, and treating ticks as nanoseconds on those devices
    /// under-reports GPU time by up to 80×.
    pub timestamp_period_ns: f32,
    /// `VkQueueFamilyProperties::timestampValidBits` for the queue the work ran on. Only the low
    /// `valid_bits` of a query result are meaningful; the rest are undefined and must be masked
    /// off before any arithmetic. `0` means the queue supports no timestamps at all and the
    /// engine must not report intervals from it.
    pub valid_bits: u32,
    /// A device tick and the host microsecond that tick corresponds to.
    ///
    /// Ideally from `VK_EXT_calibrated_timestamps` (`vkGetCalibratedTimestampsEXT` with
    /// `DEVICE` + a host time domain), which samples both clocks together and reports a maximum
    /// deviation. Where that extension is absent the engine anchors by bracketing: read the host
    /// clock, submit a command buffer whose only content is one timestamp write, wait, read the
    /// host clock again, and take the midpoint. That fallback's error is half the submit
    /// round-trip and it must be reported in [`Self::anchor_uncertainty_us`] so nobody reads a
    /// 3 µs kernel span placed by a 200 µs anchor as if the placement were meaningful.
    pub host_anchor_us: u64,
    /// The device tick that [`Self::host_anchor_us`] corresponds to (already masked).
    pub device_anchor_ticks: u64,
    /// Half-width of the anchor's uncertainty, in microseconds. `0` is a claim of perfect
    /// correlation and should only ever come from `VK_EXT_calibrated_timestamps` reporting a
    /// deviation of zero.
    pub anchor_uncertainty_us: u64,
}

impl GpuTimestampCalibration {
    /// Whether this calibration can be used at all.
    pub fn is_usable(&self) -> bool {
        self.valid_bits > 0 && self.timestamp_period_ns > 0.0
    }

    /// Mask a raw query result down to the bits the queue actually defines.
    ///
    /// `valid_bits >= 64` means every bit is valid (Vulkan permits 64) and the mask is a no-op —
    /// computed without shifting by 64, which is undefined in Rust and in C.
    pub fn mask(&self, raw_ticks: u64) -> u64 {
        mask_ticks(raw_ticks, self.valid_bits)
    }

    /// Tick delta → nanoseconds, handling counter wrap within the valid-bit width.
    ///
    /// A 32-valid-bit counter at a 1 ns period wraps every ~4.3 seconds, which is well inside a
    /// benchmark run, so `end < begin` is a wrap and not an error. It is only recoverable for a
    /// single wrap; an interval longer than the counter period is unrecoverable and reported as
    /// `None` rather than as a plausible-looking wrong number.
    pub fn ticks_to_ns(&self, begin_ticks: u64, end_ticks: u64) -> Option<f64> {
        if !self.is_usable() {
            return None;
        }
        let begin = self.mask(begin_ticks);
        let end = self.mask(end_ticks);
        let span = if end >= begin {
            end - begin
        } else {
            // One wrap of the valid-bit-wide counter.
            let modulus = tick_modulus(self.valid_bits)?;
            modulus.checked_sub(begin)?.checked_add(end)?
        };
        Some(span as f64 * f64::from(self.timestamp_period_ns))
    }

    /// Place a device tick on the shared absolute microsecond axis.
    ///
    /// Returns `None` when the calibration is unusable. The conversion is affine and signed: a
    /// tick *before* the anchor is normal (the anchor is usually taken at device init and the
    /// work happens later, but a bracketing anchor taken after the fact is also legal).
    pub fn ticks_to_axis_us(&self, ticks: u64) -> Option<u64> {
        if !self.is_usable() {
            return None;
        }
        let ticks = self.mask(ticks) as i128;
        let anchor = self.mask(self.device_anchor_ticks) as i128;
        let delta_ns = (ticks - anchor) as f64 * f64::from(self.timestamp_period_ns);
        let us = self.host_anchor_us as i128 + (delta_ns / 1000.0).round() as i128;
        u64::try_from(us.max(0)).ok()
    }
}

/// Mask a raw timestamp to the queue's valid-bit width. Free function so it can be unit-tested
/// without constructing a calibration.
fn mask_ticks(raw_ticks: u64, valid_bits: u32) -> u64 {
    if valid_bits >= 64 {
        raw_ticks
    } else if valid_bits == 0 {
        0
    } else {
        raw_ticks & ((1u64 << valid_bits) - 1)
    }
}

/// The wrap modulus of a `valid_bits`-wide counter, or `None` when it does not fit in a `u64`
/// (i.e. the counter is 64 bits wide and never wraps within `u64` arithmetic).
fn tick_modulus(valid_bits: u32) -> Option<u64> {
    if valid_bits == 0 || valid_bits >= 64 {
        None
    } else {
        Some(1u64 << valid_bits)
    }
}

/// One measured GPU interval: a pair of timestamp query results and what they bracket.
#[derive(Clone, Debug)]
pub struct GpuInterval {
    /// Span name, e.g. `"MatMulNBits"` or `"subgraph"`. Rendered as `vulkan.gpu.<label>`.
    pub label: String,
    /// Raw result of the timestamp written **before** the work (unmasked is fine).
    pub begin_ticks: u64,
    /// Raw result of the timestamp written **after** the work.
    pub end_ticks: u64,
    /// Index of the node this interval covers within the subgraph, when it is per-node.
    pub node_index: Option<u64>,
    /// FLOP estimate for the covered work, when the op module supplied one. With this and the
    /// duration, the trace carries achieved FLOP/s directly — which is what makes a roofline
    /// argument possible instead of a vibe.
    pub flops: Option<u64>,
    /// Bytes read+written by the covered work, when known. With duration, achieved bandwidth.
    pub bytes: Option<u64>,
}

/// One submission's worth of GPU timing, as handed back by the engine after the fence signals.
#[derive(Clone, Debug)]
pub struct GpuTimestampReport {
    pub calibration: GpuTimestampCalibration,
    /// Queue family index the work ran on — used to pick the device lane, so two queues do not
    /// draw on top of each other.
    pub queue_family: u32,
    pub intervals: Vec<GpuInterval>,
}

// -------------------------------------------------------------------------------------------
// Summary
// -------------------------------------------------------------------------------------------

/// The part of `record` that no child span accounts for — the `vkCmd*` calls themselves.
///
/// `record` is an inclusive bracket (R11): every named child is already inside it, so the only
/// honest statement about "command-buffer recording cost" is the residual. It is a subtraction and
/// not a measurement, and it is printed as such.
///
/// `xfer_us` must be the LARGER of the two transfer accountings (`upload`+`readback` from
/// `record_transfer`, and the `cmd_upload` sub-span), never their sum: the two can bracket the
/// same memcpy, and adding them would under-report the residual by inventing child time.
fn record_residual_us(record_us: u64, xfer_us: u64, desc_alloc_us: u64, pipeline_us: u64) -> u64 {
    record_us.saturating_sub(xfer_us + desc_alloc_us + pipeline_us)
}

/// Cumulative, human-readable digest emitted on teardown. Cheap to update; only touched when
/// tracing or the verbose flag is on.
#[derive(Default)]
struct Summary {
    getcap_calls: u64,
    total_nodes: u64,
    claimed_nodes: u64,
    island_count: u64,
    largest_island_nodes: u64,
    largest_island_flops: u64,
    concentration: f64,
    boundary_bytes_per_inference: u64,
    boundary_time_fraction: f64,
    /// `qualified_op -> (count, last reason)` for declined nodes — Mouse's backlog.
    declined: BTreeMap<String, (u64, String)>,

    /// `[first_record, replay, rerecord]`.
    record_paths: [u64; 3],
    /// Shape keys already seen, per subgraph tag, so a REPLAY on a new key is called what it is.
    seen_shape_keys: HashMap<String, HashSet<String>>,

    upload_bytes: u64,
    upload_count: u64,
    readback_bytes: u64,
    readback_count: u64,

    /// `phase -> (total_us, count)`.
    phase_us: BTreeMap<Phase, (u64, u64)>,

    /// GPU intervals actually measured, `label -> (total_ns, count)`.
    gpu_ns: BTreeMap<String, (u64, u64)>,
    /// Set when at least one usable calibration was seen; drives the "GPU time is real" line.
    gpu_measured: bool,
    /// Largest anchor uncertainty seen, printed with the GPU numbers.
    gpu_anchor_uncertainty_us: u64,
}

// -------------------------------------------------------------------------------------------
// Tracer
// -------------------------------------------------------------------------------------------

/// Process-wide tracer singleton. All sessions share one timeline and one output file, stamped
/// with the real pid so events merge into a host trace under the same process, on their own
/// lanes.
static TRACER: OnceLock<VulkanTracer> = OnceLock::new();

/// The shared tracer. First access reads the environment and wires everything up.
pub fn tracer() -> &'static VulkanTracer {
    TRACER.get_or_init(VulkanTracer::new)
}

thread_local! {
    static THREAD_NAMED: Cell<bool> = const { Cell::new(false) };
}

/// One sampled counter point (rendered as a Chrome `"C"` event at export).
struct CounterSample {
    track: String,
    key: String,
    value: f64,
    ts: u64,
}

/// The env-gated tracer. Cheap to leave wired in when disabled.
pub struct VulkanTracer {
    ctx: TraceContext,
    mem: Option<Arc<MemoryCollector>>,
    path: Option<PathBuf>,
    counters: Mutex<Vec<CounterSample>>,
    op_times: Mutex<HashMap<String, (u64, u64)>>,
    summary: Mutex<Summary>,
    verbose: bool,
    /// Whether the engine was asked for GPU timestamp queries ([`ENV_GPU_TIMESTAMPS`]).
    gpu_timestamps_requested: bool,
}

impl VulkanTracer {
    fn new() -> Self {
        let path = std::env::var(ENV_TRACE).ok().filter(|s| !s.is_empty());
        let trace_on = path.is_some();

        let (ctx, mem) = if trace_on {
            let (ctx, mem) = TraceContext::in_memory();
            ctx.set_process_name("onnxruntime-ep-vulkan");
            (ctx, Some(mem))
        } else {
            (TraceContext::noop(), None)
        };

        let verbose = trace_on
            || std::env::var(ENV_VERBOSE)
                .map(|v| v == "1")
                .unwrap_or(false);

        VulkanTracer {
            ctx,
            mem,
            path: path.map(PathBuf::from),
            counters: Mutex::new(Vec::new()),
            op_times: Mutex::new(HashMap::new()),
            summary: Mutex::new(Summary::default()),
            verbose,
            gpu_timestamps_requested: gpu_timestamps_requested(),
        }
    }

    /// Whether JSON tracing is enabled — the hot-path gate.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.ctx.is_enabled()
    }

    /// Whether *any* observability is active: JSON tracing or the verbose summary. When neither
    /// is on this is one bool load and every recorder early-returns.
    #[inline]
    pub fn active(&self) -> bool {
        self.is_enabled() || self.verbose
    }

    /// Whether the engine should write `VkQueryPool` timestamps this run ([`ENV_GPU_TIMESTAMPS`]).
    ///
    /// The engine calls this instead of reading the environment itself, so there is one answer.
    #[inline]
    pub fn wants_gpu_timestamps(&self) -> bool {
        self.gpu_timestamps_requested
    }

    fn summary(&self) -> std::sync::MutexGuard<'_, Summary> {
        match self.summary.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        }
    }

    fn push_counter(&self, track: &str, key: &str, value: f64) {
        let ts = self.ctx.clock().now_micros();
        let mut c = match self.counters.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        c.push(CounterSample {
            track: track.to_string(),
            key: key.to_string(),
            value,
            ts,
        });
    }

    /// Name the current OS thread's lane once (idempotent per thread).
    pub fn note_thread(&self, name: &str) {
        if !self.is_enabled() {
            return;
        }
        THREAD_NAMED.with(|n| {
            if !n.get() {
                self.ctx.set_thread_name(name);
                n.set(true);
            }
        });
    }

    // --- Claiming view -----------------------------------------------------------------

    /// Record one `GetCapability` partition, in Mouse's metric vocabulary.
    ///
    /// Emits a `vulkan.getcapability` instant plus counter tracks for the metrics that must
    /// appear in every benchmark run, and folds them into the summary. The declined histogram is
    /// carried verbatim: it is the auto-generated op backlog, and summarising it away would
    /// destroy the one output Mouse asked for by name.
    pub fn record_partition(&self, stats: &PartitionStats) {
        if !self.active() {
            return;
        }
        {
            let mut s = self.summary();
            s.getcap_calls += 1;
            s.total_nodes += stats.total_nodes;
            s.claimed_nodes += stats.claimed_nodes;
            s.island_count += stats.island_count;
            s.largest_island_nodes = s.largest_island_nodes.max(stats.largest_island_nodes);
            s.largest_island_flops = s.largest_island_flops.max(stats.largest_island_flops);
            s.concentration = stats.concentration;
            s.boundary_bytes_per_inference = stats.boundary_bytes_per_inference;
            s.boundary_time_fraction = stats.boundary_time_fraction;
            for (op, n, reason) in &stats.declined {
                let e = s.declined.entry(op.clone()).or_insert((0, String::new()));
                e.0 += *n;
                if !reason.is_empty() {
                    e.1 = reason.clone();
                }
            }
        }
        if !self.is_enabled() {
            return;
        }
        let mut args = Args::new()
            .with("total_nodes", stats.total_nodes)
            .with("claimed_nodes", stats.claimed_nodes)
            .with("island_count", stats.island_count)
            .with("largest_island_nodes", stats.largest_island_nodes)
            .with("largest_island_flops", stats.largest_island_flops)
            .with("concentration", stats.concentration)
            .with(
                "boundary_bytes_per_inference",
                stats.boundary_bytes_per_inference,
            )
            .with("boundary_time_fraction", stats.boundary_time_fraction);
        for (op, n, reason) in &stats.declined {
            args = args.with(format!("declined_{op}"), format!("x{n}: {reason}"));
        }
        self.ctx
            .instant("vulkan.getcapability", "ep.claim", Some(args));

        // The four numbers that must appear in every benchmark run (OP_COVERAGE.md §7.3).
        self.push_counter("vulkan.island_count", "islands", stats.island_count as f64);
        self.push_counter(
            "vulkan.largest_island_flops",
            "flops",
            stats.largest_island_flops as f64,
        );
        self.push_counter("vulkan.concentration", "ratio", stats.concentration);
        self.push_counter(
            "vulkan.boundary_bytes",
            "bytes",
            stats.boundary_bytes_per_inference as f64,
        );
    }

    // --- Execution view ----------------------------------------------------------------

    /// Span around one fused subgraph's dispatch region, opened inside `dispatch_ort`.
    ///
    /// Host wall time bounding this subgraph's upload, record-or-replay, submit, fence wait and
    /// readback. Entry-side work in `compute`/`compute_impl` that runs *before* `dispatch_ort`
    /// is called, and anything the callback does *after* `dispatch_ort` returns (including a
    /// benchmark harness's own instrumentation), lies outside this span — the wider bracket that
    /// contains both sides is [`compute_region`](Self::compute_region). It is deliberately not
    /// called "GPU time" anywhere.
    pub fn subgraph_region(&self, node_count: usize) -> SpanGuard {
        if !self.is_enabled() {
            return self.ctx.span(SPAN_SUBGRAPH, "ep");
        }
        self.ctx.span(SPAN_SUBGRAPH, "ep").with_args(
            Args::new()
                .with("nodes", node_count as u64)
                .with(ARG_DEVICE, DEVICE_HOST),
        )
    }

    /// Span around the instrumented success path of the `Compute` callback.
    ///
    /// It is **not** ORT's literal `Compute` entry: `compute` does the null check, resolves
    /// `this_info`, and calls `compute_impl` through `guard_ffi_status`; this span opens inside
    /// `compute_impl` and closes before `disclose_broken_commitment` runs. So the FFI guard, the
    /// null check and the post-call disclosure are outside it, and a call that early-outs before
    /// `compute_impl` emits no span at all. Read it as *the widest bracket the EP instruments on
    /// the success path*, not as the callback's true wall time.
    ///
    /// [`subgraph_region`](Self::subgraph_region) opens inside `dispatch_ort`, deeper still, so it
    /// cannot see anything the callback does on either side of that call. Measured on Phi-3.5
    /// `prefill_1`, the two spans differed by a large term, nearly all of it landing *after*
    /// `vulkan.subgraph` closed rather than before it.
    ///
    /// That term was **not the EP**. Re-measured on the same workload with the benchmark
    /// harness's counters-file dump moved out of the timed region, it collapses to a small
    /// fraction of a millisecond. No magnitude is quoted here: this branch carries the
    /// instrument and none of its output, so there is no committed artifact to cite, and
    /// `bench/test_cuda_profile.py`'s citation pin reads the figure out of the committed
    /// profile wherever one exists. The dump ran on every
    /// timed inference, from `counters::record_dispatches` after `dispatch_ort` returns — which
    /// is the side the region was on. The span is kept because that is the finding: without an
    /// outer bracket the harness's own cost was inside the EP's numbers and nothing could see it.
    ///
    /// This span is **structural, not a [`Phase`]**: it is `cat == "ep"` like `vulkan.subgraph`,
    /// it is never summed into any sibling total, and it adds no level to the phase tree. It
    /// exists so a reduction has an anchor wider than `vulkan.subgraph` — the instrumented
    /// success-path region inside `compute_impl`, not ORT's literal `Compute` entry-to-return
    /// wall — and can therefore state how much of *that region* no phase covers, instead of
    /// charging that time to whatever span it can see.
    ///
    /// It does not by itself explain the region it exposes. What is measured is the size and the
    /// side of the region; naming its cause needs a separate instrument, and
    /// `bench/cuda_profile.py` reports it as `outside_subgraph_us` — measured and unattributed.
    pub fn compute_region(&self) -> SpanGuard {
        self.ctx.span(SPAN_COMPUTE_CALL, "ep")
    }

    /// Start a timing phase: a span plus a summary fold on drop. `None` (zero cost) when nothing
    /// is listening.
    #[inline]
    pub fn phase(&self, phase: Phase) -> Option<PhaseGuard> {
        if !self.active() {
            return None;
        }
        let span = self
            .ctx
            .span(format!("vulkan.{}", phase.as_str()), "ep.phase")
            .with_args(
                Args::new()
                    .with(ARG_DEVICE, DEVICE_HOST)
                    .with("caveat", phase.caveat())
                    // Machine-readable parentage. An aggregator that sums `ph:"X"` spans by name
                    // must skip any span carrying `nested_in`, or it attributes a child's cost to
                    // its parent and reports a memcpy as command-buffer recording.
                    .with(
                        "nested_in",
                        phase.nested_in().map(Phase::as_str).unwrap_or("none"),
                    ),
            );
        Some(PhaseGuard {
            phase,
            start: Instant::now(),
            _span: span,
        })
    }

    /// Fold a phase duration into the summary without opening a span. For call sites that
    /// already have their own timing.
    pub fn record_phase(&self, phase: Phase, dur: Duration) {
        if !self.active() {
            return;
        }
        let mut s = self.summary();
        let e = s.phase_us.entry(phase).or_insert((0, 0));
        e.0 += dur.as_micros() as u64;
        e.1 += 1;
    }

    /// Record which recording path a `Compute` call took, resolving `Replay` against the shape
    /// keys already seen for this subgraph.
    ///
    /// A caller reports `Replay` when it reused a cached command buffer. If the shape key is one
    /// this subgraph has never presented before, the reuse cannot have been legitimate for that
    /// key and the call is reclassified as [`RecordPath::Rerecord`] — the signal that a
    /// benchmark's "steady state" is not steady. Pass an empty `shape_key` for subgraphs with no
    /// dynamic shapes.
    pub fn record_path(
        &self,
        subgraph_tag: &str,
        path: RecordPath,
        shape_key: &str,
        node_count: usize,
    ) -> RecordPath {
        if !self.active() {
            return path;
        }
        let resolved = {
            let mut s = self.summary();
            let resolved = match path {
                RecordPath::Replay if !shape_key.is_empty() => {
                    let seen = s
                        .seen_shape_keys
                        .get(subgraph_tag)
                        .is_some_and(|keys| keys.contains(shape_key));
                    if seen {
                        RecordPath::Replay
                    } else {
                        RecordPath::Rerecord
                    }
                }
                other => other,
            };
            if !shape_key.is_empty() {
                s.seen_shape_keys
                    .entry(subgraph_tag.to_string())
                    .or_default()
                    .insert(shape_key.to_string());
            }
            let idx = match resolved {
                RecordPath::FirstRecord => 0,
                RecordPath::Replay => 1,
                RecordPath::Rerecord => 2,
            };
            s.record_paths[idx] += 1;
            resolved
        };
        if self.is_enabled() {
            let mut args = Args::new()
                .with("path", resolved.as_str())
                .with("nodes", node_count as u64)
                .with("subgraph", subgraph_tag.to_string());
            if !shape_key.is_empty() {
                args = args.with("shape_key", shape_key.to_string());
            }
            self.ctx.instant(
                format!("{}[{}]", INSTANT_RECORD_PATH, resolved.as_str()),
                "ep.path",
                Some(args),
            );
        }
        resolved
    }

    /// Record one explicit host↔device transfer: direction, bytes and host wall time.
    ///
    /// These are counted separately from compute because on a discrete GPU they are frequently
    /// the dominant cost, and because `boundary_bytes_per_inference` (the number MVS minimises)
    /// is only checkable against reality if we measure the bytes we actually moved.
    pub fn record_transfer(&self, direction: Transfer, bytes: u64, dur: Duration) {
        // UNCONDITIONAL, AND BEFORE THE GUARD ON PURPOSE.
        //
        // Everything below early-returns unless tracing or verbose is on. That is right for the
        // trace document and wrong for the bytes: staging upload is ~71% of wall and the run that
        // gets quoted is the one nobody set a flag on. `alloc_device_upload_bytes` read 0 on a run
        // whose cmd_upload phase was 15.2 s, because it counts a different copy through a
        // different device. These two atomics are the byte falsifier for persistent weight
        // residency, and they must exist in the default configuration.
        let us = dur.as_micros() as u64;
        match direction {
            Transfer::Upload => crate::counters::staging::on_upload(bytes, us),
            Transfer::Readback => crate::counters::staging::on_readback(bytes, us),
        }
        if !self.active() {
            return;
        }
        {
            let mut s = self.summary();
            match direction {
                Transfer::Upload => {
                    s.upload_count += 1;
                    s.upload_bytes += bytes;
                }
                Transfer::Readback => {
                    s.readback_count += 1;
                    s.readback_bytes += bytes;
                }
            }
            let e = s.phase_us.entry(direction.phase()).or_insert((0, 0));
            e.0 += dur.as_micros() as u64;
            e.1 += 1;
        }
        if self.is_enabled() {
            self.push_counter("vulkan.transfer_bytes", direction.as_str(), bytes as f64);
            let us = dur.as_micros().max(1) as f64;
            // Effective bandwidth for this transfer, in GiB/s. Reported per transfer rather than
            // aggregated because a single large staging copy and a thousand tiny ones have very
            // different bandwidth and identical byte totals.
            let gib_s = (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (us / 1_000_000.0);
            self.push_counter("vulkan.transfer_gib_s", direction.as_str(), gib_s);
        }
    }

    /// Lightweight build-time span for one node.
    pub fn op_span(&self, node: &NodeDesc) -> SpanGuard {
        if !self.is_enabled() {
            return self.ctx.span(node.op_type.clone(), "op");
        }
        self.ctx
            .span(node.op_type.clone(), "op")
            .with_args(standard_op_args(node))
    }

    /// Start a wall-clock timer for one node's handler, or `None` when tracing is off (so a
    /// disabled run does not even read the clock).
    #[inline]
    pub fn op_timer_start(&self) -> Option<Instant> {
        if self.is_enabled() {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Emit a per-node span with shape/variant context and fold it into the slowest-ops table.
    ///
    /// **This is host translation/record time, not kernel time**, and the span says so in its
    /// `caveat` arg. Per-node GPU time comes only from [`Self::record_gpu_intervals`].
    pub fn record_op_meta(&self, node: &NodeDesc, start: Instant, dur: Duration, meta: OpMeta<'_>) {
        if !self.is_enabled() {
            return;
        }
        let mut args = standard_op_args(node)
            .with("input_shapes", meta.in_shapes.to_string())
            .with("output_shapes", meta.out_shapes.to_string())
            .with("dtype", meta.dtype.to_string())
            .with(
                "caveat",
                "host-side record/translate time — NOT GPU kernel time",
            );
        if let Some(v) = meta.variant {
            args = args.with(ARG_VARIANT, v.to_string());
        }
        if let Some(b) = meta.bytes {
            args = args.with(ARG_BYTES, b);
        }
        if let Some(f) = meta.flops {
            args = args.with(ARG_FLOPS, f);
        }
        self.ctx
            .complete(node.op_type.clone(), "op", start, dur, Some(args));
        self.record_op_time(&node.op_type, dur.as_micros() as u64);
    }

    fn record_op_time(&self, op_type: &str, us: u64) {
        let mut m = match self.op_times.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(e) = m.get_mut(op_type) {
            e.0 += us;
            e.1 += 1;
        } else {
            m.insert(op_type.to_string(), (us, 1));
        }
    }

    // --- GPU view ----------------------------------------------------------------------

    /// Convert one submission's timestamp query results into device-lane spans.
    ///
    /// Returns the number of intervals that produced a span. Intervals are dropped — never
    /// approximated — when the calibration is unusable (`timestampValidBits == 0`) or when a
    /// tick delta cannot be resolved (an interval longer than the counter's wrap period).
    ///
    /// The spans land on a synthetic per-queue lane, not on the submitting thread's lane, so GPU
    /// execution is never drawn inside the CPU work that submitted it.
    pub fn record_gpu_intervals(&self, report: &GpuTimestampReport) -> usize {
        if !self.active() {
            return 0;
        }
        let cal = &report.calibration;
        if !cal.is_usable() {
            log::debug!(
                "GPU timestamps discarded: timestampValidBits={} timestampPeriod={} — the queue \
                 reports no usable timestamp counter, so no GPU time is claimed for this \
                 submission",
                cal.valid_bits,
                cal.timestamp_period_ns
            );
            return 0;
        }

        let mut emitted = 0usize;
        for iv in &report.intervals {
            let Some(ns) = cal.ticks_to_ns(iv.begin_ticks, iv.end_ticks) else {
                continue;
            };
            {
                let mut s = self.summary();
                s.gpu_measured = true;
                s.gpu_anchor_uncertainty_us =
                    s.gpu_anchor_uncertainty_us.max(cal.anchor_uncertainty_us);
                let e = s.gpu_ns.entry(iv.label.clone()).or_insert((0, 0));
                e.0 += ns as u64;
                e.1 += 1;
            }
            if !self.is_enabled() {
                emitted += 1;
                continue;
            }
            let Some(ts_us) = cal.ticks_to_axis_us(iv.begin_ticks) else {
                continue;
            };
            let mut args = Args::new()
                .with(ARG_DEVICE, DEVICE_GPU)
                .with("gpu_ns", ns)
                .with("timestamp_period_ns", f64::from(cal.timestamp_period_ns))
                .with("timestamp_valid_bits", u64::from(cal.valid_bits))
                .with("anchor_uncertainty_us", cal.anchor_uncertainty_us)
                .with("queue_family", u64::from(report.queue_family));
            if let Some(n) = iv.node_index {
                args = args.with("node_index", n);
            }
            if let Some(f) = iv.flops {
                args = args.with(ARG_FLOPS, f);
                if ns > 0.0 {
                    // Achieved GFLOP/s for this kernel. With the device's peak this is a roofline
                    // position rather than an anecdote.
                    args = args.with("achieved_gflops", f as f64 / ns);
                }
            }
            if let Some(b) = iv.bytes {
                args = args.with(ARG_BYTES, b);
                if ns > 0.0 {
                    args = args.with("achieved_gib_s", (b as f64 / 1.073_741_824e9) / (ns / 1e9));
                }
            }
            self.emit_gpu_span(
                &format!("vulkan.gpu.{}", iv.label),
                ts_us,
                ns as u64 / 1000,
                report.queue_family,
                args,
            );
            emitted += 1;
        }
        emitted
    }

    /// Emit one complete event on a device lane at an explicit absolute timestamp.
    ///
    /// `TraceContext::complete` takes an `Instant`, which a device tick is not and cannot be
    /// converted into, so the event is constructed directly. This is the only reason this module
    /// touches `TraceEvent` by hand.
    fn emit_gpu_span(&self, name: &str, ts_us: u64, dur_us: u64, queue_family: u32, args: Args) {
        use onnx_runtime_tracer::{TraceEvent, TracePhase};
        let event = TraceEvent {
            name: name.to_string(),
            cat: "gpu".to_string(),
            ph: TracePhase::Complete,
            ts: ts_us,
            dur: Some(dur_us),
            pid: self.ctx.pid(),
            tid: gpu_lane(queue_family),
            scope: None,
            args: Some(args.into_value()),
        };
        self.ctx.emit(&event);
    }

    // --- Output ------------------------------------------------------------------------

    /// Print the end-of-run session summary.
    ///
    /// Goes to the log (stderr and, when attached, ORT's logger) whenever tracing or verbose is
    /// on, and is additionally embedded in the trace as a `vulkan.session_summary` instant.
    pub fn log_summary(&self) {
        if !self.active() {
            return;
        }
        let s = self.summary();
        if s.getcap_calls == 0 && s.record_paths.iter().all(|&n| n == 0) {
            return;
        }

        let mut out = String::from("===== Vulkan EP session summary =====\n");
        let coverage = if s.total_nodes > 0 {
            (s.claimed_nodes as f64 / s.total_nodes as f64) * 100.0
        } else {
            0.0
        };
        out.push_str(&format!(
            "  claim:    {}/{} nodes ({coverage:.1}% — diagnostic only) in {} island(s); \
             largest island {} nodes / {} FLOPs; concentration {:.3}\n",
            s.claimed_nodes,
            s.total_nodes,
            s.island_count,
            s.largest_island_nodes,
            s.largest_island_flops,
            s.concentration
        ));
        out.push_str(&format!(
            "  boundary: {} bytes/inference, boundary_time_fraction {:.3}{}\n",
            s.boundary_bytes_per_inference,
            s.boundary_time_fraction,
            if s.boundary_time_fraction > 0.20 {
                "  ⚠ >0.20: the fix is partitioning, not another op"
            } else {
                ""
            }
        ));
        if !s.declined.is_empty() {
            let mut items: Vec<(&String, &(u64, String))> = s.declined.iter().collect();
            items.sort_by_key(|a| std::cmp::Reverse(a.1.0));
            out.push_str("            declined (→ CPU):\n");
            for (op, (n, reason)) in items.iter().take(8) {
                out.push_str(&format!("              - {op} x{n}: {reason}\n"));
            }
        }
        // `Tracer::record_path` has no production caller (audited 2026-07-30: zero call sites
        // outside this file; the classifier is unit-tested, which is why review never caught it).
        // Printing "first-record=0 replay=0 rerecord=0" reads as "this run recorded nothing",
        // which is a measurement. It is not one. It is the absence of a measurement, and the
        // difference matters here more than most: `Phase::Record`'s caveat asserted that recording
        // is "amortised across replays", and this is the only instrument that could falsify that.
        if s.record_paths.iter().all(|&n| n == 0) {
            out.push_str(
                "  compute:  record-path breakdown NOT WIRED — Tracer::record_path() has no \
                 production caller, so first-record/replay/rerecord are unmeasured, NOT zero. \
                 Nothing in this build can tell you whether command buffers are re-recorded per \
                 inference or replayed.\n",
            );
        } else {
            out.push_str(&format!(
                "  compute:  first-record={} replay={} rerecord={}\n",
                s.record_paths[0], s.record_paths[1], s.record_paths[2]
            ));
        }
        out.push_str(&format!(
            "  transfer: upload {} calls / {:.2} MiB; readback {} calls / {:.2} MiB\n",
            s.upload_count,
            s.upload_bytes as f64 / (1024.0 * 1024.0),
            s.readback_count,
            s.readback_bytes as f64 / (1024.0 * 1024.0),
        ));

        if !s.phase_us.is_empty() {
            // NESTING MATTERS AND THIS TABLE USED TO HIDE IT.
            //
            // `compile`, `record`, `submit` and `fence_wait` are SIBLINGS — they partition the
            // host thread's wall time. `upload` and `readback` are CHILDREN OF `record`: the
            // staging memcpy is timed inside the `Phase::Record` guard (see `vk/session.rs`) and
            // fed here through `record_transfer`. Summing this column therefore double-counts
            // transfer, and printing it as a flat list invited exactly that.
            //
            // Children are printed indented under their parent with an explicit marker, and the
            // sibling total is computed and printed so nobody has to add the column by hand.
            // Both the child set and the sibling total are DERIVED from `Phase::nested_in()`
            // rather than restated here. They used to be two hardcoded lists, which is the same
            // duplicate-truth defect one level down: changing the bracketing in `vk::session`
            // would have had to be remembered in three places, and the one that gets forgotten is
            // the one that prints.
            let get = |p: Phase| s.phase_us.get(&p).copied().unwrap_or((0, 0));
            // TRANSFER IS NAMED, NOT INFERRED FROM "IS A CHILD". `desc_alloc` and
            // `pipeline_lookup` are also children of `record` and are not transfer; summing every
            // child under the label "xfer" would be the same misnaming one level down.
            let child_us: u64 = get(Phase::Upload).0 + get(Phase::Readback).0;
            let sibling_us: u64 = Phase::ALL
                .iter()
                .filter(|p| p.is_sibling())
                .map(|p| get(*p).0)
                .sum();
            // `upload` (record_transfer totals) and `cmd_upload` (Switch's per-Compute sub-span)
            // can bracket the same memcpy, so the nested rows are NOT disjoint and are never
            // added together here.
            let overlap = get(Phase::CmdUpload).1 > 0 && get(Phase::Upload).1 > 0;

            out.push_str(
                "  host time (wall clock on the CPU thread). `upload`/`readback` are NESTED \
                 INSIDE `record` — do NOT add this column:\n",
            );
            for phase in Phase::ALL {
                let Some((us, calls)) = s.phase_us.get(&phase) else {
                    continue;
                };
                out.push_str(&format!(
                    "              {}{:<11} {:>10} us (x{}) — {}\n",
                    if phase.is_sibling() {
                        "    "
                    } else {
                        "  └─ "
                    },
                    phase.as_str(),
                    us,
                    calls,
                    phase.caveat()
                ));
            }
            out.push_str(&format!(
                "              {:<15} {:>10} us  (siblings only: {}; excludes the nested rows)\n",
                "SIBLING TOTAL",
                sibling_us,
                Phase::ALL
                    .iter()
                    .filter(|p| p.is_sibling())
                    .map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join("+")
            ));
            out.push_str(&format!(
                "              {:<15} {:>10} us  ({:.1}% of sibling total) — upload+readback only, \
                 already counted inside `record`\n",
                "of which xfer",
                child_us,
                100.0 * child_us as f64 / (sibling_us.max(1)) as f64,
            ));
            if overlap {
                out.push_str(
                    "              NOTE: the nested rows are NOT disjoint — `upload` and \
                     `cmd_upload` can bracket the same memcpy. Never add the nested rows to each \
                     other; each is only comparable to its parent `record`.\n",
                );
            }
            // THE UNNAMED PART OF `record` GETS A ROW (R11).
            //
            // Every child of `record` is named and printed above, which makes the decomposition
            // *look* closed. It is not: on a warm weight cache the named children fall to ~1-2 ms
            // of an ~18-23 ms `record`, so ~90% of the phase has no span of its own. That
            // residual is the vkCmd* recording itself — the thing the phase is named after — and
            // it was invisible precisely because nothing printed it. A residual that is not
            // printed is a residual nobody computes.
            //
            // `upload`/`readback` (record_transfer totals) and `cmd_upload` (the per-Compute
            // sub-span) can bracket the same memcpy, so the child set summed here takes the
            // larger of the two transfer accountings rather than their sum.
            let record_us = get(Phase::Record).0;
            if record_us > 0 {
                let residual = record_residual_us(
                    record_us,
                    child_us.max(get(Phase::CmdUpload).0),
                    get(Phase::DescAlloc).0,
                    get(Phase::PipelineLookup).0,
                );
                out.push_str(&format!(
                    "              {:<15} {:>10} us  ({:.1}% of `record`) — `record` minus every \
                     named child: the vkCmd* calls themselves, the ONLY part of `record` that is \
                     command-buffer recording. CUMULATIVE over every Compute call, so a session \
                     with one cold call and many warm ones reports a MIXTURE of the two regimes \
                     and this share belongs to no single call; for the per-call split read the \
                     `record` spans and their children in the trace JSON\n",
                    "record RESIDUAL",
                    residual,
                    100.0 * residual as f64 / record_us as f64,
                ));
            }
        }

        if s.gpu_measured {
            out.push_str(&format!(
                "  GPU time (device timestamp queries, anchor ±{} us):\n",
                s.gpu_anchor_uncertainty_us
            ));
            let mut ranked: Vec<(&String, &(u64, u64))> = s.gpu_ns.iter().collect();
            ranked.sort_by_key(|a| std::cmp::Reverse(a.1.0));
            for (label, (ns, calls)) in ranked.iter().take(10) {
                out.push_str(&format!(
                    "              {:<24} {:>12} ns (x{})\n",
                    label, ns, calls
                ));
            }
        } else {
            out.push_str(
                "  GPU time: NOT MEASURED. No timestamp queries were reported this run, so \
                 nothing above is kernel time. Set ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1 on an \
                 engine build that writes VkQueryPool timestamps.\n",
            );
        }
        out.push_str("=====================================");
        log::info!("{out}");

        if self.is_enabled() {
            let args = Args::new()
                .with("claimed_nodes", s.claimed_nodes)
                .with("total_nodes", s.total_nodes)
                .with("island_count", s.island_count)
                .with("largest_island_nodes", s.largest_island_nodes)
                .with("largest_island_flops", s.largest_island_flops)
                .with("concentration", s.concentration)
                .with(
                    "boundary_bytes_per_inference",
                    s.boundary_bytes_per_inference,
                )
                .with("boundary_time_fraction", s.boundary_time_fraction)
                .with("first_record", s.record_paths[0])
                .with("replay", s.record_paths[1])
                .with("rerecord", s.record_paths[2])
                .with("upload_bytes", s.upload_bytes)
                .with("readback_bytes", s.readback_bytes)
                .with("gpu_time_measured", s.gpu_measured);
            self.ctx
                .instant("vulkan.session_summary", "summary", Some(args));
        }
    }

    /// Log the top-10 op types by **host** record/translate time.
    ///
    /// Named for what it is. This is not a kernel profile and the log line says so; a kernel
    /// profile requires [`Self::record_gpu_intervals`] to have received real timestamps.
    pub fn log_slowest_ops(&self) {
        if !self.active() {
            return;
        }
        let snapshot: Vec<(String, u64, u64)> = {
            let m = match self.op_times.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            m.iter().map(|(k, v)| (k.clone(), v.0, v.1)).collect()
        };
        if snapshot.is_empty() {
            // Returning silently here is how this table came to be missing from every artifact
            // without anyone noticing. `record_op_meta` — the only producer — has no production
            // caller (audited 2026-07-30), so this map is empty on every real run, and an empty
            // map used to mean "print nothing". Absence of a table is not absence of slow ops.
            log::info!(
                "VulkanExecutionProvider: per-op host time NOT WIRED — Tracer::record_op_meta() \
                 has no production caller, so there is no per-op breakdown for this run. This is \
                 a missing instrument, not a run without slow ops."
            );
            return;
        }
        let total: u64 = snapshot.iter().map(|(_, us, _)| *us).sum();
        let mut ranked = snapshot;
        ranked.sort_by_key(|a| std::cmp::Reverse(a.1));
        ranked.truncate(10);
        let denom = total.max(1) as f64;

        let mut lines = format!(
            "slowest ops by HOST record/translate time (not GPU time), total {total} us:\n"
        );
        for (i, (op, us, calls)) in ranked.iter().enumerate() {
            lines.push_str(&format!(
                "  {:>2}. {:<24} {:>10} us  {:>5.1}%  ({} call(s))\n",
                i + 1,
                op,
                us,
                (*us as f64 / denom) * 100.0,
                calls
            ));
        }
        log::info!("{lines}");
    }

    /// Host-side staging traffic recorded through [`Self::record_transfer`], as
    /// `(upload_count, upload_bytes, readback_count, readback_bytes, upload_us, readback_us)`.
    ///
    /// **Tracer-scoped and therefore NOT the accounting to quote.** These fields are populated
    /// only when the tracer is active, which is why the allocator's staging verdict no longer
    /// reads them: it reads `counters::staging`, which is unconditional. Kept for the trace
    /// summary and for tests; a third upload accounting is the last thing this EP needs.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn transfer_totals(&self) -> (u64, u64, u64, u64, u64, u64) {
        let s = self.summary();
        let up = s.phase_us.get(&Phase::Upload).copied().unwrap_or((0, 0));
        let rb = s.phase_us.get(&Phase::Readback).copied().unwrap_or((0, 0));
        (
            s.upload_count,
            s.upload_bytes,
            s.readback_count,
            s.readback_bytes,
            up.0,
            rb.0,
        )
    }

    /// Write the accumulated trace as Chrome Trace JSON to the configured path.
    ///
    /// Called on EP teardown. The collector accumulates across every session in the process, so
    /// each call rewrites the full cumulative trace (last writer wins); the final teardown leaves
    /// the complete file on disk.
    pub fn export(&self) {
        if !self.is_enabled() {
            return;
        }
        let (Some(mem), Some(path)) = (&self.mem, &self.path) else {
            return;
        };
        let mut out = mem.to_chrome_json();
        let counters = match self.counters.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let tail = counter_tail(&counters, self.ctx.pid());
        splice_counters(&mut out, &tail);

        match std::fs::write(path, &out) {
            Ok(()) => log::info!(
                "wrote Vulkan EP trace ({} span event(s), {} counter sample(s)) to {}",
                mem.len(),
                counters.len(),
                path.display()
            ),
            Err(e) => log::error!("trace export to {} failed: {e}", path.display()),
        }
    }
}

/// Whether [`ENV_GPU_TIMESTAMPS`] asks for timestamp queries.
fn gpu_timestamps_requested() -> bool {
    std::env::var(ENV_GPU_TIMESTAMPS)
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Render counter samples as Chrome `"C"` events, each prefixed with a comma.
///
/// The tracer's `TracePhase` has no counter variant, so counter tracks are spliced into the
/// exported array by hand — the same approach the MLX EP takes, kept identical on purpose so the
/// two exports can be read by one tool.
fn counter_tail(counters: &[CounterSample], pid: u64) -> String {
    let mut tail = String::new();
    for c in counters {
        tail.push_str(&format!(
            ",{{\"name\":\"{}\",\"cat\":\"counter\",\"ph\":\"C\",\"ts\":{},\
             \"pid\":{},\"tid\":0,\"args\":{{\"{}\":{}}}}}",
            c.track, c.ts, pid, c.key, c.value
        ));
    }
    tail
}

/// Splice a comma-prefixed `tail` into a Chrome Trace JSON array before its closing bracket.
fn splice_counters(out: &mut String, tail: &str) {
    if !out.ends_with(']') {
        return;
    }
    out.pop();
    let had_events = out.trim_end().len() > 1; // more than just "["
    if !tail.is_empty() {
        if had_events {
            out.push_str(tail);
        } else {
            out.push_str(&tail[1..]); // strip the leading comma
        }
    }
    out.push(']');
}

fn is_default_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}

fn standard_op_args(node: &NodeDesc) -> Args {
    let mut args = Args::new()
        .with(ARG_NODE, node.name.clone())
        .with(ARG_DEVICE, DEVICE_HOST)
        .with("op_type", node.op_type.clone())
        .with("opset", i64::from(node.since_version));
    if !is_default_domain(&node.domain) {
        args = args.with(ARG_DOMAIN, node.domain.clone());
    }
    args
}

/// Per-node context for [`VulkanTracer::record_op_meta`].
///
/// A struct rather than eight positional arguments: the MLX version needed
/// `#[allow(clippy::too_many_arguments)]`, and the field names are self-documenting at the call
/// site.
#[derive(Default)]
pub struct OpMeta<'a> {
    pub in_shapes: &'a str,
    pub out_shapes: &'a str,
    pub dtype: &'a str,
    /// Selected shader variant stem, e.g. `matmul_nbits_q4_b32_gemv`.
    pub variant: Option<&'a str>,
    pub bytes: Option<u64>,
    pub flops: Option<u64>,
}

/// RAII guard for a timing phase. Emits a span and folds its wall time into the summary on drop.
#[must_use = "a PhaseGuard times its region only while alive; drop it at the end of the phase"]
pub struct PhaseGuard {
    phase: Phase,
    start: Instant,
    _span: SpanGuard,
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        tracer().record_phase(self.phase, self.start.elapsed());
    }
}

// -------------------------------------------------------------------------------------------
// Tests — host-side only. Nothing here has executed on a GPU (DESIGN.md §9.1.2).
// -------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // The residual is a subtraction, so the test that matters is that its VALUE VARIES WITH ITS
    // INPUT in both regimes (R10) — a residual that is always ~90% would be a constant wearing a
    // measurement's clothes, and one that is always 0 would hide the phase's whole cost.
    #[test]
    fn record_residual_varies_with_regime_and_never_double_counts_transfers() {
        // Warm regime: named children are ~2 ms of a 20 ms record; the vkCmd* calls are the rest.
        let warm = record_residual_us(20_000, 100, 1_500, 400);
        assert_eq!(
            warm, 18_000,
            "warm residual must be `record` minus named children"
        );
        assert!(
            warm * 100 / 20_000 >= 85,
            "warm residual must dominate `record` — that is the finding the row exists to expose"
        );

        // Cold regime: the same arithmetic must collapse to a SMALL residual when upload owns the
        // phase. If this ever equals the warm answer the row is not reading its input.
        let cold = record_residual_us(1_185_000, 1_148_000, 2_000, 30_000);
        assert_eq!(cold, 5_000);
        assert!(
            cold * 100 / 1_185_000 < 5,
            "cold residual must be a small slice — otherwise `record` is not upload-dominated"
        );
        assert_ne!(
            warm, cold,
            "residual must vary with its input, not with its name"
        );

        // `upload`/`readback` totals and the `cmd_upload` sub-span can bracket the same memcpy.
        // The caller passes the LARGER, never the sum; summing would invent child time and
        // under-report the residual. Guard the contract at the boundary.
        let (transfer_total, cmd_upload) = (9_000u64, 4_000u64);
        assert_eq!(record_residual_us(10_000, cmd_upload, 0, 0), 6_000);
        assert_eq!(
            record_residual_us(10_000, transfer_total.max(cmd_upload), 0, 0),
            1_000,
            "max(), not sum: 9000+4000 would saturate to 0 and erase the residual"
        );

        // Children may exceed the parent when a phase is re-entered; saturate rather than wrap.
        assert_eq!(record_residual_us(1_000, 5_000, 0, 0), 0);
    }

    fn cal(period: f32, bits: u32) -> GpuTimestampCalibration {
        GpuTimestampCalibration {
            timestamp_period_ns: period,
            valid_bits: bits,
            host_anchor_us: 1_000_000,
            device_anchor_ticks: 0,
            anchor_uncertainty_us: 0,
        }
    }

    #[test]
    fn masking_respects_valid_bits_and_never_shifts_by_64() {
        assert_eq!(mask_ticks(u64::MAX, 64), u64::MAX);
        assert_eq!(mask_ticks(u64::MAX, 32), 0xFFFF_FFFF);
        assert_eq!(mask_ticks(0x1_0000_0001, 32), 1);
        // A queue reporting zero valid bits has no usable counter at all.
        assert_eq!(mask_ticks(u64::MAX, 0), 0);
    }

    #[test]
    fn a_zero_valid_bits_queue_yields_no_gpu_time() {
        let c = cal(1.0, 0);
        assert!(!c.is_usable());
        assert_eq!(c.ticks_to_ns(0, 1000), None);
        assert_eq!(c.ticks_to_axis_us(1000), None);
    }

    #[test]
    fn timestamp_period_is_applied_and_is_not_assumed_to_be_one() {
        // NVIDIA: 1 ns/tick.
        assert_eq!(cal(1.0, 64).ticks_to_ns(100, 1100), Some(1000.0));
        // An AMD part reporting ~40 ns/tick: the same tick delta is 40x longer, and treating
        // ticks as nanoseconds would under-report GPU time by that factor.
        assert_eq!(cal(40.0, 64).ticks_to_ns(100, 1100), Some(40_000.0));
    }

    /// `timestampPeriod` and `timestampValidBits` as measured on the two devices on this desk,
    /// 2026-07-30, by two independent instruments that agree: the EP's own capability probe
    /// (`epctl --probe-loader`, i.e. `vk/caps.rs`) and `vulkaninfoSDK`. Cross-checked by
    /// `bench/timestamp_audit.py`, which exits non-zero if they ever disagree.
    ///
    /// These are used instead of invented constants so that the tests below fail when the
    /// arithmetic would be wrong *on hardware this project has*, rather than on hardware
    /// somebody imagined. lavapipe (the CI rasteriser) reports 1.0 / 64 — the same as NVIDIA —
    /// so **neither CI nor the discrete GPU can falsify the period or the mask**. The Intel
    /// part is the only local instrument that can, which is the concrete form of "Intel is the
    /// spec-conformance oracle".
    const IRIS_XE_PERIOD_NS: f32 = 52.0833;
    const IRIS_XE_VALID_BITS: u32 = 36;
    const RTX_4060_PERIOD_NS: f32 = 1.0;
    const RTX_4060_VALID_BITS: u32 = 64;

    #[test]
    fn treating_intel_ticks_as_nanoseconds_is_wrong_by_fifty_two_times() {
        // The plausible-but-wrong reading: ticks reported as nanoseconds. It is exactly right on
        // the RTX 4060 and on lavapipe, and it under-reports every Intel duration by ~52x. This
        // is the constant-factor error described in the module docs: nothing is negative,
        // nothing is absurd, and the number is wrong.
        let ticks = 100_000u64;
        let nvidia = cal(RTX_4060_PERIOD_NS, RTX_4060_VALID_BITS)
            .ticks_to_ns(0, ticks)
            .expect("usable");
        let intel = cal(IRIS_XE_PERIOD_NS, IRIS_XE_VALID_BITS)
            .ticks_to_ns(0, ticks)
            .expect("usable");

        // On the discrete part the naive reading and the correct one coincide — which is *why*
        // this class of bug survives, and is asserted so that the coincidence is on the record.
        assert_eq!(nvidia, ticks as f64);
        // On the integrated part they do not, by more than a factor of fifty.
        assert!(
            (intel / ticks as f64 - 52.0833).abs() < 1e-3,
            "period scaling not applied: {intel} ns for {ticks} ticks"
        );
        assert!(intel > nvidia * 50.0);
    }

    #[test]
    fn an_intel_counter_wrap_does_not_produce_a_negative_or_absurd_duration() {
        // 36 valid bits at 52.0833 ns/tick wraps roughly every 3579 s — about an hour of GPU
        // uptime, which is not exotic for a benchmark session. A wrap *during* a measurement
        // must yield the short true duration, not a negative one and not ~an hour.
        let c = cal(IRIS_XE_PERIOD_NS, IRIS_XE_VALID_BITS);
        let modulus = 1u64 << IRIS_XE_VALID_BITS;
        let begin = modulus - 1_000;
        let end = 500; // wrapped
        let ns = c.ticks_to_ns(begin, end).expect("usable");
        assert!(
            (ns - 1_500.0 * f64::from(IRIS_XE_PERIOD_NS)).abs() < 1e-3,
            "{ns}"
        );
        assert!(ns > 0.0, "a wrap must not produce a negative duration");
        // The unrecovered reading would be an entire wrap period. Assert we are nowhere near it.
        assert!(ns < 1e6, "{ns} ns is a wrap period, not a kernel");
    }

    #[test]
    fn undefined_upper_bits_on_a_thirty_six_bit_counter_are_masked_away() {
        // Vulkan leaves bits above `timestampValidBits` *undefined*, not zero. A driver that
        // returns garbage there must not turn a microsecond kernel into a geological era.
        let c = cal(IRIS_XE_PERIOD_NS, IRIS_XE_VALID_BITS);
        let garbage = (0xDEADu64 << IRIS_XE_VALID_BITS) | 4_000;
        let ns = c.ticks_to_ns(1_000, garbage).expect("usable");
        assert!(
            (ns - 3_000.0 * f64::from(IRIS_XE_PERIOD_NS)).abs() < 1e-3,
            "{ns}"
        );
    }

    #[test]
    fn a_wrap_is_only_recoverable_when_the_counter_is_narrower_than_a_u64() {
        // The 64-bit case has no modulus to add, so an end-before-start reading cannot be a
        // recoverable wrap — it is a bad pair, and is reported as no measurement rather than as
        // a very large positive one obtained by unsigned underflow.
        let c = cal(RTX_4060_PERIOD_NS, RTX_4060_VALID_BITS);
        assert_eq!(c.ticks_to_ns(1_000, 500), None);
    }

    #[test]
    fn a_single_counter_wrap_is_recovered() {
        // 32 valid bits: begin near the top, end just after wrapping.
        let c = cal(1.0, 32);
        let begin = 0xFFFF_FFF0u64;
        let end = 0x0000_000Fu64;
        assert_eq!(c.ticks_to_ns(begin, end), Some(31.0));
    }

    #[test]
    fn ticks_land_on_the_shared_microsecond_axis() {
        let c = GpuTimestampCalibration {
            timestamp_period_ns: 1.0,
            valid_bits: 64,
            host_anchor_us: 1_000_000,
            device_anchor_ticks: 1_000_000_000,
            anchor_uncertainty_us: 3,
        };
        // 2 ms of ticks after the anchor -> 2000 us after the anchored host time.
        assert_eq!(c.ticks_to_axis_us(1_002_000_000), Some(1_002_000));
        // Before the anchor is legal and must not saturate to the anchor.
        assert_eq!(c.ticks_to_axis_us(999_000_000), Some(999_000));
    }

    #[test]
    fn submit_is_the_only_phase_that_observes_no_gpu_work() {
        for p in Phase::ALL {
            assert_eq!(p.observes_gpu_work(), p != Phase::Submit, "{:?}", p);
        }
        assert!(Phase::Submit.caveat().contains("NOT GPU time"));
        assert!(Phase::FenceWait.caveat().contains("UPPER BOUND"));
    }

    /// Parentage must be structural, and every nested phase must say so in its own caveat.
    ///
    /// `Phase::Record` was wired, invoked, emitted on every span, and its caveat said "amortised
    /// across replays" while 96% of its measured time was a staging memcpy nested inside it. The
    /// exclusion mechanism was present and its content was false. This is the falsifier for the
    /// content: if a phase is nested and its caveat stops saying so, or a phase stops being
    /// declared nested while `vk::session` still brackets it, this goes red.
    /// Switch's per-dispatch sub-phases merged in cleanly and, under the previous `_ => None`
    /// catch-all, would have been classified as top-level siblings and added into SIBLING TOTAL —
    /// double-counting their own parent. This is the falsifier for that whole class: every phase
    /// whose caveat calls itself a sub-phase must have a parent.
    #[test]
    fn a_sub_record_phase_can_never_be_counted_as_a_sibling() {
        for p in Phase::ALL {
            if p.caveat().contains("sub-record") {
                assert_eq!(
                    p.nested_in(),
                    Some(Phase::Record),
                    "{p:?} describes itself as sub-record but is not modelled as nested"
                );
                assert!(!p.is_sibling(), "{p:?} would be summed into SIBLING TOTAL");
            }
        }
        assert!(!Phase::CmdUpload.is_sibling());
        assert!(!Phase::DescAlloc.is_sibling());
        assert!(!Phase::PipelineLookup.is_sibling());
        assert!(
            Phase::CmdUpload.caveat().contains("OVERLAPS `upload`"),
            "cmd_upload and upload bracket the same memcpy; the artifact must say so"
        );
    }

    #[test]
    fn nested_phases_are_declared_both_structurally_and_in_their_caveat() {
        assert_eq!(Phase::Upload.nested_in(), Some(Phase::Record));
        assert_eq!(Phase::Readback.nested_in(), Some(Phase::Record));
        for p in Phase::ALL {
            match p.nested_in() {
                Some(parent) => {
                    assert!(
                        p.caveat().contains("NESTED INSIDE"),
                        "{p:?} is nested in {parent:?} but its caveat does not say so — an \
                         aggregator that reads prose will double-count it"
                    );
                    assert!(
                        parent.nested_in().is_none(),
                        "only one level of nesting is modelled; {parent:?} gained a parent"
                    );
                    assert!(!p.is_sibling());
                }
                None => assert!(p.is_sibling()),
            }
        }
        // The parent must warn that it CONTAINS its children, not just the reverse: whoever reads
        // the 68% row reads the parent's caveat, not the child's.
        assert!(
            Phase::Record.caveat().contains("CONTAINS"),
            "the phase that swallowed a memcpy must say what it swallowed"
        );
        assert!(
            !Phase::Record.caveat().contains("amortised across replays"),
            "this claim was falsified by measurement and by an unwired record-path counter"
        );
    }

    #[test]
    fn phase_tags_are_unique_and_stable() {
        let mut seen = HashSet::new();
        for p in Phase::ALL {
            assert!(
                seen.insert(p.as_str()),
                "duplicate phase tag {}",
                p.as_str()
            );
        }
        assert_eq!(Phase::FenceWait.as_str(), "fence_wait");
    }

    /// No structural span name may be a prefix of another trace name.
    ///
    /// Every subgraph matcher in `bench/` matched with `startswith`. A compute-call bracket span
    /// named `vulkan.compute` was therefore captured by the same matcher as the `record_path()`
    /// instants
    /// `vulkan.compute[REPLAY]` — separable only by `cat`/`ph`, which the matchers did not read.
    /// The two vocabularies would have been mixed inside one reduction with nothing raising.
    ///
    /// This is the falsifier for the whole class, not for that one pair: it fails if any future
    /// span, phase or instant name is a prefix of any other.
    #[test]
    fn no_trace_name_is_a_prefix_of_another() {
        let mut names: Vec<String> = vec![
            SPAN_COMPUTE_CALL.to_string(),
            SPAN_SUBGRAPH.to_string(),
            format!("{INSTANT_RECORD_PATH}["),
            "vulkan.gpu.".to_string(),
        ];
        names.extend(Phase::ALL.iter().map(|p| format!("vulkan.{}", p.as_str())));
        for a in &names {
            for b in &names {
                if std::ptr::eq(a, b) {
                    continue;
                }
                assert!(
                    !(a != b && b.starts_with(a.as_str())),
                    "{b:?} starts with {a:?}; a prefix matcher cannot tell them apart"
                );
            }
        }
    }

    /// The span-vocabulary table in this module's header must list every name the module emits.
    ///
    /// `bench/phases.py` cites that table as "the artifact declares its own structure" and
    /// `bench/trace_vocabulary.py` parses it. A span added to the code and not to the table is a
    /// module of record that cannot see the phase — which is exactly how a phase reached a
    /// committed profile that no consumer could read.
    #[test]
    fn every_emitted_span_name_appears_in_the_header_vocabulary_table() {
        let header = include_str!("trace.rs");
        let header = &header[..header
            .find("use std::cell::Cell;")
            .expect("header ends at `use`")];
        for name in [SPAN_COMPUTE_CALL, SPAN_SUBGRAPH, INSTANT_RECORD_PATH] {
            assert!(
                header.contains(name),
                "{name} is emitted but not declared in the span vocabulary table"
            );
        }
        for p in Phase::ALL {
            let name = format!("vulkan.{}", p.as_str());
            assert!(
                header.contains(&name),
                "{name} is emitted but not declared in the span vocabulary table"
            );
        }
    }

    /// The vocabulary table's row for `vulkan.compute_call` must describe the instrumented
    /// success-path region inside `compute_impl`, never ORT's literal whole `Compute` callback.
    ///
    /// This row regressed to "The **whole** `Compute` callback, ORT entry to return" once, which
    /// contradicts [`SPAN_COMPUTE_CALL`]'s own doc comment a few lines below it in this same
    /// file, contradicts [`VulkanTracer::compute_region`]'s doc, and is exactly the wording
    /// `bench/cuda_profile.py`, `bench/phases.py` and `bench/trace_vocabulary.py` were separately
    /// corrected away from. This is the falsifier for that regression: it reads the row out of
    /// this file's own header table and fails if the contradictory wording reappears.
    #[test]
    fn compute_call_vocabulary_row_does_not_claim_the_literal_whole_callback() {
        let header = include_str!("trace.rs");
        let header = &header[..header
            .find("use std::cell::Cell;")
            .expect("header ends at `use`")];
        let row_start = header
            .find("| `vulkan.compute_call` |")
            .expect("the vocabulary table must have a vulkan.compute_call row");
        let row_end =
            row_start + header[row_start..].find('\n').unwrap_or(header.len() - row_start);
        let row = &header[row_start..row_end];
        for banned in ["**whole**", "ORT entry to return"] {
            assert!(
                !row.contains(banned),
                "the vulkan.compute_call vocabulary row reintroduced literal-whole-callback \
                 wording: {banned:?}. It is the instrumented success-path region inside \
                 compute_impl, absent on early-outs before it — not ORT's literal Compute entry."
            );
        }
        assert!(
            row.contains("compute_impl") && row.contains("absent"),
            "the vulkan.compute_call row must state it opens inside compute_impl and is absent \
             on early-outs before it"
        );
    }

    /// `subgraph_region`'s doc must describe the `dispatch_ort` dispatch bracket for one
    /// subgraph, never the whole `Compute` callback or the caller's end-to-end latency.
    ///
    /// The doc regressed to exactly that once: it called the span "one fused subgraph's whole
    /// `Compute` call" and "the end-to-end latency of the subgraph *as the caller experiences
    /// it*, which is the number a user pays" — that is `compute_region`'s territory, not
    /// `subgraph_region`'s, and it contradicts this module's own vocabulary table, which already
    /// (correctly) describes `vulkan.subgraph` as "opened inside `dispatch_ort`". This is the
    /// falsifier for that regression: it reads the doc comment immediately above
    /// `pub fn subgraph_region` out of this file's own source and fails if the contradictory
    /// wording reappears.
    #[test]
    fn subgraph_region_doc_never_claims_the_whole_compute_call() {
        let src = include_str!("trace.rs");
        let fn_pos = src
            .find("pub fn subgraph_region(")
            .expect("subgraph_region must exist");
        let section_start = src
            .find("// --- Execution view")
            .expect("the execution-view section marker must exist");
        assert!(section_start < fn_pos, "the section marker must precede subgraph_region");
        let doc = &src[section_start..fn_pos];
        for banned in [
            "whole `Compute` call",
            "whole `Compute`",
            "end-to-end latency",
            "the number a user pays",
        ] {
            assert!(
                !doc.contains(banned),
                "subgraph_region's doc reintroduced whole-Compute wording: {banned:?}. \
                 subgraph_region is the dispatch_ort dispatch bracket for one subgraph, not the \
                 wider compute-call bracket -- that is compute_region, which is itself the \
                 instrumented success-path region inside compute_impl and not ORT's literal \
                 Compute callback either."
            );
        }
        assert!(
            doc.contains("dispatch_ort"),
            "subgraph_region's doc must describe it as the dispatch region opened inside \
             dispatch_ort"
        );
    }

    /// `compute_region` brackets the instrumented success path inside `compute_impl`; it must not
    /// be modelled as a phase.
    ///
    /// `Phase::BindCheck` was added to this enum to close a large blind spot and accounted for a
    /// tiny fraction of the region it was documented as explaining — the exact figures came from
    /// a run that is not a committed artifact, so they are not quoted here. It was also the first
    /// sibling phase structurally *outside* `vulkan.subgraph`, which the containment contract in
    /// `docs/PERF.md` and `bench/phases.py` does not admit, so every traced run self-reported a
    /// phase-tree disagreement. The region is now bracketed by a structural span instead, which
    /// needs no place in the phase tree.
    #[test]
    fn the_compute_call_bracket_is_not_a_phase() {
        for p in Phase::ALL {
            assert_ne!(
                format!("vulkan.{}", p.as_str()),
                SPAN_COMPUTE_CALL,
                "the compute-call bracket must not be a summable phase"
            );
            assert_ne!(format!("vulkan.{}", p.as_str()), SPAN_SUBGRAPH);
        }
        assert_eq!(Phase::ALL.len(), 10);
    }

    #[test]
    fn a_replay_on_an_unseen_shape_key_is_reported_as_a_rerecord() {
        // The tracer singleton may be disabled in this process, so exercise the classification
        // logic through a local summary rather than the global tracer.
        let mut seen: HashMap<String, HashSet<String>> = HashMap::new();
        let classify = |seen: &mut HashMap<String, HashSet<String>>, tag: &str, key: &str| {
            let known = seen.get(tag).is_some_and(|k| k.contains(key));
            seen.entry(tag.to_string())
                .or_default()
                .insert(key.to_string());
            if known {
                RecordPath::Replay
            } else {
                RecordPath::Rerecord
            }
        };
        assert_eq!(classify(&mut seen, "sg0", "1x128"), RecordPath::Rerecord);
        assert_eq!(classify(&mut seen, "sg0", "1x128"), RecordPath::Replay);
        assert_eq!(classify(&mut seen, "sg0", "1x256"), RecordPath::Rerecord);
        // A second subgraph has its own key set — one shared "last key" would mislabel every
        // alternating call.
        assert_eq!(classify(&mut seen, "sg1", "1x128"), RecordPath::Rerecord);
    }

    #[test]
    fn counter_events_splice_into_an_empty_and_a_populated_array() {
        let counters = vec![CounterSample {
            track: "vulkan.island_count".into(),
            key: "islands".into(),
            value: 3.0,
            ts: 42,
        }];
        let tail = counter_tail(&counters, 7);

        let mut empty = String::from("[]");
        splice_counters(&mut empty, &tail);
        assert!(empty.starts_with("[{"), "{empty}");
        assert!(!empty.contains("[,"), "leading comma survived: {empty}");
        assert!(empty.ends_with("]"));

        let mut populated = String::from("[{\"name\":\"x\"}]");
        splice_counters(&mut populated, &tail);
        assert!(populated.contains("},{"), "{populated}");
        assert!(populated.ends_with("]"));
    }

    #[test]
    fn the_gpu_lane_is_not_a_plausible_thread_id() {
        // GPU spans must not land on a real thread's lane; the base is far outside the range any
        // OS assigns, and each queue family gets its own lane.
        const { assert!(GPU_LANE_BASE > 1 << 24) };
        // Distinct queue families get distinct lanes.
        assert_ne!(gpu_lane(0), gpu_lane(1));
    }

    #[test]
    fn the_tracer_is_inert_without_the_env_var() {
        // The singleton reads the environment once. In the test process the trace env is not set
        // (nothing in this suite sets it), so the tracer must be disabled and every recorder must
        // be a no-op that neither panics nor allocates a trace.
        let t = tracer();
        if std::env::var_os(ENV_TRACE).is_none() {
            assert!(!t.is_enabled());
            t.record_partition(&PartitionStats::default());
            t.record_transfer(Transfer::Upload, 1024, Duration::from_micros(10));
            assert!(t.phase(Phase::Submit).is_none() || t.active());
            t.export(); // must not write anything
        }
    }

    #[test]
    fn unusable_calibrations_produce_no_gpu_spans() {
        let report = GpuTimestampReport {
            calibration: cal(1.0, 0),
            queue_family: 0,
            intervals: vec![GpuInterval {
                label: "Add".into(),
                begin_ticks: 0,
                end_ticks: 1000,
                node_index: Some(0),
                flops: Some(1024),
                bytes: Some(4096),
            }],
        };
        assert_eq!(tracer().record_gpu_intervals(&report), 0);
    }
}
