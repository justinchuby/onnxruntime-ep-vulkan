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
//! | Span | cat | Clock | What it means |
//! |---|---|---|---|
//! | `vulkan.compute_call` | `ep` | host | **The whole ORT `Compute` callback**, opened as the first statement of the callback body and closed on every return, including every refusal. Carries `outcome`. This is the only span entitled to be called a `Compute` total. |
//! | `vulkan.subgraph` | `ep` | host | The **instrumented dispatch success path** — `dispatch_ort` entry to `dispatch_ort` return. Strictly nested inside `vulkan.compute_call`, and strictly narrower than it: the binding checks that run before dispatch are outside this span. |
//! | `vulkan.compile` | `ep.phase` | host | `Compile`: plan build, pipeline/SPIR-V creation, descriptor layout. Once per subgraph. |
//! | `vulkan.prepack` | `ep.phase` | host | Weight prepack + upload of block-quantised initializers. Once per `PackKey`. |
//! | `vulkan.record` | `ep.phase` | host | The `Compute` recording bracket, `vkBeginCommandBuffer`..`vkEndCommandBuffer`. **Despite the name, dominated by the staging upload prepared inside it (~96-98% on Phi-3.5), not by command recording (1-3%).** The host upload `memcpy` is nested here as `vulkan.cmd_upload` — it is *not* buffer allocation. See `Phase::Record::caveat`. |
//! | `vulkan.desc_alloc` | `ep.phase` | host | **Nested inside `record`.** `vkCreateDescriptorPool` + `vkAllocateDescriptorSets` + `vkUpdateDescriptorSets`. |
//! | `vulkan.pipeline_lookup` | `ep.phase` | host | **Nested inside `record`.** `PipelineCache::get_or_create`. |
//! | `vulkan.cmd_upload` | `ep.phase` | host | **Nested inside `record`.** Host `memcpy` into staging + `vkCmdCopyBuffer` recording for all inputs. This is where host upload cost lives; buffer allocation is a different, unmeasured region. |
//! | `vulkan.upload` | `ep.phase` | host | **Nested inside `record`.** Host→device staging copy of inference inputs. Carries `bytes`. |
//! | `vulkan.submit` | `ep.phase` | host | **`vkQueueSubmit` only.** Host bookkeeping. Measures no GPU work. |
//! | `vulkan.fence_wait` | `ep.phase` | host | CPU blocked on the fence. Upper bound on GPU time, not GPU time. |
//! | `vulkan.readback` | `ep.phase` | host | **Nested inside `record`.** Device→host copy of outputs. Carries `bytes`. |
//! | `vulkan.compute[<PATH>]` | `ep.path` | — | **Instant, not a span.** The recording path one `Compute` took. Shares a prefix with `vulkan.compute_call` and means something else; match on the exact name or the `cat`, never on a prefix. |
//! | `vulkan.gpu.*` | `gpu` | **device** | GPU execution, from `VkQueryPool` timestamp queries only. Emitted on a separate device lane. |
//!
//! # The attribution model
//!
//! There are exactly three tiers and they are not interchangeable:
//!
//! * **Total** — `vulkan.compute_call`. The wall an ORT caller pays for one `Compute`.
//! * **Sibling** — the `ep.phase` spans with `nested_in: "none"`. Disjoint by construction, so
//!   they may be summed. Every one of them is inside `vulkan.subgraph`, which is inside the
//!   total; none of them covers the pre-dispatch binding checks.
//! * **Child** — the `ep.phase` spans with `nested_in: "<parent>"`. **Never summable with their
//!   parent**, because their cost is already inside it.
//!
//! `Total - Σ Sibling` is a **residual that is computed, never assumed to be zero**. It is real
//! host cost this vocabulary does not name (binding checks, ORT-side entry, the `dispatch_ort`
//! prologue and epilogue), and it is only meaningful on a call whose `outcome` is `ok`: a refused
//! call left the span early and its sibling set is a prefix, not a decomposition. A negative
//! residual is oversubscription — a contradiction in the instrument, never a number to report.
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

/// How one ORT `Compute` callback ended, as carried on the `vulkan.compute_call` span.
///
/// The default is [`ComputeOutcome::Unresolved`] and it is load-bearing: the guard is opened
/// before any check runs and is resolved by exactly one statement on the way out. A path that
/// leaves the callback without passing through that statement — a panic converted to a status by
/// `guard_ffi_status`, a future early return someone forgets to route — publishes `unresolved`,
/// and every consumer treats `unresolved` the same way it treats `failed`: not a decomposition.
///
/// The failure mode this removes is the one that matters for a public number: a partial call
/// whose phase spans are a *prefix* of the vocabulary looking exactly like a complete call whose
/// phase spans are a *decomposition* of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ComputeOutcome {
    /// The callback has not reached its resolution point. See the type docs.
    #[default]
    Unresolved,
    /// The callback returned a null `OrtStatus`: ORT was handed a completed inference.
    Ok,
    /// The callback returned a non-null `OrtStatus`. Some phases may have run; the set of them
    /// that did is a prefix of the vocabulary and must not be read as an attribution.
    Failed,
}

impl ComputeOutcome {
    /// Stable lowercase tag written into the span's `outcome` arg.
    ///
    /// The admissibility rule this tag feeds — *only `ok` admits a residual* — is enforced where
    /// the residual is actually computed, in `bench/phases.py::COMPUTE_CALL_OUTCOMES`. It is
    /// deliberately not duplicated as a predicate here: a second copy in a language that never
    /// subtracts anything would be an instrument nobody invokes, and two spellings of one rule is
    /// how they drift.
    pub fn as_str(self) -> &'static str {
        match self {
            ComputeOutcome::Unresolved => "unresolved",
            ComputeOutcome::Ok => "ok",
            ComputeOutcome::Failed => "failed",
        }
    }
}

/// The value carried in the `boundary` arg of every `vulkan.compute_call` span.
///
/// Named rather than described so a consumer can assert on it: the whole point of this span is
/// *where it starts*, and a reader that cannot check where it starts is trusting prose.
pub const COMPUTE_CALL_BOUNDARY: &str = "ort_compute_callback";

/// Guard for the whole ORT `Compute` callback. See [`VulkanTracer::compute_call_region`].
///
/// Resolve it with [`ComputeCallGuard::resolve`] on the way out. Dropping it writes the args and
/// closes the span; an unresolved drop is recorded honestly rather than assumed successful.
#[must_use = "the compute-call span is recorded only while the guard is alive"]
pub struct ComputeCallGuard {
    nodes: usize,
    outcome: ComputeOutcome,
    // Dropped after `Drop::drop` runs, so the args written there land on this span.
    span: SpanGuard,
}

impl ComputeCallGuard {
    /// Record how the callback ended. Called once, on the single return path.
    pub fn resolve(&mut self, outcome: ComputeOutcome) {
        self.outcome = outcome;
    }
}

impl Drop for ComputeCallGuard {
    fn drop(&mut self) {
        self.span.set_args(
            Args::new()
                .with("nodes", self.nodes as u64)
                .with(ARG_DEVICE, DEVICE_HOST)
                .with("boundary", COMPUTE_CALL_BOUNDARY)
                .with("outcome", self.outcome.as_str()),
        );
    }
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

    /// A tracer wired to an in-memory collector, for tests that need to read back the events an
    /// instrument actually emitted rather than trust its docstring.
    #[cfg(test)]
    fn for_test(enabled: bool) -> Self {
        let (ctx, mem) = if enabled {
            let (ctx, mem) = TraceContext::in_memory();
            (ctx, Some(mem))
        } else {
            (TraceContext::noop(), None)
        };
        VulkanTracer {
            ctx,
            mem,
            path: None,
            counters: Mutex::new(Vec::new()),
            op_times: Mutex::new(HashMap::new()),
            summary: Mutex::new(Summary::default()),
            verbose: false,
            gpu_timestamps_requested: false,
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

    /// Span around the **whole ORT `Compute` callback**, from the first statement of the callback
    /// body to its return, on every path including every refusal.
    ///
    /// This is the only span in this module entitled to be described as a `Compute` total.
    /// [`subgraph_region`](Self::subgraph_region) is narrower — it starts inside the engine, after
    /// the binding checks — and calling *that* one a whole `Compute` (as this module's own
    /// docstring did until this change) overstates its coverage by however long the checks take,
    /// which is exactly the cost this instrument was asked to make visible.
    ///
    /// `None` when nothing is listening. The disabled path is one relaxed atomic load and a
    /// branch: no clock read, no allocation, no formatting. The args are built once, on drop, and
    /// only for a guard that exists.
    #[inline]
    pub fn compute_call_region(&self, node_count: usize) -> Option<ComputeCallGuard> {
        if !self.is_enabled() {
            return None;
        }
        Some(ComputeCallGuard {
            nodes: node_count,
            outcome: ComputeOutcome::default(),
            span: self.ctx.span("vulkan.compute_call", "ep"),
        })
    }

    /// Span around the engine's **instrumented dispatch success path** — `dispatch_ort` entry to
    /// `dispatch_ort` return.
    ///
    /// Host wall time. It covers upload, record-or-replay, submit, fence wait and readback. It
    /// does **not** cover the ORT callback's binding checks, which run before the engine is
    /// entered: for the caller-visible total use [`compute_call_region`](Self::compute_call_region)
    /// and read the two together. Deliberately not called "GPU time" anywhere, and — since this
    /// change — deliberately not called the whole `Compute` call either.
    pub fn subgraph_region(&self, node_count: usize) -> SpanGuard {
        if !self.is_enabled() {
            return self.ctx.span("vulkan.subgraph", "ep");
        }
        self.ctx.span("vulkan.subgraph", "ep").with_args(
            Args::new()
                .with("nodes", node_count as u64)
                .with(ARG_DEVICE, DEVICE_HOST),
        )
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
                format!("vulkan.compute[{}]", resolved.as_str()),
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

    // ── issue #88: the total boundary, and the guards that keep its name honest ─────────────

    /// The instrument is only worth having if the span it emits says where it starts, what it
    /// covers, and how the call ended. Read it back from the collector rather than from prose.
    #[test]
    fn the_compute_call_span_publishes_its_boundary_and_a_resolved_outcome() {
        let t = VulkanTracer::for_test(true);
        {
            let mut g = t.compute_call_region(7).expect("tracing is on");
            g.resolve(ComputeOutcome::Ok);
        }
        let events = t.mem.as_ref().unwrap().events();
        let span = events
            .iter()
            .find(|e| e.name == "vulkan.compute_call")
            .expect("the total span must be emitted under its own name");
        assert_eq!(
            span.cat, "ep",
            "the total is an `ep` span, not an `ep.phase`"
        );
        let args = span.args.as_ref().expect("args are written on drop");
        assert_eq!(
            args.get("boundary").and_then(|v| v.as_str()),
            Some(COMPUTE_CALL_BOUNDARY),
            "a consumer must be able to CHECK where this span starts, not read that it does"
        );
        assert_eq!(args.get("outcome").and_then(|v| v.as_str()), Some("ok"));
        assert_eq!(args.get("nodes").and_then(|v| v.as_u64()), Some(7));
    }

    /// The whole point of the outcome arg: a call that left early must not look complete.
    #[test]
    fn an_unresolved_or_failed_compute_call_is_never_published_as_complete() {
        for (resolve, want) in [
            (None, "unresolved"),
            (Some(ComputeOutcome::Failed), "failed"),
        ] {
            let t = VulkanTracer::for_test(true);
            {
                let mut g = t.compute_call_region(1).unwrap();
                if let Some(o) = resolve {
                    g.resolve(o);
                }
            }
            let events = t.mem.as_ref().unwrap().events();
            let span = events
                .iter()
                .find(|e| e.name == "vulkan.compute_call")
                .unwrap();
            let outcome = span
                .args
                .as_ref()
                .and_then(|a| a.get("outcome"))
                .and_then(|v| v.as_str());
            assert_eq!(
                outcome,
                Some(want),
                "a call that never reached its resolution point must say so; defaulting to `ok` \
                 is how a refusal acquires a complete-looking attribution"
            );
        }
        // …and the tag every consumer keys its admissibility rule on is stable.
        assert_eq!(ComputeOutcome::Ok.as_str(), "ok");
        assert_eq!(ComputeOutcome::Failed.as_str(), "failed");
        assert_eq!(ComputeOutcome::Unresolved.as_str(), "unresolved");
        assert_eq!(ComputeOutcome::default(), ComputeOutcome::Unresolved);
    }

    /// Requirement 6: the disabled path is a branch, not a span.
    #[test]
    fn the_compute_call_region_costs_nothing_when_nothing_is_listening() {
        let t = VulkanTracer::for_test(false);
        assert!(
            t.compute_call_region(3).is_none(),
            "a disabled tracer must hand back no guard at all — an inert guard would still run \
             Drop, build Args and format three values on every Compute"
        );
        assert!(!t.is_enabled());
    }

    /// D4 / the name collision. `vulkan.compute_call` (span, `ep`) and `vulkan.compute[<PATH>]`
    /// (instant, `ep.path`) share a prefix and mean different things. A consumer that matches on
    /// a prefix will read a recording-path marker as a total.
    #[test]
    fn the_total_span_and_the_record_path_instant_are_told_apart_by_name_and_cat() {
        let t = VulkanTracer::for_test(true);
        {
            let mut g = t.compute_call_region(1).unwrap();
            g.resolve(ComputeOutcome::Ok);
        }
        t.record_path("subgraph0", RecordPath::FirstRecord, "4x4", 1);
        let events = t.mem.as_ref().unwrap().events();
        let totals: Vec<_> = events
            .iter()
            .filter(|e| e.name == "vulkan.compute_call")
            .collect();
        let paths: Vec<_> = events.iter().filter(|e| e.cat == "ep.path").collect();
        assert_eq!(totals.len(), 1, "exactly one total span per call");
        assert_eq!(paths.len(), 1, "the record-path marker is still emitted");
        assert!(
            paths[0].name.starts_with("vulkan.compute["),
            "the record-path instant keeps its published name; got {:?}",
            paths[0].name
        );
        assert_ne!(
            paths[0].name, totals[0].name,
            "the two must never collide on the exact name"
        );
        assert_ne!(paths[0].cat, totals[0].cat, "…nor on the cat");
        assert!(
            events
                .iter()
                .filter(|e| e.name.starts_with("vulkan.compute"))
                .count()
                > 1,
            "a PREFIX match sees both, which is exactly why neither may be matched by prefix"
        );
    }

    /// D2, made executable. DESIGN's §9.5 row for `record` claims the host upload memcpy is
    /// nested inside it, in `cmd_upload`, and that buffer allocation is a different region. The
    /// claim is only worth making if the production phase table says the same thing.
    #[test]
    fn the_host_upload_memcpy_is_declared_inside_record_and_not_in_an_allocation_phase() {
        assert_eq!(
            Phase::CmdUpload.nested_in(),
            Some(Phase::Record),
            "the memcpy phase is a CHILD of record; a sibling would be summed with its parent"
        );
        assert!(
            Phase::CmdUpload.caveat().contains("memcpy"),
            "cmd_upload's caveat must name the copy it contains"
        );
        assert!(
            Phase::Record.caveat().contains("cmd_upload"),
            "record's row must DISCLOSE the upload preparation nested in it — a reader who sees \
             only `record` would attribute a staging memcpy to command recording"
        );
        // There is no allocation phase in this vocabulary, and none of the ten phases may quietly
        // become one by claiming the memcpy. Exactly two rows may name it: `cmd_upload`, which
        // measures it, and `record`, which discloses that it is nested inside. Any third row
        // naming a host memcpy is a second claim on the same cost.
        let naming_memcpy: Vec<&str> = Phase::ALL
            .iter()
            .filter(|p| p.caveat().contains("memcpy"))
            .map(|p| p.as_str())
            .collect();
        assert_eq!(
            naming_memcpy,
            vec!["record", "cmd_upload"],
            "the host upload memcpy is measured by `cmd_upload` and disclosed by its parent \
             `record`, and by nothing else; DESIGN §9.5 says the same and an allocation row \
             saying it would be a second claim on one cost"
        );
        assert_eq!(
            Phase::Record.nested_in(),
            None,
            "`record` names the memcpy as a PARENT disclosing a child, so it must be a sibling"
        );
    }

    /// D3, made executable, without a Vulkan device. The rule is about a *position in the source*
    /// — the increment must be downstream of the fence result — so the guard reads the source.
    /// A mutation that moves the call next to `vkQueueSubmit` turns this red.
    #[test]
    fn queue_submits_are_counted_only_after_the_fence_returned_success() {
        let src = include_str!("vk/session.rs");
        let calls: Vec<_> = src.match_indices("on_submit_completed()").collect();
        assert_eq!(
            calls.len(),
            1,
            "exactly one production increment of queue_submits_completed; found {}",
            calls.len()
        );
        let at = calls[0].0;
        let before = &src[..at];
        // The nearest preceding conditional must be the fence result, not the submit result.
        let fence_gate = before
            .rfind("if fence_ok {")
            .expect("the increment must be inside an `if fence_ok` block");
        let host_t1 = before
            .rfind("let host_t1 = onnx_runtime_tracer::absolute_now_us();")
            .expect("the increment must be after the fence wait has been timed out");
        assert!(
            host_t1 < fence_gate && fence_gate < at,
            "order must be: fence wait completes -> host_t1 -> `if fence_ok` -> increment. A \
             submission counted before the fence returns is a submission the host cannot know \
             executed anything."
        );
        assert!(
            !before[fence_gate..].contains('}'),
            "the increment must be the guarded statement itself, not a statement after the block"
        );
        // And the counter is a real counter, not a constant.
        let (_, _, _, before_n) = crate::counters::dispatch_resources::all();
        crate::counters::dispatch_resources::on_submit_completed();
        let (_, _, _, after_n) = crate::counters::dispatch_resources::all();
        assert_eq!(after_n, before_n + 1);
    }

    /// The three resource counters must be incremented on the success side of their Vulkan call.
    /// Same reasoning as the fence gate above, same held-out mutation: move the call up past the
    /// `return None` and the test goes red.
    #[test]
    fn dispatch_resource_counters_sit_downstream_of_the_call_that_can_fail() {
        for (src, needle, err) in [
            (
                include_str!("vk/pipeline.rs"),
                "on_pool_created()",
                "vkCreateDescriptorPool failed",
            ),
            (
                include_str!("vk/pipeline.rs"),
                "on_sets_allocated(",
                "vkAllocateDescriptorSets failed",
            ),
            (
                include_str!("vk/cmd.rs"),
                "on_command_buffers_allocated(",
                "vkAllocateCommandBuffers failed",
            ),
        ] {
            let at = src
                .find(needle)
                .unwrap_or_else(|| panic!("no production call to {needle}"));
            let fail = src
                .find(err)
                .unwrap_or_else(|| panic!("no failure arm mentioning {err}"));
            assert!(
                fail < at,
                "{needle} must come after the failure arm for {err}; counting before it means a \
                 driver refusal is reported as resource consumption"
            );
        }
    }

    /// Requirement 1: the three tiers are distinct and the sibling set is disjoint. A phase that
    /// is inside another phase may never be summed with it, and the total is neither.
    #[test]
    fn the_attribution_tiers_are_disjoint_and_the_total_is_not_a_phase() {
        let siblings: Vec<&str> = Phase::ALL
            .iter()
            .filter(|p| p.is_sibling())
            .map(|p| p.as_str())
            .collect();
        let children: Vec<&str> = Phase::ALL
            .iter()
            .filter(|p| !p.is_sibling())
            .map(|p| p.as_str())
            .collect();
        assert!(!siblings.is_empty() && !children.is_empty());
        for c in &children {
            assert!(
                !siblings.contains(c),
                "{c} cannot be both a sibling and a child"
            );
        }
        // Every child's parent is itself a sibling: one level of nesting, so `Σ siblings` is a
        // complete partition of the named cost and the residual is everything else.
        for p in Phase::ALL {
            if let Some(parent) = p.nested_in() {
                assert!(
                    parent.is_sibling(),
                    "{}'s parent {} must be a sibling, or Σ siblings is not a partition",
                    p.as_str(),
                    parent.as_str()
                );
            }
        }
        // The total is not in the phase vocabulary at all — it cannot be summed by accident.
        assert!(
            !Phase::ALL
                .iter()
                .any(|p| format!("vulkan.{}", p.as_str()) == "vulkan.compute_call"),
            "the total must not be reachable as a phase"
        );
    }

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
