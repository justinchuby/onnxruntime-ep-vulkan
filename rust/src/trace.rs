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
//!   counted separately with byte counts ([`Phase::Upload`], [`Phase::Readback`]), because on a
//!   discrete GPU they are frequently the whole story and a benchmark that hides them is
//!   marketing (charter, and `DESIGN.md` §9.2). They are folded into the summary by
//!   [`VulkanTracer::record_transfer`] and **emit no span of their own** — see
//!   [`Phase::emits_span`].
//! * **Recording is NOT amortised in this engine, and now the instrument says so.**
//!   `ENGINE.md` §6.1 describes a record-once / replay-many model. This engine does not implement
//!   it: `dispatch_ort` resets and re-records the whole command buffer on every `Compute` call,
//!   so [`RecordPath::Replay`] is unreachable here by construction and
//!   [`RecordPath::RecordedAgain`] is what every call after the first reports. That is a
//!   different fact from [`RecordPath::Rerecord`], which names a re-record *caused by a shape-key
//!   change* — the Vulkan analogue of MLX's compile-cache `RETRACE`. Do not read a zero in the
//!   replay column as "replays were rare"; it is zero because there is no replay path.
//! * **Real GPU time comes from the device's own clock or not at all.** See below.
//!
//! # The span vocabulary
//!
//! Three tiers, and they are **strictly nested but never interchangeable**. Read
//! [`the two-level attribution model`](#the-two-level-attribution-model) below before subtracting
//! anything.
//!
//! | Span | cat | tier | Clock | What it means |
//! |---|---|---|---|---|
//! | `vulkan.ort_compute_callback` | `ep.compute_call` | callback | host | The **whole `OrtNodeComputeInfo::Compute` callback body**, opened inside the `extern "C"` entry point and closed as it returns to ORT. Carries `outcome`. |
//! | `vulkan.subgraph` | `ep` | dispatch | host | The **engine dispatch only** (`vk::session::dispatch_ort`). Entered after the callback's null/liveness/binding checks and **never opened at all** when one of them refuses. It is NOT the whole `Compute` call. |
//! | `vulkan.compile` | `ep.phase` | phase | host | `Compile`: plan build, pipeline/SPIR-V creation, descriptor layout. Once per subgraph. |
//! | `vulkan.prepack` | `ep.phase` | phase | host | Weight prepack + upload of block-quantised initializers. Once per `PackKey`. |
//! | `vulkan.record` | `ep.phase` | phase | host | The `Compute` recording bracket. **Despite the name, dominated by the staging upload it contains (~96-98% on Phi-3.5), not by command recording (1-3%).** See `Phase::Record::caveat`. |
//! | `vulkan.submit` | `ep.phase` | phase | host | **`vkQueueSubmit` only.** Host bookkeeping. Measures no GPU work. |
//! | `vulkan.fence_wait` | `ep.phase` | phase | host | CPU blocked on the fence. Upper bound on GPU time, not GPU time. |
//! | `vulkan.desc_alloc`, `vulkan.pipeline_lookup`, `vulkan.cmd_upload` | `ep.phase` | phase | host | Sub-phases **nested inside `vulkan.record`**; they carry `nested_in=record`. |
//! | `vulkan.gpu.*` | `gpu` | — | **device** | GPU execution, from `VkQueryPool` timestamp queries only. Emitted on a separate device lane. |
//!
//! ## Phases that are folded into the summary but emit NO span
//!
//! [`Phase::Upload`] and [`Phase::Readback`] appear in the printed summary table and in
//! [`Summary::phase_us`], but **production never calls [`VulkanTracer::phase`] for them**: the
//! engine reports them through [`VulkanTracer::record_transfer`], which folds a duration without
//! opening a span. A trace-JSON reader will therefore find no `vulkan.upload` / `vulkan.readback`
//! events, and an analyser must not treat their absence as "no transfer happened". This is stated
//! here because listing them as emitted spans is exactly the error that made
//! `Phase::Readback::nested_in()` claim a parent it never had.
//!
//! ## The two-level attribution model
//!
//! Issue #88 asks where the host cost of a `Compute` call goes. There are **two** residuals and
//! they live at different levels. They are disjoint quantities about different intervals, they
//! have different denominators, and **they must never be added to each other or substituted for
//! each other**:
//!
//! * **outer** = `Σ vulkan.ort_compute_callback − Σ vulkan.subgraph` — everything the callback
//!   body does *around* the engine dispatch: ORT context reads, liveness and binding checks,
//!   status construction, the post-return broken-commitment disclosure, and any call that
//!   refused before a dispatch was attempted. Denominator: the callback total.
//! * **inner** = `Σ vulkan.subgraph − Σ (top-level ep.phase spans)` — everything the dispatch
//!   does *around* its own top-level phases. Denominator: the dispatch total.
//!
//! `bench/phases.py::two_level_attribution` computes both, labels them, and refuses to report
//! either when the spans it needs are missing, empty, escaping, overlapping or unknown.
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
/// Trace arg key: which code region a span brackets, named in the source's own vocabulary.
///
/// Present so that an analyser identifies a span by a declared property rather than by matching
/// its name, which is what let `vulkan.subgraph` be read as "the whole Compute call" for as long
/// as its docstring said so.
pub const ARG_BOUNDARY: &str = "boundary";
/// Trace arg key: the level of the two-level attribution model this span belongs to.
///
/// One of [`TIER_CALLBACK`], [`TIER_DISPATCH`], [`TIER_PHASE`]. A residual computed between two
/// spans of the *same* tier is a bug; carrying the tier is what makes that mechanically
/// detectable instead of a naming convention.
pub const ARG_TIER: &str = "tier";
/// Trace arg key: what the ORT `Compute` callback returned. See [`ComputeOutcome`].
pub const ARG_OUTCOME: &str = "outcome";

/// [`ARG_BOUNDARY`] value for the `extern "C"` ORT `Compute` callback body (`ep::compute`).
pub const BOUNDARY_ORT_COMPUTE_CALLBACK: &str = "ort_compute_callback";
/// [`ARG_BOUNDARY`] value for the engine dispatch (`vk::session::dispatch_ort`).
pub const BOUNDARY_ENGINE_DISPATCH: &str = "engine_dispatch";

/// [`ARG_TIER`] value: the outer level — the whole callback body.
pub const TIER_CALLBACK: &str = "callback";
/// [`ARG_TIER`] value: the middle level — one engine dispatch, strictly inside a callback.
pub const TIER_DISPATCH: &str = "dispatch";
/// [`ARG_TIER`] value: the inner level — one phase, strictly inside a dispatch.
pub const TIER_PHASE: &str = "phase";

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
    ///
    /// **Summary-only: production emits no `vulkan.upload` span.** The engine reports this
    /// through [`VulkanTracer::record_transfer`], which folds a duration into the summary
    /// without opening a span. See the module docs.
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
    ///
    /// **Summary-only: production emits no `vulkan.readback` span**, and — unlike
    /// [`Phase::Upload`] — this work does **not** happen inside the `record` bracket. See
    /// [`Phase::nested_in`].
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
                 NESTED INSIDE `record` — already counted there, do not add to the sibling total. \
                 SUMMARY-ONLY: no `vulkan.upload` span is emitted; this row is folded in through \
                 record_transfer"
            }
            Phase::Record => {
                "host: the whole vkBeginCommandBuffer..vkEndCommandBuffer bracket. It CONTAINS \
                 `upload`/`cmd_upload` (the staging memcpy), `desc_alloc` and `pipeline_lookup`, \
                 so it is an INCLUSIVE interval and its name describes its bracket, not its \
                 content (R11). It does NOT contain `readback`, which runs after the fence wait. \
                 The split is regime-dependent and must be read from the child rows of THIS run, \
                 never from a remembered ratio: with a cold weight cache `cmd_upload` dominates \
                 it (measured 1148 of 1185 ms on Phi-3.5's first Compute), and with a warm cache \
                 the children collapse to ~1-2 ms and the UNNAMED RESIDUAL — the vkCmd* calls \
                 themselves — is ~90% of it. The summary prints that residual as its own row, but \
                 CUMULATIVELY over all calls, so it mixes the two regimes; the per-call split is \
                 only in the trace spans"
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
                "host: device->host copy; counts toward end-to-end latency. TOP-LEVEL, NOT nested \
                 in `record`: production reads outputs back AFTER the record bracket closes, \
                 after vkQueueSubmit and after the fence wait (see vk/session.rs). \
                 SUMMARY-ONLY: no `vulkan.readback` span is emitted; this row is folded in \
                 through record_transfer"
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
            // vkEndCommandBuffer; the input staging loop runs inside that bracket. See
            // session.rs (Record guard) — this is a fact about the call graph, not a policy, and
            // it must be re-checked if that guard moves.
            Phase::Upload => Some(Phase::Record),
            // Switch's per-dispatch sub-phases, added in `692e7d0`. They are documented in their
            // own caveats as "sub-record" and they are opened inside the Record guard.
            Phase::DescAlloc | Phase::PipelineLookup | Phase::CmdUpload => Some(Phase::Record),
            // READBACK IS NOT A CHILD OF `record`, AND SAYING SO WAS A FALSE STRUCTURAL CLAIM.
            //
            // It was declared `Some(Phase::Record)` alongside `Upload` because the two look
            // symmetrical from the enum. They are not symmetrical in the engine: the staging
            // upload is recorded into the command buffer inside the bracket, while the output
            // download happens in `dispatch_ort` *after* `drop(_record_guard)`, *after*
            // `vkQueueSubmit`, and *after* the fence wait — the outputs do not exist until the
            // GPU has drained. The false parentage made `SIBLING TOTAL` drop a genuine
            // top-level interval and told every aggregator that reads `nested_in` to subtract
            // readback from a bracket that never contained it.
            //
            // `Readback` therefore has no parent. It is disjoint from `record`, `submit` and
            // `fence_wait`, which is what makes summing the siblings legitimate.
            Phase::Readback => None,
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

    /// Whether production opens a `vulkan.<phase>` span for this phase, or only folds a duration
    /// into the summary.
    ///
    /// # Why a reader needs this
    ///
    /// The module's span table used to list `vulkan.upload` and `vulkan.readback` as emitted
    /// spans. They are not: `vk::session` reports both through
    /// [`VulkanTracer::record_transfer`], which never touches the trace document. An analyser
    /// that looks for them in the JSON finds nothing and — with no way to tell "this phase never
    /// emits" from "this phase did not run" — reports a transfer-free inference. Absence of a
    /// span is not absence of the work (R7), and this predicate is what makes the difference
    /// checkable instead of remembered.
    ///
    /// Exhaustive on purpose, for the same reason [`Phase::nested_in`] is.
    pub fn emits_span(self) -> bool {
        match self {
            // Summary-only: folded in by `record_transfer`, no `ph:"X"` event exists.
            Phase::Upload | Phase::Readback => false,
            Phase::Compile
            | Phase::Prepack
            | Phase::Record
            | Phase::DescAlloc
            | Phase::PipelineLookup
            | Phase::CmdUpload
            | Phase::Submit
            | Phase::FenceWait => true,
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
///
/// **This engine does not implement that model.** `vk::session::dispatch_ort` calls
/// `CommandPool::begin`, which resets the single pre-allocated command buffer, on every
/// `Compute`. There is no cached `VkCommandBuffer` to replay, so [`RecordPath::Replay`] is
/// unreachable from production and the steady state is [`RecordPath::RecordedAgain`]. Keeping
/// `Replay` in the vocabulary is deliberate: the column has to exist for its zero to mean
/// anything, and the day a replay path lands the instrument is already there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RecordPath {
    /// First recording of this subgraph's command buffer.
    FirstRecord,
    /// Replayed the cached `VkCommandBuffer` — the steady-state path in a record-once engine.
    /// **Never reported by this engine**; see the type docs.
    Replay,
    /// Re-recorded because the input shape key changed. A benchmark that reports a median over
    /// runs where this fired is measuring the recording path, not the steady state.
    Rerecord,
    /// Recorded from scratch again because this engine holds **no cached command buffer at all**
    /// — not a replay, and not a shape-driven re-record.
    ///
    /// Separate from [`RecordPath::Rerecord`] because the two have different causes and different
    /// fixes: a `Rerecord` says the benchmark's shapes are not steady, and a `RecordedAgain` says
    /// the engine never had a steady state to leave. Folding this into `Rerecord` would have
    /// reported every Phi-3.5 inference as a shape-key change, which is a false statement about
    /// the model.
    RecordedAgain,
}

impl RecordPath {
    pub fn as_str(self) -> &'static str {
        match self {
            RecordPath::FirstRecord => "FIRST_RECORD",
            RecordPath::Replay => "REPLAY",
            RecordPath::Rerecord => "RERECORD",
            RecordPath::RecordedAgain => "RECORDED_AGAIN",
        }
    }

    /// Index into [`Summary::record_paths`]. One mapping, declared once.
    fn slot(self) -> usize {
        match self {
            RecordPath::FirstRecord => 0,
            RecordPath::Replay => 1,
            RecordPath::Rerecord => 2,
            RecordPath::RecordedAgain => 3,
        }
    }

    /// Every path, in reporting order. Index `i` is the path whose `slot()` is `i`.
    pub const ALL: [RecordPath; 4] = [
        RecordPath::FirstRecord,
        RecordPath::Replay,
        RecordPath::Rerecord,
        RecordPath::RecordedAgain,
    ];
}

/// What the ORT `Compute` callback returned, as seen at the callback boundary.
///
/// Carried as the `outcome` arg on `vulkan.ort_compute_callback`. An analyser must not silently
/// drop a non-`Ok` call from an attribution total — a refused call is host cost that a user paid
/// — but it must not blend it into a "time per successful inference" either, which is why the
/// state is on the span rather than inferred from a counter difference.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ComputeOutcome {
    /// The callback did not reach its own return — a panic unwound through the guard, or the
    /// caller forgot to resolve the outcome. **The default**, so a lost outcome fails closed as
    /// "unknown" instead of silently reading as success.
    #[default]
    Unresolved,
    /// The callback returned a null `OrtStatusPtr`.
    Ok,
    /// The callback returned an `OrtStatus` — a broken commitment (`ep::disclose_broken_commitment`).
    Failed,
}

impl ComputeOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            ComputeOutcome::Unresolved => "unresolved",
            ComputeOutcome::Ok => "ok",
            ComputeOutcome::Failed => "failed",
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

    /// One slot per [`RecordPath`], indexed by `RecordPath::slot()`.
    record_paths: [u64; 4],
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

    /// Span around the **engine dispatch** for one fused subgraph — `vk::session::dispatch_ort`,
    /// and nothing else.
    ///
    /// # What this is NOT
    ///
    /// It is **not** the `Compute` call. This span opens inside `dispatch_ort`, which the
    /// callback reaches only after ORT's kernel context has been read and after the bound-count,
    /// bound-size and addressability checks have all passed; a `Compute` that refuses at any of
    /// those never opens this span at all, and the fault-injection control never reaches it
    /// either. Its docstring said "one fused subgraph's whole `Compute` call" from the day it
    /// landed, which is how an analyser came to compute a "cost outside the dispatch" of exactly
    /// zero by definition.
    ///
    /// The callback body is [`VulkanTracer::ort_compute_callback`]
    /// (`vulkan.ort_compute_callback`). This span nests strictly inside it, and the difference
    /// between the two is the *outer* residual of the two-level attribution model (module docs).
    ///
    /// Host wall time. Within the dispatch this covers upload, recording, submit, fence wait and
    /// readback — the latency of the dispatch *as the caller experiences it*, which is
    /// deliberately not called "GPU time" anywhere.
    pub fn subgraph_region(&self, node_count: usize) -> SpanGuard {
        if !self.is_enabled() {
            return self.ctx.span("vulkan.subgraph", "ep");
        }
        self.ctx.span("vulkan.subgraph", "ep").with_args(
            Args::new()
                .with("nodes", node_count as u64)
                // Machine-readable tier, so an analyser never has to infer the level from the
                // span name. `boundary` names the code region; `tier` names the level in the
                // attribution model. A residual computed from two spans of the same tier is a
                // bug, and these args are what makes that checkable.
                .with(ARG_BOUNDARY, BOUNDARY_ENGINE_DISPATCH)
                .with(ARG_TIER, TIER_DISPATCH)
                .with(ARG_DEVICE, DEVICE_HOST),
        )
    }

    /// Span around the **whole ORT `Compute` callback body** — the outer level of the two-level
    /// attribution model (module docs).
    ///
    /// # Exactly where the clock opens and closes
    ///
    /// `ep::compute` is the `extern "C"` function ORT calls. The guard is created there, after
    /// the one branch that cannot be timed truthfully (a null `OrtNodeComputeInfo`, which leaves
    /// no `OrtApi` to report through and which ORT never produces), and dropped as `compute`
    /// returns. It therefore contains:
    ///
    /// * `crate::guard_ffi_status` and all of `compute_impl` — the fault-injection control, the
    ///   null-context check, the liveness check, all three binding checks, the dispatch, and the
    ///   construction of any `OrtStatus`;
    /// * `ep::disclose_broken_commitment`, which runs *after* `compute_impl` returns and is
    ///   therefore host cost inside the callback that no dispatch-level span can see.
    ///
    /// It excludes only the null-`OrtNodeComputeInfo` guard above it. No claim is made here about
    /// how many ways `compute_impl` can return: the guard is scope-based, so a return that nobody
    /// enumerated is still inside it.
    ///
    /// # Outcome
    ///
    /// The outcome is not known until the callback is about to return, so it is set on the guard
    /// rather than at construction, and it defaults to [`ComputeOutcome::Unresolved`] — a panic
    /// that unwinds past the resolve point leaves the span honestly labelled instead of quietly
    /// labelled as a success.
    ///
    /// Returns a guard that emits nothing when tracing is off; the guard is otherwise the only
    /// producer of `vulkan.ort_compute_callback`.
    pub fn ort_compute_callback(&self, subgraph_id: u64, node_count: usize) -> ComputeCallGuard {
        ComputeCallGuard {
            enabled: self.is_enabled(),
            subgraph_id,
            node_count,
            start: Instant::now(),
            outcome: ComputeOutcome::Unresolved,
        }
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
                    .with(ARG_TIER, TIER_PHASE)
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
    ///
    /// The resolved path is folded into the process counters **unconditionally**, for the same
    /// reason [`Self::record_transfer`] counts bytes unconditionally: the run that gets quoted is
    /// the one nobody set `ONNXRUNTIME_EP_VULKAN_TRACE` on, and "was the command buffer
    /// re-recorded every inference?" is not a question a reader should have to re-run the model
    /// to answer.
    pub fn record_path(
        &self,
        subgraph_tag: &str,
        path: RecordPath,
        shape_key: &str,
        node_count: usize,
    ) -> RecordPath {
        let resolved = if !self.active() {
            // No shape-key history is kept when nothing is listening. Production passes an empty
            // shape key — for which the resolution below is the identity — so the counters read
            // the same either way, and the summary's history is not worth a mutex on the compute
            // path of a run that asked for no observability.
            path
        } else {
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
            let idx = resolved.slot();
            s.record_paths[idx] += 1;
            resolved
        };
        // ONE call site, after the branch, so the counter and the summary can never disagree
        // about which path was taken.
        crate::counters::record_command_buffer_path(resolved);
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
        // `Tracer::record_path` was unwired until issue #88: it had zero production call sites,
        // and printing "first-record=0 replay=0 rerecord=0" read as "this run recorded nothing",
        // which is a measurement. It is not one — it is the absence of a measurement. The NOT
        // WIRED branch is kept because it is still the truthful thing to print for a session that
        // never dispatched, and because the distinction it draws is the one this instrument
        // exists for.
        if s.record_paths.iter().all(|&n| n == 0) {
            out.push_str(
                "  compute:  record-path breakdown UNOBSERVED — no Compute call reached the \
                 recording decision in this session, so first-record/replay/rerecord/\
                 recorded-again are unmeasured, NOT zero.\n",
            );
        } else {
            out.push_str("  compute:  ");
            for p in RecordPath::ALL {
                out.push_str(&format!("{}={} ", p.as_str(), s.record_paths[p.slot()]));
            }
            out.push_str(
                "\n            (REPLAY is 0 by construction: dispatch_ort resets and re-records \
                 the command buffer on every Compute, so there is no replay path to take. Read \
                 that 0 as `no such path`, not as `replays were rare`.)\n",
            );
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
                "  host time (wall clock on the CPU thread). `upload` is NESTED INSIDE `record` \
                 — do NOT add this column. Rows marked [summary-only] emit no trace span:\n",
            );
            for phase in Phase::ALL {
                let Some((us, calls)) = s.phase_us.get(&phase) else {
                    continue;
                };
                out.push_str(&format!(
                    "              {}{:<11} {:>10} us (x{}){} — {}\n",
                    if phase.is_sibling() {
                        "    "
                    } else {
                        "  └─ "
                    },
                    phase.as_str(),
                    us,
                    calls,
                    // DERIVED, not restated. A phase folded in by `record_transfer` has no
                    // `ph:"X"` event, so a reader who greps the trace JSON for `vulkan.readback`
                    // finds nothing and must not read that absence as "no readback happened".
                    if phase.emits_span() {
                        ""
                    } else {
                        " [summary-only: no span]"
                    },
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
                .with(
                    "first_record",
                    s.record_paths[RecordPath::FirstRecord.slot()],
                )
                .with("replay", s.record_paths[RecordPath::Replay.slot()])
                .with("rerecord", s.record_paths[RecordPath::Rerecord.slot()])
                .with(
                    "recorded_again",
                    s.record_paths[RecordPath::RecordedAgain.slot()],
                )
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

/// RAII guard for the **outer** attribution level: the whole ORT `Compute` callback body.
///
/// Emits one `vulkan.ort_compute_callback` complete event on drop, carrying the outcome. Not a
/// [`SpanGuard`] because the outcome is only knowable at the end of the callback, and a span
/// whose args are fixed at construction cannot carry it.
///
/// The guard is created even when tracing is off (it is one `Instant::now()` and a bool) so that
/// the production call site has no `if traced { … }` branch to get wrong; `Drop` emits nothing
/// unless the trace document is being written.
#[must_use = "a ComputeCallGuard times the callback body only while alive; hold it until the \
              callback returns"]
pub struct ComputeCallGuard {
    enabled: bool,
    subgraph_id: u64,
    node_count: usize,
    start: Instant,
    outcome: ComputeOutcome,
}

impl ComputeCallGuard {
    /// Record what the callback is about to return to ORT.
    ///
    /// Call this on the *only* path that returns to ORT. Anything that unwinds past it leaves
    /// [`ComputeOutcome::Unresolved`] on the span, which is the honest label for a callback whose
    /// return nobody observed.
    pub fn set_outcome(&mut self, outcome: ComputeOutcome) {
        self.outcome = outcome;
    }

    /// The outcome the span will carry if it were dropped now. Exists so a test can assert on the
    /// fail-closed default without waiting for the emission.
    pub fn outcome(&self) -> ComputeOutcome {
        self.outcome
    }
}

impl Drop for ComputeCallGuard {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        let t = tracer();
        t.ctx.complete(
            "vulkan.ort_compute_callback",
            "ep.compute_call",
            self.start,
            self.start.elapsed(),
            Some(
                Args::new()
                    .with(ARG_DEVICE, DEVICE_HOST)
                    .with(ARG_BOUNDARY, BOUNDARY_ORT_COMPUTE_CALLBACK)
                    .with(ARG_TIER, TIER_CALLBACK)
                    .with(ARG_OUTCOME, self.outcome.as_str())
                    .with("subgraph_id", self.subgraph_id)
                    .with("nodes", self.node_count as u64),
            ),
        );
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
        // READBACK IS NOT NESTED, and this line used to assert that it was.
        //
        // Nothing in the engine ever made it true: `dispatch_ort` drops the `Phase::Record`
        // guard, submits, waits on the fence, and only then downloads the outputs. The
        // assertion passed because it was checking the enum against itself — both halves of
        // the claim lived in this file, and neither of them was `vk::session`. It is inverted
        // here so that restoring the false parentage goes red.
        assert_eq!(
            Phase::Readback.nested_in(),
            None,
            "readback happens after the record bracket closes, after submit and after the fence \
             wait; declaring it a child of `record` subtracts it from a bracket that never \
             contained it"
        );
        assert!(
            Phase::Readback.caveat().contains("TOP-LEVEL"),
            "the caveat a reader sees must agree with the structure"
        );
        assert!(
            !Phase::Record.caveat().contains("`readback`, `desc_alloc`"),
            "`record`'s caveat must not list readback among the children it contains"
        );
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

    // ---------------------------------------------------------------------------------------
    // Issue #88 — the two-level attribution vocabulary
    // ---------------------------------------------------------------------------------------

    /// `RecordPath` slots are a published wire position (`counters.rs` mirrors them into the C
    /// ABI), so a reorder of the enum must not silently re-label four counters.
    #[test]
    fn record_path_slots_are_unique_stable_and_agree_with_the_reporting_order() {
        let mut seen = HashSet::new();
        for (i, p) in RecordPath::ALL.iter().enumerate() {
            assert!(seen.insert(p.slot()), "duplicate slot for {p:?}");
            assert_eq!(p.slot(), i, "{p:?} is not at its own reporting position");
            assert!(p.slot() < 4);
        }
        assert_eq!(RecordPath::FirstRecord.slot(), 0);
        assert_eq!(RecordPath::Replay.slot(), 1);
        assert_eq!(RecordPath::Rerecord.slot(), 2);
        assert_eq!(RecordPath::RecordedAgain.slot(), 3);
    }

    /// `RecordedAgain` and `Rerecord` are different facts and must not share a token: one says
    /// the engine has no cache, the other says the benchmark's shapes moved.
    #[test]
    fn recorded_again_is_not_spelled_like_a_shape_driven_rerecord() {
        let tags: Vec<&str> = RecordPath::ALL.iter().map(|p| p.as_str()).collect();
        let unique: HashSet<&&str> = tags.iter().collect();
        assert_eq!(
            unique.len(),
            tags.len(),
            "duplicate record-path token: {tags:?}"
        );
        assert_ne!(
            RecordPath::RecordedAgain.as_str(),
            RecordPath::Rerecord.as_str()
        );
    }

    /// A callback guard that is never told its outcome must not read as a success.
    #[test]
    fn an_unresolved_compute_callback_never_reads_as_a_success() {
        let g = tracer().ort_compute_callback(7, 2);
        assert_eq!(
            g.outcome(),
            ComputeOutcome::Unresolved,
            "the default must fail closed: a panic that unwinds past the resolve point would \
             otherwise label a lost call `ok`"
        );
        assert_eq!(ComputeOutcome::default(), ComputeOutcome::Unresolved);
        drop(g);

        let mut g = tracer().ort_compute_callback(7, 2);
        g.set_outcome(ComputeOutcome::Failed);
        assert_eq!(g.outcome(), ComputeOutcome::Failed);
        assert_eq!(ComputeOutcome::Failed.as_str(), "failed");
        assert_eq!(ComputeOutcome::Ok.as_str(), "ok");
        assert_eq!(ComputeOutcome::Unresolved.as_str(), "unresolved");
    }

    /// The two attribution levels must be distinguishable by a declared property, not by span
    /// name, and the callback tier must not share a token with the dispatch tier.
    #[test]
    fn the_two_attribution_levels_carry_distinct_tier_tokens() {
        let tiers = [TIER_CALLBACK, TIER_DISPATCH, TIER_PHASE];
        let unique: HashSet<&&str> = tiers.iter().collect();
        assert_eq!(unique.len(), 3, "tier tokens must be distinguishable");
        assert_ne!(BOUNDARY_ORT_COMPUTE_CALLBACK, BOUNDARY_ENGINE_DISPATCH);
        // The callback span name must not be a prefix of, or prefixed by, the dispatch span
        // name: an analyser that matches on `startswith` would otherwise fold the levels.
        assert!(!"vulkan.ort_compute_callback".starts_with("vulkan.subgraph"));
        assert!(!"vulkan.subgraph".starts_with("vulkan.ort_compute_callback"));
    }

    /// The phases production emits as spans, and the ones it only folds into the summary.
    ///
    /// This is the structural half of B3: `upload` and `readback` were listed in the module's
    /// span table as emitted spans and are not. An analyser that trusts the table looks for
    /// `vulkan.readback` in the trace, finds nothing, and reports a transfer-free inference.
    #[test]
    fn only_the_phases_production_opens_a_span_for_are_declared_as_emitting() {
        assert!(!Phase::Upload.emits_span());
        assert!(!Phase::Readback.emits_span());
        for p in [
            Phase::Compile,
            Phase::Prepack,
            Phase::Record,
            Phase::DescAlloc,
            Phase::PipelineLookup,
            Phase::CmdUpload,
            Phase::Submit,
            Phase::FenceWait,
        ] {
            assert!(p.emits_span(), "{p:?} is opened by production as a span");
        }
        // Every summary-only phase must say so in its caveat, so the fact survives into the
        // artifact a human reads.
        for p in Phase::ALL {
            if !p.emits_span() {
                assert!(
                    p.caveat().contains("SUMMARY-ONLY"),
                    "{p:?} emits no span but its caveat does not disclose that"
                );
            }
        }
    }

    /// The sibling set is what may be summed. Readback rejoining it is the visible consequence
    /// of B3, and it is asserted here so a silent revert changes a number this test reads.
    #[test]
    fn readback_is_a_top_level_sibling_and_record_no_longer_claims_it() {
        let siblings: Vec<&str> = Phase::ALL
            .iter()
            .filter(|p| p.is_sibling())
            .map(|p| p.as_str())
            .collect();
        assert!(
            siblings.contains(&"readback"),
            "readback partitions the dispatch's wall time with record/submit/fence_wait: {siblings:?}"
        );
        assert!(siblings.contains(&"record"));
        assert!(siblings.contains(&"submit"));
        assert!(siblings.contains(&"fence_wait"));
        assert!(
            !siblings.contains(&"upload"),
            "upload really is inside the record bracket and must stay out of the sibling total"
        );
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
