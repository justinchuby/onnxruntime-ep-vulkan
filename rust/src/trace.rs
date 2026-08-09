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
//! | `vulkan.subgraph` | `ep` | host | One fused subgraph's whole `Compute` call. |
//! | `vulkan.compile` | `ep.phase` | host | `Compile`: plan build, pipeline/SPIR-V creation, descriptor layout. Once per subgraph. |
//! | `vulkan.prepack` | `ep.phase` | host | Weight prepack + upload of block-quantised initializers. Once per `PackKey`. |
//! | `vulkan.record` | `ep.phase` | host | The `Compute` recording bracket. **Despite the name, dominated by the staging upload it contains (~96-98% on Phi-3.5), not by command recording (1-3%).** See `Phase::Record::caveat`. |
//! | `vulkan.upload` | `ep.phase` | host | Host→device staging copy of inference inputs. Carries `bytes`. |
//! | `vulkan.submit` | `ep.phase` | host | **`vkQueueSubmit` only.** Host bookkeeping. Measures no GPU work. |
//! | `vulkan.fence_wait` | `ep.phase` | host | CPU blocked on the fence. Upper bound on GPU time, not GPU time. |
//! | `vulkan.readback` | `ep.phase` | host | Device→host copy of outputs. Carries `bytes`. |
//! | `vulkan.gpu.*` | `gpu` | **device** | GPU execution, from `VkQueryPool` timestamp queries only. Emitted on a separate device lane. |
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
    /// **The whole `Compute` call's host wall time** — the total every other Compute-time phase
    /// is a part of. A [`PhaseRole::Total`], never summed with the siblings it contains.
    ///
    /// Opened at the top of the EP's dispatch entry point and closed on **every** exit from it,
    /// error paths included, so `execute` is the denominator a share may be taken against and the
    /// whole an attribution must close on. Before this phase existed, everything the EP did
    /// before `vkBeginCommandBuffer` — ORT tensor pointer reads, the dynamic-shape pre-pass, plan
    /// and cache lookup, binding preparation, per-call buffer allocation — was outside every
    /// phase, and no row said so.
    Execute,
    /// Plan/cache lookup and shape/binding preparation, before any GPU object is created for this
    /// call: reading ORT's input tensor pointers, resolving dynamic byte sizes, the shape-only
    /// pre-pass that re-runs dynamic translate handlers, and the device-residency resolution that
    /// decides which inputs are already on the device.
    Prepare,
    /// Per-call GPU buffer and staging allocation: device-local input/output/intermediate/temp
    /// buffers, their staging counterparts, and the output-binding decision.
    BufferAlloc,
    /// Host→device staging copy of this inference's inputs.
    Upload,
    /// The `Compute` recording bracket. The name is historical: this span's host wall time is
    /// **dominated by the staging upload nested inside it** (measured 96-98% of the phase on
    /// Phi-3.5), while actual `vkBeginCommandBuffer..vkEndCommandBuffer` recording is 1-3% of wall
    /// (87-229 ms). R11: a measurement's name is not its definition — subtract `upload`+`readback`
    /// before attributing anything here to recording.
    Record,
    /// Sub-phase of `Record`: command-buffer acquisition — `vkResetCommandBuffer` followed by
    /// `vkBeginCommandBuffer`. Emitted once per `Compute` call, inside the `Record` bracket.
    ///
    /// Separated from the rest of `Record` because "the command buffer is reused" and "the
    /// recording is reused" are different claims and this project has conflated them: the buffer
    /// object is allocated once per session, but it is **reset and re-recorded on every call**,
    /// and only a timer around the reset/begin pair can show which of the two a reader is looking
    /// at.
    CmdAlloc,
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
    /// Everything after the fence signals: copying the downloaded staging bytes into ORT's own
    /// output tensors, inserting freshly-uploaded constant inputs into the weight cache, and
    /// releasing this call's buffers.
    ///
    /// A top-level sibling, **not** part of `Record`: it runs after the recording bracket has
    /// closed and after the submission has completed.
    Writeback,
    /// Device→host copy of this inference's outputs.
    Readback,
}

impl Phase {
    /// Stable lowercase tag used in span names, counters and the summary.
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Compile => "compile",
            Phase::Prepack => "prepack",
            Phase::Execute => "execute",
            Phase::Prepare => "prepare",
            Phase::BufferAlloc => "buffer_alloc",
            Phase::Upload => "upload",
            Phase::Record => "record",
            Phase::CmdAlloc => "cmd_alloc",
            Phase::DescAlloc => "desc_alloc",
            Phase::PipelineLookup => "pipeline_lookup",
            Phase::CmdUpload => "cmd_upload",
            Phase::Submit => "submit",
            Phase::FenceWait => "fence_wait",
            Phase::Writeback => "writeback",
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
            Phase::Execute => {
                "TOTAL, not a part: the whole Compute call's host wall time, from EP entry to EP \
                 return on every exit path including errors. It CONTAINS `prepare`, \
                 `buffer_alloc`, `record` (and record's children), `submit`, `fence_wait` and \
                 `writeback`. Never add it to any of them — it is the denominator, and the part \
                 of it no other phase names is printed as UNATTRIBUTED"
            }
            Phase::Prepare => {
                "host: plan/cache lookup and shape/binding preparation before any GPU object is \
                 created for this call — ORT input pointer reads, dynamic byte-size resolution, \
                 the shape-only pre-pass, and the device-residency resolution. Contains NO \
                 Vulkan work and NO transfer. Disjoint from every other Compute-time phase"
            }
            Phase::BufferAlloc => {
                "host: per-call vkCreateBuffer/vkAllocateMemory for device-local and staging \
                 buffers, plus the output-binding decision. Allocation only — the bytes are \
                 moved later, under `cmd_upload`. Disjoint from every other Compute-time phase"
            }
            Phase::Upload => {
                "host: staging copy; on a discrete GPU this is PCIe time and users pay it. \
                 NESTED INSIDE `record` — already counted there, do not add to the sibling total"
            }
            Phase::Record => {
                "host: the whole vkBeginCommandBuffer..vkEndCommandBuffer bracket. It CONTAINS \
                 `upload`/`cmd_upload` (the staging memcpy), `cmd_alloc`, `desc_alloc` and \
                 `pipeline_lookup`, so it is an INCLUSIVE interval and its name describes its \
                 bracket, not its content (R11). It does NOT contain `readback`, which is folded \
                 under `writeback` after the fence. The split is regime-dependent and must be \
                 read from the child rows of THIS run, never from a remembered ratio: with a cold \
                 weight cache `cmd_upload` dominates it, and with a warm cache the children \
                 collapse and the UNNAMED RESIDUAL — the vkCmd* calls themselves — dominates. The \
                 summary prints that residual as its own row, but CUMULATIVELY over all calls, so \
                 it mixes the two regimes; the per-call split is only in the trace spans"
            }
            Phase::CmdAlloc => {
                "host/sub-record: vkResetCommandBuffer + vkBeginCommandBuffer, once per Compute \
                 call. This is command-buffer ACQUISITION, not recording: a non-zero count here \
                 on every call is the witness that the recording is rebuilt per call rather than \
                 replayed. NESTED INSIDE `record` — already counted there, do not add to the \
                 sibling total"
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
            Phase::Writeback => {
                "host: after the fence — copying downloaded staging bytes into ORT's output \
                 tensors, weight-cache insertion, and this call's buffer frees. It CONTAINS \
                 `readback`. Disjoint from `record`: it begins after vkEndCommandBuffer and after \
                 the fence has signalled"
            }
            Phase::Readback => {
                "host: device->host copy; counts toward end-to-end latency. NESTED INSIDE \
                 `writeback` — already counted there, do not add to the sibling total"
            }
        }
    }

    /// The phase whose wall time already contains this one, if any.
    ///
    /// Emitted as the `nested_in` span arg. A phase with `nested_in == Some(p)` must never be
    /// added to a total that also contains `p`.
    pub fn nested_in(self) -> Option<Phase> {
        match self.role() {
            PhaseRole::Child(parent) => Some(parent),
            PhaseRole::Total | PhaseRole::Sibling => None,
        }
    }

    /// What kind of quantity this phase's wall time is. **The single exhaustive declaration** —
    /// [`Self::nested_in`], [`Self::is_sibling`] and [`Self::is_total`] all derive from it.
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
    /// The third state — [`PhaseRole::Total`] — was added with [`Phase::Execute`]. A total is not
    /// a sibling and it is not a child: it is the *whole* the siblings decompose, and the
    /// two-state model had no way to say that. Without it `execute` would have been summed into
    /// SIBLING TOTAL and doubled the host time of every run.
    pub fn role(self) -> PhaseRole {
        match self {
            // The whole Compute call. Contains every Compute-time phase below.
            Phase::Execute => PhaseRole::Total,
            // `vk::session` opens Phase::Record before vkBeginCommandBuffer and drops it after
            // vkEndCommandBuffer; the input staging loop runs inside that bracket. See session.rs
            // (Record guard) — this is a fact about the call graph, not a policy, and it must be
            // re-checked if that guard moves.
            Phase::Upload => PhaseRole::Child(Phase::Record),
            // Switch's per-dispatch sub-phases, added in `692e7d0`, plus `CmdAlloc`. They are
            // documented in their own caveats as "sub-record" and are opened inside the Record
            // guard.
            Phase::CmdAlloc | Phase::DescAlloc | Phase::PipelineLookup | Phase::CmdUpload => {
                PhaseRole::Child(Phase::Record)
            }
            // CORRECTED 2026-08-08 (issue #88). `Readback` was declared a child of `Record`, and
            // it is not one: `record_transfer(Transfer::Readback, ..)` is called in the EP's
            // Step 5, *after* the Record guard is dropped and after the fence has signalled. The
            // declaration therefore removed readback from the sibling total and charged it to a
            // bracket it does not lie inside — `bench/phases.py` never listed it under
            // `PHASE_CHILDREN["record"]`, so the two sides of the same fact disagreed. Its real
            // parent is the post-fence `Writeback` bracket, which now exists to hold it.
            Phase::Readback => PhaseRole::Child(Phase::Writeback),
            // EXHAUSTIVE ON PURPOSE — do not add a `_` arm. A catch-all here classifies every
            // future phase as a top-level sibling by default, which means a new sub-phase gets
            // silently added into SIBLING TOTAL and double-counts its parent. That is how three
            // phases arrived in one session: they merged cleanly and would have been summed.
            // Make the compiler ask.
            Phase::Compile
            | Phase::Prepack
            | Phase::Prepare
            | Phase::BufferAlloc
            | Phase::Record
            | Phase::Submit
            | Phase::FenceWait
            | Phase::Writeback => PhaseRole::Sibling,
        }
    }

    /// Phases that are top-level parts: their wall times may be summed.
    ///
    /// False for [`Phase::Execute`], which is the *whole* those parts decompose, and false for
    /// every child.
    pub fn is_sibling(self) -> bool {
        matches!(self.role(), PhaseRole::Sibling)
    }

    /// Whether this phase is a whole rather than a part — the denominator, never an addend.
    pub fn is_total(self) -> bool {
        matches!(self.role(), PhaseRole::Total)
    }

    /// Whether this phase happens **inside one `Compute` call**.
    ///
    /// `Compile` and `Prepack` are session-build phases: they are real host time and they are
    /// genuinely siblings, but they are not part of any `Compute` call, so adding them to a
    /// decomposition of [`Phase::Execute`] would over-subscribe the total. This is the predicate
    /// that keeps the attribution honest about which whole it is closing on.
    ///
    /// Exhaustive for the same reason [`Self::role`] is.
    pub fn in_compute(self) -> bool {
        match self {
            Phase::Compile | Phase::Prepack => false,
            Phase::Execute
            | Phase::Prepare
            | Phase::BufferAlloc
            | Phase::Upload
            | Phase::Record
            | Phase::CmdAlloc
            | Phase::DescAlloc
            | Phase::PipelineLookup
            | Phase::CmdUpload
            | Phase::Submit
            | Phase::FenceWait
            | Phase::Writeback
            | Phase::Readback => true,
        }
    }

    /// Whether [`VulkanTracer::phase`] may open a `ph:"X"` span for this phase.
    ///
    /// `false` for the three phases whose duration is folded into the summary from a caller that
    /// already owns the clock:
    ///
    /// * [`Phase::Execute`] — its bracket is already drawn on the timeline as `vulkan.subgraph`.
    ///   Emitting a second, coincident `vulkan.execute` span would give every trace aggregator
    ///   two spans covering the same microseconds under different names, which is the exact
    ///   double-count this module exists to prevent.
    /// * [`Phase::Upload`] / [`Phase::Readback`] — reported through
    ///   [`VulkanTracer::record_transfer`], which emits byte and bandwidth *counters* rather than
    ///   a span, because the same memcpy is already bracketed by `cmd_upload` / `writeback`.
    pub fn emits_span(self) -> bool {
        !matches!(self, Phase::Execute | Phase::Upload | Phase::Readback)
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
    pub const ALL: [Phase; 15] = [
        Phase::Compile,
        Phase::Prepack,
        Phase::Execute,
        Phase::Prepare,
        Phase::BufferAlloc,
        Phase::Upload,
        Phase::Record,
        Phase::CmdAlloc,
        Phase::DescAlloc,
        Phase::PipelineLookup,
        Phase::CmdUpload,
        Phase::Submit,
        Phase::FenceWait,
        Phase::Writeback,
        Phase::Readback,
    ];

    /// The top-level parts of one `Compute` call, in the order they run.
    ///
    /// This is the **only** set that may be summed and compared against [`Phase::Execute`]. It is
    /// derived, not restated: a new phase joins it by being a `Sibling` that is `in_compute`, and
    /// a phase that stops being either leaves it, without anyone editing a list.
    pub fn compute_siblings() -> impl Iterator<Item = Phase> {
        Phase::ALL
            .into_iter()
            .filter(|p| p.is_sibling() && p.in_compute())
    }
}

/// What kind of quantity a [`Phase`]'s wall time is. See [`Phase::role`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PhaseRole {
    /// A whole that other phases decompose. Never summed with anything; used as a denominator.
    Total,
    /// A top-level part. Siblings within the same scope are disjoint and may be summed.
    Sibling,
    /// A part of the named parent's interval. Never added to a total that contains the parent.
    Child(Phase),
}

/// Which recording path one `Compute` call took — the Vulkan analogue of MLX's compile-cache
/// state, over `ENGINE.md` §6.1's record-once / replay-many model.
///
/// # What this build actually does, stated because the vocabulary predates it
///
/// The vocabulary was written for a design in which a `VkCommandBuffer` is recorded once and
/// replayed. **This build does not do that.** `VulkanSession::dispatch_ort` calls
/// `CommandPool::begin` — `vkResetCommandBuffer` + `vkBeginCommandBuffer` — unconditionally on
/// every `Compute` call and re-records every dispatch. So [`RecordPath::Replay`] is a state this
/// EP cannot currently reach, and the production call site never reports it. That is the finding,
/// not a gap in the instrument: the aspiration is in the vocabulary and the measurement is what
/// says whether it was met.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RecordPath {
    /// First recording of this subgraph's command buffer.
    FirstRecord,
    /// Replayed the cached `VkCommandBuffer` — the steady-state path. **Unreachable in this
    /// build**; see the enum docs.
    Replay,
    /// The command buffer was recorded again for a subgraph that had already been recorded.
    ///
    /// Originally documented as "re-recorded because the input shape key changed". It has two
    /// causes now, and only a caller can tell them apart: a shape-key change (reported by passing
    /// [`RecordPath::Replay`] with a key this subgraph has not presented), or — the case this EP
    /// is in — a build with no command-buffer cache at all, which re-records unconditionally.
    /// Either way, a benchmark that reports a median over runs where this fired is measuring the
    /// recording path, not a steady state.
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

/// How much of the `Compute` wall the phase table can name, and — the part that matters — how
/// much it cannot.
///
/// # The whole and the parts
///
/// The whole is [`Phase::Execute`]: one bracket around the EP's dispatch entry point, closed on
/// every exit path. The parts are [`Phase::compute_siblings`] — the top-level phases that happen
/// inside a `Compute` call. They are disjoint by construction (each guard is dropped before the
/// next is opened; see `vk/session.rs`), so summing them is legitimate and comparing that sum to
/// the whole is a real identity rather than a tautology: numerator and denominator are read from
/// different `Instant`s at different points in the call.
///
/// # Why the residual is the headline and not a footnote
///
/// R11: a decomposition that appears to close is the hardest kind of wrong. Every phase this EP
/// has ever emitted was a *part*, so a reader summing them had no way to know what fraction of
/// the call they covered — and the answer, before issue #88, was "everything after
/// `vkBeginCommandBuffer`", which on a decode step is not most of it. [`Self::unattributed_us`]
/// is that gap, computed rather than assumed, and it is printed even when it is zero.
///
/// # What this is NOT
///
/// * It is **not** a claim that the parts are causally exclusive. They are wall-clock disjoint on
///   the calling thread. Asynchronous device work started in one part and completed in another
///   moves cost between them, and no host clock can undo that — which is exactly why `submit` is
///   labelled host-only and `fence_wait` an upper bound.
/// * It is **not** per-call. Every field is cumulative over the process, so a session mixing one
///   cold call with many warm ones reports a mixture. The per-call split is only in the spans.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct HostAttribution {
    /// Cumulative [`Phase::Execute`] wall, in microseconds. The denominator.
    pub execute_us: u64,
    /// Number of `Compute` calls the total was measured over.
    pub execute_calls: u64,
    /// Sum of the top-level in-`Compute` phases.
    pub attributed_us: u64,
    /// `execute_us - attributed_us`, floored at zero. Host time inside `Compute` that no phase
    /// names.
    pub unattributed_us: u64,
    /// The parts summed to more than the whole. Impossible if the guards are disjoint and the
    /// total encloses them, so it is a defect in the instrumentation — reported, never smoothed.
    pub over_subscribed: bool,
    /// In-`Compute` sibling phases that recorded **no calls at all** while `Compute` calls
    /// happened. A phase that silently stops being invoked would otherwise shrink the attributed
    /// total and grow the residual with nothing raising.
    pub silent_phases: Vec<Phase>,
    /// Whether this attribution may be quoted as a decomposition of `Compute`.
    ///
    /// False when there is no total to divide by, when a part is missing, or when the parts
    /// over-subscribe the whole. A false verdict does not suppress the numbers — it forbids
    /// reading them as complete.
    pub admissible: bool,
}

impl HostAttribution {
    /// Derive the attribution from a phase table of `phase -> (total_us, calls)`.
    ///
    /// A free-standing function of its input so the arithmetic is testable without a tracer, a
    /// device, or an environment variable.
    pub fn from_phase_table(table: &BTreeMap<Phase, (u64, u64)>) -> Self {
        let get = |p: Phase| table.get(&p).copied().unwrap_or((0, 0));
        let (execute_us, execute_calls) = get(Phase::Execute);
        let attributed_us: u64 = Phase::compute_siblings().map(|p| get(p).0).sum();
        let silent_phases: Vec<Phase> = if execute_calls > 0 {
            Phase::compute_siblings()
                .filter(|&p| get(p).1 == 0)
                .collect()
        } else {
            Vec::new()
        };
        let over_subscribed = attributed_us > execute_us;
        HostAttribution {
            execute_us,
            execute_calls,
            attributed_us,
            unattributed_us: execute_us.saturating_sub(attributed_us),
            over_subscribed,
            admissible: execute_calls > 0
                && execute_us > 0
                && !over_subscribed
                && silent_phases.is_empty(),
            silent_phases,
        }
    }

    /// The share of `Compute` wall no phase names, in percent. `None` when there is no whole to
    /// take a share of — absence of a denominator is not a share of zero.
    pub fn unattributed_pct(&self) -> Option<f64> {
        (self.execute_us > 0).then(|| 100.0 * self.unattributed_us as f64 / self.execute_us as f64)
    }

    /// One line naming why the attribution may not be read as complete, or `None` when it may.
    pub fn refusal(&self) -> Option<String> {
        if self.execute_calls == 0 || self.execute_us == 0 {
            return Some(
                "NO WHOLE: Phase::Execute recorded no duration, so there is no denominator and \
                 no share below is a share of anything. This is the absence of a measurement, \
                 not a measurement of zero."
                    .to_string(),
            );
        }
        if self.over_subscribed {
            return Some(format!(
                "OVER-SUBSCRIBED: the top-level parts sum to {} us inside a {} us whole. The \
                 parts are supposed to be wall-clock disjoint and enclosed by the total, so this \
                 is a defect in the instrumentation and the decomposition below is not usable.",
                self.attributed_us, self.execute_us
            ));
        }
        if !self.silent_phases.is_empty() {
            return Some(format!(
                "INCOMPLETE: {} Compute call(s) ran and these top-level phases recorded none: \
                 {}. Their cost is inside UNATTRIBUTED under a name that is not theirs, so the \
                 breakdown must not be quoted as a decomposition.",
                self.execute_calls,
                self.silent_phases
                    .iter()
                    .map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        None
    }
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
    /// Shape keys already seen, per subgraph id, so a REPLAY on a new key is called what it is.
    seen_shape_keys: HashMap<u64, HashSet<String>>,

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

    /// Span around one fused subgraph's whole `Compute` call.
    ///
    /// Host wall time. On a Vulkan EP this covers upload, record-or-replay, submit, fence wait
    /// and readback — i.e. it is the end-to-end latency of the subgraph *as the caller
    /// experiences it*, which is the number a user pays, and which is deliberately not called
    /// "GPU time" anywhere.
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

    /// Bracket the whole `Compute` call: the denominator every phase share is a share of.
    ///
    /// Returns `None` (a single relaxed atomic load, no clock read, no allocation) when nothing
    /// is listening. Hold it for the entire EP dispatch entry point — including every early error
    /// return — so the total encloses the parts on every path, not just the happy one. A total
    /// that only covers successful calls would make a failing run look faster than it was.
    ///
    /// Deliberately emits no span of its own: `vulkan.subgraph` from [`Self::subgraph_region`]
    /// already brackets the same microseconds, and a second coincident span under a different
    /// name would let an aggregator count the call twice.
    #[inline]
    pub fn execute_region(&self) -> Option<ExecuteGuard> {
        if !self.active() {
            return None;
        }
        Some(ExecuteGuard {
            start: Instant::now(),
        })
    }

    /// Start a timing phase: a span plus a summary fold on drop. `None` (zero cost) when nothing
    /// is listening.
    ///
    /// Only for phases where [`Phase::emits_span`] is true. A phase whose duration is owned by a
    /// caller with its own clock ([`Phase::Execute`], [`Phase::Upload`], [`Phase::Readback`]) is
    /// folded through [`Self::record_phase`] or [`Self::record_transfer`] instead; opening a span
    /// for it here would draw two coincident spans over the same microseconds.
    #[inline]
    pub fn phase(&self, phase: Phase) -> Option<PhaseGuard> {
        debug_assert!(
            phase.emits_span(),
            "Phase::{phase:?} does not emit its own span — fold it through record_phase/\
             record_transfer instead of opening a coincident second span"
        );
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
    /// dynamic shapes, or when the caller re-records unconditionally and the key is therefore not
    /// consulted by anything.
    ///
    /// `subgraph_id` is the EP's own subgraph identifier, **not** a formatted string: the caller
    /// is on the per-inference path and must not allocate to name a subgraph the tracer will
    /// discard when tracing is off.
    pub fn record_path(
        &self,
        subgraph_id: u64,
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
                        .get(&subgraph_id)
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
                    .entry(subgraph_id)
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
                .with("subgraph", subgraph_id);
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
        // A session that ran Compute must print, even if GetCapability was never observed and the
        // record-path witness was somehow not reached: the "the witness vanished" branch below is
        // the only thing that can report that failure, and bailing out here would suppress it.
        if s.getcap_calls == 0 && s.record_paths.iter().all(|&n| n == 0) && s.phase_us.is_empty() {
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
        // `Tracer::record_path` was wired to a production caller in issue #88 (`vk/session.rs`
        // classifies every Compute call before it touches the command pool). Before that it had
        // none, and this line printed "NOT WIRED" — which was the honest thing to print, because
        // "first-record=0 replay=0 rerecord=0" reads as "this run recorded nothing" and that is a
        // measurement, not the absence of one.
        //
        // The failure mode has now inverted. With a caller in place, all-zeros while Compute
        // calls happened means the witness was REMOVED — the call site deleted, refactored away,
        // or moved behind a branch that never runs — and the summary must say so instead of
        // reverting to a "not wired yet" story that is no longer true.
        if s.record_paths.iter().all(|&n| n == 0) {
            let execute_calls = s
                .phase_us
                .get(&Phase::Execute)
                .map(|&(_, calls)| calls)
                .unwrap_or(0);
            if execute_calls > 0 {
                out.push_str(&format!(
                    "  compute:  INSTRUMENT DEFECT — {execute_calls} Compute call(s) ran and the \
                     record-path witness classified none of them. `Tracer::record_path()` has a \
                     production caller (vk/session.rs); all-zeros here means it stopped being \
                     reached, so nothing in this run can tell you whether command buffers were \
                     re-recorded or replayed.\n"
                ));
            } else {
                out.push_str(
                    "  compute:  no Compute call in this process — record-path is unmeasured, \
                     NOT zero.\n",
                );
            }
        } else {
            out.push_str(&format!(
                "  compute:  first-record={} replay={} rerecord={} (over {} distinct subgraph(s))\n",
                s.record_paths[0],
                s.record_paths[1],
                s.record_paths[2],
                s.seen_shape_keys.len()
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
            // Three roles, not two (see `Phase::role`). `execute` is the WHOLE: one bracket
            // around the entire Compute call. `prepare`/`buffer_alloc`/`record`/`submit`/
            // `fence_wait`/`writeback` are SIBLINGS that partition it. `upload`, `cmd_alloc`,
            // `desc_alloc`, `pipeline_lookup`, `cmd_upload` and `readback` are CHILDREN of a
            // sibling. Summing the column double-counts the children AND counts the whole twice.
            //
            // Children are printed indented under their parent with an explicit marker, and the
            // sibling total is computed and printed so nobody has to add the column by hand.
            // Both the child set and the sibling total are DERIVED from `Phase::role()` rather
            // than restated here. They used to be two hardcoded lists, which is the same
            // duplicate-truth defect one level down: changing the bracketing in `vk::session`
            // would have had to be remembered in three places, and the one that gets forgotten is
            // the one that prints.
            let get = |p: Phase| s.phase_us.get(&p).copied().unwrap_or((0, 0));
            // TRANSFER IS NAMED, NOT INFERRED FROM "IS A CHILD". `desc_alloc` and
            // `pipeline_lookup` are also children and are not transfer; summing every child under
            // the label "xfer" would be the same misnaming one level down.
            let child_us: u64 = get(Phase::Upload).0 + get(Phase::Readback).0;
            let attribution = HostAttribution::from_phase_table(&s.phase_us);
            // `upload` (record_transfer totals) and `cmd_upload` (Switch's per-Compute sub-span)
            // can bracket the same memcpy, so the nested rows are NOT disjoint and are never
            // added together here.
            let overlap = get(Phase::CmdUpload).1 > 0 && get(Phase::Upload).1 > 0;

            out.push_str(
                "  host time (wall clock on the CPU thread). `execute` is the WHOLE Compute \
                 call; indented rows are NESTED inside the row above them — do NOT add this \
                 column:\n",
            );
            for phase in Phase::ALL {
                let Some((us, calls)) = s.phase_us.get(&phase) else {
                    continue;
                };
                let marker = if phase.is_total() {
                    "  = "
                } else if phase.is_sibling() {
                    "    "
                } else {
                    "  └─"
                };
                out.push_str(&format!(
                    "            {}{:<15} {:>10} us (x{}) — {}\n",
                    marker,
                    phase.as_str(),
                    us,
                    calls,
                    phase.caveat()
                ));
            }
            // THE UNNAMED PART OF THE COMPUTE CALL GETS A ROW, AND A VERDICT (R11).
            //
            // This is the point of issue #88. Every phase this EP emitted before it was a PART;
            // there was no whole, so a reader summing the siblings could not tell whether they
            // covered 95% of the call or 40% of it. `execute` supplies the denominator and
            // UNATTRIBUTED is the subtraction, printed even when it is zero, with an explicit
            // admissibility verdict so a partial trace cannot be quoted as a decomposition.
            out.push_str(&format!(
                "            {:<17} {:>10} us  (siblings only: {}; excludes the nested rows)\n",
                "SIBLING TOTAL",
                attribution.attributed_us,
                Phase::compute_siblings()
                    .map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join("+")
            ));
            match attribution.unattributed_pct() {
                Some(pct) => out.push_str(&format!(
                    "            {:<17} {:>10} us  ({pct:.1}% of `execute`) — host time inside \
                     Compute that NO phase names. CUMULATIVE over {} call(s), so a session \
                     mixing one cold call with many warm ones reports a MIXTURE and this share \
                     belongs to no single call; for the per-call split read the spans in the \
                     trace JSON\n",
                    "UNATTRIBUTED", attribution.unattributed_us, attribution.execute_calls
                )),
                None => out.push_str(&format!(
                    "            {:<17} {:>10} us  (no `execute` total — share undefined)\n",
                    "UNATTRIBUTED", attribution.unattributed_us
                )),
            }
            match attribution.refusal() {
                Some(reason) => out.push_str(&format!(
                    "            ATTRIBUTION: NOT ADMISSIBLE — {reason}\n"
                )),
                None => out.push_str(
                    "            ATTRIBUTION: admissible — every top-level phase was invoked and \
                     the parts fit inside the whole. Still host wall clock only: async device \
                     work started in one phase and completed in another moves cost between them, \
                     and no host clock can undo that.\n",
                ),
            }
            out.push_str(&format!(
                "            {:<17} {:>10} us  ({:.1}% of sibling total) — upload+readback only, \
                 already counted inside their parents\n",
                "of which xfer",
                child_us,
                100.0 * child_us as f64 / (attribution.attributed_us.max(1)) as f64,
            ));
            if overlap {
                out.push_str(
                    "            NOTE: the nested rows are NOT disjoint — `upload` and \
                     `cmd_upload` can bracket the same memcpy. Never add the nested rows to each \
                     other; each is only comparable to its parent.\n",
                );
            }
            // THE UNNAMED PART OF `record` GETS A ROW TOO (R11, one level down).
            //
            // Every child of `record` is named and printed above, which makes that decomposition
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
                    get(Phase::Upload).0.max(get(Phase::CmdUpload).0),
                    get(Phase::DescAlloc).0 + get(Phase::CmdAlloc).0,
                    get(Phase::PipelineLookup).0,
                );
                out.push_str(&format!(
                    "            {:<17} {:>10} us  ({:.1}% of `record`) — `record` minus every \
                     named child: the vkCmd* calls themselves, the ONLY part of `record` that is \
                     command-buffer recording. CUMULATIVE over every Compute call, so a session \
                     with one cold call and many warm ones reports a MIXTURE of the two regimes \
                     and this share belongs to no single call\n",
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
            let attribution = HostAttribution::from_phase_table(&s.phase_us);
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
                // The record-path witness is wired in production (issue #88). Carrying the
                // wiring state as a fact rather than letting a consumer infer it from three
                // zeros, which is ambiguous between "nothing ran" and "the witness vanished".
                .with(
                    "record_path_wired",
                    s.record_paths.iter().any(|&n| n > 0) || attribution.execute_calls == 0,
                )
                // Host attribution (issue #88). `execute_us` is the WHOLE Compute wall;
                // `attributed_us` is the sum of the top-level phases inside it; the difference is
                // host cost no phase names. `attribution_admissible` is false whenever the
                // breakdown must not be read as a decomposition — a refused or partial trace
                // therefore cannot be quoted as complete.
                .with("execute_us", attribution.execute_us)
                .with("execute_calls", attribution.execute_calls)
                .with("attributed_us", attribution.attributed_us)
                .with("unattributed_us", attribution.unattributed_us)
                .with("attribution_admissible", attribution.admissible)
                .with(
                    "attribution_refusal",
                    attribution.refusal().unwrap_or_default(),
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

/// RAII bracket for [`Phase::Execute`] — the whole `Compute` call. See
/// [`VulkanTracer::execute_region`].
pub struct ExecuteGuard {
    start: Instant,
}

impl Drop for ExecuteGuard {
    fn drop(&mut self) {
        tracer().record_phase(Phase::Execute, self.start.elapsed());
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
        assert!(!Phase::CmdAlloc.is_sibling());
        assert!(
            Phase::CmdUpload.caveat().contains("OVERLAPS `upload`"),
            "cmd_upload and upload bracket the same memcpy; the artifact must say so"
        );
    }

    #[test]
    fn nested_phases_are_declared_both_structurally_and_in_their_caveat() {
        assert_eq!(Phase::Upload.nested_in(), Some(Phase::Record));
        // CORRECTED IN ISSUE #88. `readback` used to be modelled as a child of `record`, and it
        // never was: `record_transfer(Transfer::Readback, ..)` fires in Step 5, after the record
        // guard is dropped AND after the fence. Under the old model its microseconds were
        // subtracted from `record`'s residual, which inflated the reported command-recording
        // share of a phase that had already ended.
        assert_eq!(Phase::Readback.nested_in(), Some(Phase::Writeback));
        for p in Phase::ALL {
            match p.role() {
                PhaseRole::Child(parent) => {
                    assert!(
                        p.caveat().contains("NESTED INSIDE"),
                        "{p:?} is nested in {parent:?} but its caveat does not say so — an \
                         aggregator that reads prose will double-count it"
                    );
                    assert!(
                        matches!(parent.role(), PhaseRole::Sibling),
                        "only one level of nesting under a sibling is modelled; {parent:?} is \
                         {:?}",
                        parent.role()
                    );
                    assert!(!p.is_sibling());
                    assert!(!p.is_total());
                }
                PhaseRole::Sibling => {
                    assert!(p.is_sibling());
                    assert!(p.nested_in().is_none());
                    assert!(!p.is_total());
                }
                PhaseRole::Total => {
                    // A total is neither summable nor nested. If it were classified as a sibling
                    // it would be added into SIBLING TOTAL and double every host number.
                    assert!(
                        !p.is_sibling(),
                        "{p:?} is the whole and must never be summed"
                    );
                    assert!(p.nested_in().is_none());
                    assert!(p.is_total());
                }
            }
        }
        // The parent must warn that it CONTAINS its children, not just the reverse: whoever reads
        // the 68% row reads the parent's caveat, not the child's.
        assert!(
            Phase::Record.caveat().contains("CONTAINS"),
            "the phase that swallowed a memcpy must say what it swallowed"
        );
        assert!(
            Phase::Writeback.caveat().contains("CONTAINS"),
            "readback's real parent must name what it contains"
        );
        assert!(
            Phase::Record
                .caveat()
                .contains("does NOT contain `readback`"),
            "`record` no longer contains readback (issue #88) and must say so explicitly; a \
             caveat silent on the correction would send a reader subtracting post-fence memcpy \
             time from a phase that had already ended"
        );
        assert!(
            !Phase::Record.caveat().contains("amortised across replays"),
            "this claim was falsified by measurement and by an unwired record-path counter"
        );
    }

    /// Exactly one phase may be the whole, and it must be the one the session actually brackets.
    ///
    /// Two totals would mean two denominators and no way to tell which share is which; zero
    /// totals is the pre-#88 state, where every phase was a part and nothing said what of.
    #[test]
    fn there_is_exactly_one_total_and_it_is_execute() {
        let totals: Vec<Phase> = Phase::ALL.into_iter().filter(|p| p.is_total()).collect();
        assert_eq!(
            totals,
            vec![Phase::Execute],
            "the Compute call has exactly one whole"
        );
    }

    /// The sibling set that gets summed must be exactly the in-Compute top-level phases —
    /// no total (double-counts the call), no child (double-counts its parent), and nothing
    /// that happens outside Compute (`compile`/`prepack` are session-setup, and folding them
    /// into a per-inference denominator would make the first inference look free).
    #[test]
    fn compute_siblings_excludes_the_total_the_children_and_the_setup_phases() {
        let sibs: Vec<Phase> = Phase::compute_siblings().collect();
        for p in &sibs {
            assert!(
                matches!(p.role(), PhaseRole::Sibling),
                "{p:?} is not a sibling"
            );
            assert!(
                p.in_compute(),
                "{p:?} does not happen inside a Compute call"
            );
        }
        assert!(!sibs.contains(&Phase::Execute), "the whole is not a part");
        assert!(!sibs.contains(&Phase::Compile), "compile is session setup");
        assert!(!sibs.contains(&Phase::Prepack), "prepack is session setup");
        assert!(!sibs.contains(&Phase::Readback), "readback is a child");
        // The set the production path actually opens, in order. If a guard is added to
        // `vk::session` without appearing here — or removed from here without the guard going —
        // this list is the thing that has to change, so the drift is visible in a diff.
        assert_eq!(
            sibs,
            vec![
                Phase::Prepare,
                Phase::BufferAlloc,
                Phase::Record,
                Phase::Submit,
                Phase::FenceWait,
                Phase::Writeback,
            ]
        );
    }

    /// A phase whose duration is owned by another clock must not also draw its own span.
    ///
    /// `execute` shares its microseconds with `vulkan.subgraph`; `upload`/`readback` are folded
    /// in through `record_transfer` from inside a parent span. Any of them emitting a second
    /// coincident `ph:"X"` span would let an aggregator count the same time twice under two
    /// different names, which is the exact failure `nested_in` was added to prevent one level up.
    #[test]
    fn phases_owned_by_another_clock_emit_no_span_of_their_own() {
        assert!(!Phase::Execute.emits_span());
        assert!(!Phase::Upload.emits_span());
        assert!(!Phase::Readback.emits_span());
        for p in Phase::ALL {
            if !p.emits_span() {
                continue;
            }
            assert!(
                matches!(p.role(), PhaseRole::Sibling | PhaseRole::Child(_)),
                "{p:?} emits a span but is the whole"
            );
        }
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

    // ── Host attribution (issue #88) ────────────────────────────────────────────────────────
    //
    // The arithmetic is deliberately testable without a device, an environment variable, or a
    // tracer: `HostAttribution::from_phase_table` is a pure function of the phase table, so
    // every failure mode below is a real behavioural assertion and not a restatement of the
    // source.

    /// A table built the way a healthy warm decode step builds one.
    fn full_table(execute_us: u64, part_us: u64) -> BTreeMap<Phase, (u64, u64)> {
        let mut t = BTreeMap::new();
        t.insert(Phase::Execute, (execute_us, 10));
        for p in Phase::compute_siblings() {
            t.insert(p, (part_us, 10));
        }
        t
    }

    #[test]
    fn the_unattributed_remainder_is_the_whole_minus_the_named_parts() {
        // 6 siblings x 100 us = 600 us named inside a 1000 us call.
        let a = HostAttribution::from_phase_table(&full_table(1000, 100));
        assert_eq!(a.execute_us, 1000);
        assert_eq!(a.attributed_us, 600);
        assert_eq!(a.unattributed_us, 400);
        assert_eq!(a.unattributed_pct(), Some(40.0));
        assert!(a.admissible, "{:?}", a.refusal());
        assert_eq!(a.refusal(), None);

        // R10: the value must VARY with its input. A residual that reads the same for a call
        // whose parts cover 95% as for one where they cover 60% is a constant in a
        // measurement's clothes.
        let tight = HostAttribution::from_phase_table(&full_table(1000, 160));
        assert_eq!(tight.unattributed_us, 40);
        assert_ne!(tight.unattributed_us, a.unattributed_us);
    }

    /// The failure this whole model exists to make impossible: a phase stops being invoked, its
    /// cost silently moves into the residual, and the table still reads as a decomposition.
    #[test]
    fn a_phase_that_stops_being_invoked_makes_the_attribution_inadmissible() {
        for missing in Phase::compute_siblings() {
            let mut t = full_table(1000, 100);
            t.remove(&missing);
            let a = HostAttribution::from_phase_table(&t);
            assert!(
                !a.admissible,
                "{missing:?} disappeared and the breakdown still claimed to be complete"
            );
            assert_eq!(a.silent_phases, vec![missing]);
            let refusal = a
                .refusal()
                .expect("an inadmissible attribution must say why");
            assert!(
                refusal.contains(missing.as_str()),
                "the refusal must NAME the phase that vanished, or a reader cannot tell which \
                 number moved: {refusal}"
            );
            // The numbers are still reported — refusing to interpret is not refusing to show.
            assert_eq!(a.execute_us, 1000);
            assert_eq!(a.unattributed_us, 500);
        }
    }

    /// A phase present in the table but with zero calls is the same defect wearing a nicer hat:
    /// the key exists, the sum reads plausible, and nothing ran.
    #[test]
    fn a_phase_present_with_zero_calls_is_still_silent() {
        let mut t = full_table(1000, 100);
        t.insert(Phase::Submit, (0, 0));
        let a = HostAttribution::from_phase_table(&t);
        assert!(!a.admissible);
        assert_eq!(a.silent_phases, vec![Phase::Submit]);
    }

    /// The parts cannot exceed the whole. If they do, the guards overlap or the total does not
    /// enclose them — an instrumentation defect, and the one thing a plausible-looking
    /// percentage would hide most effectively.
    #[test]
    fn parts_larger_than_the_whole_are_refused_not_clamped() {
        let a = HostAttribution::from_phase_table(&full_table(500, 100));
        assert!(a.over_subscribed);
        assert!(!a.admissible);
        assert_eq!(
            a.unattributed_us, 0,
            "saturating, so the residual never goes negative — but the flag is what makes the \
             zero readable as a defect instead of as perfect coverage"
        );
        assert!(a.refusal().unwrap().contains("OVER-SUBSCRIBED"));
    }

    /// No Compute call means no denominator. The share must be `None`, not 0.0: a run that
    /// measured nothing and a run that measured 0% unattributed are opposite findings.
    #[test]
    fn an_empty_run_has_no_share_rather_than_a_zero_share() {
        let a = HostAttribution::from_phase_table(&BTreeMap::new());
        assert_eq!(a.execute_calls, 0);
        assert_eq!(a.unattributed_pct(), None);
        assert!(!a.admissible);
        assert!(a.refusal().unwrap().contains("NO WHOLE"));
        assert!(
            a.silent_phases.is_empty(),
            "with no Compute call, a phase that did not fire is not evidence of anything"
        );
    }

    /// The summary must not be able to print a complete-looking attribution for a partial trace.
    /// This exercises the rendered text, not the struct, because the text is what a reader
    /// quotes.
    #[test]
    fn the_rendered_summary_refuses_a_partial_attribution_in_words() {
        let full = HostAttribution::from_phase_table(&full_table(1000, 100));
        assert!(full.refusal().is_none());

        let mut partial_tbl = full_table(1000, 100);
        partial_tbl.remove(&Phase::FenceWait);
        let partial = HostAttribution::from_phase_table(&partial_tbl);
        let refusal = partial.refusal().expect("must refuse");
        assert!(refusal.starts_with("INCOMPLETE"));
        assert!(
            refusal.contains("fence_wait"),
            "must name the missing phase: {refusal}"
        );
    }
}
