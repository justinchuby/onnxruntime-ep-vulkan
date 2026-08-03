//! Always-on execution counters for the ORT boundary — the evidence behind M0 criterion 8.
//!
//! # Why this module exists
//!
//! Morpheus's reworded criterion 8 requires **a non-zero executed-dispatch count per lane,
//! reported**. The word *reported* is doing the work. Until now a CI lane could:
//!
//! * skip every test (no Vulkan device, an `xfail`, a collection error) and exit 0;
//! * claim nothing, decline every node honestly, run the whole graph on CPU, and exit 0;
//! * load the EP, compile a subgraph, and never reach `Compute`, and exit 0.
//!
//! All three are green, and none of them has executed a single shader. That is a false green, and
//! a false green is the failure mode that let four consecutive red runs go unnoticed: it sends you
//! looking somewhere else. The fix is not another assertion inside a test — a lane where the tests
//! never ran cannot be fixed by an assertion inside a test. It has to be a count that the lane
//! *produces* and a gate that reads it from outside.
//!
//! # What a "dispatch" means here, exactly
//!
//! `dispatches_executed` is incremented **after `dispatch_ort` returns success**, by the number of
//! `CompiledKernel`s in the subgraph — one per `vkCmdDispatch` recorded. `dispatch_ort` submits and
//! then waits on a fence before returning, so a success return means the GPU ran that command
//! buffer to completion. It is therefore a count of dispatches that *executed*, not of dispatches
//! that were *recorded* or *submitted*.
//!
//! It is deliberately **not** a count of anything the shader computed. A dispatch that executed and
//! produced wrong numbers still counts here. This module answers "did our code run on a device",
//! which is a strictly weaker claim than "the answer is right" — that one belongs to the
//! differential test against the CPU EP. Keeping the two claims separate is the whole point; the
//! project has already shipped two fabricated speedups by letting a weak claim stand in for a
//! strong one.
//!
//! # Why not `trace.rs`
//!
//! `trace.rs` is Niobe's, and it is *off unless asked for* — a tracer you have to enable cannot be
//! the thing that proves a lane did work, because the lane that forgets to enable it looks exactly
//! like the lane that executed nothing. These counters are unconditional: eight relaxed atomic
//! adds on a path that has already submitted a command buffer and waited on a fence.
//!
//! # How CI reads them
//!
//! Two ways, both from outside the test suite:
//!
//! 1. **In-process, live** — `OrtEpVulkanGetExecutionCounters` is exported from the cdylib.
//!    Python can `ctypes.CDLL(path)` the already-loaded library and read the counters at any
//!    point, including per-test.
//! 2. **Out-of-process, after the fact** — set `ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` and the EP
//!    writes a JSON snapshot to it, then `epctl --check-counters <file>` gates on it.
//!
//! The second is what a lane gate wants, and it has a property worth stating: if the process dies,
//! the file is either missing or reports the counts as of the last successful dispatch. A missing
//! file is exit code 3 — "the lane did not report" — which is distinct from exit 1, "the lane
//! reported zero". A crashed lane must not be able to look like a passing one, and a lane that
//! executed nothing must not be able to look like a lane that was never asked to.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Bumped when a field is **added**. Fields are never removed or reordered, so a reader that
/// knows version *n* can read the first *n* generations' worth of a version *n+k* struct.
pub const COUNTERS_ABI_VERSION: u32 = 3;

/// Set to a path to have the EP write a JSON counter snapshot there.
pub const ENV_COUNTERS_FILE: &str = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE";

// ---------------------------------------------------------------------------
// model_output_equivalence — §9.1.3 / §10.0 verdict
//
// This is a JSON-only field. It is NOT part of VulkanEpCounters (the C ABI struct) because:
//   1. The struct is published and consumed by epctl, probe_allocator.py, and test_phi35.py.
//      Renaming or growing it breaks callers. Compatibility outranks API elegance (standing ruling).
//   2. The EP has no access to the CPU oracle. The verdict is set by Trinity's Python harness
//      after running a VulkanEP-vs-CPU comparison on the same artifact.
//   3. Absent means UNMEASURED — not MATCH. This is R7: absence of an instrument is not a
//      negative result. A run that did not compare says so explicitly.
//
// Cross-owner note (counters.rs is Switch's file; Trinity adds these constants and the
// corresponding JSON emission in to_json/dump_observations_if_requested):
//   - to_json() gains a default "UNMEASURED" verdict so every counter dump is self-describing.
//   - dump_observations_if_requested() reads the existing file's verdict before overwriting,
//     so Trinity's Python write (MATCH or DIVERGENT) survives the teardown rebuild.
//   - No abi_version bump: the C struct is unchanged; this is a JSON-only addition.
// ---------------------------------------------------------------------------

/// The JSON key for the model-level correctness verdict.
pub const EQUIVALENCE_KEY: &str = "model_output_equivalence";
/// Verdict value: VulkanEP and CPU oracle agree within §9.1 tolerance on all outputs.
pub const EQUIVALENCE_MATCH: &str = "MATCH";
/// Verdict value: at least one output disagrees — the kernel is wrong.
pub const EQUIVALENCE_DIVERGENT: &str = "DIVERGENT";
/// Verdict value (default): no CPU comparison was performed in this run.
pub const EQUIVALENCE_UNMEASURED: &str = "UNMEASURED";
/// Verdict value: the comparison was performed and **this EP executed zero nodes**.
///
/// Added 2026-07-31 by §10.0's third metric amendment. This is **not** `DIVERGENT`:
/// `DIVERGENT` says our kernels computed the wrong answer, `UNATTRIBUTED` says our kernels
/// did not run and the comparison was CPU-vs-CPU. Different owners, different fixes,
/// different next questions — "a lane that prints one red for both is a lane with R13's
/// defect". Written by the Python harness (`tests/ops/_verdict.py`), which derives it and
/// cannot be made to emit `MATCH` at a zero own-provider count.
pub const EQUIVALENCE_UNATTRIBUTED: &str = "UNATTRIBUTED";
/// Verdict value: the two attribution witnesses disagree about whether this EP ran.
///
/// The ORT profile (an instrument we do not own) and `dispatches_executed` (ours) must
/// agree about *presence*, not magnitude. One saying "ran" while the other says "did not"
/// means one of the two instruments is lying and we do not know which; nothing may be
/// reported until we do (§10.0 third amendment, clause 2).
pub const EQUIVALENCE_SPLIT_FRAME: &str = "SPLIT-FRAME";
/// The JSON key for the full verdict record: `executed_by`, `attribution_source`,
/// `attribution_witnesses`, `artifact`, device identity. Written beside the token by
/// `tests/ops/_verdict.py::write_equivalence_record`, because *a caveat that lives in a
/// different artifact from the number it qualifies is not attached to it*.
pub const EQUIVALENCE_RECORD_KEY: &str = "model_output_equivalence_record";

/// Extract the `model_output_equivalence` value from an existing JSON snapshot string.
///
/// Returns one of the `EQUIVALENCE_*` constants. Returns `EQUIVALENCE_UNMEASURED` if the
/// field is absent, unreadable, or carries an unrecognised value — so the caller never has
/// to special-case the absent-field case.
///
/// Hand-rolled for the same reason as [`json_u64`] in `epctl.rs`: no serialiser dependency
/// at an ABI boundary.
pub fn extract_equivalence(doc: &str) -> &'static str {
    let needle = format!("\"{EQUIVALENCE_KEY}\"");
    let Some(start) = doc.find(&needle) else {
        return EQUIVALENCE_UNMEASURED;
    };
    let rest = doc[start + needle.len()..].trim_start();
    let Some(rest) = rest.strip_prefix(':').map(str::trim_start) else {
        return EQUIVALENCE_UNMEASURED;
    };
    let Some(rest) = rest.strip_prefix('"') else {
        return EQUIVALENCE_UNMEASURED;
    };
    if rest.starts_with(EQUIVALENCE_MATCH) {
        return EQUIVALENCE_MATCH;
    }
    if rest.starts_with(EQUIVALENCE_DIVERGENT) {
        return EQUIVALENCE_DIVERGENT;
    }
    if rest.starts_with(EQUIVALENCE_UNATTRIBUTED) {
        return EQUIVALENCE_UNATTRIBUTED;
    }
    if rest.starts_with(EQUIVALENCE_SPLIT_FRAME) {
        return EQUIVALENCE_SPLIT_FRAME;
    }
    EQUIVALENCE_UNMEASURED
}

/// Extract the raw `model_output_equivalence_record` **object** from an existing snapshot.
///
/// Returns the object text including its braces, or `None` if the key is absent or the value
/// is not a brace-delimited object.
///
/// # Why this exists
///
/// `dump_observations_if_requested` rebuilds the counters file from scratch at teardown and
/// carried the verdict *token* across but not the *record*. The observable consequence was a
/// file on disk reading `"model_output_equivalence": "MATCH"` with no `executed_by` frame
/// anywhere in it — which is precisely the shape §10.0's third metric amendment forbids, and
/// precisely the shape `epctl` now refuses. The verdict was correct in the process that wrote
/// it and lost its attribution on the way to the artifact a human reads.
///
/// This is the same defect as Defect C (two writers, one artifact, different schemas), and it
/// is a *caveat detached from the number it qualifies*: the token survived, the sentence
/// saying which world it was about did not.
///
/// Nesting-aware because the record contains nested objects (`executed_by`,
/// `attribution_witnesses`); string-aware because a value could contain a brace.
pub fn extract_equivalence_record(doc: &str) -> Option<&str> {
    let needle = format!("\"{EQUIVALENCE_RECORD_KEY}\"");
    let start = doc.find(&needle)?;
    let rest = &doc[start + needle.len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    if !rest.starts_with('{') {
        return None;
    }
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Host↔device staging traffic, counted **unconditionally**, outside the tracer.
///
/// # Why this exists, and why it is not in the tracer
///
/// Two upload accountings existed and the one everybody quotes was blind. `alloc_device_upload_*`
/// counts copies through the *allocator's device-memory provider*, which is only engaged when
/// `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1`. The upload that actually dominates the run —
/// `vk::session` re-staging the whole weight set on every inference, measured at 1997.6 MiB per
/// inference and 71% of wall — went through `Tracer::record_transfer`, which **early-returns
/// unless tracing or verbose is on**. So the default configuration produced
/// `alloc_device_upload_bytes: 0` on a run whose `cmd_upload` phase was 15.2 seconds, and the
/// bytes that matter were recorded nowhere at all.
///
/// Persistent weight residency (the ~95% lever) is to be verified **on bytes, not wall time**,
/// because bytes are deterministic and wall time swings 9.5× under contention. A byte falsifier
/// that only exists when an opt-in flag is set is a falsifier that will not be set on the run
/// somebody quotes. These are therefore plain atomics on the recording path, in the counters
/// module, and they land in the counters JSON every run.
///
/// # What they are NOT
///
/// They are **not** an independent measurement of `alloc_device_upload_bytes` and the two must
/// never be added or reconciled: they count different copies made by different code through
/// different devices. R11 — an identity whose two sides come from the same source cannot fire;
/// so can an identity between two quantities that were never the same quantity.
pub mod staging {
    use super::ORD;
    use std::sync::atomic::AtomicU64;

    static UPLOADS: AtomicU64 = AtomicU64::new(0);
    static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
    static UPLOAD_US: AtomicU64 = AtomicU64::new(0);
    static READBACKS: AtomicU64 = AtomicU64::new(0);
    static READBACK_BYTES: AtomicU64 = AtomicU64::new(0);
    static READBACK_US: AtomicU64 = AtomicU64::new(0);

    /// One host→device staging copy. Called from `Tracer::record_transfer`, *before* its
    /// `active()` guard, so it is recorded whether or not anybody asked for a trace.
    pub fn on_upload(bytes: u64, us: u64) {
        UPLOADS.fetch_add(1, ORD);
        UPLOAD_BYTES.fetch_add(bytes, ORD);
        UPLOAD_US.fetch_add(us, ORD);
    }

    /// One device→host readback. See [`on_upload`].
    pub fn on_readback(bytes: u64, us: u64) {
        READBACKS.fetch_add(1, ORD);
        READBACK_BYTES.fetch_add(bytes, ORD);
        READBACK_US.fetch_add(us, ORD);
    }

    /// `(uploads, upload_bytes, upload_us, readbacks, readback_bytes, readback_us)`.
    pub fn snapshot() -> (u64, u64, u64, u64, u64, u64) {
        (
            UPLOADS.load(ORD),
            UPLOAD_BYTES.load(ORD),
            UPLOAD_US.load(ORD),
            READBACKS.load(ORD),
            READBACK_BYTES.load(ORD),
            READBACK_US.load(ORD),
        )
    }

    pub fn reset() {
        for c in [
            &UPLOADS,
            &UPLOAD_BYTES,
            &UPLOAD_US,
            &READBACKS,
            &READBACK_BYTES,
            &READBACK_US,
        ] {
            c.store(0, ORD);
        }
    }

    /// What a zero means here, said out loud, because zero is the value residency is supposed to
    /// produce and it is also the value a moved hook produces.
    ///
    /// R7: absence of an instrument must not read as a negative result. When no staging bytes were
    /// recorded across a run that executed Compute calls, this counter alone **cannot** tell
    /// "the weights are resident" from "`record_transfer` no longer brackets the copy". The
    /// independent cross-check is the `cmd_upload` phase in `vk::session`, which is a different
    /// call site around the same memcpy.
    pub fn sentence(compute_calls: u64) -> String {
        let (up_n, up_b, up_us, rb_n, rb_b, _) = snapshot();
        if up_n == 0 && rb_n == 0 {
            if compute_calls == 0 {
                return "SESSION STAGING: none recorded, and no Compute call ran — nothing to say."
                    .to_string();
            }
            return format!(
                "SESSION STAGING: 0 bytes recorded across {compute_calls} Compute call(s). This \
                 counter alone CANNOT distinguish 'the weights are resident' (the win) from \
                 'record_transfer no longer brackets the staging copy' (an instrument failure). \
                 Cross-check the `cmd_upload` phase, which is an independent bracket around the \
                 same memcpy in vk::session, before quoting this as residency."
            );
        }
        let mib = up_b as f64 / (1024.0 * 1024.0);
        let per_call = if compute_calls > 0 {
            format!(
                " {:.2} MiB per Compute call across {} call(s)",
                mib / compute_calls as f64,
                compute_calls
            )
        } else {
            String::new()
        };
        format!(
            "SESSION STAGING: {up_n} host->device copy/copies totalling {mib:.1} MiB in {:.1} ms, \
             {rb_n} readback(s) totalling {:.2} MiB.{per_call} This is per-inference traffic and \
             it is counted SEPARATELY from every alloc_device_upload_* number, which only sees \
             copies through the allocator's device-memory provider: the two are different copies \
             and must never be added.",
            up_us as f64 / 1000.0,
            rb_b as f64 / (1024.0 * 1024.0),
        )
    }
}

/// Session-scoped device-memory residency and weight-cache lifetime.
///
/// # Why this frame exists (R12)
///
/// The `alloc_*` tallies in [`crate::allocator::tally`] observe Tank's device-backed allocator,
/// which uses its **own** `VkDevice` (§6.5 split frame). They are structurally blind to the
/// session's `gpu-allocator` (`vk::alloc::Allocator`) — the one that holds the weight cache. A
/// weight-cache leak is therefore **UNOBSERVABLE** in `alloc_high_water_bytes`: that counter reads
/// a different world. This module is the instrument for the session allocator's frame.
///
/// # What each counter falsifies (R9/R10)
///
/// * `session_device_high_water_bytes` — deterministic (byte counts do not swing with CPU
///   contention). Predict it before a run, then read it: it is the peak device-local bytes the
///   session allocator ever held simultaneously. For a session that caches N MiB of weights and
///   nothing else, the high-water is bounded at ≈ N MiB regardless of how many inferences run.
///   If it grows with the run count, the cache lifetime is leaking.
/// * `weight_cache_release_calls` — **the R10 wiring artifact.** `release_weight_cache` in the
///   source tree is indistinguishable from one never written until an artifact varies with input.
///   This counter is 0 on a run where the release path never executes, and non-zero the instant it
///   does — its value is produced by the call graph, not by review.
/// * `session_device_bytes_in_use` / `weight_cache_bytes_resident` at teardown — if the session's
///   lifetime owns the cache, both reach 0 by the time the process observations are dumped. A
///   non-zero residual is a leak the session never released.
pub mod weights {
    use super::ORD;
    use std::sync::atomic::AtomicU64;

    static DEV_ALLOCS: AtomicU64 = AtomicU64::new(0);
    static DEV_FREES: AtomicU64 = AtomicU64::new(0);
    static DEV_BYTES_IN_USE: AtomicU64 = AtomicU64::new(0);
    static DEV_HIGH_WATER: AtomicU64 = AtomicU64::new(0);
    static WC_RELEASE_CALLS: AtomicU64 = AtomicU64::new(0);
    static WC_RELEASE_BUFFERS: AtomicU64 = AtomicU64::new(0);
    static WC_RELEASE_BYTES: AtomicU64 = AtomicU64::new(0);
    static WC_BYTES_RESIDENT: AtomicU64 = AtomicU64::new(0);

    /// One device-local (`DeviceLocal` / `PackedWeights`) buffer was allocated through the session
    /// allocator. Raises the high-water mark if the new in-use total exceeds any prior peak.
    pub fn on_device_alloc(bytes: u64) {
        DEV_ALLOCS.fetch_add(1, ORD);
        let now = DEV_BYTES_IN_USE.fetch_add(bytes, ORD) + bytes;
        // Monotonic-max update. Relaxed is fine: the only reader is a post-run snapshot, and a
        // transient under-report during a race resolves on the next alloc.
        DEV_HIGH_WATER.fetch_max(now, ORD);
    }

    /// One device-local buffer was freed through the session allocator.
    pub fn on_device_free(bytes: u64) {
        DEV_FREES.fetch_add(1, ORD);
        DEV_BYTES_IN_USE.fetch_sub(bytes, ORD);
    }

    /// A weight-tensor buffer entered the cache (moved out of the per-call free path).
    pub fn on_cache_insert(bytes: u64) {
        WC_BYTES_RESIDENT.fetch_add(bytes, ORD);
    }

    /// A cache entry was evicted mid-run (stale key overwrite), not released at subgraph teardown.
    /// Adjusts resident bytes without touching the release-call wiring artifact.
    pub fn on_cache_evict(bytes: u64) {
        WC_BYTES_RESIDENT.fetch_sub(bytes, ORD);
    }

    /// `release_weight_cache` freed a cache for one subgraph: `buffers` entries, `bytes` total.
    /// Called once per subgraph release even when the cache was empty, so the **call count** is the
    /// wiring artifact and the **byte total** is the residency reclaimed.
    pub fn on_cache_release(buffers: u64, bytes: u64) {
        WC_RELEASE_CALLS.fetch_add(1, ORD);
        WC_RELEASE_BUFFERS.fetch_add(buffers, ORD);
        WC_RELEASE_BYTES.fetch_add(bytes, ORD);
        WC_BYTES_RESIDENT.fetch_sub(bytes, ORD);
    }

    /// `(dev_allocs, dev_frees, dev_bytes_in_use, dev_high_water, wc_release_calls,
    ///   wc_release_buffers, wc_release_bytes, wc_bytes_resident)`.
    #[allow(clippy::type_complexity)]
    pub fn snapshot() -> (u64, u64, u64, u64, u64, u64, u64, u64) {
        (
            DEV_ALLOCS.load(ORD),
            DEV_FREES.load(ORD),
            DEV_BYTES_IN_USE.load(ORD),
            DEV_HIGH_WATER.load(ORD),
            WC_RELEASE_CALLS.load(ORD),
            WC_RELEASE_BUFFERS.load(ORD),
            WC_RELEASE_BYTES.load(ORD),
            WC_BYTES_RESIDENT.load(ORD),
        )
    }

    pub fn reset() {
        for c in [
            &DEV_ALLOCS,
            &DEV_FREES,
            &DEV_BYTES_IN_USE,
            &DEV_HIGH_WATER,
            &WC_RELEASE_CALLS,
            &WC_RELEASE_BUFFERS,
            &WC_RELEASE_BYTES,
            &WC_BYTES_RESIDENT,
        ] {
            c.store(0, ORD);
        }
    }
}

/// The wire format of [`snapshot`], and the C ABI `OrtEpVulkanGetExecutionCounters` fills in.
///
/// `#[repr(C)]` with `struct_size` and `abi_version` first so a reader can validate before it
/// trusts any other field. Growth is additive: append only.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VulkanEpCounters {
    /// `size_of::<VulkanEpCounters>()` as the *library* knows it.
    pub struct_size: u32,
    /// [`COUNTERS_ABI_VERSION`].
    pub abi_version: u32,
    /// `OrtEp::Compile` entries, successful or not.
    pub compile_calls: u64,
    /// Fused subgraphs that compiled to at least one dispatchable kernel.
    pub subgraphs_live: u64,
    /// Fused subgraphs that compiled to a stub — no device, or translation produced no kernels.
    /// A lane with `subgraphs_live == 0` and `subgraphs_stub > 0` claimed work it cannot do.
    pub subgraphs_stub: u64,
    /// `OrtNodeComputeInfo::Compute` entries.
    pub compute_calls: u64,
    /// Compute calls that returned an `OrtStatus` instead of success.
    pub compute_failures: u64,
    /// `VK_ERROR_DEVICE_LOST` observations on a fence wait.
    ///
    /// Separate from `compute_failures` because it is not one failure among many: once the
    /// device is lost every later submission fails, the host process keeps running, and ORT's
    /// Python fallback re-runs the graph on the CPU EP and returns a plausible answer with exit
    /// status 0. Any run with `device_losses > 0` is `ERROR(instrument)`, not a measurement —
    /// including the parts of it that completed before the loss.
    pub device_losses: u64,
    /// Dispatches that ran to fence completion. **This is the criterion-8 number.**
    pub dispatches_executed: u64,
    /// Islands that passed `partition::evaluate` (net-benefit gate, §7.0.2) in multi-cluster
    /// `GetCapability` calls.  Single-cluster bypass does not increment this.  Present even
    /// when 0 so that the wiring census (R10) can distinguish "gate ran, all rejected" from
    /// "UNWIRED" (key absent).  Added in ABI version 2.
    pub viable_islands_retained: u64,
    /// Nodes for which the §8.9 proof ledger was consulted at all (`Ready` rows).  **Added in
    /// ABI version 3.**
    ///
    /// The unconditional half of the pair, so a run in which the gate never executed reads `0`
    /// here regardless of what `ledger_hits` says.  Present even when 0, for the same reason
    /// `viable_islands_retained` is: absence of the field and a value of zero are different
    /// facts and must not share a spelling (R12).
    ///
    /// The C ABI carries the *counts* only. The three-state token (`"UNWIRED"` /
    /// `"UNOBSERVABLE"` / int) lives in the JSON artifact, because a `u64` cannot express it —
    /// a reader of this struct must derive the state from `proven_key_lookups == 0` and
    /// `ledger_entries == 0` rather than from a sentinel value, since every sentinel a `u64`
    /// could carry is also a legitimate count.
    pub proven_key_lookups: u64,
    /// Of those lookups, the ones the ledger held a proof for.  **ABI version 3.**
    pub ledger_hits: u64,
    /// Nodes declined with `[unproven]` — a kernel exists and no proof covers the form.
    /// **ABI version 3.**
    pub unproven_declines: u64,
    /// Entries in the ledger compiled into this binary.  **ABI version 3.**
    ///
    /// Needed to tell `UNOBSERVABLE` (an empty ledger — a hit could not have occurred) from a
    /// real miss. Without it, `ledger_hits == 0` is the ambiguous digit RAI-011 was raised over.
    pub ledger_entries: u64,
    /// Forms claimed through the `CLAIM_UNPROVEN` escape hatch rather than on evidence.
    /// **ABI version 3.**  Non-zero means this run's claims are not fully backed by proofs.
    pub unproven_forms_claimed: u64,
}

static COMPILE_CALLS: AtomicU64 = AtomicU64::new(0);
static SUBGRAPHS_LIVE: AtomicU64 = AtomicU64::new(0);
static SUBGRAPHS_STUB: AtomicU64 = AtomicU64::new(0);
static COMPUTE_CALLS: AtomicU64 = AtomicU64::new(0);
static COMPUTE_FAILURES: AtomicU64 = AtomicU64::new(0);
static DEVICE_LOSSES: AtomicU64 = AtomicU64::new(0);
static DISPATCHES_EXECUTED: AtomicU64 = AtomicU64::new(0);
/// Successful and failed writes of the JSON snapshot, reported *by the next successful write*.
///
/// **Why this exists (Tank, 2026-08-02, measured).** The snapshot is rewritten on the dispatch
/// path and again at teardown, and the write is best-effort: a failure is logged at `warn` and
/// ignored, because a diagnostic that can fail the run it was only supposed to observe is a
/// liability. That is still the right policy — but it has a consequence nobody had written down.
/// **A failed write leaves the previous snapshot on disk, and the previous snapshot is a
/// well-formed document that looks complete.** A reader cannot tell it from the final one.
///
/// Specimen: a two-lane KV run on a contended box produced `session_staging_uploads = 3,
/// session_staging_readbacks = 2` for a 5-iteration loop — an inference caught in flight, with
/// upload counted and readback not. Differencing those byte totals measured *where the observation
/// stopped*, not what the run did, and the resulting slope looked like a 6.7% KV saving. Re-running
/// the same point produced the complete document and the byte-exact expected value.
///
/// So the pair is emitted into the document: `counters_snapshot_writes` is how many snapshots
/// preceded this one, and `counters_snapshot_write_failures` is how many of those did not reach
/// the disk. A stale file is now self-announcing unless *every* subsequent write fails, and a
/// reader who sees a non-zero failure count knows the file may be a prefix. This is R13 in the
/// artifact rather than in the harness: an instrument that could not record is not a measurement.
static SNAPSHOT_WRITES: AtomicU64 = AtomicU64::new(0);
static SNAPSHOT_WRITE_FAILURES: AtomicU64 = AtomicU64::new(0);
/// Total nodes that passed the claim predicate across all `GetCapability` calls.
///
/// JSON-only (not in the C ABI struct). Together with `islands_offered`, this is the
/// **partition falsifier**: `islands_offered == claimed_nodes` means every node is its own
/// island — partitioning produced no merges.
static CLAIMED_NODES: AtomicU64 = AtomicU64::new(0);
/// Islands offered to ORT across all `GetCapability` calls (surviving partition evaluation).
///
/// JSON-only. See [`CLAIMED_NODES`].
static ISLANDS_OFFERED: AtomicU64 = AtomicU64::new(0);

/// Islands that were evaluated by `partition::evaluate` (the net-benefit / `retain_viable` gate)
/// and passed — i.e., survived the economics gate in multi-cluster graphs.
///
/// JSON-only. This is the R10 wiring observable for the net-benefit predicate (§7.0.2 §10.0.1):
/// its value is 0 for single-cluster graphs (bypassed) and varies with the island graph for
/// multi-cluster runs. A counter file that contains this key — even at 0 — proves the mechanism
/// is in the call graph. Owner: Mouse (`partition.rs`, `ep.rs`).
static VIABLE_ISLANDS_RETAINED: AtomicU64 = AtomicU64::new(0);

/// Clusters that reached the net-benefit decision point at all, whether the gate ran or was
/// bypassed. Unconditional half of [`record_net_benefit_decision`] — RAI-011.
static NET_BENEFIT_CLUSTERS_SEEN: AtomicU64 = AtomicU64::new(0);
/// Clusters actually put through `partition::evaluate`.
static NET_BENEFIT_GATE_EVALUATIONS: AtomicU64 = AtomicU64::new(0);
/// Clusters that went around the gate via the single-cluster bypass (§7, line ~1114).
///
/// **Expected to be permanently 0 from `GetCapability` as of 2026-08-01 (RAI-011, Mouse):** the
/// bypass was removed and replaced by a post-evaluation override. A non-zero value here now means
/// a second, un-evaluated path into the partitioner has been reintroduced.
static NET_BENEFIT_GATE_BYPASSES: AtomicU64 = AtomicU64::new(0);
/// Islands the gate **rejected** whose rejection was overridden because they were the graph's
/// only island — RAI-011's replacement for the bypass.
///
/// Distinct from every other state here on purpose. Before this counter existed the three facts
/// "the gate retained the island", "the gate rejected every island" and "the gate never ran"
/// compressed onto one digit. They now occupy three different fields, one of which is a string.
static NET_BENEFIT_SOLE_ISLAND_OVERRIDES: AtomicU64 = AtomicU64::new(0);
/// Which rejection(s) the override suppressed, as a bitmask over [`OverriddenVerdict`].
///
/// A count of overrides tells you the override fired; it does not tell you **what it overrode**,
/// and those are different facts. `net_benefit_sole_island_overrides: 1` was reported in the
/// wiring census with no way to learn whether the gate had said *too small* or *transfer
/// dominated* — the reason existed in memory (`GateOutcome::SoleIslandOverride` carries it) and
/// died at the counter boundary. This is the missing half.
///
/// A bitmask rather than a last-writer-wins slot, so two overrides with different reasons in one
/// process render as `MIXED` instead of silently reporting whichever ran last.
static NET_BENEFIT_OVERRIDE_REASONS: AtomicU64 = AtomicU64::new(0);

/// The gate verdict a sole-island override suppressed.
///
/// Deliberately *not* `partition::RejectReason`: this module must not depend on the partitioner,
/// and the counter needs a token, not the arithmetic that produced it. The mapping happens once,
/// at the single call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverriddenVerdict {
    /// The gate rejected on the size arm (`RejectReason::TooSmall`).
    TooSmall,
    /// The gate rejected on the economics arm (`RejectReason::TransferDominated`).
    TransferDominated,
}

impl OverriddenVerdict {
    fn bit(self) -> u64 {
        match self {
            OverriddenVerdict::TooSmall => 1,
            OverriddenVerdict::TransferDominated => 2,
        }
    }
}

// --- §8.9 proof-ledger gate (criterion 11). Owner: Mouse (`registry.rs`). ---
//
// The same three-state shape as the net-benefit gate above, and for the same reason. Before
// RAI-011, `viable_islands_retained: 0` meant *bypassed* and *all-rejected* and *never ran*, all
// on one digit. `proven_key_lookups` is the unconditional half here: every node with a `Ready`
// row increments it whether the ledger hit or missed, so a run in which the mechanism never
// executed is `0` — a state the JSON reports as the string `"UNWIRED"`, not as a number.
/// Nodes for which the ledger was consulted at all.
static LEDGER_LOOKUPS: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones the ledger held a proof for.
static LEDGER_HITS: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones where the ledger parsed and simply held no proof for that form.
/// **Distinct from a faulted lookup** (RAI-008(d)): this one says *regenerate this form*.
static LEDGER_KEY_ABSENT: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones that got no reading because the ledger itself failed. R13: an instrument
/// error is never a detection, so these are never counted as misses about a form.
static LEDGER_FAULTED_LOOKUPS: AtomicU64 = AtomicU64::new(0);
/// Nodes declined with `[unproven]`: a kernel exists, no proof covers the form, and the form is
/// not in the §8.9.4 allowlist.
static UNPROVEN_DECLINES: AtomicU64 = AtomicU64::new(0);
/// Nodes claimed only because their key was in `ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN`.
static UNPROVEN_FORMS_CLAIMED: AtomicU64 = AtomicU64::new(0);
/// The distinct keys that the escape hatch actually let through, for
/// `unproven_forms_enabled: [...]` (§8.9.4 item 3). A list rather than a count on purpose: the
/// CI check has to be able to *name* what a lane claimed without evidence.
static UNPROVEN_KEYS_USED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Shader stems this process actually dispatched, sorted and deduplicated.
///
/// Added 2026-08-02 (Mouse, cross-owner into Tank's file, declared) for §8.9.11. A proof-ledger
/// entry says a form matched the CPU oracle; it has to be able to say *what code* it matched
/// with, or the entry silently outlives its subject when that code is replaced. The stems are
/// `&'static str` from the shader table, so this is the set of embedded SPIR-V modules the run
/// bound to a pipeline — the run's own account of what it executed, not a re-derivation from the
/// registry.
static SHADERS_DISPATCHED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

/// Claimed nodes whose `Compute()` returned a non-OK status — RAI Ruling 2's broken commitment.
static BROKEN_COMMITMENTS: AtomicU64 = AtomicU64::new(0);/// Of those, the ones whose mandatory WARN reached ORT's own logging sink.
static BROKEN_COMMITMENT_WARNS_TO_ORT: AtomicU64 = AtomicU64::new(0);

/// How many §8.9.7 session-creation disclosures ran in this process.
///
/// The in-frame witness for [`claimed_form_evidence`]: zero here means the claim set was never
/// examined, which is not the same as a claim set that examined clean.
static SESSION_DISCLOSURES: AtomicU64 = AtomicU64::new(0);
/// Distinct claimed forms backed by a ledger `MATCH`.
static CLAIMED_FORMS_PROVEN: AtomicU64 = AtomicU64::new(0);
/// Distinct claimed forms with no proof at all.
static CLAIMED_FORMS_UNMEASURED: AtomicU64 = AtomicU64::new(0);
/// Distinct claimed forms whose recorded verdict is not `MATCH`.
static CLAIMED_FORMS_DIVERGENT: AtomicU64 = AtomicU64::new(0);
/// Distinct claimed forms whose evidence could not be read because the ledger is faulted.
static CLAIMED_FORMS_LEDGER_FAULTED: AtomicU64 = AtomicU64::new(0);
/// Disclosures that emitted the mandatory WARN.
static SESSION_DISCLOSURE_WARNS: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones whose WARN reached ORT's own logging sink.
static SESSION_DISCLOSURE_WARNS_TO_ORT: AtomicU64 = AtomicU64::new(0);
/// Compute failures produced by the fault-injection control rather than suffered.
static COMPUTE_FAILURES_INJECTED: AtomicU64 = AtomicU64::new(0);

/// `Relaxed` is correct here and the reasoning is worth stating rather than assuming.
///
/// These counters are diagnostics: nothing in the EP branches on them, so no other memory access
/// needs to be ordered against them. The only consumer is a reader in another thread (or another
/// process, via the JSON file) that wants a number, and for that `Relaxed` gives monotonic
/// per-counter values with no ordering guarantee *between* counters. A snapshot can therefore
/// catch `compute_calls` incremented and `dispatches_executed` not yet — which is a real state the
/// program passes through anyway, so it is not even a lie.
const ORD: Ordering = Ordering::Relaxed;

pub fn record_compile_call() {
    COMPILE_CALLS.fetch_add(1, ORD);
}

pub fn record_subgraph(live: bool) {
    if live {
        SUBGRAPHS_LIVE.fetch_add(1, ORD);
    } else {
        SUBGRAPHS_STUB.fetch_add(1, ORD);
    }
}

pub fn record_compute_call() {
    COMPUTE_CALLS.fetch_add(1, ORD);
}

pub fn record_compute_failure() {
    COMPUTE_FAILURES.fetch_add(1, ORD);
}

/// One VK_ERROR_DEVICE_LOST on a fence wait. Sticky for the life of the process: a device
/// that came back is still a device that went away, and everything measured either side of it
/// belongs to two different machines.
pub fn record_device_lost() {
    DEVICE_LOSSES.fetch_add(1, ORD);
}

/// How many times the device has been lost in this process.
pub fn device_losses() -> u64 {
    DEVICE_LOSSES.load(ORD)
}

/// Record a `GetCapability` call's partition results.
///
/// `claimed` is the number of nodes that passed `claim_decision`; `islands` is the number of
/// connected clusters offered to ORT after partition evaluation. Together they form the
/// **partition falsifier**: when `islands == claimed` and both are `> 1`, partitioning produced
/// no merges — every node is its own island.
pub fn record_capability(claimed: u64, islands: u64) {
    CLAIMED_NODES.fetch_add(claimed, ORD);
    ISLANDS_OFFERED.fetch_add(islands, ORD);
}

/// Record the number of islands that passed `partition::evaluate` (the net-benefit gate) in a
/// multi-cluster `GetCapability` call. Single-cluster calls bypass the gate and must not call
/// this function — the absence of an increment from a single-cluster run is correct behaviour.
///
/// R10 observable: the presence of `viable_islands_retained` in the counters JSON proves the
/// net-benefit gate is in the production call graph.  The value varies with the island graph:
/// 0 when every candidate island is rejected by TooSmall or TransferDominated, N > 0 when N
/// islands survive.  An always-0 result is distinguishable from UNWIRED (key absent) because the
/// key is present — UNWIRED would not emit the key at all.
pub fn record_viable_islands_retained(n: u64) {
    VIABLE_ISLANDS_RETAINED.fetch_add(n, ORD);
}

/// Record one cluster's passage past — or around — the net-benefit gate. **RAI-011.**
///
/// One call site, two counters, and only one of them is conditional: an author who wants
/// `net_benefit_gate_evaluations` to be non-zero has to go through the path that also records the
/// bypass, so the pair cannot be forged from either end. Same shape as
/// `allocator::on_residency_evaluated`.
///
/// The finding this exists for: `viable_islands_retained == 0` was *structurally ambiguous*
/// between "the gate ran on every cluster and rejected all of them" and "the gate was never
/// consulted because there was exactly one cluster". Those are opposite facts about the system
/// and they printed the same digit. Phi-3.5 now converges to a single fused island, so the
/// ambiguous reading is the one our only real model produces.
pub fn record_net_benefit_decision(evaluated: bool) {
    NET_BENEFIT_CLUSTERS_SEEN.fetch_add(1, ORD);
    if evaluated {
        NET_BENEFIT_GATE_EVALUATIONS.fetch_add(1, ORD);
    } else {
        NET_BENEFIT_GATE_BYPASSES.fetch_add(1, ORD);
    }
}

/// Record that the sole-island exemption overrode a rejection the gate actually computed.
///
/// Must be called *in addition to* [`record_net_benefit_decision`]`(true)`, never instead of it:
/// the island was evaluated, and then its verdict was overridden. An override that suppressed the
/// evaluation record would be the bypass again.
///
/// The reason is a required argument rather than a second optional call, because an entry point
/// that records the override without the reason is the same shape as the bypass it replaced: one
/// path that reports and one that stays quiet.
pub fn record_sole_island_override(reason: Option<OverriddenVerdict>) {
    NET_BENEFIT_SOLE_ISLAND_OVERRIDES.fetch_add(1, ORD);
    if let Some(reason) = reason {
        NET_BENEFIT_OVERRIDE_REASONS.fetch_or(reason.bit(), ORD);
    }
}

/// Record one §8.9 proof-ledger consultation.
///
/// Unconditional half first, exactly as [`record_net_benefit_decision`] does it: the only way to
/// make `ledger_hits` non-zero is through the path that also increments `proven_key_lookups`, so
/// a hit count cannot exist without a lookup count to divide it by.
///
/// RAI-008(d) — the argument is not a `bool`. A miss is three findings (`KEY-ABSENT`,
/// `LEDGER-FAULTED`, `NEVER-ATTEMPTED`) that call for three different actions, and a `bool`
/// spells all three `false`. That is the same collapse R12 made this project undo when *bypassed*
/// and *all-rejected* were sharing one `0`.
pub fn record_ledger_lookup(outcome: crate::registry::LedgerLookup) {
    use crate::registry::LedgerLookup as L;
    if outcome == L::NeverAttempted {
        // Recording one would be a lookup, which is precisely what it asserts did not happen.
        // It is derived from `proven_key_lookups == 0`, never counted.
        return;
    }
    LEDGER_LOOKUPS.fetch_add(1, ORD);
    match outcome {
        L::Hit => {
            LEDGER_HITS.fetch_add(1, ORD);
        }
        L::KeyAbsent => {
            LEDGER_KEY_ABSENT.fetch_add(1, ORD);
        }
        L::Faulted => {
            LEDGER_FAULTED_LOOKUPS.fetch_add(1, ORD);
        }
        // `NeverAttempted` is derived from `proven_key_lookups == 0`; recording one would be a
        // lookup, which is what it asserts did not happen.
        L::NeverAttempted => {}
    }
}

/// Which of R13's three miss states this run is in, as one token.
///
/// The precedence is deliberate and is the R13 order: an instrument outage outranks a finding.
/// A run whose ledger faulted has *no* reading about any form, so reporting `KEY-ABSENT` — a
/// statement about the form — would spell an outage exactly like a detection.
fn ledger_miss_state() -> &'static str {
    if !crate::registry::ledger().faults.is_empty() || LEDGER_FAULTED_LOOKUPS.load(ORD) > 0 {
        return crate::registry::LedgerLookup::Faulted.token();
    }
    if LEDGER_LOOKUPS.load(ORD) == 0 {
        return crate::registry::LedgerLookup::NeverAttempted.token();
    }
    if LEDGER_KEY_ABSENT.load(ORD) > 0 {
        return crate::registry::LedgerLookup::KeyAbsent.token();
    }
    crate::registry::LedgerLookup::Hit.token()
}

/// Record a node declined because nothing has proven its form (§8.9, `DeclineCode::Unproven`).
pub fn record_unproven_decline() {
    UNPROVEN_DECLINES.fetch_add(1, ORD);
}

/// Record that the §8.9.4 escape hatch let one node through, and which key did it.
///
/// The key is kept, not merely counted. `epctl --check-counters` fails on a non-empty
/// `unproven_forms_enabled` list, and a CI lane that has to be told *what* it claimed without
/// evidence is a lane whose operator can act on the message.
pub fn record_unproven_form_enabled(key: &str) {
    UNPROVEN_FORMS_CLAIMED.fetch_add(1, ORD);
    if let Ok(mut used) = UNPROVEN_KEYS_USED.lock()
        && !used.iter().any(|k| k == key)
    {
        used.push(key.to_string());
    }
}

/// Escape a string for a JSON string literal. Proof keys are `[A-Za-z0-9:./,+_-]` by
/// construction, so this exists to keep a malformed key from producing malformed JSON rather
/// than because one is expected.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// The three-state reading of `ledger_hits`, as a JSON fragment. Same shape, same reasons, as
/// [`viable_islands_retained_json`] — one mechanism's vocabulary, not two (R10 sub-rule).
///
/// * `"UNWIRED"` — no node reached the ledger check at all, so the gate did not run.
/// * `"UNOBSERVABLE"` — the check ran, and the ledger baked into this artifact holds **no
///   entries**, so a hit could not have occurred in this frame. `0` here would read as "we
///   looked and nothing matched", which is a different fact about a different world (R12).
/// * an integer — the ledger has entries and this is how many of the lookups found one.
fn ledger_hits_json() -> String {
    let lookups = LEDGER_LOOKUPS.load(ORD);
    if lookups == 0 {
        "\"UNWIRED\"".to_string()
    } else if crate::registry::ledger().is_empty() {
        "\"UNOBSERVABLE\"".to_string()
    } else {
        LEDGER_HITS.load(ORD).to_string()
    }
}

/// Token naming what the ledger gate did on this run, for a reader grepping a word.
///
/// `FAULTED` is R13's instrument state and outranks the rest: a ledger that failed its own
/// digest or parse check has produced no reading, and reporting that as `DECLINED` would spell
/// an instrument outage exactly like a detection.
fn ledger_gate_state() -> &'static str {
    let ledger = crate::registry::ledger();
    if !ledger.faults.is_empty() {
        return "FAULTED";
    }
    let lookups = LEDGER_LOOKUPS.load(ORD);
    let hits = LEDGER_HITS.load(ORD);
    let declines = UNPROVEN_DECLINES.load(ORD);
    match (lookups, hits, declines) {
        (0, _, _) => "UNWIRED",
        (_, 0, 0) => "MIXED",
        (_, _, 0) => "ALL-PROVEN",
        (_, 0, _) => "ALL-DECLINED",
        _ => "MIXED",
    }
}

/// `unproven_forms_enabled` — the §8.9.4 item-3 disclosure, as a JSON array fragment.
///
/// Absent-meaning-empty is what §8.9.4 specifies, and an empty array is how JSON spells it. This
/// is the value `epctl --check-counters` fails on when it is non-empty.
fn unproven_forms_enabled_json() -> String {
    let Ok(used) = UNPROVEN_KEYS_USED.lock() else {
        return "\"INSTRUMENT-ERROR\"".to_string();
    };
    let body: Vec<String> = used.iter().map(|k| format!("\"{}\"", json_escape(k))).collect();
    format!("[{}]", body.join(", "))
}

/// The three-state reading of `viable_islands_retained`, as a JSON fragment.
///
/// * `"UNWIRED"` — no cluster has reached the decision point at all (R10: uninvoked is not empty).
/// * `"UNOBSERVABLE"` — clusters were seen and **every one of them bypassed** the gate, so the
///   event `viable_islands_retained` counts could not occur in this run's frame (R12).
/// * an integer — the gate ran on at least one cluster, so `0` is a *result*: all rejected.
///
/// A type change, not a value change: an increment can forge a number and cannot forge a type.
fn viable_islands_retained_json() -> String {
    let seen = NET_BENEFIT_CLUSTERS_SEEN.load(ORD);
    let evaluated = NET_BENEFIT_GATE_EVALUATIONS.load(ORD);
    if seen == 0 {
        "\"UNWIRED\"".to_string()
    } else if evaluated == 0 {
        "\"UNOBSERVABLE\"".to_string()
    } else {
        VIABLE_ISLANDS_RETAINED.load(ORD).to_string()
    }
}

/// Token naming what happened to the net-benefit gate on this run, for a reader who is grepping a
/// word rather than doing arithmetic.
fn net_benefit_gate_state() -> &'static str {
    let seen = NET_BENEFIT_CLUSTERS_SEEN.load(ORD);
    let evaluated = NET_BENEFIT_GATE_EVALUATIONS.load(ORD);
    let bypassed = NET_BENEFIT_GATE_BYPASSES.load(ORD);
    match (seen, evaluated, bypassed) {
        (0, _, _) => "UNWIRED",
        (_, 0, _) => "BYPASSED",
        (_, _, 0) => "EVALUATED",
        _ => "MIXED",
    }
}

/// Token naming *what the sole-island override overrode*, which is a different fact from *whether*
/// it fired.
///
/// `UNOBSERVABLE` when no override happened: the event whose reason is being asked for did not
/// occur in this frame, so there is no reason to report and reporting a reason-shaped `NONE` would
/// invite a reader to treat absence as a verdict (R12).
fn net_benefit_override_reason() -> &'static str {
    let overrides = NET_BENEFIT_SOLE_ISLAND_OVERRIDES.load(ORD);
    let mask = NET_BENEFIT_OVERRIDE_REASONS.load(ORD);
    match (overrides, mask) {
        (0, _) => "UNOBSERVABLE",
        (_, 1) => "TOO_SMALL",
        (_, 2) => "TRANSFER_DOMINATED",
        (_, 3) => "MIXED",
        // An override was counted but no reason reached the mask: the two halves of one call site
        // have drifted apart. Named rather than defaulted, because a silent fallback here would
        // hide exactly the wiring failure this field exists to expose.
        (_, _) => "UNRECORDED",
    }
}

/// Record a broken commitment: a node this EP **claimed** whose `Compute()` returned non-OK.
///
/// `delivered_to_ort_sink` is whether the mandatory WARN actually reached ORT's own logger, not
/// whether we tried. Two counters, one call site, one of them unconditional — the delivered count
/// cannot be raised without also raising the count of events it is a subset of.
///
/// **This writes the counters artifact immediately.** The R12 hazard that bit `alloc_*` at
/// shutdown applies with full force here: a broken commitment is followed by ORT's silent fallback
/// and, in the incidents this instrument exists for, by a session that never reaches an orderly
/// teardown on the path a reader is watching. An observable that can only be read at a moment that
/// no longer occurs is out-of-frame by construction, so this one is read at the *instant of the
/// event* rather than at a shutdown that is not guaranteed to arrive.
pub fn record_broken_commitment(delivered_to_ort_sink: bool) {
    BROKEN_COMMITMENTS.fetch_add(1, ORD);
    if delivered_to_ort_sink {
        BROKEN_COMMITMENT_WARNS_TO_ORT.fetch_add(1, ORD);
    }
    dump_if_requested();
}

/// Record that a Compute failure was **planted** by the fault-injection control rather than
/// suffered. Counted separately and published separately so that an injected failure can never be
/// read as a real one, in either direction.
pub fn record_injected_compute_failure() {
    COMPUTE_FAILURES_INJECTED.fetch_add(1, ORD);
}

/// Record one §8.9.7 session-creation disclosure.
///
/// Counts **distinct forms**, not nodes: the disclosure itself is per form, and a counter whose
/// units differ from the disclosure it measures cannot be reconciled with it.
///
/// **This writes the counters artifact immediately**, for the same R12 reason as
/// [`record_broken_commitment`]. The moment this observable must be read is session creation; a
/// session that goes on to claim unproven forms is by construction a session that may end
/// abnormally, and an observable that can only be read at a shutdown that no longer occurs is
/// out-of-frame by construction. Read it at the instant of the event instead.
pub fn record_session_disclosure(
    proven: usize,
    unmeasured: usize,
    divergent: usize,
    ledger_faulted: usize,
    warned: bool,
    warn_reached_ort_sink: bool,
) {
    SESSION_DISCLOSURES.fetch_add(1, ORD);
    CLAIMED_FORMS_PROVEN.fetch_add(proven as u64, ORD);
    CLAIMED_FORMS_UNMEASURED.fetch_add(unmeasured as u64, ORD);
    CLAIMED_FORMS_DIVERGENT.fetch_add(divergent as u64, ORD);
    CLAIMED_FORMS_LEDGER_FAULTED.fetch_add(ledger_faulted as u64, ORD);
    if warned {
        SESSION_DISCLOSURE_WARNS.fetch_add(1, ORD);
        if warn_reached_ort_sink {
            SESSION_DISCLOSURE_WARNS_TO_ORT.fetch_add(1, ORD);
        }
    }
    dump_if_requested();
}

/// The evidence standing of every form this process's sessions claimed, as one token.
///
/// * `"UNOBSERVABLE"` — no session-creation disclosure ran in this frame, so no claim set has
///   been examined and nothing is known. **Never `"ALL-PROVEN"`**: a claim set that was never
///   assembled is not a claim set that was proven, and that substitution is precisely the §6.5
///   coincidence — an agreement produced by the absence of the event rather than by its outcome.
/// * `"ALL-PROVEN"` — at least one form was claimed and every claimed form has a ledger `MATCH`.
/// * `"NO-CLAIMS"` — a disclosure ran and the sessions claimed nothing at all.
/// * `"UNMEASURED-PRESENT"` / `"DIVERGENT-PRESENT"` / `"LEDGER-FAULTED"` — the escalating
///   findings, reported most-specific-first so a divergence is never hidden behind an absence.
fn claimed_form_evidence() -> &'static str {
    if SESSION_DISCLOSURES.load(ORD) == 0 {
        return "UNOBSERVABLE";
    }
    if CLAIMED_FORMS_LEDGER_FAULTED.load(ORD) > 0 {
        "LEDGER-FAULTED"
    } else if CLAIMED_FORMS_DIVERGENT.load(ORD) > 0 {
        "DIVERGENT-PRESENT"
    } else if CLAIMED_FORMS_UNMEASURED.load(ORD) > 0 {
        "UNMEASURED-PRESENT"
    } else if CLAIMED_FORMS_PROVEN.load(ORD) > 0 {
        "ALL-PROVEN"
    } else {
        "NO-CLAIMS"
    }
}

/// Which channel carried the session-creation WARNs, as a token. Same three states, and the same
/// argument, as [`broken_commitment_channel`]: silence does not prove a channel.
fn session_disclosure_channel() -> &'static str {
    let warns = SESSION_DISCLOSURE_WARNS.load(ORD);
    let delivered = SESSION_DISCLOSURE_WARNS_TO_ORT.load(ORD);
    if warns == 0 {
        "UNOBSERVABLE"
    } else if delivered == warns {
        "ORT_SINK"
    } else {
        "PRIVATE_LOG_ONLY"
    }
}

/// The three-state reading of `broken_commitments`.
///
/// `0` is only printable when at least one `Compute` ran: with `compute_calls == 0` this EP never
/// executed a claim, so a claim it made cannot have been broken, and the event is out of frame
/// (R12) rather than absent. This is the same forgery `_verdict.py` refuses for `MATCH` — a
/// CPU-only run must not be able to produce the clean-run token.
fn broken_commitments_json() -> String {
    if COMPUTE_CALLS.load(ORD) == 0 {
        "\"UNOBSERVABLE\"".to_string()
    } else {
        BROKEN_COMMITMENTS.load(ORD).to_string()
    }
}

/// Which channel carried the broken-commitment WARNs, as a token.
///
/// * `"UNOBSERVABLE"` — no broken commitment occurred, so the channel has not been exercised and
///   nothing about it is known from this run. Not `"ORT_SINK"`; a channel is not proven by the
///   absence of traffic.
/// * `"ORT_SINK"` — every WARN reached `Logger_LogMessage`.
/// * `"PRIVATE_LOG_ONLY"` — at least one WARN could not reach ORT's logger (none attached) and
///   exists only on our stderr, i.e. invisible to exactly the audience Ruling 2 names.
fn broken_commitment_channel() -> &'static str {
    let events = BROKEN_COMMITMENTS.load(ORD);
    let delivered = BROKEN_COMMITMENT_WARNS_TO_ORT.load(ORD);
    if events == 0 {
        "UNOBSERVABLE"
    } else if delivered == events {
        "ORT_SINK"
    } else {
        "PRIVATE_LOG_ONLY"
    }
}

///
/// Writing on every successful dispatch means: a crash *after* real work still leaves evidence of
/// that work, and successive reads of the file always reflect the latest accumulated state rather
/// than a snapshot from the process's first dispatch.
pub fn record_dispatches(n: u64) {
    if n == 0 {
        return;
    }
    DISPATCHES_EXECUTED.fetch_add(n, ORD);
    dump_if_requested();
}

/// Record that this process bound and dispatched the embedded SPIR-V module `stem`.
///
/// Idempotent and order-independent: the value of interest is the *set*, because a proof entry's
/// shader digest must not depend on how many times a kernel ran.
pub fn record_shader_dispatched(stem: &'static str) {
    if stem.is_empty() {
        return;
    }
    if let Ok(mut used) = SHADERS_DISPATCHED.lock() {
        if let Err(pos) = used.binary_search(&stem) {
            used.insert(pos, stem);
        }
    }
}

/// The stems recorded by [`record_shader_dispatched`], sorted.
pub fn shaders_dispatched() -> Vec<&'static str> {
    SHADERS_DISPATCHED.lock().map(|u| u.clone()).unwrap_or_default()
}

/// `shaders_dispatched` and `shaders_dispatched_digest` as JSON fragments.
///
/// The digest is over the SPIR-V **bytes** of exactly those modules, so it changes when the
/// compiled kernel changes and does not change when an unrelated shader does. Its frame — what it
/// covers and what it deliberately does not — is stated in `docs/OP_COVERAGE.md` §8.9.11.
fn shaders_dispatched_json() -> (String, String) {
    let stems = shaders_dispatched();
    let list: Vec<String> = stems.iter().map(|s| format!("\"{}\"", json_escape(s))).collect();
    let digest = match crate::registry::shader_digest_for(&stems) {
        Some(d) => d,
        // R12: no module was dispatched is a different fact from "the digest is zero".
        None => "NONE-DISPATCHED".to_string(),
    };
    (format!("[{}]", list.join(", ")), digest)
}

/// Current values. Not a consistent cross-counter snapshot; see [`ORD`].
pub fn snapshot() -> VulkanEpCounters {
    VulkanEpCounters {
        struct_size: std::mem::size_of::<VulkanEpCounters>() as u32,
        abi_version: COUNTERS_ABI_VERSION,
        compile_calls: COMPILE_CALLS.load(ORD),
        subgraphs_live: SUBGRAPHS_LIVE.load(ORD),
        subgraphs_stub: SUBGRAPHS_STUB.load(ORD),
        compute_calls: COMPUTE_CALLS.load(ORD),
        compute_failures: COMPUTE_FAILURES.load(ORD),
        device_losses: DEVICE_LOSSES.load(ORD),
        dispatches_executed: DISPATCHES_EXECUTED.load(ORD),
        viable_islands_retained: VIABLE_ISLANDS_RETAINED.load(ORD),
        proven_key_lookups: LEDGER_LOOKUPS.load(ORD),
        ledger_hits: LEDGER_HITS.load(ORD),
        unproven_declines: UNPROVEN_DECLINES.load(ORD),
        ledger_entries: crate::registry::ledger().len() as u64,
        unproven_forms_claimed: UNPROVEN_FORMS_CLAIMED.load(ORD),
    }
}

/// Zero every counter. Exported so a test can scope a claim to one model run.
pub fn reset() {
    COMPILE_CALLS.store(0, ORD);
    SUBGRAPHS_LIVE.store(0, ORD);
    SUBGRAPHS_STUB.store(0, ORD);
    COMPUTE_CALLS.store(0, ORD);
    COMPUTE_FAILURES.store(0, ORD);
    DEVICE_LOSSES.store(0, ORD);
    DISPATCHES_EXECUTED.store(0, ORD);
    if let Ok(mut used) = SHADERS_DISPATCHED.lock() {
        used.clear();
    }
    CLAIMED_NODES.store(0, ORD);
    ISLANDS_OFFERED.store(0, ORD);
    // Both sides: Tank's staging tally and Mouse's retained-island counter. Neither excludes the
    // other; a reset that clears one and not the other is how a test reads another test's traffic.
    staging::reset();
    weights::reset();
    VIABLE_ISLANDS_RETAINED.store(0, ORD);
    NET_BENEFIT_CLUSTERS_SEEN.store(0, ORD);
    NET_BENEFIT_GATE_EVALUATIONS.store(0, ORD);
    NET_BENEFIT_GATE_BYPASSES.store(0, ORD);
    NET_BENEFIT_SOLE_ISLAND_OVERRIDES.store(0, ORD);
    NET_BENEFIT_OVERRIDE_REASONS.store(0, ORD);
    BROKEN_COMMITMENTS.store(0, ORD);
    BROKEN_COMMITMENT_WARNS_TO_ORT.store(0, ORD);
    SESSION_DISCLOSURES.store(0, ORD);
    CLAIMED_FORMS_PROVEN.store(0, ORD);
    CLAIMED_FORMS_UNMEASURED.store(0, ORD);
    CLAIMED_FORMS_DIVERGENT.store(0, ORD);
    CLAIMED_FORMS_LEDGER_FAULTED.store(0, ORD);
    SESSION_DISCLOSURE_WARNS.store(0, ORD);
    SESSION_DISCLOSURE_WARNS_TO_ORT.store(0, ORD);
    COMPUTE_FAILURES_INJECTED.store(0, ORD);
    LEDGER_LOOKUPS.store(0, ORD);
    LEDGER_HITS.store(0, ORD);
    LEDGER_KEY_ABSENT.store(0, ORD);
    LEDGER_FAULTED_LOOKUPS.store(0, ORD);
    UNPROVEN_DECLINES.store(0, ORD);
    UNPROVEN_FORMS_CLAIMED.store(0, ORD);
    if let Ok(mut used) = UNPROVEN_KEYS_USED.lock() {
        used.clear();
    }
}

impl VulkanEpCounters {
    /// The JSON `epctl --check-counters` reads. Hand-rolled because every value is a `u64` and
    /// pulling in a serialiser for eight integers would be the wrong trade at an ABI boundary.
    ///
    /// Calls [`to_json_with_equiv`] with `UNMEASURED` as the default verdict. The Python
    /// comparison harness overwrites that default by calling `write_equivalence_verdict()` after
    /// running the VulkanEP-vs-CPU comparison.
    pub fn to_json(&self) -> String {
        self.to_json_with_equiv(EQUIVALENCE_UNMEASURED)
    }

    /// Emit the counters JSON with an explicit `model_output_equivalence` verdict.
    ///
    /// `equiv` must be one of `EQUIVALENCE_MATCH`, `EQUIVALENCE_DIVERGENT`, or
    /// `EQUIVALENCE_UNMEASURED`. The field is always present so a reader never has to distinguish
    /// "absent" from "UNMEASURED" — absence and UNMEASURED have the same meaning (R7: absence of
    /// an instrument is not a negative result), but the explicit value makes the state visible.
    pub fn to_json_with_equiv(&self, equiv: &str) -> String {
        let claimed = CLAIMED_NODES.load(ORD);
        let islands = ISLANDS_OFFERED.load(ORD);
        let viable = viable_islands_retained_json();
        let (shaders_list, shaders_digest) = shaders_dispatched_json();
        format!(
            "{{\n  \"abi_version\": {},\n  \"compile_calls\": {},\n  \"subgraphs_live\": {},\n  \
             \"subgraphs_stub\": {},\n  \"compute_calls\": {},\n  \"compute_failures\": {},\n  \"device_losses\": {},\n  \
             \"dispatches_executed\": {},\n  \"claimed_nodes\": {},\n  \"islands_offered\": {},\n  \
             \"viable_islands_retained\": {},\n  \
             \"net_benefit_gate\": \"{}\",\n  \
             \"net_benefit_gate_clusters_seen\": {},\n  \
             \"net_benefit_gate_evaluations\": {},\n  \
             \"net_benefit_gate_bypasses\": {},\n  \
             \"net_benefit_sole_island_overrides\": {},\n  \
             \"net_benefit_override_reason\": \"{}\",\n  \
             \"broken_commitments\": {},\n  \
             \"broken_commitment_warns_to_ort_sink\": {},\n  \
             \"broken_commitment_warn_channel\": \"{}\",\n  \
             \"compute_failures_injected\": {},\n  \
             \"fault_injection\": \"{}\",\n  \
             \"proven_key_lookups\": {},\n  \
             \"ledger_hits\": {},\n  \
             \"ledger_gate\": \"{}\",\n  \
             \"ledger_miss\": \"{}\",\n  \
             \"ledger_entries\": {},\n  \
             \"ledger_faults\": {},\n  \
             \"unproven_declines\": {},\n  \
             \"unproven_forms_claimed\": {},\n  \
             \"unproven_forms_enabled\": {},\n  \
             \"shaders_dispatched\": {},\n  \
             \"shaders_dispatched_digest\": \"{}\",\n  \
             \"session_disclosures\": {},\n  \
             \"claimed_forms_proven\": {},\n  \
             \"claimed_forms_unmeasured\": {},\n  \
             \"claimed_forms_divergent\": {},\n  \
             \"claimed_forms_ledger_faulted\": {},\n  \
             \"claimed_form_evidence\": \"{}\",\n  \
             \"session_disclosure_warns\": {},\n  \
             \"session_disclosure_warns_to_ort_sink\": {},\n  \
             \"session_disclosure_channel\": \"{}\",\n  \
             \"model_output_equivalence\": \"{}\"\n}}\n",
            self.abi_version,
            self.compile_calls,
            self.subgraphs_live,
            self.subgraphs_stub,
            self.compute_calls,
            self.compute_failures,
            self.device_losses,
            self.dispatches_executed,
            claimed,
            islands,
            viable,
            net_benefit_gate_state(),
            NET_BENEFIT_CLUSTERS_SEEN.load(ORD),
            NET_BENEFIT_GATE_EVALUATIONS.load(ORD),
            NET_BENEFIT_GATE_BYPASSES.load(ORD),
            NET_BENEFIT_SOLE_ISLAND_OVERRIDES.load(ORD),
            net_benefit_override_reason(),
            broken_commitments_json(),
            BROKEN_COMMITMENT_WARNS_TO_ORT.load(ORD),
            broken_commitment_channel(),
            COMPUTE_FAILURES_INJECTED.load(ORD),
            if crate::ep::fault_injection_active() {
                "ACTIVE"
            } else {
                "NONE"
            },
            LEDGER_LOOKUPS.load(ORD),
            ledger_hits_json(),
            ledger_gate_state(),
            ledger_miss_state(),
            crate::registry::ledger().len(),
            crate::registry::ledger().faults.len(),
            UNPROVEN_DECLINES.load(ORD),
            UNPROVEN_FORMS_CLAIMED.load(ORD),
            unproven_forms_enabled_json(),
            shaders_list,
            shaders_digest,
            SESSION_DISCLOSURES.load(ORD),
            CLAIMED_FORMS_PROVEN.load(ORD),
            CLAIMED_FORMS_UNMEASURED.load(ORD),
            CLAIMED_FORMS_DIVERGENT.load(ORD),
            CLAIMED_FORMS_LEDGER_FAULTED.load(ORD),
            claimed_form_evidence(),
            SESSION_DISCLOSURE_WARNS.load(ORD),
            SESSION_DISCLOSURE_WARNS_TO_ORT.load(ORD),
            session_disclosure_channel(),
            equiv,
        )
    }

    /// One line for a log or a CI step summary.
    pub fn summary(&self) -> String {
        format!(
            "VulkanExecutionProvider counters: {} dispatch(es) executed across {} Compute call(s) \
             ({} failed); {} subgraph(s) compiled live, {} stub, from {} Compile call(s)",
            self.dispatches_executed,
            self.compute_calls,
            self.compute_failures,
            self.subgraphs_live,
            self.subgraphs_stub,
            self.compile_calls,
        )
    }
}

/// Write the snapshot to `$ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE`, if set.
///
/// Best-effort by design: a diagnostic that can fail a run it was only supposed to observe is a
/// liability. A write failure is logged at `warn` and otherwise ignored.
///
/// # Why this writes the *full* document
///
/// This is called from [`record_dispatches`] — on every successful dispatch — and it used to write
/// `snapshot().to_json()`, a strict subset of what [`dump_observations_if_requested`] writes.
/// Both write the same path, and last write wins. So on any run whose final write was a dispatch
/// rather than a teardown, every `alloc_*` and `pointers_*` key **vanished from the file**, and
/// the reader saw a well-formed document with the interesting half missing. Measured: the
/// `DEVICE_MEMORY=0` cells of the transfer-bound matrix produced counters files containing only
/// the ten base keys, while the `DEVICE_MEMORY=1` cells of the same matrix carried all thirty —
/// a difference that looks like a property of device memory and is purely a write-order artefact.
///
/// It also clobbered `model_output_equivalence` back to `UNMEASURED` after Trinity had written a
/// real verdict, for the same reason.
///
/// Emitting one document from one function removes the ordering dependence entirely. The extra
/// cost is a file read and ~30 atomic loads per dispatch, on a path that was already doing a file
/// write per dispatch.
pub fn dump_if_requested() {
    dump_observations_if_requested();
}

/// Append the allocator's pointer-observation tallies to the snapshot file.
///
/// Separate from [`dump_if_requested`] and deliberately *not* part of [`VulkanEpCounters`]: that
/// struct is a C ABI other processes read through [`fill`], and growing it to carry a diagnostic
/// would make every consumer's copy of the header a version problem. The JSON document has no such
/// constraint — `epctl`'s reader looks up keys by name and ignores ones it does not know — so the
/// numbers can live there additively.
///
/// This exists because the observations are only complete at teardown, and a process that has torn
/// down cannot print to a test's captured stdout. A file survives the process; a log line does not.
///
/// **Verdict preservation (§9.1.3):** Trinity's Python harness writes `model_output_equivalence`
/// (MATCH or DIVERGENT) to this file during the test, before the session is destroyed. This
/// function is called at transfer teardown — after the Python comparison but before the process
/// exits. To avoid overwriting Trinity's verdict with the default UNMEASURED, this function reads
/// the existing file's verdict first and preserves it in the rebuilt document.
pub fn dump_observations_if_requested() {
    let Some(path) = std::env::var_os(ENV_COUNTERS_FILE) else {
        return;
    };

    // Preserve any verdict that Trinity wrote during the comparison run.
    // Default to UNMEASURED if the file is absent or the field is missing.
    //
    // The RECORD travels with the token (§10.0 third metric amendment). Carrying the token
    // alone was a real defect: the artifact on disk read `"model_output_equivalence": "MATCH"`
    // with no `executed_by` anywhere in it, which is exactly the unattributed shape the
    // amendment forbids and `epctl` now refuses. A verdict that loses its attribution between
    // the process that derived it and the file a human reads is not a verdict about this EP.
    let existing_raw = std::fs::read_to_string(&path).ok();
    let existing_equiv = existing_raw
        .as_deref()
        .map(extract_equivalence)
        .unwrap_or(EQUIVALENCE_UNMEASURED);
    let existing_record = existing_raw
        .as_deref()
        .and_then(extract_equivalence_record)
        .map(str::to_string);

    let o = crate::allocator::ledger::snapshot();
    let t = crate::allocator::tally::snapshot();
    let st = staging::snapshot();
    let wc = weights::snapshot();
    // §10.0.1 R12 — frame provenance. `alloc_device_authoritative_spans` can only ever be non-zero
    // when the provider's buffers are on the device the engine dispatches on (§6.5). In any other
    // frame the event it counts cannot occur, so the artifact prints the JSON *string*
    // `"UNOBSERVABLE"` rather than the number 0: a reader doing arithmetic on it fails loudly
    // instead of quietly reading a structural pin as a measurement.
    //
    // R10 adds the second string. Even in the shared frame, the counter is only a measurement if
    // the predicate that feeds it has actually run on something; before it does, its zero is
    // `"UNWIRED"`. So the key has THREE JSON types across its life —
    // `"UNOBSERVABLE"` → `"UNWIRED"` → an integer — and each transition is a *type* change rather
    // than a value change. That is deliberate and it is the falsifier: an increment can forge a
    // number, and no increment can forge a type.
    let (frame, frame_device) = crate::allocator::tally::device_frame();
    let authoritative = if !crate::allocator::tally::device_authoritative_observable() {
        "\"UNOBSERVABLE\"".to_string()
    } else if !crate::allocator::tally::device_authoritative_wired() {
        "\"UNWIRED\"".to_string()
    } else {
        t.device_authoritative_spans.to_string()
    };
    let frame_sides = crate::allocator::tally::frame_sides_sentence();
    let frames_declared = crate::allocator::tally::frames_declared();
    let session_devices = crate::allocator::tally::session_devices();
    let alloc_device_index = crate::allocator::tally::allocator_device_index();
    let snap = snapshot();
    let mut doc = snap.to_json_with_equiv(existing_equiv);
    // Splice the observation keys in before the closing brace rather than appending after it, so
    // the file stays valid JSON for anything less forgiving than our own reader.
    if let Some(cut) = doc.rfind('}') {
        doc.truncate(cut);
        doc = doc.trim_end().trim_end_matches('\n').to_string();
        doc.push_str(&format!(
            ",\n  \"pointers_observed\": {},\n  \"pointers_host\": {},\n  \
             \"pointers_at_base\": {},\n  \"pointers_interior\": {},\n  \
             \"pointers_in_guard_band\": {},\n  \"pointers_use_after_free\": {},\n  \
             \"pointer_max_offset\": {},\n  \"alloc_allocations\": {},\n  \
             \"alloc_frees\": {},\n  \"alloc_bytes\": {},\n  \
             \"alloc_high_water_bytes\": {},\n  \"alloc_device_backed_spans\": {},\n  \
             \"alloc_staged_spans\": {},\n  \"alloc_staged_bytes\": {},\n  \
             \"alloc_allocators_released\": {},\n  \"alloc_allocators_live\": {},\n  \
             \"alloc_frees_after_release\": {},\n  \
             \"alloc_live_at_release_spans\": {},\n  \"alloc_live_at_release_bytes\": {},\n  \
             \"alloc_device_uploads\": {},\n  \"alloc_device_upload_bytes\": {},\n  \
             \"alloc_device_downloads\": {},\n  \"alloc_device_download_bytes\": {},\n  \
             \"alloc_unified_memory\": {},\n  \
             \"alloc_quarantine_peak_spans\": {},\n  \
             \"alloc_quarantine_retired\": {},\n  \
             \"alloc_device_buffer_binds\": {},\n  \
             \"alloc_failed_lookups\": {},\n  \
             \"alloc_device_authoritative_ceiling\": {},\n  \
             \"alloc_device_residency_evaluations\": {},\n  \
             \"alloc_device_authoritative_spans\": {},\n  \
             \"alloc_device_frame\": \"{}\",\n  \
             \"alloc_device_frame_device\": \"{}\",\n  \
             \"alloc_device_frames_declared\": {},\n  \
             \"alloc_device_frame_allocator_index\": \"{}\",\n  \
             \"alloc_device_frame_session_devices\": \"{}\",\n  \
             \"alloc_device_frame_sides\": \"{}\",\n  \
             \"session_staging_uploads\": {},\n  \
             \"session_staging_upload_bytes\": {},\n  \
             \"session_staging_upload_us\": {},\n  \
             \"session_staging_readbacks\": {},\n  \
             \"session_staging_readback_bytes\": {},\n  \
             \"session_staging_readback_us\": {},\n  \
             \"session_device_allocs\": {},\n  \
             \"session_device_frees\": {},\n  \
             \"session_device_bytes_in_use\": {},\n  \
             \"session_device_high_water_bytes\": {},\n  \
             \"weight_cache_release_calls\": {},\n  \
             \"weight_cache_release_buffers\": {},\n  \
             \"weight_cache_release_bytes\": {},\n  \
             \"weight_cache_bytes_resident\": {},\n  \
             \"counters_snapshot_writes\": {},\n  \
             \"counters_snapshot_write_failures\": {}\n}}\n",
            o.observed,
            o.host,
            o.at_base,
            o.interior,
            o.in_guard_band,
            o.use_after_free,
            o.max_offset,
            t.allocations,
            t.frees,
            t.bytes,
            t.high_water_bytes,
            t.device_backed_spans,
            t.staged_spans,
            t.staged_bytes,
            t.allocators_released,
            t.allocators_live,
            t.frees_after_release,
            t.live_at_release_spans,
            t.live_at_release_bytes,
            t.device_uploads,
            t.device_upload_bytes,
            t.device_downloads,
            t.device_download_bytes,
            t.unified_memory,
            t.quarantine_peak_spans,
            t.quarantine_retired,
            t.device_buffer_binds,
            t.failed_lookups,
            t.device_authoritative_ceiling,
            t.device_residency_evaluations,
            authoritative,
            frame,
            frame_device.replace('\\', "\\\\").replace('"', "\\\""),
            frames_declared,
            alloc_device_index
                .replace('\\', "\\\\")
                .replace('"', "\\\""),
            session_devices.replace('\\', "\\\\").replace('"', "\\\""),
            frame_sides.replace('\\', "\\\\").replace('"', "\\\""),
            st.0,
            st.1,
            st.2,
            st.3,
            st.4,
            st.5,
            wc.0,
            wc.1,
            wc.2,
            wc.3,
            wc.4,
            wc.5,
            wc.6,
            wc.7,
            SNAPSHOT_WRITES.load(ORD),
            SNAPSHOT_WRITE_FAILURES.load(ORD),
        ));
    }
    // Splice the preserved record back in, same technique, so the token and the frame that
    // says which world it is about stay in one artifact.
    if let Some(record) = existing_record.as_deref() {
        if let Some(cut) = doc.rfind('}') {
            doc.truncate(cut);
            doc = doc.trim_end().trim_end_matches('\n').to_string();
            doc.push_str(&format!(
                ",\n  \"{EQUIVALENCE_RECORD_KEY}\": {record}\n}}\n"
            ));
        }
    }
    if let Err(e) = std::fs::write(&path, doc) {
        SNAPSHOT_WRITE_FAILURES.fetch_add(1, ORD);
        log::warn!(
            "could not write EP observations to {}: {e}. The file on disk is now the PREVIOUS \
             snapshot, which is well-formed and is a prefix of this run; \
             counters_snapshot_write_failures says so in the next document that reaches the disk.",
            path.to_string_lossy()
        );
    } else {
        SNAPSHOT_WRITES.fetch_add(1, ORD);
    }
    // The trace is prose, so it goes beside the JSON rather than into it. It is written for the
    // same reason the JSON is: by the time these lines exist, ORT's logger has usually already
    // been torn down, so logging them reaches nobody.
    let trace = crate::allocator::ledger::trace_lines();
    if !trace.is_empty() {
        let mut p = std::path::PathBuf::from(&path);
        p.set_extension("trace.txt");
        let _ = std::fs::write(p, trace.join("\n") + "\n");
    }
}

/// Copy the current counters into a caller-provided buffer.
///
/// Returns the number of bytes written, or 0 if `out` is null or `out_bytes` is too small to carry
/// even `struct_size` + `abi_version`. Writing `min(out_bytes, size_of)` is what makes the struct
/// safe to extend: an old caller with a small buffer gets a correct prefix rather than a stomped
/// stack.
///
/// # Safety
/// `out` must be null, or writable for `out_bytes` bytes.
pub unsafe fn fill(out: *mut c_void, out_bytes: usize) -> usize {
    const HEADER: usize = 8; // struct_size + abi_version
    if out.is_null() || out_bytes < HEADER {
        return 0;
    }
    let snap = snapshot();
    let n = out_bytes.min(std::mem::size_of::<VulkanEpCounters>());
    // SAFETY: `out` is writable for `out_bytes` bytes (caller contract) and `n <= out_bytes`.
    // `snap` is a live local of exactly `size_of::<VulkanEpCounters>()` bytes and `n` is at most
    // that, so the read side is in bounds too. The two regions cannot overlap: `snap` is on our
    // stack frame, which the caller has no pointer to.
    unsafe {
        std::ptr::copy_nonoverlapping((&raw const snap).cast::<u8>(), out.cast::<u8>(), n);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lost device must be visible in the artifact a harness reads, not only in a log line.
    ///
    /// Found 2026-08-02: at ctx 512 the device was lost, ORT re-ran the island on the CPU EP,
    /// the process exited 0, and a *complete* counters file was written. Differencing that pair
    /// against its sibling produced an apparent 6.7% KV saving that was an observation ending
    /// early. `compute_failures` alone does not separate "a dispatch failed" from "the device is
    /// gone, and every number in this file describes a partial run"; `device_losses` does, and it
    /// has to be in the JSON because the JSON is what the probes read.
    #[test]
    fn a_lost_device_is_recorded_and_reaches_the_json_artifact() {
        // Process-global statics: serialise with every other test that touches them.
        let _g = crate::allocator::ledger::test_lock();
        reset();
        assert_eq!(device_losses(), 0, "reset must clear the device-loss count");
        assert!(
            snapshot().to_json().contains("\"device_losses\""),
            "the key must be present at zero: a probe cannot refuse on a key that only \
             appears once the failure has happened"
        );
        record_device_lost();
        record_device_lost();
        assert_eq!(device_losses(), 2);
        assert_eq!(snapshot().device_losses, 2);
        assert!(
            snapshot().to_json().contains("\"device_losses\": 2"),
            "got: {}",
            snapshot().to_json()
        );
        reset();
    }

    /// The bytes that dominate the run must be counted with **no flag set**.
    ///
    /// This is the falsifier for the whole "verify residency on bytes" plan: if the recording hook
    /// is behind the tracer's `active()` guard, the default run records nothing and the sweep that
    /// is supposed to prove residency is taken on an instrument that was never on.
    #[test]
    fn staging_bytes_are_recorded_without_any_tracing_flag() {
        // Process-global statics: serialise with every other test that touches them.
        let _g = crate::allocator::ledger::test_lock();
        staging::reset();
        // A tracer with no ONNXRUNTIME_EP_VULKAN_TRACE and no verbose is inert by construction;
        // the process-wide tracer in a test run is exactly that.
        crate::trace::tracer().record_transfer(
            crate::trace::Transfer::Upload,
            4096,
            std::time::Duration::from_micros(25),
        );
        let (n, bytes, us, ..) = staging::snapshot();
        assert_eq!((n, bytes), (1, 4096), "staging upload was not counted");
        assert!(us >= 25, "staging time was not counted; got {us}");
        staging::reset();
    }

    /// Zero is the value residency produces AND the value a moved hook produces. The artifact must
    /// refuse to call it a win. R7.
    #[test]
    fn zero_staging_bytes_refuses_to_claim_residency() {
        // Process-global statics: serialise with every other test that touches them.
        let _g = crate::allocator::ledger::test_lock();
        staging::reset();
        let s = staging::sentence(661);
        assert!(s.contains("CANNOT distinguish"), "got: {s}");
        assert!(
            s.contains("cmd_upload"),
            "must name the independent bracket; got: {s}"
        );
        assert!(
            !s.contains("resident (the win)."),
            "must not assert the win; got: {s}"
        );
    }

    /// The two upload accountings count different copies and the artifact must forbid adding them.
    #[test]
    fn the_staging_sentence_separates_itself_from_the_allocator_upload_counters() {
        // Process-global statics: serialise with every other test that touches them.
        let _g = crate::allocator::ledger::test_lock();
        staging::reset();
        staging::on_upload(1024 * 1024, 1000);
        let s = staging::sentence(2);
        assert!(s.contains("alloc_device_upload_"), "got: {s}");
        assert!(s.contains("must never be added"), "got: {s}");
        assert!(s.contains("per Compute call"), "got: {s}");
        staging::reset();
    }

    // The `session_staging_*` keys are asserted inside
    // `the_dispatch_path_dump_carries_the_allocator_keys_too` rather than in a test of their own:
    // that test already owns `ENV_COUNTERS_FILE`, and a second writer to one artifact is the bug
    // this module exists to remember.

    /// The dispatch-path dump must not write a poorer document than the teardown path.
    ///
    /// Both write the same file and last-write-wins, so a subset document on the hot path erases
    /// the `alloc_*`/`pointers_*` keys on any run that ends with a dispatch. This is the falsifier:
    /// if `dump_if_requested` ever stops emitting the full document, this goes red.
    ///
    /// It also covers the `session_staging_*` keys, deliberately in the SAME test rather than a
    /// second one: these tests share one process-wide env var and one output path, and two tests
    /// writing one artifact is the defect this file exists to remember.
    #[test]
    fn the_dispatch_path_dump_carries_the_allocator_keys_too() {
        // Process-global statics: serialise with every other test that touches them.
        let _g = crate::allocator::ledger::test_lock();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("counters_dump_parity_test.json");
        std::fs::remove_file(&path).ok();

        staging::on_upload(2048, 7);

        // SAFETY: single-threaded test; the variable is removed before returning on every path.
        unsafe { std::env::set_var(ENV_COUNTERS_FILE, &path) };
        dump_if_requested();
        // SAFETY: see above.
        unsafe { std::env::remove_var(ENV_COUNTERS_FILE) };

        let doc = std::fs::read_to_string(&path).expect("dump must have written the file");
        std::fs::remove_file(&path).ok();
        for key in [
            "alloc_allocations",
            "alloc_device_backed_spans",
            "alloc_device_authoritative_spans",
            "alloc_quarantine_retired",
            "alloc_device_frame",
            "alloc_device_frame_device",
            "pointers_use_after_free",
            "session_staging_uploads",
            "session_staging_upload_bytes",
            "session_staging_upload_us",
            "session_staging_readbacks",
            "session_staging_readback_bytes",
            "session_staging_readback_us",
        ] {
            assert!(
                doc.contains(key),
                "the per-dispatch dump dropped `{key}`; it overwrites the teardown document, so \
                 the key disappears from any run that ends with a dispatch. Document was:\n{doc}"
            );
        }
        assert!(
            !doc.contains("\"session_staging_upload_bytes\": 0,"),
            "the staging bytes reached the counters module but not the document; got:\n{doc}"
        );
    }

    /// **R12 / R10 falsifier for the §8.9.7 disclosure observable.**
    ///
    /// Two things at once. First: with no session-creation disclosure in this frame,
    /// `claimed_form_evidence` must read `UNOBSERVABLE` and never `ALL-PROVEN` — a claim set that
    /// was never assembled is not a claim set that came back clean, and the substitution of the
    /// second for the first is the §6.5 coincidence. Second: the token's content must **vary with
    /// its input**, which is the only thing that distinguishes a wired observable from a constant.
    #[test]
    fn the_claimed_form_evidence_token_is_unobservable_before_it_is_anything_else() {
        let _g = crate::allocator::ledger::test_lock();
        reset();
        let doc = snapshot().to_json();
        assert!(
            doc.contains("\"claimed_form_evidence\": \"UNOBSERVABLE\""),
            "no disclosure has run, so nothing is known about any claim set. Got:\n{doc}"
        );
        assert!(
            doc.contains("\"session_disclosure_channel\": \"UNOBSERVABLE\""),
            "a channel with no traffic is not a proven channel. Got:\n{doc}"
        );

        record_session_disclosure(3, 0, 0, 0, false, false);
        let doc = snapshot().to_json();
        assert!(
            doc.contains("\"claimed_form_evidence\": \"ALL-PROVEN\""),
            "three proven forms and nothing else should read ALL-PROVEN. Got:\n{doc}"
        );
        assert!(
            doc.contains("\"session_disclosure_channel\": \"UNOBSERVABLE\""),
            "no WARN was emitted, so the WARN channel is still unexercised. Got:\n{doc}"
        );

        record_session_disclosure(1, 1, 0, 0, true, true);
        let doc = snapshot().to_json();
        assert!(
            doc.contains("\"claimed_form_evidence\": \"UNMEASURED-PRESENT\""),
            "an unmeasured claimed form must surface. Got:\n{doc}"
        );
        assert!(
            doc.contains("\"session_disclosure_channel\": \"ORT_SINK\""),
            "the WARN reached ORT's logger and must say so. Got:\n{doc}"
        );

        record_session_disclosure(0, 0, 1, 0, true, false);
        let doc = snapshot().to_json();
        assert!(
            doc.contains("\"claimed_form_evidence\": \"DIVERGENT-PRESENT\""),
            "a divergent form must not hide behind an unmeasured one. Got:\n{doc}"
        );
        assert!(
            doc.contains("\"session_disclosure_channel\": \"PRIVATE_LOG_ONLY\""),
            "a WARN that did not reach ORT's logger is invisible to the audience that matters, \
             and the artifact must say so. Got:\n{doc}"
        );

        record_session_disclosure(0, 0, 0, 1, true, true);
        let doc = snapshot().to_json();
        assert!(
            doc.contains("\"claimed_form_evidence\": \"LEDGER-FAULTED\""),
            "R13: an instrument error outranks the findings it makes unreadable. Got:\n{doc}"
        );
        reset();
    }

    /// **R12 falsifier.** `alloc_device_authoritative_spans` may never appear as the number `0` in
    /// a frame where the event it counts cannot occur (§6.5 split device, or no provider at all).
    ///
    /// This goes red if anyone "simplifies" the artifact back to always emitting a number — which
    /// is exactly how a structural pin gets read as a measurement.
    #[test]
    fn a_pinned_authoritative_counter_reports_unobservable_and_never_zero() {
        let _g = crate::allocator::ledger::test_lock();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("counters_r12_frame_test.json");
        std::fs::remove_file(&path).ok();

        crate::allocator::tally::reset_for_test();

        // Frame 1: no provider at all.
        // SAFETY: single-threaded test; removed before returning on every path.
        unsafe { std::env::set_var(ENV_COUNTERS_FILE, &path) };
        dump_observations_if_requested();
        let doc = std::fs::read_to_string(&path).expect("dump must have written the file");
        assert!(
            doc.contains("\"alloc_device_authoritative_spans\": \"UNOBSERVABLE\""),
            "with no device-memory provider the counter is pinned, not measured; got:\n{doc}"
        );
        assert!(
            doc.contains("\"alloc_device_frame\": \"OFF\""),
            "the frame must be reported even when there is none; got:\n{doc}"
        );

        // Frame 2: a second VkDevice — the §6.5 defect. Still pinned.
        crate::allocator::tally::set_only_frame_for_test(
            crate::allocator::tally::FRAME_SPLIT,
            "Some Other Device",
        );
        dump_observations_if_requested();
        let doc = std::fs::read_to_string(&path).expect("dump must have written the file");
        assert!(
            doc.contains("\"alloc_device_authoritative_spans\": \"UNOBSERVABLE\""),
            "on a split device a compute dispatch cannot bind these buffers, so 0 is not a \
             measurement; got:\n{doc}"
        );
        assert!(
            doc.contains("\"alloc_device_frame\": \"SPLIT-DEVICE\""),
            "the split frame must be named in the artifact; got:\n{doc}"
        );

        // Frame 2b: the split frame must NAME BOTH SIDES, not merely say that they differ.
        // A reader holding only "SPLIT-DEVICE" cannot tell a selector-1 run from a selector-0 one,
        // which is the third time two index spaces have produced a correct counter about the
        // wrong situation on this project.
        crate::allocator::tally::set_allocator_device_index(0);
        crate::allocator::tally::note_session_device(1, "The Session's Device");
        dump_observations_if_requested();
        let doc = std::fs::read_to_string(&path).expect("dump must have written the file");
        assert!(
            doc.contains("\"alloc_device_frame_session_devices\": \"1=The Session's Device\""),
            "the split artifact must name the SESSION side's device; got:\n{doc}"
        );
        assert!(
            doc.contains("\"alloc_device_frame_allocator_index\": \"0\""),
            "the split artifact must name which index the ALLOCATOR side was stood up for; \
             got:\n{doc}"
        );
        assert!(
            doc.contains("ALLOCATOR side: 'Some Other Device'"),
            "the frame-sides sentence must name the allocator's device in prose; got:\n{doc}"
        );

        // Frame 3: §6.5 satisfied — observable at last, and STILL not a measurement, because the
        // residency predicate has not run on anything. R10's third state, in the artifact's type
        // system: `"UNWIRED"` is a JSON string, so arithmetic on it fails loudly.
        crate::allocator::tally::set_only_frame_for_test(
            crate::allocator::tally::FRAME_SHARED,
            "The Session's Device",
        );
        dump_observations_if_requested();
        let doc = std::fs::read_to_string(&path).expect("dump must have written the file");
        assert!(
            doc.contains("\"alloc_device_authoritative_spans\": \"UNWIRED\""),
            "a shared frame with zero residency evaluations is UNWIRED, not a measured 0; \
             got:\n{doc}"
        );
        assert!(
            doc.contains("\"alloc_device_residency_evaluations\": 0"),
            "the evaluation count is what distinguishes UNWIRED from measured and must be in the \
             artifact; got:\n{doc}"
        );

        // Frame 4: the predicate runs on one span and finds it staged. NOW the zero is evidence,
        // and the key's JSON type changes from string to integer. An increment cannot forge a
        // type change, which is why this transition is proof rather than assertion.
        crate::allocator::tally::on_residency_evaluated(false);
        dump_observations_if_requested();
        let doc = std::fs::read_to_string(&path).expect("dump must have written the file");
        assert!(
            doc.contains("\"alloc_device_authoritative_spans\": 0"),
            "once the predicate has run, its zero is a measurement and must be emitted as a \
             number; got:\n{doc}"
        );
        assert!(
            doc.contains("\"alloc_device_residency_evaluations\": 1"),
            "the measured zero must carry the evaluation count that earns it; got:\n{doc}"
        );

        // SAFETY: see above.
        unsafe { std::env::remove_var(ENV_COUNTERS_FILE) };
        std::fs::remove_file(&path).ok();
        crate::allocator::tally::reset_for_test();
    }

    /// **The frame is a property of the POPULATION, not of the last writer.**
    ///
    /// `PROVIDERS` in `vk::host_device_memory` is a map keyed by factory device index, so a single
    /// process can stand up a device-memory provider for device 0 and another for device 1 — two
    /// ORT sessions on two GPUs is an ordinary thing for a user to do. Both call `set_device_frame`.
    /// The allocator tallies they feed are process-global, so the artifact ends up with **one
    /// frame label over a population drawn from two frames**, and which label wins depends on
    /// which session registered last.
    ///
    /// That is an R12 defect and not a test defect: in a mixed process the old code would emit
    /// `alloc_device_authoritative_spans` as a *number* — a measurement — whenever the SHARED
    /// provider happened to register second, even though every span from the SPLIT provider is
    /// in a frame where the counted event cannot occur. R12 says a counter whose event cannot
    /// occur in its frame reports `UNOBSERVABLE`; a population spanning two frames has no single
    /// frame in which to ask the question, so the honest answer is the same one.
    ///
    /// This is the falsifier for the claim "the frame is confined". It runs the two registrations
    /// concurrently, because the defect's signature is *scheduling-dependent output*: before the
    /// fix this test reported `SHARED` or `SPLIT-DEVICE` depending on which thread won, and now
    /// it reports `MIXED` under every interleaving.
    #[test]
    fn two_frames_in_one_process_are_never_reported_as_one_frame() {
        use crate::allocator::tally;
        let _g = crate::allocator::ledger::test_lock();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("counters_r12_mixed_frame_test.json");
        std::fs::remove_file(&path).ok();
        tally::reset_for_test();

        // Two providers, two devices, one process, no ordering between them.
        std::thread::scope(|s| {
            s.spawn(|| {
                tally::set_device_frame(tally::FRAME_SHARED, "Device A");
                tally::on_residency_evaluated(false);
            });
            s.spawn(|| {
                tally::set_device_frame(tally::FRAME_SPLIT, "Device B");
                tally::on_residency_evaluated(false);
            });
        });

        // SAFETY: single-threaded again by the time this runs; removed on the way out.
        unsafe { std::env::set_var(ENV_COUNTERS_FILE, &path) };
        dump_observations_if_requested();
        let doc = std::fs::read_to_string(&path).expect("dump must have written the file");
        // Leave the artifact behind. R10: the falsifier for "the frame is a property of the
        // population, not of the last writer" is a document the mechanism produced.
        std::fs::write(dir.join("mixed_frame_dump.json"), &doc).ok();
        // SAFETY: see above.
        unsafe { std::env::remove_var(ENV_COUNTERS_FILE) };
        std::fs::remove_file(&path).ok();

        assert!(
            doc.contains("\"alloc_device_frame\": \"MIXED\""),
            "two providers declared two different frames in this process, so no single frame \
             describes these numbers; reporting either one of them as THE frame is how a \
             selector-1 population gets read as a selector-0 measurement. Got:\n{doc}"
        );
        assert!(
            doc.contains("\"alloc_device_authoritative_spans\": \"UNOBSERVABLE\""),
            "the residency screen ran (so the counter is wired) but the population spans two \
             frames, and there is no frame in which to ask whether the event could occur. R12 \
             says that is UNOBSERVABLE, never a number. Got:\n{doc}"
        );
        assert!(
            doc.contains("\"alloc_device_frames_declared\": 2"),
            "the artifact must say how many frames it is standing on, or MIXED is an assertion \
             rather than a count; got:\n{doc}"
        );
        assert!(
            doc.contains("Device A") && doc.contains("Device B"),
            "a mixed frame must NAME the frames it mixed — `MIXED` alone is the same detection \
             -without-description that made `SPLIT-DEVICE` unreadable. Got:\n{doc}"
        );

        tally::reset_for_test();
    }

    /// The counters are process-wide statics, so tests that touch them must not run concurrently
    /// with each other. One test, sequenced by hand, is simpler than a mutex and cannot deadlock.
    #[test]
    fn counters_record_what_they_claim_to_record() {
        // Process-global statics, hand-sequenced below. Other tests in this binary drive the real
        // `Compute` entry point and legitimately record broken commitments while this one runs;
        // without the shared lock this test reads their events as its own and fails intermittently
        // — a flake that is really a frame error, since the counters are process-wide and the
        // assertions here are about *this* sequence.
        let _g = crate::allocator::ledger::test_lock();
        reset();
        assert_eq!(snapshot().dispatches_executed, 0);

        record_compile_call();
        record_subgraph(true);
        record_subgraph(false);
        record_compute_call();
        record_dispatches(3);
        record_compute_call();
        record_compute_failure();

        let s = snapshot();
        assert_eq!(s.compile_calls, 1);
        assert_eq!(s.subgraphs_live, 1);
        assert_eq!(s.subgraphs_stub, 1);
        assert_eq!(s.compute_calls, 2);
        assert_eq!(s.compute_failures, 1);
        assert_eq!(s.dispatches_executed, 3);
        assert_eq!(s.abi_version, COUNTERS_ABI_VERSION);
        assert_eq!(
            s.struct_size as usize,
            std::mem::size_of::<VulkanEpCounters>()
        );

        // A zero-dispatch record must not move the number. `Compute` calls this unconditionally,
        // so "success with no kernels" must not be able to inflate the criterion-8 count.
        record_dispatches(0);
        assert_eq!(snapshot().dispatches_executed, 3);

        // The JSON is what `epctl --check-counters` parses; keep the two in step.
        let json = snapshot().to_json();
        assert!(json.contains("\"dispatches_executed\": 3"));
        assert!(json.contains("\"compute_failures\": 1"));
        assert!(
            json.contains("\"model_output_equivalence\": \"UNMEASURED\""),
            "to_json() must include UNMEASURED by default — an uncompared run says so explicitly"
        );

        // A short buffer gets a correct prefix, not a stomp.
        let mut buf = [0u8; 8];
        // SAFETY: `buf` is 8 writable bytes and we say so.
        let n = unsafe { fill(buf.as_mut_ptr().cast(), buf.len()) };
        assert_eq!(n, 8);
        assert_eq!(
            u32::from_ne_bytes(buf[0..4].try_into().expect("4 bytes")) as usize,
            std::mem::size_of::<VulkanEpCounters>(),
            "a short reader still learns how big the real struct is"
        );

        // A buffer too small to carry the header is refused rather than partially filled.
        let mut tiny = [0u8; 4];
        // SAFETY: `tiny` is 4 writable bytes.
        assert_eq!(unsafe { fill(tiny.as_mut_ptr().cast(), tiny.len()) }, 0);
        // SAFETY: null is explicitly permitted and returns 0.
        assert_eq!(unsafe { fill(std::ptr::null_mut(), 64) }, 0);

        // ---------------------------------------------------------------------------------
        // RAI Ruling 2 / RAI-011: the two states that must never print the same digit.
        // ---------------------------------------------------------------------------------
        reset();

        // No Compute has run, so no claim this EP made can have been broken: the event is out of
        // frame, and R12 says that reads UNOBSERVABLE and never 0. This is what stops a CPU-only
        // fallback run — the 2026-07-30 specimen — from producing the clean-run token.
        let json = snapshot().to_json();
        assert!(
            json.contains("\"broken_commitments\": \"UNOBSERVABLE\""),
            "a run with zero Compute calls cannot have broken a commitment; got:\n{json}"
        );
        assert!(
            json.contains("\"broken_commitment_warn_channel\": \"UNOBSERVABLE\""),
            "an unexercised channel is not a working channel; got:\n{json}"
        );

        // One Compute later, the zero is a result: the negative polarity's assertion.
        record_compute_call();
        let json = snapshot().to_json();
        assert!(
            json.contains("\"broken_commitments\": 0"),
            "once a claim has been executed, 0 broken commitments is a measurement; got:\n{json}"
        );

        // The WARN fired and reached ORT's sink.
        record_broken_commitment(true);
        let json = snapshot().to_json();
        assert!(json.contains("\"broken_commitments\": 1"), "{json}");
        assert!(
            json.contains("\"broken_commitment_warns_to_ort_sink\": 1"),
            "{json}"
        );
        assert!(
            json.contains("\"broken_commitment_warn_channel\": \"ORT_SINK\""),
            "{json}"
        );

        // A WARN that reached nobody must not be reported as a delivered disclosure.
        record_broken_commitment(false);
        let json = snapshot().to_json();
        assert!(
            json.contains("\"broken_commitment_warn_channel\": \"PRIVATE_LOG_ONLY\""),
            "a WARN that never reached ORT's logger must say so; got:\n{json}"
        );

        // RAI-011: bypassed and all-rejected are opposite facts and printed the same digit.
        reset();
        let json = snapshot().to_json();
        assert!(
            json.contains("\"viable_islands_retained\": \"UNWIRED\""),
            "no cluster has reached the decision point; got:\n{json}"
        );

        record_net_benefit_decision(false); // single-cluster bypass — Phi-3.5's fused island
        let json = snapshot().to_json();
        assert!(
            json.contains("\"viable_islands_retained\": \"UNOBSERVABLE\""),
            "a bypassed gate cannot retain an island, so its 0 is out of frame; got:\n{json}"
        );
        assert!(
            json.contains("\"net_benefit_gate\": \"BYPASSED\""),
            "{json}"
        );

        record_net_benefit_decision(true); // the gate actually ran on a cluster and rejected it
        let json = snapshot().to_json();
        assert!(
            json.contains("\"viable_islands_retained\": 0"),
            "once the gate has run, 0 retained is a rejection and must be a number; got:\n{json}"
        );
        assert!(json.contains("\"net_benefit_gate\": \"MIXED\""), "{json}");
        assert!(
            json.contains("\"net_benefit_gate_evaluations\": 1"),
            "{json}"
        );
        assert!(json.contains("\"net_benefit_gate_bypasses\": 1"), "{json}");

        record_viable_islands_retained(2);
        let json = snapshot().to_json();
        assert!(json.contains("\"viable_islands_retained\": 2"), "{json}");

        reset();
        assert_eq!(snapshot().compute_calls, 0);
    }

    /// An override count says *that* the gate was overruled; it does not say *what* was overruled.
    ///
    /// The wiring census reported `sole_island_overrides=1` with no way to learn whether the gate
    /// had said "too small" or "transfer dominated". Those are different findings — the first is
    /// the designed behaviour of a one-node graph, the second would mean the economics arm is
    /// declining a graph we ship. This pins that they no longer read alike.
    #[test]
    fn an_override_reports_which_verdict_it_overrode() {
        reset();
        let json = snapshot().to_json();
        assert!(
            json.contains("\"net_benefit_override_reason\": \"UNOBSERVABLE\""),
            "no override happened, so there is no reason to report and it must not read as a \
             verdict (R12); got:\n{json}"
        );

        record_net_benefit_decision(true);
        record_sole_island_override(Some(OverriddenVerdict::TooSmall));
        let json = snapshot().to_json();
        assert!(
            json.contains("\"net_benefit_override_reason\": \"TOO_SMALL\""),
            "{json}"
        );
        assert!(
            json.contains("\"net_benefit_sole_island_overrides\": 1"),
            "{json}"
        );

        // A second override with a different reason must not overwrite the first: the artifact
        // has to say both happened, not whichever ran last.
        record_net_benefit_decision(true);
        record_sole_island_override(Some(OverriddenVerdict::TransferDominated));
        let json = snapshot().to_json();
        assert!(
            json.contains("\"net_benefit_override_reason\": \"MIXED\""),
            "two overrides with different reasons must not collapse onto one token; got:\n{json}"
        );

        reset();
        record_net_benefit_decision(true);
        record_sole_island_override(None);
        let json = snapshot().to_json();
        assert!(
            json.contains("\"net_benefit_override_reason\": \"UNRECORDED\""),
            "an override whose reason never arrived must say so rather than default to a \
             plausible verdict; got:\n{json}"
        );
        reset();
    }

    /// `UNOBSERVABLE` and `TOO_SMALL` are tokens, not numbers, so a reader cannot average them,
    /// sum them, or mistake an absent override for a zero-valued one.
    #[test]
    fn the_override_reason_is_typed_so_arithmetic_on_it_fails_loudly() {
        reset();
        assert_eq!(net_benefit_override_reason(), "UNOBSERVABLE");
        assert!(net_benefit_override_reason().parse::<u64>().is_err());
        record_net_benefit_decision(true);
        record_sole_island_override(Some(OverriddenVerdict::TransferDominated));
        assert!(net_benefit_override_reason().parse::<u64>().is_err());
        reset();
    }

    /// The verdict vocabulary is duplicated in `tests/ops/_verdict.py`. Duplicated vocabularies
    /// drift, and a drifted one fails *quietly*: `extract_equivalence` maps any token it does not
    /// recognise to `UNMEASURED`, so a token the Python side renamed would arrive here as "no
    /// comparison was performed" — a red turned into a shrug, which is precisely the two-token
    /// disease R13 names. This test reads the Python file and asserts the two sets are equal.
    ///
    /// A missing or unparsable file is an **instrument error**, never a pass: it panics with a
    /// message saying so. A cross-language check that silently skips when it cannot find its
    /// subject is not a check.
    #[test]
    fn verdict_vocabulary_cannot_drift_from_the_python_harness() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("ops")
            .join("_verdict.py");
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "ERROR(instrument=verdict_vocabulary_unreadable): {} — {e}. This test cannot \
                 report a vocabulary as undrifted when it never read the other half of it.",
                path.display()
            )
        });

        // `VERDICT_MATCH: str = "MATCH"` -> "MATCH". Deliberately literal: an alias or a computed
        // name would not be picked up, and that is correct — the Rust side can only compare
        // against literals it can see.
        let mut python: Vec<String> = src
            .lines()
            .filter_map(|line| {
                let rest = line.strip_prefix("VERDICT_")?;
                let (_, value) = rest.split_once("= \"")?;
                Some(value.trim_end_matches(['"', ' ']).to_string())
            })
            .collect();
        python.sort();
        assert!(
            !python.is_empty(),
            "ERROR(instrument=verdict_vocabulary_unparsable): found no `VERDICT_* : str = \"...\"` \
             definitions in {}. The file's shape changed and this test is now reading nothing, \
             which would let every future drift pass.",
            path.display()
        );

        let mut rust = vec![
            EQUIVALENCE_MATCH.to_string(),
            EQUIVALENCE_DIVERGENT.to_string(),
            EQUIVALENCE_UNMEASURED.to_string(),
            EQUIVALENCE_UNATTRIBUTED.to_string(),
            EQUIVALENCE_SPLIT_FRAME.to_string(),
        ];
        rust.sort();

        assert_eq!(
            python, rust,
            "the verdict vocabulary drifted between tests/ops/_verdict.py and counters.rs. \
             Every token the harness can write must be a token this crate can read back, or the \
             unreadable one silently becomes UNMEASURED."
        );

        // The keys the token and the record travel under must match too: a correct token written
        // under a key nobody reads is the same silence with an extra step.
        for (rust_key, python_name) in [
            (EQUIVALENCE_KEY, "EQUIVALENCE_KEY"),
            (EQUIVALENCE_RECORD_KEY, "EQUIVALENCE_RECORD_KEY"),
        ] {
            let needle = format!("{python_name}: str = \"{rust_key}\"");
            assert!(
                src.contains(&needle),
                "tests/ops/_verdict.py does not define {python_name} as {rust_key:?}; the JSON \
                 key carrying the verdict drifted between the writer and the reader"
            );
        }
    }

    #[test]
    fn extract_equivalence_parses_the_five_states() {
        let match_doc = snapshot().to_json_with_equiv(EQUIVALENCE_MATCH);
        let div_doc = snapshot().to_json_with_equiv(EQUIVALENCE_DIVERGENT);
        let unm_doc = snapshot().to_json_with_equiv(EQUIVALENCE_UNMEASURED);
        let unattr_doc = snapshot().to_json_with_equiv(EQUIVALENCE_UNATTRIBUTED);
        let split_doc = snapshot().to_json_with_equiv(EQUIVALENCE_SPLIT_FRAME);
        // Build a document that physically lacks the field (old snapshot format).
        // to_json() writes `"model_output_equivalence": "UNMEASURED"` — strip both key and value.
        let without_field = {
            let raw = snapshot().to_json();
            let key_prefix = format!(",\n  \"{EQUIVALENCE_KEY}\"");
            if let Some(pos) = raw.find(&key_prefix) {
                // Remove key through the closing quote of the value.
                let after_key = &raw[pos + key_prefix.len()..];
                let value_end = after_key.find('\n').unwrap_or(after_key.len());
                format!("{}{}", &raw[..pos], &after_key[value_end..])
            } else {
                raw
            }
        };

        assert_eq!(extract_equivalence(&match_doc), EQUIVALENCE_MATCH);
        assert_eq!(extract_equivalence(&div_doc), EQUIVALENCE_DIVERGENT);
        assert_eq!(extract_equivalence(&unm_doc), EQUIVALENCE_UNMEASURED);

        // §10.0 third metric amendment (2026-07-31): two states more. UNATTRIBUTED must not
        // fold into DIVERGENT — a run we could not attribute is not a run that disagreed —
        // and SPLIT-FRAME must not fold into either, because it is a statement about the
        // instruments and not about the arithmetic.
        assert_eq!(extract_equivalence(&unattr_doc), EQUIVALENCE_UNATTRIBUTED);
        assert_ne!(extract_equivalence(&unattr_doc), EQUIVALENCE_DIVERGENT);
        assert_ne!(extract_equivalence(&unattr_doc), EQUIVALENCE_UNMEASURED);
        assert_eq!(extract_equivalence(&split_doc), EQUIVALENCE_SPLIT_FRAME);
        assert_ne!(extract_equivalence(&split_doc), EQUIVALENCE_UNATTRIBUTED);

        // Mutation control: a parser that returned a constant would satisfy every equality
        // above if that constant were UNMEASURED, so prove the five tokens are five values.
        let seen = [
            extract_equivalence(&match_doc),
            extract_equivalence(&div_doc),
            extract_equivalence(&unm_doc),
            extract_equivalence(&unattr_doc),
            extract_equivalence(&split_doc),
        ];
        let mut uniq = seen.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            5,
            "five tokens must parse to five distinct values"
        );
        // Absence must be treated the same as UNMEASURED (R7: absence ≠ negative).
        assert_eq!(
            extract_equivalence(&without_field),
            EQUIVALENCE_UNMEASURED,
            "a snapshot without the field predates the verdict; absence = UNMEASURED"
        );
        assert_eq!(extract_equivalence("{}"), EQUIVALENCE_UNMEASURED);
    }

    /// §10.0 third metric amendment: the RECORD must survive the teardown rebuild, not just
    /// the token.
    ///
    /// The specimen this test was written from: a real device-0 Phi-3.5 run on 2026-07-31
    /// left `bench/results/counters-phi35-dev0.json` reading
    /// `"model_output_equivalence": "MATCH"` with `model_output_equivalence_record: null` —
    /// the Python harness wrote both keys, `dump_observations_if_requested` rebuilt the file
    /// and carried only the token. An unattributed MATCH on disk, produced by a correctly
    /// attributed run. Defect C's shape (two writers, one artifact) applied to a caveat.
    #[test]
    fn the_verdict_record_survives_a_rebuild_and_is_not_confused_by_nesting() {
        let record = "{ \"verdict\": \"MATCH\", \"executed_by\": { \
                      \"VulkanExecutionProvider\": 3, \"CPUExecutionProvider\": 30 }, \
                      \"detail\": \"a } brace inside a string must not end the object\" }";
        let doc = format!(
            "{{\n  \"{EQUIVALENCE_KEY}\": \"MATCH\",\n  \
             \"{EQUIVALENCE_RECORD_KEY}\": {record},\n  \"dispatches_executed\": 3883\n}}\n"
        );

        let got = extract_equivalence_record(&doc).expect("the record must be found");
        assert_eq!(
            got, record,
            "nested objects and braces-in-strings must not truncate it"
        );
        assert!(got.contains("VulkanExecutionProvider"));

        // Absence is None, not an empty object and not a panic: a snapshot predating the
        // amendment simply has no record, and that is the UNATTRIBUTED case, not a crash.
        let bare = snapshot().to_json_with_equiv(EQUIVALENCE_MATCH);
        assert!(
            extract_equivalence_record(&bare).is_none(),
            "a MATCH with no record must read as absent so the gate can refuse it"
        );
        assert!(extract_equivalence_record("{}").is_none());
        assert!(
            extract_equivalence_record(&format!("{{\"{EQUIVALENCE_RECORD_KEY}\": null}}"))
                .is_none(),
            "a null record is not an object and must not be spliced back as one"
        );
    }
    /// **RAI-008(d)** — `ledger_miss` moves with its input, and never spells two states alike.
    ///
    /// R12's shape: the artifact must distinguish *no lookup happened here* from *the ledger has
    /// no proof for this form* from *the ledger itself is unreadable*. This drives all three and
    /// asserts three different tokens come out, which is the assertion a single `bool` could not
    /// have supported however carefully it was read.
    #[test]
    fn the_ledger_miss_token_names_which_of_three_things_happened() {
        use crate::registry::LedgerLookup as L;
        let _g = crate::allocator::ledger::test_lock();

        reset();
        assert_eq!(
            ledger_miss_state(),
            "NEVER-ATTEMPTED",
            "a run that consulted the ledger zero times has not found a form missing"
        );
        assert!(snapshot().to_json().contains("\"ledger_miss\": \"NEVER-ATTEMPTED\""));

        reset();
        record_ledger_lookup(L::Hit);
        record_ledger_lookup(L::Hit);
        assert_eq!(ledger_miss_state(), "HIT", "no miss occurred");
        assert_eq!(snapshot().ledger_hits, 2);

        reset();
        record_ledger_lookup(L::Hit);
        record_ledger_lookup(L::KeyAbsent);
        assert_eq!(
            ledger_miss_state(),
            "KEY-ABSENT",
            "one form absent from a healthy ledger is a finding about that form"
        );
        // The unconditional half still counts it: a miss is a lookup.
        assert_eq!(snapshot().proven_key_lookups, 2);
        assert_eq!(snapshot().ledger_hits, 1);

        reset();
        record_ledger_lookup(L::KeyAbsent);
        record_ledger_lookup(L::Faulted);
        assert_eq!(
            ledger_miss_state(),
            "LEDGER-FAULTED",
            "R13: an instrument outage outranks a finding, or an outage reads as a detection"
        );
        assert!(snapshot().to_json().contains("\"ledger_miss\": \"LEDGER-FAULTED\""));

        // `NeverAttempted` is not a lookup, and recording one must not manufacture the lookup it
        // asserts did not happen.
        reset();
        record_ledger_lookup(L::NeverAttempted);
        assert_eq!(snapshot().proven_key_lookups, 0);
        assert_eq!(snapshot().ledger_hits, 0);

        reset();
    }
}
