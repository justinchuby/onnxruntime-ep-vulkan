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

/// Bumped when a field is **added**. Fields are never removed or reordered, so a reader that
/// knows version *n* can read the first *n* generations' worth of a version *n+k* struct.
pub const COUNTERS_ABI_VERSION: u32 = 1;

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
    EQUIVALENCE_UNMEASURED
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
    /// Dispatches that ran to fence completion. **This is the criterion-8 number.**
    pub dispatches_executed: u64,
}

static COMPILE_CALLS: AtomicU64 = AtomicU64::new(0);
static SUBGRAPHS_LIVE: AtomicU64 = AtomicU64::new(0);
static SUBGRAPHS_STUB: AtomicU64 = AtomicU64::new(0);
static COMPUTE_CALLS: AtomicU64 = AtomicU64::new(0);
static COMPUTE_FAILURES: AtomicU64 = AtomicU64::new(0);
static DISPATCHES_EXECUTED: AtomicU64 = AtomicU64::new(0);
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
        dispatches_executed: DISPATCHES_EXECUTED.load(ORD),
    }
}

/// Zero every counter. Exported so a test can scope a claim to one model run.
pub fn reset() {
    COMPILE_CALLS.store(0, ORD);
    SUBGRAPHS_LIVE.store(0, ORD);
    SUBGRAPHS_STUB.store(0, ORD);
    COMPUTE_CALLS.store(0, ORD);
    COMPUTE_FAILURES.store(0, ORD);
    DISPATCHES_EXECUTED.store(0, ORD);
    CLAIMED_NODES.store(0, ORD);
    ISLANDS_OFFERED.store(0, ORD);
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
        format!(
            "{{\n  \"abi_version\": {},\n  \"compile_calls\": {},\n  \"subgraphs_live\": {},\n  \
             \"subgraphs_stub\": {},\n  \"compute_calls\": {},\n  \"compute_failures\": {},\n  \
             \"dispatches_executed\": {},\n  \"claimed_nodes\": {},\n  \"islands_offered\": {},\n  \
             \"model_output_equivalence\": \"{}\"\n}}\n",
            self.abi_version,
            self.compile_calls,
            self.subgraphs_live,
            self.subgraphs_stub,
            self.compute_calls,
            self.compute_failures,
            self.dispatches_executed,
            claimed,
            islands,
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
    let existing_equiv = std::fs::read_to_string(&path)
        .ok()
        .as_deref()
        .map(extract_equivalence)
        .unwrap_or(EQUIVALENCE_UNMEASURED);

    let o = crate::allocator::ledger::snapshot();
    let t = crate::allocator::tally::snapshot();
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
             \"alloc_device_authoritative_spans\": {}\n}}\n",
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
            t.device_authoritative_spans,
        ));
    }
    if let Err(e) = std::fs::write(&path, doc) {
        log::warn!(
            "could not write EP observations to {}: {e}",
            path.to_string_lossy()
        );
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

    /// The dispatch-path dump must not write a poorer document than the teardown path.
    ///
    /// Both write the same file and last-write-wins, so a subset document on the hot path erases
    /// the `alloc_*`/`pointers_*` keys on any run that ends with a dispatch. This is the falsifier:
    /// if `dump_if_requested` ever stops emitting the full document, this goes red.
    #[test]
    fn the_dispatch_path_dump_carries_the_allocator_keys_too() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("counters_dump_parity_test.json");
        std::fs::remove_file(&path).ok();

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
            "pointers_use_after_free",
        ] {
            assert!(
                doc.contains(key),
                "the per-dispatch dump dropped `{key}`; it overwrites the teardown document, so \
                 the key disappears from any run that ends with a dispatch. Document was:\n{doc}"
            );
        }
    }

    /// The counters are process-wide statics, so tests that touch them must not run concurrently
    /// with each other. One test, sequenced by hand, is simpler than a mutex and cannot deadlock.
    #[test]
    fn counters_record_what_they_claim_to_record() {
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

        reset();
        assert_eq!(snapshot().compute_calls, 0);
    }

    #[test]
    fn extract_equivalence_parses_the_three_states() {
        let match_doc = snapshot().to_json_with_equiv(EQUIVALENCE_MATCH);
        let div_doc = snapshot().to_json_with_equiv(EQUIVALENCE_DIVERGENT);
        let unm_doc = snapshot().to_json_with_equiv(EQUIVALENCE_UNMEASURED);
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
        // Absence must be treated the same as UNMEASURED (R7: absence ≠ negative).
        assert_eq!(
            extract_equivalence(&without_field),
            EQUIVALENCE_UNMEASURED,
            "a snapshot without the field predates the verdict; absence = UNMEASURED"
        );
        assert_eq!(extract_equivalence("{}"), EQUIVALENCE_UNMEASURED);
    }
}
