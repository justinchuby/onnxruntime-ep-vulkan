"""Tests for the Phase-2 gap-attribution reductions in :mod:`bench.cuda_profile`.

These exist because every reduction in that module has been wrong at least once, and each
time the wrongness was invisible in the output: a plausible number in a well-formatted
table, with no internal inconsistency to notice.

The four defects that motivated these tests, in the order they were found:

1. **Cumulative totals read as per-run.** ``total_us / call_count`` on this EP is dominated
   by the cold first ``Compute``, which uploads the whole weight set — measured 980 ms of
   the 983 ms total ``cmd_upload`` on Phi-3.5 ``prefill_1``. The naive average reported
   70 ms/run of staging for a steady state that pays 0.15 ms, which made a host-bound
   workload look transfer-bound.

2. **An absent marker read as a negative result.** ``record_paths`` returns ``{}`` on this
   EP build because it emits no ``ep.path`` instants. ``{}`` was being read as "no
   re-recording observed" when it means "not measured" — the same unmeasured-is-zero error
   the fallback fractions were fixed for. The command buffer is in fact rebuilt on every
   single inference.

3. **A blind spot charged to whatever was visible.** The trace's outermost span opened
   inside ``dispatch_ort``, not at the ``Compute`` entry point, so 27 ms per inference had
   no row. That 27 ms turned out to be the harness's own counters-file dump.

4. **The instrument that saw it was emitted and never read.** ``vulkan.compute_call`` was
   added to close (3) and ``compute_calls()`` went on anchoring on ``vulkan.subgraph``, so
   every span in the 27 ms region landed in the ``None`` bucket and appeared in no per-call
   or steady-state total. A blind spot that is measurable but unattributed is renamed, not
   closed — and the accompanying ``Phase::BindCheck``, documented as the explanation for the
   region, measured a warm median of 0.073 ms against it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cuda_profile as cp  # noqa: E402


def _span(name, cat, ts, dur, args=None):
    ev = {"name": name, "cat": cat, "ph": "X", "ts": ts, "dur": dur}
    if args:
        ev["args"] = args
    return ev


def _phase(name, ts, dur, nested_in="none"):
    return _span(f"vulkan.{name}", "ep.phase", ts, dur, {"nested_in": nested_in})


def _gpu(label, ts, dur_us, ns=None):
    return _span(f"vulkan.gpu.{label}", "gpu", ts, dur_us,
                 {"gpu_ns": ns if ns is not None else dur_us * 1000})


def _call(ts, dur, inner_ts, inner_dur, nodes=1):
    """One ``Compute`` call as the current EP emits it: an outer bracket around an inner one."""
    return [_span(cp.COMPUTE_CALL_SPAN, "ep", ts, dur),
            _span(cp.SUBGRAPH_SPAN, "ep", inner_ts, inner_dur, {"nodes": nodes})]


# ---------------------------------------------------------------------------
# The anchor: which span brackets a Compute call
# ---------------------------------------------------------------------------

def test_the_outer_bracket_is_preferred_over_the_inner_one():
    """The defect: the instrument that saw the region was emitted and never read.

    ``vulkan.compute_call`` was added to close a 27 ms blind spot; ``compute_calls()`` went
    on anchoring on ``vulkan.subgraph``, so the region stayed in the ``None`` bucket and no
    per-call or steady-state total contained it. Renamed, not closed.
    """
    events = _call(0, 100_000, 1_000, 40_000)
    anchor = cp.choose_anchor(events)
    assert anchor["span"] == cp.COMPUTE_CALL_SPAN
    assert anchor["sees_whole_compute"] is True
    calls = cp.compute_calls(events)
    assert len(calls) == 1
    assert calls[0]["dur"] == 100_000, "the anchor must be the outer bracket, not the inner"


def test_a_trace_without_the_outer_bracket_still_reduces_and_says_it_is_bounded():
    """Stored traces predate the outer span. Refusing to read them discards the evidence.

    But a reduction anchored on the inner span cannot say anything about the region outside
    it, so the limit is written onto the artifact rather than left to be inferred from a
    missing field.
    """
    events = [_span(cp.SUBGRAPH_SPAN, "ep", 1_000, 40_000, {"nodes": 1}),
              _phase("record", 1_010, 5_000)]
    anchor = cp.choose_anchor(events)
    assert anchor["span"] == cp.SUBGRAPH_SPAN
    assert anchor["sees_whole_compute"] is False
    assert "FALLBACK" in anchor["basis"]
    traced = {"workload": "w", "median_ms": 50.0, "warmup_ms": [], "steady_ms": [50.0]}
    tmp = Path(__file__).resolve().parent / "results" / "_cuda69"
    del tmp  # the report path is exercised in the round-trip test below
    out = cp._summarise_bucket(cp.bucket_by_call(events, cp.compute_calls(events))[0])
    assert out["subgraph_us"] is None, (
        "with no outer bracket there is no inner-vs-outer difference to report; None means "
        "'this anchor cannot see it', and 0 would mean 'it costs nothing'")
    assert out["unattributed_in_call_us"] is None, (
        "the tier-1 remainder is `call - bind_check - subgraph`, and this anchor cannot see "
        "the callback the tier-1 spans partition")
    assert out["unattributed_in_subgraph_us"] is not None, (
        "the tier-2 residual IS computable here: the fallback anchor is the dispatch region "
        "itself, so `call_us` is the right denominator for it")
    assert traced["median_ms"] == 50.0


def test_the_region_between_the_two_brackets_is_measured_and_reported():
    """The number Tank's review is about: 62.3 ms outer, 33.9 ms inner, 27.0 ms between.

    Shaped exactly like the real trace — almost all of the difference falls *after* the
    inner span closes, not before it. Under Morpheus's tier model the region is decomposed one
    step further: `bind_check` is measured on its own span, and what neither it nor the dispatch
    region covers is `unattributed_in_call_us` — never charged to `bind_check`.
    """
    events = []
    for i in range(4):
        base = i * 1_000_000
        events += _call(base, 62_329, base + 152, 33_868)
        events.append(_phase("bind_check", base + 60, 73))
        events.append(_phase("record", base + 200, 5_581))
        events.append(_phase("submit", base + 6_000, 236))
        events.append(_phase("fence_wait", base + 6_300, 20_825))
    per_call = [cp._summarise_bucket(b)
                for b in cp.bucket_by_call(events, cp.compute_calls(events))]
    steady = cp.steady_state(per_call)
    assert steady["median_call_us"] == pytest.approx(62_329)
    assert steady["median_subgraph_us"] == pytest.approx(33_868)
    assert steady["median_bind_check_us"] == pytest.approx(73)
    assert steady["median_unattributed_in_call_us"] == pytest.approx(62_329 - 73 - 33_868)
    # bind_check is 0.27% of the region it was once documented as explaining. Asserted as a
    # number so the withdrawn claim cannot be restated in prose without this going red.
    assert (steady["median_bind_check_us"]
            / steady["median_unattributed_in_call_us"]) < 0.01
    # And the tier-2 phases still reconcile inside the dispatch region, which is what the
    # containment contract is about.
    assert steady["median_tier2_total_us"] == pytest.approx(26_642)
    assert steady["median_unattributed_in_subgraph_us"] == pytest.approx(33_868 - 26_642)
    # The tier-1 sum must NOT include the tier-2 phases: they are already inside subgraph_us.
    assert steady["median_tier1_total_us"] == pytest.approx(73)


def test_the_reconciliation_terms_do_not_overlap_and_stay_on_one_axis():
    """A decomposition that mixes a traced term with an untraced one closes falsely."""
    events = []
    for i in range(4):
        base = i * 1_000_000
        events += _call(base, 60_000, base + 100, 30_000)
        events.append(_phase("bind_check", base + 20, 60))
        events.append(_phase("record", base + 200, 5_000))
        events.append(_phase("fence_wait", base + 6_000, 20_000))
    per_call = [cp._summarise_bucket(b)
                for b in cp.bucket_by_call(events, cp.compute_calls(events))]
    steady = cp.steady_state(per_call)
    rec = cp.compute_reconciliation(steady, traced_median_ms=70.0, overhead_ratio=1.14,
                                    anchor=cp.choose_anchor(events))
    assert rec["available"] is True
    # Tier 1 partitions the anchor exactly: bind_check + subgraph + unattributed.
    assert (rec["bind_check_ms"] + rec["subgraph_ms"] + rec["unattributed_in_call_ms"]
            == pytest.approx(rec["compute_call_ms"])), (
        "the tier-1 terms must exactly partition their parent")
    # Tier 2 partitions the dispatch region exactly.
    assert (rec["tier2_phases_ms"] + rec["unattributed_in_subgraph_ms"]
            == pytest.approx(rec["subgraph_ms"]))
    # And the two tiers are NOT additive against one parent: tier-2 time is inside subgraph_ms,
    # so adding it to the tier-1 terms overshoots the anchor. This is the category error the
    # ruling forbids, asserted as arithmetic.
    assert (rec["bind_check_ms"] + rec["subgraph_ms"] + rec["unattributed_in_call_ms"]
            + rec["tier2_phases_ms"]) > rec["compute_call_ms"]
    assert rec["outside_compute_call_ms"] == pytest.approx(70.0 - 60.0)
    assert "traced" in rec["axis"]
    assert "untraced" in rec["do_not"]
    assert "UNATTRIBUTED" in rec["unattributed_in_call_attribution"], (
        "the region is measured, not explained; the last claim made about it "
        "(Phase::BindCheck, 'the binding checks') was 0.073 ms against 27.0 ms")
    assert "not bind-check time" in rec["unattributed_in_call_attribution"], (
        "the withdrawn attribution must be denied by name, next to the number")


def test_a_reduction_of_an_admissible_traced_run_issues_no_refusals():
    """Refusals are for evidence that cannot be read, not for the module disagreeing with itself.

    The committed ``profile_prefill_1_v2.json`` shipped
    ``refusals: ["trace.rs and cuda_profile.py disagree about the phase tree"]`` beside
    ``verdict: "GPU_TIME_MEASURED"`` — a self-declared instrument disagreement passing as a
    green result. Every phase this EP emits must be classifiable by this module without a
    word of complaint.
    """
    events = []
    for i in range(3):
        base = i * 1_000_000
        events += _call(base, 60_000, base + 100, 40_000)
        for name in cp.SIBLING_PHASES:
            events.append(_phase(name, base + 200, 1_000))
        for name in cp.NESTED_PHASES:
            events.append(_phase(name, base + 300, 100, nested_in="record"))
        events.append(_gpu("matmul", base + 500, 50, ns=50_000))
    traced = {"workload": "w", "median_ms": 65.0, "warmup_ms": [1.0],
              "steady_ms": [65.0, 65.0], "verdict": "ADMISSIBLE", "refusals": []}
    trace_file = Path(__file__).resolve().parent / "_test_trace_no_refusals.json"
    trace_file.write_text(json.dumps(events), "utf-8")
    try:
        out = cp.attribute(traced, {"median_ms": 57.0}, trace_file)
    finally:
        trace_file.unlink()
    assert out["verdict"] == cp.GPU_TIME_MEASURED
    assert out["refusals"] == [], (
        f"an admissible traced run must produce no refusals; got {out['refusals']}")
    assert out["phases"]["phase_tree_disagreements"] == []
    assert out.get("scope_limits") is None, "the outer anchor was present; nothing is bounded"
    assert out["anchor"]["span"] == cp.COMPUTE_CALL_SPAN


def test_a_refusal_demotes_the_verdict_and_cannot_ride_along_with_it():
    """Directive 8, as the negative case: ``GPU_TIME_MEASURED`` requires ``refusals == []``.

    The test above proves a clean run refuses nothing. That is the easy half, and it is the
    half the rejected revision also passed. This is the half it failed: when a refusal *does*
    arise, the verdict must move. The shipped artifact carried a phase-tree refusal under
    ``"verdict": "GPU_TIME_MEASURED"`` precisely because the verdict was decided from the GPU
    spans early and never re-examined.

    The trace here is fully measurable — real GPU spans, a real anchor — so the *only* reason
    the verdict may not read MEASURED is the refusal itself. An unknown phase name is used to
    provoke it, because that is a disagreement this module cannot resolve by looking harder.
    """
    events = []
    for i in range(3):
        base = i * 1_000_000
        events += _call(base, 60_000, base + 100, 40_000)
        for name in cp.SIBLING_PHASES:
            events.append(_phase(name, base + 200, 1_000))
        events.append(_gpu("matmul", base + 500, 50, ns=50_000))
    # A phase this module has never heard of: trace.rs grew a phase and nobody told the
    # reduction. It cannot be summed, so it must be refused, not quietly dropped.
    events.append(_phase("teleport", 200, 1_000))
    traced = {"workload": "w", "median_ms": 65.0, "warmup_ms": [1.0],
              "steady_ms": [65.0, 65.0], "verdict": "ADMISSIBLE", "refusals": []}
    trace_file = Path(__file__).resolve().parent / "_test_trace_refusal_demotes.json"
    trace_file.write_text(json.dumps(events), "utf-8")
    try:
        out = cp.attribute(traced, {"median_ms": 57.0}, trace_file)
    finally:
        trace_file.unlink()
    assert out["refusals"], "an unknown phase must be refused, not dropped"
    assert any("teleport" in r for r in out["refusals"])
    assert out["verdict"] != cp.GPU_TIME_MEASURED, (
        f"the reduction refused and still reported {out['verdict']!r}; that is the exact "
        f"pairing Tank rejected")
    assert out["verdict"] == cp.ATTRIBUTION_REFUSED
    # And the GPU numbers are still there. A refusal bounds what may be *concluded*; it does
    # not delete the evidence, which is what makes the demotion honest rather than punitive.
    assert out["gpu"]["total_ns"] > 0


def test_the_verdict_and_refusal_invariant_holds_for_every_verdict():
    """The invariant stated over the whole enum, not just the one pairing that was caught.

    ``_finalise_verdict`` is a single choke point, so it is cheap to assert the property for
    every verdict this module can emit rather than for the one that shipped broken.
    """
    for verdict in cp.VERDICTS:
        clean = cp._finalise_verdict({"verdict": verdict, "refusals": []})
        assert clean["verdict"] == verdict, "a clean report's verdict is never rewritten"
        dirty = cp._finalise_verdict({"verdict": verdict, "refusals": ["cannot read x"]})
        assert dirty["verdict"] != cp.GPU_TIME_MEASURED, (
            f"{verdict!r} plus a refusal resolved to GPU_TIME_MEASURED")


# ---------------------------------------------------------------------------
# Per-call bucketing
# ---------------------------------------------------------------------------

def test_cold_first_call_does_not_contaminate_the_steady_state():
    """The exact shape of the real defect: one huge cold call, many cheap warm ones.

    A mean over all calls reports ~201 ms of upload per run. No call in this trace costs
    that: the cold one costs 2000 ms and every warm one costs 1 ms. The median of the warm
    calls is the only number that describes a run a user actually experiences.
    """
    events = [_span("vulkan.subgraph", "ep", 0, 2_100_000, {"nodes": 3})]
    events.append(_phase("record", 10, 2_050_000))
    events.append(_phase("cmd_upload", 20, 2_000_000, nested_in="record"))
    for i in range(1, 14):
        base = 3_000_000 + i * 100_000
        events.append(_span("vulkan.subgraph", "ep", base, 50_000, {"nodes": 3}))
        events.append(_phase("record", base + 10, 5_000))
        events.append(_phase("cmd_upload", base + 20, 1_000, nested_in="record"))

    calls = cp.compute_calls(events)
    assert len(calls) == 14
    per_call = [cp._summarise_bucket(b) for b in cp.bucket_by_call(events, calls)]
    steady = cp.steady_state(per_call)

    assert steady["warm_calls"] == 13
    assert steady["median_nested_us"]["cmd_upload"] == pytest.approx(1_000)
    assert steady["cold"]["nested_us"]["cmd_upload"]["us"] == pytest.approx(2_000_000)

    naive_mean = 2_000_000 + 13 * 1_000
    assert naive_mean / 14 > 100 * steady["median_nested_us"]["cmd_upload"], (
        "the naive average must be shown to be wildly unlike any real call, which is "
        "the whole reason the split exists")


def test_spans_outside_every_compute_call_are_kept_not_dropped():
    """Session setup, compile and prepack happen outside any ``Compute``.

    Dropping them would make the per-call totals silently fail to reconcile with the
    trace they came from, which removes the only cheap way to notice a bucketing bug.
    """
    events = [
        _phase("compile", 0, 500_000),          # before any Compute
        _span("vulkan.subgraph", "ep", 1_000_000, 10_000, {"nodes": 1}),
        _phase("record", 1_000_010, 5_000),
        _phase("prepack", 5_000_000, 700_000),  # after the last Compute
    ]
    calls = cp.compute_calls(events)
    buckets = cp.bucket_by_call(events, calls)
    outside = [b for b in buckets if b["index"] is None]
    assert len(outside) == 1
    names = {e["name"] for e in outside[0]["phases"]}
    assert names == {"vulkan.compile", "vulkan.prepack"}


def test_nested_spans_are_not_added_to_the_sibling_total():
    """``record`` is an inclusive bracket; its children must not be summed beside it."""
    events = [
        _span("vulkan.subgraph", "ep", 0, 100_000, {"nodes": 1}),
        _phase("record", 10, 60_000),
        _phase("cmd_upload", 20, 50_000, nested_in="record"),
        _phase("fence_wait", 60_020, 30_000),
    ]
    calls = cp.compute_calls(events)
    summary = cp._summarise_bucket(cp.bucket_by_call(events, calls)[0])
    assert summary["sibling_total_us"] == pytest.approx(90_000)
    assert summary["record_residual_us"] == pytest.approx(10_000)


def test_upload_and_cmd_upload_are_not_double_counted():
    """The two phases bracket the same memcpy — take the larger, never the sum."""
    events = [
        _span("vulkan.subgraph", "ep", 0, 100_000, {"nodes": 1}),
        _phase("record", 10, 60_000),
        _phase("cmd_upload", 20, 50_000, nested_in="record"),
        _phase("upload", 25, 49_000, nested_in="record"),
    ]
    calls = cp.compute_calls(events)
    summary = cp._summarise_bucket(cp.bucket_by_call(events, calls)[0])
    # 60_000 - max(50_000, 49_000) == 10_000.  Summing them would underflow to 0 and be
    # clamped, silently reporting that command recording is free.
    assert summary["record_residual_us"] == pytest.approx(10_000)


# ---------------------------------------------------------------------------
# Re-record detection without the marker
# ---------------------------------------------------------------------------

def test_rerecording_every_call_is_detected_without_ep_path_events():
    """The real observation: 355 descriptor sets allocated on every warm call.

    ``record_paths`` sees nothing here because this EP emits no ``ep.path`` instants. The
    per-dispatch recording work still proves the command buffer was rebuilt.
    """
    events = []
    for i in range(6):
        base = i * 1_000_000
        events.append(_span("vulkan.subgraph", "ep", base, 500_000, {"nodes": 355}))
        events.append(_phase("record", base + 10, 400_000))
        for d in range(355):
            events.append(_phase("desc_alloc", base + 20 + d, 1, nested_in="record"))

    assert cp.record_paths(events) == {}, "precondition: no marker to read"
    calls = cp.compute_calls(events)
    per_call = [cp._summarise_bucket(b) for b in cp.bucket_by_call(events, calls)]
    ev = cp.rerecord_evidence(per_call)
    assert ev["verdict"] == "RERECORDED_EVERY_CALL"
    assert ev["warm_desc_alloc_median"] == 355


def test_a_replayed_command_buffer_reads_as_replayed():
    """Warm calls that record nothing must not be called re-recorders."""
    events = [_span("vulkan.subgraph", "ep", 0, 500_000, {"nodes": 8}),
              _phase("record", 10, 400_000)]
    events += [_phase("desc_alloc", 20 + d, 1, nested_in="record") for d in range(8)]
    for i in range(1, 6):
        base = i * 1_000_000
        events.append(_span("vulkan.subgraph", "ep", base, 50_000, {"nodes": 8}))
        events.append(_phase("submit", base + 10, 100))
    per_call = [cp._summarise_bucket(b)
                for b in cp.bucket_by_call(events, cp.compute_calls(events))]
    assert cp.rerecord_evidence(per_call)["verdict"] == "REPLAYED"


def test_rerecord_verdict_is_unknown_without_a_warm_call():
    """One call is not a steady state and must not be described as one."""
    events = [_span("vulkan.subgraph", "ep", 0, 500_000, {"nodes": 8})]
    per_call = [cp._summarise_bucket(b)
                for b in cp.bucket_by_call(events, cp.compute_calls(events))]
    assert cp.rerecord_evidence(per_call)["verdict"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Device time is never inferred
# ---------------------------------------------------------------------------

def test_absent_gpu_spans_are_unavailable_not_zero():
    """No timestamps means no measurement.  ``fence_wait`` must not stand in for one."""
    events = [
        _span("vulkan.subgraph", "ep", 0, 100_000, {"nodes": 1}),
        _phase("fence_wait", 10, 90_000),
    ]
    gpu = cp.gpu_breakdown(events)
    assert gpu["span_count"] == 0
    assert gpu["total_ns"] == 0
    traced = {"workload": "w", "median_ms": 100.0, "warmup_ms": [], "steady_ms": [100.0]}
    out = cp.attribute(traced, {"median_ms": 100.0}, Path("does-not-exist.json"))
    assert out["verdict"] == cp.TRACE_ABSENT
    assert any("no phase split" in r for r in out["refusals"])


def test_gpu_time_is_summed_from_gpu_ns_not_span_duration():
    """``dur`` is the same number rounded to microseconds; ``gpu_ns`` is authoritative."""
    events = [_gpu("matmul", 0, 5, ns=5_432), _gpu("matmul", 10, 5, ns=5_678)]
    gpu = cp.gpu_breakdown(events)
    assert gpu["total_ns"] == 11_110
    assert gpu["per_label_ns"]["matmul"]["count"] == 2


def test_op_comparison_is_restricted_to_ops_both_providers_ran():
    """Comparing an op only one provider executed is not a comparison."""
    vk = {"MatMul|VulkanExecutionProvider": {"op_type": "MatMul",
                                             "provider": "VulkanExecutionProvider",
                                             "us": 100.0, "nodes": 1},
          "Cast|VulkanExecutionProvider": {"op_type": "Cast",
                                           "provider": "VulkanExecutionProvider",
                                           "us": 5.0, "nodes": 1}}
    cu = {"MatMul|CUDAExecutionProvider": {"op_type": "MatMul",
                                           "provider": "CUDAExecutionProvider",
                                           "us": 50.0, "nodes": 1},
          "Gather|CUDAExecutionProvider": {"op_type": "Gather",
                                           "provider": "CUDAExecutionProvider",
                                           "us": 3.0, "nodes": 1}}
    out = cp.op_type_comparison(vk, cu)
    shared = {r["op_type"] for r in out["shared_op_types"]}
    assert shared == {"MatMul"}
