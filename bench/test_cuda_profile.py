"""Tests for the Phase-2 gap-attribution reductions in :mod:`bench.cuda_profile`.

These exist because every reduction in that module has been wrong at least once, and each
time the wrongness was invisible in the output: a plausible number in a well-formatted
table, with no internal inconsistency to notice.

The four defects that motivated these tests, in the order they were found:

1. **Cumulative totals read as per-run.** ``total_us / call_count`` on this EP is dominated
   by the cold first ``Compute``, which uploads the whole weight set — nearly the entire
   ``cmd_upload`` total on Phi-3.5 ``prefill_1`` lands in that one call. The naive average
   charged a large per-run staging cost to a steady state that pays almost none, which made
   a host-bound workload look transfer-bound. (Magnitudes are quoted with the committed
   profile, which this instrumentation head does not carry.)

2. **An absent marker read as a negative result.** ``record_paths`` returns ``{}`` on this
   EP build because it emits no ``ep.path`` instants. ``{}`` was being read as "no
   re-recording observed" when it means "not measured" — the same unmeasured-is-zero error
   the fallback fractions were fixed for. The command buffer is in fact rebuilt on every
   single inference.

3. **A blind spot charged to whatever was visible.** The trace's outermost span opened
   inside ``dispatch_ort``, not at the ``Compute`` entry point, so a large per-inference
   term had no row. It turned out to be the harness's own counters-file dump. (Neither the
   pre-fix run nor the corrected one is a committed artifact on this branch, so no figure
   is quoted; the corrected term is ``outside_subgraph_ms``, read out of the committed
   profile by the citation pin below wherever one exists.)

4. **The instrument that saw it was emitted and never read.** ``vulkan.compute_call`` was
   added to close (3) and ``compute_calls()`` went on anchoring on ``vulkan.subgraph``, so
   every span in that region landed in the ``None`` bucket and appeared in no per-call
   or steady-state total. A blind spot that is measurable but unattributed is renamed, not
   closed — and the accompanying ``Phase::BindCheck``, documented as the explanation for the
   region, accounted for a negligible fraction of it.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _polarity
import cuda_competition as cc
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

    ``vulkan.compute_call`` was added to close a large blind spot; ``compute_calls()`` went
    on anchoring on ``vulkan.subgraph``, so the region stayed in the ``None`` bucket and no
    per-call or steady-state total contained it. Renamed, not closed.
    """
    events = _call(0, 100_000, 1_000, 40_000)
    anchor = cp.choose_anchor(events)
    assert anchor["span"] == cp.COMPUTE_CALL_SPAN
    assert anchor["sees_compute_call_bracket"] is True
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
    assert anchor["sees_compute_call_bracket"] is False
    assert "FALLBACK" in anchor["basis"]
    traced = {"workload": "w", "median_ms": 50.0, "warmup_ms": [], "steady_ms": [50.0]}
    tmp = Path(__file__).resolve().parent / "results" / "_cuda69"
    del tmp  # the report path is exercised in the round-trip test below
    out = cp._summarise_bucket(cp.bucket_by_call(events, cp.compute_calls(events))[0])
    assert out["subgraph_us"] is None, (
        "with no outer bracket there is no inner-vs-outer difference to report; None means "
        "'this anchor cannot see it', and 0 would mean 'it costs nothing'")
    assert out["outside_subgraph_us"] is None
    assert traced["median_ms"] == 50.0


def test_the_region_between_the_two_brackets_is_measured_and_reported():
    """The region Tank's review is about: a large term between the outer and inner brackets.

    The microsecond figures below are a **synthetic** trace shaped like the real one — every
    call splits identically, which is what lets this test assert an exact partition. They are
    not measurements and must not be quoted as any; the committed figures live in
    ``bench/results/_cuda69/profile_prefill_1.json``. What is modelled from the real trace is
    the *shape*: almost all of the difference falls **after** the inner span closes, not
    before it.
    """
    events = []
    for i in range(4):
        base = i * 1_000_000
        events += _call(base, 62_329, base + 152, 33_868)
        events.append(_phase("record", base + 200, 5_581))
        events.append(_phase("submit", base + 6_000, 236))
        events.append(_phase("fence_wait", base + 6_300, 20_825))
    per_call = [cp._summarise_bucket(b)
                for b in cp.bucket_by_call(events, cp.compute_calls(events))]
    steady = cp.steady_state(per_call)
    assert steady["median_call_us"] == pytest.approx(62_329)
    assert steady["median_subgraph_us"] == pytest.approx(33_868)
    assert steady["median_outside_subgraph_us"] == pytest.approx(62_329 - 33_868)
    # And the phases still reconcile inside the INNER bracket, which is what the containment
    # contract is about.
    assert steady["median_sibling_total_us"] == pytest.approx(26_642)
    assert steady["median_unattributed_in_subgraph_us"] == pytest.approx(33_868 - 26_642)


def test_the_reconciliation_terms_do_not_overlap_and_stay_on_one_axis():
    """A decomposition that mixes a traced term with an untraced one closes falsely."""
    events = []
    for i in range(4):
        base = i * 1_000_000
        events += _call(base, 60_000, base + 100, 30_000)
        events.append(_phase("record", base + 200, 5_000))
        events.append(_phase("fence_wait", base + 6_000, 20_000))
    per_call = [cp._summarise_bucket(b)
                for b in cp.bucket_by_call(events, cp.compute_calls(events))]
    steady = cp.steady_state(per_call)
    rec = cp.compute_reconciliation(steady, traced_median_ms=70.0, overhead_ratio=1.14,
                                    anchor=cp.choose_anchor(events))
    assert rec["available"] is True
    assert rec["subgraph_ms"] + rec["outside_subgraph_ms"] == pytest.approx(
        rec["compute_call_ms"]), "the two children must exactly partition their parent"
    assert (rec["sibling_phases_ms"] + rec["unattributed_in_subgraph_ms"]
            == pytest.approx(rec["subgraph_ms"]))
    assert rec["outside_compute_call_ms"] == pytest.approx(70.0 - 60.0)
    assert "traced" in rec["axis"]
    assert "untraced" in rec["do_not"]
    assert "UNATTRIBUTED" in rec["outside_subgraph_attribution"], (
        "the region is measured, not explained; the last claim made about it "
        "(Phase::BindCheck, 'the binding checks') accounted for a negligible fraction of it "
        "and was withdrawn")


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


# ---------------------------------------------------------------------------
# Per-call bucketing
# ---------------------------------------------------------------------------

def test_cold_first_call_does_not_contaminate_the_steady_state():
    """The exact shape of the real defect: one huge cold call, many cheap warm ones.

    The figures below are a **synthetic trace**, built in this test rather than observed: a
    cold call that costs orders of magnitude more than a warm one, so a mean over all calls
    lands on a number no call in the trace costs. The median of the warm calls is the only
    number that describes a run a user actually experiences.
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
    """One descriptor set allocated per claimed node, on every warm call.

    A **synthetic trace**: the island size used here matches Phi-3.5's `claimed_nodes` 355
    (committed in ``bench/results/barrier_ab-post-dev0-0.json``) so the fixture is the shape
    the EP really produces, but the per-call recording work below is constructed, not
    measured.  ``record_paths`` sees nothing here because this EP emits no ``ep.path``
    instants. The per-dispatch recording work still proves the command buffer was rebuilt.
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


# ---------------------------------------------------------------------------
# Cross-process arithmetic is withdrawn, not renamed
# ---------------------------------------------------------------------------

def _admissible_traced_run():
    """A small admissible traced reduction, with a deliberately different untraced wall."""
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
    trace_file = Path(__file__).resolve().parent / "_test_trace_withdrawn.json"
    trace_file.write_text(json.dumps(events), "utf-8")
    try:
        return cp.attribute(traced, {"median_ms": 57.0}, trace_file)
    finally:
        trace_file.unlink()


def test_no_term_is_published_that_mixes_the_traced_and_untraced_processes():
    """The untraced arm emits no device timestamps, so it has no term to combine with one.

    `gpu_ms_per_run` can only come from the traced process -- the tracer is what arms the
    query pool. `untraced_median_ms` is a second process's wall clock. Dividing or
    subtracting one by the other produces a number that looks like a within-run
    decomposition and is not one: whatever run-to-run variation separated the two
    processes lands in the result by construction.

    This is the defect that shipped as `gpu_share_untraced_bound` and
    `host_ms_per_run_residual`. The test is written against the *arithmetic*, not the two
    names, so re-introducing the same quantity under a third name fails here too.
    """
    out = _admissible_traced_run()
    assert out["verdict"] == cp.GPU_TIME_MEASURED
    gpu = out["gpu_ms_per_run"]
    untraced = out["untraced_median_ms"]
    assert gpu and untraced and gpu != untraced

    forbidden = {round(gpu / untraced, 12), round(untraced - gpu, 12)}
    offenders = [k for k, v in out.items()
                 if isinstance(v, float) and round(v, 12) in forbidden]
    assert not offenders, (
        f"{offenders} equal a cross-process ratio or difference; the traced device median "
        f"and the untraced wall come from two separate runs and may not be combined")


def test_the_withdrawn_terms_are_named_rather_than_silently_dropped():
    """A field that vanishes reads exactly like a field that was never populated.

    A reader who knew the old number has to be able to find out what happened to it, so
    the withdrawal carries the formula, the reason, and the fact that both operands are
    still on the record and the old value can be recomputed by anyone who disagrees.
    """
    out = _admissible_traced_run()
    tomb = out.get("withdrawn_terms")
    assert isinstance(tomb, dict)
    for name in ("gpu_share_untraced_bound", "host_ms_per_run_residual"):
        assert name not in out, f"{name} is withdrawn and must not be published"
        assert name in tomb, f"{name} vanished without a tombstone"
        assert "Withdrawn:" in tomb[name]
    assert "untraced_median_ms" in out and "gpu_ms_per_run" in out, (
        "the operands stay published so the withdrawal can be checked")


def test_the_withdrawal_does_not_smuggle_in_a_replacement_measurement():
    """Withdrawing a number is not licence to publish a new one.

    This revision has no eligible author for a replacement metric and did not re-run the
    profiler, so the tombstone must say so rather than quietly substituting a differently
    named quantity in the same slot.
    """
    out = _admissible_traced_run()
    tomb = out["withdrawn_terms"]
    assert "no_replacement_in_this_revision" in tomb
    assert all(isinstance(v, str) for v in tomb.values()), (
        "a tombstone entry is prose, never a number")


def test_the_committed_profile_artifact_agrees_with_the_module():
    """The shipped artifact must not still carry what the module stopped computing.

    An artifact and the code that produced it disagreeing is how a withdrawn number
    survives in the file people actually read.
    """
    art = (Path(__file__).resolve().parents[1] / "bench" / "results" / "_cuda69"
           / "profile_prefill_1.json")
    if not art.exists():
        pytest.skip("committed artifact not present in this tree")
    doc = json.loads(art.read_text("utf-8"))
    for name in ("gpu_share_untraced_bound", "host_ms_per_run_residual"):
        assert name not in doc, f"the committed artifact still publishes {name}"
    assert "withdrawn_terms" in doc, "the artifact drops the terms without saying so"
    md = art.with_suffix(".md").read_text("utf-8")
    assert "tracer-overhead-immune" not in md, (
        "the rendered report still makes the withdrawn claim in prose")
    assert "withdrawn" in md.lower()
# ---------------------------------------------------------------------------
# B3: a record may not be green and refusing at the same time
# ---------------------------------------------------------------------------
#
# `GPU_TIME_MEASURED` beside a non-empty `refusals` list is a record disagreeing with
# itself, and the verdict is the part people read.  These tests are on `seal_verdict`
# and on `attribute()`'s live behaviour -- the committed-artifact check further down is
# deliberately secondary, because an artifact can be regenerated and the invariant has
# to hold for runs nobody has taken yet.

def test_seal_withholds_a_green_verdict_from_a_refusing_record():
    out = _polarity.withholds(
        cp.seal_verdict({"verdict": cp.GPU_TIME_MEASURED, "refusals": ["the trace lied"]}),
        because="a record that refuses may not also call itself measured")
    assert out == cp.GPU_TIME_MEASURED


def test_seal_leaves_a_clean_green_verdict_alone():
    """The negative control. A gate that fires on everything screens nothing."""
    _polarity.publishes(
        cp.seal_verdict({"verdict": cp.GPU_TIME_MEASURED, "refusals": []}),
        cp.GPU_TIME_MEASURED, because="nothing refused, so nothing is withheld")


def test_seal_does_not_relabel_a_verdict_that_was_never_green():
    """`TRACE_ABSENT` is already not a claim; downgrading it would erase why it refused."""
    for v in (cp.TRACE_ABSENT, cp.GPU_TIME_UNAVAILABLE):
        out = cp.seal_verdict({"verdict": v, "refusals": ["something"]})
        assert out["verdict"] == v
        assert "withheld_from" not in out


def test_attribute_withholds_the_verdict_when_the_run_refuses():
    """End to end: a refusal raised anywhere inside `attribute` costs the green token.

    Driven through a *real* refusal path -- the traced record arrives already refusing --
    rather than by poking `out` directly, so this fails if a future exit forgets the seal.
    """
    events = []
    for i in range(4):
        base = i * 1_000_000
        events += _call(base, 60_000, base + 100, 30_000)
        events.append(_phase("record", base + 200, 5_000))
        events.append(_phase("fence_wait", base + 6_000, 20_000))
        events.append(_gpu("matmul", base + 500, 50, ns=50_000))
    traced = {"workload": "w", "median_ms": 65.0, "warmup_ms": [1.0],
              "steady_ms": [65.0, 65.0], "verdict": "ADMISSIBLE",
              "counters_scope": cc.COUNTERS_SCOPE_FIRST_RUN,
              "refusals": ["the device clock was not read on this run"]}
    trace_file = Path(__file__).resolve().parent / "_test_trace_seal.json"
    trace_file.write_text(json.dumps(events), "utf-8")
    try:
        out = cp.attribute(traced, {"median_ms": 57.0,
                                    "counters_scope": cc.COUNTERS_SCOPE_FIRST_RUN},
                           trace_file)
    finally:
        trace_file.unlink()
    assert _polarity.withholds(out, because="the traced record arrived refusing") == (
        cp.GPU_TIME_MEASURED)
    assert out["verdict"] == cp.GPU_TIME_WITHHELD
    assert "device clock" in " ".join(out["withheld_because"])
    assert out["refusals"], "the refusal itself is not consumed by the downgrade"


def test_every_exit_from_attribute_is_sealed():
    """Structural, not behavioural: the gate must be on the *code path*, not on one case.

    Reads `attribute`'s source and requires every `return` in it to go through
    `seal_verdict`. A fifth early exit added later cannot quietly bypass the invariant --
    which is the failure mode that produced the committed artifact this blocker is about.
    """
    src = inspect.getsource(cp.attribute)
    returns = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("return ")]
    assert returns, "attribute() has no returns; this test is reading the wrong function"
    for ln in returns:
        assert "seal_verdict(" in ln, (
            f"unsealed exit from attribute(): {ln!r}. Every return must pass through "
            f"seal_verdict so a refusal cannot ship beside a green verdict.")


def test_render_leads_with_the_withheld_verdict_not_a_footnote():
    """The renderer is what gets pasted into a PR; the withholding has to be above the fold."""
    md = cp.render({"workload": "w", "verdict": cp.GPU_TIME_MEASURED,
                    "refusals": ["the device clock was not read"]})
    headline = next(ln for ln in md.splitlines() if ln.startswith("verdict: "))
    assert headline == f"verdict: **{cp.GPU_TIME_WITHHELD}**", (
        f"the headline still reads {headline!r}; the verdict is the part people quote")
    preamble = md.split("## ")[0]
    assert "withheld" in preamble.lower(), (
        "the withholding must appear before the first section, not under the numbers")
    assert "the device clock was not read" in md, "the reason survives the downgrade"

    clean = cp.render({"workload": "w", "verdict": cp.GPU_TIME_MEASURED, "refusals": []})
    assert f"verdict: **{cp.GPU_TIME_MEASURED}**" in clean, (
        "negative control: a renderer that withholds unconditionally teaches readers to "
        "ignore the token")
    assert "withheld" not in clean.lower()


def test_the_committed_artifact_is_not_green_while_refusing():
    """Secondary to the structural tests above: one artifact, checked because it is public."""
    art = (Path(__file__).resolve().parents[1] / "bench" / "results" / "_cuda69"
           / "profile_prefill_1.json")
    if not art.exists():
        pytest.skip("committed artifact not present in this tree")
    doc = json.loads(art.read_text("utf-8"))
    if doc.get("verdict") in cp.GREEN_VERDICTS:
        assert not doc.get("refusals"), (
            f"{art.name} publishes {doc['verdict']} beside {len(doc['refusals'])} refusal(s)")


# ---------------------------------------------------------------------------
# B2: the documented figure is pinned to the committed artifact
# ---------------------------------------------------------------------------

#: Every place in the tree that quotes the post-fix `outside_subgraph` figure in prose.
#: Adding a citation without adding it here is caught by the "no stale value" screen below.
_OUTSIDE_SUBGRAPH_CITATION_SITES = (
    "docs/PERF.md",
    "rust/src/trace.rs",
    "bench/cuda_profile.py",
)

#: Synthetic fixture values only -- not a record of any real historical measurement.
#: These are not witnessed by any artifact committed on this branch, so no specific past
#: figure is asserted here; the tuple exists solely to exercise the "no stale value" scan
#: below (`test_every_documented_outside_subgraph_citation_matches_the_artifact` and its
#: negative control) with a shape that looks like a plausible drifted figure. If a real
#: superseded figure is ever discovered with a committed witness for both the stale and
#: the corrected value, it belongs here as a *cited* pair, not a bare literal.
_SUPERSEDED_OUTSIDE_SUBGRAPH_MS = ("9.999",)


def _committed_profile():
    art = (Path(__file__).resolve().parents[1] / "bench" / "results" / "_cuda69"
           / "profile_prefill_1.json")
    if not art.exists():
        pytest.skip("committed artifact not present in this tree")
    return json.loads(art.read_text("utf-8"))


def _committed_outside_subgraph_ms():
    """The pinned figure, or a failure that says what happened rather than a KeyError.

    A restructuring of `compute_reconciliation` that renames this term is a legitimate
    thing to do, but it must not be able to silently disarm the pin: an absent key would
    otherwise surface as a `KeyError` in a test whose name is about documents, and the
    obvious repair would be to delete the test.
    """
    doc = _committed_profile()
    rec = doc.get("compute_reconciliation") or {}
    ms = rec.get("outside_subgraph_ms")
    if ms is None:
        pytest.fail(
            "the committed profile no longer publishes "
            "`compute_reconciliation.outside_subgraph_ms`, which is the figure "
            f"{list(_OUTSIDE_SUBGRAPH_CITATION_SITES)} cite. If the reduction was "
            "restructured, repoint this pin at the term that replaced it and update the "
            "prose in the same change -- do not drop the pin, which is the failure this "
            "test exists to prevent.")
    return ms


def test_the_committed_artifact_agrees_with_itself_about_outside_subgraph():
    """`outside_subgraph_ms` and `steady.median_outside_subgraph_us` are the same measurement.

    No magnitude is asserted here. The results branch pins one, because it documents one;
    this head documents no figure at all, so hardcoding an expected value would be asserting
    a number that nothing on this branch publishes and no artifact here backs -- the exact
    defect the pin exists to catch, committed inside the pin. The two-units agreement is the
    part that is true of any profile, and it is what this test is named for.
    """
    doc = _committed_profile()
    ms = _committed_outside_subgraph_ms()
    us = doc["steady"]["median_outside_subgraph_us"]
    assert us / 1000.0 == pytest.approx(ms), (
        f"the artifact publishes {ms} ms and {us} us; they are one number in two units")


def test_every_documented_outside_subgraph_citation_matches_the_artifact():
    """The provenance pin. Prose may not drift from the artifact it claims to cite.

    The rejection that produced this test was documents quoting a figure that had
    drifted from the committed profile -- a plausible-looking number that was not the
    one in the tree. No specific historical figures are asserted here, since neither
    side of that drift is witnessed by a committed artifact on this branch; the figure
    is read *out of the artifact* here and required to be the one the prose says,
    whatever it currently is.
    """
    doc = _committed_profile()
    ms = _committed_outside_subgraph_ms()
    expected = f"{ms:.3f}".rstrip("0")
    root = Path(__file__).resolve().parents[1]
    for rel in _OUTSIDE_SUBGRAPH_CITATION_SITES:
        text = (root / rel).read_text("utf-8")
        assert expected in text, (
            f"{rel} cites the outside-subgraph figure but not as {expected} ms, which is "
            f"what {doc.get('workload')}'s committed profile actually measured")
        for stale in _SUPERSEDED_OUTSIDE_SUBGRAPH_MS:
            assert stale not in text, (
                f"{rel} still quotes the superseded {stale} ms figure; "
                f"the committed artifact says {expected} ms")


def test_the_pin_would_notice_a_drifting_document():
    """Negative control: a pin that cannot fail is decoration."""
    ms = _committed_outside_subgraph_ms()
    expected = f"{ms:.3f}".rstrip("0")
    for stale in _SUPERSEDED_OUTSIDE_SUBGRAPH_MS:
        drifted = f"the term reads {stale} ms"
        assert expected not in drifted
        assert stale in drifted


# ---------------------------------------------------------------------------
# Advisory: the medians do not algebraically partition
# ---------------------------------------------------------------------------

def test_the_reconciliation_says_out_loud_that_medians_do_not_partition():
    """`sibling + unattributed != subgraph` in the committed artifact, by a small residual.

    Not a leak: each term is an independently-taken median over the warm calls, and the
    median of a sum is not the sum of the medians. A reader of the JSON alone has no way
    to know that, so the record has to say it. The residual is read out of whatever
    profile is committed rather than quoted here, so this test does not itself become a
    place where a stale figure survives.
    """
    doc = _committed_profile()
    rec = doc["compute_reconciliation"]
    lhs = rec["sibling_phases_ms"] + rec["unattributed_in_subgraph_ms"]
    assert lhs != pytest.approx(rec["subgraph_ms"], abs=1e-9), (
        "if these now partition exactly the artifact was regenerated; keep the note anyway, "
        "it is a property of the statistic and not of this run")
    note = rec.get("partition_note")
    assert note, "the artifact leaves the residual unexplained"
    assert "not" in note.lower() and "median" in note.lower()
    assert f"{lhs:.3f}" in note and f"{rec['subgraph_ms']:.3f}" in note, (
        "the note must show the arithmetic it is explaining, not just assert it")


def test_the_partition_note_is_generated_not_pasted():
    """Computed from the terms, so it cannot survive them changing underneath it."""
    events = []
    for i in range(4):
        base = i * 1_000_000
        events += _call(base, 60_000, base + 100, 30_000)
        events.append(_phase("record", base + 200, 5_000))
        events.append(_phase("fence_wait", base + 6_000, 20_000))
    per_call = [cp._summarise_bucket(b)
                for b in cp.bucket_by_call(events, cp.compute_calls(events))]
    rec = cp.compute_reconciliation(cp.steady_state(per_call), traced_median_ms=70.0,
                                    overhead_ratio=1.14, anchor=cp.choose_anchor(events))
    assert "30.000 ms against subgraph_ms 30.000" in rec["partition_note"], rec["partition_note"]
    assert "+0.000 ms" in rec["partition_note"], (
        "this synthetic trace splits identically on every call, so its residual is zero; "
        "the note still states the arithmetic rather than suppressing itself")


# ---------------------------------------------------------------------------
# B1: NO SPEED FIELD WITHOUT ADMISSIBILITY AND OUTPUT EQUIVALENCE
#
# `gap_ms` and `speedup_vulkan_over_cuda` were two medians divided by each other. They
# were written whatever the two arms had computed, whatever their provider attribution
# said, and `--reanalyse` copied them forward whatever the new analysis concluded — so a
# reanalysis that refused republished the refused number.
#
# The controls below are planted at the tensor and driven through the REAL `cp.main` and
# the REAL `dump_public_json`, because "the reduction returned no field" and "no field is
# in the file a reader opens" are different claims and only the second one matters. Both
# polarities: a refusing profile publishes nothing, and an equivalent admissible pair
# still publishes, because a gate that refuses everything protects nothing.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

import public_paths as pp  # noqa: E402

_REF_LOGITS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_WRONG_BY_1000 = [v * 1000.0 for v in _REF_LOGITS]

#: Vulkan is nine times faster than CUDA on every sample in every fixture below, so a
#: control that stops working stops disqualifying a *winner*.
_FAST_VULKAN = [1.0] * 5
_SLOW_CUDA = [9.0] * 5


def _traced_events():
    events = []
    for i in range(4):
        base = i * 1_000_000
        events += _call(base, 60_000, base + 100, 30_000)
        events.append(_phase("record", base + 200, 5_000))
        events.append(_phase("fence_wait", base + 6_000, 20_000))
        events.append(_gpu("matmul", base + 500, 50, ns=50_000))
    return events


def _arm_record(arm: str, scratch: Path, logits, steady) -> dict:
    """One arm's record with a real tensor on disk, shaped exactly as `dispatch_arm` returns.

    Both spellings of the two directories, because the whole defect class lives in the
    difference between them: rooted for the reader, absolute under the runtime-only key
    for the parent, and nothing that is both.
    """
    scratch = Path(scratch)
    dump = scratch / f"outputs_{arm}_prefill_1"
    dump.mkdir(parents=True, exist_ok=True)
    manifest = cc._npy_dump([np.asarray(logits, dtype=np.float32)], ["logits"], dump)
    return {"arm": arm, "workload": "prefill_1", "model_key": "m",
            "verdict": cc.ADMISSIBLE, "refusals": [], "instrument_errors": [],
            "median_ms": sorted(steady)[len(steady) // 2], "steady_ms": list(steady),
            "warmup_ms": [1.0], "compute_regime": "float32",
            "counters_scope": cc.COUNTERS_SCOPE_FIRST_RUN,
            "outputs_manifest": manifest,
            "outputs_dir": pp.public_path(dump),
            "scratch_dir": pp.public_path(scratch),
            pp.RUNTIME_ONLY_KEY: {"outputs_dir": str(dump.resolve()),
                                  "scratch_dir": str(scratch.resolve())}}


def _run_profile(tmp_path, monkeypatch, *, vulkan_logits=None, cuda_logits=None,
                 vulkan_verdict=cc.ADMISSIBLE, skip_reference=False, argv=None):
    """Drive the real `cp.main` with synthetic arms, and return the file it wrote.

    Only the two things that need a GPU are replaced — `cc.dispatch_arm` and
    `cp.run_traced_vulkan`. Everything after them is the shipped code path: the
    equivalence instrument, the speed gate, the seals, `dump_public_json`, `render`.

    `pp.REPO` is pointed at the fixture directory for the same reason
    `test_equivalence_re_derives_from_a_committed_record_without_the_runtime_channel`
    does it: a record has to root at a token this machine can resolve BACK, and pytest's
    own temp directory is named after the operator's account, which the sanitiser
    correctly scrubs and therefore cannot invert. Rooting at `<repo>` is what a real run
    does, so this is the shipped shape rather than a convenience.
    """
    monkeypatch.setattr(pp, "REPO", tmp_path)
    logits = {cc.ARM_VULKAN: vulkan_logits or _REF_LOGITS,
              cc.ARM_CUDA: cuda_logits or _REF_LOGITS,
              cc.ARM_CPU_HOST: _REF_LOGITS}
    steady = {cc.ARM_VULKAN: _FAST_VULKAN, cc.ARM_CUDA: _SLOW_CUDA,
              cc.ARM_CPU_HOST: [50.0] * 5}

    def fake_dispatch(arm, workload, *, iters, warmup, scratch, device, seed,
                      timeout=3600):
        if skip_reference and arm == cp.SPEED_REFERENCE_ARM:
            return {"arm": arm, "workload": workload.key, "verdict": cc.UNMEASURED,
                    "refusals": ["no interpreter for this arm on this machine"]}
        rec = _arm_record(arm, scratch, logits[arm], steady[arm])
        if arm == cc.ARM_VULKAN:
            rec["verdict"] = vulkan_verdict
        return rec

    def fake_traced(workload, *, iters, warmup, scratch, device, seed,
                    gpu_timestamps=True, timeout=3600):
        rec = _arm_record(cc.ARM_VULKAN, scratch, _REF_LOGITS, [1.2] * 5)
        trace = Path(scratch) / "trace_prefill_1.json"
        trace.write_text(json.dumps(_traced_events()), encoding="utf-8")
        rec["trace_rel"] = "trace_prefill_1.json"
        rec["trace_path"] = pp.public_path(trace)
        return rec

    monkeypatch.setattr(cc, "dispatch_arm", fake_dispatch)
    monkeypatch.setattr(cp, "run_traced_vulkan", fake_traced)
    out = tmp_path / "profile.json"
    rc = cp.main(argv or ["--workload", "prefill_1", "--out", str(out),
                          "--scratch", str(tmp_path / "scratch")])
    assert rc == 0
    return out, json.loads(out.read_text(encoding="utf-8"))


def test_an_equivalent_admissible_pair_still_publishes_a_speed_field(tmp_path, monkeypatch):
    """THE MUST-PASS POLARITY, through the real serialisation boundary.

    A gate that removes the fields unconditionally would pass every negative control below
    and be worth nothing. Vulkan here is nine times faster, admissible, and agrees with the
    CPU reference, so the claim stands and the numbers reach the file.
    """
    out, doc = _run_profile(tmp_path, monkeypatch)

    assert doc["speed_claim"]["verdict"] == cp.SPEED_COMPARED, doc["speed_claim"]["refusals"]
    assert doc["speed_claim"]["equivalence"]["arms"][cc.ARM_VULKAN]["verdict"] == "MATCH"
    assert doc["speed_claim"]["equivalence"]["arms"][cc.ARM_CUDA]["verdict"] == "MATCH"
    assert doc["speedup_vulkan_over_cuda"] == pytest.approx(9.0)
    assert doc["gap_ms"] == pytest.approx(-8.0)
    assert not pp.scan(out.read_text(encoding="utf-8"))
    assert pp.RUNTIME_ONLY_KEY not in out.read_text(encoding="utf-8")


def test_a_thousandfold_wrong_vulkan_output_publishes_no_speed_field(tmp_path, monkeypatch):
    """The defect, planted at the tensor and read back out of the file on disk.

    The Vulkan arm is nine times faster and returns logits a thousand times the reference's.
    The report may say so; it may not carry a number that reads as a performance result.
    """
    out, doc = _run_profile(tmp_path, monkeypatch, vulkan_logits=_WRONG_BY_1000)

    assert doc["speed_claim"]["verdict"] == cp.SPEED_REFUSED
    for field in cp.SPEED_FIELDS:
        assert field not in doc, f"{field} survived a refusing claim into the artifact"
    assert any("DIVERGENT" in r for r in doc["speed_claim"]["refusals"]), \
        doc["speed_claim"]["refusals"]
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "No Vulkan-vs-CUDA speed result is published" in md
    assert "speedup_vulkan_over_cuda" not in md.replace("`speedup_vulkan_over_cuda`", "")


def test_an_inadmissible_vulkan_arm_publishes_no_speed_field(tmp_path, monkeypatch):
    """Equivalence is not the only gate. An arm the provider never claimed is not fast."""
    out, doc = _run_profile(tmp_path, monkeypatch, vulkan_verdict=cc.SPLIT_FRAME)

    assert doc["speed_claim"]["verdict"] == cp.SPEED_REFUSED
    assert any(cc.SPLIT_FRAME in r for r in doc["speed_claim"]["refusals"])
    for field in cp.SPEED_FIELDS:
        assert field not in doc


def test_a_profile_with_no_reference_arm_publishes_no_speed_field(tmp_path, monkeypatch):
    """"We did not check" must not be spelled the same way as "they agreed"."""
    out, doc = _run_profile(tmp_path, monkeypatch, skip_reference=True)

    assert doc["speed_claim"]["verdict"] == cp.SPEED_REFUSED
    for field in cp.SPEED_FIELDS:
        assert field not in doc


def test_skip_cuda_makes_no_cross_arm_claim_at_all(tmp_path, monkeypatch):
    """No CUDA arm, no ratio, and no speed claim to refuse — the fields are simply absent."""
    out = tmp_path / "profile.json"
    _, doc = _run_profile(
        tmp_path, monkeypatch,
        argv=["--workload", "prefill_1", "--out", str(out), "--skip-cuda",
              "--scratch", str(tmp_path / "scratch")])

    assert "speed_claim" not in doc
    for field in cp.SPEED_FIELDS:
        assert field not in doc


def test_reanalysis_cannot_carry_a_refused_speed_field_forward(tmp_path, monkeypatch):
    """`--reanalyse` used to copy these two fields across whatever it concluded.

    Driven the way the defect would actually arrive: a run that legitimately published a
    speed result, then the same record reanalysed after the Vulkan arm's tensor turns out
    to disagree. Nothing about the stored numbers changes; what changes is that they are no
    longer witnessed, and a refusal that leaves the number in the file has refused nothing.
    """
    out, first = _run_profile(tmp_path, monkeypatch)
    assert first["speedup_vulkan_over_cuda"] == pytest.approx(9.0)

    vk_dir = pp.resolve_public_path(first["untraced_record"]["outputs_dir"])
    assert vk_dir is not None, first["untraced_record"]["outputs_dir"]
    np.save(vk_dir / cc.OUTPUT_HANDLE, np.asarray(_WRONG_BY_1000, dtype=np.float32))

    assert cp.main(["--workload", "prefill_1", "--out", str(out), "--reanalyse"]) == 0
    again = json.loads(out.read_text(encoding="utf-8"))

    assert again["reanalysed_from_schema"] == cp.SCHEMA
    assert again["speed_claim"]["verdict"] == cp.SPEED_REFUSED
    for field in cp.SPEED_FIELDS:
        assert field not in again, (
            f"{field} was copied forward from a report whose reanalysis refuses; the "
            f"stored number outlived the evidence for it")


def test_reanalysis_republishes_a_speed_field_it_can_still_stand_behind(tmp_path,
                                                                        monkeypatch):
    """The other polarity: reanalysis of a still-equivalent record re-derives the fields."""
    out, first = _run_profile(tmp_path, monkeypatch)

    assert cp.main(["--workload", "prefill_1", "--out", str(out), "--reanalyse"]) == 0
    again = json.loads(out.read_text(encoding="utf-8"))

    assert again["speed_claim"]["verdict"] == cp.SPEED_COMPARED
    assert again["speedup_vulkan_over_cuda"] == pytest.approx(
        first["speedup_vulkan_over_cuda"])
    assert again["gap_ms"] == pytest.approx(first["gap_ms"])


def test_a_hand_planted_speed_field_cannot_survive_a_reanalysis_that_refuses(
        tmp_path, monkeypatch):
    """The direct falsifier, with the stale number planted rather than earned.

    A stored document from any earlier revision may carry these fields; the boundary has
    to strip them on the way out, not merely decline to add them.
    """
    out, first = _run_profile(tmp_path, monkeypatch, vulkan_logits=_WRONG_BY_1000)
    assert "speedup_vulkan_over_cuda" not in first
    first["speedup_vulkan_over_cuda"] = 9.0
    first["gap_ms"] = -8.0
    out.write_text(json.dumps(first), encoding="utf-8")

    assert cp.main(["--workload", "prefill_1", "--out", str(out), "--reanalyse"]) == 0
    again = json.loads(out.read_text(encoding="utf-8"))

    for field in cp.SPEED_FIELDS:
        assert field not in again


def test_seal_speed_claim_removes_rather_than_relabels():
    """Both polarities of the seal itself, without a run around it."""
    refusing = {"speed_claim": {"verdict": cp.SPEED_REFUSED, "refusals": ["x"]},
                "gap_ms": -8.0, "speedup_vulkan_over_cuda": 9.0}
    sealed = cp.seal_speed_claim(refusing)
    assert all(f not in sealed for f in cp.SPEED_FIELDS)

    standing = {"speed_claim": {"verdict": cp.SPEED_COMPARED, "refusals": []},
                "gap_ms": -8.0, "speedup_vulkan_over_cuda": 9.0}
    assert cp.seal_speed_claim(standing)["speedup_vulkan_over_cuda"] == 9.0

    # No claim at all is a refusal for the same reason an empty equivalence map is.
    assert all(f not in cp.seal_speed_claim({"gap_ms": 1.0, "speedup_vulkan_over_cuda": 2.0})
               for f in cp.SPEED_FIELDS)


def test_speed_claim_names_every_arm_it_was_asked_about():
    """Total: a verdict and a reason for every input, and it never raises."""
    for untraced, cuda, reference in (
        (None, None, None),
        ({"arm": cc.ARM_VULKAN, "verdict": cc.ADMISSIBLE}, None, None),
        (None, {"arm": cc.ARM_CUDA, "verdict": cc.ADMISSIBLE}, None),
        ({"arm": cc.ARM_VULKAN, "verdict": cc.INSTRUMENT_ERROR},
         {"arm": cc.ARM_CUDA, "verdict": cc.ADMISSIBLE}, None),
    ):
        claim = cp.speed_claim(untraced, cuda, reference)
        assert claim["verdict"] == cp.SPEED_REFUSED
        assert claim["refusals"], (untraced, cuda, reference)
        assert claim["contributing_arms"] == list(cp.SPEED_SUBJECT_ARMS)


def test_no_report_is_serialised_without_both_seals():
    """Structural, not behavioural: one funnel, and the source is what says so.

    `dump_public_json` may appear exactly once in this module — inside `dump_profile_json`,
    which seals the verdict and the speed claim first. A future exit that writes a report
    another way cannot quietly bypass either gate, which is the failure mode that produced
    the ungated speed fields in the first place.
    """
    src = Path(cp.__file__).read_text(encoding="utf-8")
    calls = [ln.strip() for ln in src.splitlines()
             if "dump_public_json(" in ln and not ln.lstrip().startswith("from ")]
    funnel = inspect.getsource(cp.dump_profile_json)

    assert len(calls) == 1, (
        f"{len(calls)} call(s) to dump_public_json in cuda_profile.py; every report must "
        f"be written through dump_profile_json:\n  " + "\n  ".join(calls))
    assert calls[0] in funnel
    assert "seal_verdict(" in funnel and "seal_speed_claim(" in funnel


def test_the_committed_artifact_carries_no_unwitnessed_speed_field():
    """Secondary to the structural gate: one artifact, checked because it is public."""
    art = (Path(__file__).resolve().parents[1] / "bench" / "results" / "_cuda69"
           / "profile_prefill_1.json")
    if not art.exists():
        pytest.skip("committed artifact not present in this tree")
    doc = json.loads(art.read_text("utf-8"))
    published = [f for f in cp.SPEED_FIELDS if f in doc]
    if published:
        assert (doc.get("speed_claim") or {}).get("verdict") == cp.SPEED_COMPARED, (
            f"{art.name} publishes {published} with no standing speed claim")

# ---------------------------------------------------------------------------
# B6: THE SPEED TOKEN AND THE SPEED FIELDS ARE ONE STATEMENT
#
# `seal_speed_claim` asked one question — is the token green? — and deleted the fields when
# it was not. It never asked the converse, and the converse was reachable: `attribute`
# returned at `TRACE_ABSENT` BEFORE it recorded `untraced_median_ms`, so on `--reanalyse`
# of a record whose trace cannot be read, `publish_speed_claim` had no medians to divide,
# wrote no fields, and left a claim reading `SPEED_COMPARED` with
# `fields: [gap_ms, speedup_vulkan_over_cuda]` and neither field on the record.
#
# A token that says a comparison was made, naming two numbers that are not there, is worse
# than a refusal: a reader who checks the verdict and not the payload has been told the
# comparison happened. Two changes, and both are needed. The medians are recorded before
# any exit, so the honest record keeps its comparison instead of losing it to an ordering
# accident; and the seal fails closed in both directions, so an incomplete claim is
# downgraded and its partial fields deleted rather than published under a green token.
# ---------------------------------------------------------------------------

def test_a_speed_compared_token_without_its_fields_is_downgraded():
    """The invariant, directly: token and fields are consistent or the token goes.

    Unit-level and deliberately crude — a report assembled by hand in exactly the state the
    early return produced, so the seal is judged on the shape rather than on the route that
    reaches it.
    """
    report = {"speed_claim": {"verdict": cp.SPEED_COMPARED, "fields": list(cp.SPEED_FIELDS),
                              "refusals": []}}

    sealed = cp.seal_speed_claim(report)

    assert sealed["speed_claim"]["verdict"] == cp.SPEED_REFUSED
    assert sealed["speed_claim"]["incomplete_fields"] == list(cp.SPEED_FIELDS)
    for field in cp.SPEED_FIELDS:
        assert field not in sealed


def test_a_partially_complete_speed_claim_is_refused_and_stripped():
    """Half the fields is not half a comparison; it is a comparison nobody can check."""
    report = {"speed_claim": {"verdict": cp.SPEED_COMPARED, "fields": list(cp.SPEED_FIELDS)},
              cp.SPEED_FIELDS[0]: 1.25}

    sealed = cp.seal_speed_claim(report)

    assert sealed["speed_claim"]["verdict"] == cp.SPEED_REFUSED
    assert sealed["speed_claim"]["incomplete_fields"] == [cp.SPEED_FIELDS[1]]
    for field in cp.SPEED_FIELDS:
        assert field not in sealed


def test_a_complete_speed_claim_survives_the_same_seal():
    """MUST-PASS POLARITY. A seal that downgrades every claim publishes nothing, ever."""
    report = {"speed_claim": {"verdict": cp.SPEED_COMPARED, "fields": list(cp.SPEED_FIELDS)},
              "gap_ms": -8.0, "speedup_vulkan_over_cuda": 9.0}

    sealed = cp.seal_speed_claim(report)

    assert sealed["speed_claim"]["verdict"] == cp.SPEED_COMPARED
    assert sealed["gap_ms"] == -8.0
    assert sealed["speedup_vulkan_over_cuda"] == 9.0


def test_a_boolean_is_not_a_speed_field():
    """`True` is an `int` in Python, and a ratio that is `True` is not a ratio."""
    report = {"speed_claim": {"verdict": cp.SPEED_COMPARED, "fields": list(cp.SPEED_FIELDS)},
              "gap_ms": True, "speedup_vulkan_over_cuda": True}

    sealed = cp.seal_speed_claim(report)

    assert sealed["speed_claim"]["verdict"] == cp.SPEED_REFUSED


def test_reanalysis_of_a_record_whose_trace_is_gone_keeps_token_and_fields_together(
        tmp_path, monkeypatch):
    """THE REAL REANALYSIS PATH, end to end through `cp.main`.

    The route the defect arrives by: a report that legitimately published a speed result,
    reanalysed on a machine where the trace file is no longer there. `attribute` returns
    `TRACE_ABSENT`, and everything downstream of that early return used to be skipped —
    including the medians the speed fields are derived from.

    The medians do not come from the trace. They came off the dispatched records and were
    already on the stored document, so the honest outcome is that the cross-arm comparison
    survives a missing trace intact: `TRACE_ABSENT` is a statement about the phase split,
    not about the ratio.
    """
    out, first = _run_profile(tmp_path, monkeypatch)
    assert first["speed_claim"]["verdict"] == cp.SPEED_COMPARED
    trace = pp.resolve_public_path(first["traced_record"]["trace_path"])
    assert trace is not None and trace.is_file(), first["traced_record"]["trace_path"]
    trace.unlink()

    assert cp.main(["--workload", "prefill_1", "--out", str(out), "--reanalyse"]) == 0
    again = json.loads(out.read_text(encoding="utf-8"))

    assert again["verdict"] == cp.TRACE_ABSENT, again["verdict"]
    assert again["untraced_median_ms"] == pytest.approx(first["untraced_median_ms"]), (
        "the medians were lost to an early return; they are not a conclusion of the trace "
        "reduction and nothing about a missing trace makes them unknown")

    claim = again["speed_claim"]["verdict"]
    present = [f for f in cp.SPEED_FIELDS if f in again]
    assert (claim == cp.SPEED_COMPARED) == (len(present) == len(cp.SPEED_FIELDS)), (
        f"the token says {claim} and the record carries {present}; a SPEED_COMPARED token "
        f"naming fields that are not on the record is a claim a reader cannot check")
    assert claim == cp.SPEED_COMPARED, again["speed_claim"].get("refusals")
    assert again["speedup_vulkan_over_cuda"] == pytest.approx(
        first["speedup_vulkan_over_cuda"])


def test_reanalysis_without_a_median_refuses_rather_than_publishing_a_bare_token(
        tmp_path, monkeypatch):
    """The fail-closed half, on the same real path.

    Here the medians genuinely are absent — a stored record whose untraced arm produced
    none — so there is nothing to divide and nothing to publish. The token must say so
    instead of standing alone.
    """
    out, first = _run_profile(tmp_path, monkeypatch)
    first["untraced_record"].pop("median_ms", None)
    out.write_text(json.dumps(first), encoding="utf-8")

    assert cp.main(["--workload", "prefill_1", "--out", str(out), "--reanalyse"]) == 0
    again = json.loads(out.read_text(encoding="utf-8"))

    assert again["speed_claim"]["verdict"] == cp.SPEED_REFUSED
    assert again["speed_claim"]["incomplete_fields"], again["speed_claim"]
    for field in cp.SPEED_FIELDS:
        assert field not in again


def test_every_exit_from_attribute_carries_the_median_fields():
    """Structural, for the same reason the sealing guard is structural.

    The defect was an ordering one — a `return` above the assignment — and a behavioural
    test of the two exits that exist today says nothing about the third somebody adds. This
    reads `attribute`'s source and requires the medians to be recorded before any return.
    """
    src = inspect.getsource(cp.attribute)
    lines = src.splitlines()
    assigns = [i for i, ln in enumerate(lines)
               if ln.strip().startswith('out["untraced_median_ms"]')]
    returns = [i for i, ln in enumerate(lines) if ln.strip().startswith("return ")]

    assert len(assigns) == 1, (
        f"`untraced_median_ms` is assigned {len(assigns)} times in attribute(); two "
        f"assignments is two contracts and a reader cannot tell which one an exit took")
    assert returns, "attribute() has no returns; this test is reading the wrong function"
    assert assigns[0] < min(returns), (
        "attribute() can return before it records `untraced_median_ms`; that exit produces "
        "a record whose speed claim has no numbers to be about")


def test_the_serialiser_still_funnels_through_both_seals():
    """The seal is only structural while `dump_profile_json` is the only writer."""
    src = inspect.getsource(cp.dump_profile_json)

    assert "seal_verdict(" in src and "seal_speed_claim(" in src
