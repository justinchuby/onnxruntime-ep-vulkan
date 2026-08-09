"""Tests for the two-level host-cost attribution of issue #88.

WHAT IS BEING DEFENDED
======================
Issue #88 asks a single question — *where does the host cost of a `Compute` call go?* — and the
answer has **two** parts that live at different levels and must never be substituted for each
other:

* the **outer** residual, ``sum(vulkan.ort_compute_callback) - sum(vulkan.subgraph)``: what the
  ORT callback body costs *around* the engine dispatch;
* the **inner** residual, ``sum(vulkan.subgraph) - sum(top-level ep.phase spans)``: what the
  engine dispatch costs *around* its own phases.

The first attempt at this shipped one analyser that subtracted the phase spans from the callback
total and printed the result under the outer residual's definition. That number is
``outer + inner``. On the canonical case below it reads **300 µs where the answer is 100 µs**,
and a reviewer reading the label would have attributed 300 µs of cost to the callback body.

So the load-bearing property here is not "the arithmetic is right". It is **"the two levels
cannot be conflated"**: the canonical case is chosen so that outer (100), inner (200) and their
forbidden sum (300) are three different numbers, and the test asserts that 300 appears nowhere in
the artifact under any key.

MUTATION CONTROLS
=================
Every gate below that could pass vacuously is paired with a *held-out* reimplementation carrying
the exact defect it is written to catch, run through the identical protocol, and required to go
red. A test that has never been shown to fail is a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pytest  # noqa: E402

import phases  # noqa: E402


# ---------------------------------------------------------------------------
# Trace construction — microseconds, as Chrome Trace uses.
# ---------------------------------------------------------------------------

def _callback(ts, dur, *, outcome="ok", subgraph_id=0, nodes=3, tid=0, args=None):
    a = {
        "device": "host",
        "subgraph_id": subgraph_id,
        "nodes": nodes,
        phases.ARG_BOUNDARY: phases.BOUNDARY_ORT_COMPUTE_CALLBACK,
        phases.ARG_TIER: phases.TIER_CALLBACK,
        phases.ARG_OUTCOME: outcome,
    }
    a.update(args or {})
    return {"name": phases.CALLBACK, "cat": phases.CAT_COMPUTE_CALL, "ph": "X",
            "ts": ts, "dur": dur, "pid": 1, "tid": tid, "args": a}


def _dispatch(ts, dur, *, nodes=3, tid=0, args=None):
    a = {
        "device": "host",
        "nodes": nodes,
        phases.ARG_BOUNDARY: phases.BOUNDARY_ENGINE_DISPATCH,
        phases.ARG_TIER: phases.TIER_DISPATCH,
    }
    a.update(args or {})
    return {"name": phases.SUBGRAPH, "cat": phases.CAT_DISPATCH, "ph": "X",
            "ts": ts, "dur": dur, "pid": 1, "tid": tid, "args": a}


def _phase(name, ts, dur, *, tid=0, nested_in=None, caveat="x"):
    a = {"device": "host", "caveat": caveat, phases.ARG_TIER: phases.TIER_PHASE}
    if nested_in:
        a["nested_in"] = nested_in
        a["caveat"] = phases.SUB_PHASE_CAVEAT_PREFIX + " " + caveat
    return {"name": f"vulkan.{name}", "cat": phases.CAT_PHASE, "ph": "X",
            "ts": ts, "dur": dur, "pid": 1, "tid": tid, "args": a}


def canonical_trace():
    """The case Seraph named: compute=1000 us, subgraph=900 us, top-level phases=700 us.

    Laid out so every containment relation the analyser checks actually holds::

        callback   [1000, 2000)   1000 us
          dispatch [1050, 1950)    900 us
            record [1060, 1460)    400 us      <- top-level
            submit [1460, 1560)    100 us      <- top-level
            fence  [1560, 1760)    200 us      <- top-level      sum = 700
              cmd_upload [1070, 1450)          <- NESTED inside record, never summed here

    outer = 1000 - 900 = **100**    (callback body around the dispatch)
    inner =  900 - 700 = **200**    (dispatch around its own phases)
    outer + inner = 300, which is the number the rejected analyser printed under the outer's
    name, and which must not appear anywhere in the artifact.
    """
    return [
        _callback(1000, 1000),
        _dispatch(1050, 900),
        _phase("record", 1060, 400),
        _phase("submit", 1460, 100),
        _phase("fence_wait", 1560, 200),
        _phase("cmd_upload", 1070, 380, nested_in="record"),
    ]


def _values(obj):
    """Every scalar reachable in a nested dict/list, for the 'no key holds 300' assertion."""
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_values(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_values(v))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(obj)
    return out


# ---------------------------------------------------------------------------
# B1 — the canonical case, and the number that must not exist
# ---------------------------------------------------------------------------

def test_the_canonical_case_reports_outer_100_and_inner_200_separately():
    r = phases.two_level_attribution(canonical_trace())
    assert r["state"] == "PASS", r["detail"]
    assert r["callback_total_us"] == 1000
    assert r["dispatch_total_us"] == 900
    assert r["top_level_phase_total_us"] == 700
    assert r["outer_residual_us"] == 100
    assert r["inner_residual_us"] == 200


def test_the_two_residuals_are_never_conflated_into_300():
    """The defect this whole module exists for: 300 is outer+inner and means nothing."""
    r = phases.two_level_attribution(canonical_trace())
    assert 300 not in _values(r), (
        "a value of 300 appeared in the attribution artifact. 300 = outer(100) + inner(200), a "
        "sum of two residuals of different intervals against different denominators. Publishing "
        "it — under any key — is the rejected reading of issue #88.")
    assert r["outer_residual_us"] != r["inner_residual_us"]
    for key in r:
        assert "combined" not in key, key
        assert key not in ("residual_us", "host_residual_us", "total_residual_us",
                           "unattributed_us"), (
            f"{key!r} names a residual without saying WHICH level it belongs to. The names in "
            f"this dict are the only thing preventing substitution.")


def test_each_residual_is_a_share_of_its_own_denominator_and_says_which():
    r = phases.two_level_attribution(canonical_trace())
    assert r["outer_residual_pct_of_callback_total"] == pytest.approx(10.0)
    assert r["inner_residual_pct_of_dispatch_total"] == pytest.approx(200 / 900 * 100)
    # The inner residual as a share of the CALLBACK total would be 20% — a real number about
    # nothing, and the easiest substitution to make by accident. It is not published.
    assert 20.0 not in _values(r)
    assert "sum(vulkan.ort_compute_callback.dur) - sum(vulkan.subgraph.dur)" in \
        r["outer_residual_def"]
    assert "sum(vulkan.subgraph.dur)" in r["inner_residual_def"]
    assert r["residuals_are_disjoint_levels"] is True
    assert "not two halves of one quantity" in r["never_sum_note"]


def test_the_dispatch_span_is_used_not_discarded():
    """The rejected analyser had vulkan.subgraph in the fixture and ignored it."""
    r = phases.two_level_attribution(canonical_trace())
    assert r["dispatch_spans"] == 1
    assert r["dispatch_total_us"] == 900
    # Removing the dispatch span must change the answer. If it does not, the analyser never
    # read it, which is exactly the rejected behaviour.
    without = [e for e in canonical_trace() if e["name"] != phases.SUBGRAPH]
    r2 = phases.two_level_attribution(without)
    assert r2["outer_residual_us"] != r["outer_residual_us"]


# ---------------------------------------------------------------------------
# B1 — held-out mutants of the analyser itself
# ---------------------------------------------------------------------------

def _outer_from(report):
    """The protocol: whatever an analyser publishes as issue #88's outer residual."""
    return report.get("outer_residual_us")


def test_the_rejected_analyser_would_fail_this_module():
    """The exact mutant Seraph rejected, run through this module's own gate.

    `phases_minus_toplevel` computes `callback - phases`, which is `outer + inner`, and labels
    it as the outer residual. It must not be able to pass.
    """
    ev = canonical_trace()
    calls = phases.callback_spans(ev)
    sib = phases.sibling_phases(phases.phase_spans(ev))
    mutant = {
        "outer_residual_us": sum(c["dur"] for c in calls) - sum(p["dur"] for p in sib),
    }
    assert _outer_from(mutant) == 300
    assert _outer_from(mutant) != _outer_from(phases.two_level_attribution(ev)), (
        "the shipped analyser agrees with the rejected one; the gate cannot fire")


def test_held_out_mutants_of_the_two_level_split_all_disagree_with_the_shipped_one():
    ev = canonical_trace()
    shipped = phases.two_level_attribution(ev)
    calls = phases.callback_spans(ev)
    disp = [e for e in ev if e["name"] == phases.SUBGRAPH]
    sib = phases.sibling_phases(phases.phase_spans(ev))
    cb, dp, ph = (sum(c["dur"] for c in calls),
                  sum(d["dur"] for d in disp),
                  sum(p["dur"] for p in sib))
    mutants = {
        "subtracts_phases_from_the_callback (the rejected one)": cb - ph,
        "adds_the_two_levels": (cb - dp) + (dp - ph),
        "substitutes_inner_for_outer": dp - ph,
        "forgets_the_nesting_and_uses_all_phase_spans": cb - dp - ph,
        "uses_the_dispatch_as_the_callback": dp - dp,
    }
    for label, value in mutants.items():
        assert value != shipped["outer_residual_us"] or label.startswith("uses_the_dispatch"), (
            f"mutant {label!r} produced the shipped outer residual; this gate is not load-bearing")
    # The one mutant that could collide numerically is stated explicitly rather than hidden:
    # `dp - dp` is 0, and 0 is not 100, so it too disagrees on this fixture.
    assert mutants["uses_the_dispatch_as_the_callback"] == 0 != shipped["outer_residual_us"]


# ---------------------------------------------------------------------------
# B1 — refusal: every way the evidence can be inadmissible
# ---------------------------------------------------------------------------

def _assert_refused(report):
    assert report["state"] == "REFUSED", report
    assert report["outer_residual_us"] is None
    assert report["inner_residual_us"] is None
    assert report["refusals"], "a refusal with no stated reason is a silent drop"
    for key in report:
        assert "_pct_" not in key, (
            f"{key!r} survived a refusal. A stripped percentage must be ABSENT, not 0.0 — a "
            f"reader formatting a refusal as '0.0%' is the failure this rule prevents.")


def test_a_trace_with_no_callback_span_is_vacuous_not_zero():
    ev = [e for e in canonical_trace() if e["name"] != phases.CALLBACK]
    r = phases.two_level_attribution(ev)
    assert r["state"] == "VACUOUS"
    assert r["outer_residual_us"] is None
    assert r["callback_total_us"] is None
    assert "not evidence that the callback body costs nothing" in r["detail"]


def test_an_empty_trace_is_vacuous():
    r = phases.two_level_attribution([])
    assert r["state"] == "VACUOUS"
    assert r["outer_residual_us"] is None


def test_empty_callback_spans_refuse_rather_than_divide_by_zero():
    _assert_refused(phases.two_level_attribution([_callback(1000, 0), _dispatch(1000, 0)]))


def test_an_escaped_dispatch_span_is_refused():
    ev = [_callback(1000, 1000), _dispatch(5000, 900)]
    _assert_refused(phases.two_level_attribution(ev))


def test_nested_callback_spans_are_refused():
    ev = [_callback(1000, 1000), _callback(1200, 100), _dispatch(1050, 900)]
    _assert_refused(phases.two_level_attribution(ev))


def test_overlapping_callback_spans_are_refused():
    ev = [_callback(1000, 1000), _callback(1900, 500), _dispatch(1050, 900)]
    _assert_refused(phases.two_level_attribution(ev))


def test_an_unknown_phase_span_refuses_the_whole_attribution():
    ev = canonical_trace() + [_phase("teleport", 1800, 50)]
    _assert_refused(phases.two_level_attribution(ev))


def test_a_callback_missing_its_boundary_arg_is_refused():
    ev = canonical_trace()
    ev[0]["args"].pop(phases.ARG_BOUNDARY)
    _assert_refused(phases.two_level_attribution(ev))


def test_a_dispatch_missing_its_tier_arg_is_refused():
    ev = canonical_trace()
    ev[1]["args"][phases.ARG_TIER] = phases.TIER_CALLBACK
    _assert_refused(phases.two_level_attribution(ev))


def test_a_callback_with_no_duration_is_refused_not_read_as_zero():
    ev = canonical_trace()
    ev[0].pop("dur")
    _assert_refused(phases.two_level_attribution(ev))


def test_a_negative_duration_is_refused():
    ev = canonical_trace()
    ev[1]["dur"] = -5
    _assert_refused(phases.two_level_attribution(ev))


def test_a_dispatch_longer_than_its_callback_is_refused_not_reported_negative():
    ev = [_callback(1000, 100), _dispatch(1000, 900)]
    _assert_refused(phases.two_level_attribution(ev))


def test_phases_summing_past_their_dispatch_are_refused():
    ev = [_callback(1000, 5000), _dispatch(1000, 900),
          _phase("record", 1000, 800), _phase("submit", 1000, 800)]
    _assert_refused(phases.two_level_attribution(ev))


def test_an_unresolved_outcome_is_refused_rather_than_read_as_success():
    ev = canonical_trace()
    ev[0]["args"][phases.ARG_OUTCOME] = "unresolved"
    r = phases.two_level_attribution(ev)
    _assert_refused(r)
    assert r["outcomes"]["unresolved"] == 1


def test_a_failed_call_is_disclosed_and_never_silently_dropped():
    """A failing Compute is a real callback body. Dropping it shrinks the denominator."""
    ev = canonical_trace() + [_callback(3000, 40, outcome="failed", subgraph_id=1, nodes=0)]
    r = phases.two_level_attribution(ev)
    assert r["state"] == "PASS", r["detail"]
    assert r["outcomes"] == {"ok": 1, "failed": 1}
    assert r["callback_total_us"] == 1040
    assert r["outer_residual_us"] == 140


def test_a_phase_span_outside_every_dispatch_is_refused():
    ev = canonical_trace() + [_phase("submit", 5000, 10)]
    _assert_refused(phases.two_level_attribution(ev))


def test_compile_and_prepack_run_outside_a_dispatch_and_are_not_treated_as_escapes():
    ev = canonical_trace() + [_phase("compile", 10, 500), _phase("prepack", 520, 300)]
    r = phases.two_level_attribution(ev)
    assert r["state"] == "PASS", r["detail"]
    assert r["top_level_phase_total_us"] == 700
    assert r["inner_residual_us"] == 200


def test_nested_sub_record_spans_never_enter_the_inner_numerator():
    """cmd_upload lives inside record; adding it counts the same microseconds twice."""
    r = phases.two_level_attribution(canonical_trace())
    assert "cmd_upload" not in r["top_level_phase_us"]
    assert sorted(r["top_level_phase_us"]) == ["fence_wait", "record", "submit"]


# ---------------------------------------------------------------------------
# unknown_phase_spans — the fail-closed drift screen
# ---------------------------------------------------------------------------

def test_unknown_phase_spans_is_quiet_on_a_clean_trace():
    r = phases.unknown_phase_spans(canonical_trace())
    assert r["red"] is False
    assert r["state"] == "PASS"
    assert r["unknown_phase_spans"] == []


def test_unknown_phase_spans_goes_red_on_a_drifted_phase_name():
    r = phases.unknown_phase_spans(canonical_trace() + [_phase("teleport", 1800, 50)])
    assert r["red"] is True
    assert r["unknown_phase_spans"] == ["vulkan.teleport"]
    assert r["unknown_counts"]["vulkan.teleport"] == 1


def test_a_clean_trace_holds_no_entry_for_a_name_it_never_saw():
    """Reject polarity: the drift map must be EMPTY, not populated with zeros."""
    with pytest.raises(KeyError):
        phases.unknown_phase_spans(canonical_trace())["unknown_counts"]["vulkan.teleport"]


def test_a_refused_attribution_publishes_no_percentage_key_at_all():
    """Reject polarity: the percentage is absent, so reading it raises."""
    ev = canonical_trace() + [_phase("teleport", 1800, 50)]
    with pytest.raises(KeyError):
        phases.two_level_attribution(ev)["outer_residual_pct_of_callback_total"]


def test_unknown_phase_spans_does_not_fire_on_a_non_phase_category():
    """A gpu span or a counter is not a drifted phase; firing on it would cost the screen its
    authority the first time somebody adds an unrelated event."""
    ev = canonical_trace() + [
        {"name": "vulkan.gpu.k", "cat": "gpu", "ph": "X", "ts": 1, "dur": 1, "args": {}},
        {"name": "vulkan.compute", "cat": "ep", "ph": "i", "ts": 1, "args": {}},
    ]
    assert phases.unknown_phase_spans(ev)["red"] is False


# ---------------------------------------------------------------------------
# callback_spans — selection by category, never by name prefix
# ---------------------------------------------------------------------------

def test_callback_spans_reads_the_args_the_ep_stamps():
    got = phases.callback_spans(canonical_trace())
    assert len(got) == 1
    assert got[0]["valid"] is True
    assert got[0]["outcome"] == "ok"
    assert got[0]["tier"] == phases.TIER_CALLBACK
    assert got[0]["boundary"] == phases.BOUNDARY_ORT_COMPUTE_CALLBACK
    assert got[0]["index"] == 0


def test_callback_spans_marks_a_malformed_span_invalid_instead_of_repairing_it():
    ev = [_callback(1000, 1000, args={phases.ARG_TIER: "phase"})]
    got = phases.callback_spans(ev)
    assert got[0]["valid"] is False
    assert "tier" in got[0]["invalid_reason"]


def test_the_record_path_instant_is_not_mistaken_for_a_callback_span():
    """Reject polarity: selection is by cat+name, so a prefix-matching INSTANT selects nothing.

    `vulkan.compute[FIRST_RECORD]` is a `ph:"i"` event emitted by the record-path counter. An
    analyser that matched on a `vulkan.compute` prefix would pick it up, give it no duration,
    and quietly inflate the callback count.
    """
    ev = [{"name": "vulkan.compute[FIRST_RECORD]", "cat": "ep", "ph": "i", "ts": 1,
           "pid": 1, "tid": 0, "args": {"path": "FIRST_RECORD"}}]
    with pytest.raises(IndexError):
        phases.callback_spans(ev)[0]


# ---------------------------------------------------------------------------
# Integration with analyse()
# ---------------------------------------------------------------------------

def test_analyse_carries_the_attribution_and_the_drift_screen():
    report = phases.analyse(canonical_trace())
    tla = report["two_level_attribution"]
    assert tla["state"] == "PASS", tla["detail"]
    assert tla["outer_residual_us"] == 100
    assert tla["inner_residual_us"] == 200
    assert report["falsifiers"]["unknown_phase_spans"]["red"] is False
    assert 300 not in _values(tla)


def test_analyse_still_labels_time_in_compute_as_the_dispatch_total():
    report = phases.analyse(canonical_trace())
    note = report["time_in_compute_note"]
    assert "ENGINE DISPATCH" in note
    assert "outer_residual_us" in note
    assert report["time_in_compute_ms"] == pytest.approx(0.9)


def test_a_refused_attribution_is_listed_by_red_flags_as_not_a_detection():
    ev = canonical_trace() + [_phase("teleport", 1800, 50)]
    report = phases.analyse(ev)
    flags = phases.red_flags(report)
    assert any(f.startswith("two_level_attribution: REFUSED") for f in flags), flags
    assert any("NOT a detection" in f for f in flags)


def test_describe_prints_both_levels_with_their_own_denominators():
    lines = phases.describe(phases.analyse(canonical_trace()))
    text = "\n".join(lines)
    assert "OUTER" in text and "INNER" in text
    assert "callback total" in text and "dispatch total" in text
    assert "Do not add them" in text
