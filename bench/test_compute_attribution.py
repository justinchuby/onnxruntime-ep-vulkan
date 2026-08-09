"""Two-polarity screen and mutation battery for the issue #88 attribution instruments.

WHAT IS UNDER TEST
==================
Two **total** instruments in ``bench/phases.py`` — they return ``(value, why)`` and never raise,
so ``rust/tools/audit_instruments.py`` can only see their polarities through
``bench/_polarity.py``'s ``refuses`` / ``selects``:

* :func:`phases.unknown_phase_spans` — refuses a trace containing an ``ep.phase`` span this
  module cannot account for. It closes a **silent drop**: ``phase_spans`` filters on
  ``HOST_PHASES`` membership, so a phase added to ``trace.rs`` and not added here vanishes from
  the table with no warning and every share becomes a percentage of the wrong denominator.
* :func:`phases.compute_call_attribution` — decomposes each whole ``Compute`` callback into
  disjoint siblings plus a **computed** residual, and refuses when the total span is absent,
  when its boundary is undeclared, when the phase vocabulary is unknown, when the siblings
  oversubscribe the total, or when no call ended ``ok``.

WHY THE MUTATION BATTERY IS HERE AND NOT ONLY THE HAPPY PATH
===========================================================
The census's ``screened`` verdict credits a test file for containing both polarities. That is a
claim about the file's *shape*. What makes it a claim about the *instrument* is section 3: eight
deliberately defective reimplementations, each of which must fail the identical protocol the
real instrument passes. The model is ``bench/test_devices_identity.py``, and the protocol is
factored into one callable for the same reason it is there — a battery that runs a weaker
protocol than the real tests proves nothing about the real tests.

Nothing in this file is gated on a device, a model or a platform. The census disqualifies a
polarity test that carries ``require_vulkan``, ``skipif``, ``slow`` or ``gpu``, and it is right
to: an instrument screened only on a machine with a GPU is unscreened on the machine where the
regression lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import phases  # noqa: E402
from _polarity import PolarityError, refuses, selects  # noqa: E402


# ---------------------------------------------------------------------------
# 1. TRACE FIXTURES — the smallest artifacts that carry the properties under test
# ---------------------------------------------------------------------------


def _phase(name: str, ts: int, dur: int, nested_in: str = "none") -> dict:
    return {
        "ph": "X",
        "cat": "ep.phase",
        "name": f"vulkan.{name}",
        "ts": ts,
        "dur": dur,
        "args": {"device": "host", "caveat": "…", "nested_in": nested_in},
    }


def _total(ts: int, dur: int, outcome: str = "ok", boundary: "str | None" = None) -> dict:
    args = {"device": "host", "nodes": 3, "outcome": outcome}
    if boundary is not None:
        args["boundary"] = boundary
    elif boundary is None:
        args["boundary"] = phases.COMPUTE_CALL_BOUNDARY
    return {
        "ph": "X",
        "cat": "ep",
        "name": phases.COMPUTE_CALL,
        "ts": ts,
        "dur": dur,
        "args": args,
    }


def _subgraph(ts: int, dur: int) -> dict:
    return {
        "ph": "X",
        "cat": "ep",
        "name": phases.SUBGRAPH,
        "ts": ts,
        "dur": dur,
        "args": {"device": "host", "nodes": 3},
    }


def _good_trace() -> "list[dict]":
    """One admissible call: a 1000 µs total containing 700 µs of disjoint siblings.

    The 300 µs difference is the point of the whole instrument — it is the binding checks and
    the dispatch prologue, real host cost that no phase span names, and it must come out of the
    subtraction rather than being assumed to be zero.
    """
    return [
        _total(1_000, 1_000),
        _subgraph(1_050, 900),
        _phase("record", 1_060, 500),
        # A CHILD: inside `record`, and therefore never added to the sibling total.
        _phase("cmd_upload", 1_070, 400, nested_in="record"),
        _phase("submit", 1_600, 50),
        _phase("fence_wait", 1_660, 150),
        # The `ep.path` INSTANT, which shares a prefix with the total span and means something
        # else. It must not be mistaken for either a phase or a total.
        {
            "ph": "i",
            "cat": "ep.path",
            "name": "vulkan.compute[FIRST_RECORD]",
            "ts": 1_560,
            "args": {"path": "FIRST_RECORD"},
        },
    ]


def _trace_with_unknown_phase() -> "list[dict]":
    """The exact regression this screen exists for: a new sibling phase in trace.rs only."""
    return _good_trace() + [_phase("bind_check", 1_010, 40)]


# ---------------------------------------------------------------------------
# 2. THE REAL INSTRUMENTS, IN BOTH POLARITIES
# ---------------------------------------------------------------------------


def test_a_trace_whose_phase_vocabulary_is_fully_known_is_certified_by_identity():
    """Accept polarity. Compared with ``is``: a look-alike set must not pass for the real one."""
    selects(phases.unknown_phase_spans(_good_trace()), phases.KNOWN_SPAN_NAMES)
    # An empty trace has no unknown phases. Vacuous, and honest about being so.
    selects(phases.unknown_phase_spans([]), phases.KNOWN_SPAN_NAMES)


def test_a_phase_span_this_module_cannot_account_for_refuses_and_names_it():
    """Reject polarity, and the reason must be actionable, not merely non-empty."""
    why = refuses(phases.unknown_phase_spans(_trace_with_unknown_phase()))
    assert "vulkan.bind_check" in why, (
        "a refusal that does not name the offending span leaves the reader to diff two "
        "vocabularies by hand, which is the work the instrument was supposed to do"
    )
    assert "HOST_PHASES" in why, "the refusal must say where the repair goes"


def test_the_ep_path_instant_is_not_mistaken_for_an_unknown_phase():
    """``vulkan.compute[<PATH>]`` shares a prefix with ``vulkan.compute_call``.

    A prefix match, or a match that ignores ``cat``, reads a recording-path marker as a phase and
    refuses a perfectly good trace. That is a false refusal, and a screen that fires on good
    artifacts gets turned off.
    """
    only_the_instant = [e for e in _good_trace() if e.get("cat") == "ep.path"]
    selects(phases.unknown_phase_spans(only_the_instant), phases.KNOWN_SPAN_NAMES)


def test_an_admissible_call_is_decomposed_into_siblings_and_a_computed_residual():
    rows, why = phases.compute_call_attribution(_good_trace())
    assert rows is not None, why
    assert len(rows) == 1
    row = rows[0]
    assert row["total_ms"] == pytest.approx(1.0)
    # record + submit + fence_wait = 500 + 50 + 150. `cmd_upload` is a CHILD and is excluded.
    assert row["sibling_ms"] == pytest.approx(0.7), (
        "the sibling total must exclude nested phases; including cmd_upload would double-count "
        "400 of record's own 500 µs"
    )
    assert row["residual_ms"] == pytest.approx(0.3), (
        "the residual is a subtraction, and it is the quantity issue #88 asked for"
    )
    assert "never assumed zero" in why


def test_the_residual_varies_with_its_input_rather_than_with_its_name():
    """A residual that does not move when the siblings move is a constant in disguise (R10)."""
    trace = _good_trace()
    base, _ = phases.compute_call_attribution(trace)
    # Halve the one large sibling; the residual must grow by exactly what record lost.
    widened = [dict(e) for e in trace]
    for e in widened:
        if e["name"] == "vulkan.record":
            e["dur"] = 250
    grown, _ = phases.compute_call_attribution(widened)
    assert grown[0]["residual_ms"] == pytest.approx(base[0]["residual_ms"] + 0.25)
    assert grown[0]["residual_ms"] != base[0]["residual_ms"]


def test_attribution_refuses_a_trace_with_no_total_span():
    why = refuses(phases.compute_call_attribution([e for e in _good_trace()
                                                   if e["name"] != phases.COMPUTE_CALL]))
    assert phases.SUBGRAPH in why, (
        "the refusal must say why vulkan.subgraph is not a substitute — that substitution is "
        "the exact error the boundary move corrects"
    )


def test_attribution_refuses_a_total_that_does_not_declare_where_it_starts():
    trace = [dict(e) for e in _good_trace()]
    for e in trace:
        if e["name"] == phases.COMPUTE_CALL:
            e["args"] = {k: v for k, v in e["args"].items() if k != "boundary"}
    why = refuses(phases.compute_call_attribution(trace))
    assert "boundary" in why


def test_attribution_refuses_when_the_only_calls_present_were_refused_or_unresolved():
    for outcome in ("failed", "unresolved"):
        trace = [dict(e) for e in _good_trace()]
        for e in trace:
            if e["name"] == phases.COMPUTE_CALL:
                e["args"] = dict(e["args"], outcome=outcome)
        why = refuses(phases.compute_call_attribution(trace))
        assert outcome in why, "the refusal must disclose WHICH inadmissible population it saw"
        assert "prefix" in why


def test_attribution_refuses_oversubscription_independently_of_the_outcome():
    """A negative residual is a broken instrument, not a small number.

    Checked on every call, admissible or not: siblings that exceed the total they are inside
    mean the spans are not nested the way the model says, and that is true whether or not the
    call succeeded.
    """
    # Three siblings that all fit inside the 1000 us total but OVERLAP each other, so their sum
    # is 2700 us. Overlapping siblings are the interesting case: each one is individually
    # contained, and only the sum reveals that they cannot all be disjoint intervals.
    trace = [
        _total(1_000, 1_000),
        _phase("record", 1_010, 900),
        _phase("submit", 1_020, 900),
        _phase("fence_wait", 1_030, 900),
    ]
    why = refuses(phases.compute_call_attribution(trace))
    assert "OVERSUBSCRIBED" in why
    # …and the same trace with a `failed` outcome still refuses for the oversubscription reason,
    # not for the outcome reason. Order matters: reporting "no admissible calls" would hide a
    # structurally broken instrument behind a routine-looking refusal.
    failed = [
        dict(e, args=dict(e["args"], outcome="failed"))
        if e["name"] == phases.COMPUTE_CALL
        else e
        for e in trace
    ]
    assert "OVERSUBSCRIBED" in refuses(phases.compute_call_attribution(failed))


def test_attribution_refuses_a_sibling_that_outlives_the_total_it_started_inside():
    """The silent drop, one level up.

    A plain containment filter skips a span that starts inside the total and ends after it. Its
    whole cost then leaves the sibling sum and reappears in the residual, labelled as host cost
    nobody has named — which is the opposite of what happened.
    """
    trace = [dict(e) for e in _good_trace()]
    for e in trace:
        if e["name"] == "vulkan.record":
            e["dur"] = 5_000  # starts at 1060, ends at 6060; the total closes at 2000.
    why = refuses(phases.compute_call_attribution(trace))
    assert "ESCAPES" in why and "record" in why
    assert "unattributed" in why


def test_attribution_refuses_a_trace_whose_phase_vocabulary_is_unknown():
    """Fail closed: an unknown phase removes the attribution, it does not warn about it."""
    why = refuses(phases.compute_call_attribution(_trace_with_unknown_phase()))
    assert "vulkan.bind_check" in why


def test_inadmissible_calls_are_disclosed_rather_than_dropped():
    trace = _good_trace() + [
        _total(10_000, 800, outcome="failed"),
        _phase("record", 10_010, 100),
    ]
    rows, why = phases.compute_call_attribution(trace)
    assert rows is not None and len(rows) == 1, "only the `ok` call may be decomposed"
    assert "excluded as failed" in why, (
        "a call that was silently dropped is indistinguishable from a call that never happened"
    )


# ---------------------------------------------------------------------------
# 3. THE MUTATION BATTERY — every planted defect must fail the real protocol
# ---------------------------------------------------------------------------


def _mutant_ignores_cat_and_matches_by_prefix(events):
    """Reads the ``ep.path`` instant as a phase. Refuses a good trace."""
    bad = sorted(
        {
            e["name"]
            for e in events
            if isinstance(e.get("name"), str)
            and e["name"].startswith("vulkan.")
            and e["name"] not in phases.KNOWN_SPAN_NAMES
        }
    )
    if bad:
        return None, f"unknown: {bad}"
    return phases.KNOWN_SPAN_NAMES, "ok"


def _mutant_accepts_everything(events):
    """The silent drop, preserved. This is the pre-#88 behaviour of ``phase_spans``."""
    return phases.KNOWN_SPAN_NAMES, "no unknown phases"


def _mutant_returns_an_equal_but_different_set(events):
    """Passes ``==`` and fails ``is`` — the confusion ``selects`` exists to end.

    ``frozenset(a_frozenset)`` returns the *same object* in CPython, so the look-alike has to be
    built from a plain ``set`` to actually be a different object. That detail is why this mutant
    is here: a battery that accidentally re-used the real object would prove nothing.
    """
    for e in events:
        if e.get("cat") == "ep.phase" and e.get("name") not in phases.KNOWN_SPAN_NAMES:
            return None, f"unknown: {e['name']}"
    look_alike = frozenset(set(phases.KNOWN_SPAN_NAMES))
    assert look_alike == phases.KNOWN_SPAN_NAMES and look_alike is not phases.KNOWN_SPAN_NAMES
    return look_alike, "ok"


def _mutant_refuses_without_saying_which(events):
    for e in events:
        if e.get("cat") == "ep.phase" and e.get("name") not in phases.KNOWN_SPAN_NAMES:
            return None, "unknown phase"
    return phases.KNOWN_SPAN_NAMES, "ok"


_UNKNOWN_MUTANTS = {
    "ignores_cat_and_matches_by_prefix": _mutant_ignores_cat_and_matches_by_prefix,
    "accepts_everything": _mutant_accepts_everything,
    "returns_an_equal_but_different_set": _mutant_returns_an_equal_but_different_set,
    "refuses_without_saying_which": _mutant_refuses_without_saying_which,
}


def _unknown_protocol(fn) -> None:
    """The protocol the real ``unknown_phase_spans`` passes in section 2, as one callable."""
    selects(fn(_good_trace()), phases.KNOWN_SPAN_NAMES)
    selects(fn([e for e in _good_trace() if e.get("cat") == "ep.path"]),
            phases.KNOWN_SPAN_NAMES)
    why = refuses(fn(_trace_with_unknown_phase()))
    if "vulkan.bind_check" not in why:
        raise PolarityError(
            f"the refusal did not name the offending span: {why!r}. A refusal that does not say "
            f"WHICH phase is unknown cannot be acted on."
        )


def test_the_real_unknown_phase_spans_passes_the_protocol():
    """The control's control: the battery below means nothing if the real one fails here."""
    _unknown_protocol(phases.unknown_phase_spans)


@pytest.mark.parametrize("mutant_name", sorted(_UNKNOWN_MUTANTS))
def test_a_defective_unknown_phase_spans_is_caught_by_this_protocol(mutant_name: str):
    with pytest.raises(PolarityError):
        _unknown_protocol(_UNKNOWN_MUTANTS[mutant_name])


def _mutant_attr_sums_children_too(events):
    """Adds nested spans to the sibling total: double-counts ``cmd_upload`` inside ``record``."""
    known, why = phases.unknown_phase_spans(events)
    if known is None:
        return None, why
    rows, w = phases.compute_call_attribution(events)
    if rows is None:
        return None, w
    for r in rows:
        r["sibling_ms"] += 0.4
        r["residual_ms"] -= 0.4
    return rows, w


def _mutant_attr_assumes_a_zero_residual(events):
    rows, w = phases.compute_call_attribution(events)
    if rows is None:
        return None, w
    for r in rows:
        r["residual_ms"] = 0.0
    return rows, w


def _mutant_attr_accepts_a_failed_call(events):
    """Reports a refused call's prefix of spans as though it were a decomposition."""
    patched = [
        dict(e, args=dict(e.get("args") or {}, outcome="ok"))
        if e.get("name") == phases.COMPUTE_CALL
        else e
        for e in events
    ]
    return phases.compute_call_attribution(patched)


def _mutant_attr_falls_back_to_the_subgraph_span(events):
    """Substitutes ``vulkan.subgraph`` for the total — the pre-#88 boundary error itself."""
    rewritten = [
        dict(e, name=phases.COMPUTE_CALL,
             args=dict(e.get("args") or {},
                       boundary=phases.COMPUTE_CALL_BOUNDARY, outcome="ok"))
        if e.get("name") == phases.SUBGRAPH
        else e
        for e in events
        if e.get("name") != phases.COMPUTE_CALL
    ]
    return phases.compute_call_attribution(rewritten)


_ATTR_MUTANTS = {
    "sums_children_too": _mutant_attr_sums_children_too,
    "assumes_a_zero_residual": _mutant_attr_assumes_a_zero_residual,
    "accepts_a_failed_call": _mutant_attr_accepts_a_failed_call,
    "falls_back_to_the_subgraph_span": _mutant_attr_falls_back_to_the_subgraph_span,
}


def _attr_protocol(fn) -> None:
    """The protocol the real ``compute_call_attribution`` passes in section 2."""
    rows, why = fn(_good_trace())
    if rows is None:
        raise PolarityError(f"refused a good trace: {why!r}")
    if len(rows) != 1:
        raise PolarityError(f"expected exactly one admissible call, got {len(rows)}")
    r = rows[0]
    if abs(r["sibling_ms"] - 0.7) > 1e-9:
        raise PolarityError(
            f"sibling total is {r['sibling_ms']} ms, expected 0.7 — nested spans must not be "
            f"added to their parent's siblings"
        )
    if abs(r["residual_ms"] - 0.3) > 1e-9:
        raise PolarityError(
            f"residual is {r['residual_ms']} ms, expected 0.3 — it must be computed from the "
            f"total, never assumed"
        )
    # …and the refusals.
    failed = [
        dict(e, args=dict(e["args"], outcome="failed"))
        if e.get("name") == phases.COMPUTE_CALL
        else e
        for e in _good_trace()
    ]
    refuses(fn(failed), because="every call in this trace was refused")
    no_total = [e for e in _good_trace() if e.get("name") != phases.COMPUTE_CALL]
    refuses(fn(no_total), because="there is no caller-visible total to subtract from")
    refuses(fn(_trace_with_unknown_phase()), because="the phase vocabulary is not accounted for")


def test_the_real_compute_call_attribution_passes_the_protocol():
    _attr_protocol(phases.compute_call_attribution)


@pytest.mark.parametrize("mutant_name", sorted(_ATTR_MUTANTS))
def test_a_defective_compute_call_attribution_is_caught_by_this_protocol(mutant_name: str):
    with pytest.raises(PolarityError):
        _attr_protocol(_ATTR_MUTANTS[mutant_name])
