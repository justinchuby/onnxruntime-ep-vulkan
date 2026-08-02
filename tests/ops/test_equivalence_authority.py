"""Which `model_output_equivalence` is of record — round 28, item 2.

`bench/results/phi35-certified-dev0.json`, the artifact behind our only quotable figure,
carries two fields of the same name, adjacent, disagreeing:

    results[0].model_output_equivalence          = MATCH
    results[0].counters.model_output_equivalence = UNMEASURED

The answer is that they are not two sources disagreeing. The nested one is
`rust/src/counters.rs::to_json()`'s **default** for a field only the Python comparison
harness ever writes. It is a hole, not a reading — and R12 says a hole must say so.

These tests are always-on and need no device: they read artifacts already on disk. That is
deliberate. The defect they guard is a *reader* defect — a human quoting the wrong one of
two same-named values — and a reader defect is present the moment the file is, not the
moment a GPU is.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _verdict  # noqa: E402

REPO = HERE.parent.parent
RESULTS = REPO / "bench" / "results"

#: The key an artifact carries once it has been reconciled.
STAMP = "model_output_equivalence_authority"


def _records_with_both() -> list[tuple[pathlib.Path, int, dict]]:
    """Every bench result element carrying an outer token AND a nested counters copy."""
    found = []
    if not RESULTS.is_dir():
        return found
    for path in sorted(RESULTS.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a malformed sibling artifact is not our subject
            continue
        if not isinstance(doc, dict):
            continue
        for i, rec in enumerate(doc.get("results", []) or []):
            if not isinstance(rec, dict):
                continue
            counters = rec.get("counters")
            if _verdict.EQUIVALENCE_KEY in rec and isinstance(counters, dict) \
                    and _verdict.EQUIVALENCE_KEY in counters:
                found.append((path, i, rec))
    return found


def test_the_authority_rule_reads_the_record_not_the_token():
    """The rule must key off the record's presence, never off the token's value.

    Keying off ``token == "UNMEASURED"`` would answer the question by reading the very
    field whose trustworthiness is in question — and would mislabel a genuine comparison
    that legitimately concluded UNMEASURED as "nobody wrote it". Both polarities are
    asserted, because either alone is satisfiable by the wrong rule.
    """
    measured_unmeasured = {
        _verdict.EQUIVALENCE_KEY: "UNMEASURED",
        _verdict.EQUIVALENCE_RECORD_KEY: {"executed_by": "cpu", "artifact": "x.onnx"},
    }
    authority, token, reason = _verdict.equivalence_authority(measured_unmeasured)
    assert authority == _verdict.EQUIVALENCE_AUTHORITY_MEASURED, (
        "a comparison that ran and concluded UNMEASURED was reported as never having run. "
        f"The rule is reading the token, not the record. reason={reason!r}"
    )
    assert token == "UNMEASURED"

    default_match = {_verdict.EQUIVALENCE_KEY: "MATCH"}
    authority, token, reason = _verdict.equivalence_authority(default_match)
    assert authority == _verdict.EQUIVALENCE_AUTHORITY_UNSET, (
        "a bare MATCH with no record beside it was accepted as a reading. Nothing in that "
        "frame ran a comparison; the token is whatever its emitter defaulted to."
    )
    assert _verdict.EQUIVALENCE_RECORD_KEY in reason


def test_a_default_never_reads_as_agreement():
    """R12 at the value level: two values, one of which nobody wrote, have not agreed.

    The tempting shortcut is `agreement = (outer == inner)`, which for the phi35 artifact
    yields DISAGREE and sends a reader hunting for a contradiction that does not exist —
    and for a hypothetical artifact where both happened to read MATCH would yield AGREE
    from a single measurement. Both are wrong for the same reason (R11).
    """
    default_side = _verdict.reconcile_equivalence(
        {_verdict.EQUIVALENCE_KEY: "MATCH", "counters": {_verdict.EQUIVALENCE_KEY: "MATCH"}}
    )
    assert default_side["agreement"] == _verdict.WITNESS_AGREEMENT_UNOBSERVABLE, (
        "two identical tokens were reported as AGREE although only one of them was "
        f"written by a comparison: {default_side}"
    )

    both_read = _verdict.reconcile_equivalence(
        {
            _verdict.EQUIVALENCE_KEY: "MATCH",
            "counters": {
                _verdict.EQUIVALENCE_KEY: "MATCH",
                _verdict.EQUIVALENCE_RECORD_KEY: {"executed_by": "VulkanExecutionProvider"},
            },
        }
    )
    assert both_read["agreement"] == _verdict.WITNESS_AGREEMENT_AGREE, (
        f"two genuine readings that match were not reported as AGREE: {both_read}"
    )

    disagreeing = _verdict.reconcile_equivalence(
        {
            _verdict.EQUIVALENCE_KEY: "MATCH",
            "counters": {
                _verdict.EQUIVALENCE_KEY: "DIVERGENT",
                _verdict.EQUIVALENCE_RECORD_KEY: {"executed_by": "VulkanExecutionProvider"},
            },
        }
    )
    assert disagreeing["agreement"] == _verdict.WITNESS_AGREEMENT_DISAGREE, (
        "two genuine readings that contradict each other were not reported as DISAGREE — "
        f"which is the one case that IS a finding: {disagreeing}"
    )


def _is_certified(path: pathlib.Path) -> bool:
    """Whether this artifact is one anyone quotes from.

    Scope matters here, and getting it wrong in either direction is a real cost. Gating on
    *every* historical bench artifact would turn ~20 frozen records — written before the
    rule existed, owned by another agent, and not regenerable without re-running the
    benchmark — into a permanent red, and a permanently red gate is one that gets ignored
    or loosened. Gating on none would leave the artifact of record ambiguous, which is the
    actual defect. So the gate is the certified set; the rest are reported.
    """
    return "certified" in path.name


def test_the_certified_artifacts_say_which_equivalence_is_of_record():
    """The lane membership: no *certified* artifact may carry both values unreconciled.

    A reader given two same-named fields and no rule picks one. Which one they pick is not
    a property of the evidence.
    """
    records = [r for r in _records_with_both() if _is_certified(r[0])]
    if not records:
        pytest.skip("no certified bench artifact currently carries both values")

    unstamped = []
    genuinely_disagreeing = []
    for path, i, rec in records:
        stamp = rec.get(STAMP)
        if not isinstance(stamp, dict):
            unstamped.append(f"{path.name}[results/{i}]")
            continue
        recomputed = _verdict.reconcile_equivalence(rec)
        assert stamp.get("agreement") == recomputed["agreement"], (
            f"{path.name}[results/{i}] carries a stale reconciliation stamp: it says "
            f"{stamp.get('agreement')!r}, recomputing from the record says "
            f"{recomputed['agreement']!r}. A stamp that no longer follows from the "
            "artifact it sits in is worse than no stamp."
        )
        if recomputed["agreement"] == _verdict.WITNESS_AGREEMENT_DISAGREE:
            genuinely_disagreeing.append(f"{path.name}[results/{i}]")

    assert not unstamped, (
        "these artifacts carry two `model_output_equivalence` values with nothing saying "
        "which is of record, so a reader chooses by accident:\n  "
        + "\n  ".join(unstamped)
        + f"\nStamp each with `{STAMP}` = _verdict.reconcile_equivalence(record)."
    )
    assert not genuinely_disagreeing, (
        "two GENUINE readings of model_output_equivalence contradict each other in these "
        "artifacts — unlike the default case, this one IS a finding:\n  "
        + "\n  ".join(genuinely_disagreeing)
    )


def test_uncertified_artifacts_are_reported_not_gated(capsys):
    """The rest of the population, as a PRECONDITION and never a gate (R9 A5).

    These are frozen records written before the rule existed. Their silence here is a
    statement about scope, not about them, and this test says so out loud rather than
    printing a bare PASS — the same reason `tests/union_check.py` prints
    ``PRECONDITION(...)`` for its non-gating tiers.

    The one thing it *does* assert is that no uncertified artifact holds two GENUINE
    readings that contradict each other. That case is a finding wherever it appears, and
    scoping it away would be scoping away the only half of this that is about the EP.
    """
    others = [r for r in _records_with_both() if not _is_certified(r[0])]
    disagreeing = [
        f"{p.name}[results/{i}]"
        for p, i, rec in others
        if _verdict.reconcile_equivalence(rec)["agreement"]
        == _verdict.WITNESS_AGREEMENT_DISAGREE
    ]
    unstamped = [f"{p.name}[results/{i}]" for p, i, rec in others if STAMP not in rec]
    with capsys.disabled():
        print(
            f"\nPRECONDITION(equivalence authority): {len(others)} uncertified record(s) "
            f"carry both values; {len(unstamped)} are unreconciled. Not a gate — see "
            "_is_certified()."
        )
    assert not disagreeing, (
        "two GENUINE readings contradict each other in an uncertified artifact. This is "
        "in scope wherever it appears:\n  " + "\n  ".join(disagreeing)
    )
