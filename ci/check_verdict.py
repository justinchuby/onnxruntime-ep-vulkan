#!/usr/bin/env python3
"""``check_verdict`` — read the criterion-10 verdict record and decide the lane.

This is the reader half of ``ci/gate_chain_fp32.py``, kept as a **separate process** on
purpose.  A step that both produces a verdict and decides on it can be deleted in one
edit and the lane goes quiet rather than red; two steps mean the decision survives the
producer, and the producer's artifact has to exist on disk for the decision to be made at
all.  R10's falsifier for "X is wired" is *an artifact X produced whose content varies
with its input* — so this reads the artifact and refuses when it is absent.

It is also the epctl-independent half.  ``epctl --check-counters`` is the canonical gate
and it reads the counters snapshot; this reads the verdict record.  Two readers of the
same verdict, with different failure modes and different parsers, is R13 obligation 3 at
the lane level.

**Vocabulary is Trinity's** (``tests/ops/_verdict.py``): ``MATCH`` / ``DIVERGENT`` /
``UNMEASURED`` / ``UNATTRIBUTED`` / ``SPLIT-FRAME``.  This file defines none of its own.
Note in particular that ``UNATTRIBUTED`` is reported as ``UNATTRIBUTED`` and never folded
into ``DIVERGENT``: they have different owners, different fixes and different next
questions (DESIGN.md §10.0 third amendment, clause 4).

Terminal states, per R13 — an instrument error never counts as a detection:

    0  VERDICT-CHECK: PASS
    1  VERDICT-CHECK: FAIL(condition=<verdict token>)
    4  VERDICT-CHECK: ERROR(instrument=...)

USAGE
    python ci/check_verdict.py <verdict-record.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4


def _instrument_error(instrument: str, detail: str) -> int:
    print(f"VERDICT-CHECK: ERROR(instrument={instrument})", flush=True)
    print(detail, flush=True)
    print(
        "VERDICT-CHECK: the check did not reach its observation. Per DESIGN.md §10.0.1 "
        "R13 this is NOT a detection and NOT a pass — a lane with an instrument error "
        "is not a lane that ran.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, flush=True)
        return EXIT_USAGE
    path = Path(argv[0])

    sys.path.insert(0, str(REPO_ROOT / "tests" / "ops"))
    try:
        import _verdict  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return _instrument_error(
            "verdict_vocabulary_unavailable",
            f"Could not import tests/ops/_verdict.py: {exc!r}",
        )

    if not path.exists():
        # Absence of a check is a refusal, not a default green (Niobe's
        # bench/admissible.py carries the same principle).  A missing verdict record
        # means the gate step did not complete, which is UNMEASURED, which is a
        # failing lane — never a skipped one.
        print(
            f"VERDICT-CHECK: FAIL(condition={_verdict.VERDICT_UNMEASURED})", flush=True
        )
        print(
            f"No verdict record at {path}.\n"
            "The gate step writes this file to UNMEASURED *before* opening a session and "
            "overwrites it only after a completed comparison, so an absent file means "
            "the step did not run at all. Absence of a measurement is not a passing "
            "measurement.",
            flush=True,
        )
        return EXIT_FAIL_CONDITION

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return _instrument_error(
            "verdict_record_unparseable", f"{path}: {exc!r}"
        )
    if not isinstance(record, dict):
        return _instrument_error(
            "verdict_record_not_an_object", f"{path} does not contain a JSON object."
        )

    token = record.get("verdict")
    executed_by = record.get("executed_by")
    source = record.get("attribution_source")

    print(f"VERDICT-CHECK: record at {path}:", flush=True)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)

    if token not in _verdict.VERDICTS:
        return _instrument_error(
            "verdict_token_unknown",
            f"verdict={token!r} is not one of {_verdict.VERDICTS}. This reader and the "
            "writer disagree about the vocabulary, which is a harness fault and says "
            "nothing about the EP.",
        )

    # A MATCH that carries no frame is the 2026-07-30 specimen: arithmetically correct
    # and about a different world.  Clause 3 makes it unconstructible upstream; this
    # rejects it downstream too, because a gate that trusts its input is a gate that
    # trusts whatever replaced its input.
    if token == _verdict.VERDICT_MATCH:
        if not isinstance(executed_by, dict) or not executed_by:
            print(
                f"VERDICT-CHECK: FAIL(condition={_verdict.VERDICT_UNATTRIBUTED})",
                flush=True,
            )
            print(
                f"verdict={token} arrived with executed_by={executed_by!r}. A verdict "
                "without a frame is not a verdict about this EP (DESIGN.md §10.0 third "
                "amendment). Treating it as UNATTRIBUTED.",
                flush=True,
            )
            return EXIT_FAIL_CONDITION
        own = int(executed_by.get(_verdict.EP_NAME, 0) or 0)
        if own <= 0:
            print(
                f"VERDICT-CHECK: FAIL(condition={_verdict.VERDICT_UNATTRIBUTED})",
                flush=True,
            )
            print(
                f"verdict={token} with executed_by={executed_by} — zero fused-island "
                f"executions for {_verdict.EP_NAME}. MATCH is unrepresentable at a zero "
                "own-provider count; the comparison is arithmetically correct and it is "
                "about CPU-vs-CPU.",
                flush=True,
            )
            return EXIT_FAIL_CONDITION
        if source != _verdict.ATTRIBUTION_SOURCE_PROFILE:
            print(
                f"VERDICT-CHECK: FAIL(condition={_verdict.VERDICT_UNATTRIBUTED})",
                flush=True,
            )
            print(
                f"attribution_source={source!r}, expected "
                f"{_verdict.ATTRIBUTION_SOURCE_PROFILE!r}. The attribution must come "
                "from an instrument we do not own; our own counters may corroborate it "
                "but may not be its primary witness (clause 1).",
                flush=True,
            )
            return EXIT_FAIL_CONDITION

        print("VERDICT-CHECK: PASS", flush=True)
        print(
            f"verdict={token}, executed_by={executed_by}, attribution_source={source}, "
            f"artifact={record.get('artifact')!r}, device_index="
            f"{record.get('device_index')!r}.",
            flush=True,
        )
        print(
            "This lane is `green` for this artifact only. The verdict travels with the "
            "artifact at producer-at-version and never generalises to another one.",
            flush=True,
        )
        return EXIT_PASS

    # Every non-MATCH token is a distinct condition and is printed as itself.  Folding
    # UNATTRIBUTED into DIVERGENT would lose the whole finding.
    print(f"VERDICT-CHECK: FAIL(condition={token})", flush=True)
    print(_EXPLANATIONS.get(token, "No explanation registered for this token."), flush=True)
    print(f"executed_by={executed_by!r} attribution_source={source!r}", flush=True)
    print(f"detail={record.get('detail')!r}", flush=True)
    return EXIT_FAIL_CONDITION


_EXPLANATIONS = {
    "DIVERGENT": (
        "Our kernels ran and computed the wrong answer. Owner: Switch/Mouse. "
        "Next question: which output, and what is its max-abs-diff."
    ),
    "UNMEASURED": (
        "No CPU-only comparison was completed on this artifact in this run. This is the "
        "default the gate writes before opening a session, so seeing it here means the "
        "comparison never finished. It is not a soft PASS and it is not a FAIL of the "
        "EP: nothing was measured. Owner: Link (the lane)."
    ),
    "UNATTRIBUTED": (
        "The comparison was performed and agreed, and this EP executed ZERO nodes in the "
        "run that produced the outputs — CPU-vs-CPU. This is NOT DIVERGENT: the model "
        "was not wrong, the subject was. Owner: Switch/Mouse (why did the EP decline or "
        "fall back), not the numerics. Next question: what does ORT's log say about "
        "`Falling back`, and what is in executed_by."
    ),
    "SPLIT-FRAME": (
        "The two witnesses disagree: ORT's profile and our own counters describe "
        "different worlds. Report no triple and no ratio. Owner: Switch (counters side). "
        "Two witnesses that can only ever agree are one witness; this is the value of "
        "having two."
    ),
}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
