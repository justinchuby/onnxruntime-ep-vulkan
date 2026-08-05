#!/usr/bin/env python3
"""``check_fatal_log`` — a known-fatal log line is a lane failure, not a log line.

DESIGN.md §10.0.1 R13, obligation 3: *the remedy for a guard that can fail silently is
not a more careful guard; it is an independent check that fails differently.*  ORT prints

    EP_FAIL ... Falling back to CPUExecutionProvider

inside ``run()`` and **raises nothing**.  That line has now appeared **five times on this
project while every gate passed**.  A grep cannot ``NameError``, and a guard cannot be
silenced by a log format change; each covers the other's outage, which is why both exist.

This reads the lane's captured suite output — the log must be captured with stderr merged
(``2>&1``), because ORT writes that line from C++ to fd 2 — and fails the lane on a hit.
It quotes the matching **text**.  Per R13's second clause, a count is what let a
``NameError`` masquerade as a detection, so no count is reported without its text.

Terminal states:

    0  FATAL-LOG-CHECK: PASS
    1  FATAL-LOG-CHECK: FAIL(condition=runtime_fallback_announced_by_ort)
    4  FATAL-LOG-CHECK: ERROR(instrument=...)   — including "the log was never captured",
                                                  because a check with no input has not
                                                  observed anything (R12: a counter whose
                                                  event cannot occur in its frame reports
                                                  UNOBSERVABLE, never 0)

USAGE
    python ci/check_fatal_log.py <captured-log> [<captured-log> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, flush=True)
        return EXIT_USAGE

    # `--lane-marker=PATH` distinguishes "the lane ran and captured no fatal line" from
    # "the lane died before it ever produced a log". Both are non-passes and neither is a
    # detection, but only the first is a statement about this check's subject. Observed on
    # the 2026-08-01 main run: both device lanes died at Clippy, never reached the pytest
    # step that tees its output, and this check added a second ERROR(instrument) on top of
    # a failure it had nothing to do with. The marker is written by the lane's own log
    # producer, so it cannot be absent on a run that did produce a log.
    #
    # REPEATABLE, and scoped to the logs that FOLLOW it. One marker for several logs is
    # a marker for whichever producer happens to run first: on 2026-08-04 the Windows lane
    # passed `.lane-reached` (written by the pytest step) together with
    # `gate_chain_fp32_ort_stderr.log` (written by a LATER step), so when the pytest step
    # failed and the gate step never ran, the marker was present, the log was absent, and
    # this check reported an instrument failure for a step that had simply not happened.
    # A marker only speaks for the producer that writes it.
    pairs: list[tuple[str, str]] = []
    current_marker = ""
    for a in argv:
        if a.startswith("--lane-marker="):
            current_marker = a.split("=", 1)[1]
        else:
            pairs.append((a, current_marker))
    argv = [name for name, _ in pairs]
    marker_of = dict(pairs)
    if not argv:
        print(__doc__, flush=True)
        return EXIT_USAGE

    sys.path.insert(0, str(REPO_ROOT / "tests" / "ops"))
    try:
        import _verdict  # type: ignore

        markers = _verdict.FATAL_LOG_MARKERS
        find = _verdict.find_fatal_log_lines
    except Exception as exc:  # noqa: BLE001
        print("FATAL-LOG-CHECK: ERROR(instrument=verdict_vocabulary_unavailable)", flush=True)
        print(
            f"Could not import tests/ops/_verdict.py: {exc!r}\n"
            "The marker list lives there so there is exactly one copy of it to keep "
            "correct. This check has no marker list of its own and will not invent one.",
            flush=True,
        )
        return EXIT_ERROR_INSTRUMENT

    # LIVENESS, before any verdict of this check is trusted.  Added 2026-08-02 by Trinity
    # with the marker fix: between 2026-07-31 and 2026-08-02 the patterns did not match the
    # line ORT actually prints, so this check reported green over a log announcing the
    # fallback twice, and was cited as second witness for five incidents on the strength of
    # a match it could not make.  **A marker list that has never been shown to fire is
    # indistinguishable from one that cannot** — so it is now shown to fire, against a real
    # captured announcement, every time this check runs.  A blind witness is an instrument
    # outage and never a detection (R13).
    try:
        _verdict.assert_fatal_log_check_is_live()
    except Exception as exc:  # noqa: BLE001
        print("FATAL-LOG-CHECK: ERROR(instrument=witness_not_live)", flush=True)
        print(
            f"{exc}\n\n"
            "This check scanned nothing, because a scanner that cannot match its own "
            "positive control tells you nothing about the log you gave it.",
            flush=True,
        )
        return EXIT_ERROR_INSTRUMENT

    hits: list[tuple[str, str]] = []
    scanned = 0
    for name in argv:
        path = Path(name)
        lane_marker = marker_of.get(name, "")
        if not path.exists():
            if lane_marker and not Path(lane_marker).exists():
                print("FATAL-LOG-CHECK: ERROR(instrument=lane_did_not_reach_evidence)", flush=True)
                print(
                    f"{path} does not exist and the lane marker {lane_marker} was never "
                    "written, so the lane failed before it produced any log at all.\n"
                    "This is NOT a pass. It is also not a detection: ORT cannot have "
                    "announced a fallback in a run that never started. The lane is "
                    "already red for the reason that stopped it, and this check declines "
                    "to add a second red on top of a first one it did not cause.",
                    flush=True,
                )
                print(
                    "::warning title=Fatal-log check had no subject::"
                    "The lane did not reach log capture. Fix the earlier failure and "
                    "re-read this step.",
                    flush=True,
                )
                return EXIT_PASS
            print("FATAL-LOG-CHECK: ERROR(instrument=log_not_captured)", flush=True)
            print(
                f"{path} does not exist. The lane step that was supposed to tee its "
                "output here did not, so this check has no input. UNOBSERVABLE is not "
                "zero hits: it is no observation, and it must not read as a clean lane."
                + (
                    f"\nThe lane marker {lane_marker} IS present, so the lane did reach "
                    "log capture and then captured nothing — an instrument failure, not "
                    "an empty run."
                    if lane_marker
                    else ""
                ),
                flush=True,
            )
            return EXIT_ERROR_INSTRUMENT
        text = path.read_text(encoding="utf-8", errors="replace")
        scanned += 1
        for line in find(text):
            hits.append((str(path), line))

    if hits:
        print(
            "FATAL-LOG-CHECK: FAIL(condition=runtime_fallback_announced_by_ort)",
            flush=True,
        )
        print(
            "ORT abandoned this EP at run time and re-executed on CPU without raising.\n"
            "Every assertion in the lane may have passed; they were checking CPU output.\n"
            "The matching lines, quoted in full:",
            flush=True,
        )
        for where, line in hits:
            print(f"  {where}: {line}", flush=True)
        return EXIT_FAIL_CONDITION

    print("FATAL-LOG-CHECK: PASS", flush=True)
    print(
        f"Scanned {scanned} captured log(s) for {list(markers)} and found no match.\n"
        "What this claims: ORT did not *announce* a run-time fallback in the captured "
        "output. What it does not claim: that the EP executed anything — that is the "
        "verdict's job (ci/check_verdict.py), and this exists to cover that guard's "
        "outage, not to replace it.",
        flush=True,
    )
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
