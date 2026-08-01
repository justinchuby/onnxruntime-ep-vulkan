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

    hits: list[tuple[str, str]] = []
    scanned = 0
    for name in argv:
        path = Path(name)
        if not path.exists():
            print("FATAL-LOG-CHECK: ERROR(instrument=log_not_captured)", flush=True)
            print(
                f"{path} does not exist. The lane step that was supposed to tee its "
                "output here did not, so this check has no input. UNOBSERVABLE is not "
                "zero hits: it is no observation, and it must not read as a clean lane.",
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
