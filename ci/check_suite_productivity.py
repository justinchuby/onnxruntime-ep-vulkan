#!/usr/bin/env python3
"""``check_suite_productivity`` — a step that asserted nothing must not report success.

THE DEFECT CLASS THIS EXISTS FOR
--------------------------------

On 2026-08-02 the op-correctness lane was found to be one missing *optional* Python
dependency away from asserting nothing at all. ``tests/ops/test_shape_inference_delta.py``
built its coverage report at **module scope**, so the import of ``onnx_shape_inference``
ran at COLLECTION time; without the package pytest printed
``Interrupted: 1 error during collection`` and abandoned the whole ``tests/ops``
directory. 665 tests, none of them run, and nothing in the lane whose job it was to
notice.

That is the CI form of an observable that is true whatever happens — the same shape as
``check_fatal_log``'s twelve historical "hits" that were its own docstring, the INFO
channel that had never carried anything, and ``elementwise::EXERCISED`` consulted before a
proof key exists. **A lane that reports success for doing nothing is not a weaker lane
than one that reports success for doing the work. It is a different kind of object: it
has no positive state, so its green carries no information.**

Fixing the dependency is the small half. The large half is making a lane that asserted
nothing *unable* to report success, and that is a mechanism, not a convention. This is
the mechanism.

WHAT IT ASSERTS, IN THE ORDER IT ASSERTS IT
-------------------------------------------

Given a captured pytest log (the lanes already ``tee`` one, so this reads the evidence
the lane published rather than re-running anything):

1. **The log exists.** If it does not and the lane marker was never written, the lane
   died before it could produce evidence — not a pass, not a detection, and this check
   declines to add a second red on top of a first one it did not cause (the
   ``--lane-marker`` convention from ``check_fatal_log.py``). If the marker IS present
   and the log is not, that is an instrument failure.
2. **No collection error.** ``Interrupted:``/``errors during collection``/``ERROR
   collecting`` are a hard fail, whatever the counts say. A directory that aborted
   during collection asserted nothing about the tests it never imported.
3. **The suite ran at all.** ``no tests ran`` and a terminal summary with zero
   executed outcomes are hard fails. All-skipped is the quiet form of the same thing.
4. **A floor on COLLECTED tests.** The collected count is environment-independent — it
   does not depend on a GPU, an ICD, a driver, or an ORT build — so a drop below the
   recorded floor means tests stopped being *collected*, which is exactly what a
   collection-time import error, a deleted file, or a selector that matches nothing
   produces. This is the ratchet.
5. **A floor on EXECUTED tests, per lane.** Executed = passed + failed + xpassed +
   xfailed. Skipped is not executed; deselected is not executed; an error is not an
   assertion. Executed floors are calibrated per lane because skip counts legitimately
   differ between a GPU lane and a lavapipe lane.

Floors live in ``ci/suite_floor.json`` — committed, with provenance per entry. A floor
that can be lowered by a command-line flag is a waiver with a flag, so there is no
``--relax``; lowering a floor is an edit to a tracked file with a reason next to it.

WHAT IT DOES NOT CLAIM
----------------------

That the assertions were *good* ones. This check counts outcomes; it cannot see a test
that runs and asserts something vacuous — that is ``check_tautological_assertions.py``'s
job, and this exists to cover the case that screen cannot reach: a test that never ran.

Terminal states (R13):

    0  SUITE-PRODUCTIVITY: PASS
    1  SUITE-PRODUCTIVITY: FAIL(condition=collection_error)
    1  SUITE-PRODUCTIVITY: FAIL(condition=no_tests_ran)
    1  SUITE-PRODUCTIVITY: FAIL(condition=asserted_nothing)
    1  SUITE-PRODUCTIVITY: FAIL(condition=collected_below_floor)
    1  SUITE-PRODUCTIVITY: FAIL(condition=executed_below_floor)
    4  SUITE-PRODUCTIVITY: ERROR(instrument=...)

USAGE
    python ci/check_suite_productivity.py --suite tests/ops --lane build-test-linux \
        [--lane-marker=PATH] <captured-pytest-log>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLOOR_PATH = REPO_ROOT / "ci" / "suite_floor.json"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

#: Outcomes that mean a test body ran and its assertions were evaluated.
#:
#: `xfailed` counts: an xfail test executes and its failure is checked. `skipped` and
#: `deselected` do not: no assertion was evaluated. `error` does not either — a setup or
#: collection error is a red, and reds are welcome, but they are not evidence that the
#: subject was exercised.
EXECUTED_OUTCOMES = ("passed", "failed", "xpassed", "xfailed")

#: Every outcome word pytest can print in its terminal summary line. Parsed rather than
#: assumed so an unrecognised word is visible instead of silently dropped into a zero.
KNOWN_OUTCOMES = EXECUTED_OUTCOMES + (
    "skipped",
    "deselected",
    "error",
    "errors",
    "warning",
    "warnings",
)

_SUMMARY_ITEM_RE = re.compile(r"(\d+)\s+([a-z]+)")
_COLLECTED_RE = re.compile(r"(\d+)\s+tests?\s+collected", re.IGNORECASE)
_COLLECTED_DESELECTED_RE = re.compile(
    r"collected\s+(\d+)\s+items?(?:\s*/\s*(\d+)\s+deselected)?", re.IGNORECASE
)

#: Rust's libtest harness. `cargo test` has exactly the same hole as pytest and it is
#: quieter: a target with no tests prints `running 0 tests` / `test result: ok.` and exits
#: ZERO. Eleven steps across the two device lanes are `cargo test` invocations, so the
#: defect class does not stop at Python.
_LIBTEST_RUNNING_RE = re.compile(r"^running (\d+) tests?$", re.MULTILINE)
_LIBTEST_RESULT_RE = re.compile(
    r"test result:\s+(ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed;\s+(\d+) ignored;"
    r"\s+(\d+) measured;\s+(\d+) filtered out",
    re.MULTILINE,
)

#: The three ways pytest announces that it gave up before running anything.
_COLLECTION_ERROR_MARKERS = (
    "errors during collection",
    "error during collection",
    "ERROR collecting",
    "Interrupted:",
)


class Totals:
    """Outcome counts parsed out of a captured pytest log."""

    def __init__(self) -> None:
        self.by_outcome: dict[str, int] = {}
        self.collected: int | None = None
        self.summary_line: str | None = None
        self.no_tests_ran: bool = False
        self.unknown_words: list[str] = []

    @property
    def executed(self) -> int:
        return sum(self.by_outcome.get(k, 0) for k in EXECUTED_OUTCOMES)

    @property
    def reported_total(self) -> int:
        """Everything the summary accounted for, executed or not."""
        return sum(
            self.by_outcome.get(k, 0)
            for k in ("passed", "failed", "xpassed", "xfailed", "skipped", "error", "errors")
        )

    def describe(self) -> str:
        if not self.by_outcome:
            return "(no terminal summary line found)"
        return ", ".join(f"{v} {k}" for k, v in sorted(self.by_outcome.items()))


def parse_pytest_log(text: str) -> Totals:
    """Parse a ``pytest -q`` capture.

    Deliberately tolerant about *where* the summary is: the lanes merge ORT's native
    stderr into the same file (``2>&1``), so the terminal summary is not reliably the
    last line. The last line that looks like a summary wins.
    """
    totals = Totals()
    lines = text.splitlines()

    for line in lines:
        m = _COLLECTED_RE.search(line)
        if m:
            totals.collected = int(m.group(1))
        m = _COLLECTED_DESELECTED_RE.search(line)
        if m:
            totals.collected = int(m.group(1))

    # pytest's terminal summary always ends with a duration: `... in 12.34s`.
    # Anchor on that so a stray "3 passed" inside captured stdout is not mistaken for it.
    summary_re = re.compile(r"^=*\s*(.*?)\s+in\s+[\d.]+s\s*(?:\([^)]*\))?\s*=*$")
    for line in lines:
        stripped = line.strip()
        m = summary_re.match(stripped)
        if not m:
            continue
        body = m.group(1)
        if body.strip().lower() in ("no tests ran", "no tests ran"):
            totals.no_tests_ran = True
            totals.summary_line = stripped
            totals.by_outcome = {}
            continue
        items = _SUMMARY_ITEM_RE.findall(body)
        if not items:
            continue
        parsed: dict[str, int] = {}
        unknown: list[str] = []
        for count, word in items:
            key = "errors" if word == "error" else word
            if word not in KNOWN_OUTCOMES:
                unknown.append(word)
                continue
            parsed[key] = parsed.get(key, 0) + int(count)
        # A summary body that is entirely unrecognised words is not a summary.
        if not parsed:
            continue
        totals.summary_line = stripped
        totals.by_outcome = parsed
        totals.unknown_words = unknown
        totals.no_tests_ran = False

    return totals


def parse_libtest_log(text: str) -> Totals:
    """Parse a ``cargo test`` capture.

    Rust's harness is mapped onto the same vocabulary rather than given its own, because
    the question is identical and should be answerable in one place: ``passed`` and
    ``failed`` are executed, ``ignored`` is a skip, ``filtered out`` is a deselection.

    A ``cargo test`` invocation can produce SEVERAL ``test result:`` blocks (one per
    target, plus doc-tests). They are summed: the step's claim is about the invocation,
    not about one of its binaries.
    """
    totals = Totals()
    blocks = _LIBTEST_RESULT_RE.findall(text)
    if not blocks:
        return totals
    passed = failed = ignored = filtered = 0
    for _status, p, f, i, _m, fo in blocks:
        passed += int(p)
        failed += int(f)
        ignored += int(i)
        filtered += int(fo)
    totals.by_outcome = {"passed": passed, "failed": failed, "skipped": ignored}
    if filtered:
        totals.by_outcome["deselected"] = filtered
    running = [int(n) for n in _LIBTEST_RUNNING_RE.findall(text)]
    totals.collected = (sum(running) + filtered) if running else (passed + failed + ignored)
    totals.summary_line = (
        f"cargo test: {len(blocks)} target block(s); {passed} passed; {failed} failed; "
        f"{ignored} ignored; {filtered} filtered out"
    )
    return totals


def has_collection_error(text: str) -> list[str]:
    """Quote the lines, never just count them (R13: a count is what let a NameError pass)."""
    hits = []
    for line in text.splitlines():
        for marker in _COLLECTION_ERROR_MARKERS:
            if marker in line:
                hits.append(line.strip())
                break
    return hits


def load_floors(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "suites" not in data:
        raise ValueError("suite_floor.json has no 'suites' object")
    return data


def _provenance(entry: dict) -> str:
    """Render a floor entry's provenance.

    Stored as a list of lines in ``suite_floor.json`` because a single long string is
    unreadable in a diff, and that file is meant to be read in diffs: lowering a floor
    without rewriting the reason next to it should be conspicuous.
    """
    prov = entry.get("provenance")
    if prov is None:
        return "(none recorded)"
    if isinstance(prov, list):
        return "\n    " + "\n    ".join(str(x) for x in prov)
    return str(prov)


def _fail(condition: str, *lines: str) -> int:
    print(f"SUITE-PRODUCTIVITY: FAIL(condition={condition})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_FAIL_CONDITION


def _error(instrument: str, *lines: str) -> int:
    print(f"SUITE-PRODUCTIVITY: ERROR(instrument={instrument})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_ERROR_INSTRUMENT


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__)
    ap.add_argument("log", nargs="?", help="captured pytest output (stderr merged)")
    ap.add_argument("--suite", required=False, default="tests/ops")
    ap.add_argument("--lane", required=False, default=None)
    ap.add_argument(
        "--harness",
        choices=("pytest", "libtest"),
        default="pytest",
        help="pytest (default) or libtest for `cargo test` output",
    )
    ap.add_argument("--lane-marker", default="")
    ap.add_argument("--floors", default=str(FLOOR_PATH))
    ap.add_argument(
        "--emit-json",
        default="",
        help="write the parsed totals here, so the numbers this check acted on are "
        "readable without re-parsing the log",
    )
    if not argv:
        print(__doc__, flush=True)
        return EXIT_USAGE
    args = ap.parse_args(argv)
    if not args.log:
        print(__doc__, flush=True)
        return EXIT_USAGE

    log_path = Path(args.log)
    if not log_path.exists():
        if args.lane_marker and not Path(args.lane_marker).exists():
            print("SUITE-PRODUCTIVITY: ERROR(instrument=lane_did_not_reach_evidence)", flush=True)
            print(
                f"{log_path} does not exist and the lane marker {args.lane_marker} was "
                "never written, so the lane failed before it ran any tests at all.\n"
                "This is NOT a pass, and it is not a finding about productivity either: "
                "a suite that never started cannot be said to have asserted nothing in "
                "the sense this check means. The lane is already red for the reason that "
                "stopped it.",
                flush=True,
            )
            print(
                "::warning title=Suite-productivity check had no subject::"
                "The lane did not reach the test step. Fix the earlier failure and "
                "re-read this step.",
                flush=True,
            )
            return EXIT_PASS
        return _error(
            "log_not_captured",
            f"{log_path} does not exist. The step that tees the suite output did not, so "
            "this check has no input."
            + (
                f"\nThe lane marker {args.lane_marker} IS present, so the lane did reach "
                "the test step and then captured nothing — an instrument failure, not an "
                "empty run."
                if args.lane_marker
                else ""
            ),
        )

    text = log_path.read_text(encoding="utf-8", errors="replace")

    try:
        floors = load_floors(Path(args.floors))
    except Exception as exc:  # noqa: BLE001
        return _error(
            "floors_unreadable",
            f"Could not read {args.floors}: {exc!r}\n"
            "The floors are the only thing that makes this check a mechanism rather than "
            "a convention. Without them it would pass on any log with a summary line, "
            "which is the state it was written to end.",
        )

    entry = floors["suites"].get(args.suite)
    if entry is None:
        return _error(
            "suite_has_no_floor",
            f"{args.suite!r} has no entry in {args.floors}.\n"
            "An unclassified suite is exactly the state the op-correctness step lived in: "
            "running, green, and answerable to nothing. Add an entry with a measured "
            "value and a provenance note rather than passing here by default.",
        )

    totals = parse_pytest_log(text) if args.harness == "pytest" else parse_libtest_log(text)

    if args.emit_json:
        out = Path(args.emit_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "suite": args.suite,
                    "lane": args.lane,
                    "log": str(log_path),
                    "collected": totals.collected,
                    "by_outcome": totals.by_outcome,
                    "executed": totals.executed,
                    "reported_total": totals.reported_total,
                    "summary_line": totals.summary_line,
                    "no_tests_ran": totals.no_tests_ran,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    # ---- 2. collection errors -------------------------------------------------
    # pytest only: libtest has no collection phase — a Rust target that does not compile
    # is a cargo error with a non-zero exit, which is self-announcing.
    collection_hits = has_collection_error(text) if args.harness == "pytest" else []
    if collection_hits:
        return _fail(
            "collection_error",
            "pytest aborted or errored during COLLECTION. Whatever the counts below say, "
            "the tests in the affected files were never imported, so nothing was asserted "
            "about them. A collection error is not a subset of a test failure: a failing "
            "test is evidence, a file that never loaded is an absence of evidence.",
            "The matching lines, quoted in full:",
            *[f"  {log_path}: {h}" for h in collection_hits[:20]],
            f"\nParsed outcome summary: {totals.describe()}",
            "\nMost common cause on this project: a module-level import of an OPTIONAL "
            "dependency. Move the import inside the tests that need it so the blast "
            "radius is those tests, not the directory.",
        )

    # ---- instrument: no summary at all ---------------------------------------
    if totals.summary_line is None:
        return _error(
            "summary_not_found",
            f"No {'pytest terminal summary' if args.harness == 'pytest' else 'libtest `test result:` block'} "
            f"was found in {log_path} ({len(text.splitlines())} lines read).\n"
            "This check reads counts out of the lane's own captured output; a log with no "
            "summary means the run was killed, the tee lost the tail, or the output format "
            "changed. UNOBSERVABLE is not zero, and it must not read as a clean lane.",
        )

    if totals.unknown_words:
        return _error(
            "unrecognised_outcome_word",
            f"Terminal summary contained outcome word(s) this check does not know: "
            f"{totals.unknown_words}.\nLine: {totals.summary_line}\n"
            "Reporting a number derived from a partly-understood line would be a count "
            "without its text. Add the word to KNOWN_OUTCOMES with a decision about "
            "whether it counts as executed.",
        )

    # ---- 3. the suite ran at all ---------------------------------------------
    if totals.no_tests_ran:
        return _fail(
            "no_tests_ran",
            f"pytest reported `no tests ran` for {args.suite}.\n"
            f"Line: {totals.summary_line}\n"
            "A selector that matches nothing exits ZERO in several pytest configurations "
            "and reads in a lane log exactly like a suite that passed. It is not one.",
        )

    if totals.executed == 0:
        return _fail(
            "asserted_nothing",
            f"{args.suite} accounted for {totals.reported_total} test(s) and executed NONE "
            f"of them.\n"
            f"Outcomes: {totals.describe()}\n"
            f"Line: {totals.summary_line}\n"
            "Every outcome was a skip, a deselection, or an error. This step therefore "
            "asserted nothing about the subject it is named for, and a step that asserted "
            "nothing must not be able to report success — that is the whole reason this "
            "check exists.",
        )

    # ---- 4. collected floor (environment-independent) -------------------------
    min_collected = entry.get("min_collected")
    if min_collected is not None:
        observed_collected = totals.collected
        if observed_collected is None:
            # `-q` does not always print the collected line; the reported total is a
            # sound lower bound on it, and using a lower bound can only make this
            # check MORE willing to fail, never less.
            observed_collected = totals.reported_total
            source = "reported total (no explicit `collected` line in this log)"
        else:
            source = "pytest's own `collected` line"
        if observed_collected < min_collected:
            return _fail(
                "collected_below_floor",
                f"{args.suite}: {observed_collected} test(s) collected, floor is "
                f"{min_collected} (source: {source}).\n"
                f"Floor provenance: {_provenance(entry)}\n"
                "The collected count does not depend on a GPU, an ICD, a driver or an ORT "
                "build — it is the same number on every lane. A drop means tests stopped "
                "being COLLECTED: an import error, a deleted file, a renamed selector, or "
                "a conftest that bailed. None of those are things a green step should be "
                "able to hide.\n"
                "If the drop is intentional, lower the floor in ci/suite_floor.json and "
                "say why in its `provenance`. There is deliberately no flag for this.",
            )

    # ---- 5. executed floor (per lane) ----------------------------------------
    lane_floors = entry.get("min_executed_by_lane", {})
    lane_key = args.lane or "*"
    min_executed = lane_floors.get(lane_key, lane_floors.get("*"))
    if min_executed is not None and totals.executed < min_executed:
        return _fail(
            "executed_below_floor",
            f"{args.suite} on lane {lane_key!r}: {totals.executed} test(s) executed "
            f"(passed+failed+xpassed+xfailed), floor is {min_executed}.\n"
            f"Outcomes: {totals.describe()}\n"
            f"Floor provenance: {_provenance(entry)}\n"
            "Tests moved from executing to skipping. A rising skip count is the quiet "
            "form of a lane doing less work while still reporting success.",
        )

    print("SUITE-PRODUCTIVITY: PASS", flush=True)
    print(
        f"{args.suite} on lane {lane_key!r}: {totals.describe()}.\n"
        f"Executed (passed+failed+xpassed+xfailed): {totals.executed} "
        f"(floor {min_executed if min_executed is not None else 'unset'}).\n"
        f"Collected: {totals.collected if totals.collected is not None else totals.reported_total} "
        f"(floor {min_collected if min_collected is not None else 'unset'}).\n"
        f"Summary line read: {totals.summary_line}\n"
        "What this claims: the step ran a floor's worth of tests and evaluated their "
        "assertions. What it does not claim: that those assertions are non-vacuous "
        "(ci/check_tautological_assertions.py), that the EP executed anything "
        "(ci/check_verdict.py), or that the tests PASSED — a red suite is a red lane by "
        "its own exit code, and this check is about the case where there is no exit code "
        "to be red.",
        flush=True,
    )
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
