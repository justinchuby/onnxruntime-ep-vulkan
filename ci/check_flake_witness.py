#!/usr/bin/env python3
"""``check_flake_witness`` — a name that survives a truncated log.

THE DEFECT THIS EXISTS FOR
--------------------------

On 2026-08-03 the coordinator's merge gate went red once in seven runs and **the name
of the failing test was lost to a truncated log tail**. Six greens and one red with no
subject is worse than no signal at all: it is a signal that trains people to press the
button again. The re-run passed, the merge went in, and the only durable record of the
event is a sentence in a message.

Two separate things went wrong there and they need separate mechanisms:

1. **The name did not survive the transport.** GitHub Actions truncates step logs, and a
   pytest/libtest failure name appears in the middle of a long capture. This file's first
   job is to lift failing test IDs out of the captured logs and re-emit them as
   ``::error title=...::`` workflow annotations, which are stored as check-run
   annotations rather than as log bytes and therefore do not truncate. The tail block it
   prints last is belt-and-braces: log tails survive when heads do not.

2. **Nothing accumulated across runs.** A 1-in-40 intermittent is invisible to any single
   run by construction — every individual run of it is either an ordinary red or an
   ordinary green. It is only visible in the JOIN across runs, and nothing in this repo
   was doing that join. This file's second job is an append-only ledger keyed by
   (commit, lane, suite, run, test id) and one query over it: **did the same test id, at
   the same commit, in the same suite, both fail and not-fail?** If it did, the test is
   intermittent, and the *commit* is exonerated — which is the fact the person staring
   at one red and six greens actually needs.

WHY "NOT-FAILED" AND NOT "PASSED"
---------------------------------

libtest prints every outcome by name (``test foo ... ok``). pytest, without ``-v``, prints
only failures by name. So for pytest the ledger can record ``FAILED`` positively and can
only *infer* the complement. It records ``NOT_FAILED``, not ``PASSED``, and the inference
is stated rather than hidden: an ID is NOT_FAILED in a run when that run's log parsed
cleanly, reported a terminal summary, and did not name the ID among its failures.

That distinction matters because ``NOT_FAILED`` includes *skipped* and *deselected*. A
test that failed on Monday and was skipped on Tuesday is NOT intermittent — it is a test
that stopped running, which is ``check_suite_productivity``'s defect class, not this
one. So the intermittency rule requires the two observations to come from runs whose
executed counts are within a tolerance of each other, and where they are not, it says
``INCOMPARABLE`` and explains why instead of claiming a flake.

WHAT IT DOES NOT CLAIM
----------------------

* It does not say WHY a test is intermittent. ``vk::barrier::tests::
  backend_probe_writes_legacy_token`` is intermittent because ``backend_probe_*`` is a
  process-global env var and the tests race for it; that is Trinity's env-var auditor,
  a different mechanism, and this one would report the same word for a genuine data race
  in the EP.
* It does not detect an intermittent that has only ever been observed once. One
  observation is not a rate. The ledger is what makes the second observation cheap.
* It cannot see a test that is intermittent between commits rather than within one.
  Keying on the commit is deliberate: the whole point is to separate "your change broke
  it" from "it does that".
* A ledger under ``bench/results/`` is per-checkout. On a hosted runner with no cache the
  join has exactly one run in it and this check can only annotate, never conclude. That
  is a real limit and ``--require-history`` exists so a lane can demand otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

DEFAULT_LEDGER = Path("bench/results/link-flake-witness/ledger.jsonl")

# pytest short-test-summary lines: `FAILED tests/ops/test_x.py::test_y - AssertionError`
_PYTEST_FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s.*)?$")
# pytest terminal summary. TWO spellings, and the second one is the one this project
# actually produces. Non-quiet pytest decorates it — `=== 8 failed, 624 passed in 1.2s ===`
# — and that is what this regex was written against. Under `-q`, which BOTH device lanes
# and the lane-check job use, the same line is emitted UNDECORATED:
#     6 failed, 805 passed, 49 skipped, 3 xfailed, 58 warnings in 410.23s (0:06:50)
# So for every pytest log this repository has ever captured, `saw_summary` was False and
# this check reported ERROR(instrument=log_unparsed). It declined rather than lying, which
# is why it is a repairable defect and not a false green — but a witness that has never
# parsed a single real subject is a witness with no demonstrated positive state, and it
# went unnoticed because it only reaches this branch on a run that got as far as pytest.
# The undecorated form is anchored on the trailing ` in <n>s`, which pytest always emits
# and which no ordinary output line carries, and it must START with a count so that a test
# name or an assertion message mentioning "passed" cannot be mistaken for a summary.
_PYTEST_SUMMARY_DECORATED_RE = re.compile(
    r"^=+\s*(?P<body>.*?\b(?:passed|failed|error|no tests ran).*?)\s*=+\s*$"
)
_PYTEST_SUMMARY_BARE_RE = re.compile(
    r"^(?P<body>(?:\d+\s+\w+(?:,\s*)?)+|no tests ran)\s+in\s+\d+(?:\.\d+)?s(?:\s+\(.*\))?$"
)
_PYTEST_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|xpassed|xfailed|skipped|error|errors|deselected)")


def _pytest_summary(line: str) -> re.Match | None:
    """Either spelling. Decorated first: a decorated line also matches nothing else."""
    m = _PYTEST_SUMMARY_DECORATED_RE.match(line)
    if m:
        return m
    m = _PYTEST_SUMMARY_BARE_RE.match(line)
    if m and re.search(r"\b(?:passed|failed|error|errors|no tests ran)\b", m.group("body")):
        return m
    return None

# libtest: `test vk::barrier::tests::backend_probe_writes_legacy_token ... FAILED`
_LIBTEST_OUTCOME_RE = re.compile(r"^test\s+(?P<name>\S+)\s+\.\.\.\s+(?P<outcome>ok|FAILED|ignored)\b")
# `Running unittests src/lib.rs (target/debug/deps/...)` — the word `unittests` is not a
# target name, and using it as one collapses every lib test in the repo under one prefix.
_LIBTEST_RUNNING_RE = re.compile(r"^\s*Running\s+(?:unittests\s+)?(?P<target>\S+)")
_LIBTEST_RESULT_RE = re.compile(
    r"^test result:\s+(?:ok|FAILED)\.\s+(?P<passed>\d+) passed;\s+(?P<failed>\d+) failed;"
    r"\s+(?P<ignored>\d+) ignored"
)

FAILED = "FAILED"
NOT_FAILED = "NOT_FAILED"
#: One record per parsed log, carrying no test id.  It is what makes the pytest
#: complement reconstructible: without it a run in which nothing failed leaves no trace
#: at all in the ledger, and the run that exonerates a commit is exactly the run whose
#: existence the join needs.  See `synthesise_not_failed`.
RUN_SEEN = "RUN_SEEN"


@dataclass
class RunObservation:
    """One captured log, parsed into named outcomes."""

    suite: str
    lane: str
    commit: str
    run_id: str
    harness: str
    log: str
    failed: list[str] = field(default_factory=list)
    not_failed: list[str] = field(default_factory=list)
    executed: int = 0
    parsed: bool = False
    note: str = ""

    def records(self) -> list[dict]:
        out = []
        for tid in self.failed:
            out.append(self._rec(tid, FAILED))
        for tid in self.not_failed:
            out.append(self._rec(tid, NOT_FAILED))
        # The run itself, named once, with no test id.  Written even when nothing failed
        # and nothing was named — a green run is a datum, and until 2026-08-04 it was the
        # one datum this ledger threw away.
        if self.parsed:
            out.append(self._rec("", RUN_SEEN))
        return out

    def _rec(self, tid: str, outcome: str) -> dict:
        return {
            "commit": self.commit,
            "lane": self.lane,
            "suite": self.suite,
            "run_id": self.run_id,
            "harness": self.harness,
            "test_id": tid,
            "outcome": outcome,
            "executed": self.executed,
            "log": self.log,
        }


def parse_pytest(text: str) -> tuple[list[str], list[str], int, bool, str]:
    """Return (failed ids, not-failed ids, executed, parsed, note).

    pytest names only its failures without ``-v``, so ``not_failed`` is empty here and
    the complement is reconstructed at query time by ``synthesise_not_failed`` from the
    union of ids the ledger has seen for this ``(commit, lane, suite)`` and the
    ``RUN_SEEN`` marker each parsed run leaves.  Between 2026-08-03 and 2026-08-04 this
    sentence described a function that did not exist and the pytest join could never
    fire; it is now the name of a callable.  Saying so is the point: an empty
    ``not_failed`` is not a claim that nothing passed.
    """
    failed: list[str] = []
    executed = 0
    saw_summary = False
    for raw in text.splitlines():
        line = raw.rstrip()
        m = _PYTEST_FAILED_RE.match(line.strip())
        if m and ("::" in m.group(1) or m.group(1).endswith(".py")):
            tid = m.group(1)
            if tid not in failed:
                failed.append(tid)
            continue
        s = _pytest_summary(line.strip())
        if s:
            saw_summary = True
            for count, kind in _PYTEST_COUNT_RE.findall(s.group("body")):
                if kind in ("passed", "failed", "xpassed", "xfailed"):
                    executed += int(count)
    note = "" if saw_summary else "no pytest terminal summary line found"
    return failed, [], executed, saw_summary, note


def parse_libtest(text: str) -> tuple[list[str], list[str], int, bool, str]:
    """libtest names EVERY outcome, so both polarities are observed rather than inferred."""
    failed: list[str] = []
    not_failed: list[str] = []
    executed = 0
    target = "?"
    saw_result = False
    for raw in text.splitlines():
        line = raw.rstrip()
        r = _LIBTEST_RUNNING_RE.match(line)
        if r:
            target = r.group("target")
            continue
        m = _LIBTEST_OUTCOME_RE.match(line.strip())
        if m:
            tid = f"{target}::{m.group('name')}"
            if m.group("outcome") == "FAILED":
                if tid not in failed:
                    failed.append(tid)
            elif m.group("outcome") == "ok":
                if tid not in not_failed:
                    not_failed.append(tid)
            continue
        res = _LIBTEST_RESULT_RE.match(line.strip())
        if res:
            saw_result = True
            executed += int(res.group("passed")) + int(res.group("failed"))
    note = "" if saw_result else "no libtest `test result:` line found"
    return failed, not_failed, executed, saw_result, note


def observe(path: Path, harness: str, suite: str, lane: str, commit: str, run_id: str) -> RunObservation:
    text = path.read_text(encoding="utf-8", errors="replace")
    if harness == "pytest":
        failed, not_failed, executed, parsed, note = parse_pytest(text)
    else:
        failed, not_failed, executed, parsed, note = parse_libtest(text)
    return RunObservation(
        suite=suite,
        lane=lane,
        commit=commit,
        run_id=run_id,
        harness=harness,
        log=str(path),
        failed=failed,
        not_failed=not_failed,
        executed=executed,
        parsed=parsed,
        note=note,
    )


def append_ledger(ledger: Path, records: list[dict]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def read_ledger(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    out = []
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # An append-only ledger can be torn by a killed process mid-write. A torn
            # last line is not a reason to lose the whole history, but it IS a reason
            # not to pretend the file is intact, so it is counted and reported.
            out.append({"__torn__": line[:120]})
    return out


@dataclass
class Verdict:
    intermittent: list[dict] = field(default_factory=list)
    incomparable: list[dict] = field(default_factory=list)
    torn: int = 0
    runs: int = 0


def synthesise_not_failed(clean: list[dict]) -> list[dict]:
    """Reconstruct the pytest complement, which the parser cannot observe.

    THE DEFECT THIS CLOSES
    ----------------------
    ``parse_pytest``'s own docstring said "the complement is reconstructed at query time
    from the union of ids the ledger has ever seen for this (suite, lane)".  Nothing in
    this file did that.  ``parse_pytest`` returned ``[]`` for ``not_failed`` and ``join``
    read the ledger as written, so for pytest every key had ``passes == []`` and the
    ``if not fails or not passes: continue`` line took the branch **every time, at any
    history depth**.  The parse defect Link fixed on 2026-08-04 stopped the check reading
    its logs; this one stopped it concluding anything from them, and it was invisible for
    the same reason — the file's second job has no positive state to notice the absence
    of.  Verified on a real captured lane ledger (run 30974825118, 609 records): 8 pytest
    FAILED records, 0 NOT_FAILED records, join necessarily empty.

    THE RECONSTRUCTION, STATED
    --------------------------
    An id is ``NOT_FAILED`` in a run when **that run's log parsed**, the run belongs to
    the same ``(commit, lane, suite)``, and the run did not name the id among its
    failures.  ``RUN_SEEN`` is what makes "that run's log parsed" a fact in the ledger
    rather than an assumption; a run with no ``RUN_SEEN`` record contributes nothing,
    because a run that did not parse cannot be said to have not-failed anything.

    This is an inference, not an observation, and the synthesised records say so with
    ``"inferred": True`` so a reader can tell them from libtest's directly observed ones.
    It stays inside the existing caveat: ``NOT_FAILED`` still includes skipped and
    deselected, which is what ``join``'s executed-count tolerance is for.
    """
    runs: dict[tuple[str, str, str, str], dict] = {}
    failed_in_run: dict[tuple[str, str, str, str], set[str]] = {}
    ids_by_scope: dict[tuple[str, str, str], set[str]] = {}
    observed: set[tuple[str, str, str, str, str]] = set()

    for r in clean:
        scope = (r["commit"], r["lane"], r["suite"])
        run = (*scope, r["run_id"])
        outcome = r.get("outcome")
        if outcome == RUN_SEEN:
            runs[run] = r
            continue
        tid = r.get("test_id") or ""
        if not tid:
            continue
        ids_by_scope.setdefault(scope, set()).add(tid)
        observed.add((*run, tid))
        if outcome == FAILED:
            failed_in_run.setdefault(run, set()).add(tid)

    out: list[dict] = []
    for run, marker in runs.items():
        scope = run[:3]
        named = failed_in_run.get(run, set())
        for tid in ids_by_scope.get(scope, set()):
            if tid in named or (*run, tid) in observed:
                continue
            out.append(
                {
                    "commit": run[0],
                    "lane": run[1],
                    "suite": run[2],
                    "run_id": run[3],
                    "harness": marker.get("harness", "pytest"),
                    "test_id": tid,
                    "outcome": NOT_FAILED,
                    "executed": marker.get("executed", 0),
                    "log": marker.get("log", ""),
                    "inferred": True,
                }
            )
    return out


def join(records: list[dict], tolerance: float) -> Verdict:
    """The whole mechanism: did one id, at one commit, in one suite, both fail and not-fail?"""
    v = Verdict()
    clean = []
    for r in records:
        if "__torn__" in r:
            v.torn += 1
        else:
            clean.append(r)
    v.runs = len({(r["commit"], r["lane"], r["suite"], r["run_id"]) for r in clean})
    clean = clean + synthesise_not_failed(clean)

    by_key: dict[tuple[str, str, str, str], dict[str, list[dict]]] = {}
    for r in clean:
        if r.get("outcome") == RUN_SEEN:
            continue
        key = (r["commit"], r["lane"], r["suite"], r["test_id"])
        by_key.setdefault(key, {}).setdefault(r["outcome"], []).append(r)

    for (commit, lane, suite, test_id), outcomes in sorted(by_key.items()):
        fails = outcomes.get(FAILED, [])
        passes = outcomes.get(NOT_FAILED, [])
        if not fails or not passes:
            continue
        # NOT_FAILED includes skipped/deselected. Two runs that executed wildly
        # different amounts of work are not two samples of the same experiment.
        fe = max(r["executed"] for r in fails)
        pe = max(r["executed"] for r in passes)
        hi = max(fe, pe)
        entry = {
            "commit": commit,
            "lane": lane,
            "suite": suite,
            "test_id": test_id,
            "failed_runs": sorted({r["run_id"] for r in fails}),
            "not_failed_runs": sorted({r["run_id"] for r in passes}),
            "executed_when_failed": fe,
            "executed_when_not_failed": pe,
            "not_failed_inferred": all(r.get("inferred") for r in passes),
        }
        if hi and abs(fe - pe) / hi > tolerance:
            entry["why_incomparable"] = (
                f"the two runs executed {fe} and {pe} tests. A NOT_FAILED that comes from a "
                f"run which executed far less work is more likely a test that stopped running "
                f"than a test that passed; that is check_suite_productivity's defect class."
            )
            v.incomparable.append(entry)
        else:
            v.intermittent.append(entry)
    return v


def _annotate(title: str, message: str) -> None:
    """A check-run annotation is stored outside the log body, so it cannot truncate."""
    if os.environ.get("GITHUB_ACTIONS") != "true" and not os.environ.get("FLAKE_WITNESS_FORCE_ANNOTATE"):
        return
    safe = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title={title}::{safe}")


def _fail(condition: str, *lines: str) -> int:
    print(f"FLAKE-WITNESS: FAIL(condition={condition})")
    for line in lines:
        print(f"  {line}")
    return EXIT_FAIL_CONDITION


def _error(instrument: str, *lines: str) -> int:
    print(f"FLAKE-WITNESS: ERROR(instrument={instrument})")
    for line in lines:
        print(f"  {line}")
    return EXIT_ERROR_INSTRUMENT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="*", help="captured test log(s) to read")
    ap.add_argument("--harness", choices=("pytest", "libtest"), default="pytest")
    ap.add_argument("--suite", default="unnamed-suite")
    ap.add_argument("--lane", default="*")
    ap.add_argument("--commit", default=os.environ.get("GITHUB_SHA", "LOCAL"))
    ap.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID") or os.environ.get("GITHUB_RUN_ATTEMPT") or "local")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--no-append", action="store_true", help="query the ledger without adding to it")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="how far two runs' executed counts may differ and still count as two samples "
        "of the same experiment (default 0.10)",
    )
    ap.add_argument(
        "--require-history",
        type=int,
        default=0,
        metavar="N",
        help="ERROR(instrument) unless the ledger holds at least N distinct runs. A join "
        "over one run cannot find an intermittent, and a green from it means nothing.",
    )
    ap.add_argument("--lane-marker", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.tolerance < 0:
        print("FLAKE-WITNESS: usage error — --tolerance may not be negative", file=sys.stderr)
        return EXIT_USAGE

    observations: list[RunObservation] = []
    for i, raw in enumerate(args.log):
        p = Path(raw)
        if not p.exists():
            if args.lane_marker is not None and not args.lane_marker.exists():
                print(
                    f"FLAKE-WITNESS: DECLINED — {p} is absent and the lane marker "
                    f"{args.lane_marker} was never written, so the lane died before it "
                    "could produce evidence. Declining to add a second red to a first "
                    "one this check did not cause."
                )
                return EXIT_PASS
            return _error("log_absent", f"{p} does not exist but the lane marker says the lane reached this step")
        run_id = args.run_id if len(args.log) == 1 else f"{args.run_id}#{i}"
        observations.append(observe(p, args.harness, args.suite, args.lane, args.commit, run_id))

    if not observations and args.require_history == 0:
        print("FLAKE-WITNESS: usage error — give at least one log, or --require-history N to query", file=sys.stderr)
        return EXIT_USAGE

    for obs in observations:
        if not obs.parsed:
            return _error(
                "log_unparsed",
                f"{obs.log}: {obs.note}.",
                "A log this check could not parse is UNOBSERVABLE, not clean. Reporting "
                "PASS here would be the exact defect class this repo keeps finding: an "
                "observable that is true whatever happened.",
            )

    if not args.no_append:
        recs: list[dict] = []
        for obs in observations:
            recs.extend(obs.records())
        try:
            append_ledger(args.ledger, recs)
        except OSError as exc:
            return _error("ledger_unwritable", f"{args.ledger}: {exc}")

    # --- the name must survive the transport, whatever the verdict turns out to be ---
    named_now: list[tuple[str, str]] = []
    for obs in observations:
        for tid in obs.failed:
            named_now.append((obs.lane, tid))
            _annotate(
                f"FAILED on {obs.lane}",
                f"{tid}\nsuite={obs.suite} commit={obs.commit} run={obs.run_id}\n"
                f"Emitted as an annotation because a log tail can truncate and a name that "
                f"does not survive the transport is a red with no subject.",
            )

    ledger_records = read_ledger(args.ledger)
    verdict = join(ledger_records, args.tolerance)

    if args.require_history and verdict.runs < args.require_history:
        return _error(
            "history_too_short",
            f"the ledger at {args.ledger} holds {verdict.runs} distinct run(s); "
            f"--require-history demands {args.require_history}.",
            "An intermittent is only visible in the join across runs. A join over fewer "
            "runs than that cannot find one, so a PASS from it would be a green that "
            "carries no information.",
        )

    def _tail_block() -> None:
        print()
        print("---- FLAKE-WITNESS NAMES (printed last so a truncated head cannot lose them) ----")
        if named_now:
            for lane, tid in named_now:
                print(f"  FAILED  [{lane}]  {tid}")
        else:
            print("  (no failing test named in the log(s) read on this run)")
        for e in verdict.intermittent:
            print(f"  FLAKY   [{e['lane']}]  {e['test_id']}  @{e['commit'][:12]}")
        print(f"---- ledger {args.ledger} : {verdict.runs} run(s), {len(ledger_records)} record(s) ----")

    if verdict.torn:
        print(
            f"FLAKE-WITNESS: NOTE — {verdict.torn} torn ledger line(s) skipped. An "
            "append-only file can be cut mid-write by a killed process; the history is "
            "usable, and saying so is not the same as pretending the file is intact."
        )

    for e in verdict.incomparable:
        print(f"FLAKE-WITNESS: INCOMPARABLE — {e['test_id']} @{e['commit'][:12]} on {e['lane']}")
        print(f"  {e['why_incomparable']}")

    if verdict.intermittent:
        lines = []
        for e in verdict.intermittent:
            msg = (
                f"{e['test_id']} both FAILED and did not fail at commit {e['commit'][:12]} "
                f"on lane {e['lane']} (suite {e['suite']}): failed in run(s) "
                f"{', '.join(e['failed_runs'])}, did not fail in run(s) "
                f"{', '.join(e['not_failed_runs'])}."
            )
            lines.append(msg)
            lines.append(
                "  THE COMMIT IS EXONERATED AND THE TEST IS NOT. A re-run that goes green "
                "does not make this go away; it is the second observation that produced it."
            )
            _annotate("Intermittent test (commit exonerated)", msg)
        code = _fail("intermittent", *lines)
        _tail_block()
        return code

    if named_now:
        print(
            f"FLAKE-WITNESS: PASS — {len(named_now)} failing test(s) named and annotated; "
            "no id has yet both failed and not-failed at this commit."
        )
    else:
        print(
            f"FLAKE-WITNESS: PASS — nothing failed in the log(s) read; "
            f"{verdict.runs} run(s) in the ledger and no intermittency found among them."
        )
    print(
        "What this claims: that every failing test id in these logs has been re-emitted "
        "where truncation cannot reach it, and that the accumulated ledger shows no id "
        "both failing and not failing at one commit. What it does not claim: that a green "
        "run proves absence of a flake — a 1-in-40 needs roughly that many runs before "
        "the join can see it, which is why the ledger is append-only and not per-run."
    )
    _tail_block()
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
