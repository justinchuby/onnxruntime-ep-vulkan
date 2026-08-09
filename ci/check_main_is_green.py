#!/usr/bin/env python3
"""Read the colour of `main` on GitHub, and refuse to let a local gate run stand in for it.

WHY THIS EXISTS
===============
CI was RED on `main` for at least ten consecutive pushes and nobody noticed. Every merge
in that window was verified locally — 574 lib tests passed, clippy clean, `cargo fmt
--check` clean, ledger identical — and every report quoted those numbers. Not one of them
quoted the badge, because nothing in anybody's workflow reads it.

That is not a CI defect. Every one of those reds printed its failure in full, with the
condition named, on GitHub, for days. It is the exact defect `ci/check_open_reds.py` was
built for, one level up: **a check whose failure reaches no one is a check that has
stopped being one.** There the failure reached no one because no register recorded which
reds were owned; here it reached no one because the reader was a human who had a green
local transcript in front of them and no reason to open a browser.

The four registers in `ci/` all have the same shape: take a thing that was implicit —
"the known reds", "the lane really ran", "that digest move was deliberate", "that proof
was withdrawn on purpose" — and make it a file something reads. The implicit thing here
is **"main is green"**, asserted from memory by whoever is merging. This screen makes it
an observation with a URL in it.

WHAT IT DOES
============
It fills a window of N **applicable** runs and rules on that window's colour.

    gh run list --branch main --limit <raw> --json conclusion,status,headSha,...,event,headBranch

"Applicable" means a run that actually answers this question: workflow `CI` (the one
badge.svg'd in README.md, `--workflow` overridable), `event == push` (`--events`
overridable), `headBranch == <branch>`, and not a `skipped` completed run — a skipped run
verified nothing about the tree, so counting it would let a workflow that never executed
stand in for a reading of main's colour.

The raw `--limit` is NOT the window. It is escalated (50 -> 200 -> 500) until N applicable
runs have been collected, because the naive top-N of *anything* can be — and on
2026-08-08 was — fully evicted by high-volume issue automation (`Squad Triage`, `Squad
Issue Assign`, `Squad Heartbeat (Ralph)`, all `event: issues`, none ever a BAD
conclusion). At that moment this screen read ten bot runs, found no failure among them,
and printed PASS while three unresolved CI reds sat just outside the window it looked at.
A window that bot traffic can dilute is not an observation of main's colour; it is a coin
flip on how much bot traffic landed between two reads. So: the window is over applicable
runs, and an under-filled window is DECLINED, never truncated into a green.

Every source — the live `gh` calls, `--from-json`, `--from-json-map` — is a page fetcher
handed to ONE loop (`collect_applicable_window`). There is exactly one implementation of
filter/escalate/decide in this file, so a test that drives any source drives the same
decision code the production path runs. That is deliberate: the previous shape of this fix
had three copies of that logic, and every test drove a copy that production did not use.

It rules in R13 vocabulary:

  every completed applicable run succeeded    -> PASS
  any completed applicable run failed         -> FAIL(condition=main_is_red), exit 1,
                                                 with the run URL and the head sha
  fewer than N applicable runs in all history -> ERROR(instrument=insufficient_history), 4
  escalation cap reached without N            -> ERROR(instrument=search_capped), 4
  window has zero COMPLETED applicable runs   -> ERROR(instrument=no_completed_evidence), 4
  a run record that is not an object, or a
    payload that is not an array of them      -> ERROR(instrument=malformed_payload), 4
  the whole-screen time budget ran out        -> ERROR(instrument=deadline_exhausted), 4
  gh absent / unauthenticated / API error     -> ERROR(instrument=github_unreachable), 4

WHAT IT DELIBERATELY DOES NOT DO
================================
It does not treat "I could not ask" as "the answer was yes". Offline is UNOBSERVABLE, not
green — that is the whole R13 point and it is the reason this is exit 4 and not exit 0
with a shrug. A coordinator who runs this on a plane learns that they do not know the
colour of `main`, which is a true and useful thing to learn, and is precisely the state
that held for ten pushes while everybody believed the opposite.

It does not gate on `in_progress` runs. A run that has not finished has not failed, and
blocking on it would make the screen a wait rather than a reading. It reports them so the
reader knows the answer is partial. But a window in which *nothing* has completed carries
no evidence at all, and that is ERROR(instrument=no_completed_evidence), not a PASS with a
footnote: zero completed runs is a denominator of zero, and "no completed run failed" is
true of a branch nobody has ever built.

It does not read the *badge*. A badge is an image of the default workflow's latest
conclusion on the default branch and it silently omits every other workflow. This asks
for the runs.

COST
====
Up to three network round trips, not one: the raw `--limit` escalates 50 -> 200 -> 500
until the applicable window is full, so a repo with heavy non-CI automation pays two or
three `gh run list` calls instead of one (on 2026-08-08 this branch needed two — a raw 50
contained zero applicable runs, a raw 200 contained twenty-seven). The whole screen runs
under a single end-to-end deadline (`--deadline-seconds`, default 90s) shared across every
call, so the worst case is bounded by that one number rather than by 3x an independent
per-call timeout — and it is set below the 120s `timeout` the `main_is_green` register
entry declares, so a slow GitHub produces this screen's own
ERROR(instrument=deadline_exhausted) rather than the register's opaque ERROR(timeout).
Requires `gh` on PATH and an authenticated token with `repo` read. No build, no toolchain,
no device. That is still cheap enough that "I did not run it" cannot be a schedule
argument — which matters, because the gate it replaces ("I remembered to look") costs
nothing at all and was still skipped ten times running.

WIRING
======
Declared in `ci/open_reds.json` so it runs in CI like every other screen, and — the part
that reaches the reader who failed — invoked by the local pre-merge routine, so a report
that quotes green local gates cannot be produced without also having read the badge. The
`--for-merge` flag prints the one-line sentence a merge report is expected to contain,
including the run URL, so the sentence cannot be written from memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Callable, NamedTuple

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

BAD = {"failure", "timed_out", "cancelled", "startup_failure", "action_required"}
SKIPPED = "skipped"
FIELDS = (
    "conclusion,status,headSha,displayTitle,url,workflowName,createdAt,event,"
    "headBranch,databaseId"
)

# The instrument vocabulary. Every declined read names exactly one of these, and they are
# deliberately distinct: "I could not reach GitHub", "GitHub answered with something that
# is not a run list", "there is not enough history to fill the window", "I stopped looking
# rather than paginate forever", "nothing in the window has finished", and "I ran out of
# time" are six different states, and collapsing them would make the reason a reader gets
# useless for deciding what to do next. None of them is a green.
INSTRUMENT_UNREACHABLE = "github_unreachable"
INSTRUMENT_MALFORMED = "malformed_payload"
INSTRUMENT_INSUFFICIENT = "insufficient_history"
INSTRUMENT_CAPPED = "search_capped"
INSTRUMENT_NO_EVIDENCE = "no_completed_evidence"
INSTRUMENT_DEADLINE = "deadline_exhausted"

# The window this screen reads is defined over APPLICABLE runs, not the naive top-N of
# whatever `gh run list` hands back. "Applicable" means: the workflow that actually
# exercises the tree on a push to the target branch — by default the `CI` workflow
# (`.github/workflows/ci.yml`, the one badge.svg'd in README.md) triggered by `event ==
# push` on `headBranch == <branch>`. Every other workflow in this repo's
# `.github/workflows/` is either issue/PR-bot automation (`Squad Triage`, `Squad Issue
# Assign`, `Squad Heartbeat (Ralph)` — all `on: issues`, never a reading of the tree) or
# opt-in (`conformance.yml`, `workflow_dispatch` only). None of those answer the question
# this screen exists to answer, and a naive unfiltered window can be fully displaced by
# them.
DEFAULT_APPLICABLE_WORKFLOW = "CI"
DEFAULT_APPLICABLE_EVENTS = frozenset({"push"})

# When a page does not contain enough applicable runs, fetch progressively more raw
# history rather than accept a truncated read. Capped, so a repo whose automation
# out-runs its pushes by more than 50:1 makes this screen fail closed (ERROR, not a
# silent green) instead of paginating forever.
RAW_FETCH_STEPS = (50, 200, 500)

# One end-to-end budget for the whole screen, shared by every `gh` call it makes. It is
# NOT a per-call timeout: three independent 60s per-call timeouts would let a slow GitHub
# push this screen to 180s, past the 120s `timeout` its own `main_is_green` register entry
# declares, and the register would then report ERROR(timeout) — an outage of the harness
# rather than this screen's own, much more informative, deadline_exhausted.
DEFAULT_DEADLINE_SECONDS = 90.0

# ── THE CLOSURE RULE — code, because a register's `closes_when` is an instruction ──────
#
# `ci/open_reds.json` accepts this screen's red with a `closes_when` sentence, and
# `ci/check_open_reds.py` quotes that sentence verbatim the moment the screen turns green
# (FAIL(condition=stale_acceptance)). It is not commentary about the acceptance; it is the
# discharge procedure the next reader follows, and a wrong one sends them to do the wrong
# thing with the register's authority behind it.
#
# It was wrong. The entry was written when this screen ruled on the newest run, and said a
# push to `main` whose CI run completes green closes it. Issue #84's fix (PR #91) made the
# whole applicable WINDOW authoritative: `screen()` returns PASS iff the window filled, at
# least one run in it completed, and NO completed run in it is BAD. By 2026-08-09 that was
# the difference between a register instructing the next reader that one more green push
# discharges the acceptance, and a branch where eight green pushes had already landed in
# front of two reds still inside the window — the screen correctly red, the instruction
# correctly followed, and the two saying opposite things (issue #103).
#
# Two mechanisms keep them together, and they fail in different directions on purpose:
#
#   * `CLOSURE_RULE_ID` is owned HERE, beside the verdict it names. A register sentence
#     must carry it. If the rule below ever changes, this identifier changes with it and
#     every register still quoting the old one goes red — that is a version stamp, not a
#     keyword search, and it cannot be satisfied by prose that happens to use the right
#     nouns.
#   * `closure_prose_defects` additionally screens for the superseded single-event
#     readings, and `ci/check_open_reds.py` runs it before it will grant the acceptance.
#     The pin therefore lives in a production lane rather than only in a test.
#
# Neither of them can prove the identifier describes the verdict; only exercising the
# verdict can. `ci/test_lane_checks.py` does that on the production `gh` path in both
# polarities, and the case that matters is the one the old sentence got wrong: a FULL
# window whose newest run is green with a red still behind it.
CLOSURE_RULE_ID = "zero_red_completed_runs_in_the_applicable_window"

CLOSURE_STATEMENT = (
    "This screen goes green when the applicable window it rules on contains zero red "
    "runs — closure rule `" + CLOSURE_RULE_ID + "`. The window is the newest `--limit` "
    "APPLICABLE runs (workflow `CI`, event `push`, on the target branch, `skipped` "
    "excluded); every completed run in it must have succeeded, and at least one must have "
    "completed. The newest run's colour is not the window's colour: a green push in front "
    "of a red still inside the window is exactly the state this screen exists to convict."
)

# Claims a `closes_when` may not make about this screen, each paired with the wrong thing
# it would send a discharging reader off to do. These are the superseded single-event
# readings specifically — not a style guide, and not a substitute for CLOSURE_RULE_ID.
#
# THE LIMIT, STATED. This is a heuristic backstop and it is applied per sentence with a
# negation carve-out, so that a sentence which exists to DENY the wrong reading ("a single
# green push does not close this") is not convicted for containing it. A sufficiently
# contorted double negative would slip past. That is tolerable because it is the SECOND
# guard: a sentence carrying the wrong rule identifier is refused by the first one no
# matter how it is worded, and this one only narrows the ways a sentence carrying the
# right identifier can still mislead.
_CLOSURE_NEGATIONS = re.compile(r"\b(not|never|no longer|nothing|rather than|instead of)\b",
                                re.IGNORECASE)

_CLOSURE_FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\ba (?:single )?push to [`']?\w+[`']? whose\b", re.IGNORECASE),
        "it makes ONE push the closing event. The window is authoritative since PR #91, "
        "so a reader following this would flip the entry to expect=green while reds "
        "remained inside the window and this screen went on failing",
    ),
    (
        re.compile(r"\b(?:one|a single|the next|the latest|the newest)\s+green\s+"
                   r"(?:push|run|build|commit|merge)\b", re.IGNORECASE),
        "it names a single green run as sufficient. Zero red runs IN THE WINDOW is the "
        "condition; one green run is not the same claim and is weaker in exactly the "
        "direction that discharges an acceptance early",
    ),
    (
        re.compile(r"\b(?:latest|newest|most recent|head)\s+run\b.{0,40}\b"
                   r"(?:green|succe\w+|passes)\b", re.IGNORECASE),
        "it rules on the head run rather than the window. That is the pre-PR-#91 "
        "semantics and it is the defect issue #84 was filed for",
    ),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+")


def closure_prose_defects(closes_when: object) -> list[str]:
    """Reasons `closes_when` does not describe THIS screen's closure rule. Empty == agrees.

    Shipped here rather than in a test so that `ci/check_open_reds.py` — the production
    consumer of `closes_when` — can refuse an acceptance whose discharge instruction has
    drifted from the verdict it is an instruction about. A rule that only a test knows is
    a rule the register can violate in production for as long as nobody runs the test.

    Fails closed on a non-string: a register field that is not prose cannot be an
    instruction, and reporting "no defects" about it would be this screen's own
    unobservable-is-not-green error one level up.
    """
    if not isinstance(closes_when, str):
        return [
            f"`closes_when` is {type(closes_when).__name__}, not a string; a discharge "
            "instruction that is not prose instructs nobody"
        ]
    defects: list[str] = []
    if CLOSURE_RULE_ID not in closes_when:
        defects.append(
            f"it does not carry this screen's closure rule identifier "
            f"{CLOSURE_RULE_ID!r}. That token is owned beside the verdict in "
            "ci/check_main_is_green.py and changes when the verdict does, so a register "
            "that omits it is describing a rule nothing keeps it tied to. The rule is: "
            + CLOSURE_STATEMENT
        )
    for sentence in _SENTENCE_SPLIT.split(closes_when):
        if _CLOSURE_NEGATIONS.search(sentence):
            continue
        for pattern, why in _CLOSURE_FORBIDDEN:
            if pattern.search(sentence):
                defects.append(
                    f"the sentence {sentence.strip()[:160]!r} claims a closing condition "
                    f"this screen does not have: {why}"
                )
                break
    return defects


KNOWN_LIMITS = {
    "github_unreachable_is_unobservable_not_green": (
        "This screen's subject lives on github.com. With `gh` absent, unauthenticated, or "
        "offline it returns ERROR(instrument=github_unreachable), exit 4 — never a pass — "
        "so a disconnected reader gets `the colour of main is unknown to me` and every gate "
        "depending on it is UNOBSERVABLE rather than clean. That is the honest answer and it "
        "is still a gap: the failure this screen exists to prevent is a reader proceeding "
        "without a colour, and offline is a state in which no colour exists to give them. It "
        "is bounded (up to three `gh run list` calls under one shared deadline, `repo` read "
        "scope) and it cannot silently invert into a green. See ci/open_reds.json "
        "known_limits id=main_is_green_cannot_be_read_without_a_network."
    ),
}


def _assert_known_limit(name: str) -> int:
    if name not in KNOWN_LIMITS:
        print(
            f"ERROR(instrument=unknown_limit): {name!r} is not a declared limit of this "
            f"screen. Declared: {sorted(KNOWN_LIMITS)}. A register entry accepting a limit "
            "the screen does not admit to is an acceptance of nothing."
        )
        return EXIT_USAGE
    print(f"KNOWN-LIMIT {name}")
    print(f"  {KNOWN_LIMITS[name]}")
    print(
        "\nFAIL(condition=known_limit_still_open): declared, owned, bounded, and red on "
        "purpose. It goes green when the limit is closed, not when somebody stops looking."
    )
    return EXIT_FAIL_CONDITION



def _annotating() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _annotate(level: str, title: str, message: str) -> None:
    if _annotating():
        one = message.replace("\r", " ").replace("\n", "%0A")
        print(f"::{level} title={title}::{one}")


class Deadline:
    """One end-to-end time budget for the whole screen.

    Every `gh` call this screen makes draws from the same budget rather than getting its
    own independent timeout, so the worst-case wall time of the escalating search is the
    budget itself and not the number of steps times a per-call limit. That is what makes
    the register's declared `timeout` an upper bound this screen respects instead of a
    number it can quietly exceed."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds = float(seconds)
        self._clock = clock
        self._start = clock()

    def remaining(self) -> float:
        return self.seconds - (self._clock() - self._start)

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def spent(self) -> float:
        return self._clock() - self._start


class Page(NamedTuple):
    """What a page fetcher hands back for one requested raw `limit`.

    `instrument is None` means the fetch succeeded and `payload` is the *undissected*
    thing the source produced — normally the array `gh ... --json` prints, but possibly a
    JSON `null`, which is why success is signalled by the absent instrument and not by a
    non-None payload. The payload is deliberately not validated here: validation,
    filtering, escalation and the decision all live in `collect_applicable_window`, so
    that no source can drift into having its own private idea of what a run list is."""

    payload: object | None
    error: str
    instrument: str | None


class LiveRunListFetcher:
    """THE production source: `gh run list` against github.com.

    This is the fetcher `check_main_is_green.py` uses when it is doing its actual job,
    and the loop it is handed to is the same loop `--from-json`/`--from-json-map` drive.
    A test that swaps a `gh` shim onto PATH exercises this class and everything after
    it — which is the point: a seam nobody's production path crosses proves nothing
    about production."""

    kind = "live"

    def __init__(self, branch: str, deadline: Deadline) -> None:
        self.branch = branch
        self.deadline = deadline
        self.calls: list[int] = []

    def describe(self, limit: int) -> str:
        return f"`gh run list --branch {self.branch} --limit {limit}`"

    def __call__(self, limit: int) -> Page:
        self.calls.append(limit)
        gh = shutil.which("gh")
        if gh is None:
            return Page(None, "`gh` is not on PATH, so the colour of the branch was not "
                              "read", INSTRUMENT_UNREACHABLE)
        budget = self.deadline.remaining()
        if budget <= 0:
            return Page(None, f"the {self.deadline.seconds:g}s budget for this screen was "
                              f"spent before {self.describe(limit)} could be issued",
                        INSTRUMENT_DEADLINE)
        argv = ["gh", "run", "list", "--branch", self.branch, "--limit", str(limit),
                "--json", FIELDS]
        # Invoke the RESOLVED absolute path, not the bare name. On Windows CreateProcess
        # only appends `.exe`, so a `gh` that is a `.bat`/`.cmd` wrapper — or a shim a
        # test drops on PATH to drive this exact code path with no network — is found by
        # shutil.which() above and then not started at all, turning a scripted GitHub
        # into a spurious `gh could not be started`. Resolving keeps the guard and the
        # call talking about the same binary.
        argv[0] = gh
        try:
            proc = subprocess.run(
                argv,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=budget,
            )
        except subprocess.TimeoutExpired:
            return Page(None, f"{self.describe(limit)} did not answer inside the "
                              f"{budget:.1f}s left of this screen's "
                              f"{self.deadline.seconds:g}s budget", INSTRUMENT_DEADLINE)
        except OSError as exc:
            return Page(None, f"`gh run list` could not be started: {exc}",
                        INSTRUMENT_UNREACHABLE)
        if proc.returncode != 0:
            return Page(None, f"{self.describe(limit)} exited {proc.returncode}: "
                              f"{(proc.stderr or '').strip()[:400]}", INSTRUMENT_UNREACHABLE)
        try:
            return Page(json.loads(proc.stdout), "", None)
        except json.JSONDecodeError as exc:
            return Page(None, f"{self.describe(limit)} printed something that is not JSON: "
                              f"{exc}", INSTRUMENT_MALFORMED)


class FixedJsonFetcher:
    """`--from-json`: one fixed page, the same for every requested limit.

    A file is all the history there is, so when the loop asks for a bigger page and gets
    the same (shorter-than-asked-for) list back, its ordinary `this page was not full`
    rule concludes `history exhausted` without needing a rule of its own. That is why
    this class carries no filter, no escalation and no verdict."""

    kind = "from-json"

    def __init__(self, path: str) -> None:
        self.path = path
        self.calls: list[int] = []

    def describe(self, limit: int) -> str:
        return f"--from-json {self.path}"

    def __call__(self, limit: int) -> Page:
        self.calls.append(limit)
        try:
            with open(self.path, encoding="utf-8") as fh:
                return Page(json.load(fh), "", None)
        except OSError as exc:
            return Page(None, f"cannot read --from-json {self.path}: {exc}",
                        INSTRUMENT_UNREACHABLE)
        except json.JSONDecodeError as exc:
            return Page(None, f"--from-json {self.path} is not JSON: {exc}",
                        INSTRUMENT_MALFORMED)


class JsonMapFetcher:
    """`--from-json-map`: a JSON object keyed by the raw limit each page simulates.

    Lets a test script the *sequence* of pages an escalating search would see without a
    network. A key that is absent is an empty page, which the loop reads as `this source
    has nothing more`, exactly as a short page from `gh` would be."""

    kind = "from-json-map"

    def __init__(self, path: str) -> None:
        self.path = path
        self.calls: list[int] = []

    def describe(self, limit: int) -> str:
        return f"--from-json-map {self.path} key {str(limit)!r}"

    def __call__(self, limit: int) -> Page:
        self.calls.append(limit)
        try:
            with open(self.path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except OSError as exc:
            return Page(None, f"cannot read --from-json-map {self.path}: {exc}",
                        INSTRUMENT_UNREACHABLE)
        except json.JSONDecodeError as exc:
            return Page(None, f"--from-json-map {self.path} is not JSON: {exc}",
                        INSTRUMENT_MALFORMED)
        if not isinstance(doc, dict):
            return Page(None, f"--from-json-map {self.path} must be a JSON object keyed by "
                              "the raw limit each page simulates", INSTRUMENT_MALFORMED)
        return Page(doc.get(str(limit), []), "", None)


def _coerce_rows(payload: object, origin: str) -> tuple[list[dict] | None, str]:
    """The one place a payload becomes a list of run records.

    A `gh --json` payload is an array of objects. Anything else — a bare object, a
    string, `null`, or an array with a non-object (or `null`) element in it — is a
    payload this screen cannot read, and reading it anyway is how an AttributeError ends
    up on stderr wearing exit 1, the same exit code a genuine red uses. That collision is
    the whole reason this returns a named instrument instead of raising."""
    if not isinstance(payload, list):
        return None, (f"{origin} did not produce the array `gh ... --json` prints; it "
                      f"produced {type(payload).__name__}")
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            return None, (f"{origin} element {i} is {type(row).__name__}, not a run "
                          "object; a list this screen cannot read is not a list of "
                          "successful runs")
    return list(payload), ""


def _run_identity(run: dict) -> object | None:
    """A stable identity for one run, or None when the record carries none.

    Used to keep a duplicated record from filling two slots of the window: a window of
    ten in which the same run appears twice is a window of nine, and if the duplicate is
    green it has quietly diluted the sample the screen rules on."""
    ident = run.get("databaseId")
    if ident is not None:
        return ("databaseId", ident)
    url = run.get("url") or ""
    if url:
        return ("url", url)
    return None


def _is_applicable(run: dict, branch: str, workflow: str, events: frozenset[str]) -> bool:
    """A run answers the question this screen asks only if it is a `workflow` run,
    triggered by one of `events`, on `branch` itself, and actually ran. `skipped` is
    excluded even though it is `status: completed`: a skipped run verified nothing about
    the tree, so counting it toward the window would let a workflow that never executed
    stand in for a reading of main's colour — the same substitution this screen exists to
    refuse for an unread badge."""
    if (run.get("workflowName") or "") != workflow:
        return False
    if (run.get("event") or "") not in events:
        return False
    if (run.get("headBranch") or "") != branch:
        return False
    if (run.get("status") or "") == "completed" and (run.get("conclusion") or "") == SKIPPED:
        return False
    return True


class Window(NamedTuple):
    """`runs is None` means the window could NOT be filled, and `instrument` names why.
    A window that could not be filled is never a colour — that is the entire contract."""

    runs: list[dict] | None
    error: str
    instrument: str | None
    meta: dict


def collect_applicable_window(
    fetch_page: Callable[[int], Page],
    *,
    need: int,
    branch: str,
    workflow: str,
    events: frozenset[str],
    steps: tuple[int, ...] = RAW_FETCH_STEPS,
    deadline: Deadline | None = None,
) -> Window:
    """THE loop. Filter, escalate, decide — once, for every source.

    `fetch_page(limit) -> Page` is the only thing that differs between production (`gh`)
    and the two file-driven sources, and none of them gets an opinion about applicability,
    escalation, or when a partial read becomes an error. That is not tidiness: the
    previous shape of this fix carried three copies of this logic, one per source, and
    the shipped tests all drove copies that production never executed — so deleting the
    filter from the live copy, or making the live copy return a truncated raw window at
    the cap, left every test green while reintroducing the exact defect being fixed.

    Returns a Window whose `runs` is the newest `need` applicable runs, or `None` with a
    named instrument. It never returns fewer than `need` runs and never substitutes raw
    (unfiltered) runs for missing applicable ones: a diluted window truncated back to ten
    is the defect, not a fallback for it."""
    meta: dict = {
        "raw_scanned": 0, "steps_tried": 0, "steps": tuple(steps),
        "applicable_found": 0, "duplicates_dropped": 0, "last_page_size": 0,
    }
    ladder = tuple(s for s in steps if s >= need) or (need,)
    meta["steps"] = ladder

    for step in ladder:
        if deadline is not None and deadline.expired():
            return Window(None, (
                f"this screen's {deadline.seconds:g}s budget was spent after "
                f"{meta['steps_tried']} page(s); the window was never filled"
            ), INSTRUMENT_DEADLINE, meta)

        page = fetch_page(step)
        meta["steps_tried"] += 1
        if page.instrument is not None:
            return Window(None, page.error, page.instrument, meta)

        rows, err = _coerce_rows(page.payload, _describe(fetch_page, step))
        if rows is None:
            return Window(None, err, INSTRUMENT_MALFORMED, meta)

        meta["last_page_size"] = len(rows)
        meta["raw_scanned"] = max(meta["raw_scanned"], len(rows))

        applicable: list[dict] = []
        seen: set[object] = set()
        for run in rows:
            if not _is_applicable(run, branch, workflow, events):
                continue
            ident = _run_identity(run)
            if ident is not None:
                if ident in seen:
                    meta["duplicates_dropped"] += 1
                    continue
                seen.add(ident)
            applicable.append(run)
        meta["applicable_found"] = len(applicable)

        if len(applicable) >= need:
            return Window(applicable[:need], "", None, meta)

        if len(rows) < step:
            # The source returned fewer runs than asked for: that is every run there is,
            # and a bigger raw limit cannot find more of them.
            return Window(None, (
                f"only {len(applicable)} applicable run(s) of {need} needed were found "
                f"across all {len(rows)} run(s) that exist for `{branch}` (history "
                "exhausted)"
            ), INSTRUMENT_INSUFFICIENT, meta)

    return Window(None, (
        f"only {meta['applicable_found']} applicable run(s) of {need} needed were found "
        f"after scanning the last {ladder[-1]} run(s) on `{branch}`; the search is capped "
        "there rather than paginated indefinitely, and an under-filled window is declined "
        "rather than truncated into a colour"
    ), INSTRUMENT_CAPPED, meta)


def _describe(fetch_page: object, limit: int) -> str:
    describe = getattr(fetch_page, "describe", None)
    return describe(limit) if callable(describe) else f"the run source (limit {limit})"


def make_fetcher(args, deadline: Deadline):
    """Pick the page fetcher. This is the ONLY place the three sources differ, and none of
    them carries any of the decision logic they feed."""
    if args.from_json_map:
        return JsonMapFetcher(args.from_json_map)
    if args.from_json:
        return FixedJsonFetcher(args.from_json)
    return LiveRunListFetcher(args.branch, deadline)


def screen(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--branch", default="main")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument(
        "--from-json",
        dest="from_json",
        help="read the run list from a file instead of calling gh; this is how the "
             "two-polarity test drives both colours without a network. It is a FIXED "
             "page — the same list for every raw limit the search asks for — so the "
             "escalating loop reads it as `this is all the history there is`.",
    )
    ap.add_argument(
        "--from-json-map",
        dest="from_json_map",
        help="read escalating raw pages from a JSON object keyed by the raw limit each "
             "page simulates, instead of calling gh; this scripts the SEQUENCE of pages "
             "an escalating search sees, with no network. It feeds the same loop the "
             "live `gh` path feeds.",
    )
    ap.add_argument(
        "--workflow",
        default=DEFAULT_APPLICABLE_WORKFLOW,
        help="the workflow name that answers this question (default: %(default)s, the "
             "one badge.svg'd in README.md). Runs from any other workflow — including "
             "issue automation like Squad Triage — are not applicable and are excluded "
             "from the window no matter how recent they are.",
    )
    ap.add_argument(
        "--events",
        default=",".join(sorted(DEFAULT_APPLICABLE_EVENTS)),
        help="comma-separated GitHub event names that answer this question (default: "
             "%(default)s). A run triggered by `issues`, `schedule`, or "
             "`workflow_dispatch` is not a reading of a push to the branch.",
    )
    ap.add_argument(
        "--deadline-seconds",
        dest="deadline_seconds",
        type=float,
        default=DEFAULT_DEADLINE_SECONDS,
        help="one end-to-end budget for the whole screen, shared by every gh call it "
             "makes (default: %(default)s). Deliberately below the timeout the "
             "main_is_green register entry declares, so a slow GitHub yields this "
             "screen's ERROR(instrument=deadline_exhausted) rather than the register's "
             "ERROR(timeout).",
    )
    ap.add_argument(
        "--assert-known-limit",
        dest="assert_known_limit",
        help="name a declared limit of this screen and fail on it deliberately; the "
             "register uses this so an accepted gap has to be one the screen admits to.",
    )
    ap.add_argument(
        "--for-merge",
        action="store_true",
        help="print the sentence a merge report must contain, with the run URL in it.",
    )
    args = ap.parse_args(argv)

    if args.assert_known_limit:
        return _assert_known_limit(args.assert_known_limit)

    if args.limit < 1:
        print("MAIN-GREEN: ERROR(instrument=usage) --limit must be >= 1")
        return EXIT_USAGE

    events = frozenset(e.strip() for e in args.events.split(",") if e.strip())
    if not events:
        print("MAIN-GREEN: ERROR(instrument=usage) --events must name at least one event")
        return EXIT_USAGE

    if args.deadline_seconds <= 0:
        print("MAIN-GREEN: ERROR(instrument=usage) --deadline-seconds must be > 0")
        return EXIT_USAGE

    deadline = Deadline(args.deadline_seconds)
    window = collect_applicable_window(
        make_fetcher(args, deadline),
        need=args.limit,
        branch=args.branch,
        workflow=args.workflow,
        events=events,
        deadline=deadline,
    )
    runs, err, meta = window.runs, window.error, window.meta
    if runs is None:
        instrument = window.instrument or INSTRUMENT_UNREACHABLE
        print(f"MAIN-GREEN: ERROR(instrument={instrument})")
        print(f"  {err}")
        print(
            f"  This screen's window is the last {args.limit} run(s) of workflow "
            f"`{args.workflow}` triggered by event(s) {sorted(events)} on "
            f"`{args.branch}` — not the last {args.limit} run(s) of any workflow or "
            f"event. {meta['steps_tried']} raw page(s) were read (largest "
            f"{meta['raw_scanned']} run(s)) and the window was not filled."
        )
        print(
            f"  The colour of `{args.branch}` was NOT observed. This is UNOBSERVABLE, not "
            "green: a local gate run says nothing about the branch's CI, and the ten "
            "consecutive red pushes that made this screen necessary were all made by "
            "someone holding a green local transcript. An under-filled window is declined "
            "for the same reason an unreachable GitHub is — a read that did not happen "
            "must never render as a read that came back clean."
        )
        _annotate("error", "main colour unread", err)
        return EXIT_ERROR_INSTRUMENT

    completed = [r for r in runs if (r.get("status") or "") == "completed"]
    pending = [r for r in runs if (r.get("status") or "") != "completed"]
    red = [r for r in completed if (r.get("conclusion") or "") in BAD]

    head = runs[0]
    print(f"MAIN-GREEN census of `{args.branch}` (applicable = workflow "
          f"`{args.workflow}`, event(s) {sorted(events)}): {len(runs)} run(s) in window, "
          f"{len(completed)} completed, {len(pending)} still running, {len(red)} RED")
    print(f"  window filled from {meta['steps_tried']} raw page(s), largest "
          f"{meta['raw_scanned']} run(s)"
          + (f", {meta['duplicates_dropped']} duplicate record(s) dropped"
             if meta["duplicates_dropped"] else ""))
    print(f"  latest: {head.get('workflowName', '?')} — {head.get('conclusion') or head.get('status')}")
    print(f"          {head.get('headSha', '')[:12]}  {head.get('displayTitle', '')}")
    print(f"          {head.get('url', '')}")

    if not completed:
        print("")
        print(f"MAIN-GREEN: ERROR(instrument={INSTRUMENT_NO_EVIDENCE})")
        print(
            f"  All {len(runs)} applicable run(s) in the window are still in flight; not "
            "one of them has finished. `no completed run failed` is true of a branch that "
            "has never been built, so a full window with an empty completed set carries no "
            "evidence about the colour of "
            f"`{args.branch}` and is declined rather than reported green."
        )
        _annotate(
            "error", "main colour unread",
            f"{len(runs)} applicable run(s), none completed",
        )
        return EXIT_ERROR_INSTRUMENT

    if red:
        print("")
        print(
            f"FAIL(condition=main_is_red): {len(red)} completed run(s) on `{args.branch}` "
            "did not succeed. By the discipline stated in this repository's README a red "
            "badge blocks merges, and a green local gate set is not a second opinion about "
            "it — it is an answer to a different question."
        )
        for r in red:
            print(f"  - {r.get('conclusion')}  {r.get('workflowName', '?')}  "
                  f"{r.get('headSha', '')[:12]}  {r.get('displayTitle', '')}")
            print(f"      {r.get('url', '')}")
        _annotate(
            "error", f"{args.branch} is red",
            f"{len(red)} failed run(s); newest {red[0].get('url', '')}",
        )
        if args.for_merge:
            print("")
            print(f"MERGE REPORT SENTENCE: `{args.branch}` is RED at "
                  f"{red[0].get('headSha', '')[:12]} — {red[0].get('url', '')} — "
                  "this merge is blocked until it is green or the red is declared in "
                  "ci/open_reds.json with an owner.")
        return EXIT_FAIL_CONDITION

    if pending:
        print("")
        print(
            f"  {len(pending)} run(s) have not finished. They have not failed, so they are "
            "not counted red, but the reading is partial and the merge report should say so."
        )
        _annotate(
            "warning", f"{args.branch} colour is partial",
            f"{len(pending)} run(s) still in progress",
        )

    print("")
    print(f"PASS: every completed applicable run on `{args.branch}` in the window of "
          f"{len(runs)} succeeded.")
    if args.for_merge:
        print(f"MERGE REPORT SENTENCE: `{args.branch}` is GREEN at "
              f"{head.get('headSha', '')[:12]} — {head.get('url', '')}"
              + (f" ({len(pending)} run(s) still in flight)" if pending else ""))
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(screen())
