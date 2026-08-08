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
`gh` call(s), escalating a raw --limit until enough APPLICABLE runs are found:

    gh run list --branch main --limit N --json conclusion,status,headSha,displayTitle,url,workflowName,event,headBranch

"Applicable" means: workflow `CI` (the one badge.svg'd in README.md), `event == push`,
`headBranch == main`, not a `skipped` conclusion — i.e. an actual reading of a push to
`main`, not issue/PR automation (Squad Triage, Squad Issue Assign, Squad Heartbeat — all
`event: issues`) or an opt-in workflow (`conformance.yml`, `workflow_dispatch` only) that
happens to have run recently. A naive top-N window with no such filter can be fully
evicted by high-volume non-applicable noise, at which point the screen would report green
having read zero real CI history — that defect (found live 2026-08-08, ref: the issue
this fix closes) is exactly the "a check whose failure reaches no one" pattern this
screen was built to name, one level further in.

It rules in R13 vocabulary:

  every completed applicable run on main succeeded  -> PASS
  any completed applicable run on main failed        -> FAIL(condition=main_is_red), exit 1,
                                                         with the run URL and the head sha
  fewer than N applicable runs exist, or the search
    was capped before finding N                      -> ERROR(instrument=insufficient_history
                                                         |search_capped), exit 4 — declined,
                                                         never a silent green
  gh absent / unauthenticated / offline               -> ERROR(instrument=github_unreachable),
                                                         exit 4

WHAT IT DELIBERATELY DOES NOT DO
================================
It does not treat "I could not ask" as "the answer was yes". Offline is UNOBSERVABLE, not
green — that is the whole R13 point and it is the reason this is exit 4 and not exit 0
with a shrug. A coordinator who runs this on a plane learns that they do not know the
colour of `main`, which is a true and useful thing to learn, and is precisely the state
that held for ten pushes while everybody believed the opposite.

It does not gate on `in_progress` runs. A run that has not finished has not failed, and
blocking on it would make the screen a wait rather than a reading. It reports them so the
reader knows the answer is partial.

It does not read the *badge*. A badge is an image of the default workflow's latest
conclusion on the default branch and it silently omits every other workflow. This asks
for the runs.

COST
====
One network round trip, typically under a second, requiring `gh` on PATH and an
authenticated token with `repo` read. No build, no toolchain, no device. That is cheap
enough that "I did not run it" cannot be a schedule argument — which matters, because the
gate it replaces ("I remembered to look") costs nothing at all and was still skipped ten
times running.

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
import shutil
import subprocess
import sys

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

BAD = {"failure", "timed_out", "cancelled", "startup_failure", "action_required"}
SKIPPED = "skipped"
FIELDS = "conclusion,status,headSha,displayTitle,url,workflowName,createdAt,event,headBranch"

# The window this screen reads is defined over APPLICABLE runs, not the naive top-N of
# whatever `gh run list` hands back. "Applicable" means: the workflow that actually
# exercises the tree on a push to the target branch — by default the `CI` workflow
# (`.github/workflows/ci.yml`, the one badge.svg'd in README.md) triggered by `event ==
# push` on `headBranch == <branch>`. Every other workflow in this repo's
# `.github/workflows/` is either issue/PR-bot automation (`Squad Triage`, `Squad Issue
# Assign`, `Squad Heartbeat (Ralph)` — all `on: issues`/`pull_request`, never a reading of
# the tree) or opt-in (`conformance.yml`, `workflow_dispatch` only). None of those answer
# the question this screen exists to answer, and a naive unfiltered window can be fully
# displaced by them: high-volume bot noise fills all N slots and evicts real CI history,
# so the screen reports green while unresolved CI reds sit just outside the window it
# actually looked at. That is not an observation of main's colour; it is a coin flip on
# how much bot traffic happened to land between two reads.
DEFAULT_APPLICABLE_WORKFLOW = "CI"
DEFAULT_APPLICABLE_EVENTS = frozenset({"push"})

# When the naive top window doesn't contain enough applicable runs, fetch progressively
# more raw history rather than accept a truncated read. Capped, so a noisy repo makes
# this screen fail closed (ERROR, not a silent green) instead of paginating forever.
RAW_FETCH_STEPS = (50, 200, 500)

KNOWN_LIMITS = {
    "github_unreachable_is_unobservable_not_green": (
        "This screen's subject lives on github.com. With `gh` absent, unauthenticated, or "
        "offline it returns ERROR(instrument=github_unreachable), exit 4 — never a pass — "
        "so a disconnected reader gets `the colour of main is unknown to me` and every gate "
        "depending on it is UNOBSERVABLE rather than clean. That is the honest answer and it "
        "is still a gap: the failure this screen exists to prevent is a reader proceeding "
        "without a colour, and offline is a state in which no colour exists to give them. It "
        "is bounded (one API call, ~1s, `repo` read scope) and it cannot silently invert "
        "into a green. See ci/open_reds.json known_limits "
        "id=main_is_green_cannot_be_read_without_a_network."
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


def fetch(branch: str, limit: int, source: str | None) -> tuple[list[dict] | None, str]:
    """Return (runs, error). `runs is None` means the instrument did not reach GitHub."""
    if source:
        try:
            with open(source, encoding="utf-8") as fh:
                doc = json.load(fh)
        except OSError as exc:
            return None, f"cannot read --from-json {source}: {exc}"
        except json.JSONDecodeError as exc:
            return None, f"--from-json {source} is not JSON: {exc}"
        if not isinstance(doc, list):
            return None, f"--from-json {source} must be the array `gh ... --json` prints"
        return doc, ""

    if shutil.which("gh") is None:
        return None, "`gh` is not on PATH, so the colour of the branch was not read"
    try:
        proc = subprocess.run(
            ["gh", "run", "list", "--branch", branch, "--limit", str(limit),
             "--json", FIELDS],
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None, "`gh run list` did not answer within 60s"
    except OSError as exc:
        return None, f"`gh run list` could not be started: {exc}"
    if proc.returncode != 0:
        return None, f"`gh run list` exited {proc.returncode}: {proc.stderr.strip()[:400]}"
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return None, f"`gh run list --json` printed something that is not JSON: {exc}"
    if not isinstance(doc, list):
        return None, "`gh run list --json` did not print an array"
    return doc, ""


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


def fetch_applicable(
    branch: str,
    need: int,
    workflow: str,
    events: frozenset[str],
    source: str | None,
    source_map: str | None,
) -> tuple[list[dict] | None, str, dict]:
    """Return (applicable_runs, error, meta). `applicable_runs is None` means the window
    could not be filled: either the instrument did not reach GitHub (`meta["kind"] ==
    "unreachable"`, propagated verbatim from `fetch`), or fewer than `need` applicable
    runs exist in all available history (`meta["kind"] == "insufficient_history"`), or the
    escalating search was capped before finding `need` (`meta["kind"] ==
    "search_capped"`). None of those are a green — they are declined reads, reported with
    an explicit reason, exactly like ERROR(instrument=github_unreachable) already is for
    this screen.
    """
    meta: dict = {"raw_scanned": 0, "steps_tried": 0, "kind": None}

    if source_map:
        try:
            with open(source_map, encoding="utf-8") as fh:
                pages = json.load(fh)
        except OSError as exc:
            meta["kind"] = "unreachable"
            return None, f"cannot read --from-json-map {source_map}: {exc}", meta
        except json.JSONDecodeError as exc:
            meta["kind"] = "unreachable"
            return None, f"--from-json-map {source_map} is not JSON: {exc}", meta
        if not isinstance(pages, dict):
            meta["kind"] = "unreachable"
            return None, f"--from-json-map {source_map} must be a JSON object keyed by limit", meta
        steps = tuple(s for s in RAW_FETCH_STEPS if s >= need) or (need,)
        last_raw_len = 0
        last_applicable = 0
        for step in steps:
            meta["steps_tried"] += 1
            raw = pages.get(str(step), [])
            if not isinstance(raw, list):
                meta["kind"] = "unreachable"
                return None, f"--from-json-map {source_map} key {step!r} is not an array", meta
            last_raw_len = len(raw)
            meta["raw_scanned"] = max(meta["raw_scanned"], last_raw_len)
            applicable = [r for r in raw if _is_applicable(r, branch, workflow, events)]
            last_applicable = len(applicable)
            if len(applicable) >= need:
                return applicable[:need], "", meta
            if last_raw_len < step:
                meta["kind"] = "insufficient_history"
                return None, (
                    f"only {last_applicable} applicable run(s) of {need} needed found across "
                    f"all {last_raw_len} run(s) available for `{branch}` (history exhausted)"
                ), meta
        meta["kind"] = "search_capped"
        return None, (
            f"only {last_applicable} applicable run(s) of {need} needed found through a raw "
            f"window of {steps[-1]}; the search is capped there rather than paginated "
            "indefinitely"
        ), meta

    if source:
        raw, err = fetch(branch, need, source)
        if raw is None:
            meta["kind"] = "unreachable"
            return None, err, meta
        meta["raw_scanned"] = len(raw)
        meta["steps_tried"] = 1
        applicable = [r for r in raw if _is_applicable(r, branch, workflow, events)]
        if len(applicable) >= need:
            return applicable[:need], "", meta
        meta["kind"] = "insufficient_history"
        return None, (
            f"only {len(applicable)} applicable run(s) of {need} needed found in the "
            f"{len(raw)} run(s) provided by --from-json (a fixed, non-paginating source)"
        ), meta

    steps = tuple(s for s in RAW_FETCH_STEPS if s >= need) or (need,)
    last_raw_len = 0
    last_applicable = 0
    for step in steps:
        meta["steps_tried"] += 1
        raw, err = fetch(branch, step, None)
        if raw is None:
            meta["kind"] = "unreachable"
            return None, err, meta
        last_raw_len = len(raw)
        meta["raw_scanned"] = max(meta["raw_scanned"], last_raw_len)
        applicable = [r for r in raw if _is_applicable(r, branch, workflow, events)]
        last_applicable = len(applicable)
        if len(applicable) >= need:
            return applicable[:need], "", meta
        if last_raw_len < step:
            # gh returned fewer runs than asked for: that is every run there is, and a
            # bigger --limit would not find more of them.
            meta["kind"] = "insufficient_history"
            return None, (
                f"only {last_applicable} applicable run(s) of {need} needed found across "
                f"all {last_raw_len} run(s) that exist for `{branch}` (history exhausted)"
            ), meta

    meta["kind"] = "search_capped"
    return None, (
        f"only {last_applicable} applicable run(s) of {need} needed found after scanning "
        f"the last {steps[-1]} run(s) on `{branch}`; declining to search further rather "
        "than pass a possibly-incomplete read as a colour"
    ), meta


def screen(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--branch", default="main")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument(
        "--from-json",
        dest="from_json",
        help="read the run list from a file instead of calling gh; this is how the "
             "two-polarity test drives both colours without a network.",
    )
    ap.add_argument(
        "--from-json-map",
        dest="from_json_map",
        help="read escalating raw pages from a JSON object (keyed by the raw --limit "
             "each page simulates) instead of calling gh; this is how the pagination "
             "tests drive the applicable-run search past a first, too-small page with "
             "no network.",
    )
    ap.add_argument(
        "--workflow",
        default=DEFAULT_APPLICABLE_WORKFLOW,
        help="the workflow name that answers this question (default: %(default)s, the "
             "one badge.svg'd in README.md). Runs from any other workflow — including "
             "issue/PR automation like Squad Triage — are not applicable and are "
             "excluded from the window regardless of how recent they are.",
    )
    ap.add_argument(
        "--events",
        default=",".join(sorted(DEFAULT_APPLICABLE_EVENTS)),
        help="comma-separated GitHub event names that answer this question (default: "
             "%(default)s). A run triggered by `issues`, `schedule`, or "
             "`workflow_dispatch` is not a reading of a push to the branch.",
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

    runs, err, meta = fetch_applicable(
        args.branch, args.limit, args.workflow, events, args.from_json, args.from_json_map,
    )
    if runs is None:
        kind = meta.get("kind")
        if kind in ("insufficient_history", "search_capped"):
            print(f"MAIN-GREEN: ERROR(instrument={kind})")
            print(f"  {err}")
            print(
                f"  This screen's window is the last {args.limit} run(s) of workflow "
                f"`{args.workflow}` triggered by event(s) {sorted(events)} on `{args.branch}` "
                f"— not the last {args.limit} run(s) of any workflow/event. "
                f"{meta['steps_tried']} raw page(s) were scanned (up to {meta['raw_scanned']} "
                "run(s) in the largest) and that was not enough to fill it. The colour of "
                f"`{args.branch}` was NOT observed: an incomplete applicable window must "
                "never be reported as a clean one, the same way an unreachable GitHub must "
                "never be reported as a green one."
            )
        else:
            print("MAIN-GREEN: ERROR(instrument=github_unreachable)")
            print(f"  {err}")
            print(
                f"  The colour of `{args.branch}` was NOT observed. This is UNOBSERVABLE, not "
                "green: a local gate run says nothing about the branch's CI, and the ten "
                "consecutive red pushes that made this screen necessary were all made by "
                "someone holding a green local transcript."
            )
        _annotate("error", "main colour unread", err)
        return EXIT_ERROR_INSTRUMENT

    if not runs:
        print("MAIN-GREEN: ERROR(instrument=no_runs_listed)")
        print(
            f"  GitHub listed zero workflow runs for `{args.branch}`. A branch with no runs "
            "has not been screened; that is not the same as a branch that passed."
        )
        _annotate("error", "main colour unread", f"no runs listed for {args.branch}")
        return EXIT_ERROR_INSTRUMENT

    completed = [r for r in runs if (r.get("status") or "") == "completed"]
    pending = [r for r in runs if (r.get("status") or "") != "completed"]
    red = [r for r in completed if (r.get("conclusion") or "") in BAD]

    head = runs[0]
    print(f"MAIN-GREEN census of `{args.branch}` (applicable = workflow `{args.workflow}`, "
          f"event(s) {sorted(events)}): {len(runs)} run(s) in window, "
          f"{len(completed)} completed, {len(pending)} still running, {len(red)} RED")
    print(f"  latest: {head.get('workflowName', '?')} — {head.get('conclusion') or head.get('status')}")
    print(f"          {head.get('headSha', '')[:12]}  {head.get('displayTitle', '')}")
    print(f"          {head.get('url', '')}")

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
    print(f"PASS: every completed run on `{args.branch}` in the last {len(runs)} succeeded.")
    if args.for_merge:
        print(f"MERGE REPORT SENTENCE: `{args.branch}` is GREEN at "
              f"{head.get('headSha', '')[:12]} — {head.get('url', '')}"
              + (f" ({len(pending)} run(s) still in flight)" if pending else ""))
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(screen())
