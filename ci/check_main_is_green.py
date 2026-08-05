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
One `gh` call:

    gh run list --branch main --limit N --json conclusion,status,headSha,displayTitle,url,workflowName

and it rules in R13 vocabulary:

  every completed run on main succeeded      -> PASS
  any completed run on main failed/cancelled -> FAIL(condition=main_is_red), exit 1,
                                                with the run URL and the head sha
  gh absent / unauthenticated / offline      -> ERROR(instrument=github_unreachable), exit 4

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
FIELDS = "conclusion,status,headSha,displayTitle,url,workflowName,createdAt"

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

    runs, err = fetch(args.branch, args.limit, args.from_json)
    if runs is None:
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
    print(f"MAIN-GREEN census of `{args.branch}`: {len(runs)} run(s) listed, "
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
