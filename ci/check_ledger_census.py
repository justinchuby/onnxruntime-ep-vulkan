#!/usr/bin/env python3
"""Census the proof ledger against its own history: has an entry ever gone MISSING?

WHY THIS EXISTS
===============
`gen_proof_ledger.py --check` asks, of every entry in the file, whether it agrees with the
build. It is a good question and it is answered well. It is also the wrong question for one
specific failure, because it can only be asked **about entries that are still there**.

On 2026-08-03 Tank found that `26fd93f` proved and committed three `Cast` forms —
`f32>i32`, `i32>f32`, `f32>bool` — and that they were absent from `main`. `git log -S` on
the ledger found the addition and **no removal**: the deletion happened inside a conflict
resolution in the merge `eb84364`, and history simplification then hid the original commit
from the file's own log. So:

* `--check` was green, because every entry that remained agreed with the build.
* the shrinking-write guard did not fire, because that guard covers writes by **the tool**,
  and a merge is not one.
* `check_open_reds.py` did not fire, because the ledger is not a check.

The only instrument that noticed was the op suite, which went red at that merge and stayed
red, **unaccounted** — and an unaccounted red is how a deletion survives. That is this
repository's own lesson from `check_open_reds.py`, one artifact over: a sum cannot see one
of its terms go silent, and a *proof* is a term.

THE ARITHMETIC
==============
The same shape `check_open_reds.py` applies to subjects, applied to proofs:

    N ever proven = M present now + K retired + D VANISHED

`D > 0` is the failure. Everything else is disclosure.

WHERE `N` COMES FROM, AND WHY IT IS NOT A FILE
==============================================
`check_open_reds.py` keeps its denominator in an append-only `subjects` list inside the
register it screens. That works there because a human edits that register deliberately.

It would **not** work here, and the reason is the whole point of this screen: the event
being detected is a merge silently rewriting the ledger. A denominator that lives in a
tracked file next to the ledger is rewritten by exactly the same merge, in the same
conflict resolution, by the same hand. It would have been deleted alongside the three Cast
entries and the census would have balanced.

So `N` is derived from **git history**: the union of every `key` that has ever appeared in
any revision of the ledger reachable from the repository's refs. History is append-only in
the sense that matters here — a merge commit adds to it and cannot subtract from it. The
denominator is therefore held somewhere the failure mode cannot reach.

`--simplify` is deliberately NOT used when listing revisions (`git rev-list --all`), because
history simplification is precisely what hid `26fd93f` from the file's own log.

RETIREMENT
==========
A proof may legitimately go away: a form is removed from the EP, an op is withdrawn, a
duplicate key is collapsed. That is a written act, in `evidence/proof_retired.json`, with an
owner, a date and a reason — the same three fields `check_open_reds.py` requires to retire a
subject, for the same argument. Retiring is not suppression: a retired key is *named* in the
output every run.

WHAT THIS DOES NOT CLAIM
========================
* Not that the surviving entries are correct — `gen_proof_ledger.py --check` is that screen
  and the two must stay separate. This one never opens the built EP.
* Not that every key ever written was *deliberately* written; a key added by mistake and
  removed on purpose is a retirement, and must be recorded as one.
* Not that the ledger is complete. "Which forms ought to be proven" is the census question
  `probe_model_op_census.py` asks. This asks only whether something that WAS proven has
  quietly stopped being proven.

USAGE
=====
    python ci/check_ledger_census.py                    # screen HEAD's ledger against history
    python ci/check_ledger_census.py --at <rev>         # replay: screen an older revision
    python ci/check_ledger_census.py --json <path>      # machine-readable record
    python ci/check_ledger_census.py --list-retired

Standard library only, and `git`. No flag suppresses a failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LEDGER_REL = "evidence/proof_ledger.jsonl"
RETIRED = REPO / "evidence" / "proof_retired.json"
RETIRED_FIELDS = ("owner", "date", "reason")


def _git(args: list[str], repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, encoding="utf-8"
    )


def _keys_of(text: str) -> set[str]:
    """Every `key` in a ledger body. A line without one is a header, not an entry."""
    out = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = entry.get("key")
        if isinstance(key, str) and key:
            out.add(key)
    return out


def ever_proven(repo: Path, upto: str | None = None) -> dict[str, str]:
    """key -> the EARLIEST revision that carried it.

    `git rev-list --all --full-history` and not `--simplify-merges`/`--follow`: the removal
    this screen exists for was invisible to a simplified log, so a simplified log cannot be
    the input.

    `--full-history` is load-bearing and was added after the first replay arm reported PASS on
    the very merge it was written to convict. Default history simplification drops commits
    whose change is "not interesting" for the path once a merge has picked a side — at
    `eb84364`, `git rev-list <rev> -- evidence/proof_ledger.jsonl` lists **13** revisions and
    `26fd93f`, the commit that proved the three `Cast` forms, is not one of them; with
    `--full-history` it lists **55** and `26fd93f` is there. That is not a detail: the
    simplification that hid the proving commit from the file's own log is the same
    simplification that would hide it from this census, and a screen for a deletion must not
    take its denominator from the view the deletion is invisible in.

    `upto` bounds the walk to the ancestors of one revision, and it is not an optimisation.
    Without it the replay arm is dishonest in both directions: a key proven *after* the
    revision under test would be reported VANISHED from it (it was never there to lose), and
    the count would answer a question about the future. The denominator must be "what this
    revision's own history had already proven".

    `rev-list` is reverse-chronological, so the walk is reversed to make "first", first.
    """
    scope = [upto] if upto else ["--all"]
    revs = _git(["rev-list", "--full-history", *scope, "--", LEDGER_REL], repo).stdout.split()
    seen: dict[str, str] = {}
    for rev in reversed(revs):
        blob = _git(["show", f"{rev}:{LEDGER_REL}"], repo)
        if blob.returncode != 0:
            continue
        for key in _keys_of(blob.stdout):
            seen.setdefault(key, rev)
    return seen


def present_at(repo: Path, rev: str | None) -> set[str]:
    if rev is None:
        path = repo / LEDGER_REL
        if not path.is_file():
            raise FileNotFoundError(
                f"no ledger at {path}; the comparison input is missing, so nothing was ruled "
                "on either way — that is UNOBSERVABLE, not PASS"
            )
        return _keys_of(path.read_text(encoding="utf-8"))
    blob = _git(["show", f"{rev}:{LEDGER_REL}"], repo)
    if blob.returncode != 0:
        raise FileNotFoundError(f"{rev} has no {LEDGER_REL}: {blob.stderr.strip()}")
    return _keys_of(blob.stdout)


def load_retired(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    retired = doc.get("retired", {})
    if not isinstance(retired, dict):
        raise ValueError(f"{path}: `retired` must be an object keyed by proof key")
    for key, rec in retired.items():
        missing = [f for f in RETIRED_FIELDS if not (isinstance(rec, dict) and rec.get(f))]
        if missing:
            raise ValueError(
                f"{path}: retired proof {key!r} is missing {missing}. Withdrawing a proof "
                "needs a name and a reason for the same argument that accepting a red does."
            )
    return retired


def screen(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument(
        "--at",
        default=None,
        help="screen the ledger as of REV instead of the working tree. This is the replay "
             "arm: it is how the screen is shown to convict a real, historical deletion "
             "rather than only a planted one.",
    )
    ap.add_argument("--json", default="")
    ap.add_argument("--list-retired", action="store_true")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()

    retired = load_retired(RETIRED)
    if args.list_retired:
        if not retired:
            print("no proofs are retired.")
            return 0
        for key, rec in sorted(retired.items()):
            print(f"{key}\n    {rec['owner']} {rec['date']}: {rec['reason']}")
        return 0

    ever = ever_proven(repo, args.at)
    now = present_at(repo, args.at)
    where = args.at or "the working tree"

    vanished = sorted(k for k in ever if k not in now and k not in retired)
    retired_present = sorted(k for k in retired if k in now)
    unhistoried = sorted(k for k in now if k not in ever)

    n, m, k, d = len(ever), len(now), len([x for x in retired if x not in now]), len(vanished)
    print(f"LEDGER CENSUS of {LEDGER_REL} at {where}")
    print(
        f"  {n} ever proven = {m - len(unhistoried)} present now + {k} retired + {d} VANISHED"
        + (f" (+{len(unhistoried)} present but not yet in history — uncommitted)" if unhistoried else "")
    )
    if retired:
        print(f"  retired ({len(retired)}):")
        for key in sorted(retired):
            rec = retired[key]
            print(f"    - {key}  [{rec['owner']} {rec['date']}: {rec['reason']}]")

    if retired_present:
        print("")
        print(
            f"FAIL(condition=retired_but_present): {len(retired_present)} key(s) are recorded "
            "as withdrawn and are in the ledger anyway. A retirement is a statement that the "
            "proof is gone; if it came back, the retirement is now a false record of the "
            "file's contents and must be deleted, not left to agree with nothing."
        )
        for key in retired_present:
            print(f"  - {key}")
        return 1

    if vanished:
        print("")
        print(
            f"FAIL(condition=proof_vanished): {len(vanished)} proof(s) were committed to this "
            "ledger and are no longer in it, with no retirement record."
        )
        for key in vanished:
            print(f"  - {key}\n      first proven in {ever[key][:12]}; "
                  f"`git log -S {key!r} -- {LEDGER_REL}` finds the addition")
        print(
            "\n  A proof does not leave this ledger by being absent. If the form is genuinely "
            f"withdrawn, record it in {RETIRED.relative_to(REPO).as_posix()} with an owner, a "
            "date and a reason. If it is not, this is a deletion — most likely inside a merge "
            "conflict resolution, which is the one write neither `--check` nor the "
            "shrinking-write guard can see."
        )
        rc = 1
    else:
        print("")
        print(
            f"PASS: every key this ledger has ever carried is either present or retired with a "
            f"reason. Read as: nothing has been proven and then quietly unproven."
        )
        rc = 0

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "at": args.at or "worktree",
                    "ever_proven": n,
                    "present": m,
                    "retired": sorted(retired),
                    "vanished": vanished,
                    "retired_but_present": retired_present,
                    "not_yet_in_history": unhistoried,
                    "verdict": "PASS" if rc == 0 else "FAIL(condition)",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return rc


if __name__ == "__main__":
    sys.exit(screen())
