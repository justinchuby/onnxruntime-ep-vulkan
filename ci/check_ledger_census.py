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
any revision of the ledger reachable from HEAD (see DEFAULT_SCOPE). History is append-only in
the sense that matters here — a merge commit adds to it and cannot subtract from it. The
denominator is therefore held somewhere the failure mode cannot reach.

`--simplify` is deliberately NOT used when listing revisions (`git rev-list --full-history HEAD`), because
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

# WAS `--all`, AND `--all` WAS WRONG — found on 2026-08-04 by running this screen twice.
#
# The first run of the session was green. The second, minutes after a teammate pushed an
# in-progress branch, reported 28 VANISHED proofs "first proven in 18ddece" — a commit on
# `squad/mouse` that is not an ancestor of anything I have. The screen was convicting my
# branch for not containing somebody else's unmerged work, and the sentence it printed
# ("committed to this ledger and no longer in it") was false: they were never in this line
# of history at all.
#
# `HEAD` is the right denominator and loses nothing the screen exists for. The failure it
# was built to catch is a proof dropped inside a merge conflict resolution, and BOTH merge
# parents are reachable from HEAD, so `--full-history` still sees the side the deletion
# came from. What `--all` added was only refs that were never merged — which is not history,
# it is other people's drafts.
#
# This is the third time in one session that a framing choice, not a value test, was the
# defect: a symmetric value comparison convicted my own repair; an unresolvable boundary
# made every comparison vacuous; and now too WIDE a scope convicted a branch for a proof it
# never had. `--full-history` stays: that one is load-bearing and is separately asserted.
DEFAULT_SCOPE = "HEAD"
RETIRED = REPO / "evidence" / "proof_retired.json"
RETIRED_FIELDS = ("owner", "date", "reason")
FRAME_WITNESSES = ("source_digest", "toolchain")
REWITNESS = REPO / "evidence" / "proof_rewitness.json"
REWITNESS_FIELDS = ("revision", "field", "owner", "date", "reason")
WORKTREE = "WORKTREE"


def _git(args: list[str], repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, encoding="utf-8"
    )


def _keys_of(text: str) -> set[str]:
    """Every `key` in a ledger body. A line without one is a header, not an entry."""
    return set(_entries_of(text))


def _entries_of(text: str) -> dict[str, dict]:
    """key -> entry for a ledger body. A line without a key is a header, not an entry."""
    out: dict[str, dict] = {}
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
            out[key] = entry
    return out


def history_is_complete(repo: Path) -> tuple[bool, str]:
    """Can this checkout see the history the census takes its denominator from?

    DECLARED AND NOT GUARDED IN SESSION 18; GUARDED HERE. The census answers "was this
    key ever proven" by walking every revision that touched the ledger. In a shallow or
    partial clone that walk terminates at the graft boundary, the denominator silently
    shrinks to whatever was fetched, and a key deleted *before* the boundary reads as
    never-proven rather than as VANISHED. The screen would then print PASS — which is the
    exact failure mode this screen was written to make impossible, arriving through the
    clone depth instead of through a deletion.

    `--depth=1` is the CI default for most runners, so this is not a hypothetical: it is
    the configuration the screen is most likely to be run under. A check that cannot see
    its subject has not checked anything, so this returns an instrument error rather than
    a colour.
    """
    shallow = _git(["rev-parse", "--is-shallow-repository"], repo)
    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
        return False, (
            "this is a SHALLOW clone. `git rev-list --full-history` stops at the graft "
            "boundary, so 'ever proven' would mean 'proven since the fetch depth' and a "
            "proof deleted before it would read as never-existing. Re-run with a full "
            "clone (`git fetch --unshallow`)."
        )
    filt = _git(["config", "--get", "remote.origin.promisor"], repo)
    if filt.returncode == 0 and filt.stdout.strip() == "true":
        return False, (
            "this is a PARTIAL (promisor) clone. Blob fetches are lazy, so `git show "
            "<rev>:ledger` can fail per revision and those revisions drop silently out of "
            "the denominator. Re-run with a full clone."
        )
    if (repo / ".git" / "shallow").exists():
        return False, (
            "a .git/shallow graft file is present. The revision walk is truncated and the "
            "denominator is not 'every revision that ever touched the ledger'."
        )
    return True, ""


def witness_transitions(
    repo: Path, upto: str | None = None
) -> tuple[list[tuple[str, str, str, str, str]], list[str]]:
    """-> (transitions, walk_order). Each transition is (revision, key, field, old, new).

    FOUND BY THE LINUX LANE, 2026-08-04, ONE DAY AFTER THE REPAIR IT UNDID.

    115 of 121 entries had `source_digest` re-witnessed under the current hashing rule at
    `aea0147`, and `eee65aa` — an author regenerating the ledger from a base that predated
    that merge — put the withdrawn values back. Nothing saw it. No key went missing, so the
    key census and the loss invariant both reported 0; and on Windows a stale source digest
    with matching SPIR-V is `SOURCE-COSMETIC`, which forgives, so `--check` printed PASS
    over all 115. Only Linux declined, and it took the whole op suite with it.

    THE FIRST VERSION OF THIS ARM WAS WRONG AND THE WAY IT WAS WRONG IS THE POINT. It asked
    "did this value go back to something a later revision replaced", which is SYMMETRIC:
    it convicted my own repair exactly as loudly as it convicted the regression, because
    from the values alone the two are the same event seen from opposite ends. A screen
    cannot rank two alternating values without a frame. So the question asked here is not
    "which value is right" — it is "did the writer say they were moving it", which has an
    answer in the repository and needs no build and no platform.
    """
    scope = [upto] if upto else [DEFAULT_SCOPE]
    revs = _git(
        ["rev-list", "--full-history", "--topo-order", *scope, "--", LEDGER_REL], repo
    ).stdout.split()
    walk = list(reversed(revs))
    last: dict[tuple[str, str], str] = {}
    out: list[tuple[str, str, str, str, str]] = []
    for rev in walk:
        blob = _git(["show", f"{rev}:{LEDGER_REL}"], repo)
        if blob.returncode != 0:
            continue
        for key, entry in _entries_of(blob.stdout).items():
            for field in FRAME_WITNESSES:
                val = entry.get(field)
                if not val:
                    continue
                prev = last.get((key, field))
                if prev is not None and prev != val:
                    out.append((rev, key, field, prev, val))
                last[(key, field)] = val
    if upto is None:
        for key, entry in present_entries(repo, None).items():
            for field in FRAME_WITNESSES:
                val = entry.get(field)
                if not val:
                    continue
                prev = last.get((key, field))
                if prev is not None and prev != val:
                    out.append((WORKTREE, key, field, prev, val))
    return out, walk + [WORKTREE]


def load_rewitness(path: Path) -> dict:
    """The declarations that make a witness move legible.

    Same discipline as `open_reds.json` and `proof_retired.json`: an event that is allowed
    to happen silently is an event nobody can tell from its opposite. `screened_since`
    bounds the window so adopting this screen does not retroactively convict history that
    predates it — transitions older than that revision are counted and named, not failed.
    """
    if not path.is_file():
        return {"screened_since": "", "rewitness": []}
    doc = json.loads(path.read_text(encoding="utf-8"))
    for rec in doc.get("rewitness", []):
        missing = [f for f in REWITNESS_FIELDS if not rec.get(f)]
        if missing:
            raise ValueError(
                f"{path}: a rewitness record is missing {missing}. Moving a frame witness "
                "needs a name and a reason for the same argument accepting a red does: "
                "otherwise a regeneration from a stale base is indistinguishable from a "
                "deliberate re-witness, and on 2026-08-04 it was."
            )
    return doc


def screen_transitions(
    repo: Path,
    transitions: list[tuple[str, str, str, str, str]],
    walk: list[str],
    doc: dict,
    at_scope: str | None = None,
) -> tuple[list, list, int]:
    """-> (undeclared, stale_declarations, out_of_frame_count).

    The frame boundary is a POSITION IN THE WALK, not an ancestry test, and that is the
    second thing this arm got wrong before it got it right. `merge-base --is-ancestor` said
    the offending commit was out of frame — correctly, because it was authored on a side
    branch that forked BEFORE the boundary and only reached main through a later merge.
    That is not an edge case: it is exactly how the regression happened, so a boundary that
    excuses it excuses the only event the arm exists for.
    """
    since = (doc.get("screened_since") or "").strip()
    cut = -1
    if since:
        ok = _git(["rev-parse", "--verify", "--quiet", since + "^{commit}"], repo)
        if ok.returncode != 0:
            raise ValueError(
                f"screened_since={since!r} does not resolve to a commit in this repository. "
                "This is not a detail: an unresolvable boundary puts EVERY transition out of "
                "frame and the screen prints PASS having ruled on nothing. That is the same "
                "failure this whole file exists to prevent, arriving through a typo — and it "
                "happened on the first run, with a hand-written sha."
            )
        full = ok.stdout.strip()
        if full in walk:
            cut = walk.index(full)
        elif at_scope is None:
            raise ValueError(
                f"screened_since={since!r} resolves, but never touched {LEDGER_REL}, so it "
                "names no position in this walk and cannot bound it. Use a revision that "
                "wrote the ledger."
            )
        else:
            # A replay bounded at an EARLIER revision than the boundary. Everything in this
            # walk predates the rule, which is the honest answer, not an error: adopting a
            # screen must not retroactively convict history it could not have governed.
            cut = len(walk)
    pos = {rev: i for i, rev in enumerate(walk)}
    decls = doc.get("rewitness", [])
    in_frame, out_of_frame = [], 0
    for t in transitions:
        if cut >= 0 and pos.get(t[0], len(walk)) < cut:
            out_of_frame += 1
            continue
        in_frame.append(t)
    matched: set[int] = set()
    undeclared = []
    for rev, key, field, old, new in in_frame:
        hit = None
        for i, d in enumerate(decls):
            if d["field"] != field:
                continue
            dr = d["revision"]
            if dr == rev or (dr != WORKTREE and rev != WORKTREE and rev.startswith(dr)):
                hit = i
                break
        if hit is None:
            undeclared.append((rev, key, field, old, new))
        else:
            matched.add(hit)
    stale = []
    for i, d in enumerate(decls):
        if i in matched:
            continue
        dr = d["revision"]
        if dr == WORKTREE:
            in_scope = at_scope is None
        else:
            in_scope = any(r == dr or r.startswith(dr) for r in walk)
        if in_scope:
            stale.append(d)
    return undeclared, stale, out_of_frame


def accidental(doc: dict) -> list[dict]:
    """Declared moves whose declaration says they were NOT deliberate.

    A record is not a suppression. `deliberate: false` means the move happened and was an
    accident; the entry stays so the check keeps ruling on that revision and so the next
    reader can see that this class of accident has happened, how often, and what it cost.
    Printing the count on every run is the difference between a record and a silence.
    """
    return [d for d in doc.get("rewitness", []) if d.get("deliberate") is False]


def ever_proven(repo: Path, upto: str | None = None) -> dict[str, str]:
    """key -> the EARLIEST revision that carried it.

    `git rev-list --full-history HEAD` and not `--simplify-merges`/`--follow`: the removal
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
    scope = [upto] if upto else [DEFAULT_SCOPE]
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
    return set(present_entries(repo, rev))


def present_entries(repo: Path, rev: str | None) -> dict[str, dict]:
    if rev is None:
        path = repo / LEDGER_REL
        if not path.is_file():
            raise FileNotFoundError(
                f"no ledger at {path}; the comparison input is missing, so nothing was ruled "
                "on either way — that is UNOBSERVABLE, not PASS"
            )
        return _entries_of(path.read_text(encoding="utf-8"))
    blob = _git(["show", f"{rev}:{LEDGER_REL}"], repo)
    if blob.returncode != 0:
        raise FileNotFoundError(f"{rev} has no {LEDGER_REL}: {blob.stderr.strip()}")
    return _entries_of(blob.stdout)


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


KNOWN_LIMITS = {
    "shallow_clone_is_unobservable_not_clean": (
        "This census walks every revision that touched the ledger. In a shallow clone most "
        "of them are absent, so the walk is short and the answer 'nothing vanished' is a "
        "statement about the fetch depth, not about the project. history_is_complete() now "
        "GUARDS this — ERROR(instrument=truncated_history), exit 2 — but a guard is not a "
        "fix: in a shallow CI checkout the screen still rules on nothing, and a lane that "
        "only tests for exit 1 would read that as tolerable. See ci/open_reds.json "
        "known_limits id=ledger_census_is_unobservable_in_a_shallow_clone."
    ),
}


def _assert_known_limit(name: str) -> int:
    if name not in KNOWN_LIMITS:
        print(
            f"ERROR(instrument=unknown_limit): {name!r} is not a declared limit of this "
            f"screen. Declared: {sorted(KNOWN_LIMITS)}. A register entry accepting a limit "
            "the screen does not admit to is an acceptance of nothing."
        )
        return 2
    print(f"KNOWN-LIMIT {name}")
    print(f"  {KNOWN_LIMITS[name]}")
    print(
        "\nFAIL(condition=known_limit_still_open): declared, owned, bounded, and red on "
        "purpose. It goes green when the limit is closed, not when somebody stops looking."
    )
    return 1


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
    ap.add_argument(
        "--allow-shallow",
        action="store_true",
        help="run the census against a truncated history anyway. The result is not a "
             "census: 'ever proven' becomes 'proven since the fetch depth'. Exists so the "
             "guard is testable in both polarities, and it is reported in the frame line.",
    )
    ap.add_argument("--assert-known-limit", default="")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()

    if args.assert_known_limit:
        return _assert_known_limit(args.assert_known_limit)

    complete, why = history_is_complete(repo)
    if not complete and not args.allow_shallow:
        print(f"ERROR(instrument=truncated_history): {LEDGER_REL} census cannot run here.")
        print(f"  {why}")
        print(
            "  This is deliberately NOT a PASS and NOT a FAIL. The screen's whole claim is "
            "about what the history contains; with the history truncated it has no "
            "denominator, and an answer computed from a denominator that silently shrank is "
            "the failure this screen exists to prevent, arriving through the clone depth."
        )
        return 2

    retired = load_retired(repo / "evidence" / "proof_retired.json")
    if args.list_retired:
        if not retired:
            print("no proofs are retired.")
            return 0
        for key, rec in sorted(retired.items()):
            print(f"{key}\n    {rec['owner']} {rec['date']}: {rec['reason']}")
        return 0

    ever = ever_proven(repo, args.at)
    now = present_at(repo, args.at)
    rw_doc = load_rewitness(repo / "evidence" / "proof_rewitness.json")
    _trans, _walk = witness_transitions(repo, args.at)
    undeclared, stale_decl, out_of_frame = screen_transitions(
        repo, _trans, _walk, rw_doc, args.at
    )
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
    print(
        f"  frame witnesses {list(FRAME_WITNESSES)}: {len(undeclared)} UNDECLARED move(s), "
        f"{len(stale_decl)} declaration(s) matching nothing, {out_of_frame} out of frame "
        f"(before screened_since={rw_doc.get('screened_since') or '<unset>'})"
        + ("  [history truncated: --allow-shallow]" if not complete else "")
    )
    for d in accidental(rw_doc):
        if not any(r == d["revision"] or r.startswith(d["revision"]) for r in _walk):
            continue
        print(
            f"  DECLARED ACCIDENT: {d['revision'][:12]} moved `{d['field']}` on "
            f"{d.get('keys', '?')} entr(ies) and nobody meant to "
            f"[{d['owner']} {d['date']}]. Recorded, not excused; the check still rules on it."
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
    elif undeclared:
        print("")
        print(
            f"FAIL(condition=undeclared_witness_move): {len(undeclared)} §8.9.19 frame "
            "witness(es) moved with no record of anyone deciding to move them. No key went "
            "missing, so the census above and the loss invariant both read clean; the loss "
            "is INSIDE surviving entries."
        )
        seen_rev: dict[tuple[str, str], int] = {}
        for rev, key, field, old, new in undeclared:
            seen_rev[(rev, field)] = seen_rev.get((rev, field), 0) + 1
        for (rev, field), cnt in sorted(seen_rev.items(), key=lambda x: -x[1]):
            sample = next(t for t in undeclared if t[0] == rev and t[2] == field)
            print(
                f"  - {rev[:12]}: {field} moved on {cnt} entr(ies), "
                f"e.g. {sample[1]}\n      {sample[3]!r} -> {sample[4]!r}"
            )
        print(
            f"\n  A frame witness is the field that decides, on a SECOND platform, whether a "
            f"difference was the compiler or the kernel. Moving one is legitimate and routine "
            f"(--reprove, --backfill-frame --rewitness-source); moving one WITHOUT SAYING SO "
            f"is not, because a regeneration from a stale base then looks exactly like a "
            f"deliberate re-witness. On 2026-08-04 it looked exactly like one on 115 of 121 "
            f"entries, Windows forgave every one of them as SOURCE-COSMETIC, and the Linux "
            f"lane declined all 115. Declare the move in "
            f"{REWITNESS.relative_to(REPO).as_posix()} with an owner, a date and a reason — "
            f"or, if it was not deliberate, repair it."
        )
        rc = 1
    elif stale_decl:
        print("")
        print(
            f"FAIL(condition=stale_rewitness_declaration): {len(stale_decl)} declaration(s) "
            "in " + REWITNESS.relative_to(REPO).as_posix() + " match no witness move in the "
            "history. Good news, and the same arm as `stale_acceptance`: a declaration that "
            "has stopped describing anything is a record of a decision nobody can check, and "
            "deleting it is what stops this register rotting."
        )
        for d in stale_decl:
            print(f"  - {d['revision']} {d['field']} [{d['owner']} {d['date']}: {d['reason']}]")
        rc = 1
    else:
        print("")
        print(
            f"PASS: every key this ledger has ever carried is either present or retired with a "
            f"reason, and no §8.9.19 frame witness has moved without a record of someone "
            f"deciding to move it. Read as: nothing has been proven and then quietly "
            f"unproven, and nothing has been re-witnessed and then quietly un-re-witnessed."
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
                    "witnesses_undeclared_moves": [
                        {"revision": r, "key": k2, "field": f, "from": o, "to": nv}
                        for r, k2, f, o, nv in undeclared
                    ],
                    "rewitness_declarations_matching_nothing": stale_decl,
                    "witness_moves_out_of_frame": out_of_frame,
                    "history_complete": complete,
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
    try:
        sys.exit(screen())
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR(instrument=register_unusable): {exc}")
        print(
            "  Exit 2, deliberately distinct from 1. Nothing was ruled on either way — that "
            "is UNOBSERVABLE, not PASS and not FAIL."
        )
        sys.exit(2)


