"""Land THIS branch three ways, for real, and screen the census on each landing.

WHY THIS EXISTS, AND WHY IT IS NOT ONE LANDING ANY MORE
======================================================
Round 1 of PR #44 asked one question — does the declaration survive the squash that lands
it — and answered it by building the squash. That was right, and it was not enough. PR #53
was green on its own head, green under a merge commit, and RED under a squash, and the red
would have arrived on `main`, after the merge, in a job nobody was watching, because
`ci/check_ledger_census.py` runs on `push: branches: [main]`. A screen whose colour depends
on which merge button a human presses is not a screen; it is a coin.

So the acceptance question is not "does the head pass", and not even "does the squash
pass". It is: **does every landing this repository can produce pass, and do the planted
defects fail on every one of them.** This script answers exactly that, against the real
repository, with no network and no GitHub.

    squash   `gh pr merge --squash`   the MERGE of the branch into main, committed as one
                                      brand-new commit with main as its only parent; every
                                      branch commit un-ancestored. Built with `git merge
                                      --squash`, not an overlay of the branch's paths — see
                                      `_land` and issue #60: when main has moved inside the
                                      merge window those are different trees, and only this
                                      one is the tree GitHub actually lands.
    merge    `gh pr merge --merge`    a real two-parent commit; branch commits reachable
    rebase   `gh pr merge --rebase`   each non-merge commit replayed under a new sha

and, because a declaration must keep being true after it lands, a fourth column:

    replay   one more unrelated commit on top of the landing, census re-run

CONTROLS, BECAUSE THREE GREENS PROVE NOTHING ON THEIR OWN
=========================================================
A green matrix is compatible with a screen that has stopped reading the register at all. So
every landing is also run against two deliberately broken registers:

  * `v2-branch-sha`  — the record rewritten in `rewitness/2` naming the branch commit that
    actually made the change, which is the best a v2 author could do. It must FAIL on the
    squash and the rebase (the sha is erased) and it is allowed to pass on the merge — that
    asymmetry IS the defect this change removes, printed rather than argued.
  * `v3-wrong-new`   — the same v3 record with one character of the cause's `new` content
    id changed. It must FAIL on EVERY landing: a cause the trees do not corroborate is not
    a cause, whichever way the change arrives.

WHICH COMMIT IS THE BRANCH HEAD — `--head`, AND WHY IT IS NOT `git rev-parse HEAD`
==================================================================================
On a `pull_request` event GitHub checks out `refs/pull/N/merge`: a synthetic two-parent
commit merging the PR head into the CURRENT tip of the base. `git rev-parse HEAD` on that
checkout is the MERGE, whose first parent IS the base — so `merge-base(base, HEAD)` collapses
to the base itself and "how many paths has the base moved since the merge base" is
identically **zero on the one event this instrument exists for**. Every downstream
consequence follows from that one number: the merge-window `lost` non-vacuity guard in
`_land` evaluates an empty path set, and this script prints *"this run cannot exhibit the
merge-window collision"* directly above the merge-window collision.

`ci/check_landing_simulation.py` already resolves the real head (`HEAD^2` on that shape) to
make its own rule R2 non-vacuous. It now PASSES that commit here with `--head`, and this
script uses it for the merge base, the `pr` branch it builds, the squash tree, the
non-vacuity guard and every printed diagnostic. A gate and the engine it invokes must agree
about which commit is the branch head; when they do not, the engine is measuring a different
branch than the gate decided about.

`--head` that does not resolve is `ERROR(instrument=head_unavailable)`, exit 2 — the same
answer as an unresolvable base and for the same reason: the subject could not be observed.

Usage:
    python ci/simulate_squash_rewitness.py                 # all landings + controls
    python ci/simulate_squash_rewitness.py --landing squash
    python ci/simulate_squash_rewitness.py --base origin/main
    python ci/simulate_squash_rewitness.py --head HEAD^2   # a pull_request merge ref

IN THE LANE: this script is the ENGINE, not the step. `ci/check_landing_simulation.py`
decides whether a given branch can exhibit a landing-dependent verdict at all and runs this
when it can; `ci/negative_control_landing_simulation.py` asserts both polarities of that
decision against the real `q_gemv.comp` collision from issue #60.
"""
import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PY = sys.executable
LANDINGS = ("squash", "merge", "rebase")


class BaseUnavailable(RuntimeError):
    """The landing target could not be resolved or fetched.

    A DISTINCT TYPE BECAUSE THE ANSWER IS DISTINCT. Every other failure in here is a verdict
    about the branch; this one is a verdict about the checkout, and the two must not share an
    exit code. `ci/check_landing_simulation.py` turns it into `ERROR(instrument=...)` and
    exit 2 — never a skip, because a simulation that quietly does not run on the day
    `origin/main` was not fetched is green for the same reason it is useless.
    """


class HeadUnavailable(RuntimeError):
    """The branch head could not be resolved.

    Same class of answer as `BaseUnavailable`, about the other end of the landing. It exists
    separately because `--head` is now passed IN by the gate rather than read off the
    checkout, so "the caller named a commit this repository does not have" is a distinct and
    newly reachable failure — and the one shape that must never degrade into silently
    simulating `git rev-parse HEAD` instead, which on a `pull_request` merge ref is the
    vacuous simulation this argument exists to prevent.
    """


def _force_writable(func, path, _exc):
    """Git marks objects and packs read-only, and Windows honours that on unlink.

    `rmtree(..., ignore_errors=True)` therefore leaves a multi-hundred-megabyte clone
    behind next to the repository every time this runs, silently. Clear the read-only
    bit and retry, and let a failure that is not that one be seen.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def git(args, cwd):
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="sim", GIT_AUTHOR_EMAIL="sim@sim",
        GIT_COMMITTER_NAME="sim", GIT_COMMITTER_EMAIL="sim@sim",
    )
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )


def census(clone: Path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run(
        [PY, str(clone / "ci" / "check_ledger_census.py"), "--repo", str(clone)],
        cwd=str(clone), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def condition(out: str) -> str:
    """The verdict token, short enough to tabulate: `PASS`, or the condition that fired."""
    for line in out.splitlines():
        if line.startswith(("FAIL", "ERROR")):
            head = line.split(":")[0].strip()
            inner = head[head.find("(") + 1: head.rfind(")")] if "(" in head else head
            return inner.split("=", 1)[-1] if "=" in inner else head
    return "PASS" if "PASS" in out else "?"


def _prepare(tmp: Path, base: str, head: str = "HEAD"):
    """A clone of this repository holding `base` and a `pr` branch with THIS tree on it.

    The PR branch keeps the real commits (so the merge landing has real parents to reach)
    and, if the working tree differs from HEAD, gains one more commit carrying it — the
    register is edited in the same change as the ledger, so a simulation that could only
    read committed state would answer one commit too late, after the point where a bad
    declaration can still be fixed without a follow-up.

    `head` IS AN ARGUMENT, NOT `git rev-parse HEAD`. On a `pull_request` event the checkout
    is `refs/pull/N/merge`, whose first parent is the base; building `pr` from it makes
    `merge-base(base, pr)` the base itself and every merge-window measurement below vacuous.
    `ci/check_landing_simulation.py` resolves the real head and passes it here.

    THE WORKING-TREE OVERLAY IS CONDITIONAL ON THE CHECKOUT BEING THAT HEAD. When `head` is
    NOT the checked-out commit — the merge-ref case — the files on disk are the MERGE's
    tree, which already carries the base's side. Copying them onto the branch head would
    attribute the base's concurrent edit to the branch, which is the exact content confusion
    the squash landing exists to expose, injected into the fixture. So the overlay runs only
    when the checkout is the head being simulated, and says so when it declines.
    """
    tmp.mkdir(parents=True, exist_ok=True)
    clone = tmp / "clone"
    checkout = git(["rev-parse", "HEAD"], HERE).stdout.strip()
    head_sha = git(["rev-parse", f"{head}^{{commit}}"], HERE).stdout.strip()
    if not head_sha:
        raise HeadUnavailable(
            f"head {head!r} does not resolve to a commit in {HERE}. The branch head is what "
            "every landing here is built FROM, so this is an instrument error and never a "
            "fallback to the checkout: on a pull_request merge ref the checkout is not the "
            "branch, and simulating it instead would be a vacuous green."
        )
    # Resolve the base HERE, not in the clone: cloning rewrites `origin` to point at this
    # worktree, so `origin/main` inside the clone would silently mean "this checkout's local
    # main", which can be many commits behind the branch this PR would really land on. A
    # simulation against the wrong base answers a question nobody asked.
    base_sha = git(["rev-parse", f"{base}^{{commit}}"], HERE).stdout.strip()
    if not base_sha:
        raise BaseUnavailable(
            f"base {base!r} does not resolve to a commit in {HERE}. A landing simulation "
            "with no landing target is not a weaker simulation, it is none — so this is an "
            "instrument error and never a skip. Fetch the base ref (`git fetch --no-tags "
            "origin main`, with NO --depth) and re-run."
        )
    r = git(["clone", "-q", "--no-local", str(HERE), str(clone)], HERE.parent)
    if r.returncode:
        raise SystemExit(f"cannot clone: {r.stderr}")
    git(["fetch", "-q", "origin", head_sha, base_sha], clone)
    if git(["cat-file", "-e", f"{base_sha}^{{commit}}"], clone).returncode:
        raise BaseUnavailable(f"base {base_sha[:12]} did not arrive in the clone")
    if git(["cat-file", "-e", f"{head_sha}^{{commit}}"], clone).returncode:
        raise HeadUnavailable(f"head {head_sha[:12]} did not arrive in the clone")
    git(["checkout", "-q", "-B", "pr", head_sha], clone)

    if head_sha != checkout:
        print(
            f"  the checkout {checkout[:12]} is not the simulated head {head_sha[:12]} "
            "(a pull_request merge ref, or an explicit --head), so the working tree is NOT "
            "overlaid: those files are the merge's content and copying them would credit "
            "the base's side of the collision to this branch"
        )
        return clone, base_sha, head_sha

    tracked = [p for p in git(["ls-files"], HERE).stdout.splitlines() if p]
    for rel in git(["ls-files"], clone).stdout.splitlines():
        if rel:
            (clone / rel).unlink(missing_ok=True)
    for rel in tracked:
        src = HERE / rel
        if not src.is_file():
            continue
        dst = clone / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    git(["add", "-A"], clone)
    if git(["diff", "--cached", "--quiet"], clone).returncode != 0:
        git(["commit", "-q", "-m", "working tree, as it would be committed"], clone)
    pr = git(["rev-parse", "HEAD"], clone).stdout.strip()
    return clone, base_sha, pr


def _fanout(source: Path, dest: Path, base_sha: str, pr: str) -> Path:
    """A per-landing working copy of `source`, made the cheap way.

    `_prepare` pays for one `--no-local` clone of the real repository and one file-by-file
    copy of the working tree. Doing that three times — once per landing — was two thirds of
    this script's wall clock for no added truth, because the three landings differ only in
    what is built AFTER the clone. A `--local` clone hardlinks the object store instead of
    re-transferring it, which is safe here for the reason it is always safe: git objects are
    immutable, and every landing only ever ADDS objects.
    """
    r = git(["clone", "-q", "--local", "--no-hardlinks", str(source), str(dest)], dest.parent)
    if r.returncode:
        raise SystemExit(f"cannot fan out the simulation clone: {r.stderr}")
    for sha in (base_sha, pr):
        if git(["cat-file", "-e", f"{sha}^{{commit}}"], dest).returncode:
            git(["fetch", "-q", "origin", sha], dest)
            if git(["cat-file", "-e", f"{sha}^{{commit}}"], dest).returncode:
                raise BaseUnavailable(f"{sha[:12]} did not arrive in the landing clone")
    git(["checkout", "-q", "-B", "pr", pr], dest)
    return dest


def _land(clone: Path, base: str, pr: str, mode: str, prune_refs: bool = False):
    """Put `pr`'s content onto `base` as `main`, the way GitHub's button of that name does.

    -> (landing sha, "") or ("", why it could not be built). A simulation that silently
    degraded into a different shape than the one it names would be worse than no
    simulation, so a failure here is reported, never worked around.

    ISSUE #60 — WHY THE SQUASH IS A MERGE AND NOT A CHECKOUT. This used to build the squash
    with `git checkout <pr> -- .`, which overlays the PR's paths onto the base's tree. When
    the base has not moved since the merge base those are the same tree and the shortcut was
    invisible. When the base HAS moved — which is the entire merge window this simulation
    exists to cover — they are not: a file edited on BOTH sides gets the PR's version and the
    base's edit is silently dropped. GitHub does the opposite. `gh pr merge --squash` commits
    the MERGE RESULT with the base as its single parent, so a concurrent edit to a declared
    `caused_by_content` path is present in the landed tree, the declared `new` no longer
    matches it, and the census is red. Reproduced on this repository: base `bb09871` plus a
    one-line edit to `rust/shaders/glsl/templates/q_gemv.comp`, PR `8f12b32` — real squash
    FAIL(condition=uncorroborated_rewitness_cause), overlay squash PASS. The shortcut printed
    green on the exact defect the simulation is named after.

    `prune_refs` DEFAULTS TO FALSE, AND THAT IS THE SAFE DEFAULT. Deleting the `pr` and
    `replay` branches is needed in the fan-out clones — while those refs exist the census's
    revision walk can still reach the un-ancestored branch commits, so the squash would be
    screened as if it were the merge. It is also a destructive edit to whatever repository
    this function is handed, which is why the caller has to ask for it. Production asks; the
    tests that build a landing in a fixture repository do not, and assert their own refs
    survive. This function also refuses outright on a dirty working tree: a landing cannot
    be built on top of uncommitted work, and clobbering a caller's uncommitted work is the
    exact accident the default guards.
    """
    dirty = git(["status", "--porcelain", "-z", "--untracked-files=no"], clone).stdout
    if dirty.strip("\0").strip():
        return "", (
            f"refusing to build the {mode} landing in {clone}: the working tree has "
            "uncommitted changes, and this function checks branches out over it. A landing "
            "is built in a scratch clone, never in a checkout somebody is using."
        )
    git(["checkout", "-q", "-B", "main", base], clone)
    if mode == "squash":
        r = git(["merge", "-q", "--squash", pr], clone)
        if r.returncode:
            git(["merge", "--abort"], clone)
            git(["reset", "-q", "--hard", base], clone)
            return "", (
                "the squash landing does not merge cleanly, so GitHub could not build it "
                f"either: {r.stdout}{r.stderr}"
            )
        git(["add", "-A"], clone)
        if git(["diff", "--cached", "--quiet"], clone).returncode == 0:
            return "", "the squash landing is empty: this branch changes nothing against the base"
        git(["commit", "-q", "-m", "squash landing (new sha, branch history erased)"], clone)
    elif mode == "merge":
        r = git(["merge", "-q", "--no-ff", "--no-edit", "-m", "merge landing", pr], clone)
        if r.returncode:
            return "", f"merge failed: {r.stdout}{r.stderr}"
    elif mode == "rebase":
        mb = git(["merge-base", base, pr], clone).stdout.strip()
        git(["checkout", "-q", "-B", "replay", pr], clone)
        r = git(["rebase", "--onto", base, mb, "replay"], clone)
        if r.returncode:
            git(["rebase", "--abort"], clone)
            return "", f"rebase failed: {r.stdout}{r.stderr}"
        replayed = git(["rev-parse", "HEAD"], clone).stdout.strip()
        git(["checkout", "-q", "-B", "main", replayed], clone)
    else:
        return "", f"unknown landing {mode!r}"
    landed = git(["rev-parse", "HEAD"], clone).stdout.strip()
    git(["checkout", "-q", "main"], clone)
    if mode != "merge":
        if git(["merge-base", "--is-ancestor", pr, landed], clone).returncode == 0:
            return "", f"the {mode} landing still has the PR head as an ancestor"
        why = _refuse_if_vacuous(clone, base, pr, landed, mode)
        if why:
            return "", why
        if prune_refs:
            git(["branch", "-q", "-D", "pr"], clone)
            git(["branch", "-q", "-D", "replay"], clone)
    return landed, ""


def lost_base_edits(clone: Path, base: str, pr: str, landed: str) -> list[str]:
    """Paths the base moved since the merge base that `landed` does NOT carry. -> sorted.

    NON-VACUITY FOR THE MERGE WINDOW (issue #60). If the base moved since the merge base, the
    landing MUST carry those base-side edits — that is the whole difference between the tree
    GitHub builds and the branch's own tree, and it is what the overlay squash used to throw
    away. Measured by CONTENT, on the paths that can show it, so a future rewrite of `_land`
    cannot quietly go back to an overlay.

    A path counts as lost only when the base changed it AND the branch did not touch it: if
    both sides edited it, a landing that resolves to the branch's version is a legitimate
    merge result, not a dropped edit.

    ALL FOUR REVISIONS COME FROM THE CALLER. `pr` is the head the gate resolved, not
    `git rev-parse HEAD` — on a `pull_request` merge ref those differ, `merge-base(base, pr)`
    collapses to the base, `moved_on_base` is empty, and this function then returns `[]` for
    every landing on the one event it exists for. That was the defect; the head is now
    plumbed through so this set is real.
    """
    mb = git(["merge-base", base, pr], clone).stdout.strip()
    if not mb:
        return []
    return [
        rel
        for rel in sorted(_changed_paths(clone, mb, base))
        if git(["diff", "--quiet", base, landed, "--", rel], clone).returncode != 0
        and git(["diff", "--quiet", mb, pr, "--", rel], clone).returncode == 0
    ]


def _refuse_if_vacuous(clone: Path, base: str, pr: str, landed: str, mode: str) -> str:
    """-> "" if `landed` is a landing GitHub could produce, else why it is not.

    Stated as a REFUSAL rather than a colour: a simulation of a tree nobody will ever have
    proves nothing, so it is not screened at all.
    """
    lost = lost_base_edits(clone, base, pr, landed)
    if not lost:
        return ""
    return (
        f"the {mode} landing dropped {len(lost)} path(s) the base changed and this "
        f"branch did not touch, e.g. {lost[:3]}. That is not a landing GitHub can "
        "produce, and a simulation of a tree nobody will ever have proves nothing"
    )


def _changed_paths(clone: Path, a: str, b: str) -> set[str]:
    r = git(["diff", "--name-only", f"{a}..{b}"], clone)
    return {line for line in r.stdout.splitlines() if line}


def _v3_record(doc: dict):
    for i, rec in enumerate(doc.get("rewitness", [])):
        if rec.get("schema") == "rewitness/3":
            return i, rec
    return None, None


def _write_register(clone: Path, doc: dict, message: str) -> None:
    (clone / "evidence" / "proof_rewitness.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    git(["add", "-A"], clone)
    git(["commit", "-q", "-m", message], clone)


def _control_v2_branch_sha(doc: dict, branch_sha: str):
    """The v3 record rewritten as the best a `rewitness/2` author could have written it."""
    i, rec = _v3_record(doc)
    if i is None:
        return None
    out = json.loads(json.dumps(doc))
    v2 = {k: v for k, v in rec.items() if k not in ("schema", "caused_by_content")}
    v2["schema"] = "rewitness/2"
    v2["caused_by"] = branch_sha
    out["rewitness"][i] = v2
    return out


def _control_v3_wrong_new(doc: dict):
    i, _ = _v3_record(doc)
    if i is None:
        return None
    out = json.loads(json.dumps(doc))
    p = out["rewitness"][i]["caused_by_content"]["paths"][0]
    p["new"] = ("0" if p["new"][0] != "0" else "1") + p["new"][1:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landing", default="all", choices=(*LANDINGS, "all"))
    ap.add_argument("--base", default="origin/main")
    ap.add_argument(
        "--head", default="HEAD",
        help="the BRANCH head to land. Defaults to the checkout, which is right on a direct "
             "push and WRONG on a `pull_request` event, where the checkout is "
             "`refs/pull/N/merge` and its first parent is the base — making the merge base "
             "the base itself and every merge-window measurement here vacuous. "
             "ci/check_landing_simulation.py resolves it (HEAD^2 on that shape) and passes "
             "it. Unresolvable is exit 2, never a fallback to the checkout.",
    )
    ap.add_argument(
        "--no-replay",
        action="store_true",
        help="skip the 'one commit later' re-screen. For a CALLER that is asserting one "
             "polarity of this script itself; never for the lane, where a declaration that "
             "stops being true the commit after it lands is a defect worth a census.",
    )
    ap.add_argument(
        "--no-controls",
        action="store_true",
        help="skip the two broken-register controls. Same caveat as --no-replay: this exists "
             "so ci/negative_control_landing_simulation.py can exercise the landing shape "
             "cheaply, and a lane run that used it would be reporting three greens with "
             "nothing to say they are not the greens of a screen that stopped reading.",
    )
    args = ap.parse_args()
    modes = LANDINGS if args.landing == "all" else (args.landing,)

    head = git(["rev-parse", f"{args.head}^{{commit}}"], HERE).stdout.strip()
    checkout = git(["rev-parse", "HEAD"], HERE).stdout.strip()
    shallow = git(["rev-parse", "--is-shallow-repository"], HERE).stdout.strip()
    if shallow == "true" or (HERE / ".git" / "shallow").exists():
        print("SIM: ERROR(instrument=shallow_checkout)")
        print(
            "  This checkout is SHALLOW. Every landing is built by merging or replaying "
            "against a base, and `ci/check_ledger_census.py` then takes its denominator from "
            "the full revision walk — both are truncated at the graft boundary, so the matrix "
            "would be three greens computed from a history that is not this repository's. "
            "Check out with `fetch-depth: 0` (and fetch the base with NO --depth: a "
            "depth-limited fetch marks its own graft point even in a full clone — issue #28)."
        )
        return 2
    if not head:
        print("SIM: ERROR(instrument=head_unavailable)")
        print(
            f"  --head {args.head!r} does not resolve to a commit in {HERE}. This never falls "
            "back to the checkout: on a `pull_request` event the checkout is the synthetic "
            "merge ref, whose first parent is the base, and simulating it would report zero "
            "paths moved on the base and 'this run cannot exhibit the merge-window collision' "
            "on the one event the collision exists for."
        )
        return 2
    print(
        f"branch head {head[:12]}  base {args.base}"
        + ("" if head == checkout else f"  (checkout is {checkout[:12]}, --head {args.head})")
    )

    rows = []
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="landsim_", dir=str(HERE.parent)))
    try:
        try:
            source, base_sha, pr = _prepare(tmp / "source", args.base, head)
        except BaseUnavailable as exc:
            print("SIM: ERROR(instrument=base_unavailable)")
            print(f"  {exc}")
            return 2
        except HeadUnavailable as exc:
            print("SIM: ERROR(instrument=head_unavailable)")
            print(f"  {exc}")
            return 2
        mb = git(["merge-base", base_sha, pr], source).stdout.strip()
        ahead = len(_changed_paths(source, mb, base_sha)) if mb else 0
        print(
            f"  base {base_sha[:12]}, pr {pr[:12]}, merge-base {mb[:12] if mb else '(none)'}; "
            f"the base has moved {ahead} path(s) since the merge base"
        )
        if not ahead:
            print(
                "  (nothing has landed on the base since this branch left it, so the squash "
                "tree and the branch tree are the same content — this run cannot exhibit the "
                "merge-window collision, only the landing-shape properties. This sentence is "
                f"a measurement of merge-base {mb[:12] if mb else '(none)'}..{base_sha[:12]} "
                f"with the branch head {pr[:12]}, NOT of the checkout: issue #60's second "
                "blocker was printing it from a merge ref, where it is vacuously true.)"
            )

        for mode in modes:
            clone = _fanout(source, tmp / mode, base_sha, pr)
            print(f"\n-- {mode}: base {base_sha[:12]}, pr {pr[:12]}")
            landed, why = _land(clone, base_sha, pr, mode, prune_refs=True)
            if not landed:
                print(f"   SIM INVALID: {why}")
                rows.append((mode, "SIM-INVALID", "-", "-", "-"))
                ok = False
                continue
            print(f"   landing commit {landed[:12]}")

            rc, out = census(clone)
            real = condition(out)
            print(f"   register as written : exit={rc} {real}")
            for line in out.splitlines():
                if "frame witnesses" in line or line.startswith(("FAIL", "ERROR")):
                    print("     " + line.strip()[:170])
            if rc != 0:
                ok = False
                for line in out.splitlines():
                    if line.lstrip().startswith("- "):
                        print("     " + line.strip()[:220])

            rc_replay, replay = 0, "-"
            if not args.no_replay:
                # the declaration must keep being true AFTER it lands, not only at the landing
                readme = clone / "README.md"
                readme.write_text(
                    readme.read_text(encoding="utf-8") + "\n<!-- a later, unrelated commit -->\n",
                    encoding="utf-8",
                )
                git(["add", "-A"], clone)
                git(["commit", "-q", "-m", "a later, unrelated commit"], clone)
                rc_replay, out_replay = census(clone)
                replay = condition(out_replay)
                print(f"   one commit later    : exit={rc_replay} {replay}")
                if rc_replay != 0:
                    ok = False
                    for line in out_replay.splitlines():
                        if line.startswith(("FAIL", "ERROR")):
                            print("     " + line.strip()[:170])

            if args.no_controls:
                rows.append((mode, f"{real} ({rc})", f"{replay} ({rc_replay})", "-", "-"))
                continue

            doc = json.loads(
                (clone / "evidence" / "proof_rewitness.json").read_text(encoding="utf-8")
            )
            if _v3_record(doc)[0] is None:
                print("   no rewitness/3 record present; the controls cannot run")
                return 2

            # control 1: the v2 form, naming the real branch commit that made the change.
            _write_register(clone, _control_v2_branch_sha(doc, pr),
                            "control: v2 naming the branch commit")
            rc_v2, out_v2 = census(clone)
            c_v2 = condition(out_v2)
            print(f"   ctl v2-branch-sha   : exit={rc_v2} {c_v2}")

            # control 2: the same v3 record with a cause the tree does not corroborate.
            _write_register(clone, _control_v3_wrong_new(doc),
                            "control: v3 with a wrong new content id")
            rc_w, out_w = census(clone)
            c_w = condition(out_w)
            print(f"   ctl v3-wrong-new    : exit={rc_w} {c_w}")
            if rc_w == 0:
                print("     CONTROL DID NOT FIRE: an uncorroborated cause passed")
                ok = False

            rows.append((mode, f"{real} ({rc})", f"{replay} ({rc_replay})",
                         f"{c_v2} ({rc_v2})", f"{c_w} ({rc_w})"))

            if mode in ("squash", "rebase") and rc_v2 == 0:
                print(f"     CONTROL DID NOT FIRE: a branch-only v2 `caused_by` survived a "
                      f"{mode}, which is the defect this schema removes")
                ok = False

        print("\nLANDING MATRIX")
        print("  {:<8} {:<26} {:<26} {:<36} {:<36}".format(
            "landing", "as written", "one commit later", "ctl v2-branch-sha",
            "ctl v3-wrong-new"))
        for row in rows:
            print("  {:<8} {:<26} {:<26} {:<36} {:<36}".format(*row))
        print(
            "\n  Read the third column as the reason this schema exists: the SAME correct\n"
            "  declaration, written the only way rewitness/2 allows, is landing-strategy\n"
            "  dependent. The fourth is the reason to believe the first: a cause the trees\n"
            "  do not corroborate fails on every landing."
        )
        if ok:
            print("\nPASS: rewitness/3 survives every landing; both controls fire.")
            return 0
        print("\nFAIL")
        return 1
    finally:
        shutil.rmtree(tmp, onerror=_force_writable)
        if tmp.exists():
            print(f"WARNING: could not remove the simulation clone at {tmp}")


if __name__ == "__main__":
    sys.exit(main())
