"""Squash-merge simulation for the cleanroom index-URL privacy control (issue #55).

PR #57 was rejected twice. The second rejection was not about the redaction at all: the
negative control fetched the rejected module with `git show d5bab5d:...`, and this
repository squash-merges and deletes branches. The moment the fix landed, `d5bab5d` would
be unreachable, the REPLAYED arms -- the only evidence in the control not written by the
same hand as the tests -- would stop firing, and the control would be a control of nothing.

That claim is checkable now rather than after the fact. This clones the repository, resets
to origin/main, applies this branch's TREE as one brand-new commit (exactly what
`gh pr merge --squash` produces), deletes every other ref, expires the reflog and prunes,
so that the branch commits are not merely non-ancestors -- their objects are GONE. Then it
runs the privacy suite and the negative control in that tree and demands both green.

A simulation that only demanded green would prove nothing, so it also runs two A/B arms in
the same landed tree:

  * the module replaced by a rejected head's fixture bytes  -> the control must go RED;
  * one fixture byte flipped                                -> the control must REFUSE it.

Run: python ci/simulate_squash_cleanroom_redaction.py
"""
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
FIXTURES = "ci/fixtures/cleanroom-redaction"
MODULE = "python/verify_cleanroom.py"
CONTROL = "ci/negative_control_cleanroom_redaction.py"
SUITE = "tests/packaging/test_verify_cleanroom_redaction.py"


def _force_writable(func, path, _exc):
    """Git marks objects and packs read-only, and Windows honours that on unlink."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def run(args, cwd):
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)


def _land(clone: Path, base: str) -> str:
    """Apply this worktree's tracked files onto *base* as a single new commit."""
    git(["checkout", "-q", "--detach", base], clone)
    for rel in git(["ls-files"], clone).stdout.splitlines():
        if rel:
            (clone / rel).unlink(missing_ok=True)
    for rel in git(["ls-files"], HERE).stdout.splitlines():
        src = HERE / rel
        if not rel or not src.is_file():
            continue
        dst = clone / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    git(["add", "-A"], clone)
    git(["-c", "user.name=sim", "-c", "user.email=sim@sim", "commit", "-q",
         "-m", "PowerShell exit-status verdict and index-URL redaction (#55)\n\n"
               "squash simulation of PR #57"], clone)
    git(["branch", "-q", "-f", "main"], clone)
    git(["checkout", "-q", "main"], clone)
    return git(["rev-parse", "HEAD"], clone).stdout.strip()


def _forget_everything_else(clone: Path) -> None:
    """Delete every ref but `main`, expire the reflog and prune. After this the branch
    objects are not just unreachable, they are collected -- which is the state a reviewer
    pulling `main` a week after the merge is actually in."""
    for line in git(["for-each-ref", "--format=%(refname)"], clone).stdout.splitlines():
        if line.strip() and line.strip() != "refs/heads/main":
            git(["update-ref", "-d", line.strip()], clone)
    git(["remote", "remove", "origin"], clone)
    git(["reflog", "expire", "--expire=now", "--expire-unreachable=now", "--all"], clone)
    git(["gc", "--prune=now", "-q"], clone)


def main() -> int:
    head = git(["rev-parse", "HEAD"], HERE).stdout.strip()
    base = git(["rev-parse", "origin/main"], HERE).stdout.strip()
    print(f"branch head {head[:12]}   base origin/main {base[:12]}")

    manifest = json.loads((HERE / FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    rejected = [f["origin"]["ref_at_authoring"] for f in manifest["fixtures"]]
    print(f"fixture origins (provenance only): {', '.join(c[:12] for c in rejected)}")

    tmp = Path(tempfile.mkdtemp(prefix="squashsim55_", dir=str(HERE.parent)))
    failures = []
    try:
        clone = tmp / "clone"
        r = git(["clone", "-q", "--no-local", str(HERE), str(clone)], HERE.parent)
        if r.returncode:
            print(r.stderr)
            return 2
        git(["fetch", "-q", "origin", head, base], clone)
        squash = _land(clone, base)
        if git(["merge-base", "--is-ancestor", head, squash], clone).returncode == 0:
            print("SIM INVALID: the branch head is still an ancestor; not a squash")
            return 2
        _forget_everything_else(clone)
        print(f"squash commit {squash[:12]} on main; branch deleted, reflog expired, gc'd")

        # The premise of the whole exercise: the refs the OLD control depended on are gone.
        for sha in [head, *rejected]:
            gone = git(["cat-file", "-e", f"{sha}^{{commit}}"], clone).returncode != 0
            print(f"   {sha[:12]} reachable after landing? {'no' if gone else 'YES'}")
            if not gone:
                failures.append(f"{sha[:12]} survived the squash; the simulation is weak")

        # 1. the fixtures themselves must have been carried across, byte for byte
        for name in ("verify_cleanroom.rejected-r1.pysrc",
                     "verify_cleanroom.rejected-r2.pysrc", "manifest.json"):
            same = ((clone / FIXTURES / name).read_bytes()
                    == (HERE / FIXTURES / name).read_bytes())
            print(f"   fixture {name} replayed byte-identically? {same}")
            if not same:
                failures.append(f"fixture {name} did not survive the landing byte-exact")

        # 2. the suite and the control must both be green on main-only history
        r = run([PY, "-m", "pytest", SUITE, "-q", "-p", "no:randomly"], clone)
        tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-1:] or [""]
        print(f"\n   privacy suite on the SQUASH commit: exit={r.returncode}  {tail[0]}")
        if r.returncode != 0:
            failures.append("the privacy suite is not green after the landing")

        r = run([PY, CONTROL], clone)
        verdict = [ln for ln in r.stdout.splitlines() if ln.startswith("NEGATIVE-CONTROL")]
        print(f"   negative control on the SQUASH commit: exit={r.returncode}")
        for line in verdict:
            print("      " + line[:120])
        if r.returncode != 0:
            failures.append("the negative control does not fire after the landing")
        replayed = sum(1 for ln in r.stdout.splitlines() if "[REPLAYED " in ln)
        print(f"   REPLAYED arms that fired with only main history: {replayed}")
        if replayed < 10:
            failures.append(f"only {replayed} replay arms fired after the landing")

        # 3. A/B: the control must go RED on a rejected head's bytes, in the LANDED tree.
        live = (clone / MODULE).read_bytes()
        try:
            (clone / MODULE).write_bytes(
                (clone / FIXTURES / "verify_cleanroom.rejected-r2.pysrc").read_bytes())
            r = run([PY, CONTROL], clone)
            print(f"\n   A/B, rejected r2 bytes in place of the module: exit={r.returncode}")
            if r.returncode == 0:
                failures.append("the control passes on the rejected bytes AFTER landing")
        finally:
            (clone / MODULE).write_bytes(live)

        # 4. A/B: a tampered fixture must be refused, not replayed.
        fix = clone / FIXTURES / "verify_cleanroom.rejected-r1.pysrc"
        good = fix.read_bytes()
        try:
            fix.write_bytes(good.replace(b"REDACTED", b"redacted", 1))
            r = run([PY, CONTROL], clone)
            refused = "fixture_digest_mismatch" in (r.stdout + r.stderr)
            print(f"   A/B, one fixture tampered: exit={r.returncode} "
                  f"digest_mismatch_reported={refused}")
            if r.returncode == 0 or not refused:
                failures.append("a tampered fixture was not refused after landing")
        finally:
            fix.write_bytes(good)

        print()
        if failures:
            for f in failures:
                print(f"FAIL: {f}")
            return 1
        print("PASS: after a real squash onto main -- branch deleted, reflog expired, "
              "objects pruned -- the privacy suite and all of the control's arms still "
              "fire, the control still goes red on the bytes it was written against, and "
              "a tampered fixture is still refused.")
        return 0
    finally:
        shutil.rmtree(tmp, onerror=_force_writable)
        if tmp.exists():
            print(f"WARNING: could not remove the simulation clone at {tmp}")


if __name__ == "__main__":
    sys.exit(main())
