"""Squash-merge simulation for THIS branch, run against the real repository.

Blocker #1's claim is that the declaration in evidence/proof_rewitness.json survives the
squash that lands it. That is checkable now, not after the fact: clone the repo, reset a
worktree to origin/main, apply this branch's TREE as one brand-new commit (exactly what
`gh pr merge --squash` produces), and run the census there. The branch commits are not
ancestors of that commit, so a v1 record naming one would self-invalidate.
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


def _force_writable(func, path, _exc):
    """Git marks objects and packs read-only, and Windows honours that on unlink.

    `rmtree(..., ignore_errors=True)` therefore leaves a multi-hundred-megabyte clone
    behind next to the repository every time this runs, silently. Clear the read-only
    bit and retry, and let a failure that is not that one be seen.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def main() -> int:
    head = git(["rev-parse", "HEAD"], HERE).stdout.strip()
    base = git(["rev-parse", "origin/main"], HERE).stdout.strip()
    print(f"branch head {head[:12]}  base {base[:12]}")

    tmp = Path(tempfile.mkdtemp(prefix="squashsim_", dir=str(HERE.parent)))
    try:
        clone = tmp / "clone"
        r = git(["clone", "-q", "--no-local", str(HERE), str(clone)], HERE.parent)
        if r.returncode:
            print(r.stderr)
            return 2
        git(["fetch", "-q", "origin", head, base], clone)
        git(["checkout", "-q", "--detach", base], clone)

        # Apply THE WORKING TREE, not HEAD's tree. The register is edited in the same commit
        # that edits the ledger, so a simulation that could only read committed state would
        # answer the question one commit too late — after the point where a bad declaration
        # can still be fixed without a follow-up, which is the entire defect being repaired.
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
        git(["-c", "user.name=sim", "-c", "user.email=sim@sim",
             "commit", "-q", "-m", "squash simulation of PR #44"], clone)
        git(["branch", "-q", "-f", "main"], clone)
        git(["checkout", "-q", "main"], clone)
        squash = git(["rev-parse", "HEAD"], clone).stdout.strip()

        anc = git(["merge-base", "--is-ancestor", head, squash], clone).returncode
        print(f"squash commit {squash[:12]}; branch head is ancestor? {anc == 0}")
        if anc == 0:
            print("SIM INVALID: the branch head is still an ancestor, this is not a squash")
            return 2

        reg_same = ((clone / "evidence" / "proof_rewitness.json").read_bytes()
                    == (HERE / "evidence" / "proof_rewitness.json").read_bytes())
        led_same = ((clone / "evidence" / "proof_ledger.jsonl").read_bytes()
                    == (HERE / "evidence" / "proof_ledger.jsonl").read_bytes())
        print(f"register and ledger replayed byte-identically? {reg_same and led_same}")
        if not (reg_same and led_same):
            return 2

        r = subprocess.run([PY, str(clone / "ci" / "check_ledger_census.py"),
                            "--repo", str(clone)], cwd=clone, capture_output=True, text=True)
        ok = r.returncode == 0
        for line in r.stdout.splitlines():
            if "frame witnesses" in line or line.startswith(("PASS", "FAIL", "ERROR")):
                print("   " + line)
        print(f"census on the SQUASH commit: exit={r.returncode}")

        # The negative control: the same tree with the record written the v1 way it was
        # written before this fix. It must go red, or the simulation proves nothing.
        reg = clone / "evidence" / "proof_rewitness.json"
        doc = json.loads(reg.read_text(encoding="utf-8"))
        v2 = next((r for r in doc["rewitness"] if r.get("schema") == "rewitness/2"), None)
        if v2 is None:
            print("no rewitness/2 record to downgrade; the control cannot run")
            return 2
        doc["rewitness"][doc["rewitness"].index(v2)] = {
            "revision": head,
            "field": v2["field"],
            "owner": v2["owner"],
            "date": v2["date"],
            "reason": "v1 control: the same declaration, indexed by the branch commit",
        }
        reg.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        git(["add", "-A"], clone)
        git(["-c", "user.name=sim", "-c", "user.email=sim@sim",
             "commit", "-q", "-m", "v1 control"], clone)
        r2 = subprocess.run([PY, str(clone / "ci" / "check_ledger_census.py"),
                             "--repo", str(clone)], cwd=clone, capture_output=True, text=True)
        red = r2.returncode == 1 and "undeclared_witness_move" in r2.stdout
        print(f"v1 control on the SAME squash: exit={r2.returncode} "
              f"undeclared_witness_move={'undeclared_witness_move' in r2.stdout}")

        if ok and red:
            print("\nPASS: rewitness/2 survives the squash; the v1 form of the same "
                  "declaration does not.")
            return 0
        print("\nFAIL")
        return 1
    finally:
        shutil.rmtree(tmp, onerror=_force_writable)
        if tmp.exists():
            print(f"WARNING: could not remove the simulation clone at {tmp}")


if __name__ == "__main__":
    sys.exit(main())
