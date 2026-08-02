#!/usr/bin/env python
"""Union check — "does my change break a caller that does not exist on my branch yet?"

Three defects on 2026-08-01 had one shape, in three languages and three subsystems:

- **Tank:** wiring a residency screen made previously tally-neutral tests start moving a
  process-global counter. The lock was correct; the population that needed it grew.
- **Niobe:** a ``sys.path.insert`` in a new ``bench/`` test file rebound imports for
  Link's ``ci/`` lane checks. Each file green alone, three red together.
- **Trinity + Mouse:** a required keyword-only ``guard`` parameter added on one branch, a
  caller added on another, into the same file. Each branch green alone; the union raised
  ``TypeError: missing 1 required keyword-only argument``.

In all three, nobody did anything wrong locally, and **no command either author could have
run on their own branch would have shown it**. Our discipline verifies branches; the
defects live in unions.

R9 amendment 5, applied to this tool
------------------------------------
Ask which way each tier moves when its subject is wrong.

*Tier 3* — merge and run — moves with its subject: if the union is broken, the merged
suite goes red. It is a **gate**.

*Tiers 1 and 2* — file intersection and global-side-effect scan — do **not**. A union
defect can exist with neither an intersecting file nor a scanned side effect (Niobe's
would have been caught by tier 2; a defect through a shared C ABI or a shared artifact
would be caught by neither). Their green means only "the two cheap shapes are absent",
which is a statement about this tool, not about the union. **They are preconditions that
route attention, never gates**, and this script never prints a bare PASS for them — it
prints ``PRECONDITION`` with the shapes it looked for named, so a reader cannot mistake
the absence of a report for the presence of evidence.

That distinction is the whole reason this file is not called ``check_union.py``: the
``check_*`` names in ``ci/`` are gates with exit code 1 for a detection, and this is only
one of those in ``--run`` mode.

Usage
-----
::

    python tests/union_check.py                 # tiers 1+2, seconds, from a branch
    python tests/union_check.py --run           # tier 3: trial-merge and run the union
    python tests/union_check.py --base main --run --pytest-args "-q -x"

Exit codes follow the R13 vocabulary the rest of the harness uses:
0 ``PASS`` / 1 ``FAIL(condition=...)`` / 4 ``ERROR(instrument=...)``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Directories whose ``.py`` files can end up on ``sys.path`` during a suite run.
_IMPORT_DIRS = ("tests/ops", "bench", "bench/results", "ci", "rust/tools", "tests")

#: Shapes that reach out of their own file at import time. Position **0** is the
#: shadowing position: an insert there makes that directory's modules win over every
#: module already importable under the same basename. ``append`` cannot shadow, so it is
#: not flagged — flagging it produced 20 lines of noise on the first run of this tool,
#: and a report that names everything routes attention nowhere.
_GLOBAL_SIDE_EFFECTS = (
    (
        r"sys\.path\.insert\s*\(\s*0\s*,",
        "sys.path.insert(0, ...) — the shadowing position; this directory's modules now "
        "win over same-named modules anywhere else on the path",
    ),
    (
        r"^os\.environ\[",
        "module-scope os.environ assignment — leaks into every test imported afterwards",
    ),
)


def _module_basename_collisions() -> dict[str, list[str]]:
    """Module basenames importable from more than one directory.

    This is the **population** that makes a ``sys.path.insert(0, ...)`` dangerous, and it
    is what turned Niobe's insert into three red lanes in Link's ``ci/`` checks: the
    insert was correct and local, and the collision was somewhere else entirely. Neither
    author could see it from their own file, which is the whole subject of this tool.

    Computed from the working tree, not from the diff: a collision that already exists is
    exactly as dangerous as one being added, and more likely to be forgotten.
    """
    seen: dict[str, list[str]] = {}
    for d in _IMPORT_DIRS:
        base = REPO / d
        if not base.is_dir():
            continue
        for py in base.glob("*.py"):
            seen.setdefault(py.stem, []).append(d)
    return {k: v for k, v in sorted(seen.items()) if len(v) > 1}



def _git(*args: str, cwd: Path = REPO) -> str:
    out = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def _changed(a: str, b: str) -> set[str]:
    """Files changed on *b* since it diverged from *a* (``a...b``)."""
    return {ln for ln in _git("diff", "--name-only", f"{a}...{b}").splitlines() if ln}


def _added_or_modified_lines(base: str, ref: str, path: str) -> list[str]:
    try:
        diff = _git("diff", "-U0", f"{base}...{ref}", "--", path)
    except RuntimeError:
        return []
    return [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def tier1_intersection(base: str, head: str) -> dict:
    """Files edited on **both** sides. Neither branch's green covers the merged text."""
    mine = _changed(base, head)
    theirs = _changed(head, base)
    both = sorted(mine & theirs)
    return {
        "tier": 1,
        "name": "file_intersection",
        "changed_here": len(mine),
        "changed_there": len(theirs),
        "intersecting": both,
        "shape": (
            "the same file was edited on both sides; a signature changed on one and a "
            "caller added on the other is invisible to either branch's suite"
        ),
    }


def tier2_global_side_effects(base: str, head: str) -> dict:
    """Lines *added on either side* that change state outside their own file."""
    findings: list[dict] = []
    for side, (a, b) in (("here", (base, head)), ("there", (head, base))):
        for path in sorted(_changed(a, b)):
            if not path.endswith(".py"):
                continue
            for line in _added_or_modified_lines(a, b, path):
                for pattern, why in _GLOBAL_SIDE_EFFECTS:
                    if re.search(pattern, line):
                        findings.append(
                            {"side": side, "path": path, "line": line.strip()[:160], "why": why}
                        )
                        break
    # An insert at position 0 only *shadows* if some other directory on the path offers a
    # module of the same basename. The collision map is the signal; the insert alone is
    # an idiom. So a finding is elevated only when the editing file lives in a directory
    # that actually participates in a collision — otherwise it is counted, not named.
    # Naming all 16 was the first draft, and a report that names everything routes
    # attention nowhere.
    collisions = _module_basename_collisions()
    hot = {d for dirs in collisions.values() for d in dirs}
    elevated = [f for f in findings if str(Path(f["path"]).parent).replace("\\", "/") in hot]
    return {
        "tier": 2,
        "name": "global_side_effects",
        "findings": elevated,
        "suppressed": len(findings) - len(elevated),
        "collisions": collisions,
        "shape": (
            "a file that is green alone can change the meaning of a file it never "
            "mentions; an insert at sys.path position 0 plus a module-basename collision "
            "elsewhere in the repo is the pair that does it"
        ),
    }


def _conflicts(tree: Path) -> set[str]:
    out = _git("diff", "--name-only", "--diff-filter=U", cwd=tree)
    return {ln for ln in out.splitlines() if ln}


#: Paths that are regenerated *outputs*, never inputs to a lane. A conflict here is two
#: runs disagreeing about what they measured, not a defect in the union of the code, so
#: the tool may take HEAD's copy and go on to test the code. Anything outside this set is
#: still ERROR(instrument=merge_conflict): the union does not exist yet, and an instrument
#: that cannot construct its subject has not observed it.
_ARTIFACT_PREFIXES = ("bench/results/",)


def _resolve_artifact_conflicts(tree: Path) -> set[str]:
    """Take HEAD's copy of conflicted *artifacts*; return whatever is still conflicted."""
    remaining = set()
    for path in _conflicts(tree):
        if path.startswith(_ARTIFACT_PREFIXES) and path.endswith((".json", ".log", ".md")):
            _git("checkout", "--ours", "--", path, cwd=tree)
            _git("add", "--", path, cwd=tree)
        else:
            remaining.add(path)
    return remaining


def tier3_trial_merge_and_run(
    base: str, head: str, pytest_args: list[str], targets: list[str],
    resolve_artifacts: bool = False,
) -> dict:
    """The only tier that is a gate: merge into a scratch worktree and run the union."""
    scratch = Path(tempfile.mkdtemp(prefix="union-check-", dir=str(REPO.parent)))
    tree = scratch / "tree"
    branch = f"union-check/{os.getpid()}"
    try:
        _git("worktree", "add", "--detach", str(tree), head)
        merge = subprocess.run(
            ["git", "merge", "--no-edit", base],
            cwd=str(tree), capture_output=True, text=True, check=False,
        )
        if merge.returncode != 0:
            unresolved = _resolve_artifact_conflicts(tree) if resolve_artifacts else _conflicts(tree)
            if unresolved:
                return {
                    "tier": 3, "name": "merged_run", "state": "ERROR(instrument=merge_conflict)",
                    "detail": (merge.stdout + merge.stderr)[-2000:],
                    "unresolved": sorted(unresolved),
                    "shape": "the union does not exist yet; resolve the conflict, then re-run",
                }
        run = subprocess.run(
            [sys.executable, "-m", "pytest", *targets, *pytest_args],
            cwd=str(tree), capture_output=True, text=True, check=False,
        )
        tail = (run.stdout + run.stderr).splitlines()[-40:]
        return {
            "tier": 3, "name": "merged_run",
            "state": "PASS" if run.returncode == 0 else "FAIL(condition=union_red)",
            "returncode": run.returncode,
            "targets": targets,
            "tail": tail,
            "shape": "this tier moves with its subject: a broken union is a red suite",
        }
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(tree)],
                       cwd=str(REPO), capture_output=True, text=True, check=False)
        subprocess.run(["git", "branch", "-D", branch],
                       cwd=str(REPO), capture_output=True, text=True, check=False)
        shutil.rmtree(scratch, ignore_errors=True)


def _targets_for(intersecting: list[str], side_effects: list[dict]) -> list[str]:
    """Test paths worth running in the merged tree, widest-first."""
    touched = {f["path"] for f in side_effects} | set(intersecting)
    dirs = {p.split("/")[0] for p in touched if "/" in p}
    targets = []
    if "tests" in dirs or not dirs:
        targets.append("tests/ops")
    for extra in ("bench", "ci"):
        if extra in dirs and Path(REPO / extra).is_dir():
            targets.append(extra)
    return targets or ["tests/ops"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="main", help="the branch this will merge into")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--run", action="store_true", help="tier 3: trial-merge and run")
    ap.add_argument("--pytest-args", default="-q -p no:cacheprovider")
    ap.add_argument("--json", default="", help="also write the report here")
    ap.add_argument(
        "--resolve-artifacts", action="store_true",
        help="take HEAD's copy of conflicted bench/results artifacts so the code union can be tested",
    )
    ap.add_argument(
        "--targets", default="",
        help="comma-separated pytest targets for tier 3; default is derived from tiers 1-2",
    )
    args = ap.parse_args(argv)

    try:
        t1 = tier1_intersection(args.base, args.head)
        t2 = tier2_global_side_effects(args.base, args.head)
    except RuntimeError as exc:
        print(f"UNION-CHECK: ERROR(instrument=git) {exc}")
        return 4

    report = {"base": args.base, "head": args.head, "tiers": [t1, t2]}

    print("=== UNION CHECK — 'does my change break a caller that is not on my branch?' ===")
    print(f"  base={args.base}  changed here={t1['changed_here']}  changed there={t1['changed_there']}")
    print()
    print("  TIER 1 — file intersection")
    if t1["intersecting"]:
        for p in t1["intersecting"]:
            print(f"    OVERLAP   {p}")
        print("    Neither branch's green covers the merged text of these files.")
    else:
        print("    no file was edited on both sides")
    print()
    print("  TIER 2 — import-path shadowing added in a directory that collides")
    if t2["findings"]:
        for f in t2["findings"]:
            print(f"    SHADOWING   [{f['side']}] {f['path']}")
            print(f"                {f['line']}")
    else:
        print("    no insert at sys.path position 0 was added in a colliding directory")
    print(f"    ({t2['suppressed']} further insert(s) added in non-colliding directories, not named)")
    print()
    print("  TIER 2b — module basenames importable from more than one directory")
    print("    (the population that makes an insert above dangerous; existing, not added)")
    if t2["collisions"]:
        for name, dirs in t2["collisions"].items():
            print(f"    COLLISION   {name}.py  <-  {', '.join(dirs)}")
    else:
        print("    none")
    print()
    print("  Tiers 1 and 2 are PRECONDITIONS, not gates (R9 amendment 5). They do not move")
    print("  with their subject: a union can be broken with no intersecting file and no")
    print("  scanned side effect. Their silence is a statement about this tool, not about")
    print("  the union. Only tier 3 detects.")
    print()

    if not args.run:
        verdict = "PRECONDITION(tiers=1,2; tier 3 not run)"
        print(f"UNION-CHECK: {verdict}")
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    targets = (
        [t.strip() for t in args.targets.split(",") if t.strip()]
        or _targets_for(t1["intersecting"], t2["findings"])
    )
    print(f"  TIER 3 — trial merge of {args.base} into {args.head}, running {targets}")
    t3 = tier3_trial_merge_and_run(
        args.base, args.head, args.pytest_args.split(), targets,
        resolve_artifacts=args.resolve_artifacts,
    )
    report["tiers"].append(t3)
    for line in t3.get("tail", []):
        print(f"    {line}")
    print()
    print(f"UNION-CHECK: {t3['state']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if t3["state"].startswith("ERROR"):
        print(t3.get("detail", ""))
        return 4
    return 0 if t3["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
