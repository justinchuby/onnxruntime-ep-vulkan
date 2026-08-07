#!/usr/bin/env python3
"""Negative control for the landing-simulation lane step: does it fire, and does it acquit?

WHAT IS UNDER TEST, AND WHY IT NEEDS ITS OWN CONTROL
====================================================
`ci/check_landing_simulation.py` runs `ci/simulate_squash_rewitness.py` on the branches whose
verdict can depend on which merge button a maintainer presses, and skips it on the ones whose
verdict cannot. Both halves of that sentence are load-bearing and both can fail silently:

* a gate that never says REQUIRED is a step that runs nothing, and a lane full of green steps
  that ran nothing is what issue #60 already found once — `simulate_squash_rewitness.py` sat
  in `ci/` for two sessions in no workflow at all;
* a simulator whose squash landing is not the tree GitHub builds is a step that runs and
  cannot see. That is not hypothetical either: until this change the squash was built by
  OVERLAYING the branch's paths onto the base, which throws away a concurrent base-side edit
  to a file the branch also touched — precisely the collision. Arm `overlay_squash_is_blind`
  below still builds it that way and requires it to be GREEN, so the repair is printed rather
  than asserted.

ARM KINDS, ratio printed, for the same reason `negative_control_ledger_census.py` prints it:

  STRUCTURAL  read of the workflow and the inventory; no repository is built
  REPLAYED    built from this repository's REAL history — `bb09871` (the base PR #53 left)
              and `ca61252` (the squash that landed it) — with one planted main-side edit
  PLANTED     built from the current tree with a synthetic base or branch

THE REPLAYED ARMS ARE THE ONES THAT MATTER, and they replay issue #60 exactly: the declaring
change is PR #53's real content, the collision is a one-line edit to
`rust/shaders/glsl/templates/q_gemv.comp` — the real declared cause path of the repository's
only `rewitness/3` record — landed on the base inside the merge window. One of them runs from
a synthetic `refs/pull/N/merge` checkout rather than from the branch, because that is what
GitHub actually hands CI and because taking that checkout as the branch head is what made the
merge-window guard vacuous in the first place.

The historical branch `squad/7-tile-matmulnbits-prefill` (head `8f12b32`) was deleted when it
landed, so it is unreachable in a CI checkout. The PR is therefore reconstructed from
main-only history as `bb09871` + `ca61252`'s tree, which is the same content by construction.

THE INSTRUMENT IS THIS TREE'S, THE SUBJECT IS THE HISTORICAL ONE
================================================================
Each replayed scenario is built in a scratch clone whose `ci/` machinery is overwritten with
THIS working tree's copies before anything runs. Otherwise the arms would exercise PR #53's
simulator against PR #53's census and say nothing about the change they are shipped with.
Nothing in `ci/` is in any shader source closure and every path under it is a refused cause
root, so swapping it cannot move a single content id the census reads.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PY = sys.executable

#: The base PR #53 branched from, and the squash commit that landed it. Both are on `main`,
#: so both are reachable in any full checkout; neither is reachable in a shallow one, which is
#: why the lane job that runs this declares `fetch-depth: 0`.
BASE_BEFORE_53 = "bb09871"
LANDED_53 = "ca61252"
#: The real declared cause path of this repository's only `rewitness/3` record.
CAUSE_PATH = "rust/shaders/glsl/templates/q_gemv.comp"
#: The `ci/` files the scenarios must run THIS tree's copy of. Everything the two entry points
#: import, transitively, and nothing else.
INSTRUMENT_FILES = (
    "ci/check_landing_simulation.py",
    "ci/simulate_squash_rewitness.py",
    "ci/check_ledger_census.py",
    "ci/proof_retirement.py",
)

RESULTS: list[tuple[str, str, bool, str]] = []


def record(kind: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((kind, name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {kind:10s} {name}" + (f" — {detail}" if detail else ""))


def _force_writable(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="ctl", GIT_AUTHOR_EMAIL="ctl@example.invalid",
        GIT_COMMITTER_NAME="ctl", GIT_COMMITTER_EMAIL="ctl@example.invalid",
    )
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )


def py(script: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    # utf-8 on both ends: the register's reasons carry `§`, and a Windows runner's console
    # codepage has killed a control harness in this repository before (see
    # negative_control_ledger_census.run).
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(
        [PY, str(Path(cwd) / script), *args], cwd=str(cwd), capture_output=True,
        text=True, encoding="utf-8", errors="replace", env=env,
    )


def _install_instrument(clone: Path) -> None:
    for rel in INSTRUMENT_FILES:
        src = REPO / rel
        if src.is_file():
            shutil.copyfile(src, clone / rel)


def _scratch_clone(tmp: Path, name: str) -> Path:
    dest = tmp / name
    r = git(["clone", "-q", "--no-local", str(REPO), str(dest)], tmp)
    if r.returncode:
        raise SystemExit(f"cannot build the control clone: {r.stderr}")
    return dest


def _plant_cause_edit(clone: Path, message: str) -> str:
    """One line appended to the real declared cause path, committed. -> the new sha.

    An APPEND, deliberately: it must merge cleanly with the branch's own edit to the same
    file, because a conflict is a landing GitHub would refuse to build and the arm would then
    be measuring the merge driver instead of the screen.
    """
    path = clone / CAUSE_PATH
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("// a concurrent main-side edit, planted by the landing-simulation control\n")
    git(["add", "-A"], clone)
    git(["commit", "-q", "-m", message], clone)
    return git(["rev-parse", "HEAD"], clone).stdout.strip()


def _build_replay(tmp: Path, name: str, collide: bool) -> tuple[Path, str]:
    """PR #53 reconstructed on `bb09871`, against a base that did or did not move `q_gemv`.

    -> (clone, base ref name). HEAD is left on the reconstructed PR branch, because that is
    what both entry points read as "this branch".
    """
    clone = _scratch_clone(tmp, name)
    git(["checkout", "-q", "-B", "sim_base", BASE_BEFORE_53], clone)
    if collide:
        _plant_cause_edit(clone, "main-side edit to the declared cause path")

    # The PR: `bb09871` with `ca61252`'s tree on top, which is PR #53's content without
    # needing the deleted branch.
    git(["checkout", "-q", "-B", "sim_pr", BASE_BEFORE_53], clone)
    git(["read-tree", "-u", "--reset", LANDED_53], clone)
    git(["add", "-A"], clone)
    git(["commit", "-q", "-m", "PR #53, reconstructed from its landed tree"], clone)
    _install_instrument(clone)
    git(["add", "-A"], clone)
    if git(["diff", "--cached", "--quiet"], clone).returncode != 0:
        git(["commit", "-q", "-m", "this tree's ci/ machinery, as the instrument"], clone)
    return clone, "sim_base"


def _build_gate_scenario(tmp: Path, name: str, move_base: bool) -> tuple[Path, str]:
    """The CURRENT tree as the PR, with an unrelated edit, against a base that did or did not
    land an edit to a declared cause path. Isolates rule R2 from rules R1 and R3."""
    clone = _scratch_clone(tmp, name)
    head = git(["rev-parse", "HEAD"], REPO).stdout.strip()
    git(["fetch", "-q", "origin", head], clone)
    git(["checkout", "-q", "-B", "gate_base", head], clone)
    if move_base:
        _plant_cause_edit(clone, "main-side edit to the declared cause path")
    git(["checkout", "-q", "-B", "gate_pr", head], clone)
    readme = clone / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\n<!-- an edit in no shader closure -->\n",
        encoding="utf-8",
    )
    git(["add", "-A"], clone)
    git(["commit", "-q", "-m", "an unrelated edit"], clone)
    _install_instrument(clone)
    git(["add", "-A"], clone)
    if git(["diff", "--cached", "--quiet"], clone).returncode != 0:
        git(["commit", "-q", "-m", "this tree's ci/ machinery, as the instrument"], clone)
    return clone, "gate_base"


# ══════════════════════════════════════════════════════════════════════════════════════════
# STRUCTURAL — the step exists, and it is the one the inventory names
# ══════════════════════════════════════════════════════════════════════════════════════════
def arm_wired_into_the_workflow() -> None:
    wf = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    gate = "ci/check_landing_simulation.py" in wf
    ctl = "ci/negative_control_landing_simulation.py" in wf
    record(
        "STRUCTURAL", "the gate and this control are both steps in ci.yml",
        gate and ctl,
        f"gate={'wired' if gate else 'ABSENT'} control={'wired' if ctl else 'ABSENT'}. "
        "An unwired tool is invisible to the lane-coverage census by construction "
        "(ci/check_verification_subjects.py's own note), so this is asserted from the "
        "workflow text and not from anybody's memory of adding it",
    )


def arm_declared_in_the_lane_inventory() -> None:
    sys.path.insert(0, str(HERE))
    import lane_inventory  # noqa: E402

    ids = {c.id for c in lane_inventory.CHECKS}
    want = {"hostfree.landing_simulation", "hostfree.landing_simulation_negative_control"}
    missing = sorted(want - ids)
    steps = {c.step for c in lane_inventory.CHECKS if c.id in want}
    wf = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    unmatched = sorted(s for s in steps if f"- name: {s}" not in wf)
    record(
        "STRUCTURAL", "both steps are declared Checks whose names match the workflow",
        not missing and not unmatched and len(steps) == 2,
        f"missing from the inventory: {missing or 'none'}; declared step names not found in "
        f"ci.yml: {unmatched or 'none'}",
    )


# ══════════════════════════════════════════════════════════════════════════════════════════
# PLANTED — the gate's two polarities, isolated to rule R2
# ══════════════════════════════════════════════════════════════════════════════════════════
def arm_gate_acquits_an_unrelated_edit(tmp: Path) -> None:
    clone, base = _build_gate_scenario(tmp, "gate_green", move_base=False)
    r = py("ci/check_landing_simulation.py", ["--explain-only", "--base", base], clone)
    ok = r.returncode == 0 and "VERDICT: NOT-REQUIRED" in r.stdout
    record(
        "PLANTED", "a branch that touches no corroborated path is NOT-REQUIRED", ok,
        f"exit={r.returncode}; "
        + ("said NOT-REQUIRED" if "NOT-REQUIRED" in r.stdout else r.stdout[-200:].strip()),
    )


def arm_gate_fires_on_the_merge_window_collision(tmp: Path) -> None:
    clone, base = _build_gate_scenario(tmp, "gate_red", move_base=True)
    r = py("ci/check_landing_simulation.py", ["--explain-only", "--base", base], clone)
    ok = (
        r.returncode == 0
        and "VERDICT: REQUIRED" in r.stdout
        and "R2 " in r.stdout
        and CAUSE_PATH in r.stdout
    )
    record(
        "PLANTED", "the SAME branch is REQUIRED once the base moves the cause path", ok,
        f"exit={r.returncode}; rule R2 {'named' if 'R2 ' in r.stdout else 'MISSING'}, "
        f"{CAUSE_PATH} {'named' if CAUSE_PATH in r.stdout else 'MISSING'}. Same branch, same "
        "register, same rules — only the base moved, which is issue #60's whole shape",
    )


def arm_an_unresolvable_base_is_an_instrument_error(tmp: Path) -> None:
    clone, _ = _build_gate_scenario(tmp, "gate_nobase", move_base=False)
    r = py(
        "ci/check_landing_simulation.py",
        ["--explain-only", "--base", "refs/heads/a-base-that-does-not-exist"], clone,
    )
    ok = r.returncode == 2 and "ERROR(instrument=base_unavailable)" in r.stdout
    record(
        "PLANTED", "an unresolvable base is exit 2, never a skip", ok,
        f"exit={r.returncode} (want 2); "
        + ("token present" if "base_unavailable" in r.stdout else r.stdout[-200:].strip())
        + ". A simulation that quietly does not run on the day origin/main was not fetched "
        "is green for the same reason it is useless",
    )


def arm_a_shallow_checkout_is_an_instrument_error(tmp: Path) -> None:
    dest = tmp / "gate_shallow"
    r = git(["clone", "-q", "--depth", "1", "--no-local", str(REPO), str(dest)], tmp)
    if r.returncode:
        record("PLANTED", "a shallow checkout is exit 2, never a skip", False,
               f"could not build a shallow clone: {r.stderr.strip()[:160]}")
        return
    _install_instrument(dest)
    rr = py("ci/check_landing_simulation.py", ["--explain-only"], dest)
    ok = rr.returncode == 2 and "shallow_checkout" in rr.stdout
    record(
        "PLANTED", "a shallow checkout is exit 2, never a skip", ok,
        f"exit={rr.returncode} (want 2); "
        + ("token present" if "shallow_checkout" in rr.stdout else rr.stdout[-200:].strip())
        + ". Issue #28's trap is one level down from here: a --depth fetch marks its own "
        "graft point even inside a fetch-depth: 0 checkout",
    )


# ══════════════════════════════════════════════════════════════════════════════════════════
# REPLAYED — the real q_gemv collision, both polarities
# ══════════════════════════════════════════════════════════════════════════════════════════
def _run_squash(clone: Path, base: str) -> subprocess.CompletedProcess:
    # One landing, no inner controls, no replay commit: this arm is asserting the polarity of
    # the SQUASH, and the inner controls have their own arms in
    # ci/negative_control_ledger_census.py. The lane step runs the full matrix.
    return py(
        "ci/simulate_squash_rewitness.py",
        ["--landing", "squash", "--base", base, "--no-controls", "--no-replay"], clone,
    )


def arm_squash_is_red_on_the_collision(tmp: Path) -> Path:
    clone, base = _build_replay(tmp, "replay_red", collide=True)
    r = _run_squash(clone, base)
    ok = (
        r.returncode == 1
        and "uncorroborated_rewitness_cause" in r.stdout
        and "SIM INVALID" not in r.stdout
    )
    record(
        "REPLAYED", "the real squash is RED on the concurrent q_gemv cause-path edit", ok,
        f"exit={r.returncode} (want 1); condition "
        + ("uncorroborated_rewitness_cause" if "uncorroborated_rewitness_cause" in r.stdout
           else "NOT FOUND")
        + f". Base {BASE_BEFORE_53} + one line in {CAUSE_PATH}; PR = {LANDED_53}'s tree",
    )
    if not ok:
        print("      " + "\n      ".join(r.stdout.strip().splitlines()[-14:]))
    return clone


def arm_squash_is_green_without_the_collision(tmp: Path) -> None:
    clone, base = _build_replay(tmp, "replay_green", collide=False)
    r = _run_squash(clone, base)
    ok = r.returncode == 0 and "SIM INVALID" not in r.stdout
    record(
        "REPLAYED", "the SAME PR is GREEN when nothing lands on the cause path", ok,
        f"exit={r.returncode} (want 0). This is the arm that stops the one above from being a "
        "constant: the branch, the register and the declared content ids are byte-identical "
        "in both, and only the base differs",
    )
    if not ok:
        print("      " + "\n      ".join(r.stdout.strip().splitlines()[-14:]))


def arm_a_pull_request_merge_ref_still_measures_the_merge_window(tmp: Path) -> None:
    """THE ARM FOR ISSUE #60's SECOND BLOCKER, end to end, on the real collision.

    Every other arm here runs from a checkout that IS the branch. GitHub does not hand CI
    that: on a `pull_request` event the checkout is `refs/pull/N/merge`, a synthetic
    two-parent commit whose FIRST parent is the base. `git rev-parse HEAD` on it therefore
    makes `merge-base(base, HEAD)` the base itself — the base has "moved" zero paths, the
    merge-window `lost` guard in `_land` evaluates an empty set, and the engine prints "this
    run cannot exhibit the merge-window collision" directly above the merge-window collision.

    So this arm reproduces that checkout exactly, runs the LANE'S ENTRY POINT (the gate, which
    resolves `HEAD^2` and forwards it as `--head`), and requires three things at once: the
    gate fires with R2 naming the real cause path, the engine reports a NON-ZERO base-moved
    count, and the sentence that would have been a lie is absent. A gate and the engine it
    invokes must agree about which commit is the branch head.
    """
    clone, base = _build_replay(tmp, "merge_ref", collide=True)
    pr = git(["rev-parse", "sim_pr"], clone).stdout.strip()
    git(["checkout", "-q", "-B", "pull_60_merge", base], clone)
    m = git(["merge", "-q", "--no-ff", "-m", "Merge pull request #60", pr], clone)
    if m.returncode:
        record("REPLAYED", "a pull_request merge ref still measures the merge window", False,
               f"could not build the synthetic merge ref: {(m.stdout + m.stderr)[:200]}")
        return
    merge_ref = git(["rev-parse", "HEAD"], clone).stdout.strip()

    r = py(
        "ci/check_landing_simulation.py",
        ["--base", base, "--sim-arg=--landing", "--sim-arg=squash",
         "--sim-arg=--no-controls", "--sim-arg=--no-replay"],
        clone,
    )
    out = r.stdout + r.stderr
    fired = "VERDICT: REQUIRED" in out and "R2 " in out and CAUSE_PATH in out
    resolved = f"--head {pr}" in out or pr[:12] in out
    moved = re.search(r"the base has moved (\d+) path\(s\) since the merge base", out)
    nonzero = bool(moved) and int(moved.group(1)) > 0
    honest = "cannot exhibit the merge-window collision" not in out
    red = r.returncode == 1 and "uncorroborated_rewitness_cause" in out
    ok = fired and resolved and nonzero and honest and red
    record(
        "REPLAYED", "a pull_request merge ref still measures the merge window", ok,
        f"checkout {merge_ref[:12]} is the merge ref, branch head {pr[:12]}; gate "
        f"{'fired on R2' if fired else 'DID NOT FIRE'}; head {'forwarded' if resolved else 'NOT FORWARDED'}; "
        f"base moved {moved.group(1) if moved else '(not printed)'} path(s) "
        f"{'(non-vacuous)' if nonzero else 'WHICH IS VACUOUS'}; "
        f"{'no' if honest else 'STILL PRINTS THE'} 'cannot exhibit' claim; "
        f"exit={r.returncode} (want 1) {'red on the collision' if red else 'NOT RED'}",
    )
    if not ok:
        print("      " + "\n      ".join(out.strip().splitlines()[-18:]))


def arm_overlay_squash_is_blind(tmp: Path) -> None:
    """The contrast arm: build the squash the way this script used to, demand GREEN.

    `git checkout <pr> -- .` overlays the branch's paths onto the base's tree. For a file both
    sides changed, the branch wins and the base's edit is gone — so the collision is overlaid
    away and the census, correctly, finds nothing wrong with the tree it was handed. Wiring
    THAT into the lane would have produced a green step on the exact defect it was wired in
    for, which is why this arm asserts the old shape is blind rather than trusting that the
    new one is not.
    """
    clone, base = _build_replay(tmp, "overlay", collide=True)
    pr = git(["rev-parse", "sim_pr"], clone).stdout.strip()
    git(["checkout", "-q", "-B", "overlay_main", base], clone)
    git(["checkout", "-q", pr, "--", "."], clone)
    git(["add", "-A"], clone)
    git(["commit", "-q", "-m", "overlay squash landing (the shape this used to build)"], clone)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run(
        [PY, str(clone / "ci" / "check_ledger_census.py"), "--repo", str(clone)],
        cwd=str(clone), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env,
    )
    still_there = git(
        ["diff", "--quiet", base, "HEAD", "--", CAUSE_PATH], clone
    ).returncode != 0
    ok = r.returncode == 0 and still_there
    record(
        "REPLAYED", "the OLD overlay squash is green on the same collision (contrast)", ok,
        f"census exit={r.returncode} (want 0) on a tree whose {CAUSE_PATH} "
        + ("no longer matches the base's" if still_there else "DID match the base's")
        + ". The repair is the landing shape, not the census",
    )


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    tmp = Path(tempfile.mkdtemp(prefix="landsimctl_", dir=str(REPO.parent)))
    try:
        print("\n-- structural")
        arm_wired_into_the_workflow()
        arm_declared_in_the_lane_inventory()

        print("\n-- the gate, both polarities")
        arm_gate_acquits_an_unrelated_edit(tmp)
        arm_gate_fires_on_the_merge_window_collision(tmp)
        arm_an_unresolvable_base_is_an_instrument_error(tmp)
        arm_a_shallow_checkout_is_an_instrument_error(tmp)

        print("\n-- the simulator, both polarities, on the real issue #60 collision")
        arm_squash_is_red_on_the_collision(tmp)
        arm_squash_is_green_without_the_collision(tmp)
        arm_a_pull_request_merge_ref_still_measures_the_merge_window(tmp)
        arm_overlay_squash_is_blind(tmp)
    finally:
        shutil.rmtree(tmp, onerror=_force_writable)
        if tmp.exists():
            print(f"WARNING: could not remove the control scratch at {tmp}")

    kinds = {k: 0 for k in ("STRUCTURAL", "PLANTED", "REPLAYED")}
    for kind, _n, _ok, _d in RESULTS:
        kinds[kind] = kinds.get(kind, 0) + 1
    failed = [n for _k, n, ok, _d in RESULTS if not ok]
    print(
        f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} arms passed "
        f"({kinds['STRUCTURAL']} STRUCTURAL, {kinds['PLANTED']} PLANTED, "
        f"{kinds['REPLAYED']} REPLAYED). The ratio is printed because a control made only of "
        "plants proves only that the planting works; the REPLAYED arms are built from this "
        f"repository's own {BASE_BEFORE_53}..{LANDED_53}."
    )
    if failed:
        print("FAIL(condition=control_arm_did_not_fire):")
        for n in failed:
            print(f"  - {n}")
        return 1
    print(
        "PASS: the gate fires on the merge-window collision and acquits an unrelated edit; "
        "the real squash is red on it and green without it; the overlay squash it replaced "
        "is blind to it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
