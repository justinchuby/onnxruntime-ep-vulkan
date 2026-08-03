#!/usr/bin/env python3
"""Negative control for ``check_build_precondition`` — arms counted by provenance.

A screen never seen failing has no demonstrated positive state, and a screen over
*workflow files* is unusually easy to write in a way that has none: the tree it runs on
is, by construction, the tree its author just fixed.

Provenance, same vocabulary as the other controls in this directory:

* ``LIVE``     — this repository's own workflow files as they stand right now.
* ``REPLAYED`` — the real defect, read out of this repository's history with ``git
                 show``. Not text written to make the screen fire: the exact bytes that
                 were on ``main`` until 2026-08-03, when both device lanes carried
                 ``BUILD_SKIPPED=1; exit 0`` and thirty steps were gated on it.
* ``PLANTED``  — text written to exercise a path. Proves the path is wired; proves
                 nothing about whether that shape occurs in the wild.

The REPLAYED arm is the one that matters. Every other arm here is me writing a defect I
already know how to write. That one is the defect as it actually existed, retrieved
rather than reconstructed, and if a future refactor of this screen stops catching it the
arm goes red for a reason nobody can argue with.

Exit 0 when every arm fired as declared, 1 otherwise.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
SCRIPT = CI_DIR / "check_build_precondition.py"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

#: The commit at which both device lanes still carried the escape.
HISTORICAL_REF = "607056a"
WORKFLOW_REL = ".github/workflows/ci.yml"

MINIMAL_HEAD = """name: control
on: [push]
jobs:
  demo:
"""


def run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


def write(tmp: Path, name: str, text: str) -> Path:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


def historical_workflow() -> str | None:
    """The real ci.yml at the commit that still carried the escape.

    Returns None rather than raising: a shallow clone legitimately may not have the
    object, and a control that cannot read its subject must say UNOBSERVABLE rather than
    quietly dropping an arm. The caller records that as a failed arm, because a control
    that silently loses its only non-planted arm is the shape this whole round is about.
    """
    proc = subprocess.run(
        ["git", "show", f"{HISTORICAL_REF}:{WORKFLOW_REL}"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def main() -> int:
    arms: list[tuple[str, str, bool, str]] = []  # (provenance, name, ok, note)

    with tempfile.TemporaryDirectory(prefix="buildprecond-") as td:
        tmp = Path(td)

        # ---- LIVE ------------------------------------------------------------
        rc, out = run([
            str(REPO_ROOT / ".github/workflows/ci.yml"),
            str(REPO_ROOT / ".github/workflows/conformance.yml"),
        ])
        arms.append((
            "LIVE",
            "this repository's own workflows pass all three rules",
            rc == EXIT_PASS and "BUILD-PRECONDITION: PASS" in out,
            "the green arm: a screen that only ever reddens is a lock on a broken state",
        ))

        # ---- REPLAYED: the defect as it really was ---------------------------
        hist = historical_workflow()
        if hist is None:
            arms.append((
                "REPLAYED",
                f"the real BUILD_SKIPPED escape at {HISTORICAL_REF} goes red",
                False,
                f"could not `git show {HISTORICAL_REF}:{WORKFLOW_REL}` — UNOBSERVABLE, "
                "and an unreadable subject is not a passing arm",
            ))
        else:
            path = write(tmp, "historical-ci.yml", hist)
            rc, out = run([str(path)])
            arms.append((
                "REPLAYED",
                f"the real BUILD_SKIPPED escape at {HISTORICAL_REF} goes red",
                rc == EXIT_FAIL_CONDITION and "skip_flag_with_exit_zero" in out,
                "BP1 on the exact bytes that were on main until 2026-08-03",
            ))
            arms.append((
                "REPLAYED",
                "and it is BP1 that catches it, not BP2 by accident",
                "BP1 —" in out and "BP2 —" not in out,
                "the writer was present then, so the guards were live, not dormant",
            ))

        # ---- PLANTED: BP1, the shape rather than the string -------------------
        rc, out = run([str(write(tmp, "bp1-other-name.yml", MINIMAL_HEAD + """    steps:
      - name: Provision the device
        run: |
          if ! command -v vulkaninfo; then
            echo "NO_GPU=yes" >> "$GITHUB_ENV"
            exit 0
          fi
      - name: Correctness tests
        if: env.NO_GPU != 'yes'
        run: pytest tests/
"""))])
        arms.append((
            "PLANTED",
            "BP1 fires on a different name, a different step and a different reason",
            rc == EXIT_FAIL_CONDITION and "skip_flag_with_exit_zero" in out,
            "the rule is the conjunction write+exit-0+gated, not the string BUILD_SKIPPED",
        ))

        # ---- PLANTED: BP1 does NOT fire on either half alone ------------------
        rc, out = run([str(write(tmp, "bp1-write-only.yml", MINIMAL_HEAD + """    steps:
      - name: Provision ONNX Runtime
        run: echo "ORT_HOME=$PWD/ort" >> "$GITHUB_ENV"
      - name: Build the world
        run: |
          cargo build --release
          test -f target/release/libx.so || exit 1
      - name: Tests
        if: env.ORT_HOME != ''
        run: pytest tests/
"""))])
        arms.append((
            "PLANTED",
            "a provisioning step that publishes a gated name and does NOT exit 0 is clean",
            rc == EXIT_PASS,
            "half the shape is not the shape; a rule that fired here would be ignored",
        ))

        rc, out = run([str(write(tmp, "bp1-exit-only.yml", MINIMAL_HEAD + """    steps:
      - name: Optional cache warm
        run: |
          if [ ! -d .cache ]; then
            echo "no cache to warm"
            exit 0
          fi
          warm-cache
      - name: Build the world
        run: |
          cargo build --release
          test -f target/release/libx.so || exit 1
"""))])
        arms.append((
            "PLANTED",
            "a step that exits 0 early but gates nothing is clean",
            rc == EXIT_PASS,
            "`exit 0` is not a defect; `exit 0` that silently deletes other steps is",
        ))

        # ---- PLANTED: BP2, the dormant guard ---------------------------------
        rc, out = run([str(write(tmp, "bp2-dormant.yml", MINIMAL_HEAD + """    steps:
      - name: Build the world
        run: |
          cargo build --release
          test -f target/release/libx.so || exit 1
      - name: Tests
        if: env.BUILD_SKIPPED != '1'
        run: pytest tests/
"""))])
        arms.append((
            "PLANTED",
            "BP2 fires on a guard whose writer does not exist",
            rc == EXIT_FAIL_CONDITION and "dead_guard" in out,
            "this is the state my own 2026-08-03 fix left the tree in",
        ))

        # ---- PLANTED: BP2 does not fire on a declared env: key ----------------
        rc, out = run([str(write(tmp, "bp2-declared.yml", """name: control
on: [push]
env:
  RUN_HEAVY: "0"
jobs:
  demo:
    steps:
      - name: Build the world
        run: |
          cargo build --release
          test -f target/release/libx.so || exit 1
      - name: Heavy tests
        if: env.RUN_HEAVY == '1'
        run: pytest tests/heavy
"""))])
        arms.append((
            "PLANTED",
            "a guard reading a DECLARED env: key is clean",
            rc == EXIT_PASS,
            "a workflow-level default is a writer; treating it as dormant would train "
            "people to ignore BP2",
        ))

        # ---- PLANTED: BP3, the build step that never looks at its output ------
        rc, out = run([str(write(tmp, "bp3-unverified.yml", MINIMAL_HEAD + """    steps:
      - name: Build Vulkan EP (cargo build --release)
        run: |
          cargo build --release --manifest-path rust/Cargo.toml
          echo "EP_LIB=$PWD/rust/target/release/libx.so" >> "$GITHUB_ENV"
      - name: Tests
        run: pytest tests/
"""))])
        arms.append((
            "PLANTED",
            "BP3 fires on a build step that publishes a path it never checked",
            rc == EXIT_FAIL_CONDITION and "build_step_does_not_verify_its_artifact" in out,
            "a path string is written whether or not anything is at the end of it",
        ))

        # ---- PLANTED: BP3 is satisfied by a real assertion --------------------
        rc, out = run([str(write(tmp, "bp3-verified.yml", MINIMAL_HEAD + """    steps:
      - name: Build Vulkan EP (cargo build --release)
        run: |
          cargo build --release --manifest-path rust/Cargo.toml
          test -f rust/target/release/libx.so || exit 1
          echo "EP_LIB=$PWD/rust/target/release/libx.so" >> "$GITHUB_ENV"
      - name: Tests
        run: pytest tests/
"""))])
        arms.append((
            "PLANTED",
            "BP3 is satisfied by an actual existence assertion",
            rc == EXIT_PASS,
            "the green arm for BP3",
        ))

        # ---- PLANTED: instrument paths ---------------------------------------
        rc, out = run([str(tmp / "does-not-exist.yml")])
        arms.append((
            "PLANTED",
            "a workflow that is not there is ERROR(instrument), never a pass",
            rc == EXIT_ERROR_INSTRUMENT and "workflow_not_found" in out,
            "a screen that read nothing must not report a clean tree",
        ))

        rc, out = run([str(write(tmp, "empty.yml", "name: nothing\non: [push]\n"))])
        arms.append((
            "PLANTED",
            "a file with no steps is ERROR(instrument=no_steps_parsed)",
            rc == EXIT_ERROR_INSTRUMENT and "no_steps_parsed" in out,
            "UNOBSERVABLE is not zero — this screen has no YAML parser and must say so "
            "when the block structure it relies on stops matching",
        ))

        rc, out = run([
            str(REPO_ROOT / ".github/workflows/ci.yml"),
            "--allowlist",
            str(tmp / "not-json.json"),
        ])
        arms.append((
            "PLANTED",
            "a missing allowlist is an empty allowlist, not an error",
            rc == EXIT_PASS,
            "the file is optional; only an UNREADABLE one is an instrument failure",
        ))

        bad = write(tmp, "bad-allowlist.json", "{ this is not json")
        rc, out = run([str(REPO_ROOT / ".github/workflows/ci.yml"), "--allowlist", str(bad)])
        arms.append((
            "PLANTED",
            "an unreadable allowlist is ERROR(instrument), not an empty one",
            rc == EXIT_ERROR_INSTRUMENT and "allowlist_unreadable" in out,
            "reading it as empty would turn every recorded judgement into a fresh red, "
            "and the fastest way past that is to delete the judgement",
        ))

        rc, out = run([])
        arms.append((
            "PLANTED",
            "no arguments prints usage and exits 2, never 0",
            rc == EXIT_USAGE,
            "a screen invoked with nothing must not report a clean tree",
        ))

        # ---- PLANTED: the allowlist works, and only where it is aimed ---------
        allow = write(
            tmp,
            "allow.json",
            '{"names": {"BUILD_SKIPPED": "recorded judgement, for the control"}, '
            '"steps": {}}',
        )
        rc, out = run([str(tmp / "bp2-dormant.yml"), "--allowlist", str(allow)])
        arms.append((
            "PLANTED",
            "an allowlisted name is reported as allowlisted, not silently dropped",
            rc == EXIT_PASS and "BP2 allowlisted" in out,
            "a waiver that leaves no line in the output is a waiver nobody re-reads",
        ))

    width = max(len(name) for _, name, _, _ in arms)
    failures = 0
    for prov, name, ok, note in arms:
        mark = "ok  " if ok else "FAIL"
        suffix = f"   ({note})" if note else ""
        print(f"  [{prov:<8}] {mark}  {name.ljust(width)}{suffix}")
        if not ok:
            failures += 1

    counts = {"LIVE": 0, "REPLAYED": 0, "PLANTED": 0}
    for prov, _, _, _ in arms:
        counts[prov] += 1
    print()
    print(
        f"NEGATIVE-CONTROL: {counts['LIVE']} LIVE / {counts['REPLAYED']} REPLAYED / "
        f"{counts['PLANTED']} PLANTED."
    )
    print(
        "NEGATIVE-CONTROL: the REPLAYED arms are the load-bearing ones. Every PLANTED arm "
        "is a defect written by the person who wrote the rule that catches it; the "
        f"REPLAYED arms are the defect as it really stood on main at {HISTORICAL_REF}, "
        "retrieved with `git show` rather than reconstructed from memory."
    )
    if failures:
        print(f"NEGATIVE-CONTROL: FAIL(condition=arm_did_not_fire) — {failures} arm(s).")
        return 1
    print("NEGATIVE-CONTROL: PASS — every arm fired as declared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
