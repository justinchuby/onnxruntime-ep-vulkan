#!/usr/bin/env python3
"""Negative control for ``check_suite_productivity`` — arms counted by provenance.

A guard never seen failing has no demonstrated positive state. This exercises every
condition and every instrument path of ``ci/check_suite_productivity.py`` and reports the
arms by **where the input came from**, because those are not the same evidence:

* ``LIVE``     — a run that actually happened here, in the course of this work, whose log
                 was produced by really running the suite. Not text written to make the
                 check fire.
* ``REPLAYED`` — a real captured log from an earlier run by someone else, re-fed here.
* ``PLANTED``  — text written to exercise a path. Proves the path is wired; proves nothing
                 about whether that shape occurs in the wild.

The two LIVE arms are the ones the brief demanded and they are a matched pair from the
same scratch virtual environment, built **without** the optional ``onnx-shape-inference``
package — a real deletion, not an import hook:

* ``ARM-A-prefix-nodep.log``  — the tree as it stood before this round. The module built
  its report at import time, so pytest printed ``Interrupted: 1 error during collection``
  and the whole ``tests/ops`` directory asserted nothing. The check goes **red**.
* ``ARM-B-postfix-nodep.log`` — the same environment, same missing package, after the
  report was made lazy. 665 collected, 316 executed, and the two tests that genuinely need
  the package fail *individually*. The check goes **green** — which matters as much as the
  red arm, because a check that fails on the repaired state is a lock on the broken one.

The REPLAYED arm is the more uncomfortable one: ``bench/results/linux_lavapipe_optests.txt``
is a real log from a real green run reading ``2 passed, 36 skipped``. That run was a
deliberate partial selection, so it is **not** an accusation about that run — but it is a
log that a whole-directory step could have produced, and it is the exact text that would
have read as a healthy lane to every instrument this project had before today.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
SCRIPT = CI_DIR / "check_suite_productivity.py"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

LIVE_DIR = REPO_ROOT / "bench" / "results" / "link-suite-productivity"
LIVE_PREFIX = LIVE_DIR / "ARM-A-prefix-nodep.log"
LIVE_POSTFIX = LIVE_DIR / "ARM-B-postfix-nodep.log"
REPLAY_PARTIAL = REPO_ROOT / "bench" / "results" / "linux_lavapipe_optests.txt"

SUITE = "tests/ops"
SINGLE = "tests/ops/test_claim_diagnostics.py::test_no_vulkan_icd_falls_back_to_cpu"


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


def main() -> int:
    arms: list[tuple[str, str, bool, str]] = []  # (provenance, name, ok, note)

    with tempfile.TemporaryDirectory(prefix="suiteprod-") as td:
        tmp = Path(td)

        # ---- LIVE ------------------------------------------------------------
        if LIVE_PREFIX.exists():
            rc, out = run(["--suite", SUITE, str(LIVE_PREFIX)])
            arms.append((
                "LIVE",
                "real collection abort (dep really absent) goes red",
                rc == EXIT_FAIL_CONDITION and "collection_error" in out,
                "665 collected -> 0; the directory asserted nothing",
            ))
        else:
            arms.append(("LIVE", "pre-fix live log present", False, f"missing {LIVE_PREFIX}"))

        if LIVE_POSTFIX.exists():
            rc, out = run(["--suite", SUITE, str(LIVE_POSTFIX)])
            arms.append((
                "LIVE",
                "same environment after the fix stays green",
                rc == EXIT_PASS and "PASS" in out,
                "the repaired state must not be locked out by its own control",
            ))
        else:
            arms.append(("LIVE", "post-fix live log present", False, f"missing {LIVE_POSTFIX}"))

        # ---- LIVE (libtest) ---------------------------------------------------
        live_cargo = LIVE_DIR / "cargo-test-lib-windows.log"
        if live_cargo.exists():
            rc, out = run([
                "--suite", "cargo test --lib", "--harness", "libtest",
                "--lane", "build-test-windows", str(live_cargo),
            ])
            arms.append((
                "LIVE",
                "the real Windows cargo test --lib run stays green",
                rc == EXIT_PASS,
                "510 passed / 0 failed / 4 ignored, captured on this tree",
            ))
        else:
            arms.append(("LIVE", "windows cargo test --lib log present", False,
                         f"missing {live_cargo}"))

        # ---- REPLAYED --------------------------------------------------------
        if REPLAY_PARTIAL.exists():
            rc, out = run(["--suite", SUITE, str(REPLAY_PARTIAL)])
            arms.append((
                "REPLAYED",
                "a real '2 passed, 36 skipped' green run trips the collected floor",
                rc == EXIT_FAIL_CONDITION and "collected_below_floor" in out,
                "that run exited 0 and no instrument on this project could tell",
            ))
        else:
            arms.append(("REPLAYED", "historical lavapipe op-test log present", False,
                         f"missing {REPLAY_PARTIAL}"))

        # ---- PLANTED: the conditions -----------------------------------------
        p = write(tmp, "all-skipped.log", "665 skipped in 12.00s\n")
        rc, out = run(["--suite", SUITE, str(p)])
        arms.append(("PLANTED", "every test skipped is asserted_nothing, not a pass",
                     rc == EXIT_FAIL_CONDITION and "asserted_nothing" in out,
                     "exit code 0 from pytest; the quiet form of the same defect"))

        p = write(tmp, "no-tests.log", "no tests ran in 0.01s\n")
        rc, out = run(["--suite", SUITE, str(p)])
        arms.append(("PLANTED", "a selector that matches nothing is no_tests_ran",
                     rc == EXIT_FAIL_CONDITION and "no_tests_ran" in out, ""))

        p = write(tmp, "single-empty.log", "no tests ran in 0.01s\n")
        rc, out = run(["--suite", SINGLE, str(p)])
        arms.append(("PLANTED", "the single-test step is held to a floor of 1",
                     rc == EXIT_FAIL_CONDITION and "no_tests_ran" in out,
                     "a renamed node id makes this step run nothing"))

        p = write(tmp, "single-ok.log", "1 passed in 0.40s\n")
        rc, out = run(["--suite", SINGLE, str(p)])
        arms.append(("PLANTED", "the single-test step passes on one executed test",
                     rc == EXIT_PASS, "the floor is not set so high it can never be met"))

        p = write(tmp, "shrunk.log", "400 passed, 100 skipped in 30.00s\n")
        rc, out = run(["--suite", SUITE, str(p)])
        arms.append(("PLANTED", "a shrunken collection trips collected_below_floor",
                     rc == EXIT_FAIL_CONDITION and "collected_below_floor" in out,
                     "500 accounted for out of 665; a whole file stopped importing"))

        p = write(tmp, "skipped-up.log", "200 passed, 465 skipped in 30.00s\n")
        rc, out = run(["--suite", SUITE, "--lane", "build-test-linux", str(p)])
        arms.append(("PLANTED", "full collection but fewer executed trips executed_below_floor",
                     rc == EXIT_FAIL_CONDITION and "executed_below_floor" in out,
                     "collection intact, work quietly moved into skips"))

        p = write(tmp, "healthy.log", "50 failed, 272 passed, 343 skipped in 300.00s\n")
        rc, out = run(["--suite", SUITE, "--lane", "build-test-linux", str(p)])
        arms.append(("PLANTED", "a red-but-productive suite is NOT this check's failure",
                     rc == EXIT_PASS,
                     "50 real failures are evidence; this check is about the absence of it"))

        p = write(tmp, "xfail.log", "600 xfailed, 65 skipped in 30.00s\n")
        rc, out = run(["--suite", SUITE, "--lane", "build-test-linux", str(p)])
        arms.append(("PLANTED", "xfailed counts as executed",
                     rc == EXIT_PASS,
                     "an xfail test runs and its failure is checked; a skip does not"))

        # ---- PLANTED: the libtest harness ------------------------------------
        p = write(tmp, "cargo-empty.log",
                  "running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; "
                  "0 measured; 0 filtered out; finished in 0.00s\n")
        rc, out = run(["--suite", "cargo test --lib", "--harness", "libtest", str(p)])
        arms.append(("PLANTED", "cargo test with zero tests is asserted_nothing",
                     rc == EXIT_FAIL_CONDITION and "asserted_nothing" in out,
                     "libtest prints `test result: ok.` and exits 0; there is no --strict"))

        p = write(tmp, "cargo-filtered.log",
                  "running 3 tests\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; "
                  "0 measured; 507 filtered out; finished in 0.10s\n")
        rc, out = run(["--suite", "cargo test --lib", "--harness", "libtest", str(p)])
        arms.append(("PLANTED", "a stray --lib filter leaving 3 of 510 trips the floor",
                     rc == EXIT_FAIL_CONDITION and "executed_below_floor" in out,
                     "filtered out is a deselection, not work"))

        p = write(tmp, "cargo-nothing.log", "   Compiling onnxruntime_vulkan_ep v0.1.0\n")
        rc, out = run(["--suite", "cargo test --lib", "--harness", "libtest", str(p)])
        arms.append(("PLANTED", "a cargo log with no `test result:` block is ERROR",
                     rc == EXIT_ERROR_INSTRUMENT and "summary_not_found" in out, ""))

        # ---- PLANTED: instrument paths ---------------------------------------
        rc, out = run(["--suite", SUITE, str(tmp / "absent.log")])
        arms.append(("PLANTED", "a missing log with no marker is ERROR(instrument)",
                     rc == EXIT_ERROR_INSTRUMENT and "log_not_captured" in out, ""))

        marker = write(tmp, ".lane-reached", "")
        rc, out = run(["--suite", SUITE, f"--lane-marker={marker}", str(tmp / "absent.log")])
        arms.append(("PLANTED",
                     "missing log WITH the marker present is still ERROR(instrument)",
                     rc == EXIT_ERROR_INSTRUMENT and "log_not_captured" in out,
                     "the lane reached the step and captured nothing"))

        rc, out = run([
            "--suite", SUITE,
            f"--lane-marker={tmp / 'never-written'}",
            str(tmp / "absent.log"),
        ])
        arms.append(("PLANTED",
                     "missing log AND missing marker declines to add a second red",
                     rc == EXIT_PASS and "lane_did_not_reach_evidence" in out,
                     "the lane is already red for the reason that stopped it"))

        p = write(tmp, "no-summary.log", "some ORT chatter\nand more\n")
        rc, out = run(["--suite", SUITE, str(p)])
        arms.append(("PLANTED", "a log with no terminal summary is ERROR(instrument)",
                     rc == EXIT_ERROR_INSTRUMENT and "summary_not_found" in out,
                     "UNOBSERVABLE is not zero and must not read as clean"))

        p = write(tmp, "weird.log", "3 passed, 4 flooped in 1.00s\n")
        rc, out = run(["--suite", SUITE, str(p)])
        arms.append(("PLANTED", "an unknown outcome word is ERROR(instrument), not a guess",
                     rc == EXIT_ERROR_INSTRUMENT and "unrecognised_outcome_word" in out,
                     "a number read off a partly-understood line is a count without text"))

        p = write(tmp, "ok.log", "665 passed in 30.00s\n")
        rc, out = run(["--suite", "tests/nowhere", str(p)])
        arms.append(("PLANTED", "a suite with no floor entry is ERROR, never a default pass",
                     rc == EXIT_ERROR_INSTRUMENT and "suite_has_no_floor" in out,
                     "an unclassified suite is answerable to nothing"))

        bad = write(tmp, "floors.json", "{not json")
        rc, out = run(["--suite", SUITE, "--floors", str(bad), str(p)])
        arms.append(("PLANTED", "an unreadable floors file is ERROR, never a default pass",
                     rc == EXIT_ERROR_INSTRUMENT and "floors_unreadable" in out, ""))

        rc, out = run([])
        arms.append(("PLANTED", "no log named prints usage and does not pass",
                     rc == EXIT_USAGE, "'I was not given a run' is not 'the run was clean'"))

        # ---- PLANTED: the floors cannot be waived at the command line --------
        rc, out = run(["--suite", SUITE, "--min-collected", "1", str(
            write(tmp, "shrunk2.log", "10 passed in 1.00s\n"))])
        arms.append(("PLANTED", "there is no flag that lowers a floor",
                     rc == EXIT_USAGE,
                     "argparse rejects it; lowering a floor is a tracked-file edit"))

        # ---- PLANTED: the floors file is itself well-formed -------------------
        try:
            data = json.loads((CI_DIR / "suite_floor.json").read_text(encoding="utf-8"))
            ok = (
                isinstance(data.get("suites"), dict)
                and all(
                    "provenance" in v and (v.get("min_collected") or v.get("min_executed_by_lane"))
                    for v in data["suites"].values()
                )
            )
            note = "every entry carries a floor and a provenance"
        except Exception as exc:  # noqa: BLE001
            ok, note = False, repr(exc)
        arms.append(("PLANTED", "every floor entry states where its number came from", ok, note))

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
        "NEGATIVE-CONTROL: the LIVE pair is a matched control — the SAME environment with "
        "the SAME package missing, red before the fix and green after it. A red arm alone "
        "would only prove the check can fail; the green arm is what proves it is not a "
        "lock on the broken state."
    )
    if failures:
        print(f"NEGATIVE-CONTROL: FAIL(condition=arm_did_not_fire) — {failures} arm(s).")
        return 1
    print("NEGATIVE-CONTROL: PASS — every arm fired as declared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
