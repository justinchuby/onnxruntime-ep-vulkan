#!/usr/bin/env python3
"""Negative control for ``ci/check_flake_witness.py`` — demand red on the shape it names.

A flake witness on a repo with no reproducing flake is silent, and silence from a check
is indistinguishable from silence from no check. This file makes the witness go red on
demand so that its quiet is worth something.

ARMS
----

* **REPLAYED** — real libtest and pytest logs captured on this box, with a real failing
  name lifted out of one of them. The failing *text* is not invented; only the pairing
  of two runs at one commit is arranged.
* **PLANTED** — synthesised logs for the shapes no real capture happened to contain:
  a torn ledger line, a NOT_FAILED that came from a much smaller run (INCOMPARABLE), a
  log with no terminal summary (UNPARSED), a ledger too short to conclude from.
* **LIVE** — the real ledger produced by ci/link-linux-repro/link_linux_flake_hunt.sh,
  which must NOT report an intermittent, because in those runs nothing was intermittent.
  A control whose arms only ever go red proves the check is a constant just as surely as
  one whose arms only ever go green.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CHECK = HERE / "check_flake_witness.py"

REPLAYED = "REPLAYED"
PLANTED = "PLANTED"
LIVE = "LIVE"


def run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(CHECK), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    return proc.returncode, proc.stdout + proc.stderr


LIBTEST_GREEN = """\
   Compiling onnxruntime-ep-vulkan v0.28.0
     Running unittests src/lib.rs (target/debug/deps/onnxruntime_ep_vulkan-1)

running 3 tests
test vk::barrier::tests::backend_probe_writes_legacy_token ... ok
test vk::barrier::tests::a_second_test ... ok
test ops::norm::tests::a_third_test ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s
"""

LIBTEST_RED = """\
   Compiling onnxruntime-ep-vulkan v0.28.0
     Running unittests src/lib.rs (target/debug/deps/onnxruntime_ep_vulkan-1)

running 3 tests
test vk::barrier::tests::backend_probe_writes_legacy_token ... FAILED
test vk::barrier::tests::a_second_test ... ok
test ops::norm::tests::a_third_test ... ok

failures:

---- vk::barrier::tests::backend_probe_writes_legacy_token stdout ----
thread 'vk::barrier::tests::backend_probe_writes_legacy_token' panicked at rust/src/vk/barrier.rs:1:1:
assertion `left == right` failed

test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s
"""

LIBTEST_RED_TINY = """\
     Running unittests src/lib.rs (target/debug/deps/onnxruntime_ep_vulkan-1)

running 1 test
test vk::barrier::tests::backend_probe_writes_legacy_token ... FAILED

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 500 filtered out; finished in 0.01s
"""

PYTEST_RED = """\
=================================== FAILURES ===================================
______________ test_criterion_10_three_consecutive_attributed_match ____________
E   AssertionError: DIVERGENT
=========================== short test summary info ============================
FAILED tests/ops/test_criterion10.py::test_criterion_10_three_consecutive_attributed_match - AssertionError: DIVERGENT
========================= 1 failed, 40 passed in 12.00s =========================
"""

PYTEST_GREEN = """\
=========================== short test summary info ============================
========================= 41 passed in 11.00s ==================================
"""

PYTEST_NO_SUMMARY = """\
collecting ...
tests/ops/test_thing.py ....
"""


def _scratch_root() -> Path:
    """Scratch stays inside the checkout, never in a system temp dir.

    A control that writes outside the workspace is a control whose leftovers nobody
    ever sees; keeping them under bench/results means they land where the lane's own
    evidence upload can pick them up if one ever survives a crash.
    """
    root = REPO / "bench" / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


def main() -> int:
    fired: list[tuple[str, str, str]] = []
    failures: list[str] = []

    def arm(kind: str, name: str, args: list[str], want_code: int, want_text: str, tmp: Path) -> None:
        code, out = run(args)
        ok = code == want_code and want_text in out
        if ok:
            fired.append((kind, name, f"exit {code}"))
        else:
            failures.append(
                f"[{kind}] {name}: wanted exit {want_code} containing {want_text!r}, "
                f"got exit {code}.\n--- output ---\n{out.strip()[:2000]}\n"
            )

    with tempfile.TemporaryDirectory(prefix="flake-witness-control-", dir=str(_scratch_root())) as td:
        tmp = Path(td)

        # ---- 1. the core join: one id, one commit, one suite, both polarities -------
        led = tmp / "ledger-core.jsonl"
        red = write(tmp, "lib-red.log", LIBTEST_RED)
        green = write(tmp, "lib-green.log", LIBTEST_GREEN)
        run(["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "deadbeefcafe",
             "--run-id", "r1", "--ledger", str(led), str(red)])
        arm(
            PLANTED,
            "same id fails in r1 and does not fail in r2 at one commit",
            ["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "deadbeefcafe",
             "--run-id", "r2", "--ledger", str(led), str(green)],
            1,
            "FLAKE-WITNESS: FAIL(condition=intermittent)",
            tmp,
        )

        # ---- 2. the commit is what separates a flake from a regression --------------
        led2 = tmp / "ledger-diff-commit.jsonl"
        run(["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "AAAA",
             "--run-id", "r1", "--ledger", str(led2), str(red)])
        arm(
            PLANTED,
            "the SAME id failing at one commit and passing at another is NOT a flake",
            ["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "BBBB",
             "--run-id", "r2", "--ledger", str(led2), str(green)],
            0,
            "FLAKE-WITNESS: PASS",
            tmp,
        )

        # ---- 3. lanes are not interchangeable --------------------------------------
        led3 = tmp / "ledger-diff-lane.jsonl"
        run(["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "CCCC",
             "--run-id", "r1", "--ledger", str(led3), str(red)])
        arm(
            PLANTED,
            "failing on linux and passing on windows is a portability finding, not a flake",
            ["--harness", "libtest", "--suite", "lib", "--lane", "windows", "--commit", "CCCC",
             "--run-id", "r2", "--ledger", str(led3), str(green)],
            0,
            "FLAKE-WITNESS: PASS",
            tmp,
        )

        # ---- 4. a NOT_FAILED from a much smaller run is not a sample ----------------
        led4 = tmp / "ledger-incomparable.jsonl"
        tiny = write(tmp, "lib-red-tiny.log", LIBTEST_RED_TINY)
        run(["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "DDDD",
             "--run-id", "r1", "--ledger", str(led4), str(tiny)])
        arm(
            PLANTED,
            "1-executed vs 3-executed is INCOMPARABLE, not intermittent",
            ["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "DDDD",
             "--run-id", "r2", "--ledger", str(led4), str(green)],
            0,
            "FLAKE-WITNESS: INCOMPARABLE",
            tmp,
        )

        # ---- 5. an unparsed log is UNOBSERVABLE, not clean --------------------------
        arm(
            PLANTED,
            "a pytest log with no terminal summary is an instrument error, not a PASS",
            ["--harness", "pytest", "--suite", "ops", "--ledger", str(tmp / "l5.jsonl"),
             str(write(tmp, "no-summary.log", PYTEST_NO_SUMMARY))],
            4,
            "FLAKE-WITNESS: ERROR(instrument=log_unparsed)",
            tmp,
        )

        # ---- 6. a missing log with no lane marker is a DECLINE, not a second red ----
        arm(
            PLANTED,
            "absent log + absent lane marker declines rather than piling on",
            ["--harness", "pytest", "--suite", "ops", "--ledger", str(tmp / "l6.jsonl"),
             "--lane-marker", str(tmp / "never-written"), str(tmp / "absent.log")],
            0,
            "FLAKE-WITNESS: DECLINED",
            tmp,
        )

        # ---- 7. a missing log WITH the marker present is an instrument failure ------
        marker = write(tmp, "reached", "")
        arm(
            PLANTED,
            "absent log + present lane marker is ERROR(instrument=log_absent)",
            ["--harness", "pytest", "--suite", "ops", "--ledger", str(tmp / "l7.jsonl"),
             "--lane-marker", str(marker), str(tmp / "absent.log")],
            4,
            "FLAKE-WITNESS: ERROR(instrument=log_absent)",
            tmp,
        )

        # ---- 8. a join over one run cannot conclude, and must not pretend to --------
        arm(
            PLANTED,
            "--require-history refuses a verdict the ledger is too short to support",
            ["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "EEEE",
             "--run-id", "r1", "--ledger", str(tmp / "l8.jsonl"), "--require-history", "5", str(green)],
            4,
            "FLAKE-WITNESS: ERROR(instrument=history_too_short)",
            tmp,
        )

        # ---- 9. a torn ledger line is reported, not swallowed, and not fatal --------
        led9 = tmp / "ledger-torn.jsonl"
        run(["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "FFFF",
             "--run-id", "r1", "--ledger", str(led9), str(green)])
        with led9.open("a", encoding="utf-8") as fh:
            fh.write('{"commit": "FFFF", "lane": "lin')  # killed mid-write
        arm(
            PLANTED,
            "a torn append-only line is counted and named, and does not lose the history",
            ["--harness", "libtest", "--suite", "lib", "--lane", "linux", "--commit", "FFFF",
             "--run-id", "r2", "--ledger", str(led9), "--no-append", str(green)],
            0,
            "torn ledger line(s) skipped",
            tmp,
        )

        # ---- 10. pytest failures are named even though pytest names no passes -------
        arm(
            PLANTED,
            "a pytest FAILED id is lifted into the tail block",
            ["--harness", "pytest", "--suite", "ops", "--lane", "windows", "--commit", "GGGG",
             "--run-id", "r1", "--ledger", str(tmp / "l10.jsonl"),
             str(write(tmp, "pytest-red.log", PYTEST_RED))],
            0,
            "FAILED  [windows]  tests/ops/test_criterion10.py::test_criterion_10_three_consecutive_attributed_match",
            tmp,
        )

        # ---- 11. and a pytest id that fails once and not again IS intermittent ------
        led11 = tmp / "l11.jsonl"
        run(["--harness", "pytest", "--suite", "ops", "--lane", "windows", "--commit", "HHHH",
             "--run-id", "r1", "--ledger", str(led11), str(tmp / "pytest-red.log")])
        # pytest names no passes, so the complement is supplied explicitly by a second
        # observation recorded as NOT_FAILED. This is the inference the docstring warns
        # about, made visible: without it, pytest flakes are invisible to the join.
        with led11.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "commit": "HHHH", "lane": "windows", "suite": "ops", "run_id": "r2",
                "harness": "pytest",
                "test_id": "tests/ops/test_criterion10.py::test_criterion_10_three_consecutive_attributed_match",
                "outcome": "NOT_FAILED", "executed": 41, "log": "synthetic",
            }, sort_keys=True) + "\n")
        arm(
            PLANTED,
            "a pytest id failing in r1 and not in r2 at one commit is intermittent",
            ["--harness", "pytest", "--suite", "ops", "--lane", "windows", "--commit", "HHHH",
             "--run-id", "r3", "--ledger", str(led11), "--no-append",
             str(write(tmp, "pytest-green.log", PYTEST_GREEN))],
            1,
            "THE COMMIT IS EXONERATED AND THE TEST IS NOT",
            tmp,
        )

        # ---- 12. LIVE: real captured runs must NOT claim an intermittent -----------
        # Two of the forty runs are TRACKED under ci/fixtures/flake-witness/ so this arm
        # exists on a hosted runner and not only on the box that produced them. The other
        # thirty-eight are used too when they are present locally; a control whose only
        # non-planted arm depends on an untracked directory is a control that quietly
        # becomes all-planted the moment it runs somewhere else.
        fixtures = sorted((HERE / "fixtures" / "flake-witness").glob("real-lib-linux-run-*.log"))
        local_runs = sorted((REPO / "bench" / "results" / "link-flake-witness" / "runs").glob("lib-linux-run*.log"))
        live_runs = fixtures + [p for p in local_runs if p.name not in {f.name for f in fixtures}]
        if not fixtures:
            failures.append(
                "[LIVE] ci/fixtures/flake-witness/real-lib-linux-run-*.log are missing. "
                "These are real captured `cargo test --lib` runs and they are tracked "
                "precisely so this arm cannot silently disappear."
            )
        if live_runs:
            live_led = tmp / "live.jsonl"
            code, out = run(
                ["--harness", "libtest", "--suite", "cargo test --lib", "--lane", "build-test-linux",
                 "--commit", "d46327b-worktree", "--run-id", "hunt", "--ledger", str(live_led),
                 *[str(p) for p in live_runs]]
            )
            if code == 0 and "FLAKE-WITNESS: PASS" in out:
                fired.append((LIVE, f"{len(live_runs)} real captured Linux run log(s) report no intermittent", "exit 0"))
            else:
                failures.append(
                    f"[{LIVE}] real run ledger: wanted exit 0 / PASS, got {code}.\n{out.strip()[:2000]}\n"
                )
            # and the same ledger, told a lie about one run, must go red
            recs = [json.loads(l) for l in live_led.read_text(encoding="utf-8").splitlines() if l.strip()]
            target = next((r for r in recs if r["outcome"] == "NOT_FAILED"), None)
            if target is not None:
                flipped = dict(target)
                flipped["outcome"] = "FAILED"
                flipped["run_id"] = "hunt#flipped"
                with live_led.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(flipped, sort_keys=True) + "\n")
                arm(
                    REPLAYED,
                    f"one flipped outcome in the real ledger reddens it ({target['test_id']})",
                    ["--harness", "libtest", "--suite", "cargo test --lib", "--lane", "build-test-linux",
                     "--commit", "d46327b-worktree", "--run-id", "q", "--ledger", str(live_led),
                     "--no-append", str(live_runs[0])],
                    1,
                    "FLAKE-WITNESS: FAIL(condition=intermittent)",
                    tmp,
                )

    print("FLAKE-WITNESS NEGATIVE CONTROL")
    print("=" * 78)
    for kind, name, detail in fired:
        print(f"  [{kind:<8}] {name} -> {detail}")
    if failures:
        print()
        for f in failures:
            print(f"  MISFIRE: {f}")
        print(f"\nFAIL — {len(failures)} arm(s) did not behave as specified.")
        return 1
    kinds = {k for k, _, _ in fired}
    print(f"\nPASS — {len(fired)} arm(s) fired as specified ({', '.join(sorted(kinds))}).")
    print(
        "What this proves: the witness distinguishes intermittent from regression, from "
        "portability difference, from a test that stopped running, and from a log it "
        "could not read. What it does not prove: that any real intermittent is currently "
        "reproducible on this box — 40 consecutive Linux runs of `cargo test --lib` at "
        "d46327b did not produce one, including the `backend_probe_*` test that was "
        "1-in-9 a round ago. That is a fact about the flake, not about the check."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
