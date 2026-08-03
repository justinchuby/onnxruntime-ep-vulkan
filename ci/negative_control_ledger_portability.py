#!/usr/bin/env python3
"""Negative control for ``check_ledger_portability`` — arms counted by provenance.

A check that has only ever been observed passing is one step from a check that cannot
fail. This exercises every condition and every UNOBSERVABLE path, and it reports the
arms **by where the input came from** rather than as a single total, because those are
not the same evidence:

* ``LIVE``     — a run that actually happened on real hardware in the course of this
                 work, not constructed to make the check fire.
* ``REPLAYED`` — a real artifact from a run someone else made earlier, re-fed here.
* ``PLANTED``  — text I wrote to exercise a path. Proves the path is wired; proves
                 nothing about whether the path is ever reached in the wild.

Only the LIVE arms are evidence that the condition occurs. Everything else is evidence
that the code runs. ``check_ledger_portability`` currently stands at **3 LIVE / 0
REPLAYED / 7 PLANTED**, and all three of its conditions have a LIVE arm — which is
unusual here and only because Linux was genuinely broken when I looked.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
SCRIPT = CI_DIR / "check_ledger_portability.py"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_ERROR_INSTRUMENT = 4

LIVE_LINUX = REPO_ROOT / "bench" / "results" / "linux_lavapipe_probe.txt"
LIVE_LINUX_TESTS = REPO_ROOT / "bench" / "results" / "linux_lavapipe_optests.txt"
LIVE_LOADER = REPO_ROOT / "bench" / "results" / "linux_lavapipe_loader_probe.txt"
LIVE_WINDOWS = REPO_ROOT / "bench" / "results" / "windows_nvidia_probe_control.txt"

FAULT_TEXT = (
    '[vulkan-ep] WARN: [VulkanEP] proof ledger fault: ledger entry for '
    '"ai.onnx::Tanh/6+/f32>f32/ew_unary_tanh_f32/static/n1" was proven against shader '
    "digest 16a64dbeb2dbf63d but this build's modules [\"ew_unary_tanh_f32\"] hash to "
    "8f87214a7ca41ca9."
)
CLAIMS_NOTHING_TEXT = (
    "[vulkan-ep] INFO: [§8.9.7] this session claims 0/1 nodes; all work runs on the CPU EP."
)
CLAIMS_SOMETHING_TEXT = (
    "[vulkan-ep] INFO: session claims 1 proven form(s) [§8.9.7]: com.microsoft::MatMulNBits x1"
)
ABSENT_TEXT = (
    "SKIPPED [1] tests/ops/test_elementwise.py:231: No Vulkan device available — either "
    "no ICD is installed or all devices failed the capability gate."
)
GATE_PASS_TEXT = (
    "Device 0 [Vulkan enum index 0]: llvmpipe (LLVM 20.1.2, 256 bits) [Vulkan 1.4.318] "
    " — gate PASS"
)


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

    with tempfile.TemporaryDirectory(prefix="ledgerctl-") as td:
        tmp = Path(td)

        # ---- LIVE ------------------------------------------------------------
        if LIVE_LINUX.exists() and LIVE_LOADER.exists():
            rc, out = run(
                [
                    "--run-log", str(LIVE_LINUX),
                    "--device-lane",
                    "--loader-artifact", str(LIVE_LOADER),
                ]
            )
            arms.append((
                "LIVE",
                "linux lavapipe probe goes red on ledger_fault",
                rc == EXIT_FAIL_CONDITION and "ledger_fault" in out,
                "the run that produced this exited 0",
            ))
        else:
            arms.append(("LIVE", "linux lavapipe probe artifact present", False,
                         f"missing {LIVE_LINUX}"))

        if LIVE_LINUX_TESTS.exists() and LIVE_LOADER.exists():
            rc, out = run(
                [
                    "--run-log", str(LIVE_LINUX_TESTS),
                    "--device-lane",
                    "--loader-artifact", str(LIVE_LOADER),
                ]
            )
            arms.append((
                "LIVE",
                "linux op-test run goes red on device_absence_misnamed",
                rc == EXIT_FAIL_CONDITION and "device_absence_misnamed" in out,
                "'2 passed, 36 skipped' — the lane would have been green",
            ))
        else:
            arms.append(("LIVE", "linux op-test artifact present", False,
                         f"missing {LIVE_LINUX_TESTS}"))

        if LIVE_WINDOWS.exists():
            rc, out = run(["--run-log", str(LIVE_WINDOWS), "--device-lane"])
            arms.append((
                "LIVE",
                "windows nvidia control stays green",
                rc == EXIT_PASS,
                "same commit, same ledger, different toolchain",
            ))
        else:
            arms.append(("LIVE", "windows control artifact present", False,
                         f"missing {LIVE_WINDOWS}"))

        # ---- PLANTED ---------------------------------------------------------
        p = write(tmp, "fault.txt", FAULT_TEXT + "\n" + CLAIMS_NOTHING_TEXT + "\n")
        rc, out = run(["--run-log", str(p), "--device-lane"])
        arms.append(("PLANTED", "synthetic ledger fault fires", rc == EXIT_FAIL_CONDITION, ""))

        p = write(tmp, "nothing.txt", CLAIMS_NOTHING_TEXT + "\n")
        rc, out = run(["--run-log", str(p), "--device-lane"])
        arms.append(("PLANTED", "claims 0/N fires on a declared device lane",
                     rc == EXIT_FAIL_CONDITION and "claimed_nothing" in out, ""))

        rc, out = run(["--run-log", str(p)])
        arms.append(("PLANTED", "claims 0/N is UNOBSERVABLE without --device-lane",
                     rc == EXIT_PASS and "UNOBSERVABLE" in out,
                     "a build-only run claims nothing correctly"))

        p = write(tmp, "absent.txt", ABSENT_TEXT + "\n")
        rc, out = run(["--run-log", str(p), "--device-lane"])
        arms.append(("PLANTED",
                     "device absence is UNOBSERVABLE without a loader artifact",
                     rc == EXIT_PASS and "UNOBSERVABLE" in out,
                     "no ICD at all looks identical without it"))

        g = write(tmp, "gate.txt", GATE_PASS_TEXT + "\n")
        rc, out = run(["--run-log", str(p), "--device-lane", "--loader-artifact", str(g)])
        arms.append(("PLANTED", "device absence fires once a gate PASS is supplied",
                     rc == EXIT_FAIL_CONDITION and "device_absence_misnamed" in out, ""))

        p = write(tmp, "clean.txt", CLAIMS_SOMETHING_TEXT + "\n")
        rc, out = run(["--run-log", str(p), "--device-lane", "--loader-artifact", str(g)])
        arms.append(("PLANTED", "a clean device run stays green with all conditions armed",
                     rc == EXIT_PASS, ""))

        rc, out = run(["--run-log", str(tmp / "does-not-exist.txt"), "--device-lane"])
        arms.append(("PLANTED", "a missing artifact is ERROR(instrument), not a detection",
                     rc == EXIT_ERROR_INSTRUMENT and "artifact_unreadable" in out, ""))

        rc, out = run([])
        arms.append(("PLANTED", "no run named is ERROR(instrument), never PASS",
                     rc == EXIT_ERROR_INSTRUMENT and "no_run_named" in out,
                     "'I was not given a run' is not 'the run was clean'"))

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
        "NEGATIVE-CONTROL: only the LIVE arms are evidence that the condition occurs. "
        "The PLANTED arms are evidence that the code runs, which is a weaker claim and "
        "is not interchangeable with it."
    )
    if failures:
        print(f"NEGATIVE-CONTROL: FAIL(condition=arm_did_not_fire) — {failures} arm(s).")
        return 1
    print("NEGATIVE-CONTROL: PASS — every arm fired as declared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
