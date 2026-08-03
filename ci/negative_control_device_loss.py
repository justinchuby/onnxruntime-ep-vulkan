#!/usr/bin/env python3
"""Negative control for ci/check_device_loss.py — does the screen actually go red?

A screen that has only ever been observed passing is one step from a screen that cannot
fail. This file constructs inputs on which `check_device_loss.py` MUST report a specific
condition, and asserts it reports that condition and not merely "some failure". §10.0.1
R9: evidence scales only with falsifying instruments.

Arm provenance is stated per arm and never blurred, because "we tested it" reads the same
whether the input was real or synthesised and the two are worth very different amounts:

  REPLAYED  the input is a real artifact of a real incident, fed to the screen after the
            fact. Proves the screen would have caught it. Does not prove the screen fires
            during a live run.
  LIVE      the screen found this on a file it was not written against, in a normal run.
  PLANTED   the input was synthesised here to exercise one rule. Proves the rule fires.
            Proves nothing about whether the rule's event ever occurs in reality.

Run: python ci/negative_control_device_loss.py     (exit 0 = every arm fired)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK = REPO_ROOT / "ci" / "check_device_loss.py"
TANK_ARTIFACT = REPO_ROOT / "bench" / "results" / "ctx512_device_lost.txt"
TRINITY_LOG = REPO_ROOT / "bench" / "results" / "trinity-suite-dev1.log"

CLEAN_LOG = """\
[vulkan-ep] INFO: VulkanExecutionProvider: device 0 (NVIDIA GeForce RTX 4060)
[vulkan-ep] INFO: claimed fused subgraph #1 (355 nodes)
2026-08-02 12:00:00 [I:onnxruntime:, session_state.cc] Node placements
2026-08-02 12:00:00 [V:onnxruntime:, session_state.cc]  VulkanExecutionProvider: [Fused]
argmax 30751
all 65 outputs bit-identical to the CPU reference
EXIT = 0
"""


def run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(CHECK), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


class Arms:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, bool, str]] = []

    def arm(self, name: str, provenance: str, ok: bool, note: str) -> None:
        self.results.append((name, provenance, ok, note))

    def expect_condition(
        self, name: str, provenance: str, args: list[str], condition: str
    ) -> None:
        code, out = run(args)
        want = f"FAIL(condition={condition})"
        ok = code == 1 and want in out
        self.expect_note = ""
        self.arm(
            name,
            provenance,
            ok,
            f"exit={code} " + (f"saw {want}" if ok else f"did NOT see {want}"),
        )

    def expect_reported(
        self, name: str, provenance: str, args: list[str], condition: str
    ) -> None:
        """Red for any condition, with `condition` named among them."""
        code, out = run(args)
        ok = code == 1 and f"[{condition}]" in out
        self.arm(
            name,
            provenance,
            ok,
            f"exit={code} " + (f"reported [{condition}]" if ok else f"no [{condition}]"),
        )

    def expect_pass(self, name: str, provenance: str, args: list[str]) -> None:
        code, out = run(args)
        ok = code == 0 and ": PASS" in out
        self.arm(name, provenance, ok, f"exit={code} " + ("PASS" if ok else "not PASS"))

    def expect_error(
        self, name: str, provenance: str, args: list[str], instrument: str
    ) -> None:
        code, out = run(args)
        want = f"ERROR(instrument={instrument})"
        ok = code == 4 and want in out
        self.arm(
            name,
            provenance,
            ok,
            f"exit={code} " + (f"saw {want}" if ok else f"did NOT see {want}"),
        )


def main() -> int:
    arms = Arms()
    tmp = Path(tempfile.mkdtemp(prefix="device-loss-control-"))

    # --- The red arm that matters, on real evidence -------------------------
    if TANK_ARTIFACT.exists():
        arms.expect_condition(
            "real device loss (Tank, ctx 512, vkWaitForFences)",
            "REPLAYED",
            [str(TANK_ARTIFACT)],
            "device_lost_reported",
        )
        arms.expect_reported(
            "the same log's ORT fallback announcement",
            "REPLAYED",
            [str(TANK_ARTIFACT)],
            "runtime_fallback_announced",
        )
        arms.expect_reported(
            "_verdict.FATAL_LOG_MARKERS misses that log's real line",
            "REPLAYED",
            [str(TANK_ARTIFACT)],
            "marker_list_misses_real_line",
        )
    else:
        arms.arm(
            "real device loss",
            "REPLAYED",
            False,
            f"{TANK_ARTIFACT} is missing — the red arm cannot be run, which is an "
            "instrument outage in this control, not a pass",
        )

    if TRINITY_LOG.exists():
        arms.expect_condition(
            "second device loss, dev1 vkQueueSubmit, 2026-07-31",
            "LIVE (found by the screen on a file it was not written against)",
            [str(TRINITY_LOG)],
            "device_lost_reported",
        )

    # --- The green arm ------------------------------------------------------
    clean = tmp / "clean.log"
    clean.write_text(CLEAN_LOG, encoding="utf-8")
    arms.expect_pass("a run that did not lose the device", "PLANTED", [str(clean)])

    # --- The structural rule, which needs no text at all --------------------
    short = tmp / "short.json"
    short.write_text(
        json.dumps({"points": [{"iters": 25, "compute_calls": 9}]}), encoding="utf-8"
    )
    arms.expect_condition(
        "declared 25 inferences, observed 9, no log text anywhere",
        "PLANTED",
        [str(short)],
        "observation_ended_early",
    )

    whole = tmp / "whole.json"
    whole.write_text(
        json.dumps({"points": [{"iters": 25, "compute_calls": 25}]}), encoding="utf-8"
    )
    arms.expect_pass("declared 25, observed 25", "PLANTED", [str(whole)])

    inflight = tmp / "inflight.json"
    inflight.write_text(json.dumps({"uploads": 5, "readbacks": 4}), encoding="utf-8")
    arms.expect_condition(
        "uploads == readbacks + 1 (an inference caught in flight)",
        "PLANTED",
        [str(inflight)],
        "observation_ended_early",
    )

    rejected = tmp / "rejected.json"
    rejected.write_text(
        json.dumps({"rejected_points": [{"iters": 25, "compute_calls": 9}]}),
        encoding="utf-8",
    )
    arms.expect_pass(
        "the SAME truncation, but the producer already rejected it", "PLANTED", [str(rejected)]
    )

    # --- The reach arm: the evidence for the extents ruling ------------------
    # A device loss the EP reports and ORT never announces. check_fatal_log reads
    # ORT's announcement, so it is green here; this check is red. That difference is
    # the whole reason both exist, and it is demonstrated rather than argued.
    ep_only = tmp / "ep_only.log"
    ep_only.write_text(
        "[vulkan-ep] ERROR: vkQueueSubmit failed: The logical device has been lost.\n"
        "inference 3 of 25\n",
        encoding="utf-8",
    )
    arms.expect_condition(
        "EP reports the loss, ORT announces nothing (reach beyond check_fatal_log)",
        "PLANTED",
        [str(ep_only)],
        "device_lost_reported",
    )
    fatal_log = REPO_ROOT / "ci" / "check_fatal_log.py"
    if fatal_log.exists():
        proc = subprocess.run(
            [sys.executable, str(fatal_log), str(ep_only)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        arms.arm(
            "  ...and check_fatal_log is NOT red on that same file",
            "PLANTED",
            proc.returncode != 1,
            f"check_fatal_log exit={proc.returncode} (1 would mean it caught it too, and "
            "this check would then have no reach of its own to justify it)",
        )

    # --- Instrument-outage arms: an outage is never a pass ------------------
    arms.expect_error("no path named at all", "PLANTED", [], "no_paths_given")
    empty = tmp / "empty_dir"
    empty.mkdir()
    arms.expect_error(
        "a directory with nothing readable in it",
        "PLANTED",
        [str(empty)],
        "nothing_scanned",
    )
    arms.expect_error(
        "a path that does not exist",
        "PLANTED",
        [str(tmp / "absent.log")],
        "nothing_scanned",
    )

    # --- Report -------------------------------------------------------------
    width = max(len(n) for n, _, _, _ in arms.results)
    print("\nNEGATIVE CONTROL: ci/check_device_loss.py")
    print("=" * (width + 34))
    failed = 0
    for name, provenance, ok, note in arms.results:
        status = "FIRED" if ok else "DID NOT FIRE"
        print(f"  [{status:>12}] {name:<{width}}  {provenance}")
        print(f"                 {note}")
        if not ok:
            failed += 1
    live = sum(1 for _, p, _, _ in arms.results if p.startswith("LIVE"))
    replayed = sum(1 for _, p, _, _ in arms.results if p.startswith("REPLAYED"))
    planted = sum(1 for _, p, _, _ in arms.results if p.startswith("PLANTED"))
    print(
        f"\n  provenance: {live} LIVE, {replayed} REPLAYED, {planted} PLANTED "
        f"of {len(arms.results)} arms."
    )
    print(
        "  A PLANTED arm proves the rule fires on an input built to make it fire. It does "
        "not evidence that the rule's event occurs in reality; only the LIVE and REPLAYED "
        "arms do that."
    )
    if failed:
        print(f"\nNEGATIVE CONTROL: FAIL — {failed} arm(s) did not fire.")
        print(
            "An arm that does not fire means the screen cannot detect the thing that arm "
            "describes. Quote the arm, not the count."
        )
        return 1
    print("\nNEGATIVE CONTROL: PASS — every arm fired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
