"""Four-cell demonstration that the census stall detector separates hang from contention.

`test_stall_guard.py` proves the property against a fake clock, deterministically and in
milliseconds.  This probe proves it against the **real census on the real machine**, which
is the thing that was actually excluded from the suite.

The cells are the 2x2 of {census healthy, census hung} x {machine quiet, machine loaded}:

    ------------------+---------------------------+---------------------------
                      | machine quiet             | machine loaded
    ------------------+---------------------------+---------------------------
    census healthy    | PASS  (must not fire)     | PASS  (must not fire)
    census hung       | FAIL  (must fire)         | FAIL  (must fire)
    ------------------+---------------------------+---------------------------

A one-armed demonstration proves nothing — a detector that never fires passes the top row
too, and a detector that always fires passes the bottom row.  ``arms_must_differ`` in the
artifact records that the two rows produced different verdicts; the demonstration is void
without it.

The bottom-right cell is the one no wall-clock timeout can deliver.  A timeout wide enough
to survive the measured 4.4x suite inflation on this host is wide enough that a genuine
hang sits inside it, so its bottom-right cell reads PASS — the detector is a decoration
exactly when it is needed.  A work-unit budget's bottom-right cell reads FAIL, later in
wall time and at the same point in work.

Run:
    python probe_stall_guard.py [--device 0] [--quick]

Writes bench/results/stall_guard_arms-dev{N}.json.  Not collected by pytest (the filename
is `probe_`, not `test_`): it costs real minutes and the always-on falsifier is
`test_stall_guard.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
_REPO_ROOT = HERE.parent.parent
_RESULTS = _REPO_ROOT / "bench" / "results"
_CENSUS_TEST = "test_wiring_census.py::test_wiring_census"

#: The step to hang.  `validation_probe` is chosen because it carries the smallest budget,
#: so the hung cells resolve in bounded work; the mechanism is the same for every step.
_INJECT_STEP = "validation_probe"

#: Load: this many busy processes.  The point is not a specific ratio — the detector is
#: supposed to be indifferent to the ratio — it is that the ratio is visibly not 1.
_LOAD_PROCS = max(4, 2 * (os.cpu_count() or 4))

_LOAD_SRC = (
    "import hashlib,time\n"
    "b=bytes(range(256))*4096\n"
    "end=time.time()+{secs}\n"
    "while time.time()<end: hashlib.sha256(b).digest()\n"
)


def _start_load(seconds: int) -> list[subprocess.Popen]:
    return [
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", _LOAD_SRC.format(secs=seconds)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(_LOAD_PROCS)
    ]


def _stop_load(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            p.kill()
        except Exception:  # noqa: BLE001, S110
            pass
    for p in procs:
        try:
            p.wait(timeout=15)
        except Exception:  # noqa: BLE001, S110
            pass


def _spin_rate() -> float:
    """Reference computations per second by a thread running FLAT OUT, no duty cycle.

    The work clock deliberately sleeps between units, and on a 20-core host a handful of
    busy processes leave it a free core, so its rate barely moves — which is correct
    behaviour (a machine with spare cores is not contending with the census either) but
    makes it a poor witness for "the load arm was actually loaded".  This measurement has
    no sleep in it, so it drops as soon as the cores are genuinely oversubscribed and can
    be quoted as evidence that the arms differed in load.
    """
    block = bytes(range(256)) * 1024
    n = 0
    end = time.perf_counter() + 2.0
    while time.perf_counter() < end:
        hashlib.sha256(block).digest()
        n += 1
    return n / 2.0


def _clock_rate() -> float:
    """Reference units per second right now.  Reported so the two load arms differ visibly.

    This is a rate, not a duration, and it is a property of the *instrument*, not of the
    EP — it is the denominator the budgets are expressed in, and stating it is what lets a
    reader check that the loaded arm was actually loaded (§10.0 obligation 8 governs
    claims about the subject; this is a claim about the measuring rod).
    """
    sys.path.insert(0, str(HERE))
    from _watchdog import WorkClock  # noqa: PLC0415

    clock = WorkClock().start()
    t0 = time.perf_counter()
    start = clock.units
    time.sleep(4.0)
    rate = (clock.units - start) / (time.perf_counter() - t0)
    clock.stop()
    return rate


def _run_cell(name: str, *, inject: bool, loaded: bool, device: str) -> dict:
    env = dict(os.environ)
    env.pop("ONNXRUNTIME_EP_VULKAN_TEST_STALL", None)
    if inject:
        env["ONNXRUNTIME_EP_VULKAN_TEST_STALL"] = _INJECT_STEP
    env["ONNXRUNTIME_EP_VULKAN_DEVICE"] = device

    load = _start_load(3600) if loaded else []
    try:
        if loaded:
            time.sleep(5.0)  # let the load actually take hold before measuring anything
        rate = _clock_rate()
        spin = _spin_rate()
        proc = subprocess.run(  # noqa: S603
            [
                sys.executable, "-m", "pytest", _CENSUS_TEST,
                "-q", "-p", "no:cacheprovider", "-s",
            ],
            cwd=str(HERE),
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
        )
    finally:
        _stop_load(load)

    combined = (proc.stdout or "") + (proc.stderr or "")
    fired = "NO_PROGRESS" in combined
    verdict = "PASS" if proc.returncode == 0 else "FAIL"

    census_path = _RESULTS / f"wiring_census-dev{device}.json"
    census = {}
    if census_path.is_file():
        try:
            census = json.loads(census_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            census = {}
        # `stall_guard_census-`, NOT `wiring_census-`, and the rename is a defect fix.
        #
        # `ci/check_census_completeness.py` globs `wiring_census-*.json` and treats every
        # match as a CENSUS ARM in its name-against-content analysis.  These four cells are
        # snapshots taken by a stall-guard probe, not arms the census ran, and because they
        # are snapshots they freeze the census's output FORMAT at the moment they were
        # written.  A later round that improves an observation's text then reads as
        # `VARIES` in Link's report — content responding to input — when what actually
        # varied was the format between an old snapshot and a new run.  Found round 29,
        # when six mechanisms flipped to VARIES and only two of them had actually moved
        # between the two device arms.
        (_RESULTS / f"stall_guard_census-dev{device}-{name}.json").write_text(
            json.dumps(census, indent=2), encoding="utf-8"
        )

    stall_lines = [
        ln.strip() for ln in combined.splitlines()
        if "NO_PROGRESS" in ln or "STALL-INJECT" in ln
    ]
    return {
        "cell": name,
        "fault_injected": _INJECT_STEP if inject else "NONE",
        "machine": "LOADED" if loaded else "QUIET",
        "load_processes": _LOAD_PROCS if loaded else 0,
        "reference_units_per_second": round(rate, 1),
        "flat_out_units_per_second": round(spin, 1),
        "pytest_exit": proc.returncode,
        "verdict": verdict,
        "detector_fired": fired,
        # R13: the text is the evidence.  No count of failures appears anywhere here.
        "stall_text": stall_lines[:6],
        "census_stalled": census.get("stalled", []),
        "observed_units": census.get("stall_detector", {}).get("observed_units", {}),
        "observed_max_silence_units": census.get("stall_detector", {}).get(
            "observed_max_silence_units", {}
        ),
        "clock_units_total": census.get("stall_detector", {}).get("clock_units_total"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0"))
    ap.add_argument(
        "--cells",
        default="healthy_quiet,healthy_loaded,hung_quiet,hung_loaded",
        help="comma-separated subset, for re-running one cell",
    )
    args = ap.parse_args(argv)

    plan = {
        "healthy_quiet": dict(inject=False, loaded=False),
        "healthy_loaded": dict(inject=False, loaded=True),
        "hung_quiet": dict(inject=True, loaded=False),
        "hung_loaded": dict(inject=True, loaded=True),
    }
    wanted = [c.strip() for c in args.cells.split(",") if c.strip()]

    cells = []
    for name in wanted:
        print(f"\n=== cell {name} ===", file=sys.stderr, flush=True)
        cell = _run_cell(name, device=args.device, **plan[name])
        print(
            f"  verdict={cell['verdict']} detector_fired={cell['detector_fired']} "
            f"units/s={cell['reference_units_per_second']}",
            file=sys.stderr,
            flush=True,
        )
        cells.append(cell)

    by_name = {c["cell"]: c for c in cells}
    healthy = [c for c in cells if c["fault_injected"] == "NONE"]
    hung = [c for c in cells if c["fault_injected"] != "NONE"]

    arms_differ = (
        bool(healthy) and bool(hung)
        and all(not c["detector_fired"] for c in healthy)
        and all(c["detector_fired"] for c in hung)
    )
    load_arms_differ = None
    load_witness = {}
    if "healthy_quiet" in by_name and "healthy_loaded" in by_name:
        q, ld = by_name["healthy_quiet"], by_name["healthy_loaded"]
        # Two independent witnesses that the load arm was actually loaded, because the
        # first one alone is weak on a 20-core host: the duty-cycled work clock keeps a
        # free core and barely slows, which is correct behaviour but proves nothing about
        # the load.  The flat-out rate has no sleep in it and the census's own work-unit
        # consumption is measured over the whole run.
        load_witness = {
            "flat_out_units_per_second": {
                "quiet": q["flat_out_units_per_second"],
                "loaded": ld["flat_out_units_per_second"],
            },
            "census_clock_units_total": {
                "quiet": q["clock_units_total"],
                "loaded": ld["clock_units_total"],
            },
            "what_the_second_witness_shows": (
                "The healthy census consumed more of this machine's reference work when "
                "loaded than when quiet.  That is the CPU-only limit of the work clock, "
                "measured rather than asserted: a duty-cycled CPU-bound unit under-tracks "
                "load that also hits the GPU, the disk and the scheduler, so the budgets "
                "carry margin for it and this run reports how much of that margin the "
                "load consumed."
            ),
        }
        load_arms_differ = (
            ld["flat_out_units_per_second"] < q["flat_out_units_per_second"]
            or (ld["clock_units_total"] or 0) > (q["clock_units_total"] or 0)
        )

    doc = {
        "probe": "stall_guard_arms",
        "device_selector": args.device,
        "claim": (
            "The census stall detector fires on a hang and does not fire on contention, "
            "and does both regardless of which of the two is happening at the time."
        ),
        "why_not_a_bigger_timeout": (
            "A wall-clock threshold moves the same way when the box is loaded and when "
            "the census hangs, so no value of it separates them (R9 amendment 5).  The "
            "hung_loaded cell is the discriminator: a timeout loose enough to survive "
            "this host's measured suite inflation passes there, and a budget denominated "
            "in work units fails there."
        ),
        "arms_must_differ": arms_differ,
        "load_arms_must_differ": load_arms_differ,
        "load_witness": load_witness,
        "injected_step": _INJECT_STEP,
        "fault_injection_marking": (
            "Every injected cell announces [STALL-INJECT] on the child's stderr and the "
            "census artifact records stall_detector.fault_injection, so a planted stall "
            "can never be read as a suffered one."
        ),
        "no_duration_quoted": (
            "§10.0 obligation 8 — reference_units_per_second describes the measuring rod, "
            "not the EP.  No cell quotes how long anything took, and no verdict here "
            "depends on a duration."
        ),
        "cells": cells,
    }
    _RESULTS.mkdir(parents=True, exist_ok=True)
    out = _RESULTS / f"stall_guard_arms-dev{args.device}.json"
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", file=sys.stderr)

    if len(wanted) == 4 and not arms_differ:
        print(
            "DEMONSTRATION VOID: the arms did not differ.  A detector that fires in every "
            "cell, or in none, has demonstrated nothing.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
