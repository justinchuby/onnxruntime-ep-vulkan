"""Falsify ``gpu_steady_tail()`` against foreign GPU work, using real contended measurements.

The question, and why it could not be answered by argument
----------------------------------------------------------

The coordinator corrected an attribution: the load on this box is not his orchestration, it is
another project running CPU **and GPU** tests. Everything established about contention until now
covered the CPU case — Fact Checker's finding that Intel's 52.0833 ns/tick counter is
reference-clock based, so host load changes work-per-tick and not the tick. **That argument does
not cover a second process submitting GPU work**, and ``gpu_steady_tail()`` is the only
measurement channel this project is currently allowed to quote from.

So: does foreign GPU work move the steady tail, and does the gate fire when it does?

The gate is ``rsd <= 2%`` over a **suffix** of at least 5 inferences. Two properties follow
before any measurement, and they pull in opposite directions:

* RSD is a **variance** test. Contention that is *intermittent* raises variance and is caught.
  Contention that is *sustained* raises variance not at all — it moves the whole series to a new
  level which is perfectly steady about its own wrong mean. **A variance test cannot see a bias.**
* The window is a **suffix**, not an arbitrary window. So contention that *ends before the run
  does* is excluded automatically, however severe it was.

Together those say the gate's protection depends on something outside the gate: **whether the
contention stops before the measurement stops.** That is a property of the other project's
schedule, not of our instrument. This module tests that claim against measured runs rather than
resting on the reasoning.

The data
--------

Three real runs of the same build on the same device, taken within the hour:

* ``soloA``      — sole tenant, verified by process ancestry against ``nvidia-smi``.
* ``contended``  — one foreign GPU process.
* ``contended3`` — three foreign GPU processes, which is the sustained case.

The truncation
--------------

``contended3`` ends with its foreign load finishing, so the series decays and every suffix
contains the decay — which is why the gate refused it. **Truncating that run is not synthesis.**
It asks what this exact instrument would have reported about this exact device state had the
measurement ended while contention was still present. That is the realistic case and not the
exotic one: the other project's test runs last hours; a probe run lasts twenty-five seconds. The
truncated series is measured data, sampled from a real device under real foreign load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import phases  # noqa: E402

#: Runs to compare, with the device-state verdict each was taken under.
RUNS = [
    ("soloA", "SOLE_TENANT"),
    ("contended", "FOREIGN_GPU_WORK (1 process)"),
    ("contended3", "FOREIGN_GPU_WORK (3 processes)"),
]

#: The second failure mode, which needed no foreign process at all. Same build family, same
#: sole-tenant board, differing only in whether the driver let the board leave its idle power
#: state. ``gpustate_*.json`` carries the SM clock each was taken at.
CLOCK_RUNS = [
    ("after_coldboard", "post-barrier build, cold idle board"),
    ("base_b", "pre-barrier build, cold idle board"),
    ("baseline_certified", "pre-barrier build, cold idle board (repeat)"),
]


def series_of(tag: str) -> "list[float]":
    """The per-inference GPU-busy series in microseconds, as the gate consumes it."""
    d = json.loads((HERE / f"gemv_{tag}_dev0.json").read_text("utf-8"))
    return [v * 1000.0 for v in d["gpu_steady_tail"]["series_ms"]]


def verdict_line(name: str, busy_us: "list[float]") -> str:
    r = phases.gpu_steady_tail(busy_us)
    if r["verdict"] == "STEADY":
        return (f"{name:<44} {r['verdict']:<15} {r['median_ms']:>9.3f} ms   "
                f"RSD {r['rsd']:>8.4%}   n={r['n']:<3} discarded={r['discarded_inferences']}")
    return f"{name:<44} {r['verdict']:<15} {'—':>9}      {r.get('detail','')[:44]}"


def main() -> int:
    print("== does foreign GPU work move gpu_steady_tail(), and does the gate fire? ==\n")
    print(f"{'run':<44} {'verdict':<15} {'median':>9}      detail")
    print("-" * 108)

    baseline = None
    for tag, state in RUNS:
        try:
            s = series_of(tag)
        except FileNotFoundError:
            print(f"ERROR(instrument): gemv_{tag}_dev0.json absent — run not taken")
            continue
        r = phases.gpu_steady_tail(s)
        if tag == "soloA" and r["verdict"] == "STEADY":
            baseline = r["median_ms"]
        print(verdict_line(f"{tag}  [{state}]", s))

    print()
    print("-- the same contended run, truncated to end while its contention was still present --")
    print("   (measured data; only the run's *length* differs, which is the other project's")
    print("    schedule and not ours)")
    print()

    full = series_of("contended3")
    for cut in (20, 28, 34):
        print(verdict_line(f"contended3 truncated to {cut} inferences", full[:cut]))

    if baseline:
        print()
        for cut in (20, 28, 34):
            r = phases.gpu_steady_tail(full[:cut])
            if r["verdict"] == "STEADY":
                print(f"   at {cut} inferences the gate reports STEADY at "
                      f"{r['median_ms']:.3f} ms, RSD {r['rsd']:.4%} — "
                      f"**{r['median_ms'] / baseline:.2f}x** the sole-tenant "
                      f"{baseline:.3f} ms, with no warning of any kind")

    print()
    print("=" * 108)
    print("== the second failure mode, which needed no foreign process at all ==")
    print()
    print("The gate is blind to the GPU's clock. This board idles at 210 MHz against a 3105 MHz")
    print("maximum, and a workload that never raises utilisation enough to trigger a boost runs")
    print("the whole way at idle clock. That is *perfectly steady* — so it produces the gate's")
    print("most confident possible verdict. A low clock does not raise RSD; it lowers it.")
    print()
    print(f"{'run':<52} {'tenancy':<18} {'SM clock':<22} {'gate says'}")
    print("-" * 108)
    for tag, what in CLOCK_RUNS:
        try:
            st = json.loads((HERE / f"gemv_{tag}_dev0.json").read_text("utf-8"))["gpu_steady_tail"]
            gs = json.loads((HERE / f"gpustate_{tag}.json").read_text("utf-8"))
        except FileNotFoundError:
            print(f"ERROR(instrument): {tag} artifacts absent")
            continue
        clk = gs["sm_mhz"]
        clock_s = f"{clk['min']:.0f}-{clk['max']:.0f} MHz"
        said = (f"STEADY {st['median_ms']:.3f} ms @ RSD {st['rsd']:.4%}"
                if st["verdict"] == "STEADY" else st["verdict"])
        print(f"{what:<52} {gs['verdict']:<18} {clock_s:<22} {said}")

    print()
    print("  Same build family, same sole-tenant board, minutes apart. The pre-barrier build held")
    print("  the board at 210 MHz for 594 seconds and never boosted; the post-barrier build")
    print("  reached 2490 MHz within 15 inferences. GPU busy differs by 21.4x and BOTH runs are")
    print("  reported STEADY, the slower one at the tighter RSD.")
    print()
    print("  Mechanism: cutting 11.7 ms of host time per inference raised GPU duty cycle from")
    print("  ~11.5/30.7 = 37% to ~11.5/17.2 = 67%, crossing the driver's boost threshold. So a")
    print("  host-side fix has a second-order effect on device clock. That is real, but it is")
    print("  NOT a 21.4x kernel speedup and must never be quoted as one.")
    print()
    print("  The one piece of good news, and it retro-certifies the project's history: the two")
    print("  clock regimes are 21x apart and do not overlap. A phi-3.5 GPU-busy figure near")
    print("  12 ms is necessarily a boosted-clock figure, because the idle-clock regime for this")
    print("  workload is ~247 ms. So every figure this project has quoted in the 11-41 ms band")
    print("  was taken in the boosted regime and the comparisons between them stand.")

    print()
    print("CONCLUSION")
    print("  `gpu_steady_tail` is a variance test over a suffix. It cannot see a bias, from")
    print("  either source measured here:")
    print("    1. foreign GPU work that outlives the run  -> STEADY at 11.0x, RSD 0.79%")
    print("    2. a board that never leaves its idle clock -> STEADY at 21.4x, RSD 0.12%")
    print("  In both cases the WRONG number carried a BETTER RSD than the right one. Low RSD is")
    print("  evidence of a steady device; it is not evidence of a correctly clocked, sole-tenant")
    print("  one, and it was being read as if it were.")
    print()
    print("  A device-clock figure is quotable only when accompanied, over the same window, by")
    print("  a tenancy verdict AND an SM-clock record. That is what probe_gpustate.py produces,")
    print("  and it is now a required companion rather than a diagnostic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
