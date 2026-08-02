#!/usr/bin/env python3
"""``check_run_disturbance`` — a run whose repetitions disagree may not publish a timing figure.

The finding this exists for
---------------------------

Contention has cost this project repeatedly and expensively: the suite inflates 4.4x under load
(161 s quiet, 708 s loaded), ``record`` inflates 9.5x, and a contention-induced ``68 failed`` was
once read as a regression that did not exist. Every guard considered for it has been wall-clock
threshold shaped, and every one fails R9 amendment 5 — a slow machine and a broken build both make
numbers go up, so the check moves with the reader's confidence rather than with the condition.

``bench/run_disturbance.py`` supplies a quantity that does not have that shape: the RSD of one
dispatch **ordinal** across repetitions. Ordinal ``k`` is the same node with the same shape every
inference, so its dispersion is the machine failing to repeat itself, and it is dimensionless,
in-band, and unmoved by how fast the machine or how large the model is.

Why this is not the device-state check again
---------------------------------------------

It is the orthogonal half, and both are required:

* obligation 8 (``check_device_state``) catches a **clock that was wrong** — a board pinned at
  210 MHz, uniformly inflating every dispatch.
* this catches a **run that was disturbed** — repetitions disagreeing at whatever clock.

Neither sees the other's failure. A run can be disturbed at a perfectly steady clock; and we have
observed a run perfectly stationary at an entirely wrong clock, at 21.4x, carrying *better* RSD
than the correct one. Passing this check is not evidence for the other and never substitutes.

The three terminal states
-------------------------

* ``PASS`` — the scan published no timing figure, **or** every run behind one repeated itself.
* ``FAIL(condition=RUN_DISTURBED)`` — a figure was published from a run whose repetitions disagree
  beyond the census threshold.
* ``ERROR(instrument=...)`` — the check could not reach its observation: an unparseable trace, too
  few repetitions, or a dispatch count that wanders so that ordinal ``k`` is not one node. Per
  §10.0.1 R13 this is **not** a detection and **not** a pass.

What a PASS does not mean
--------------------------

It does not mean the machine was quiet. The statistic is blind to level: a run in which a
competing process took a fixed share of the device throughout is *uniformly* inflated, its
repetitions agree perfectly, and it passes. ``run_disturbance.synthetic_uniform_slowdown``
constructs that run and ``bench/test_run_disturbance.py`` asserts this check passes it. The claim
is scoped to stationarity, deliberately and demonstrably.

USAGE
    python ci/check_run_disturbance.py --scan bench/results [--summary <file>] [--explain]
    python ci/check_run_disturbance.py --trace bench/results/trace_gemv_contended_dev0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import run_disturbance as rd  # noqa: E402

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

LABEL = "RUN-DISTURBANCE"


def report_pass(detail: str) -> int:
    print(f"{LABEL}: PASS", flush=True)
    print(detail, flush=True)
    return EXIT_PASS


def report_fail(condition: str, detail: str) -> int:
    print(f"{LABEL}: FAIL(condition={condition})", flush=True)
    print(detail, flush=True)
    return EXIT_FAIL_CONDITION


def report_instrument_error(instrument: str, detail: str) -> int:
    print(f"{LABEL}: ERROR(instrument={instrument})", flush=True)
    print(detail, flush=True)
    print(
        f"{LABEL}: the check did not reach its observation. Per DESIGN.md §10.0.1 R13 "
        "this is NOT a detection and NOT a pass.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


def assess(trace: Path) -> dict:
    m = rd.measure(rd.per_inference_kernel_us(trace))
    out = rd.classify(m)
    out["measurement"] = m
    out["trace"] = trace.name
    out["path"] = str(trace)
    return out


def corroborate(results: "list[dict]") -> dict:
    """How much does this check add over ``gpu_steady_tail``'s own floors? Measured, not assumed.

    A guard that only ever refuses runs something else already refused is corroborating rather
    than protecting, and that is worth knowing and worth re-checking as the trace set grows. This
    computes it instead of leaving it to a claim in a docstring.
    """
    import phases

    both, new, tail_only, compared, new_traces = 0, 0, 0, 0, []
    for r in results:
        path = Path(r.get("path") or "")
        if not path.exists():
            continue
        try:
            events = phases.load(path)
            subs = phases.subgraph_spans(events)
            gpus = phases.gpu_spans(events)
            busy = phases.attribute_gpu_ordinally(subs, gpus)["busy_us"]
            tail = phases.gpu_steady_tail([busy.get(s["index"]) for s in subs])
        except Exception:
            continue
        compared += 1
        publishes = tail.get("verdict") == "STEADY"
        if r["verdict"] == "FAIL" and publishes:
            new += 1
            new_traces.append(r["trace"])
        elif r["verdict"] == "FAIL":
            both += 1
        elif not publishes:
            tail_only += 1
    reading = (
        "Every run this check refuses is already refused by the tail's own floors, so on this "
        "evidence it ADDS NO REFUSALS. And the reason is now known rather than puzzling: "
        "same-ordinal RSD correlates with whole-series per-inference RSD at Spearman 0.919 "
        "(log-log r 0.970), because a disturbance that scales a whole submission moves every "
        "dispatch inside it together. The agreement is REDUNDANCY, NOT CORROBORATION. What this "
        "module still offers that nothing else does is localisation -- see --localise -- and the "
        "fact that it refuses for a reason independent of suffix selection, which matters if a "
        "MARGINAL_TAIL's withheld median is ever published." if new == 0 else
        f"{new} run(s) would be published by the tail and are refused here. This check is "
        "protecting, not merely corroborating, and those runs need attention.")
    return {"compared": compared, "both_refuse": both, "new_refusals": new,
            "new_refusal_traces": new_traces, "tail_only": tail_only, "reading": reading}


def sensitivity(results: "list[dict]", grid=None) -> dict:
    """Flagged count against threshold, published whole rather than reduced to one number.

    Niobe's discipline: she set a threshold, saw the flagged count, noticed her instinct was to
    move it until the count reached zero, named that *choosing the answer*, and published the
    table. A boundary is defensible when the answer is flat around it, and that flatness is
    something a reader must be able to see rather than take on trust.
    """
    grid = grid or [0.05, 0.10, 0.11, 0.15, 0.20, 0.25, 0.30, 0.35, 0.36, 0.40, 0.50, 0.75, 1.00]
    meds = [r["measurement"]["same_ordinal_rsd_median"] for r in results
            if r.get("measurement")]
    table = [{"threshold": t, "flagged": sum(1 for m in meds if m > t)} for t in grid]
    # The widest run of thresholds giving the answer at the shipped default.
    at_default = sum(1 for m in meds if m > rd.DISTURBANCE_RSD_MAX)
    stable = [row["threshold"] for row in table if row["flagged"] == at_default]
    return {
        "n_traces": len(meds),
        "table": table,
        "flagged_at_default": at_default,
        "default": rd.DISTURBANCE_RSD_MAX,
        "unchanged_over": [min(stable), max(stable)] if stable else None,
        "reading": (
            f"The flagged count is {at_default} for every threshold from {min(stable):.0%} to "
            f"{max(stable):.0%}." if stable else "No threshold on the grid reproduces the default."
        ) + " Flatness around the boundary is the argument for it; if this band is ever narrow, "
            "the threshold is doing the deciding and should be reported as a judgement.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="refuse a timing figure from a disturbed run")
    ap.add_argument("--scan", type=Path, help="directory of traces to assess")
    ap.add_argument("--trace", type=Path, action="append", help="assess these traces")
    ap.add_argument("--glob", default="trace_*.json", help="pattern used with --scan")
    ap.add_argument("--summary", type=Path, help="write the full assessment here as JSON")
    ap.add_argument("--corroborate", action="store_true",
                    help="also report agreement with gpu_steady_tail's verdict, including how "
                         "many refusals this check adds that the tail's own floors do not")
    ap.add_argument("--explain", action="store_true", help="print what the check does and stop")
    ap.add_argument("--localise", action="store_true",
                    help="decompose each run's dispersion into whole-submission scaling versus "
                         "per-dispatch disagreement -- the part that is NOT redundant with "
                         "whole-series RSD")
    ap.add_argument("--sensitivity", action="store_true",
                    help="print the flagged-count-against-threshold table instead of defending "
                         "a single boundary")
    args = ap.parse_args(argv)

    if args.explain:
        print(__doc__)
        return EXIT_PASS
    if not args.scan and not args.trace:
        print(f"{LABEL}: usage — one of --scan or --trace is required", file=sys.stderr)
        return EXIT_USAGE

    traces = list(args.trace or [])
    if args.scan:
        if not args.scan.is_dir():
            return report_instrument_error(
                "scan_root_absent", f"--scan {args.scan} is not a directory")
        traces += sorted(args.scan.glob(args.glob))
    if not traces:
        return report_pass(
            "no trace carrying a timing figure was found in this scan, so there is no run whose "
            "stationarity could be in question. This is the 'published nothing timed' pass.")

    results, errors = [], []
    for t in traces:
        try:
            results.append(assess(t))
        except rd.InstrumentError as e:
            errors.append({"trace": t.name, "instrument_error": str(e)})
        except Exception as e:  # a trace we cannot read is our failure, never a detection
            errors.append({"trace": t.name, "instrument_error": f"{type(e).__name__}: {e}"})

    failed = [r for r in results if r["verdict"] == "FAIL"]
    summary = {
        "check": "run_disturbance",
        "threshold": rd.DISTURBANCE_RSD_MAX,
        "assessed": len(results),
        "instrument_errors": errors,
        "failed": [r["trace"] for r in failed],
        "results": results,
    }
    if args.corroborate:
        summary["corroboration"] = corroborate(results)
    if args.sensitivity:
        summary["sensitivity"] = sensitivity(results)
    if args.localise:
        loc = []
        for r in results:
            p = Path(r.get("path") or "")
            try:
                loc.append({"trace": r["trace"], **rd.localise(rd.per_inference_kernel_us(p))})
            except Exception as e:  # noqa: BLE001
                loc.append({"trace": r["trace"], "instrument_error": f"{type(e).__name__}: {e}"})
        summary["localisation"] = loc
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2))

    for r in sorted(results, key=lambda r: -r["same_ordinal_rsd_median"]):
        mark = "FAIL" if r["verdict"] == "FAIL" else "pass"
        print(f"  {mark}  {r['same_ordinal_rsd_median']:>8.2%}  {r['trace']}")

    if args.corroborate and "corroboration" in summary:
        c = summary["corroboration"]
        print(f"\n  vs gpu_steady_tail over {c['compared']} trace(s):")
        print(f"    flagged here AND already refused by the tail : {c['both_refuse']}")
        print(f"    flagged here but the tail would PUBLISH      : {c['new_refusals']}"
              f"   {c['new_refusal_traces'] or ''}")
        print(f"    tail refuses but this check passes           : {c['tail_only']}")
        print(f"  {c['reading']}")

    if args.sensitivity and "sensitivity" in summary:
        s = summary["sensitivity"]
        print(f"\n  threshold sensitivity over {s['n_traces']} trace(s):")
        for row in s["table"]:
            mark = "  <- default" if row["threshold"] == s["default"] else ""
            print(f"    {row['threshold']:>6.0%}   {row['flagged']:>3d} flagged{mark}")
        print(f"  {s['reading']}")

    if args.localise and "localisation" in summary:
        print("\n  localisation -- a DECOMPOSITION, not a second opinion "
              "(normalised statistic still correlates with whole-series RSD at rho 0.710):")
        print(f"    {'trace':34s} {'ordinal':>9s} {'norm':>9s} {'expl':>7s}  character")
        for e in sorted(summary["localisation"],
                        key=lambda x: -(x.get("same_ordinal_rsd_median") or 0)):
            if "instrument_error" in e:
                print(f"    {e['trace']:34s}  ERROR(instrument): {e['instrument_error']}")
                continue
            ex = e["explained_by_level"]
            print(f"    {e['trace']:34s} {e['same_ordinal_rsd_median']:>8.2%} "
                  f"{e['level_normalised_ordinal_rsd']:>8.2%} "
                  f"{(ex if ex is not None else float('nan')):>6.1%}  "
                  f"{e['character'].split(':')[0]}")

    if errors and not results:
        return report_instrument_error(
            "no_trace_assessable",
            "every trace in the scan raised an instrument error:\n  "
            + "\n  ".join(f"{e['trace']}: {e['instrument_error']}" for e in errors))
    if failed:
        return report_fail(
            "RUN_DISTURBED",
            f"{len(failed)} of {len(results)} run(s) exceed a same-ordinal RSD of "
            f"{rd.DISTURBANCE_RSD_MAX:.0%}; repetitions of identical work disagree, so no timing "
            "figure from them is publishable:\n  "
            + "\n  ".join(f"{r['trace']}: {r['same_ordinal_rsd_median']:.2%}" for r in failed))
    return report_pass(
        f"{len(results)} run(s) repeated themselves within {rd.DISTURBANCE_RSD_MAX:.0%} "
        "same-ordinal RSD."
        + (f" {len(errors)} trace(s) could not be assessed and are reported, not counted as "
           "passes." if errors else "")
        + "\nThis says the runs were stationary. It does NOT say the machine was quiet or the "
          "clock was right — see obligation 8's device-state record, which is still required.")


if __name__ == "__main__":
    raise SystemExit(main())
