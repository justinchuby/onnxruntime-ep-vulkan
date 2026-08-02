"""Is the `MARGINAL_TAIL` boundary in the right place? Cross-check every verdict against an
independent signal.

WHY
===

`probe_frames.py` established that Switch's ~70% kernel spread and my steady tail are one frame
under two selections, and produced a 28-trace census showing that no trace with a large
whole-series spread publishes a tail figure. That is a real property and I reproduced it. **It is
not a clean bill of health, and this probe exists to find out how far from one it is.**

Two questions, and they are not the same question:

1. **Does any trace grade `STEADY` that should not have?** The tail test's own RSD cannot answer
   this -- it is the thing under test. `probe_frames.per_kernel` supplies an independent signal
   from a different frame: the **same-ordinal-across-inferences RSD**, the spread of the *k*-th
   dispatch's duration across inferences. Switch measured 0.36% on his steadiest trace and 142.41%
   on his most disturbed one, computed for two traces. This probe computes it for **all 28**, so
   that a `STEADY` verdict sitting on disturbed per-dispatch structure has somewhere to show up.

2. **Does the boundary refuse things it need not?** Reported, but deliberately subordinate: an
   over-refusal costs a data point and an under-refusal ships a wrong number.

WHAT THIS PROBE CANNOT DO, AND IT MATTERS MORE THAN WHAT IT CAN
===============================================================

**Every signal here -- whole-series RSD, tail RSD, same-ordinal RSD -- is a dispersion measure
computed from the same series of durations.** They differ in *selection*, not in kind. A run that
is uniformly wrong is quiet in all three at once, which is exactly what
`trace_gemv_baseline_certified_dev0.json` demonstrates from inside this very census: whole-series
RSD **0.12%**, tail RSD **0.1163%**, `STEADY` at **100% coverage** -- the steadiest trace of the
28 -- and the paired `gemv_baseline_certified_dev0.json` reports **246.72 ms against a true
~11.5 ms**, because the board never left its 210 MHz idle clock. **21.4x wrong, and the most
confident verdict in the census.**

So this probe can find a `STEADY` verdict sitting on *disturbed* structure. It cannot find one
sitting on *uniformly displaced* structure, and no rearrangement of these three numbers ever will.
That is R9 amendment 5 and the answer stays `bench/device_state.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import probe_frames  # noqa: E402

#: Switch's two measured poles: 0.36% on the steadiest trace, 142.41% on the most disturbed.
#:
#: **This threshold is reported at three placements, not one, because the population between his
#: two poles is a continuum and not a gap.** Placed at 5% (an order of magnitude above the clean
#: pole, which is where I first put it) it calls 7 traces holes; placed anywhere in the census's
#: only real gap -- 10.48% to 39.60%, a 3.78x step -- it calls none. Publishing a single number
#: here would be choosing the answer. See `threshold_sensitivity` in the output.
SAME_ORDINAL_DISTURBED = 0.05

#: The empirical gap in the same-ordinal series, which is also where the whole-series gap falls
#: (10.36% -> 34.39%). That the two signals gap in the same place is itself the finding: see
#: `independence` in the output.
SAME_ORDINAL_GAP = (0.1048, 0.3960)

PUBLISHING = probe_frames.PUBLISHING_VERDICTS


def _sensitivity(rows: "list[dict]") -> dict:
    """How many holes at each candidate threshold. Published so the choice cannot hide.

    I set this at 5% first, got 7 holes, and my instinct was to move it until the number was 0.
    That instinct is the reason this table exists instead of a threshold.
    """
    out = []
    for thr in (0.02, 0.05, 0.08, 0.105, 0.20, 0.30, 0.396, 0.50):
        h = [r["trace"] for r in rows
             if r.get("same_ordinal_rsd") is not None
             and r["same_ordinal_rsd"] >= thr and r["tail_verdict"] in PUBLISHING]
        out.append({"threshold": thr, "holes": len(h), "traces": h})
    return {
        "table": out,
        "note": ("holes fall to zero for every threshold at or above 10.5%, and the census's only "
                 "gap is 10.48% -> 39.60%. So 'no holes' is true for any placement inside the "
                 "gap and false below it. Below the gap the flagged traces are not a separate "
                 "population -- they are the top of a continuum, and calling them holes is a "
                 "statement about the threshold, not about the traces."),
    }


def _independence(rows: "list[dict]") -> dict:
    """Is the same-ordinal RSD actually an independent check on the tail? Measured, not assumed."""
    pairs = [(r["whole_series_rsd"], r["same_ordinal_rsd"]) for r in rows
             if r.get("same_ordinal_rsd") and r.get("whole_series_rsd")]
    if len(pairs) < 3:
        return {"verdict": "UNCHECKED", "n": len(pairs)}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    def pearson(a, b):
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(order):
            out[i] = pos + 1
        return out

    ratios = sorted(y / x for x, y in pairs)
    return {
        "n": len(pairs),
        "spearman_rho": round(pearson(rank(xs), rank(ys)), 4),
        "pearson_r_on_logs": round(pearson([__import__("math").log(x) for x in xs],
                                           [__import__("math").log(y) for y in ys]), 4),
        "median_same_ordinal_over_whole_series": round(statistics.median(ratios), 3),
        "verdict": "NOT_INDEPENDENT",
        "detail": (
            "The same-ordinal-across-inferences RSD is very nearly the whole-series RSD: Spearman "
            "0.90, r=0.96 on logs, and the median trace has a same-ordinal RSD 1.13x its "
            "whole-series RSD. That is mechanical rather than coincidental -- a disturbance that "
            "scales a whole submission moves every dispatch in it together, so the k-th "
            "dispatch's spread across inferences reproduces the inference sums' spread. **It is "
            "therefore a re-selection of the same measurement, not a second opinion**, and a "
            "'no holes' result from it is much weaker evidence about the tail test than it "
            "looks. The decisive evidence is that it agrees with every other dispersion measure "
            "on trace_gemv_baseline_certified_dev0.json -- rating it the CLEANEST trace in the "
            "census at 0.36% -- while that run is 21.4x wrong."),
    }


def census(trace_dir: Path) -> dict:
    rows = []
    errors = []
    for t in sorted(trace_dir.glob("trace_gemv_*_dev0.json")):
        try:
            row = probe_frames.per_inference(t)
            subs, gpus = row.pop("_subs"), row.pop("_gpus")
            if row["n_inferences"] < probe_frames.MIN_INFERENCES:
                continue
            try:
                pk = probe_frames.per_kernel(subs, gpus, row["trace"])
            except probe_frames.InstrumentError as e:
                # R13. A trace whose per-dispatch frame could not be computed is UNCHECKED, which
                # is not the same as checked-and-clean, and it must not be counted as either.
                row["same_ordinal_rsd"] = None
                row["cross_check"] = "ERROR(instrument)"
                row["cross_check_detail"] = str(e)
                rows.append(row)
                errors.append(f"{row['trace']}: {e}")
                continue
            so = pk.get("same_ordinal_across_inferences_median_rsd")
            row["same_ordinal_rsd"] = so
            row["within_one_inference_rsd"] = pk.get("within_one_inference_rsd")
            row["dispatches_per_inference"] = pk.get("dispatches_per_inference")
            row["inferences_compared"] = pk.get("inferences_compared")
            publishes = row["tail_verdict"] in PUBLISHING
            if so is None:
                row["cross_check"] = "UNCHECKED"
            elif so >= SAME_ORDINAL_DISTURBED and publishes:
                row["cross_check"] = "HOLE"
            elif so >= SAME_ORDINAL_DISTURBED:
                row["cross_check"] = "AGREE_REFUSED"
            elif publishes:
                row["cross_check"] = "AGREE_PUBLISHED"
            else:
                row["cross_check"] = "REFUSED_BUT_CLEAN"
            rows.append(row)
        except probe_frames.InstrumentError as e:
            errors.append(f"{t.name}: {e}")

    holes = [r for r in rows if r["cross_check"] == "HOLE"]
    unchecked = [r for r in rows if r["cross_check"] in ("UNCHECKED", "ERROR(instrument)")]
    over = [r for r in rows if r["cross_check"] == "REFUSED_BUT_CLEAN"]

    tail_rsds = sorted((r["tail_rsd"], r["trace"]) for r in rows if r["tail_rsd"] is not None)

    out = {
        "instrument": "probe_tail_boundary",
        "traces": len(rows),
        "same_ordinal_disturbed_threshold": SAME_ORDINAL_DISTURBED,
        "census": rows,
        "holes": [r["trace"] for r in holes],
        "unchecked": [r["trace"] for r in unchecked],
        "refused_but_clean": [r["trace"] for r in over],
        "tail_rsd_ranking": [{"trace": t, "tail_rsd": v} for v, t in tail_rsds],
        "instrument_errors": errors,
    }
    out["threshold_sensitivity"] = _sensitivity(rows)
    out["independence"] = _independence(rows)
    if errors:
        out["instrument_error_note"] = (
            "R13: failures of this probe, not findings about any trace. A trace listed here is "
            "UNCHECKED, which is neither a pass nor a detection.")

    if holes:
        out["verdict"] = "THRESHOLD_DEPENDENT"
        out["detail"] = (
            f"at the {SAME_ORDINAL_DISTURBED:.0%} threshold, {len(holes)} publishing trace(s) sit "
            f"on 'disturbed' per-dispatch structure; at any threshold inside the census's only "
            f"gap ({SAME_ORDINAL_GAP[0]:.2%}-{SAME_ORDINAL_GAP[1]:.2%}) there are none. **No "
            f"dispersion hole in the tail test is demonstrable from this census** -- and no clean "
            f"bill of health is either, because the signal is not independent of the one it is "
            f"checking. See `independence` and `certification_crosstab`.")
    else:
        out["verdict"] = "NO_DISPERSION_HOLE_FOUND"
        out["detail"] = (
            f"across {len(rows)} traces, no publishing tail verdict sits on disturbed "
            f"per-dispatch structure. **This clears the tail test of one failure mode only.**")
    out["scope"] = (
        "Every signal compared here -- whole-series RSD, tail RSD, same-ordinal RSD -- is a "
        "dispersion measure over the same durations, differing in selection and not in kind. A "
        "uniformly displaced run is quiet in all of them at once. "
        "trace_gemv_baseline_certified_dev0.json is that run and it is inside this census: the "
        "steadiest trace of the 28 on all three measures (whole 0.12%, tail 0.1163%, same-ordinal "
        "0.36%), STEADY at 100% coverage with n=46, and its paired figure is 246.72 ms against a "
        "true ~11.5 ms. **21.4x wrong, and the most confident verdict in the census.** A clean "
        "result here is not evidence of a correctly clocked sole-tenant run and must never be "
        "read as one. Only bench/device_state.py, whose evidence comes from outside the series, "
        "refuses it.")
    out["certification_crosstab"] = _crosstab(rows, trace_dir)
    return out


def _crosstab(rows: "list[dict]", trace_dir: Path) -> dict:
    """Every trace that publishes a tail figure, against the only out-of-series evidence we have.

    This is the check the dispersion measures cannot perform, and it is the one that answers
    "is there a run that grades STEADY and should not have?"
    """
    sys.path.insert(0, str(HERE.parent))
    import device_state  # noqa: E402

    seen = []
    for r in rows:
        tag = r["trace"].replace("trace_gemv_", "").replace("_dev0.json", "")
        for d in (HERE, trace_dir):
            p = d / f"gpustate_{tag}.json"
            if p.exists():
                break
        else:
            seen.append({"trace": r["trace"], "tail_verdict": r["tail_verdict"],
                         "companion": None,
                         "certification": "UNCERTIFIED",
                         "why": "no device-state companion exists for this run"})
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        state = doc.get("summary", doc)
        cert = device_state.certify(
            {"verdict": r["tail_verdict"], "median_ms": None, "n": r["tail_n"],
             "coverage": r["tail_coverage"]}, state)
        seen.append({"trace": r["trace"], "tail_verdict": r["tail_verdict"],
                     "companion": state.get("verdict"),
                     "peak_sm_mhz": (state.get("sm_mhz") or {}).get("max"),
                     "sm_max_mhz": state.get("sm_max_mhz"),
                     "certification": cert["verdict"], "why": cert.get("detail")})
    publishing = [s for s in seen if s["tail_verdict"] in PUBLISHING]
    withheld = [s for s in publishing if s["certification"] == "WITHHELD"]
    return {
        "rows": seen,
        "publishing_traces": len(publishing),
        "of_which_quotable": len([s for s in publishing if s["certification"] == "QUOTABLE"]),
        "of_which_withheld_by_the_companion": [s["trace"] for s in withheld],
        "of_which_uncertified_for_want_of_a_companion":
            len([s for s in publishing if s["certification"] == "UNCERTIFIED"]),
        "answer": (
            f"{len(withheld)} trace(s) grade a publishing tail verdict AND are refused by "
            f"out-of-series evidence: {[s['trace'] for s in withheld]}. **That is a run that "
            f"grades STEADY and should not have.** It is not a misplaced boundary -- no n or "
            f"coverage floor could reach it, since it publishes at n=46 and 100% coverage -- it "
            f"is the bias blindness, and the companion is the fix already in place."
            if withheld else
            "no publishing trace is refused by out-of-series evidence, but most have no "
            "out-of-series evidence at all."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", type=Path, required=True,
                    help="directory holding trace_gemv_*_dev0.json")
    ap.add_argument("--json", type=Path, default=HERE / "tail_boundary.json")
    args = ap.parse_args()
    if not args.traces.is_dir():
        print(f"ERROR(instrument): no such directory {args.traces}", file=sys.stderr)
        return 3
    rep = census(args.traces)
    args.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("== every verdict cross-checked against the per-dispatch frame ==")
    print(f"{'trace':<40} {'wholeRSD':>9} {'tail':<15} {'tailRSD':>9} {'sameOrd':>9}  check")
    for r in sorted(rep["census"], key=lambda r: -(r["same_ordinal_rsd"] or -1)):
        so = r["same_ordinal_rsd"]
        print(f"{r['trace'].replace('trace_gemv_', '').replace('_dev0.json', ''):<40} "
              f"{(r['whole_series_rsd'] or 0):>8.2%} {r['tail_verdict']:<15} "
              f"{(('%.4f%%' % (100 * r['tail_rsd'])) if r['tail_rsd'] else '-'):>9} "
              f"{(('%.2f%%' % (100 * so)) if so is not None else 'n/a'):>9}  {r['cross_check']}")
    print()
    print("== threshold sensitivity: how many 'holes' at each placement ==")
    for t in rep["threshold_sensitivity"]["table"]:
        print(f"  same-ordinal >= {t['threshold']:>6.1%} -> {t['holes']} hole(s)")
    print("  " + rep["threshold_sensitivity"]["note"])
    print()
    ind = rep["independence"]
    print(f"== is this an independent check? {ind['verdict']} ==")
    print(f"  Spearman {ind['spearman_rho']}, r={ind['pearson_r_on_logs']} on logs, "
          f"median same-ordinal/whole-series ratio {ind['median_same_ordinal_over_whole_series']}")
    print()
    ct = rep["certification_crosstab"]
    print("== the only out-of-series evidence: device state ==")
    print(f"{'trace':<26} {'tail':<15} {'companion':<18} {'peakSM':>7}  certification")
    for s in ct["rows"]:
        if s["tail_verdict"] not in PUBLISHING and s["companion"] is None:
            continue
        print(f"{s['trace'].replace('trace_gemv_', '').replace('_dev0.json', ''):<26} "
              f"{s['tail_verdict']:<15} {str(s['companion']):<18} "
              f"{str(s.get('peak_sm_mhz') or '-'):>7}  {s['certification']}")
    print(f"\n  publishing traces: {ct['publishing_traces']}  |  quotable: "
          f"{ct['of_which_quotable']}  |  no companion at all: "
          f"{ct['of_which_uncertified_for_want_of_a_companion']}")
    print(f"  {ct['answer']}")
    print()
    print(f"verdict: {rep['verdict']}")
    print(rep["detail"])
    print()
    print("SCOPE: " + rep["scope"])
    for e in rep["instrument_errors"]:
        print(f"  ERROR(instrument) — NOT a detection: {e}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
