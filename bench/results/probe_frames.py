"""Do Niobe's steady tail and Switch's 70% kernel-time spread measure the same thing?

The question
------------
Two statements sit in tension in the project record:

* Niobe: ``gpu_steady_tail`` reported **STEADY** on the NVIDIA figure, at **0.033% RSD**.
* Switch: kernel time swung **49.4 -> 83.8 -> 71.0 -> 58.5 ms for the same 353 dispatches**, a
  **70% spread**.

The coordinator offered two cases: either the instruments cover **different frames** and both are
right (R12), or they cover the **same frame** and one of them is wrong, in which case we need to
know which before either certifies anything. The answer is a third case, and it is stronger than
both: **the frames coincide, and the two statements can never disagree about a published number,
because the tail refuses every run whose spread is large.**

Establishing the frames -- not from their names (R11)
------------------------------------------------------
This module **calls Niobe's own functions** for her frame rather than reimplementing them, so the
comparison cannot be accused of getting her frame wrong:

* :func:`phases.attribute_gpu_ordinally` -> ``busy_us[i]`` = **the sum of ``gpu_ns`` over every
  kernel span of inference i**. One number per inference, itself a sum over ~355 kernels.
* :func:`phases.gpu_steady_tail` -> the longest **suffix** of that series holding within 2% RSD,
  with coverage and count floors below which it publishes nothing.

Switch's frame comes from the history entry that produced the numbers, read rather than recalled
(``.squad/agents/switch/history.md``): *"cluster (group GPU spans into submissions by the large
gaps, alignment-free, durations only) ... gpu_busy 49.4/83.8/71.0/58.5 ms (n=353 each)"*. That is
**also a per-submission sum over all of that submission's kernels** -- the same physical quantity,
reached by gap-clustering instead of ordinal attribution, and taken over **every** submission with
nothing discarded.

So the frames coincide. What differs is the **selection**: the tail discards a leading ramp and
reports that it did; the spread summarised the whole series including it.

The decisive test, and it is deliberately not a comparison of two numbers
--------------------------------------------------------------------------
A ratio between one whole-series RSD and one tail RSD is an anecdote, and picking the trace that
produces the prettiest ratio is how this project got three fabricated figures. The falsifiable
question is a census, not a specimen:

    **Does any run exist in which the tail publishes a quotable number AND the whole-series
    spread is large?**

If none exists, the two statements are not in conflict and no conflict can be constructed from the
evidence we hold -- they are *ordered*, because a large spread is precisely the condition under
which the tail returns ``NO_STEADY_TAIL`` or a ``MARGINAL_TAIL`` that withholds its median. This
module answers that over **every committed device-0 trace**.

This is a real falsifier: a single trace with a large whole-series spread and a publishing tail
returns ``CONFLICT`` and sends the question back for resolution.

The per-kernel hypothesis, and the correction it needs
--------------------------------------------------------
The coordinator suggested that a population of 355 kernels with a wide per-kernel spread could
produce a steady per-inference total because *"the variance averages out"*. The direction is right;
the mechanism is not, and the difference is load-bearing.

Within a single inference the individual ``q_gemv`` dispatch durations do have a large spread. But
that is **not variance** -- it is **population heterogeneity**. The 161 dispatches cover five
distinct node shapes (K=3072 with N in {3072, 8192, 9216, 32064}, and K=8192 with N=3072), and the
durations fall into discrete clusters around those shapes. The same kernel *ordinal* measured
across inferences is stable to a fraction of a percent.

There is therefore almost no variance to average. The sum is steady because each term is
individually steady and the spread *between* terms is deterministic and repeats identically every
inference. "The variance averages out" makes a checkable prediction -- that the sum's RSD is
``per_kernel_rsd / sqrt(N)`` -- and this module prints that prediction next to the observed value
so the reader can see the observed one fall far below it. That gap is the signature of a
deterministic spread, not an averaged random one.

R13
---
A trace this module cannot parse, or whose kernel spans do not line up with its submissions, is
``ERROR(instrument)``. That is a fault in this file and is never a finding about either instrument
under comparison.

Usage::

    python bench/results/probe_frames.py                 # the census, over every dev0 trace
    python bench/results/probe_frames.py --trace <path>  # restrict to named traces
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "bench"))

import phases  # noqa: E402  (sys.path set immediately above)

HOT_KERNEL = "q_gemv_matmul_nbits_f16"

# A whole-series RSD at or above this is "a large spread" in the sense Switch's 70% figure meant.
LARGE_SPREAD_RSD = 0.30
# The `gpu_steady_tail` verdicts that release a median a reader could quote. MARGINAL_TAIL is
# excluded on purpose: it parks its median under `withheld_median_ms` precisely so it cannot be.
PUBLISHING_VERDICTS = {"STEADY"}
# A series shorter than this cannot support a whole-vs-suffix comparison at all.
MIN_INFERENCES = 6


class InstrumentError(RuntimeError):
    """R13: this module failed. Never a finding about either instrument under comparison."""


def rsd(xs: "list[float]") -> "float | None":
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = statistics.fmean(xs)
    return (statistics.stdev(xs) / m) if m else None


def per_inference(path: Path) -> dict:
    """Frame A, computed with Niobe's own attribution and her own tail function."""
    events = phases.load(path)
    subs = phases.subgraph_spans(events)
    gpus = phases.gpu_spans(events)
    if not subs or not gpus:
        raise InstrumentError(
            f"{path.name}: {len(subs)} subgraph spans, {len(gpus)} gpu spans -- unparseable")
    ordinal = phases.attribute_gpu_ordinally(subs, gpus)
    busy = ordinal["busy_us"]
    series = [v for v in (busy.get(s["index"], 0.0) / 1000.0 for s in subs) if v]
    tail = phases.gpu_steady_tail([busy.get(s["index"]) for s in subs])
    return {
        "trace": path.name,
        "n_inferences": len(series),
        "whole_series_rsd": rsd(series),
        "whole_series_spread_x": (max(series) / min(series)) if series and min(series) else None,
        "whole_series_min_ms": min(series) if series else None,
        "whole_series_max_ms": max(series) if series else None,
        "tail_verdict": tail.get("verdict"),
        "tail_n": tail.get("n"),
        "tail_coverage": tail.get("coverage"),
        "tail_rsd": tail.get("rsd"),
        "tail_publishes_a_number": tail.get("verdict") in PUBLISHING_VERDICTS,
        "_subs": subs,
        "_gpus": gpus,
    }


def per_kernel(subs: list, gpus: list, trace_name: str) -> dict:
    """Frame B: the individual dispatch. Same cursor walk as `attribute_gpu_ordinally`."""
    per_sub: "list[list[float]]" = []
    cursor = 0
    for s in subs:
        n = s.get("nodes")
        if not isinstance(n, int) or n < 0:
            continue
        take = gpus[cursor:cursor + n]
        cursor += len(take)
        per_sub.append([g["gpu_ns"] / 1000.0 for g in take
                        if g.get("kernel") == HOT_KERNEL and g.get("gpu_ns") is not None])
    per_sub = [p for p in per_sub if p]
    if not per_sub:
        raise InstrumentError(f"{trace_name}: no {HOT_KERNEL} spans attributed to any submission")

    mid = sorted(per_sub, key=len)[len(per_sub) // 2]
    n_hot = len(mid)
    within = rsd(mid)
    full = [p for p in per_sub if len(p) == n_hot]
    across = []
    if len(full) > 1:
        across = [r for r in (rsd([p[k] for p in full]) for k in range(n_hot)) if r is not None]
    med_across = statistics.median(across) if across else None

    # Heterogeneity, not variance: cluster the durations of one inference and count the groups.
    clusters: "list[list[float]]" = []
    for d in sorted(mid):
        if clusters and d <= clusters[-1][-1] * 1.25:
            clusters[-1].append(d)
        else:
            clusters.append([d])

    out = {
        "from_trace": trace_name,
        "dispatches_per_inference": n_hot,
        "within_one_inference_rsd": within,
        "within_one_inference_spread_x": (max(mid) / min(mid)) if min(mid) else None,
        "same_ordinal_across_inferences_median_rsd": med_across,
        "inferences_compared": len(full),
        "duration_clusters": [
            {"n": len(c), "median_us": statistics.median(c)} for c in clusters],
        "n_duration_clusters": len(clusters),
        "predicted_sum_rsd_if_the_spread_were_variance":
            (within / math.sqrt(n_hot)) if within and n_hot else None,
    }
    # The label is DERIVED, not asserted. An earlier revision of this file hard-coded
    # "heterogeneity, not variance" into the output string and then printed it over a contended
    # trace whose same-ordinal RSD was 173% -- the data said variance and the instrument said
    # heterogeneity anyway. A conclusion that survives its own refutation is not a measurement.
    if not across or not min(mid) or within is None:
        out["classification"] = "INSUFFICIENT_DATA"
        out["interpretation"] = "not enough attributed dispatches to classify the spread"
    elif med_across >= within:
        out["classification"] = "VARIANCE_DOMINATED"
        out["interpretation"] = (
            f"The same ordinal varies {med_across:.2%} across inferences against a "
            f"{within:.2%} spread within one inference. Repeating the identical dispatch moves "
            "it as much as changing which dispatch you look at, so this run's spread is "
            "genuine run-to-run variance and the population structure is buried under it. "
            "This is what a disturbed run looks like -- and its tail refuses to publish.")
    else:
        out["classification"] = "HETEROGENEITY_DOMINATED"
        out["interpretation"] = (
            f"The same ordinal repeats across inferences at only {med_across:.2%} RSD while the "
            f"population within one inference spans {(max(mid) / min(mid)):.1f}x across "
            f"{len(clusters)} discrete duration clusters. The spread is population heterogeneity "
            "across node shapes, not run-to-run variance: there is almost no variance to average "
            f"out. 'Averaging out' would predict a sum RSD of {within / math.sqrt(n_hot):.2%}; "
            "the sum is steadier than that because its terms are individually steady and the "
            "spread between them is deterministic and repeats every inference.")
    return out


def verdict_for(census: "list[dict]") -> dict:
    large = [r for r in census if (r["whole_series_rsd"] or 0) >= LARGE_SPREAD_RSD]
    publishing = [r for r in census if r["tail_publishes_a_number"]]
    conflict = [r for r in large if r["tail_publishes_a_number"]]
    counts = {
        "traces": len(census),
        "large_spread": len(large),
        "tail_publishes": len(publishing),
        "both_at_once": len(conflict),
        "large_spread_rsd_threshold": LARGE_SPREAD_RSD,
        "max_whole_rsd_among_publishing":
            max((r["whole_series_rsd"] or 0) for r in publishing) if publishing else None,
        "min_whole_rsd_among_large":
            min((r["whole_series_rsd"] or 0) for r in large) if large else None,
    }
    if not census:
        return {"verdict": "ERROR(instrument)", "counts": counts,
                "detail": "no trace carried enough inferences to compare a whole series to a "
                          "suffix of it"}
    if not large:
        return {"verdict": "INCONCLUSIVE(no large-spread run)", "counts": counts,
                "detail": (f"no committed trace reaches a whole-series RSD of "
                           f"{LARGE_SPREAD_RSD:.0%}, so this evidence cannot exercise the "
                           "question. The 70% run must be re-taken or located before the census "
                           "can adjudicate.")}
    if conflict:
        return {"verdict": "CONFLICT", "counts": counts,
                "conflicting_traces": [r["trace"] for r in conflict],
                "detail": (
                    f"{len(conflict)} trace(s) carry a whole-series RSD >= "
                    f"{LARGE_SPREAD_RSD:.0%} AND a tail that publishes a number. The two "
                    "statements can therefore disagree about a quotable figure, and one of the "
                    "instruments is wrong. Resolve before either certifies anything.")}
    return {"verdict": "SAME_FRAME_ORDERED_SELECTION", "counts": counts,
            "detail": (
                f"Same frame, and no conflict is constructible from the evidence we hold. Across "
                f"{len(census)} committed traces, {len(large)} carry a whole-series RSD >= "
                f"{LARGE_SPREAD_RSD:.0%} and NOT ONE of them publishes a tail figure -- every one "
                "returns NO_STEADY_TAIL, or a MARGINAL_TAIL that withholds its median. "
                f"Conversely the largest whole-series RSD among the {len(publishing)} publishing "
                f"traces is {counts['max_whole_rsd_among_publishing']:.2%}, against a smallest of "
                f"{counts['min_whole_rsd_among_large']:.2%} among the large-spread set: the two "
                "sets are disjoint and well separated. Switch's 70% spread and Niobe's steady "
                "tail have never described the same published number, and cannot -- a large "
                "spread is exactly the condition under which the tail refuses. Both instruments "
                "are correct over their own selection and neither needs repair on this account.")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="reconcile the 70% kernel-time spread against gpu_steady_tail")
    ap.add_argument("--trace", type=Path, action="append",
                    help="restrict the census to these traces (default: every dev0 gemv trace)")
    ap.add_argument("--json", type=Path, default=HERE / "frames.json")
    args = ap.parse_args()

    traces = args.trace or sorted(HERE.glob("trace_gemv_*_dev0.json"))
    report: dict = {
        "instrument": "probe_frames",
        "frame_a": ("per-inference GPU busy = the sum of gpu_ns over every kernel span of that "
                    "inference (phases.attribute_gpu_ordinally). Niobe's tail is a SUFFIX of this "
                    "series; Switch's 70% spread was the WHOLE of it. Same quantity, different "
                    "selection."),
        "frame_b": ("the duration of one q_gemv dispatch -- a genuinely different frame, and the "
                    "one the coordinator hypothesised about."),
        "census": [],
    }
    keep: "dict[str, tuple]" = {}
    try:
        for t in traces:
            if not t.exists():
                raise InstrumentError(f"no trace at {t}")
            row = per_inference(t)
            subs, gpus = row.pop("_subs"), row.pop("_gpus")
            if row["n_inferences"] < MIN_INFERENCES:
                continue
            report["census"].append(row)
            keep[row["trace"]] = (subs, gpus)

        # Frame B is computed on two DELIBERATELY chosen traces, not on whichever came first:
        # the steadiest run in the census and the most disturbed one. Letting a single specimen
        # stand for the population is how the first revision of this file printed
        # "heterogeneity" over a run whose numbers said variance.
        by_rsd = sorted(report["census"], key=lambda r: (r["whole_series_rsd"] or 0))
        picks = {"steadiest": by_rsd[0], "most_disturbed": by_rsd[-1]} if by_rsd else {}
        report["per_kernel"] = {
            label: per_kernel(*keep[row["trace"]], row["trace"])
            for label, row in picks.items()
        }
    except InstrumentError as e:
        report.update(verdict="ERROR(instrument)", reason=str(e))
        args.json.write_text(json.dumps(report, indent=2))
        print(f"ERROR(instrument): {e}", file=sys.stderr)
        return 3

    report.update(verdict_for(report["census"]))
    args.json.write_text(json.dumps(report, indent=2))

    print("== frame A: per-inference GPU busy, whole series vs the steady tail ==")
    print(f"{'trace':<24s} {'n':>4s} {'whole RSD':>10s} {'spread':>8s}  "
          f"{'tail verdict':<15s} {'tail RSD':>9s}")
    for r in sorted(report["census"], key=lambda r: -(r["whole_series_rsd"] or 0)):
        tr = f"{r['tail_rsd']:.4%}" if r["tail_rsd"] is not None else "-"
        flag = ("  <-- large spread, publishes NOTHING"
                if (r["whole_series_rsd"] or 0) >= LARGE_SPREAD_RSD else "")
        print(f"{r['trace'][11:-10]:<24s} {r['n_inferences']:>4d} "
              f"{(r['whole_series_rsd'] or 0):>9.2%} {(r['whole_series_spread_x'] or 0):>7.2f}x  "
              f"{str(r['tail_verdict']):<15s} {tr:>9s}{flag}")

    for label, d in (report.get("per_kernel") or {}).items():
        print(f"\n== frame B ({label}): the individual dispatch, {d['from_trace']} ==")
        print(f"  {d['dispatches_per_inference']} q_gemv dispatches per inference, "
              f"{d['inferences_compared']} inferences compared")
        print(f"  within ONE inference           : RSD {d['within_one_inference_rsd']:.2%}, "
              f"spread {d['within_one_inference_spread_x']:.1f}x, "
              f"{d['n_duration_clusters']} discrete clusters")
        print(f"  same ordinal ACROSS inferences : RSD "
              f"{d['same_ordinal_across_inferences_median_rsd']:.2%}")
        print(f"  [{d['classification']}] {d['interpretation']}")

    print(f"\nverdict: {report['verdict']}\n{report['detail']}")
    print(f"wrote {args.json}")
    return 1 if report["verdict"].startswith(("CONFLICT", "ERROR")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
