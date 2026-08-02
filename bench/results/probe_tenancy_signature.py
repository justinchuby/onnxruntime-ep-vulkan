"""``probe_tenancy_signature`` — does ``PER_DISPATCH`` track witnessed foreign tenancy, and does
the tail statistic move under it?

THE QUESTION, AND WHY IT IS NOT THE ONE I WAS ASKED
=====================================================
I was handed a two-trace correspondence and a reading of it:

    ab_p1_long   35.31% -> 3.90%   ( 88.9% explained)  SUBMISSION_LEVEL
    contended3   41.11% -> 50.43%  (-22.7% explained)  PER_DISPATCH

    "PER_DISPATCH is the signature that foreign GPU work should produce."

Both traces have a committed device-state companion, so the correspondence is checkable rather than
plausible, and on those two traces **it holds**: ``contended3`` is ``FOREIGN_GPU_WORK`` at
``foreign_sample_fraction = 1.0``, ``ab_p1_long`` is ``SOLE_TENANT`` at 0.0031.

Two traces that agree is where four wrong conclusions started this week. The census has **nine**
device-0 traces with a companion, **four** of them witnessed ``FOREIGN_GPU_WORK``, and the
hypothesis is falsifiable against all of them at once. So this probe cross-tabulates every trace
that has both a localisation and a tenancy record, and reports the disagreements first.

WHAT IT IMPORTS RATHER THAN REBUILDS
-------------------------------------
``localise`` and ``per_inference_kernel_us`` are Switch's, from ``bench/run_disturbance.py``.
``gpu_steady_tail`` and ``steady_state_split`` are mine, from ``bench/phases.py``. Neither is
reimplemented here: a second copy of a classifier is a second thing to be wrong, and this probe's
whole value is that the two sides are the *same* code that produced the figures being compared.
The tenancy verdicts are read from Switch's committed ``gpustate_*.json`` companions.

WHAT IT MEASURES, AND THE TWO QUESTIONS ARE NOT THE SAME
----------------------------------------------------------
1. **Correspondence.** Does ``character`` (``PER_DISPATCH`` / ``SUBMISSION_LEVEL`` / ``MIXED``)
   track the witnessed tenancy verdict?
2. **Does the tail statistic move?** Not *does it publish* — large-spread traces already refuse,
   which is established and is a different question. Whether the **level the tail reports** shifts
   under witnessed foreign work. A tail that refuses is a tail that saw something; a tail that
   publishes an unchanged number under witnessed contention is blind to it.

R13: a trace that cannot be read is ``ERROR(instrument)`` and is listed as such, never counted as
a clean or a disturbed one.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent

_SYS_PATH_BEFORE = list(sys.path)
sys.path.insert(0, str(BENCH))
try:
    import phases  # noqa: E402
    import run_disturbance as rd  # noqa: E402
finally:
    sys.path[:] = _SYS_PATH_BEFORE

#: The raw traces are not in this repository. They are Switch's, read-only, and the default points
#: at his worktree; ``--traces`` overrides. If they are absent the probe reports ERROR(instrument)
#: for every trace rather than reporting a census of zero disturbed runs.
DEFAULT_TRACES = Path(os.environ.get(
    "NIOBE_TRACE_DIR", r"C:\Users\justinchu\dev\ep-vulkan-switch\bench\results"))

#: A companion is only evidence about the window it covers. These are Switch's, one per tag.
COMPANIONS = HERE


def _tag_of(trace_name: str) -> str:
    """``trace_gemv_contended3_dev0.json`` -> ``contended3``."""
    stem = trace_name[len("trace_gemv_"):] if trace_name.startswith("trace_gemv_") else trace_name
    for suffix in ("_dev0.json", "_dev1.json", ".json"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _companion(tag: str) -> "dict | None":
    p = COMPANIONS / f"gpustate_{tag}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _per_inference_totals(inferences: "list[list[float]]") -> "list[float]":
    """Per-inference GPU-busy in **microseconds**, summed over the inference's dispatches.

    This is the series ``gpu_steady_tail`` consumes, and it consumes microseconds --- its parameter
    is ``busy_us`` and it does the conversion to ms itself. The first cut of this function divided
    by 1000 here as well, and every printed level came out 1000x low (``soloA`` as 0.0115 rather
    than 11.53). ERROR(instrument), caught before publication and recorded rather than quietly
    corrected: every verdict, RSD and ratio in this probe is scale-invariant and did not move, which
    is exactly why the mistake survived a first reading of the output. A units error that changes no
    verdict is the kind that gets published.
    """
    return [sum(inf) for inf in inferences]


def survey(trace_dir: Path) -> dict:
    rows, errors = [], []
    for path in sorted(trace_dir.glob("trace_gemv_*_dev0.json")):
        tag = _tag_of(path.name)
        try:
            inferences = rd.per_inference_kernel_us(path)
            if len(inferences) < 3:
                raise ValueError(f"only {len(inferences)} inference(s) recoverable")
            loc = rd.localise(inferences)
            totals = _per_inference_totals(inferences)
            tail = phases.gpu_steady_tail(totals)
        except Exception as exc:  # noqa: BLE001
            errors.append({"trace": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        comp = _companion(tag)
        whole_rsd = rd._rsd(totals)
        rows.append({
            "tag": tag,
            "trace": path.name,
            "n_inferences": len(inferences),
            "character": loc.get("character", "").split(":")[0] or None,
            "same_ordinal_rsd": loc.get("same_ordinal_rsd_median"),
            "level_normalised": loc.get("level_normalised_ordinal_rsd"),
            "explained_by_level": loc.get("explained_by_level"),
            "whole_series_rsd": whole_rsd,
            "median_ms": statistics.median(totals) / 1000.0,
            "tail_verdict": tail.get("verdict"),
            "tail_median_ms": tail.get("median_ms"),
            "tail_withheld_ms": tail.get("withheld_median_ms"),
            "tail_rsd": tail.get("rsd"),
            "tail_n": tail.get("n"),
            "tenancy": (comp or {}).get("verdict"),
            "foreign_fraction": (comp or {}).get("foreign_sample_fraction"),
            "sm_peak_mhz": ((comp or {}).get("sm_mhz") or {}).get("max"),
            "sm_max_mhz": (comp or {}).get("sm_max_mhz"),
        })
    return {"rows": rows, "instrument_errors": errors}


def correspondence(rows: "list[dict]") -> dict:
    """Question 1: does ``PER_DISPATCH`` track witnessed foreign tenancy?

    Reported as a 2x2 over the traces that have **both** signals. Agreements and disagreements are
    both listed by name, because a count of disagreements is not actionable and a named trace is.
    """
    witnessed = [r for r in rows if r["tenancy"] in ("FOREIGN_GPU_WORK", "SOLE_TENANT")
                 and r["character"]]
    cells = {"foreign_per_dispatch": [], "foreign_not_per_dispatch": [],
             "sole_per_dispatch": [], "sole_not_per_dispatch": []}
    for r in witnessed:
        foreign = r["tenancy"] == "FOREIGN_GPU_WORK"
        pd = r["character"] == "PER_DISPATCH"
        key = ("foreign_" if foreign else "sole_") + ("per_dispatch" if pd else "not_per_dispatch")
        cells[key].append({"tag": r["tag"], "character": r["character"],
                           "foreign_fraction": r["foreign_fraction"],
                           "explained_by_level": r["explained_by_level"]})
    tp = len(cells["foreign_per_dispatch"])
    fn = len(cells["foreign_not_per_dispatch"])
    fp = len(cells["sole_per_dispatch"])
    tn = len(cells["sole_not_per_dispatch"])
    return {
        "n_with_both_signals": len(witnessed),
        "cells": cells,
        "counts": {"foreign_and_per_dispatch": tp, "foreign_but_not_per_dispatch": fn,
                   "sole_but_per_dispatch": fp, "sole_and_not_per_dispatch": tn},
        "sensitivity": (tp / (tp + fn)) if (tp + fn) else None,
        "specificity": (tn / (tn + fp)) if (tn + fp) else None,
    }


def tail_movement(rows: "list[dict]") -> dict:
    """Question 2: does the tail statistic *move* under witnessed foreign work?

    The comparison is only meaningful within one probe on one device, so it is restricted to the
    ``probe_gemv.py`` family the companions cover, and the reference is the sole-tenant traces that
    are also at a boosted clock -- a sole-tenant run pinned at idle clock is the 21.4x specimen and
    must not be used as a reference for anything.
    """
    ref, foreign, excluded = [], [], []
    for r in rows:
        if r["tenancy"] is None:
            continue
        boosted = (r["sm_peak_mhz"] or 0) >= 0.5 * (r["sm_max_mhz"] or 1)
        published = r["tail_verdict"] == "STEADY"
        level = r["tail_median_ms"] if r["tail_median_ms"] is not None else r["tail_withheld_ms"]
        entry = {"tag": r["tag"], "tenancy": r["tenancy"], "tail_verdict": r["tail_verdict"],
                 "tail_level_ms": level, "level_is_withheld": not published,
                 "tail_rsd": r["tail_rsd"], "tail_n": r["tail_n"],
                 "median_ms": r["median_ms"], "sm_peak_mhz": r["sm_peak_mhz"],
                 "character": r["character"]}
        if not boosted:
            entry["excluded_because"] = ("peak SM clock never left idle; this is the bias specimen, "
                                         "not a usable reference or subject")
            excluded.append(entry)
        elif r["tenancy"] == "SOLE_TENANT":
            ref.append(entry)
        else:
            foreign.append(entry)

    # The reference is built ONLY from tails that published. A withheld MARGINAL_TAIL median used
    # as a denominator is that median published by the back door, and §16 says it never publishes.
    # The first cut of this function took the median over all four sole-tenant levels and got
    # 15.5159 ms -- a reference that exists nowhere, half-built from numbers the instrument refused
    # to release. Caught here; ERROR(instrument), not a result.
    ref_levels = [e["tail_level_ms"] for e in ref
                  if e["tail_level_ms"] is not None and not e["level_is_withheld"]]
    baseline = statistics.median(ref_levels) if ref_levels else None
    for e in foreign:
        e["x_vs_sole_tenant"] = (e["tail_level_ms"] / baseline) if (baseline and
                                                                   e["tail_level_ms"]) else None
    return {"sole_tenant_reference_ms": baseline,
            "reference_built_from": [e["tag"] for e in ref if not e["level_is_withheld"]],
            "sole_tenant": ref,
            "foreign_gpu_work": foreign, "excluded": excluded}


def truncation_sweep(trace_dir: Path, tag: str, steps=(20, 28, 34, 40, 46, 52, 60)) -> dict:
    """The falsifier for "the tail refused, so the tail saw it".

    ``contended3`` refuses only because it ran long enough to decay. Feed the same series to the
    same function at increasing lengths and watch the verdict flip. If a prefix publishes, the
    refusal is a property of the run's length, not of the instrument's sensitivity.
    """
    path = trace_dir / f"trace_gemv_{tag}_dev0.json"
    if not path.is_file():
        return {"verdict": "ERROR(instrument)", "detail": f"{path.name} not readable"}
    inferences = rd.per_inference_kernel_us(path)
    totals = _per_inference_totals(inferences)
    out = []
    for k in steps:
        if k > len(totals):
            continue
        t = phases.gpu_steady_tail(totals[:k])
        out.append({"truncated_to": k, "verdict": t.get("verdict"),
                    "median_ms": t.get("median_ms"), "withheld_ms": t.get("withheld_median_ms"),
                    "rsd": t.get("rsd"), "n": t.get("n")})
    out.append({"truncated_to": len(totals), "verdict": phases.gpu_steady_tail(totals).get("verdict"),
                "median_ms": phases.gpu_steady_tail(totals).get("median_ms"),
                "withheld_ms": phases.gpu_steady_tail(totals).get("withheld_median_ms"),
                "rsd": phases.gpu_steady_tail(totals).get("rsd"),
                "n": phases.gpu_steady_tail(totals).get("n")})
    return {"tag": tag, "n_inferences": len(totals), "sweep": out}


def character_boundary_sweep(rows: "list[dict]") -> dict:
    """Is the correspondence's failure an artifact of where the cut sits?

    ``_character`` calls it ``PER_DISPATCH`` at ``explained_by_level <= 0.20`` and
    ``SUBMISSION_LEVEL`` at ``>= 0.60``. Reporting that the correspondence fails at one cut point
    is weak: the obvious reply is "then move the cut". So sweep every cut from -0.5 to 1.0 and
    publish the whole table, the way the 5%-vs-7 threshold episode should have been handled the
    first time. If the correspondence never separates at any cut, that is a statement about the
    signal and not about the boundary, and no boundary needs choosing.

    The rule swept is the strongest form of the hypothesis: "low ``explained_by_level`` means
    foreign GPU work". Cut ``c`` calls a trace foreign when ``explained_by_level <= c``.
    """
    both = [r for r in rows if r["tenancy"] in ("FOREIGN_GPU_WORK", "SOLE_TENANT")
            and r["explained_by_level"] is not None]
    table = []
    for i in range(-50, 105, 5):
        c = i / 100.0
        tp = sum(1 for r in both if r["tenancy"] == "FOREIGN_GPU_WORK"
                 and r["explained_by_level"] <= c)
        fn = sum(1 for r in both if r["tenancy"] == "FOREIGN_GPU_WORK"
                 and r["explained_by_level"] > c)
        fp = sum(1 for r in both if r["tenancy"] == "SOLE_TENANT"
                 and r["explained_by_level"] <= c)
        tn = sum(1 for r in both if r["tenancy"] == "SOLE_TENANT"
                 and r["explained_by_level"] > c)
        table.append({"cut": c, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                      "correct": tp + tn, "of": len(both),
                      "youden_j": ((tp / (tp + fn)) if (tp + fn) else 0.0)
                                  - ((fp / (fp + tn)) if (fp + tn) else 0.0)})
    best = max(table, key=lambda t: t["youden_j"]) if table else None
    return {
        "n": len(both),
        "n_foreign": sum(1 for r in both if r["tenancy"] == "FOREIGN_GPU_WORK"),
        "n_sole": sum(1 for r in both if r["tenancy"] == "SOLE_TENANT"),
        "table": table,
        "best_cut_by_youden_j": best,
        "values_by_trace": sorted(
            [{"tag": r["tag"], "tenancy": r["tenancy"],
              "explained_by_level": r["explained_by_level"]} for r in both],
            key=lambda e: e["explained_by_level"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    ap.add_argument("--out", type=Path, default=HERE / "tenancy_signature.json")
    args = ap.parse_args(argv)

    if not args.traces.is_dir():
        print(f"ERROR(instrument): trace directory not readable: {args.traces}")
        return 2

    s = survey(args.traces)
    rows = s["rows"]
    report = {
        "probe": "probe_tenancy_signature",
        "traces_dir": str(args.traces),
        "n_traces": len(rows),
        "instrument_errors": s["instrument_errors"],
        "correspondence": correspondence(rows),
        "tail_movement": tail_movement(rows),
        "character_boundary_sweep": character_boundary_sweep(rows),
        "contended3_truncation": truncation_sweep(args.traces, "contended3"),
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    c = report["correspondence"]
    print(f"traces read: {len(rows)}   instrument errors: {len(s['instrument_errors'])}")
    for e in s["instrument_errors"]:
        print(f"  ERROR(instrument) {e['trace']}: {e['error']}")
    print()
    print("=== Q1  does PER_DISPATCH track witnessed foreign tenancy? ===")
    print(f"traces with both signals: {c['n_with_both_signals']}")
    for k, v in c["counts"].items():
        print(f"  {k:<32} {v}")
    for k in ("foreign_but_not_per_dispatch", "sole_but_per_dispatch"):
        for e in c["cells"][k.replace("foreign_but_not", "foreign_not")
                            .replace("sole_but", "sole")]:
            print(f"  DISAGREES  {e['tag']:<22} {e['character']:<18} "
                  f"foreign_fraction={e['foreign_fraction']}")
    print()
    print("=== Q2  does the tail LEVEL move under witnessed foreign work? ===")
    tm = report["tail_movement"]
    print(f"sole-tenant, boosted-clock reference: {tm['sole_tenant_reference_ms']}")
    for e in tm["sole_tenant"] + tm["foreign_gpu_work"]:
        print(f"  {e['tag']:<22} {e['tenancy']:<18} {str(e['tail_verdict']):<15} "
              f"level={e['tail_level_ms']}  x={e.get('x_vs_sole_tenant')}")
    for e in tm["excluded"]:
        print(f"  EXCLUDED {e['tag']:<20} {e['excluded_because']}")
    print()
    print("=== Q1b  does ANY cut of explained_by_level separate tenancy? ===")
    bs = report["character_boundary_sweep"]
    print(f"n={bs['n']}  foreign={bs['n_foreign']}  sole={bs['n_sole']}")
    for e in bs["values_by_trace"]:
        print(f"  {e['tag']:<22} {e['tenancy']:<18} explained_by_level={e['explained_by_level']:+.4f}")
    print("  cut     tp fn fp tn  correct  YoudenJ")
    for t in bs["table"]:
        print(f"  {t['cut']:+.2f}   {t['tp']}  {t['fn']}  {t['fp']}  {t['tn']}   "
              f"{t['correct']}/{t['of']}     {t['youden_j']:+.3f}")
    b = bs["best_cut_by_youden_j"]
    print(f"  best achievable: cut={b['cut']:+.2f} correct={b['correct']}/{b['of']} J={b['youden_j']:+.3f}")
    print()
    print("=== contended3 truncation sweep ===")
    for r in report["contended3_truncation"].get("sweep", []):
        print(f"  n<={r['truncated_to']:<4} {str(r['verdict']):<15} "
              f"median={r['median_ms']}  withheld={r['withheld_ms']}  rsd={r['rsd']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
