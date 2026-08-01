"""Where the host span goes, and a lower bound on each phase that contention cannot inflate.

After the column tile and packed loads, NVIDIA GPU busy is ~12.2 ms/inference while the
`vulkan.subgraph` host span is 28-50 ms. The next order of magnitude is host-side, but this
machine cannot be quiet (the coordinator's orchestration is the load), so an ordinary mean over
host phases is not quotable.

**The estimator that survives contention: the minimum over inferences.** Contention can only ADD
host time to a phase — it delays the thread, it does not make work disappear. So `min` over a
series of identical repetitions is a *lower bound* on that phase's uncontended cost, and the sum
of the minima is a lower bound on the uncontended host span. Bounds are quotable where means are
not. The median is printed alongside, clearly labelled as contended.

**R11 discipline.** Naming every child of a parent makes a decomposition look closed. This prints
the UNACCOUNTED remainder of every `vulkan.subgraph` span as its own column, so a decomposition
that covers 40% of the span cannot read as one that covers all of it. `pipeline_lookup` and
`desc_alloc` are nested inside `record` and are reported separately, never added to the top-level
sum.

Usage:
    python bench/results/probe_hostphase.py trace_a.json [trace_b.json ...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import phases  # noqa: E402

SUB = "vulkan.subgraph"
# Top-level children of one subgraph span. These are summed; the remainder is reported.
TOP = ["vulkan.record", "vulkan.submit", "vulkan.fence_wait"]
# Nested inside `record`. Reported for shape, NEVER added to the top-level sum.
NESTED = ["vulkan.cmd_upload", "vulkan.pipeline_lookup", "vulkan.desc_alloc"]


def decompose(trace: Path) -> dict:
    events = [e for e in phases.load(trace) if e.get("ph") == "X"]
    subs = sorted((e for e in events if e.get("name") == SUB), key=lambda e: e["ts"])
    if len(subs) < 3:
        return {"trace": trace.name, "state": "ERROR(instrument)",
                "detail": f"{len(subs)} subgraph span(s)"}
    by = {n: sorted((e for e in events if e.get("name") == n), key=lambda e: e["ts"])
          for n in TOP + NESTED}

    gpus = phases.gpu_spans(events)
    busy = phases.attribute_gpu_ordinally(phases.subgraph_spans(events), gpus)["busy_us"]

    rows = []
    for i, s in enumerate(subs):
        lo, hi = s["ts"], s["ts"] + s.get("dur", 0)
        v = {n: sum(e.get("dur", 0) for e in by[n] if lo <= e["ts"] < hi) for n in TOP + NESTED}
        v["subgraph"] = s.get("dur", 0)
        v["unaccounted"] = v["subgraph"] - sum(v[n] for n in TOP)
        v["gpu"] = busy.get(i, 0.0)
        rows.append(v)

    # Inference 0 is pipeline compile — a different regime, excluded from every statistic and
    # printed on its own line so it cannot be quietly averaged in.
    warm = rows[1:]
    keys = TOP + ["unaccounted", "subgraph", "gpu"] + NESTED
    stat = {
        k: {
            "min_ms": round(min(r[k] for r in warm) / 1000, 3),
            "median_ms": round(statistics.median(r[k] for r in warm) / 1000, 3),
            "max_ms": round(max(r[k] for r in warm) / 1000, 3),
        }
        for k in keys
    }
    lower_bound = sum(stat[k]["min_ms"] for k in TOP + ["unaccounted"])
    return {
        "trace": trace.name,
        "state": "OK",
        "inferences": len(rows),
        "cold": {k: round(rows[0][k] / 1000, 3) for k in keys},
        "warm": stat,
        "host_lower_bound_ms": round(lower_bound, 3),
        "accounted_share_of_median": round(
            sum(stat[k]["median_ms"] for k in TOP) / stat["subgraph"]["median_ms"], 4),
    }


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    here = Path(__file__).resolve().parent
    out = []
    for a in args:
        p = Path(a)
        if not p.is_absolute():
            p = here / a
        if not p.is_file():
            print(f"ERROR(instrument): no such trace: {p}")
            return 2
        r = decompose(p)
        out.append(r)
        if r["state"] != "OK":
            print(f"{r['trace']}: {r['state']} — {r['detail']}")
            continue
        print(f"\n== {r['trace']} · {r['inferences']} inferences ==")
        print(f"{'phase':<26} {'min ms':>9} {'median ms':>10} {'max ms':>9}   "
              f"(cold inference 0: ms)")
        for k in TOP + ["unaccounted", "subgraph", "gpu"]:
            s = r["warm"][k]
            print(f"{k:<26} {s['min_ms']:>9.3f} {s['median_ms']:>10.3f} {s['max_ms']:>9.3f}   "
                  f"{r['cold'][k]:>10.3f}")
        print("  -- nested inside record, NOT part of the sum above --")
        for k in NESTED:
            s = r["warm"][k]
            print(f"{k:<26} {s['min_ms']:>9.3f} {s['median_ms']:>10.3f} {s['max_ms']:>9.3f}   "
                  f"{r['cold'][k]:>10.3f}")
        print(f"\n  host span LOWER BOUND (sum of per-phase minima): "
              f"{r['host_lower_bound_ms']:.3f} ms   "
              f"vs GPU busy min {r['warm']['gpu']['min_ms']:.3f} ms")
        print(f"  named children cover {r['accounted_share_of_median']:.1%} of the median span "
              f"— the rest is the 'unaccounted' row, not zero")

    outp = here / "hostphase.json"
    outp.write_text(json.dumps(out, indent=2), "utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
