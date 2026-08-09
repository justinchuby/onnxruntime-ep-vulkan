"""Where the host span goes, and the tightest bound on each phase that contention allows.

After the column tile and packed loads, NVIDIA GPU busy is ~12.2 ms/inference while the
`vulkan.subgraph` host span is 28-50 ms. The next order of magnitude is host-side, but this
machine cannot be quiet — it is shared with another project running CPU and GPU tests — so an
ordinary mean over host phases is not quotable.

**The estimator that survives contention: the minimum over inferences — and it bounds from
ABOVE, not below.** This is a correction to what an earlier version of this docstring and of
`switch-record-is-host-bound.md` said, and the direction is the whole point of the estimator.

Contention can only ADD host time to a phase: it delays the thread, it does not make work
disappear. Write ``observed_i = true + delay_i`` with ``delay_i >= 0``. Then
``min_i(observed_i) = true + min_i(delay_i) >= true``. So the minimum is an **upper bound on the
uncontended cost**, and the tightest one the data can give. It is not a lower bound. I had the
inequality backwards.

**The consequence, which is not cosmetic.** Two upper bounds do not bound a difference from
below. "record was <= 14.414 ms before" and "record is <= 2.704 ms after" does not, by itself,
prove the change was an improvement at all, let alone by 5.33x. What establishes the direction
here is not timing: it is a **count** — 147,618 `VkBufferMemoryBarrier` structs per inference
before against 354 after, which is exact, contention-independent, and read off the code and the
island shape rather than a clock. The direction is certain; the 5.33x is an estimate under
contention and must be quoted as such, never as a bound.

The median is printed alongside, clearly labelled as contended.

**R11 discipline.** Naming every child of a parent makes a decomposition look closed. This prints
the UNACCOUNTED remainder of every `vulkan.subgraph` span as its own column, so a decomposition
that covers 40% of the span cannot read as one that covers all of it. Phases that open and close
*inside* another phase — `upload`, `cmd_alloc`, `desc_alloc`, `pipeline_lookup`, `cmd_upload`,
`readback` — are reported separately, never added to the top-level sum.

The two lists are **derived from `phases.PHASE_CHILDREN`**, not written out here, so a phase added
to `trace.rs` cannot be silently omitted from the sum and quietly inflate UNACCOUNTED. Issue #88
added three top-level phases (`prepare`, `buffer_alloc`, `writeback`); against the hardcoded list
this probe would have reported all of them as unaccounted and read as if nothing had improved.

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
#
# DERIVED, not listed. An earlier revision hardcoded three names, and when `trace.rs` grew
# `prepare`, `buffer_alloc` and `writeback` this probe kept summing the old three and kept
# reporting the new work as UNACCOUNTED — a decomposition that had improved would have read as
# one that had not. The parenthood now comes from `phases.PHASE_CHILDREN`, which
# `phases.phase_nesting` independently re-derives from timestamp containment and goes red on
# disagreement, so a phase added tomorrow lands in one of these two lists by itself.
_NESTED_NAMES = {c for kids in phases.PHASE_CHILDREN.values() for c in kids}
# `compile` and `prepack` are session-setup phases that do not run inside a Compute call.
_NOT_IN_COMPUTE = {"compile", "prepack"}
TOP = [f"vulkan.{p}" for p in phases.HOST_PHASES
       if p not in _NESTED_NAMES and p not in _NOT_IN_COMPUTE]
# Nested inside a top-level phase. Reported for shape, NEVER added to the top-level sum.
NESTED = [f"vulkan.{p}" for p in phases.HOST_PHASES if p in _NESTED_NAMES]


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
    upper_bound = sum(stat[k]["min_ms"] for k in TOP + ["unaccounted"])
    return {
        "trace": trace.name,
        "state": "OK",
        "inferences": len(rows),
        "cold": {k: round(rows[0][k] / 1000, 3) for k in keys},
        "warm": stat,
        "host_upper_bound_ms": round(upper_bound, 3),
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
        print(f"\n  host span UPPER BOUND (sum of per-phase minima; contention only adds): "
              f"{r['host_upper_bound_ms']:.3f} ms   "
              f"vs GPU busy min {r['warm']['gpu']['min_ms']:.3f} ms")
        print(f"  named children cover {r['accounted_share_of_median']:.1%} of the median span "
              f"— the rest is the 'unaccounted' row, not zero")

    outp = here / "hostphase.json"
    outp.write_text(json.dumps(out, indent=2), "utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
