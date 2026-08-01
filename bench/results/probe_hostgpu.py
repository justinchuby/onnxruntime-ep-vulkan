"""Is the spread host-side or device-side? Same trace, both clocks, per inference.

The coordinator's `HOST_SIDE_EXCURSIONS` signature reports host spread >= 2.0x against GPU
spread <= 1.25x for repetitions of identical work. That is a *within-run* ratio, so contention
cannot manufacture it. This probe reproduces the same comparison from my own traces at my own
commit, and extends it in the one direction that decides my open question:

    WITHIN-RUN  spread of GPU busy across inferences of one process
    BETWEEN-RUN spread of GPU busy across repetitions of the SAME BUILD in separate processes

My withdrawn "70% kernel-time spread" (49.4 -> 83.8 -> 71.0 -> 58.5 ms, n=353 dispatches each)
was a *between-run* device-clock figure, not a host one. Niobe's 0.033% RSD is a within-run
steady suffix. Those are different quantities and this prints both, so the reconciliation is a
measurement rather than an argument.

Host time here is the sum of `vulkan.subgraph` span durations per inference -- the host axis,
`dur` in microseconds. GPU time is `gpu_ns` attributed ordinally, never by timestamp.

Usage:
    python bench/results/probe_hostgpu.py trace_a.json trace_b.json ...
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import phases  # noqa: E402


def per_inference(trace: Path) -> dict:
    """Host µs and GPU µs per inference, from one trace, paired by inference index."""
    events = phases.load(trace)
    subs = phases.subgraph_spans(events)
    gpus = phases.gpu_spans(events)
    if not subs or not gpus:
        return {"trace": trace.name, "state": "ERROR(instrument)",
                "detail": f"{len(subs)} subgraph spans, {len(gpus)} gpu spans"}
    acc = phases.gpu_span_accounting(subs, gpus)
    busy = phases.attribute_gpu_ordinally(subs, gpus)["busy_us"]

    period = phases._cycle_period([s["nodes"] for s in subs]) or 1
    n_inf = len(subs) // period
    if n_inf < 2:
        return {"trace": trace.name, "state": "ERROR(instrument)",
                "detail": f"{n_inf} inference(s) — nothing to spread"}

    host, gpu = [], []
    for c in range(n_inf):
        window = subs[c * period:(c + 1) * period]
        host.append(sum(s["dur"] for s in window))
        gpu.append(sum(busy.get(s["index"], 0.0) for s in window))

    tail = phases.gpu_steady_tail(gpu)
    drop = tail.get("discarded_inferences", 0) if tail["verdict"] == "STEADY" else 0

    # Inference 0 is a known regime of its own: cold caches, first submission, and on the host
    # side the whole pipeline compile. Reporting it inside a "spread" would let one known event
    # stand in for variability. Reported separately instead, never averaged in silently.
    warm_h, warm_g = host[1:], gpu[1:]

    return {
        "trace": trace.name,
        "state": "OK",
        "accounting": acc.get("verdict", acc) if isinstance(acc, dict) else acc,
        "inferences": n_inf,
        "dispatches_per_inference": sum(s["nodes"] or 0 for s in subs[:period]),
        "host_ms": [round(v / 1000, 3) for v in host],
        "gpu_ms": [round(v / 1000, 3) for v in gpu],
        "cold_host_ms": round(host[0] / 1000, 3),
        "cold_gpu_ms": round(gpu[0] / 1000, 3),
        "host_spread_warm": round(max(warm_h) / min(warm_h), 4) if min(warm_h) else None,
        "gpu_spread_warm": round(max(warm_g) / min(warm_g), 4) if min(warm_g) else None,
        "gpu_steady_tail": tail,
        "steady_drop": drop,
        # The tail is the regime Niobe quotes. Report the host spread over the SAME window, so
        # the two are compared on the same inferences and not across a regime boundary.
        "host_spread_on_tail": (round(max(host[drop:]) / min(host[drop:]), 4)
                                if len(host[drop:]) >= 2 and min(host[drop:]) else None),
        "gpu_spread_on_tail": (round(max(gpu[drop:]) / min(gpu[drop:]), 4)
                               if len(gpu[drop:]) >= 2 and min(gpu[drop:]) else None),
        "gpu_median_tail_ms": round(statistics.median(gpu[drop:]) / 1000, 3) if gpu[drop:] else None,
    }


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    here = Path(__file__).resolve().parent
    reps = []
    for a in args:
        p = Path(a)
        if not p.is_absolute():
            p = here / a
        if not p.is_file():
            print(f"ERROR(instrument): no such trace: {p}")
            return 2
        reps.append(per_inference(p))

    print(f"{'trace':<40} {'n':>3} {'cold h':>8} {'cold g':>7} {'host x':>7} {'gpu x':>7} "
          f"{'gpu med ms':>10}  tail")
    for r in reps:
        if r["state"] != "OK":
            print(f"{r['trace']:<40} {r['state']}: {r['detail']}")
            continue
        t = r["gpu_steady_tail"]
        print(f"{r['trace']:<40} {r['inferences']:>3} "
              f"{r['cold_host_ms']:>8.1f} {r['cold_gpu_ms']:>7.1f} "
              f"{r['host_spread_warm']:>7.3f} {r['gpu_spread_warm']:>7.3f} "
              f"{(r['gpu_median_tail_ms'] or 0):>10.3f}  {t['verdict']}")

    ok = [r for r in reps if r["state"] == "OK" and r["gpu_median_tail_ms"]]
    if len(ok) >= 2:
        meds = [r["gpu_median_tail_ms"] for r in ok]
        hosts = [statistics.median(r["host_ms"][r["steady_drop"]:]) for r in ok]
        print(f"\nBETWEEN-RUN, steady tails only, {len(ok)} process(es):")
        print(f"  GPU  median-of-medians {statistics.median(meds):.3f} ms, "
              f"spread {max(meds) / min(meds):.4f}x  ({min(meds):.3f} .. {max(meds):.3f})")
        print(f"  HOST median-of-medians {statistics.median(hosts):.3f} ms, "
              f"spread {max(hosts) / min(hosts):.4f}x  ({min(hosts):.3f} .. {max(hosts):.3f})")

    out = here / "hostgpu_spread.json"
    out.write_text(json.dumps(reps, indent=2), "utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
