"""Reproduce the two Switch findings that the summary table cannot show on its own.

Both are R11 exposures: a phase whose NAME is not its CONTENT.

  1. `record` residual — `record` is an inclusive bracket. Its named children (`cmd_upload`,
     `desc_alloc`, `pipeline_lookup`) collapse to ~1-2 ms once the weight cache is warm, so the
     unnamed remainder (the vkCmd* calls themselves) is ~90% of the phase. The EP summary prints
     the residual, but CUMULATIVELY across every Compute call, which mixes the cold and warm
     regimes into a share belonging to no single call. Per-call is only visible here.

  2. `fence_wait` idle — the share of the host's fence wait during which no kernel was running.
     Two estimators are printed on purpose because they DISAGREE:

       overlap:  intersect GPU spans with the fence_wait window on the host axis.
       cluster:  compare the fence_wait duration to the summed duration of the kernel cluster it
                 waited on, grouping GPU spans by the large gaps between submissions.

     `overlap` depends on the GPU->host calibration anchor being accurate; `cluster` uses only
     durations and is alignment-free. When they disagree the difference is anchor drift, i.e.
     ERROR(instrument), not a measurement of idle (R13). Quote the cluster figure, and only from a
     run whose verdict is an attributed MATCH.

Usage:
    python bench/results/probe_record_residual.py <trace.json>

Produce the trace with:
    $env:ONNXRUNTIME_EP_VULKAN_TRACE  = "...\\bench\\results\\switch_trace.json"
    $env:ONNXRUNTIME_EP_VULKAN_TRACE_GPU = "1"
    python bench/phi35.py --worker --device 0 --iters 2 --warmup 1 --out ... --scratch ...
"""

from __future__ import annotations

import collections
import json
import sys

NAMED_RECORD_CHILDREN = ("cmd_upload", "desc_alloc", "pipeline_lookup")


def phase(event: dict) -> str:
    return event["name"].split("vulkan.")[-1]


def load(path: str) -> list[dict]:
    doc = json.load(open(path, encoding="utf-8"))
    events = doc["traceEvents"] if isinstance(doc, dict) else doc
    return [e for e in events if e.get("ph") == "X"]


def by_phase(spans: list[dict], name: str) -> list[dict]:
    return sorted((e for e in spans if phase(e) == name), key=lambda e: e["ts"])


def record_residuals(spans: list[dict]) -> None:
    print("\n-- `record` residual, per Compute call (us -> ms) --")
    for i, rec in enumerate(by_phase(spans, "record")):
        lo, hi = rec["ts"], rec["ts"] + rec["dur"]
        kids: collections.Counter = collections.Counter()
        for e in spans:
            if e is rec:
                continue
            nm = phase(e)
            if nm in NAMED_RECORD_CHILDREN and e["ts"] >= lo and e["ts"] + e.get("dur", 0) <= hi:
                kids[nm] += e["dur"]
        named = sum(kids.values())
        residual = rec["dur"] - named
        print(
            f"  call{i}: record {rec['dur'] / 1000:9.1f} ms | "
            + " ".join(f"{k} {kids[k] / 1000:7.1f}" for k in NAMED_RECORD_CHILDREN)
            + f" | RESIDUAL {residual / 1000:9.1f} ms ({100 * residual / rec['dur']:5.1f}%)"
        )


def gpu_union(spans: list[dict]) -> list[tuple[int, int]]:
    raw = sorted((e["ts"], e["ts"] + e["dur"]) for e in spans if phase(e).startswith("gpu."))
    merged: list[tuple[int, int]] = []
    for a, b in raw:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def gpu_clusters(spans: list[dict], k: int) -> list[list[tuple[int, int]]]:
    """Group GPU spans into `k` submissions by the largest inter-span gaps.

    Alignment-free: only relative GPU-side ordering is used, never the host anchor.
    """
    g = sorted((e["ts"], e["ts"] + e["dur"]) for e in spans if phase(e).startswith("gpu."))
    if not g or k <= 1:
        return [g]
    gaps = sorted(((g[i + 1][0] - g[i][1], i) for i in range(len(g) - 1)), reverse=True)[: k - 1]
    out, start = [], 0
    for cut in sorted(i for _, i in gaps) + [len(g) - 1]:
        out.append(g[start : cut + 1])
        start = cut + 1
    return out


def fence_idle(spans: list[dict]) -> None:
    waits = by_phase(spans, "fence_wait")
    if not waits:
        print("\n-- fence_wait: UNOBSERVABLE (no fence_wait spans in this trace) --")
        return
    union = gpu_union(spans)
    clusters = gpu_clusters(spans, len(waits))
    print("\n-- fence_wait idle, two estimators (they are supposed to be compared, not averaged) --")
    for i, wait in enumerate(waits):
        lo, hi = wait["ts"], wait["ts"] + wait["dur"]
        covered = sum(max(0, min(hi, b) - max(lo, a)) for a, b in union)
        cluster = clusters[i] if i < len(clusters) else []
        busy = sum(b - a for a, b in cluster)
        wall = (cluster[-1][1] - cluster[0][0]) if cluster else 0
        print(
            f"  call{i}: fence_wait {wait['dur'] / 1000:7.1f} ms | "
            f"overlap idle {100 * (1 - covered / wait['dur']):5.1f}% | "
            f"cluster busy {busy / 1000:7.1f} ms (n={len(cluster)}) "
            f"idle {100 * (1 - busy / wait['dur']):5.1f}% | "
            f"between-kernel bubbles {(wall - busy) / 1000:5.1f} ms"
        )
    print(
        "  NOTE: 'between-kernel bubbles' near zero means the idle is at the SUBMISSION EDGES\n"
        "  (submit->first kernel, last kernel->host wake), not between dispatches."
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    spans = load(sys.argv[1])
    print(f"spans: {dict(collections.Counter(phase(e).split('.')[0] for e in spans))}")
    record_residuals(spans)
    fence_idle(spans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
