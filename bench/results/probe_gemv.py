"""GPU-busy per inference and the per-kernel breakdown, from the device clock only.

Why this exists: `q_gemv_matmul_nbits_f16` is ~95% of all GPU time on the discrete part and
~98% on the integrated one, so it is the only kernel Amdahl allows anyone to work on. This probe
is the falsifier for "the change made it faster": it runs the traced worker, reconstructs the
per-inference GPU-busy series from `gpu_ns` (device timestamp counter, which host load cannot
touch), and hands it to `phases.gpu_steady_tail`, which REFUSES to report a number unless a
suffix of >= 5 inferences holds within 2% RSD.

Three terminal states, never two (R13):
    STEADY           -> a quotable GPU-busy ms/inference
    NO_STEADY_TAIL   -> FAIL(condition): the device never settled. No number.
    INSUFFICIENT     -> ERROR(instrument): too few usable inferences. No number.

Wall clock is deliberately not reported. It is contended on this machine and withheld by three
independent instruments; the device counter is not.

Usage:
    python bench/results/probe_gemv.py --device 0 --iters 14 --tag baseline
    python bench/results/probe_gemv.py --trace <existing.trace.json>       # re-analyse only
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import phases  # noqa: E402

TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"
TRACE_GPU_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS"


def run_worker(device: int, iters: int, warmup: int, scratch: Path, trace: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f"probe_gemv_dev{device}.json"
    counters = scratch / f"probe_gemv_counters_dev{device}.json"
    for p in (out, counters, trace):
        p.unlink(missing_ok=True)
    env = dict(os.environ)
    env[TRACE_ENV] = str(trace)
    env[TRACE_GPU_ENV] = "1"
    env[COUNTERS_ENV] = str(counters)
    cmd = [sys.executable, str(BENCH / "phi35.py"), "--worker", "--device", str(device),
           "--iters", str(iters), "--warmup", str(warmup),
           "--out", str(out), "--scratch", str(scratch)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    rec = json.loads(out.read_text("utf-8")) if out.exists() else {}
    rec["_worker_exit"] = proc.returncode
    rec["_stderr_tail"] = (proc.stderr or "").strip()[-1200:]
    return rec


def analyse(trace: Path, verdict: "str | None") -> dict:
    events = phases.load(trace)
    subs = phases.subgraph_spans(events)
    gpus = phases.gpu_spans(events)
    ordinal = phases.attribute_gpu_ordinally(subs, gpus)
    busy = ordinal["busy_us"]
    accounting = phases.gpu_span_accounting(subs, gpus)

    # Per inference = per cycle of islands, not per island span.
    order = [s["nodes"] for s in subs]
    period = phases._cycle_period(order) or 1
    per_inf: "list[float]" = []
    for c in range(len(subs) // period):
        tot = 0.0
        for s in subs[c * period:(c + 1) * period]:
            tot += busy.get(s["index"], 0.0)
        per_inf.append(tot)

    tail = phases.gpu_steady_tail(per_inf)

    # Per-kernel totals over the steady tail only, so a cold pipeline-compile pass cannot
    # contribute. `discarded_inferences` is the tail's own start index.
    drop = tail.get("discarded_inferences", 0) if tail["verdict"] == "STEADY" else 0
    keep = set()
    for c in range(drop, len(subs) // period):
        for s in subs[c * period:(c + 1) * period]:
            keep.add(s["index"])
    n_kept = max(1, (len(subs) // period) - drop)

    ns = collections.Counter()
    cnt = collections.Counter()
    # Ordinal attribution: gpu spans are in submission order, same order the subgraphs are.
    # Re-derive membership the same way `attribute_gpu_ordinally` does, by node_index ranges.
    for g in gpus:
        ns[g["kernel"]] += g["gpu_ns"] or 0
        cnt[g["kernel"]] += 1
    total_ns = sum(ns.values()) or 1

    return {
        "trace": str(trace),
        "verdict_of_run": verdict,
        "gpu_span_accounting": accounting,
        "ordinal_left_over": ordinal["left_over"],
        "inferences": len(per_inf),
        "islands_per_inference": period,
        "gpu_busy_series_ms": [round(v / 1000.0, 3) for v in per_inf],
        "gpu_steady_tail": tail,
        "kernels_all_inferences": [
            {"kernel": k,
             "total_ms": round(v / 1e6, 3),
             "share": round(v / total_ns, 4),
             "dispatches": cnt[k],
             "dispatches_per_inference": round(cnt[k] / max(1, len(per_inf)), 1),
             "mean_us": round(v / 1e3 / cnt[k], 2)}
            for k, v in ns.most_common()
        ],
        "kept_inferences_for_kernels": n_kept,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--trace", help="re-analyse an existing trace instead of running")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    verdict = None
    if a.trace:
        trace = Path(a.trace)
    else:
        trace = here / f"trace_gemv_{a.tag}_dev{a.device}.json"
        rec = run_worker(a.device, a.iters, a.warmup, here / "_scratch", trace)
        verdict = rec.get("model_output_equivalence")
        print(f"worker exit {rec.get('_worker_exit')}  verdict={verdict}")
        if rec.get("refusals"):
            print("REFUSALS:", rec["refusals"])
        if not trace.exists():
            print("ERROR(instrument): no trace written.")
            print(rec.get("_stderr_tail"))
            return 2

    rep = analyse(trace, verdict)
    outp = here / f"gemv_{a.tag}_dev{a.device}.json"
    outp.write_text(json.dumps(rep, indent=2), "utf-8")

    t = rep["gpu_steady_tail"]
    print(f"\n== device {a.device} · {a.tag} ==")
    print(f"verdict(model_output_equivalence) = {verdict}")
    print(f"series ms: {rep['gpu_busy_series_ms']}")
    print(f"gpu_steady_tail: {t['verdict']}")
    if t["verdict"] == "STEADY":
        print(f"  GPU busy {t['median_ms']:.3f} ms/inference  (mean {t['mean_ms']:.3f}, "
              f"n={t['n']}, RSD {t['rsd']:.4%}, discarded {t['discarded_inferences']})")
    else:
        print(f"  {t['detail']}")
    print("\n  kernel                                total_ms   share  disp/inf   mean_us")
    for k in rep["kernels_all_inferences"][:8]:
        print(f"  {k['kernel']:<36} {k['total_ms']:>9.2f} {k['share']:>7.2%} "
              f"{k['dispatches_per_inference']:>9} {k['mean_us']:>9.2f}")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
