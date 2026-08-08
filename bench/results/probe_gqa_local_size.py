"""Sweep `gqa_f16`'s workgroup size on the real Phi-3.5 graph, and prove the packing is inert.

WHY THIS EXISTS SEPARATELY FROM `probe_real_model_latency.py`
------------------------------------------------------------
`real_model_diagnostics.json` says *which* kernel dominates; it does not say what to do about it.
This probe answers the follow-on question — **what workgroup size should `gqa_f16` be dispatched
at?** — and it exists as its own committed artifact because the host rule
`ops::attention::gqa_local_size` embeds two constants (`GQA_MAX_LOCAL_SIZE`, `GQA_MIN_GROUPS`)
whose only honest justification is a table. A constant defended by an argument about subgroup
widths is a guess with a citation; this is the measurement it has to agree with.

WHAT IT MEASURES, AND WHAT IT REFUSES TO CONFLATE
-------------------------------------------------
Two questions, deliberately not merged into one number:

1. **Is it faster?** For each case and each candidate size, the *kernel's own* GPU nanoseconds,
   read from the EP's GPU tracer (`ONNXRUNTIME_EP_VULKAN_TRACE` + `..._TRACE_GPU`) and attributed
   by name, plus the whole-graph GPU total from the same trace. Wall clock is deliberately NOT the
   headline here: this box is shared (PERF.md §20), the decode arm carries a large host-side KV
   round trip, and the claim under test is about one kernel. The wall number is recorded anyway,
   because a kernel win that does not move the graph is worth knowing about.

2. **Is it the same answer?** The packing changes which invocations share a workgroup; it changes
   nothing about what any invocation computes, reads or writes. That is a claim about the source
   text, so it is falsifiable byte-for-byte: every size's outputs are compared to size 1's with
   `real_model.bitwise_identical`, and anything short of BITWISE-IDENTICAL is a failure of the
   change, not a tolerance to be widened. A speedup that is only measured, never verified, is the
   defect this repository's ledger exists to prevent.

The equivalence pass is run FIRST and its verdict is printed before any timing table, so a
correctness failure cannot be buried under a performance win.

ONE SUBPROCESS PER POINT
------------------------
ORT registers an EP process-globally and the EP writes its counters from an exit hook, so a
loop inside one process would measure the second size against the first size's pipeline cache and
report the union of both. The parent spawns `probe_real_model_latency.py --worker-diagnose` for
timing and this file's own `--worker-outputs` for equivalence.

Usage:
    python bench/results/probe_gqa_local_size.py                # full sweep -> JSON
    python bench/results/probe_gqa_local_size.py --sizes 1,32   # narrow
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1]
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import public_paths
import real_model as rm  # noqa: E402

SCHEMA = "real_model_gqa_local_size/1"
LOCAL_SIZE_ENV = "ONNXRUNTIME_EP_VULKAN_GQA_LOCAL_SIZE"
TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"
TRACE_GPU_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
GQA_EVENT = "vulkan.gpu.gqa_f16"

DEFAULT_SIZES = (1, 2, 4, 8, 16, 32, 64)
# The cases are the diagnostic set, not the whole matrix: these are the six points where
# `real_model_diagnostics.json` already established what fraction of GPU time GQA holds, so the
# sweep is measured exactly where the diagnosis was made.
DEFAULT_CASES = (
    ("prefill", 1, 0),
    ("prefill", 8, 0),
    ("prefill", 32, 0),
    ("prefill", 128, 0),
    ("decode", 1, 512),
    ("decode", 1, 1024),
)


def case_label(phase: str, m: int, past: int) -> str:
    return f"{phase}/M{m}/past{past}"


def gqa_invocations(phase: str, m: int) -> int:
    """`B * Nq * S` for Phi-3.5 — the number of GQA invocations the dispatch must cover.

    Batch is 1 and Phi-3.5-mini has 32 query heads, so this is `32 * S`, and `S` is the number of
    new tokens (M at prefill, 1 at decode). It is spelled out because it is the *independent
    variable* of the whole sweep: decode has 32 invocations and prefill at M=128 has 4096, and a
    rule that is right for one is not automatically right for the other.
    """
    seq = m if phase == "prefill" else 1
    return 1 * 32 * seq


# --------------------------------------------------------------------------------------------
# Equivalence: the packing must be inert
# --------------------------------------------------------------------------------------------


def outputs_worker(argv) -> int:
    """Run one inference at one workgroup size and dump every output, for a bitwise compare."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-outputs", action="store_true")
    ap.add_argument("--phase", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--past", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort

    model_rec = rm.resolve_model(rm.PHI35)
    arm = rm.VULKAN_TILED
    arm.apply_env(os.environ)
    lib = os.environ.get(EP_LIB_ENV)
    if not lib or not Path(lib).is_file():
        print(f"{EP_LIB_ENV} unset or missing", file=sys.stderr)
        return 2
    try:
        ort.register_execution_provider_library(rm.EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:
        if "already registered" not in str(exc):
            print(f"registration failed: {exc}", file=sys.stderr)
            return 2

    case = rm.Case(rm.PHI35.key, a.phase, a.m, a.past,
                   tokens=(a.m if a.phase == "prefill" else 1), unit="tokens")
    feeds = rm.build_feeds(case, np)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("ep.device_index", str(a.device))
    sess = ort.InferenceSession(str(model_rec["path"]), opts, providers=list(arm.providers))
    providers = list(sess.get_providers())
    outs = sess.run(None, feeds)
    np.savez(a.out, **{f"o{i}": o for i, o in enumerate(outs)})
    public_paths.dump_public_json(
        {"providers": providers, "count": len(outs)},
        Path(a.out).with_suffix(".meta.json"))
    return 0


def _run_outputs(py: str, scratch: Path, phase: str, m: int, past: int, local: int,
                 device: int) -> "Path | None":
    out = scratch / f"gqa_ls_out_{phase}_{m}_{past}_{local}.npz"
    out.unlink(missing_ok=True)
    Path(str(out) + ".meta.json").unlink(missing_ok=True)
    env = dict(os.environ)
    env[LOCAL_SIZE_ENV] = str(local)
    proc = subprocess.run(
        [py, str(Path(__file__).resolve()), "--worker-outputs", "--phase", phase,
         "--m", str(m), "--past", str(past), "--device", str(device), "--out", str(out)],
        cwd=str(_ROOT), env=env, capture_output=True)
    if not out.is_file():
        sys.stderr.write(f"[gqa-ls] outputs worker exit {proc.returncode} for local={local}\n")
        return None
    return out


def equivalence_pass(py: str, scratch: Path, cases, sizes, device: int) -> list:
    """Every size's outputs against size 1's, byte for byte. No tolerance, by design."""
    import numpy as np

    records = []
    for (phase, m, past) in cases:
        base_path = _run_outputs(py, scratch, phase, m, past, 1, device)
        rec: dict = {"case": case_label(phase, m, past),
                     "invocations": gqa_invocations(phase, m),
                     "reference_local_size": 1, "comparisons": []}
        if base_path is None:
            rec["error"] = "reference (local=1) run failed"
            records.append(rec)
            continue
        with np.load(base_path) as z:
            ref = [z[k] for k in sorted(z.files)]
        for local in sizes:
            if local == 1:
                continue
            p = _run_outputs(py, scratch, phase, m, past, local, device)
            if p is None:
                rec["comparisons"].append({"local_size": local, "verdict": "UNMEASURED",
                                           "detail": "worker failed"})
                continue
            with np.load(p) as z:
                cand = [z[k] for k in sorted(z.files)]
            verdict = rm.bitwise_identical(cand, ref, np)
            verdict["verdict"] = ("BITWISE-IDENTICAL" if verdict.get("identical")
                                  else "DIFFERENT")
            rec["comparisons"].append({"local_size": local, **verdict})
            print(f"[gqa-ls] equivalence {rec['case']} local={local}: {verdict['verdict']}",
                  flush=True)
        rec["all_identical"] = bool(rec["comparisons"]) and all(
            c.get("verdict") == "BITWISE-IDENTICAL" for c in rec["comparisons"])
        records.append(rec)
    return records


# --------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------


def _gpu_totals(trace_path: Path, iters: int) -> dict:
    """Per-kernel GPU microseconds per inference, from the EP's own Chrome trace.

    ORT's profiler attributes every Vulkan node to the single fused node it handed the EP, so it
    cannot answer "which kernel". The EP's tracer can, and `cat == "gpu"` events carry the
    timestamp-query durations rather than host wall time.
    """
    d = json.loads(trace_path.read_text(encoding="utf-8"))
    events = d["traceEvents"] if isinstance(d, dict) else d
    agg: collections.Counter = collections.Counter()
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "gpu":
            agg[e.get("name", "?")] += e.get("dur", 0)
    total = sum(agg.values())
    return {
        "gqa_us": agg.get(GQA_EVENT, 0) / iters,
        "total_us": total / iters,
        "by_kernel_us": {k: v / iters for k, v in sorted(agg.items(), key=lambda kv: -kv[1])},
    }


def _time_point(py: str, scratch: Path, phase: str, m: int, past: int, local: int,
                device: int, iters: int) -> dict:
    trace = scratch / f"gqa_ls_trace_{phase}_{m}_{past}_{local}.json"
    out = scratch / f"gqa_ls_diag_{phase}_{m}_{past}_{local}.json"
    for p in (trace, out):
        p.unlink(missing_ok=True)
    env = dict(os.environ)
    env[TRACE_ENV] = str(trace)
    env[TRACE_GPU_ENV] = "1"
    env[LOCAL_SIZE_ENV] = str(local)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [py, str(_BENCH / "results" / "probe_real_model_latency.py"), "--worker-diagnose",
         "--model", rm.PHI35.key, "--arm", "vulkan_tiled", "--phase", phase, "--m", str(m),
         "--past", str(past), "--device", str(device), "--iters", str(iters), "--out", str(out)],
        cwd=str(_ROOT), env=env, capture_output=True)
    wall_s = time.perf_counter() - t0
    rec: dict = {"local_size": local, "process_wall_s": round(wall_s, 3)}
    if not trace.is_file():
        rec["error"] = f"no trace; worker exit {proc.returncode}"
        return rec
    rec.update(_gpu_totals(trace, iters))
    if out.is_file():
        try:
            worker = json.loads(out.read_text(encoding="utf-8"))
            rec["providers"] = worker.get("providers")
        except json.JSONDecodeError:
            pass
    return rec


def timing_pass(py: str, scratch: Path, cases, sizes, device: int, iters: int) -> list:
    records = []
    for (phase, m, past) in cases:
        points = [_time_point(py, scratch, phase, m, past, s, device, iters) for s in sizes]
        ok = [p for p in points if "gqa_us" in p and p["gqa_us"] > 0]
        rec: dict = {
            "case": case_label(phase, m, past),
            "invocations": gqa_invocations(phase, m),
            "points": points,
        }
        if ok:
            base = next((p for p in points if p["local_size"] == 1 and "gqa_us" in p), None)
            best = min(ok, key=lambda p: p["gqa_us"])
            rec["best_local_size"] = best["local_size"]
            rec["best_gqa_us"] = best["gqa_us"]
            if base:
                rec["baseline_gqa_us"] = base["gqa_us"]
                rec["speedup_vs_size_1"] = (base["gqa_us"] / best["gqa_us"]
                                            if best["gqa_us"] else None)
                rec["graph_speedup_vs_size_1"] = (base["total_us"] / best["total_us"]
                                                  if best.get("total_us") else None)
        records.append(rec)
        cells = " ".join(f"{p.get('gqa_us', 0) / 1000.0:8.2f}" for p in points)
        print(f"[gqa-ls] {rec['case']:22s} {cells}  best={rec.get('best_local_size')} "
              f"({rec.get('speedup_vs_size_1', 0) or 0:.2f}x)", flush=True)
    return records


# --------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker-outputs" in argv:
        return outputs_worker(argv)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("real_model_gqa_local_size.json")))
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--scratch", default=str(_BENCH / "results" / "_issue56_scratch"))
    ap.add_argument("--skip-equivalence", action="store_true",
                    help="timing only. Never use for a committed artifact: the artifact's whole "
                         "claim is that the packing is inert, and this switch removes the "
                         "evidence for it.")
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",") if s]
    if 1 not in sizes:
        print("[gqa-ls] refusing: size 1 is the reference arm for both passes", file=sys.stderr)
        return 2
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    lib = os.environ.get(EP_LIB_ENV)
    if not lib or not Path(lib).is_file():
        print(f"[gqa-ls] refusing: {EP_LIB_ENV} unset or missing", file=sys.stderr)
        return 2

    model_rec = rm.resolve_model(rm.PHI35)
    print(f"[gqa-ls] {rm.PHI35.key}: {model_rec['path']}")
    print(f"[gqa-ls]   sha256 {model_rec['sha256']} ({model_rec['provenance']})")

    equivalence = []
    if not args.skip_equivalence:
        equivalence = equivalence_pass(py, scratch, DEFAULT_CASES, sizes, args.device)
    timing = timing_pass(py, scratch, DEFAULT_CASES, sizes, args.device, args.iters)

    report = {
        "schema": SCHEMA,
        "issue": 56,
        "subject": "rust/shaders/glsl/gqa_f16.comp local_size_x (specialisation constant 0)",
        "depends_on": "PR #53 (squad/7-tile-matmulnbits-prefill)",
        "model": model_rec,
        "sizes": sizes,
        "iters_per_point": args.iters,
        "ep_library": lib,
        "ep_library_sha256": rm.sha256_file(lib),
        "device_index": args.device,
        "equivalence": equivalence,
        "equivalence_complete": bool(equivalence) and all(
            r.get("all_identical") for r in equivalence),
        "timing": timing,
        "limitations": [
            "GPU time is the EP's own timestamp-query total per kernel, not a hardware "
            "occupancy counter: it says the kernel finished sooner, not why.",
            "Wall clock on this box is STEADY_UNCERTIFIED (PERF.md §20); the kernel-time "
            "column is the claim and process_wall_s is context, not evidence.",
            "One device (RTX A1000). The rule this table justifies is a function of the "
            "invocation count, which is a property of the model, but the best size is a "
            "property of the machine — hence the environment kill switch.",
        ],
    }
    public_paths.dump_public_json(report, Path(args.out))
    print(f"\n  wrote {args.out}")
    if equivalence and not report["equivalence_complete"]:
        print("[gqa-ls] EQUIVALENCE FAILED — the packing is not inert", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
