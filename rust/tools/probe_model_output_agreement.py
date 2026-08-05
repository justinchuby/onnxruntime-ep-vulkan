#!/usr/bin/env python
"""Does a model this EP claims nodes on actually produce the right answers? Owner: Mouse.

WHY THIS EXISTS
---------------
The census counts claims. It cannot tell a claim that computes the right answer from one that
computes a wrong answer of the right shape, and the charter's first line is that the second is
worse than declining. On BERT-SQuAD-12 the census reads 480 claimed nodes; **476 of those 480
carry at least one edge whose shape ORT reported as rank 0** (an empty dimension list). ORT
returns rank 0 for a genuine scalar *and* for a tensor whose rank shape inference never
established, and the C API's `GetDimensionsCount` cannot tell the two apart. So the claim
predicates on that model may have been reading "unknown rank" as "rank-0 scalar" and claiming
on it.

This probe runs the model twice -- once with the Vulkan EP in the provider list and once on the
CPU EP alone -- with identical feeds, and compares every output. It is a correctness screen for
a *whole model*, not a kernel.

TWO GUARDS, BOTH REQUIRED
-------------------------
1. The Vulkan session must actually have the Vulkan EP in `get_providers()`. A session that
   silently fell back to CPU compares CPU against CPU and passes vacuously -- the failure mode
   `test_phi35_cpu_output_matches_between_sessions` had.
2. `dispatches_executed > 0` off the counters, so a run that claimed nodes and dispatched
   nothing is reported as an instrument failure rather than as agreement.

NO CLOCK. Correctness only.

USAGE
    python rust/tools/probe_model_output_agreement.py --model <path.onnx>
                                                      [--out bench/results/agreement_<tag>.json]
Requires `ONNXRUNTIME_VULKAN_EP_LIB`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np


def _feed(sess, seed: int = 0) -> dict:
    """Feeds shaped from the session's own declared inputs, with symbolic extents pinned to 1."""
    rng = np.random.default_rng(seed)
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        t = i.type
        if "int64" in t:
            # Token ids and masks. Kept small and non-negative: an id outside the embedding
            # table is an out-of-range gather, which is a different bug than the one under test.
            feeds[i.name] = rng.integers(0, 2, size=shape, dtype=np.int64)
        elif "int32" in t:
            feeds[i.name] = rng.integers(0, 2, size=shape, dtype=np.int32)
        elif "float16" in t:
            feeds[i.name] = rng.standard_normal(shape).astype(np.float16)
        else:
            feeds[i.name] = rng.standard_normal(shape).astype(np.float32)
    return feeds


def main() -> int:
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB is not set")
        return 2
    ort.register_execution_provider_library("VulkanExecutionProvider", lib)

    so = ort.SessionOptions()
    so.log_severity_level = 3

    cpu = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    feeds = _feed(cpu, args.seed)
    want = cpu.run(None, feeds)

    vk = ort.InferenceSession(
        args.model, so, providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    used = vk.get_providers()
    if "VulkanExecutionProvider" not in used:
        print(f"ERROR(instrument): the Vulkan EP is not in the session providers: {used}")
        return 2
    got = vk.run(None, feeds)

    report = {"model": args.model, "providers": used, "outputs": []}
    worst = 0.0
    for o, w, g in zip(cpu.get_outputs(), want, got, strict=True):
        w = np.asarray(w)
        g = np.asarray(g)
        rec = {"name": o.name, "shape": list(w.shape), "dtype": str(w.dtype)}
        if w.shape != g.shape:
            rec["verdict"] = f"SHAPE MISMATCH cpu={w.shape} vk={g.shape}"
        elif not np.issubdtype(w.dtype, np.floating):
            same = bool(np.array_equal(w, g))
            rec["verdict"] = "EXACT" if same else "DIFFERENT"
        else:
            wf, gf = w.astype(np.float64), g.astype(np.float64)
            denom = np.maximum(np.abs(wf), 1e-6)
            rel = float(np.max(np.abs(wf - gf) / denom))
            rec["max_abs"] = float(np.max(np.abs(wf - gf)))
            rec["max_rel"] = rel
            rec["all_zero_vk"] = bool(np.all(gf == 0.0))
            worst = max(worst, rel)
            rec["verdict"] = "AGREE" if rel < 1e-2 else "DISAGREE"
        report["outputs"].append(rec)
        print(f"  {rec['verdict']:<16} {o.name}  {rec}")

    report["worst_max_rel"] = worst
    bad = [r for r in report["outputs"] if r["verdict"] not in ("AGREE", "EXACT")]
    report["verdict"] = "AGREE" if not bad else "DISAGREE"
    print(f"\n{report['verdict']}: {len(bad)} of {len(report['outputs'])} outputs disagree")

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {p}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
