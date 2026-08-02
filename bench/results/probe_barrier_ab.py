#!/usr/bin/env python3
"""Interleaved A/B of Switch's barrier fix, on one harness, one machine, one artifact.

WHY INTERLEAVED AND NOT TWO RUNS
================================
The prediction under test (Switch's, and it has a falsifier) is that replacing 147,618
per-buffer ``VkBufferMemoryBarrier`` structs with one global ``VkMemoryBarrier`` moves **host**
time sharply and **device** time barely. Testing that needs a before and an after, and the naive
form — run the old DLL, run the new one, subtract — has a confound this project has already been
burned by: **host time is exactly what machine load moves**, this box is not quiet, and the load
is not constant between two runs minutes apart. A single before/after pair cannot separate "the
fix moved host time" from "the machine was busier during the before run".

So the two DLLs are alternated, A B A B, within one sitting, and each run carries its own
out-of-band load survey. If the host delta survives while the load ordering scrambles, load is
not what produced it. If it tracks the load instead, that is the answer and it is the one worth
having.

The device-clock figure (``phases.gpu_steady_tail``) is carried through the same way. It is
believed contention-robust — Switch's own hog experiment landed within 0.08% of solo, or refused
outright — and this run is a second, independent occasion to see whether that holds.

Nothing here is quotable as an end-to-end number: the wall clock stays withheld under the
standing gate. What is produced is a *paired difference* in two decomposed quantities.

    python bench/results/probe_barrier_ab.py --repeats 2 --trace-iters 40

Artifacts land beside this file in ``bench/results/``; nothing is written to the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent
POST_DLL = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
PRE_DLL = (REPO.parent / "ep-vulkan-niobe-ab" / "rust" / "target" / "release"
           / "onnxruntime_vulkan_ep.dll")


def one(tag: str, dll: Path, device: int, trace_iters: int, iters: int, seq: int) -> dict:
    out = HERE / f"barrier_ab-{tag}-dev{device}-{seq}.json"
    env = dict(os.environ)
    env["ONNXRUNTIME_VULKAN_EP_LIB"] = str(dll)
    cmd = [sys.executable, str(BENCH / "phi35.py"), "--device", str(device),
           "--iters", str(iters), "--warmup", "2", "--trace-iters", str(trace_iters),
           "--out", str(out)]
    # Bytes, never text=True: the worker's stderr carries ORT's UTF-16LE sink lines.
    subprocess.run(cmd, env=env, capture_output=True)
    if not out.is_file():
        return {"tag": tag, "seq": seq, "state": "ERROR(instrument=no_artifact)"}
    rec = json.loads(out.read_text("utf-8"))["results"][0]
    an = ((rec.get("phase_pass") or {}).get("analysis") or {})
    hp = (an.get("host_phases_ms") or {}).get("record") or {}
    tail = ((an.get("steady_state") or {}).get("gpu_steady_tail") or {})
    survey = (rec.get("machine_quiescence") or {}).get("survey") or {}
    return {
        "tag": tag, "seq": seq, "artifact": str(out),
        "model_output_equivalence": rec.get("model_output_equivalence"),
        "foreign_busy_cores_mean": survey.get("mean_foreign_busy_cores"),
        "quiescence": (rec.get("machine_quiescence") or {}).get("verdict"),
        "record_host_median_ms": hp.get("median_ms"),
        "record_host_min_ms": hp.get("min_ms"),
        "record_host_leaf_ms_total": hp.get("leaf_ms"),
        "record_n": hp.get("n"),
        "gpu_tail_verdict": tail.get("verdict"),
        "gpu_tail_median_ms": tail.get("median_ms"),
        "gpu_tail_n": tail.get("n"),
        "gpu_tail_discarded": tail.get("discarded_inferences"),
        "gpu_tail_rsd": tail.get("rsd"),
        "warm_gpu_busy_per_inference_ms": (
            round((an.get("steady_state") or {}).get("warm_gpu_busy_ms", 0)
                  / ((an.get("steady_state") or {}).get("warm_inferences") or 1), 4)
            if (an.get("steady_state") or {}).get("warm_gpu_busy_ms") else None),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--trace-iters", type=int, default=40)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "barrier_ab.json"))
    a = ap.parse_args(argv)

    if not POST_DLL.is_file() or not PRE_DLL.is_file():
        print(f"ERROR(instrument=missing_dll): post={POST_DLL.is_file()} pre={PRE_DLL.is_file()}")
        return 4

    rows = []
    for seq in range(a.repeats):
        for tag, dll in (("post", POST_DLL), ("pre", PRE_DLL)):
            r = one(tag, dll, a.device, a.trace_iters, a.iters, seq)
            rows.append(r)
            print(json.dumps(r))

    doc = {
        "kind": "barrier-fix A/B, interleaved",
        "pre_dll": str(PRE_DLL), "post_dll": str(POST_DLL),
        "note": ("host record time and device-clock GPU busy, same harness and machine, DLL the "
                 "only difference. Wall clock is not reported: it is withheld by the standing "
                 "contention gate and this probe does not have a second opinion about that."),
        "runs": rows,
    }
    Path(a.out).write_text(json.dumps(doc, indent=2), "utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
