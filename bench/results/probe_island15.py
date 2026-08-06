#!/usr/bin/env python3
"""Does Phi-3.5 actually execute on the Vulkan EP? A count of node executions, by provider.

The ledger is populated, the gate approves, 323 of 363 nodes are claimed and 33 islands are
retained -- and before this fix every one of them executed **zero** times, because the patch
loop in `vk/session.rs` left island outputs that are also consumed internally as `None` and the
translate handler correctly refused a descriptor it could not interpret.

This probe reads ORT's own profiling trace, which records what *ran*, not what was claimed.
Claimed-but-never-executed is precisely the failure it exists to tell apart from working: both
produce correct logits, because CPU fallback is always correct.

Run:  python bench/results/probe_island15.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EP_NAME = "VulkanExecutionProvider"
LIB = ROOT / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
# ARCHIVAL: pinned to the exact Foundry cache layout this investigation measured against
# (the pre-2026-08-05 "...-cuda-gpu/cuda-int4-rtn-block-32/..." catalog revision, issue
# #11). Intentionally NOT auto-resolved against the live cache: a live resolver could
# silently pick a *different* cached revision than the one this result was measured
# against, which would misattribute a new run to an old artifact. Override PHI35_MODEL to
# replay against a different artifact explicitly (issue #19).
MODEL = (
    os.environ.get(
        "PHI35_MODEL",
        r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
        r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
        r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    )
)


def main() -> int:
    os.environ.setdefault("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    if not LIB.is_file():
        print(f"ERROR(instrument=lib_absent): {LIB}", file=sys.stderr)
        return 4
    digest = hashlib.sha256(LIB.read_bytes()).hexdigest()
    print(f"EP library : {LIB}")
    print(f"sha256     : {digest}")

    import numpy as np
    import onnxruntime as ort

    try:
        ort.register_execution_provider_library(EP_NAME, str(LIB))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            print(f"ERROR(instrument=register_failed): {exc}", file=sys.stderr)
            return 4

    opts = ort.SessionOptions()
    opts.enable_profiling = True
    sess = ort.InferenceSession(MODEL, opts, providers=[EP_NAME, "CPUExecutionProvider"])
    print(f"providers  : {sess.get_providers()}")

    seq = 4
    feeds = {
        "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int64),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
    }
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values"):
            feeds[i.name] = np.zeros((1, 32, 0, 96), dtype=np.float16)

    outs = sess.run(None, feeds)
    profile = sess.end_profiling()

    events = json.loads(pathlib.Path(profile).read_text())
    by_provider: collections.Counter = collections.Counter()
    by_op: collections.Counter = collections.Counter()
    for e in events:
        if e.get("cat") != "Node" or not isinstance(e.get("args"), dict):
            continue
        prov = e["args"].get("provider", "?")
        by_provider[prov] += 1
        if prov == EP_NAME:
            by_op[e["args"].get("op_name", "?")] += 1
    pathlib.Path(profile).unlink(missing_ok=True)

    vk = by_provider.get(EP_NAME, 0)
    total = sum(by_provider.values())
    print()
    print("node executions by provider (from ORT's own trace — what RAN, not what was claimed)")
    for k, v in by_provider.most_common():
        print(f"    {k:34s} {v:5d}")
    print(f"    {'TOTAL':34s} {total:5d}")
    if vk:
        print()
        print("  Vulkan-executed nodes by op:")
        for k, v in by_op.most_common():
            print(f"    {k:34s} {v:5d}")
    print()
    print(f"  logits {outs[0].shape}  finite={bool(np.isfinite(outs[0]).all())}  "
          f"outputs={len(outs)}")
    print()
    print(f"  VERDICT: {'EXECUTES' if vk else 'ZERO VULKAN EXECUTIONS'} "
          f"({vk}/{total} node executions on the EP)")

    out = ROOT / "bench" / "results" / "island15_execution.json"
    out.write_text(json.dumps({
        "probe": "phi35_node_executions_by_provider",
        "ep_library_sha256": digest,
        "by_provider": dict(by_provider),
        "vulkan_by_op": dict(by_op),
        "vulkan_executions": vk,
        "total_executions": total,
        "n_outputs": len(outs),
        "logits_finite": bool(np.isfinite(outs[0]).all()),
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
