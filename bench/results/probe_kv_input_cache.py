#!/usr/bin/env python3
"""Does the >=32 KiB input cache serve stale KV when the past changes at the same address?

WHY THIS PROBE EXISTS
=====================
Niobe measured that per-inference staging upload is FLAT across past_len 0 / 128 / 512 --
399,376 B every time -- while readback grows by exactly 393,216 B per past token. She refused
to call the upload `0`, on the grounds that the past KV must reach the device somehow and a
`0` would claim the read side is free. She was right to refuse, and the resolution is that the
upload is in the INTERCEPT, not the slope:

    5-iteration base at past 128 - base at past   0 = 50,331,648 = 128 x 393,216
    5-iteration base at past 512 - base at past 128 = 150,994,944 = 384 x 393,216
    marginal per inference at all three contexts   = 399,376 (identical to the byte)

The whole past cache is uploaded ONCE PER SESSION. Five iterations at past 128 would have
added 251,658,240 B if it were uploaded per inference; they added nothing.

The mechanism is `vk/session.rs`'s weight cache: any input >= 32 KiB is cached on
`(cpu_ptr, byte_size)` and skipped on later inferences. Its comment states the assumption
outright -- "if ORT gives us the same CPU pointer as a prior inference, the tensor is a model
constant (initialiser/weight)". Phi-3.5's `past_key_values.N.key` is `past_len x 6144` bytes,
so it crosses 32 KiB at **past_len >= 6** and is cached as though it were a weight.

THE TWO DEFINITIONS THAT COINCIDE
=================================
"same address and size" and "same tensor contents" are the same thing for a weight and
different things for a KV cache. Every case we run agrees with the cache:

* criterion 10 runs at past_len 0 -- the KV inputs are empty, below the threshold, never cached
* Niobe's harness feeds the SAME past tensor every iteration -- the cache is correct there
* every evidence case has a past of 512 B -- below the threshold

So this probe does the one thing none of those do: **it changes the contents at the same
address and asks whether the answer follows the data.**

WHAT IT REFUSES TO DO
=====================
It refuses to report unless (a) the two reference answers actually differ, or a stale read
would be indistinguishable from a correct one, and (b) the Vulkan EP actually executed the
node, or the comparison is between the CPU and itself. Both are guards this project has been
burned by the absence of.

It also reports whether the two runs saw the SAME cpu pointer. If ORT hands us a different
address the cache never fires, the probe proves nothing, and it says so rather than reporting
a pass -- an unreached branch cannot support a null result.

Run:  python bench/results/probe_kv_input_cache.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIB = pathlib.Path(
    os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(ROOT / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )
)

B, S, NQ, NKV, D = 1, 1, 2, 2, 32
PAST = 512  # past_key bytes = 1*2*512*32*2 = 65,536 >= the 32 KiB cache threshold
TOTAL = PAST + S
MAXSEQ = 4096
ROT = D  # full rotary

#: past_key bytes, computed so the record states why this geometry was chosen.
PAST_BYTES = B * NKV * PAST * D * 2
CACHE_MIN_BYTES = 32 * 1024


def build_model(path: pathlib.Path) -> None:
    node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "packed_qkv",
            "",
            "",
            "past_key",
            "past_value",
            "seqlens_k",
            "total_seq",
            "cos_cache",
            "sin_cache",
        ],
        outputs=["attn_out", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=NQ,
        kv_num_heads=NKV,
        scale=1.0 / np.sqrt(D),
        do_rotary=1,
        rotary_interleaved=0,
    )
    g = helper.make_graph(
        [node],
        "gqa_cache_probe",
        [
            helper.make_tensor_value_info(
                "packed_qkv", TensorProto.FLOAT16, [B, S, (NQ + 2 * NKV) * D]
            ),
            helper.make_tensor_value_info(
                "past_key", TensorProto.FLOAT16, [B, NKV, PAST, D]
            ),
            helper.make_tensor_value_info(
                "past_value", TensorProto.FLOAT16, [B, NKV, PAST, D]
            ),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [B]),
            helper.make_tensor_value_info("total_seq", TensorProto.INT32, []),
            helper.make_tensor_value_info("cos_cache", TensorProto.FLOAT16, [MAXSEQ, ROT // 2]),
            helper.make_tensor_value_info("sin_cache", TensorProto.FLOAT16, [MAXSEQ, ROT // 2]),
        ],
        [
            helper.make_tensor_value_info("attn_out", TensorProto.FLOAT16, [B, S, NQ * D]),
            helper.make_tensor_value_info(
                "present_key", TensorProto.FLOAT16, [B, NKV, TOTAL, D]
            ),
            helper.make_tensor_value_info(
                "present_value", TensorProto.FLOAT16, [B, NKV, TOTAL, D]
            ),
        ],
    )
    m = helper.make_model(
        g,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    m.ir_version = 10
    onnx.save(m, str(path))


def make_feeds(rng: np.random.Generator, past_k: np.ndarray, past_v: np.ndarray) -> dict:
    """Feeds sharing the CALLER's past arrays, so the two runs can reuse one buffer."""
    pos = np.arange(MAXSEQ, dtype=np.float32)[:, None] / 10000.0 ** (
        np.arange(ROT // 2, dtype=np.float32)[None, :] * 2.0 / ROT
    )
    return {
        "packed_qkv": rng.standard_normal((B, S, (NQ + 2 * NKV) * D)).astype(np.float16) * 0.1,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": np.array([TOTAL - 1], dtype=np.int32),
        "total_seq": np.array(TOTAL, dtype=np.int32),
        "cos_cache": np.cos(pos).astype(np.float16),
        "sin_cache": np.sin(pos).astype(np.float16),
    }


def run(sess, feeds) -> list:
    return sess.run(None, feeds)


def main() -> int:
    tmp = ROOT / "bench" / "results" / "_gqa_cache_probe.onnx"
    build_model(tmp)
    rng = np.random.default_rng(20260802)

    # ONE pair of buffers, mutated in place between runs. Reusing the same numpy arrays is
    # what gives ORT the chance to hand the EP the same address twice -- which is the only
    # configuration in which the cache can fire at all.
    past_k = (rng.standard_normal((B, NKV, PAST, D)) * 0.1).astype(np.float16)
    past_v = (rng.standard_normal((B, NKV, PAST, D)) * 0.1).astype(np.float16)
    feeds = make_feeds(rng, past_k, past_v)

    cpu = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    ref_a = run(cpu, feeds)

    ort.register_execution_provider_library("VulkanExecutionProvider", str(LIB))
    so = ort.SessionOptions()
    so.enable_profiling = True
    vk = ort.InferenceSession(
        str(tmp), so, providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    got_a = run(vk, feeds)

    # ---- mutate the past IN PLACE: same address, same size, different bytes -------------
    past_k[...] = (rng.standard_normal((B, NKV, PAST, D)) * 0.1).astype(np.float16)
    past_v[...] = (rng.standard_normal((B, NKV, PAST, D)) * 0.1).astype(np.float16)

    ref_b = run(cpu, feeds)
    got_b = run(vk, feeds)
    prof = vk.end_profiling()

    # ---- guards ------------------------------------------------------------------------
    n_vk = 0
    try:
        with open(prof, "r", encoding="utf-8") as f:
            for e in json.load(f):
                if e.get("cat") == "Node" and e.get("args", {}).get(
                    "provider"
                ) == "VulkanExecutionProvider":
                    n_vk += 1
    finally:
        pathlib.Path(prof).unlink(missing_ok=True)

    ref_moved = float(np.max(np.abs(ref_a[0].astype(np.float32) - ref_b[0].astype(np.float32))))
    if ref_moved == 0.0:
        print("  REFUSE: the reference answer did not move between runs -- a stale read")
        print("          would be indistinguishable from a correct one. Probe proves nothing.")
        tmp.unlink(missing_ok=True)
        return 2
    if n_vk == 0:
        print("  REFUSE: no node executed on VulkanExecutionProvider -- the comparison is")
        print("          the CPU against itself and would report a perfect match.")
        tmp.unlink(missing_ok=True)
        return 2

    def worst(x, y):
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        assert x.shape == y.shape, f"shape mismatch {x.shape} vs {y.shape}"
        d = np.abs(x - y)
        return float(d.max()), float((d / np.maximum(np.abs(y), 1e-6)).max())

    names = ["attn_out", "present_key", "present_value"]
    # present_* have different shapes only if the model changed; compare index 0 primarily.
    abs_b, rel_b = worst(got_b[0], ref_b[0])
    abs_stale, _ = worst(got_b[0], got_a[0])

    print()
    print(f"  geometry: past={PAST} past_key bytes={PAST_BYTES} threshold={CACHE_MIN_BYTES}")
    print(f"  cacheable by size: {PAST_BYTES >= CACHE_MIN_BYTES}")
    print(f"  vulkan node executions: {n_vk}")
    print(f"  reference moved between runs by {ref_moved:.6g}  (guard: must be > 0)")
    print()
    print(f"  run B vs CORRECT answer (cpu, new past)   max_abs={abs_b:.6g} rel={rel_b:.6g}")
    print(f"  run B vs STALE answer   (vk,  old past)   max_abs={abs_stale:.6g}")
    print()
    if abs_stale == 0.0 and abs_b > 0.0:
        verdict = "STALE_CACHE"
        print("  VERDICT: STALE_CACHE — run B returned run A's answer exactly. The >=32 KiB")
        print("           input cache served the OLD past KV after the contents changed at")
        print("           the same address. In generation this silently freezes the cache.")
    elif abs_b <= 2e-3:
        verdict = "FOLLOWS_DATA"
        print("  VERDICT: FOLLOWS_DATA — the answer tracked the new past. The cache either")
        print("           did not fire or is content-safe on this path.")
    else:
        verdict = "DIVERGENT"
        print("  VERDICT: DIVERGENT — run B matches neither the correct nor the stale answer.")

    rec = ROOT / "bench" / "results" / "kv_input_cache.json"
    rec.write_text(
        json.dumps(
            {
                "kind": "kv_input_cache",
                "question": "does the >=32 KiB input cache serve stale KV at a reused address?",
                "past_len": PAST,
                "past_key_bytes": PAST_BYTES,
                "cache_min_bytes": CACHE_MIN_BYTES,
                "cacheable_by_size": PAST_BYTES >= CACHE_MIN_BYTES,
                "vulkan_node_executions": n_vk,
                "reference_moved_between_runs": ref_moved,
                "run_b_vs_correct_max_abs": abs_b,
                "run_b_vs_stale_max_abs": abs_stale,
                "outputs": names,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  record: {rec}")
    tmp.unlink(missing_ok=True)
    return 0 if verdict == "FOLLOWS_DATA" else 1


if __name__ == "__main__":
    sys.exit(main())
