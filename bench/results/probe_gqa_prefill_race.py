"""The sibling-key race: prefill is the only regime that reaches the branch.

`gqa_f16.comp` used to read `present_key`/`present_value` at `t` in
`[past_len, tok_pos)` -- positions written by *sibling invocations of the same
dispatch*. local_size is 1 and there is one workgroup per (b, h, s_local), and Vulkan
orders neither execution nor memory between workgroups of one dispatch. So that read
was a data race.

It is unreachable at decode: `seq_len == 1` makes `tok_pos == past_len`, the branch
range is empty, and every case we own is decode. A test suite of decode cases passes
whether the race exists or not -- which is why it survived.

This probe builds the regime that reaches it: `seq_len > 1` with a non-empty past. It
does three things a single comparison cannot:

  1. **Proves the branch is reached**, by counting the (t, s_local) pairs that land in
     it from the same geometry the shader uses. A null result on an unreached branch is
     `0.0 == 0.0` (see `switch-a-null-control-must-be-shown-to-discriminate.md`).
  2. **Proves the node ran on Vulkan**, from ORT's own trace, so the comparison is not
     the CPU against itself.
  3. **Repeats the run**, because a race that happens to win is indistinguishable from
     no race in a single sample. Cross-run identity is necessary, not sufficient --
     but non-identity would be decisive.

Correctness of the *replacement* is the stronger claim and it is checked by the CPU
differential: the recomputed key is the same arithmetic step 2 applies to the same
`packed_qkv` row and the same cos/sin row, so it should be bit-identical, not close.
"""

import json
import os
import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LIB = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
EP = "VulkanExecutionProvider"

import onnxruntime as ort  # noqa: E402

# Geometry. Nq/Nkv = 4 is a *genuine* grouping, unlike Phi-3.5's 32/32 -- the branch
# under test is per-kv-head, so a degenerate grouping would exercise less of it.
B, S, NQ, NKV, D = 1, 3, 8, 2, 32
PAST = 4
TOTAL = PAST + S
MAX_POS = 64
ROT_HALF = D // 2
SEED = 20260802
RUNS = 3


def build_model(path: pathlib.Path) -> None:
    qkv_dim = (NQ + 2 * NKV) * D
    node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "packed_qkv", "", "", "past_key", "past_value",
            "seqlens_k", "total_seq", "cos_cache", "sin_cache",
        ],
        outputs=["attn_out", "present_key", "present_value"],
        domain="com.microsoft",
        name="gqa_prefill",
        num_heads=NQ,
        kv_num_heads=NKV,
        scale=float(1.0 / np.sqrt(D)),
        do_rotary=1,
        rotary_interleaved=0,
        smooth_softmax=0,
        local_window_size=-1,
    )
    f16 = TensorProto.FLOAT16
    graph = helper.make_graph(
        [node],
        "gqa_prefill_race",
        [
            helper.make_tensor_value_info("packed_qkv", f16, [B, S, qkv_dim]),
            helper.make_tensor_value_info("past_key", f16, [B, NKV, PAST, D]),
            helper.make_tensor_value_info("past_value", f16, [B, NKV, PAST, D]),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [B]),
            helper.make_tensor_value_info("total_seq", TensorProto.INT32, []),
            helper.make_tensor_value_info("cos_cache", f16, [MAX_POS, ROT_HALF]),
            helper.make_tensor_value_info("sin_cache", f16, [MAX_POS, ROT_HALF]),
        ],
        [
            helper.make_tensor_value_info("attn_out", f16, [B, S, NQ * D]),
            helper.make_tensor_value_info("present_key", f16, [B, NKV, TOTAL, D]),
            helper.make_tensor_value_info("present_value", f16, [B, NKV, TOTAL, D]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.save(model, str(path))


def build_feeds():
    rng = np.random.default_rng(SEED)
    pos = np.arange(MAX_POS, dtype=np.float32)[:, None]
    inv = (10000.0 ** (-np.arange(ROT_HALF, dtype=np.float32) / ROT_HALF))[None, :]
    ang = pos * inv
    return {
        "packed_qkv": (rng.standard_normal((B, S, (NQ + 2 * NKV) * D)) * 0.1).astype(
            np.float16
        ),
        "past_key": (rng.standard_normal((B, NKV, PAST, D)) * 0.1).astype(np.float16),
        "past_value": (rng.standard_normal((B, NKV, PAST, D)) * 0.1).astype(np.float16),
        # seqlens_k is total-1, matching the harness's convention for the evidence case.
        "seqlens_k": np.array([TOTAL - 1], dtype=np.int32),
        "total_seq": np.array(TOTAL, dtype=np.int32),
        "cos_cache": np.cos(ang).astype(np.float16),
        "sin_cache": np.sin(ang).astype(np.float16),
    }


def branch_reach_count() -> int:
    """(h, s_local, t) triples that land in the sibling-key branch.

    The shader's own geometry: past_len comes from seqlens_k, tok_pos = past_len +
    s_local, and the branch is t in [past_len, tok_pos). Counted here rather than
    asserted, because "the branch is reached" is the precondition that makes a null
    result mean anything.
    """
    past_len = TOTAL - 1 - (S - 1)  # seqlens_k - (S-1): the true past for s_local=0
    n = 0
    for _h in range(NQ):
        for s_local in range(S):
            tok_pos = past_len + s_local
            n += max(0, tok_pos - past_len)
    return n


def run(model_path, feeds, vulkan):
    so = ort.SessionOptions()
    if vulkan:
        so.enable_profiling = True
        so.profile_file_prefix = str(HERE / "gqa_prefill_prof")
        sess = ort.InferenceSession(
            str(model_path), so, providers=[EP, "CPUExecutionProvider"]
        )
    else:
        sess = ort.InferenceSession(
            str(model_path), so, providers=["CPUExecutionProvider"]
        )
    outs = sess.run(None, feeds)
    execs = 0
    if vulkan:
        p = pathlib.Path(sess.end_profiling())
        for e in json.loads(p.read_text(encoding="utf-8")):
            if e.get("cat") == "Node" and e.get("args", {}).get("provider") == EP:
                execs += 1
        p.unlink(missing_ok=True)
    return outs, execs


def main():
    ort.register_execution_provider_library(EP, str(LIB))
    model_path = HERE / "_gqa_prefill_race.onnx"
    build_model(model_path)
    feeds = build_feeds()

    reach = branch_reach_count()
    print(f"  geometry: B={B} S={S} Nq={NQ} Nkv={NKV} D={D} past={PAST} total={TOTAL}")
    print(f"  sibling-key branch reached by {reach} (head, position, t) triple(s)")
    if reach == 0:
        print("  REFUSING: the changed branch is never reached; a null result here")
        print("            would be 0.0 == 0.0, not evidence.")
        return 2

    cpu, _ = run(model_path, feeds, False)
    runs = []
    execs = 0
    for _ in range(RUNS):
        vk, e = run(model_path, feeds, True)
        execs = e
        runs.append(vk)

    print(f"  vulkan node executions (last run): {execs}")
    if execs == 0:
        print("  REFUSING: the EP did not execute the node; both arms are the CPU's.")
        return 2

    rec = {
        "geometry": {
            "B": B, "S": S, "Nq": NQ, "Nkv": NKV, "D": D,
            "past": PAST, "total": TOTAL,
        },
        "sibling_branch_triples_reached": reach,
        "vulkan_node_executions": execs,
        "runs": RUNS,
        "outputs": {},
    }

    names = ["attn_out", "present_key", "present_value"]
    ok = True
    for i, name in enumerate(names):
        c = cpu[i].astype(np.float32)
        v = runs[0][i].astype(np.float32)
        max_abs = float(np.max(np.abs(c)))
        diff = float(np.max(np.abs(v - c)))
        denom = np.maximum(np.abs(c), 1e-6)
        rel = float(np.max(np.abs(v - c) / denom))
        identical = all(
            bool(np.array_equal(runs[j][i], runs[0][i])) for j in range(1, RUNS)
        )
        rec["outputs"][name] = {
            "max_abs": max_abs,
            "max_abs_diff": diff,
            "worst_rel": rel,
            "identical_across_runs": identical,
        }
        print(
            f"    {name:14s} max_abs={max_abs:8.4f} max_abs_diff={diff:10.6g} "
            f"worst_rel={rel:10.6g} identical_across_{RUNS}_runs={identical}"
        )
        if not identical:
            ok = False

    # Ledger tolerance for this form.
    RTOL, ATOL = 1e-3, 1e-5
    worst = max(r["worst_rel"] for r in rec["outputs"].values())
    within = worst <= RTOL
    rec["ledger_rtol"] = RTOL
    rec["ledger_atol"] = ATOL
    rec["worst_rel_over_all_outputs"] = worst
    rec["verdict"] = "MATCH" if (within and ok) else "DIVERGENT"
    print(f"\n  worst_rel over all outputs: {worst:.6g}  (ledger rtol {RTOL})")
    print(f"  VERDICT: {rec['verdict']}")

    out = HERE / "gqa_prefill_race.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"  record: {out}")
    model_path.unlink(missing_ok=True)
    return 0 if rec["verdict"] == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
