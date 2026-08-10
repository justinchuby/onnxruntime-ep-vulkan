"""
test_gqa.py — conformance tests for GroupQueryAttention (decode path)

Tests a single GQA node against ORT CPU EP as oracle.
Covers:
  - past_len=0 (benchmark case, decode step 0)
  - past_len=N (subsequent decode steps with KV cache)
  - GQA head grouping (kv_num_heads < num_heads)
Both devices (via conftest fixture).

All tests use the packed-QKV input form that Phi-3.5 GenAI emits:
  input 0: packed_qkv [B, S, (H+2*Hkv)*D]
  input 3: past_key    [B, Hkv, past_seq, D]
  input 4: past_value  [B, Hkv, past_seq, D]
  input 5: seqlens_k   [B] int32
  input 6: total_sequence_length  scalar int32 (unused by serial kernel, required by schema)
  input 7: cos_cache   [max_seq, D//2]
  input 8: sin_cache   [max_seq, D//2]

WHICH KERNEL THESE TESTS EXERCISE, and why that is deliberate (issue #90).
Every model built here declares `past_key`/`past_value` with a STATIC past extent — 0, 4 or 16 —
so `total_len = past + 1` is 1, 5 or 17.  The KV-parallel decode selector
(`ops::attention::gqa_decode_kv_lanes_with`) needs `total_len / W >= 32` for some W >= 2, which
none of those extents can reach, so it refuses and every test in this file dispatches the serial
`gqa_f16` module exactly as it did before #90.  That is what makes this file the REGRESSION
control for the old path: if a change to the selector ever made one of these forms take the new
module, these tests would be measuring something other than what they say they measure.
Coverage for the new module lives in `test_gqa_decode_kv_parallel.py`, which builds forms whose
extents clear the ladder.  Do not "modernise" the extents here to make them exercise the new
kernel — add cases there instead.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

import _models as m

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 32
ROTARY_DIM = 16   # partial rotary (neox, interleaved=0)
BATCH = 1
MAX_SEQ = 64
SCALE = float(HEAD_DIM ** -0.5)


def _cos_sin(max_seq: int, rot_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Simple sinusoidal cos/sin cache, shape [max_seq, rot_dim//2]."""
    pos = np.arange(max_seq, dtype=np.float32)[:, None]
    freq = 1.0 / (10000 ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    angles = pos * freq
    return np.cos(angles).astype(np.float16), np.sin(angles).astype(np.float16)


def _build_gqa_model(past_seq: int) -> bytes:
    """Build a one-node GQA ONNX model.  past_seq is the KV-cache size."""
    packed_qkv_dim = (NUM_HEADS + 2 * KV_HEADS) * HEAD_DIM
    schema_version = 1

    query_shape = ["B", "S", packed_qkv_dim]
    past_kv_shape = ["B", KV_HEADS, past_seq, HEAD_DIM]
    seqlens_shape = ["B"]

    in_types = {
        "packed_qkv": (TensorProto.FLOAT16, query_shape),
        "past_key":   (TensorProto.FLOAT16, past_kv_shape),
        "past_value": (TensorProto.FLOAT16, past_kv_shape),
        "seqlens_k":  (TensorProto.INT32,   seqlens_shape),
        "total_seq":  (TensorProto.INT32,   []),
        "cos_cache":  (TensorProto.FLOAT16, [MAX_SEQ, ROTARY_DIM // 2]),
        "sin_cache":  (TensorProto.FLOAT16, [MAX_SEQ, ROTARY_DIM // 2]),
    }

    out_shape = ["B", "S", NUM_HEADS * HEAD_DIM]
    pres_kv_shape = ["B", KV_HEADS, past_seq + 1, HEAD_DIM]

    inputs  = [helper.make_tensor_value_info(n, t, s) for n, (t, s) in in_types.items()]
    outputs = [
        helper.make_tensor_value_info("attn_out",   TensorProto.FLOAT16, out_shape),
        helper.make_tensor_value_info("present_key",   TensorProto.FLOAT16, pres_kv_shape),
        helper.make_tensor_value_info("present_value", TensorProto.FLOAT16, pres_kv_shape),
    ]

    gqa = helper.make_node(
        "GroupQueryAttention",
        inputs=["packed_qkv", "", "", "past_key", "past_value", "seqlens_k", "total_seq",
                "cos_cache", "sin_cache"],
        outputs=["attn_out", "present_key", "present_value"],
        domain="com.microsoft",
        name="gqa_0",
        num_heads=NUM_HEADS,
        kv_num_heads=KV_HEADS,
        scale=SCALE,
        local_window_size=-1,
        do_rotary=1,
        rotary_interleaved=0,
        smooth_softmax=0,
    )

    graph = helper.make_graph([gqa], "gqa_test", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", schema_version),
        ],
    )
    return model.SerializeToString()


def _make_inputs(past_seq: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    packed_dim = (NUM_HEADS + 2 * KV_HEADS) * HEAD_DIM
    packed_qkv = rng.standard_normal((BATCH, 1, packed_dim)).astype(np.float16) * 0.1
    past_key   = rng.standard_normal((BATCH, KV_HEADS, past_seq, HEAD_DIM)).astype(np.float16) * 0.1
    past_value = rng.standard_normal((BATCH, KV_HEADS, past_seq, HEAD_DIM)).astype(np.float16) * 0.1
    seqlens_k  = np.array([past_seq], dtype=np.int32)
    total_seq  = np.array(past_seq + 1, dtype=np.int32)
    cos, sin   = _cos_sin(MAX_SEQ, ROTARY_DIM)
    return {
        "packed_qkv": packed_qkv,
        "past_key":   past_key,
        "past_value": past_value,
        "seqlens_k":  seqlens_k,
        "total_seq":  total_seq,
        "cos_cache":  cos,
        "sin_cache":  sin,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Known-failing guard
# ---------------------------------------------------------------------------

# GQA Compute-path bug — RETIRED 2026-08-02 (Mouse).
#
# What it recorded: absent optional inputs (slots 1 and 2 are empty strings in the packed-QKV
# form) produced size=0 alloc requests at inference time, so the EP claimed the node at
# GetCapability and then fell back to CPU at Compute.  `strict=True` was there so the suite would
# go red (XPASS) the moment it was fixed.
#
# It did, and this is that red.  Switch's two GQA fixes (index space, then prefill) and the proof
# ledger entry that lets the form be claimed at all now put every test in this file at MATCH:
# `FAIL(condition): 0`, seven `XPASS(stale)`.  The marker's own removal condition — "remove when
# alloc handles absent optional inputs and these tests produce MATCH" — is met, so it is removed
# rather than relaxed.
#
# R13: an XPASS is neither a detection nor an outage. Leaving it in place would keep seven tests
# red for somebody else's good news, and a red nobody can act on is the one that gets ignored.

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("past_seq", [0, 4, 16])
def test_gqa_attn_output_matches_cpu(past_seq: int, vulkan_device_available) -> None:
    """attn_out from Vulkan EP must match CPU EP within f16 tolerance."""
    model_bytes = _build_gqa_model(past_seq)
    feeds = _make_inputs(past_seq)

    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("past_seq", [0, 4])
def test_gqa_present_kv_shape(past_seq: int, vulkan_device_available) -> None:
    """present_key and present_value must have shape [B, Hkv, past+1, D]."""
    model_bytes = _build_gqa_model(past_seq)
    feeds = _make_inputs(past_seq, seed=42)

    # assert_vulkan_claims first — vacuous-pass guard.
    m.assert_vulkan_claims(model_bytes, feeds)
    outs = m.run_vulkan(model_bytes, feeds)

    expected_kv = (BATCH, KV_HEADS, past_seq + 1, HEAD_DIM)
    assert outs[1].shape == expected_kv, f"present_key shape {outs[1].shape} != {expected_kv}"
    assert outs[2].shape == expected_kv, f"present_value shape {outs[2].shape} != {expected_kv}"


@pytest.mark.parametrize("past_seq", [0, 4])
def test_gqa_present_kv_matches_cpu(past_seq: int, vulkan_device_available) -> None:
    """present_key/value from Vulkan EP must match CPU EP (KV cache correctness)."""
    model_bytes = _build_gqa_model(past_seq)
    feeds = _make_inputs(past_seq, seed=7)

    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, rtol=1e-2, atol=1e-2)
