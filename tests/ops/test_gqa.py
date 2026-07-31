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

# GQA Compute-path bug: absent optional inputs (slots 1 and 2 are empty strings in the
# packed-QKV form) produce size=0 alloc requests at inference time.  The EP correctly
# claims the node at GetCapability; it fails at Compute because the allocator returns
# None for a 0-byte buffer, causing ORT to fall back to CPU.
#
# Root cause: packed-QKV GQA schema has 9 inputs; slots 1 and 2 ("key", "value") are
# absent in the Phi-3.5 packed form.  The kernel/allocator path does not skip absent
# optional inputs.  Owner: Switch (alloc.rs / ops/gqa.rs).
#
# strict=True so the suite goes red (XPASS) the moment the bug is fixed — matching
# Trinity's Phi-3.5 logits gate pattern.
_GQA_COMPUTE_BUG = pytest.mark.xfail(
    strict=True,
    reason=(
        "GQA Compute path: absent optional inputs (packed-QKV slots 1/2) produce "
        "size=0 alloc requests; EP falls back to CPU at inference time. "
        "Owner: Switch (alloc.rs / ops/gqa.rs). Remove when alloc handles absent "
        "optional inputs and these tests produce MATCH."
    ),
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_GQA_COMPUTE_BUG
@pytest.mark.parametrize("past_seq", [0, 4, 16])
def test_gqa_attn_output_matches_cpu(past_seq: int, vulkan_device_available) -> None:
    """attn_out from Vulkan EP must match CPU EP within f16 tolerance."""
    model_bytes = _build_gqa_model(past_seq)
    feeds = _make_inputs(past_seq)

    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, rtol=1e-2, atol=1e-2)


@_GQA_COMPUTE_BUG
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


@_GQA_COMPUTE_BUG
@pytest.mark.parametrize("past_seq", [0, 4])
def test_gqa_present_kv_matches_cpu(past_seq: int, vulkan_device_available) -> None:
    """present_key/value from Vulkan EP must match CPU EP (KV cache correctness)."""
    model_bytes = _build_gqa_model(past_seq)
    feeds = _make_inputs(past_seq, seed=7)

    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, rtol=1e-2, atol=1e-2)
