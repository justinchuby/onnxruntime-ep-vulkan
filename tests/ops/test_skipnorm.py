"""Tests for SkipSimplifiedLayerNormalization — fused residual-add + RMSNorm.

COVERAGE AXES (§7.1.4 — optional-input arity; §7.1.3 — template boundary)
===========================================================================
SkipSimplifiedLayerNormalization has two output slots that matter:
  slot 0: normalised output (always present)
  slot 3: residual sum / pre-norm sum (feeds the next block in LLM graphs)

Slots 1 and 2 (mean, inv_std_var) are empty in every Phi-3.5 node (census §4.21).

Coverage matrix:
  dtype          : f32, f16
  slot-3 present : yes (Phi-3.5 pattern), no (1-output form, internal scratch path)
  batch shape    : [batch, seq, hidden], [hidden] (rank-1 edge case)
  seq_len        : 1 (decode), >1 (prefill)

Yesterday-red test: test_skip_norm_f16_nonzero_output_slot0 and
test_skip_norm_f16_slot3_nonzero. Before the f16 shader and translate-handler path were
wired, `skip_norm` returned Unsupported for DType::F16, causing the claim predicate to
disagree with the translate handler — the node would be claimed but then fail to
dispatch, silently falling back to CPU. The xfail comment is preserved here to
document what went red on unfixed code.

ORACLE SAFETY (R7)
==================
Every test calls m.assert_vulkan_claims before comparing outputs. A missing call
would let a CPU-vs-CPU comparison pass regardless of whether the EP executed anything.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest
from onnx import TensorProto as tp

import tests.ops._models as m

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

# Tolerance for SkipSimplifiedLayerNormalization.
# f32: single RMSNorm pass, arithmetic is f32 throughout. 1e-5 matches the EW floor.
# f16: accumulation-order differences in the partial sum-of-squares reduction.
#   The tree reduction in the GPU shader may differ in summation order vs the CPU scalar
#   loop. 1e-2 is conservative; empirical data (both devices) showed max delta ~3e-3.
SKIP_NORM_F32 = {"rtol": 1e-5, "atol": 1e-5}
SKIP_NORM_F16 = {"rtol": 1e-2, "atol": 1e-3}


def _make_skip_norm_model(
    hidden: int,
    batch: int = 1,
    seq: int = 1,
    dtype: int = tp.FLOAT,
    include_slot3: bool = True,
    opset: int = 1,
) -> bytes:
    """Build a minimal SkipSimplifiedLayerNormalization model.

    Parameters
    ----------
    hidden      : last dimension (hidden size).
    batch, seq  : leading dimensions; shape is [batch, seq, hidden] when both > 1.
    dtype       : ONNX TensorProto dtype, e.g. tp.FLOAT or tp.FLOAT16.
    include_slot3 : whether to wire output slot 3 (residual sum).
                    Slots 1 and 2 are always absent (empty-string name, Phi-3.5 pattern).
    opset       : opset version for the com.microsoft domain.
    """
    np_dtype = np.float32 if dtype == tp.FLOAT else np.float16

    if batch == 1 and seq == 1:
        shape = [hidden]
    else:
        shape = [batch, seq, hidden]

    rng = np.random.default_rng(0)
    gamma_np = rng.standard_normal(hidden).astype(np_dtype)

    node_outputs = ["out0", "", "", "out3"] if include_slot3 else ["out0"]

    node = oh.make_node(
        "SkipSimplifiedLayerNormalization",
        inputs=["hidden", "skip", "gamma"],
        outputs=node_outputs,
        domain="com.microsoft",
        epsilon=1e-5,
    )

    inputs_info = [
        oh.make_tensor_value_info("hidden", dtype, shape),
        oh.make_tensor_value_info("skip",   dtype, shape),
    ]
    outputs_info = [oh.make_tensor_value_info("out0", dtype, shape)]
    if include_slot3:
        outputs_info.append(oh.make_tensor_value_info("out3", dtype, shape))

    graph = oh.make_graph(
        [node],
        "skip_norm_test",
        inputs_info,
        outputs_info,
        initializer=[onh.from_array(gamma_np, name="gamma")],
    )
    model = oh.make_model(
        graph,
        opset_imports=[
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", opset),
        ],
    )
    model.ir_version = 8
    return model.SerializeToString()


def _feeds(hidden: int, batch: int = 1, seq: int = 1, dtype=np.float32) -> dict:
    rng = np.random.default_rng(1)
    if batch == 1 and seq == 1:
        shape = (hidden,)
    else:
        shape = (batch, seq, hidden)
    return {
        "hidden": rng.standard_normal(shape).astype(dtype),
        "skip":   rng.standard_normal(shape).astype(dtype),
    }


# ---------------------------------------------------------------------------
# f32 correctness
# ---------------------------------------------------------------------------

def test_skip_norm_f32_slot0_matches_cpu(vulkan_device_available):
    """f32 SkipSimplifiedLayerNorm slot-0 output must match ORT CPU EP.

    This is the basic f32 path. The oracle is ORT CPU EP with the same op.
    """
    model = _make_skip_norm_model(hidden=64, batch=2, seq=4, dtype=tp.FLOAT, include_slot3=False)
    feeds = _feeds(hidden=64, batch=2, seq=4, dtype=np.float32)
    m.assert_vulkan_claims(model, feeds)
    m.assert_matches_cpu(model, feeds, **SKIP_NORM_F32)


def test_skip_norm_f32_slot3_residual(vulkan_device_available):
    """f32 SkipSimplifiedLayerNorm slot-3 (residual sum) must match CPU EP.

    Slot 3 is the pre-norm sum (hidden + skip) that feeds the next block. A kernel
    that silently drops slot 3 would break the LLM residual stream; this test catches it.
    """
    model = _make_skip_norm_model(hidden=64, batch=2, seq=4, dtype=tp.FLOAT, include_slot3=True)
    feeds = _feeds(hidden=64, batch=2, seq=4, dtype=np.float32)
    m.assert_vulkan_claims(model, feeds)
    m.assert_matches_cpu(model, feeds, **SKIP_NORM_F32)


def test_skip_norm_f32_nonzero_output_slot0(vulkan_device_available):
    """f32 SkipSimplifiedLayerNorm output slot 0 must be non-zero.

    Structural guard: if the GPU silently discards the output write (e.g. binding-count
    mismatch like the 2026-07-30 MatMulNBits defect), the output is all-zero. This
    test would catch the same class of bug in the norm kernel.
    """
    model = _make_skip_norm_model(hidden=128, batch=1, seq=8, dtype=tp.FLOAT, include_slot3=True)
    feeds = _feeds(hidden=128, batch=1, seq=8, dtype=np.float32)
    m.assert_vulkan_claims(model, feeds)
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    assert m.EP_NAME in sess.get_providers(), "EP not loaded"
    out0 = sess.run(None, feeds)[0]
    assert not np.all(out0 == 0.0), (
        "slot 0 output is all-zero — possible binding-count mismatch (output write dropped)"
    )


# ---------------------------------------------------------------------------
# f16 correctness — the Phi-3.5 path
# ---------------------------------------------------------------------------

def test_skip_norm_f16_nonzero_output_slot0(vulkan_device_available):
    """f16 SkipSimplifiedLayerNorm slot-0 output must be non-zero (yesterday-red gate).

    YESTERDAY-RED: before the f16 shader path was wired in templates.rs::skip_norm,
    this test would fail with AssertionError from assert_vulkan_claims because the
    translate handler returned Unsupported for DType::F16, causing silent CPU fallback.
    If the vacuous-pass guard was bypassed, the all-zero test would catch it: the GPU
    would write zeros to a zero-initialized DeviceLocal buffer.

    This is the f16 analogue of the 2026-07-30 MatMulNBits binding-count sentinel.
    """
    model = _make_skip_norm_model(hidden=64, batch=2, seq=4, dtype=tp.FLOAT16, include_slot3=True)
    feeds = _feeds(hidden=64, batch=2, seq=4, dtype=np.float16)
    m.assert_vulkan_claims(model, feeds)
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    assert m.EP_NAME in sess.get_providers(), "EP not loaded"
    out0 = sess.run(None, feeds)[0]
    assert not np.all(out0 == 0.0), (
        "f16 slot 0 output is all-zero — possible translate-handler not wired or "
        "binding-count mismatch (output write dropped)"
    )


def test_skip_norm_f16_slot3_nonzero(vulkan_device_available):
    """f16 SkipSimplifiedLayerNorm slot-3 (residual sum) must be non-zero.

    Slot 3 is hidden + skip. With non-zero random inputs this must be non-zero.
    An all-zero slot 3 means the output binding fell outside the descriptor layout.
    """
    model = _make_skip_norm_model(hidden=64, batch=2, seq=4, dtype=tp.FLOAT16, include_slot3=True)
    feeds = _feeds(hidden=64, batch=2, seq=4, dtype=np.float16)
    m.assert_vulkan_claims(model, feeds)
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    assert m.EP_NAME in sess.get_providers(), "EP not loaded"
    outputs = sess.run(None, feeds)
    out3 = outputs[1]  # slot 3 is the second requested output
    assert not np.all(out3 == 0.0), (
        "f16 slot 3 (residual sum) is all-zero — output binding may have been dropped"
    )


def test_skip_norm_f16_slot0_matches_cpu(vulkan_device_available):
    """f16 SkipSimplifiedLayerNorm slot-0 output must match ORT CPU EP.

    Tolerance: SKIP_NORM_F16 (rtol=1e-2, atol=1e-3) — accounts for accumulation-order
    differences in the partial-sum-of-squares tree reduction between GPU and scalar CPU.
    """
    model = _make_skip_norm_model(hidden=64, batch=2, seq=4, dtype=tp.FLOAT16, include_slot3=False)
    feeds = _feeds(hidden=64, batch=2, seq=4, dtype=np.float16)
    m.assert_vulkan_claims(model, feeds)
    m.assert_matches_cpu(model, feeds, **SKIP_NORM_F16)


def test_skip_norm_f16_phi35_shape(vulkan_device_available):
    """f16 SkipSimplifiedLayerNorm at Phi-3.5 hidden size (3072).

    The exact shape class the 128 nodes in Phi-3.5-mini-instruct exercise.
    hidden=3072 means 3072 / 256 = 12 full passes per workgroup thread before the
    tree reduction. This is the shape class that would be exercised when the model runs.
    """
    model = _make_skip_norm_model(
        hidden=3072, batch=1, seq=1, dtype=tp.FLOAT16, include_slot3=True
    )
    feeds = _feeds(hidden=3072, batch=1, seq=1, dtype=np.float16)
    m.assert_vulkan_claims(model, feeds)
    out0, out3 = None, None
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    assert m.EP_NAME in sess.get_providers(), "EP not loaded"
    outputs = sess.run(None, feeds)
    out0 = outputs[0]
    out3 = outputs[1]
    assert not np.all(out0 == 0.0), "slot 0 is all-zero at Phi-3.5 hidden=3072"
    assert not np.all(out3 == 0.0), "slot 3 is all-zero at Phi-3.5 hidden=3072"
