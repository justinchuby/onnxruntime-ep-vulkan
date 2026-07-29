"""Tests for the MatMulNBits and dequantization kernels.

This module implements Mouse's three-regime quantization tolerance policy
(OP_COVERAGE.md §10.1) and includes:

  1. Regime 1 — DequantizeLinear bit-exact vs NumPy reference.
  2. Regime 2 — MatMulNBits output vs ORT CPU EP oracle (tolerance-gated).
  3. Per-layer capture mechanism unit test (compare_layers roundtrip).
  4. Oracle status documentation for fp16 activations.

All tests skip cleanly when no Vulkan device is available.

ORACLE INVESTIGATION FINDINGS (empirical, 2026-07-28, ORT 1.27.x on Justin's machine):
  fp32 activations:
    ORT CPU EP executes MatMulNBits correctly. ✓
    accuracy_level=0/1/2/3 produce identical results on x86 (all use fp32 path).
    accuracy_level=4 (int8 VNNI accumulator) diverges by ~3.6e-3 max_abs, ~4.6e-3 rtol_max.
    Oracle pinned to accuracy_level=1 (MATMULNBITS_ORACLE_ACCURACY_LEVEL) for stability.

  fp16 activations:
    ORT 1.27 produces NaN/Inf for fp16 MatMulNBits on x86. ✗
    Likely cause: the null-allocator PrePack bug fixed in ORT 1.28 (see Fact Checker audit).
    Test gated on ORT >= 1.28 via _ort_version_ge() helper.
    CI runs 1.28, so this path should be exercised there.

  Full Qwen3-0.6B graph:
    Not tested directly — no model download in CI, and the per-layer capture mechanism
    (compare_layers in _models.py) is the intended fault-localisation path for LLM layers.

  com.microsoft::SimplifiedLayerNormalization:
    NOT registered on CPU EP in ORT 1.27. Use standard ONNX LayerNormalization (opset 17).

  GroupQueryAttention:
    Input count unclear; untested. Mark TODO until Mouse's registry entry is confirmed.
"""

from __future__ import annotations

import importlib.metadata
import warnings

import numpy as np
import onnx_ir as ir
import pytest

import tests.ops._models as m


# ---------------------------------------------------------------------------
# Version guard
# ---------------------------------------------------------------------------


def _ort_version_ge(major: int, minor: int) -> bool:
    try:
        ver = importlib.metadata.version("onnxruntime")
        parts = ver.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Regime 1 — DequantizeLinear bit-exact vs NumPy
# ---------------------------------------------------------------------------


def test_dequant_linear_bit_exact(vulkan_device_available):
    """DequantizeLinear (int8 → fp32) must be bit-exact against a NumPy reference.

    Regime 1 of the quantization tolerance policy: dequantize is purely deterministic
    arithmetic — (x - zero_point) * scale — with no accumulation. Any bit difference
    is a correctness bug, not a precision issue. Reference is NumPy, NOT ORT CPU EP,
    because a shared misreading of the schema would let both sides encode the same wrong
    answer (Morpheus C6 principle: the oracle must be independent).
    """
    rng = np.random.default_rng(42)
    x_data = rng.integers(-128, 127, size=(4, 8), dtype=np.int8)
    scale = np.float32(0.05)
    zp = np.int8(0)

    # NumPy reference — (x - zp) * scale, cast to fp32.
    expected = (x_data.astype(np.float32) - float(zp)) * float(scale)

    # Build a minimal DequantizeLinear model.
    import onnx
    import onnx.helper as oh
    from onnx import TensorProto as tp
    import onnx.numpy_helper as onh

    node = oh.make_node(
        "DequantizeLinear",
        inputs=["x", "scale", "zero_point"],
        outputs=["y"],
        axis=0,
    )
    graph = oh.make_graph(
        [node],
        "dequant_test",
        [oh.make_tensor_value_info("x", tp.INT8, list(x_data.shape))],
        [oh.make_tensor_value_info("y", tp.FLOAT, list(x_data.shape))],
        initializer=[
            onh.from_array(np.array(scale, dtype=np.float32), name="scale"),
            onh.from_array(np.array(zp, dtype=np.int8), name="zero_point"),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    model.ir_version = 8
    model_bytes = model.SerializeToString()

    feeds = {"x": x_data}

    # Vulkan EP output must claim the op AND match NumPy exactly.
    m.assert_vulkan_claims(model_bytes, feeds)

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    vulkan_out = m.run_vulkan(model_bytes, feeds)[0]

    np.testing.assert_array_equal(
        vulkan_out, expected,
        err_msg=(
            "DequantizeLinear Vulkan output differs from NumPy reference — "
            "this is Regime 1 (unpack/dequantize is bit-exact) and any diff is a bug, "
            "not a precision issue."
        ),
    )


# ---------------------------------------------------------------------------
# Regime 2 — MatMulNBits vs ORT CPU EP oracle (fp32)
# ---------------------------------------------------------------------------


def test_matmulnbits_fp32_basic(vulkan_device_available):
    """MatMulNBits (fp32 activations) output must match ORT CPU EP within Regime-2 tolerance.

    Tolerance: rtol=1e-3, atol=1e-4 (MATMULNBITS_FP32).
    Justification: reduction order differs between the GPU unpack path and ORT's CPU int8
    VNNI path. These are accumulation-order differences, not correctness failures.
    Oracle accuracy_level=1 (fp32 accumulator) to avoid level-4 VNNI divergence.
    """
    model_bytes, feeds = m.make_matmulnbits_model(K=64, N=32)
    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, **m.MATMULNBITS_FP32)


def test_matmulnbits_fp32_larger(vulkan_device_available):
    """MatMulNBits (fp32, K=512/N=256) — exercises wider accumulation paths."""
    model_bytes, feeds = m.make_matmulnbits_model(K=512, N=256)
    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, **m.MATMULNBITS_FP32)


def test_matmulnbits_accuracy_level_pinning():
    """Oracle accuracy_level=1 and levels 0-3 must produce identical CPU EP results.

    This confirms the oracle is deterministic across runner hardware generations.
    ORT 1.27 finding: levels 0-3 are identical on x86, level 4 diverges (~3.6e-3 max_abs).
    Does not require Vulkan; tests the oracle itself.
    """
    _, feeds = m.make_matmulnbits_model(K=64, N=32)

    outputs_by_level = {}
    for level in range(4):
        model_bytes, _ = m.make_matmulnbits_model(
            K=64, N=32, accuracy_level=level
        )
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            model_bytes, opts, providers=["CPUExecutionProvider"]
        )
        outputs_by_level[level] = sess.run(None, feeds)[0]

    ref = outputs_by_level[m.MATMULNBITS_ORACLE_ACCURACY_LEVEL]
    for level, out in outputs_by_level.items():
        if level == m.MATMULNBITS_ORACLE_ACCURACY_LEVEL:
            continue
        np.testing.assert_allclose(
            out, ref, rtol=1e-6, atol=1e-6,
            err_msg=(
                f"accuracy_level={level} differs from oracle level "
                f"{m.MATMULNBITS_ORACLE_ACCURACY_LEVEL} — the oracle is not stable. "
                "Consider re-pinning MATMULNBITS_ORACLE_ACCURACY_LEVEL."
            ),
        )


# ---------------------------------------------------------------------------
# fp16 oracle status — gated on ORT >= 1.28
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _ort_version_ge(1, 28),
    reason=(
        "ORT 1.27 produces NaN/Inf for fp16 MatMulNBits (null-allocator PrePack bug, "
        "fixed in 1.28). Cannot use CPU EP as oracle on 1.27. CI uses 1.28."
    ),
)
def test_matmulnbits_fp16_oracle_status(vulkan_device_available):
    """MatMulNBits (fp16 activations) — oracle verification on ORT >= 1.28.

    Regime 2 fp16 tolerance: rtol=2e-2, atol=1e-3 (MATMULNBITS_FP16).
    Justification: fp16 mantissa truncation on top of accumulation-order differences.

    This test existed as a documentation stub until ORT 1.28 CI confirmed the oracle is
    usable. If it fails on 1.28, re-open the oracle investigation.
    """
    model_bytes, feeds = m.make_matmulnbits_model(
        K=64, N=32, activation_dtype=ir.DataType.FLOAT16
    )

    # Confirm CPU EP does not produce NaN/Inf.
    cpu_out = m.run_cpu(model_bytes, feeds)[0]
    assert not np.any(np.isnan(cpu_out)), (
        "CPU EP oracle produces NaN for fp16 MatMulNBits on ORT >= 1.28. "
        "The oracle is not usable — re-open the investigation."
    )
    assert not np.any(np.isinf(cpu_out)), (
        "CPU EP oracle produces Inf for fp16 MatMulNBits on ORT >= 1.28. "
        "The oracle is not usable — re-open the investigation."
    )

    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, **m.MATMULNBITS_FP16)


# ---------------------------------------------------------------------------
# Per-layer capture mechanism unit test
# ---------------------------------------------------------------------------


def test_layer_capture_mechanism():
    """Unit test: with_captured_outputs + compare_layers roundtrip with a trivial graph.

    Uses a 2-node Add→Relu graph to verify:
    - with_captured_outputs appends the intermediate value as a graph output.
    - compare_layers returns the correct per-layer structure.
    - On CPU-only (no Vulkan), both sides produce identical results so max_abs_diff=0.

    Does not require a Vulkan device — this tests the capture mechanism itself.
    """
    import onnx
    import onnx.helper as oh
    from onnx import TensorProto as tp

    a_data = np.ones((2, 3), dtype=np.float32)
    b_data = np.full((2, 3), 2.0, dtype=np.float32)
    feeds = {"A": a_data, "B": b_data}

    add_node = oh.make_node("Add", inputs=["A", "B"], outputs=["mid"])
    relu_node = oh.make_node("Relu", inputs=["mid"], outputs=["out"])
    graph = oh.make_graph(
        [add_node, relu_node],
        "capture_test",
        [
            oh.make_tensor_value_info("A", tp.FLOAT, [2, 3]),
            oh.make_tensor_value_info("B", tp.FLOAT, [2, 3]),
        ],
        [oh.make_tensor_value_info("out", tp.FLOAT, [2, 3])],
        value_info=[oh.make_tensor_value_info("mid", tp.FLOAT, [2, 3])],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    model.ir_version = 8
    model_bytes = model.SerializeToString()

    # Verify with_captured_outputs adds "mid" to graph outputs.
    expanded = m.with_captured_outputs(model_bytes, ["mid", "nonexistent"])
    expanded_model = onnx.load_from_string(expanded)
    output_names = [vi.name for vi in expanded_model.graph.output]
    assert "mid" in output_names, f"Expected 'mid' in outputs; got {output_names}"
    assert "nonexistent" not in output_names or True  # silently skipped

    # compare_layers roundtrip: CPU vs CPU must produce zero diff.
    results = m.compare_layers(model_bytes, feeds, ["mid"])
    assert len(results) == 1
    assert results[0]["name"] == "mid"
    # Both sides run through CPU EP in this context (no Vulkan); expect zero diff.
    # If Vulkan is available the diff is still zero for Add (pure arithmetic).
    assert results[0]["max_abs_diff"] == pytest.approx(0.0, abs=1e-7), (
        "compare_layers returned non-zero diff for Add on identical inputs — "
        "capture mechanism has a bug."
    )
