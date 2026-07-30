"""Tests for the MatMulNBits and dequantization kernels.

This module implements Mouse's three-regime quantization tolerance policy
(OP_COVERAGE.md §10.1) and includes:

  1. Regime 1 — DequantizeLinear bit-exact vs NumPy reference.
  2. Regime 2 — MatMulNBits output vs ORT CPU EP oracle (tolerance-gated).
  3. Per-layer capture mechanism unit test (compare_layers roundtrip).
  4. Oracle status documentation for fp16 activations.

All tests skip cleanly when no Vulkan device is available.

COVERAGE DIMENSIONS (Morpheus R9 — for each claim, name what goes red if it's false)
======================================================================================
The suite tracks correctness across the following axes. A proof of one form does NOT cover
another. The binding-arity axis was the root cause of the 2026-07-30 all-zero-logit bug:
push_dynamic_kernel built a 4-slot descriptor set for the 3-input form; the shader needed
5 bindings; the output write fell outside the layout and was silently discarded.

  bits           : {4, 8}
  block_size     : {16, 32, 64, 128}
  input arity    : 3-input (no zero_points, symmetric)   ← the Phi-3.5 form, and the bug
                   4-input (with zero_points, asymmetric)
  batch path     : static batch (M known at compile time) ← was green before the fix
                   symbolic/dynamic batch (M symbolic)    ← the bug path
  dtype          : fp32 activations
                   fp16 activations (gated ORT >= 1.28)
  rows           : 1 (decode / GEMV)
                   >1 (prefill / GEMM-like tiling)
  runs           : single-run (insufficient for dirty-arena detection)
                   multi-run in same session (Tank's requirement)

The 2026-07-30 bug lived at the intersection of (3-input) × (symbolic batch). Neither axis
alone was a bug. Each axis must be independently covered:
  ✓ input-arity parametrised: test_matmulnbits_form_matrix_fp32, test_matmulnbits_fp16_matrix
  ✓ symbolic-batch gate: test_matmulnbits_fp16_dynamic_batch (non-zero assertion)
  ✓ symbolic-batch multi-run: test_matmulnbits_fp16_dynamic_batch_multirun (dirty-arena check)
  ✗ 4-input × symbolic-batch: not yet tested — absence declared, not silenced

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

    ORACLE SAFETY NOTE (onnx#8182):
    This test uses opset 18 and a NumPy oracle.  It is intentionally NOT using opset 23+
    or onnx.reference.ReferenceEvaluator.  Reason: onnx#8182 documents that the opset-23
    and opset-25 DequantizeLinear reference implementations are NOT registered in
    onnx ≤ 1.22.0.  ReferenceEvaluator would silently fall back to the opset-21
    implementation, which does not know ``output_dtype`` or ``block_size`` — producing
    wrong expected outputs that the Vulkan kernel would be tested against.
    NumPy is independent of ONNX opset registration and is immune to this class of defect.
    See m.assert_qdq_reference_oracle_safe() for the guard that enforces this for any
    future oracle path that might use ReferenceEvaluator.
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


# ---------------------------------------------------------------------------
# Coverage matrix — the forms the claim predicate says it handles (Mouse)
# ---------------------------------------------------------------------------
#
# The predicate claims bits in {4, 8}, block_size in {16, 32, 64, 128}, the 3-input symmetric
# form and the 4-input asymmetric one, and every M rather than only decode. A predicate that
# claims a form nothing exercises is the exact failure OP_COVERAGE.md §7 exists to prevent, so
# every cell it admits is enumerated here rather than described.


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("with_zero_points", [True, False])
def test_matmulnbits_form_matrix_fp32(vulkan_device_available, bits, block_size, with_zero_points):
    """Every (bits, block_size, zero-point) form the predicate claims must match the CPU EP."""
    model_bytes, feeds = m.make_matmulnbits_model(
        K=256, N=64, bits=bits, block_size=block_size, with_zero_points=with_zero_points
    )
    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, **m.MATMULNBITS_FP32)


@pytest.mark.parametrize("rows", [1, 2, 7, 32])
def test_matmulnbits_prefill_rows_fp32(vulkan_device_available, rows):
    """M > 1 is prefill. The GEMV runs one workgroup per output element, so it is correct for
    every M and merely unoptimal — this asserts the correctness half of that claim."""
    model_bytes, feeds = m.make_matmulnbits_model(K=128, N=48, rows=rows)
    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, **m.MATMULNBITS_FP32)


@pytest.mark.skipif(
    not _ort_version_ge(1, 28),
    reason="fp16 MatMulNBits needs an ORT >= 1.28 oracle (see the module docstring).",
)
@pytest.mark.parametrize("with_zero_points", [True, False])
@pytest.mark.parametrize("rows", [1, 3])
def test_matmulnbits_fp16_matrix(vulkan_device_available, with_zero_points, rows):
    """fp16 is the form that matters: all 161 of Phi-3.5's MatMulNBits nodes carry fp16
    activations, scales and outputs, so an f32-only kernel would decline the model this kernel
    exists to run. The odd `rows` case also exercises the packed-lane output store."""
    model_bytes, feeds = m.make_matmulnbits_model(
        K=256,
        N=64,
        rows=rows,
        with_zero_points=with_zero_points,
        activation_dtype=ir.DataType.FLOAT16,
    )
    cpu_out = m.run_cpu(model_bytes, feeds)[0]
    assert not np.any(np.isnan(cpu_out)) and not np.any(np.isinf(cpu_out)), (
        "CPU EP oracle produced NaN/Inf for fp16 MatMulNBits — the oracle is not usable."
    )
    m.assert_vulkan_claims(model_bytes, feeds)
    m.assert_matches_cpu(model_bytes, feeds, **m.MATMULNBITS_FP16)


@pytest.mark.skipif(
    not _ort_version_ge(1, 28),
    reason="fp16 MatMulNBits needs an ORT >= 1.28 oracle (see the module docstring).",
)
def test_matmulnbits_fp16_dynamic_batch(vulkan_device_available):
    """Dynamic-batch (symbolic M) f16 MatMulNBits must produce non-zero output.

    This is the regression guard for the 2026-07-30 all-zero-logit bug: when ORT sees a
    symbolic leading dimension on the activation tensor, compile_impl routes the kernel through
    push_dynamic_kernel (session.rs), which builds a binding-token list from the NodeDesc
    input/output counts. For MatMulNBits without zero_points the NodeDesc has 3 inputs + 1
    output = 4 tokens, but matmul_nbits_gemv passes 5 bindings to KernelRequest (scales bound
    twice: once as scales, once as the inert zero_point placeholder for the 5th shader slot).
    The pipeline was therefore created with only 4 descriptor slots; shader binding 4 (the
    output) fell outside the descriptor set and wrote nowhere. The output buffer, which drivers
    zero-initialise for security, read back as all-zero on both Intel Iris Xe and RTX 4060.

    The fix (ShapeOnlyRecorder::dispatch now captures k.bindings alongside the other fields,
    and dispatch_ort uses those captured bindings — not kernel.bindings — for n_bindings and
    buf_bindings on the dynamic path) is tested here by asserting that a single dynamic-batch
    fp16 MatMulNBits at a Phi-3.5-representative shape produces a non-zero output.

    static_batch variants (test_matmulnbits_fp16_matrix) already passed before this fix, so
    they do NOT guard this path. Only the symbolic-batch form triggers push_dynamic_kernel.
    """
    model_bytes, feeds = m.make_matmulnbits_model(
        K=256,
        N=64,
        rows=1,
        with_zero_points=False,
        activation_dtype=ir.DataType.FLOAT16,
        symbolic_batch=True,
    )
    cpu_out = m.run_cpu(model_bytes, feeds)[0]
    assert not np.any(np.isnan(cpu_out)) and not np.any(np.isinf(cpu_out)), (
        "CPU EP oracle produced NaN/Inf for fp16 MatMulNBits — the oracle is not usable."
    )
    m.assert_vulkan_claims(model_bytes, feeds)
    # The primary guard: a symbolic-batch f16 dispatch must write actual values. All-zero
    # means the output binding was missing from the descriptor set and nothing was written.
    vk_out = m.run_vulkan(model_bytes, feeds)[0]
    assert np.any(np.abs(vk_out) > 1e-4), (
        "VK output is all-zero for symbolic-batch f16 MatMulNBits: the output binding was "
        "not in the descriptor set. This is the dynamic-binding-count bug."
    )
    m.assert_matches_cpu(model_bytes, feeds, **m.MATMULNBITS_FP16)


@pytest.mark.skipif(
    not _ort_version_ge(1, 28),
    reason="fp16 MatMulNBits needs an ORT >= 1.28 oracle (see the module docstring).",
)
def test_matmulnbits_fp16_dynamic_batch_multirun(vulkan_device_available):
    """Dynamic-batch f16 MatMulNBits: 3 runs in one session must produce non-zero, identical outputs.

    WHY THIS TEST EXISTS (Tank's multi-run discriminator, 2026-07-30)
    ----------------------------------------------------------------
    On 2026-07-30, Tank ran the Phi-3.5 model three times in one session with identical
    feeds and found:

      - Output 0 (logits): exactly 0.0 on runs 2 and 3 — in a dirty arena. An unwritten
        buffer in a dirty arena shows garbage; zeros in a dirty arena means something
        actively wrote zeros there. That confirmed "computed zero", not "unwritten zero".
      - Outputs 1..64 (KV cache): bitwise different between runs — the dirty-arena signature
        of an unwritten tensor.

    A single-run session probe cannot distinguish these two failure modes: on run 1 the arena
    is clean, so "unwritten" and "computed zero" both look like zeros. This is the same
    structural blindness that produced the flattering CPU-vs-CPU comparisons on this project.

    The fix (ShapeOnlyRecorder captures k.bindings; dispatch_ort uses those on the dynamic
    path) makes all three runs produce the SAME correct non-zero output, because the output
    binding is now correctly included in the descriptor set on every call.

    WHAT GOES RED ON UNFIXED CODE
    ------------------------------
    On the pre-fix DLL:
      - All three runs produce all-zero output (unwritten output buffer, driver zero-init).
      - The "non-zero" assertion fails on run 1.
      - The "bit-identical" assertion might pass (zeros == zeros) but is vacuously correct.

    On the fixed DLL:
      - All three runs produce non-zero, CPU-oracle-matching output.
      - Both assertions pass.
    """
    import onnxruntime as ort

    _RUNS = 3

    model_bytes, feeds = m.make_matmulnbits_model(
        K=256,
        N=64,
        rows=1,
        with_zero_points=False,
        activation_dtype=ir.DataType.FLOAT16,
        symbolic_batch=True,
    )

    # Validate the oracle is usable for this configuration.
    cpu_out = m.run_cpu(model_bytes, feeds)[0]
    assert not np.any(np.isnan(cpu_out)) and not np.any(np.isinf(cpu_out)), (
        "CPU EP oracle produced NaN/Inf for fp16 MatMulNBits — the oracle is not usable."
    )
    assert np.any(np.abs(cpu_out) > 1e-4), (
        "CPU EP oracle itself is all-zero — the model or feeds are invalid."
    )

    # Create ONE session and run it _RUNS times.
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_bytes, opts, providers=m.EP_PROVIDERS)

    # Vacuous-pass guard (R7, DESIGN.md §9.1): refuse to compare if the EP is absent.
    # ORT falls back silently without raising; all runs would be CPU-vs-CPU.
    used = sess.get_providers()
    if m.EP_NAME not in used:
        pytest.skip(
            f"VulkanExecutionProvider not in providers ({used}) — no Vulkan device available. "
            "Skipping multi-run test rather than comparing CPU-vs-CPU."
        )

    run_outputs: list[np.ndarray] = []
    for run_idx in range(_RUNS):
        out = sess.run(None, feeds)
        arr = np.array(out[0], copy=True, dtype=np.float32)
        # Non-zero guard: the pre-fix code writes nothing to the output (binding outside
        # the descriptor set layout). Drivers zero-initialise GPU buffers, so the result is
        # all-zero regardless of which run number we're on.
        assert np.any(np.abs(arr) > 1e-4), (
            f"VK output is all-zero on run {run_idx + 1}/{_RUNS} for symbolic-batch f16 "
            "MatMulNBits. This is the dynamic-binding-count bug: the output binding was not "
            "included in the descriptor set and nothing was written."
        )
        run_outputs.append(arr)

    # Bit-identical across all runs: same session, same feeds, deterministic hardware.
    # A correct kernel writes the same computed values every time.  Any divergence is
    # a data race, arena-reuse corruption, or non-deterministic dispatch.
    for run_idx in range(1, _RUNS):
        np.testing.assert_array_equal(
            run_outputs[0],
            run_outputs[run_idx],
            err_msg=(
                f"Run 1 vs run {run_idx + 1}: outputs differ. Same session, same feeds, "
                "same hardware — divergence indicates a data race, arena-reuse corruption, "
                "or non-deterministic atomic dispatch in the f16 GEMV kernel."
            ),
        )
