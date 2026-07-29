"""ONNX-IR model builders, session helpers, and assertion utilities for the Vulkan EP tests.

Models are constructed with ``onnx_ir`` (``ir.Value`` / ``ir.Node`` / ``ir.Graph`` /
``ir.Model``), never ``onnx.helper``. Comparisons run the model through the Vulkan EP (with
ORT CPU fallback available) and check the result against ORT's CPU EP, tolerance-gated.

TOLERANCE POLICY
================
Tolerances are per op-family, documented below. Widening any tolerance requires:
  1. Trinity's code-review sign-off.
  2. A comment in the test that names the driver exhibiting wider error and explains why
     the wider tolerance is acceptable.
Never widen a tolerance to turn a red test green without following this protocol.

fp32 elementwise (Add, Sub, Mul, Div, Neg, Abs, Sqrt, Reciprocal, Sign, Pow, Min, Max):
    rtol=1e-5, atol=1e-5
    Justification: IEEE 754 single-precision arithmetic with one rounding per operation.
    GPU and CPU should produce identical or near-identical results for these simple ops.
    1e-5 ≈ 1 ULP at typical activation magnitudes (~1.0). Tighter is not warranted because
    GPU and CPU may use slightly different FMA forms.

fp32 non-linear / transcendental (Exp, Log, Erf, Sin, Cos, Tan, etc.):
    rtol=1e-5, atol=1e-5
    Justification: Hardware transcendental units (GPU) may differ from libm (CPU) in the last
    1-2 ULPs at edge inputs. 1e-5 tolerates this. Do NOT widen without a per-driver data point.

fp32 activations (Relu, Clip, Sigmoid, Tanh, LeakyRelu, Elu, HardSigmoid, Softplus, Gelu):
    rtol=1e-5, atol=1e-5
    Justification: Composed of elementwise + non-linear operations; same tolerance applies.

fp32 comparison/logic (Equal, Greater, Less, And, Or, Not, Where):
    rtol=0, atol=0 (exact)
    Justification: Boolean/comparison results must be bit-exact — GPU and CPU must agree on
    every element. Any divergence is a correctness bug, not a tolerance issue.

fp16 any (M1+, gated on DeviceCapabilities.fp16_arithmetic):
    rtol=1e-3, atol=1e-3
    Justification: fp16 has 10-bit mantissa (~3.01 decimal digits). 1e-3 ≈ 0.5 ULP at
    typical activation magnitudes. Tighter than this is unreliable across GPU vendors
    because fp16 transcendentals vary by 1-2 ULPs. This is a measured, not pessimistic,
    bound: validated against NVIDIA RTX, AMD RDNA2, and lavapipe fp16 paths.

Reductions, GEMM, MatMul (M2+, OQ-10 in DESIGN.md §11):
    TBD — tolerance is accumulation-order-dependent and MUST be derived from test data
    per vendor (NVIDIA/AMD/lavapipe). A placeholder will be set when M2 ops land, with an
    explicit derivation comment. Do not guess; do not copy from fp32 elementwise.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EP_NAME = "VulkanExecutionProvider"
EP_PROVIDERS = [EP_NAME, "CPUExecutionProvider"]

# Tolerance constants — use these names in tests, not magic numbers.
FP32_ELEMENTWISE = {"rtol": 1e-5, "atol": 1e-5}
FP32_TRANSCENDENTAL = {"rtol": 1e-5, "atol": 1e-5}
FP32_ACTIVATION = {"rtol": 1e-5, "atol": 1e-5}
FP32_EXACT = {"rtol": 0, "atol": 0}
FP16_ANY = {"rtol": 1e-3, "atol": 1e-3}


# ---------------------------------------------------------------------------
# IR construction helpers
# ---------------------------------------------------------------------------


def tensor(name: str, dtype: ir.DataType, shape: list[int]) -> ir.Value:
    """A named, typed, shaped IR value — used for graph inputs and outputs."""
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def make_model(
    op_type: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
    opset: int = 21,
) -> bytes:
    """Build a single-node ONNX model. Pass fresh ir.Value instances per call."""
    node = ir.node(
        op_type,
        inputs,
        attributes=attributes or {},
        domain=domain,
        outputs=outputs,
    )
    opset_imports: dict[str, int] = {"": opset}
    if domain:
        opset_imports[domain] = 1
    # Absent optional inputs are represented as unnamed values (empty string) on the node
    # — they must not appear as graph inputs (would collide or be ambiguous).
    graph_inputs = [v for v in inputs if v.name]
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        name=f"vulkan_{op_type}",
        opset_imports=opset_imports,
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _make_session_options(*, profiling: bool = False, prefix: str = "vulkan_test") -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if profiling:
        opts.enable_profiling = True
        opts.profile_file_prefix = prefix
    validate = os.environ.get("ONNXRUNTIME_EP_VULKAN_VALIDATE", "").strip()
    if validate == "1":
        opts.add_session_config_entry("ep.enable_validation", "1")
    return opts


def _session(model: bytes, providers: list[str], *, profiling: bool = False) -> ort.InferenceSession:
    opts = _make_session_options(profiling=profiling)
    return ort.InferenceSession(model, opts, providers=providers)


def run_vulkan(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run a model through the Vulkan EP (CPU fallback available) and return its outputs."""
    return _session(model, EP_PROVIDERS).run(None, feeds)


def run_cpu(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run a model through ORT's CPU EP and return its outputs."""
    opts = _make_session_options()
    return ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)


# ---------------------------------------------------------------------------
# Claim assertion — the vacuous-pass guard
# ---------------------------------------------------------------------------


def assert_vulkan_claims(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert the Vulkan EP actually executed at least one node of *model*.

    Mechanism: ORT built-in profiling JSON (structured, not text parsing).
    - SessionOptions.enable_profiling = True
    - sess.end_profiling() → path to the trace JSON
    - Parse events where cat=="Node" and check args.provider for VulkanExecutionProvider.

    A test that passes *without* calling this function is vacuously correct: because CPU
    fallback is always correct, the output comparison will pass whether or not the EP ran
    anything. This function is the guard against that failure mode (DESIGN.md §9.1).
    """
    opts = _make_session_options(profiling=True, prefix="_vulkan_claim_probe")
    try:
        sess = ort.InferenceSession(model, opts, providers=EP_PROVIDERS)
        sess.run(None, feeds)
        profile_path = sess.end_profiling()
    except Exception as exc:
        raise AssertionError(
            f"Vulkan EP session failed — the EP may not be built or registered: {exc}"
        ) from exc

    try:
        with open(profile_path) as fh:
            events = json.load(fh)
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    providers_seen = {
        e["args"]["provider"]
        for e in events
        if e.get("cat") == "Node"
        and isinstance(e.get("args"), dict)
        and "provider" in e["args"]
    }
    assert EP_NAME in providers_seen, (
        f"{EP_NAME} did not execute any node of this model — the CPU-match check would be "
        f"a vacuous pass. Providers seen: {sorted(providers_seen) or 'none'}.\n"
        "Possible causes:\n"
        "  • No Vulkan device passed the capability gate (no ICD, or device below baseline).\n"
        "  • The claim predicate for this op rejected the node form — check "
        "    ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1 for decline reasons.\n"
        "  • The crate was not built yet (run cargo build --release first)."
    )


# ---------------------------------------------------------------------------
# Output comparison against the CPU oracle
# ---------------------------------------------------------------------------


def assert_matches_cpu(
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    rtol: float = FP32_ELEMENTWISE["rtol"],
    atol: float = FP32_ELEMENTWISE["atol"],
) -> None:
    """Assert the Vulkan EP output equals ORT's CPU EP output, tolerance-gated.

    The CPU EP is the sole correctness oracle (DESIGN.md §9.1). Never use numpy as the
    reference for ops that ORT's CPU EP supports.
    """
    expected = run_cpu(model, feeds)
    actual = run_vulkan(model, feeds)
    assert len(actual) == len(expected), (
        f"Output count mismatch: Vulkan EP returned {len(actual)}, CPU returned {len(expected)}"
    )
    for idx, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(
            got, want, rtol=rtol, atol=atol,
            err_msg=f"output[{idx}]: Vulkan EP vs ORT CPU EP mismatch"
        )


def cpu_can_run(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """Return True if ORT's CPU EP can execute this model (used to skip fp16 comparisons)."""
    try:
        opts = _make_session_options()
        ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
        return True
    except Exception:
        return False


def check(
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    rtol: float = FP32_ELEMENTWISE["rtol"],
    atol: float = FP32_ELEMENTWISE["atol"],
) -> None:
    """Assert the Vulkan EP claims the node; then (if CPU has a kernel) compare outputs.

    This is the standard test entry point: it first calls assert_vulkan_claims (the
    vacuous-pass guard), then assert_matches_cpu if the CPU EP can run the model. Use this
    in almost every test. Use assert_vulkan_claims alone only when the CPU EP lacks a kernel
    for the dtype and you only need to prove device placement.
    """
    assert_vulkan_claims(model, feeds)
    if cpu_can_run(model, feeds):
        assert_matches_cpu(model, feeds, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Inverse claim assertion — the conservative-claiming guard
# ---------------------------------------------------------------------------


def assert_vulkan_does_not_claim(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert that VulkanExecutionProvider executed ZERO nodes of *model*.

    This is the guard against over-claiming: if the EP claims an op form whose contract is
    not fully implemented it produces silent wrong results on some inputs or drivers. Use
    this for permanent CPU fallback ops (NonZero, fp64 arithmetic, data-dependent-shape ops)
    and for unsupported attribute combinations (e.g., Cast to fp64, dynamic shape metadata).

    The 'no Vulkan device' case is vacuously correct: with zero devices, the EP can never
    claim any node, so this assertion always passes — which is the right behaviour (no
    device → CPU fallback is guaranteed correct without further checks).
    """
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "_vulkan_noclaim_probe"
    try:
        sess = ort.InferenceSession(model, opts, providers=EP_PROVIDERS)
        sess.run(None, feeds)
        profile_path = sess.end_profiling()
    except Exception as exc:
        raise AssertionError(
            f"Session failed during not-claimed assertion — EP may be broken: {exc}"
        ) from exc

    try:
        with open(profile_path) as fh:
            events = json.load(fh)
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    providers_seen = {
        e["args"]["provider"]
        for e in events
        if e.get("cat") == "Node"
        and isinstance(e.get("args"), dict)
        and "provider" in e["args"]
    }
    assert EP_NAME not in providers_seen, (
        f"{EP_NAME} claimed a node it should NOT claim. "
        f"Providers seen: {sorted(providers_seen)}.\n"
        "This is a conservative-claiming violation: claiming an op form whose contract "
        "is not fully implemented produces silent wrong results on some inputs or drivers.\n"
        "Check the claim predicate for this op in rust/src/ops/ and ensure it rejects "
        "this input form."
    )
