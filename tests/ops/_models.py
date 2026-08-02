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

Quantized ops — three-regime policy (Mouse's spec, OP_COVERAGE.md §10.1):

  Regime 1 — Unpack / dequantize (DequantizeLinear, int4→fp):
    rtol=0, atol=0 (bit-exact against NumPy reference)
    Justification: dequantize is purely deterministic arithmetic — (x - zp) * scale — with
    no accumulation. Any bit difference is a correctness bug, not a precision issue.
    Reference: NumPy (NOT ORT CPU EP, because a shared misreading of the schema would let
    both sides of the comparison encode the same wrong answer).

  Regime 2 — MatMulNBits output vs ORT CPU EP:
    fp32 activations: rtol=1e-3, atol=1e-4
    fp16 activations: rtol=2e-2, atol=1e-3
    Justification: the GPU unpack and accumulate in a different order from ORT's CPU int8
    VNNI path. The differences are accumulation-order-only, not correctness failures.
    Empirically measured (2026-07-28, Justin's dev machine, K=1024, N=512):
      accuracy_level=4 (int8 accumulator) vs accuracy_level=1 (fp32):
        max_abs=3.6e-3, mean_abs=7.5e-4, rtol_max=4.6e-3
    Oracle is pinned at accuracy_level=MATMULNBITS_ORACLE_ACCURACY_LEVEL (1, fp32 accumulator).
    The 2e-2 rtol for fp16 accommodates fp16 mantissa truncation on top of accumulation order.
    Source: Mouse's OP_COVERAGE.md §10.1; derived from measurement, not guessed.

    ACCURACY_LEVEL RULING (Mouse, 2026-07-30 — for Trinity's oracle question):
    Phi-3.5-mini-instruct declares accuracy_level=0.  Trinity's oracle is pinned at
    accuracy_level=1.  Ruling: the oracle pinning is correct and introduces no error.

    ORT's CPU MatMulNBits kernel ignores accuracy_level for values 0-3 — they all map to
    SQNBIT_CompFp32 (fp32 accumulation), confirmed empirically on ORT 1.27.x with
    test_matmulnbits_accuracy_level_pinning() and identically in ORT source (SQNBitGemm.cpp,
    `ComputeType` selection).  Only accuracy_level=4 changes computation (int8 VNNI accumulator),
    and that diverges by ~4.6e-3 rtol — which is why we pin away from it.

    The GPU shader always uses float accumulation regardless of the attribute:
      float acc = 0.0;
      ... unpack and accumulate in fp32 ...
    so accuracy_level is not readable by the GPU path at all.

    Consequence: a comparison between a model with accuracy_level=0 and an oracle at
    accuracy_level=1 compares two instances of the same computation.  Pinning at 1 is correct.

    IMPORTANT: fp16 activations require ORT 1.28+ for a usable oracle. ORT 1.27 produces
    NaN/Inf for fp16 MatMulNBits on x86 (empirically confirmed 2026-07-28; suspected null-
    allocator PrePack bug, fixed in 1.28). Tests gated on ORT >= 1.28 for fp16 paths.

  Regime 3 — End-to-end LLM: top-1 token agreement over 64 greedy steps + KL bound.
    Tolerance: per-model, derived at M3+ when actual LLM tests land.
    Justification: final logit per-element comparison on a 150k-vocab model is meaningless —
    a broken kernel can maintain top-1 accuracy for many tokens before divergence is visible
    (Morpheus C6, DESIGN.md §9.1). Per-layer intermediate capture (compare_layers() below)
    is the correct fault-localisation mechanism.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

import _verdict

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
# Shape inference helpers (DESIGN.md §8.6 — coverage work, not harness polish)
# ---------------------------------------------------------------------------


def apply_shape_inference(model_bytes: bytes, *, warn_on_missing: bool = False) -> bytes:
    """Apply onnx_shape_inference.infer_symbolic_shapes to serialised model bytes.

    Converts bytes → ir.Model, runs infer_symbolic_shapes (in-place), serialises
    back to bytes. This fills in output shapes that were left None/unspecified in
    the exported model, turning dynamic-shape declines into potential claims without
    any kernel changes.

    IMPORTANT CAVEAT: A node claimed only *after* this step is claimable exclusively
    in a pipeline that includes this preprocessing. Report coverage as two separate
    numbers — "without preprocessing" and "additionally after preprocessing" — and
    never add them without noting the precondition. Coordinate with Mouse on whether
    the registry should distinguish "declined for dynamic shape (always)" from
    "claimable if shapes were known" for correct RESULTS.md entries.

    Parameters
    ----------
    model_bytes :
        Serialised ONNX proto (any builder in this module, or a real exported model).
    warn_on_missing :
        Forwarded to infer_symbolic_shapes. False (default) keeps test output clean.
    """
    try:
        from onnx_shape_inference import infer_symbolic_shapes
    except ImportError as exc:
        raise ImportError(
            "onnx-shape-inference is required for shape-inference preprocessing.\n"
            "Install: pip install onnx-shape-inference\n"
            "Or: pip install -r tests/requirements.txt"
        ) from exc

    import onnx as _onnx  # noqa: PLC0415 — onnx is already a transitive dep (onnx_ir)
    try:
        onnx_proto = _onnx.ModelProto.FromString(model_bytes)
    except Exception as exc:
        raise ValueError(f"apply_shape_inference: could not deserialise model bytes: {exc}") from exc

    ir_model = ir.from_proto(onnx_proto)
    infer_symbolic_shapes(ir_model, warn_on_missing=warn_on_missing)
    return ir.to_proto(ir_model).SerializeToString()


def make_model_dynamic_output(
    op_type: str,
    inputs: list[ir.Value],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
    opset: int = 21,
    n_outputs: int = 1,
    output_dtype: ir.DataType = ir.DataType.FLOAT,
    output_names: list[str] | None = None,
) -> bytes:
    """Build a single-node model where output shapes are intentionally left unspecified.

    Unlike make_model, this function does NOT annotate output shapes. This replicates
    the common case of models exported from frameworks that do not fully annotate
    intermediate shapes. The EP declines such nodes with code="dynamic-shape".

    After calling apply_shape_inference(), the output shapes will be propagated from
    the (concrete-shaped) input values, turning declines into claims.

    Parameters
    ----------
    op_type :
        ONNX op type, e.g. "Add", "Relu".
    inputs :
        Input values with explicit shapes. Shapes must be concrete integers or named
        symbolic dims; the inferred output shapes derive from them.
    domain :
        Op domain. Default "" = ai.onnx.
    attributes :
        Op attributes, if any.
    opset :
        Opset version.
    n_outputs :
        Number of outputs (default 1).
    output_dtype :
        Data type for all outputs. Default FLOAT.
    output_names :
        Output value names. If omitted, uses "out0", "out1", ...
    """
    names = output_names or [f"out{i}" for i in range(n_outputs)]
    outputs = [
        ir.Value(name=n, type=ir.TensorType(output_dtype), shape=None)
        for n in names
    ]
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
    graph_inputs = [v for v in inputs if v.name]
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        name=f"dyn_{op_type}",
        opset_imports=opset_imports,
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


# ---------------------------------------------------------------------------
# Machine-readable claim log helpers (Mouse's OP_COVERAGE.md §10.2)
# ---------------------------------------------------------------------------


def read_claim_log(log_path: "Path | str") -> dict[str, dict]:
    """Parse a CLAIM_LOG JSON Lines file into a dict keyed by qualified op name.

    Set ONNXRUNTIME_EP_VULKAN_CLAIM_LOG=<path> before session creation and the EP writes
    one JSON Lines record per claim decision, flushed immediately. Each record has:
        op      — domain-qualified op type  (e.g. "com.microsoft::MatMulNBits")
        node    — graph node name (or "" if unnamed)
        opset   — resolved since_version (or 0 if unresolved)
        claimed — bool
        code    — DeclineCode tag ("not-registered", "attribute", etc.) or null if claimed
        reason  — human-readable explanation string

    Returns {op_name: last_record} for each op seen in the file.
    Returns {} if the file doesn't exist (EP not built, or log not written).

    Usage:
        log_path = Path(__file__).parent / f"_claim_{os.getpid()}.jsonl"
        os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log_path)
        try:
            InferenceSession(model, ...)
            claims = read_claim_log(log_path)
            assert claims["Add"]["claimed"]
            assert claims["com.microsoft::NotARealOp"]["code"] == "not-registered"
        finally:
            del os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"]
            log_path.unlink(missing_ok=True)
    """
    from pathlib import Path as _Path

    p = _Path(log_path)
    if not p.exists():
        return {}
    result: dict[str, dict] = {}
    try:
        for line in p.read_text("utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    result[record["op"]] = record
                except (json.JSONDecodeError, KeyError):
                    pass
    except OSError:
        pass
    return result


# ---------------------------------------------------------------------------
# Quantization tolerance constants (Mouse's three-regime policy, OP_COVERAGE §10.1)
# ---------------------------------------------------------------------------

# Regime 1: Unpack / dequantize — must be bit-exact vs NumPy.
# Reference: NumPy (not ORT CPU EP — see module docstring for why).
DEQUANT_EXACT = {"rtol": 0, "atol": 0}

# Regime 2: MatMulNBits output vs ORT CPU EP oracle.
# Empirically derived 2026-07-28 (see module docstring for measurement details).
MATMULNBITS_FP32 = {"rtol": 1e-3, "atol": 1e-4}   # fp32 activations
MATMULNBITS_FP16 = {"rtol": 2e-2, "atol": 1e-3}   # fp16 activations

# Oracle pinning: always use accuracy_level=1 (fp32 accumulator) for the CPU EP oracle.
# Levels 0-3 are identical on x86 (all use fp32 path) but level 4 (int8 VNNI) diverges
# by ~4.6e-3 rtol — larger than MATMULNBITS_FP32.rtol. Pinning makes the reference
# deterministic and reproducible across runner hardware generations.
MATMULNBITS_ORACLE_ACCURACY_LEVEL: int = 1


# ---------------------------------------------------------------------------
# Q/DQ opset-23+ oracle registration guard (onnx#8182)
# ---------------------------------------------------------------------------
#
# onnx#8182: the opset-23 and opset-25 reference implementations for
# QuantizeLinear / DequantizeLinear are NOT registered in onnx ≤ 1.22.0.
# onnx.reference.ReferenceEvaluator silently falls back to the opset-21 implementation,
# which does not know the ``output_dtype`` attribute (new in opset 23) or the
# ``block_size`` attribute (new in opset 23).  The fallback either:
#   a. raises TypeError (onnx 1.22.0, observed) — detectable, therefore safe;
#   b. returns a number using the wrong semantics — undetectable, therefore dangerous.
#
# The fix is targeted at onnx 1.23.0, which does not exist as of 2026-07-30.
# Detection and refusal are the only options available until then.
#
# IMPORTANT: our Regime-1 oracle for DequantizeLinear is NumPy, NOT ReferenceEvaluator.
# Our Regime-2 oracle for MatMulNBits is ORT CPU EP, NOT ReferenceEvaluator.
# Neither is affected by onnx#8182 today.  This guard exists so any future test that
# adds a ReferenceEvaluator-based Q/DQ oracle will see an immediate, named refusal rather
# than a silent wrong expected value.
#
# No-bump behavioral correction class (broader context, §9.4.1):
#   onnx#8182 is one of at least 9 known instances of behavioral corrections applied to
#   ONNX ops without an opset version bump.  This class is invisible to opset-version
#   fingerprinting and to ContribSchema baseline checks by construction.
#   Other instances in our plan: onnx#8099 (ScatterND min/max), onnx#8194 (TopK sorted=0).
#   Detection via code: Trinity runs assert_qdq_reference_oracle_safe() before any
#   ReferenceEvaluator oracle path.
#   Detection of new instances across all ops: requires a per-onnx-release human audit.
#   See decisions inbox: trinity-qdq-oracle-guard-no-bump-audit-2026-07-30.md.

# Probed once at pytest_configure time in conftest.py; the conftest updates this via
# its own module-level globals.  Tests read this to decide whether to call the guard.
# Default: False — conservatively refuses until the conftest probe confirms safety.
QDQOPSET23_REFERENCE_SAFE: bool = False
QDQOPSET23_REFERENCE_STATUS: str = "not probed — run via pytest to get conftest.py probe"


def assert_qdq_reference_oracle_safe(
    opset: int,
    attributes: Sequence[str],
) -> None:
    """Refuse to proceed if ReferenceEvaluator cannot correctly evaluate Q/DQ at *opset*.

    ALWAYS call this before constructing any oracle that uses
    ``onnx.reference.ReferenceEvaluator`` for QuantizeLinear or DequantizeLinear nodes.

    Args:
        opset: The ONNX opset of the Q/DQ node being evaluated.
        attributes: The attribute names present on the node (e.g. ``["output_dtype"]``).

    Raises:
        RuntimeError: If the opset and attributes combination is affected by onnx#8182
            and the current environment cannot produce a correct result.

    Note:
        For opset < 23 and for forms without ``output_dtype`` or ``block_size``, the
        opset-21 fallback is semantically correct and this function does not raise.
        Our current Regime-1 test uses opset 18 + NumPy oracle — it does NOT call this.

    Background (onnx#8182):
        The opset-23 and opset-25 Q/DQ reference implementations are not registered in
        onnx ≤ 1.22.0.  ReferenceEvaluator falls back to opset-21, which does not know
        ``output_dtype`` or ``block_size``.  In onnx 1.22.0 the fallback raises TypeError
        (detectable).  In a future release that registers the op incorrectly, the fallback
        could silently return a wrong number (the dangerous case).  Either way, this
        function refuses before the wrong result reaches the test oracle.
    """
    _AFFECTED_ATTRIBUTES = frozenset({"output_dtype", "block_size"})
    _affected = opset >= 23 and bool(_AFFECTED_ATTRIBUTES.intersection(attributes))

    if not _affected:
        return  # opset < 23 or no affected attributes — safe to proceed

    if not QDQOPSET23_REFERENCE_SAFE:
        raise RuntimeError(
            f"Q/DQ oracle refused: opset {opset} with attributes {sorted(attributes)!r} "
            f"is affected by onnx#8182 (unregistered opset-23/25 Q/DQ reference "
            f"implementations in onnx ≤ 1.22.0). "
            f"Current environment status: {QDQOPSET23_REFERENCE_STATUS}. "
            f"The fix is targeted at onnx 1.23.0 (not yet released as of 2026-07-30). "
            f"Use a NumPy oracle (Regime 1) or ORT CPU EP oracle (Regime 2) instead. "
            f"Do NOT use ReferenceEvaluator for Q/DQ at opset >= 23 with these attributes."
        )

    import warnings
    warnings.warn(
        f"Q/DQ oracle at opset {opset} with attributes {sorted(attributes)!r}: "
        f"this environment raises for the affected form (onnx#8182 fallback is live but "
        f"detectable). The oracle will fail with TypeError rather than returning a wrong "
        f"number. Use a NumPy or ORT CPU EP oracle instead of ReferenceEvaluator.",
        stacklevel=2,
    )




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


def outputs_bit_equal(a: list[np.ndarray], b: list[np.ndarray]) -> tuple[bool, list[int]]:
    """Test bit-exact equality between two output lists using raw memory comparison.

    Returns (all_equal, list_of_differing_output_indices).

    WHY RAW BYTES, NOT max|a-b|
    ============================
    ``np.max(np.abs(a - b))`` returns ``nan`` whenever *either* array contains a NaN,
    even if both arrays are bit-identical (e.g., same NaN bit pattern in both). This was
    observed by Tank when comparing two fp16 KV-cache outputs near the representable limit
    (~65472 / fp16-max 65504): the first diff used max|a-b| and reported nan for all 64
    outputs — implying all 64 differed — when in fact the two arrays were bit-identical.
    He rewrote to raw-byte equality before trusting the result.

    ``np.array_equal(a, b, equal_nan=True)`` is almost correct but has a subtlety: it
    treats two NaN *values* as equal even if they have different bit patterns (different
    sign/payload bits). For correctness gates on fp16 KV outputs, two NaN payloads from
    different compute paths can carry different bits; calling them equal would mask a real
    difference. Raw byte equality has no such ambiguity: bit-identical means bit-identical.

    LIMITATION
    ==========
    Raw byte equality is strictly tighter than "numerically close". This function is only
    appropriate for cross-run determinism checks (same computation → same bits) and for
    detecting unwritten buffers (all-zero). For EP-vs-CPU numeric tolerance, use
    ``np.testing.assert_allclose``.
    """
    if len(a) != len(b):
        return False, list(range(max(len(a), len(b))))
    differing = [
        i for i, (x, y) in enumerate(zip(a, b))
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes()
    ]
    return len(differing) == 0, differing


# =========================================================================================
# All-output CPU-oracle comparison (criterion 10's missing arm)
# =========================================================================================
#
# WHY THIS EXISTS
# ===============
# Criterion 10 closed on 2026-08-01 and was reopened three hours later.  ``_compare_run_to
# _cpu`` in test_criterion10.py binds ``vk_out[0]`` / ``cpu_out[0]`` and derives every
# oracle fact from the logits.  **No KV output was compared against CPU anywhere in the
# tree.**  The tree does have an all-65 gate, but it is cross-run identity — Vulkan run 1
# == run 2 == run 3 — which proves determinism, not correctness:
#
#     gate                  outputs   property
#     cross-run identity    all 65    determinism only
#     CPU oracle             1 of 65  correctness
#
# A deterministically wrong KV write passes both.  That is not a hypothetical: I planted
# one before writing this and every gate stayed green — see
# ``bench/results/planted_kv_probe.json``.  Sixty-four of sixty-five outputs were zero and
# the series verdict was ``MATCH``.
#
# The row had been closed on *divergence*, which is the symptom of a dirty arena.  On a
# clean arena the same binding-arity defect is stable and silent.  Morpheus's sentence: the
# closure certified the symptom was gone; it never established the defect was fixed.
#
# NON-TRIVIALITY IS NOT OPTIONAL HERE
# ===================================
# Sixty-four pairs of zeros pass an all-output allclose *perfectly*.  That is ``0.0 == 0.0``
# in another costume, and it is the precise failure this arm exists to catch — so a
# comparison over degenerate tensors reports ``NOT_PERFORMED``, never ``AGREE``.  Absence of
# evidence gets its own token rather than borrowing the passing one.

#: KV-cache outputs of the int4 Phi-3.5 artifact are fp16 and are produced by the
#: MatMulNBits qkv path.  The tolerance is therefore **not chosen for this gate** — it is the
#: fp16 MatMulNBits tolerance already derived from measured data in the header of this file
#: (max_abs=3.6e-3, rtol_max=4.6e-3 observed; 2e-2/1e-3 pinned above it).  Reusing the
#: tolerance justified for the arithmetic that produces the tensor is the justification;
#: picking a number that makes this gate green would not be.
KV_CACHE_FP16 = dict(MATMULNBITS_FP16)

#: fp32 outputs on the same path get the fp32 MatMulNBits tolerance, same reasoning.
KV_CACHE_FP32 = dict(MATMULNBITS_FP32)


def tolerance_for_output(arr: np.ndarray) -> tuple[dict, str]:
    """Return ``(tolerance, justification)`` for one output, keyed on its dtype.

    The justification travels with the number.  A tolerance whose reason is not recorded
    beside it is a tolerance nobody can re-derive, and this project has already been bitten
    by a default defended by a reason its own documentation did not give.
    """
    if arr.dtype == np.float16:
        return KV_CACHE_FP16, (
            "fp16 MatMulNBits tolerance (_models.py header, derived from measured data: "
            "max_abs=3.6e-3, rtol_max=4.6e-3); the qkv path that produces this tensor is "
            "the arithmetic that tolerance was justified for"
        )
    if arr.dtype in (np.float32, np.float64):
        return KV_CACHE_FP32, (
            "fp32 MatMulNBits tolerance (_models.py header); same path, fp32 activations"
        )
    return FP32_EXACT, (
        f"{arr.dtype} is integral or non-floating; a tolerance would be meaningless and any "
        "difference is a defect"
    )


def _is_degenerate(arr: np.ndarray) -> bool:
    """A tensor carrying no information: empty, all-NaN, or every element identical.

    All-zero is the case that matters — an output outside the descriptor set is never
    written and reads back zero-initialised on both Intel and NVIDIA drivers — but the guard
    is written on *constancy* rather than on zero, because a buffer left holding one repeated
    residue value is the same absence of evidence and a zero-only test would miss it.
    """
    if arr.size == 0:
        return True
    flat = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.floating):
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return True
        return bool(finite.min() == finite.max())
    return bool(flat.min() == flat.max())


def compare_all_outputs_to_cpu(
    vk_out: list[np.ndarray],
    cpu_out: list[np.ndarray],
) -> tuple[str, dict]:
    """Compare **every** output against the CPU oracle, with a non-triviality guard.

    Returns ``(outcome, facts)`` where outcome is ``COMPARISON_AGREE`` /
    ``COMPARISON_DISAGREE`` / ``COMPARISON_NOT_PERFORMED`` — a comparison outcome, never a
    verdict.  The verdict is derived downstream from this plus attribution (§10.0 amendment
    3: a comparison of tensors cannot, on its own, say ``MATCH``).

    The three outcomes are kept distinct on purpose:

    ``AGREE``
        Every output was compared, every one was within its justified tolerance, and every
        one carried information on **both** sides.
    ``DISAGREE``
        Some output exceeded its tolerance, or the two runs disagree in arity/shape/dtype.
    ``NOT_PERFORMED``
        Some output pair was degenerate, so the comparison over it is vacuous.  This is
        **absence of evidence, not evidence of agreement**, and collapsing it into either of
        the other two is the mistake that produced the reopened row.

    ``facts`` carries ``oracle_outputs_compared`` — and note that name.  The artifact this
    replaces carried ``outputs_compared: 65`` *listed among the oracle facts* while counting
    **cross-run** comparisons; it was read as sixty-five oracle comparisons and a
    ``max_abs_diff`` over one tensor was quoted beside it.  The two counts now have two names
    and cannot be read off one key.
    """
    facts: dict = {
        "oracle_outputs_total": len(vk_out),
        "oracle_outputs_compared": 0,
        "oracle_outputs_within_tolerance": 0,
        "oracle_outputs_degenerate": 0,
        "oracle_degenerate_indices": [],
        "oracle_failing_indices": [],
        "oracle_max_abs_diff_over_all_outputs": 0.0,
        "oracle_worst_output_index": None,
        "per_output": [],
    }

    if len(vk_out) != len(cpu_out):
        facts["oracle_arity_mismatch"] = {"vk": len(vk_out), "cpu": len(cpu_out)}
        return COMPARISON_DISAGREE, facts

    worst = -1.0
    for i, (v, c) in enumerate(zip(vk_out, cpu_out)):
        entry: dict = {"index": i, "shape": list(v.shape), "dtype": str(v.dtype)}

        if v.shape != c.shape or v.dtype != c.dtype:
            entry["status"] = "SHAPE_OR_DTYPE_MISMATCH"
            entry["cpu_shape"] = list(c.shape)
            entry["cpu_dtype"] = str(c.dtype)
            facts["oracle_failing_indices"].append(i)
            facts["per_output"].append(entry)
            continue

        vk_degenerate = _is_degenerate(v)
        cpu_degenerate = _is_degenerate(c)
        entry["vk_degenerate"] = vk_degenerate
        entry["cpu_degenerate"] = cpu_degenerate

        if vk_degenerate or cpu_degenerate:
            # Non-triviality guard, on BOTH sides.  Two constant tensors compare equal to
            # any tolerance, so this pair proves nothing whichever way it falls.
            entry["status"] = "DEGENERATE"
            facts["oracle_outputs_degenerate"] += 1
            facts["oracle_degenerate_indices"].append(i)
            facts["per_output"].append(entry)
            continue

        tol, why = tolerance_for_output(v)
        a = v.astype(np.float64)
        b = c.astype(np.float64)
        abs_diff = np.abs(a - b)
        max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
        denom = tol["atol"] + np.abs(b)
        max_rel = float(np.max(abs_diff / np.where(denom == 0, 1.0, denom))) if a.size else 0.0
        within = bool(np.allclose(a, b, rtol=tol["rtol"], atol=tol["atol"], equal_nan=True))

        entry.update(
            {
                "status": "WITHIN_TOLERANCE" if within else "OUTSIDE_TOLERANCE",
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "rtol": tol["rtol"],
                "atol": tol["atol"],
                "tolerance_justification": why,
            }
        )
        facts["oracle_outputs_compared"] += 1
        if within:
            facts["oracle_outputs_within_tolerance"] += 1
        else:
            facts["oracle_failing_indices"].append(i)
        if max_abs > worst:
            worst = max_abs
            facts["oracle_worst_output_index"] = i
        facts["per_output"].append(entry)

    facts["oracle_max_abs_diff_over_all_outputs"] = max(worst, 0.0)

    if facts["oracle_failing_indices"]:
        return COMPARISON_DISAGREE, facts
    if facts["oracle_outputs_degenerate"]:
        return COMPARISON_NOT_PERFORMED, facts
    if facts["oracle_outputs_compared"] != facts["oracle_outputs_total"]:
        return COMPARISON_NOT_PERFORMED, facts
    return COMPARISON_AGREE, facts


def run_session_n_times(
    sess: "ort.InferenceSession",
    feeds: dict[str, np.ndarray],
    n: int,
) -> list[list[np.ndarray]]:
    """Run *sess* with identical *feeds* exactly *n* times and return all output lists.

    The returned list has length *n*; element *i* is the output list for run *i+1*.

    WHY MULTI-RUN IS STRUCTURALLY REQUIRED
    =======================================
    ORT's memory-pattern planner does not engage on the first run of a session. From run 2
    onward it records tensor lifetimes and sub-divides the arena — handing back interior
    pointers derived by arithmetic on the allocator's return value (e.g. ``base + n``).
    Before that, the arena is freshly zeroed by the OS. This creates an arena-cleanliness
    boundary at the run-1/run-2 boundary:

    - **Run 1:** arena is clean (OS-zeroed). An unwritten output buffer shows zeros.
      A kernel that *computes* zeros is indistinguishable from a kernel that *never writes*.
    - **Run 2+:** arena is dirty (residues from run 1). An unwritten output buffer now
      shows the residue of whatever lived there before — usually a very different value.
      A kernel that correctly computes zeros will still show zeros on run 2. A kernel that
      never writes will show garbage.

    Tank's three-run experiment on Phi-3.5 confirmed this:
    - Outputs 1..64 (KV cache, fp16): bit-DIFFERENT between run 1 and runs 2/3 with
      identical feeds. Signature: nobody writes them, dirty arena shows residue.
    - Output 0 (logits, fp32): exactly 0.0 in all 32064 positions on runs 2 and 3 as
      well. The arena is dirty around it, yet it shows zeros. Something writes zeros.

    These are two different bugs with two different owners. A single-run gate cannot
    distinguish them. A multi-run gate distinguishes them immediately: if run1 and run2
    differ bitwise for a given output index, the output was *never written* on run1 either
    (it just happened to be in a clean arena). If run1 and run2 agree (both zero), the
    kernel writes zeros deliberately.

    MINIMUM N
    =========
    N ≥ 2 is required to observe the planner boundary. N ≥ 3 adds a confirmatory run
    that distinguishes planner-boundary effects from first-run fluke. Use N=3 for any
    gate that claims MATCH; use N=2 for cheaper signal checks.
    """
    return [sess.run(None, feeds) for _ in range(n)]


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


def is_vulkan_claimed(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """Return True iff VulkanExecutionProvider claimed at least one node of *model*.

    Non-asserting probe used by test_barrier_parity to skip (not fail) when an op is
    not yet Ready. Uses ORT profiling JSON (same mechanism as assert_vulkan_claims),
    with a broad exception catch so that:
      - A CPU fallback (not claimed) → returns False
      - An EP crash during session creation → returns False (conservative)
      - A genuine claim → returns True

    Note: Some ops crash ORT with profiling=True on Intel Vulkan (EP bug in Compile path
    for unimplemented ops). These crashes are caught here and return False, masking them
    as "not claimed". The EP-side crash is still visible in stderr. Route to Tank.
    """
    opts = _make_session_options(profiling=True, prefix="_vulkan_isclaimed_probe")
    try:
        sess = ort.InferenceSession(model, opts, providers=EP_PROVIDERS)
        sess.run(None, feeds)
        profile_path = sess.end_profiling()
    except Exception:
        return False

    try:
        with open(profile_path) as fh:
            events = json.load(fh)
    except Exception:
        return False
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    return any(
        e.get("cat") == "Node"
        and isinstance(e.get("args"), dict)
        and e["args"].get("provider") == EP_NAME
        for e in events
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

    NaN NOTE
    ========
    ``np.testing.assert_allclose`` treats NaN as not equal to itself (equal_nan=False by
    default in assert_allclose). If both arrays contain the same NaN at a position, this
    will raise. However, `assert_allclose` does NOT reduce via max|a-b| — it checks each
    element individually — so it does not produce the nan-contamination that ``max(abs(a-b))``
    does when NaN is present. The NaN hole (Tank's finding) affects callers who write their
    own ``max(abs(...))`` reduction, not this function directly.

    For bit-exact equality (including NaN-bit-identity), use ``outputs_bit_equal`` instead.
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


# ---------------------------------------------------------------------------
# EP provider placement guard — universal vacuous-pass prevention
# ---------------------------------------------------------------------------


def assert_ep_in_session(sess: "ort.InferenceSession") -> None:
    """Assert that EP_NAME is listed in *sess*.get_providers().

    ORT does NOT raise when a plugin EP fails to load or advertises zero devices.
    It silently falls back to CPUExecutionProvider and returns an empty provider list
    for the plugin.  Any test that compares a VulkanEP session against a CPU oracle
    WITHOUT calling this guard is comparing CPU-vs-CPU when the EP is absent — a
    guaranteed spurious pass.

    WHEN TO CALL
    ============
    Call this immediately after ``ort.InferenceSession(... providers=EP_PROVIDERS)``
    and before ``sess.run()`` in any test that:
      (a) compares the VulkanEP session output against a CPU oracle, OR
      (b) intends to assert that GPU dispatch actually occurred.

    Tests that only assert crash-absence (session loads, run returns output) do NOT
    need this guard — the session is useful even in pure CPU fallback for those purposes.

    NOTE ON ``assert_vulkan_claims``
    ================================
    ``assert_vulkan_claims`` (above) is more expensive: it runs a separate profiling
    session to detect which nodes ran on the EP.  Use ``assert_ep_in_session`` as a
    fast pre-check in tests that own the session; use ``assert_vulkan_claims`` when
    you need to verify device placement for a model passed in as bytes.

    COORDINATOR REFERENCE
    =====================
    The coordinator's ``phi35_vk_vs_cpu.py`` script includes this same guard as its
    "HARD GATE":

        used = vk_sess.get_providers()
        if EP_NAME not in used:
            print("FAIL ... Refusing to compare.")
            return 1

    That script discovered the all-zero logits bug *because* it had the guard; the
    earlier ``test_phi35_cpu_output_matches_between_sessions`` did not and so it
    passed vacuously while reporting nothing meaningful.
    """
    used = sess.get_providers()
    assert EP_NAME in used, (
        f"{EP_NAME} not in session.get_providers(): {used}.\n"
        "ORT fell back to CPU silently — the comparison would be CPU-vs-CPU (vacuous pass).\n"
        "Possible causes:\n"
        "  • ONNXRUNTIME_VULKAN_EP_LIB not set or file not found.\n"
        "  • ort.register_execution_provider_library was not called before session creation.\n"
        "  • The EP registered but enumerated zero Vulkan devices (no ICD, capability gate).\n"
        "Check conftest.py register_vulkan_ep fixture and require_vulkan fixture.\n"
        "\n"
        "NOTE: This guard catches load-time fallback only (get_providers() is fixed at\n"
        "session-create time). It does NOT catch run-time fallback, where ORT prints\n"
        "'EP_FAIL ... Falling back' during sess.run() and silently re-executes on CPU.\n"
        "Use assert_vulkan_executed_runtime() after sess.run() to close that gap."
    )


# ---------------------------------------------------------------------------
# Runtime-fallback guard — post-run Vulkan execution check
# ---------------------------------------------------------------------------


def count_vulkan_executions_from_profile(events: list[dict]) -> int:
    """Return the number of *fused-island executions* the Vulkan EP performed in *events*.

    R11 WARNING — THE NAME IS NOT THE DEFINITION
    ============================================
    The name says "executions".  A reader will hear "nodes".  It is neither: ORT emits one
    ``cat == "Node"`` profiling event per **fused island** execution, and one fused island
    can cover hundreds of graph nodes.  On Phi-3.5 today, **354 of 364 graph nodes execute
    inside a single island**, so a healthy run reports ``1``.  Read bare, that ``1`` looks
    like a catastrophe (``1 node ran on Vulkan``) when it is the best result the project has
    produced.  Never print this number without printing what it counts; use
    :func:`describe_vulkan_execution_count`.

    The number is therefore a *presence* signal, not a *volume* signal:

    - ``0``  — conclusive: the Vulkan EP ran nothing at all (runtime fallback).
    - ``>=1`` — the Vulkan EP participated.  How MUCH work it did is a different
      instrument (``claimed_count`` / ``dispatches_executed`` in the counters JSON).
      Do not derive coverage from this value; it will not scale with it.

    Counts entries where ``cat == "Node"`` and ``args["provider"] == EP_NAME``.
    This is the post-run complement to ``assert_ep_in_session``: where that function
    checks at session-create time (which is fixed before ``run()``), this function checks
    what actually executed during ``run()``.
    """
    return _verdict.tally_providers(events).get(EP_NAME, 0)


def describe_vulkan_execution_count(count: int) -> str:
    """Render a Guard D count together with its definition, for printing.

    R11: a measurement's name is not its definition, and the definition has to travel with
    the number or the next reader supplies their own.  Every artifact that quotes a Guard D
    count goes through this function so there is exactly one wording to keep correct.
    """
    if count == 0:
        return "0 fused-island executions — the Vulkan EP ran NOTHING (fallback)"
    plural = "" if count == 1 else "s"
    return (
        f"{count} fused-island execution{plural} — NOT {count} graph node{plural}; one "
        "island can cover hundreds of graph nodes (Phi-3.5: 354 of 364 nodes in one island, "
        "so 1 is healthy). Presence signal only; read coverage from the counters JSON."
    )


def assert_vulkan_executed_runtime(profile_path: "str | os.PathLike[str]") -> int:
    """Assert that at least one Node event in *profile_path* was executed by the Vulkan EP.

    This is Guard D for tests that own their session: ORT prints ``EP_FAIL ... Falling back``
    during ``sess.run()`` and retries on CPU without raising.  ``get_providers()`` remains
    non-empty (the EP registered successfully), so ``assert_ep_in_session`` cannot detect
    this.  Only the profiling trace, which records what actually ran, can close the gap.

    DESIGN.md §10.0.1 R10 / R7 note: a comparison gate without this guard is wired to
    the wrong signal — it reads CPU-vs-CPU and reports MATCH.  That reading is structurally
    indistinguishable from "VulkanEP executed and agreed with CPU", which is the reading the
    caller intends.

    WHAT THE COUNT MEANS
    ====================
    ORT records one ``Node`` profiling event per *fused island* execution, not per graph node.
    A single VulkanEP event may represent hundreds of graph nodes fused into one island.
    The count returned here is the number of fused-island executions observed — not the
    number of graph nodes dispatched.  Zero is the only value that is conclusively bad:
    it means the Vulkan EP ran nothing at all (runtime fallback confirmed).  Any value ≥ 1
    proves Vulkan participated; the graph-node count is a separate instrument.

    FAILURE MODE DISTINCTION
    ========================
    Two distinct failure modes must be separated because they require different actions:

    - ``AssertionError`` — *fallback detected*: the profiling trace was read successfully
      but zero VulkanEP Node events were found.  This is a real finding — the EP fell back
      at run time.  Route to Switch/Mouse (allocation or dispatch failure).

    - ``RuntimeError`` — *guard instrument broken*: the trace file could not be read, the
      JSON could not be parsed, or another infrastructure failure occurred.  This is an
      instrument failure, not a finding about the EP.  Do not route as an EP bug; fix the
      harness first.

    Parameters
    ----------
    profile_path:
        Path returned by ``sess.end_profiling()``.  The file is read and deleted here.

    Returns
    -------
    int
        Number of VulkanEP fused-island execution events (always ≥ 1 on success).

    Raises
    ------
    AssertionError
        Fallback detected — the EP ran zero fused islands at run time.
    RuntimeError
        Guard instrument broken — trace file unreadable or unparseable.  Concretely a
        :class:`_verdict.InstrumentError`, which subclasses ``RuntimeError`` so that
        callers written against the old signature keep working.
    """
    return attribution_from_profile(profile_path).assert_executed()


def attribution_from_profile(
    profile_path: "str | os.PathLike[str]",
) -> "_verdict.ExecutionAttribution":
    """Guard D's *observation*, without its assertion — §10.0 third amendment, clause 5.

    This is the function callers should use when they are about to emit a verdict: the
    observation becomes the verdict's constructor argument, so the guard and the verdict
    stop being separable.  ``assert_vulkan_executed_runtime`` is the same observation with
    the convenience assertion applied.
    """
    return _verdict.ExecutionAttribution.from_profile(profile_path)


# ---------------------------------------------------------------------------
# model_output_equivalence verdict — §9.1.3 / §10.0 (Morpheus ruling)
#
# The verdict is a RECORD, not a string, and it is DERIVED, not chosen.  The whole
# mechanism lives in ``_verdict.py``; these names are re-exported so existing call sites
# and test modules keep one import.  See ``_verdict.py``'s module docstring for the five
# binding clauses and the vocabulary table.
# ---------------------------------------------------------------------------

EQUIVALENCE_MATCH: str = _verdict.VERDICT_MATCH
EQUIVALENCE_DIVERGENT: str = _verdict.VERDICT_DIVERGENT
EQUIVALENCE_UNMEASURED: str = _verdict.VERDICT_UNMEASURED
#: A comparison was performed and this EP executed zero nodes.  NOT ``DIVERGENT``.
EQUIVALENCE_UNATTRIBUTED: str = _verdict.VERDICT_UNATTRIBUTED
#: The profile witness and the counters witness disagree about whether this EP ran.
EQUIVALENCE_SPLIT_FRAME: str = _verdict.VERDICT_SPLIT_FRAME
#: Every token a reader may meet in `model_output_equivalence`.  One vocabulary, not two:
#: mirrored in rust/src/counters.rs, rust/src/bin/epctl.rs and bench/admissible.py.
EQUIVALENCE_VERDICTS: tuple[str, ...] = _verdict.VERDICTS
#: JSON keys in the counters artifact: the token (string) and the full record (object).
EQUIVALENCE_KEY: str = _verdict.EQUIVALENCE_KEY
EQUIVALENCE_RECORD_KEY: str = _verdict.EQUIVALENCE_RECORD_KEY

COMPARISON_AGREE: str = _verdict.COMPARISON_AGREE
COMPARISON_DISAGREE: str = _verdict.COMPARISON_DISAGREE
COMPARISON_NOT_PERFORMED: str = _verdict.COMPARISON_NOT_PERFORMED

ExecutionAttribution = _verdict.ExecutionAttribution
EquivalenceVerdict = _verdict.EquivalenceVerdict
AttributedRunSeries = _verdict.AttributedRunSeries
InstrumentError = _verdict.InstrumentError
write_equivalence_record = _verdict.write_equivalence_record
read_equivalence_record = _verdict.read_equivalence_record
read_counters_dispatches = _verdict.read_counters_dispatches
read_counters_witness = _verdict.read_counters_witness
WITNESS_UNOBSERVABLE: str = _verdict.WITNESS_UNOBSERVABLE
WITNESS_AGREEMENT_AGREE: str = _verdict.WITNESS_AGREEMENT_AGREE
WITNESS_AGREEMENT_DISAGREE: str = _verdict.WITNESS_AGREEMENT_DISAGREE
WITNESS_AGREEMENT_UNOBSERVABLE: str = _verdict.WITNESS_AGREEMENT_UNOBSERVABLE
find_fatal_log_lines = _verdict.find_fatal_log_lines


def write_unmeasured_verdict(
    counters_path: "str | os.PathLike[str] | None",
    reason: str,
    *,
    device_index: str = "",
    artifact: str = "",
) -> None:
    """Best-effort ``UNMEASURED`` write for the paths that bail out before comparing.

    Swallows write failures deliberately: these call sites are already failing for a
    different, better-diagnosed reason, and an instrument outage in the *recorder* must
    not overwrite the finding that is on its way up the stack (R13 — an instrument error
    never counts as a detection, and it must not erase one either).
    """
    if not counters_path:
        return
    try:
        _verdict.write_equivalence_record(
            counters_path,
            _verdict.EquivalenceVerdict.unmeasured(
                reason=reason, artifact=artifact, device_index=device_index
            ),
        )
    except Exception:  # noqa: BLE001 - see docstring
        pass


# ---------------------------------------------------------------------------
# Barrier-backend parity helper — used by test_barrier_parity.py
# ---------------------------------------------------------------------------


def run_with_backend(
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    force_legacy: bool = False,
) -> tuple[list[np.ndarray], str]:
    """Run *model* on the Vulkan EP with the specified barrier backend.

    Parameters
    ----------
    model :
        Serialised ONNX model bytes.
    feeds :
        Input name → ndarray mapping.
    force_legacy :
        If ``True``, adds ``ep.force_legacy_barriers=1`` to the session options, forcing
        ``Barriers::Legacy`` (``vkCmdPipelineBarrier``) even on a device that supports
        ``synchronization2``.  If ``False``, the device's natural selection from
        ``Barriers::select`` is used (sync2 if available, legacy otherwise).

    Returns
    -------
    (outputs, active_backend) where *active_backend* is one of:
        ``"sync2"``   — ``Sync2Backend`` was selected (``VK_KHR_synchronization2`` path).
        ``"legacy"``  — ``LegacyBackend`` was selected (``vkCmdPipelineBarrier`` path).
        ``"unknown"`` — EP has not yet implemented ``ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE``
                        (see TODO below); probe file was not written.

    Implementation contract — what Switch must implement
    ----------------------------------------------------
    When the environment variable ``ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE=<absolute-path>``
    is set at session creation time, the EP must:

    1. Write exactly ``"sync2"`` or ``"legacy"`` (no trailing newline, UTF-8) to ``<path>``
       **before ``CreateSession`` returns**, during ``Barriers::select`` inside ``Device::new``.
    2. Overwrite any previous content (create-or-truncate semantics).
    3. If the file cannot be written (permissions, etc.), log a warning and continue —
       the probe is test-only and must not affect correctness.

    This is the simplest IPC mechanism that requires zero ORT API changes.  The path is
    chosen by the test harness (unique per process, project-relative) to avoid /tmp and to
    ensure cleanup regardless of test outcome.

    The mechanism ties to DESIGN.md §7.5 item 5: "Trinity runs the differential suite twice
    per lane — once default, once forced."  Without the probe, a ``force_legacy=True`` session
    that silently ignores the option would make the parity test a false green.
    """
    # Project-relative probe file, unique per process (safe for pytest-xdist).
    # Lives in tests/ops/ alongside the test files; always cleaned up in finally.
    probe_path = Path(__file__).parent / f"_barrier_probe_{os.getpid()}.txt"
    if probe_path.exists():
        try:
            probe_path.unlink()
        except OSError:
            pass

    old_probe = os.environ.pop("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE", None)
    os.environ["ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE"] = str(probe_path.absolute())
    try:
        opts = _make_session_options()
        if force_legacy:
            opts.add_session_config_entry("ep.force_legacy_barriers", "1")
        sess = ort.InferenceSession(model, opts, providers=EP_PROVIDERS)
        outputs = sess.run(None, feeds)
    finally:
        # Always restore the env var, whether the session succeeded or not.
        if old_probe is not None:
            os.environ["ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE"] = old_probe
        else:
            os.environ.pop("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE", None)

    active_backend = "unknown"
    if probe_path.exists():
        try:
            active_backend = probe_path.read_text("utf-8").strip()
        finally:
            try:
                probe_path.unlink()
            except OSError:
                pass

    return outputs, active_backend


# ---------------------------------------------------------------------------
# MatMulNBits model builder
# ---------------------------------------------------------------------------


def make_matmulnbits_model(
    K: int,
    N: int,
    *,
    bits: int = 4,
    block_size: int = 32,
    accuracy_level: int = MATMULNBITS_ORACLE_ACCURACY_LEVEL,
    activation_dtype: ir.DataType = ir.DataType.FLOAT,
    with_zero_points: bool = True,
    rows: int = 1,
    symbolic_batch: bool = False,
) -> tuple[bytes, dict[str, np.ndarray]]:
    """Build a minimal MatMulNBits ONNX graph and return (model_bytes, feeds).

    The packed weights and zero-points are random but deterministically seeded so that
    test results are reproducible across runs and machines.

    Parameters
    ----------
    K, N :
        Input/output feature dimensions (activations shape is [1, K]).
    bits :
        Quantization bit-width (4 or 8). Default 4.
    block_size :
        Quantization block size. Must be a multiple of 32.
    accuracy_level :
        CPU EP oracle accumulator precision (0=default, 1=fp32, 2=fp16, 3=bf16, 4=int8).
        Pin this to MATMULNBITS_ORACLE_ACCURACY_LEVEL in oracle comparisons.
    activation_dtype :
        ORT data type for the activations input (FLOAT or FLOAT16).
    with_zero_points :
        Emit the 4-input asymmetric form. `False` gives the 3-input symmetric-RTN form, which
        is what every one of Phi-3.5's 161 MatMulNBits nodes actually is — the implied zero
        point is then `1 << (bits-1)`, a fact derived from the CPU EP rather than the schema
        prose (OP_COVERAGE.md §8.1.1).
    rows :
        Number of activation rows. 1 is decode; >1 is prefill, which the GEMV handles by
        running one workgroup per output element rather than by tiling.
    symbolic_batch :
        If `True`, declare the leading activation dimension as a symbolic string ("batch")
        rather than the concrete `rows` integer. This triggers ORT's dynamic-shape path,
        which exercises the session-layer runtime-extents dispatch in `session.rs`. The feeds
        are unchanged (rows-row activation). Use this to catch bugs where a kernel computes
        correctly with static shapes but produces zeros when the leading dim is symbolic —
        the exact failure mode found in the 2026-07-30 all-zero-logits investigation.

    Returns
    -------
    (model_bytes, feeds) — model_bytes is a serialized ONNX proto; feeds contains only the
    activation array (packed weights are baked in as initializers).
    """
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    from onnx import TensorProto as tp

    rng = np.random.default_rng(42)

    # --- Packed weight tensor ---
    # Shape: [N, ceil(K / block_size), block_size * bits / 8]
    blocks_per_col = -(-K // block_size)  # ceil division
    packed_bytes = block_size * bits // 8
    packed_shape = [N, blocks_per_col, packed_bytes]
    packed_data = rng.integers(0, 256, size=packed_shape, dtype=np.uint8)

    # --- Activation (the only dynamic input) ---
    np_dtype = np.float32 if activation_dtype == ir.DataType.FLOAT else np.float16
    act = rng.standard_normal((rows, K)).astype(np_dtype)
    feeds = {"X": act}

    # --- Scale tensor ---
    # Shape: [N * blocks_per_col]. MatMulNBits binds `A`, `scales` and `Y` to the SAME type
    # parameter T1, so leaving the scales fp32 under fp16 activations makes ORT reject the model
    # outright ("T1 bound to different types"). Found while landing the fp16 GEMV: Phi-3.5's 161
    # MatMulNBits nodes are all fp16, so the fp16 path is the one that matters, not the spare.
    scale_shape = [N * blocks_per_col]
    scale_data = rng.uniform(0.001, 0.1, size=scale_shape).astype(np_dtype)

    # --- Zero-point tensor (optional; packed at `bits`, one run per column, byte-padded) ---
    zp_bytes_per_col = -(-(blocks_per_col * bits) // 8)
    zp_shape = [N, zp_bytes_per_col]
    zp_data = rng.integers(0, 256, size=zp_shape, dtype=np.uint8)

    # Map ir.DataType → ONNX TensorProto type for the graph input declaration.
    _dtype_to_tp = {ir.DataType.FLOAT: tp.FLOAT, ir.DataType.FLOAT16: tp.FLOAT16}
    act_tp_dtype = _dtype_to_tp[activation_dtype]

    # Build initializers.
    b_tensor = onh.from_array(packed_data, name="B")
    scale_tensor = onh.from_array(scale_data, name="scale")
    zp_tensor = onh.from_array(zp_data, name="zero_points")

    # Build node.
    node = oh.make_node(
        "MatMulNBits",
        inputs=["X", "B", "scale", "zero_points"] if with_zero_points else ["X", "B", "scale"],
        outputs=["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=bits,
        block_size=block_size,
        accuracy_level=accuracy_level,
    )

    # Build graph.
    batch_dim = "batch" if symbolic_batch else rows
    x_info = oh.make_tensor_value_info("X", act_tp_dtype, [batch_dim, K])
    y_info = oh.make_tensor_value_info("Y", act_tp_dtype, [batch_dim, N])

    graph = oh.make_graph(
        [node],
        "matmulnbits_test",
        [x_info],
        [y_info],
        initializer=[b_tensor, scale_tensor, zp_tensor]
        if with_zero_points
        else [b_tensor, scale_tensor],
    )
    model = oh.make_model(
        graph,
        opset_imports=[
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 8
    return model.SerializeToString(), feeds


# ---------------------------------------------------------------------------
# Per-layer intermediate capture — fault localisation for multi-layer models
# ---------------------------------------------------------------------------


def with_captured_outputs(model_bytes: bytes, extra_output_names: list[str]) -> bytes:
    """Return a copy of *model_bytes* with *extra_output_names* added as graph outputs.

    Use this to expose intermediate activations as outputs so that compare_layers() can
    check each layer independently instead of comparing only the final logits.

    Rationale (Morpheus C6, DESIGN.md §9.1): a broken kernel in a deep model can maintain
    correct top-1 token accuracy for many steps before divergence becomes visible. Per-layer
    capture makes a fault localisable in minutes rather than days.

    Only adds outputs that are not already graph outputs; silently skips unknown names
    (so callers can pass a superset list and let the graph answer what it has).

    Parameters
    ----------
    model_bytes :
        Serialised ONNX model bytes (not mutated; a modified copy is returned).
    extra_output_names :
        Names of internal nodes whose outputs should be exposed. Each name must match a
        value_info entry or a node output in the model.

    Returns
    -------
    Modified model bytes with the requested outputs appended to the graph's output list.
    """
    import onnx

    model = onnx.load_from_string(model_bytes)
    graph = model.graph

    existing_outputs = {vi.name for vi in graph.output}
    all_values: dict[str, onnx.TypeProto] = {}

    # Collect type information from value_info, initializers, and node outputs.
    for vi in graph.value_info:
        all_values[vi.name] = vi.type
    for init in graph.initializer:
        all_values.setdefault(init.name, onnx.TypeProto())
    for node in graph.node:
        for out in node.output:
            all_values.setdefault(out, onnx.TypeProto())

    for name in extra_output_names:
        if name in existing_outputs:
            continue
        tp = onnx.TypeProto()
        if name in all_values:
            tp.CopyFrom(all_values[name])
        vi = onnx.ValueInfoProto()
        vi.name = name
        vi.type.CopyFrom(tp)
        graph.output.append(vi)

    return model.SerializeToString()


def compare_layers(
    model_bytes: bytes,
    feeds: dict[str, np.ndarray],
    layer_names: list[str],
) -> list[dict]:
    """Run *model_bytes* on both the Vulkan EP and CPU EP, returning per-layer diffs.

    This is the primary fault-localisation tool: pass in the names of intermediate output
    nodes (obtained from model inspection or with_captured_outputs) and get back a sorted
    list showing which layers diverge most.

    Parameters
    ----------
    model_bytes :
        Serialised ONNX model bytes.  The model should already have the desired intermediate
        outputs exposed (e.g., via with_captured_outputs).
    feeds :
        Input feeds.
    layer_names :
        Intermediate output names to compare.  Names that are not present in the model
        output list are silently ignored.

    Returns
    -------
    A list of dicts (one per layer), sorted by max_abs_diff descending:
        {
            "name": str,          # layer name
            "max_abs_diff": float,
            "mean_abs_diff": float,
            "rtol_max": float,    # max(abs(a-b) / (atol + abs(b))) for elementwise comparison
        }
    The list is sorted by max_abs_diff descending so the worst-diverging layer is first.
    """
    expanded = with_captured_outputs(model_bytes, layer_names)

    vulkan_outputs = run_vulkan(expanded, feeds)
    cpu_outputs = run_cpu(expanded, feeds)

    # The graph outputs are the original outputs + layer_names (in that order).
    import onnx
    model = onnx.load_from_string(expanded)
    output_names = [vi.name for vi in model.graph.output]

    results = []
    for name, vulkan_val, cpu_val in zip(output_names, vulkan_outputs, cpu_outputs):
        if name not in layer_names:
            continue
        v = np.asarray(vulkan_val, dtype=float)
        c = np.asarray(cpu_val, dtype=float)
        abs_diff = np.abs(v - c)
        denom = 1e-8 + np.abs(c)
        # NaN-SAFE reductions: np.nanmax/nanmean skip NaN elements rather than propagating
        # them. Background: numpy returns NaN for max(a-b) whenever EITHER array contains
        # a NaN at the same position, even if both values are bit-identical NaN. This was
        # observed by Tank when comparing fp16 KV-cache outputs near the representable limit.
        # nanmax reports the worst finite diff; nan_count separately tells the caller how
        # many positions have NaN in either output. See also outputs_bit_equal().
        nan_count = int(np.sum(np.isnan(abs_diff)))
        results.append({
            "name": name,
            "max_abs_diff": float(np.nanmax(abs_diff)) if abs_diff.size else 0.0,
            "mean_abs_diff": float(np.nanmean(abs_diff)) if abs_diff.size else 0.0,
            "rtol_max": float(np.nanmax(abs_diff / denom)) if abs_diff.size else 0.0,
            "nan_count": nan_count,
        })

    results.sort(key=lambda d: d["max_abs_diff"], reverse=True)
    return results
