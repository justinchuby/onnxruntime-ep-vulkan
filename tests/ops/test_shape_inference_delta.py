"""Shape-inference coverage delta measurement.

PURPOSE
=======
Measures how many nodes previously declined due to missing output-shape annotations
can be claimed after applying ``onnx_shape_inference.infer_symbolic_shapes`` as a
preprocessing step.

This is the **first concrete coverage measurement** this project can make, and it is
measurable without a single shader executing -- only EP claim-or-decline decisions
matter here, not numerical correctness.

WHAT THE DELTA MEANS
====================
The coordinator (Morpheus section 8.6) sequenced this as coverage work, not harness polish.
But the number must be interpreted carefully:

  "X ops claimed without preprocessing"
  vs
  "Y ops additionally claimed after preprocessing"

are different guarantees. The Y-ops are claimable **only in a pipeline that runs
shape inference first**. A raw exported model without shape annotations will only
see this coverage gain if the user's inference pipeline includes the preprocessing
step -- or if the serving framework runs it automatically.

THREE-CLASS SHAPE TAXONOMY (claim.rs, DESIGN.md §8.8)
======================================================
Switch's runtime-extents work (merged 2026-07-29, ENGINE_ACCEPTS_RUNTIME_EXTENTS = true)
introduced a three-way split among shapes the EP previously bundled as "dynamic":

  CLASS 1 — rank-known, extents-symbolic (DeclineCode::DynamicShape):
    The rank is fully determined by the op spec from input ranks; only the extent
    values are unknown at compile time.  Switch's fix makes these claimable because
    extents arrive as push-constants at Compute time.  All _DELTA_CASES with concrete
    input shapes and no output annotation fall here — the rank is inferable from
    the op semantics (elementwise -> output rank = input rank; broadcast -> max rank).

  CLASS 2 — rank-unknown (DeclineCode::UnknownRank):
    Even the rank is unknown, so descriptor layout cannot be determined.  Not claimable
    even with ENGINE_ACCEPTS_RUNTIME_EXTENTS.  Examples: Reshape (output rank set by
    input values), Squeeze/Unsqueeze with data-dependent axes.

  CLASS 3 — data-dependent shape (DeclineCode::DataDependentShape):
    Output shape depends on input *values*, not just shapes.  Permanently unclaimable
    under one-command-buffer-per-subgraph.  Examples: NonZero, Compress.

HOW THIS CHANGED test_uninferred_shape_ep_declines
===================================================
Before Switch's fix: all elementwise ops (Relu, Add, Sub, ...) without output-shape
annotation were CLASS 1 by declaration, but ENGINE_ACCEPTS_RUNTIME_EXTENTS was false,
so they declined anyway.  ``test_uninferred_shape_ep_declines`` correctly observed declines.

After Switch's fix: ENGINE_ACCEPTS_RUNTIME_EXTENTS = true.  CLASS 1 ops now claim even
without output annotation, because the EP infers the rank from input shapes + op spec
and treats symbolic extents as runtime parameters.  The old test's premise became false.

The false premise was baked into the test's *name*, not its docstring, making it
invisible to docstring-grep audit.  ``test_uninferred_shape_ep_declines`` was replaced
by ``test_ep_claims_without_output_annotation`` when the premise was corrected (2026-07-30).

AUDIT METHOD EXTENSION
======================
The previous audit (2026-07-30) keyed on docstring phrases such as "with N claimed
nodes", "all X are declined", "falls back entirely to".  That audit caught six tests in
test_phi35.py but missed this file entirely.

Tests whose only statement of a claim/decline premise is their function name are
invisible to docstring grep.  The next release-time audit must also run:

  grep -r "def test_.*_ep_declines\\|def test_.*_ep_claims" tests/ops/
  grep -r "assert_vulkan_does_not_claim\\|assert_vulkan_claims" tests/ops/

For each hit, verify the claim/decline expectation against the current state of
claim.rs and ENGINE_ACCEPTS_RUNTIME_EXTENTS.  This catches the shape of defect found
here: a test that announces its premise in its identifier, not its body.

EXCEPTION — MAX
===============
Max (ONNX variadic-input elementwise max) fails test_inferred_shape_ep_claims even
after apply_shape_inference produces a concrete output shape.  It also declines without
annotation.  Its claim predicate is declining for a reason not yet determined; it is
excluded from test_ep_claims_without_output_annotation and marked xfail(strict=True)
in the inferred test.  See CANNOT-CLASSIFY note in the test body.

HOW TO RUN
==========
  # Measure only (no Vulkan EP needed):
  pytest tests/ops/test_shape_inference_delta.py::test_shape_inference_increases_resolved_count
  pytest tests/ops/test_shape_inference_delta.py::test_inferred_model_cpu_correctness

  # Full EP measurement (needs ONNXRUNTIME_VULKAN_EP_LIB set):
  $env:ONNXRUNTIME_VULKAN_EP_LIB = "rust\\target\\release\\onnxruntime_vulkan_ep.dll"
  pytest tests/ops/test_shape_inference_delta.py -v

DETERMINISM
===========
All random inputs are constructed from a fixed seed (separate from test_op_table.py's
_RNG to avoid coupling). Module-level construction only; never inside a test function.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

# ---------------------------------------------------------------------------
# Deterministic RNG -- independent seed from test_op_table.py (seed 42)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(137)

_S8x16 = (8, 16)
_S4x12 = (4, 12)
_S3x4x5 = (3, 4, 5)


def _f32(shape: tuple) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32)


def _f32_pos(shape: tuple) -> np.ndarray:
    return (np.abs(_RNG.standard_normal(shape)) + 1e-3).astype(np.float32)


# ---------------------------------------------------------------------------
# InputDesc -- lightweight descriptor for graph inputs.
#
# We store descriptors, NOT live ir.Value objects, in _DELTA_CASES.
# ir.Graph owns its input Values after construction; reusing the same
# ir.Value across multiple make_model_dynamic_output calls raises:
#   "Value is already owned by a different graph."
# Descriptors are reconstructed into fresh ir.Value objects inside each
# builder call, mirroring the pattern in test_op_table.py.
# ---------------------------------------------------------------------------

class InputDesc(NamedTuple):
    name: str
    shape: tuple
    dtype: ir.DataType = ir.DataType.FLOAT

    def fresh(self) -> ir.Value:
        return ir.Value(
            name=self.name,
            type=ir.TensorType(self.dtype),
            shape=ir.Shape(list(self.shape)),
        )


def _inp(name: str, shape: tuple, dtype: ir.DataType = ir.DataType.FLOAT) -> InputDesc:
    return InputDesc(name=name, shape=shape, dtype=dtype)


# ---------------------------------------------------------------------------
# DeltaCase -- a single-op model whose output shape is intentionally missing.
# ---------------------------------------------------------------------------

class DeltaCase(NamedTuple):
    id: str
    op: str
    inputs: list
    feeds: dict
    tol: dict
    domain: str = ""
    attrs: dict = {}


_DELTA_CASES = [
    # --- binary elementwise: output shape = broadcast(a.shape, b.shape) ---
    DeltaCase(
        id="Add-fp32-dyn", op="Add",
        inputs=[_inp("a", _S8x16), _inp("b", _S8x16)],
        feeds={"a": _f32(_S8x16), "b": _f32(_S8x16)},
        tol=dict(m.FP32_ELEMENTWISE),
    ),
    DeltaCase(
        id="Sub-fp32-dyn", op="Sub",
        inputs=[_inp("a", _S8x16), _inp("b", _S8x16)],
        feeds={"a": _f32(_S8x16), "b": _f32(_S8x16)},
        tol=dict(m.FP32_ELEMENTWISE),
    ),
    DeltaCase(
        id="Mul-fp32-dyn", op="Mul",
        inputs=[_inp("a", _S8x16), _inp("b", _S8x16)],
        feeds={"a": _f32(_S8x16), "b": _f32(_S8x16)},
        tol=dict(m.FP32_ELEMENTWISE),
    ),
    DeltaCase(
        id="Div-fp32-dyn", op="Div",
        inputs=[_inp("a", _S8x16), _inp("b", _S8x16)],
        feeds={"a": _f32(_S8x16), "b": _f32_pos(_S8x16)},
        tol=dict(m.FP32_ELEMENTWISE),
    ),
    DeltaCase(
        id="Max-fp32-dyn", op="Max",
        inputs=[_inp("a", _S8x16), _inp("b", _S8x16)],
        feeds={"a": _f32(_S8x16), "b": _f32(_S8x16)},
        tol=dict(m.FP32_ELEMENTWISE),
    ),
    # --- unary elementwise: output shape = input shape ---
    DeltaCase(
        id="Relu-fp32-dyn", op="Relu",
        inputs=[_inp("x", _S8x16)],
        feeds={"x": _f32(_S8x16)},
        tol=dict(m.FP32_ACTIVATION),
    ),
    DeltaCase(
        id="Neg-fp32-dyn", op="Neg",
        inputs=[_inp("x", _S8x16)],
        feeds={"x": _f32(_S8x16)},
        tol=dict(m.FP32_ELEMENTWISE),
    ),
    DeltaCase(
        id="Abs-fp32-dyn", op="Abs",
        inputs=[_inp("x", _S8x16)],
        feeds={"x": _f32(_S8x16)},
        tol=dict(m.FP32_ELEMENTWISE),
    ),
    DeltaCase(
        id="Exp-fp32-dyn", op="Exp",
        inputs=[_inp("x", _S8x16)],
        feeds={"x": _f32(_S8x16)},
        tol=dict(m.FP32_TRANSCENDENTAL),
    ),
    DeltaCase(
        id="Log-fp32-dyn", op="Log",
        inputs=[_inp("x", _S8x16)],
        feeds={"x": _f32_pos(_S8x16)},
        tol=dict(m.FP32_TRANSCENDENTAL),
    ),
    DeltaCase(
        id="Sqrt-fp32-dyn", op="Sqrt",
        inputs=[_inp("x", _S8x16)],
        feeds={"x": _f32_pos(_S8x16)},
        tol=dict(m.FP32_TRANSCENDENTAL),
    ),
    DeltaCase(
        id="Sigmoid-fp32-dyn", op="Sigmoid",
        inputs=[_inp("x", _S8x16)],
        feeds={"x": _f32(_S8x16)},
        tol=dict(m.FP32_TRANSCENDENTAL),
    ),
    DeltaCase(
        id="Tanh-fp32-dyn", op="Tanh",
        inputs=[_inp("x", _S4x12)],
        feeds={"x": _f32(_S4x12)},
        tol=dict(m.FP32_TRANSCENDENTAL),
    ),
    # --- rank-3 (tests that inference propagates rank, not just 2-D) ---
    DeltaCase(
        id="Relu-fp32-dyn-3d", op="Relu",
        inputs=[_inp("x", _S3x4x5)],
        feeds={"x": _f32(_S3x4x5)},
        tol=dict(m.FP32_ACTIVATION),
    ),
    DeltaCase(
        id="Add-fp32-dyn-3d", op="Add",
        inputs=[_inp("a", _S3x4x5), _inp("b", _S3x4x5)],
        feeds={"a": _f32(_S3x4x5), "b": _f32(_S3x4x5)},
        tol=dict(m.FP32_ELEMENTWISE),
    ),
]

_EXPECTED_DELTA = len(_DELTA_CASES)


def _build_dynamic_model(case: DeltaCase) -> bytes:
    """Build the dynamic-output model for a case, creating fresh ir.Value inputs."""
    fresh_inputs = [desc.fresh() for desc in case.inputs]
    return m.make_model_dynamic_output(
        case.op,
        fresh_inputs,
        domain=case.domain,
        attributes=case.attrs,
        output_dtype=ir.DataType.FLOAT,
    )


# ---------------------------------------------------------------------------
# Shape-resolution measurement (pure Python, no EP, no Vulkan needed)
# ---------------------------------------------------------------------------


def _count_resolved(cases: list, *, inferred: bool) -> int:
    """Count cases whose output shape is non-None after optional inference.

    A "resolved" output has every dimension as a concrete integer (not None, not symbolic).
    This is a fast proxy for coverage gain.
    """
    import onnx as _onnx

    resolved = 0
    for case in cases:
        model_bytes = _build_dynamic_model(case)
        if inferred:
            model_bytes = m.apply_shape_inference(model_bytes)
        ir_model = ir.from_proto(_onnx.ModelProto.FromString(model_bytes))
        for output in ir_model.graph.outputs:
            shape = output.shape
            if shape is not None and all(isinstance(d, int) for d in shape):
                resolved += 1
                break
    return resolved


class _DeltaReport(NamedTuple):
    before: int
    after: int

    @property
    def delta(self) -> int:
        return self.after - self.before

    def __str__(self) -> str:
        return (
            "\nShape-inference coverage delta (shape-resolution proxy):\n"
            f"  Without preprocessing:       {self.before}/{len(_DELTA_CASES)} ops have resolved output shapes\n"
            f"  After apply_shape_inference: {self.after}/{len(_DELTA_CASES)} ops have resolved output shapes\n"
            f"  Delta: +{self.delta} ops become shape-resolvable via preprocessing\n"
            f"\n  Claim caveat: these {self.delta} ops are claimable ONLY in a pipeline\n"
            "  that runs onnx_shape_inference preprocessing. Without it, the EP sees\n"
            "  dynamic output shapes and declines them.\n"
            "\n  (EP-based delta requires ONNXRUNTIME_VULKAN_EP_LIB -- see test_inferred_shape_ep_claims)"
        )


_REPORT = _DeltaReport(
    before=_count_resolved(_DELTA_CASES, inferred=False),
    after=_count_resolved(_DELTA_CASES, inferred=True),
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shape_inference_increases_resolved_count() -> None:
    """ALL delta cases must gain concrete shapes after apply_shape_inference.

    Pure-Python check -- no EP, no Vulkan. Verifies:
      1. Baseline: output shapes are None without inference
         (make_model_dynamic_output leaves them unspecified).
      2. After inference: all outputs have concrete integer shapes.
      3. Delta = len(_DELTA_CASES) (every designed case is inferable).
    """
    print(_REPORT)
    assert _REPORT.before == 0, (
        f"Expected all {len(_DELTA_CASES)} cases to have UNKNOWN output shapes before inference "
        f"(make_model_dynamic_output leaves them None), but {_REPORT.before} already had shapes. "
        "make_model_dynamic_output must produce shape=None outputs."
    )
    assert _REPORT.after == _EXPECTED_DELTA, (
        f"Expected all {_EXPECTED_DELTA} cases to gain concrete shapes after inference, "
        f"but only {_REPORT.after} did. "
        "Check onnx_shape_inference supports these op types."
    )
    assert _REPORT.delta == _EXPECTED_DELTA, (
        f"Delta should be {_EXPECTED_DELTA} but got {_REPORT.delta}."
    )


def test_inferred_model_cpu_correctness() -> None:
    """CPU EP correctness is preserved after apply_shape_inference.

    After shape inference, the model is semantically identical -- the only change
    is that output shapes are now annotated. ORT CPU EP must produce the same
    outputs on both the original and the inferred model.

    No Vulkan EP needed.
    """
    for case in _DELTA_CASES:
        original_bytes = _build_dynamic_model(case)
        inferred_bytes = m.apply_shape_inference(original_bytes)
        original_out = m.run_cpu(original_bytes, case.feeds)
        inferred_out = m.run_cpu(inferred_bytes, case.feeds)
        assert len(original_out) == len(inferred_out), (
            f"{case.id}: output count changed after inference"
        )
        for idx, (orig, inf) in enumerate(zip(original_out, inferred_out)):
            np.testing.assert_array_equal(
                orig, inf,
                err_msg=(
                    f"{case.id}: output[{idx}] differs between original and inferred model "
                    "-- apply_shape_inference must not alter semantics."
                ),
            )


@pytest.mark.parametrize("case", _DELTA_CASES, ids=[c.id for c in _DELTA_CASES])
def test_inferred_shape_ep_claims(case, require_vulkan):
    """After apply_shape_inference, the Vulkan EP must claim the node.

    EP-based delta measurement. For each case:
      1. Build model with dynamic output shape (no shape annotation).
      2. Apply shape inference -> concrete output shape.
      3. Assert EP claims the node (vacuous-pass guard).
      4. Assert output matches CPU EP.

    SKIPS if ONNXRUNTIME_VULKAN_EP_LIB is not set.

    CANNOT-CLASSIFY: Max-fp32-dyn declines even after shape inference annotates a
    concrete output shape.  The Max claim predicate is declining for an undetermined
    reason — neither dtype, rank, arity, nor shape annotation explains it given the
    concrete inputs used here.  Marked xfail(strict=True) so it flips loudly when the
    predicate is fixed, rather than training people to ignore a permanently-red test.
    """
    if case.op == "Max":
        pytest.xfail(
            "Max claim predicate declines even with annotated output shape — "
            "root cause undetermined (not dtype, rank, arity, or missing annotation). "
            "Remove this xfail when the Max predicate is fixed and test_inferred_shape_ep_claims"
            "[Max-fp32-dyn] goes green."
        )
    inferred_bytes = m.apply_shape_inference(_build_dynamic_model(case))
    m.check(inferred_bytes, case.feeds, **case.tol)


# Ops that ENGINE_ACCEPTS_RUNTIME_EXTENTS = true makes claimable without output annotation.
# Before Switch's runtime-extents fix, all of these declined when output shape was absent.
# After the fix, all CLASS-1 elementwise ops claim regardless of output annotation: the
# EP infers output rank from input ranks + op semantics and treats extents as runtime params.
#
# Exception: Max is excluded — it declines both with and without annotation (CANNOT-CLASSIFY).
# Exception: Add was already in this set before Switch's fix (was the first discovered case).
#
# Updated when claim predicates change; must remain in sync with OP_COVERAGE.md.
_CLAIMS_WITHOUT_OUTPUT_ANNOTATION: frozenset[str] = frozenset(
    # All elementwise ops in _DELTA_CASES except Max (which declines for unknown reason).
    {"Add", "Sub", "Mul", "Div", "Relu", "Neg", "Abs", "Exp", "Log", "Sqrt", "Sigmoid", "Tanh"}
)

# Ops that decline without output annotation even after Switch's runtime-extents fix.
# A test that expects DECLINE should key on this set.
_DECLINES_WITHOUT_OUTPUT_ANNOTATION: frozenset[str] = frozenset({"Max"})


@pytest.mark.parametrize("case", _DELTA_CASES, ids=[c.id for c in _DELTA_CASES])
def test_ep_claims_without_output_annotation(case, require_vulkan):
    """After Switch's ENGINE_ACCEPTS_RUNTIME_EXTENTS fix, CLASS-1 ops claim without annotation.

    What changed (2026-07-29):
      Before: EP declined all nodes with missing output-shape annotation, regardless of
        whether the rank was determinable from inputs.  This was the premise of the old
        test_uninferred_shape_ep_declines, which is now replaced by this test.
      After: ENGINE_ACCEPTS_RUNTIME_EXTENTS = true.  For CLASS-1 ops (rank-known from
        inputs, extents claimable at Compute time), the EP infers output rank from input
        rank + op spec and claims the node.  Output shapes are pushed as runtime constants
        rather than baked into the pipeline at compile time.

    Three-class taxonomy (see module docstring):
      CLASS 1 (this test): rank-known → claimable.  All binary/unary elementwise ops with
        concrete input shapes fall here.  The EP does not need output annotation.
      CLASS 2: rank-unknown → not claimable.  Not in _DELTA_CASES.
      CLASS 3: data-dependent → permanently unclaimable.  Not in _DELTA_CASES.

    SKIPS: Max (CANNOT-CLASSIFY: declines even with annotated output; excluded from this
      test and marked xfail in test_inferred_shape_ep_claims).
    SKIPS if ONNXRUNTIME_VULKAN_EP_LIB is not set.

    R9 instrument: this test goes red if ENGINE_ACCEPTS_RUNTIME_EXTENTS is set back to
    false or if the elementwise claim predicate adds an output-annotation requirement.
    The claim the test makes ("CLASS-1 ops claim without annotation") would go silent if
    these tests were skipped or vacuously passing.
    """
    if case.op in _DECLINES_WITHOUT_OUTPUT_ANNOTATION:
        pytest.skip(
            f"{case.id}: {case.op} is CANNOT-CLASSIFY — declines without output annotation "
            f"even after Switch's ENGINE_ACCEPTS_RUNTIME_EXTENTS fix.  Excluded from this "
            f"test; see test_inferred_shape_ep_claims[{case.id}] (xfail) for tracking."
        )
    model_bytes = _build_dynamic_model(case)
    m.check(model_bytes, case.feeds, **case.tol)
