"""Vulkan EP op-correctness tests — edge-case and structural coverage.

ADDING A NEW OP? Go to ``test_op_table.py``, not this file.
``test_op_table.py`` is the flat table where a new op is a single row in ``_CASES``.

This file covers structural edge cases that do not fit naturally in the flat table:
  - Scalar–scalar inputs (0-dim tensors)
  - Large-rank inputs (rank-4, exercises workgroup indexing)
  - 1-element tensors (workgroup boundary conditions)
  - Clip with tensor-input min/max (requires 3-input graph, awkward in a flat row)
  - M0 canonical test: ``test_binary_elementwise[Add-fp32]`` — the single required
    proof-of-life for M0 exit criteria. Do NOT delete this test.

For the full breadth of tier-1 ops (§4.1–§4.5 of OP_COVERAGE.md), see ``test_op_table.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m

# ---------------------------------------------------------------------------
# Random seed — all input generation is deterministic.
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)


def _fp32(shape: tuple[int, ...]) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32)


def _fp32_positive(shape: tuple[int, ...]) -> np.ndarray:
    """Positive fp32 values — for ops that require positive input (Sqrt, Log, Reciprocal)."""
    return np.abs(_fp32(shape)) + 1e-3


# ---------------------------------------------------------------------------
# Binary elementwise cases
# Each entry: (op, dtype, np_dtype, shapes, feed_fn, tol_group, requires_positive_a)
# ---------------------------------------------------------------------------

_BINARY_CASES = [
    # M0: Add is the single required op. Explicitly marked; must not be deleted.
    ("Add", DT.FLOAT, np.float32, ([3, 4], [3, 4]), m.FP32_ELEMENTWISE),
    # M1: remaining binary ops (pending Tank's crate claiming them)
    ("Sub",  DT.FLOAT, np.float32, ([3, 4], [3, 4]), m.FP32_ELEMENTWISE),
    ("Mul",  DT.FLOAT, np.float32, ([3, 4], [3, 4]), m.FP32_ELEMENTWISE),
    ("Div",  DT.FLOAT, np.float32, ([3, 4], [3, 4]), m.FP32_ELEMENTWISE),
    ("Pow",  DT.FLOAT, np.float32, ([3, 4], [3, 4]), m.FP32_ELEMENTWISE),
    ("Min",  DT.FLOAT, np.float32, ([3, 4], [3, 4]), m.FP32_ELEMENTWISE),
    ("Max",  DT.FLOAT, np.float32, ([3, 4], [3, 4]), m.FP32_ELEMENTWISE),
]

# Scalar broadcast: shape [3, 4] × scalar []
_BROADCAST_CASES = [
    ("Add", DT.FLOAT, np.float32, ([3, 4], []), m.FP32_ELEMENTWISE),
    ("Mul", DT.FLOAT, np.float32, ([3, 4], []), m.FP32_ELEMENTWISE),
]

# Edge shape: 0-dim (scalar × scalar)
_SCALAR_CASES = [
    ("Add", DT.FLOAT, np.float32, ([], []), m.FP32_ELEMENTWISE),
]


@pytest.mark.parametrize("op,dtype,np_dtype,shapes,tol", _BINARY_CASES,
                         ids=[f"{op}-fp32" for op, *_ in _BINARY_CASES])
def test_binary_elementwise(op, dtype, np_dtype, shapes, tol, require_vulkan) -> None:
    """fp32 binary elementwise: equal shapes, seeded random inputs."""
    a_shape, b_shape = shapes
    feeds = {
        "a": _fp32(tuple(a_shape)) if op not in ("Pow",) else _fp32_positive(tuple(a_shape)),
        "b": (_fp32_positive(tuple(b_shape)) if op == "Pow" else
              _fp32_positive(tuple(b_shape)) if op == "Div" else  # avoid div-by-zero
              _fp32(tuple(b_shape))),
    }
    model = m.make_model(
        op,
        [m.tensor("a", dtype, a_shape), m.tensor("b", dtype, b_shape)],
        [m.tensor("out", dtype, a_shape)],
    )
    m.check(model, feeds, **tol)


@pytest.mark.parametrize("op,dtype,np_dtype,shapes,tol", _BROADCAST_CASES,
                         ids=[f"{op}-broadcast" for op, *_ in _BROADCAST_CASES])
def test_binary_broadcast_scalar(op, dtype, np_dtype, shapes, tol, require_vulkan) -> None:
    """Binary elementwise with scalar-broadcast second input."""
    a_shape, b_shape = shapes
    feeds = {
        "a": _fp32(tuple(a_shape)),
        "b": np.array(2.0, dtype=np_dtype),
    }
    model = m.make_model(
        op,
        [m.tensor("a", dtype, a_shape), m.tensor("b", dtype, b_shape)],
        [m.tensor("out", dtype, a_shape)],
    )
    m.check(model, feeds, **tol)


@pytest.mark.parametrize("op,dtype,np_dtype,shapes,tol", _SCALAR_CASES,
                         ids=[f"{op}-scalar" for op, *_ in _SCALAR_CASES])
def test_binary_scalar_scalar(op, dtype, np_dtype, shapes, tol, require_vulkan) -> None:
    """Binary elementwise on scalar (0-dim) inputs — edge-shape coverage."""
    feeds = {"a": np.array(1.5, dtype=np_dtype), "b": np.array(2.5, dtype=np_dtype)}
    model = m.make_model(
        op,
        [m.tensor("a", dtype, []), m.tensor("b", dtype, [])],
        [m.tensor("out", dtype, [])],
    )
    m.check(model, feeds, **tol)


# ---------------------------------------------------------------------------
# Unary elementwise cases
# ---------------------------------------------------------------------------

_UNARY_INPUTS: dict[str, np.ndarray] = {
    "Neg":        np.array([[-3.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32),
    "Abs":        np.array([[-3.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32),
    "Sign":       np.array([[-3.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32),
    "Floor":      np.array([[-2.7, -0.1, 0.0, 0.9, 3.2]], dtype=np.float32),
    "Ceil":       np.array([[-2.7, -0.1, 0.0, 0.9, 3.2]], dtype=np.float32),
    "Round":      np.array([[-2.5, -0.5, 0.5, 1.5, 2.5]], dtype=np.float32),  # ties-to-even
    "Sqrt":       np.array([[0.01, 0.5, 1.0, 4.0, 9.0]], dtype=np.float32),
    "Reciprocal": np.array([[-4.0, -0.5, 0.25, 1.0, 2.0]], dtype=np.float32),
    "Exp":        np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]], dtype=np.float32),
    "Log":        np.array([[0.1, 0.5, 1.0, 2.0, 10.0]], dtype=np.float32),
    "Erf":        np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]], dtype=np.float32),
}

_UNARY_CASES = [
    (op, DT.FLOAT, np.float32, m.FP32_ELEMENTWISE if op not in ("Exp", "Log", "Erf") else m.FP32_TRANSCENDENTAL)
    for op in _UNARY_INPUTS
]


@pytest.mark.parametrize("op,dtype,np_dtype,tol", _UNARY_CASES, ids=[c[0] for c in _UNARY_CASES])
def test_unary_elementwise(op, dtype, np_dtype, tol, require_vulkan) -> None:
    """fp32 unary elementwise: seeded fixed inputs chosen to exercise normal value ranges."""
    x = _UNARY_INPUTS[op]
    model = m.make_model(
        op,
        [m.tensor("x", dtype, list(x.shape))],
        [m.tensor("out", dtype, list(x.shape))],
    )
    m.check(model, {"x": x}, **tol)


# ---------------------------------------------------------------------------
# Activation cases
# ---------------------------------------------------------------------------

_ACT_X = np.array([[-3.0, -0.5, 0.0], [0.5, 2.0, 4.0]], dtype=np.float32)

_ACT_CASES: list[tuple[str, dict[str, object], dict]] = [
    ("Relu",          {},                          m.FP32_ACTIVATION),
    ("Sigmoid",       {},                          m.FP32_TRANSCENDENTAL),
    ("Tanh",          {},                          m.FP32_TRANSCENDENTAL),
    ("LeakyRelu",     {},                          m.FP32_ACTIVATION),
    ("LeakyRelu",     {"alpha": 0.1},              m.FP32_ACTIVATION),
    ("Elu",           {},                          m.FP32_ACTIVATION),
    ("Elu",           {"alpha": 1.5},              m.FP32_ACTIVATION),
    ("HardSigmoid",   {},                          m.FP32_ACTIVATION),
    ("HardSigmoid",   {"alpha": 0.15, "beta": 0.4}, m.FP32_ACTIVATION),
    ("Softplus",      {},                          m.FP32_TRANSCENDENTAL),
    # Gelu — default (erf-based, opset 20+)
    ("Gelu",          {},                          m.FP32_TRANSCENDENTAL),
]


def _act_id(case: tuple) -> str:
    op, attrs, _ = case
    suffix = "-".join(f"{k}{v}" for k, v in sorted(attrs.items())) if attrs else "default"
    return f"{op}-{suffix}"


@pytest.mark.parametrize("op,attrs,tol", _ACT_CASES, ids=[_act_id(c) for c in _ACT_CASES])
def test_activation(op, attrs, tol, require_vulkan) -> None:
    """fp32 activations with standard input range."""
    model = m.make_model(
        op,
        [m.tensor("x", DT.FLOAT, list(_ACT_X.shape))],
        [m.tensor("out", DT.FLOAT, list(_ACT_X.shape))],
        attributes=attrs,
    )
    m.check(model, {"x": _ACT_X}, **tol)


# ---------------------------------------------------------------------------
# Clip — special (min/max as tensor inputs, opset 11+)
# ---------------------------------------------------------------------------


def test_clip_min_max(require_vulkan) -> None:
    """Clip with both min and max supplied as scalar tensor inputs."""
    model = m.make_model(
        "Clip",
        [m.tensor("x", DT.FLOAT, [2, 3]),
         m.tensor("min", DT.FLOAT, []),
         m.tensor("max", DT.FLOAT, [])],
        [m.tensor("out", DT.FLOAT, [2, 3])],
    )
    feeds = {
        "x": _ACT_X,
        "min": np.array(-1.0, dtype=np.float32),
        "max": np.array(1.5, dtype=np.float32),
    }
    m.check(model, feeds, **m.FP32_ACTIVATION)


def test_clip_no_bounds(require_vulkan) -> None:
    """Clip with no min or max — should be identity."""
    model = m.make_model(
        "Clip",
        [m.tensor("x", DT.FLOAT, [2, 3])],
        [m.tensor("out", DT.FLOAT, [2, 3])],
    )
    m.check(model, {"x": _ACT_X}, **m.FP32_ACTIVATION)


# ---------------------------------------------------------------------------
# Large rank (edge shape: rank-4 tensor with 1-element dims)
# ---------------------------------------------------------------------------


def test_add_rank4_large(require_vulkan) -> None:
    """Add on a rank-4 tensor — exercises workgroup indexing across multiple dimensions."""
    shape = [2, 8, 8, 4]
    feeds = {"a": _fp32(tuple(shape)), "b": _fp32(tuple(shape))}
    model = m.make_model(
        "Add",
        [m.tensor("a", DT.FLOAT, shape), m.tensor("b", DT.FLOAT, shape)],
        [m.tensor("out", DT.FLOAT, shape)],
    )
    m.check(model, feeds, **m.FP32_ELEMENTWISE)


def test_add_one_element(require_vulkan) -> None:
    """Add on a 1-element tensor — edge case for boundary-condition workgroup dispatch."""
    feeds = {"a": np.array([[1.5]], dtype=np.float32), "b": np.array([[2.5]], dtype=np.float32)}
    model = m.make_model(
        "Add",
        [m.tensor("a", DT.FLOAT, [1, 1]), m.tensor("b", DT.FLOAT, [1, 1])],
        [m.tensor("out", DT.FLOAT, [1, 1])],
    )
    m.check(model, feeds, **m.FP32_ELEMENTWISE)
