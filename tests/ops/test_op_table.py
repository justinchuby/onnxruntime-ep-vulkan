"""Unified op-table test harness for the Vulkan EP.

HOW TO ADD AN OP
================
Add a single row to ``_CASES``. That is all. No new file, no new fixture, no new function.

Each row is a ``CaseSpec`` with:
  - ``id``      : unique pytest identifier, e.g. ``"Add-fp32"``
  - ``op``      : ONNX op name
  - ``domain``  : ``""`` for ai.onnx, ``"com.microsoft"`` for contrib ops
  - ``attrs``   : dict of op attributes (may be empty)
  - ``inputs``  : list of ``(name, DataType, shape)`` triples
  - ``feeds``   : ``{name: np.ndarray}`` — seeded, deterministic inputs
  - ``outputs`` : list of ``(name, DataType, shape)`` triples
  - ``tol``     : tolerance dict from ``_models`` (``FP32_ELEMENTWISE``, etc.)
  - ``claim``   : ``True`` → assert EP claims the node; ``False`` → assert EP does NOT claim
  - ``live``    : ``True`` → kernel is confirmed working end-to-end; enables barrier parity

For ``claim=True``: the test calls ``m.check()`` which asserts:
  1. VulkanExecutionProvider actually executed the node (vacuous-pass guard).
  2. Outputs match ORT CPU EP within ``tol``.

For ``claim=False``: the test calls ``m.assert_vulkan_does_not_claim()`` + ``m.assert_matches_cpu()``.
  This is the conservative-claiming guard: proves the EP declined the node AND CPU produced
  correct results — catches over-claiming before silent wrong results reach a user.

LIVE FLAG — BARRIER PARITY GATE
=================================
``live=True`` means Mouse has confirmed the kernel dispatches end-to-end on real hardware.
The barrier-parity test (``test_barrier_parity.py``) reads this flag to decide whether to
run rather than creating an ORT probe session. This prevents Intel Iris Xe AV crashes in
the EP's Compile path for Staged ops (C-level crash, not catchable by Python's except).
Crash was localised to Atan-fp32 (case index 39, deterministic order) on 2026-07-29.

HOW TO MARK AN OP LIVE:
  1. Mouse marks the op Ready in ``rust/src/ops/registry.rs`` (kernel dispatches correctly).
  2. Add ``live=True`` to the corresponding ``CaseSpec`` row here.
  3. ``test_op_table[{id}]`` must pass; ``test_barrier_parity[{id}]`` will then run instead of skip.

TOLERANCE POLICY
================
Use named constants from ``_models``; never hardcode numbers.
See ``_models.py`` module docstring for full policy and justifications.

DETERMINISM
===========
All input arrays are constructed from ``_RNG`` (seeded 42). Do not use ``np.random``
directly; do not call ``_RNG`` at module level outside of the ``_CASES`` table
construction — that would make the seed position depend on import order.

OP COVERAGE MAPPING
===================
Sections below mirror OP_COVERAGE.md §4.1–§4.5 + declined set.
When Mouse's crate adds an op family, add the corresponding rows in the matching section.
A row with ``claim=True`` is a pending test: it will be red until Tank implements the op.
That is intentional — the table is the implementation checklist.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m

# ---------------------------------------------------------------------------
# Deterministic RNG — seed fixed; all inputs constructed once at import time.
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# CaseSpec — one row in the op table.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CaseSpec:
    id: str
    op: str
    feeds: dict[str, np.ndarray]
    claim: bool = True
    # live=True means the kernel is confirmed working end-to-end (dispatch succeeds on real
    # hardware). Set to True by Mouse when marking an op Ready in rust/src/ops/registry.rs.
    # The barrier-parity test uses this flag — not a probe session — to decide whether to run.
    # This prevents Intel-only AV crashes in the EP's Compile path for Staged ops when the
    # parity guard creates a profiling ORT session (C-level crash, not catchable by Python).
    #
    # INVARIANT: live=True must not be set without claim=True.
    # VALIDATION: test_op_table[{id}] passes only when live=True is correct — a live=True
    # row that doesn't actually execute on VulkanEP will fail the claim assertion.
    live: bool = False
    domain: str = ""
    attrs: dict[str, object] = dataclasses.field(default_factory=dict)
    inputs: list[tuple[str, DT, list[int]]] = dataclasses.field(default_factory=list)
    outputs: list[tuple[str, DT, list[int]]] = dataclasses.field(default_factory=list)
    tol: dict[str, float] = dataclasses.field(default_factory=lambda: dict(m.FP32_ELEMENTWISE))


# ---------------------------------------------------------------------------
# Factory helpers — reduce per-row verbosity for common patterns.
# These are called ONCE at module load to populate _CASES; do not call them
# inside test functions.
# ---------------------------------------------------------------------------

def _f32(shape: tuple[int, ...]) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32)

def _f32_pos(shape: tuple[int, ...]) -> np.ndarray:
    """Strictly positive fp32 — for Sqrt, Log, Reciprocal, etc."""
    return (np.abs(_RNG.standard_normal(shape)) + 1e-3).astype(np.float32)

def _f32_unit(shape: tuple[int, ...]) -> np.ndarray:
    """fp32 in (-1, 1) — for Asin, Acos, Atanh."""
    return ((_RNG.random(shape) * 1.8) - 0.9).astype(np.float32)

def _f32_ge1(shape: tuple[int, ...]) -> np.ndarray:
    """fp32 >= 1 — for Acosh."""
    return (_RNG.random(shape).astype(np.float32) + 1.01)

def _bool(shape: tuple[int, ...]) -> np.ndarray:
    return _RNG.integers(0, 2, shape).astype(bool)

def _f16(shape: tuple[int, ...]) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float16)

def _f16_pos(shape: tuple[int, ...]) -> np.ndarray:
    """Strictly positive fp16 — for Sqrt, Div denominators, Log."""
    return (np.abs(_RNG.standard_normal(shape)) + 0.05).astype(np.float16)

def _i32(shape: tuple[int, ...]) -> np.ndarray:
    return _RNG.integers(-100, 100, shape, dtype=np.int32)

_S = (3, 4)  # standard test shape
_S16 = (3, 5)  # odd element count: exercises the high half of the final packed fp16 word

def _ew2(
    op_id: str,
    op: str,
    a: np.ndarray,
    b: np.ndarray,
    *,
    in_dt: DT = DT.FLOAT,
    out_dt: DT = DT.FLOAT,
    attrs: dict | None = None,
    tol: dict | None = None,
    claim: bool = True,
    live: bool = False,
    domain: str = "",
) -> CaseSpec:
    """Two-input elementwise op: output shape = a.shape."""
    shape = list(a.shape)
    return CaseSpec(
        id=op_id, op=op, domain=domain,
        attrs=attrs or {},
        inputs=[("a", in_dt, shape), ("b", in_dt, list(b.shape))],
        feeds={"a": a, "b": b},
        outputs=[("out", out_dt, shape)],
        tol=tol or dict(m.FP32_ELEMENTWISE),
        claim=claim,
        live=live,
    )


def _ew1(
    op_id: str,
    op: str,
    x: np.ndarray,
    *,
    in_dt: DT = DT.FLOAT,
    out_dt: DT = DT.FLOAT,
    attrs: dict | None = None,
    tol: dict | None = None,
    claim: bool = True,
    live: bool = False,
) -> CaseSpec:
    """One-input elementwise op: output shape = x.shape."""
    shape = list(x.shape)
    return CaseSpec(
        id=op_id, op=op,
        attrs=attrs or {},
        inputs=[("x", in_dt, shape)],
        feeds={"x": x},
        outputs=[("out", out_dt, shape)],
        tol=tol or dict(m.FP32_ELEMENTWISE),
        claim=claim,
        live=live,
    )


# ---------------------------------------------------------------------------
# _CASES — THE TABLE.  Add an op here; it will be tested automatically.
#
# claim=True  → test will PASS only when Tank's crate claims the op.
# claim=False → test will PASS only when the EP declines the op.
# ---------------------------------------------------------------------------

_CASES: list[CaseSpec] = [

    # ======================================================================
    # §4.1  Binary / variadic elementwise — EW-B (23 ops)
    # ======================================================================

    # --- fp32 arithmetic (all S, all EW-B) ---
    _ew2("Add-fp32",  "Add",  _f32(_S), _f32(_S), live=True),
    _ew2("Sub-fp32",  "Sub",  _f32(_S), _f32(_S), live=True),
    _ew2("Mul-fp32",  "Mul",  _f32(_S), _f32(_S), live=True),
    _ew2("Div-fp32",  "Div",  _f32(_S), _f32_pos(_S), live=True),        # avoid div-by-zero
    _ew2("Pow-fp32",  "Pow",  _f32_pos(_S), _f32_pos(_S), live=True),    # pow(pos, pos) stays real
    _ew2("Min-fp32",  "Min",  _f32(_S), _f32(_S)),
    _ew2("Max-fp32",  "Max",  _f32(_S), _f32(_S)),
    _ew2("PRelu-fp32","PRelu",_f32(_S), _f32_pos((4,)),        # slope broadcast on last dim
        tol=dict(m.FP32_ACTIVATION)),

    # --- variadic fold — claim only when variadic support is complete ---
    # Sum/Mean claim 2-input form; variadic expansion is gated separately.
    _ew2("Sum-fp32-2inp", "Sum",  _f32(_S), _f32(_S)),
    _ew2("Mean-fp32-2inp","Mean", _f32(_S), _f32(_S)),

    # --- comparison ops → bool output ---
    _ew2("Equal-fp32",          "Equal",          _f32(_S), _f32(_S),
         out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),
    _ew2("Greater-fp32",        "Greater",        _f32(_S), _f32(_S),
         out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),
    _ew2("Less-fp32",           "Less",           _f32(_S), _f32(_S),
         out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),
    _ew2("GreaterOrEqual-fp32", "GreaterOrEqual", _f32(_S), _f32(_S),
         out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),
    _ew2("LessOrEqual-fp32",    "LessOrEqual",    _f32(_S), _f32(_S),
         out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),

    # --- integer arithmetic ---
    _ew2("Add-i32",  "Add",  _i32(_S), _i32(_S), in_dt=DT.INT32, out_dt=DT.INT32,
         tol=dict(m.FP32_EXACT)),
    _ew2("Mul-i32",  "Mul",  _i32(_S), _i32(_S), in_dt=DT.INT32, out_dt=DT.INT32,
         tol=dict(m.FP32_EXACT)),

    # --- boolean logic ---
    _ew2("And-bool", "And", _bool(_S), _bool(_S),
         in_dt=DT.BOOL, out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),
    _ew2("Or-bool",  "Or",  _bool(_S), _bool(_S),
         in_dt=DT.BOOL, out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),
    _ew2("Xor-bool", "Xor", _bool(_S), _bool(_S),
         in_dt=DT.BOOL, out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),

    # --- bitwise integer ---
    _ew2("BitwiseAnd-i32", "BitwiseAnd", _i32(_S), _i32(_S),
         in_dt=DT.INT32, out_dt=DT.INT32, tol=dict(m.FP32_EXACT)),
    _ew2("BitwiseOr-i32",  "BitwiseOr",  _i32(_S), _i32(_S),
         in_dt=DT.INT32, out_dt=DT.INT32, tol=dict(m.FP32_EXACT)),
    _ew2("BitwiseXor-i32", "BitwiseXor", _i32(_S), _i32(_S),
         in_dt=DT.INT32, out_dt=DT.INT32, tol=dict(m.FP32_EXACT)),

    # ======================================================================
    # §4.2  Unary elementwise — EW-U (27 ops)
    # ======================================================================

    _ew1("Neg-fp32",        "Neg",        _f32(_S), live=True),
    _ew1("Abs-fp32",        "Abs",        _f32(_S), live=True),
    _ew1("Sign-fp32",       "Sign",       _f32(_S), tol=dict(m.FP32_EXACT), live=True),
    _ew1("Floor-fp32",      "Floor",      _f32(_S), tol=dict(m.FP32_EXACT), live=True),
    _ew1("Ceil-fp32",       "Ceil",       _f32(_S), tol=dict(m.FP32_EXACT), live=True),
    _ew1("Round-fp32",      "Round",      _f32(_S), tol=dict(m.FP32_EXACT), live=True),
    _ew1("Sqrt-fp32",       "Sqrt",       _f32_pos(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Reciprocal-fp32", "Reciprocal", _f32_pos(_S), live=True),
    _ew1("Exp-fp32",        "Exp",        _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Log-fp32",        "Log",        _f32_pos(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Erf-fp32",        "Erf",        _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Sin-fp32",        "Sin",        _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Cos-fp32",        "Cos",        _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Tan-fp32",        "Tan",        _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Asin-fp32",       "Asin",       _f32_unit(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Acos-fp32",       "Acos",       _f32_unit(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Atan-fp32",       "Atan",       _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Sinh-fp32",       "Sinh",       _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Cosh-fp32",       "Cosh",       _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Tanh-fp32",       "Tanh",       _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Asinh-fp32",      "Asinh",      _f32(_S),     tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Acosh-fp32",      "Acosh",      _f32_ge1(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Atanh-fp32",      "Atanh",      _f32_unit(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Not-bool",        "Not",        _bool(_S),
         in_dt=DT.BOOL, out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),
    _ew1("BitwiseNot-i32",  "BitwiseNot", _i32(_S),
         in_dt=DT.INT32, out_dt=DT.INT32, tol=dict(m.FP32_EXACT)),
    _ew1("IsNaN-fp32",      "IsNaN",      _f32(_S),
         out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),
    _ew1("IsInf-fp32",      "IsInf",      _f32(_S),
         out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),

    # ======================================================================
    # §4.3  Activations — EW-U with push-constant params (16 ops)
    # ======================================================================

    _ew1("Relu-fp32",            "Relu",            _f32(_S), tol=dict(m.FP32_ACTIVATION), live=True),
    _ew1("Sigmoid-fp32",         "Sigmoid",         _f32(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Tanh-act-fp32",        "Tanh",            _f32(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("LeakyRelu-default",    "LeakyRelu",       _f32(_S), tol=dict(m.FP32_ACTIVATION), live=True),
    _ew1("LeakyRelu-alpha0.1",   "LeakyRelu",       _f32(_S), tol=dict(m.FP32_ACTIVATION),
         attrs={"alpha": 0.1}, live=True),
    _ew1("Elu-default",          "Elu",             _f32(_S), tol=dict(m.FP32_ACTIVATION), live=True),
    _ew1("Elu-alpha1.5",         "Elu",             _f32(_S), tol=dict(m.FP32_ACTIVATION),
         attrs={"alpha": 1.5}, live=True),
    _ew1("Selu-default",         "Selu",            _f32(_S), tol=dict(m.FP32_ACTIVATION), live=True),
    _ew1("Celu-default",         "Celu",            _f32(_S), tol=dict(m.FP32_ACTIVATION), live=True),
    _ew1("HardSigmoid-default",  "HardSigmoid",     _f32(_S), tol=dict(m.FP32_ACTIVATION), live=True),
    _ew1("HardSigmoid-custom",   "HardSigmoid",     _f32(_S), tol=dict(m.FP32_ACTIVATION),
         attrs={"alpha": 0.15, "beta": 0.4}, live=True),
    _ew1("HardSwish-fp32",       "HardSwish",       _f32(_S), tol=dict(m.FP32_ACTIVATION), live=True),
    _ew1("Softplus-fp32",        "Softplus",        _f32(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Softsign-fp32",        "Softsign",        _f32(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Gelu-fp32",            "Gelu",            _f32(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("Mish-fp32",            "Mish",            _f32(_S), tol=dict(m.FP32_TRANSCENDENTAL), live=True),
    _ew1("ThresholdedRelu-fp32", "ThresholdedRelu", _f32(_S), tol=dict(m.FP32_ACTIVATION),
         attrs={"alpha": 1.0}, live=True),

    # ======================================================================
    # §4.3b  fp16 elementwise — the dtype a real decoder is actually made of
    # ======================================================================
    #
    # Phi-3.5 is fp16 throughout: 64 Mul and 32 Sigmoid nodes, and the fp32 rows above are worth
    # zero of them. These exercise a genuinely different storage path, not a different expression
    # — f16 tensors are packed two to a uint word and stored through atomicAnd/atomicOr on
    # disjoint 16-bit lanes, so a wrong lane or word index is invisible to every fp32 row here.
    #
    # An odd element count is deliberate on the odd rows below (_S16 = (3, 5) = 15 elements): it
    # puts a live element in a partial final word, which is exactly where the packing breaks.
    _ew2("Add-fp16",  "Add",  _f16(_S), _f16(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    _ew2("Sub-fp16",  "Sub",  _f16(_S), _f16(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    _ew2("Mul-fp16",  "Mul",  _f16(_S), _f16(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    _ew2("Div-fp16",  "Div",  _f16(_S), _f16_pos(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    # Broadcast at fp16: b is rank-1 over the last axis, so the general indexing path runs rather
    # than the EW_IDENTICAL fast path. That distinction is what the whole 258-node dynamic-shape
    # story turns on, and it has never been exercised at fp16.
    _ew2("Mul-fp16-bcast", "Mul", _f16(_S), _f16((_S[1],)), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),

    # An odd element count is deliberately declined, not claimed: _S16 = (3, 5) = 15 elements puts
    # a live element in a partial final word, whose store lands outside the bound buffer range.
    # Device selector 0 (NVIDIA) absorbed that and returned the right answer; device selector 1
    # (Intel Iris Xe, the spec-conformance oracle) applied robustBufferAccess and left a zero.
    # Labels corrected 2026-07-30T21:23:53-07:00 — the selector indices were inverted in all
    # earlier comments (Tank's finding: discrete sorts first, so selector 0 = NVIDIA).
    # These rows are here as the regression guard for that vendor split — claim=False
    # asserts the EP declines rather than answering differently on two GPUs.
    _ew1("Relu-fp16-odd", "Relu", _f16(_S16), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), claim=False),
    _ew1("Sigmoid-fp16-odd", "Sigmoid", _f16(_S16), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), claim=False),

    _ew1("Relu-fp16",    "Relu",    _f16(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    _ew1("Sigmoid-fp16", "Sigmoid", _f16(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    _ew1("Sqrt-fp16",    "Sqrt",    _f16_pos(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    _ew1("Exp-fp16",     "Exp",     _f16(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    _ew1("Tanh-fp16",    "Tanh",    _f16(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    _ew1("Erf-fp16",     "Erf",     _f16(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),
    # Gelu carries no attribute but travels the parameter tail; proving it at fp16 proves the tail
    # offset is dtype-independent.
    _ew1("Gelu-fp16",    "Gelu",    _f16(_S), in_dt=DT.FLOAT16, out_dt=DT.FLOAT16,
         tol=dict(m.FP16_ANY), live=True),

    # ======================================================================
    # §4.4  Select / cast — EW-T (3 ops)
    # ======================================================================

    # Cast fp32 → int32
    #
    # The feed is scaled and given exact zeros on purpose. A standard normal truncates to zero on
    # ~68% of its elements and to ±1 on most of the rest, so a kernel that wrote zeros everywhere
    # would have matched the CPU oracle closely enough to look right. The property under test is
    # "truncation toward zero, at magnitudes that make truncation observable", and the feed has to
    # contain such magnitudes for the assertion to mean it.
    CaseSpec(
        id="Cast-fp32-to-i32", op="Cast",
        attrs={"to": int(DT.INT32)},  # ONNX TensorProto.INT32 = 6
        inputs=[("x", DT.FLOAT, list(_S))],
        feeds={"x": (100.0 * _f32(_S)).astype(np.float32)},
        outputs=[("out", DT.INT32, list(_S))],
        tol=dict(m.FP32_EXACT),
    ),

    # Cast int32 → fp32
    CaseSpec(
        id="Cast-i32-to-fp32", op="Cast",
        attrs={"to": int(DT.FLOAT)},  # ONNX TensorProto.FLOAT = 1
        inputs=[("x", DT.INT32, list(_S))],
        feeds={"x": (1000 * _i32(_S)).astype(np.int32)},
        outputs=[("out", DT.FLOAT, list(_S))],
        tol=dict(m.FP32_EXACT),
    ),

    # Cast fp32 → bool
    #
    # ONNX casts to bool with `x != 0`, so a continuous feed is all-True — a constant reference,
    # which any kernel that returned a constant would match. The zeros here are what make the
    # False arm reachable at all.
    CaseSpec(
        id="Cast-fp32-to-bool", op="Cast",
        attrs={"to": int(DT.BOOL)},   # ONNX TensorProto.BOOL = 9
        inputs=[("x", DT.FLOAT, list(_S))],
        feeds={"x": np.trunc(2.0 * _f32(_S)).astype(np.float32)},
        outputs=[("out", DT.BOOL, list(_S))],
        tol=dict(m.FP32_EXACT),
    ),

    # Cast fp32 → fp64 is DECLINED (fp64 is a permanent CPU fallback — §4.4 "no f64 ever")
    CaseSpec(
        id="Cast-fp32-to-fp64-declined", op="Cast",
        attrs={"to": int(DT.DOUBLE)},
        inputs=[("x", DT.FLOAT, list(_S))],
        feeds={"x": _f32(_S)},
        outputs=[("out", DT.DOUBLE, list(_S))],
        tol=dict(m.FP32_EXACT),
        claim=False,
    ),

    # Where: 3-way broadcast; condition is BOOL, x/y are fp32
    CaseSpec(
        id="Where-fp32", op="Where",
        inputs=[
            ("cond", DT.BOOL,  list(_S)),
            ("x",    DT.FLOAT, list(_S)),
            ("y",    DT.FLOAT, list(_S)),
        ],
        feeds={
            "cond": _bool(_S),
            "x":    _f32(_S),
            "y":    _f32(_S),
        },
        outputs=[("out", DT.FLOAT, list(_S))],
        tol=dict(m.FP32_EXACT),
    ),

    # ======================================================================
    # §4.5  Shape metadata — zero-dispatch ops (9 ops)
    # ======================================================================

    # Identity: an ew_unary copy kernel, claimed even as a one-node island.
    _ew1("Identity-fp32", "Identity", _f32(_S), live=True),

    # ----------------------------------------------------------------------
    # Flatten — DECLINED.  Reshape — CLAIMED as of 2026-08-04, by the falsifier
    # this comment named.
    #
    # These two rows asserted `claim=True` for three rounds and were red every
    # time. The row comment then read "Row is present to mark the claimed op" —
    # which is the tell: it asserted an *intention*, not a property. They were
    # flipped to `claim=False` with a ruling and, crucially, with the run that
    # would overturn it.
    #
    # THE RULING WAS: `Reshape` and `Flatten` perform no arithmetic, their only
    # value is not breaking an island, and the Phi-3.5 claim log (363 node
    # records, the whole graph) contains **zero** of either — so registering
    # them would widen the claim table for this suite's benefit and no model's.
    #
    # THE FALSIFIER FIRED. The op census run on BERT-SQuAD-12 on 2026-08-04
    # (bench/results/op_census_bert_r16.json) reports **71 `Reshape` nodes in
    # the graph and 59 offered to the EP**. The premise "no real model asks"
    # is false by name and by count, exactly as this comment said it would be,
    # so the `Reshape` row flips back to `claim=True` and the row below is now
    # a positive case with a CPU oracle behind it.
    #
    # `Flatten` stays declined, and the distinction is measured rather than
    # inherited: BERT has 71 `Reshape` and **zero** `Flatten`; MobileNetV2 has
    # 1 and zero; Phi-3.5 has zero and zero. The falsifier is unchanged for it.
    #
    # WHAT THE CLAIM COSTS, so the flip is not read as free: `Reshape` is a
    # copy — one full-tensor read and one full-tensor write through
    # `ew_cast_f32_to_f32`. It is not an alias. `bind_aliased_output` exists but
    # `dispatch_ort` honours a pair only when the output is an external plan
    # output and the input an external plan input, and every `Reshape` worth
    # claiming is an interior island edge.
    #
    # AND WHAT IT DOES NOT BUY: on BERT the registered row claims **zero** of
    # those 59 nodes — 53 have no rank on any operand, 4 are i64, 2 fail
    # conservation of element count. The row is live and correct and the model
    # it was written for does not use it. That is recorded in the decision file
    # `mouse-reshape-claims-nothing.md`, not hidden behind this green test.
    # ----------------------------------------------------------------------

    # Flatten (axis=1 over [3, 4] → [3, 4])
    CaseSpec(
        id="Flatten-fp32-axis1", op="Flatten",
        attrs={"axis": 1},
        inputs=[("x", DT.FLOAT, [3, 4])],
        feeds={"x": _f32((3, 4))},
        outputs=[("out", DT.FLOAT, [3, 4])],
        tol=dict(m.FP32_EXACT),
        claim=False,
    ),

    # Reshape (static shape initializer as input 1)
    CaseSpec(
        id="Reshape-fp32-static", op="Reshape",
        attrs={},
        inputs=[("x", DT.FLOAT, [3, 4]), ("shape", DT.INT64, [2])],
        feeds={
            "x":     _f32((3, 4)),
            "shape": np.array([4, 3], dtype=np.int64),
        },
        outputs=[("out", DT.FLOAT, [4, 3])],
        tol=dict(m.FP32_EXACT),
        claim=True,
    ),

    # ======================================================================
    # §4.1 / §4.2  DECLINED cases — must NOT be claimed
    # ======================================================================

    # fp64 arithmetic: Vulkan shaderFloat64 is optional; EP must decline.
    _ew2("Add-fp64-declined", "Add",
         _f32(_S).astype(np.float64), _f32(_S).astype(np.float64),
         in_dt=DT.DOUBLE, out_dt=DT.DOUBLE,
         tol=dict(m.FP32_EXACT), claim=False),

    # NonZero: data-dependent output shape — permanent CPU fallback (OP_COVERAGE §4.7).
    CaseSpec(
        id="NonZero-declined", op="NonZero",
        inputs=[("x", DT.FLOAT, [2, 4])],
        feeds={"x": np.array([[0.0, 1.5, 0.0, 2.0], [0.0, 0.0, -1.0, 0.0]], dtype=np.float32)},
        outputs=[("out", DT.INT64, [2, -1])],
        tol=dict(m.FP32_EXACT),
        claim=False,
    ),
]


# ---------------------------------------------------------------------------
# Test dispatch — single function, single parametrize.
# Adding a row to _CASES is all that is required to add a test.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_op_table(case: CaseSpec, require_vulkan) -> None:
    """Unified op-table dispatch: claim assertion + CPU oracle comparison."""
    inputs  = [m.tensor(n, dt, s) for n, dt, s in case.inputs]
    outputs = [m.tensor(n, dt, s) for n, dt, s in case.outputs]
    model   = m.make_model(
        case.op, inputs, outputs,
        domain=case.domain,
        attributes=case.attrs,
    )
    if case.claim:
        m.check(model, case.feeds, **case.tol)
    else:
        m.assert_vulkan_does_not_claim(model, case.feeds)
        if m.cpu_can_run(model, case.feeds):
            m.assert_matches_cpu(model, case.feeds, **case.tol)
