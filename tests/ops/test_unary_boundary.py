"""Boundary semantics for the operators that SHARE `rust/shaders/glsl/templates/ew_unary.comp`.

WHY THIS FILE EXISTS (issue #43)
--------------------------------
PR #35 edited `ew_unary.comp` to give Asin/Acos a portable minimax core. That template is
shared by 42 op selectors, so the edit was felt by every operator built from it — and the
operators that broke in CI were `IsInf`, `IsNaN` and `Not`, which have nothing to do with
inverse trig. The break was in the proof ledger rather than the arithmetic, and
`tests/ops/test_proof_ledger.py::test_no_entry_carries_a_stale_source_digest` is the screen
for that. This file closes the OTHER half of the same hole: what the ops actually COMPUTE.

The gap it fills is concrete and was measured, not supposed. In `test_op_table.py` the rows

    _ew1("IsNaN-fp32", "IsNaN", _f32(_S), out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),
    _ew1("IsInf-fp32", "IsInf", _f32(_S), out_dt=DT.BOOL, tol=dict(m.FP32_EXACT)),

draw inputs from `_f32`, which is `_RNG.standard_normal(...)` — a generator that never
produces a NaN and never produces an infinity. So the only assertion those rows have ever
made about the two operators whose entire job is classifying non-finite floats is
`all-false == all-false`. A kernel that returned a constant `false` would pass both, and a
shared-template edit that broke non-finite handling would be invisible to the op table.

Every test below therefore feeds values `_f32` cannot generate, and compares the Vulkan EP
against the ORT CPU EP **exactly** — these are classification and sign-manipulation
operators, so there is no tolerance to hide behind and none is used. `m.assert_vulkan_claims`
runs first in every case: without it a declining EP silently falls back and the suite
compares the CPU EP against itself, which is the vacuous pass DESIGN.md §9.1 exists for and
is precisely how #43 reached main.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m

pytestmark = pytest.mark.usefixtures("require_vulkan")

_F = np.float32


def _nonfinite_f32() -> np.ndarray:
    """The float32 values `standard_normal` cannot produce, plus the ones it will not.

    NaN appears with several distinct payloads and both signs because a classifier that
    tested a bit pattern instead of the exponent/mantissa rule would agree with the CPU on
    the canonical quiet NaN alone. Denormals and the extremes are here because they are the
    inputs most likely to be flushed by a fast-math flag on one driver and not another.
    """
    bits = [
        0x7FC00000,  # canonical quiet NaN
        0xFFC00000,  # negative quiet NaN
        0x7FC0DEAD,  # quiet NaN, non-zero payload
        0x7F800001,  # signalling NaN, smallest payload
        0xFF800001,  # negative signalling NaN
        0x7F800000,  # +Inf
        0xFF800000,  # -Inf
        0x00000000,  # +0.0
        0x80000000,  # -0.0
        0x00000001,  # smallest positive denormal
        0x80000001,  # smallest negative denormal
        0x007FFFFF,  # largest denormal
        0x00800000,  # smallest normal
        0x7F7FFFFF,  # FLT_MAX
        0xFF7FFFFF,  # -FLT_MAX
        0x3F800000,  # 1.0
        0xBF800000,  # -1.0
    ]
    return np.array(bits, dtype=np.uint32).view(np.float32)


def _run_unary(op: str, xs: np.ndarray, out_dt: DT, out_np) -> tuple[np.ndarray, np.ndarray]:
    """Run `op` over `xs` on the Vulkan EP and the CPU EP, asserting the EP claimed the node."""
    shape = [int(xs.size)]
    in_dt = DT.BOOL if xs.dtype == np.bool_ else DT.FLOAT
    model = m.make_model(
        op, [m.tensor("x", in_dt, shape)], [m.tensor("out", out_dt, shape)]
    )
    feeds = {"x": np.ascontiguousarray(xs)}
    m.assert_vulkan_claims(model, feeds)
    vk = np.asarray(m.run_vulkan(model, feeds)[0], dtype=out_np)
    cpu = np.asarray(m.run_cpu(model, feeds)[0], dtype=out_np)
    return vk, cpu


def _bitrep(x: np.floating) -> str:
    v = np.float32(x)
    return f"{float(v)!r} (0x{int(v.view(np.uint32)):08x})"


def _report(op: str, xs: np.ndarray, vk: np.ndarray, cpu: np.ndarray) -> str:
    bad = np.flatnonzero(vk != cpu)
    return f"{op}: Vulkan and the CPU EP disagree at {bad.size}/{xs.size} input(s):\n" + "\n".join(
        f"  x={_bitrep(xs[i])}: vulkan={vk[i]!r} cpu={cpu[i]!r}" for i in bad[:12]
    )


# ---------------------------------------------------------------------------
# The three operators issue #43 names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["IsNaN", "IsInf"])
def test_classifiers_agree_with_the_cpu_on_non_finite_inputs(op: str) -> None:
    """`IsNaN`/`IsInf` over the inputs the op table cannot generate.

    Exact agreement, no tolerance: these return bool. Both directions matter — a classifier
    that says `true` everywhere is as wrong as one that says `false` everywhere, and the
    op-table rows can only ever have observed the latter.
    """
    xs = _nonfinite_f32()
    vk, cpu = _run_unary(op, xs, DT.BOOL, np.bool_)
    assert np.array_equal(vk, cpu), _report(op, xs, vk, cpu)


@pytest.mark.parametrize("op,expect", [("IsNaN", 5), ("IsInf", 2)])
def test_the_classifier_tests_are_not_vacuous(op: str, expect: int) -> None:
    """Non-vacuity: the CPU EP must answer `true` somewhere, and not everywhere.

    This is the guard the op-table rows lack. If a future edit to `_nonfinite_f32` removed
    the NaNs, the test above would go on passing while asserting nothing, exactly as
    `_f32(_S)` does today. Asserting the expected count of `true` answers makes that
    silent loss of coverage a failure instead.
    """
    xs = _nonfinite_f32()
    _, cpu = _run_unary(op, xs, DT.BOOL, np.bool_)
    n = int(np.count_nonzero(cpu))
    assert n == expect, (
        f"{op}: expected exactly {expect} true answer(s) over the boundary input set, got {n}. "
        "The input set no longer exercises what this file claims it does."
    )
    assert n != xs.size, f"{op}: answered true for every input; the set has no negative case"


def test_not_inverts_both_boolean_values_and_is_an_involution() -> None:
    """`Not` over both bool values, checked against the CPU EP and against itself.

    `Not` is the only bool>bool op on the shared template, so it is the only place a change
    to the packed-byte path (`EW_PACKED_BYTES`) shows up alone. The involution arm is a
    property the CPU EP is not needed for: whatever `Not` means, applying it twice must be
    the identity, and a kernel that wrote a nonzero-but-not-1 byte would fail here while
    comparing equal under a truthiness cast.
    """
    xs = np.array([True, False] * 8, dtype=np.bool_)
    vk, cpu = _run_unary("Not", xs, DT.BOOL, np.bool_)
    assert np.array_equal(vk, cpu), _report("Not", xs.astype(np.float32), vk, cpu)
    assert np.array_equal(vk, ~xs), f"Not: {vk!r} is not the inverse of {xs!r}"

    once = np.ascontiguousarray(vk)
    twice, _ = _run_unary("Not", once, DT.BOOL, np.bool_)
    assert np.array_equal(twice, xs), (
        f"Not(Not(x)) != x: {twice!r} != {xs!r}. The kernel is not writing a canonical bool."
    )


# ---------------------------------------------------------------------------
# The sign-manipulating neighbours on the same template. These share the
# selector machinery with Asin/Acos and are the ops where -0.0 and non-finite
# handling can diverge without any test noticing.
#
# Two device-level behaviours were MEASURED on NVIDIA RTX A1000 while writing
# this file, and both are permitted rather than defective:
#
#   * NaN payload and sign are not preserved. Every op returns the canonical
#     0x7fffffff for any NaN input, whatever the payload or sign bit. Vulkan
#     only guarantees this through `shaderSignedZeroInfNanPreserveFloat32`,
#     which is a property a device MAY expose and this EP does not require —
#     ew_unary.comp says so itself, in the comment above `ew_asin`. ONNX does
#     not fix NaN payloads either, so bit equality is the wrong question here
#     and "is it a NaN" is the right one.
#   * Denormals may be flushed to zero, and whether they are is DEVICE-DEPENDENT:
#     measured on the same build and the same SPIR-V, NVIDIA RTX A1000 answers
#     Abs(1.4e-45) with +0.0 while lavapipe answers 1.4e-45. This EP requests
#     neither `shaderDenormPreserveFloat32` nor `shaderDenormFlushToZeroFloat32`,
#     so both are conformant and neither may be asserted as THE behaviour.
#
# So the bitwise arm below runs over the PORTABLE subset and states its two
# exclusions; the other two arms assert the device behaviour that is contractual
# (NaN in, NaN out) and characterise the part that is not. None of this relaxes
# an existing tolerance — before this file, no test fed these ops a NaN, an
# infinity or a denormal at all.
# ---------------------------------------------------------------------------

_UNARY_F32_OPS = ["Abs", "Neg", "Sign", "Floor", "Ceil", "Sqrt", "Reciprocal"]


def _is_denormal(a: np.ndarray) -> np.ndarray:
    u = np.asarray(a, np.float32).view(np.uint32)
    return ((u & 0x7F800000) == 0) & ((u & 0x007FFFFF) != 0)


@pytest.mark.parametrize("op", _UNARY_F32_OPS)
def test_sign_and_rounding_ops_agree_bitwise_on_portable_boundary_inputs(op: str) -> None:
    """Bit-exact agreement with the CPU EP over the inputs whose answer is device-independent.

    Compared as BITS, not as floats: `-0.0 == 0.0` is true in IEEE, so a value comparison
    cannot see a kernel that loses the sign of zero — and `Neg(+0.0)`, `Ceil(-0.5)` and
    `Sign(-0.0)` are exactly where losing it is possible. This is the arm that would catch a
    collateral edit to the shared template's sign or zero handling.

    Excluded, with reasons stated rather than assumed: inputs and CPU results that are
    denormal (flush-to-zero is permitted; see the block comment above) and NaN (payloads are
    unspecified by both Vulkan and ONNX). `test_non_finite_inputs_produce_non_finite_outputs`
    and `test_denormal_flush_to_zero_is_consistent` cover those two separately.
    """
    xs = _nonfinite_f32()
    vk, cpu = _run_unary(op, xs, DT.FLOAT, np.float32)

    portable = ~np.isnan(xs) & ~np.isnan(cpu) & ~_is_denormal(xs) & ~_is_denormal(cpu)
    assert np.count_nonzero(portable) >= 6, (
        f"{op}: only {int(np.count_nonzero(portable))} portable input(s) survived the "
        "exclusions; this parametrisation is close to vacuous"
    )

    vb = vk.view(np.uint32)[portable]
    cb = cpu.view(np.uint32)[portable]
    if not np.array_equal(vb, cb):
        bad = np.flatnonzero(vb != cb)
        xp = xs[portable]
        raise AssertionError(
            f"{op}: {bad.size} result(s) differ in BITS from the CPU EP on inputs whose "
            f"answer does not depend on optional device float controls:\n"
            + "\n".join(
                f"  x={_bitrep(xp[i])}: vulkan=0x{vb[i]:08x} cpu=0x{cb[i]:08x}"
                for i in bad[:12]
            )
            + "\n(equal as floats is not equal as bits: -0.0 == 0.0 compares true)"
        )


@pytest.mark.parametrize("op", _UNARY_F32_OPS)
def test_non_finite_inputs_produce_non_finite_outputs(op: str) -> None:
    """NaN in, NaN out — the part of non-finite behaviour that IS contractual.

    Payloads are not asserted because neither Vulkan nor ONNX fixes them, and this device
    was measured returning canonical 0x7fffffff for every NaN input. What must hold is that
    a NaN is not silently turned into a number, which is the failure that would let a broken
    kernel look correct on ordinary data.

    `Sign` is the documented exception and is asserted as such: GLSL's `sign()` returns 0.0
    for a NaN operand, so the Vulkan EP answers 0.0 where the CPU EP propagates the NaN. That
    divergence is PRE-EXISTING and untouched by PR #35 — `git diff 7752b84..e3d3cf6 --
    ew_unary.comp` adds only comments and the four `ew_asin_*` functions, and no existing
    selector — so it is recorded here rather than fixed under issue #43. Recorded as an
    assertion and not a comment so that a future change to it fails loudly.
    """
    xs = _nonfinite_f32()
    nan_in = np.isnan(xs)
    vk, _ = _run_unary(op, xs, DT.FLOAT, np.float32)
    got = vk[nan_in]

    if op == "Sign":
        assert np.all(got == np.float32(0.0)), (
            "Sign no longer answers 0.0 for every NaN input; measured "
            f"{[_bitrep(v) for v in got[:4]]}. If this is deliberate it closes a known "
            "divergence from the ONNX CPU EP and this expectation should be tightened to "
            "NaN propagation, not deleted."
        )
        return

    bad = np.flatnonzero(~np.isnan(got))
    assert bad.size == 0, (
        f"{op}: {bad.size} NaN input(s) produced a number:\n"
        + "\n".join(f"  x={_bitrep(xs[nan_in][i])} -> {_bitrep(got[i])}" for i in bad[:8])
    )


def test_infinities_survive_the_ops_that_are_defined_on_them() -> None:
    """+/-Inf is carried through, and its sign with it.

    A kernel that clamped or saturated would pass every random-input test in the suite and
    fail here. `Reciprocal` is the inverse case: 1/inf is 0 and the sign of that zero is the
    sign of the infinity, which only a bitwise comparison can see.
    """
    xs = np.array([0x7F800000, 0xFF800000], dtype=np.uint32).view(np.float32)
    for op, expect in (
        ("Abs", [0x7F800000, 0x7F800000]),
        ("Neg", [0xFF800000, 0x7F800000]),
        ("Floor", [0x7F800000, 0xFF800000]),
        ("Ceil", [0x7F800000, 0xFF800000]),
        ("Reciprocal", [0x00000000, 0x80000000]),
    ):
        vk, cpu = _run_unary(op, xs, DT.FLOAT, np.float32)
        assert list(vk.view(np.uint32)) == expect, (
            f"{op}(+/-inf): got {[hex(int(b)) for b in vk.view(np.uint32)]}, "
            f"expected {[hex(b) for b in expect]}"
        )
        assert np.array_equal(vk.view(np.uint32), cpu.view(np.uint32)), (
            f"{op}(+/-inf): Vulkan {[hex(int(b)) for b in vk.view(np.uint32)]} != "
            f"CPU {[hex(int(b)) for b in cpu.view(np.uint32)]}"
        )


def test_denormal_handling_is_uniform_across_the_shared_template() -> None:
    """Characterisation, not conformance: the POLICY is the device's, the UNIFORMITY is ours.

    This EP requests neither `shaderDenormPreserveFloat32` nor `shaderDenormFlushToZeroFloat32`,
    so denormal handling is the device's choice and both answers are conformant. Measured
    while writing this file, on the same build and the same SPIR-V:

        NVIDIA RTX A1000            Abs(1.4e-45) -> +0.0        (flush to zero)
        lavapipe / llvmpipe         Abs(1.4e-45) -> 1.4e-45     (preserved)

    So asserting either policy would bake one driver's behaviour into the suite — the exact
    mistake issue #4 was opened to undo for `asin`/`acos`. What this EP *can* owe the reader
    is that whichever policy the device applies, it applies it CONSISTENTLY across the
    operators built from the shared template: a mixture, where some selectors flush and
    others preserve, would make `ew_unary.comp`'s behaviour depend on which op you asked
    for, and that is a defect on any device.

    So the policy is read off the device with `Abs` and then `Neg` is required to agree with
    it, sign of zero included. A collateral edit to the shared template that changed the
    denormal path for one selector and not another fails here on every device.
    """
    bits = [0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF]
    xs = np.array(bits, dtype=np.uint32).view(np.float32)

    abs_vk, _ = _run_unary("Abs", xs, DT.FLOAT, np.float32)
    flushes = bool(np.all(abs_vk == np.float32(0.0)))
    preserves = bool(np.array_equal(abs_vk.view(np.uint32), np.array([b & 0x7FFFFFFF for b in bits], dtype=np.uint32)))
    assert flushes or preserves, (
        "Abs answered denormal inputs with neither a uniform zero nor the preserved magnitude: "
        f"{[_bitrep(v) for v in abs_vk]}. That is a mixture, which no denormal policy produces."
    )

    neg_vk, _ = _run_unary("Neg", xs, DT.FLOAT, np.float32)
    expect = (
        [0x80000000, 0x00000000, 0x80000000, 0x00000000]
        if flushes
        else [b ^ 0x80000000 for b in bits]
    )
    policy = "flush-to-zero" if flushes else "denorm-preserve"
    assert list(neg_vk.view(np.uint32)) == expect, (
        f"Abs reports a {policy} device, but Neg does not follow the same policy: got "
        f"{[hex(int(b)) for b in neg_vk.view(np.uint32)]}, expected {[hex(b) for b in expect]}. "
        "Two selectors on the same template are handling denormals differently."
    )


def test_the_boundary_set_contains_what_the_op_table_generator_cannot_produce() -> None:
    """The premise of this file, asserted rather than asserted-about.

    `test_op_table._f32` is `standard_normal`. If some future change gave it NaNs and
    infinities, the argument in this module's docstring would be stale and a reader should
    be told by a failure rather than by reading. Equally, if `_nonfinite_f32` ever stopped
    containing them, every test above would quietly become the vacuous pass it was written
    to replace.
    """
    xs = _nonfinite_f32()
    assert np.any(np.isnan(xs)), "the boundary set has no NaN"
    assert np.any(np.isposinf(xs)) and np.any(np.isneginf(xs)), "the boundary set has no infinity"
    assert np.any(xs.view(np.uint32) == 0x80000000), "the boundary set has no -0.0"

    from test_op_table import _f32  # the generator whose blind spot this file covers

    sample = _f32((512,))
    assert np.all(np.isfinite(sample)), (
        "test_op_table._f32 now produces non-finite values, so the IsNaN/IsInf rows in the op "
        "table are no longer vacuous and this file's docstring is out of date"
    )
