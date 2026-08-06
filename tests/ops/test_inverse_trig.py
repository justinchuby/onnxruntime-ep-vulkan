"""`Asin`/`Acos`: the portable inverse-trig path, and the screen that keeps it honest.

WHAT ISSUE #4 ACTUALLY WAS
==========================
Not "lavapipe is buggy". Vulkan's "Precision of GLSL.std.450 Instructions" table defines
`asin(x)` by inheritance from ``atan2(x, sqrt(1 - x*x))`` and `acos(x)` from
``atan2(sqrt(1 - x*x), x)``, and gives ``atan()``/``atan2()`` an allowance of **4096 ULP** in
single precision. 4096 ULP is ~2.4e-4 relative — twenty-four times looser than
``FP32_TRANSCENDENTAL``. Mesa lavapipe measured 3831 ULP on `Asin` and 3903 ULP on `Acos`: inside
its allowance, conformant, and ~1e3x outside this project's contract. NVIDIA measured 4 and 5.

So the two available answers were: adopt 4096 ULP as the project's accuracy policy for every op
that inherits from ``atan2``, or stop calling the built-in. This suite is the second answer's
evidence, and it is written so that the numbers are checked rather than quoted.

WHAT IS ASSERTED, AND WHY AT THAT NUMBER
========================================
``STATED_ULP_BOUND = 16`` is **derived, not fitted**:

  * the core polynomial contributes <= 2.42 ULP (`asin`) / <= 1.28 ULP (`acos`), measured in exact
    float32 arithmetic over 4,000,001 points with no GPU involved — `test_reference_core_bound`
    below re-derives it on every run;
  * the reflected branch computes ``pi/2 - 2*asin_core(sqrt((1-a)/2))``, which doubles ``sqrt``'s
    error and amplifies it by ``1/sqrt(1 - s^2)``. Vulkan gives ``sqrt`` no bound directly either,
    but by inheritance from ``1.0 / inversesqrt()`` — 2 ULP plus a 2.5 ULP divide, so <= 4.5 ULP.
    Worst at the ``s = 1/2`` seam: ``2 * 4.5 * ulp(0.5) / sqrt(0.75) / ulp(pi/6)`` ~ 10.4 ULP;
  * every other operation on the path (`OpFAdd`, `OpFSub`, `OpFMul`, `OpFma`) is correctly rounded.

2.42 + 10.4 rounds up to 16 with room. It is ~1.9e-6 relative, still five times inside
``FP32_TRANSCENDENTAL``. No measurement from any driver went into choosing it, which is the whole
point — a bound fitted to a driver is that driver's behaviour wearing a contract's clothes.

The devices measure far better than the bound: <= 3 ULP against a float64 reference and <= 5 ULP
against the ORT CPU EP (itself 4 ULP from float64) on NVIDIA RTX A1000 *and* lavapipe.

THE NEGATIVE CONTROL
====================
``test_builtin_negative_control_*`` reads ``bench/results/inverse_trig_ulp_*.json`` — real sweeps
taken this session on real lavapipe, one with the built-in path (94a4bd6) and one with this
branch's. It asserts the built-in artifact **fails** the bound and the portable artifact passes.
A check that has only ever been observed to pass is not known to be a check (CI_POLICY, and the
no-ICD negative control in ci.yml). ``test_bound_rejects_the_builtin_error_magnitude`` makes the
same point without any artifact: it perturbs a correct answer by the error lavapipe produced and
requires the op-table tolerance to reject it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m

pytestmark = pytest.mark.usefixtures("require_vulkan")

_REPO = Path(__file__).resolve().parents[2]
_SHADER_DIR = _REPO / "rust" / "shaders"
_ARTIFACTS = _REPO / "bench" / "results"

#: The derived accuracy contract for `Asin`/`Acos`, in ULPs of float32 at the oracle's magnitude.
#: See the module docstring for the derivation. Do not raise this to make a device pass; a device
#: that needs it raised has found something the derivation does not account for.
STATED_ULP_BOUND = 16.0

#: Bound on the polynomial core alone, in exact float32 arithmetic. Re-derived by
#: `test_reference_core_bound`, which needs no GPU and no driver.
CORE_ULP_BOUND = {"asin": 2.5, "acos": 1.5}

_F = np.float32
_PIO2 = _F(1.5707963267948966)


# ---------------------------------------------------------------------------
# The reference implementation — the same math the shader evaluates, in numpy.
# ---------------------------------------------------------------------------

_CEPHES = [
    _F(4.2163199048e-2),
    _F(2.4181311049e-2),
    _F(4.5470025998e-2),
    _F(7.4953002686e-2),
    _F(1.6666752422e-1),
]


def _core(s: np.ndarray) -> np.ndarray:
    """`asin(s)` for s in [0, 1/2] as `s + s^3*P(s^2)`, evaluated strictly in float32.

    Every intermediate is forced back to float32 so this mirrors the shader rather than numpy's
    float64 promotion. A GPU may contract `p*z + c` into an FMA, which is *more* accurate than
    this, so this reference is the pessimistic side of the shader, not an approximation of it.
    """
    z = (s * s).astype(np.float32)
    p = np.full_like(z, _CEPHES[0])
    for c in _CEPHES[1:]:
        p = (p * z + c).astype(np.float32)
    return ((p * z).astype(np.float32) * s + s).astype(np.float32)


def _reference_asin(x: np.ndarray) -> np.ndarray:
    a = np.abs(x).astype(np.float32)
    refl = a > _F(0.5)
    z = np.where(refl, np.maximum(_F(0.5) - _F(0.5) * a, _F(0.0)), a * a).astype(np.float32)
    s = np.where(refl, np.sqrt(z), a).astype(np.float32)
    r = _core(s)
    out = np.where(refl, (_PIO2 - _F(2.0) * r).astype(np.float32), r).astype(np.float32)
    return np.copysign(out, x).astype(np.float32)


def _reference_acos(x: np.ndarray) -> np.ndarray:
    a = np.abs(x).astype(np.float32)
    refl = a > _F(0.5)
    z = np.where(refl, np.maximum(_F(0.5) - _F(0.5) * a, _F(0.0)), a * a).astype(np.float32)
    s = np.where(refl, np.sqrt(z), a).astype(np.float32)
    r = _core(s)
    two_r = (_F(2.0) * r).astype(np.float32)
    return np.where(
        refl,
        np.where(x > 0, two_r, (_F(np.pi) - two_r).astype(np.float32)),
        (_PIO2 - np.copysign(r, x)).astype(np.float32),
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(op: str, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run `op` on the Vulkan EP and the CPU EP over `xs`, asserting the EP claimed it."""
    shape = [int(xs.size)]
    model = m.make_model(
        op, [m.tensor("x", DT.FLOAT, shape)], [m.tensor("out", DT.FLOAT, shape)]
    )
    feeds = {"x": np.ascontiguousarray(xs)}
    # Without this the whole suite would compare the CPU EP against itself and read 0.0 ULP —
    # the vacuous pass DESIGN.md §9.1 exists for.
    m.assert_vulkan_claims(model, feeds)
    vk = np.asarray(m.run_vulkan(model, feeds)[0], dtype=np.float32)
    cpu = np.asarray(m.run_cpu(model, feeds)[0], dtype=np.float32)
    return vk, cpu


def _ulp(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """ULPs of float32 at the reference's magnitude, via the house `format_spacing`."""
    got64 = got.astype(np.float64)
    ref64 = ref.astype(np.float64)
    with np.errstate(invalid="ignore"):
        u = np.abs(got64 - ref64) / m.format_spacing(ref.astype(np.float32), np.float32)
    both_nan = np.isnan(got64) & np.isnan(ref64)
    return np.where(both_nan | (got64 == ref64), 0.0, u)


def _dense_grid(points: int = 40001) -> np.ndarray:
    """A dense sweep of [-1, 1] with the seams and their representable neighbours pinned in."""
    dense = np.linspace(-1.0, 1.0, points, dtype=np.float64).astype(np.float32)
    seams: list[np.float32] = []
    for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
        s = np.float32(v)
        seams += [s, np.nextafter(s, np.float32(-2.0)), np.nextafter(s, np.float32(2.0))]
    grid = np.unique(np.concatenate([dense, np.array(seams, dtype=np.float32)]))
    return grid[(grid >= -1.0) & (grid <= 1.0)]


# ---------------------------------------------------------------------------
# The core bound, with no GPU in the loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["asin", "acos"])
def test_reference_core_bound(op: str) -> None:
    """The polynomial half of the derivation, re-derived rather than quoted.

    This runs entirely in numpy float32, so it fails the same way on every machine and is the
    part of ``STATED_ULP_BOUND`` that cannot be blamed on a driver.
    """
    xs = _dense_grid(400001)
    got = (_reference_asin if op == "asin" else _reference_acos)(xs)
    exact = (np.arcsin if op == "asin" else np.arccos)(xs.astype(np.float64))
    u = _ulp(got, exact.astype(np.float32))
    worst = float(np.max(u[np.isfinite(u)]))
    assert worst <= CORE_ULP_BOUND[op], (
        f"the float32 reference for {op} reached {worst:.3f} ULP over {xs.size} points, above the "
        f"{CORE_ULP_BOUND[op]} this suite's derivation of STATED_ULP_BOUND assumes. Either the "
        f"coefficients in ew_unary.comp changed without this file changing, or the derivation in "
        f"the module docstring needs redoing — do not raise the constant to make this pass."
    )


# ---------------------------------------------------------------------------
# The device, densely
# ---------------------------------------------------------------------------


@pytest.mark.portability
@pytest.mark.parametrize("op", ["Asin", "Acos"])
def test_dense_sweep_within_stated_bound(op: str) -> None:
    """Every point of [-1, 1] this grid touches is within the derived bound of the CPU EP."""
    xs = _dense_grid()
    vk, cpu = _run(op, xs)
    u = _ulp(vk, cpu)
    finite = np.isfinite(u)
    worst = float(np.max(u[finite]))
    at = xs[np.argmax(np.where(finite, u, -1.0))]
    assert worst <= STATED_ULP_BOUND, (
        f"{op}: worst {worst:.1f} ULP vs the CPU EP at x={float(at)!r} over {xs.size} points, "
        f"above the derived bound of {STATED_ULP_BOUND}. p99={np.percentile(u[finite], 99):.1f}, "
        f"p99.9={np.percentile(u[finite], 99.9):.1f}. This bound is derived from Vulkan's own "
        f"guarantees; a device that exceeds it has done something the derivation does not cover."
    )


@pytest.mark.portability
@pytest.mark.parametrize("op", ["Asin", "Acos"])
def test_random_inputs_within_stated_bound(op: str) -> None:
    """The same bound on a seeded random sample, so the grid's regularity is not load-bearing."""
    rng = np.random.default_rng(20260805)
    xs = ((rng.random(8192) * 2.0) - 1.0).astype(np.float32)
    vk, cpu = _run(op, xs)
    u = _ulp(vk, cpu)
    worst = float(np.max(u[np.isfinite(u)]))
    assert worst <= STATED_ULP_BOUND, f"{op}: worst {worst:.1f} ULP on random inputs"


@pytest.mark.portability
@pytest.mark.parametrize("op", ["Asin", "Acos"])
def test_near_unit_magnitude_is_stable(op: str) -> None:
    """|x| within a few ULP of 1 — where the reduction degenerates and `acos` cancels.

    ``acos(x) = pi/2 - asin(x)`` loses almost every significant bit here, which is why the
    reflected form is used above 1/2. A regression to the naive form shows up in this test and
    nowhere else in the suite, because the dense grid's step never gets this close to the endpoint.
    """
    xs = [np.float32(1.0), np.float32(-1.0)]
    for _ in range(64):
        xs.append(np.nextafter(xs[-2], np.float32(0.0)))
        xs.append(np.nextafter(xs[-2], np.float32(0.0)))
    arr = np.unique(np.array(xs, dtype=np.float32))
    vk, cpu = _run(op, arr)
    u = _ulp(vk, cpu)
    worst = float(np.max(u[np.isfinite(u)]))
    assert worst <= STATED_ULP_BOUND, (
        f"{op}: worst {worst:.1f} ULP within 64 ULP of |x|=1 — the reduction is unstable there"
    )


# ---------------------------------------------------------------------------
# Boundary behaviour, asserted on bits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["Asin", "Acos"])
def test_endpoints_and_signed_zero_are_bit_exact(op: str) -> None:
    """+-1, +-0 must match the CPU EP bit for bit, including the sign of zero.

    Bits rather than values: ``repr(np.float32(-0.0))`` prints ``0.0``, so a lost signed zero is
    invisible in any float-formatted comparison and in ``np.allclose``. ONNX inherits C's
    ``asin(-0) == -0``; ``sign(x) * r`` in the shader would have broken exactly this.
    """
    xs = np.array([-1.0, -0.0, 0.0, 1.0], dtype=np.float32)
    vk, cpu = _run(op, xs)
    assert [f"0x{v:08x}" for v in vk.view(np.uint32)] == [
        f"0x{v:08x}" for v in cpu.view(np.uint32)
    ], f"{op}: endpoint/signed-zero bits differ from the CPU EP"


@pytest.mark.parametrize("op", ["Asin", "Acos"])
def test_out_of_domain_and_nan_are_nan(op: str) -> None:
    """|x| > 1, +-inf and NaN all produce NaN, as the CPU EP does.

    The shader clamps the radicand so ``sqrt`` never sees a negative operand — GLSL leaves that
    undefined — and reaches NaN through an explicit domain test instead. This asserts the
    behaviour, not the mechanism, so a different mechanism that also works is allowed.
    """
    xs = np.array(
        [1.0000001, -1.0000001, 2.0, -2.0, 1e30, np.inf, -np.inf, np.nan], dtype=np.float32
    )
    vk, cpu = _run(op, xs)
    assert np.all(np.isnan(cpu)), "precondition: the CPU EP should return NaN for all of these"
    bad = [float(x) for x, v in zip(xs, vk) if not np.isnan(v)]
    assert not bad, f"{op}: returned a non-NaN for out-of-domain inputs {bad}"


# ---------------------------------------------------------------------------
# Properties — shared math, not two drifting approximations
# ---------------------------------------------------------------------------


def test_asin_plus_acos_is_pi_over_two() -> None:
    """``asin(x) + acos(x) == pi/2`` to within both ops' bounds combined.

    This is the test that would catch `Asin` and `Acos` being fixed independently and drifting
    apart. They share ``ew_asin_core`` precisely so this holds by construction.
    """
    xs = _dense_grid(20001)
    asin_vk, _ = _run("Asin", xs)
    acos_vk, _ = _run("Acos", xs)
    total = asin_vk.astype(np.float64) + acos_vk.astype(np.float64)
    u = _ulp(total.astype(np.float32), np.full(xs.shape, np.float32(np.pi / 2)))
    worst = float(np.max(u[np.isfinite(u)]))
    assert worst <= 2 * STATED_ULP_BOUND, (
        f"asin(x) + acos(x) is {worst:.1f} ULP from pi/2 — the two ops are not sharing a core"
    )


@pytest.mark.parametrize(
    "op,relation",
    [("Asin", "odd"), ("Acos", "pi-reflected")],
    ids=["asin-is-odd", "acos-reflects-through-pi"],
)
def test_symmetry(op: str, relation: str) -> None:
    """``asin(-x) == -asin(x)`` exactly; ``acos(-x) == pi - acos(x)`` to within the bound.

    `asin`'s is exact because the sign is a bit-copy applied to a magnitude computed from |x|;
    if it ever stops being exact, the sign handling has moved back into the arithmetic.
    """
    xs = _dense_grid(20001)
    xs = xs[xs > 0]
    pos, _ = _run(op, xs)
    neg, _ = _run(op, (-xs).astype(np.float32))
    if relation == "odd":
        assert np.array_equal(neg.view(np.uint32), (-pos).view(np.uint32)), (
            "asin(-x) is not the exact negation of asin(x)"
        )
    else:
        u = _ulp(neg, (np.float32(np.pi) - pos).astype(np.float32))
        worst = float(np.max(u[np.isfinite(u)]))
        assert worst <= 2 * STATED_ULP_BOUND, f"acos(-x) vs pi - acos(x): {worst:.1f} ULP"


# ---------------------------------------------------------------------------
# Negative controls — the bound has power, and the built-in really failed
# ---------------------------------------------------------------------------


def test_bound_rejects_the_builtin_error_magnitude() -> None:
    """A check observed only to pass is not known to be a check.

    Perturb a correct answer by the error Mesa lavapipe's built-in actually produced (3831 ULP on
    `Asin`, max 3.90e-4 absolute) and require both the derived ULP bound and the op-table's
    ``FP32_TRANSCENDENTAL`` to reject it. If either accepts it, the green lanes above mean nothing.
    """
    xs = _dense_grid(4001)
    exact = np.arcsin(xs.astype(np.float64)).astype(np.float32)
    perturbed = (exact + np.float32(3.9e-4)).astype(np.float32)

    u = _ulp(perturbed, exact)
    assert float(np.max(u[np.isfinite(u)])) > STATED_ULP_BOUND, (
        "the derived ULP bound accepts lavapipe's measured built-in error — it is not a check"
    )
    assert not np.allclose(
        perturbed, exact, **m.FP32_TRANSCENDENTAL
    ), "FP32_TRANSCENDENTAL accepts lavapipe's measured built-in error"


@pytest.mark.parametrize("op", ["Asin", "Acos"])
def test_builtin_negative_control_artifact_is_red(op: str) -> None:
    """The recorded lavapipe sweep of the *built-in* path must fail the bound.

    Provenance is in ``bench/results/inverse_trig_ulp_lavapipe_builtin.json``: taken on
    ``llvmpipe (LLVM 22.1.8, 256 bits)`` with the shader tree at 94a4bd6, i.e. the code this
    branch replaced. Recorded rather than re-run because reproducing it needs a second build of
    the EP from a different commit, which no test can do.
    """
    path = _ARTIFACTS / "inverse_trig_ulp_lavapipe_builtin.json"
    if not path.is_file():
        pytest.skip(f"negative-control artifact not present: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))["ops"][op]
    assert row["max_ulp_vs_cpu"] > STATED_ULP_BOUND, (
        f"the recorded built-in sweep for {op} reads {row['max_ulp_vs_cpu']} ULP, inside the "
        f"bound. Either the artifact is not the built-in path, or the bound is too loose to have "
        f"caught the defect it was written for."
    )


@pytest.mark.parametrize("op", ["Asin", "Acos"])
def test_portable_artifact_is_green(op: str) -> None:
    """The positive arm of the same pair, on the same device, so the comparison is like-for-like."""
    path = _ARTIFACTS / "inverse_trig_ulp_lavapipe_portable.json"
    if not path.is_file():
        pytest.skip(f"artifact not present: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["verdict"] == "MEASURED", doc["verdict"]
    row = doc["ops"][op]
    assert row["max_ulp_vs_cpu"] <= STATED_ULP_BOUND
    assert row["edge_cases"]["all_agree"], f"{op}: recorded edge cases disagree with the CPU EP"


# ---------------------------------------------------------------------------
# The screen — issue #4 acceptance: every loosely-bounded built-in, with a decision
# ---------------------------------------------------------------------------

#: Every GLSL.std.450 built-in the shader tree may call, with what Vulkan's "Precision of
#: GLSL.std.450 Instructions" table promises for it in single precision and what this project
#: decided to do about that. `test_builtin_screen_is_complete` fails if a shader starts calling
#: something that is not in here, so the decision cannot be skipped by adding a call.
#:
#: "tight" = correctly rounded, exact, or a small constant ULP count.
#: "loose" = an allowance wide enough to miss FP32_TRANSCENDENTAL (1e-5) on a conformant driver.
BUILTIN_SCREEN: dict[str, tuple[str, str, str]] = {
    # name        verdict   spec allowance (single precision)              decision
    "abs": ("tight", "exact", "keep"),
    "sign": ("tight", "exact", "keep"),
    "floor": ("tight", "exact", "keep"),
    "ceil": ("tight", "exact", "keep"),
    "roundEven": ("tight", "correctly rounded", "keep — ONNX Round is half-to-even"),
    "fract": ("tight", "correctly rounded", "keep"),
    "mod": ("tight", "inherited from x - y*floor(x/y)", "keep"),
    "min": ("tight", "exact", "keep"),
    "max": ("tight", "exact", "keep"),
    "clamp": ("tight", "exact", "keep"),
    # Not GLSL.std.450 at all: core SPIR-V integer atomics and a memory barrier. They carry no
    # floating-point rounding, so the precision table has nothing to say about them. Screened
    # here rather than waved through so the screen stays a complete census of what we call.
    "atomicAnd": ("tight", "exact (integer atomic, not in the precision table)", "keep"),
    "atomicOr": ("tight", "exact (integer atomic, not in the precision table)", "keep"),
    "memoryBarrierShared": ("tight", "n/a (memory ordering, not arithmetic)", "keep"),
    "mix": ("tight", "inherited from fmul/fadd", "keep"),
    "inversesqrt": ("tight", "2 ULP", "keep"),
    "sqrt": ("tight", "inherited from 1.0/inversesqrt(): <= ~4.5 ULP", "keep"),
    "fma": ("tight", "inherited from OpFMul then OpFAdd", "keep"),
    "log": ("tight", "3 ULP outside [0.5,2]; abs < 2^-21 inside", "keep"),
    "log2": ("tight", "3 ULP outside [0.5,2]; abs < 2^-21 inside", "keep"),
    "packHalf2x16": ("tight", "correctly rounded conversion", "keep"),
    "unpackHalf2x16": ("tight", "exact widening", "keep"),
    # Loose, but bounded by a term that stays small over the ranges these ops are used on.
    "exp": (
        "loose-bounded",
        "3 + 2*|x| ULP — grows without limit in |x|",
        "keep; ~13 ULP at |x|=5 (1.5e-6 rel). Revisit if an op feeds it |x| >> 10.",
    ),
    "pow": (
        "loose-bounded",
        "inherited from exp2(y * log2(x))",
        "keep; same reasoning as exp, and only Erf/activations reach it.",
    ),
    # Loose enough that a conformant driver can miss the contract. Only asin/acos have been
    # observed to do so; the rest are recorded, not fixed, and named as follow-on work.
    "asin": (
        "loose-unusable",
        "inherited from atan2(x, sqrt(1-x*x)) -> 4096 ULP (~2.4e-4 rel)",
        "REPLACED — ew_asin() in templates/ew_unary.comp (issue #4)",
    ),
    "acos": (
        "loose-unusable",
        "inherited from atan2(sqrt(1-x*x), x) -> 4096 ULP (~2.4e-4 rel)",
        "REPLACED — ew_acos() in templates/ew_unary.comp (issue #4)",
    ),
    "atan": (
        "loose-unusable",
        "4096 ULP (~2.4e-4 rel)",
        "NOT REPLACED — green on lavapipe and NVIDIA today, but only by driver goodwill. "
        "Follow-on: same treatment as asin/acos.",
    ),
    "sin": (
        "loose-unusable",
        "absolute error <= 2^-11 (4.9e-4) inside [-pi, pi]",
        "NOT REPLACED — green on both devices today; the allowance is 49x FP32_TRANSCENDENTAL's "
        "atol. Follow-on.",
    ),
    "cos": (
        "loose-unusable",
        "absolute error <= 2^-11 (4.9e-4) inside [-pi, pi]",
        "NOT REPLACED — as sin. Follow-on.",
    ),
    "tan": ("loose-unusable", "inherited from sin()/cos()", "NOT REPLACED — as sin. Follow-on."),
    "sinh": ("loose-bounded", "inherited from (exp(x)-exp(-x))*0.5", "keep; inherits exp's bound."),
    "cosh": ("loose-bounded", "inherited from (exp(x)+exp(-x))*0.5", "keep; inherits exp's bound."),
    "tanh": ("loose-bounded", "inherited from sinh()/cosh()", "keep; inherits exp's bound."),
    "asinh": ("tight", "inherited from log(x + sqrt(x*x + 1))", "keep — log and sqrt are tight."),
    "acosh": ("tight", "inherited from log(x + sqrt(x*x - 1))", "keep — log and sqrt are tight."),
    "atanh": ("tight", "inherited from log((1+x)/(1-x))*0.5", "keep — log is tight."),
    "isnan": ("tight", "exact", "keep"),
    "isinf": ("tight", "exact", "keep"),
    "floatBitsToUint": ("tight", "exact reinterpretation", "keep"),
    "uintBitsToFloat": ("tight", "exact reinterpretation", "keep"),
}

#: Built-ins whose Vulkan allowance is wide enough to miss FP32_TRANSCENDENTAL. Every one of these
#: must carry an explicit decision above — "REPLACED" or "NOT REPLACED" with a reason.
_LOOSE = "loose-unusable"

_CALL = re.compile(r"(?<![A-Za-z0-9_])([a-zA-Z][a-zA-Z0-9_]*)\s*\(")

#: Comments are prose. `range (...)`, `set (...)` and `per (...)` all read as calls to a regex,
#: and this file is full of prose about precision, so strip comments before scanning.
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)

#: Identifiers that are GLSL keywords, project functions, macros or types rather than built-ins.
#: Anything matching `_CALL` and not in `BUILTIN_SCREEN` is reported unless it is here or is
#: defined in the shader tree itself (which the test detects).
_NOT_A_BUILTIN = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "defined",
        "layout",
        "main",
        "barrier",
        # Types used as constructors. `float`/`int`/`uint`/`bool` are handled separately below
        # because they also appear as declaration keywords.
        "int64_t",
        "uint64_t",
        "float16_t",
        "f16vec2",
        "i16vec2",
        # Names token-pasted into existence by the shared load/store macros in `indexing.glsl`.
        # They resolve to project code, not to GLSL.std.450 entry points.
        "load0",
        "load1",
        "load2",
        "load_cond",
    }
)


def _code_only(text: str) -> str:
    """The shader source with comments removed, so prose does not read as a call."""
    return _COMMENT.sub(" ", text)


def _shader_sources() -> list[Path]:
    return sorted(_SHADER_DIR.rglob("*.comp")) + sorted(_SHADER_DIR.rglob("*.glsl"))


def test_builtin_screen_is_complete() -> None:
    """Every function the shader tree calls is either project-defined or screened above.

    Issue #4's acceptance asks for a decision on each loosely-bounded built-in. A prose list would
    go stale the first time someone added a call; this reads the tree. If it fails, add the
    built-in to ``BUILTIN_SCREEN`` with its allowance from Vulkan's precision table and a decision
    — do not add it to ``_NOT_A_BUILTIN`` unless it genuinely is not a built-in.
    """
    sources = _shader_sources()
    assert sources, f"no shader sources found under {_SHADER_DIR}"

    # Functions and macros defined inside the tree are not built-ins.
    defined: set[str] = set()
    for path in sources:
        text = _code_only(path.read_text(encoding="utf-8"))
        defined |= set(re.findall(r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M))
        defined |= set(
            re.findall(
                r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{",
                text,
                re.M,
            )
        )

    unscreened: dict[str, set[str]] = {}
    for path in sources:
        for name in set(_CALL.findall(_code_only(path.read_text(encoding="utf-8")))):
            if name in BUILTIN_SCREEN or name in defined or name in _NOT_A_BUILTIN:
                continue
            if name.isupper() or name in {"float", "int", "uint", "bool", "vec2", "uvec2"}:
                continue  # macro invocation or a constructor, not a built-in call
            unscreened.setdefault(name, set()).add(path.name)

    assert not unscreened, (
        "the shader tree calls functions with no entry in BUILTIN_SCREEN: "
        + ", ".join(f"{k} ({', '.join(sorted(v))})" for k, v in sorted(unscreened.items()))
        + ". Add each with its allowance from Vulkan's 'Precision of GLSL.std.450 Instructions' "
        "table and a decision (issue #4 acceptance)."
    )


def test_loosely_bounded_builtins_carry_a_decision() -> None:
    """Each `loose-unusable` entry says REPLACED or NOT REPLACED, and names why."""
    for name, (verdict, allowance, decision) in BUILTIN_SCREEN.items():
        assert allowance, f"{name}: no spec allowance recorded"
        if verdict != _LOOSE:
            continue
        assert decision.startswith(("REPLACED", "NOT REPLACED")), (
            f"{name} is loosely bounded but its decision does not start with REPLACED or "
            f"NOT REPLACED: {decision!r}"
        )


def test_asin_acos_no_longer_call_the_builtin() -> None:
    """`ew_unary.comp` must not reach GLSL's `asin`/`acos` on the ops that were replaced.

    The ULP tests above would also catch a revert, but only on a driver that misses the contract —
    on NVIDIA the built-in measures 4-5 ULP and every one of them would stay green. This one is
    read off the source and fails everywhere.
    """
    text = (_SHADER_DIR / "glsl" / "templates" / "ew_unary.comp").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )
    for name in ("asin", "acos"):
        calls = re.findall(rf"(?<![A-Za-z0-9_]){name}\s*\(", body)
        assert not calls, (
            f"ew_unary.comp still calls the GLSL built-in {name}() — Vulkan allows it 4096 ULP "
            f"by inheritance from atan2(), which is what issue #4 removed."
        )
    assert "ew_asin_core(" in body, "the shared minimax core is gone from ew_unary.comp"
