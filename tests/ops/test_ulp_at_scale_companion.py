#!/usr/bin/env python3
"""§8.9.24(3) as a gate: a ULP-at-scale figure may not stand without its allowance.

WHY THIS FILE EXISTS
====================
`docs/DESIGN.md` §8.9.24 refuted a finding I filed. The finding was
*"`atol=1e-3` is 0.128 ULP-at-scale on the logits -- the tolerance demands finer than fp16
can express"*, and it failed twice in the same sentence:

  * `atol` is ONE TERM of `np.allclose`'s allowance `atol + rtol*|b|`. At the logits' own
    scale the whole allowance is **33.628 ULP-at-scale**, not 0.128 -- a factor of 263. On
    `present.31.key` and `present.31.value` the factors are **116** and **506**.
  * `ULP-at-scale` divides by the spacing at the **tensor maximum** while the predicate
    evaluates **per element**.

Ruling (3) fenced the statistic rather than withdrawing it, and made me its owner: any row
reporting a residual in `ULP-at-scale` also reports (a) the allowance in the *same* unit
and (b) the failing set's residual on the element basis.

WHAT THESE ARMS DEFEND, AND WHY THE POLARITY ARM IS THE LOAD-BEARING ONE
========================================================================
A companion key that is merely *present* proves nothing: the obligation is only real if a
row missing it is **refused**. `test_the_check_refuses_an_incomplete_row` is therefore the
arm that matters, and `test_a_complete_row_is_accepted` is its control -- without it the
check could reject everything and every other arm here would still be green.

The **inversion** arm is the one that carries the correction's content: on a specimen
shaped like `present.31.key` -- failing elements at ~0.011 in a tensor whose scale is 5.77,
a ratio of ~500 -- the at-scale reading says *every failure is within one representable
step* while the element basis says *every failure is more than twenty representable steps
wide*. Both statements are computed from the same residuals. That is the whole of §8.9.24's
corollary, asserted rather than believed.

No device, no model, no artifact. This is a property of the comparator's arithmetic.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _models as m  # noqa: E402
from probe_criterion10_rescore import failing_element_census  # noqa: E402


# -- the refusal, and its control ---------------------------------------------------------


@pytest.mark.parametrize("dropped", sorted({c for v in m.ULP_AT_SCALE_COMPANIONS.values() for c in v}))
def test_the_check_refuses_an_incomplete_row(dropped: str) -> None:
    """Every companion in the table is individually load-bearing.

    Parameterised over the companions rather than asserted once, because a table entry
    nobody removes is a table entry nobody has tested.
    """
    row = {
        "failing_residual_within_one_ulp_at_scale": 4,
        "failing_max_ulp_at_scale": 0.55,
        "atol_in_ulps_at_scale": 0.256,
        "max_ulp_at_scale_diff": 0.55,
        "max_ulp_diff": 31.0,
        "allowance_in_ulps_at_scale_min": 0.31,
        "allowance_in_ulps_at_scale_median": 0.31,
        "allowance_in_ulps_at_scale_max": 29.8,
        "failing_ulp_element_basis_max": 41.0,
        "ulp_element_basis_max": 41.0,
    }
    m.assert_ulp_at_scale_row_is_complete(row, where="control")

    row.pop(dropped)
    with pytest.raises(m.UlpAtScaleCompanionError) as exc:
        m.assert_ulp_at_scale_row_is_complete(row, where="mutated")
    assert dropped in str(exc.value)
    assert "8.9.24(3)" in str(exc.value)


def test_a_none_valued_companion_is_as_absent_as_a_missing_one() -> None:
    """`None` is the shape a defaulted field takes, and it must not satisfy the obligation.

    This project has been told twice that a field nobody set reads identically to a field
    someone measured (`model_output_equivalence: MATCH` beside `UNMEASURED`, round 28).
    """
    row = {
        "max_ulp_at_scale_diff": 0.55,
        "max_ulp_diff": 31.0,
        "allowance_in_ulps_at_scale_min": None,
    }
    with pytest.raises(m.UlpAtScaleCompanionError):
        m.assert_ulp_at_scale_row_is_complete(row, where="defaulted")


def test_a_row_with_no_ulp_at_scale_figure_is_not_burdened() -> None:
    """The obligation attaches to the FIGURE, not to every row in the tree.

    Without this the check would be a tax on rows that make no at-scale claim, and it
    would be removed the first time it inconvenienced someone.
    """
    m.assert_ulp_at_scale_row_is_complete({"max_abs_diff": 0.2, "median_ulp_diff": 1.0})


# -- the inversion: the same residuals, two readings, opposite conclusions -----------------


def _layer31_shaped_specimen() -> tuple[np.ndarray, np.ndarray]:
    """A tensor whose scale is ~500x the magnitude of its failing elements.

    Built to `present.31.key`'s measured shape: scale 5.77, failing references ~0.011.
    """
    cpu = np.full(64, 5.75, dtype=np.float16)
    cpu[:4] = np.float16(0.011)
    vk = cpu.copy()
    vk[:4] = (cpu[:4].astype(np.float32) + 0.0025).astype(np.float16)
    return vk, cpu


def test_the_at_scale_reading_and_the_element_reading_disagree_by_more_than_twenty() -> None:
    """§8.9.24(1)'s corollary, computed rather than quoted.

    `failing_residual_within_one_ulp_at_scale == failing_elements` was read as *these fail
    by less than one representable step*. On the element basis the same failures are more
    than twenty steps wide. The two numbers come from one residual array and one
    predicate; only the denominator's location differs.
    """
    vk, cpu = _layer31_shaped_specimen()
    tol, _ = m.tolerance_for_output(vk)
    c = failing_element_census(vk, cpu, tol)

    assert c["failing_elements"] == 4
    # the reading that was made
    assert c["failing_residual_within_one_ulp_at_scale"] == c["failing_elements"]
    assert c["failing_max_ulp_at_scale"] <= 1.0
    # the reading the predicate's own granularity supports
    assert c["failing_ulp_element_basis_min"] > 20.0, c
    assert c["failing_ulp_element_basis_min"] / max(c["failing_max_ulp_at_scale"], 1e-30) > 20.0


def test_the_satisfiability_bound_holds_on_the_specimen_and_is_recomputed_not_quoted() -> None:
    """`allowance/ulp(b) >= rtol*2**10` -- measured on this tensor, not copied from prose.

    A bound this project quotes and never evaluates is a `PREDICTION` wearing a
    `MEASUREMENT`'s clothes, which §8.9.24(6) named this week.
    """
    vk, cpu = _layer31_shaped_specimen()
    tol, _ = m.tolerance_for_output(vk)
    c = failing_element_census(vk, cpu, tol)
    assert c["satisfiability_bound_element_basis"] == pytest.approx(tol["rtol"] * 1024.0)
    assert c["allowance_in_ulps_element_basis_min"] >= c["satisfiability_bound_element_basis"]


@pytest.mark.parametrize(
    "value",
    [2.0**-14, 1e-3, 0.011, 1.0, 5.75, 100.0, 1000.0, 32768.0, 65504.0],
)
def test_the_bound_is_independent_of_magnitude_across_the_whole_fp16_normal_range(value):
    """The one-line argument that settles satisfiability, swept.

    §8.9.24 states the minimum is 20.48 attained at |b| = 32768. If that is wrong anywhere
    in the fp16 normal range, the ruling's central claim is wrong and this fails.
    """
    b = np.float16(value)
    tol = m.MATMULNBITS_FP16
    allowance = tol["atol"] + tol["rtol"] * float(b)
    ulp = float(m.format_spacing(b, np.float16))
    assert allowance / ulp >= tol["rtol"] * 1024.0 - 1e-9, (value, allowance / ulp)


# -- the spacing-overflow defect this sweep found -----------------------------------------


def test_np_spacing_overflows_at_the_top_of_the_finite_range() -> None:
    """The ground truth for the arm below, stated against numpy rather than against us.

    Without this the repair looks like a defensive branch nobody can justify. `np.spacing`
    looks upward; at 65504 the next fp16 value is infinity, so it answers `inf`.
    """
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.spacing(np.float16(65504.0)))


def test_a_504_unit_error_at_fp16s_maximum_does_not_read_as_zero_ulp() -> None:
    """The first observable this project has found that fails in the ACQUITTING direction.

    Before the repair, `ulp_residual` and `ulp_at_scale_residual` both reported **0.0 ULP**
    for a 504-unit error at 65504, because the denominator was `inf`. Every other
    instrument defect found here (§8.9.22's degenerate denominator, §8.9.24's borrowed
    step) made a sound residual look wrong. This one made a wrong residual look sound,
    which is strictly worse: nobody re-derives a clean number.

    The verdict was never at risk -- `np.allclose` reads no ULP -- and that is the point
    §8.9.24(3) turns on: the *report* could have acquitted a saturating tensor while the
    *gate* failed it, and the two would have sat on the same row.
    """
    cpu = np.full(8, 65504.0, dtype=np.float16)
    vk = cpu.copy()
    vk[0] = np.float16(65000.0)

    elem, basis = m.ulp_residual(vk, cpu)
    at_scale, _ = m.ulp_at_scale_residual(vk, cpu)

    assert elem.max() > 10.0, elem
    assert at_scale.max() > 10.0, at_scale
    assert "overflows to inf" in basis
    # and the §8.9.24 bound survives the repair: ulp(b) <= |b| * 2**-10
    assert float(m.format_spacing(np.float16(65504.0), np.float16)) <= 65504.0 * 2.0**-10


def test_the_repaired_spacing_is_the_formats_own_step_and_not_an_invented_number() -> None:
    """32, not a fudge: the gap between 65504 and the previous representable fp16."""
    b = np.float16(65504.0)
    prev = np.nextafter(b, np.float16(-np.inf))
    assert float(m.format_spacing(b, np.float16)) == float(b) - float(prev) == 32.0


def test_the_repair_changes_nothing_away_from_the_boundary() -> None:
    """A repair that moved ordinary numbers would have invalidated every prior reading.

    Every ULP figure this project has published sits well inside the finite range, so the
    repair must be a no-op there or the record needs re-deriving rather than annotating.
    """
    rng = np.random.default_rng(3)
    vals = rng.normal(0, 4.0, size=4096).astype(np.float16)
    vals = vals[np.isfinite(vals)]
    with np.errstate(over="ignore"):
        old = np.abs(np.spacing(vals)).astype(np.float64)
    new = m.format_spacing(vals, np.float16)
    assert np.array_equal(old, new)


def test_a_subnormal_reference_gets_the_atol_term_and_not_a_starved_allowance() -> None:
    """The other half of the bound: subnormals are >= 16,777 ULP, from `atol` alone.

    This is where the unsatisfiability argument would have been strongest if it held, so
    it gets its own arm.
    """
    b = np.float16(2.0**-20)
    tol = m.MATMULNBITS_FP16
    ulp = abs(float(np.spacing(b)))
    assert tol["atol"] / ulp > 16000.0, tol["atol"] / ulp


# -- the comparator itself carries the companions -----------------------------------------


def test_the_comparator_emits_the_allowance_on_every_compared_output() -> None:
    """The obligation lands in the comparator, not only in the probe that misused it.

    §8.9.24(3) names "the comparator that already carries `verdict_predicate`" as the
    place. A row from `compare_all_outputs_to_cpu` is what a reader quotes, so it is the
    row that must be complete.
    """
    rng = np.random.default_rng(7)
    cpu = [
        rng.normal(0, 4.0, size=(1, 1, 128)).astype(np.float32),
        rng.normal(0, 1.0, size=(1, 4, 8, 16)).astype(np.float16),
    ]
    vk = [(a.astype(np.float64) * (1 + 1e-4)).astype(a.dtype) for a in cpu]
    _outcome, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    for e in facts["per_output"]:
        assert e["status"] in {"WITHIN_TOLERANCE", "OUTSIDE_TOLERANCE"}, e
        assert e["allowance_in_ulps_at_scale_min"] is not None, e
        assert e["ulp_element_basis_max"] is not None, e
        assert "rtol*2**10" in e["satisfiability_bound_note"]
        m.assert_ulp_at_scale_row_is_complete(e, where="comparator row")


def test_the_allowance_is_the_whole_sum_and_not_the_atol_term() -> None:
    """The arithmetic the finding got wrong, asserted as an inequality.

    On a tensor with any appreciable scale the allowance is orders above `atol` alone; a
    regression that silently reported the `atol` term under the allowance's name would
    make these equal.
    """
    cpu = np.full(256, 13.0859375, dtype=np.float16)
    cpu[0] = np.float16(0.011)
    tol, _ = m.tolerance_for_output(cpu)
    one_ulp = abs(float(np.spacing(np.float16(np.abs(cpu).max()))))
    d = m.allowance_in_ulps_at_scale(np.abs(cpu.astype(np.float64)), tol, one_ulp)
    atol_only = tol["atol"] / one_ulp
    assert d["allowance_in_ulps_at_scale_max"] / atol_only > 100.0, (d, atol_only)
    assert "WHOLE predicate allowance" in d["allowance_in_ulps_at_scale_basis"]


def test_an_undefined_scale_reports_absence_rather_than_zero() -> None:
    """A statistic over an undefined denominator is an instrument state, not a 0."""
    tol = m.MATMULNBITS_FP16
    d = m.allowance_in_ulps_at_scale(np.zeros(4), tol, 0.0)
    assert d["allowance_in_ulps_at_scale_min"] is None
    assert "reported as absent rather than as 0" in d["allowance_in_ulps_at_scale_basis"]
