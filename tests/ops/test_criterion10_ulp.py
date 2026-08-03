"""The unit of the criterion-10 residual: ULPs, and why relative error is not it.

THE DEFECT THIS FIXES IS THE UNIT, NOT THE KERNELS
==================================================
Morpheus, 2026-08-02, having checked the premise of the question he was asked before
reasoning about it — f16 kernels already accumulate in fp32 and always have, so fp16 is a
*storage* format here and never was an accumulation format:

    Of the 65 per-output residuals, 64 are exact negative powers of two and the 65th is
    3 * 2**-9 — small integer multiples of the fp16 ULP.  KV magnitude grows with depth,
    the ULP grows with it, so the absolute residual rises with depth *for a correct
    implementation*.  The curve is a plot of magnitude.

    Express the residual in ULPs.  **Predicted flat at 1-3 across all 32 layers.**
    Flat => no defect; a step => a located one.

The prediction is on record in ``bench/results/criterion10-ulp-prediction.md``, written
before any ULP number existed.

THIS IS NOT A RELAXATION
========================
A 1-ULP residual at layer 0 passes comfortably on an absolute ``atol`` that is generous at
small magnitudes; in ULPs it has nowhere to hide.  Fixing the unit may make the gate
*tighter*.  It is also why this does not fall under Switch's earlier refusal: he declined
a scale-set tolerance because it lost to the incumbent at 9.3x against a 10x bar, missing
"1 ULP added everywhere" — and a ULP-denominated residual is exactly the observable that
catches that.

WHAT THIS FILE ASSERTS
======================
The arms below are all GPU-free and test the *instrument*, not the EP.  The load-bearing
one is ``test_ulp_does_not_blow_up_near_zero_which_is_the_whole_argument_for_the_unit``:
relative error's failure near zero is the reason for the change, so if ULP inherited it
the change would be ornamental.

NOT DONE HERE, DELIBERATELY
===========================
``atol`` is untouched and the criterion-10 verdict is not flipped.  ``DIVERGENT`` is
honest right now; the verdict changes when the unit is right, or it does not change.
"""

from __future__ import annotations

import numpy as np
import pytest

import _models as m


# ---------------------------------------------------------------------------
# The unit itself
# ---------------------------------------------------------------------------


def test_one_ulp_apart_reads_as_one_ulp_at_every_magnitude():
    """The defining property: the unit is scale-free where the absolute bound is not.

    Two fp16 values one representable step apart must read 1.0 ULP whether they sit at
    0.001 or at 1000 — which is precisely what an absolute ``atol`` cannot do, and the
    reason the monotone curve looked like a defect.
    """
    for magnitude in (0.001, 0.5, 1.0, 17.0, 1000.0):
        base = np.float16(magnitude)
        stepped = np.nextafter(base, np.float16(np.inf), dtype=np.float16)
        ulps, _ = m.ulp_residual(np.array([stepped]), np.array([base]))
        assert ulps[0] == pytest.approx(1.0), (magnitude, ulps[0])


def test_the_absolute_residual_of_one_ulp_grows_with_magnitude_which_is_the_whole_point():
    """The observation Morpheus made, asserted rather than quoted.

    A one-step residual is a *rising* absolute number as magnitude rises.  So a monotone
    absolute curve across 32 layers of growing KV magnitude is what a correct
    implementation produces, and reading it as a defect is reading a plot of magnitude.
    """
    absolutes = []
    for magnitude in (0.5, 4.0, 32.0, 256.0):
        base = np.float16(magnitude)
        stepped = np.nextafter(base, np.float16(np.inf), dtype=np.float16)
        absolutes.append(abs(float(stepped) - float(base)))
        ulps, _ = m.ulp_residual(np.array([stepped]), np.array([base]))
        assert ulps[0] == pytest.approx(1.0)
    assert absolutes == sorted(absolutes), absolutes
    assert absolutes[-1] > absolutes[0] * 100, absolutes


def test_ulp_survives_a_denormal_scale_residual_where_relative_error_does_not():
    """Half of the argument for the unit — and only half, which the next arm states.

    Below the smallest normal, spacing is constant, so a residual that is *itself* of
    denormal size reads 1 ULP where a bare relative error reads 1e20.
    """
    tiny = np.float16(6e-8)  # one denormal step
    ulps, basis = m.ulp_residual(np.array([tiny], np.float16), np.array([0.0], np.float16))
    assert ulps[0] == pytest.approx(1.0), (ulps[0], basis)

    relative = abs(float(tiny)) / 1e-30
    assert relative > 1e20, relative


def test_ulp_DOES_blow_up_at_a_cancellation_element_which_i_first_claimed_it_could_not():
    """R11, on my own instrument, kept executable.

    My first docstring for ``ulp_residual`` said a near-zero element "cannot manufacture a
    large ULP count" because spacing floors.  Spacing does floor — and the claim is still
    false in the case that matters: when the reference element **cancels to zero while the
    tensor's scale is ~1**, a residual of one ULP *at the tensor's scale* is measured
    against the denormal spacing and reads 16384 ULP.

    So ``max_ulp_diff`` alone cannot acquit — Morpheus's sentence pointed at my own
    instrument.  The consumer records median, p99 and a cancellation count beside it.
    """
    cpu = np.array([1.0, 0.0], np.float16)
    vk = np.array([1.0, 2.0**-10], np.float16)
    ulps, _ = m.ulp_residual(vk, cpu)
    assert ulps[0] == 0.0
    assert ulps[1] > 10_000, ulps[1]


def test_the_record_reports_a_distribution_so_one_cancellation_element_cannot_dominate():
    """The fix for the arm above: flat-at-1-3 must be readable off a distribution.

    A tensor with a single cancellation element and an otherwise one-ULP residual must
    show a huge ``max_ulp_diff`` **and** a small median, with the cancellation counted —
    so a reader can tell "one bad element" from "wrong everywhere" without being told.
    """
    rng = np.random.default_rng(7)
    cpu = (rng.standard_normal(512) + 4.0).astype(np.float16)
    cpu[17] = np.float16(0.0)
    vk = np.array([np.nextafter(x, np.float16(np.inf), dtype=np.float16) for x in cpu],
                  dtype=np.float16)
    vk[17] = np.float16(2.0**-10)
    _, facts = _pair(vk, cpu)
    entry = facts["per_output"][0]
    assert entry["max_ulp_diff"] > 10_000, entry["max_ulp_diff"]
    assert entry["median_ulp_diff"] == pytest.approx(1.0), entry["median_ulp_diff"]
    assert entry["ulp_cancellation_elements"] >= 1


def test_zero_residual_is_zero_ulps_and_not_a_division_by_zero():
    ulps, _ = m.ulp_residual(np.zeros(8, np.float16), np.zeros(8, np.float16))
    assert np.all(ulps == 0.0)
    assert np.all(np.isfinite(ulps))


def test_the_spacing_basis_is_the_oracle_not_our_own_output():
    """A wrong answer must not get to choose its own denominator.

    That is the failure this project's verdict machinery exists to prevent one level down,
    and it would be an odd thing to reintroduce in the tolerance.  Asserted by making the
    two sides differ in magnitude enough that the choice of basis changes the count.
    """
    cpu = np.array([1.0], np.float16)
    vk = np.array([1024.0], np.float16)
    ulps_correct, _ = m.ulp_residual(vk, cpu)
    ulps_if_we_chose, _ = m.ulp_residual(cpu, vk)
    assert ulps_correct > ulps_if_we_chose * 100, (ulps_correct, ulps_if_we_chose)


def test_fp32_and_fp16_are_measured_in_their_own_dtypes_ulp():
    """One ULP of fp32 is not one ULP of fp16; the basis is the stored dtype."""
    for dt in (np.float16, np.float32):
        base = dt(1.0)
        stepped = np.nextafter(base, dt(np.inf), dtype=dt)
        ulps, basis = m.ulp_residual(np.array([stepped], dt), np.array([base], dt))
        assert ulps[0] == pytest.approx(1.0)
        assert str(np.dtype(dt)) in basis


def test_an_integral_tensor_counts_one_as_one_ulp():
    ulps, basis = m.ulp_residual(np.array([5], np.int64), np.array([3], np.int64))
    assert ulps[0] == 2.0
    assert "defect" in basis


# ---------------------------------------------------------------------------
# The record — the headline moves, the demoted number stays visible
# ---------------------------------------------------------------------------


def _pair(vk: np.ndarray, cpu: np.ndarray):
    return m.compare_all_outputs_to_cpu([vk], [cpu])


def test_the_per_output_record_names_its_headline_and_disowns_the_old_one():
    """`max_rel_diff` misled two people today, including the coordinator.

    It is kept — deleting a number others have already quoted makes a record unreadable —
    but it must arrive labelled, with the reason attached, so nobody has to already know.
    """
    rng = np.random.default_rng(0)
    cpu = rng.standard_normal(64).astype(np.float16)
    vk = cpu.copy()
    _, facts = _pair(vk, cpu)
    entry = facts["per_output"][0]
    assert entry["headline_statistic"] == "median_ulp_diff"
    assert entry["max_rel_diff_is_headline"] is False
    assert "denominator artefact" in entry["max_rel_diff_caveat"]
    assert "max_ulp_diff" in entry
    # The new unit must arrive with its own weakness stated, or it is the old artefact
    # wearing a fresh unit (R11): a decomposition that appears to close.
    assert "cancellation-sensitive" in entry["headline_note"]
    for name in ("p99_ulp_diff", "max_ulp_diff", "ulp_cancellation_elements"):
        assert name in entry["headline_secondary"], entry["headline_secondary"]
        assert name in entry


def test_relative_error_is_shown_to_be_the_artefact_the_caveat_claims_it_is():
    """The four curves, on one specimen that models the real situation.

    Four "layers" of geometrically growing magnitude, each with a residual of **exactly one
    ULP** in the bulk — and a single **cancellation element at layer 2 only**, where the
    reference cancels to ~0 while the tensor's scale is not.  That is what produces the
    0.4559 in the ruling, and it is the shape the criterion must read correctly.

    * ``max_abs_diff``    rises monotonically — a plot of magnitude, which is the mistake.
    * ``max_rel_diff``    spikes at layer 2 and is **not monotone** — the ruling's complaint.
    * ``max_ulp_diff``    **also spikes at layer 2.**  Stated, not hidden: max ULP is
      cancellation-sensitive too, so swapping one max for another would have been the old
      artefact in a fresh costume (R11).
    * ``median_ulp_diff`` is **flat at 1** through the spike — the statistic that actually
      delivers "flat at 1-3", and therefore the headline.

    Two of my earlier attempts at this arm were wrong and both are worth naming.  The first
    asserted our ``max_rel_diff`` diverges near zero; it does not, because it divides by
    ``atol + |b|`` and is atol-floored.  The second used a one-ULP residual everywhere and
    found ``max_rel_diff`` flat — correctly, because the relative error of a one-ULP
    residual is scale-free.  The non-monotonicity needs a cancellation element to exist at
    all, and finding that out is what located the right headline.
    """
    abs_curve, rel_curve, max_ulp_curve, med_ulp_curve = [], [], [], []
    for layer, magnitude in enumerate((0.25, 2.0, 16.0, 128.0)):
        rng = np.random.default_rng(100 + layer)
        cpu = (rng.standard_normal(256) * 0.1 + magnitude).astype(np.float16)
        vk = np.array(
            [np.nextafter(x, np.float16(np.inf), dtype=np.float16) for x in cpu],
            dtype=np.float16,
        )
        if layer == 2:
            cpu[5] = np.float16(0.0)
            vk[5] = np.float16(2.0**-10)
        _, facts = _pair(vk, cpu)
        entry = facts["per_output"][0]
        abs_curve.append(entry["max_abs_diff"])
        rel_curve.append(entry["max_rel_diff"])
        max_ulp_curve.append(entry["max_ulp_diff"])
        med_ulp_curve.append(entry["median_ulp_diff"])

    assert abs_curve == sorted(abs_curve), abs_curve
    assert abs_curve[-1] > abs_curve[0] * 100, abs_curve

    assert rel_curve != sorted(rel_curve), rel_curve
    assert rel_curve[2] > rel_curve[3] * 10, rel_curve

    # Honesty about the new unit: its max spikes on the same element.
    assert max_ulp_curve[2] > max_ulp_curve[3] * 100, max_ulp_curve

    # And the statistic that survives it.
    assert all(u == pytest.approx(1.0) for u in med_ulp_curve), med_ulp_curve


def test_the_facts_carry_a_worst_ulp_index_that_can_differ_from_the_worst_abs_index():
    """Two different questions, so two different indices, so two different keys.

    The artifact this replaces carried one worst-index and invited the reader to assume it
    answered both.  Both tensors carry a one-ULP residual; only their magnitudes differ.

    (The first version of this arm used constant tensors and was correctly caught by the
    degeneracy guard — two constants compare equal to any tolerance, so nothing was
    compared and both indices were ``None``.  Left in the record because the guard doing
    its job to my own fixture is worth more than a tidy history.)
    """
    small = np.array([1.0, 1.25, 1.5, 1.75], np.float16)
    big = np.array([1024.0, 1088.0, 1152.0, 1216.0], np.float16)
    vk_small = small.copy()
    vk_small[0] = np.nextafter(small[0], np.float16(np.inf), dtype=np.float16)
    vk_big = big.copy()
    vk_big[0] = np.nextafter(big[0], np.float16(np.inf), dtype=np.float16)
    _, facts = m.compare_all_outputs_to_cpu([vk_small, vk_big], [small, big])
    assert facts["oracle_outputs_compared"] == 2, facts["per_output"]
    # index 1 has the larger absolute residual; both are 1 ULP.
    assert facts["oracle_worst_output_index"] == 1
    assert facts["oracle_max_ulp_diff_over_all_outputs"] == pytest.approx(1.0)
    assert facts["oracle_max_abs_diff_over_all_outputs"] > 0.5


def test_the_ulp_statistic_does_not_move_the_pass_fail_decision_yet():
    """Explicitly asserted, because the constraint is easy to violate by accident.

    Criterion 10 stays open on the unit alone and ``DIVERGENT`` is honest right now.  The
    ULP number is *recorded*; ``atol`` is untouched and ``within`` is still decided by
    ``np.allclose`` on the incumbent tolerance.  If a future change makes the gate
    ULP-denominated that is a deliberate act, and this arm should fail and be rewritten
    rather than quietly pass.
    """
    import inspect

    src = inspect.getsource(m.compare_all_outputs_to_cpu)
    within_line = [ln for ln in src.splitlines() if "within = " in ln]
    assert len(within_line) == 1, within_line
    assert "np.allclose" in within_line[0]
    assert "ulp" not in within_line[0].lower(), (
        "the gate has become ULP-denominated; that is a deliberate act and this arm must "
        "be rewritten to say so, not deleted"
    )


def test_the_exceedance_rule_uses_the_prediction_and_not_a_number_i_chose_afterwards():
    """The predicate is Morpheus's band, recorded before the instrument existed.

    My first version of this rule was "more than 3x the observed baseline". I wrote it
    before dumping the real curve. When I dumped it, the deepest two KV outputs read 4
    against a baseline of 1 -- so my rule flagged them, and my next instinct was to widen
    the multiple until it did not. **That instinct is the defect**: a threshold chosen
    after seeing the data is a threshold fitted to the data, and it would have made the
    instrument incapable of contradicting me. So the predicate is the prediction: flat at
    1-3, his number, on record in bench/results/criterion10-ulp-prediction.md before a
    single ULP had been measured.

    An exceedance therefore reports *itself* and lets the reader see its size. A smooth
    accumulation curve that tops out at 4 and a step that lands at 12 both exceed the
    band; ``multiple_of_baseline`` is what separates a one-ULP overshoot from a located
    defect, and it is reported rather than thresholded.
    """
    drift = [1.0] * 48 + [2.0] * 12 + [3.0, 3.0, 4.0, 4.0]
    drift_out = m.ulp_outliers(drift)
    assert [o["output_index"] for o in drift_out] == [62, 63]
    assert all(o["multiple_of_baseline"] == pytest.approx(4.0) for o in drift_out)

    step = [1.0] * 63 + [12.0]
    step_out = m.ulp_outliers(step)
    assert [o["output_index"] for o in step_out] == [63]
    assert step_out[0]["multiple_of_baseline"] == pytest.approx(12.0)

    # The distinction the record must preserve: same predicate, very different finding.
    assert step_out[0]["multiple_of_baseline"] >= 3 * drift_out[0]["multiple_of_baseline"]

    # R12: an empty curve is UNOBSERVABLE, never "no outliers".
    assert m.ulp_outliers([]) == m.OUTPUT_COVERAGE_NOT_COMPUTED


def test_the_suspect_output_is_kept_out_of_its_own_denominator():
    """A wrong answer must not choose its own basis.

    Output 0 is the output under suspicion, so it is excluded from the baseline it is
    measured against -- the same rule that makes :func:`ulp_residual` denominate in the
    oracle's spacing and never in ours. Without the exclusion a single large output drags
    the baseline toward itself and shrinks its own multiple.
    """
    curve = [12.0] + [1.0] * 4
    with_exclusion = m.ulp_outliers(curve)
    without = m.ulp_outliers(curve, exclude_from_baseline=())
    assert with_exclusion[0]["baseline_median_ulp_diff"] == pytest.approx(1.0)
    assert with_exclusion[0]["multiple_of_baseline"] == pytest.approx(12.0)
    assert without[0]["multiple_of_baseline"] <= with_exclusion[0]["multiple_of_baseline"]
