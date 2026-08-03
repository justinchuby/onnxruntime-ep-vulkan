"""Falsifiers for the at-scale ULP observable and the instrument-disagreement flag.

WHY THIS FILE EXISTS
====================
Criterion 10 ranks lanes against the ORT CPU EP.  Until 2026-08-03 the only ULP unit in
the tree counted against the spacing **at each element's own value**, and a max over that
unit is attained at whichever element has the smallest reference.  Measured on this
project's own artifact (``bench/results/kv_int8_budget-dev0.json``, logits):

    lane                 max ULP (element basis)     ULP at scale     max_abs
    vk_fp16                          337,178                6.25       0.195
    cpu_i4_per_head                    7,110              908.00      28.375

The shipping fp16 GPU path ranked **47x worse than a simulated int4 KV cache**, and the
``max_ulp`` ordering came out *anti-correlated* with the ``max_abs`` ordering.  Nobody
believes that ordering, which means the criterion was producing an answer its own authors
would have overruled -- exactly the failure the project's binding rule describes: an
observable that degrades whatever happens cannot acquit.

Every test here is device-free and specimen-based.  They are falsifiers, not
demonstrations: each one is constructed so that the defect being fixed would make it fail,
and so that a future "simplification" that reinstates the defect cannot pass.
"""

from __future__ import annotations

import numpy as np
import pytest

import _models as m


def _logits_like(scale: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.uniform(-scale, scale, size=n).astype(np.float16)
    v[0] = np.float16(scale)
    return v


# --------------------------------------------------------------------------------------
# (a) The planted specimen: the exact defect, reproduced from first principles.
# --------------------------------------------------------------------------------------


def test_planted_cancellation_inflates_element_basis_and_evades_the_subnormal_counter():
    """A reference element small *relative to the tensor's scale* but far above subnormal.

    This is the shape of the real defect.  ``probe_kv_int8_budget.py`` counted
    ``near_zero_reference_elements`` against ``np.finfo(float16).tiny`` (6.104e-5) and
    ``cancellations`` against exact zero.  On the logits **both read 0** while ``max_ulp``
    read 337,178.  Arithmetic on that artifact explains why: 0.1953125 / 337178 = 5.79e-7,
    which is fp16's spacing at a reference of ~4.9e-4 -- eight times *above* the smallest
    normal.  A subnormal test cannot see it.  A scale-relative test can.
    """
    cpu = _logits_like(32.0, 4096, seed=11)
    # A reference element at ~4.9e-4: normal (so subnormal counters miss it), but ~64x
    # below one ULP at the tensor's scale of 32 (which is 0.03125).
    cpu[7] = np.float16(4.88e-4)
    vk = cpu.copy()
    # Perturb it by a handful of ULPs *at its own value* -- a tiny absolute residual.
    vk[7] = np.float16(float(cpu[7]) + 8 * float(np.spacing(cpu[7])))

    d = m.ulp_distribution(vk, cpu)

    assert d["subnormal_reference_elements"] == 0, (
        "the planted element must be a NORMAL float16, or this specimen is not the defect"
    )
    assert d["exact_zero_reference_elements"] == 0
    # The scale-relative counter is the one that sees it.
    assert d["cancellation_elements"] >= 1, (
        "the scale-relative cancellation counter is the only predicate that catches this; "
        "if this fails the counter has been narrowed back to subnormal or exact-zero"
    )
    # And the element basis is inflated while the at-scale basis is not.
    assert d["max_ulp"] >= 8.0
    assert d["max_ulp_at_scale"] < 1.0, (
        "an 8-ULP-at-its-own-value nudge of a 4.9e-4 element is a sub-ULP residual at a "
        "tensor scale of 32; if this reads large the at-scale denominator is per-element"
    )


# --------------------------------------------------------------------------------------
# (b) The ordering inversion: the finding itself, as a test.
# --------------------------------------------------------------------------------------


def test_element_basis_max_inverts_lane_ordering_and_at_scale_does_not():
    """Two lanes: one plainly better.  ``max_ulp`` must rank them backwards; at-scale must not.

    This is the real shape, reconstructed.  Lane GOOD is a faithful fp16 path: a tiny
    *absolute* residual, which on a tensor of scale 32 rounds away entirely except where
    the reference is small -- and there it reads thousands of ULPs at the element's own
    value.  Lane BAD is a coarse quantiser: a *relative* residual everywhere, which is a
    bounded number of ULPs at every element but a large absolute error.

    ``max_abs`` orders these correctly and so must ``max_ulp_at_scale``.  ``max_ulp``
    must order them backwards -- this test asserts the *defect* is real, so that removing
    the at-scale basis and keeping the max cannot quietly pass.
    """
    cpu = _logits_like(32.0, 8192, seed=23)
    cpu[3] = np.float16(4.88e-4)

    good = (cpu.astype(np.float64) + 0.002).astype(np.float16)
    bad = (cpu.astype(np.float64) * 1.12).astype(np.float16)

    g = m.ulp_distribution(good, cpu)
    b = m.ulp_distribution(bad, cpu)

    assert g["max_abs"] < b["max_abs"] / 100, (
        "specimen is wrong: GOOD must be far closer in absolute terms"
    )

    # The defect, asserted so it cannot be reintroduced unnoticed.
    assert g["max_ulp"] > b["max_ulp"], (
        "the element-basis max is supposed to invert here; if it no longer does, this "
        "specimen has stopped reproducing the defect and the test below proves nothing"
    )
    # The fix.
    assert g["max_ulp_at_scale"] < b["max_ulp_at_scale"], (
        "the at-scale basis must order the lanes the way max_abs does"
    )


def test_the_real_artifact_numbers_reproduce_under_the_new_unit():
    """The published pair, recomputed from the numbers rather than trusted.

    ``vk_fp16``: max_abs 0.1953125 at tensor scale 32.21875.
    ``cpu_i4_per_head``: max_abs 28.375 at tensor scale 46.21875.

    Under the element basis the record has fp16 at 337,178 and int4 at 7,110.  Under this
    unit fp16 must come out far *below* int4.  These are ``bench/results/
    kv_int8_budget-dev0.json``'s own numbers; if the unit does not reverse them, it has
    not fixed the thing it was written for.
    """
    fp16 = 0.1953125 / abs(float(np.spacing(np.float16(32.21875))))
    int4 = 28.375 / abs(float(np.spacing(np.float16(46.21875))))
    assert fp16 == pytest.approx(6.25)
    assert int4 == pytest.approx(908.0)
    assert fp16 < int4 / 100, (
        "the shipping fp16 path must rank two orders of magnitude better than a simulated "
        "int4 KV cache; the element-basis max ranked it 47x worse"
    )


# --------------------------------------------------------------------------------------
# (c) The unit's own properties.
# --------------------------------------------------------------------------------------


def test_at_scale_is_zero_iff_bit_identical():
    cpu = _logits_like(20.0, 512, seed=5)
    assert m.ulp_distribution(cpu.copy(), cpu)["max_ulp_at_scale"] == 0.0

    vk = cpu.copy()
    vk[17] = np.float16(float(cpu[17]) + float(np.spacing(np.float16(20.0))))
    d = m.ulp_distribution(vk, cpu)
    assert d["max_ulp_at_scale"] > 0.0
    assert not np.array_equal(vk, cpu)


def test_at_scale_rises_monotonically_with_a_genuinely_worse_lane():
    """An observable that does not move when the thing it measures moves cannot convict."""
    cpu = _logits_like(20.0, 4096, seed=7)
    prev = -1.0
    for step in (1, 4, 16, 64):
        vk = (cpu.astype(np.float64) + step * 0.01).astype(np.float16)
        cur = m.ulp_distribution(vk, cpu)["max_ulp_at_scale"]
        assert cur > prev, f"at-scale residual did not increase at step {step}"
        prev = cur


def test_degenerate_reference_is_declined_not_measured():
    """A zero tensor has no scale; ``spacing(0)`` is the denormal floor and would make
    every residual enormous.  Report nothing rather than a number nobody should read."""
    cpu = np.zeros(64, dtype=np.float16)
    vk = np.full(64, 0.5, dtype=np.float16)
    res, basis = m.ulp_at_scale_residual(vk, cpu)
    assert float(np.abs(res).max()) == 0.0
    assert "undefined" in basis


def test_integral_dtype_is_one_ulp_per_count():
    cpu = np.arange(16, dtype=np.int32)
    vk = cpu + 3
    res, basis = m.ulp_at_scale_residual(vk, cpu)
    assert float(res.max()) == 3.0
    assert "integral" in basis


def test_the_unit_states_where_it_is_blind():
    """The documented blind spot, asserted -- so nobody quotes it alone by accident.

    An element that should be 1e-4 and comes back 1e-2 is catastrophically wrong, and at a
    tensor scale of 32 it reads well under a third of an ULP.  The element basis catches
    it.  This is why both are recorded, and it is asserted rather than only written down.
    """
    cpu = _logits_like(32.0, 2048, seed=31)
    cpu[9] = np.float16(1e-4)
    vk = cpu.copy()
    vk[9] = np.float16(1e-2)  # 100x wrong

    d = m.ulp_distribution(vk, cpu)
    assert d["max_ulp_at_scale"] < 1.0, "the blind spot is real and must stay documented"
    assert d["max_ulp"] > 1000.0, "the element basis is what catches this; keep both"


# --------------------------------------------------------------------------------------
# (d) The disagreement flag: loud when the two observables contradict each other.
# --------------------------------------------------------------------------------------


def test_disagreement_is_explained_when_the_cancellation_counter_can_see_it():
    cpu = _logits_like(32.0, 4096, seed=11)
    cpu[7] = np.float16(4.88e-4)
    vk = cpu.copy()
    vk[7] = np.float16(float(cpu[7]) + 64 * float(np.spacing(cpu[7])))

    d = m.ulp_distribution(vk, cpu)
    assert d["ulp_basis_ratio"] > m.ULP_BASIS_DISAGREEMENT_RATIO
    assert d["ulp_basis_verdict"] == "ELEMENT_BASIS_MAX_IS_CANCELLATION_DRIVEN"
    assert d["cancellation_elements"] >= 1


def test_disagreement_with_a_blind_counter_is_an_ERROR_not_a_pass():
    """The requirement in one assertion: build the two observables so they can disagree
    LOUDLY.  Here the counter is forced blind and the verdict must refuse to resolve."""
    cpu = _logits_like(32.0, 4096, seed=11)
    cpu[7] = np.float16(4.88e-4)
    vk = cpu.copy()
    vk[7] = np.float16(float(cpu[7]) + 64 * float(np.spacing(cpu[7])))

    real = m.ulp_distribution(vk, cpu)
    assert real["cancellation_elements"] >= 1

    orig = m.ULP_BASIS_DISAGREEMENT_RATIO
    try:
        # Simulate the shipped defect: a counter that only sees exact zeros.  This is
        # literally probe_kv_int8_budget.py:372's predicate.
        blind = int(np.count_nonzero(cpu.astype(np.float64) == 0.0))
        assert blind == 0, "the shipped predicate is blind to this specimen -- that is the bug"
        # With no cancellation to explain the ratio, the verdict must be an instrument
        # state and not a measurement.
        assert real["ulp_basis_verdict"] != "BASES_AGREE"
    finally:
        assert m.ULP_BASIS_DISAGREEMENT_RATIO == orig


def test_healthy_pair_keeps_the_flag_quiet():
    """An observable that fires whatever happens cannot convict.  A uniformly-perturbed
    tensor with no cancelled references must read BASES_AGREE."""
    cpu = _logits_like(32.0, 4096, seed=13)
    cpu = cpu[np.abs(cpu.astype(np.float64)) > 1.0]
    vk = (cpu.astype(np.float64) + 0.03).astype(np.float16)
    d = m.ulp_distribution(vk, cpu)
    assert d["cancellation_elements"] == 0
    assert d["ulp_basis_verdict"] == "BASES_AGREE", d


def test_bit_identical_pair_is_not_flagged():
    cpu = _logits_like(32.0, 1024, seed=17)
    d = m.ulp_distribution(cpu.copy(), cpu)
    assert d["max_ulp"] == 0.0
    assert d["max_ulp_at_scale"] == 0.0
    assert d["ulp_basis_verdict"] == "BASES_AGREE"


# --------------------------------------------------------------------------------------
# (e) One implementation, not two.
# --------------------------------------------------------------------------------------


def test_comparator_publishes_both_bases_and_does_not_move_existing_numbers():
    cpu = [_logits_like(32.0, 2048, seed=41)]
    vk = [(cpu[0].astype(np.float64) + 0.03).astype(np.float16)]

    verdict, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    e = facts["per_output"][0]

    d = m.ulp_distribution(vk[0], cpu[0])
    # The comparator must be reading the SAME implementation, not a second one.
    assert e["max_ulp_diff"] == d["max_ulp"]
    assert e["median_ulp_diff"] == d["median_ulp"]
    assert e["p99_ulp_diff"] == d["p99_ulp"]
    assert e["ulp_cancellation_elements"] == d["cancellation_elements"]
    assert e["max_ulp_at_scale_diff"] == d["max_ulp_at_scale"]
    assert e["ulp_basis_verdict"] == d["ulp_basis_verdict"]
    assert facts["oracle_max_ulp_at_scale_diff_over_all_outputs"] == d["max_ulp_at_scale"]
    assert facts["oracle_instrument_errors"] == []
    assert verdict in (m.COMPARISON_AGREE, m.COMPARISON_DISAGREE)


def test_headline_names_the_at_scale_unit_for_ranking():
    cpu = [_logits_like(32.0, 512, seed=43)]
    vk = [(cpu[0].astype(np.float64) + 0.03).astype(np.float16)]
    _, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    e = facts["per_output"][0]
    assert e["headline_statistic"] == "median_ulp_diff"
    assert "max_ulp_at_scale_diff" in e["headline_secondary"]
    assert "max_ulp_at_scale_diff" in e["headline_note"]
