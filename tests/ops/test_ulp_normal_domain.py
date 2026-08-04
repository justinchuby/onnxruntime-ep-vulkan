"""§8.9.22's ruled observable, and the admits-nothing test applied to the repair itself.

WHY THIS FILE EXISTS
====================
`docs/DESIGN.md` §8.9.22 rules that criterion 10's logits observable is not a max-ULP over
the whole tensor: the replacement reports **the residual over references at or above the
smallest normal** and **the count and fraction below it**, as two named quantities, and
publishes the subnormal population rather than dropping it.

Ruling (2) is the load-bearing half: *"A change to a criterion that makes nothing pass
which did not pass before is a repair. One that admits the thing whose measurement
prompted it is a narrowing."*  Morpheus applied that test to int8.  **This file applies it
to the repair itself** — every arm below asks whether the ruled statistic lets anything
through that the element basis caught.

Device-free by construction: numpy only, no session, no GPU, deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _models as m  # noqa: E402
import _verdict  # noqa: E402


FP16_TINY = float(np.finfo(np.float16).tiny)  # 6.103515625e-05


# --------------------------------------------------------------------------------------
# 1. The mechanism §8.9.22 named, reproduced from first principles
# --------------------------------------------------------------------------------------


def test_subnormal_reference_inflates_the_element_basis_max_by_three_orders():
    """The ruling's mechanism, built rather than cited.

    `spacing()` collapses by ~3 orders of magnitude below the smallest normal, so the SAME
    absolute residual reads three orders larger when its reference is subnormal.  If this
    arm ever goes green for the wrong reason the whole ruling is unsupported here.
    """
    residual = np.float16(2.0**-14)  # one subnormal step's worth of absolute error

    normal_ref = np.array([1.0], dtype=np.float16)
    sub_ref = np.array([FP16_TINY / 2], dtype=np.float16)

    u_normal, _ = m.ulp_residual((normal_ref + residual).astype(np.float16), normal_ref)
    u_sub, _ = m.ulp_residual((sub_ref + residual).astype(np.float16), sub_ref)

    assert u_normal[0] < 20.0, u_normal
    assert u_sub[0] > 1.0, u_sub
    # The ratio is the defect: identical absolute error, three orders apart in ULPs.
    assert u_sub[0] / max(u_normal[0], 1e-30) > 100.0, (u_sub[0], u_normal[0])


def test_the_max_is_located_at_the_least_informative_elements():
    """0.45% of the tensor sat below the smallest normal and owned the max.

    Specimen: a scale-32 tensor whose bulk carries a real 2-ULP residual, plus a small
    subnormal population carrying a residual that is *smaller in absolute terms* than the
    bulk's.  The element-basis max must be attained in the subnormal population — i.e. the
    statistic reports the degeneracy, not the subject.
    """
    rng = np.random.default_rng(1022)
    n = 4096
    cpu = (rng.uniform(1.0, 32.0, n)).astype(np.float16)
    n_sub = 18  # 0.44% -- the ruling's fraction, at this size
    cpu[:n_sub] = np.float16(FP16_TINY / 4)

    vk = cpu.astype(np.float64)
    vk[n_sub:] += 2.0 * np.abs(np.spacing(cpu[n_sub:].astype(np.float16))).astype(
        np.float64
    )
    bulk_abs = float(
        np.abs(vk[n_sub:] - cpu[n_sub:].astype(np.float64)).max()
    )
    # ABSOLUTELY smaller than the bulk residual -- and yet it will own the max, because
    # its denominator is the denormal spacing.
    vk[:n_sub] += bulk_abs / 2.0
    vk = vk.astype(np.float16)

    ulps, _ = m.ulp_residual(vk, cpu)
    assert int(np.argmax(ulps)) < n_sub, "the max must be owned by the subnormal population"

    sub_abs = float(np.abs(vk[:n_sub].astype(np.float64) - cpu[:n_sub].astype(np.float64)).max())
    assert sub_abs < bulk_abs, (sub_abs, bulk_abs)

    nd = m.ulp_normal_domain(vk, cpu)
    assert nd["subnormal_reference_elements"] == n_sub
    assert 0.003 < nd["subnormal_reference_fraction"] < 0.006
    # And the ruled statistic reports the subject: the bulk's real 2 ULP.
    assert nd["max_ulp_normal_domain"] <= 4.0, nd["max_ulp_normal_domain"]
    assert float(ulps.max()) / nd["max_ulp_normal_domain"] > 100.0


# --------------------------------------------------------------------------------------
# 2. Two numbers, never one — enforced, not promised
# --------------------------------------------------------------------------------------


def test_report_refuses_to_format_the_residual_without_the_population():
    nd = m.ulp_normal_domain(
        np.array([1.0, 2.0], dtype=np.float16), np.array([1.0, 2.0], dtype=np.float16)
    )
    assert "subnormal references" in m.ulp_normal_domain_report(nd)

    for missing in m.ULP_NORMAL_DOMAIN_KEYS:
        stripped = {k: v for k, v in nd.items() if k != missing}
        with pytest.raises(_verdict.InstrumentError) as e:
            m.ulp_normal_domain_report(stripped)
        assert missing in str(e.value)


def test_the_population_is_published_not_dropped():
    """An excluded element must leave a visible count, or the domain was narrowed."""
    cpu = np.array([8.0, 8.0, FP16_TINY / 2, 8.0], dtype=np.float16)
    vk = np.array([8.0, 8.0, 0.0, 8.0], dtype=np.float16)
    nd = m.ulp_normal_domain(vk, cpu)
    assert nd["subnormal_reference_elements"] == 1
    assert nd["subnormal_reference_fraction"] == 0.25
    assert "PUBLISHED, not dropped" in nd["normal_domain_declared"]
    assert "0.2500%" not in nd["normal_domain_declared"]  # it is 25%, not 0.25%
    assert "25.0000%" in nd["normal_domain_declared"]


def test_an_empty_normal_domain_is_an_instrument_state_not_a_zero():
    cpu = np.array([FP16_TINY / 2, FP16_TINY / 4], dtype=np.float16)
    vk = np.array([0.0, 0.0], dtype=np.float16)
    nd = m.ulp_normal_domain(vk, cpu)
    assert nd["normal_domain_verdict"] == "ERROR(instrument=empty_normal_domain)"
    assert nd["max_ulp_normal_domain"] is None
    assert "UNMEASURED(empty domain)" in m.ulp_normal_domain_report(nd)


# --------------------------------------------------------------------------------------
# 3. THE ADMITS-NOTHING TEST, applied to my own repair (§8.9.22 ruling 2)
# --------------------------------------------------------------------------------------


def test_the_gate_never_consumed_any_ulp_basis_so_the_repair_cannot_loosen_it():
    """The decisive arm.

    Criterion 10's pass/fail predicate is `np.allclose(rtol, atol)`.  If the ruled
    statistic could flip a verdict, this repair would be a narrowing.  It cannot, and the
    proof is mechanical: perturb the ULP statistics arbitrarily and the status is
    unmoved, because the status is not a function of them.

    Constructed as a pair: the SAME tensor pair scored under the comparator, and scored
    under a hand-written allclose.  They must agree element-for-element on every output,
    which is only true if no ULP number participates.
    """
    rng = np.random.default_rng(7)
    cpu_list, vk_list = [], []
    for k in range(6):
        c = (rng.uniform(-8.0, 8.0, 512)).astype(np.float16)
        v = c.astype(np.float64)
        v += (k * 4) * np.abs(np.spacing(c)).astype(np.float64)
        # a subnormal population of varying size, to move the ruled statistic around
        c = c.copy()
        c[: k * 17] = np.float16(FP16_TINY / 3)
        cpu_list.append(c)
        vk_list.append(v.astype(np.float16))

    _outcome, facts = m.compare_all_outputs_to_cpu(vk_list, cpu_list)
    for i, e in enumerate(facts["per_output"]):
        tol, _ = m.tolerance_for_output(vk_list[i])
        expected = bool(
            np.allclose(
                vk_list[i].astype(np.float64),
                cpu_list[i].astype(np.float64),
                rtol=tol["rtol"],
                atol=tol["atol"],
                equal_nan=True,
            )
        )
        assert (e["status"] == "WITHIN_TOLERANCE") == expected, (i, e["status"])
        assert "np.allclose" in e["verdict_predicate"]
        assert "no ULP statistic" in e["verdict_predicate"]


def test_a_genuinely_wrong_subnormal_element_is_still_caught_by_the_gate():
    """The admission the ruled statistic *would* make if it were a gate — and is not.

    An element whose reference is subnormal and whose value is catastrophically wrong
    leaves the normal domain entirely.  The old element-basis max would have shown a huge
    number.  The ruled residual does not.  The criterion must still fail it.
    """
    cpu = np.full(256, 4.0, dtype=np.float16)
    cpu[3] = np.float16(FP16_TINY / 2)
    vk = cpu.copy()
    vk[3] = np.float16(0.5)  # absolutely and relatively wrong

    dist = m.ulp_distribution(vk, cpu)
    assert dist["max_ulp"] > 1e3, dist["max_ulp"]  # old statistic screams
    assert dist["max_ulp_normal_domain"] == 0.0  # ruled statistic is silent
    assert dist["subnormal_reference_elements"] == 1  # but the population is published

    _outcome, facts = m.compare_all_outputs_to_cpu([vk], [cpu])
    assert facts["per_output"][0]["status"] == "OUTSIDE_TOLERANCE"
    assert facts["oracle_failing_indices"] == [0]


def test_the_element_basis_max_is_still_recorded_unchanged():
    """Additive, not replacing.  Nothing that was published before this ruling is gone."""
    rng = np.random.default_rng(19)
    cpu = rng.uniform(-2.0, 2.0, 300).astype(np.float16)
    cpu[:5] = np.float16(FP16_TINY / 8)
    vk = (cpu.astype(np.float64) + 2.0**-16).astype(np.float16)

    dist = m.ulp_distribution(vk, cpu)
    for legacy in (
        "median_ulp",
        "p99_ulp",
        "max_ulp",
        "max_ulp_at_scale",
        "max_abs",
        "cancellation_elements",
        "ulp_basis_verdict",
    ):
        assert legacy in dist, legacy
    expected_max, _ = m.ulp_residual(vk, cpu)
    finite = expected_max[np.isfinite(expected_max)]
    assert dist["max_ulp"] == float(finite.max())


def test_a_pass_cannot_be_manufactured_by_declaring_the_domain():
    """The narrowing this ruling forbids, attempted, and observed not to work.

    Take a tensor that FAILS the criterion because of a residual on normal references.
    Declaring the subnormal domain out cannot rescue it, because the failing elements are
    not in the excluded set.  If this ever passes, someone has made the domain split a
    gate.
    """
    cpu = np.full(128, 6.0, dtype=np.float16)
    cpu[:20] = np.float16(FP16_TINY / 2)
    vk = cpu.astype(np.float64)
    vk[64:] += 0.5  # far outside rtol=2e-2 * 6.0 = 0.12
    vk = vk.astype(np.float16)

    dist = m.ulp_distribution(vk, cpu)
    assert dist["subnormal_reference_elements"] == 20
    assert dist["max_ulp_normal_domain"] > 0.0

    _outcome, facts = m.compare_all_outputs_to_cpu([vk], [cpu])
    assert facts["per_output"][0]["status"] == "OUTSIDE_TOLERANCE"


# --------------------------------------------------------------------------------------
# 4. The instrument can go red — a check that never fails checks nothing
# --------------------------------------------------------------------------------------


def test_selftest_the_domain_split_can_disagree_with_the_element_basis():
    """A separating case.  Without one, every arm above could be passing vacuously."""
    cpu = np.full(64, 10.0, dtype=np.float16)
    cpu[0] = np.float16(FP16_TINY / 4)
    vk = cpu.copy()
    vk[0] = np.float16(FP16_TINY / 4 + 2.0**-24)

    dist = m.ulp_distribution(vk, cpu)
    assert dist["max_ulp"] > 0.0
    assert dist["max_ulp_normal_domain"] == 0.0
    assert dist["max_ulp"] != dist["max_ulp_normal_domain"]


def test_integral_dtypes_declare_no_domain_rather_than_pretending_to_one():
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([1, 2, 4], dtype=np.int64)
    nd = m.ulp_normal_domain(a, b)
    assert nd["normal_domain_verdict"].startswith("NOT_APPLICABLE")
    assert nd["subnormal_reference_elements"] == 0
    assert nd["max_ulp_normal_domain"] == 1.0
