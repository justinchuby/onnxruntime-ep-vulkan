#!/usr/bin/env python3
"""Falsifiers for criterion 10's all-output CPU oracle. No GPU, no model, no ORT session.

WHY THIS FILE EXISTS AND WHY IT NEEDS NO DEVICE
===============================================
Criterion 10 closed on 2026-08-01 and was reopened three hours later: the oracle compared
**one output out of sixty-five**, and no KV output was compared against CPU anywhere in the
tree.  ``m.compare_all_outputs_to_cpu`` is the missing arm.  These are its falsifiers.

They deliberately require **no device and no artifact**.  The Phi-3.5 model is not present
on every machine — it is absent from the one this was written on — so a falsifier that
needs it would silently skip on exactly the machines where nobody is watching, and a
skipped test reports the same green as a passing one.  The gate's *reading* is what these
pin, and that is a property of the comparison function, not of a GPU.

THE PLANT IS STABLE ON PURPOSE
==============================
Morpheus's discharge condition (b): the planted control must be **wrong and stable**.  An
unstable plant is caught by cross-run identity and would prove nothing about this gap — the
row was originally closed on divergence, which is the symptom of a dirty arena, while the
same binding-arity defect on a clean arena is stable and silent.  Every plant here is
byte-identical across runs.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _models as m  # noqa: E402

VOCAB = 512
KV_SHAPE = (1, 4, 8, 16)
N_OUTPUTS = 9


def _correct(seed: int = 11) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 4.0, size=(1, 1, VOCAB)).astype(np.float32)
    outs = [logits]
    for _ in range(1, N_OUTPUTS):
        outs.append(rng.normal(0.0, 1.0, size=KV_SHAPE).astype(np.float16))
    return outs


def _zeroed_kv(correct: list[np.ndarray]) -> list[np.ndarray]:
    """The planted defect: logits untouched, every KV output all-zero. Wrong and stable."""
    return [correct[0].copy()] + [np.zeros_like(a) for a in correct[1:]]


# -- the arm that must go red -------------------------------------------------------------


def test_all_zero_kv_is_refused_even_though_the_logits_agree():
    """The reopened defect, in one assertion.

    The logits are bit-identical to the oracle, so the old one-output comparison returns
    AGREE. Sixty-four zeroed KV outputs must nevertheless refuse the run.
    """
    correct = _correct()
    planted = _zeroed_kv(correct)

    outcome, facts = m.compare_all_outputs_to_cpu(planted, correct)

    assert outcome != m.COMPARISON_AGREE, (
        "an all-zero KV write agreed with the CPU oracle; this is the exact defect "
        f"criterion 10 was reopened for. facts={facts}"
    )
    assert facts["oracle_outputs_degenerate"] == N_OUTPUTS - 1
    assert facts["oracle_degenerate_indices"] == list(range(1, N_OUTPUTS))


def test_the_old_one_output_comparison_would_have_passed_the_same_plant():
    """The ground-truth arm, without which the test above proves nothing.

    A falsifier that fires is only evidence if the thing it replaces would NOT have fired
    on the same input. This asserts the gap directly: identical logits, so the logits-only
    oracle sees agreement, on data the all-output oracle refuses.
    """
    from test_criterion10 import _compare_run_to_cpu

    correct = _correct()
    planted = _zeroed_kv(correct)

    old_outcome, _ = _compare_run_to_cpu(planted, correct)
    new_outcome, _ = m.compare_all_outputs_to_cpu(planted, correct)

    assert old_outcome == m.COMPARISON_AGREE, (
        "the one-output comparison did not agree with the plant, so this plant does not "
        "demonstrate the gap and the test above is measuring something else"
    )
    assert new_outcome != m.COMPARISON_AGREE


def test_a_stable_plant_is_invisible_to_cross_run_identity():
    """Why the plant had to be stable, asserted rather than asserted-in-prose.

    Cross-run identity is the only all-65 gate that existed. It compares Vulkan to Vulkan,
    so a defect that is the same every run passes it perfectly.
    """
    planted = _zeroed_kv(_correct())
    identical, differing = m.outputs_bit_equal(planted, planted)
    assert identical and differing == [], (
        "the stable plant was not stable; cross-run identity would have caught it and the "
        "control would prove nothing about this gap"
    )


def test_one_wrong_kv_output_out_of_many_is_refused():
    """Not just all-zero: a single output outside tolerance must fail the whole run."""
    correct = _correct()
    planted = [a.copy() for a in correct]
    planted[5] = (planted[5].astype(np.float32) + 3.0).astype(np.float16)

    outcome, facts = m.compare_all_outputs_to_cpu(planted, correct)

    assert outcome == m.COMPARISON_DISAGREE
    assert facts["oracle_failing_indices"] == [5]
    assert facts["oracle_worst_output_index"] == 5


def test_arity_mismatch_is_a_disagreement_not_a_crash():
    correct = _correct()
    outcome, facts = m.compare_all_outputs_to_cpu(correct[:-1], correct)
    assert outcome == m.COMPARISON_DISAGREE
    assert facts["oracle_arity_mismatch"] == {"vk": N_OUTPUTS - 1, "cpu": N_OUTPUTS}


# -- the non-triviality guard, on both sides ----------------------------------------------


def test_zeros_on_both_sides_are_not_evidence():
    """Morpheus's discharge condition (c), stated as he stated it.

    64 pairs of zeros satisfy an all-output comparison perfectly, which is `0.0 == 0.0` in
    a fourth costume. A degenerate oracle is an absence of evidence and must report
    NOT_PERFORMED — never AGREE, and never DISAGREE either, because nothing disagreed.
    """
    zeros = [np.zeros((1, 1, VOCAB), np.float32)] + [
        np.zeros(KV_SHAPE, np.float16) for _ in range(N_OUTPUTS - 1)
    ]

    outcome, facts = m.compare_all_outputs_to_cpu(zeros, [a.copy() for a in zeros])

    assert outcome == m.COMPARISON_NOT_PERFORMED, (
        "two all-zero runs reported agreement; that is 0.0 == 0.0 wearing the gate's "
        f"passing token. facts={facts}"
    )
    assert facts["oracle_outputs_degenerate"] == N_OUTPUTS
    assert facts["oracle_outputs_compared"] == 0


def test_a_degenerate_oracle_side_alone_is_also_refused():
    """The guard is on both sides, not only on the EP's.

    If the CPU reference itself is constant the comparison is vacuous whatever the EP did.
    """
    correct = _correct()
    blind_oracle = [correct[0]] + [np.zeros_like(a) for a in correct[1:]]

    outcome, facts = m.compare_all_outputs_to_cpu(correct, blind_oracle)
    assert outcome == m.COMPARISON_NOT_PERFORMED
    assert facts["oracle_outputs_degenerate"] == N_OUTPUTS - 1


def test_constant_nonzero_is_degenerate_too():
    """Residue, not just zero-initialisation.

    A buffer left holding one repeated value carries no information either, and a guard
    written on `== 0` would pass it. That is why the guard is written on constancy.
    """
    correct = _correct()
    planted = [correct[0].copy()] + [np.full_like(a, 0.25) for a in correct[1:]]
    outcome, facts = m.compare_all_outputs_to_cpu(planted, correct)
    assert outcome != m.COMPARISON_AGREE
    assert facts["oracle_outputs_degenerate"] == N_OUTPUTS - 1


# -- the arm that must stay green ---------------------------------------------------------


def test_a_correct_run_agrees_and_says_over_how_many_outputs():
    correct = _correct()
    perturbed = [
        (a.astype(np.float64) * (1.0 + 1e-4)).astype(a.dtype) for a in correct
    ]

    outcome, facts = m.compare_all_outputs_to_cpu(perturbed, correct)

    assert outcome == m.COMPARISON_AGREE, facts
    assert facts["oracle_outputs_compared"] == N_OUTPUTS
    assert facts["oracle_outputs_within_tolerance"] == N_OUTPUTS
    assert facts["oracle_outputs_degenerate"] == 0


def test_the_two_counts_have_two_names():
    """R11, mechanically. The reopened row's proximate cause was one key read as another.

    `outputs_compared: 65` counted cross-run comparisons, sat among the oracle facts, and
    was quoted into the criteria table as sixty-five oracle comparisons beside a
    `max_abs_diff` covering one tensor. Both names are now unambiguous and neither may
    revert to the bare form.
    """
    correct = _correct()
    _, facts = m.compare_all_outputs_to_cpu(correct, correct)
    assert "oracle_outputs_compared" in facts
    assert "outputs_compared" not in facts, (
        "the bare key is back; it is the one that was misread and it must not exist"
    )

    from test_criterion10 import _compare_run_to_cpu

    _, logit_facts = _compare_run_to_cpu(correct, correct)
    assert "logits_max_abs_diff" in logit_facts
    assert "max_abs_diff" not in logit_facts, (
        "the unqualified diff key is back; its extent is one output out of sixty-five and "
        "its name has to say so"
    )


@pytest.mark.parametrize(
    "dtype,derived_from",
    [(np.float16, "MATMULNBITS_FP16"), (np.float32, "MATMULNBITS_FP32")],
)
def test_every_tolerance_carries_its_justification(dtype, derived_from):
    """Discharge condition (a): tolerance justified rather than assumed.

    NOTE ON THIS TEST'S OWN POLARITY. It first read

        (np.float16, m.KV_CACHE_FP16) ... assert tol == expected

    which compares the constant to itself and cannot fail — I caught it by widening
    KV_CACHE_FP16 to 1e9 as a mutation check and watching this test stay green. That is the
    third instance of the same shape in three days, and it is the exact form the screen in
    ci/check_tautological_assertions.py does NOT reach, since the two sides are different
    text.

    The claim is therefore asserted against the thing the tolerance says it derives from:
    the MatMulNBits tolerances already justified from measured data in _models.py's header,
    because MatMulNBits is the arithmetic that produces these tensors. If someone picks a
    number for this gate instead, these fail.
    """
    source = getattr(m, derived_from)
    tol, why = m.tolerance_for_output(np.zeros((2, 2), dtype))
    assert tol == source, (
        f"the {dtype} tolerance is no longer {derived_from}; a tolerance chosen to make "
        "this gate green is not a justified tolerance"
    )
    assert "MatMulNBits" in why
    assert why.strip()


def test_integral_outputs_get_no_tolerance_at_all():
    tol, why = m.tolerance_for_output(np.zeros((2, 2), np.int64))
    assert tol == m.FP32_EXACT
    assert "meaningless" in why
