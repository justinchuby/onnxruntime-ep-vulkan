"""Tests for the CUDA-competition harness (issue #69). Owner: Niobe.

These are tests of the *instrument*, not of the EP.  Every gate this harness uses
to decide whether a number may be quoted is exercised in both polarities: a case
where it must fire, and a case where it must not.  A gate with only the passing
polarity tested is a gate nobody has seen work.

The planted-defect tests are the load-bearing ones.  Each takes a profile or a
record that the harness currently calls admissible, injects exactly the defect
the gate exists to catch, and asserts the verdict flips.  If a future refactor
silently disables a gate, these fail; a test that only checks the happy path
would not.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import io
import json
import os
import re
import sys
import tokenize
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cuda_competition as cc  # noqa: E402
import cuda_workloads as cw  # noqa: E402
import bench_models as bm  # noqa: E402


# ---------------------------------------------------------------------------
# Profile fixtures
# ---------------------------------------------------------------------------

def _kernel_event(node: str, provider: str, dur: int, op: str = "MatMul") -> dict:
    return {"cat": "Node", "name": f"{node}_kernel_time", "dur": dur,
            "args": {"provider": provider, "op_name": op}}


def _write_profile(tmp_path: Path, events: "list[dict]") -> Path:
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(events), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# read_profile_partition
# ---------------------------------------------------------------------------

def test_partition_counts_nodes_not_executions(tmp_path):
    """N profiled runs of one node must count as one node, not N.

    ORT emits one kernel_time event per node *per run*.  Counting events would
    report a graph N times larger than it is and would make the fallback share
    depend on how many iterations were profiled.
    """
    events = []
    for _ in range(7):
        events.append(_kernel_event("a", "CUDAExecutionProvider", 100))
        events.append(_kernel_event("b", "CUDAExecutionProvider", 100))
    got = cc.read_profile_partition(_write_profile(tmp_path, events))
    assert got["executed_nodes"] == 2
    assert got["partition"] == {"CUDAExecutionProvider": 2}
    # kernel time, by contrast, accumulates across runs — it is a total, not a census
    assert got["kernel_time_us"]["CUDAExecutionProvider"] == 1400


def test_partition_ignores_non_node_events(tmp_path):
    """Session and fence bookkeeping are not graph nodes."""
    events = [
        _kernel_event("real", "CUDAExecutionProvider", 50),
        {"cat": "Session", "name": "model_run", "dur": 9999, "args": {}},
        {"cat": "Node", "name": "real_fence_before", "dur": 3, "args": {}},
        {"cat": "Node", "name": "real_fence_after", "dur": 3, "args": {}},
    ]
    got = cc.read_profile_partition(_write_profile(tmp_path, events))
    assert got["executed_nodes"] == 1
    assert got["kernel_time_us"] == {"CUDAExecutionProvider": 50}


def test_missing_provider_arg_is_unattributed_not_dropped(tmp_path):
    """A node whose provider ORT did not record must not vanish from the census.

    Silently dropping it would shrink the denominator and flatter the fallback
    share — the failure mode is a *better*-looking number, which is the direction
    that does not get noticed.
    """
    events = [
        _kernel_event("a", "CUDAExecutionProvider", 100),
        {"cat": "Node", "name": "mystery_kernel_time", "dur": 100, "args": {}},
    ]
    got = cc.read_profile_partition(_write_profile(tmp_path, events))
    assert got["partition"] == {"CUDAExecutionProvider": 1, "UNATTRIBUTED": 1}
    assert got["executed_nodes"] == 2


# ---------------------------------------------------------------------------
# classify_fallback — the fusion problem
# ---------------------------------------------------------------------------

def test_fusing_ep_node_share_is_misleading_and_time_share_is_not():
    """A **synthetic fixture** in the shape a fusing EP produces: one fused provider
    node beside a handful of residual host nodes, with almost all kernel time on the
    provider.

    The numbers below are constructed inputs, not an observation: no committed artifact
    in this tree records a per-provider profile node census for any arm, so this test
    quotes none and neither does `cuda_competition`'s own docstring. What is pinned is
    the *disagreement* — a partition whose node share is over the threshold while its
    time share is far under it — which is what made node-counting unusable and what a
    future refactor might quietly undo.
    """
    partition = {"VulkanExecutionProvider": 1, "CPUExecutionProvider": 8}
    kernel_us = {"VulkanExecutionProvider": 45000.0, "CPUExecutionProvider": 90.0}
    got = cc.classify_fallback(partition, kernel_us, "VulkanExecutionProvider")
    assert got["nodes"] == pytest.approx(8 / 9)
    assert got["time"] == pytest.approx(90.0 / 45090.0, rel=1e-6)
    assert got["time"] < cc.FALLBACK_SPLIT_THRESHOLD
    assert got["nodes"] > cc.FALLBACK_SPLIT_THRESHOLD


def test_nothing_executed_gives_none_not_zero():
    """0/0 is 'unmeasured', never 'no fallback'."""
    got = cc.classify_fallback({}, {}, "CUDAExecutionProvider")
    assert got["nodes"] is None
    assert got["time"] is None
    assert got["executed_nodes"] == 0


def test_cpu_arm_cannot_fall_back_to_itself():
    got = cc.classify_fallback({"CPUExecutionProvider": 40}, {"CPUExecutionProvider": 10.0},
                               "CPUExecutionProvider")
    assert got["time"] == 0.0
    assert got["nodes"] == 0.0


def test_planted_cpu_fallback_is_detected_by_time_share():
    """The negative control: inject genuine CPU work and the gate must fire.

    Same partition shape as the passing case above; only the CPU EP's kernel time
    changes.  If the gate ever stops reading kernel time, this test fails.
    """
    partition = {"CUDAExecutionProvider": 300, "CPUExecutionProvider": 31}
    healthy = cc.classify_fallback(partition, {"CUDAExecutionProvider": 10000.0,
                                               "CPUExecutionProvider": 40.0},
                                   "CUDAExecutionProvider")
    assert healthy["time"] < cc.FALLBACK_SPLIT_THRESHOLD

    planted = cc.classify_fallback(partition, {"CUDAExecutionProvider": 10000.0,
                                               "CPUExecutionProvider": 4000.0},
                                   "CUDAExecutionProvider")
    assert planted["time"] > cc.FALLBACK_SPLIT_THRESHOLD, (
        "a 4000us CPU share against 10000us of CUDA must be convicted as a split frame")


# ---------------------------------------------------------------------------
# ULP distance — the instrument Trinity's np.spacing finding rules out
# ---------------------------------------------------------------------------

def _one_ulp(a, b):
    return int(cc.ulp_distance(a, b)[0])


def test_ulp_distance_is_one_for_adjacent_float16():
    a = np.array([1.0], dtype=np.float16)
    b = np.nextafter(a, np.float16(2.0))
    assert _one_ulp(a, b) == 1


def test_ulp_distance_is_zero_for_identical():
    a = np.array([3.5, -7.25, 0.0], dtype=np.float16)
    assert list(cc.ulp_distance(a, a)) == [0, 0, 0]


def test_ulp_distance_crosses_zero_correctly():
    """+0 and -0 are the same number; their bit patterns are not.

    A naive sign-magnitude subtraction reports 32768 ULP here.
    """
    pos = np.array([0.0], dtype=np.float16)
    neg = np.array([-0.0], dtype=np.float16)
    assert int(cc.ulp_distance(pos, neg)[0]) == 0

    smallest = np.array([np.nextafter(np.float16(0.0), np.float16(1.0))], dtype=np.float16)
    neg_smallest = -smallest
    assert int(cc.ulp_distance(smallest, neg_smallest)[0]) == 2


def test_ulp_distance_at_float16_max_finite_does_not_read_zero():
    """Trinity's 2026-08-04 finding, as a regression test.

    ``np.spacing`` returns ``inf`` at fp16's largest finite value, so a real error
    there divides to 0.0 ULP — a wrong residual that looks sound.  The bit-pattern
    instrument must report the true distance.
    """
    max_finite = np.array([np.finfo(np.float16).max], dtype=np.float16)
    lower = np.array([np.finfo(np.float16).max], dtype=np.float16)
    for _ in range(504):
        lower = np.nextafter(lower, np.float16(0.0))
    assert int(cc.ulp_distance(lower, max_finite)[0]) == 504
    # and the instrument this replaces would have said zero
    assert np.isinf(np.spacing(max_finite)[0])


# ---------------------------------------------------------------------------
# compare_outputs
# ---------------------------------------------------------------------------

def _logits(seed: int, n: int = 64, rows: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, n)) * 5.0).astype(np.float16)


def _separated_logits(rows: int = 4, n: int = 64) -> np.ndarray:
    """Logits whose ranking is decidable and whose rows are genuinely distinct.

    Several checks below are about *ordering*, and ordering is only testable when
    the reference's own ranks are separated by more than the noise being injected.
    Random logits over 64 columns are frequently tied to within a ULP, which is the
    real condition the harness meets on Phi-3.5 and the reason
    :func:`cuda_competition.conditioned_top_k` exists — but a test that wants to
    assert an ordering claim must first construct an input where one exists.

    Rows are given different *orderings*, not merely different offsets.  A fixture
    whose rows are near-copies makes a row-permutation control trivially pass, which
    is a property of the fixture and not of the instrument.
    """
    rng = np.random.default_rng(20260807)
    out = np.empty((rows, n), dtype=np.float32)
    base = np.arange(n, 0, -1, dtype=np.float32)
    for i in range(rows):
        out[i] = rng.permutation(base)
    return out.astype(np.float16)


def test_identical_outputs_match():
    a = _logits(1)
    got = cc.compare_outputs([a], [a], ["logits"])
    assert got["verdict"] == "MATCH"
    assert got["outputs"][0]["max_peak_ulp"] == 0.0
    assert got["outputs"][0]["max_raw_ulp"] == 0


def test_peak_ulp_spacing_uses_the_tensor_scale_not_the_element():
    """The fix for the metric that rejected every honest arm.

    Raw ULP is measured in units of the *local* spacing, which collapses near zero, so a
    physically irrelevant difference between two near-zero logits scores enormously
    against a tight budget on every GPU arm.  No magnitude is quoted for that rejection:
    the run that produced it is not a committed artifact in this tree.  Peak-scaled ULP
    measures the same difference in units of the spacing at the tensor's largest
    magnitude, which is the scale softmax and argmax actually care about — and the
    fixture below, whose values are constructed here rather than observed, demonstrates
    both readings on one difference.
    """
    ref = np.array([[16.0, 1e-3, -1e-3]], dtype=np.float16)
    sub = ref.copy()
    # Move a near-zero element across zero: a small physical difference on a tensor
    # peaking far above it, but thousands of representable fp16 values.
    sub[0, 1] = np.float16(-1e-3)
    raw = cc.ulp_distance(sub, ref)
    assert raw.max() > 1000, "raw ULP should blow up across zero — that is its flaw"

    spacing = cc.peak_ulp_spacing(ref)
    assert spacing == pytest.approx(float(np.spacing(np.float16(16.0))))
    scaled = abs(float(sub[0, 1]) - float(ref[0, 1])) / spacing
    assert scaled < 1.0, "the same difference is sub-ULP at the tensor's own scale"


def test_peak_ulp_spacing_is_finite_at_float16_max():
    """``np.spacing`` returns inf here; an infinite budget passes every error.

    Trinity's 2026-08-04 finding, applied to the new metric: the budget's
    denominator must be finite at the top of the range or the gate cannot fail.
    """
    peak = np.finfo(np.float16).max
    assert np.isinf(np.spacing(peak))
    got = cc.peak_ulp_spacing(np.array([[peak, 0.0]], dtype=np.float16))
    assert got is not None and np.isfinite(got) and got > 0


def test_tf32_regime_widens_the_budget_and_float32_does_not():
    """The CUDA EP computes fp32 MatMul/Conv in TF32 by default — 10 mantissa bits.

    That is a large precision difference against a graph declaring 23, by construction of
    the formats and not as an observation: no peak-ULP magnitude is quoted here, because
    the MobileNetV2 run that produced one is not a committed artifact in this tree.  The
    budget must follow the declared precision or the comparison either disqualifies the
    competitor or silently licenses a precision loss the subject does not take.
    """
    strict = cc.equivalence_budget_ulp("float32", "float32")
    loose = cc.equivalence_budget_ulp("float32", "tf32")
    assert strict == cc.ROUNDING_DEPTH_BOUND
    assert loose == cc.ROUNDING_DEPTH_BOUND * 2 ** 13
    assert loose > strict


def test_compute_regime_reads_provider_options_over_the_documented_default():
    assert cc.compute_regime(cc.ARM_CUDA, None) == "tf32"
    assert cc.compute_regime(cc.ARM_CUDA, {"use_tf32": "1"}) == "tf32"
    assert cc.compute_regime(cc.ARM_CUDA, {"use_tf32": "0"}) == "float32"
    # No other arm gets the wider budget, whatever it reports.
    assert cc.compute_regime(cc.ARM_VULKAN, {"use_tf32": "1"}) == "float32"
    assert cc.compute_regime(cc.ARM_CPU_HOST, None) == "float32"


def test_argmax_flip_is_divergent_when_the_ranking_is_decidable():
    """Task equivalence gates, and it gates where it has power.

    The swap is made on logits a full unit apart so the ordering claim is testable;
    on tied logits the check must abstain instead, which the next test asserts.
    """
    ref = _separated_logits()
    sub = ref.copy()
    order = np.argsort(-ref[0].astype(np.float64))
    top, second = order[0], order[1]
    sub[0, second], sub[0, top] = ref[0, top], ref[0, second]
    got = cc.compare_outputs([sub], [ref], ["logits"])
    assert got["verdict"] == "DIVERGENT"
    assert got["outputs"][0]["task"]["argmax_agreement"] < 1.0


def test_tied_ranking_abstains_rather_than_failing_or_passing():
    """An undecidable ordering must read UNRESOLVED, never MATCH and never DIVERGENT.

    This is the condition the harness actually meets: pseudo-random feeds through a
    model with no coherent context give Phi-3.5 a top-1/top-2 gap of 0.031, smaller
    than either GPU arm's own error.  The unconditioned check scored 0.0 agreement
    for *both* arms, which is a fact about the RNG.  A check with no power must not
    be quotable in either direction.
    """
    ref = np.zeros((1, 8), dtype=np.float16)
    ref[0] = np.float16(4.0)
    ref[0, 0] = np.float16(4.0 + 2 ** -8)  # a hair above its neighbours
    sub = ref.copy()
    sub[0, 0], sub[0, 1] = ref[0, 1], ref[0, 0]
    tk = cc.conditioned_top_k(sub.astype(np.float64), ref.astype(np.float64),
                              k=4, resolution=1.0)
    assert tk["verdict"] == "UNRESOLVED"
    assert tk["argmax_resolvable_rows"] == 0
    # The unconditioned view still records the disagreement — abstaining is not
    # the same as not looking.
    assert tk["unconditioned_argmax_agreement"] < 1.0


def test_unresolved_task_check_does_not_rescue_a_failing_numeric_gate():
    """Abstention on the task axis must not upgrade an over-budget arm to MATCH."""
    ref = np.zeros((1, 8), dtype=np.float16)
    ref[0] = np.float16(4.0)
    sub = (ref.astype(np.float32) * 2.0).astype(np.float16)
    got = cc.compare_outputs([sub], [ref], ["logits"])
    assert got["outputs"][0]["task"]["verdict"] == "UNRESOLVED"
    assert got["verdict"] == "DIVERGENT"


def test_conditioning_cannot_be_widened_without_limit_by_a_wild_arm():
    """Resolution is capped at the budget, so a wrong arm cannot buy itself slack.

    Conditioning on the observed disagreement is exact, but left uncapped an arm
    with enormous error would declare every row unresolvable and evade the task
    check entirely.  The cap means that beyond the budget the numeric gate has
    already failed it.
    """
    ref = _separated_logits()
    sub = (ref.astype(np.float32) * 100.0).astype(np.float16)
    got = cc.compare_outputs([sub], [ref], ["logits"])
    entry = got["outputs"][0]
    spacing = entry["peak_ulp_spacing"]
    assert entry["task"]["resolution_abs"] <= entry["peak_ulp_budget"] * spacing
    assert got["verdict"] == "DIVERGENT"


def test_a_sparse_plant_cannot_manufacture_its_own_excuse():
    """The defect a planted control exposed, as a regression test.

    Conditioning on the tensor-wide *maximum* disagreement is self-defeating: an arm
    whose only error is an adjacent-rank swap produces, as that error, exactly the
    noise estimate that then declares the swap unresolvable.  A high quantile is
    immune — a plant touching under 1% of elements does not move p99, so the rows
    stay resolvable and the plant is caught.
    """
    ref = _separated_logits(rows=8, n=512)
    sub = ref.copy()
    order = np.argsort(-ref[0].astype(np.float64))
    sub[0, order[0]], sub[0, order[1]] = ref[0, order[1]], ref[0, order[0]]

    diff = np.abs(sub.astype(np.float64) - ref.astype(np.float64)).ravel()
    assert float(np.quantile(diff, 0.99)) == 0.0, "a 2-element plant must not move p99"
    assert diff.max() > 0.0, "but the maximum sees it, which is why it excused itself"

    got = cc.compare_outputs([sub], [ref], ["logits"])
    assert got["verdict"] == "DIVERGENT"
    assert got["outputs"][0]["task"]["argmax_agreement"] < 1.0


def test_all_zero_output_is_divergent():
    """The historical failure mode in this codebase: a kernel that writes nothing."""
    ref = _logits(3)
    got = cc.compare_outputs([np.zeros_like(ref)], [ref], ["logits"])
    assert got["verdict"] == "DIVERGENT"
    assert "identically zero" in got["outputs"][0]["detail"]


def test_nan_output_is_divergent_and_not_silently_excluded():
    """A NaN must not be filtered out of the comparison as 'non-finite'.

    Excluding non-finite pairs and comparing only the rest is how an arm that
    produced NaN over half its output passes an equivalence check.
    """
    ref = _logits(4)
    sub = ref.copy()
    sub[0, 0] = np.float16(np.nan)
    got = cc.compare_outputs([sub], [ref], ["logits"])
    assert got["verdict"] == "DIVERGENT"
    assert got["outputs"][0]["nonfinite_subject"] == 1


def test_within_budget_matches_and_reports_the_distance():
    ref = _separated_logits()
    sub = ref.copy()
    idx = 10
    sub[1, idx] = np.nextafter(np.nextafter(ref[1, idx], np.float16(0.0)),
                               np.float16(0.0))
    got = cc.compare_outputs([sub], [ref], ["logits"])
    entry = got["outputs"][0]
    assert entry["max_peak_ulp"] <= entry["peak_ulp_budget"]
    assert entry["max_raw_ulp"] == 2
    assert got["verdict"] == "MATCH"


def test_over_budget_is_divergent():
    ref = _separated_logits()
    sub = ref.copy()
    spacing = cc.peak_ulp_spacing(ref)
    budget = cc.equivalence_budget_ulp("float16", "float32")
    sub[2, 3] = np.float16(float(ref[2, 3]) + (budget + 8) * spacing)
    got = cc.compare_outputs([sub], [ref], ["logits"])
    assert got["outputs"][0]["max_peak_ulp"] > budget
    assert got["verdict"] == "DIVERGENT"


def test_planted_argmax_is_caught(request):
    """A whole row's winner replaced by a token that was not in the top 100."""
    ref = _separated_logits(rows=8)
    sub = ref.copy()
    sub[:, 40] = np.float16(float(ref[:, 0].max()) + 8.0)
    got = cc.compare_outputs([sub], [ref], ["logits"])
    assert got["verdict"] == "DIVERGENT"


def test_row_permutation_is_caught():
    """Right values, wrong rows: an indexing bug that preserves every statistic."""
    ref = _separated_logits(rows=8)
    sub = np.roll(ref, 1, axis=0)
    got = cc.compare_outputs([sub], [ref], ["logits"])
    assert got["verdict"] == "DIVERGENT"


def test_shape_mismatch_is_divergent_not_a_crash():
    got = cc.compare_outputs([np.zeros((2, 3), np.float16)],
                             [np.zeros((2, 4), np.float16)], ["logits"])
    assert got["verdict"] == "DIVERGENT"


# ---------------------------------------------------------------------------
# compare_workload — the arithmetic that produces the headline
# ---------------------------------------------------------------------------

def _rec(arm: str, samples, verdict=cc.ADMISSIBLE, repeat=0) -> dict:
    return {"arm": arm, "verdict": verdict, "steady_ms": list(samples), "repeat": repeat}


def _workload() -> cw.Workload:
    return cw.Workload(key="prefill_16", model_key="phi35_int4", family="prefill",
                       seq_len=16, past_len=0)


def _matched(*arms: str) -> dict:
    """An equivalence result in which every named arm compared and agreed.

    Supplied explicitly by the arithmetic tests below because `compare_workload` now
    refuses an arm it has no equivalence verdict for — see
    `test_an_arm_with_timings_and_no_equivalence_verdict_is_refused_by_name`. Passing
    this is not a convenience: it is the state a real run is in by the time a ratio is
    computed, and spelling it out here keeps these tests about the arithmetic instead of
    about the gate.
    """
    return {"verdict": cc.EQ_COMPARED, "reference_arm": cc.ARM_CPU_HOST,
            "arms": {arm: {"verdict": "MATCH", "outputs": [{"name": "logits"}]}
                     for arm in arms}}


_BOTH_GPU_ARMS_MATCHED = (cc.ARM_VULKAN, cc.ARM_CUDA)


def test_missing_arm_yields_no_ratio():
    """A missing arm is not a losing arm."""
    got = cc.compare_workload(_workload(), [_rec(cc.ARM_VULKAN, [10.0] * 10)],
                              equivalence=_matched(*_BOTH_GPU_ARMS_MATCHED))
    assert got["verdict"] == cc.UNMEASURED
    assert "speedup_vulkan_over_cuda" not in got


def test_split_frame_samples_are_excluded_from_the_ratio():
    """A SPLIT_FRAME arm must not contribute samples to a headline number."""
    records = [
        _rec(cc.ARM_VULKAN, [10.0] * 10, verdict=cc.SPLIT_FRAME),
        _rec(cc.ARM_CUDA, [20.0] * 10),
    ]
    got = cc.compare_workload(_workload(), records,
                              equivalence=_matched(*_BOTH_GPU_ARMS_MATCHED))
    assert got["n_vulkan"] == 0
    assert got["verdict"] == cc.UNMEASURED


def test_divergent_arm_disqualifies_the_ratio():
    """Faster-but-wrong is not faster."""
    records = [_rec(cc.ARM_VULKAN, [5.0] * 10), _rec(cc.ARM_CUDA, [20.0] * 10)]
    eq = _matched(cc.ARM_CUDA)
    eq["arms"][cc.ARM_VULKAN] = {"verdict": "DIVERGENT"}
    got = cc.compare_workload(_workload(), records, equivalence=eq)
    assert got["verdict"] == cc.NOT_EQUIVALENT
    assert "speedup_vulkan_over_cuda" not in got


def test_vulkan_faster_requires_the_whole_interval_above_one():
    records = [_rec(cc.ARM_VULKAN, [10.0 + 0.01 * i for i in range(30)]),
               _rec(cc.ARM_CUDA, [30.0 + 0.01 * i for i in range(30)])]
    got = cc.compare_workload(_workload(), records,
                              equivalence=_matched(*_BOTH_GPU_ARMS_MATCHED))
    assert got["verdict"] == "VULKAN_FASTER"
    assert got["speedup_vulkan_over_cuda"]["lo"] > 1.0


def test_overlapping_arms_are_indistinguishable_not_a_winner():
    """The gate that stops a coin-flip from becoming a headline."""
    rng = np.random.default_rng(11)
    a = list(10.0 + rng.standard_normal(40))
    b = list(10.0 + rng.standard_normal(40))
    got = cc.compare_workload(_workload(), [_rec(cc.ARM_VULKAN, a), _rec(cc.ARM_CUDA, b)],
                              equivalence=_matched(*_BOTH_GPU_ARMS_MATCHED))
    assert got["verdict"] == "INDISTINGUISHABLE"


def test_cuda_faster_requires_the_whole_interval_below_one():
    records = [_rec(cc.ARM_VULKAN, [45.0 + 0.01 * i for i in range(30)]),
               _rec(cc.ARM_CUDA, [15.0 + 0.01 * i for i in range(30)])]
    got = cc.compare_workload(_workload(), records,
                              equivalence=_matched(*_BOTH_GPU_ARMS_MATCHED))
    assert got["verdict"] == "CUDA_FASTER"
    assert got["speedup_vulkan_over_cuda"]["hi"] < 1.0


def test_runtime_version_confound_bracket_is_reported_when_both_cpu_arms_ran():
    records = [
        _rec(cc.ARM_VULKAN, [45.0] * 10), _rec(cc.ARM_CUDA, [15.0] * 10),
        _rec(cc.ARM_CPU_HOST, [104.0] * 10), _rec(cc.ARM_CPU_CUDA_RT, [100.0] * 10),
    ]
    got = cc.compare_workload(_workload(), records,
                              equivalence=_matched(*_BOTH_GPU_ARMS_MATCHED))
    conf = got["runtime_version_confound"]
    assert conf["ratio_cpu_cudart_over_cpu_host"]["ratio"] == pytest.approx(100.0 / 104.0)


def test_confound_bracket_is_absent_when_a_cpu_arm_is_missing():
    """No half-bracket: one CPU arm alone cannot bound a version difference."""
    records = [_rec(cc.ARM_VULKAN, [45.0] * 10), _rec(cc.ARM_CUDA, [15.0] * 10),
               _rec(cc.ARM_CPU_HOST, [104.0] * 10)]
    got = cc.compare_workload(_workload(), records,
                              equivalence=_matched(*_BOTH_GPU_ARMS_MATCHED))
    # The ratio itself was computed, so the absence below is the bracket's rule and not
    # a comparison that refused before it got there.
    assert got["verdict"] in ("VULKAN_FASTER", "CUDA_FASTER", "INDISTINGUISHABLE")
    assert "runtime_version_confound" not in got


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_refuses_below_three_samples():
    got = cc.bootstrap_ratio_ci([1.0, 2.0], [3.0, 4.0])
    assert got["lo"] is None
    assert "no interval is quotable" in got["note"]


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    """Two independent invocations must agree.

    Written as two named calls rather than ``f(a, b) == f(a, b)``. The claim is about
    what happens *across* invocations, so the two sides being separate calls is the
    substance of the test, not an incidental detail -- and the identical-operands form
    reads as a plain tautology to anyone who does not already know the function reseeds.
    It reads that way to the lane screen too, which is what caught it.

    The regression actually guarded here is the seed ceasing to be fixed:
    ``bootstrap_ratio_ci`` builds ``random.Random(seed)`` fresh from a constant default,
    so passing ``seed=None`` (OS entropy) makes two calls disagree and fails this test.
    Verified by negative control. The function holds no RNG state between calls, so this
    is the only way it can become non-reproducible -- which is worth saying, because
    otherwise the next reader will draw the same conclusion the screen did.
    """
    a = [10.0 + 0.1 * i for i in range(20)]
    b = [20.0 + 0.1 * i for i in range(20)]
    first = cc.bootstrap_ratio_ci(a, b)
    second = cc.bootstrap_ratio_ci(a, b)
    assert first == second
    # Determinism of a *refusal* would satisfy the assertion above while saying nothing
    # about the resampling, so require a real interval before believing the claim.
    assert first["lo"] is not None, "a refused interval makes the agreement above vacuous"
    assert first["hi"] > first["lo"], "a degenerate interval is deterministic for free"


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(7)
    a = list(10.0 + rng.standard_normal(50) * 0.5)
    b = list(20.0 + rng.standard_normal(50) * 1.0)
    got = cc.bootstrap_ratio_ci(a, b)
    assert got["lo"] <= got["ratio"] <= got["hi"]


# ---------------------------------------------------------------------------
# workloads and feeds
# ---------------------------------------------------------------------------

class _FakeMeta:
    def __init__(self, name, type_, shape):
        self.name, self.type, self.shape = name, type_, shape


class _FakeSession:
    def __init__(self, inputs):
        self._inputs = inputs

    def get_inputs(self):
        return self._inputs


def test_prefill_lengths_are_the_ones_issue_69_asks_for():
    keys = {w.key for w in cw.llm_workloads() if w.family == "prefill"}
    assert keys == {f"prefill_{n}" for n in (1, 2, 4, 8, 16, 32, 64, 128)}


def test_every_decode_workload_has_a_non_empty_kv_cache():
    """A decode benchmark with an empty cache measures a different kernel."""
    for w in cw.llm_workloads():
        if w.family == "decode":
            assert w.past_len > 0
            assert w.seq_len == 1


def test_feeds_bind_kv_dimensions_from_the_workload():
    inputs = [
        _FakeMeta("input_ids", "tensor(int64)", ["batch_size", "sequence_length"]),
        _FakeMeta("attention_mask", "tensor(int64)", ["batch_size", "total_sequence_length"]),
        _FakeMeta("past_key_values.0.key", "tensor(float16)",
                  ["batch_size", 32, "past_sequence_length", 96]),
    ]
    w = cw.Workload(key="decode_past512", model_key="m", family="decode",
                    seq_len=1, past_len=512)
    feeds = cw.build_feeds(_FakeSession(inputs), w)
    assert feeds.arrays["input_ids"].shape == (1, 1)
    assert feeds.arrays["attention_mask"].shape == (1, 513)
    assert feeds.arrays["past_key_values.0.key"].shape == (1, 32, 512, 96)


def test_feeds_refuse_an_unbound_symbolic_dimension():
    """A guessed dimension silently changes how much work the arm does."""
    inputs = [_FakeMeta("x", "tensor(float)", ["batch_size", "mystery_dim"])]
    w = cw.Workload(key="w", model_key="m", family="prefill", seq_len=4, past_len=0)
    with pytest.raises(ValueError, match="unbound symbolic dimension"):
        cw.build_feeds(_FakeSession(inputs), w)


def test_feeds_are_byte_identical_across_calls_with_the_same_seed():
    """The whole comparison rests on the arms being fed the same bytes."""
    inputs = [
        _FakeMeta("input_ids", "tensor(int64)", ["batch_size", "sequence_length"]),
        _FakeMeta("h", "tensor(float16)", ["batch_size", "sequence_length", 32]),
    ]
    w = cw.Workload(key="prefill_8", model_key="m", family="prefill", seq_len=8, past_len=0)
    a = cw.build_feeds(_FakeSession(inputs), w, seed=99)
    b = cw.build_feeds(_FakeSession(inputs), w, seed=99)
    assert a.digest == b.digest


def test_feed_digest_separates_dtype_from_content():
    """Same bytes, different dtype, must not digest the same."""
    i64 = [_FakeMeta("x", "tensor(int64)", [1, 4])]
    i32 = [_FakeMeta("x", "tensor(int32)", [1, 4])]
    w = cw.Workload(key="w", model_key="m", family="prefill", seq_len=4, past_len=0)
    assert cw.build_feeds(_FakeSession(i64), w).digest != \
        cw.build_feeds(_FakeSession(i32), w).digest


def test_position_ids_start_at_past_len():
    inputs = [_FakeMeta("position_ids", "tensor(int64)",
                        ["batch_size", "sequence_length"])]
    w = cw.Workload(key="decode_past128", model_key="m", family="decode",
                    seq_len=1, past_len=128)
    feeds = cw.build_feeds(_FakeSession(inputs), w)
    assert feeds.arrays["position_ids"].tolist() == [[128]]


# ---------------------------------------------------------------------------
# model provenance
# ---------------------------------------------------------------------------

def test_unknown_model_key_is_an_error_not_a_guess():
    got = bm.resolve("not-a-real-model")
    assert got.status == bm.MODEL_RESOLVE_ERROR


def test_url_model_absent_without_allow_download_is_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv(bm.CACHE_ENV, str(tmp_path))
    got = bm.resolve("mobilenetv2_12", allow_download=False)
    assert got.status == bm.MODEL_ABSENT
    assert "never downloads implicitly" in got.detail


def test_bundle_digest_is_order_independent():
    entries = [{"name": "a", "sha256": "aa", "bytes": 1},
               {"name": "b", "sha256": "bb", "bytes": 2}]
    assert bm._bundle_digest(entries) == bm._bundle_digest(list(reversed(entries)))


def test_bundle_digest_changes_when_a_weight_file_changes():
    """External data must be inside the identity, or two weight sets share a name."""
    base = [{"name": "m.onnx", "sha256": "aa", "bytes": 10},
            {"name": "m.onnx.data", "sha256": "bb", "bytes": 2_000_000_000}]
    swapped = [dict(base[0]), {**base[1], "sha256": "cc"}]
    assert bm._bundle_digest(base) != bm._bundle_digest(swapped)


def test_digest_mismatch_is_a_finding_not_a_repin(tmp_path, monkeypatch):
    monkeypatch.setenv(bm.CACHE_ENV, str(tmp_path))
    root = tmp_path / "mobilenetv2_12"
    root.mkdir(parents=True)
    (root / "mobilenetv2-12.onnx").write_bytes(b"not really a model")
    got = bm.resolve("mobilenetv2_12", expect_bundle_sha256="0" * 64)
    assert got.status == bm.MODEL_DIGEST_MISMATCH
    assert "not a re-pin" in got.detail


def test_every_spec_records_a_licence():
    for key, spec in bm.SPECS.items():
        assert spec.license_id, f"{key} has no licence id"
        assert spec.license_url, f"{key} has no licence url"


# ---------------------------------------------------------------------------
# arm wiring
# ---------------------------------------------------------------------------

def test_every_arm_has_a_runtime_and_a_provider():
    for arm in cc.ARMS:
        assert arm in cc.ARM_RUNTIME
        assert arm in cc.ARM_PROVIDER


def test_the_two_cpu_arms_are_the_same_provider_in_different_runtimes():
    """They are the version-confound bracket; if they ever diverge it is not one."""
    assert cc.ARM_PROVIDER[cc.ARM_CPU_HOST] == cc.ARM_PROVIDER[cc.ARM_CPU_CUDA_RT]
    assert cc.ARM_RUNTIME[cc.ARM_CPU_HOST] != cc.ARM_RUNTIME[cc.ARM_CPU_CUDA_RT]


def test_gpu_arms_live_in_the_same_runtimes_as_their_cpu_brackets():
    assert cc.ARM_RUNTIME[cc.ARM_VULKAN] == cc.ARM_RUNTIME[cc.ARM_CPU_HOST]
    assert cc.ARM_RUNTIME[cc.ARM_CUDA] == cc.ARM_RUNTIME[cc.ARM_CPU_CUDA_RT]


def test_missing_cuda_interpreter_is_a_refusal_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.delenv(cc.CUDA_PYTHON_ENV, raising=False)
    got = cc.dispatch_arm(cc.ARM_CUDA, _workload(), iters=1, warmup=0, scratch=tmp_path,
                          device=0, seed=1)
    assert got["verdict"] == cc.UNMEASURED
    assert "absent, not slow" in got["refusals"][0]


# ---------------------------------------------------------------------------
# STALE SCRATCH: a crashing worker must not inherit the previous run's answer
#
# `dispatch_arm` names its two scratch slots after the arm and the workload and nothing
# else, so the second run of an arm lands on the first run's files.  It then decides
# "the worker wrote no record" from `out.is_file()` and hands `outputs_dir` to
# `cross_arm_equivalence`.  Plant an ADMISSIBLE record and a tensor in those slots, crash
# the worker, and — before `_clear_stale_arm_scratch` — the parent reads a pass and
# compares yesterday's tensor.  These are the falsifiers for that.
# ---------------------------------------------------------------------------

def _plant_stale_arm_scratch(scratch: Path, arm: str, workload) -> "tuple[Path, Path]":
    """Write the previous run's ADMISSIBLE record and its output tensor into the slots."""
    out = scratch / f"result_{arm}_{workload.key}.json"
    dump = scratch / f"outputs_{arm}_{workload.key}"
    dump.mkdir(parents=True, exist_ok=True)
    tensor = dump / "out0.npy"
    np.save(tensor, np.arange(4, dtype=np.float32))
    out.write_text(json.dumps({
        "arm": arm, "workload": workload.key, "model_key": workload.model_key,
        "verdict": cc.ADMISSIBLE, "refusals": [], "instrument_errors": [],
        "samples_ms": [1.0, 1.0, 1.0], "stale_marker": "written by the PREVIOUS run",
        "outputs_manifest": [{"name": "logits", "dtype": "float32", "shape": [4],
                              "file_rel": cc.OUTPUT_HANDLE}],
    }), encoding="utf-8")
    return out, dump


def _crashing_worker(monkeypatch):
    """Make `subprocess.run` behave like a worker that died before writing anything."""
    class _Dead:
        returncode = -1073741819  # 0xC0000005, an access violation on Windows
        stdout = b""
        stderr = b"Fatal Python error: Segmentation fault"

    calls: "list[list[str]]" = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _Dead()

    monkeypatch.setattr(cc.subprocess, "run", _fake_run)
    return calls


def test_a_crashed_worker_does_not_inherit_the_previous_runs_record(monkeypatch, tmp_path):
    """No record written this run → INSTRUMENT_ERROR, even when the slot already held one."""
    workload = _workload()
    out, dump = _plant_stale_arm_scratch(tmp_path, cc.ARM_CPU_HOST, workload)
    _crashing_worker(monkeypatch)

    got = cc.dispatch_arm(cc.ARM_CPU_HOST, workload, iters=1, warmup=0, scratch=tmp_path,
                          device=0, seed=1)

    assert got["verdict"] == cc.INSTRUMENT_ERROR, got
    assert "without writing a record" in got["instrument_errors"][0]
    assert "stale_marker" not in got
    assert got.get("verdict") != cc.ADMISSIBLE
    assert not out.exists(), "the previous run's record survived into this run"


def test_a_crashed_worker_does_not_leave_the_previous_runs_tensors_to_be_compared(
        monkeypatch, tmp_path):
    """The equivalence input must be UNMEASURED, not yesterday's `out0.npy`."""
    workload = _workload()
    out, dump = _plant_stale_arm_scratch(tmp_path, cc.ARM_CPU_HOST, workload)
    stale_tensor = dump / "out0.npy"
    assert stale_tensor.is_file()
    _crashing_worker(monkeypatch)

    got = cc.dispatch_arm(cc.ARM_CPU_HOST, workload, iters=1, warmup=0, scratch=tmp_path,
                          device=0, seed=1)

    assert not stale_tensor.exists(), "a stale tensor is still in the equivalence path"
    assert not got.get("outputs_manifest")
    eq = cc.cross_arm_equivalence([got, {"arm": cc.ARM_VULKAN, "outputs_manifest": []}],
                                  reference_arm=cc.ARM_CPU_HOST)
    # Fail closed: the arm that could not be compared is named with a verdict, not left
    # out of an empty map that `compare_workload` would have read as "nobody was
    # disqualified".
    assert eq["verdict"] == cc.EQ_REFUSED
    assert eq["arms"][cc.ARM_VULKAN]["verdict"] == cc.UNMEASURED
    assert "UNMEASURED" in eq["detail"]


def test_the_slots_are_cleared_before_the_worker_starts_not_after_it_fails(
        monkeypatch, tmp_path):
    """Clearing after the fact would still let a *hanging* worker's timeout read stale.

    The observation is ordering, so it is made from inside the call: when the subprocess
    starts, both slots must already be empty.
    """
    workload = _workload()
    out, dump = _plant_stale_arm_scratch(tmp_path, cc.ARM_CPU_HOST, workload)
    seen: "list[tuple[bool, list[str]]]" = []

    def _fake_run(cmd, **kwargs):
        seen.append((out.exists(), sorted(p.name for p in dump.iterdir())))
        raise cc.subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(cc.subprocess, "run", _fake_run)
    got = cc.dispatch_arm(cc.ARM_CPU_HOST, workload, iters=1, warmup=0, scratch=tmp_path,
                          device=0, seed=1, timeout=1)

    assert seen == [(False, [])], seen
    assert got["verdict"] == cc.INSTRUMENT_ERROR
    assert "timed out" in got["instrument_errors"][0]


def test_a_slot_that_cannot_be_cleared_refuses_instead_of_running_the_arm(
        monkeypatch, tmp_path):
    """Fail closed: an unclearable slot is an instrument error, not a run with a fallback.

    A sub-directory in the output slot was not written by `_npy_dump`, so this module will
    not delete it — and it will not run the worker over it either, because the whole point
    of the clearing step is that what is read afterwards was written by this run.
    """
    workload = _workload()
    _plant_stale_arm_scratch(tmp_path, cc.ARM_CPU_HOST, workload)
    (tmp_path / f"outputs_{cc.ARM_CPU_HOST}_{workload.key}" / "nested").mkdir()
    calls = _crashing_worker(monkeypatch)

    got = cc.dispatch_arm(cc.ARM_CPU_HOST, workload, iters=1, warmup=0, scratch=tmp_path,
                          device=0, seed=1)

    assert got["verdict"] == cc.INSTRUMENT_ERROR, got
    assert "refusing to clear" in got["instrument_errors"][0]
    assert calls == [], "the arm was dispatched over a slot that could not be cleared"


def test_clearing_the_slots_does_not_break_the_worker_that_does_write(monkeypatch, tmp_path):
    """The must-pass side: a record written by THIS run is returned, stale marker gone."""
    workload = _workload()
    out, dump = _plant_stale_arm_scratch(tmp_path, cc.ARM_CPU_HOST, workload)

    class _Ok:
        returncode = 0
        stdout = b""
        stderr = b""

    def _fake_run(cmd, **kwargs):
        target = Path(cmd[cmd.index("--out") + 1])
        target.write_text(json.dumps({
            "arm": cc.ARM_CPU_HOST, "workload": workload.key,
            "model_key": workload.model_key, "verdict": cc.ADMISSIBLE,
            "refusals": [], "instrument_errors": [], "samples_ms": [2.0],
            "outputs_manifest": [{"name": "logits", "dtype": "float32", "shape": [2]}],
        }), encoding="utf-8")
        return _Ok()

    monkeypatch.setattr(cc.subprocess, "run", _fake_run)
    got = cc.dispatch_arm(cc.ARM_CPU_HOST, workload, iters=1, warmup=0, scratch=tmp_path,
                          device=0, seed=1)

    assert got["verdict"] == cc.ADMISSIBLE
    assert "stale_marker" not in got
    assert got["samples_ms"] == [2.0]
    assert got["worker_returncode"] == 0


def test_the_clearer_will_not_empty_a_directory_it_did_not_name(tmp_path):
    """Narrowness, directly: a path outside the slot shape keeps its contents."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    foreign = scratch / "results"
    foreign.mkdir()
    (foreign / "keep.npy").write_bytes(b"0")
    kept = scratch / "notes.json"
    kept.write_text("{}", encoding="utf-8")

    problems = cc._clear_stale_arm_scratch(kept, foreign, scratch)

    assert kept.is_file(), "a file outside the result slot shape was deleted"
    assert (foreign / "keep.npy").is_file(), "a directory outside the slot shape was emptied"
    assert any("refusing to clear" in p for p in problems), problems


# ---------------------------------------------------------------------------
# suite schema
# ---------------------------------------------------------------------------

def test_summarise_of_empty_is_not_zero():
    got = cc.summarise([])
    assert got["median_ms"] is None
    assert got["n"] == 0


def test_render_does_not_crash_on_an_empty_suite():
    suite = {"schema": "cuda_competition/1", "device_facts": {}, "iters": 1, "warmup": 0,
             "repeats": 1, "results": [], "comparisons": [], "equivalence": {}}
    assert "Vulkan EP vs ORT CUDA EP" in cc.render(suite)


@pytest.mark.parametrize("path", sorted(
    (_HERE / "results" / "_cuda69").glob("*.json")))
def test_a_committed_record_that_refuses_does_not_also_pass(path):
    """An admissible committed artifact must carry **no** refusals. `refusals == []`.

    The defect this is the falsifier for, verbatim from the artifact it was found in:

        "verdict": "GPU_TIME_MEASURED",
        "refusals": ["phase 'bind_check' declares nested_in='none' but this module lists it
                      as nested; trace.rs and cuda_profile.py disagree about the phase tree"]

    A module that says "I cannot read my own instrument" and a verdict that says "measured"
    cannot both be true of one document. The refusal was written by a cross-check the same
    change had broken, and it survived review, a green bench suite and a green Rust suite,
    because every existing assertion checked the *verdict enum* and none checked whether the
    record refused.

    A refusal is not a warning. It means a claim below it may not be read. So a committed
    record either has none, or it does not get to carry a verdict that reads like a result:

    * ``cuda_profile``: ``GPU_TIME_MEASURED`` requires ``refusals == []``. The two
      *withholding* verdicts (``GPU_TIME_UNAVAILABLE``, ``TRACE_ABSENT``) require at least
      one, because a withheld measurement with no stated reason is worse than either.
    * ``cuda_competition``: any ``ADMISSIBLE`` arm requires ``refusals == []`` — that is what
      the word means — and every non-admissible arm must say why.

    ``scope_limits`` is the separate, weaker channel: a bounded-but-sound claim. It is
    deliberately *not* a refusal, and equally deliberately not silent.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema") or ""

    if schema.startswith("cuda_profile/"):
        refusals = payload.get("refusals") or []
        if payload.get("verdict") == "GPU_TIME_MEASURED":
            assert refusals == [], (
                f"{path.name} reports GPU_TIME_MEASURED and refuses at the same time:\n  "
                + "\n  ".join(refusals))
        else:
            assert refusals, (
                f"{path.name} withholds a measurement ({payload.get('verdict')!r}) without "
                f"recording a reason")
        # The phase-tree cross-check specifically: it fires when trace.rs and cuda_profile.py
        # disagree about which phases may be summed, and a committed record must never
        # contain that disagreement.
        assert (payload.get("phases") or {}).get("phase_tree_disagreements", []) == [], (
            f"{path.name} carries a phase-tree disagreement: "
            f"{payload['phases']['phase_tree_disagreements']}")
        return

    if schema.startswith("cuda_competition/"):
        for rec in payload.get("results", []):
            refusals = rec.get("refusals") or []
            if rec.get("verdict") == cc.ADMISSIBLE:
                assert refusals == [], (
                    f"{path.name}: {rec.get('arm')}/{rec.get('workload')} is ADMISSIBLE and "
                    f"refuses: {refusals}")
            else:
                assert refusals or rec.get("instrument_errors"), (
                    f"{path.name}: {rec.get('arm')}/{rec.get('workload')} is "
                    f"{rec.get('verdict')!r} with no refusal and no instrument error — "
                    f"a verdict with no stated reason cannot be audited")


@pytest.mark.parametrize("path", sorted(
    (_HERE / "results" / "_cuda69").glob("*.json")))
def test_committed_records_are_readable_and_self_describing(path):
    """Any committed record under ``_cuda69/`` must declare a schema this file knows.

    Two schemas live here — ``cuda_competition/N`` (a suite run) and
    ``bench_models/N`` (a provenance resolution).  An unrecognised schema is a
    failure rather than a skip: a record nobody can classify is a record nobody can
    audit, and silently skipping it is how a stale artifact survives a refactor.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    assert schema, f"{path.name} carries no schema field"

    if schema.startswith("bench_models/"):
        for key, rec in payload["models"].items():
            assert rec["status"] in {bm.MODEL_OK, bm.MODEL_ABSENT,
                                     bm.MODEL_DIGEST_MISMATCH, bm.MODEL_RESOLVE_ERROR}
            if rec["status"] == bm.MODEL_OK:
                assert rec["bundle_sha256"], f"{key} is OK with no bundle digest"
                assert rec["license_id"], f"{key} is OK with no licence"
        return

    if schema.startswith("cuda_profile/"):
        assert payload.get("verdict") in {
            "GPU_TIME_MEASURED", "GPU_TIME_UNAVAILABLE", "TRACE_ABSENT",
        }, f"{path.name}: unknown profile verdict {payload.get('verdict')!r}"
        if payload["verdict"] == "GPU_TIME_MEASURED":
            # Device time must come from the warm calls, not from a mean that a cold
            # first Compute dominates.  The cold call carries the whole weight upload;
            # averaging it in describes a regime that never occurs.
            assert payload.get("gpu_ms_per_run_basis") == "warm_call_median", (
                f"{path.name}: per-run device time must be a warm-call median, got "
                f"{payload.get('gpu_ms_per_run_basis')!r}")
            assert payload.get("steady", {}).get("warm_calls", 0) > 0
            assert payload.get("rerecord", {}).get("verdict") in {
                "REPLAYED", "RERECORDED_EVERY_CALL", "PARTIALLY_RERECORDED", "UNKNOWN",
            }
            # An attribution anchored on the inner span cannot rule anything out beyond
            # it.  `vulkan.compute_call` is the outer bracket -- the instrumented
            # success-path region inside `compute_impl`, not ORT's literal `Compute`
            # entry -- and anchored on the inner `vulkan.subgraph` instead, the
            # reduction is blind to the region where the counters-dump artifact was
            # hiding — a region larger than the device time the trace *did* see.  The
            # magnitude is quoted where its profile is committed, which is not this
            # branch.
            anchor = payload.get("anchor") or {}
            assert anchor.get("span"), (
                f"{path.name}: no anchor recorded. A profile that does not say which span "
                f"it bucketed against cannot be audited: the same numbers mean different "
                f"things depending on whether the outer bracket was used.")
            if not anchor.get("sees_compute_call_bracket"):
                # Admissible, but only as a bounded claim, and the bound must be on the
                # artifact rather than in whoever remembers the run.
                assert payload.get("scope_limits"), (
                    f"{path.name}: anchored on {anchor['span']!r}, which cannot see the "
                    f"region `vulkan.compute_call` would bucket, and carries no "
                    f"scope_limits saying so.")
        return

    assert schema.startswith("cuda_competition/"), f"{path.name}: unknown schema {schema!r}"
    for rec in payload.get("results", []):
        assert rec.get("verdict") in {
            cc.ADMISSIBLE, cc.SPLIT_FRAME, cc.PROVIDER_ABSENT, cc.NOTHING_CLAIMED,
            cc.NOT_EQUIVALENT, cc.UNMEASURED, cc.INSTRUMENT_ERROR,
        }, f"{path.name}: unknown verdict {rec.get('verdict')!r}"
        if rec["verdict"] == cc.ADMISSIBLE:
            # An admissible arm must carry the evidence that made it admissible.
            assert rec.get("model_bundle_sha256"), "admissible arm with no model digest"
            assert rec.get("feed_digest"), "admissible arm with no feed digest"
            assert rec.get("providers_held"), "admissible arm with no provider list"
    for comp in payload.get("comparisons", []):
        assert comp.get("verdict") in {
            "VULKAN_FASTER", "CUDA_FASTER", "INDISTINGUISHABLE",
            cc.UNMEASURED, cc.NOT_EQUIVALENT,
        }, f"{path.name}: unknown comparison verdict {comp.get('verdict')!r}"


# ---------------------------------------------------------------------------
# Instrumentation must not be inside the measurement
# ---------------------------------------------------------------------------
#
# The defect these cover, in full, because it cost a whole baseline sweep:
#
# `dispatch_arm` set ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE before session creation and
# never unset it. In the EP, `counters::record_dispatches` — which runs after EVERY
# Compute — calls `dump_if_requested`, and that READS AND REWRITES the entire counters
# JSON document. So every timed Vulkan inference carried a file read, a JSON rebuild and
# a file write that the CUDA arm had no equivalent of.
#
# The median with the dump live was well above the median without it — all of it this
# harness measuring itself, and all of it charged to the Vulkan EP in a report whose
# entire purpose is to compare the Vulkan EP against CUDA. A meaningful share of the
# gap the first baseline attributed to CUDA being faster was, instead, the instrument.
# **No magnitude is quoted on this branch**: the A/B artifact that would demonstrate one
# is not committed here, and a ratio whose witness lives elsewhere (or nowhere) is the
# kind of claim this harness exists to refuse.
#
# The trap is that nothing looked wrong. Every gate passed, every arm read ADMISSIBLE,
# equivalence read MATCH, and the numbers were stable across repeated iterations with a
# tight confidence interval — a precisely reproducible measurement of the wrong thing.
# It was only found by asking where a large block of untraced host time went.
#
# So the invariant is not "remember to unset the variable". It is that evidence
# collection and timing are separate regions, and the code says which is which.


def test_counters_dump_is_not_left_inside_the_timed_region():
    """The counters env var must be dropped before anything that is timed.

    Asserted on the source rather than by running a session, because the failure is a
    *timing* artifact: a functional test passes cheerfully while the number it produces
    is measurably inflated. What can be checked cheaply and
    deterministically is the ordering
    — the pop must appear after the first run and before the warmup loop.
    """
    src = Path(cc.__file__).read_text(encoding="utf-8")
    pop = src.index("os.environ.pop(COUNTERS_ENV, None)")
    first_run = src.index("rec.first_run_ms =")
    warmup = src.index("for _ in range(warmup):")
    steady = src.index("for _ in range(iters):")
    assert first_run < pop < warmup < steady, (
        "the counters-file env var must be unset after the first run and before the "
        "warmup/steady loops. The EP rewrites that JSON after every Compute, so leaving "
        "it set puts a file write inside every timed inference — measurably inflating "
        "the result. No magnitude is quoted here; see the module-level comment above "
        "for why.")



def test_counters_scope_is_recorded_on_every_vulkan_arm():
    """The record must say which regime it was measured under.

    A number that was inflated by instrumentation and a number that was not are not
    interchangeable, and the withdrawn `baseline_main*.json` files contained the former:
    their Vulkan `prefill_1` median was far above the same arm's median in a clean run,
    and they said `ADMISSIBLE` with no refusals while carrying no
    `counters_scope` at all. The two medians are not quoted here — `baseline_main*` was
    withdrawn under blocker B1 and is committed nowhere, so the inflated side has no
    witness, and a comparison with one side missing is exactly what this field exists to
    prevent. Without it on the record there is nothing in the
    artifact that distinguishes the two regimes, and the only way to tell would be to
    remember — which is how the inflated numbers would have been quoted forever.

    This is the *shape* half. `test_every_committed_vulkan_record_declares_its_counters_scope`
    is the half that looks at what actually got committed.
    """
    assert "counters_scope" in {f.name for f in dataclasses.fields(cc.ArmResult)}
    rec = cc.ArmResult(arm=cc.ARM_VULKAN, workload="w", model_key="m")
    assert rec.counters_scope == "", "must start empty so 'never set' is distinguishable"


def test_keep_counters_flag_reproduces_the_artifact_on_purpose():
    """The escape hatch that proved the artifact stays wired.

    The A/B is the only evidence that the inflation claim is real, and an escape hatch that
    silently stops working takes the reproduction with it. The flag must be read with the
    same truthiness rules as the rest of the harness so that `=1` means what it says.
    """
    assert cc.KEEP_COUNTERS_ENV == "ONNXRUNTIME_EP_VULKAN_BENCH_KEEP_COUNTERS"
    for truthy in ("1", "true", "YES", "on"):
        os.environ[cc.KEEP_COUNTERS_ENV] = truthy
        try:
            assert cc._env_flag(cc.KEEP_COUNTERS_ENV) is True, truthy
        finally:
            os.environ.pop(cc.KEEP_COUNTERS_ENV, None)
    for falsy in ("0", "", "false", "no"):
        os.environ[cc.KEEP_COUNTERS_ENV] = falsy
        try:
            assert cc._env_flag(cc.KEEP_COUNTERS_ENV) is False, falsy
        finally:
            os.environ.pop(cc.KEEP_COUNTERS_ENV, None)
    assert cc._env_flag(cc.KEEP_COUNTERS_ENV) is False, "unset must be False"


# ---------------------------------------------------------------------------
# What actually got committed
# ---------------------------------------------------------------------------

def _committed_evidence_json():
    """Every JSON under `_cuda69/` that git is actually tracking.

    Reading the directory would also pick up a scratch file an operator happens to have
    left there, which is not what "committed" means and would make this screen's verdict
    depend on an untracked working tree.
    """
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    out = subprocess.run(["git", "ls-files", "-z", "bench/results/_cuda69"],
                         cwd=repo, capture_output=True, text=True, check=True).stdout
    return [repo / f for f in out.split("\0") if f.endswith(".json")]


def _measurement_records(node, path="$"):
    """Yield every `(json-path, record)` that looks like one arm's measurement."""
    if isinstance(node, dict):
        if "arm" in node and "verdict" in node:
            yield path, node
        for k, v in node.items():
            yield from _measurement_records(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _measurement_records(v, f"{path}[{i}]")


def test_committed_evidence_is_present_and_enumerable():
    """A screen that silently rules on nothing passes for the wrong reason.

    Skips, rather than fails, on a branch that carries no committed evidence at
    all. That is a real branch state, not a defect: the instrumentation head
    submitted for review deliberately publishes the harness and none of its
    output, so demanding evidence there would be demanding the thing under review
    not exist.

    That skip is not itself a hard guarantee against "the evidence vanished" on a
    branch that is supposed to publish results -- this module does not name or
    assert the existence of such a guard elsewhere, because doing so without
    verifying it in the tree that carries it would be exactly the kind of claim
    this branch exists to refuse. A results-publishing branch is responsible for
    its own hard assertion over its own committed evidence. What this test does
    guarantee is narrower and local: the moment *any* evidence is present here,
    every assertion below runs unchanged -- this is a skip on an empty tree, not
    a skip on an inconvenient result.
    """
    files = _committed_evidence_json()
    if not files:
        pytest.skip(
            "this branch publishes no committed evidence (instrumentation-only head); "
            "bench/test_result_staleness.py holds the hard guard where evidence lives")
    records = [r for f in files
               for r in _measurement_records(json.loads(f.read_text("utf-8")))]
    assert records, "committed evidence contains no measurement records"
    assert any(rec.get("arm") == cc.ARM_VULKAN for _, rec in records), (
        "no committed Vulkan record: the counters_scope screen below would be vacuous")


def test_every_committed_vulkan_record_declares_its_counters_scope():
    """A committed Vulkan number must say which instrumentation regime produced it.

    The EP rewrites its counters JSON after every `Compute`. With that dump left inside
    the timed region a Vulkan median is measurably inflated by the harness's own file
    write, and the resulting record is indistinguishable from a clean one unless it says
    so. The withdrawn `baseline_main_v2.json` was exactly that: 64 `ADMISSIBLE` results,
    no refusals, no `counters_scope`, and a Vulkan `prefill_1` median well above the
    corrected one. No ratio is quoted here: `baseline_main_v2.json` and the
    `baseline_fixed` it was compared against are both withdrawn and committed nowhere on
    this branch, so any figure computed from them would have no witness -- exactly the
    defect `test_no_measured_ratio_ships_without_a_committed_witness` below exists to
    catch structurally, not just for this one pair.

    Scoped to Vulkan arms because `counters_scope` describes *our* EP's counters; a CUDA
    or CPU arm has none and carries the field empty rather than absent.
    """
    offenders = []
    for f in _committed_evidence_json():
        doc = json.loads(f.read_text("utf-8"))
        for jpath, rec in _measurement_records(doc):
            if rec.get("arm") != cc.ARM_VULKAN:
                continue
            scope = rec.get("counters_scope")
            if "counters_scope" not in rec:
                offenders.append(f"{f.name}{jpath}: field absent")
            elif not scope:
                offenders.append(f"{f.name}{jpath}: empty ({scope!r})")
            elif scope not in cc.COUNTERS_SCOPES:
                offenders.append(f"{f.name}{jpath}: unknown scope {scope!r}")
    assert not offenders, (
        "committed Vulkan records without a declared counters scope:\n  "
        + "\n  ".join(offenders))


#: A bare "<digits>.<digits>x" literal in this module's own prose reads as a measured
#: result -- exactly the shape of the withdrawn baseline ratio this guard exists to keep
#: out. This module carries no committed evidence of its own
#: (`test_committed_evidence_is_present_and_enumerable` above skips when this branch
#: publishes none), so any such literal here would be unwitnessed by construction: a
#: real-looking number backed by nothing in the tree. A qualitative bound ("more than
#: 2x", "more than twice") is not this shape and is not the thing being guarded against
#: -- it makes no claim precise enough to need a witness.
_UNWITNESSED_RATIO_RE = re.compile(r"\b\d+\.\d+x\b")


def test_no_measured_ratio_ships_without_a_committed_witness():
    """This module may state the qualitative finding; it may not quote a magnitude for it.

    The rejected baseline ratio was a real-looking decimal, sourced from
    `baseline_main_v2.json` / `baseline_fixed`, both withdrawn and committed nowhere on
    this branch. Citing a number from an artifact nobody can read is the same defect
    `test_cuda_profile.py`'s citation-pin tests exist to catch for `outside_subgraph_ms`
    -- but *this* module has no committed artifact to pin a figure to at all, so the only
    sound fix here is to not state one. If a future run commits real evidence for a
    counters-dump ratio, it belongs in a JSON artifact under `bench/results/_cuda69/`
    with a citation pin like `test_cuda_profile.py`'s, not as a bare literal in this
    module's docstrings or assertion messages.

    The negative control below (`test_the_unwitnessed_ratio_guard_would_notice_a_regression`)
    necessarily writes an example of the forbidden shape to prove the regex can see it, so
    its own source -- read via `inspect.getsource`, not a textual marker that could collide
    with a look-alike string -- is excluded from the scan here rather than escaped in place.
    """
    src = Path(__file__).read_text("utf-8")
    exempt = inspect.getsource(test_the_unwitnessed_ratio_guard_would_notice_a_regression)
    assert src.count(exempt) == 1, "the exempted negative-control source must appear once"
    prose = src.replace(exempt, "", 1)
    offenders = _UNWITNESSED_RATIO_RE.findall(prose)
    assert not offenders, (
        f"this module quotes a decimal ratio literal with no committed witness: "
        f"{offenders}. Either delete the figure and keep the qualitative claim, or "
        f"commit the artifact that backs it under bench/results/_cuda69/ and cite it "
        f"through a pin, the way test_cuda_profile.py does for outside_subgraph_ms.")


def test_the_unwitnessed_ratio_guard_would_notice_a_regression():
    """Negative control: a guard that cannot fail is decoration.

    Exempted from the scan in `test_no_measured_ratio_ships_without_a_committed_witness`
    by source lookup, because this function's whole purpose is to contain the forbidden
    shape.
    """
    example_measured_figure = "a Vulkan prefill_1 median " + "1.61x" + " the corrected one"
    example_qualitative_bound = ("the median with the dump live was more than "
                                  "twice the median without it")
    assert _UNWITNESSED_RATIO_RE.search(example_measured_figure)
    assert not _UNWITNESSED_RATIO_RE.search(example_qualitative_bound)


# ---------------------------------------------------------------------------
# Fact Checker's de-claim scan: a general numeric-witness guard
# ---------------------------------------------------------------------------
#
# `_UNWITNESSED_RATIO_RE` above catches one shape (a bare decimal ratio). The de-claim scan
# found the same underlying defect in many other shapes across the production modules:
# decimal and integer timings, decimal and integer percentages, bare integer ratios,
# comma-grouped and compact counts, and sizes carrying a unit -- `2.3 GB`, `26 MB`,
# `2,291,238,912 bytes`, `45 ms`, `355`. So the shape half of this guard is deliberately
# wide.
#
# The *witness* half is the part the second review round rewrote, and it is the part that
# matters. The previous version accepted any figure that merely had citation-*looking* text
# nearby: the literal substring `bench/results/` within a few hundred characters was enough
# to bless it. That blesses three separate defects at once -- a path that does not exist in
# this tree, a real artifact that does not carry the figure at all, and a figure quoted from
# a *neighbouring* field of a real artifact ("2.2 GB" beside a witness that says
# 2,291,238,912 bytes). All three shipped on this branch and all three are now convictions:
#
#   1. every cited repo-relative path is parsed out and must exist;
#   2. the field the figure names *itself* -- adjacent to it in its own clause, or paired
#      with an explicit `[witness: `field`]` annotation there -- is looked up in the cited
#      artifact at that exact path. No wider scope is consulted: not a proximity window, not
#      the clause's other citations, not the enclosing paragraph. A pool is a pool whatever
#      its boundary, and a figure answered from a pool is a figure answered by a neighbour;
#   3. the figure must actually be *there* -- equal to that field, equal to it scaled by the
#      stated unit and rounded to the precision printed, or, for a percentage alone, equal to
#      a ratio of two fields named in the figure's own clause. Failing that, for an
#      unstructured witness only, the digits must at least appear verbatim in the cited file.
#
# The alternative to (3) is the standard withheld-figure disclaimer this codebase already
# uses, which claims nothing and so needs no witness.
#
# Only *prose* is scanned -- comments and string literals, extracted with `tokenize`. Code is
# not prose: `ROUNDING_DEPTH_BOUND = 128`, `MANTISSA_BITS`, a `sha256`/byte-count provenance
# table and a synthetic fixture's `{"CPUExecutionProvider": 8}` are constants and inputs, not
# claims about a run, and a guard that reads them as claims would be trained out of the tree
# within a week.

#: ISO dates (`2026-08-04`) and dotted version/serial numbers (`573.44`, `1.4.309`) are not
#: measurements. They are removed before scanning rather than special-cased in the pattern,
#: which keeps the pattern readable and keeps the exemption auditable in one place.
_NOT_A_MEASUREMENT_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d+(?:\.\d+){2,}\b|\b\d+\.\d+(?=\s*,?\s*r\d)")

#: Units a figure may carry. Unit-separated (`2.3 GB`), compact (`2.3GB`) and hyphenated
#: (`2.3-GB`) spellings are one pattern on purpose: the hyphen was a live escape hatch.
_UNIT_ALTERNATION = (
    r"x|ns|us|\u00b5s|ms|s|sec|secs|seconds|minutes|bytes|B|KB|KiB|MB|MiB|GB|GiB|TB|TiB"
    r"|ULP|ulp|dispatches|nodes|calls|spans|elements"
)

_MEASUREMENT_SHAPED_RE = re.compile(
    r"(?<![\w.$-])"
    r"(?P<approx>~\s?)?"
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:[\s-]?(?P<unit>%|(?:" + _UNIT_ALTERNATION + r")\b)(?!\s?(?:CI\b|confidence)))?"
    r"(?!\.?\d)"
)

#: Units whose figure is a scaled restatement of a byte count, with the scale to divide a
#: witnessed byte count by before comparing. Both decimal and binary readings are accepted:
#: a document may legitimately print either, and the guard's job is to catch a figure the
#: artifact cannot produce *at all*, not to legislate SI-vs-IEC.
_BYTE_SCALES = {
    "bytes": (1,), "b": (1,),
    "kb": (10 ** 3, 2 ** 10), "kib": (2 ** 10,),
    "mb": (10 ** 6, 2 ** 20), "mib": (2 ** 20,),
    "gb": (10 ** 9, 2 ** 30), "gib": (2 ** 30,),
    "tb": (10 ** 12, 2 ** 40), "tib": (2 ** 40,),
}

#: Phrases that mark a figure as deliberately withheld, or mark the surrounding numbers as a
#: constructed fixture rather than an observation. Neither kind is a claim about a run, so
#: neither needs a witness -- but the phrase excuses the *clause* it stands in and no other,
#: for the same reason a citation binds the clause it stands in and no other. Note what is
#: *not* here any more: `bench/results/`. A path-shaped substring is no longer a blessing --
#: it is now an obligation, checked below.
_WITHHELD_FIGURE_DISCLAIMERS = (
    "no specific", "no magnitude is quoted", "no figure is quoted", "no ratio is quoted",
    "not quoted here", "no dispatch count is quoted", "not witnessed by", "no committed artifact",
    "none is quoted", "none is witnessed", "no figure for it is quoted", "is not quoted",
    "synthetic fixture", "synthetic trace", "constructed fixture", "fixture, not a measurement",
)

#: Production modules this guard screens. Deliberately narrow: docs like docs/PERF.md carry
#: real, individually cited measurements throughout (see its own §26.1 provenance table) and a
#: blanket scan there would flag legitimate witnessed figures, not catch unwitnessed ones.
_NUMERIC_WITNESS_SCREENED_MODULES = ("cuda_profile.py", "cuda_competition.py", "bench_models.py")

#: Repo-relative citation targets. A citation is a *path*, and this is what "parse the path"
#: means: it is extracted, resolved against the repository root, and required to exist.
_CITED_PATH_RE = re.compile(
    r"(?<![\w/.-])((?:bench|docs|rust|tests|\.github)/[\w./+-]*[\w]+"
    r"\.(?:json|jsonl|md|txt|log|csv|py|rs|toml|yml|yaml))")

#: Field names named beside a figure, in any of this codebase's three backtick spellings.
#: A *structured* path -- dotted, and/or carrying an array index, e.g.
#: `model.external_data.files[0].bytes` -- is looked up at that exact path only; see
#: `_values_for_field` for why it is no longer also reduced to its last segment. A *bare*,
#: unstructured citation (`claimed_nodes`) is answered by its exact top-level path, or by
#: its leaf key while every path bearing that key agrees on one value; a leaf whose paths
#: disagree is ambiguous and witnesses nothing.
_CITED_FIELD_RE = re.compile(r"`{1,2}([A-Za-z_][\w.]*(?:\[\d+\])?[\w.]*)`{1,2}")

#: An explicit, claim-local escape hatch: ``[witness: `model.external_data.files[0].bytes`]``
#: binds the figure it is written beside -- one annotation to one figure, paired like
#: brackets, never pooled over the clause. It exists so an author whose sentence is too
#: tangled for the adjacency rule below has a way to *say* which field backs the figure,
#: rather than a reason to widen the rule. It is not an exemption: an annotation naming a
#: field that does not carry the value is a conviction.
_WITNESS_ANNOTATION_RE = re.compile(
    r"\[witness:\s*`{1,2}([A-Za-z_][\w.]*(?:\[\d+\])?[\w.]*)`{1,2}\s*\]")

#: --- claim segmentation ---------------------------------------------------------------
#:
#: There is deliberately no proximity window here any more, and no paragraph-level pooling
#: either. A window pools every citation within N characters and hands the union to every
#: figure in it; a paragraph does the same thing with a different boundary. Either way a
#: *false* claim passes whenever some unrelated neighbour happens to carry the value: the
#: defect the fourth review round found, where ``model.bytes`` 2291238912 (wrong; that field
#: is 26,180,848) was blessed by a correct sibling citation of
#: ``model.external_data.files[0].bytes`` 2291238912 two sentences away. Widening or
#: narrowing the scope cannot fix that -- it only moves which false claims get lucky.
#:
#: Association is structural and claim-local instead. Prose is cut into paragraphs
#: (blank-line separated) and paragraphs into clauses; a figure binds to the field citation
#: *adjacent to it in its own clause* -- nearest before or after with no other figure in
#: between, the same matching discipline as brackets -- or to the ``[witness: `field`]``
#: annotation paired with it there. That is what "``claimed_nodes`` 355" and "2291238912
#: (``model.weights_bytes``)" already mean to a reader; the guard now reads them the same
#: way, and reads nothing else. Paragraphs survive only as the scope a *path* citation and a
#: withheld-figure disclaimer may be stated once in; neither of those lends a value.
_CLAUSE_BREAK_RE = re.compile(r"(?:[.;!?](?=[\s\"')\]]|$)|\s--\s|\s[\u2014\u2013]\s)")
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")
_BACKTICKED_SPAN_RE = re.compile(r"`{1,2}[^`]*`{1,2}")
_DECIMAL_RE = re.compile(r"\d[\d,]*\.\d+")
_ABBREVIATION_RE = re.compile(r"\b(?:e\.g|i\.e|cf|vs|etc|Fig|No|approx|Dr|Mr|St)\.", re.I)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _mask_for_segmentation(text: str) -> str:
    """A same-length copy of ``text`` with everything a period may legally live inside
    overwritten, so clause splitting cannot cut a decimal, a dotted field path, a
    repo-relative artifact path or an abbreviation in half. Offsets are preserved exactly:
    every replacement is one character for one character and whitespace is left alone, so a
    span found in the mask indexes the real text.
    """
    chars = list(text)
    for rx in (_BACKTICKED_SPAN_RE, _CITED_PATH_RE, _DECIMAL_RE, _ABBREVIATION_RE):
        for m in rx.finditer(text):
            for i in range(m.start(), m.end()):
                if not chars[i].isspace():
                    chars[i] = "X"
    return "".join(chars)


def _split_spans(masked: str, pattern, lo: int, hi: int) -> "list[tuple]":
    spans, start = [], lo
    for m in pattern.finditer(masked, lo, hi):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, hi))
    return [(a, b) for a, b in spans if b > a]


def _enclosing_span(spans: "list[tuple]", pos: int, default: tuple) -> tuple:
    for lo, hi in spans:
        if lo <= pos < hi:
            return lo, hi
    return default


def _module_prose(text: str) -> str:
    """Comments and docstrings only -- the parts of a module that make claims.

    Everything else is blanked, newlines included in the blanking so offsets and therefore
    line numbers survive: a guard that names the wrong line is a guard nobody acts on.

    Code is excluded so that constants (`ROUNDING_DEPTH_BOUND = 128`), provenance tables of
    digests and byte counts, and a synthetic fixture's `{"CPUExecutionProvider": 8}` are never
    read as measurements. Non-docstring string literals are excluded for the same reason: a
    string in an expression is data the module *handles*, not a sentence it asserts.
    """
    lines = text.splitlines(keepends=True)
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line)

    def _offset(row: int, col: int) -> int:
        return starts[row - 1] + col if row - 1 < len(starts) else len(text)

    spans = []
    readline = io.StringIO(text).readline
    for tok in tokenize.generate_tokens(readline):
        if tok.type == tokenize.COMMENT:
            spans.append((_offset(*tok.start), _offset(*tok.end)))
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            spans.append((_offset(first.lineno, first.col_offset),
                          _offset(first.end_lineno, first.end_col_offset)))

    out = ["\n" if ch == "\n" else " " for ch in text]
    for begin, end in spans:
        out[begin:end] = list(text[begin:end])
    blanked = "".join(out)
    return _NOT_A_MEASUREMENT_RE.sub(lambda m: " " * (m.end() - m.start()), blanked)


def _walk_numbers(node, name: "str | None", path: str, sink: dict) -> None:
    """Index every number in a document twice: at its exact path, and under its leaf key.

    The leaf index keeps the *set of paths* that carry the name, not a flattened pool of
    their values, because "which paths does this bare name mean" is the question a bare
    citation has to answer before any value of it can be believed.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_numbers(v, str(k), f"{path}.{k}" if path else str(k), sink)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_numbers(v, name, f"{path}[{i}]", sink)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        sink["by_path"].setdefault(path.lower(), set()).add(float(node))
        if name:
            leaf = sink["by_leaf"].setdefault(name.lower(), {})
            leaf.setdefault(path.lower(), set()).add(float(node))


def _artifact_fields(path: Path) -> dict:
    """A cited artifact's numbers, indexed by exact path and by leaf key.

    ``{"by_path": {dotted.path[i]: {values}}, "by_leaf": {key: {dotted.path[i]: {values}}}}``.
    """
    sink: dict = {"by_path": {}, "by_leaf": {}}
    try:
        raw = path.read_text("utf-8")
    except OSError:
        return sink
    if path.suffix == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                _walk_numbers(json.loads(line), None, "", sink)
            except json.JSONDecodeError:
                continue
    elif path.suffix == ".json":
        try:
            _walk_numbers(json.loads(raw), None, "", sink)
        except json.JSONDecodeError:
            return sink
    return sink


def _values_for_field(raw: str, fields: dict) -> "list[float]":
    """Values in ``fields`` for one cited field name, or nothing if the citation is ambiguous.

    A citation that names a *structured* path -- dotted and/or carrying an array index,
    e.g. ``model.external_data.files[0].bytes`` -- is looked up at that exact path only,
    never widened to its bare leaf key. Widening was the defect: ``model.bytes`` (26,180,848,
    the graph proto alone) and ``model.external_data.files[0].bytes`` (2,291,238,912, the
    externalized weights blob) share the leaf name ``bytes`` but name two different fields
    with two different values, and a citation of the former was being satisfied by the
    latter's value because both were folded into one ``bytes`` bucket.

    A *bare* citation (`` `claimed_nodes` ``) resolves to its exact top-level path when the
    document has one. Otherwise it is answered by the leaf key only while that key is
    **unambiguous as to value**: every path in the document carrying that name must agree on
    one value, as ``claimed_nodes`` (355 at six paths) and ``island_count`` (1 at six) do. A
    leaf whose paths disagree -- ``bytes`` is 1,757,184, 19,665,792 *and* 2,308,799,152 in
    ``bench/results/barrier_ab-post-dev0-0.json`` -- names nothing in particular, so it
    witnesses nothing and the author is made to write the exact path or an explicit witness
    annotation instead. Bare-leaf pooling is otherwise the same borrowing defect as paragraph
    pooling, one scope down.
    """
    key = raw.lower()
    by_path, by_leaf = fields.get("by_path", {}), fields.get("by_leaf", {})
    if "." in raw or "[" in raw:
        return sorted(by_path.get(key, ()))
    if key in by_path:
        return sorted(by_path[key])
    values: "set[float]" = set()
    for found in by_leaf.get(key, {}).values():
        values |= found
    return sorted(values) if len(values) == 1 else []


def _named_field_values(window: str, fields: dict) -> "list[float]":
    """Values in ``fields`` for every field cited in ``window``.

    Kept for the percentage arm below and for nothing else, because a share is a relation
    between two fields and no single field carries it. Every other rung binds a figure to the
    citation it names itself: see ``_bound_field_citations``.
    """
    values: "list[float]" = []
    for raw in _CITED_FIELD_RE.findall(window):
        values.extend(_values_for_field(raw, fields))
    return values


def _adjacent_binder(binders: "list[tuple]", span: tuple,
                     figures: "list[tuple]") -> "list[str]":
    """The binder a figure names itself: nearest before, else nearest after, no figure between.

    "``claimed_nodes`` 355" and "2291238912 (``model.bytes``)" are the two spellings this
    codebase uses, and exactly one field is answerable in each. The trailing binder is
    consulted only when nothing precedes, because the field that *follows* a figure is
    normally the next claim's field -- admitting it too is how a swapped pair passes, each
    figure finding the other's field.
    """
    start, end = span
    before = [b for b in binders if b[1] <= start]
    if before:
        b = before[-1]
        if not any(b[1] <= a and c <= start for a, c in figures):
            return [b[2]]
    after = [b for b in binders if b[0] >= end]
    if after:
        b = after[0]
        if not any(end <= a and c <= b[0] for a, c in figures):
            return [b[2]]
    return []


def _annotation_binder(annotations: "list[tuple]", span: tuple,
                       figures: "list[tuple]") -> "list[str]":
    """One annotation, one figure -- matched like brackets, never pooled.

    ``[witness: `field`]`` is written *for* a claim, so it binds exactly the figure it
    annotates and no other: the clause is walked once and each annotation pairs with the
    figure immediately beside it, taking the figure before it when that figure is still
    unpaired ("2291238912 bytes [witness: `f`]") and otherwise the figure after it
    ("[witness: `f`] 2291238912 bytes"). A figure two annotations away is left unbound rather
    than handed the union, which is what makes several annotations in one clause unambiguous
    instead of a pool.
    """
    tokens = sorted([(a, b, field) for a, b, field in annotations]
                    + [(a, b, None) for a, b in figures])
    paired: dict = {}
    available, previous = None, None
    for start, end, field in tokens:
        if field is not None:
            if previous is not None and previous[2] is None and previous[:2] not in paired:
                paired[previous[:2]] = field
                available = None
            else:
                available = field
        elif available is not None:
            paired[(start, end)] = available
            available = None
        previous = (start, end, field)
    return [paired[span]] if span in paired else []


def _bound_field_citations(scanned: str, clause: tuple, span: tuple,
                           figures: "list[tuple]") -> "list[str]":
    """The field citations this figure itself names -- nothing else in the file.

    Adjacency inside the figure's own clause, not distance and not the enclosing paragraph.
    2291238912 beside ``model.bytes`` is answerable to ``model.bytes`` alone, however many
    correct sibling citations share the paragraph or even the clause. An explicit
    ``[witness: `field`]`` annotation in the clause takes precedence for the figure it is
    paired with, for claims too tangled for adjacency to read; a figure no annotation is
    paired with still falls to the adjacency rule, over the citations that live *outside* the
    annotations, so an annotation written for one claim is never borrowed by another.
    """
    lo, hi = clause
    local = [(a, b) for a, b in figures if lo <= a < hi]
    annotations = [(m.start(), m.end(), m.group(1))
                   for m in _WITNESS_ANNOTATION_RE.finditer(scanned, lo, hi)]
    if annotations:
        annotated = _annotation_binder(annotations, span, local)
        if annotated:
            return annotated
    fields = [(m.start(), m.end(), m.group(1))
              for m in _CITED_FIELD_RE.finditer(scanned, lo, hi)
              if not any(a <= m.start() and m.end() <= b for a, b, _ in annotations)]
    return _adjacent_binder(fields, span, local)


def _matches_at_precision(candidate: float, printed: str) -> bool:
    """Does ``candidate`` round to the figure as printed, at the precision printed?

    "2.29 GB" is checked to two decimals and "2.3 GB" to one, so a witness of
    2,291,238,912 bytes backs both spellings and backs "2.2 GB" under neither.
    """
    digits = len(printed.split(".")[1]) if "." in printed else 0
    try:
        return round(candidate, digits) == round(float(printed), digits)
    except (ValueError, OverflowError):
        return False


def _pool_carries(pool: "list[float]", plain: str, unit_key: str) -> bool:
    for value in pool:
        if unit_key in _BYTE_SCALES and any(
                _matches_at_precision(value / scale, plain)
                for scale in _BYTE_SCALES[unit_key]):
            return True
        if _matches_at_precision(value, plain):
            return True
    return False


def _figure_is_witnessed(printed: str, unit: "str | None", claim: dict,
                         cited: "list[Path]") -> bool:
    """Is the printed figure actually *in* one of the cited files, at its own citation?

    For a structured artifact (`.json`/`.jsonl`) there is exactly one way to be witnessed: the
    figure must name its own field -- adjacent to it in its own clause, or paired with an
    explicit ``[witness: `field`]`` annotation in that clause -- and that field must carry the
    value, at the unit and rounding printed. A *different* field carrying it is not an answer,
    whichever clause or paragraph that field is cited in. This is the rung that makes a false
    ``model.bytes`` 2291238912 fail beside a correct
    ``model.external_data.files[0].bytes`` 2291238912, and it fails the same way when the
    correct sibling citation is a sentence later instead of a clause later: there is no wider
    scope to fall through to any more.

    What was removed, and why. There used to be two further rungs -- the clause's other
    citations, then the enclosing *paragraph's* -- so that a paragraph could state a figure in
    one sentence and its provenance in the next. That is a value pool: any figure in the
    paragraph could be answered by any field the paragraph mentioned, so a false claim passed
    whenever some neighbour happened to carry its digits. Prose that really does read a figure
    off a field named elsewhere now says so with an annotation, which is claim-local and
    checkable; the annotation can itself be wrong, and is then a conviction.

    The one deliberate exception is a percentage, and it is recorded as an exception: a share
    is a *relation* between two fields, so neither one carries it and adjacency cannot bind
    it. It is checked against ratios of the fields named in the figure's own clause -- never
    the paragraph -- and naming two fields is necessary, not sufficient, for a share to be
    true.

    A verbatim digit search over an artifact is not a witness: these files are tens of
    thousands of lines of unrelated numbers, and "88.9" occurs in the very artifact that was
    cited to back it while backing nothing of the kind. The verbatim search is kept only for
    unstructured witnesses (`.py`, `.rs`, `.md`, logs), where there is no field to name and a
    pinned literal in a test *is* the record.
    """
    plain = printed.replace(",", "")
    try:
        numeric = float(plain)
    except ValueError:
        return False
    unit_key = (unit or "").lower()
    for path in cited:
        if path.suffix in (".json", ".jsonl"):
            fields = _artifact_fields(path)
            bound: "list[float]" = []
            for raw in claim["bound"]:
                bound.extend(_values_for_field(raw, fields))
            if _pool_carries(bound, plain, unit_key):
                return True
            if unit_key == "%":
                ratio = _named_field_values(claim["clause"], fields)
                for a in ratio:
                    for b in ratio:
                        if b and _matches_at_precision(100.0 * a / b, plain):
                            return True
            continue
        try:
            text = path.read_text("utf-8")
        except OSError:
            continue
        if plain in text or printed in text or (numeric.is_integer()
                                                and str(int(numeric)) in text):
            return True
    return False


def _unwitnessed_measurement_shaped_figures(text: str, *, prose_only: bool = False) -> list:
    """Measurement-shaped figures in ``text`` with no usable witness.

    ``text`` is prose. Pass ``prose_only=False`` (the default) for a raw fragment, which is
    what the mutants below hand it; the module-level scan extracts prose first.
    """
    offenders = []
    scanned = text if prose_only else _NOT_A_MEASUREMENT_RE.sub(
        lambda m: " " * (m.end() - m.start()), text)
    masked = _mask_for_segmentation(scanned)
    whole = (0, len(scanned))
    paragraphs = _split_spans(masked, _PARAGRAPH_BREAK_RE, 0, len(masked))
    figures = []
    for m in _MEASUREMENT_SHAPED_RE.finditer(scanned):
        bare = m.group("unit") is None and not m.group("approx")
        if bare and "," not in m.group("value") and len(m.group("value").split(".")[0]) < 3:
            continue  # a one- or two-digit bare integer is not a measurement-shaped figure
        figures.append((m.start(), m.end()))
    for m in _MEASUREMENT_SHAPED_RE.finditer(scanned):
        printed, unit = m.group("value"), m.group("unit")
        if (m.start(), m.end()) not in figures:
            continue
        paragraph = _enclosing_span(paragraphs, m.start(), whole)
        clause = _enclosing_span(
            _split_spans(masked, _CLAUSE_BREAK_RE, *paragraph), m.start(), paragraph)
        paragraph_text = scanned[paragraph[0]:paragraph[1]]
        clause_text = scanned[clause[0]:clause[1]]
        #: The disclaimer is read from the figure's own clause too. Scoped to the paragraph
        #: it was the last implicit paragraph-level exemption for a measurement claim: one
        #: sentence saying a magnitude is withheld excused every *other* figure beside it,
        #: which is the pooling defect wearing a different hat. A clause that withholds a
        #: figure claims nothing; a clause that states one has to witness it.
        if any(d in clause_text.lower() for d in _WITHHELD_FIGURE_DISCLAIMERS):
            continue
        line_no = scanned.count("\n", 0, m.start()) + 1
        where = f"line {line_no}: {m.group().strip()!r}"
        #: A figure's *field* binding is claim-local and stays that way. The artifact path
        #: may be stated once for the paragraph, because that is how these blocks are
        #: written and because a path is not a value pool: whichever cited artifact answers,
        #: the figure's own field has to carry the figure there, so a neighbouring sentence's
        #: path cannot lend a neighbouring sentence's number.
        raw_paths = _CITED_PATH_RE.findall(clause_text) or _CITED_PATH_RE.findall(
            paragraph_text)
        if not raw_paths:
            offenders.append(f"{where} — no committed witness cited and no withheld-figure "
                             f"disclaimer")
            continue
        missing = [p for p in raw_paths if not (_REPO_ROOT / p).exists()]
        if missing:
            offenders.append(f"{where} — cites {missing}, which does not exist in this tree")
            continue
        claim = {"bound": _bound_field_citations(scanned, clause, (m.start(), m.end()),
                                                 figures),
                 "clause": clause_text}
        cited = [_REPO_ROOT / p for p in raw_paths]
        if not _figure_is_witnessed(printed, unit, claim, cited):
            named = claim["bound"] or "no field named beside it"
            offenders.append(f"{where} — cited {raw_paths} carries no such value at {named}")
    return offenders


def test_production_modules_do_not_quote_unwitnessed_measurement_shaped_figures():
    """A production module may not print a number this tree cannot produce.

    Every timing, percentage, ratio, count and unit-bearing size in these modules' prose must
    either sit beside a citation whose path exists *and* whose named field carries the value,
    or be marked withheld. "Nearby text that looks like a citation" is not enough — that was
    the hole the second review round found, and the mutants below are its falsifiers.
    """
    root = Path(__file__).resolve().parent
    offenders = {}
    for name in _NUMERIC_WITNESS_SCREENED_MODULES:
        prose = _module_prose((root / name).read_text("utf-8"))
        found = _unwitnessed_measurement_shaped_figures(prose, prose_only=True)
        if found:
            offenders[name] = found
    assert not offenders, (
        "unwitnessed measurement-shaped figures (no witness that carries the value):\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in offenders.items()))


# --- the guard's own falsifiers -------------------------------------------------------
#
# A guard with no must-fire mutant is decoration, and a guard with no must-not-fire mutant
# gets deleted the first time it convicts an honest line. Both directions are pinned.

_REAL_MODEL_WITNESS = "bench/results/real_model_gqa_local_size.json"


def test_a_size_figure_with_no_witness_is_convicted():
    """The `2.3 GB` shape: a unit-bearing size, correctly rounded, and backed by nothing.

    Correct rounding is not a witness. This figure happens to be right, and it is still a
    conviction, because nothing beside it says where it came from.
    """
    assert _unwitnessed_measurement_shaped_figures(
        "a 2.3 GB model left resident from a previous arm changes the allocator state")


def test_a_size_figure_whose_witness_disagrees_is_convicted():
    """The defect the previous guard blessed: a real citation, a real field, a wrong figure.

    The witness says 2,291,238,912 bytes. That backs "2.29 GB" and it backs "2.3 GB"; it does
    not back "2.2 GB", and the old proximity rule could not tell the difference because it
    never opened the artifact.
    """
    wrong = (f"the weights are 2.2 GB (``model.weights_bytes``, committed in "
             f"{_REAL_MODEL_WITNESS})")
    right = (f"the weights are 2.3 GB (``model.weights_bytes`` 2291238912, committed in "
             f"{_REAL_MODEL_WITNESS})")
    exact = (f"the weights are 2.29 GB (``model.weights_bytes``, committed in "
             f"{_REAL_MODEL_WITNESS})")
    assert _unwitnessed_measurement_shaped_figures(wrong)
    assert not _unwitnessed_measurement_shaped_figures(right)
    assert not _unwitnessed_measurement_shaped_figures(exact)


def test_a_dotted_citation_does_not_borrow_a_sibling_fields_same_leaf_key():
    """Issue #69's second review-round defect: two fields share a leaf name, one value wins.

    ``model.bytes`` (the graph proto alone) is 26,180,848 in the committed witness.
    ``model.external_data.files[0].bytes`` (the externalized weights blob beside it) is
    2,291,238,912 -- roughly 88x larger -- and the *only* thing the two fields share is the
    bare key ``bytes``. A citation of ``model.bytes`` claiming 2.29 GB must be convicted: that
    figure is not what ``model.bytes`` holds, however loudly a same-named sibling field
    elsewhere in the same artifact agrees with it. Must-fire mutant for the fix in
    ``_values_for_field``: before it, this exact wrong claim passed, because the dotted
    citation was widened to the bare leaf key ``bytes`` and matched against *every* field in
    the document named ``bytes``, not just the one actually cited.
    """
    wrong = (f"the model.onnx graph proto alone is 2291238912 bytes (``model.bytes``, "
             f"committed in {_REAL_MODEL_WITNESS})")
    right = (f"the model.onnx graph proto alone is 26180848 bytes (``model.bytes``, "
             f"committed in {_REAL_MODEL_WITNESS})")
    assert _unwitnessed_measurement_shaped_figures(wrong)
    assert not _unwitnessed_measurement_shaped_figures(right)


def test_an_array_indexed_citation_is_witnessed_at_its_own_exact_structured_path():
    """Valid exact-field control: the array-indexed sibling field, cited precisely.

    ``model.external_data.files[0].bytes`` is a different, exact path from ``model.bytes``
    even though both end in the leaf ``bytes``. Naming it in full -- including the array
    index selector -- witnesses its own value (2,291,238,912, i.e. 2.29 GB) without needing,
    and without permitting, any help from the unrelated ``model.bytes`` field.
    """
    good = (f"the externalized weights blob is 2.29 GB "
            f"(``model.external_data.files[0].bytes``, committed in {_REAL_MODEL_WITNESS})")
    assert not _unwitnessed_measurement_shaped_figures(good)


def test_a_false_claim_is_not_rescued_by_a_correct_sibling_claim_nearby():
    """Must-fire, mixed window: the fourth review round's blocker, in one paragraph.

    Both sentences quote 2,291,238,912. The second one is *right* -- that is the size of the
    externalized blob at ``model.external_data.files[0].bytes``. The first one is a lie:
    ``model.bytes`` is the graph proto, 26,180,848. Under a proximity window the two share a
    pool of citations and the lie inherits its neighbour's witness; under claim-local binding
    the first figure names ``model.bytes`` and is answerable only to ``model.bytes``.
    """
    mixed = (f"The graph proto alone is 2291238912 bytes (``model.bytes``, committed in "
             f"{_REAL_MODEL_WITNESS}).  The externalized weights blob is 2291238912 bytes "
             f"(``model.external_data.files[0].bytes``, committed in {_REAL_MODEL_WITNESS}).")
    offenders = _unwitnessed_measurement_shaped_figures(mixed)
    assert offenders and "model.bytes" in offenders[0]
    #: and not by collapsing the distance: the same two claims inside a single clause, where
    #: no window however small could separate them, are separated by adjacency alone.
    one_clause = (f"2291238912 (``model.bytes``) and 2291238912 "
                  f"(``model.external_data.files[0].bytes``), committed in "
                  f"{_REAL_MODEL_WITNESS}")
    assert _unwitnessed_measurement_shaped_figures(one_clause)


def test_two_true_values_attached_to_the_wrong_two_fields_are_convicted():
    """Must-fire, swapped values: every number real, every pairing false.

    26,180,848 and 2,291,238,912 are both genuinely in the witness, and the paragraph cites
    both fields that hold them -- so a pooled check finds every figure and passes the lot.
    The claim the prose actually makes is that ``model.bytes`` is the larger, and it is not.
    """
    swapped = (f"``model.bytes`` 2291238912 and ``model.weights_bytes`` 26180848, both "
               f"committed in {_REAL_MODEL_WITNESS}")
    assert len(_unwitnessed_measurement_shaped_figures(swapped)) == 2


def test_a_duplicate_leaf_key_does_not_let_one_path_answer_for_the_other():
    """Must-fire, duplicate leaf: two exact paths ending in the same leaf, one lying.

    ``model.bytes`` (26,180,848) and ``model.external_data.files[0].bytes`` (2,291,238,912)
    end in the same leaf ``bytes``. Here the *first* claim is true and the second attaches
    the proto's size to the blob's path. Exact-path lookup alone does not catch this -- the
    values still pool -- so it is the claim-local binding that convicts the second sentence
    and leaves the first alone.
    """
    dup = (f"The graph proto is 26180848 bytes (``model.bytes``).  The externalized blob is "
           f"26180848 bytes (``model.external_data.files[0].bytes``).  Both committed in "
           f"{_REAL_MODEL_WITNESS}.")
    offenders = _unwitnessed_measurement_shaped_figures(dup)
    assert len(offenders) == 1 and "files[0]" in offenders[0]


def test_two_exactly_cited_claims_may_share_one_paragraph():
    """Must-not-fire control: the honest version of all three mutants above.

    Same paragraph, same leaf key, two figures, both bound to the field that actually holds
    them -- in both orders, because "``field`` 123" and "123 (``field``)" are the two
    spellings this codebase uses. A guard that cannot pass this convicts the tree's own
    provenance prose and gets deleted.
    """
    field_first = (f"``model.bytes`` 26180848 is the graph proto and "
                   f"``model.external_data.files[0].bytes`` 2291238912 is the externalized "
                   f"blob, both committed in {_REAL_MODEL_WITNESS}")
    figure_first = (f"26180848 (``model.bytes``) is the proto; 2291238912 "
                    f"(``model.external_data.files[0].bytes``) is the blob.  Both committed "
                    f"in {_REAL_MODEL_WITNESS}.")
    assert not _unwitnessed_measurement_shaped_figures(field_first)
    assert not _unwitnessed_measurement_shaped_figures(figure_first)


def test_a_figure_may_not_take_its_provenance_from_the_next_sentence():
    """Must-fire, false cross-clause neighbour: the fifth review round's blocker.

    "The graph proto alone is 2291238912 bytes" is false -- ``model.bytes`` is 26,180,848 --
    and the sentence after it correctly cites the field that *does* hold 2,291,238,912. While
    the paragraph was a fallback scope, the false clause borrowed its neighbour's field and
    passed, which is the same borrowing defect as a proximity window with a different
    boundary. Nothing binds this figure in its own clause, so it is now a conviction, and the
    honest repair is to say which field it came from.
    """
    cross_clause = (f"The graph proto alone is 2291238912 bytes.  It was read off "
                    f"``model.external_data.files[0].bytes`` in {_REAL_MODEL_WITNESS}.")
    offenders = _unwitnessed_measurement_shaped_figures(cross_clause)
    assert len(offenders) == 1 and "no field named beside it" in offenders[0]
    #: The same shape with the figure told the truth about its own field is still convicted
    #: while it names no field: the fallback is gone for true and false claims alike.
    unbound_but_true = (f"The graph proto alone is 26180848 bytes.  It was read off "
                        f"``model.bytes`` in {_REAL_MODEL_WITNESS}.")
    assert _unwitnessed_measurement_shaped_figures(unbound_but_true)


def test_provenance_in_a_following_sentence_is_written_as_an_annotation():
    """Must-not-fire control: the sanctioned repair for the mutant above.

    An author whose provenance really does live in the next sentence says so with a
    claim-local annotation. It is checked exactly like any other binding, so the honest
    version passes and the version that annotates the wrong field does not.
    """
    honest = (f"The externalized blob is 2291238912 bytes [witness: "
              f"``model.external_data.files[0].bytes``].  It was read off that field in "
              f"{_REAL_MODEL_WITNESS}.")
    lying = (f"The graph proto alone is 2291238912 bytes [witness: ``model.bytes``].  It was "
             f"read off that field in {_REAL_MODEL_WITNESS}.")
    assert not _unwitnessed_measurement_shaped_figures(honest)
    assert _unwitnessed_measurement_shaped_figures(lying)


def test_two_annotations_in_one_clause_bind_one_figure_each():
    """Must-fire and must-not-fire together: annotations pair, they do not pool.

    Both fields named here are real and both values are real. If the two annotations were
    pooled over the clause, the swap would pass -- each figure would find the other's field.
    Paired one-to-one, the swapped clause convicts twice and the honest one not at all.
    """
    honest = (f"the proto is 26180848 bytes [witness: ``model.bytes``] and the blob is "
              f"2291238912 bytes [witness: ``model.external_data.files[0].bytes``], both "
              f"committed in {_REAL_MODEL_WITNESS}")
    swapped = (f"the proto is 2291238912 bytes [witness: ``model.bytes``] and the blob is "
               f"26180848 bytes [witness: ``model.external_data.files[0].bytes``], both "
               f"committed in {_REAL_MODEL_WITNESS}")
    assert not _unwitnessed_measurement_shaped_figures(honest)
    assert len(_unwitnessed_measurement_shaped_figures(swapped)) == 2


def test_an_ambiguous_bare_leaf_citation_witnesses_nothing():
    """Must-fire: a bare key that names three different fields names none of them.

    ``bytes`` occurs at three paths in the barrier artifact -- the built library's 1,757,184,
    a readback's 19,665,792 and an upload's 2,308,799,152 -- and a bare `` `bytes` ``
    citation cannot say which was meant. Answering it from the pooled leaf is bare-leaf
    pooling, the same borrowing defect one scope down, so it is refused and the author is
    made to name the exact path. A leaf that *does* agree everywhere it appears --
    ``claimed_nodes``, 355 at all six of its paths -- still answers, and is the control here.
    """
    witness = "bench/results/barrier_ab-post-dev0-0.json"
    ambiguous = f"the readback moved 19665792 bytes (``bytes``), committed in {witness}"
    exact = (f"the readback moved 19665792 bytes "
             f"(``results[0].phase_pass.analysis.transfers.readback.bytes``), committed in "
             f"{witness}")
    agreeing_leaf = f"``claimed_nodes`` 355 nodes were claimed, committed in {witness}"
    assert _unwitnessed_measurement_shaped_figures(ambiguous)
    assert not _unwitnessed_measurement_shaped_figures(exact)
    assert not _unwitnessed_measurement_shaped_figures(agreeing_leaf)


def test_an_explicit_witness_annotation_binds_and_can_itself_be_wrong():
    """The escape hatch is an annotation, not an exemption.

    ``[witness: `field`]`` lets an author bind a figure in a clause too tangled for
    adjacency. It binds the same way everything else does: naming a field that does not carry
    the value is a conviction, not a pass.
    """
    truthful = (f"the blob is 2291238912 bytes [witness: ``model.external_data.files[0]"
                f".bytes``], committed in {_REAL_MODEL_WITNESS}")
    lying = (f"the proto is 2291238912 bytes [witness: ``model.bytes``], committed in "
             f"{_REAL_MODEL_WITNESS}")
    assert not _unwitnessed_measurement_shaped_figures(truthful)
    assert _unwitnessed_measurement_shaped_figures(lying)


def test_clause_segmentation_does_not_cut_decimals_paths_or_dotted_fields():
    """The segmentation is only sound if a period inside a figure is not a clause end.

    Every one of these would be split mid-token by a naive `.`-splitter, orphaning the figure
    from its citation and convicting an honest line. Offsets in the mask are checked to line
    up character-for-character with the real text, because the binding indexes through it.
    """
    text = (f"the weights are 2.29 GB (``model.external_data.files[0].bytes``, e.g. as "
            f"committed in {_REAL_MODEL_WITNESS})")
    masked = _mask_for_segmentation(text)
    assert len(masked) == len(text)
    assert all(a == b for a, b in zip(masked, text) if a.isspace() or b.isspace())
    assert not _CLAUSE_BREAK_RE.search(masked)
    assert not _unwitnessed_measurement_shaped_figures(text)


def test_a_citation_to_a_path_that_does_not_exist_is_convicted():
    """A withdrawn artifact is not a witness, however precisely it is named.

    `bench/results/_cuda69/profile_prefill_1.json` is exactly the path this branch cited
    while deleting: the de-claim scan's original finding.
    """
    offenders = _unwitnessed_measurement_shaped_figures(
        "the warm call spent 355 nodes' worth of time, committed in "
        "bench/results/_cuda69/profile_prefill_1.json")
    assert offenders and "does not exist" in offenders[0]


def test_a_timing_a_percentage_and_a_count_are_all_convicted_uncited():
    """The shape half: unit-separated, compact, percentage, ratio, count.

    Each of these was legible to a reader as a measurement and illegible to the previous
    pattern, which only knew decimal ms, decimal percent, bare integer ratios and
    comma-grouped counts.
    """
    for fragment in (
        "the warm median was 45 ms",
        "the warm median was 45ms",
        "the warm median was 45-ms",
        "the arm was 12% slower",
        "the arm was 12x slower",
        "the run recorded 18,060 ULP of error",
        "the run recorded 2291238912 bytes of weights",
        "roughly ~3200 dispatches per inference",
    ):
        assert _unwitnessed_measurement_shaped_figures(fragment), fragment


def test_a_ratio_of_two_named_fields_is_witnessed_by_the_artifact():
    """97.8% is not a literal in the artifact; it is `claimed_nodes` over `nodes_probed`.

    A guard that demanded the printed digits appear verbatim would force every derived share
    to be deleted, so shares are checked against the ratio of the fields actually named. A
    share whose fields are *not* named, or which no pair of them produces, still convicts.

    This arm is deliberately weaker than the rule the reviewers settled on, and cannot
    replace it. Naming the two fields a share divides is necessary, not sufficient: the
    artifact must also identify the *quantity* being claimed. The 88.9% below is the worked
    example -- `claimed_nodes` and `total_nodes_probed` are real fields, but neither is a
    per-provider profile-node count, so no ratio of them witnesses a 1-Vulkan/8-CPU split.
    It convicts here because no pair of fields produces the figure; had one coincidentally
    produced it, the claim would still have to go, and that judgement is a reader's, not
    this screen's.
    """
    good = ("`claimed_nodes` 355 of `total_nodes_probed` 363, so 97.8% of the probed graph, "
            "committed in bench/results/barrier_ab-post-dev0-0.json")
    bad = ("`claimed_nodes` and `total_nodes_probed` put 88.9% of the graph on the host, "
           "committed in bench/results/barrier_ab-post-dev0-0.json")
    assert not _unwitnessed_measurement_shaped_figures(good)
    assert _unwitnessed_measurement_shaped_figures(bad)


def test_methodology_code_constants_and_synthetic_fixtures_do_not_fire():
    """The three must-not-fire classes, each for a different reason.

    Methodology says the figure is withheld and so claims nothing. Code constants are not
    prose and never reach the scan. A fixture's numbers are inputs the test constructs, and
    saying so out loud is what makes them not a claim about a run.
    """
    methodology = ("the term collapses to a small fraction of a millisecond; no magnitude is "
                   "quoted here, since none is witnessed by a committed artifact in this tree")
    fixture = ("the synthetic fixture below is built with 4000 us of CPU kernel time beside "
               "10000 us of CUDA, so the gate has something to convict")
    assert not _unwitnessed_measurement_shaped_figures(methodology)
    assert not _unwitnessed_measurement_shaped_figures(fixture)

    code_only = (
        "ROUNDING_DEPTH_BOUND = 128\n"
        "MANTISSA_BITS = {'float32': 23, 'tf32': 10, 'float16': 10}\n"
        "PINNED = {'sha256': 'c0c3f7', 'bytes': 2291238912}\n"
        "PARTITION = {'VulkanExecutionProvider': 1, 'CPUExecutionProvider': 8}\n"
    )
    assert _module_prose(code_only).strip() == ""
    assert not _unwitnessed_measurement_shaped_figures(_module_prose(code_only),
                                                       prose_only=True)


def test_a_disclaimer_excuses_its_own_clause_and_not_its_neighbours():
    """Must-fire: the last paragraph-scoped exemption, closed.

    A withheld-figure disclaimer used to excuse every figure in its paragraph, so one
    sentence saying a magnitude is withheld blessed the unwitnessed figures beside it -- the
    pooling defect in another costume. The disclaimer now excuses the clause it stands in,
    the way a citation binds the clause it stands in. The withheld clause still claims
    nothing and still passes; the quoted figure in the next sentence has to stand on its own.
    """
    mixed = (f"No magnitude is quoted here for the outer term.  The warm median was 45 ms, "
             f"committed in {_REAL_MODEL_WITNESS}.")
    offenders = _unwitnessed_measurement_shaped_figures(mixed)
    assert len(offenders) == 1 and "45 ms" in offenders[0]


def test_the_prose_extractor_keeps_offsets_so_line_numbers_are_real():
    """A blanked-out module must still report the line a figure is actually on.

    Deleting non-prose would shift every line number after the first function body, and a
    guard that names the wrong line is a guard nobody acts on.
    """
    module = (
        "X = 1\n"
        "def f():\n"
        "    return 2\n"
        "# the warm median was 45 ms\n"
    )
    offenders = _unwitnessed_measurement_shaped_figures(_module_prose(module), prose_only=True)
    assert offenders and offenders[0].startswith("line 4:")


def test_no_committed_record_calls_itself_admissible_while_refusing():
    """`ADMISSIBLE` beside a non-empty `refusals` list is a record disagreeing with itself.

    Secondary to the structural gate in `cuda_profile.attribute`; this one catches an
    artifact produced before that gate existed that is still sitting in the tree.
    """
    offenders = []
    for f in _committed_evidence_json():
        doc = json.loads(f.read_text("utf-8"))
        for jpath, rec in _measurement_records(doc):
            refusals = rec.get("refusals") or []
            if refusals and rec.get("verdict") in cc.GREEN_VERDICTS:
                offenders.append(
                    f"{f.name}{jpath}: {rec['verdict']} with {len(refusals)} refusal(s)")
    assert not offenders, "\n  ".join(["records that refuse and pass at once:"] + offenders)


# ---------------------------------------------------------------------------
# GATE 4 OUTPUT EQUIVALENCE: the internal channel and the public artifact are
# two different things, and the check fails closed
#
# The defect these are the falsifiers for shipped as a passing suite.  The worker
# put an absolute `out0.npy` into its record; `dump_public_json` correctly refused
# to publish an absolute path and rewrote it `<repo>/...`; the parent then asked
# `Path("<repo>/...").is_file()`, got False, said "reference output file missing",
# and returned `{"arms": {}}`.  `compare_workload` asked only which arms were
# disqualified, got an empty set, and published VULKAN_FASTER for an arm whose
# logits were a thousand times the reference's.  Every gate in the harness was
# working; the wiring between two of them was the entire failure.
#
# So both halves are tested here.  The separation (a relative handle plus a
# runtime-only channel, resolved with containment) and the closure (nothing that
# could not be compared is allowed to read as something that was).  Each control
# plants exactly one defect and asserts a refusal, and the last two are the
# must-pass polarity: a correct arm still compares, and a correct arm is still
# allowed to win.
# ---------------------------------------------------------------------------

import public_paths as pp  # noqa: E402

#: A Vulkan arm that is nine times faster than CUDA on every sample. Every control
#: below uses it, so a control that stops working stops disqualifying a *winner* --
#: which is the only way this class of defect ever reaches a reader.
_FAST_VULKAN = [1.0, 1.0, 1.0, 1.0, 1.0]
_SLOW_CUDA = [9.0, 9.0, 9.0, 9.0, 9.0]

#: The reference logits and a thousandfold-wrong copy of them.
_TRUE_LOGITS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_WRONG_BY_1000 = [v * 1000.0 for v in _TRUE_LOGITS]


def _worker_record(scratch: Path, arm: str, logits, steady) -> dict:
    """One arm's record, written and read back **through the real boundary**.

    Not a hand-built dict: the defect lived in what `dump_public_json` does to a
    record on its way to disk, so a control that skips the serialisation step tests
    the version of the code that never had the bug.
    """
    dump = scratch / f"outputs_{arm}_wl"
    dump.mkdir(parents=True, exist_ok=True)
    manifest = cc._npy_dump([np.asarray(logits, dtype=np.float32)], ["logits"], dump)
    payload = {"arm": arm, "workload": "prefill_1", "model_key": "m",
               "verdict": cc.ADMISSIBLE, "refusals": [], "instrument_errors": [],
               "steady_ms": list(steady), "compute_regime": "float32",
               "outputs_manifest": manifest}
    out = scratch / f"result_{arm}_wl.json"
    pp.dump_public_json(payload, out)
    rec = json.loads(out.read_text(encoding="utf-8"))
    # Exactly what `dispatch_arm` attaches after reading the worker's record.
    rec["outputs_dir"] = pp.public_path(dump)
    rec["scratch_dir"] = pp.public_path(scratch)
    rec[pp.RUNTIME_ONLY_KEY] = {"outputs_dir": str(dump.resolve()),
                                "scratch_dir": str(scratch.resolve())}
    return rec


def _three_arms(scratch: Path, vulkan_logits) -> "list[dict]":
    return [
        _worker_record(scratch, cc.ARM_CPU_HOST, _TRUE_LOGITS, [5.0]),
        _worker_record(scratch, cc.ARM_VULKAN, vulkan_logits, _FAST_VULKAN),
        _worker_record(scratch, cc.ARM_CUDA, _TRUE_LOGITS, _SLOW_CUDA),
    ]


def _verdict_for(records, *, equivalence=None) -> dict:
    eq = cc.cross_arm_equivalence(records) if equivalence is None else equivalence
    return {"eq": eq, "cmp": cc.compare_workload(_workload(), records, equivalence=eq)}


def test_a_thousandfold_wrong_vulkan_output_cannot_be_published_as_faster(tmp_path):
    """The end-to-end control for the shipped defect, planted at the tensor.

    Vulkan is nine times faster on every sample and returns logits 1000x the CPU
    reference's.  Before the fix this printed VULKAN_FASTER.
    """
    records = _three_arms(tmp_path, _WRONG_BY_1000)
    got = _verdict_for(records)

    assert got["eq"]["arms"][cc.ARM_VULKAN]["verdict"] == "DIVERGENT"
    assert got["cmp"]["verdict"] == cc.NOT_EQUIVALENT
    assert got["cmp"]["verdict"] != "VULKAN_FASTER"
    assert cc.ARM_VULKAN in got["cmp"]["equivalence_disqualified_arms"]
    assert "speedup_vulkan_over_cuda" not in got["cmp"]


def test_a_correct_arm_still_compares_and_is_still_allowed_to_win(tmp_path):
    """The must-pass polarity. A screen that refuses everything protects nothing."""
    records = _three_arms(tmp_path, _TRUE_LOGITS)
    got = _verdict_for(records)

    assert got["eq"]["verdict"] == cc.EQ_COMPARED, got["eq"].get("detail")
    assert got["eq"]["arms"][cc.ARM_VULKAN]["verdict"] == "MATCH"
    assert got["eq"]["compared_arms"] == sorted([cc.ARM_VULKAN, cc.ARM_CUDA])
    assert got["cmp"]["equivalence_disqualified_arms"] == []
    assert got["cmp"]["equivalence_unchecked_arms"] == []
    assert got["cmp"]["verdict"] == "VULKAN_FASTER"


def test_a_serialised_path_in_the_manifest_is_refused_and_not_read_as_missing(tmp_path):
    """The defect itself, planted: a manifest that names a path instead of a handle.

    This is the shape the old worker wrote and the sanitiser rooted. It must be a
    refusal that disqualifies, never "the file is missing" folded into an empty map.
    """
    records = _three_arms(tmp_path, _TRUE_LOGITS)
    ref = next(r for r in records if r["arm"] == cc.ARM_CPU_HOST)
    entry = ref["outputs_manifest"][0]
    entry.pop("file_rel")
    entry["file"] = "<repo>/bench/results/_cuda69/scratch/outputs_cpu_host_wl/out0.npy"

    got = _verdict_for(records)

    assert got["eq"]["verdict"] == cc.EQ_REFUSED
    assert "refused rather than read" in got["eq"]["detail"]
    assert set(got["eq"]["arms"]) == {cc.ARM_VULKAN, cc.ARM_CUDA}
    assert got["cmp"]["verdict"] == cc.NOT_EQUIVALENT
    assert got["cmp"]["verdict"] != "VULKAN_FASTER"


def test_a_tokenised_outputs_dir_this_machine_cannot_resolve_refuses(tmp_path):
    """`<elsewhere>` is a place that is, by construction, not here."""
    records = _three_arms(tmp_path, _TRUE_LOGITS)
    for rec in records:
        rec.pop(pp.RUNTIME_ONLY_KEY)
        rec["outputs_dir"] = "<elsewhere>/outputs_cpu_host_wl"

    got = _verdict_for(records)

    assert got["eq"]["verdict"] == cc.EQ_REFUSED
    assert "cannot resolve" in got["eq"]["detail"]
    assert got["cmp"]["verdict"] == cc.NOT_EQUIVALENT


def test_a_missing_cpu_reference_disqualifies_every_other_arm_by_name(tmp_path):
    """No reference is not "nothing to compare against"; it is "nothing is comparable"."""
    records = [r for r in _three_arms(tmp_path, _TRUE_LOGITS)
               if r["arm"] != cc.ARM_CPU_HOST]

    got = _verdict_for(records)

    assert got["eq"]["verdict"] == cc.EQ_REFUSED
    assert set(got["eq"]["arms"]) == {cc.ARM_VULKAN, cc.ARM_CUDA}
    assert all(v["verdict"] == cc.UNMEASURED for v in got["eq"]["arms"].values())
    assert got["cmp"]["verdict"] == cc.NOT_EQUIVALENT
    assert got["cmp"]["verdict"] != "VULKAN_FASTER"


def test_a_reference_whose_tensor_is_absent_refuses(tmp_path):
    """The handle is well-formed and contained; the file it names is simply not there."""
    records = _three_arms(tmp_path, _TRUE_LOGITS)
    (tmp_path / f"outputs_{cc.ARM_CPU_HOST}_wl" / cc.OUTPUT_HANDLE).unlink()

    got = _verdict_for(records)

    assert got["eq"]["verdict"] == cc.EQ_REFUSED
    assert "nothing to compare" in got["eq"]["detail"]
    assert got["cmp"]["verdict"] == cc.NOT_EQUIVALENT


def test_a_comparison_that_examined_zero_tensors_is_an_instrument_error(tmp_path, monkeypatch):
    """A comparison with no outputs looked at nothing, whatever verdict it carries.

    Planted at the comparator, because that is the only place the state is reachable
    from: `compare_outputs` returns an empty `outputs` list for an arity mismatch and
    for an absent side, and neither may reach the reader as a pass.
    """
    records = _three_arms(tmp_path, _TRUE_LOGITS)
    monkeypatch.setattr(cc, "compare_outputs",
                        lambda *a, **k: {"verdict": "MATCH", "outputs": []})

    got = _verdict_for(records)

    assert got["eq"]["arms"][cc.ARM_VULKAN]["verdict"] == cc.INSTRUMENT_ERROR
    assert "zero tensors" in got["eq"]["arms"][cc.ARM_VULKAN]["detail"]
    assert got["eq"]["compared_arms"] == []
    assert got["cmp"]["verdict"] == cc.NOT_EQUIVALENT


@pytest.mark.parametrize("handle", [
    "../outputs_cpu_host_wl/out0.npy",     # traversal to a sibling arm's tensor
    "sub/../../out0.npy",                  # traversal spelled through a subdirectory
    "out0.npy/../../../out0.npy",          # traversal after a legitimate leading name
])
def test_a_handle_that_escapes_its_own_output_directory_is_refused(tmp_path, handle):
    """Containment, planted. A handle out of the arm's own slot is another arm's answer."""
    records = _three_arms(tmp_path, _WRONG_BY_1000)
    subject = next(r for r in records if r["arm"] == cc.ARM_VULKAN)
    subject["outputs_manifest"][0]["file_rel"] = handle

    got = _verdict_for(records)

    assert got["eq"]["arms"][cc.ARM_VULKAN]["verdict"] == cc.UNMEASURED
    assert got["cmp"]["verdict"] == cc.NOT_EQUIVALENT
    assert got["cmp"]["verdict"] != "VULKAN_FASTER"


def test_an_absolute_handle_outside_the_output_directory_is_refused(tmp_path):
    """Not traversal but the same class: a handle that is not a handle at all."""
    outside = tmp_path / "outside"
    outside.mkdir()
    np.save(outside / cc.OUTPUT_HANDLE, np.asarray(_TRUE_LOGITS, dtype=np.float32))
    records = _three_arms(tmp_path, _WRONG_BY_1000)
    subject = next(r for r in records if r["arm"] == cc.ARM_VULKAN)
    subject["outputs_manifest"][0]["file_rel"] = str(outside / cc.OUTPUT_HANDLE)

    got = _verdict_for(records)

    assert got["eq"]["arms"][cc.ARM_VULKAN]["verdict"] == cc.UNMEASURED
    assert got["cmp"]["verdict"] == cc.NOT_EQUIVALENT


def test_an_arm_with_timings_and_no_equivalence_verdict_is_refused_by_name(tmp_path):
    """The fail-open itself: an empty `arms` map used to mean "nobody was disqualified"."""
    records = _three_arms(tmp_path, _WRONG_BY_1000)

    entry = cc.compare_workload(_workload(), records, equivalence={"arms": {}})

    assert entry["equivalence_unchecked_arms"] == sorted([cc.ARM_CUDA, cc.ARM_VULKAN])
    assert entry["verdict"] == cc.INSTRUMENT_ERROR
    assert entry["verdict"] != "VULKAN_FASTER"
    assert "speedup_vulkan_over_cuda" not in entry


def test_no_equivalence_argument_at_all_is_refused_rather_than_assumed(tmp_path):
    """The same hole reached by the other door: the caller that forgot to pass one."""
    records = _three_arms(tmp_path, _WRONG_BY_1000)

    entry = cc.compare_workload(_workload(), records)

    assert entry["verdict"] == cc.INSTRUMENT_ERROR
    assert "no equivalence result was supplied" in entry["detail"]


def test_a_committed_record_carries_no_absolute_path_and_no_runtime_channel(tmp_path):
    """The other obligation: closing the read path must not reopen the leak."""
    records = _three_arms(tmp_path, _TRUE_LOGITS)
    suite = {"schema": "cuda_competition/1", "results": records,
             "equivalence": {"prefill_1": cc.cross_arm_equivalence(records)}}
    out = tmp_path / "suite.json"

    pp.dump_public_json(suite, out)
    text = out.read_text(encoding="utf-8")
    published = json.loads(text)

    assert not pp.scan(text), pp.scan(text)[:4]
    assert pp.RUNTIME_ONLY_KEY not in text
    for rec in published["results"]:
        assert pp.RUNTIME_ONLY_KEY not in rec
        assert not Path(rec["outputs_dir"]).is_absolute()
        assert rec["outputs_manifest"][0]["file_rel"] == cc.OUTPUT_HANDLE


def test_the_runtime_channel_cannot_be_published_by_hand(tmp_path):
    """`sanitise=False` asserts the payload is already public. This one is not."""
    payload = {"arm": cc.ARM_VULKAN,
               pp.RUNTIME_ONLY_KEY: {"outputs_dir": "/srv/run/outputs_vulkan_wl"}}

    with pytest.raises(pp.PathLeak) as exc:
        pp.dump_public_json(payload, tmp_path / "leaky.json", sanitise=False)

    # Structural, not a leak scan: `/srv/run/...` names no machine and no account, and
    # is still an in-process handle with no meaning to a reader.
    assert "runtime_only_key" in str(exc.value)
    assert not (tmp_path / "leaky.json").exists()


def test_equivalence_re_derives_from_a_committed_record_without_the_runtime_channel(
        tmp_path, monkeypatch):
    """`--reanalyse` on a second machine: rooted paths must resolve back to real files.

    This is why the public field is a *rooted* path rather than a deleted one. The
    record that reaches a reviewer has no absolute path and no runtime channel, and it
    still has to be enough to re-run the comparison against tensors under their own
    checkout.
    """
    monkeypatch.setattr(pp, "REPO", tmp_path)
    records = _three_arms(tmp_path, _WRONG_BY_1000)
    out = tmp_path / "suite.json"
    pp.dump_public_json({"results": records}, out)
    committed = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert all(pp.RUNTIME_ONLY_KEY not in r for r in committed)
    assert committed[0]["outputs_dir"].startswith("<repo>/")

    got = _verdict_for(committed)

    assert got["eq"]["verdict"] == cc.EQ_REFUSED
    assert got["eq"]["arms"][cc.ARM_VULKAN]["verdict"] == "DIVERGENT"
    assert got["eq"]["arms"][cc.ARM_CUDA]["verdict"] == "MATCH"
    assert got["cmp"]["verdict"] == cc.NOT_EQUIVALENT


def test_a_rooted_path_resolves_only_to_somewhere_it_could_have_come_from(tmp_path,
                                                                          monkeypatch):
    """The resolver's own polarities, without a suite around them."""
    monkeypatch.setattr(pp, "REPO", tmp_path)
    (tmp_path / "bench").mkdir()

    assert pp.resolve_public_path("<repo>/bench") == (tmp_path / "bench").resolve()
    assert pp.resolve_public_path("<elsewhere>/out0.npy") is None
    assert pp.resolve_public_path("<foreign-repo>/bench") is None
    assert pp.resolve_public_path("<no-such-root>/bench") is None
    assert pp.resolve_public_path(str(tmp_path / "bench")) is None
    assert pp.resolve_public_path("<repo>/../escaped") is None


def test_containment_is_decided_after_resolution_not_on_the_text(tmp_path):
    """A name that passes every string test and still leaves the directory."""
    base = tmp_path / "slot"
    base.mkdir()
    (base / cc.OUTPUT_HANDLE).write_bytes(b"0")

    assert pp.contained_child(base, cc.OUTPUT_HANDLE) == (base / cc.OUTPUT_HANDLE).resolve()
    assert pp.contained_child(base, "sub/out0.npy") == (base / "sub" / "out0.npy").resolve()
    assert pp.contained_child(base, "../out0.npy") is None
    assert pp.contained_child(base, "/etc/passwd") is None
    assert pp.contained_child(base, "C:out0.npy") is None
    assert pp.contained_child(base, "<repo>/out0.npy") is None
    assert pp.contained_child(base, "") is None


def test_the_profile_and_trace_handles_use_the_same_contract(tmp_path):
    """One contract for every file that crosses a process boundary in this suite.

    `cuda_profile` read `rec["profile_path"]` — the sanitiser's rooted output — and got
    an empty op comparison, which reads as "these providers share no op types" rather
    than "this file was never opened". Same fix, same falsifiers.
    """
    import cuda_profile as cp

    scratch = tmp_path / "traced"
    scratch.mkdir()
    (scratch / "profile_x.json").write_text("[]", encoding="utf-8")
    (scratch / "trace_prefill_1.json").write_text("[]", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("[]", encoding="utf-8")

    rec = {"arm": cc.ARM_VULKAN,
           "scratch_dir": pp.public_path(scratch),
           "profile_rel": "profile_x.json",
           "trace_rel": "trace_prefill_1.json",
           pp.RUNTIME_ONLY_KEY: {"scratch_dir": str(scratch.resolve())}}

    got, why = cc.resolve_scratch_file(rec, rec["profile_rel"])
    assert got == (scratch / "profile_x.json").resolve() and why == ""
    assert cp.resolve_trace(rec)[0] == (scratch / "trace_prefill_1.json").resolve()

    assert cc.resolve_scratch_file(rec, "../outside.json")[0] is None
    assert cc.resolve_scratch_file(rec, str(outside))[0] is None
    assert cc.resolve_scratch_file(rec, None)[0] is None
    assert cc.resolve_scratch_file({"arm": cc.ARM_VULKAN}, "profile_x.json")[0] is None


def test_a_trace_that_cannot_be_resolved_is_refused_with_its_reason(tmp_path):
    """`attribute` must say "unresolvable handle", not "the EP wrote no trace"."""
    import cuda_profile as cp

    traced = {"arm": cc.ARM_VULKAN, "workload": "prefill_1", "verdict": cc.ADMISSIBLE,
              "trace_rel": "trace_prefill_1.json", "outputs_dir": "<elsewhere>/x",
              "scratch_dir": "<elsewhere>/traced"}
    resolved, why = cp.resolve_trace(traced)

    assert resolved is None and why
    out = cp.attribute(traced, None, resolved, unresolved=why)

    assert out["verdict"] == cp.TRACE_ABSENT
    assert any("cannot resolve" in r for r in out["refusals"]), out["refusals"]


def test_the_worker_writes_a_handle_and_never_a_path(tmp_path):
    """The writer's half, asserted where it is written rather than inferred downstream."""
    manifest = cc._npy_dump(
        [np.asarray(_TRUE_LOGITS, dtype=np.float32), np.asarray([1.0, 2.0])],
        ["logits", "present_key_values"], tmp_path / "slot")

    assert manifest[0]["file_rel"] == cc.OUTPUT_HANDLE
    assert not any(k in manifest[0] for k in cc._LEGACY_ABS_HANDLE)
    assert (tmp_path / "slot" / cc.OUTPUT_HANDLE).is_file()
    assert "sha256" in manifest[1] and "file_rel" not in manifest[1]
    assert not pp.scan(json.dumps(manifest))


# ---------------------------------------------------------------------------
# The four handle instruments are TOTAL: they return `(value, why)` rather than
# raising, because every caller needs the `why` on the refusal path — it is what
# `cross_arm_equivalence` puts in the arm's verdict and what `attribute` puts in
# its refusals.  So their reject polarity is asserted through `bench/_polarity.py`,
# which fails the test at run time when the thing inside it did not refuse, exactly
# as `pytest.raises` does for a guard that raises.  See the header of that module.
# ---------------------------------------------------------------------------

import _polarity  # noqa: E402


def _resolvable_record(tmp_path: Path) -> "tuple[dict, dict]":
    rec = _worker_record(tmp_path, cc.ARM_VULKAN, _TRUE_LOGITS, _FAST_VULKAN)
    return rec, rec["outputs_manifest"][0]


def test_resolve_arm_output_refuses_a_handle_it_cannot_stand_behind(tmp_path):
    """Reject polarity, five ways, each an assertion rather than an annotation."""
    rec, entry = _resolvable_record(tmp_path)
    dump = tmp_path / f"outputs_{cc.ARM_VULKAN}_wl"

    _polarity.refuses(cc.resolve_arm_output(rec, {"name": "logits"}),
                      because="no handle at all")
    _polarity.refuses(cc.resolve_arm_output(rec, {"file": str(dump / cc.OUTPUT_HANDLE)}),
                      because="a path where a handle belongs")
    _polarity.refuses(cc.resolve_arm_output(rec, {"file_rel": "../out0.npy"}),
                      because="a handle that leaves the arm's own slot")
    _polarity.refuses(cc.resolve_arm_output(rec, {"file_rel": "somethingelse.npy"}),
                      because="a name this module never writes")
    _polarity.refuses(cc.resolve_arm_output({"arm": cc.ARM_VULKAN}, entry),
                      because="a record with no outputs_dir to root the handle at")


def test_resolve_arm_output_accepts_the_handle_its_own_writer_produced(tmp_path):
    """Accept polarity. An instrument that refuses everything screens nothing."""
    rec, entry = _resolvable_record(tmp_path)

    got, why = cc.resolve_arm_output(rec, entry)

    assert got == (tmp_path / f"outputs_{cc.ARM_VULKAN}_wl" / cc.OUTPUT_HANDLE).resolve()
    assert got.is_file() and why == ""


def test_resolve_scratch_file_refuses_what_it_cannot_root_or_contain(tmp_path):
    rec, _ = _resolvable_record(tmp_path)
    (tmp_path / "profile_x.json").write_text("[]", encoding="utf-8")

    _polarity.refuses(cc.resolve_scratch_file(rec, None), because="no handle")
    _polarity.refuses(cc.resolve_scratch_file(rec, "../profile_x.json"),
                      because="a handle outside the scratch directory")
    _polarity.refuses(cc.resolve_scratch_file(rec, "profile_missing.json"),
                      because="a contained handle naming no file")
    _polarity.refuses(cc.resolve_scratch_file({"arm": cc.ARM_VULKAN}, "profile_x.json"),
                      because="a record with no scratch_dir")


def test_resolve_scratch_file_accepts_a_contained_handle(tmp_path):
    rec, _ = _resolvable_record(tmp_path)
    (tmp_path / "profile_x.json").write_text("[]", encoding="utf-8")

    got, why = cc.resolve_scratch_file(rec, "profile_x.json")

    assert got == (tmp_path / "profile_x.json").resolve() and why == ""


def test_resolve_trace_refuses_a_record_whose_trace_it_cannot_open(tmp_path):
    import cuda_profile as cp

    _polarity.refuses(cp.resolve_trace({"arm": cc.ARM_VULKAN}),
                      because="a record with no trace handle")
    _polarity.refuses(
        cp.resolve_trace({"arm": cc.ARM_VULKAN, "trace_rel": "trace_x.json",
                          "scratch_dir": "<elsewhere>/traced"}),
        because="a rooted directory this machine cannot resolve")


def test_resolve_trace_accepts_the_handle_run_traced_vulkan_writes(tmp_path):
    import cuda_profile as cp

    scratch = tmp_path / "traced"
    scratch.mkdir()
    (scratch / "trace_prefill_1.json").write_text("[]", encoding="utf-8")
    rec = {"arm": cc.ARM_VULKAN, "trace_rel": "trace_prefill_1.json",
           pp.RUNTIME_ONLY_KEY: {"scratch_dir": str(scratch.resolve())}}

    got, why = cp.resolve_trace(rec)

    assert got == (scratch / "trace_prefill_1.json").resolve() and why == ""


def test_relative_handle_refuses_to_name_a_file_outside_the_directory(tmp_path):
    """Reject polarity for the writer's half: no handle is better than a wrong one."""
    base = tmp_path / "slot"
    base.mkdir()

    _polarity.refuses(cc.relative_handle("", base), because="nothing was written")
    _polarity.refuses(cc.relative_handle(tmp_path / "elsewhere.json", base),
                      because="the file is not under the directory being rooted at")
    _polarity.refuses(cc.relative_handle(base, base),
                      because="the directory itself is not a file inside it")


def test_relative_handle_names_a_file_it_can_stand_behind(tmp_path):
    """Accept polarity, and the round trip: what it writes, the resolver reads."""
    base = tmp_path / "slot"
    (base / "sub").mkdir(parents=True)
    target = base / "sub" / "profile.json"
    target.write_text("[]", encoding="utf-8")

    handle, why = cc.relative_handle(target, base)

    assert handle == "sub/profile.json" and why == ""
    assert pp.contained_child(base, handle) == target.resolve()


