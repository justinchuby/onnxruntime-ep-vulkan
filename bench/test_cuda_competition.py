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

import dataclasses
import json
import os
import sys
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
    """The real Phi-3.5 shape: 1 fused Vulkan node beside 8 residual host nodes.

    Node share says 88.9% CPU.  Time share says 0.2%.  This test pins the exact
    disagreement that made node-counting unusable, so nobody re-adopts it by
    accident.
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

    Raw ULP is measured in units of the *local* spacing, which collapses near zero,
    so a physically irrelevant difference between two near-zero logits scored ~18000
    against a 32-ULP budget on every GPU arm.  Peak-scaled ULP measures the same
    difference in units of the spacing at the tensor's largest magnitude, which is
    the scale softmax and argmax actually care about.
    """
    ref = np.array([[16.0, 1e-3, -1e-3]], dtype=np.float16)
    sub = ref.copy()
    # Move a near-zero element across zero: physically 2e-3 on a tensor peaking at
    # 16, but thousands of representable fp16 values.
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

    Measured on MobileNetV2 that is 18060 peak-ULP of error against the CPU EP where
    the Vulkan EP has 23, on identical model bytes.  The budget must follow the
    declared precision or the comparison either disqualifies the competitor or
    silently licenses a precision loss the subject does not take.
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


def test_missing_arm_yields_no_ratio():
    """A missing arm is not a losing arm."""
    got = cc.compare_workload(_workload(), [_rec(cc.ARM_VULKAN, [10.0] * 10)])
    assert got["verdict"] == cc.UNMEASURED
    assert "speedup_vulkan_over_cuda" not in got


def test_split_frame_samples_are_excluded_from_the_ratio():
    """A SPLIT_FRAME arm must not contribute samples to a headline number."""
    records = [
        _rec(cc.ARM_VULKAN, [10.0] * 10, verdict=cc.SPLIT_FRAME),
        _rec(cc.ARM_CUDA, [20.0] * 10),
    ]
    got = cc.compare_workload(_workload(), records)
    assert got["n_vulkan"] == 0
    assert got["verdict"] == cc.UNMEASURED


def test_divergent_arm_disqualifies_the_ratio():
    """Faster-but-wrong is not faster."""
    records = [_rec(cc.ARM_VULKAN, [5.0] * 10), _rec(cc.ARM_CUDA, [20.0] * 10)]
    eq = {"arms": {cc.ARM_VULKAN: {"verdict": "DIVERGENT"}}}
    got = cc.compare_workload(_workload(), records, equivalence=eq)
    assert got["verdict"] == cc.NOT_EQUIVALENT
    assert "speedup_vulkan_over_cuda" not in got


def test_vulkan_faster_requires_the_whole_interval_above_one():
    records = [_rec(cc.ARM_VULKAN, [10.0 + 0.01 * i for i in range(30)]),
               _rec(cc.ARM_CUDA, [30.0 + 0.01 * i for i in range(30)])]
    got = cc.compare_workload(_workload(), records)
    assert got["verdict"] == "VULKAN_FASTER"
    assert got["speedup_vulkan_over_cuda"]["lo"] > 1.0


def test_overlapping_arms_are_indistinguishable_not_a_winner():
    """The gate that stops a coin-flip from becoming a headline."""
    rng = np.random.default_rng(11)
    a = list(10.0 + rng.standard_normal(40))
    b = list(10.0 + rng.standard_normal(40))
    got = cc.compare_workload(_workload(), [_rec(cc.ARM_VULKAN, a), _rec(cc.ARM_CUDA, b)])
    assert got["verdict"] == "INDISTINGUISHABLE"


def test_cuda_faster_requires_the_whole_interval_below_one():
    records = [_rec(cc.ARM_VULKAN, [45.0 + 0.01 * i for i in range(30)]),
               _rec(cc.ARM_CUDA, [15.0 + 0.01 * i for i in range(30)])]
    got = cc.compare_workload(_workload(), records)
    assert got["verdict"] == "CUDA_FASTER"
    assert got["speedup_vulkan_over_cuda"]["hi"] < 1.0


def test_runtime_version_confound_bracket_is_reported_when_both_cpu_arms_ran():
    records = [
        _rec(cc.ARM_VULKAN, [45.0] * 10), _rec(cc.ARM_CUDA, [15.0] * 10),
        _rec(cc.ARM_CPU_HOST, [104.0] * 10), _rec(cc.ARM_CPU_CUDA_RT, [100.0] * 10),
    ]
    got = cc.compare_workload(_workload(), records)
    conf = got["runtime_version_confound"]
    assert conf["ratio_cpu_cudart_over_cpu_host"]["ratio"] == pytest.approx(100.0 / 104.0)


def test_confound_bracket_is_absent_when_a_cpu_arm_is_missing():
    """No half-bracket: one CPU arm alone cannot bound a version difference."""
    records = [_rec(cc.ARM_VULKAN, [45.0] * 10), _rec(cc.ARM_CUDA, [15.0] * 10),
               _rec(cc.ARM_CPU_HOST, [104.0] * 10)]
    got = cc.compare_workload(_workload(), records)
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
    change had broken, and it survived review, 81/81 bench tests and 19/19 Rust tests,
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
            # An attribution that cannot see the whole Compute call cannot rule anything
            # out.  `vulkan.compute_call` is the outer bracket; anchored on the inner
            # `vulkan.subgraph` the reduction is blind to the region where the
            # counters-dump artifact was hiding — measured at a warm median of 27.0 ms
            # per call, which is larger than the device time the trace *did* see.
            anchor = payload.get("anchor") or {}
            assert anchor.get("span"), (
                f"{path.name}: no anchor recorded. A profile that does not say which span "
                f"it bucketed against cannot be audited: the same numbers mean different "
                f"things depending on whether the outer bracket was used.")
            if not anchor.get("sees_whole_compute"):
                # Admissible, but only as a bounded claim, and the bound must be on the
                # artifact rather than in whoever remembers the run.
                assert payload.get("scope_limits"), (
                    f"{path.name}: anchored on {anchor['span']!r}, which cannot see the "
                    f"whole Compute callback, and carries no scope_limits saying so.")
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
# Measured on Phi-3.5 `prefill_1`: median 68.53 ms with the dump live against 30.36 ms
# without it. **2.26x**, all of it this harness measuring itself, and all of it charged
# to the Vulkan EP in a report whose entire purpose is to compare the Vulkan EP against
# CUDA. The first baseline concluded "CUDA is 2.9x faster on prefill_1"; over half of
# that gap was the instrument.
#
# The trap is that nothing looked wrong. Every gate passed, every arm read ADMISSIBLE,
# equivalence read MATCH, and the numbers were stable across 20 iterations with a tight
# confidence interval — a precisely reproducible measurement of the wrong thing. It was
# only found by asking where 27 ms of untraced host time went.
#
# So the invariant is not "remember to unset the variable". It is that evidence
# collection and timing are separate regions, and the code says which is which.


def test_counters_dump_is_not_left_inside_the_timed_region():
    """The counters env var must be dropped before anything that is timed.

    Asserted on the source rather than by running a session, because the failure is a
    *timing* artifact: a functional test passes cheerfully while the number it produces
    is inflated 2.26x. What can be checked cheaply and deterministically is the ordering
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
        "it set puts a file write inside every timed inference — measured at 2.26x on "
        "Phi-3.5 prefill_1.")


def test_counters_scope_is_recorded_on_every_vulkan_arm():
    """The record must say which regime it was measured under.

    A number that was inflated by instrumentation and a number that was not are not
    interchangeable, and the withdrawn `baseline_main*.json` files contained the former:
    their Vulkan `prefill_1` median read 44.605 ms against 27.733 ms for the same arm in
    `baseline_fixed.json`, and they said `ADMISSIBLE` with no refusals while carrying no
    `counters_scope` at all. Without this field on the record there is nothing in the
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

    The A/B is the only evidence that the 2.26x claim is real, and an escape hatch that
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
    """A screen that silently rules on nothing passes for the wrong reason."""
    files = _committed_evidence_json()
    assert files, "no committed evidence JSON found; the screens below would rule on nothing"
    records = [r for f in files
               for r in _measurement_records(json.loads(f.read_text("utf-8")))]
    assert records, "committed evidence contains no measurement records"
    assert any(rec.get("arm") == cc.ARM_VULKAN for _, rec in records), (
        "no committed Vulkan record: the counters_scope screen below would be vacuous")


def test_every_committed_vulkan_record_declares_its_counters_scope():
    """A committed Vulkan number must say which instrumentation regime produced it.

    The EP rewrites its counters JSON after every `Compute`. With that dump left inside
    the timed region a Vulkan median is inflated by the harness's own file write, and the
    resulting record is indistinguishable from a clean one unless it says so. The
    withdrawn `baseline_main_v2.json` was exactly that: 64 `ADMISSIBLE` results, no
    refusals, no `counters_scope`, and a Vulkan `prefill_1` median 1.61x the corrected
    one.

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
