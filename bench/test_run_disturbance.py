"""Both arms of the disturbance guard, on committed traces and on constructed ground truth.

A guard that has only been shown to fire is a printed opinion; a guard that has only been shown
not to fire is a no-op. Every claim below is paired.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import run_disturbance as rd  # noqa: E402

RESULTS = ROOT / "bench" / "results"
DISTURBED = RESULTS / "trace_gemv_contended_dev0.json"
CLEAN = RESULTS / "trace_gemv_baseline_certified_dev0.json"


def _assess(path: Path) -> dict:
    return rd.classify(rd.measure(rd.per_inference_kernel_us(path)))


# --------------------------------------------------------------------------- real traces, both arms

@pytest.mark.skipif(not DISTURBED.exists(), reason="committed trace absent")
def test_fails_on_the_most_disturbed_committed_run():
    """`contended` carries a 129.5% per-inference spread. It must not publish."""
    v = _assess(DISTURBED)
    assert v["verdict"] == "FAIL", v
    assert v["condition"] == "RUN_DISTURBED"
    assert v["same_ordinal_rsd_median"] > rd.DISTURBANCE_RSD_MAX


@pytest.mark.skipif(not CLEAN.exists(), reason="committed trace absent")
def test_passes_on_the_steadiest_committed_run():
    """The negative control. Without this the guard could be a constant FAIL."""
    v = _assess(CLEAN)
    assert v["verdict"] == "PASS", v
    assert v["condition"] is None
    assert v["same_ordinal_rsd_median"] < rd.DISTURBANCE_RSD_MAX


@pytest.mark.skipif(not (CLEAN.exists() and DISTURBED.exists()), reason="committed traces absent")
def test_the_two_arms_are_separated_by_more_than_the_threshold_placement():
    """The verdicts must not hinge on where in the gap the threshold sits.

    Both committed arms keep their verdict anywhere in the census gap (10.507% .. 35.313%), so the
    result is a property of the runs and not of a fitted constant.
    """
    clean = _assess(CLEAN)["same_ordinal_rsd_median"]
    dirty = _assess(DISTURBED)["same_ordinal_rsd_median"]
    for candidate in (0.11, 0.15, 0.20, 0.25, 0.30, 0.35):
        assert clean < candidate < dirty, (candidate, clean, dirty)


@pytest.mark.skipif(not (CLEAN.exists() and DISTURBED.exists()), reason="committed traces absent")
def test_the_tail_suffix_frame_would_NOT_work_and_that_is_why_the_frame_is_the_whole_run():
    """Pins the negative result that decided the frame.

    The obvious frame for a companion to a published figure is the suffix that figure covers. It
    was tried first and it has no discriminating power: restricted to its own steady suffix, the
    most disturbed run in the census reads about as quiet as the steadiest one, because the tail's
    selection has already found a quiet window.

    If this ever starts failing, the suffix frame has become viable and the module's frame choice
    (and its scoping to "a statement about the run, not the suffix cut from it") should be
    revisited. That is the point of pinning it.
    """
    import phases

    def suffix_rsd(path):
        events = phases.load(path)
        subs = phases.subgraph_spans(events)
        gpus = phases.gpu_spans(events)
        busy = phases.attribute_gpu_ordinally(subs, gpus)["busy_us"]
        tail = phases.gpu_steady_tail([busy.get(s["index"]) for s in subs])
        runs = rd.per_inference_kernel_us(path)
        n = tail.get("n")
        assert isinstance(n, int) and n >= rd.MIN_REPETITIONS, (path.name, n)
        return rd.measure(runs[-n:])["same_ordinal_rsd_median"]

    dirty_suffix = suffix_rsd(DISTURBED)
    clean_suffix = suffix_rsd(CLEAN)

    # Over the whole run the separation is enormous...
    assert _assess(DISTURBED)["same_ordinal_rsd_median"] > 20 * \
        _assess(CLEAN)["same_ordinal_rsd_median"]
    # ...and inside the tail's own suffix it collapses to nothing usable.
    assert dirty_suffix < rd.DISTURBANCE_RSD_MAX, dirty_suffix
    assert dirty_suffix < 10 * clean_suffix, (dirty_suffix, clean_suffix)


# ------------------------------------------------------------------ constructed ground truth

def test_jitter_is_detected():
    """Alternating inflation, the condition the guard exists for."""
    v = rd.classify(rd.measure(rd.synthetic_jittered()))
    assert v["verdict"] == "FAIL"
    assert v["condition"] == "RUN_DISTURBED"


def test_perfectly_repeatable_heterogeneous_work_passes():
    """A wide spread of kernel *shapes* with no variance must not read as disturbance.

    This is the failure mode of a naive per-kernel spread check: within one inference these
    durations span 30x, and none of it is disturbance.
    """
    v = rd.classify(rd.measure(rd.synthetic_clean()))
    assert v["verdict"] == "PASS"
    assert v["same_ordinal_rsd_median"] == pytest.approx(0.0, abs=1e-12)


def test_a_uniformly_slow_run_PASSES_and_that_is_the_documented_hole():
    """The boundary of the claim, asserted rather than described.

    Every dispatch doubled for the whole run -- exactly what a competing process holding a fixed
    share of the device produces. Repetitions agree perfectly, so the guard passes it. If this
    test ever starts failing, the guard has gained bias sensitivity it does not claim, and the
    docstring scoping (and the obligation-8 complement argument) must be revisited.
    """
    slow = rd.synthetic_uniform_slowdown(factor=2.0)
    v = rd.classify(rd.measure(slow))
    assert v["verdict"] == "PASS"
    assert "uniformly slow run" in v["does_not_detect"]
    # And it really is 2x slower, so the pass is not an artefact of an empty input.
    assert sum(slow[0]) == pytest.approx(2.0 * sum(rd.synthetic_clean()[0]))


# ----------------------------------------------------------------------------- R13: instrument errors

def test_too_few_repetitions_is_an_instrument_error_not_a_pass():
    with pytest.raises(rd.InstrumentError, match="repetitions"):
        rd.measure(rd.synthetic_clean(n_inferences=2))


def test_a_wandering_dispatch_count_is_an_instrument_error_not_a_pass():
    """If ordinal k is not the same node every inference, comparing it compares different work."""
    runs = rd.synthetic_clean(n_inferences=8, n_kernels=40)
    for i, r in enumerate(runs):
        del r[: i % 5]  # dispatch count wanders
    with pytest.raises(rd.InstrumentError, match="modal"):
        rd.measure(runs)


def test_an_unreadable_trace_is_an_instrument_error(tmp_path):
    bad = tmp_path / "trace_broken.json"
    bad.write_text("{}")
    with pytest.raises(Exception):
        rd.per_inference_kernel_us(bad)


# --------------------------------------------------------------------------------- the CI wrapper

def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "ci" / "check_run_disturbance.py"), *args],
        capture_output=True, text=True)


@pytest.mark.skipif(not DISTURBED.exists(), reason="committed trace absent")
def test_cli_exits_1_on_a_disturbed_run():
    p = _run("--trace", str(DISTURBED))
    assert p.returncode == 1, p.stdout + p.stderr
    assert "RUN-DISTURBANCE: FAIL(condition=RUN_DISTURBED)" in p.stdout


@pytest.mark.skipif(not CLEAN.exists(), reason="committed trace absent")
def test_cli_exits_0_on_a_clean_run():
    p = _run("--trace", str(CLEAN))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "RUN-DISTURBANCE: PASS" in p.stdout


def test_cli_reports_instrument_error_rather_than_passing_on_junk(tmp_path):
    bad = tmp_path / "trace_junk.json"
    bad.write_text("not json")
    p = _run("--trace", str(bad))
    assert p.returncode == 4, p.stdout + p.stderr
    assert "ERROR(instrument=" in p.stdout
    assert "NOT a detection and NOT a pass" in p.stdout


def test_cli_usage_error_without_a_target():
    assert _run().returncode == 2
