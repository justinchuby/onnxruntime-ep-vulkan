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


# ---------------------------------------------------- the correction: this is NOT an independent check

@pytest.mark.skipif(not RESULTS.exists(), reason="committed traces absent")
def test_same_ordinal_rsd_is_REDUNDANT_with_whole_series_rsd_and_that_must_stay_recorded():
    """The claim this module was originally sold on, pinned as false so it cannot come back.

    An earlier docstring said same-ordinal RSD and the whole-series per-inference spread were
    "two different statistics over two different frames ... neither derived from the other". They
    are strongly rank-correlated: a disturbance that scales a whole submission moves every dispatch
    inside it together. Niobe falsified it over the census; this recomputes it here so the
    refutation lives in the test suite and not only in prose someone may trim.

    The assertion is deliberately loose (rho > 0.8): the point is that these are NOT independent,
    not that rho has a particular value on a particular trace set.
    """
    xs, ys = [], []
    for p in sorted(RESULTS.glob("trace_*_dev0.json")):
        try:
            infs = rd.per_inference_kernel_us(p)
            m = rd.measure(infs)
        except Exception:  # noqa: BLE001
            continue
        totals = [sum(f) for f in infs if len(f) == m["dispatches_per_inference"]]
        if len(totals) < 3:
            continue
        mean = sum(totals) / len(totals)
        var = sum((t - mean) ** 2 for t in totals) / (len(totals) - 1)
        xs.append((var ** 0.5) / mean)
        ys.append(m["same_ordinal_rsd_median"])

    if len(xs) < 10:
        pytest.skip(f"only {len(xs)} usable traces; the correlation is not a statistic here")

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r

    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) ** 0.5) * (sum((b - my) ** 2 for b in ry) ** 0.5)
    rho = num / den

    assert rho > 0.8, (
        f"Spearman rho = {rho:.3f}. If this ever drops, the redundancy documented at the top of "
        "run_disturbance.py has changed and the module's scoping must be re-derived -- do not "
        "simply relax this bound.")


# ------------------------------------------------------------------- localisation, the part that is new

@pytest.mark.skipif(not DISTURBED.exists(), reason="committed trace absent")
def test_localise_explains_most_of_the_dispersion_on_a_submission_level_disturbance():
    """`contended`: removing per-inference level collapses same-ordinal RSD 137% -> ~19%.

    This is Niobe's mechanism measured rather than accepted. If it ever stops holding, her
    explanation for the correlation is wrong and the scoping above rests on it.
    """
    loc = rd.localise(rd.per_inference_kernel_us(DISTURBED))
    assert loc["explained_by_level"] > 0.6, loc
    assert loc["level_normalised_ordinal_rsd"] < loc["same_ordinal_rsd_median"] / 3
    assert loc["character"].startswith("SUBMISSION_LEVEL")


PER_DISPATCH = RESULTS / "trace_gemv_contended3_dev0.json"


@pytest.mark.skipif(not PER_DISPATCH.exists(), reason="committed trace absent")
def test_localise_separates_two_runs_that_whole_series_rsd_calls_neighbours():
    """THE reason this module still exists after the redundancy finding.

    `ab_p1_long` (whole-series 37.8%) and `contended3` (34.4%) are neighbours to whole-series RSD
    and are different conditions. The decomposition must say so, in opposite directions -- if it
    ever agrees on both, it has stopped localising and is just another spread measure.
    """
    other = RESULTS / "trace_gemv_ab_p1_long_dev0.json"
    if not other.exists():
        pytest.skip("committed trace absent")
    a = rd.localise(rd.per_inference_kernel_us(other))
    b = rd.localise(rd.per_inference_kernel_us(PER_DISPATCH))
    assert a["character"].startswith("SUBMISSION_LEVEL"), a
    assert b["character"].startswith("PER_DISPATCH"), b
    assert a["explained_by_level"] > b["explained_by_level"] + 0.5


@pytest.mark.skipif(not CLEAN.exists(), reason="committed trace absent")
def test_localise_inherits_the_level_blindness_hole_rather_than_escaping_it():
    """The normalised statistic removes level ON PURPOSE, so it is *more* blind, not less.

    Asserted rather than described, because the temptation with any refinement is to believe it
    fixed the thing it was not aimed at. `baseline_certified` is the worked example: cleanest on
    every dispersion measure this project owns, and 21.4x wrong.
    """
    slow = rd.localise(rd.synthetic_uniform_slowdown(factor=2.0))
    fast = rd.localise(rd.synthetic_clean())
    assert rd.classify(rd.measure(rd.synthetic_uniform_slowdown(factor=2.0)))["verdict"] == "PASS"

    # Establish that the two inputs REALLY differ before asserting the statistic cannot tell.
    # Both normalised values are 0.0, so the equality below passes for free; without this the
    # test would be an assertion that can only succeed, which is not a check.
    a = rd.synthetic_uniform_slowdown(factor=2.0)
    b = rd.synthetic_clean()
    level_ratio = sum(sum(x) for x in a) / sum(sum(x) for x in b)
    assert abs(level_ratio - 2.0) < 1e-9, (
        f"the 'slow' control is only {level_ratio:.3f}x the clean one; the hole is not being "
        "demonstrated because the two inputs are not actually different")

    assert abs(slow["level_normalised_ordinal_rsd"] - fast["level_normalised_ordinal_rsd"]) < 1e-9, (
        "a 2x-uniformly-slow run must be INDISTINGUISHABLE from a clean one here; if it is not, "
        "this statistic has acquired level sensitivity and the scoping needs revisiting")
    assert abs(slow["same_ordinal_rsd_median"] - fast["same_ordinal_rsd_median"]) < 1e-9, (
        "the raw statistic is equally blind, which is the general result: no dispersion measure "
        "computed inside a series can see a bias that scales the series")

    # And the real trace that makes the point: certified-clean on both, and 21.4x wrong.
    cert = rd.localise(rd.per_inference_kernel_us(CLEAN))
    assert cert["same_ordinal_rsd_median"] < 0.01
    assert cert["level_normalised_ordinal_rsd"] < 0.01


@pytest.mark.skipif(not DISTURBED.exists(), reason="committed trace absent")
def test_localise_names_which_inferences_moved_not_merely_that_some_did():
    """Localisation's other half: whole-series RSD cannot point at a repetition. This must."""
    loc = rd.localise(rd.per_inference_kernel_us(DISTURBED))
    assert loc["inference_inflation_max"] > 2.0
    assert 0 < loc["inferences_over_1_10x"] < 46
    assert loc["worst_inferences"][0]["inflation_x"] == loc["inference_inflation_max"]
    assert all("index" in w for w in loc["worst_inferences"])


@pytest.mark.skipif(not RESULTS.exists(), reason="committed traces absent")
def test_the_threshold_band_narrowed_when_a_trace_landed_in_the_former_empty_gap():
    """The census falsified its own headline, and that must not be quietly re-smoothed.

    The original claim was an EMPTY gap from 10.507% to 35.313% with the threshold in it. With one
    further trace (`switch_resid`, 20.787%) the gap is populated and the disturbed-side margin
    collapsed from 1.77x to ~1.04x. This pins the fact that the separation is now a judgement on
    that trace, so nobody re-quotes the clean-separation figure.
    """
    resid = RESULTS / "trace_switch_resid_dev0.json"
    if not resid.exists():
        pytest.skip("committed trace absent")
    med = rd.measure(rd.per_inference_kernel_us(resid))["same_ordinal_rsd_median"]
    assert rd.DISTURBANCE_RSD_MAX < med < 0.35, (
        f"{med:.3%} — this trace sits inside what was reported as an empty gap")
    assert med / rd.DISTURBANCE_RSD_MAX < 1.2, (
        "its margin over the bar is small, so its verdict is a judgement rather than a separation "
        "and must be reported as one")
