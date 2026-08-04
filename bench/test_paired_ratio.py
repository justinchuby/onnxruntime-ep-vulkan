"""Locks the analysis half of `bench/results/probe_paired_ratio.py`.

The probe's whole claim is a *methodological* one: that a paired, finely interleaved A/B ratio
cancels a disturbance the two arms share, and that it can tell when it has not. A test suite for
it therefore cannot only check that the arithmetic runs. It has to feed the analysis three
synthetic worlds whose right answers are known by construction:

  * a **common-mode** disturbance (both arms slowed by the same factor) must yield PAIRING_HOLDS;
  * an **arm-specific** disturbance (one arm slowed) must yield PAIRING_FAILS(not_common_mode);
  * **no** disturbance at all must yield VACUOUS(injection_not_witnessed) — which is not a pass.

The third is the one that matters most. An instrument that reports success when nothing was
injected would have certified every one of this project's contended runs, and the negative
result this probe actually produced would never have been found.
"""
from __future__ import annotations

import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "results"))

probe = pytest.importorskip("probe_paired_ratio")


# ---------------------------------------------------------------------------- synthetic worlds
def _rows(phase, *, n_sweeps=12, n_steps=6, vk_ms=100.0, ref_ms=300.0,
          vk_mult=1.0, ref_mult=1.0, jitter=0.0, seed=7):
    """A phase's worth of rows with a known ratio and a known dispersion.

    `jitter` is a multiplicative log-normal shock applied to *both* arms of a pair identically:
    that is the machine wandering, which is exactly what a paired design is supposed to cancel.
    """
    rng = random.Random(seed)
    out = []
    t = 0.0
    for s in range(n_sweeps):
        for k in range(n_steps):
            shock = math.exp(rng.gauss(0.0, jitter)) if jitter else 1.0
            for arm, base, mult in (("vk", vk_ms, vk_mult), ("ref", ref_ms, ref_mult)):
                ms = base * mult * shock
                out.append({"phase": phase, "sweep": s, "step": k, "arm": arm,
                            "past_len": 4 + k, "ms": ms, "t0": t, "t": t + ms / 1000.0})
                t += ms / 1000.0
    return out


def _pairs(**kw):
    return probe.pair_ratios(_rows(**kw))


# ------------------------------------------------------------------------------------- pairing
def test_pair_ratios_never_pools_across_step_index():
    """Per-step cost grows with past_len; a pooled ratio would hide the axis it varies on."""
    rows = _rows("paired", n_sweeps=2, n_steps=3)
    pairs = probe.pair_ratios(rows)
    assert len(pairs) == 6
    for p in pairs:
        assert p["past_len"] == 4 + p["step"]
    # Every pair is drawn from one (sweep, step) cell, so the cells are disjoint.
    assert len({(p["sweep"], p["step"]) for p in pairs}) == len(pairs)


def test_pair_ratios_drops_unmatched_steps():
    rows = [r for r in _rows("paired", n_sweeps=1, n_steps=2) if not (r["step"] == 1 and r["arm"] == "ref")]
    pairs = probe.pair_ratios(rows)
    assert [p["step"] for p in pairs] == [0]


def test_geomean_matches_hand_value():
    assert probe.geomean([1.0, 4.0]) == pytest.approx(2.0)


def test_summarise_reports_dispersion_not_only_centre():
    s = probe.summarise(_pairs(phase="paired", jitter=0.0))
    assert s["geomean"] == pytest.approx(100.0 / 300.0, rel=1e-9)
    assert s["log_sd"] == pytest.approx(0.0, abs=1e-12)
    assert s["spread_x"] == pytest.approx(1.0, rel=1e-9)
    # A summary without dispersion is the bare number the brief forbids.
    for key in ("p10", "p90", "log_sd", "dispersion_x", "spread_x", "geomean_ci95_x", "n"):
        assert key in s


# -------------------------------------------------------------- the three verdicts, by construction
def test_common_mode_disturbance_yields_pairing_holds():
    """POSITIVE CONTROL. Both arms slowed 3x: the ratio must not move, and the test must say so."""
    base = _pairs(phase="paired", jitter=0.05, seed=1)
    hit = _pairs(phase="load", vk_mult=3.0, ref_mult=3.0, jitter=0.05, seed=2)
    rr = probe.ratio_of_ratios(base, hit, boot=400)
    lift = probe.level_lift(base, hit)
    v = probe.pairing_verdict(rr, lift)
    assert lift["vk_lift_x"] == pytest.approx(3.0, rel=0.05)
    assert lift["ref_lift_x"] == pytest.approx(3.0, rel=0.05)
    assert rr["ratio_of_ratios"] == pytest.approx(1.0, rel=0.02)
    assert v["verdict"] == "PAIRING_HOLDS"


def test_arm_specific_disturbance_yields_pairing_fails():
    """NEGATIVE CONTROL. Only the reference arm slowed: the ratio moves, and it must be caught."""
    base = _pairs(phase="paired", jitter=0.05, seed=1)
    hit = _pairs(phase="load", vk_mult=1.0, ref_mult=2.0, jitter=0.05, seed=2)
    rr = probe.ratio_of_ratios(base, hit, boot=400)
    v = probe.pairing_verdict(rr, probe.level_lift(base, hit))
    assert rr["ratio_of_ratios"] == pytest.approx(0.5, rel=0.05)
    assert not rr["contains_one"]
    assert v["verdict"] == "PAIRING_FAILS(not_common_mode)"


def test_injection_that_moves_nothing_is_vacuous_not_a_pass():
    """The failure mode that would have certified every contended run this project ever took."""
    base = _pairs(phase="paired", jitter=0.02, seed=1)
    hit = _pairs(phase="load", jitter=0.02, seed=2)
    rr = probe.ratio_of_ratios(base, hit, boot=400)
    v = probe.pairing_verdict(rr, probe.level_lift(base, hit))
    assert rr["within_tolerance"], "the synthetic world has no disturbance at all"
    assert v["verdict"] == "VACUOUS(injection_not_witnessed)"
    assert v["verdict"] != "PAIRING_HOLDS"


def test_a_barely_felt_injection_is_vacuous_below_the_stated_threshold():
    assert probe.INJECTION_MIN_LIFT == 0.15
    base = _pairs(phase="paired", seed=1)
    hit = _pairs(phase="load", vk_mult=1.10, ref_mult=1.10, seed=1)
    v = probe.pairing_verdict(probe.ratio_of_ratios(base, hit, boot=200),
                              probe.level_lift(base, hit))
    assert v["verdict"] == "VACUOUS(injection_not_witnessed)"


def test_empty_phase_is_unobservable_not_zero():
    v = probe.pairing_verdict(probe.ratio_of_ratios([], _pairs(phase="load")), {})
    assert v["verdict"] == "UNOBSERVABLE"


# ------------------------------------------------------------------------------- apparatus cost
def test_ratio_of_ratios_ci_brackets_the_point_estimate():
    rng = random.Random(3)

    def noisy(mult, n=72):
        return [{"phase": "p", "sweep": i, "step": 0, "past_len": 4, "vk_ms": 100.0,
                 "ref_ms": 300.0, "ratio": mult * math.exp(rng.gauss(0, 0.15))} for i in range(n)]

    rr = probe.ratio_of_ratios(noisy(0.33), noisy(0.50), boot=600)
    lo, hi = rr["ci95"]
    assert lo < rr["ratio_of_ratios"] < hi
    assert not rr["contains_one"]


def test_ratio_of_ratios_is_deterministic_for_a_fixed_seed():
    base, hit = _pairs(phase="paired", jitter=0.1, seed=5), _pairs(phase="load", jitter=0.1, seed=6)
    a = probe.ratio_of_ratios(base, hit, boot=300)
    b = probe.ratio_of_ratios(base, hit, boot=300)
    assert a["ci95"] == b["ci95"]


# --------------------------------------------------------------------------------- sample size
def test_pairing_buys_variance_reduction_only_when_the_shock_is_shared():
    """A shared shock is cancelled by the ratio; an independent one is not.

    This is the quantitative form of the whole argument, and it is why the real run's measured
    `variance_reduction_x` of ~1.3-1.4x — not the 5x-plus a paired design is adopted for — is
    the number that decides the design, per §10.3's 2.65x machine.
    """
    shared = probe.sample_size(_pairs(phase="paired", jitter=0.25, seed=11))
    assert shared["log_sd_ratio_paired"] == pytest.approx(0.0, abs=1e-9)
    assert shared["variance_reduction_x"] is None or shared["variance_reduction_x"] > 10

    rng = random.Random(99)
    indep = []
    for i in range(72):
        vk = 100.0 * math.exp(rng.gauss(0, 0.25))
        ref = 300.0 * math.exp(rng.gauss(0, 0.25))
        indep.append({"phase": "p", "sweep": i, "step": 0, "past_len": 4,
                      "vk_ms": vk, "ref_ms": ref, "ratio": vk / ref})
    s = probe.sample_size(indep)
    assert s["variance_reduction_x"] == pytest.approx(1.0, rel=0.25), \
        "pairing cancels nothing when the two arms wander independently"


def test_sample_size_pairs_needed_grows_with_dispersion():
    tight = probe.sample_size(_pairs(phase="p", jitter=0.0, seed=1) or [])
    rng = random.Random(3)
    loose = [{"phase": "p", "sweep": i, "step": 0, "past_len": 4, "vk_ms": 1.0, "ref_ms": 1.0,
              "ratio": math.exp(rng.gauss(0, 0.4))} for i in range(72)]
    assert probe.sample_size(loose)["pairs_for_5pct_ci"] > tight["pairs_for_5pct_ci"]


def test_sample_size_refuses_a_single_pair():
    assert probe.sample_size(_pairs(phase="p", n_sweeps=1, n_steps=1)) == {"n": 1}


# ------------------------------------------------------------------------------- clock windows
def test_clock_by_arm_attributes_samples_to_the_executing_arm():
    """A phase-wide median is dominated by the slower arm; §20.2 is about our own dispatches."""
    rows = [
        {"phase": "paired", "sweep": 0, "step": 0, "arm": "vk", "past_len": 4,
         "ms": 100.0, "t0": 0.0, "t": 1.0},
        {"phase": "paired", "sweep": 0, "step": 0, "arm": "ref", "past_len": 4,
         "ms": 300.0, "t0": 1.0, "t": 5.0},
    ]
    samples = [{"t": 0.5, "sm_mhz": 2000.0}, {"t": 3.0, "sm_mhz": 210.0},
               {"t": 4.0, "sm_mhz": 210.0}]
    got = probe.clock_by_arm(samples, rows, "paired")
    assert got["vk"]["sm_mhz_median"] == 2000.0
    assert got["ref"]["sm_mhz_median"] == 210.0
    # The phase-wide reduction reports the majority arm, which is the misreading being avoided.
    assert probe.clock_window(samples, 0.0, 5.0)["sm_mhz_median"] == 210.0


def test_clock_by_arm_says_unobservable_rather_than_inventing_a_clock():
    got = probe.clock_by_arm([], [{"phase": "p", "arm": "vk", "t0": 0.0, "t": 1.0}], "p")
    assert got["vk"]["verdict"] == "UNOBSERVABLE"
    assert got["ref"]["verdict"] == "UNOBSERVABLE"


def test_clock_window_is_unobservable_with_no_samples_in_range():
    assert probe.clock_window([{"t": 9.0, "sm_mhz": 300.0}], 0.0, 1.0)["verdict"] == "UNOBSERVABLE"


# ----------------------------------------------------------------------------- stated constants
def test_the_judgement_constants_are_stated_and_not_silently_retuned():
    assert probe.PAIRING_TOLERANCE == 0.10
    assert probe.INJECTION_MIN_LIFT == 0.15
    assert probe.BYTES_PER_PAST_TOKEN == 393216


def test_provenance_table_classifies_every_emitted_figure():
    """§22: no figure ships without a provenance class, and a ratio is never SPECIFICATION."""
    prov = probe.PROVENANCE
    assert set(prov.values()) <= {"SPECIFICATION", "MEASUREMENT", "MODEL"}
    assert prov["ratio"] == "MEASUREMENT"
    assert prov["sm_max_mhz"] == "SPECIFICATION"
    assert prov["BYTES_PER_PAST_TOKEN"] == "MODEL"
    assert prov["pairs_for_5pct_ci"] == "MODEL"
    for key in ("step_ms", "sm_mhz", "device_name", "coresidency_cost_x",
                "interleaving_cost_x", "variance_reduction_x"):
        assert prov[key] == "MEASUREMENT"
