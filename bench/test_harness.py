"""Self-tests for the benchmark harness itself.

The harness is the thing that decides what counts as a regression, so its statistics and its
honesty gates need to be tested like any other code — and unlike the benchmarks, these run
anywhere, need no GPU, and need no EP.

Run: ``python -m pytest bench/test_harness.py -q``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pytest  # noqa: E402

import environment  # noqa: E402
from stats import NOISY_RSD, Sample, comparable, relative_delta, significant  # noqa: E402
from transfer_calibration import least_squares_fit  # noqa: E402


def test_median_and_spread_reject_a_single_outlier():
    clean = Sample("s", [10.0] * 20)
    contaminated = Sample("s", [10.0] * 20 + [1000.0])
    assert clean.median == pytest.approx(10.0)
    # One 100x outlier must not move the median, and must not blow up the robust spread.
    assert contaminated.median == pytest.approx(10.0)
    assert contaminated.mad == pytest.approx(0.0)


def test_rsd_flags_a_noisy_sample():
    steady = Sample("s", [10.0, 10.1, 9.9, 10.05, 9.95] * 4)
    jittery = Sample("s", [5.0, 15.0, 6.0, 14.0, 7.0, 13.0] * 4)
    assert not steady.noisy
    assert jittery.noisy
    assert jittery.rsd > NOISY_RSD
    assert not comparable(steady, jittery)


def test_a_delta_inside_the_noise_is_not_significant():
    base = Sample("b", [10.0, 12.0, 8.0, 11.0, 9.0] * 6)  # ~20% spread
    pr = Sample("p", [11.0, 13.0, 9.0, 12.0, 10.0] * 6)  # ~10% slower median
    assert relative_delta(base, pr) > 0
    # 10% threshold, but the samples' own spread is larger — this must not be flagged.
    assert not significant(base, pr, 0.10)


def test_a_real_regression_on_steady_samples_is_significant():
    base = Sample("b", [10.0, 10.1, 9.9] * 8)
    pr = Sample("p", [13.0, 13.1, 12.9] * 8)
    assert significant(base, pr, 0.10)
    assert relative_delta(base, pr) == pytest.approx(0.3, abs=0.02)


def test_quantiles_and_empty_samples_do_not_raise():
    empty = Sample("e", [])
    assert empty.n == 0
    assert empty.median != empty.median  # NaN
    assert not empty.noisy
    d = empty.to_dict()
    assert d["n"] == 0


def test_transfer_fit_recovers_a_known_affine_model():
    # ns = 5000 + bytes / 8  (i.e. 8 bytes/ns = ~7.45 GiB/s)
    samples = [(b, 5000.0 + b / 8.0) for b in (1024, 4096, 65536, 1 << 20, 1 << 22)]
    fit = least_squares_fit(samples)
    assert fit is not None
    assert fit["fixed_ns"] == pytest.approx(5000.0, rel=1e-3)
    assert fit["bytes_per_ns"] == pytest.approx(8.0, rel=1e-3)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-6)


def test_transfer_fit_refuses_degenerate_and_impossible_data():
    assert least_squares_fit([(1024, 10.0)]) is None
    assert least_squares_fit([(1024, 10.0), (1024, 20.0)]) is None
    # A device that gets faster with more bytes is a broken measurement, not a discovery.
    assert least_squares_fit([(1024, 100.0), (2048, 50.0), (4096, 10.0)]) is None


def test_environment_capture_is_json_serialisable_and_never_raises():
    record = environment.capture()
    json.dumps(record)  # must not raise
    assert "host" in record and "build" in record and "devices" in record
    # No device on this box is a legitimate state and must be *recorded*, not hidden.
    if not record["devices"]:
        assert record["device_note"]
    assert isinstance(environment.describe(record), str)


def test_compare_reconstructs_samples_from_a_summary_only_json():
    import compare

    row = {"vulkan": {"name": "x", "median_ms": 10.0, "mad_ms": 0.5}}
    s = compare._sample(row, "vulkan")
    assert s is not None
    assert s.median == pytest.approx(10.0)
    assert compare._sample({"vulkan": None}, "vulkan") is None


def test_cases_carry_the_metadata_the_report_contract_requires():
    import cases

    built = cases.build_cases()
    assert built, "the harness must have cases"
    for c in built:
        assert c.name and c.group and c.model and isinstance(c.feeds, dict)
        # boundary_bytes_per_inference is a required reporting field (OP_COVERAGE.md §7.3).
        assert c.boundary_bytes > 0
    # Exactly one case is the OQ-12 anchor; the >=1.5x bar is defined on one shape, not on a mood.
    assert sum(1 for c in built if c.oq12_anchor) == 1


def test_the_transfer_staircase_is_log_spaced_and_covers_both_regimes():
    import cases

    steps = cases.transfer_staircase(20)
    assert steps[0] == 1024
    assert steps[-1] == 1 << 20
    assert all(b * 2 == a for a, b in zip(steps[1:], steps[:-1]))
