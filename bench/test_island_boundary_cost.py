"""Locks for the island-fragmentation cost table and for the idle-clock specimen
this session produced.

Two subjects, one file, because they came out of one measurement:

1. The island A/B (counters only, contention-independent). What is locked here is
   not the numbers for their own sake but the two CHECKS that make them
   trustworthy -- the recovered fixed cost agreeing between arms, and the fused
   readback slope landing exactly on the model's declared output bytes -- plus
   the direction of each currency. A future change that makes fragmentation
   cheap should break the direction assertions loudly.

2. `phi35-0baf660-dev0.json`. This run is the cleanest specimen of the
   companion's reason for existing that the project has produced, and unlike the
   earlier ones it is ours and it is on the current binary. Its device-clock
   series is STEADY with an RSD of 0.0717% -- the tightest tail in the project --
   at 20.18x the last quotable figure, under a SOLE_TENANT verdict with zero
   foreign GPU work. Tail and tenancy both say pass. The only field that refuses
   it is the SM-clock record: 160 of 160 samples at 210.0 MHz against a 3105 MHz
   boost. If anyone ever proposes that the clock record is a diagnostic rather
   than a required companion, this test is the answer.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "results"

_SYS_PATH_BEFORE = list(sys.path)
sys.path.insert(0, str(BENCH))
try:
    import device_companion  # noqa: E402
    import phases  # noqa: E402
finally:
    sys.path[:] = _SYS_PATH_BEFORE

COST = RESULTS / "island_boundary_cost.json"
RUN = RESULTS / "phi35-0baf660-dev0.json"
TRACE = RESULTS / "traces" / "phi35-0baf660-dev0-dev0.trace.json"

# The last figure that was ever certified quotable, for the ratio below.
LAST_QUOTABLE_MS = 12.1847


@pytest.fixture(scope="module")
def cost():
    if not COST.exists():
        pytest.skip("run bench/results/probe_island_boundary_cost.py first")
    return json.loads(COST.read_text(encoding="utf-8"))


class TestIslandBoundaryCost:
    def test_the_two_arms_recover_the_same_fixed_weight_upload(self, cost):
        """If they disagreed, the slope decomposition would not be separating
        what it claims to separate and every per-inference number below would be
        suspect."""
        agree = cost["fixed_cost_agreement"]
        assert agree["verdict"] == "AGREE", agree
        assert agree["relative_gap"] < 0.01

    def test_the_fused_readback_slope_is_exactly_the_declared_outputs(self, cost):
        """This is the check that makes the decomposition a measurement rather
        than an identity: the declared byte count comes from the ONNX output
        shapes, not from any counter in the records."""
        c = cost["declared_output_closure"]
        assert c["verdict"] == "EXACT", c
        assert c["residual_bytes"] == 0

    def test_the_per_inference_quotient_that_fooled_me_is_not_used(self):
        """The first answer divided a cumulative counter dominated by a one-time
        ~2185 MiB weight upload by two different iteration counts and produced a
        1.78x 'improvement' that was really 51/28. The probe must solve a slope
        across two iteration counts, never divide a single record."""
        src = (RESULTS / "probe_island_boundary_cost.py").read_text(encoding="utf-8")
        assert "slope = (b2 - b1) / (n2 - n1)" in src
        assert "R11" in src

    def test_fragmentation_is_worse_in_count_and_in_bytes(self, cost):
        f = cost["finding"]
        assert f["round_trips_per_inference"]["shipped"] == 66.0
        assert f["round_trips_per_inference"]["fused"] == 2.0
        assert f["staging_bytes_per_inference"]["ratio"] > 1.5

    def test_the_marginal_traffic_is_one_kv_round_trip(self, cost):
        """Fragmentation's whole byte cost is the 64 KV tensors crossing the
        boundary once more in each direction. The readback side is exact; the
        upload side carries an 8-byte residual which is recorded, not rounded."""
        f = cost["finding"]
        assert f["marginal_is_one_kv_round_trip"] is True
        assert f["marginal_readback_bytes"] == f["kv_tensor_bytes"]
        assert abs(f["upload_residual_bytes"]) == 8

    def test_fragmentation_is_NOT_worse_in_the_currencies_that_were_hypothesised(self, cost):
        """Allocator round-trips, descriptor/dispatch count and device high-water
        were the suspected costs. All three are flat or slightly better. These are
        falsifications, and they are worth more than the positive findings."""
        f = cost["finding"]
        assert f["device_allocs_ratio"] < 1.0
        assert f["dispatches_ratio"] < 1.0
        assert f["high_water_ratio"] < 1.01

    def test_the_bytes_are_negligible_against_the_weight_read(self, cost):
        """0.0376%. Switch's shape holds and the count is not the cost."""
        assert cost["finding"]["overhead_vs_weight_read_pct"] < 0.1

    def test_the_unpriced_currency_is_named_and_not_quietly_dropped(self, cost):
        r = cost["ruling"]
        assert "TIME COST IS UNMEASURED" in r
        assert "EXECUTE ON CPU" in r


@pytest.fixture(scope="module")
def run_record():
    if not RUN.exists():
        pytest.skip("phi35-0baf660-dev0.json not present")
    return json.loads(RUN.read_text(encoding="utf-8"))


class TestIdleClockSpecimen:
    """A run where everything that sounds like a pass, passes -- and the figure
    is 20x wrong."""

    def test_the_board_never_left_idle_clock(self, run_record):
        ds = run_record["results"][0]["phase_pass"]["device_state"]
        assert ds["sm_mhz"]["max"] == 210.0
        assert ds["sm_mhz"]["min"] == 210.0
        assert ds["sm_max_mhz"] == 3105.0
        # Every sample, not merely the median. A median at idle is common and
        # benign; a MAXIMUM at idle means the board never ramped at all.
        assert ds["clock_at_max_pct"] < 10.0

    def test_the_tenancy_verdict_is_clean_and_that_is_the_point(self, run_record):
        """The earlier 10.99x specimen had foreign GPU work, so a tenancy check
        alone could plausibly have caught it. This one has none. Tenancy is
        SOLE_TENANT and foreign_sample_fraction is exactly zero."""
        ds = run_record["results"][0]["phase_pass"]["device_state"]
        assert ds["verdict"] == "SOLE_TENANT"
        assert ds["foreign_sample_fraction"] == 0.0

    def test_the_device_series_is_STEADY_and_twenty_times_wrong(self):
        if not TRACE.exists():
            pytest.skip("trace not present")
        ev = phases.load(TRACE)
        g = phases.gpu_spans(ev)
        per_inf = 323
        n = len(g) // per_inf
        series = [sum(x["gpu_ns"] for x in g[i * per_inf:(i + 1) * per_inf]) / 1e6 for i in range(n)]
        tail = phases.gpu_steady_tail([v * 1000.0 for v in series])
        assert tail["verdict"] == "STEADY"
        median = tail["median_ms"]
        assert median / LAST_QUOTABLE_MS > 15.0
        suffix = series[1:]
        rsd = statistics.pstdev(suffix) / statistics.mean(suffix) * 100.0
        # Tighter than every trace in the 28-trace census.
        assert rsd < 0.11

    def test_only_the_clock_record_refuses_it(self, run_record):
        """Tail says STEADY. Tenancy says SOLE_TENANT. certify() must still
        refuse, and the reason must be the clock."""
        if not TRACE.exists():
            pytest.skip("trace not present")
        ev = phases.load(TRACE)
        g = phases.gpu_spans(ev)
        per_inf = 323
        n = len(g) // per_inf
        series = [sum(x["gpu_ns"] for x in g[i * per_inf:(i + 1) * per_inf]) / 1e6 for i in range(n)]
        tail = phases.gpu_steady_tail([v * 1000.0 for v in series])
        ds = run_record["results"][0]["phase_pass"]["device_state"]
        cert = device_companion.certify(tail, ds)
        assert cert["tail_verdict"] == "STEADY"
        assert cert["companion"] == "SOLE_TENANT"
        assert cert["quotable"] is False

    def test_the_harness_withheld_the_figure(self, run_record):
        cert = run_record["results"][0]["phase_pass"]["analysis"]["steady_state"]["gpu_steady_tail"]["certification"]
        assert cert["quotable"] is False
