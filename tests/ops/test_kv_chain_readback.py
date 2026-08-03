"""Falsifiers for `bench/results/probe_kv_chain_readback.py`'s artifact.

The probe answers a narrower question than the residency probe it follows: not *may* the KV live
in device memory, but does keeping it there actually take bytes off the host<->device link.  These
assert the properties that make its answer worth anything, and they are written so that a future
improvement does not turn them red — the slope must fall, not equal a particular number.
"""

from __future__ import annotations

import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parents[2] / "bench" / "results"
ARTIFACTS = sorted(HERE.glob("kv_chain_readback-*.json"))


@pytest.fixture(params=[p.name for p in ARTIFACTS] or ["<none>"])
def artifact(request):
    if request.param == "<none>":
        pytest.skip("no kv_chain_readback artifact has been captured yet")
    return json.loads((HERE / request.param).read_text(encoding="utf-8"))


def test_the_lane_reports_which_device_it_actually_ran_on(artifact):
    """The selector is a request, not an identity.

    `ONNXRUNTIME_EP_VULKAN_DEVICE=0` has been observed running on the device the EP enumerates as
    1.  A claim about "both devices" taken from the env var would be a claim about what was asked
    for, so the artifact carries the name the run reported.
    """
    name = artifact.get("ep_device", {}).get("vulkan.device_name")
    assert name, "the artifact must name the device the run actually used"


def test_the_round_trip_verdict_is_earned_not_asserted(artifact):
    assert artifact["verdict"] in {
        "ROUND_TRIP_REMOVED",
        "ROUND_TRIP_NOT_REMOVED",
        "RESIDENT_CHAIN_DISAGREES",
        "RESIDENT_CHAIN_RETURNS_NOTHING",
        "ERROR(instrument)",
    }


def test_correctness_is_settled_before_any_byte_count_is_read(artifact):
    """A cheaper lane that changed the answer is not a cheaper lane."""
    if artifact["verdict"] != "ROUND_TRIP_REMOVED":
        pytest.skip("this artifact makes no bandwidth claim")
    rel = artifact["resident_vs_host_final"]
    assert set(rel) >= {"attn_out", "present_key", "present_value"}
    assert max(rel.values()) <= 1e-3, (
        "keeping the KV on the device must not change the result; it agreed to the digit when "
        "this was captured"
    )


def test_an_all_zero_result_cannot_pass_as_agreement(artifact):
    """Two all-zero tensors agree perfectly and that has already shipped once on this branch."""
    if artifact["verdict"] != "ROUND_TRIP_REMOVED":
        pytest.skip("this artifact makes no bandwidth claim")
    nonzero = artifact["nonzero_final"]
    assert nonzero, "the degeneracy guard must be present, not merely satisfied"
    assert all(v > 0 for v in nonzero.values()), (
        "every output must contain something; a lane that returns nothing scores perfectly "
        "against another lane that returns nothing"
    )


def test_the_two_lanes_did_the_same_work(artifact):
    """The non-triviality guard: a lane that stopped dispatching would have a beautiful slope.

    This project has already been handed a 6.7% 'saving' that was a run ending early.
    """
    if artifact["verdict"] != "ROUND_TRIP_REMOVED":
        pytest.skip("this artifact makes no bandwidth claim")
    host = artifact["host_lane_dispatches_per_step"]
    res = artifact["resident_lane_dispatches_per_step"]
    assert sum(host) > 0, "a zero-dispatch lane measured nothing"
    assert sum(host) == sum(res), (
        "the lanes must execute the same dispatches for the byte difference to be about "
        "residency rather than about work"
    )


def test_the_claim_is_a_slope_and_the_slope_falls(artifact):
    """A total can be beaten by doing less; a per-step slope at equal work cannot.

    Asserted as an inequality on purpose. Writing `== 0` would turn red the day someone removes
    the remaining seeding transfer, which is a fix — the same trap that would have broken
    Trinity's liveness control had she asserted `== 14`.
    """
    if artifact["verdict"] != "ROUND_TRIP_REMOVED":
        pytest.skip("this artifact makes no bandwidth claim")
    host = artifact["host_lane_steady_bytes_per_step"]
    res = artifact["resident_lane_steady_bytes_per_step"]
    assert host > 0, (
        "the control lane must actually pay for the round trip, or there is nothing to remove "
        "and the comparison is UNOBSERVABLE rather than favourable"
    )
    assert res < host, "device residency must reduce the per-step link traffic"


def test_the_unmeasured_extent_is_disclosed_rather_than_implied(artifact):
    """The growing-context term is where the win is largest and this case cannot exercise it.

    The GQA evidence case fixes `past` at 4 — ORT rejects any other extent, measured. A reader
    must not be able to take this number for the one at ctx 8192.
    """
    if artifact["verdict"] != "ROUND_TRIP_REMOVED":
        pytest.skip("this artifact makes no bandwidth claim")
    why = " ".join(artifact["why"]).lower()
    assert "fixed past extent" in why, (
        "the artifact must say which extent it held constant, so its number is not mistaken for "
        "the growing-context one"
    )


def test_no_wall_clock_is_quoted(artifact):
    """The box is permanently contended (PERF.md §20). Counts and bounds only."""
    if artifact["verdict"] == "ERROR(instrument)":
        pytest.skip("no claim was made")
    flat = json.dumps(artifact).lower()
    for banned in ("ms/inference", "milliseconds", "elapsed_s", "wall_clock"):
        assert banned not in flat, f"{banned} is a timing claim and this box cannot support one"
