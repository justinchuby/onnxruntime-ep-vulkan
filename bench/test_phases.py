"""Tests for `bench/phases.py` — the phase-attribution module and its falsifiers.

Every test here is written the way the module is: a synthetic trace is built with a *known*
defect, and the test asserts that the corresponding falsifier goes red. A falsifier that has
never been shown to fail is not an instrument; it is a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pytest  # noqa: E402

import devices  # noqa: E402
import phases  # noqa: E402


# ---------------------------------------------------------------------------
# Trace construction helpers — all times in microseconds, as Chrome Trace uses.
# ---------------------------------------------------------------------------

def _sub(ts, dur, nodes):
    return {"name": "vulkan.subgraph", "cat": "ep", "ph": "X", "ts": ts, "dur": dur,
            "pid": 1, "tid": 0, "args": {"nodes": nodes, "device": "host"}}


def _phase(name, ts, dur):
    return {"name": f"vulkan.{name}", "cat": "ep.phase", "ph": "X", "ts": ts, "dur": dur,
            "pid": 1, "tid": 0, "args": {"device": "host", "caveat": "x"}}


def _gpu(ts, gpu_ns, period=1.0, bits=64, kernel="k"):
    return {"name": f"vulkan.gpu.{kernel}", "cat": "gpu", "ph": "X", "ts": ts,
            "dur": int(gpu_ns / 1000), "pid": 1, "tid": 9,
            "args": {"gpu_ns": gpu_ns, "timestamp_period_ns": period,
                     "timestamp_valid_bits": bits, "node_index": 0, "device": "vulkan-gpu"}}


def _transfer(ts, direction, nbytes, us):
    gib_s = (nbytes / (1024 ** 3)) / (us / 1e6)
    return [
        {"name": "vulkan.transfer_bytes", "cat": "counter", "ph": "C", "ts": ts, "pid": 1,
         "tid": 0, "args": {direction: nbytes}},
        {"name": "vulkan.transfer_gib_s", "cat": "counter", "ph": "C", "ts": ts + 1, "pid": 1,
         "tid": 0, "args": {direction: gib_s}},
    ]


def _island(ts, nodes, record_us, submit_us, fence_us, gpu_ns=None, upload=None):
    """One complete subgraph invocation: subgraph span with record/submit/fence nested inside."""
    ev = [_sub(ts, record_us + submit_us + fence_us + 100, nodes),
          _phase("record", ts + 10, record_us),
          _phase("submit", ts + 10 + record_us, submit_us),
          _phase("fence_wait", ts + 10 + record_us + submit_us, fence_us)]
    if upload:
        nbytes, us = upload
        ev += _transfer(ts + 20, "upload", nbytes, us)
    if gpu_ns:
        ev.append(_gpu(ts + 10 + record_us + submit_us + 5, gpu_ns))
    return ev


def _run(cycles, sizes, record_by_cycle=None, upload_bytes=None):
    """A whole run: `cycles` inferences, each executing `sizes` islands in the same order."""
    ev = []
    t = 1_000_000
    for c in range(cycles):
        for k, n in enumerate(sizes):
            rec = (record_by_cycle[c] if record_by_cycle else 1000) * (1 + k * 0)
            up = ((upload_bytes[k], rec * 0.9) if upload_bytes else None)
            ev += _island(t, n, rec, 20, 300, gpu_ns=50_000, upload=up)
            t += rec + 20 + 300 + 5000
    return ev


# ---------------------------------------------------------------------------
# Structure and attribution
# ---------------------------------------------------------------------------

def test_phases_are_attributed_to_the_subgraph_that_contains_them():
    ev = _run(2, [1, 4])
    r = phases.analyse(ev)
    assert r["subgraph_spans"] == 4
    assert r["host_phases_ms"]["record"]["n"] == 4
    assert r["falsifiers"]["phase_containment"]["red"] is False


def test_a_phase_span_outside_every_subgraph_goes_red():
    """If the nesting assumption is wrong, no phase share below it may be read."""
    ev = _run(2, [1])
    ev.append(_phase("record", 99_000_000, 500))  # nowhere near any subgraph
    r = phases.analyse(ev)
    f = r["falsifiers"]["phase_containment"]
    assert f["red"] is True
    assert f["orphan_phase_spans"] == 1


def test_phases_summing_past_their_own_subgraph_goes_red():
    ev = [_sub(1000, 500, 1), _phase("record", 1010, 5000)]
    r = phases.analyse(ev)
    assert r["falsifiers"]["phase_containment"]["red"] is True
    assert r["falsifiers"]["phase_containment"]["over_subscribed_subgraphs"] == 1


def test_unattributed_time_inside_compute_is_reported_not_folded_away():
    """A phase split whose parts do not sum to the whole must say so."""
    ev = _run(1, [1])
    r = phases.analyse(ev)
    assert r["unattributed_in_compute_ms"] > 0
    total = sum(v["total_ms"] for v in r["host_phases_ms"].values())
    assert total + r["unattributed_in_compute_ms"] == pytest.approx(
        r["time_in_compute_ms"], rel=1e-6)


# ---------------------------------------------------------------------------
# GPU time and the 52x trap
# ---------------------------------------------------------------------------

def test_gpu_totals_come_from_gpu_ns_not_from_truncated_dur():
    """`dur` is integer microseconds; a 2.7 us kernel truncates to 2, a 26% error."""
    ev = [_sub(1000, 10_000, 1), _phase("submit", 1010, 20), _phase("fence_wait", 1030, 5000)]
    ev += [_gpu(1100 + i, 2_700.0) for i in range(100)]
    r = phases.analyse(ev)
    assert r["gpu"]["all"]["total_ms"] == pytest.approx(0.27, rel=1e-9)
    # What the naive reading would have produced, for contrast:
    assert sum(g["dur_us_truncated"] for g in phases.gpu_spans(ev)) == 200  # i.e. 0.20 ms


def test_the_52x_trap_is_caught_end_to_end_on_a_non_unit_period():
    """A dropped period scale emits raw ticks; ticks / 52.0833 is not a whole number."""
    period = 52.0833
    good = [_sub(1000, 90_000, 1), _phase("submit", 1010, 20), _phase("fence_wait", 1030, 80_000)]
    good += [_gpu(1100, ticks * period, period=period, bits=36) for ticks in (100, 250, 3777)]
    r = phases.analyse(good)
    f = r["falsifiers"]["timestamp_conversion_integrality"]
    assert f["decisive"] is True and f["red"] is False

    bad = [_sub(1000, 90_000, 1), _phase("submit", 1010, 20), _phase("fence_wait", 1030, 80_000)]
    bad += [_gpu(1100, float(ticks), period=period, bits=36) for ticks in (100, 250, 3777)]
    rb = phases.analyse(bad)
    fb = rb["falsifiers"]["timestamp_conversion_integrality"]
    assert fb["red"] is True
    assert "under-reported" in fb["results"][0]["detail"]


def test_the_52x_check_reports_vacuous_on_a_unit_period_rather_than_passing():
    """NVIDIA and lavapipe both report 1.0; the check cannot fail there and must not pass."""
    ev = _run(1, [1])
    r = phases.analyse(ev)
    f = r["falsifiers"]["timestamp_conversion_integrality"]
    assert f["decisive"] is False
    assert f["results"][0]["verdict"] == "VACUOUS"
    assert any("NOT DECISIVE" in m for m in phases.red_flags(r))


def test_gpu_time_exceeding_the_fence_wait_goes_red():
    """The CPU was blocked for that long; the GPU cannot have been busy longer."""
    ev = [_sub(1000, 10_000, 1), _phase("submit", 1010, 20), _phase("fence_wait", 1030, 1_000)]
    ev.append(_gpu(1100, 9_000_000.0))  # 9 ms of GPU inside a 1 ms fence wait
    r = phases.analyse(ev)
    assert r["falsifiers"]["gpu_containment"]["red"] is True


def test_a_duration_wider_than_the_wrap_period_goes_red():
    period, bits = 52.0833, 36
    wrap_ns = (1 << bits) * period
    ev = [_sub(1000, 10_000_000, 1), _phase("submit", 1010, 20),
          _phase("fence_wait", 1030, 9_000_000)]
    ev.append(_gpu(1100, wrap_ns * 1.5, period=period, bits=bits))
    r = phases.analyse(ev)
    assert r["falsifiers"]["valid_bits_applied"]["red"] is True


# ---------------------------------------------------------------------------
# Trace vs counters
# ---------------------------------------------------------------------------

def test_trace_and_counters_must_have_watched_the_same_executions():
    ev = _run(3, [1, 1])
    assert phases.analyse(ev, {"compute_calls": 6})["falsifiers"][
        "trace_matches_counters"]["red"] is False
    bad = phases.analyse(ev, {"compute_calls": 7})["falsifiers"]["trace_matches_counters"]
    assert bad["red"] is True
    assert "wrong denominator" in bad["detail"]


def test_without_a_counters_file_the_check_refuses_rather_than_passes():
    r = phases.analyse(_run(2, [1]), None)
    assert r["falsifiers"]["trace_matches_counters"]["red"] is None


# ---------------------------------------------------------------------------
# Question 4 — recording: does it scale, and is the decline real?
# ---------------------------------------------------------------------------

def test_the_island_cycle_is_recovered_from_the_trace_not_taken_from_a_counter():
    ev = _run(4, [1, 10, 10])
    sc = phases.analyse(ev)["record_scaling"]
    assert sc["islands_per_inference_inferred"] == 3
    assert sc["cycle_count"] == 4


def test_a_pooled_decline_that_is_really_island_mix_is_not_called_warmup():
    """Islands run in a fixed order, so the first k spans ARE one inference.

    Here every island records at a perfectly constant cost across all four inferences; only the
    *identity* of the island differs within a cycle. A first-k-vs-last-k comparison over a
    misaligned window would see a change. The per-island control must not.
    """
    ev = []
    t = 1_000_000
    costs = [4000, 1000, 1000]  # island 0 is expensive, always
    for _ in range(4):
        for k, n in enumerate([1, 10, 10]):
            ev += _island(t, n, costs[k], 20, 300)
            t += costs[k] + 6000
    sc = phases.analyse(ev)["record_scaling"]
    assert sc["decline_verdict"] == "NO_DECLINE"
    assert sc["per_island_last_over_first"]["fraction_declining"] == 0.0


def test_a_real_warmup_survives_the_per_island_control():
    ev = _run(5, [1, 10, 10], record_by_cycle=[8000, 3000, 2000, 2000, 2000])
    sc = phases.analyse(ev)["record_scaling"]
    assert sc["decline_verdict"] == "REAL_WARMUP"
    assert sc["per_island_last_over_first"]["fraction_declining"] == 1.0
    assert sc["flattened"] is True
    assert "flattened" in sc["decline_detail"]


def test_a_decline_still_falling_at_the_last_cycle_is_reported_as_not_flattened():
    ev = _run(5, [1, 10], record_by_cycle=[16000, 8000, 4000, 2000, 1000])
    sc = phases.analyse(ev)["record_scaling"]
    assert sc["decline_verdict"] == "REAL_WARMUP"
    assert sc["flattened"] is False
    assert "NOT flattened" in sc["decline_detail"]


def test_the_size_question_is_undecidable_when_every_island_is_the_same_size():
    """Not a refutation. A partition with one island size cannot answer the question."""
    ev = _run(3, [1, 1, 1])
    sc = phases.analyse(ev)["record_scaling"]
    assert sc["size_verdict"] == "UNDECIDABLE"
    assert "Not a refutation" in sc["size_detail"]


def test_the_upload_memcpy_is_split_out_of_the_record_span():
    """`record` is not one activity: the staging memcpy happens inside it and has a different fix."""
    ev = _island(1_000_000, 10, record_us=10_000, submit_us=20, fence_us=300,
                 upload=(64 * 1024 ** 2, 9_000))
    r = phases.analyse(ev)
    sc = r["record_scaling"]
    assert sc["upload_inside_record_ms"] == pytest.approx(9.0, rel=1e-3)
    assert sc["command_construction_ms"] == pytest.approx(1.0, rel=1e-3)
    assert sc["upload_share_of_record"] == pytest.approx(0.9, rel=1e-3)
    assert r["transfers"]["upload"]["inside_record_span"] is True


def test_recording_that_scales_one_to_one_with_dispatch_count_is_detected():
    ev = []
    t = 1_000_000
    for _ in range(4):
        for n, cost in ((1, 1000), (10, 10_000)):
            ev += _island(t, n, cost, 20, 300)
            t += cost + 6000
    sc = phases.analyse(ev)["record_scaling"]
    assert sc["size_verdict"] == "SCALES_WITH_DISPATCH_COUNT"


def test_recording_that_ignores_dispatch_count_is_detected():
    ev = []
    t = 1_000_000
    for _ in range(4):
        for n in (1, 10):
            ev += _island(t, n, 5000, 20, 300)
            t += 5000 + 6000
    sc = phases.analyse(ev)["record_scaling"]
    assert sc["size_verdict"] == "DOES_NOT_SCALE_WITH_DISPATCH_COUNT"


def test_a_bytes_cost_wearing_a_size_label_is_named_as_such():
    """The strongest discriminator on two island sizes predicts a value, not an ordering.

    If recording is a byte-throughput cost, the implied bandwidth is the *same* for every island
    size even though the durations differ by 4x. A rank correlation on two points cannot tell
    that apart from a per-dispatch cost; an implied-bandwidth comparison can.
    """
    ev = []
    t = 1_000_000
    gib = 1024 ** 3
    for _ in range(4):
        # both islands record at exactly 1 GiB/s of upload; sizes differ 10x, bytes differ 4x
        for n, nbytes in ((1, 16 * 1024 ** 2), (10, 64 * 1024 ** 2)):
            us = nbytes / gib * 1e6
            ev += _island(t, n, int(us * 1.02), 20, 300, upload=(nbytes, us))
            t += int(us * 1.02) + 6000
    sc = phases.analyse(ev)["record_scaling"]
    assert sc["size_verdict"] == "SCALES_WITH_BYTES_NOT_DISPATCHES"
    assert sc["implied_bandwidth_spread"] == pytest.approx(1.0, abs=0.05)
    assert "bytes effect wearing a size label" in sc["size_confound"]


def test_a_one_off_first_inference_cost_is_not_called_a_ramp():
    """"Warmup" and "gradual approach" are different shapes and imply different fixes."""
    ev = _run(6, [1, 10], record_by_cycle=[40_000, 5000, 5000, 5000, 5000, 5000])
    sc = phases.analyse(ev)["record_scaling"]
    assert sc["warmup_shape"] == "ONE_OFF_FIRST_INFERENCE"
    assert sc["cycles_to_steady"] == 2

    ramp = phases.analyse(
        _run(6, [1, 10], record_by_cycle=[40_000, 24_000, 14_000, 8000, 5200, 5000])
    )["record_scaling"]
    assert ramp["cycles_to_steady"] == 5
    assert ramp["warmup_shape"] == "RAMP_OVER_5_INFERENCES"


# ---------------------------------------------------------------------------
# The metric of record's third slot
# ---------------------------------------------------------------------------

def test_an_emitted_zero_for_largest_island_flops_is_not_a_measurement():
    ev = _run(1, [1])
    ev.append({"name": "vulkan.getcapability", "cat": "ep.claim", "ph": "i", "ts": 1, "pid": 1,
               "tid": 0, "s": "t",
               "args": {"claimed_nodes": 321, "island_count": 0, "largest_island_flops": 0,
                        "concentration": 0.0, "declined_Cast": "x2: ..."}})
    ps = phases.analyse(ev)["partition_stats"]
    assert ps["third_slot_state"] == "UNPOPULATED"
    assert "must not be quoted as 0" in ps["third_slot_note"]
    assert ps["island_count_state"].startswith("UNPOPULATED")
    assert ps["declined_kinds"] == 1


def test_a_missing_getcapability_event_is_absent_not_zero():
    ps = phases.analyse(_run(1, [1]))["partition_stats"]
    assert ps["present"] is False


# ---------------------------------------------------------------------------------------------
# Ordinal GPU attribution, and the device-identity trap.
#
# Both of these were found by running the instruments against real traces from the 2026-07-30
# re-measurement, not by reasoning about them. The first produced 14 phantom violations; the
# second printed the Intel name over NVIDIA numbers on every row of the results table.
# ---------------------------------------------------------------------------------------------


def test_ordinal_attribution_is_immune_to_a_huge_calibration_anchor_uncertainty():
    """A 300 ms anchor error must not move a single kernel between submissions.

    On the Intel part `anchor_uncertainty_us` reaches 314618. Timestamp-containment attribution
    misfiles spans under that error and invents "GPU busier than its own fence" violations on a
    build whose conversion arithmetic is provably correct.
    """
    events = []
    # Two submissions, 2 dispatches each, each with a generous fence window.
    for k in range(2):
        base = 1000 + k * 10_000
        events.append(_sub(base, 5000, 2))
        events.append(_phase("record", base + 10, 1000))
        events.append(_phase("submit", base + 1100, 100))
        events.append(_phase("fence_wait", base + 1300, 3000))
    # Every GPU span drawn far outside *any* subgraph interval: the anchor is 500 ms off.
    for j in range(4):
        g = _gpu(500_000 + j, 200_000.0)
        g["args"]["anchor_uncertainty_us"] = 314618.0
        events.append(g)

    subs = phases.subgraph_spans(events)
    gpus = phases.gpu_spans(events)

    acc = phases.gpu_span_accounting(subs, gpus)
    assert not acc["red"], acc["detail"]

    ordinal = phases.attribute_gpu_ordinally(subs, gpus)
    # Every span placed, split evenly, despite none of them landing inside any interval.
    assert ordinal["left_over"] == 0
    assert not ordinal["exhausted_early"]
    assert ordinal["spans_per_submission"] == {0: 2, 1: 2}

    cont = phases.gpu_containment(subs, phases.attribute(subs, phases.phase_spans(events)), gpus)
    assert not cont["red"], cont["detail"]
    assert "ordinal" in cont["attribution"]

    aq = phases.anchor_quality(gpus)
    assert aq["available"] and aq["max_us"] == 314618.0
    assert "not evidence" in aq["detail"]


def test_a_dispatch_that_produced_no_gpu_span_is_caught_by_integer_equality():
    """The failure that raises nothing: a kernel silently missing from the query results.

    It leaves `compute_failures` at 0 (§9.1.3 — an execution-status counter, never a correctness
    signal) and simply removes time from the GPU column. Only integer equality catches it.
    """
    events = [_sub(1000, 5000, 3),
              _phase("record", 1010, 1000),
              _phase("submit", 2100, 100),
              _phase("fence_wait", 2300, 2000)]
    events += [_gpu(2400 + j, 100_000.0) for j in range(2)]  # 2 spans for 3 dispatches

    subs = phases.subgraph_spans(events)
    acc = phases.gpu_span_accounting(subs, phases.gpu_spans(events))
    assert acc["red"]
    assert acc["expected_from_nodes"] == 3 and acc["observed_gpu_spans"] == 2
    assert "under-reported" in acc["detail"]


def test_accounting_extends_to_the_counters_file_so_a_trace_cannot_agree_with_only_itself():
    events = [_sub(1000, 5000, 2)] + [_gpu(2400 + j, 100_000.0) for j in range(2)]
    subs = phases.subgraph_spans(events)
    gpus = phases.gpu_spans(events)
    assert not phases.gpu_span_accounting(subs, gpus, {"dispatches_executed": 2})["red"]
    bad = phases.gpu_span_accounting(subs, gpus, {"dispatches_executed": 7})
    assert bad["red"] and "dispatches_executed=7" in bad["detail"]


def test_two_devices_calibration_in_one_trace_is_not_a_device_fingerprint():
    events = [_gpu(10, 1000.0, period=1.0, bits=64),
              _gpu(20, 1000.0, period=52.0833, bits=36)]
    fp = phases.device_fingerprint(phases.gpu_spans(events))
    assert not fp["consistent"]
    assert fp["timestamp_period_ns"] is None
    assert "MORE THAN ONE DEVICE" in fp["detail"]


def test_ep_device_index_is_best_first_not_enumeration_order():
    """The mislabelling bug, as a test.

    `ep.device_index` indexes `engine.rs::probe_devices`, documented "sorted best-first ... so
    index 0 is the default device". `vulkaninfo`/`probe()` return enumeration order. On a laptop
    with an iGPU at enum 0 and a dGPU at enum 1 the two orderings are reversed.
    """
    igpu = devices.DeviceFacts(index=0, name="Intel Iris Xe",
                               device_type="VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
                               timestamp_period_ns=52.0833, timestamp_valid_bits=36)
    dgpu = devices.DeviceFacts(index=1, name="NVIDIA RTX 4060",
                               device_type="VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                               timestamp_period_ns=1.0, timestamp_valid_bits=64)
    probed = [igpu, dgpu]

    assert [d.name for d in devices.ep_selection_order(probed)] == [dgpu.name, igpu.name]
    assert devices.by_ep_index(probed, 0).name == dgpu.name
    assert devices.ep_index_of(probed, 0) == 1  # enum 0 (Intel) is ep index 1


def test_device_identity_check_relabels_from_the_trace_and_never_guesses():
    igpu = devices.DeviceFacts(index=0, name="Intel Iris Xe",
                               device_type="VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
                               timestamp_period_ns=52.0833, timestamp_valid_bits=36)
    dgpu = devices.DeviceFacts(index=1, name="NVIDIA RTX 4060",
                               device_type="VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                               timestamp_period_ns=1.0, timestamp_valid_bits=64)
    probed = [igpu, dgpu]

    ok = devices.device_identity_check(probed, 0, 1.0, 64)
    assert ok["ok"] and ok["verdict"] == "MATCH"
    assert ok["device"].name == dgpu.name

    # A build that reverted to enumeration order would put Intel's period on ep index 0.
    bad = devices.device_identity_check(probed, 0, 52.0833, 36)
    assert bad["ok"] is False and bad["verdict"] == "MISLABELLED"
    assert bad["device"].name == igpu.name

    # No fingerprint at all: no name may be quoted. A plausible wrong name is worse than none.
    none = devices.device_identity_check(probed, 0, None, None)
    assert none["ok"] is None and none["verdict"] == "UNVERIFIED"
    assert none["name_may_be_quoted"] is False


def test_an_unknown_timestamp_fingerprint_identifies_nothing():
    dgpu = devices.DeviceFacts(index=0, name="NVIDIA RTX 4060",
                               device_type="VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                               timestamp_period_ns=1.0, timestamp_valid_bits=64)
    got, why = devices.identify_by_timestamp([dgpu], 38.4615, 32)
    assert got is None and "matches NO probed device" in why


def test_lavapipe_and_nvidia_share_a_fingerprint_so_identity_is_refused_not_guessed():
    """1.0/64 on both. An ambiguous discriminator must return no answer, not the first match."""
    nv = devices.DeviceFacts(index=0, name="NVIDIA RTX 4060",
                             device_type="VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                             timestamp_period_ns=1.0, timestamp_valid_bits=64)
    lava = devices.DeviceFacts(index=1, name="llvmpipe",
                               device_type="VK_PHYSICAL_DEVICE_TYPE_CPU",
                               timestamp_period_ns=1.0, timestamp_valid_bits=64)
    got, why = devices.identify_by_timestamp([nv, lava], 1.0, 64)
    assert got is None and "ambiguous" in why


