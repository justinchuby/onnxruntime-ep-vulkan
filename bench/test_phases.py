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


# ---------------------------------------------------------------------------
# The field defect: a parent summed together with its own children (2026-07-31)
#
# `phase_containment` was handed the *unfiltered* phase list while every total and share was
# computed over `sibling_phases()`. On the one-island Phi-3.5 graph `record` is 8.32 s and the
# `cmd_upload` inside it is 7.97 s, so the sum came to 16.66 s against a 13.67 s subgraph and
# reported RED for three days with nothing wrong in the EP. Not one test below the line existed,
# because not one synthetic trace in this file contained a sub-record span.
# ---------------------------------------------------------------------------

def _nested_phase(name, ts, dur, parent="record", declare_caveat=True, declare_arg=True):
    """A sub-record span as `trace.rs` emits it: inside `record`, self-declaring its parent."""
    e = _phase(name, ts, dur)
    if declare_caveat:
        e["args"]["caveat"] = f"{phases.SUB_PHASE_CAVEAT_PREFIX} synthetic"
    if declare_arg:
        e["args"]["nested_in"] = parent
    return e


def test_a_parent_summed_with_its_own_children_is_not_a_containment_violation():
    """The exact shape that withheld every phase share in this project for three days."""
    ev = [_sub(1000, 1000, 353),
          _phase("record", 1010, 800),
          _nested_phase("cmd_upload", 1015, 780)]
    naive = 800 + 780
    assert naive > 1000, "the synthetic trace must reproduce the over-subscription arithmetic"
    f = phases.analyse(ev)["falsifiers"]["phase_containment"]
    assert f["state"] == "PASS", f["detail"]
    assert f["red"] is False
    assert f["nested_spans_checked"] == 1
    assert f["sibling_spans_checked"] == 1


def test_nested_spans_in_the_sibling_set_are_an_instrument_error_not_a_detection():
    """R13: ERROR(instrument) is a third state. It is neither a pass nor a detection."""
    subs = [{"ts": 1000, "dur": 1000, "end": 2000, "nodes": 1, "index": 0}]
    bad = phases.attribute(subs, phases.phase_spans(
        [_phase("record", 1010, 800), _nested_phase("cmd_upload", 1015, 780)]))
    f = phases.phase_containment(subs, bad)
    assert f["state"] == "ERROR"
    assert f["red"] is False, "an instrument error must never be counted as a detection"
    assert f["over_subscribed_subgraphs"] is None, "no verdict may be issued"
    assert "cmd_upload" in f["instrument_error"]


def test_the_instrument_error_reaches_red_flags_labelled_as_such():
    """Unquotable, but never reported as a defect that was found."""
    subs = [{"ts": 1000, "dur": 1000, "end": 2000, "nodes": 1, "index": 0}]
    bad = phases.attribute(subs, phases.phase_spans(
        [_phase("record", 1010, 800), _nested_phase("cmd_upload", 1015, 780)]))
    flags = phases.red_flags({"falsifiers": {"phase_containment":
                                             phases.phase_containment(subs, bad)}})
    assert any("ERROR(instrument)" in f and "NOT a detection" in f for f in flags)


def test_tier_two_fires_when_children_exceed_their_own_record_span():
    """Excluding children from tier 1 must not stop them being checked at all.

    The children each fit inside `record` individually — that is the only interesting case, since
    a child that runs past its parent's end is not contained by it and is caught as a stray
    instead.
    """
    ev = [_sub(1000, 5000, 1),
          _phase("record", 1010, 100),
          _nested_phase("cmd_upload", 1015, 60),
          _nested_phase("desc_alloc", 1020, 60)]
    f = phases.analyse(ev)["falsifiers"]["phase_containment"]
    assert f["state"] == "FAIL"
    assert f["over_subscribed_record_spans"] == 1
    assert f["stray_sub_record_spans"] == 0


def test_a_child_running_past_its_parents_end_is_caught_as_a_stray():
    ev = [_sub(1000, 5000, 1),
          _phase("record", 1010, 100),
          _nested_phase("cmd_upload", 1015, 400)]
    f = phases.analyse(ev)["falsifiers"]["phase_containment"]
    assert f["state"] == "FAIL"
    assert f["stray_sub_record_spans"] == 1


def test_tier_two_fires_when_a_sub_record_span_lies_outside_every_record_span():
    ev = [_sub(1000, 5000, 1),
          _phase("record", 1010, 100),
          _nested_phase("cmd_upload", 3000, 50)]  # inside the subgraph, outside `record`
    f = phases.analyse(ev)["falsifiers"]["phase_containment"]
    assert f["state"] == "FAIL"
    assert f["stray_sub_record_spans"] == 1


def test_a_child_declared_only_by_the_nested_in_arg_is_still_excluded():
    """Three sources, unioned. A caveat rewrite alone cannot re-promote a child to a sibling."""
    spans = phases.phase_spans([_phase("record", 1010, 800),
                                _nested_phase("readback", 1015, 780,
                                              declare_caveat=False)])
    assert "readback" in phases.nested_phase_names(spans)
    assert {p["phase"] for p in phases.sibling_phases(spans)} == {"record"}


def test_a_child_declared_only_by_its_caveat_is_still_excluded():
    spans = phases.phase_spans([_phase("record", 1010, 800),
                                _nested_phase("readback", 1015, 780, declare_arg=False)])
    assert "readback" in phases.nested_phase_names(spans)


def test_containment_still_fires_on_siblings_that_really_do_over_subscribe():
    """The fix must not be a mute button: siblings alone exceeding the subgraph is still FAIL."""
    ev = [_sub(1000, 1000, 1),
          _phase("record", 1010, 800),
          _phase("submit", 1810, 400),
          _nested_phase("cmd_upload", 1015, 700)]
    f = phases.analyse(ev)["falsifiers"]["phase_containment"]
    assert f["state"] == "FAIL"
    assert f["over_subscribed_subgraphs"] == 1


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


# ---------------------------------------------------------------------------
# Stable device identity (issue #18) — uuid/luid/pci carried by DeviceFacts.
# ---------------------------------------------------------------------------

def test_bytes_to_hex_matches_the_rust_query_device_identity_encoding():
    # Real RTX A1000 uuid bytes, cross-validated against nvidia-smi and the Rust EP.
    raw = [170, 223, 51, 212, 209, 24, 21, 95, 204, 96, 194, 43, 92, 53, 36, 99]
    assert devices._bytes_to_hex(raw) == "aadf33d4d118155fcc60c22b5c352463"
    assert devices._bytes_to_hex(None) is None
    assert devices._bytes_to_hex([]) is None
    assert devices._bytes_to_hex("not-a-list") is None


def test_parse_json_profile_extracts_uuid_luid_and_pci():
    payload = {
        "capabilities": {
            "device": {
                "properties": {
                    "VkPhysicalDeviceProperties": {
                        "deviceName": "NVIDIA RTX A1000",
                        "limits": {},
                    },
                    "VkPhysicalDeviceVulkan11Properties": {
                        "deviceUUID": [170, 223, 51, 212, 209, 24, 21, 95,
                                       204, 96, 194, 43, 92, 53, 36, 99],
                        "deviceLUID": [122, 223, 7, 0, 0, 0, 0, 0],
                        "deviceLUIDValid": True,
                    },
                    "VkPhysicalDevicePCIBusInfoPropertiesEXT": {
                        "pciDomain": 0,
                        "pciBus": 159,
                        "pciDevice": 0,
                        "pciFunction": 0,
                    },
                },
                "queueFamiliesProperties": [],
            }
        }
    }
    facts = devices._parse_json_profile(payload, index=0)
    assert facts.uuid == "aadf33d4d118155fcc60c22b5c352463"
    assert facts.luid == "7adf070000000000"
    assert facts.pci == "0000:9f:00.0"


def test_parse_json_profile_leaves_identity_none_when_extensions_are_absent():
    """MoltenVK / some mobile ICDs never report LUID or VK_EXT_pci_bus_info: absence, not guess."""
    payload = {
        "capabilities": {
            "device": {
                "properties": {
                    "VkPhysicalDeviceProperties": {"deviceName": "Apple M-series", "limits": {}},
                    "VkPhysicalDeviceVulkan11Properties": {
                        "deviceUUID": [1] * 16,
                        "deviceLUIDValid": False,
                    },
                },
                "queueFamiliesProperties": [],
            }
        }
    }
    facts = devices._parse_json_profile(payload, index=0)
    assert facts.uuid == "01" * 16
    assert facts.luid is None
    assert facts.pci is None


def test_identify_by_uuid_is_exact_even_for_two_identical_gpus():
    """Two identical GPUs share every timestamp fingerprint; uuid still tells them apart."""
    gpu0 = devices.DeviceFacts(index=0, name="NVIDIA RTX A1000", uuid="aa" * 16)
    gpu1 = devices.DeviceFacts(index=1, name="NVIDIA RTX A1000", uuid="bb" * 16)
    got, why = devices.identify_by_uuid([gpu0, gpu1], "BB" * 16)  # case-insensitive match
    assert got is gpu1 and "matches exactly one" in why


def test_identify_by_uuid_refuses_when_absent_or_unmatched():
    gpu0 = devices.DeviceFacts(index=0, name="NVIDIA RTX A1000", uuid="aa" * 16)
    got, why = devices.identify_by_uuid([gpu0], None)
    assert got is None and "no device uuid" in why
    got, why = devices.identify_by_uuid([gpu0], "cc" * 16)
    assert got is None and "matches NO probed device" in why


def test_device_identity_check_prefers_uuid_over_timestamp_when_both_are_present():
    """A uuid mismatch must win over an ambiguous/matching timestamp fingerprint."""
    gpu0 = devices.DeviceFacts(index=0, name="NVIDIA RTX A1000",
                               timestamp_period_ns=1.0, timestamp_valid_bits=64, uuid="aa" * 16)
    gpu1 = devices.DeviceFacts(index=1, name="NVIDIA RTX A1000 (second)",
                               timestamp_period_ns=1.0, timestamp_valid_bits=64, uuid="bb" * 16)
    probed = [gpu0, gpu1]

    # Timestamps alone are ambiguous between identical GPUs; uuid resolves it exactly.
    result = devices.device_identity_check(probed, 0, 1.0, 64, uuid="bb" * 16)
    assert result["ok"] is False and result["verdict"] == "MISLABELLED"
    assert result["device"].name == gpu1.name

    result_match = devices.device_identity_check(probed, 0, 1.0, 64, uuid="aa" * 16)
    assert result_match["ok"] and result_match["verdict"] == "MATCH"



# ---------------------------------------------------------------------------
# Steady state — the cold inference is a different workload after residency
# ---------------------------------------------------------------------------

def test_steady_state_excludes_the_cold_inference_and_reports_it_separately():
    """A 1997 MiB one-time upload averaged into three 0.4 MiB ones is not a per-inference cost."""
    ev = []
    t = 1_000_000
    for i, rec in enumerate([8000, 150, 150, 150]):
        ev += [_sub(t, rec + 400, 353), _phase("record", t + 5, rec),
               _nested_phase("cmd_upload", t + 6, rec - 20),
               _phase("fence_wait", t + 10 + rec, 300)]
        ev += _transfer(t + 7, "upload", (2_000_000_000 if i == 0 else 400_000), rec - 20)
        t += rec + 5000
    ss = phases.analyse(ev)["steady_state"]
    assert ss["verdict"] == "OK"
    assert ss["cycles"] == 4 and ss["warm_inferences"] == 3
    assert ss["upload_mib_cold"] > 1000 and ss["upload_mib_warm_mean"] < 1
    # The cold record must not appear in the warm total at all.
    assert ss["warm_phases_ms"]["record"] == pytest.approx(0.45, abs=0.01)
    assert ss["cold_phases_ms"]["record"] == pytest.approx(8.0, abs=0.01)
    assert ss["cold_over_warm_ratio"] > 10


def test_steady_state_refuses_with_fewer_than_three_cycles():
    """Two warm samples cannot show that they agree with each other."""
    ev = _run(2, [1])
    ss = phases.analyse(ev)["steady_state"]
    assert ss["verdict"] == "INSUFFICIENT"
    assert "at least 3" in ss["detail"]


def test_steady_state_gpu_idle_is_fence_minus_gpu_never_the_whole_fence():
    """`fence_wait` is an upper bound on GPU time; the idle part is what is left after it."""
    ev = []
    t = 1_000_000
    for _ in range(4):
        ev += [_sub(t, 1400, 1), _phase("record", t + 5, 100),
               _phase("submit", t + 105, 5), _phase("fence_wait", t + 110, 1000),
               _gpu(t + 200, 800_000)]
        t += 6000
    ss = phases.analyse(ev)["steady_state"]
    assert ss["warm_shares"]["fence_wait"] > ss["warm_shares"]["fence_wait_gpu_idle"]
    assert ss["warm_shares"]["fence_wait_gpu_idle"] == pytest.approx(
        (1000 - 800) * 3 / 1000.0 / ss["warm_in_compute_ms"], rel=0.02)


def test_multi_island_steady_state_drops_one_whole_cycle_not_one_span():
    ev = _run(4, [1, 4])
    ss = phases.analyse(ev)["steady_state"]
    assert ss["island_count"] == 2
    assert ss["cycles"] == 4 and ss["warm_inferences"] == 3

# ---------------------------------------------------------------------------
# The in-band control must survive the partitioner succeeding (2026-07-31)
#
# `contention_signature` refused any trace whose cycle period was 1. The period is the island
# count, so the check switched itself off permanently the moment the partitioner started
# producing a single fused island — the configuration the project is trying to reach. Nothing in
# the statistic needs two slots.
# ---------------------------------------------------------------------------

def _single_island_run(host_ms, gpu_ns_each):
    """One island, one dispatch each, so ordinal GPU attribution has the spans it needs."""
    ev = []
    t = 1_000_000
    for h, g in zip(host_ms, gpu_ns_each):
        dur = int(h * 1000)
        ev += [_sub(t, dur + 2000, 1), _phase("record", t + 5, dur),
               _phase("fence_wait", t + 10 + dur, 1500), _gpu(t + 20 + dur, g)]
        t += dur + 20_000
    return ev


def test_a_single_island_trace_is_still_testable_for_contention():
    """period == 1 is the goal state, not a reason to stop checking."""
    n = 16
    ev = _single_island_run([12.0] * n, [400_000] * n)
    cs = phases.analyse(ev)["contention_signature"]
    assert cs["verdict"] != "UNTESTABLE", cs.get("reason")
    assert cs["single_slot"] is True
    assert cs["slots_testable"] == 1


def test_single_island_host_excursions_are_detected_against_a_steady_gpu():
    n = 16
    host = [12.0] * n
    host[4] = 60.0          # a stall the device did not see
    ev = _single_island_run(host, [400_000] * n)
    cs = phases.analyse(ev)["contention_signature"]
    assert cs["verdict"] == "HOST_SIDE_EXCURSIONS"
    assert cs["stalled_slot_fraction"] == 1.0


def test_single_island_steady_run_is_not_called_contended():
    """The fix must not turn the check into one that always fires."""
    n = 16
    ev = _single_island_run([12.0 + (i % 3) * 0.2 for i in range(n)], [400_000] * n)
    cs = phases.analyse(ev)["contention_signature"]
    assert cs["verdict"] not in ("HOST_SIDE_EXCURSIONS", "UNTESTABLE"), cs


def test_untestable_reason_names_the_condition_that_actually_failed():
    """R13: quote the failure text. The old text said 'cycles=15; need >= 4' when 15 >= 4."""
    ev = _single_island_run([12.0] * 13, [400_000] * 13)  # 13 spans, only 13 cycles
    cs = phases.analyse(ev)["contention_signature"]
    if cs["verdict"] == "UNTESTABLE":
        assert "need >= 4" not in cs["reason"] or "13" not in cs["reason"]
    ev = _single_island_run([12.0] * 11, [400_000] * 11)  # under the 12-span floor
    cs = phases.analyse(ev)["contention_signature"]
    assert cs["verdict"] == "UNTESTABLE"
    assert "fewer than 12" in cs["reason"]

# ---------------------------------------------------------------------------
# One claim, one constant (2026-07-31)
#
# `phi35.baseline_disagreement` carried factor=2.0 while `admissible.baseline_comparability`
# used tol=0.25. The 2026-07-31 run's CPU baseline moved 291.8 -> 228.7 ms (1.276x) between its
# two device passes: inadmissible by one rule, silent by the other.
# ---------------------------------------------------------------------------

def test_the_two_baseline_checks_share_one_threshold():
    import admissible
    import phi35
    assert phi35.baseline_disagreement([]) is None
    moved = [{"device_index": 0, "cpu": {"median_ms": 291.7576}},
             {"device_index": 1, "cpu": {"median_ms": 228.7379}}]
    warn = phi35.baseline_disagreement(moved)
    assert warn is not None, "1.276x must not be silent when the gate refuses at 1.25x"
    assert "1.276" in warn
    cross = admissible.baseline_comparability(moved[0], moved[1], "a", "b")
    assert cross["ok"] is False and cross["verdict"] == "BASELINE_MOVED"
    # the same pair must not be admissible in one file and unremarkable in the other
    assert (warn is not None) == (cross["ok"] is False)


def test_a_steady_baseline_is_not_flagged_by_either_check():
    import admissible
    import phi35
    steady = [{"device_index": 0, "cpu": {"median_ms": 230.0}},
              {"device_index": 1, "cpu": {"median_ms": 240.0}}]
    assert phi35.baseline_disagreement(steady) is None
    assert admissible.baseline_comparability(steady[0], steady[1], "a", "b")["ok"] is True

# ---------------------------------------------------------------------------
# The GPU ramp that dropping one cold cycle does not remove (2026-07-31)
# ---------------------------------------------------------------------------

def test_gpu_steady_tail_finds_the_step_and_quotes_only_the_tail():
    """RTX 4060 shape: five inferences near 48.9 ms, then a step to 40.2 and rock stable."""
    series = [48.85, 48.91, 48.87, 48.88, 47.82] + [40.19, 40.22, 40.22, 40.17, 40.19,
                                                    40.21, 40.20, 40.21, 40.20, 40.19]
    t = phases.gpu_steady_tail([v * 1000 for v in series])
    assert t["verdict"] == "STEADY"
    assert t["discarded_inferences"] == 5
    assert t["n"] == 10
    assert t["median_ms"] == pytest.approx(40.20, abs=0.02)
    assert t["rsd"] < 0.001


def test_gpu_steady_tail_refuses_a_device_that_never_settles():
    """Intel Iris Xe shape: wanders 542-629 ms for the whole run. No figure may be quoted."""
    series = [577.34, 590.79, 555.24, 545.43, 547.24, 542.70, 553.56, 543.83,
              545.79, 545.36, 584.84, 542.39, 559.39, 557.87, 628.75]
    t = phases.gpu_steady_tail([v * 1000 for v in series])
    assert t["verdict"] == "NO_STEADY_TAIL"
    assert "never settled" in t["detail"]


def test_gpu_steady_tail_refuses_a_short_series_rather_than_guessing():
    t = phases.gpu_steady_tail([40_000] * 4)
    assert t["verdict"] == "INSUFFICIENT"


def test_gpu_steady_tail_reaches_the_steady_state_block():
    n = 16
    ev = _single_island_run([12.0] * n, [400_000] * 5 + [300_000] * (n - 5))
    t = phases.analyse(ev)["steady_state"]["gpu_steady_tail"]
    assert t["verdict"] == "STEADY"
    assert t["discarded_inferences"] == 5
    assert t["median_ms"] == pytest.approx(0.3, abs=0.001)


# ---------------------------------------------------------------------------
# The minimum-n / coverage floor on a settled tail (2026-08-01)
#
# The 2% RSD bar constrains the tail's internal spread, not its agreement with the device's
# true steady rate. Both specimens below are real: Switch's `contended` row passed at n=8 while
# sitting 2.1% above his solo figure, and my own pre-barrier-fix A/B produced tails at n=7 and
# n=5 -- 20.06 ms and 37.56 ms -- on a device whose two clean runs from the same DLL both read
# 13.346 ms. A flat suffix that is a small share of the run is an excursion, not a steady state.
# ---------------------------------------------------------------------------

def test_a_flat_suffix_that_is_a_small_share_of_the_run_is_not_a_number():
    """The n=5-of-43 specimen: RSD 1.3%, median 37.56, and the run's warm mean was 26.4."""
    wandering = [26.0, 41.2, 22.9, 30.1, 44.8, 21.7, 33.4, 45.9, 20.8, 28.6,
                 39.7, 24.3, 47.1, 22.2, 31.8, 43.6, 25.5, 36.9, 21.1, 46.4,
                 29.7, 34.2, 23.8, 42.5, 27.3, 45.1, 20.4, 38.8, 24.9, 32.6,
                 47.8, 21.9, 35.7, 43.1, 26.8, 30.4, 22.6, 44.3, 28.1]
    steady_excursion = [37.5, 37.6, 37.4, 37.6, 37.5]
    t = phases.gpu_steady_tail([v * 1000 for v in wandering + steady_excursion])
    assert t["verdict"] == "MARGINAL_TAIL"
    assert t["median_ms"] is None, "a marginal tail must not offer a median to quote"
    assert t["withheld_median_ms"] == pytest.approx(37.5, abs=0.1)
    assert t["n"] == 5
    assert t["coverage"] < phases.GPU_TAIL_MIN_COVERAGE
    # R13: quote the failure text, never the failure count.
    assert "is 11% of the 44 usable inferences" in t["detail"]
    assert "internal spread" in t["detail"]


def test_the_floor_that_would_have_refused_switchs_contended_row():
    """n=8 after discarding 38 — passed the RSD bar, sat 2.1% above solo."""
    noisy = [11.2 + (i % 7) * 0.9 for i in range(38)]
    t = phases.gpu_steady_tail([v * 1000 for v in noisy + [11.77] * 8])
    assert t["verdict"] == "MARGINAL_TAIL"
    assert t["n"] == 8
    assert t["median_ms"] is None


def test_a_long_tail_after_a_short_warmup_ramp_is_still_quotable():
    """The floor must not refuse the shape it was built to preserve: ramp then settle."""
    series = [14.96, 14.95, 13.62] + [13.34 + (i % 3) * 0.01 for i in range(40)]
    t = phases.gpu_steady_tail([v * 1000 for v in series])
    assert t["verdict"] == "STEADY"
    assert t["n"] >= 40
    assert t["coverage"] >= phases.GPU_TAIL_MIN_COVERAGE
    assert t["median_ms"] == pytest.approx(13.35, abs=0.02)


def test_the_marginal_verdict_is_not_a_steady_one_to_any_consumer():
    """Consumers gate on ``verdict == "STEADY"``; MARGINAL_TAIL must never satisfy that."""
    t = phases.gpu_steady_tail([v * 1000 for v in
                                [20.0, 45.0, 22.0, 48.0, 21.0, 44.0, 23.0, 47.0,
                                 20.5, 46.0, 24.0, 43.0, 25.0, 41.0, 26.0, 40.0,
                                 12.18, 12.19, 12.18, 12.17, 12.18]])
    assert t["verdict"] != "STEADY"
    assert t["verdict"] in ("MARGINAL_TAIL", "NO_STEADY_TAIL")


# ---------------------------------------------------------------------------
# Issue #88 — host cost attribution for the whole ORT `Compute` callback.
#
# Two brackets were added outside everything this file already knew about:
# `compute_call` (the whole ORT callback, opened in ep.rs) and `subgraph_dispatch` (the
# engine's dispatch inside it). They produce TWO residuals over TWO denominators, and the
# one rule that governs them is that they are never summed.
#
# The other correction in the same change: `readback` is a child of `subgraph_dispatch`,
# not of `record`. The EP drops the Record guard, submits, waits on the fence and only
# then copies outputs back. Both this module and `trace.rs` had declared otherwise.
# ---------------------------------------------------------------------------

def _decl(name, ts, dur, parent=None, caveat="x"):
    """A phase span that declares its own parent the way `trace.rs` emits it."""
    e = _phase(name, ts, dur)
    e["args"]["caveat"] = caveat
    e["args"]["nested_in"] = parent or "none"
    return e


def _call88(ts=1000, call=1000, dispatch=800, record=400, submit=50, fence=200, readback=100,
            cmd_upload=None):
    """One #88-shaped Compute callback: compute_call > subgraph_dispatch > {record, ...}.

    The subgraph span is co-terminous with `subgraph_dispatch`, as the EP emits it: the
    dispatch guard opens immediately inside `subgraph_region`.
    """
    d0 = ts + (call - dispatch) // 2
    ev = [_sub(d0, dispatch, 3),
          _decl("compute_call", ts, call),
          _decl("subgraph_dispatch", d0, dispatch, parent="compute_call"),
          _decl("record", d0 + 1, record, parent="subgraph_dispatch"),
          _decl("submit", d0 + 1 + record, submit, parent="subgraph_dispatch"),
          _decl("fence_wait", d0 + 1 + record + submit, fence, parent="subgraph_dispatch"),
          _decl("readback", d0 + 1 + record + submit + fence, readback,
                parent="subgraph_dispatch")]
    if cmd_upload:
        ev.append(_decl("cmd_upload", d0 + 2, cmd_upload, parent="record",
                        caveat=f"{phases.SUB_PHASE_CAVEAT_PREFIX} synthetic"))
    return ev


def test_the_two_residuals_are_two_subtractions_over_two_denominators():
    r = phases.host_residuals(phases.phase_spans(_call88()))
    assert r["verdict"] == "OK"
    # outer = compute_call - subgraph_dispatch, over compute_call
    assert r["outer"]["residual_us"] == 1000 - 800
    assert r["outer"]["denominator_us"] == 1000
    # inner = subgraph_dispatch - its own children, over subgraph_dispatch
    assert r["inner"]["residual_us"] == 800 - (400 + 50 + 200 + 100)
    assert r["inner"]["denominator_us"] == 800
    assert r["inner"]["subtracted"] == ["fence_wait", "readback", "record", "submit"]


def test_the_two_residuals_are_never_summed_and_no_field_offers_their_sum():
    """The invariant, enforced on the artifact and not only stated in prose.

    A consumer cannot add two numbers this module never emits together. The check is that
    no key anywhere in the record holds `outer + inner`, and that the invariant string
    reaches the artifact so a reader of the JSON is told why.
    """
    r = phases.host_residuals(phases.phase_spans(_call88()))
    forbidden = r["outer"]["residual_us"] + r["inner"]["residual_us"]

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)
        else:
            yield node

    assert forbidden not in [v for v in walk(r) if isinstance(v, int)]
    assert "NEVER SUMMED" in r["invariant"]
    assert r["outer"]["denominator_us"] != r["inner"]["denominator_us"]


def test_each_residual_moves_with_its_own_input_and_not_with_the_others():
    base = phases.host_residuals(phases.phase_spans(_call88()))
    wider_call = phases.host_residuals(phases.phase_spans(_call88(call=1400)))
    smaller_record = phases.host_residuals(phases.phase_spans(_call88(record=300)))

    assert wider_call["outer"]["residual_us"] == base["outer"]["residual_us"] + 400
    assert wider_call["inner"] == base["inner"], "the inner residual is not a function of the " \
                                                 "callback's own width"
    assert smaller_record["inner"]["residual_us"] == base["inner"]["residual_us"] + 100
    assert smaller_record["outer"] == base["outer"], "the outer residual is not a function of " \
                                                     "what the dispatch spent its time on"


def test_a_share_of_a_zero_denominator_is_undefined_and_not_zero():
    """0.0 reads as 'fully accounted for' — the single most misleading value here."""
    ev = _call88(call=0, dispatch=0, record=0, submit=0, fence=0, readback=0)
    r = phases.host_residuals(phases.phase_spans(ev))
    assert r["outer"]["share"] is None
    assert r["inner"]["share"] is None


def test_a_dispatch_that_outlasts_its_callback_is_refused_and_never_clamped():
    ev = phases.phase_spans(_call88(call=500, dispatch=800))
    with pytest.raises(phases.TraceRefused):
        phases.host_residuals(ev)


def test_children_that_outlast_their_dispatch_are_refused_and_never_clamped():
    ev = phases.phase_spans(_call88(dispatch=200, record=400))
    with pytest.raises(phases.TraceRefused):
        phases.host_residuals(ev)


def test_two_spans_of_one_phase_that_overlap_are_refused():
    """Concatenated traces and double-opened guards both land here.

    `sum(dur)` over overlapping spans over-counts the wall interval, so the residual taken
    against it is always SMALLER than the truth — the attribution looks better than it is.
    """
    ev = _call88()
    ev.append(_decl("record", ev[3]["ts"] + 10, 100, parent="subgraph_dispatch"))
    with pytest.raises(phases.TraceRefused):
        phases.host_residuals(phases.phase_spans(ev))


def test_a_trace_from_a_build_with_no_brackets_is_unavailable_not_zero():
    r = phases.host_residuals(phases.phase_spans(_run(1, [1])))
    assert r["verdict"] == "UNAVAILABLE"
    assert r["outer"] is None and r["inner"] is None
    assert "Absent, not zero" in r["detail"]


def test_a_parent_this_build_does_not_know_is_refused_rather_than_absorbed():
    ev = [_decl("record", 1000, 100, parent="a_phase_from_the_future")]
    with pytest.raises(phases.TraceRefused):
        phases.declared_parents(phases.phase_spans(ev))


def test_one_phase_declaring_two_parents_in_one_trace_is_refused():
    ev = [_decl("readback", 1000, 10, parent="record"),
          _decl("readback", 1100, 10, parent="subgraph_dispatch")]
    with pytest.raises(phases.TraceRefused):
        phases.declared_parents(phases.phase_spans(ev))


def test_a_cycle_in_the_declared_nesting_is_refused():
    ev = [_decl("record", 1000, 100, parent="submit"),
          _decl("submit", 1000, 100, parent="record")]
    with pytest.raises(phases.TraceRefused):
        phases.declared_parents(phases.phase_spans(ev))


def test_the_trace_outranks_this_modules_table_and_the_disagreement_is_reported():
    """Accept polarity for `declared_parents`, and the reason the table is not enforced.

    `PHASE_CHILDREN` said `readback` was a child of `record` for the whole life of that
    model and it was wrong. A checker that believed its own table could not have reported
    the table as the thing at fault.
    """
    spans = phases.phase_spans([_phase("record", 1000, 500),
                                _decl("readback", 1100, 50, parent="record")])
    assert phases.declared_parents(spans)["readback"] == "record"
    assert phases.PARENT_OF["readback"] == "subgraph_dispatch"
    v = phases.phase_nesting(spans)
    assert any("readback" in d for d in v["table_disagrees_with_trace"])


def test_readback_is_declared_inside_the_dispatch_and_not_inside_record():
    """Regression lock, matching `Phase::nested_in()` in rust/src/trace.rs.

    Production drops the Record guard, submits, waits on the fence, and only then reads
    back. Every 'recording residual' computed from the old declaration was too small by
    exactly the readback.
    """
    assert phases.PARENT_OF["readback"] == "subgraph_dispatch"
    assert "readback" not in phases.PHASE_CHILDREN["record"]
    assert set(phases.PHASE_CHILDREN["subgraph_dispatch"]) == {
        "record", "submit", "fence_wait", "readback"}


def test_a_readback_that_really_did_sit_inside_record_would_be_caught():
    """The evidence arm: containment disagreeing with the declaration goes red."""
    ev = _call88()
    # Move the readback inside the record span while it still declares the dispatch.
    ev[6]["ts"] = ev[3]["ts"] + 10
    v = phases.phase_nesting(phases.phase_spans(ev))
    assert v["ok"] is True, "containment by a grandparent's child is not itself a violation"
    # ...but declaring `record` while sitting outside it is:
    ev[6]["args"]["nested_in"] = "record"
    ev[6]["ts"] = 10_000_000
    v = phases.phase_nesting(phases.phase_spans(ev))
    assert v["ok"] is False and v["verdict"] == "MISMATCH"
    assert "readback" in v["detail"]


def test_the_brackets_are_not_summed_with_the_phases_they_contain():
    ev = _call88(cmd_upload=200)
    allp = phases.phase_spans(ev)
    sib = {p["phase"] for p in phases.sibling_phases(allp)}
    assert sib == {"record", "submit", "fence_wait", "readback"}
    assert phases.outer_bracket_names(allp) == {"compute_call", "subgraph_dispatch"}
    r = phases.analyse(ev)
    assert "compute_call" not in r["host_phases_ms"]
    assert "subgraph_dispatch" not in r["host_phases_ms"]
    assert set(r["outer_brackets_ms"]) == {"compute_call", "subgraph_dispatch"}
    assert r["falsifiers"]["phase_containment"]["state"] == "PASS"


def test_no_share_of_time_in_compute_is_ever_taken_over_a_bracket():
    """A span that CONTAINS the denominator cannot be a share of it."""
    r = phases.analyse(_call88())
    for key in r["shares_of_time_in_compute"]:
        assert not key.startswith("compute_call"), key
        assert not key.startswith("subgraph_dispatch"), key
    assert all(v <= 1.0 for v in r["shares_of_time_in_compute"].values())


def test_the_residuals_reach_the_report_and_the_rendered_description():
    r = phases.analyse(_call88())
    assert r["host_residuals"]["verdict"] == "OK"
    text = "\n".join(phases.describe(r))
    assert "TWO subtractions, TWO denominators" in text
    assert "NEVER SUMMED" in text


def test_a_build_without_the_brackets_reads_exactly_as_it_did_before():
    """No pre-#88 trace changes its numbers because this module learned two new phases."""
    ev = _run(2, [1, 4])
    r = phases.analyse(ev)
    assert r["host_phases_ms"]["record"]["n"] == 4
    assert r["outer_brackets_ms"] == {}
    assert r["host_residuals"]["verdict"] == "UNAVAILABLE"
    assert r["falsifiers"]["phase_containment"]["red"] is False


# ---------------------------------------------------------------------------
# HELD-OUT MUTATION BATTERY for the residual arithmetic (issue #88).
#
# The tests above are green. Green is evidence that the world is currently the way the
# checks expect, not that the checks would notice if it were not. So the protocol those
# tests apply is applied here to deliberately defective reimplementations of
# `host_residuals`, held out of the module under test, and every one must be caught.
#
# A mutant that survives names an assertion above that is decoration.
# ---------------------------------------------------------------------------

def _spans_of(ev):
    return phases.phase_spans(ev)


def _total(spans, name):
    return sum(p["dur"] for p in spans if p["phase"] == name)


def _mutant_sums_the_two_residuals(spans):
    """The exact error the invariant forbids: a single 'unaccounted' number over one
    denominator, counting the dispatch interval twice."""
    call = _total(spans, "compute_call")
    disp = _total(spans, "subgraph_dispatch")
    kids = sum(_total(spans, k) for k in phases.PHASE_CHILDREN["subgraph_dispatch"])
    combined = (call - disp) + (disp - kids)
    return {"verdict": "OK",
            "outer": {"residual_us": combined, "denominator_us": call,
                      "share": combined / call if call else 0.0,
                      "whole": "compute_call", "subtracted": ["subgraph_dispatch"]},
            "inner": {"residual_us": combined, "denominator_us": call,
                      "share": combined / call if call else 0.0,
                      "whole": "subgraph_dispatch", "subtracted": []},
            "invariant": "", "levels": list(phases.RESIDUAL_LEVELS)}


def _mutant_one_denominator(spans):
    """Both shares taken against the callback — the inner share is then not a share of the
    thing it was subtracted from."""
    r = phases.host_residuals(spans)
    call = r["outer"]["denominator_us"]
    r["inner"] = {**r["inner"], "denominator_us": call,
                  "share": (r["inner"]["residual_us"] / call) if call else None}
    return r


def _mutant_saturates(spans):
    """`max(x, 0)` instead of refusing: reports a fully accounted dispatch precisely when
    the instrument is most wrong."""
    call = _total(spans, "compute_call")
    disp = _total(spans, "subgraph_dispatch")
    kids = sum(_total(spans, k) for k in phases.PHASE_CHILDREN["subgraph_dispatch"])
    return {"verdict": "OK",
            "outer": {"residual_us": max(call - disp, 0), "denominator_us": call,
                      "share": (max(call - disp, 0) / call) if call else 0.0,
                      "whole": "compute_call", "subtracted": ["subgraph_dispatch"]},
            "inner": {"residual_us": max(disp - kids, 0), "denominator_us": disp,
                      "share": (max(disp - kids, 0) / disp) if disp else 0.0,
                      "whole": "subgraph_dispatch",
                      "subtracted": list(phases.PHASE_CHILDREN["subgraph_dispatch"])},
            "invariant": "NEVER SUMMED", "levels": list(phases.RESIDUAL_LEVELS)}


def _mutant_zero_share_for_zero_denominator(spans):
    r = phases.host_residuals(spans)
    for level in ("outer", "inner"):
        if r.get(level) and r[level]["share"] is None:
            r[level] = {**r[level], "share": 0.0}
    return r


def _mutant_readback_under_record(spans):
    """The pre-#88 declaration: readback subtracted from `record` instead of from the
    dispatch, so the inner residual absorbs the whole readback."""
    kids = [k for k in phases.PHASE_CHILDREN["subgraph_dispatch"] if k != "readback"]
    call = _total(spans, "compute_call")
    disp = _total(spans, "subgraph_dispatch")
    used = sum(_total(spans, k) for k in kids)
    return {"verdict": "OK",
            "outer": {"residual_us": call - disp, "denominator_us": call,
                      "share": ((call - disp) / call) if call else None,
                      "whole": "compute_call", "subtracted": ["subgraph_dispatch"]},
            "inner": {"residual_us": disp - used, "denominator_us": disp,
                      "share": ((disp - used) / disp) if disp else None,
                      "whole": "subgraph_dispatch", "subtracted": kids},
            "invariant": "NEVER SUMMED", "levels": list(phases.RESIDUAL_LEVELS)}


def _mutant_absent_brackets_read_as_zero(spans):
    call = _total(spans, "compute_call")
    disp = _total(spans, "subgraph_dispatch")
    kids = sum(_total(spans, k) for k in phases.PHASE_CHILDREN["subgraph_dispatch"])
    return {"verdict": "OK",
            "outer": {"residual_us": call - disp, "denominator_us": call, "share": 0.0,
                      "whole": "compute_call", "subtracted": ["subgraph_dispatch"]},
            "inner": {"residual_us": disp - kids, "denominator_us": disp, "share": 0.0,
                      "whole": "subgraph_dispatch",
                      "subtracted": list(phases.PHASE_CHILDREN["subgraph_dispatch"])},
            "invariant": "NEVER SUMMED", "levels": list(phases.RESIDUAL_LEVELS)}


_RESIDUAL_MUTANTS = {
    "sums_the_two_residuals": _mutant_sums_the_two_residuals,
    "one_denominator_for_both": _mutant_one_denominator,
    "saturates_instead_of_refusing": _mutant_saturates,
    "zero_share_for_zero_denominator": _mutant_zero_share_for_zero_denominator,
    "readback_subtracted_under_record": _mutant_readback_under_record,
    "absent_brackets_read_as_zero": _mutant_absent_brackets_read_as_zero,
}


def _residual_protocol(impl):
    """The properties issue #88 requires, applied to any implementation.

    Raises AssertionError on the first one violated. The real `host_residuals` must pass;
    every mutant must fail.
    """
    base = impl(_spans_of(_call88()))
    assert base["verdict"] == "OK", "a well-formed #88 trace must produce both residuals"

    # 1. Two subtractions over two denominators.
    assert base["outer"]["denominator_us"] == 1000
    assert base["inner"]["denominator_us"] == 800
    assert base["outer"]["residual_us"] == 200
    assert base["inner"]["residual_us"] == 50

    # 2. The invariant reaches the artifact.
    assert "NEVER SUMMED" in (base.get("invariant") or "")

    # 3. Each residual is a function of its own level only.
    wider = impl(_spans_of(_call88(call=1400)))
    assert wider["inner"]["residual_us"] == base["inner"]["residual_us"]
    assert wider["outer"]["residual_us"] == base["outer"]["residual_us"] + 400
    thinner = impl(_spans_of(_call88(record=300)))
    assert thinner["outer"]["residual_us"] == base["outer"]["residual_us"]
    assert thinner["inner"]["residual_us"] == base["inner"]["residual_us"] + 100

    # 4. `readback` is subtracted at the dispatch level, not under `record`.
    assert "readback" in base["inner"]["subtracted"]
    no_rb = impl(_spans_of(_call88(readback=0)))
    assert no_rb["inner"]["residual_us"] == base["inner"]["residual_us"] + 100

    # 5. A share of a zero denominator is undefined, never zero.
    empty = impl(_spans_of(_call88(call=0, dispatch=0, record=0, submit=0, fence=0,
                                   readback=0)))
    assert empty["outer"]["share"] is None
    assert empty["inner"]["share"] is None

    # 6. An impossible trace is refused, not clamped.
    for bad in (_call88(call=500, dispatch=800), _call88(dispatch=200, record=400)):
        try:
            impl(_spans_of(bad))
        except phases.TraceRefused:
            continue
        raise AssertionError("a child outlasting its parent must be refused, not clamped")

    # 7. A build with no brackets is UNAVAILABLE, not a residual of zero.
    absent = impl(_spans_of(_run(1, [1])))
    assert absent["verdict"] == "UNAVAILABLE"


def test_the_real_residual_instrument_passes_the_protocol():
    _residual_protocol(phases.host_residuals)


@pytest.mark.parametrize("mutant", sorted(_RESIDUAL_MUTANTS))
def test_a_defective_residual_implementation_is_caught_by_this_protocol(mutant):
    with pytest.raises((AssertionError, KeyError, TypeError, ZeroDivisionError)):
        _residual_protocol(_RESIDUAL_MUTANTS[mutant])


def test_the_bracket_set_propagates_a_refusal_rather_than_swallowing_it():
    """`outer_bracket_names` decides which spans are excluded from every sum. On a trace
    whose nesting is unrecoverable it must refuse with the rest of the module, not return
    an empty set — an empty bracket set silently re-promotes the brackets to siblings and
    double-counts the whole dispatch."""
    ev = [_decl("subgraph_dispatch", 1000, 100, parent="not_a_phase_this_build_knows")]
    with pytest.raises(phases.TraceRefused):
        phases.outer_bracket_names(phases.phase_spans(ev))


def test_the_bracket_set_is_empty_for_a_build_that_emits_no_brackets():
    """Accept polarity. Absence of the instrument is not a refusal."""
    assert phases.outer_bracket_names(phases.phase_spans(_run(1, [1]))) == set()
    assert phases.outer_bracket_names(phases.phase_spans(_call88())) == {
        "compute_call", "subgraph_dispatch"}
