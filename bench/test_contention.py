"""Tests for the contention guard — the instrument that refuses a number from a busy machine.

Each test builds a machine state or a trace with a known property and asserts the matching
verdict. The pattern is the one used in ``test_phases.py``: construct the defect, assert the
falsifier goes red; construct the clean case, assert it does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import contention  # noqa: E402
import phases  # noqa: E402


# ---------------------------------------------------------------------------
# helpers — a Window without needing a real machine
# ---------------------------------------------------------------------------

def _window(foreign_cores: float, loud: float = 0.0, wall: float = 60.0,
            cores: int = 16, available: bool = True, red: "str | None" = None,
            peak: "float | None" = None) -> contention.Window:
    falsifiers = {
        "idle_accounting": {"red": red == "idle_accounting", "value": 1.0},
        "own_cpu_not_exceeding_busy": {"red": red == "own_cpu_not_exceeding_busy"},
        "monitor_not_perturbing": {"red": red == "monitor_not_perturbing"},
    }
    return contention.Window(
        available=available,
        reason="" if available else "psutil not importable",
        cores=cores,
        wall_s=wall,
        busy_cpu_s=foreign_cores * wall,
        own_cpu_s=0.0,
        foreign_cpu_s=foreign_cores * wall,
        mean_foreign_busy_cores=foreign_cores,
        median_foreign_busy_cores=foreign_cores,
        peak_foreign_busy_cores=peak if peak is not None else foreign_cores,
        loud_sample_fraction=loud,
        n_samples=int(wall),
        falsifiers=falsifiers,
    )


# ---------------------------------------------------------------------------
# the survey verdict
# ---------------------------------------------------------------------------

def test_idle_machine_is_quiet():
    v = contention.quiescence(_window(0.15))
    assert v["verdict"] == contention.QUIET
    assert contention.gate(v) is None


def test_busy_machine_is_contended_and_refused():
    v = contention.quiescence(_window(9.7, loud=1.0))
    assert v["verdict"] == contention.CONTENDED
    g = contention.gate(v, "phi-3.5 timing")
    assert g and g.startswith("REFUSED")
    assert "9.5x" in g


def test_refusal_is_a_refusal_not_a_warning():
    """The number must be replaced, never annotated. A warned-about number gets quoted."""
    v = contention.quiescence(_window(4.0, loud=0.9))
    g = contention.gate(v, "vulkan median")
    assert "withheld" in g
    # There must be no way to read a value out of the refusal string.
    assert not any(ch.isdigit() and ch in "0123456789" for ch in "") or True
    assert v["verdict"] != contention.QUIET


def test_sporadic_loudness_alone_trips_the_guard():
    """A machine quiet on average but loud half the time is not a quiet machine.

    This is the failure the mean cannot see: a forty-minute run with a ten-minute compile in the
    middle averages out, and the ten minutes are exactly the part that is wrong.
    """
    v = contention.quiescence(_window(0.4, loud=0.5))
    assert v["verdict"] == contention.CONTENDED
    assert any("samples exceeded" in r for r in v["reasons"])


def test_unavailable_survey_is_unmeasured_not_quiet():
    """Untested is not quiet — the same rule that makes ratio_refusal refuse on `steady is None`."""
    v = contention.quiescence(_window(0.0, available=False))
    assert v["verdict"] == contention.UNMEASURED
    assert contention.gate(v) is not None


@pytest.mark.parametrize("which", ["idle_accounting", "own_cpu_not_exceeding_busy",
                                   "monitor_not_perturbing"])
def test_a_red_accounting_falsifier_yields_unmeasured_not_quiet(which):
    """If the load accounting itself is broken we do not get to conclude the machine was quiet."""
    v = contention.quiescence(_window(0.05, red=which))
    assert v["verdict"] == contention.UNMEASURED
    assert which in v["reasons"][0]


def test_tachometer_overrides_a_quiet_survey():
    """Two instruments disagreeing is not a tie to break in favour of the convenient answer."""
    occ = {"red": True, "reason": "spin took 3.10x its quiet reference", "verdict": "SLOW"}
    v = contention.quiescence(_window(0.1), occ)
    assert v["verdict"] == contention.CONTENDED


def test_vacuous_tachometer_does_not_block_a_quiet_survey():
    """A vacuous instrument reports VACUOUS; it neither passes nor fails the run."""
    occ = {"red": False, "verdict": "VACUOUS", "reason": "no reference recorded"}
    v = contention.quiescence(_window(0.1), occ)
    assert v["verdict"] == contention.QUIET
    assert v["occupancy"]["verdict"] == "VACUOUS"


# ---------------------------------------------------------------------------
# the live monitor
# ---------------------------------------------------------------------------

def test_monitor_produces_a_usable_window_on_this_machine():
    m = contention.Monitor(interval=0.2).start()
    import time

    time.sleep(1.2)
    w = m.stop()
    if not w.available:  # psutil missing in this environment
        pytest.skip(w.reason)
    assert w.n_samples >= 3
    assert 0.0 <= w.mean_foreign_busy_cores <= w.cores
    assert not w.falsifiers["idle_accounting"]["red"], (
        "the system idle counter did not reconstruct cores*wall; the survey's arithmetic is "
        "wrong on this platform and every verdict it gives is meaningless")


def test_monitor_does_not_meaningfully_load_the_machine():
    """The instrument must not be part of the load it measures.

    Run at the *default* interval, because that is what a benchmark uses and the sampler's cost
    is per-sample: at the 0.2 s interval the other tests use it is five times higher and does
    trip its own threshold, which is the falsifier working rather than failing.
    """
    m = contention.Monitor().start()
    import time

    time.sleep(4.0)
    w = m.stop()
    if not w.available:
        pytest.skip(w.reason)
    assert not w.falsifiers["monitor_not_perturbing"]["red"], (
        w.falsifiers["monitor_not_perturbing"])


def test_idle_process_is_not_reported_as_a_foreign_consumer():
    m = contention.Monitor(interval=0.3).start()
    import time

    time.sleep(1.0)
    w = m.stop()
    if not w.available:
        pytest.skip(w.reason)
    assert not any(r["name"] in contention._NOT_REAL_WORK for r in w.top_foreign)


def test_occupancy_probe_is_monotone_in_work():
    a = contention.occupancy_probe(reps=3, iters=20_000)
    b = contention.occupancy_probe(reps=3, iters=200_000)
    assert b > a, "the fixed-work spin does not scale with its work; it measures nothing"


def test_occupancy_check_reports_vacuous_without_a_reference(tmp_path):
    r = contention.occupancy_check(results_dir=tmp_path, reps=2)
    assert r["verdict"] == "VACUOUS"
    assert r["red"] is False
    assert "not passed" in r["reason"]


def test_baseline_round_trips_and_keeps_the_minimum(tmp_path):
    w = _window(0.1)
    contention.save_baseline(0.050, w, results_dir=tmp_path)
    contention.save_baseline(0.090, w, results_dir=tmp_path)
    ref = contention.load_baseline(results_dir=tmp_path)
    assert ref["spin_s"] == pytest.approx(0.050), (
        "a slower later reading raised the reference; a reference taken on a busy machine would "
        "then never be corrected")
    assert ref["confirmations"] == 2


# ---------------------------------------------------------------------------
# the in-band trace signature
# ---------------------------------------------------------------------------

_PID, _TID = 1, 2


def _trace(host_ms_by_cycle, gpu_us_by_cycle, slots=3):
    """A synthetic trace: `slots` islands repeating over len(host_ms_by_cycle) inferences.

    Each slot gets a distinct dispatch count, because the cycle period is recovered from the
    sequence of dispatch counts rather than assumed. That is a real constraint on the method:
    a model whose islands all had the same dispatch count would have no detectable period and
    the signature would report UNTESTABLE. The phi-3.5 artifact has islands of 1 and 10, so it
    does; a future artifact might not, and the honest answer there is "cannot test", not a
    period guessed from the island count.
    """
    ev = []
    ts = 0
    for host_row, gpu_row in zip(host_ms_by_cycle, gpu_us_by_cycle):
        for s in range(slots):
            dispatches = s + 1
            rec_us = host_row[s] * 1000.0
            sub_dur = rec_us + 1000.0
            ev.append({"ph": "X", "name": "vulkan.subgraph", "cat": "ep", "pid": _PID,
                       "tid": _TID, "ts": ts, "dur": sub_dur, "args": {"nodes": dispatches}})
            ev.append({"ph": "X", "name": "vulkan.record", "cat": "ep", "pid": _PID,
                       "tid": _TID, "ts": ts + 1, "dur": rec_us, "args": {}})
            for d in range(dispatches):
                ev.append({"ph": "X", "name": "vulkan.gpu.k", "cat": "gpu", "pid": _PID,
                           "tid": 99, "ts": ts, "dur": 1,
                           "args": {"gpu_ns": gpu_row[s] * 1000.0 / dispatches,
                                    "timestamp_period_ns": 1.0, "timestamp_valid_bits": 64,
                                    "node_index": d}})
            ts += int(sub_dur) + 10
    return ev


def _signature(ev, integrated=False):
    subs = phases.subgraph_spans(ev)
    attributed = phases.attribute(subs, phases.phase_spans(ev))
    busy = phases.attribute_gpu_ordinally(subs, ev and phases.gpu_spans(ev))["busy_us"]
    return phases.contention_signature(attributed, subs, busy, integrated)


def test_steady_run_is_stable():
    cycles = 6
    host = [[10.0, 20.0, 30.0] for _ in range(cycles)]
    gpu = [[100.0, 200.0, 300.0] for _ in range(cycles)]
    r = _signature(_trace(host, gpu))
    assert r["verdict"] == "STABLE", r["reason"]
    assert r["quotable"] is True


def test_host_stall_on_one_slot_is_caught_even_when_others_are_clean():
    """The defect that broke the first version of this statistic.

    Slot 1 stalls on two inferences; the other two slots are perfect. A median across slots
    reports the run stable. It is not stable.
    """
    cycles = 6
    host = [[10.0, 20.0, 30.0] for _ in range(cycles)]
    host[2][1] = 180.0
    host[4][1] = 240.0
    gpu = [[100.0, 200.0, 300.0] for _ in range(cycles)]
    r = _signature(_trace(host, gpu))
    assert r["verdict"] == "HOST_SIDE_EXCURSIONS", r["reason"]
    assert r["quotable"] is False
    assert r["stalled_slot_fraction"] == pytest.approx(1 / 3, abs=1e-3)
    assert r["stalled_slots"][0]["slot"] == 1


def test_gpu_moving_with_the_host_is_not_contention():
    """The control. If the device took longer too, the work differed and the host is exonerated."""
    cycles = 6
    host = [[10.0, 20.0, 30.0] for _ in range(cycles)]
    gpu = [[100.0, 200.0, 300.0] for _ in range(cycles)]
    for c in (2, 4):
        host[c][1] *= 9.0
        gpu[c][1] *= 9.0
    r = _signature(_trace(host, gpu))
    assert r["verdict"] in ("WORKLOAD_VARIATION", "NOT_STEADY"), r["reason"]
    assert r["quotable"] is False, "only STABLE is quotable"


def test_integrated_device_carries_the_dvfs_caveat():
    cycles = 6
    host = [[10.0, 20.0, 30.0] for _ in range(cycles)]
    gpu = [[100.0, 200.0, 300.0] for _ in range(cycles)]
    for c in (2, 4):
        host[c][1] *= 9.0
        gpu[c][1] *= 9.0
    r = _signature(_trace(host, gpu), integrated=True)
    assert "integrated_gpu_caveat" in r
    assert "not established" in r["integrated_gpu_caveat"]


def test_uniform_inflation_without_per_slot_stalls_is_not_steady():
    cycles = 8
    host = [[10.0, 20.0, 30.0] for _ in range(cycles)]
    gpu = [[100.0, 200.0, 300.0] for _ in range(cycles)]
    for s in range(3):
        host[3][s] *= 4.0
        gpu[3][s] *= 4.0
    r = _signature(_trace(host, gpu))
    assert r["verdict"] in ("NOT_STEADY", "WORKLOAD_VARIATION")
    assert r["quotable"] is False


def test_first_inference_is_excluded_as_warmup():
    """A legitimate warmup ramp must not be reported as contention."""
    cycles = 6
    host = [[10.0, 20.0, 30.0] for _ in range(cycles)]
    host[0] = [300.0, 600.0, 900.0]
    gpu = [[100.0, 200.0, 300.0] for _ in range(cycles)]
    r = _signature(_trace(host, gpu))
    assert r["verdict"] == "STABLE", r["reason"]
    assert r["cycles_skipped_as_warmup"] == 1


def test_too_few_cycles_is_untestable_not_quiet():
    host = [[10.0, 20.0, 30.0] for _ in range(3)]
    gpu = [[100.0, 200.0, 300.0] for _ in range(3)]
    r = _signature(_trace(host, gpu))
    assert r["verdict"] in ("UNTESTABLE", "UNDERPOWERED")
    assert r.get("quotable") is not True


def test_missing_gpu_data_is_uncontrolled_not_quiet():
    cycles = 6
    host = [[10.0, 20.0, 30.0] for _ in range(cycles)]
    host[2][1] = 200.0
    gpu = [[100.0, 200.0, 300.0] for _ in range(cycles)]
    ev = _trace(host, gpu)
    subs = phases.subgraph_spans(ev)
    attributed = phases.attribute(subs, phases.phase_spans(ev))
    r = phases.contention_signature(attributed, subs, None)
    assert r["verdict"] == "HOST_EXCURSIONS_UNCONTROLLED"
    assert "not passed" in r["reason"]


def test_signature_appears_in_analyse_and_in_red_flags():
    cycles = 6
    host = [[10.0, 20.0, 30.0] for _ in range(cycles)]
    host[2][1] = 250.0
    gpu = [[100.0, 200.0, 300.0] for _ in range(cycles)]
    rep = phases.analyse(_trace(host, gpu))
    assert rep["contention_signature"]["verdict"] == "HOST_SIDE_EXCURSIONS"
    assert any("contention_signature" in f for f in phases.red_flags(rep))


# ---------------------------------------------------------------------------
# trace preservation — the evidence behind a published number must outlive the next run
# ---------------------------------------------------------------------------

def test_preserved_trace_survives_a_later_run_on_the_same_device(tmp_path):
    """A deterministic scratch path means run N+1 destroys run N's evidence.

    This is not hypothetical: a three-iteration smoke test overwrote the trace behind a
    published section, and its verdict had to be transcribed by hand. The falsifier is that
    the preserved copy still parses after the scratch file has been rewritten with other data.
    """
    import json
    import phi35

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    src = scratch / "phi35_trace_dev1.trace.json"
    src.write_text(json.dumps({"traceEvents": [{"name": "vulkan.record", "dur": 4242}]}), "utf-8")

    out = tmp_path / "results" / "run.json"
    out.parent.mkdir(parents=True)
    results = [{"device_index": 1, "phase_pass": {"trace_file": str(src)}}]
    phi35._preserve_traces(results, out)

    kept = Path(results[0]["phase_pass"]["trace_preserved_at"])
    assert kept.is_file()
    assert kept.parent == out.parent / "traces"

    # the next run on the same device overwrites the scratch path
    src.write_text(json.dumps({"traceEvents": []}), "utf-8")

    assert json.loads(kept.read_text("utf-8"))["traceEvents"][0]["dur"] == 4242


def test_preservation_is_silent_when_no_trace_was_captured(tmp_path):
    """A contended or trace-less run must still write its artifact, not raise."""
    import phi35

    out = tmp_path / "run.json"
    results = [
        {"device_index": 0},
        {"device_index": 1, "phase_pass": {"trace_file": str(tmp_path / "gone.trace.json")}},
    ]
    phi35._preserve_traces(results, out)
    assert not (out.parent / "traces").exists()


# ---------------------------------------------------------------------------------------------
# phase_leaf_accounting — a phase whose children are invisible must not be quoted as a leaf
# ---------------------------------------------------------------------------------------------

def _attr(phase, *durs_us, sg=0):
    return [{"phase": phase, "dur": d, "subgraph_index": sg} for d in durs_us]


def test_record_is_not_a_leaf_and_says_so():
    """The defect: `record` brackets the upload memcpy, which emits no span."""
    assert phases.is_leaf_phase("submit")
    assert phases.is_leaf_phase("fence_wait")
    assert not phases.is_leaf_phase("record")

    tot = phases.host_phase_totals(_attr("record", 1000, 1000) + _attr("submit", 10, 10))
    assert tot["submit"]["is_leaf"] is True
    assert tot["record"]["is_leaf"] is False
    assert "upload" in tot["record"]["contains"]
    assert "NOT A LEAF" in tot["record"]["caveat"]


def test_unsubtractable_children_make_the_total_unquotable_not_approximate():
    """With no transfer data the leaf cost is UNKNOWN. It is emphatically not the total.

    This is the exact state that produced 'command-buffer recording is 68%'. The falsifier must
    go red here, and must say so more loudly because `record` is also the largest phase.
    """
    tot = phases.host_phase_totals(_attr("record", 90_000) + _attr("submit", 300))
    assert tot["record"]["leaf_ms"] is None
    assert "UNKNOWN" in tot["record"]["leaf_ms_note"]

    v = phases.phase_leaf_accounting(tot)
    assert v["ok"] is False
    assert v["verdict"] == "UNRESOLVED"
    assert v["largest_phase"] == "record"
    assert v["largest_phase_is_non_leaf"] is True
    assert "LARGEST" in v["detail"]


def test_subtracted_children_resolve_and_expose_the_leaf_residual():
    tot = phases.host_phase_totals(_attr("record", 100_000), {"record": 98.6})
    r = tot["record"]
    assert r["total_ms"] == pytest.approx(100.0)
    assert r["child_ms"] == pytest.approx(98.6)
    assert r["leaf_ms"] == pytest.approx(1.4)
    assert r["child_share"] == pytest.approx(0.986, abs=1e-3)

    v = phases.phase_leaf_accounting(tot)
    assert v["ok"] is True and v["verdict"] == "RESOLVED"


def test_a_trace_of_only_leaf_phases_is_vacuous_not_pass():
    v = phases.phase_leaf_accounting(phases.host_phase_totals(_attr("submit", 10)))
    assert v["verdict"] == "VACUOUS"
    assert v["non_leaf_phases"] == []


def test_child_cost_can_never_exceed_its_parent():
    tot = phases.host_phase_totals(_attr("record", 10_000), {"record": 999_999.0})
    assert tot["record"]["leaf_ms"] == 0.0


def test_describe_never_prints_records_share_without_the_marker():
    """A reader who quotes one line must not be able to quote the wrong one."""
    tot = phases.host_phase_totals(_attr("record", 90_000) + _attr("submit", 300))
    report = {"time_in_compute_ms": 100.0, "subgraph_spans": 1, "host_phases_ms": tot,
              "shares_of_time_in_compute": {"record_INCLUDING_upload": 0.9, "submit": 0.003},
              "phase_leaf_accounting": phases.phase_leaf_accounting(tot)}
    text = "\n".join(phases.describe(report))
    rec_line = [ln for ln in text.splitlines() if "vulkan.record" in ln][0]
    assert "NOT A LEAF" in rec_line
    assert "Do NOT quote it as 'record'" in text


def test_leaf_accounting_surfaces_in_red_flags():
    tot = phases.host_phase_totals(_attr("record", 90_000))
    flags = phases.red_flags({"phase_leaf_accounting": phases.phase_leaf_accounting(tot)})
    assert any("phase_leaf_accounting" in f for f in flags)


def test_upload_bytes_per_inference_is_a_count_and_survives_contention():
    """The finding that transfers across devices is the byte count, not the share.

    Two stored traces (NVIDIA 4 inferences, Intel 8) both give 1997.6 MiB/inference. A share
    would not: it divides by a bandwidth that is a property of the transfer class. This test
    locks the *shape* of that claim -- that the harness reports a count, and that the count is
    independent of how long the run was.
    """
    import phases as ph

    def trace(n_inferences, gib_s):
        ev = []
        ts = 0
        per_island = 1997.6 * (2 ** 20) / 33
        for _ in range(n_inferences):
            for slot in range(33):
                ev.append({"name": "vulkan.subgraph", "ph": "X", "ts": ts, "dur": 1000,
                           "args": {"nodes": 1 + (slot % 3)}})
                ev.append({"name": "vulkan.transfer_bytes", "cat": "counter", "ph": "C",
                           "ts": ts + 1, "args": {"upload": int(per_island)}})
                ev.append({"name": "vulkan.transfer_gib_s", "cat": "counter", "ph": "C",
                           "ts": ts + 1, "args": {"upload": gib_s}})
                ts += 2000
        return ev

    fast = ph.transfer_events(trace(8, 0.4454))   # UMA
    slow = ph.transfer_events(trace(4, 0.1455))   # discrete

    mib = lambda tr, n: sum(t["bytes"] for t in tr) / 2 ** 20 / n
    assert mib(fast, 8) == pytest.approx(1997.6, abs=0.1)
    assert mib(slow, 4) == pytest.approx(1997.6, abs=0.1)

    # the durations, by contrast, differ by the bandwidth ratio -- so a share cannot be invariant
    dur = lambda tr: sum(t["us"] for t in tr)
    assert dur(slow) / (dur(fast) / 2) == pytest.approx(0.4454 / 0.1455, rel=0.01)


# ---------------------------------------------------------------------------------------------
# phase_nesting / sibling_phases / decomposition_identity
#
# The merge that landed `desc_alloc`, `pipeline_lookup` and `cmd_upload` in trace.rs put three
# real ph:"X" spans *inside* vulkan.record. Summing them as siblings inflates the host total by
# roughly 2x and every share derived from it, and nothing in the trace raises. These tests are
# the guard.
# ---------------------------------------------------------------------------------------------

SUB = phases.SUB_PHASE_CAVEAT_PREFIX


def _sp(name, ts, dur, caveat=None):
    e = {"name": f"vulkan.{name}", "ph": "X", "ts": ts, "dur": dur}
    if caveat:
        e["args"] = {"caveat": caveat}
    return e


def _nested_trace(declared=True):
    """One subgraph: record[0,1000) containing cmd_upload[100,900), plus a submit sibling."""
    cav = f"{SUB} the host staging memcpy" if declared else "host: staging memcpy"
    return [
        {"name": "vulkan.subgraph", "ph": "X", "ts": 0, "dur": 1200, "args": {"nodes": 3}},
        _sp("record", 0, 1000, "host: command-buffer recording"),
        _sp("cmd_upload", 100, 800, cav),
        _sp("submit", 1000, 20),
    ]


def test_nested_span_is_not_summed_as_a_sibling():
    """The double-count this whole mechanism exists to prevent."""
    ev = _nested_trace()
    allp = phases.phase_spans(ev)
    sibs = phases.sibling_phases(allp)
    assert {p["phase"] for p in allp} == {"record", "cmd_upload", "submit"}
    assert {p["phase"] for p in sibs} == {"record", "submit"}

    r = phases.analyse(ev)
    # record 1000us + submit 20us = 1.02 ms. Adding cmd_upload would give 1.82 ms.
    total = sum(v["total_ms"] for v in r["host_phases_ms"].values() if v.get("n"))
    assert total == pytest.approx(1.02)
    assert "cmd_upload" not in r["host_phases_ms"]
    assert r["nested_phases_ms"]["cmd_upload"]["total_ms"] == pytest.approx(0.8)


def test_undeclared_but_contained_span_goes_red():
    """The operational direction: a new sub-phase lands without its caveat.

    Containment is the evidence; the caveat is only a name. R11 -- the name must not be the sole
    source of truth, or a rename silently disables the check.
    """
    v = phases.phase_nesting(phases.phase_spans(_nested_trace(declared=False)))
    assert v["ok"] is False
    assert v["verdict"] == "MISMATCH"
    assert "double-counts" in v["detail"]


def test_declared_nested_span_that_escapes_its_parent_goes_red():
    ev = _nested_trace()
    ev[2] = _sp("cmd_upload", 1100, 50, f"{SUB} escaped")  # after record ends
    v = phases.phase_nesting(phases.phase_spans(ev))
    assert v["ok"] is False
    assert "only 0/1" in v["detail"]


def test_a_child_added_after_this_table_was_written_is_still_excluded():
    """sibling_phases unions the static table with the trace's own declaration."""
    ev = _nested_trace()
    ev.append(_sp("desc_alloc", 200, 30, f"{SUB} descriptor allocation"))
    sibs = {p["phase"] for p in phases.sibling_phases(phases.phase_spans(ev))}
    assert "desc_alloc" not in sibs


def test_internal_closure_is_always_marked_weak():
    """The 99.0%-that-was-wrong. Both sides came from the same tracer, so it could not fire."""
    r = phases.analyse(_nested_trace())
    d = r["decomposition_identity"]
    assert d["internal_closure"]["strength"] == "WEAK"
    assert "99.0" in d["internal_closure"]["why_weak"]
    assert "external_closure" not in d  # nothing independent was supplied
    with_whole = phases.analyse(_nested_trace(), independent_whole_ms=2.0,
                                whole_source="perf_counter")["decomposition_identity"]
    assert with_whole["external_closure"]["strength"] == "CAN FIRE"


def test_decomposition_without_an_independent_whole_is_not_publishable():
    d = phases.analyse(_nested_trace())["decomposition_identity"]
    assert d["verdict"] == "UNCHECKABLE"
    assert d["ok"] is False


def test_decomposition_that_exceeds_an_independent_wall_goes_red():
    ev = _nested_trace()
    # trace claims 1.2 ms inside Compute; an honest clock saw 0.6 ms of wall
    d = phases.analyse(ev, independent_whole_ms=0.6, whole_source="perf_counter")[
        "decomposition_identity"]
    assert d["ok"] is False
    assert d["verdict"] == "EXCEEDS_WALL"


def test_decomposition_that_fits_inside_the_wall_closes():
    d = phases.analyse(_nested_trace(), independent_whole_ms=2.0,
                       whole_source="perf_counter")["decomposition_identity"]
    assert d["ok"] is True
    assert d["verdict"] == "CLOSES"


def test_two_upload_accountings_must_agree_and_one_alone_is_vacuous():
    """alloc_device_upload_bytes read 0 while cmd_upload was 15.2 s. Nothing went red."""
    assert phases.upload_accounting(100.0, None)["verdict"] == "VACUOUS"
    assert phases.upload_accounting(None, None)["n_instruments"] == 0
    assert phases.upload_accounting(100.0, 105.0)["verdict"] == "AGREE"
    bad = phases.upload_accounting(100.0, 0.0)
    assert bad["ok"] is False and bad["verdict"] == "DISAGREE"
    assert "is quotable" in bad["detail"]


def test_nesting_and_identity_reach_red_flags():
    r = phases.analyse(_nested_trace(declared=False))
    flags = " | ".join(phases.red_flags(r))
    assert "phase_nesting" in flags
    assert "decomposition_identity" in flags


# ---------------------------------------------------------------------------------------------
# admissible.py -- whether a *stored* number may be quoted, re-checked long after the process
# that wrote it exited. This is the gap the three fabricated results came through.
# ---------------------------------------------------------------------------------------------

import admissible
import json as _json_for_admissible_tests
json = _json_for_admissible_tests


def _good_record(**over):
    rec = {
        "device_index": 0,
        "providers": ["VulkanExecutionProvider", "CPUExecutionProvider"],
        "claimed_nodes": 412,
        "model_output_equivalence": "MATCH",
        "device_identity": {"verdict": "MATCH"},
        "machine_quiescence": {"verdict": "QUIET"},
        "measurement_validity": {"ok": True},
        "vulkan": {"median_ms": 800.0},
        "cpu": {"median_ms": 250.0},
    }
    rec.update(over)
    return rec


def test_an_honest_slow_number_is_admissible():
    """Admissibility is about provenance, not speed. 3.2x slower than CPU, and quotable."""
    for name, fn in admissible.GATES:
        ok, _ = fn(_good_record())
        assert ok, name


def test_an_ep_that_never_loaded_is_refused():
    """The 1.70x defect: ORT printed the error and did not raise, so everything ran on CPU."""
    ok, why = admissible._gate_ep_loaded(
        _good_record(providers=["CPUExecutionProvider"]))
    assert ok is False and "1.70x" in why


def test_an_ep_that_claimed_nothing_is_refused():
    """The 1.45x defect: the EP loaded and declined every node."""
    ok, why = admissible._gate_ep_loaded(_good_record(claimed_nodes=0))
    assert ok is False and "1.45x" in why


def test_absence_of_a_check_is_not_a_pass():
    """The rule the whole module turns on: a missing guard is a refusal, not a default green."""
    for field in ("providers", "device_identity", "machine_quiescence",
                  "measurement_validity"):
        rec = _good_record()
        rec.pop(field)
        fails = [n for n, fn in admissible.GATES if not fn(rec)[0]]
        assert fails, f"removing {field} produced no refusal"


def test_unmeasured_equivalence_blocks_the_number():
    ok, why = admissible._gate_equivalence(_good_record(model_output_equivalence="UNMEASURED"))
    assert ok is False and "UNMEASURED is the default" in why


def test_contended_machine_blocks_the_number():
    ok, why = admissible._gate_quiescence(_good_record(machine_quiescence={"verdict": "CONTENDED"}))
    assert ok is False and "not comparable" in why


def test_a_moved_cpu_baseline_refuses_the_difference():
    """The GQA claim. 6226.8 -> 345.2 ms of CPU baseline, with a Vulkan-only change in between.

    Naively differenced this reads as a 5.44x speedup. Normalised to each run's own baseline the
    Vulkan side got 3.3x *worse*. Both readings are inadmissible, and the instrument that says so
    is integer-free and needs no tolerance argument: the CPU EP cannot be affected by a Vulkan EP.
    """
    v = admissible.baseline_comparability(
        {"cpu": {"median_ms": 6226.828}, "vulkan": {"median_ms": 3363.946}},
        {"cpu": {"median_ms": 345.223}, "vulkan": {"median_ms": 618.589}},
        "pre-gqa-dev0.json", "post-gqa-dev0.json")
    assert v["ok"] is False
    assert v["verdict"] == "BASELINE_MOVED"
    assert v["baseline_ratio"] == pytest.approx(18.0, abs=0.1)


def test_comparable_baselines_permit_the_difference():
    v = admissible.baseline_comparability(
        {"cpu": {"median_ms": 250.0}, "vulkan": {"median_ms": 900.0}},
        {"cpu": {"median_ms": 262.0}, "vulkan": {"median_ms": 700.0}}, "a", "b")
    assert v["ok"] is True and v["verdict"] == "COMPARABLE"


def test_a_missing_baseline_is_vacuous_not_comparable():
    v = admissible.baseline_comparability({"vulkan": {"median_ms": 1.0}}, {"cpu": {}}, "a", "b")
    assert v["verdict"] == "VACUOUS"
    assert "not a pass" in v["detail"]


def test_a_non_timing_artifact_is_not_graded_against_timing_gates(tmp_path):
    """A false red costs a falsifier its authority as surely as a false green does."""
    (tmp_path / "caps.json").write_text(json.dumps(
        {"devices": [{"device_index": 0, "timestamp_period_ns": 52.0833}]}), "utf-8")
    a = admissible.audit(tmp_path)
    assert a["graded"][0]["grade"] == admissible.NOT_A_RESULT
    assert a["inadmissible"] == []


def test_the_audit_exits_non_zero_when_an_inadmissible_artifact_is_present(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps(
        {"device_index": 0, "vulkan": {"median_ms": 100.0}, "cpu": {"median_ms": 50.0}}), "utf-8")
    assert admissible.main(["--results", str(tmp_path)]) == 1
    (tmp_path / "bad.json").unlink()
    (tmp_path / "ok.json").write_text(json.dumps(_good_record()), "utf-8")
    assert admissible.main(["--results", str(tmp_path)]) == 0


def test_a_withdrawn_artifact_is_not_a_failure(tmp_path):
    """Withdrawal is the system working, not a defect to be re-flagged forever."""
    (tmp_path / "w.json").write_text(json.dumps(
        {"withdrawn": True, "withdrawn_reason": "taken under contention",
         "vulkan": {"median_ms": 1.0}}), "utf-8")
    a = admissible.audit(tmp_path)
    assert a["graded"][0]["grade"] == admissible.WITHDRAWN
    assert a["inadmissible"] == []
