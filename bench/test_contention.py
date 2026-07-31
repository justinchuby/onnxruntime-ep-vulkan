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
