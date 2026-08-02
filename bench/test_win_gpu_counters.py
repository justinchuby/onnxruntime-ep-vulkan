"""The vendor-neutral tenancy producer, and the rule that half a companion cannot certify.

Every failure mode this file tests was **produced**, not imagined, while building the instrument:

* a LUID join through the wrong device ordering pointed the sampler at an adapter our workload
  never touched, and the record came back a clean ``SOLE_TENANT``;
* ancestry resolution for all 204 instances cost 21.2 s per round, so a 62 s window got three
  samples and the record came back clean about a run it had not watched;
* PDH caches its instance list per process, so a sampler that opened its query before a job
  started never saw that job at all — and the record came back clean.

Three different bugs, one shape: **when the instrument is wrong, it reads clean.** That is R9
amendment 5's question answered the wrong way, and no threshold on foreign engine time repairs
any of them. They are repaired at the source and backstopped by an interlock that requires the
record to carry positive evidence that it watched the right device
(``UNOBSERVABLE(self_not_witnessed)``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import device_companion as device_state  # noqa: E402
import win_gpu_counters as wgc  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
WINDOWS = sys.platform == "win32"

REAL_INSTANCE = ("\\\\DESKTOP-POS52H4\\GPU Engine(pid_30232_luid_0x00000000_0x00010AA0_phys_0_"
                 "eng_0_engtype_3D)\\Running Time")


# --------------------------------------------------------------------------------------------
# Parsing and the adapter join
# --------------------------------------------------------------------------------------------

def test_instance_names_parse_into_pid_adapter_and_engine():
    info = wgc.parse_instance(REAL_INSTANCE)
    assert info == {"pid": 30232, "luid": "0x00000000_0x00010aa0", "phys": 0, "eng": 0,
                    "engtype": "3d"}


def test_a_name_that_is_not_an_engine_instance_is_not_guessed_at():
    assert wgc.parse_instance("\\GPU Adapter Memory(luid_0x0_0x1)\\Dedicated Usage") is None


@pytest.mark.skipif(not WINDOWS, reason="the producer is Windows-only by construction")
def test_both_adapters_on_this_desk_are_present_in_the_counters():
    live = wgc.live_luids()
    intel = wgc.luid_for_adapter("Intel(R) Iris(R) Xe Graphics")
    nvidia = wgc.luid_for_adapter("NVIDIA GeForce RTX 4060 Laptop GPU")
    assert intel in live and nvidia in live
    assert intel != nvidia


@pytest.mark.skipif(not WINDOWS, reason="the producer is Windows-only by construction")
def test_an_unknown_adapter_is_an_instrument_error_and_never_a_guess():
    with pytest.raises(wgc.CounterError):
        wgc.luid_for_adapter("Some Vendor Unknown Graphics")


def test_an_ambiguous_adapter_name_refuses_rather_than_picking_a_side(monkeypatch):
    """Two identical adapters is the case that makes a description match unsound."""
    monkeypatch.setattr(wgc, "live_luids", lambda: {"a", "b"})
    monkeypatch.setattr(wgc, "adapters", lambda: [
        {"luid": "a", "description": "Twin GPU"}, {"luid": "b", "description": "Twin GPU"}])
    with pytest.raises(wgc.CounterError) as err:
        wgc.luid_for_adapter("Twin GPU")
    assert "worse than none" in str(err.value)


# --------------------------------------------------------------------------------------------
# The record. Built from a Sampler that never touches PDH, so these run anywhere.
# --------------------------------------------------------------------------------------------

def _sampler(tracks, *, own_root=4321, rounds=30, seconds=30.0, interval=1.0, gaps=None):
    s = wgc.Sampler("0x00000000_0x00010aa0", interval=interval)
    s.own_root = own_root
    s.rounds = rounds
    s.t_start = 1000.0
    s.t_end = 1000.0 + seconds
    s.tracks = {f"i{i}": t for i, t in enumerate(tracks)}
    s._read_times = gaps if gaps is not None else [1000.0 + i for i in range(rounds)]
    return s


def _track(pid, ticks, engtype="3d", ours=None):
    t = {"pid": pid, "luid": "0x00000000_0x00010aa0", "phys": 0, "eng": 0, "engtype": engtype,
         "first": 1_000_000, "last": 1_000_000 + ticks}
    if ours is not None:
        t["ours"] = ours
    return t


def test_our_own_work_alone_on_the_adapter_is_sole_tenant():
    rec = wgc.summarise(_sampler([_track(4321, int(2.0 * wgc.TICKS_PER_SECOND), ours=True)]))
    assert rec["verdict"] == wgc.TENANCY_SOLE
    assert rec["we_were_seen_on_this_adapter"] is True
    assert rec["own_gpu_seconds"] == {"4321": 2.0}


def test_a_window_our_own_work_was_never_seen_in_is_not_sole_tenant():
    """The failure that produced this verdict: the sampler was watching the other adapter.

    It is not a detection (nothing was found) and it is not a pass (nothing was watched), and it
    reads *cleanest of all* if allowed through — the less related the adapter, the emptier the
    record.
    """
    rec = wgc.summarise(_sampler([_track(999, int(0.5 * wgc.TICKS_PER_SECOND), ours=False)]))
    assert rec["verdict"] == wgc.TENANCY_UNWITNESSED
    assert rec["verdict"].startswith("UNOBSERVABLE")
    assert "not seen on" in rec["reason"]


def test_a_window_with_no_declared_owner_is_survey_grade_only():
    rec = wgc.summarise(_sampler([_track(999, 10_000_000)], own_root=None))
    assert rec["verdict"] == wgc.TENANCY_NO_OWNER


def test_foreign_engine_time_is_a_detection():
    rec = wgc.summarise(_sampler([
        _track(4321, int(2.0 * wgc.TICKS_PER_SECOND), ours=True),
        _track(9999, int(9.0 * wgc.TICKS_PER_SECOND), ours=False),
    ]))
    assert rec["verdict"].startswith("FOREIGN_GPU_WORK")
    assert rec["foreign_busy_fraction"] > wgc.FOREIGN_BUSY_FRACTION


def test_the_compositor_holding_the_igpu_is_named_as_structural(monkeypatch):
    """A verdict that can never be cleared should say so rather than look like bad luck."""
    monkeypatch.setattr(wgc, "_process_names", lambda pids: {str(p): "dwm.exe" for p in pids})
    rec = wgc.summarise(_sampler([
        _track(4321, int(2.0 * wgc.TICKS_PER_SECOND), ours=True),
        _track(777, int(5.0 * wgc.TICKS_PER_SECOND), ours=False),
    ]))
    assert rec["verdict"] == wgc.TENANCY_STRUCTURAL


def test_the_kernel_paging_on_our_behalf_is_not_a_stranger():
    """PID 4 accrues Copy-engine time for whoever faulted. Counting it foreign fires every run."""
    rec = wgc.summarise(_sampler([
        _track(4321, int(2.0 * wgc.TICKS_PER_SECOND), ours=True),
        _track(4, int(9.0 * wgc.TICKS_PER_SECOND), engtype="copy"),
    ]))
    assert rec["verdict"] == wgc.TENANCY_SOLE
    assert rec["kernel_gpu_seconds"] == {"4": 9.0}


def test_a_sampler_that_went_blind_is_an_instrument_error_not_a_quiet_adapter():
    """R13. 62 s of window, three samples: the record must not describe what it did not watch."""
    rec = wgc.summarise(_sampler([_track(4321, 10, ours=True)], rounds=3, seconds=62.0,
                                 gaps=[1000.0, 1002.0, 1023.0, 1062.0]))
    assert rec["verdict"] == wgc.ERROR_INSTRUMENT
    assert "blind" in rec["reason"]


def test_every_record_says_it_has_no_clock_and_carries_its_silence_set():
    for rec in (wgc.summarise(_sampler([_track(4321, 100, ours=True)])),
                wgc.summarise(_sampler([_track(999, 100)], own_root=None))):
        assert rec["silence_set"]
        assert any("no clock record" in s.lower() or "NO clock record" in s
                   for s in rec["silence_set"])
    rec = wgc.summarise(_sampler([_track(4321, 100, ours=True)]))
    assert rec["clock"]["verdict"] == wgc.UNOBSERVABLE


# --------------------------------------------------------------------------------------------
# The two-axis record, and what a half companion may and may not do
# --------------------------------------------------------------------------------------------

def _tenancy_only_record():
    return device_state.compose(
        {"verdict": "SOLE_TENANT", "producer": "bench/win_gpu_counters.py",
         "own_gpu_seconds": {"4321": 2.0}},
        device_state.empty_axis("no clock producer exists for this device on this platform"))


def _steady(median=53.4, n=40):
    return {"verdict": "STEADY", "median_ms": median, "n": n, "coverage": 1.0, "rsd": 0.004}


def test_tenancy_without_clock_gets_its_own_verdict():
    rec = _tenancy_only_record()
    assert rec["verdict"] == device_state.TENANCY_ONLY
    assert rec["verdict"] != "SOLE_TENANT"
    assert rec["verdict"] != device_state.UNOBSERVABLE
    assert rec["axes_present"] == {"tenancy": True, "clock": False}


def test_a_half_companion_never_certifies():
    out = device_state.certify(_steady(), _tenancy_only_record())
    assert out["verdict"] == device_state.UNCERTIFIED_PARTIAL
    assert out["quotable"] is False
    assert out["missing"] == ["sm_clock"]


def test_the_21_4x_wrong_run_would_pass_a_tenancy_only_companion_and_must_not():
    """The loophole, stated as a test.

    ``base_b`` was **verified sole tenant** and 21.4x wrong, with the project's second-best RSD,
    because the board never left 210 MHz. A companion that reports tenancy and no clock says
    nothing at all about that run — so if `UNCERTIFIED(partial_companion)` ever becomes quotable,
    this figure comes back with it.
    """
    out = device_state.certify(_steady(median=246.7354, n=46), _tenancy_only_record())
    assert out["quotable"] is False
    assert out["verdict"] != device_state.QUOTABLE
    assert "210 MHz" in out["detail"]


def test_a_half_companion_may_still_refuse():
    """The asymmetry that makes a partial record safe to have: it can subtract, never add."""
    rec = device_state.compose(
        {"verdict": "FOREIGN_GPU_WORK", "producer": "bench/win_gpu_counters.py",
         "foreign_gpu_seconds": {"9999": 9.0}, "process_names": {"9999": "someone_else.exe"}},
        device_state.empty_axis("no clock producer"))
    assert rec["verdict"] == "FOREIGN_GPU_WORK"
    out = device_state.certify(_steady(), rec)
    assert out["verdict"] == device_state.WITHHELD
    assert out["quotable"] is False
    assert "someone_else.exe" in out["detail"]


def test_a_record_claiming_a_full_companion_with_no_clock_series_is_still_an_error():
    """Distinct from the half companion: this one *claims* both axes and has one."""
    out = device_state.certify(_steady(), {"verdict": "SOLE_TENANT", "sm_mhz": {}})
    assert out["verdict"] == device_state.ERROR


def test_an_unwitnessed_tenancy_record_is_uncertified_and_not_a_detection():
    rec = device_state.compose({"verdict": wgc.TENANCY_UNWITNESSED, "reason": "wrong adapter"},
                               device_state.empty_axis("no clock producer"))
    assert rec["verdict"] == device_state.UNOBSERVABLE
    out = device_state.certify(_steady(), rec)
    assert out["verdict"] == device_state.UNCERTIFIED
    assert out["verdict"] != device_state.WITHHELD


# --------------------------------------------------------------------------------------------
# Portability: the record exists where the producers do not
# --------------------------------------------------------------------------------------------

def test_the_record_is_emitted_in_full_where_no_producer_exists_at_all():
    """Morpheus's amendment 1, the other direction: the obligation names a record, not a tool.

    A platform with neither producer emits the same shape, the same field names, and says which
    axes are empty — rather than omitting keys, which is indistinguishable from keys nobody
    thought to write.
    """
    rec = device_state.compose(None, None)
    assert rec["verdict"] == device_state.UNOBSERVABLE
    assert set(rec) >= {"verdict", "tenancy", "clock", "axes_present", "silence_set", "obligation"}
    assert rec["tenancy"]["verdict"] == device_state.NO_PRODUCER
    assert rec["clock"]["verdict"] == device_state.NO_PRODUCER
    assert device_state.certify(_steady(), rec)["quotable"] is False


def test_an_nvidia_record_recasts_into_the_same_two_axis_shape():
    path = RESULTS / "gpustate_soloA.json"
    if not path.exists():
        pytest.skip("gpustate_soloA.json not present")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rec = device_state.from_nvidia_record(doc.get("summary", doc))
    assert rec["verdict"] == "SOLE_TENANT"
    assert rec["axes_present"] == {"tenancy": True, "clock": True}
    assert device_state.certify(_steady(median=11.5252), rec)["verdict"] == device_state.QUOTABLE


# --------------------------------------------------------------------------------------------
# The capability claim, checked against the artifact that established it
# --------------------------------------------------------------------------------------------

def test_the_intel_capability_artifact_says_what_the_counters_witnessed():
    path = RESULTS / "wingpu-intel-dev1.json"
    if not path.exists():
        pytest.skip("wingpu-intel-dev1.json not present")
    doc = json.loads(path.read_text(encoding="utf-8"))
    cap = doc["capability"]
    assert doc["adapter"]["name"] == "Intel(R) Iris(R) Xe Graphics"
    assert cap["our_work_seen_on_target"] is True, "the instrument cannot see our own work"
    assert cap["negative_control_holds"] is True, "our work also appeared on another adapter"
    assert cap["clock_axis"] == "UNOBSERVABLE"
    assert all(v == 0 for v in cap["our_engine_seconds_on_other_adapters"].values())
