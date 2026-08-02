"""The device-state companion, tested against the six artifacts that caused the retraction.

Every specimen here is a real paired ``(gpustate_*.json, gemv_*_dev0.json)`` from Switch's runs,
committed in ``bench/results/``. Two of them are runs whose ``gpu_steady_tail`` said ``STEADY``
with an excellent RSD and whose number was **21.4x wrong**. If this gate ever accepts those two
again, it has stopped working, and no amount of RSD tuning will tell us.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import device_companion as device_state  # noqa: E402
import phases  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

#: tag -> (gpustate artifact, what the paired gemv tail reported, is it the truth?)
SPECIMENS = {
    "soloA": ("gpustate_soloA.json", 11.5252, True),
    "after_coldboard": ("gpustate_after_coldboard.json", 11.5243, True),
    "contended": ("gpustate_contended.json", 11.7697, False),
    "contended3": ("gpustate_contended3.json", None, False),
    "baseline_certified": ("gpustate_baseline_certified.json", 246.7195, False),
    "base_b": ("gpustate_base_b.json", 246.7354, False),
}


def _state(tag: str) -> dict:
    path = RESULTS / SPECIMENS[tag][0]
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc.get("summary", doc)


def _steady(median: float, n: int = 33) -> dict:
    return {"verdict": "STEADY", "median_ms": median, "n": n, "coverage": 1.0, "rsd": 0.008}


@pytest.mark.parametrize("tag", ["soloA", "after_coldboard"])
def test_correctly_clocked_sole_tenant_runs_are_quotable(tag):
    median = SPECIMENS[tag][1]
    out = device_state.certify(_steady(median), _state(tag))
    assert out["verdict"] == device_state.QUOTABLE, out
    assert out["quotable"] is True


@pytest.mark.parametrize("tag", ["baseline_certified", "base_b"])
def test_idle_clock_regime_is_withheld_despite_an_excellent_rsd(tag):
    """The 21.4x-wrong runs. Their RSD was 0.12% -- better than the correct run's 0.81%.

    This is the failure that no threshold inside `gpu_steady_tail` can reach: the series is
    *uniformly* wrong, so it is *more* steady, not less.
    """
    state = _state(tag)
    assert state["verdict"] == "SOLE_TENANT", "tenancy alone would have passed this run"
    out = device_state.certify(_steady(SPECIMENS[tag][1], n=46), state)
    assert out["verdict"] == device_state.WITHHELD, out
    assert out["quotable"] is False
    assert "idle clock" in out["detail"]
    assert out["peak_sm_mhz"] == 210


@pytest.mark.parametrize("tag", ["contended", "contended3"])
def test_foreign_gpu_work_is_withheld(tag):
    state = _state(tag)
    assert state["verdict"] == "FOREIGN_GPU_WORK"
    out = device_state.certify(_steady(11.7697, n=8), state)
    assert out["verdict"] == device_state.WITHHELD, out
    assert out["quotable"] is False


def test_a_missing_companion_is_a_refusal_not_a_default_pass():
    out = device_state.certify(_steady(13.3432, n=43), None)
    assert out["verdict"] == device_state.UNCERTIFIED
    assert out["quotable"] is False


def test_a_non_nvidia_device_is_unobservable_and_that_is_not_a_pass():
    """R12. `nvidia-smi` cannot see an Intel board; the counter's event cannot occur in its frame.

    Measured, not hypothesised: on ``ep.device_index 1`` (Intel Iris Xe) ``nvidia-smi`` is present
    and exits 6. Classifying that as ``ERROR(instrument)`` would file a permanent property of the
    device as a transient fault of the harness.
    """
    state = device_state.Companion(vendor_is_nvidia=False).start().stop()
    assert state["verdict"] == "UNOBSERVABLE"
    out = device_state.certify(_steady(53.4), state)
    assert out["verdict"] == device_state.UNCERTIFIED
    assert out["quotable"] is False
    assert out["verdict"] != device_state.WITHHELD, "unobservable is not a detection"


def test_a_board_nvidia_smi_cannot_report_on_is_unobservable_not_an_error():
    state = device_state.Companion(board_index=97).start().stop()
    assert state["verdict"] in ("UNOBSERVABLE", "ERROR(instrument)")
    if state["verdict"] == "ERROR(instrument)":
        pytest.skip("nvidia-smi is not installed on this host")
    assert "outside the instrument's frame" in state["reason"]


def test_a_sampler_failure_is_error_instrument_and_never_a_detection():
    """R13. A broken companion says nothing about whether the board was contended."""
    state = {"verdict": "ERROR(instrument)", "reason": "nvidia-smi not found"}
    out = device_state.certify(_steady(11.5), state)
    assert out["verdict"] == device_state.ERROR
    assert out["quotable"] is False
    assert "nvidia-smi not found" in out["detail"]
    assert "never a detection" in out["detail"]


def test_a_clean_board_cannot_certify_a_tail_that_produced_no_number():
    out = device_state.certify({"verdict": "NO_STEADY_TAIL"}, _state("soloA"))
    assert out["verdict"] == device_state.UNCERTIFIED
    assert out["quotable"] is False


def test_a_state_record_with_no_clock_series_is_an_instrument_error():
    out = device_state.certify(_steady(11.5), {"verdict": "SOLE_TENANT", "sm_mhz": {}})
    assert out["verdict"] == device_state.ERROR
    assert out["quotable"] is False


def test_every_record_carries_its_own_silence_set():
    """R9's silence clause. A caveat that lives in the docs does not travel with the number."""
    for state in (None, _state("soloA"), {"verdict": "ERROR(instrument)", "reason": "x"}):
        out = device_state.certify(_steady(11.5), state)
        assert out["silence_set"], out
        assert any("UNOBSERVABLE" in s for s in out["silence_set"])


def test_gpu_steady_tail_is_born_uncertified():
    """The default must be a refusal. A tail that has never met the companion is not a number."""
    tail = phases.gpu_steady_tail([11500.0] * 40)
    assert tail["verdict"] == "STEADY"
    assert tail["certification"]["verdict"] == device_state.UNCERTIFIED
    assert tail["certification"]["quotable"] is False


def test_analyse_without_a_companion_leaves_the_tail_uncertified():
    report = phases.analyse([])
    assert report["device_state"]["verdict"] == "ABSENT"


def test_the_worker_pid_is_offered_while_the_child_is_still_alive():
    """Otherwise the companion cannot tell our own worker from a stranger.

    Wired the wrong way round -- PID handed over after ``communicate()`` returned -- the very
    first certified run reported ``FOREIGN_GPU_WORK`` in 93% of samples against one PID holding
    0.0 MiB, which was our own worker. That is ``ERROR(instrument)``, not a detection, and it is
    exactly the failure Switch's ``_is_ours`` docstring already warned about.
    """
    import phi35

    seen = {}

    def on_start(pid):
        seen["pid"] = pid
        seen["alive"] = psutil.pid_exists(pid)

    proc, err = phi35._run_worker(
        [sys.executable, "-c", "import time; time.sleep(0.6)"], dict(os.environ), on_start=on_start)
    assert err == ""
    assert proc is not None and proc.returncode == 0
    assert seen.get("alive") is True, "the PID arrived too late to classify any sample"
