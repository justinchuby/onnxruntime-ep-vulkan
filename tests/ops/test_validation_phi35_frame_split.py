"""Always-on, device-free polarity screen for the criterion-3(a) Phi-3.5 frame splitter.

`split_frame` and `_gate_attribution` are instruments: they decide whether a validation
message counted, and whether a transcript is allowed to be read as a validation result at
all.  An instrument that has only ever run against a clean transcript — where it prints the
answer everyone expects — is the shape this project keeps finding, and Tank's census records
it as `unfalsified`.  Both polarities of both are exercised here, on synthetic strings, with
no device and no model.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _verdict  # noqa: E402
import probe_validation_phi35 as probe  # noqa: E402
from test_validation_phi35 import _gate_attribution, device_name  # noqa: E402

B = probe.BOUNDARY


def _t(*lines: str) -> str:
    return "\n".join(lines)


# --- split_frame -----------------------------------------------------------

def test_split_frame_counts_a_vuid_before_the_boundary_as_in_frame() -> None:
    r = probe.split_frame(_t("VUID-vkCmdDispatch-none-02699 bad", B, "quiet"))
    assert r["boundary_seen"] is True
    assert len(r["in_frame_vuids"]) == 1
    assert r["teardown_vuids"] == []


def test_split_frame_counts_a_vuid_after_the_boundary_as_teardown() -> None:
    r = probe.split_frame(_t("quiet", B, "VUID-vkDestroyDevice-device-05137 leaked"))
    assert r["in_frame_vuids"] == []
    assert len(r["teardown_vuids"]) == 1


def test_split_frame_without_a_boundary_attributes_nothing_to_the_dispatch_window() -> None:
    """The dangerous polarity: a transcript with no boundary must not read as clean.

    If the boundary is missing the run did not reach the end of its inference.  Calling the
    VUIDs 'in frame' would attribute them to a dispatch that never completed; calling them
    nothing would hand back `in_frame_vuid_count: 0` for free.  They go to teardown, and
    `boundary_seen: False` is what the lane refuses on.
    """
    r = probe.split_frame(_t("VUID-a x", "VUID-b y"))
    assert r["boundary_seen"] is False
    assert r["in_frame_vuids"] == []
    assert len(r["teardown_vuids"]) == 2


def test_split_frame_counts_messenger_lines_only_inside_the_frame() -> None:
    r = probe.split_frame(
        _t("[Vulkan validation] hello", B, "[Vulkan validation] goodbye")
    )
    assert len(r["messenger_lines_in_frame"]) == 1


def test_split_frame_does_not_count_a_best_practices_line_as_a_vuid() -> None:
    """Liveness evidence must not contaminate the quantity being read."""
    r = probe.split_frame(
        _t("[Vulkan validation] BestPractices-deprecated-workgroup blah", B)
    )
    assert r["in_frame_vuids"] == []
    assert len(r["messenger_lines_in_frame"]) == 1


def test_split_frame_is_total_on_an_empty_transcript() -> None:
    r = probe.split_frame("")
    assert r["boundary_seen"] is False
    assert r["in_frame_vuids"] == [] and r["teardown_vuids"] == []


# --- _gate_attribution -----------------------------------------------------

_OK_COUNTERS = {
    "claimed_nodes": 355,
    "islands_offered": 1,
    "viable_islands_retained": 1,
    "ledger_gate": "MIXED",
    "ledger_hits": 355,
    "unproven_declines": 2,
    "device_losses": 0,
    "dispatches_executed": 355,
    "model_output_equivalence": "UNMEASURED",
    "alloc_device_frame_session_devices": "1=NVIDIA GeForce RTX 4060 Laptop GPU",
}


def _doc(**over) -> dict:
    d = {
        "child_exit_code": 0,
        "frame": {"boundary_seen": True, "in_frame_vuids": [], "messenger_lines_in_frame": []},
        "counters": dict(_OK_COUNTERS),
    }
    d.update(over)
    return d


def test_gate_attribution_accepts_a_healthy_run() -> None:
    assert _gate_attribution(_doc(), "clean")["claimed_nodes"] == 355


def test_gate_attribution_rejects_a_nonzero_exit() -> None:
    with pytest.raises(_verdict.InstrumentError) as e:
        _gate_attribution(_doc(child_exit_code=1), "clean")
    assert "exited 1" in str(e.value)


def test_gate_attribution_rejects_a_transcript_with_no_boundary() -> None:
    with pytest.raises(_verdict.InstrumentError):
        _gate_attribution(
            _doc(frame={"boundary_seen": False, "in_frame_vuids": [], "messenger_lines_in_frame": []}),
            "clean",
        )


def test_gate_attribution_rejects_a_run_with_no_counters() -> None:
    with pytest.raises(_verdict.InstrumentError):
        _gate_attribution(_doc(counters=None), "clean")


def test_gate_attribution_rejects_a_run_that_lost_the_device() -> None:
    """The fault exits 0 and writes a complete counters file, so exit code screens nothing."""
    c = dict(_OK_COUNTERS, device_losses=1)
    with pytest.raises(_verdict.InstrumentError) as e:
        _gate_attribution(_doc(counters=c), "clean")
    assert "device_losses" in str(e.value)


def test_gate_attribution_rejects_a_counters_file_that_never_recorded_device_losses() -> None:
    """Absent is not zero.  A counters file without the key cannot say the device survived."""
    c = {k: v for k, v in _OK_COUNTERS.items() if k != "device_losses"}
    with pytest.raises(_verdict.InstrumentError):
        _gate_attribution(_doc(counters=c), "clean")


# --- device_name -----------------------------------------------------------

def test_device_name_reads_the_name_the_run_reported() -> None:
    assert device_name({"alloc_device_frame_session_devices": "1=NVIDIA GeForce RTX 4060 Laptop GPU"}) == (
        "NVIDIA GeForce RTX 4060 Laptop GPU"
    )


def test_device_name_ignores_the_index_because_it_is_not_the_selector() -> None:
    """Selector 0 produced index 1 and selector 1 produced index 0 on this box."""
    assert device_name({"alloc_device_frame_session_devices": "0=Intel(R) Iris(R) Xe Graphics"}) == (
        "Intel(R) Iris(R) Xe Graphics"
    )


@pytest.mark.parametrize("raw", [None, "", "no-equals-sign", 0])
def test_device_name_refuses_a_run_that_did_not_name_its_device(raw) -> None:
    with pytest.raises(_verdict.InstrumentError):
        device_name({"alloc_device_frame_session_devices": raw})
