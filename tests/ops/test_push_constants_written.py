"""Every dispatch writes every byte of the push-constant range its layout declares.

What was wrong
--------------
`pipeline.rs` declares one push-constant range of `PUSH_CONSTANT_RANGE_BYTES` (128, the Vulkan
minimum guarantee for `maxPushConstantsSize`) for **every** pipeline layout in the engine, but
each kernel packed only the scalars it needed.  Trinity read the consequence in frame on both
devices, 2026-08-02, at WARNING severity through the EP's own messenger::

    vkCmdDispatch(): Pipeline uses a push constant range with offset 0 and size 128,
                     but 104 bytes were never set with vkCmdPushConstants

Six distinct lines, shortfalls ``{4, 20, 36, 72, 88, 104}`` — that is 128 minus the six distinct
pack sizes ``{124, 108, 92, 56, 40, 24}`` in use across 355 dispatches.

Why it mattered even though nothing was observably wrong
--------------------------------------------------------
It is not a VUID, so criterion 3(a) was never at risk.  It is still a defect: **unwritten
push-constant bytes are undefined, not zero.**  Nothing misbehaved because no shader read past
the block it declares — a property of the shaders as they happened to be written, not of the
API contract, and worth nothing the moment a shader grows a field.  The failure it was storing
up is the worst kind this project has: a new field reading undefined memory, producing plausible
numbers on one driver and different plausible numbers on another.

Why padding rather than shrinking the declared range
-----------------------------------------------------
The pipeline cache is keyed on ``(shader, spec_constants)`` with no push size in it.  Declaring
a per-kernel range would have to become part of that key, and a layout that disagreed with its
dispatch's push size is a hard error rather than a warning — trading a warning for a fault.
Padding cannot desynchronise: the range is a constant, and the recorders write all of it.

Why the zero in this file is a measurement and not a silence
-------------------------------------------------------------
The thing counted here is a *warning*, and a silent messenger is silent about warnings.  A bare
``push_constant_lines: 0`` is equally consistent with "every byte is written" and with "the
callback is dead" — and with "validation was never switched on".  Two independent conditions,
both required, both asserted below:

1. **The messenger spoke in this frame.**  Trinity's technique:
   ``VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT`` puts WARNING-severity
   ``BestPractices-`` lines on the EP's own messenger, in this process, inside the dispatch
   window.
2. **The detector has been observed in its positive state.**  ``--sensitivity`` recorded the
   same reading against the pre-fix binary, which is a *different* DLL hash.  A detector never
   seen to fire is a detector with no demonstrated positive state; this one fired with 6 lines
   on ``44D21A451D269F82`` and 0 on the fixed build.

What was observed, 2026-08-02
-----------------------------
=========================  =========================  =========================
reading                    pre-fix 44D21A451D269F82   fixed A8BAB570AB8BE38D
=========================  =========================  =========================
push_constant_lines        6                          0
shortfall_bytes_observed   {4,20,36,72,88,104}        {}
messenger_lines_in_frame   14                         8
in_frame_vuid_count        0                          0
dispatches_executed        355                        355
=========================  =========================  =========================

Both devices read 0 on the fixed build (NVIDIA RTX 4060 and Intel Iris Xe, device read off the
run, not off the selector).  Note the messenger line count moves 14 -> 8: the six lines that
disappeared are exactly the six this change removed, which is why the liveness count is asserted
``> 0`` here rather than pinned to a value.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "bench" / "results"))

# Resolved by identity through foundry_discovery (issue #11), never by a hardcoded cache
# path: Foundry Local's on-disk layout is versioned by its CLI's internal catalog revision
# (this machine's cache moved from "...-cuda-gpu" to "...-cuda-gpu-2" between when this test
# was first written and 2026-08-05) and a hardcoded path goes stale silently when that
# happens, with this lane skipping for what looks like "no GPU" rather than the real reason.
# PHI35_MODEL, if set, is still honored as an explicit direct override.
from test_phi35 import _PHI35_SPEC, _foundry_discovery  # noqa: E402

ARTIFACT = REPO / "bench" / "results" / "push_constants_written.json"
SENSITIVITY = REPO / "bench" / "results" / "push_constants_sensitivity.json"

_PHI35_DISCOVERY_ERROR: str | None = None
_override = os.environ.get("PHI35_MODEL")
if _override:
    _PHI35_MODEL_PATH: pathlib.Path | None = pathlib.Path(_override)
else:
    try:
        _PHI35_MODEL_PATH = _foundry_discovery.resolve_model_path(_PHI35_SPEC)
    except _foundry_discovery.FoundryDiscoveryError as _exc:
        _PHI35_MODEL_PATH = None
        _PHI35_DISCOVERY_ERROR = str(_exc)


def _model_present() -> bool:
    from probe_push_constants_written import PROBE_CHILD  # noqa: PLC0415, F401

    return _PHI35_MODEL_PATH is not None and _PHI35_MODEL_PATH.is_file()


_requires_lane = pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB") or not _model_present(),
    reason=(
        "needs ONNXRUNTIME_VULKAN_EP_LIB and the Phi-3.5 model to execute a real graph"
        if _PHI35_DISCOVERY_ERROR is None
        else f"Phi-3.5 model not resolvable: {_PHI35_DISCOVERY_ERROR}"
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# The scorer, tested without a GPU.  These run everywhere.
# ──────────────────────────────────────────────────────────────────────────────


def _reading(**over) -> dict:
    base = {
        "dll_sha256_16": "AAAAAAAAAAAAAAAA",
        "child_exit_code": 0,
        "dispatches_executed": 355,
        "device_losses": 0,
        "messenger_lines_in_frame": 8,
        "push_constant_lines": 0,
        "shortfall_bytes_observed": [],
    }
    base.update(over)
    return base


def _sens(**over) -> dict:
    base = {"dll_sha256_16": "BBBBBBBBBBBBBBBB", "push_constant_lines": 6,
            "shortfall_bytes_observed": [4, 20, 36, 72, 88, 104]}
    base.update(over)
    return base


def test_a_clean_reading_needs_both_liveness_and_a_proven_detector() -> None:
    from probe_push_constants_written import verdict  # noqa: PLC0415

    v, code, _ = verdict(_reading(), _sens())
    assert (v, code) == ("PUSH_CONSTANTS_FULLY_WRITTEN", 0)


def test_a_silent_messenger_makes_the_zero_unobservable_not_clean() -> None:
    """R12: a count of 0 taken through a channel never shown to carry anything is not a 0."""
    from probe_push_constants_written import verdict  # noqa: PLC0415

    v, code, _ = verdict(_reading(messenger_lines_in_frame=0), _sens())
    assert v == "UNOBSERVABLE"
    assert code == 2


def test_a_detector_never_seen_to_fire_cannot_certify_its_own_zero() -> None:
    from probe_push_constants_written import verdict  # noqa: PLC0415

    assert verdict(_reading(), None)[0] == "UNPROVEN_DETECTOR"
    assert verdict(_reading(), _sens(push_constant_lines=0))[0] == "UNPROVEN_DETECTOR"


def test_the_positive_control_may_not_be_the_subject() -> None:
    """A sensitivity record taken on this very binary would say the build both has and has not
    the defect.  That is not a control, it is a contradiction."""
    from probe_push_constants_written import verdict  # noqa: PLC0415

    v, code, _ = verdict(_reading(), _sens(dll_sha256_16="AAAAAAAAAAAAAAAA"))
    assert (v, code) == ("ERROR(instrument)", 2)


def test_a_run_that_dispatched_nothing_is_an_instrument_error_not_a_pass() -> None:
    """The ledger-decline shape: the EP fell back to CPU and the arm would report a clean
    sweep of a frame it never entered."""
    from probe_push_constants_written import verdict  # noqa: PLC0415

    for bad in ({"dispatches_executed": 0}, {"dispatches_executed": None}):
        v, code, _ = verdict(_reading(**bad), _sens())
        assert (v, code) == ("ERROR(instrument)", 2), bad


def test_a_truncated_or_device_lost_frame_is_refused() -> None:
    from probe_push_constants_written import verdict  # noqa: PLC0415

    assert verdict(_reading(child_exit_code=1), _sens())[0] == "ERROR(instrument)"
    assert verdict(_reading(device_losses=1), _sens())[0] == "ERROR(instrument)"


def test_the_defect_itself_outranks_a_missing_control() -> None:
    """If the bytes are unwritten we say so, whatever the controls look like — a finding is
    never withheld for want of a positive control that would only have confirmed it."""
    from probe_push_constants_written import verdict  # noqa: PLC0415

    v, code, why = verdict(
        _reading(push_constant_lines=6, shortfall_bytes_observed=[4, 104]), None
    )
    assert (v, code) == ("UNWRITTEN_PUSH_CONSTANT_BYTES", 1)
    assert "104" in " ".join(why)


def test_the_matcher_reads_the_layers_wording_not_ours() -> None:
    """The classifier that missed `The logical device has been lost` was tested only against
    strings we wrote ourselves.  This one is fed the layer's verbatim line."""
    from probe_push_constants_written import PUSH_NOT_SET, SHORTFALL  # noqa: PLC0415

    line = (
        "[vulkan-ep] WARN: [Vulkan validation] vkCmdDispatch(): Pipeline uses a push constant "
        "range with offset 0 and size 128, but 104 bytes were never set with "
        "vkCmdPushConstants."
    )
    assert PUSH_NOT_SET.search(line)
    assert int(SHORTFALL.search(line).group(1)) == 104
    assert not PUSH_NOT_SET.search(
        "[vulkan-ep] WARN: [Vulkan validation] vkCreateComputePipelines(): pCreateInfos[0]."
        "stage is using the SPIR-V Workgroup built-in which SPIR-V 1.6 deprecated."
    )


# ──────────────────────────────────────────────────────────────────────────────
# The reading, from the recorded artifacts.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="no recorded reading on this machine")
def test_the_recorded_reading_shows_no_unwritten_push_constant_bytes() -> None:
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["verdict"] == "PUSH_CONSTANTS_FULLY_WRITTEN", doc.get("why")
    assert doc["push_constant_lines"] == 0
    assert doc["messenger_lines_in_frame"] > 0, "the reading's own liveness condition"
    assert doc["dispatches_executed"] >= 300
    sens = doc["sensitivity_record"]
    assert sens and sens["push_constant_lines"] > 0
    assert sens["dll_sha256_16"] != doc["dll_sha256_16"]


@pytest.mark.skipif(not SENSITIVITY.is_file(), reason="no positive control on this machine")
def test_the_positive_control_recorded_the_defect_it_was_taken_for() -> None:
    """Recorded rather than asserted from memory: the pre-fix shortfalls are 128 minus each
    distinct pack size, which is what makes them evidence about *this* engine."""
    sens = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    assert sens["push_constant_lines"] == 6
    assert sens["shortfall_bytes_observed"] == [4, 20, 36, 72, 88, 104]
    assert all(0 < s < 128 for s in sens["shortfall_bytes_observed"])


@_requires_lane
@pytest.mark.slow
def test_live_run_writes_every_declared_push_constant_byte() -> None:
    """The same reading taken now, on this binary, rather than trusted from an artifact.

    ~5 minutes: one full Phi-3.5 inference under the validation layer.
    """
    from probe_push_constants_written import measure, verdict  # noqa: PLC0415

    sens = json.loads(SENSITIVITY.read_text(encoding="utf-8")) if SENSITIVITY.is_file() else None
    now = measure()
    v, code, why = verdict(now, sens)
    assert code == 0, f"{v}: {why}"
    assert now["push_constant_lines"] == 0
