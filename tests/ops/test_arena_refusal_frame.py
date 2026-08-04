"""The Intel arena refusal, turned from a one-off observation into a checkable claim.

Round 37 recorded the Iris Xe as ``UNMEASURED(refused)`` on the arena lane, identically at
G=1 and G=4, and said so honestly: because the refusal was *identical* on both group sizes
it was never a grouping finding, and it was left standing as the one device-shaped hole in
the grouping result.

This round closed it, and the interesting part is that the first explanation was wrong.

The EP's own §6.5 warning fires when ``ONNXRUNTIME_EP_VULKAN_DEVICE`` is unset and ORT
binds a device other than the one ``ep.device_index`` asked for, and that warning *predicts*
``alloc_device_frame = SPLIT-DEVICE``.  Taking the prediction for a reading, the first pass
of this work classified the refusal as a split frame.  A one-arm-per-process re-run with a
fresh counters file says the frame is ``SHARED`` in **both** polarities -- the EP honours
the explicit request, so only one VkDevice ever runs.  A right headline had a wrong cause
bolted to it, and the counters file that would have caught it earlier was
process-cumulative across two arms and therefore attributable to neither.

What actually differs between the two polarities is the index ORT keys its allocator to,
and the two output binds that the alias is about.  These arms assert exactly that, in both
polarities, off artifacts written by two separate single-arm processes.

Regenerate with (from ``tests/ops``, Vulkan SDK on PATH,
``ONNXRUNTIME_VULKAN_EP_LIB`` set)::

    $env:ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY="1"
    $env:ONNXRUNTIME_EP_VULKAN_KV_ARENA="1"
    $env:ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS="1"
    # pinned polarity
    $env:ONNXRUNTIME_EP_VULKAN_DEVICE="1"
    $env:ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE=".../arena_refusal_pinned_arena_counters.json"
    python probe_arena_refusal.py --device-index 1 --arm arena --tag dev1-pinned-arena
    # unset polarity -- NEW PROCESS, NEW COUNTERS FILE, or the reading is unattributable
    Remove-Item Env:ONNXRUNTIME_EP_VULKAN_DEVICE
    $env:ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE=".../arena_refusal_noenv_arena_counters.json"
    python probe_arena_refusal.py --device-index 1 --arm arena --tag dev1-noenv-arena
"""

from __future__ import annotations

import json
import pathlib

import pytest

_RESULTS = pathlib.Path(__file__).resolve().parents[2] / "bench" / "results"
PINNED = _RESULTS / "arena_refusal-dev1-pinned-arena.json"
NOENV = _RESULTS / "arena_refusal-dev1-noenv-arena.json"


def _load(p: pathlib.Path) -> dict:
    if not p.exists():
        pytest.skip(f"{p.name} not present; regenerate with probe_arena_refusal.py")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pinned() -> dict:
    return _load(PINNED)


@pytest.fixture(scope="module")
def noenv() -> dict:
    return _load(NOENV)


def test_each_artifact_is_one_arm_in_one_process(pinned, noenv):
    """A cumulative counters file cannot attribute a counter to an arm.

    The first no-env run of this probe ran both arms into one counters file and the
    classifier read a frame declaration that could have belonged to either.  Both records
    must declare themselves attributable or nothing below means anything.
    """
    for rec in (pinned, noenv):
        assert rec["arm_run"] == "arena", rec["arm_run"]
        assert rec["counters_attributable"] is True, rec


def test_the_device_is_read_off_the_run_and_is_the_same_one_in_both_polarities(
    pinned, noenv
):
    """The selector is a request; the name is the observation.

    On this box selector 1 is the Iris Xe and selector 0 the RTX 4060 -- the reverse of
    the factory's own index order.  Comparing two polarities is only a comparison if both
    ran on the same silicon, and that has to come off the run.
    """
    assert pinned["device_name_from_run"] == noenv["device_name_from_run"]
    assert "Intel" in pinned["device_name_from_run"], pinned["device_name_from_run"]
    for rec in (pinned, noenv):
        assert rec["running_device_names"] == rec["device_name_from_run"], rec


def test_only_the_device_env_var_differs_between_the_two_records(pinned, noenv):
    assert pinned["ep_vulkan_device_env"] == "1"
    assert noenv["ep_vulkan_device_env"] is None
    assert pinned["device_selector_requested"] == noenv["device_selector_requested"] == 1


def test_the_refusal_is_present_in_one_polarity_and_absent_in_the_other(pinned, noenv):
    """The finding itself: the Iris Xe does not refuse the arena. A harness does."""
    assert pinned["arena_verdict"] == "AGREE", pinned["arena_verdict"]
    assert noenv["arena_verdict"] == "UNMEASURED(arena_refused)", noenv["arena_verdict"]


def test_the_agreeing_arm_actually_dispatched_and_the_refused_one_did_not(pinned, noenv):
    """Switch's screen, applied here: an arm with 0 EP dispatches measured nothing.

    The AGREE is only worth reading because the EP ran.  The refusal's 0 dispatches is not
    a failure of this screen -- it is the refusal, which by construction happens before any
    dispatch.
    """
    assert pinned["dispatches_executed"] >= 1, pinned
    assert noenv["dispatches_executed"] == 0, noenv


def test_the_frame_is_SHARED_in_BOTH_polarities_so_this_is_not_a_split_device(
    pinned, noenv
):
    """The arm that refutes the first explanation, kept because it was wrong once.

    The EP's §6.5 warning predicts ``SPLIT-DEVICE`` and it is easy to write the prediction
    down as a reading.  Both records say ``SHARED``.  If a future change makes the frame
    genuinely split, this arm fails and the cause label in the decision record must be
    revisited rather than quietly inherited.
    """
    for rec in (pinned, noenv):
        sides = rec["alloc_device_frame_sides"]
        assert "SHARED" in sides, sides
        assert "SPLIT" not in sides, sides
        assert rec["counters_arena"]["alloc_device_frame"] == "SHARED", rec


def test_the_discriminant_is_the_allocator_index_and_it_moves_with_the_env_var(
    pinned, noenv
):
    """The mechanism, stated as two numbers a hostile reader can diff.

    The session's own device list has exactly one entry in both polarities, at index 0.
    Pinned, ORT keys its allocator to 0 and the indices agree.  Unset, ORT keys it to 1 --
    the index of the device the *factory* advertised and ORT bound -- and the device-memory
    OrtValues the harness makes carry an allocator identity the session's spans are not
    registered under.
    """
    assert pinned["counters_arena"]["alloc_device_frame_allocator_index"] == "0"
    assert noenv["counters_arena"]["alloc_device_frame_allocator_index"] == "1"
    for rec in (pinned, noenv):
        sides = rec["counters_arena"]["alloc_device_frame_session_devices"]
        assert sides.count("=") == 1, sides
        assert sides.startswith("0="), sides


def test_the_two_outputs_the_alias_is_about_are_exactly_the_ones_declined(pinned, noenv):
    """Three outputs are offered a bind; the arena aliases two of them.

    Pinned: 3 attempted, 0 declined, 3 bound.  Unset: 3 attempted, 2 declined, 1 bound --
    and the 1 that binds is ``attn_out``, the output the arena does *not* alias.  This is
    why ``outputs_device_bound > 0`` must not be allowed to refute an output-side cause;
    the first version of the classifier did exactly that and mis-attributed this reading.
    """
    p, n = pinned["counters_arena"], noenv["counters_arena"]
    assert p["outputs_bind_attempted"] == n["outputs_bind_attempted"] == 3
    assert p["outputs_bind_declined"] == 0 and p["outputs_device_bound"] == 3
    assert n["outputs_bind_declined"] == 2 and n["outputs_device_bound"] == 1


def test_the_authority_grants_track_the_binds(pinned, noenv):
    assert pinned["counters_arena"]["alloc_device_authority_grants"] == 3
    assert noenv["counters_arena"]["alloc_device_authority_grants"] == 1


def test_the_classifier_names_the_cause_on_the_refusal_and_names_nothing_on_the_pass(
    pinned, noenv
):
    """A classification of a run that did not refuse is a classification of nothing."""
    assert noenv["classification"]["consistent_with"] == ["F_allocator_index_mismatch"]
    assert noenv["classification"]["separated"] is True
    assert pinned["classification"]["consistent_with"] == ["NOT_REFUSED"]


def test_the_remedy_is_named_in_the_record_not_only_in_a_report(noenv):
    """Whoever reads the artifact next must not have to find this file to act on it."""
    note = " ".join(noenv["classification"]["notes"])
    assert "ONNXRUNTIME_EP_VULKAN_DEVICE" in note
    assert "before the EP library is registered" in note


def test_this_says_nothing_about_grouping(pinned, noenv):
    """The scope fence.

    Round 37's refusal was identical at G=1 and G=4, which is why it was never a grouping
    finding.  These two records are both G=1 decode.  They explain the refusal; they do not
    re-prove grouping on the Intel arena, and the ``gqa_grouping_arena_dev1_recheck``
    artifact is where that lives.
    """
    for rec in (pinned, noenv):
        assert rec["probe"] == "arena_refusal"
        assert "group_size" not in rec
