"""Two-polarity falsifier for criterion 3(a)'s frame split.  No GPU, no cargo, no EP.

``test_validation.py::classify_clean_read_frame`` decides **which moment** criterion 3(a)'s
"zero validation errors" is read at.  It is the answer to Switch's caution of 2026-08-01:

  > the now-leaked production device makes any "0 validation errors at shutdown" gate
  > ``UNOBSERVABLE``.  If your clean-run evidence depends on a shutdown-time reading, it is
  > out-of-frame by construction — check which moment you are reading.

A frame rule is an instrument, and R10 applies to it: the falsifier is an artifact whose
content varies with its input.  So the splitter is driven here with synthesised
transcripts that differ in exactly one thing at a time, and it must disagree.

Every case is a transcript, which means every case is always in the lane.  The generalisable
form, learned on Guard D: *the falsifier for an instrument does not need the world the
instrument observes; it needs a document the instrument reads.*
"""

from __future__ import annotations

import test_validation as tv

_DEV_A = "Intel(R) Iris(R) Xe Graphics"
_DEV_B = "NVIDIA GeForce RTX 4060 Laptop GPU"


def _transcript(*, capable: int, passes: list[str], during: list[str], after: list[str]) -> str:
    lines = [
        "running 1 test",
        f"[INFO] add_f32_dispatches_end_to_end: {capable} capable device(s) — running on all",
    ]
    for i, dev in enumerate(passes):
        lines.extend(during[i : i + 1])
        lines.append(f"[PASS] run_add_on_device: 1024 f32 elements verified on {dev}")
    # anything in `during` beyond one per device still belongs before the last PASS
    if len(during) > len(passes):
        extra = during[len(passes):]
        last = lines.pop()
        lines.extend(extra)
        lines.append(last)
    lines.extend(after)
    lines.append("test result: ok. 1 passed; 0 failed")
    return "\n".join(lines)


def test_a_clean_two_device_run_reads_clean() -> None:
    """The accept polarity.  Two devices dispatched, nothing said, nothing counted."""
    out = _transcript(capable=2, passes=[_DEV_A, _DEV_B], during=[], after=[])
    frame = tv.classify_clean_read_frame(out)
    assert frame["capable_devices"] == 2
    assert frame["dispatched_devices"] == 2
    assert frame["device_labels"] == [_DEV_A, _DEV_B]
    assert frame["in_frame_vuids"] == []
    assert frame["teardown_vuids"] == []


def test_a_dispatch_window_vuid_is_in_frame() -> None:
    """The reject polarity that matters: a VUID during dispatch is criterion 3's finding."""
    bad = (
        "[vulkan-ep] ERROR: VUID-vkCmdDispatch-None-08600: SPIR-V uses descriptor "
        "[Set 0, Binding 4] but the binding was not declared"
    )
    out = _transcript(capable=2, passes=[_DEV_A, _DEV_B], during=[bad], after=[])
    frame = tv.classify_clean_read_frame(out)
    assert frame["in_frame_vuids"] == [bad], frame
    assert frame["teardown_vuids"] == []


def test_a_teardown_vuid_is_not_in_frame_and_does_not_read_as_zero() -> None:
    """Switch's caution, mechanised.

    A VUID emitted after the last verified dispatch is a statement about object lifetime,
    not about the kernel path.  It must not enter the dispatch-window count — and it must
    not vanish either, or the artifact would claim a cleanliness it did not check.
    """
    leak = (
        "[vulkan-ep] ERROR: VUID-vkDestroyInstance-instance-00629: OBJ ERROR : "
        "VkDevice object still alive at vkDestroyInstance"
    )
    out = _transcript(capable=2, passes=[_DEV_A, _DEV_B], during=[], after=[leak])
    frame = tv.classify_clean_read_frame(out)
    assert frame["in_frame_vuids"] == [], (
        "a teardown-time leak VUID was counted against the dispatch window; criterion "
        "3(a) would go red for a defect in a different frame with a different owner"
    )
    assert frame["teardown_vuids"] == [leak], frame
    assert "UNOBSERVABLE" in tv.TEARDOWN_UNOBSERVABLE


def test_the_split_actually_moves_the_same_line_between_frames() -> None:
    """The paired control, and the one that makes the three above mean anything.

    The two tests above use different *text*, so a splitter that keyed on the VUID string
    itself — or on nothing at all — could pass both.  Here the SAME line is placed on
    either side of the boundary and the classification must change.  If it does not, the
    boundary is not being read and the frame rule is decorative.
    """
    line = "[vulkan-ep] ERROR: VUID-vkQueueSubmit-pCommandBuffers-00065: identical text"
    before = _transcript(capable=1, passes=[_DEV_A], during=[line], after=[])
    after = _transcript(capable=1, passes=[_DEV_A], during=[], after=[line])

    f_before = tv.classify_clean_read_frame(before)
    f_after = tv.classify_clean_read_frame(after)

    assert f_before["in_frame_vuids"] == [line] and f_before["teardown_vuids"] == []
    assert f_after["teardown_vuids"] == [line] and f_after["in_frame_vuids"] == []
    assert f_before["in_frame_vuids"] != f_after["in_frame_vuids"], (
        "the same line classified identically on both sides of the boundary — the "
        "splitter is a constant"
    )


def test_a_run_with_no_dispatch_puts_everything_out_of_frame() -> None:
    """A transcript in which nothing dispatched must not report a clean dispatch window.

    Criterion 3's own ruling: a silent validation lane and a lane with nothing to validate
    are the same reading.  ``dispatched_devices == 0`` is what the caller refuses on, and
    a VUID in such a run must not be attributed to a dispatch that never happened.
    """
    noise = "[vulkan-ep] ERROR: VUID-VkInstanceCreateInfo-pNext-pNext: something at startup"
    out = "\n".join([
        "running 1 test",
        "[SKIP] add_f32_dispatches_end_to_end: no capable Vulkan device",
        noise,
    ])
    frame = tv.classify_clean_read_frame(out)
    assert frame["dispatched_devices"] == 0
    assert frame["in_frame_vuids"] == []
    assert frame["teardown_vuids"] == [noise]


def test_one_of_two_devices_is_not_a_two_device_reading() -> None:
    """The counter that criterion 3's 'on both devices' clause is checked with.

    A run that enumerated two capable devices and verified one is a partial reading, and
    the caller asserts equality on exactly these two numbers.  Screened here because on
    the dev box they are always equal, so the branch would otherwise never execute.
    """
    out = _transcript(capable=2, passes=[_DEV_A], during=[], after=[])
    frame = tv.classify_clean_read_frame(out)
    assert frame["capable_devices"] == 2
    assert frame["dispatched_devices"] == 1
    assert frame["capable_devices"] != frame["dispatched_devices"]


def test_capable_count_absent_is_none_not_zero() -> None:
    """R12 in the splitter itself: 'the transcript never said' is not 'it said zero'."""
    out = "running 1 test\ntest result: ok. 1 passed; 0 failed"
    frame = tv.classify_clean_read_frame(out)
    assert frame["capable_devices"] is None, (
        "a transcript that never reported a capable-device count returned a number; the "
        "caller would then compare two fabricated values and pass"
    )
