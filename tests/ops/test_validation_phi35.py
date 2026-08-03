"""Criterion 3(a): validation layers clean on a run that *genuinely executes* Phi-3.5.

Why this file exists when `test_validation.py` already reads a clean frame
-------------------------------------------------------------------------
Every clean validation reading this project holds was taken over
`add_f32_dispatches_end_to_end`: one Add and a handful of dispatches.  Before today the EP
executed zero nodes of Phi-3.5, then 33 fragmented islands; only since Switch's GQA fixes
does it claim a single large fused island.  A large fused island is where descriptor
lifetime, barrier scope and aliasing defects actually appear, and it had never been put
under the layer.  A clean reading over a graph the EP barely touched is not a reading about
the graph it now runs.

The problem this file had to solve, and how
-------------------------------------------
The EP's messenger (`rust/src/vk/instance.rs`) subscribes to ERROR|WARNING of type
VALIDATION.  **A clean run is therefore silent whether the callback is live or dead.**  So
`in_frame_vuid_count == 0` is worth nothing on its own — it is R12's 0-versus-UNOBSERVABLE
exactly, and the same shape as Niobe's KV-upload hold, where refusing to write `0`
preserved a real defect.

Neither existing falsifier reaches this frame:

* `ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION` fires only in `run_add_on_device`, a
  Rust test path the ORT session never takes, and its VUID is
  `VUID-vkDestroyDevice-device-05137` — teardown, out-of-frame by construction.
* `epctl --probe-validation` proves the messenger arms, but *in a different process*.
  R12 gen-4: for a test result, the frame is the binary that ran it.

The mechanism used instead is `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT`,
which makes the layer emit best-practices messages at WARNING severity — a severity the EP's
own messenger already subscribes to — in this process, during this inference, inside this
frame.  Best-practices messages are prefixed `BestPractices-`, not `VUID-`, so they prove the
callback was live without contaminating the VUID count.  The two arms are asserted to
*differ*: an in-frame liveness demonstration that produced the same reading as the clean arm
would be a control that cannot fire.

What was observed, on both devices, 2026-08-02
----------------------------------------------
=========================  ==============  ==================
reading                    clean arm       liveness arm
=========================  ==============  ==================
in_frame_vuid_count        0               0
messenger_lines_in_frame   0               14
claimed_nodes              355             355
viable_islands_retained    1               1
device_losses              0               0
=========================  ==============  ==================

So the clean reading is a *measured* 0 and not an UNOBSERVABLE one: the same messenger, in
the same frame, on the same run shape, demonstrably carried 14 validation messages.

    Update, Switch, 2026-08-02, later the same day: the liveness arm now carries **8**, not
    14.  Six of those 14 were ``vkCmdDispatch(): ... N bytes were never set with
    vkCmdPushConstants`` — the engine declared a 128-byte push-constant range on every
    pipeline layout and each kernel wrote only what it packed, leaving the remainder
    undefined.  The recorders now write the full declared range, zero-padded, and those six
    lines are gone (``tests/ops/test_push_constants_written.py``).  The reading above is
    otherwise unchanged and the assertions in this file are unaffected: they require the
    liveness count to be ``> 0`` and to differ from the clean arm, never to equal 14.  A
    control pinned to the exact number would have gone red on a fix.

Corrections to the brief this round, recorded rather than silently adopted
-------------------------------------------------------------------------
* `ledger_gate` reads **MIXED**, not `ALL-PROVEN` — `unproven_declines: 2`.
* `dispatches_executed` reads **355**, not 8875; the counter is per claimed node, not per
  `vkCmdDispatch`.  No 8875 figure is quoted anywhere here.
* Criterion 10's verdict today is attributed but DIVERGENT.  This file records that and does
  **not** wait on it: chaining an unblocked criterion to a blocked one is how criterion 2 was
  reopened.  Whether an attributed-DIVERGENT run satisfies the row is Morpheus's ruling.

A finding, routed not swallowed
-------------------------------
The liveness arm's best-practices output contains, from inside the dispatch window:

    vkCmdDispatch(): Pipeline uses a push constant range with offset 0 and size 128, but
    104 bytes were never set with vkCmdPushConstants.

That is not a VUID and does not fail criterion 3(a), but reading unwritten push-constant
bytes is undefined, and this is `rust/src/vk/` — Switch's.  It is asserted here only as
*evidence the messenger spoke*, and reported to him as a finding.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _verdict  # noqa: E402
import probe_validation_phi35 as probe  # noqa: E402

_RESULTS = pathlib.Path(__file__).resolve().parent.parent.parent / "bench" / "results"

#: The moment criterion 3(a)'s Phi-3.5 number is read at.  A count without its moment is a
#: count from an unidentified world.  This frame is a *different object* from the old one:
#: the old dispatch window held a handful of dispatches over one Add; this one holds a
#: single fused island of 355 claimed nodes.
CLEAN_READ_FRAME = (
    "dispatch window — from process start to the instant `sess.run()` returned on a real "
    "Phi-3.5 inference, with the instance, device, descriptor sets and command buffers all "
    "still live"
)

#: R12: never `0`.
TEARDOWN_UNOBSERVABLE = (
    "UNOBSERVABLE — the production device is leaked (Switch, 2026-08-01), so anything the "
    "layer says at teardown is about object lifetimes and not about the dispatch path.  A "
    "'0 validation errors at shutdown' gate is out-of-frame by construction on this build "
    "and is not read here in either direction."
)

_CRITERION_10_STATUS = (
    "attributed DIVERGENT on the unit (Morpheus's ruling, 2026-08-02).  Criterion 3(a) is "
    "recorded against exactly that and is not blocked on it."
)

pytestmark = pytest.mark.slow


def _model_present() -> bool:
    return probe.MODEL.is_file()


_requires_lane = pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB") or not _model_present(),
    reason="needs ONNXRUNTIME_VULKAN_EP_LIB and the Phi-3.5 model to execute a real graph",
)


def _selector() -> str:
    return os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset")


@pytest.fixture(scope="module")
def arms() -> dict[str, dict]:
    """Run both arms once per module, in one process, against one binary.

    Both arms come from `probe_validation_phi35.run_arm` — one definition of the case.  Two
    builders would be two definitions, and the arms could then differ for a reason nobody
    wrote down.
    """
    out: dict[str, dict] = {}
    for arm in ("clean", "liveness"):
        try:
            out[arm] = probe.run_arm(arm)
        except Exception as exc:  # noqa: BLE001 — an outage is never a detection
            raise _verdict.InstrumentError(
                f"[criterion 3(a) instrument failure] ERROR(instrument): the '{arm}' arm "
                f"could not be run at all: {type(exc).__name__}: {exc}.  This is not a "
                f"finding about validation."
            ) from exc
    return out


def device_name(counters: dict) -> str:
    """Return the physical device this run actually used, as the run itself reported it.

    `ONNXRUNTIME_EP_VULKAN_DEVICE` is a *request*.  "Both devices were covered" asserted
    from the env var is a claim about what was asked for, not about what ran — and the two
    are demonstrably not the identity here: selector 0 produced
    `1=NVIDIA GeForce RTX 4060 Laptop GPU` and selector 1 produced
    `0=Intel(R) Iris(R) Xe Graphics`, because the allocator's enumeration index is not the
    selector.  The device *name* is the identity that survives that.
    """
    raw = counters.get("alloc_device_frame_session_devices")
    if not isinstance(raw, str) or "=" not in raw:
        raise _verdict.InstrumentError(
            "[criterion 3(a) instrument failure] ERROR(instrument): the counters file did "
            f"not name a session device (alloc_device_frame_session_devices={raw!r}), so "
            "this reading cannot be attributed to a device at all."
        )
    return raw.split("=", 1)[1].strip()


def _gate_attribution(doc: dict, arm: str) -> dict:
    """Refuse to read a validation result off a run that did not execute the model.

    Each refusal below is ERROR(instrument), not FAIL: a run that died early looks exactly
    like a run with nothing to report, and reporting its silence as cleanliness is the
    failure mode this whole file is built against.
    """
    if doc["child_exit_code"] != 0:
        raise _verdict.InstrumentError(
            f"[criterion 3(a) instrument failure] ERROR(instrument): the '{arm}' child "
            f"exited {doc['child_exit_code']}; its transcript is not a reading about "
            f"validation."
        )
    if not doc["frame"]["boundary_seen"]:
        raise _verdict.InstrumentError(
            f"[criterion 3(a) instrument failure] ERROR(instrument): the '{arm}' child "
            f"never printed the dispatch-window boundary, so no frame could be split and "
            f"'in-frame' has no referent.  An unsplit transcript reads as 0 in-frame VUIDs "
            f"for free."
        )
    counters = doc.get("counters")
    if not counters:
        raise _verdict.InstrumentError(
            f"[criterion 3(a) instrument failure] ERROR(instrument): the '{arm}' arm wrote "
            f"no counters file, so nothing attributes this transcript to an execution."
        )
    device_name(counters)  # raises if the run cannot say which device it used
    # Switch, 2026-08-02: an intermittent VK_ERROR_DEVICE_LOST on this box exits 0 and
    # writes a complete counters file.  Checked before believing any clean reading.
    if counters.get("device_losses") != 0:
        raise _verdict.InstrumentError(
            f"[criterion 3(a) instrument failure] ERROR(instrument): the '{arm}' arm "
            f"recorded device_losses={counters.get('device_losses')!r}.  A run that lost "
            f"the device is not a run whose silence means anything."
        )
    return counters


@_requires_lane
def test_messenger_is_armed_before_any_reading_is_believed() -> None:
    """An unarmed machine reports ERROR(instrument), never green.

    Switch asked for this gate explicitly.  A skip is green, and a green criterion-3
    control on a machine that cannot run it is the exact silence the criterion exists to
    remove.
    """
    state, reason = _probe_frame()
    _verdict.require_validation_armed(state, reason)


def _probe_frame() -> tuple[str, str]:
    from test_validation import _cargo_env, _probe_validation_frame  # noqa: PLC0415

    return _probe_validation_frame(env=_cargo_env())


@_requires_lane
def test_the_run_criterion_3a_reads_from_actually_executed_the_model(arms) -> None:
    """The attribution gate: 355 claimed nodes in one island, no device loss.

    Asserted as a floor rather than an equality.  Pinning `claimed_nodes == 355` would turn
    a legitimate improvement into a red, and a false red fixed by softening the assertion it
    fired on converts a working control into decoration in one commit (Morpheus).  The exact
    value is recorded in the artifact instead.
    """
    for arm, doc in arms.items():
        counters = _gate_attribution(doc, arm)
        assert counters["claimed_nodes"] >= 300, (
            f"[{arm}] the EP claimed only {counters['claimed_nodes']} nodes; this is not "
            f"the large fused island criterion 3(a) is supposed to be read over"
        )
        assert counters["viable_islands_retained"] == 1, (
            f"[{arm}] expected one retained island, got "
            f"{counters['viable_islands_retained']!r} — a fragmented graph is a different "
            f"frame than the one this reading claims"
        )
        assert counters["dispatches_executed"] > 0, (
            f"[{arm}] dispatches_executed={counters['dispatches_executed']!r}: nothing ran"
        )


@_requires_lane
def test_the_messenger_could_have_spoken_inside_this_frame(arms) -> None:
    """The load-bearing control: without it, the clean 0 is UNOBSERVABLE.

    The liveness arm must carry validation messages *through the EP's own messenger*, in
    this process, before the dispatch-window boundary.  If it cannot, criterion 3(a)'s
    reading on this build is UNOBSERVABLE and must be recorded as such — not as 0.
    """
    live = arms["liveness"]
    clean = arms["clean"]
    _gate_attribution(live, "liveness")

    live_n = live["messenger_lines_in_frame_count"]
    assert live_n > 0, (
        "ERROR-shaped result: the liveness arm produced no [Vulkan validation] line inside "
        "the dispatch window, so the messenger cannot be shown to have been able to speak "
        "in this frame.  The clean arm's in_frame_vuid_count is therefore UNOBSERVABLE, "
        "not 0, and criterion 3(a) is NOT discharged by it."
    )
    # The arms must differ, or the 'demonstration' demonstrates nothing.
    assert live_n != clean["messenger_lines_in_frame_count"], (
        f"both arms carried {live_n} messenger lines in frame; a liveness control whose "
        f"reading does not move with its input is a falsifier that cannot fire (R10)"
    )
    assert any("Vulkan validation" in ln for ln in live["frame"]["messenger_lines_in_frame"])


@_requires_lane
def test_criterion_3a_no_validation_errors_in_the_dispatch_window(arms) -> None:
    """The reading itself, with its frame stated and its liveness already proven.

    Ordered after the liveness control on purpose: this assertion is only meaningful once
    the frame has been shown capable of carrying a message.
    """
    live = arms["liveness"]
    assert live["messenger_lines_in_frame_count"] > 0, (
        "refusing to report a clean reading without in-frame liveness"
    )

    record: dict = {
        "criterion": "3(a) — validation layers clean, messenger armed, real Phi-3.5 execution",
        "device_selector": _selector(),
        "frame_read_at": CLEAN_READ_FRAME,
        "teardown_window": TEARDOWN_UNOBSERVABLE,
        "criterion_10_status": _CRITERION_10_STATUS,
        "arms": {},
    }
    for arm, doc in arms.items():
        counters = _gate_attribution(doc, arm)
        record["arms"][arm] = {
            "in_frame_vuid_count": doc["in_frame_vuid_count"],
            "in_frame_vuids": doc["frame"]["in_frame_vuids"],
            "messenger_lines_in_frame_count": doc["messenger_lines_in_frame_count"],
            "counters": counters,
        }
    record["in_frame_liveness_demonstrated"] = (
        arms["liveness"]["messenger_lines_in_frame_count"] > 0
    )
    names = {arm: device_name(v["counters"]) for arm, v in record["arms"].items()}
    record["device_name"] = names["clean"]
    record["device_name_source"] = (
        "counters.alloc_device_frame_session_devices, read off this run — not the "
        "ONNXRUNTIME_EP_VULKAN_DEVICE selector, which is a request and whose number is not "
        "the allocator's enumeration index"
    )
    assert names["clean"] == names["liveness"], (
        f"the two arms ran on different devices ({names!r}); they are then not two readings "
        f"of one frame and the liveness demonstration says nothing about the clean arm"
    )
    record["liveness_mechanism"] = (
        "VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT — best-practices "
        "messages arrive at WARNING severity, which the EP's messenger already subscribes "
        "to, in this process and inside this frame.  They are prefixed BestPractices-, not "
        "VUID-, so they do not contaminate the count being read."
    )
    record["verdict"] = (
        "PASS — 0 in-frame VUID messages in a frame demonstrated to carry validation output"
        if arms["clean"]["in_frame_vuid_count"] == 0
        else "FAIL(condition=IN_FRAME_VUIDS)"
    )
    _RESULTS.mkdir(parents=True, exist_ok=True)
    path = _RESULTS / f"criterion3a_phi35-dev{_selector()}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[CRITERION 3a] wrote {path}", file=sys.stderr)

    for arm, doc in arms.items():
        vuids = doc["frame"]["in_frame_vuids"]
        assert not vuids, (
            f"[{arm}] criterion 3(a) FAIL(condition=IN_FRAME_VUIDS): the layer reported "
            f"inside the dispatch window:\n" + "\n".join(vuids[:5])
        )
