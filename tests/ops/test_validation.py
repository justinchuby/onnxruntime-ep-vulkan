"""M0 criterion 3 — validation messenger: armed, plant fires in lane, clean after fix.

THREE OBLIGATIONS (DESIGN.md §10 M0 criterion 3, 2026-07-30):

  (a) ARMED — ``epctl --probe-validation`` exits 0.  Only when ARMED does "no errors"
      mean anything.  The EP-side ``VkDebugUtilsMessengerEXT`` was installed but not
      armed for the layer's entire first life (Switch's session-16 fix attached the
      instance-level messenger; the module-level dispatch-integration test proved it
      with ``enable_validation=true``).  Re-check still required because the fix landed
      after the last "no errors" reading.

  (b) PLANT IN LANE — ``ep_messenger_fires_for_planted_fence_leak`` runs in the lane.
      The test exists in ``vk/dispatch_integration.rs`` but is ``#[ignore]``d.
      Morpheus: "a control that must be opted into is not in the lane."  And: "a contested
      observation is not a control."  Getting it into the lane without removing the Rust
      ``#[ignore]`` (Switch owns that file) is done here: this module invokes the ignored
      test as a subprocess so it runs whenever ``pytest tests/ops`` runs.  The artifact —
      ``EP_VALIDATION_ERROR_COUNT after planted fence leak = N`` (N > 0) — travels with
      the test result.  Owner: Trinity (the wiring); Switch (the ``#[ignore]`` itself).

      The ``#[ignore]`` is now settled and STAYS: ``EP_VALIDATION_ERROR_COUNT`` is a
      process-wide static and the messenger is instance-scoped, so the assertion is only
      sound when the test owns the process.  Subprocess isolation — what this wrapper
      does — is the right way to run it, not the workaround.

      Gated (Switch's request, 2026-07-31) on the same predicate as
      ``Instance::validation_armed()``: an unarmed machine reports **ERROR(instrument)**,
      because with no messenger the counter's event cannot occur in its frame (R12) and a
      zero would be neither a pass nor a detection.  This is criterion 3's last open item.

  (c) CLEAN AFTER FIX — after Switch's binding-arity fix, running the dispatch integration
      test under validation produces no VUID errors.  The earlier reading ("no errors
      surfaced") was void because the messenger was not wired.  This takes it fresh.

The three tests below are distinct because they observe different properties:
  - Armed tests that the messenger can catch *anything*.
  - Plant tests that it catches *this specific violation* (wiring is not enough; routing must
    be correct — the messenger callback must be on the EP's instance, not an unrelated one).
  - Clean-after-fix tests that after the real bug was fixed, no new errors appear.

None implies the others.  All three are required for criterion 3.

COORDINATION NOTE (DESIGN.md §10 M0 criterion 3):
Switch owns ``rust/src/vk/`` — do not edit his files from this module.  The ``#[ignore]``
on ``ep_messenger_fires_for_planted_fence_leak`` is Switch's call.  Request below.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

import onnxruntime as ort
import pytest

import _models as m
import _registry_suppression
import _verdict

HERE = Path(__file__).parent
_REPO_ROOT = HERE.parent.parent
_CARGO_MANIFEST = _REPO_ROOT / "rust" / "Cargo.toml"
_RESULTS = _REPO_ROOT / "bench" / "results"


# ---------------------------------------------------------------------------
# CRITERION 3(a) — WHICH MOMENT THE CLEAN READ IS TAKEN AT
# ---------------------------------------------------------------------------
# Switch, 2026-08-01: *the now-leaked production device makes any "0 validation errors at
# shutdown" gate UNOBSERVABLE.  If your clean-run evidence depends on a shutdown-time
# reading, it is out-of-frame by construction — check which moment you are reading.*
#
# He is right and the first cut of this file was wrong in exactly that way: it grepped the
# WHOLE cargo transcript for `VUID-`, which includes whatever the layer says at
# `vkDestroyInstance` about objects that outlived their device.  That reading has two
# failure modes and they point opposite ways:
#
#   * a leak-time VUID turns a genuinely clean dispatch red, blaming the kernel path for a
#     teardown defect;
#   * and a reader who then "fixes" it by loosening the grep loses the dispatch-time
#     reading too.
#
# R12's answer is not a tighter grep.  It is that the shutdown window is a frame in which
# the event cannot be cleanly observed at all, so its count is `UNOBSERVABLE` — never `0`
# and never a failure — while the dispatch window is a frame in which it can, and that is
# where criterion 3(a)'s reading is taken.
#
# The split is the last per-device `[PASS] run_add_on_device` line: every dispatch has
# completed and been verified by then, and nothing after it is dispatch work.

#: The moment criterion 3(a)'s number is read at.  Recorded in the artifact, because a
#: count without its moment is a count from an unidentified world (criterion 12 (g)).
CLEAN_READ_FRAME = (
    "dispatch window — from process start to the last verified per-device dispatch, with "
    "the instance and every device still alive"
)

#: R12: never `0`.
TEARDOWN_UNOBSERVABLE = (
    "UNOBSERVABLE — the production device is leaked (Switch, 2026-08-01), so validation "
    "messages emitted at vkDestroyInstance are about object lifetimes and not about the "
    "dispatch path.  A `0 validation errors at shutdown` gate is out-of-frame by "
    "construction on this build and is not read here in either direction."
)

_VUID_RE = re.compile(r"VUID-")
_DEVICE_PASS_RE = re.compile(
    r"\[PASS\]\s+run_add_on_device:\s+\d+\s+f32 elements verified on (?P<dev>.+?)\s*$"
)
_CAPABLE_RE = re.compile(r"(?P<n>\d+) capable device\(s\)")


def classify_clean_read_frame(output: str) -> dict:
    """Split a dispatch-integration transcript into its two validation frames.

    Pure — it takes a transcript, not a machine — so both polarities are falsifiable in
    the always-on lane (``test_validation_frame_split.py``).  A frame rule that has only
    ever run against a clean transcript, where it prints the answer everyone expects, is
    the shape this project keeps finding.

    Returns ``capable_devices`` (``None`` when the transcript never said),
    ``dispatched_devices``, ``device_labels``, ``in_frame_vuids`` and ``teardown_vuids``.
    """
    lines = output.splitlines()

    capable = None
    labels: list[str] = []
    last_pass_index = -1
    for i, line in enumerate(lines):
        cap = _CAPABLE_RE.search(line)
        if cap is not None and capable is None:
            capable = int(cap.group("n"))
        dev = _DEVICE_PASS_RE.search(line)
        if dev is not None:
            labels.append(dev.group("dev").strip())
            last_pass_index = i

    # No dispatch ever completed: the whole transcript is out of the dispatch window.
    # Reporting the VUIDs as "in frame" here would attribute teardown noise to a dispatch
    # that never happened.
    boundary = last_pass_index if last_pass_index >= 0 else -1
    in_frame = [
        line for i, line in enumerate(lines) if _VUID_RE.search(line) and i <= boundary
    ]
    teardown = [
        line for i, line in enumerate(lines) if _VUID_RE.search(line) and i > boundary
    ]
    return {
        "capable_devices": capable,
        "dispatched_devices": len(labels),
        "device_labels": labels,
        "in_frame_vuids": in_frame,
        "teardown_vuids": teardown,
    }


def _write_criterion3_artifact(record: dict) -> Path:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    selector = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset")
    path = _RESULTS / f"criterion3a_clean_read-dev{selector}.json"
    import json

    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[CRITERION 3a] wrote {path}", file=sys.stderr)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _epctl_path() -> Path | None:
    """Return the path to the epctl binary, or None if not found."""
    ep_lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not ep_lib:
        return None
    ep_lib_path = Path(ep_lib).resolve()
    epctl_name = "epctl.exe" if sys.platform == "win32" else "epctl"
    candidate = ep_lib_path.parent / epctl_name
    return candidate if candidate.is_file() else None


def _cargo_env() -> dict[str, str]:
    """Return an environment dict that puts the Vulkan SDK on PATH (needed for cargo test)."""
    env = dict(os.environ)
    sdk = env.get("VULKAN_SDK", "")
    if sdk:
        bin_dir = str(Path(sdk) / "Bin")
        path = env.get("PATH", "")
        if bin_dir not in path:
            env["PATH"] = bin_dir + os.pathsep + path
    return env


def _probe_validation_frame(env: dict[str, str] | None = None) -> tuple[str, str]:
    """Return ``(state, reason)`` for the validation frame on this machine.

    The frame is what ``Instance::validation_armed()`` reports for the EP's own instance:
    is there a ``VkDebugUtilsMessengerEXT`` that could receive the layer's output at all?
    ``epctl --probe-validation`` answers it in-lane, from a process that builds exactly
    that instance.

    Raises :class:`_verdict.InstrumentError` if the probe cannot be run — an absent or
    hung probe is an outage, never a statement about validation.
    """
    epctl = _epctl_path()
    if epctl is None:
        raise _verdict.InstrumentError(
            "[criterion 3 instrument failure] ERROR(instrument): epctl was not found next "
            "to ONNXRUNTIME_VULKAN_EP_LIB, so the validation frame could not be probed. "
            "Build it: cargo build --release. This is not a finding about validation."
        )
    result = _verdict.run_subprocess_checked(
        [str(epctl), "--probe-validation"],
        what="criterion 3 validation probe",
        quiet_seconds=20,
        env=env,
    )
    return _verdict.classify_validation_probe(
        result.returncode, (result.stdout or "") + (result.stderr or "")
    )


# ---------------------------------------------------------------------------
# (a) Armed — epctl --probe-validation exits 0
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set — no epctl to run",
)
def test_validation_messenger_armed() -> None:
    """epctl --probe-validation must exit 0 (ARMED) on this machine.

    ARMED means: VK_LAYER_KHRONOS_validation is installed, was enabled, and a
    VkDebugUtilsMessengerEXT is receiving its output.

    Only in the ARMED state does 'no validation errors surfaced' mean anything —
    the same observation is produced when the layer is not loaded (Morpheus: "a silent
    instrument sitting on top of the day's worst defect for its entire life").

    Exit codes from epctl --probe-validation:
      0  ARMED — layer loaded, messenger live.  Only state that licenses a clean-run claim.
      1  fail  — (reserved for future use)
      3  absent — no answer, not a bad answer.  Install the Vulkan SDK.

    Neither "epctl was never built" nor "the validation layer is not installed" is a
    finding about validation, and neither may report green: both are lane environment,
    not a product defect, and this file's own `require_validation_armed` says exactly
    why a `pytest.skip()` here would be wrong -- "a skip is green, and a green criterion-3
    control on a machine that cannot run it is the exact silence the criterion was written
    to remove."  This test is the one place criterion 3(a) is actually decided, so it
    raises the same `_verdict.InstrumentError` every other test in this file already uses
    for the identical two conditions, rather than the `pytest.skip()` this test used to
    reach for instead (issue #1: a lane-inapplicable control must report an explicit
    instrument/unobservable outcome, never a success-shaped skip).
    """
    epctl = _epctl_path()
    if epctl is None:
        raise _verdict.InstrumentError(
            "[criterion 3a instrument failure] ERROR(instrument): epctl was not found "
            "next to ONNXRUNTIME_VULKAN_EP_LIB, so ARMED could not be probed. Build it: "
            "cargo build --release. This is not a finding about validation and it is not "
            "a pass."
        )

    # encoding="utf-8"/errors="replace": `text=True` alone decodes with the platform
    # locale, which is not UTF-8 on an English Windows runner; epctl's own diagnostics
    # (e.g. loader/layer manifest paths, "§7.2") are UTF-8, and a locale mis-decode would
    # silently corrupt them before this string is ever inspected (see issue #1).
    result = subprocess.run(
        [str(epctl), "--probe-validation"],
        capture_output=True, encoding="utf-8", errors="replace",
        timeout=_verdict.contention_tolerant_timeout(20),
    )
    output = result.stdout + result.stderr
    print(f"\n[CRITERION 3a] epctl --probe-validation output:\n{output}", file=sys.stderr)

    state, reason = _verdict.classify_validation_probe(result.returncode, output)
    if state != _verdict.VALIDATION_ARMED:
        raise _verdict.InstrumentError(
            f"[criterion 3a instrument failure] ERROR(instrument): validation is not "
            f"armed on this machine ({state} — {reason}).\n"
            "R12: a lane that cannot install or enable VK_LAYER_KHRONOS_validation has "
            "not observed 'no validation errors', it has observed nothing -- UNOBSERVABLE, "
            "never a pass and never silently skipped. Install the Vulkan SDK validation "
            "layers (VulkanSDK/runtime/...) or set VK_LAYER_PATH.\n"
            f"epctl output:\n{output}"
        )
    assert result.returncode == 0, (
        "epctl --probe-validation did not exit 0 (ARMED).\n"
        "The validation layer is installed but the debug messenger could not be created, "
        "or the layer reported an unexpected state.\n"
        f"epctl output:\n{output}"
    )
    assert "VALIDATION ARMED" in output or "ARMED" in output.upper(), (
        "epctl exited 0 but output does not confirm ARMED state. "
        f"Output:\n{output}"
    )
    print(
        "[CRITERION 3a] PASS: VK_LAYER_KHRONOS_validation is installed and messenger is live. "
        "'No errors surfaced' is now a meaningful claim.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# (b) Plant in lane — ep_messenger_fires_for_planted_fence_leak
#
# This invokes the Rust #[ignore]'d test as a subprocess so the plant runs in the
# Python lane (pytest tests/ops) without opting in separately.
#
# Switch note: removing the #[ignore] from the Rust test would move the positive control
# into the cargo-test lane as well.  That is Switch's decision.  This wiring ensures the
# control runs in the pytest lane regardless of the Rust-side #[ignore] state.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
@pytest.mark.skipif(
    not _CARGO_MANIFEST.is_file(),
    reason="rust/Cargo.toml not found — not in a source checkout",
)
@pytest.mark.slow
def test_ep_messenger_plant_fires_in_lane() -> None:
    """Criterion 3 POSITIVE CONTROL (plant): the EP's VkDebugUtilsMessengerEXT catches a planted violation.

    Runs ``ep_messenger_fires_for_planted_fence_leak`` — a ``#[ignore]``'d Rust test that:
      1. Creates an EP Instance with ``enable_validation=true`` (the same path production uses).
      2. Plants VUID-vkDestroyDevice-device-05137 by leaking a VkFence at vkDestroyDevice.
      3. Asserts EP_VALIDATION_ERROR_COUNT > 0 after the plant.

    Why this must be in the lane:
      Morpheus: "a control that must be opted into is not in the lane."
      The Rust test is #[ignore]'d so it does not run in standard cargo test.  Invoking it
      here puts the plant into the pytest lane so it runs on every ``pytest tests/ops`` without
      a separate opt-in.  The artifact (EP_VALIDATION_ERROR_COUNT printed to stdout) is the
      observation criterion 3 requires (§10.0.1 R10: "an artifact X produced whose content
      varies with X's input").

    Distinction from test_validation_messenger_armed:
      Armed proves the layer is installed.  This proves the EP's OWN instance has a live
      messenger — the production path, not a probe instance.  Armed + plant together close
      the "wired to a destination nobody read" failure mode.

    What I need from Switch (noted here per coordination protocol):
      - Remove ``#[ignore]`` from ``ep_messenger_fires_for_planted_fence_leak`` in
        ``rust/src/vk/dispatch_integration.rs`` so the plant also runs in the cargo-test lane.
      - After that, this test becomes redundant and can be demoted to a cross-check.

      RESOLVED 2026-07-31 (Switch): the ``#[ignore]`` STANDS, and the reason is sound —
      ``EP_VALIDATION_ERROR_COUNT`` is a process-wide static while the messenger is
      instance-scoped, so the assertion is only valid when the test owns the process.
      Subprocess isolation, which this wrapper already provides, is the correct way to
      run it.  In exchange he asked for the gate below.

    THE GATE (Switch's request, 2026-07-31; R12; R13).
      Before the plant is run at all, this test establishes that validation is ARMED on
      this machine.  If it is not, ``EP_VALIDATION_ERROR_COUNT`` cannot become non-zero
      for ANY state of the EP — the event cannot occur in the frame — so the control's
      ``assert count > 0`` would fail for a reason that says nothing about the messenger.
      An unarmed machine therefore reports **ERROR(instrument)**, not green and not a
      detection.  That closes criterion 3's last open item: the control can no longer be
      satisfied by a machine incapable of running it, in either direction.
    """
    state, reason = _probe_validation_frame()
    _verdict.require_validation_armed(state, reason)
    print(f"\n[CRITERION 3b] validation frame: {state} — {reason}", file=sys.stderr)

    env = _cargo_env()
    cmd = [
        "cargo", "test", "--lib", "--release",
        "--manifest-path", str(_CARGO_MANIFEST),
        "ep_messenger_fires_for_planted_fence_leak",
        "--", "--nocapture", "--ignored",
    ]
    print(f"[CRITERION 3b] running: {' '.join(cmd)}", file=sys.stderr)
    # cargo may rebuild before it runs; 300 s is the quiet-machine estimate and the
    # wrapper inflates it, because a build that is merely slow is not a messenger defect.
    result = _verdict.run_subprocess_checked(
        cmd,
        what="criterion 3b planted fence leak",
        quiet_seconds=300,
        env=env,
    )
    output = (result.stdout or "") + (result.stderr or "")
    print(f"[CRITERION 3b] cargo test output:\n{output[-3000:]}", file=sys.stderr)

    plant_state, plant_reason, count = _verdict.classify_plant_run(result.returncode, output)

    if plant_state == "ERROR":
        raise _verdict.InstrumentError(
            "[criterion 3b instrument failure] ERROR(instrument): "
            f"{plant_reason}\n"
            f"Validation was ARMED before the run ({reason}), so this outage is downstream "
            "of the frame check and is still not a finding about the messenger.\n"
            f"exit code: {result.returncode}\n"
            f"output (last 2000 chars):\n{output[-2000:]}",
            observed=count,
        )

    assert plant_state == "PASS", (
        "[CRITERION 3b] FAIL(condition): the EP's own VkDebugUtilsMessengerEXT did not "
        f"catch a planted violation.\n{plant_reason}\n"
        f"exit code: {result.returncode}\n"
        f"output (last 2000 chars):\n{output[-2000:]}"
    )
    print(
        f"[CRITERION 3b] PASS: {plant_reason}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# (c) Clean after fix — dispatch integration test shows zero VUID errors
#
# The earlier "no errors surfaced" reading was void (messenger not wired).
# This takes the reading fresh after Switch's binding-arity fix, with the messenger armed.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
@pytest.mark.skipif(
    not _CARGO_MANIFEST.is_file(),
    reason="rust/Cargo.toml not found",
)
@pytest.mark.slow
def test_validation_clean_after_binding_arity_fix() -> None:
    """Criterion 3 CLEAN READ: dispatch integration test shows no VUID errors post-fix.

    The binding-arity bug (Switch, 2026-07-30) caused:
      vkCreateComputePipelines(): SPIR-V uses descriptor [Set 0, Binding 4]
      but the binding was not declared in ... pSetLayouts[0]

    The validation layer fired the moment the messenger was armed — meaning the earlier
    "no errors surfaced" reading was a silent instrument on top of a live bug (R7 /
    DESIGN.md §10 M0 criterion 3 ruling).

    This test takes a FRESH reading after the fix.  It runs the dispatch integration test
    (``add_f32_dispatches_end_to_end``) with the messenger armed and greps the output for
    VUID-prefixed errors.  Zero VUID errors means the fix is complete and the messenger
    is now providing a meaningful "clean" signal.

    Device coverage: the Rust integration test runs on all capable devices (Intel Iris Xe
    and RTX 4060 on the local dev machine). Criterion 3's ruling requires both devices.

    R13/R12: a clean read is only a clean read inside a frame that could have been dirty.
    The armed gate runs first here for the same reason it runs in (b) — zero VUID lines
    from a machine with no messenger is not evidence of cleanliness, it is the absence of
    an instrument.
    """
    state, reason = _probe_validation_frame()
    _verdict.require_validation_armed(state, reason)
    print(f"\n[CRITERION 3c] validation frame: {state} — {reason}", file=sys.stderr)

    env = _cargo_env()
    cmd = [
        "cargo", "test", "--lib", "--release",
        "--manifest-path", str(_CARGO_MANIFEST),
        "add_f32_dispatches_end_to_end",
        "--", "--nocapture",
    ]
    print(f"[CRITERION 3c] running: {' '.join(cmd)}", file=sys.stderr)
    result = _verdict.run_subprocess_checked(
        cmd,
        what="criterion 3c clean validation read",
        quiet_seconds=300,
        env=env,
    )
    output = (result.stdout or "") + (result.stderr or "")
    print(f"[CRITERION 3c] cargo test output:\n{output[-3000:]}", file=sys.stderr)

    if result.returncode != 0 and (
        "no capable device" in output or "no shaders" in output.lower()
        or "[SKIP]" in output
    ):
        raise _verdict.InstrumentError(
            "[criterion 3c instrument failure] ERROR(instrument): "
            "add_f32_dispatches_end_to_end did not run (no ICD, no shaders, or no capable "
            "device), so no clean read was taken.  Validation was ARMED "
            f"({reason}), so the outage is downstream of the frame.  This is not a "
            "detection and it is not a pass.\n"
            f"output (last 2000 chars):\n{output[-2000:]}"
        )

    assert result.returncode == 0, (
        f"add_f32_dispatches_end_to_end failed (exit {result.returncode}).\n"
        f"Output:\n{output[-2000:]}"
    )

    frame = classify_clean_read_frame(output)

    # ── The frame could have been dirty.  Without this the read is vacuous. ───────────
    #
    # Criterion 3's own ruling: "a silent validation lane and a lane with nothing to
    # validate are the same reading."  A run that skipped every device emits zero VUID
    # lines and looks identical to a clean one, so the number of devices that actually
    # executed a dispatch is checked BEFORE the cleanliness is read.
    if frame["capable_devices"] is None or frame["dispatched_devices"] == 0:
        raise _verdict.InstrumentError(
            "[criterion 3c instrument failure] ERROR(instrument): the run produced no "
            "per-device dispatch confirmation, so there were no Vulkan calls for the "
            "layer to object to.  Zero VUID lines from this run is not a clean read, it "
            "is an empty one.\n"
            f"frame: {frame}\noutput (last 2000 chars):\n{output[-2000:]}"
        )
    assert frame["dispatched_devices"] == frame["capable_devices"], (
        f"Criterion 3(a) FAILS as a reading: {frame['capable_devices']} capable device(s) "
        f"were enumerated but only {frame['dispatched_devices']} completed a dispatch "
        f"({frame['device_labels']}).  A clean validation read that covers one of two "
        "devices is not a two-device reading, and criterion 3 asks for both.\n"
        f"output (last 2000 chars):\n{output[-2000:]}"
    )

    # ── The reading, at the moment it is taken. ──────────────────────────────────────
    assert not frame["in_frame_vuids"], (
        f"add_f32_dispatches_end_to_end produced {len(frame['in_frame_vuids'])} VUID "
        "validation error(s) inside the dispatch window, after Switch's binding-arity "
        "fix.  The criterion-3 clean reading is NOT clean.\n"
        "VUID lines:\n" + "\n".join(frame["in_frame_vuids"][:20]) + "\n"
        "This requires investigation — route to Switch."
    )

    record = {
        "criterion": "3(a)",
        "reading": "zero validation errors, messenger ARMED, post binding-arity fix",
        "frame_read_at": CLEAN_READ_FRAME,
        "validation_frame": state,
        "validation_frame_reason": reason,
        "capable_devices": frame["capable_devices"],
        "dispatched_devices": frame["dispatched_devices"],
        "device_labels": frame["device_labels"],
        "in_frame_vuid_count": 0,
        "teardown_window_vuids": TEARDOWN_UNOBSERVABLE,
        "teardown_window_lines_seen": len(frame["teardown_vuids"]),
        "device_selector_env": os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset"),
        "selector_note": (
            "add_f32_dispatches_end_to_end enumerates and runs on EVERY capable device in "
            "one process; ONNXRUNTIME_EP_VULKAN_DEVICE selects for the ORT path and does "
            "not partition this run.  Both devices are covered by the device_labels above, "
            "not by running this test twice."
        ),
        "no_duration_quoted": "§10.0 obligation 8 — no clock figure is taken or quoted here.",
    }
    _write_criterion3_artifact(record)

    print(
        "[CRITERION 3a/c] PASS: zero VUID errors in the DISPATCH WINDOW "
        f"({frame['dispatched_devices']}/{frame['capable_devices']} devices: "
        f"{frame['device_labels']}), messenger ARMED, binding-arity fix in place.\n"
        f"[CRITERION 3a/c] frame read at: {CLEAN_READ_FRAME}\n"
        f"[CRITERION 3a/c] teardown window: {TEARDOWN_UNOBSERVABLE} "
        f"({len(frame['teardown_vuids'])} line(s) seen there, NOT counted either way)",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# (d) THE GATE'S OWN FALSIFIER — on this machine, not on a synthesised string
#
# (b) and (c) now refuse to run on an unarmed machine.  That is a claim about a
# machine state this dev box does not have, so on its own it is a code reading:
# every run here is ARMED and the branch never executes.  R10's falsifier for
# "the gate is wired" is an artifact whose content varies with its input, so this
# test *removes the layer* from a real epctl process and requires the classifier
# to change its answer.
#
# The layer is removed the same way `rust/tests/validation_control.rs` does it —
# VK_LAYER_PATH pointed at a directory with no layer manifests — so no machine
# configuration is touched and the effect lasts one subprocess.
#
# Both polarities in one test, which is what makes it evidence: the armed probe
# must pass the gate and the unarmed one must not, and a gate that answered the
# same either way fails here regardless of which answer it picked.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set — no epctl to run",
)
def test_the_armed_gate_changes_its_answer_when_the_layer_is_removed() -> None:
    armed_state, armed_reason = _probe_validation_frame()
    print(f"\n[CRITERION 3d] real frame: {armed_state} — {armed_reason}", file=sys.stderr)

    # Three ways to take the layer away, tried in order, because they are neutralised by
    # different things and the ORDER is itself the reading.
    #
    #   1. VK_LAYER_PATH / VK_ADD_LAYER_PATH — how `rust/tests/validation_control.rs`
    #      does it.  The Vulkan loader documents BOTH as "ignored when running a Vulkan
    #      application with elevated privileges", the same caveat PLATFORMS.md §7.4.1
    #      records for VK_DRIVER_FILES.  On an elevated runner this arm cannot take.
    #   2. VK_LOADER_LAYERS_DISABLE — documented as a filter rather than a search path
    #      (loader 1.3.234+) with no elevation caveat in the loader's *markdown* table.
    #      PROVEN WRONG on 2026-08-06 by reading the loader's own source
    #      (`loader/loader_environment.c`, `parse_generic_filter_environment_var` →
    #      `loader_secure_getenv`): this goes through the identical `is_high_integrity()`
    #      gate as the search-path vars, so on a High-integrity runner it is dropped too.
    #      Kept because it still takes on a non-elevated dev box.
    #   3. registry_disable — flips the LunarG-registered validation layer's own value
    #      under HKLM\SOFTWARE\Khronos\Vulkan\ExplicitLayers to 1 (disabled). That key is
    #      scanned "regardless of elevation" (LoaderDriverInterface.md), so it is not
    #      gated by `is_high_integrity()` — the mechanism expected to take on the
    #      GitHub-hosted Windows runner.
    #
    # Verified on an unelevated GPU box on 2026-08-04: arm 1 alone takes here, so this
    # test PASSES on this machine and the CI red it was raised for does not reproduce on
    # a real device. Verified on the GitHub-hosted Windows runner, 2026-08-06 (this
    # file's own CI run): arms 1 and 2 both leave the gate ARMED — loader version 1.3.301
    # (modern, rules out an old-loader theory) — `registry_disable` is what actually
    # takes there.
    attempts: list[tuple[str, str, str]] = []
    stripped_state = stripped_reason = ""
    mechanism = "none"
    env_arms: "list[tuple[str, dict[str, str]]]" = [
        (
            "layer_search_path",
            {
                "VK_LAYER_PATH": str(_REPO_ROOT / "rust"),
                "VK_ADD_LAYER_PATH": str(_REPO_ROOT / "rust"),
            },
        ),
        (
            "loader_layer_filter",
            {
                "VK_LAYER_PATH": str(_REPO_ROOT / "rust"),
                "VK_ADD_LAYER_PATH": str(_REPO_ROOT / "rust"),
                "VK_LOADER_LAYERS_DISABLE": "*",
            },
        ),
    ]
    for mechanism_name, overrides in env_arms:
        no_layer_env = _cargo_env()
        no_layer_env.update(overrides)
        no_layer_env.pop("ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION", None)
        stripped_state, stripped_reason = _probe_validation_frame(no_layer_env)
        attempts.append((mechanism_name, stripped_state, stripped_reason))
        print(
            f"[CRITERION 3d] frame with {mechanism_name} applied: {stripped_state} — "
            f"{stripped_reason}",
            file=sys.stderr,
        )
        if stripped_state != _verdict.VALIDATION_ARMED:
            mechanism = mechanism_name
            break

    if stripped_state == _verdict.VALIDATION_ARMED:
        mechanism_name = "registry_disable"
        try:
            with _registry_suppression.suppress_validation_layer_registry():
                no_layer_env = _cargo_env()
                no_layer_env.pop("ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION", None)
                stripped_state, stripped_reason = _probe_validation_frame(no_layer_env)
            attempts.append((mechanism_name, stripped_state, stripped_reason))
            print(
                f"[CRITERION 3d] frame with {mechanism_name} applied: {stripped_state} — "
                f"{stripped_reason}",
                file=sys.stderr,
            )
            if stripped_state != _verdict.VALIDATION_ARMED:
                mechanism = mechanism_name
        except _registry_suppression.RegistryMechanismUnavailable as exc:
            attempts.append((mechanism_name, "mechanism_unavailable", str(exc)))
            print(
                f"[CRITERION 3d] {mechanism_name} unavailable: {exc}",
                file=sys.stderr,
            )

    assert stripped_state != _verdict.VALIDATION_PROBE_ERROR, (
        "removing the layer manifests must produce a *classified* unavailability, not a "
        "broken probe.\n"
        f"{stripped_reason}"
    )

    if stripped_state == _verdict.VALIDATION_ARMED:
        # Some loaders find the layer through registry entries VK_LAYER_PATH does not
        # override.  Then the simulation did not take, and this test has established
        # nothing — which is an instrument outage and must not be reported as a pass.
        # `rust/tests/validation_control.rs` accepts the same two outcomes for the same
        # reason; the difference is that this branch is not green.
        raise _verdict.InstrumentError(
            "[criterion 3d instrument failure] ERROR(instrument): the layer could not be "
            "taken away from a real epctl process by ANY mechanism, so the unarmed "
            "branch of the gate could not be exercised on this machine.\n"
            + "\n".join(f"  {name}: {state} — {why}" for name, state, why in attempts)
            + "\nThe gate's classification is still falsified without a machine in "
            "tests/ops/test_r13_lane.py; what is missing here is the on-hardware half.\n"
            "PROVEN root cause (2026-08-06, reading loader/loader_environment.c upstream, "
            "not assumed): VK_LAYER_PATH *and* VK_LOADER_LAYERS_DISABLE are both read "
            "through loader_secure_getenv, which returns NULL whenever the calling "
            "process token is High integrity — not a loader-age issue (real CI reports "
            "loader version 1.3.301, well past the 1.3.234 filter-var floor). If "
            "`registry_disable` is ALSO above and unavailable, the HKLM key this "
            "project's own SDK install populates "
            "(HKLM\\SOFTWARE\\Khronos\\Vulkan\\ExplicitLayers) was not writable or the "
            "validation layer manifest was not found there — see its detail above."
        )

    # The gate must let the real frame through and refuse the stripped one.
    _verdict.require_validation_armed(armed_state, armed_reason)
    with pytest.raises(_verdict.InstrumentError) as exc:
        _verdict.require_validation_armed(stripped_state, stripped_reason)
    text = str(exc.value)
    assert "UNOBSERVABLE" in text and "NOT A PASS" in text.upper(), (
        "the refusal must say what it is: R12's UNOBSERVABLE, not a zero and not a green"
    )
    print(
        f"[CRITERION 3d] PASS: the gate answers ARMED->run and {stripped_state}->"
        "ERROR(instrument) on this machine, so criterion 3's control can no longer be "
        f"satisfied by a machine incapable of running it.\n"
        f"[CRITERION 3d] mechanism that took: {mechanism} "
        f"(attempts: {[(n, s) for n, s, _ in attempts]})",
        file=sys.stderr,
    )
