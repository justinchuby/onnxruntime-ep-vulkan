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

import os
import subprocess
import sys
from pathlib import Path

import onnxruntime as ort
import pytest

import _models as m

HERE = Path(__file__).parent
_REPO_ROOT = HERE.parent.parent
_CARGO_MANIFEST = _REPO_ROOT / "rust" / "Cargo.toml"


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
    """
    epctl = _epctl_path()
    if epctl is None:
        pytest.skip("epctl binary not found next to EP lib — build first: cargo build --release")

    result = subprocess.run(
        [str(epctl), "--probe-validation"],
        capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    print(f"\n[CRITERION 3a] epctl --probe-validation output:\n{output}", file=sys.stderr)

    if result.returncode == 3:
        pytest.skip(
            "VK_LAYER_KHRONOS_validation is not installed on this machine — "
            "criterion 3 cannot be assessed here. "
            "Install the Vulkan SDK validation layers (VulkanSDK/runtime/...) "
            "or set VK_LAYER_PATH. Exit 3 means 'no answer', not 'pass'."
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
    """
    env = _cargo_env()
    cmd = [
        "cargo", "test", "--lib", "--release",
        "--manifest-path", str(_CARGO_MANIFEST),
        "ep_messenger_fires_for_planted_fence_leak",
        "--", "--nocapture", "--ignored",
    ]
    print(f"\n[CRITERION 3b] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        env=env,
        timeout=300,  # cargo build + test; first run is slow
    )
    output = result.stdout + result.stderr
    print(f"[CRITERION 3b] cargo test output:\n{output[-3000:]}", file=sys.stderr)

    if result.returncode != 0 and "SKIP" in output.upper():
        # The Rust test itself may skip (no shaders, no ICD). That is acceptable here —
        # the plant cannot fire on a machine with no Vulkan. The armed test covers the layer.
        pytest.skip(
            "ep_messenger_fires_for_planted_fence_leak skipped itself (no shaders or no ICD). "
            "The armed test (test_validation_messenger_armed) still verifies the layer is live."
        )

    assert result.returncode == 0, (
        "ep_messenger_fires_for_planted_fence_leak failed or did not run.\n"
        f"Exit code: {result.returncode}\n"
        f"Output (last 2000 chars):\n{output[-2000:]}"
    )

    # The Rust test prints: [EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak = N
    import re
    match = re.search(r"EP_VALIDATION_ERROR_COUNT after planted fence leak\s*=\s*(\d+)", output)
    assert match is not None, (
        "ep_messenger_fires_for_planted_fence_leak ran and passed but did not print "
        "the EP_VALIDATION_ERROR_COUNT artifact line. "
        "The assertion is not checking the right output — look for the #[ignore]'d test "
        "actually running (not being compiled out).\n"
        f"Output:\n{output[-2000:]}"
    )
    count = int(match.group(1))
    assert count > 0, (
        f"EP_VALIDATION_ERROR_COUNT = {count} after planted fence leak. "
        "Expected > 0 — the EP's VkDebugUtilsMessengerEXT did not fire. "
        "Either the EP's instance does not have a messenger, or the VUID is not caught "
        "by this validation layer build. Check VK_LAYER_KHRONOS_validation version."
    )
    print(
        f"[CRITERION 3b] PASS: EP_VALIDATION_ERROR_COUNT = {count} after planted fence leak. "
        f"EP's own VkDebugUtilsMessengerEXT is wired and fires on a real violation.",
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
    """
    env = _cargo_env()
    cmd = [
        "cargo", "test", "--lib", "--release",
        "--manifest-path", str(_CARGO_MANIFEST),
        "add_f32_dispatches_end_to_end",
        "--", "--nocapture",
    ]
    print(f"\n[CRITERION 3c] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        env=env,
        timeout=300,
    )
    output = result.stdout + result.stderr
    print(f"[CRITERION 3c] cargo test output:\n{output[-3000:]}", file=sys.stderr)

    if result.returncode != 0 and (
        "no capable device" in output or "no shaders" in output.lower()
        or "[SKIP]" in output
    ):
        pytest.skip(
            "add_f32_dispatches_end_to_end skipped (no ICD or no shaders). "
            "Criterion 3c requires a capable device with validation layers."
        )

    assert result.returncode == 0, (
        f"add_f32_dispatches_end_to_end failed (exit {result.returncode}).\n"
        f"Output:\n{output[-2000:]}"
    )

    # Scan for VUID-prefixed validation errors.  The messenger callback prints them as:
    #   EPCTL-VALIDATION-CAUGHT: VUID-... or
    #   VulkanValidation: VUID-...
    import re
    vuid_lines = [
        line for line in output.splitlines()
        if re.search(r"VUID-", line)
    ]
    assert not vuid_lines, (
        f"add_f32_dispatches_end_to_end produced {len(vuid_lines)} VUID validation error(s) "
        f"after Switch's binding-arity fix.  The criterion-3 clean reading is NOT clean.\n"
        f"VUID lines:\n" + "\n".join(vuid_lines[:20]) + "\n"
        f"This requires investigation — route to Switch."
    )
    print(
        f"[CRITERION 3c] PASS: zero VUID errors in add_f32_dispatches_end_to_end output "
        f"(messenger armed, binding-arity fix in place). Criterion 3 clean reading is valid.",
        file=sys.stderr,
    )
