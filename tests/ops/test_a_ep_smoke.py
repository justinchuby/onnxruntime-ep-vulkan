"""EP load / registration smoke test — intentionally minimal.

This module is collected BEFORE the full op-correctness suite (alphabetical order,
``test_a_ep_smoke.py`` < ``test_barrier_parity.py`` < ``test_claim_diagnostics.py`` …).
Its purpose is to isolate an EP registration failure into its own test result so that
a hard crash during registration is diagnosed here — with a targetted error message —
rather than inside a full op-test session where the crash swallows all 198 tests and
produces only "Windows fatal exception: access violation".

Two tests are provided:
  1. test_ep_dll_loads_via_ctypes  — OS-level DLL dependency resolution (no ORT).
  2. test_ep_ort_registers         — ORT EP registration call (no session, no inference).

If (1) passes and (2) crashes, Tank knows the fault is in EP initialization code, not in
DLL dependency resolution.  The conftest ``register_vulkan_ep`` fixture also emits the same
ctypes result to stderr before calling ORT, so the diagnostic appears even if the crash
kills the whole session before pytest can record individual test outcomes.

These tests do NOT use the ``require_vulkan`` or ``vulkan_device_available`` fixtures —
they are intentionally independent to avoid circular dependency on the fixture under test.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import pytest

EP_NAME = "VulkanExecutionProvider"


# ---------------------------------------------------------------------------
# Module-scoped lib path fixture (skip if EP lib not set)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ep_lib_path() -> Path:
    """Return the resolved Path to the EP DLL/SO; skip if not configured."""
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        pytest.skip(
            "ONNXRUNTIME_VULKAN_EP_LIB not set — EP smoke tests skipped.\n"
            "Set it to the built EP library path, e.g.:\n"
            "  Linux:   export ONNXRUNTIME_VULKAN_EP_LIB=rust/target/release/libonnxruntime_vulkan_ep.so\n"
            "  Windows: $env:ONNXRUNTIME_VULKAN_EP_LIB = 'rust\\target\\release\\onnxruntime_vulkan_ep.dll'"
        )
    path = Path(lib).resolve()
    if not path.is_file():
        pytest.skip(f"EP library path is set but file not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Smoke test 1: ctypes DLL load — no ORT
# ---------------------------------------------------------------------------


def test_ep_dll_loads_via_ctypes(ep_lib_path: Path) -> None:
    """Can the OS load the EP DLL/SO without ORT involvement?

    Tests only DLL dependency resolution (PATH, rpath, MSVC runtime, Vulkan loader).
    Does NOT call any function inside the EP — just loading the library is the assertion.

    A failure here means a missing runtime dependency, NOT a bug in EP code.
    Ensure $ORT_HOME/lib (Linux: LD_LIBRARY_PATH, Windows: PATH) is set before
    running tests so that onnxruntime.so / onnxruntime.dll can be found.
    """
    try:
        handle = ctypes.CDLL(str(ep_lib_path))
        assert handle is not None, "ctypes.CDLL returned None"
    except OSError as exc:
        pytest.fail(
            f"ctypes.CDLL failed to load the EP library.\n"
            f"\n"
            f"  File : {ep_lib_path}\n"
            f"  Error: {exc}\n"
            f"\n"
            f"  This is a DLL dependency problem, not an EP code bug.\n"
            f"  Common causes on Windows:\n"
            f"    - onnxruntime.dll not on PATH  (add $ORT_HOME\\lib to PATH)\n"
            f"    - MSVC runtime not installed   (install VC_redist.x64.exe)\n"
            f"    - Vulkan loader not found       (install VulkanSDK or ICD)\n"
            f"  Common causes on Linux:\n"
            f"    - libonnxruntime.so not found  (add $ORT_HOME/lib to LD_LIBRARY_PATH)\n"
            f"    - libvulkan.so.1 not installed (apt install libvulkan1)"
        )


# ---------------------------------------------------------------------------
# Smoke test 2: ORT EP registration — no InferenceSession, no inference
# ---------------------------------------------------------------------------


def test_ep_ort_registers(ep_lib_path: Path) -> None:
    """Can ORT register the EP without crashing, and does it advertise a device?

    This test calls ``ort.register_execution_provider_library`` and then probes
    ``ort.get_ep_devices()`` to determine whether a device was enumerated.

    IMPORTANT — API distinction:
      ``ort.get_available_providers()`` lists only ORT's compiled-in (built-in) EPs.
      Plugin EPs registered via ``register_execution_provider_library`` are NOT included
      there. They appear in ``ort.get_ep_devices()`` if they enumerate at least one device.
      Asserting ``EP_NAME in get_available_providers()`` is the WRONG check for a plugin EP.

    This test determines case 1 vs case 2:
      Case 1 (correct): EP in get_ep_devices() → the EP works; wrong API was used before.
      Case 2 (problem): EP not in get_ep_devices() → EP advertises zero devices.
                        Possible causes: capability gate rejects lavapipe, or
                        probe_devices() fails silently. Owner: Switch.

    Ordering note: this test may run before OR after the session-scoped
    ``register_vulkan_ep`` conftest fixture. If the conftest ran first, ORT will
    raise "already registered" here — that is idempotent and treated as success.
    """
    import onnxruntime as ort

    try:
        ort.register_execution_provider_library(EP_NAME, str(ep_lib_path))
    except Exception as exc:
        if "already registered" in str(exc):
            # Conftest session fixture ran first. Registration is process-scoped;
            # one call is sufficient. Proceed to the device check.
            pass
        else:
            pytest.fail(
                f"ort.register_execution_provider_library raised an unexpected exception.\n"
                f"\n"
                f"  EP  : {EP_NAME}\n"
                f"  File: {ep_lib_path}\n"
                f"  Type: {type(exc).__name__}\n"
                f"  Msg : {exc}\n"
                f"\n"
                f"  ctypes load succeeded (test_ep_dll_loads_via_ctypes passed), so DLL\n"
                f"  dependencies are resolved. The fault is in EP initialization code\n"
                f"  (OrtEpFactory, Vulkan instance creation, or device enumeration).\n"
                f"  Owner: Tank / Switch."
            )

    # -----------------------------------------------------------------------
    # Device check via get_ep_devices() — the correct API for plugin EPs.
    # -----------------------------------------------------------------------
    all_devices = ort.get_ep_devices()
    vulkan_devices = [d for d in all_devices if d.ep_name == EP_NAME]
    all_ep_names = sorted({d.ep_name for d in all_devices})

    print(
        f"\n[EP smoke] get_ep_devices() → {len(all_devices)} device(s): {all_ep_names}",
        file=sys.stderr, flush=True,
    )
    for d in vulkan_devices:
        print(
            f"[EP smoke] VulkanEP device: ep_vendor={d.ep_vendor!r}",
            file=sys.stderr, flush=True,
        )

    if not vulkan_devices:
        # ---------------------------------------------------------------
        # CASE 2: EP registered but advertises zero devices.
        # This is a real finding — it means the capability gate rejected
        # lavapipe, or probe_devices() failed silently. This would mean
        # our only software-rasterizer CI lane has no Vulkan EP coverage.
        # Owner: Switch — check capability gate vs lavapipe feature flags.
        # ---------------------------------------------------------------
        pytest.fail(
            f"CASE 2: VulkanExecutionProvider registered but advertises ZERO devices.\n"
            f"\n"
            f"  All EP devices seen: {all_ep_names}\n"
            f"\n"
            f"  Possible causes:\n"
            f"    - Capability gate rejects lavapipe (check Vulkan feature flag queries)\n"
            f"    - probe_devices() returns Ok(vec![]) silently\n"
            f"    - Shader-less safety guard active (M0 exit criterion 5) — but glslc\n"
            f"      was present and 168 shaders should have been compiled\n"
            f"\n"
            f"  Owner: Switch — diagnose probe_devices() path for lavapipe.\n"
            f"  Impact: lavapipe (our only GPU-less lane) would have no Vulkan EP coverage."
        )
    # Case 1: EP appears in get_ep_devices(). get_available_providers() was wrong API.
    # Presence in the device list is the success condition; no further assertion needed.
