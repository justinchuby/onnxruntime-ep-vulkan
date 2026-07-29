"""Pytest configuration for the Vulkan EP op-correctness suite.

The Vulkan execution-provider plugin is registered once per test session from the
``ONNXRUNTIME_VULKAN_EP_LIB`` environment variable (the cargo-built cdylib). Running
``pytest`` without that variable **skips** the entire suite with a clear message rather
than failing, so this is safe to include in any pytest invocation.

Environment variables
---------------------
ONNXRUNTIME_VULKAN_EP_LIB
    Absolute path to the built EP cdylib
    (``rust/target/release/libonnxruntime_vulkan_ep.so`` on Linux,
    ``libonnxruntime_vulkan_ep.dylib`` on macOS,
    ``onnxruntime_vulkan_ep.dll`` on Windows).

ONNXRUNTIME_EP_VULKAN_VALIDATE
    Set to ``1`` to force validation-layer checks in every test session (adds
    ``ep.enable_validation=true`` to all SessionOptions). In CI debug lanes this
    is set automatically; locally it can be opt-in. The EP panics on any Vulkan
    validation error, which surfaces as an ORT exception and fails the test.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

EP_NAME = "VulkanExecutionProvider"


# ---------------------------------------------------------------------------
# Session-level EP registration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def register_vulkan_ep() -> bool:
    """Register the Vulkan EP plugin once per session; return True iff registration succeeded.

    Returns False (rather than skipping immediately) so that tests not requiring Vulkan
    (e.g., NumPy oracle checks, per-layer capture unit tests) can still run without the EP.
    Tests that DO require the EP use ``vulkan_device_available`` or ``require_vulkan``,
    both of which depend on this fixture and skip appropriately.

    Diagnostic protocol (helps Tank when the EP crashes during registration):
    1. Probe with ctypes.CDLL first — tests OS-level DLL dependency resolution without
       ORT involvement. If this fails, the cause is a missing runtime DLL (e.g.
       onnxruntime.dll, Vulkan loader, MSVC runtime), NOT an EP code bug.
    2. If ctypes load succeeds, call ort.register_execution_provider_library. If THAT
       crashes with a hard access violation, the ctypes-OK message will have been
       flushed to stderr before the crash, telling the diagnoser that DLL resolution
       was not the problem — the fault is in EP initialization code.
    All messages go to stderr with flush=True so they survive a hard process crash.
    """
    import ctypes

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        return False
    lib_path = Path(lib).resolve()
    if not lib_path.is_file():
        print(
            f"\n[EP smoke] ONNXRUNTIME_VULKAN_EP_LIB set but file not found: {lib_path}",
            file=sys.stderr, flush=True,
        )
        return False

    # Step 1: ctypes load — DLL dependency check, no ORT involvement.
    try:
        ctypes.CDLL(str(lib_path))
        print(
            f"\n[EP smoke] ctypes.CDLL({lib_path.name}): OK — DLL dependencies resolved.",
            file=sys.stderr, flush=True,
        )
    except OSError as exc:
        print(
            f"\n[EP smoke] ctypes.CDLL FAILED: {exc}\n"
            f"[EP smoke] Cause: missing DLL dependency (onnxruntime.dll, Vulkan loader,\n"
            f"[EP smoke] or MSVC runtime). On Windows ensure $ORT_HOME\\lib is on PATH.\n"
            f"[EP smoke] EP registration skipped.",
            file=sys.stderr, flush=True,
        )
        return False

    # Step 2: ORT registration. A hard crash here means EP initialization code,
    # not DLL resolution (step 1 above already ruled that out).
    print(
        f"[EP smoke] Calling ort.register_execution_provider_library({EP_NAME!r}) ...",
        file=sys.stderr, flush=True,
    )
    ort.register_execution_provider_library(EP_NAME, str(lib_path))
    print(
        f"[EP smoke] ort.register_execution_provider_library: OK.",
        file=sys.stderr, flush=True,
    )
    return True


# ---------------------------------------------------------------------------
# Device availability probe (session-scoped, opt-in per test)
# ---------------------------------------------------------------------------


def _probe_vulkan_device() -> bool:
    """Return True if the registered Vulkan EP claims at least one node in a probe session.

    Builds a trivial single-Add model, runs it with profiling enabled, and checks
    whether any profiling event has provider=VulkanExecutionProvider. Returns False
    if the EP is registered but advertises zero devices (no Vulkan ICD, or all devices
    fail the capability gate).
    """
    # Build a minimal Add model using onnx_ir
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
    out = ir.Value(name="out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
    node = ir.node("Add", [x, y], outputs=[out])
    graph = ir.Graph([x, y], [out], nodes=[node], name="probe", opset_imports={"": 21})
    model_bytes = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()

    opts = ort.SessionOptions()
    opts.log_severity_level = 4  # suppress noise during probe
    opts.enable_profiling = True
    opts.profile_file_prefix = "_vulkan_probe"
    try:
        sess = ort.InferenceSession(
            model_bytes, opts, providers=[EP_NAME, "CPUExecutionProvider"]
        )
        feeds = {
            "x": np.array([1.0, 2.0], dtype=np.float32),
            "y": np.array([3.0, 4.0], dtype=np.float32),
        }
        sess.run(None, feeds)
        profile_path = sess.end_profiling()
    except Exception:
        return False

    try:
        with open(profile_path) as fh:
            events = json.load(fh)
        providers = {
            e.get("args", {}).get("provider")
            for e in events
            if e.get("cat") == "Node" and isinstance(e.get("args"), dict)
        }
        return EP_NAME in providers
    except Exception:
        return False
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass


_DEVICE_AVAILABLE: bool | None = None  # cached per session


@pytest.fixture(scope="session")
def vulkan_device_available(register_vulkan_ep: bool) -> bool:
    """Session-scoped fixture: skip if EP not registered; return True iff a device is present."""
    if not register_vulkan_ep:
        pytest.skip(
            "ONNXRUNTIME_VULKAN_EP_LIB is not set or not found — EP not registered. "
            "Set it to the built EP cdylib path:\n"
            "  Linux:   rust/target/release/libonnxruntime_vulkan_ep.so\n"
            "  Windows: rust\\target\\release\\onnxruntime_vulkan_ep.dll\n"
            "Run: cargo build --release",
        )
    global _DEVICE_AVAILABLE
    if _DEVICE_AVAILABLE is None:
        _DEVICE_AVAILABLE = _probe_vulkan_device()
    return _DEVICE_AVAILABLE


@pytest.fixture
def require_vulkan(vulkan_device_available: bool) -> None:
    """Skip the calling test if no Vulkan device is available."""
    if not vulkan_device_available:
        pytest.skip(
            "No Vulkan device available — either no ICD is installed or all devices "
            "failed the capability gate. Tests requiring a Vulkan device are skipped. "
            "To run on CPU software rasterizer (lavapipe): "
            "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
        )


# ---------------------------------------------------------------------------
# Session options factory (provides validation layer control)
# ---------------------------------------------------------------------------


@pytest.fixture
def session_options() -> ort.SessionOptions:
    """Standard SessionOptions for op tests: quiet logging, optional validation layers."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # WARNING and above only
    validate = os.environ.get("ONNXRUNTIME_EP_VULKAN_VALIDATE", "").strip()
    if validate == "1":
        # Force validation layers on even in release builds.
        # The EP panics on any validation error → test fails with ORT exception.
        opts.add_session_config_entry("ep.enable_validation", "1")
    return opts
