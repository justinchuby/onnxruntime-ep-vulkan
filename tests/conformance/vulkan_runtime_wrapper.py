"""Candidate runtime wrapper: runs onnx-tests models on the Vulkan EP.

onnx-tests selects the "candidate" runtime via the ``RUN_CANDIDATE`` environment variable,
resolved as a dotted import path to a
``Callable[[onnx.ModelProto], dict[str, np.ndarray]]``. By pointing ``RUN_CANDIDATE`` at
``vulkan_runtime_wrapper.run_vulkan`` (with this directory on ``PYTHONPATH``) the suite
compares Vulkan-EP output against its ONNX-reference source-of-truth **without any edit to
onnx-tests sources**.

Environment variables
---------------------
VULKAN_EP_LIB
    Absolute path to the EP cdylib (``libonnxruntime_vulkan_ep.so`` / ``.dll`` / ``.dylib``).
    Required.

VULKAN_EP_NAME
    EP name to register/select. Default ``VulkanExecutionProvider``.

VULKAN_EP_PROFILE
    Set to ``1`` to enable ORT profiling per session and record, per op-type, whether any
    node was assigned to the Vulkan EP vs CPU fallback. Attribution is written to
    ``$VULKAN_EP_ATTR_OUT`` (default ``vulkan_provider_attribution.json``) at process exit.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

import numpy as np
import onnx

_EP_NAME = os.environ.get("VULKAN_EP_NAME", "VulkanExecutionProvider")
_PROFILE = os.environ.get("VULKAN_EP_PROFILE") == "1"
_ATTR_OUT = os.environ.get("VULKAN_EP_ATTR_OUT", "vulkan_provider_attribution.json")

# op-type -> {"vulkan": bool, "cpu": bool} accumulated across the process
_attribution: dict[str, dict[str, bool]] = {}
_registered = False


def _ep_lib_path() -> str:
    lib = os.environ.get("VULKAN_EP_LIB")
    if not lib:
        raise RuntimeError(
            "VULKAN_EP_LIB is not set. Point it at the absolute path of the built EP cdylib:\n"
            "  Linux:   rust/target/release/libonnxruntime_vulkan_ep.so\n"
            "  macOS:   rust/target/release/libonnxruntime_vulkan_ep.dylib\n"
            "  Windows: rust\\target\\release\\onnxruntime_vulkan_ep.dll"
        )
    if not Path(lib).is_file():
        raise RuntimeError(f"VULKAN_EP_LIB does not point at a file: {lib!r}")
    return lib


def _ensure_registered(ort) -> None:
    global _registered
    if _registered:
        return
    lib = _ep_lib_path()
    try:
        ort.register_execution_provider_library(_EP_NAME, lib)
    except Exception as exc:
        if "already registered" not in str(exc).lower():
            raise
    _registered = True


def _record_attribution(model: onnx.ModelProto, prof_path: str) -> None:
    """Parse an ORT profiling trace and accumulate Vulkan-vs-CPU node placement."""
    try:
        with open(prof_path) as fh:
            events = json.load(fh)
    except Exception:
        return
    finally:
        try:
            os.remove(prof_path)
        except OSError:
            pass

    ran_on_vulkan = False
    cpu_ops: set[str] = set()
    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "Node":
            continue
        args = ev.get("args") or {}
        provider = args.get("provider") or ""
        op_type = args.get("op_name") or ""
        if "Vulkan" in provider:
            ran_on_vulkan = True
        elif op_type:
            cpu_ops.add(op_type)

    ignore = {"Constant", "Identity", "MemcpyFromHost", "MemcpyToHost"}
    op_types = {n.op_type for n in model.graph.node if n.op_type not in ignore}

    for op_type in op_types:
        slot = _attribution.setdefault(op_type, {"vulkan": False, "cpu": False})
        if op_type in cpu_ops:
            slot["cpu"] = True
        elif ran_on_vulkan:
            slot["vulkan"] = True


def run_vulkan(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Execute *model* on the Vulkan EP (with CPU fallback).

    The model carries all inputs as initializers, mirroring
    ``onnx_tests.runtime_wrappers.run_ort``.
    """
    import onnxruntime as ort

    _ensure_registered(ort)

    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    if _PROFILE:
        opt.enable_profiling = True

    sess = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=opt,
        providers=[_EP_NAME, "CPUExecutionProvider"],
    )
    output_names = [meta.name for meta in sess.get_outputs()]
    result = {k: v for k, v in zip(output_names, sess.run(None, {}))}

    if _PROFILE:
        prof_path = sess.end_profiling()
        _record_attribution(model, prof_path)

    return result


@atexit.register
def _flush_attribution() -> None:
    if not _PROFILE or not _attribution:
        return
    summary = {
        op: {"ran_on_vulkan": v["vulkan"], "ran_on_cpu_fallback": v["cpu"]}
        for op, v in sorted(_attribution.items())
    }
    try:
        with open(_ATTR_OUT, "w") as fh:
            json.dump(summary, fh, indent=2)
    except OSError:
        pass
