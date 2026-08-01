"""Tank — the `UNWIRED` -> measurement transition for `alloc_device_authoritative_spans`.

Byte- and state-based only.  Nothing here reads a clock, so the result is identical
under load and quiet; Switch holds the exclusive claim on device-clock measurement.

WHAT THIS FALSIFIES (R10: the falsifier for "X is wired" is an artifact X produced)

`alloc_device_authoritative_spans` has had three lives, and the artifact spells each
one as a DIFFERENT JSON TYPE so that the transition cannot be forged by an increment:

    "UNOBSERVABLE"  str   R12 — the counted event cannot occur in this run's frame
                          (no provider, or the §6.5 second VkDevice).
    "UNWIRED"       str   R10 — the frame allows it, but the residency screen has
                          never run: `alloc_device_residency_evaluations` == 0.
    0, 1, 2, ...    int    a measurement.  The screen ran on N device-backed spans at
                          their terminal state and answered for each one.

The evaluation count is the load-bearing artifact.  A zero authoritative count is
worth nothing on its own; the same zero next to `evaluations = 9` says the question
was put to nine spans and nine of them answered "mirror".  An author incrementing
the authoritative counter cannot manufacture evaluations, because both move from one
call site and only one of them moves unconditionally.

WHAT IT ALSO REPORTS (R12 obligation 2)

Which device EACH SIDE is on, by name and by index, in every frame.  `SPLIT-DEVICE`
is a detection, not a description: on its own it says the two sides differ without
saying what they are, and a reader holding only that cannot tell a selector-1 run
from a selector-0 one.

USAGE

    $env:ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY="1"
    $env:ONNXRUNTIME_EP_VULKAN_DEVICE="0"   # then "1"
    python bench/results/probe_residency_screen.py

Output goes next to this file, never to the repo root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EP_NAME = "VulkanExecutionProvider"

# The keys this probe exists to read.  Absent is a distinct state from any value.
KEYS = [
    "alloc_device_frame",
    "alloc_device_frame_device",
    "alloc_device_frame_allocator_index",
    "alloc_device_frame_session_devices",
    "alloc_device_authoritative_spans",
    "alloc_device_residency_evaluations",
    "alloc_device_authoritative_ceiling",
    "alloc_device_buffer_binds",
    "alloc_device_backed_spans",
    "alloc_staged_spans",
    "alloc_unified_memory",
]


def _child(counters: Path) -> int:
    """One process: build a tiny device-backed graph, run it, let teardown dump."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib or not Path(lib).is_file():
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB unset or missing", file=sys.stderr)
        return 2
    ort.register_execution_provider_library(EP_NAME, str(Path(lib).resolve()))

    n = 256 * 1024
    w = helper.make_tensor("w", TensorProto.FLOAT, [n], np.ones(n, dtype=np.float32))
    node = helper.make_node("Add", ["x", "w"], ["y"])
    graph = helper.make_graph(
        [node],
        "residency_screen",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    model_path = HERE / "_residency_screen_model.onnx"
    onnx.save(model, str(model_path))

    x = np.random.rand(n).astype(np.float32)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(model_path), opts, providers=[EP_NAME], provider_options=[{}])
    if EP_NAME not in sess.get_providers():
        print(f"ERROR(instrument): EP not in providers: {sess.get_providers()}", file=sys.stderr)
        return 4
    out = sess.run(None, {"x": x})[0]
    if not np.allclose(out, x + 1.0):
        print("FAIL(condition): output is wrong; every counter below is about a broken run")
        return 5
    del sess
    return 0 if counters.is_file() else 3


def _report(label: str, counters: Path) -> dict:
    if not counters.is_file():
        print(f"  ERROR(instrument): no counters artifact at {counters}")
        return {}
    data = json.loads(counters.read_text(encoding="utf-8"))
    print(f"\n=== {label} ===")
    for k in KEYS:
        if k not in data:
            print(f"  {k:<42} <ABSENT>")
            continue
        v = data[k]
        print(f"  {k:<42} {v!r}   type={type(v).__name__}")

    auth = data.get("alloc_device_authoritative_spans", "<ABSENT>")
    evals = data.get("alloc_device_residency_evaluations", "<ABSENT>")
    if auth == "UNOBSERVABLE":
        verdict = (
            "UNOBSERVABLE (R12) — the counted event cannot occur in this frame. "
            "Not a negative result; not fixable by a call site."
        )
    elif auth == "UNWIRED":
        verdict = (
            "UNWIRED (R10) — the frame allows it, but the residency screen never ran. "
            "Not a negative result either."
        )
    elif isinstance(auth, int):
        verdict = (
            f"MEASUREMENT — the screen ran on {evals} device-backed span(s) at their "
            f"terminal state and returned {auth}. This zero (or count) is a result about "
            f"the world, not about the wiring."
        )
    else:
        verdict = f"UNRECOGNISED state {auth!r} — read the code before quoting it."
    print(f"  -> {verdict}")
    print(f"  -> DEVICE IDENTITY: {data.get('alloc_device_frame_sides', '<ABSENT>')}")
    return data


def main() -> int:
    if os.environ.get("_RESIDENCY_CHILD") == "1":
        return _child(Path(os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"]))

    rc = 0
    for selector in ("0", "1"):
        counters = HERE / f"residency_screen-dev{selector}.json"
        counters.unlink(missing_ok=True)
        env = dict(os.environ)
        env["_RESIDENCY_CHILD"] = "1"
        env["ONNXRUNTIME_EP_VULKAN_DEVICE"] = selector
        env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
        env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
        p = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=env)
        if p.returncode != 0:
            print(f"\n=== selector {selector}: child exited {p.returncode} ===")
            rc = rc or p.returncode
        _report(f"selector {selector} (ONNXRUNTIME_EP_VULKAN_DEVICE={selector})", counters)

    # The R12 control: with device memory OFF there is no provider, so both selectors
    # must report OFF/UNOBSERVABLE.  A run in which this control does NOT go grey means
    # the frame reporting is not reading the frame.
    counters = HERE / "residency_screen-nodevmem.json"
    counters.unlink(missing_ok=True)
    env = dict(os.environ)
    env["_RESIDENCY_CHILD"] = "1"
    env.pop("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", None)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    p = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=env)
    if p.returncode != 0:
        print(f"\n=== control (device memory OFF): child exited {p.returncode} ===")
    _report("CONTROL — device memory OFF (must be OFF/UNOBSERVABLE)", counters)
    return rc


if __name__ == "__main__":
    sys.exit(main())
