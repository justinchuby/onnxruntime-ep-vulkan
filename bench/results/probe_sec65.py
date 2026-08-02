"""§6.5 probe — three sequential sessions in ONE process, device-backed allocation ON.

Two questions, both falsifiable:

1.  LIFETIME.  Before the process-global device owner landed, offering the session's
    VkDevice to the process-global memory provider was a use-after-free: session 0's
    Drop called vkDestroyDevice under the cached provider and session 1's inference
    died with STATUS_ACCESS_VIOLATION (0xC0000005).  This script creates and destroys
    three sessions in one process.  A crash IS the finding; "3 sessions OK" is the
    only passing outcome and it is printed with the exit code.

2.  FRAME.  Reads alloc_device_frame / alloc_device_authoritative_spans from the
    counters artifact.  SHARED + an integer (not the string "UNOBSERVABLE") is the
    §6.5-closed state transition.  The string is deliberate (R12): arithmetic on it
    fails loudly.

Output goes next to this file, never to the repo root.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent
EP_NAME = "VulkanExecutionProvider"


def tiny_model(path: Path) -> None:
    """Add(x, w) with w a 1 MiB initializer — big enough to be worth a device buffer."""
    n = 256 * 1024
    w = helper.make_tensor("w", TensorProto.FLOAT, [n], np.ones(n, dtype=np.float32))
    node = helper.make_node("Add", ["x", "w"], ["y"])
    graph = helper.make_graph(
        [node],
        "sec65",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))


def main() -> int:
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib or not Path(lib).is_file():
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB unset or missing", file=sys.stderr)
        return 2
    ort.register_execution_provider_library(EP_NAME, str(Path(lib).resolve()))

    model_path = HERE / "_sec65_model.onnx"
    tiny_model(model_path)

    counters = Path(os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"]).resolve()
    x = np.random.rand(256 * 1024).astype(np.float32)

    n_sessions = int(os.environ.get("SEC65_SESSIONS", "3"))
    for i in range(n_sessions):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            str(model_path), opts, providers=[EP_NAME], provider_options=[{}]
        )
        assert EP_NAME in sess.get_providers(), sess.get_providers()
        out = sess.run(None, {"x": x})[0]
        ok = np.allclose(out, x + 1.0)
        print(f"session {i}: ran, output correct = {ok}", flush=True)
        del sess

    print(f"SURVIVED {n_sessions} sequential sessions in one process", flush=True)

    if not counters.is_file():
        print(f"ERROR(instrument): no counters artifact at {counters}", file=sys.stderr)
        return 3
    data = json.loads(counters.read_text(encoding="utf-8"))
    # Every key here must (a) exist in the emitter and (b) carry the companion a reader needs to
    # interpret it. `alloc_device_spans` was neither: no such key is emitted anywhere in the
    # source, so this block printed `'<absent>'` on every run since it was written — a key that is
    # always absent reads like a measurement that came back empty. The real name is
    # `alloc_device_backed_spans`.
    #
    # And `alloc_device_authoritative_spans` was printed without `alloc_device_authoritative_ceiling`
    # (= backed - staged) or `alloc_device_residency_evaluations`, which are the two keys that
    # separate a measured zero from a pinned one. See `probe_indexspace.py`'s docstring: R11's
    # shape can appear in a *selection* while every field printed is individually true.
    keys = [
        "alloc_device_frame",
        "alloc_device_frame_device",
        "alloc_allocations",
        "alloc_staged_spans",
        "alloc_device_backed_spans",
        "alloc_device_buffer_binds",
        "alloc_device_authoritative_ceiling",
        "alloc_device_residency_evaluations",
        "alloc_device_authoritative_spans",
    ]
    for k in keys:
        print(f"{k} = {data.get(k, '<absent>')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
