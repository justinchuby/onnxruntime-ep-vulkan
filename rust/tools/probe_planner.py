"""Does ORT's memory-pattern planner do pointer arithmetic on our handles?

The reserved-virtual-address handle design argues that `base + n` stays in-span by
construction. That is an argument. This probe is the attempt to *observe* it.

Why this probe exists separately from `probe_allocator.py`: that one runs each session
exactly once, and ORT's memory-pattern planner does not engage on the first run. It
*records* the allocation pattern during run 1 and only from run 2 onward allocates one
block per pattern and hands out sub-ranges of it. A single-run probe therefore cannot
see interior addressing even if the planner would produce it — which is exactly the
shape of measurement error worth naming: the instrument was pointed at a moment the
phenomenon cannot occur in.

So: static shapes (the planner refuses dynamic ones), `enable_mem_pattern`, and several
runs of the same session. The answer is whatever `pointer observations:` reports at
teardown; a zero there is a real result and is reported as one.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

REPO = Path(__file__).resolve().parents[2]
LIB = Path(os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", ""))
N = 4096
RUNS = 5


def build_model(path: Path) -> None:
    """A graph with several live tensors at once, so the planner has something to pack.

    A straight chain lets the planner reuse one buffer over and over; concurrent live
    values are what make it carve a block into pieces, which is the case that would
    produce an interior pointer.
    """
    init = helper.make_tensor("w", TensorProto.FLOAT, [N], np.ones(N, np.float32))
    nodes = [
        helper.make_node("Add", ["x", "w"], ["a"]),
        helper.make_node("Mul", ["a", "w"], ["b"]),
        helper.make_node("Sub", ["b", "w"], ["c"]),
        helper.make_node("Add", ["a", "c"], ["d"]),  # keeps `a` live across b and c
        helper.make_node("Mul", ["b", "d"], ["e"]),  # keeps `b` live too
        helper.make_node("Add", ["c", "e"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "planner_probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [N])],
        [init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    path.write_bytes(model.SerializeToString())


def main() -> int:
    if not LIB.is_file():
        print(f"FAIL: ONNXRUNTIME_VULKAN_EP_LIB not found: {LIB}")
        return 1
    counters = REPO / "rust" / "target" / "probe_planner_counters.json"
    os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)

    # The observations are only complete when the EP tears down, and a process cannot read
    # its own teardown. So the session runs in a child and this process reads what it left
    # behind. That also means the numbers reported here are from a run that actually
    # finished, rather than from a snapshot taken mid-flight.
    if "--child" not in sys.argv:
        counters.unlink(missing_ok=True)
        import subprocess

        r = subprocess.run(
            [sys.executable, __file__, "--child"], env=os.environ, text=True
        )
        if r.returncode != 0:
            print(f"FAIL: child exited {r.returncode}")
            return 1
        if not counters.is_file():
            print("FAIL: the child left no counters file — it never tore the EP down")
            return 1
        doc = json.loads(counters.read_text())
        print("\n--- what the child left behind ---")
        print(json.dumps(doc, indent=2))

        observed = doc.get("pointers_observed", 0)
        if observed == 0:
            print(
                "\nRESULT: not a single pointer of ours came back. This run verifies "
                "nothing about the allocator contract."
            )
            return 1
        if doc.get("dispatches_executed", 0) == 0:
            print("\nRESULT: nothing executed on the EP — the result below is CPU fallback.")
            return 1
        interior = doc.get("pointers_interior", 0)
        print(
            f"\nRESULT: {observed} pointer(s) came back across {RUNS} runs with "
            f"mem-pattern enabled; {doc.get('pointers_at_base', 0)} at a handle base, "
            f"{interior} interior (max offset {doc.get('pointer_max_offset', 0)} B), "
            f"{doc.get('pointers_in_guard_band', 0)} in a guard band, "
            f"{doc.get('pointers_use_after_free', 0)} use-after-free."
        )
        if interior == 0:
            print(
                "OBSERVED: ORT's planner did NOT hand back a derived pointer. In-span "
                "`base + n` remains correct by construction and UNOBSERVED in a real session."
            )
        return 0

    return child()


def child() -> int:
    ort.set_default_logger_severity(0)
    os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
    tmp = REPO / "rust" / "target" / "probe_planner_model.onnx"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    build_model(tmp)
    ort.register_execution_provider_library("VulkanExecutionProvider", str(LIB))

    devices = [d for d in ort.get_ep_devices() if d.ep_name == "VulkanExecutionProvider"]
    if not devices:
        print("FAIL: no EP devices")
        return 1

    want = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE")
    chosen = devices[int(want)] if want is not None and want.isdigit() else devices[0]

    so = ort.SessionOptions()
    so.enable_mem_pattern = True  # the thing under test
    so.add_provider_for_devices([chosen], {})
    sess = ort.InferenceSession(str(tmp), so)

    x = np.arange(N, dtype=np.float32)
    last = None
    for i in range(RUNS):
        # Same shapes every run: a shape change invalidates the recorded pattern, which
        # would put us back in the situation the first run is already in.
        out = sess.run(None, {"x": x})[0]
        print(f"  run {i + 1}/{RUNS} done", flush=True)
        if last is not None and not np.array_equal(out, last):
            print("FAIL: output changed between runs of an identical input")
            return 1
        last = out

    del sess  # force teardown so the ledger is written
    return 0


if __name__ == "__main__":
    sys.exit(main())
