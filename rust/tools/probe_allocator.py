"""Local probe: does ORT 1.28 actually call our CreateAllocator, and what does its
memory-pattern planner do with the handles we return?

Not a test — a diagnostic. Run it directly:
    python rust/tools/probe_allocator.py

The question it answers is the one that made the earlier verification of the handle
scheme illegitimate: a registry that is not in ORT's path proves nothing about ORT.
"""

import json
import os
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

REPO = pathlib.Path(__file__).resolve().parents[2]
LIB = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"


def build_model(path: pathlib.Path) -> None:
    """A chain deep enough that ORT's memory-pattern planner has something to plan."""
    n = 8
    nodes = []
    prev = "x"
    for i in range(n):
        nodes.append(helper.make_node("Add", [prev, "w"], [f"t{i}"], name=f"add{i}"))
        prev = f"t{i}"
    nodes.append(helper.make_node("Identity", [prev], ["y"], name="out"))

    w = helper.make_tensor("w", TensorProto.FLOAT, [1024], np.ones(1024, np.float32))
    graph = helper.make_graph(
        nodes,
        "chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1024])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1024])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.save(model, str(path))


def main() -> int:
    if not LIB.is_file():
        print(f"FAIL: {LIB} not found - run `cargo build --release` first")
        return 2

    os.environ["ONNXRUNTIME_EP_VULKAN_VERBOSE"] = "1"
    counters = REPO / "rust" / "target" / "probe-counters.json"
    os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)

    tmp = REPO / "rust" / "target" / "probe_alloc_model.onnx"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    build_model(tmp)

    ort.set_default_logger_severity(0)
    ort.register_execution_provider_library("VulkanExecutionProvider", str(LIB))
    print("--- registered ---", flush=True)

    devices = [d for d in ort.get_ep_devices() if d.ep_name == "VulkanExecutionProvider"]
    print(f"EP devices advertised: {len(devices)}")
    for d in devices:
        print(f"  vendor={d.device.vendor_id:#06x} metadata={dict(d.ep_metadata)}")

    if not devices:
        print("FAIL: no EP devices")
        return 1

    for idx, dev in enumerate(devices):
        print(f"\n=== session on device #{idx} ===", flush=True)
        so = ort.SessionOptions()
        so.enable_mem_pattern = True  # the planner is the thing under test
        try:
            so.add_provider_for_devices([dev], {})
        except Exception as e:  # noqa: BLE001
            print(f"  add_provider_for_devices failed: {e}")
            continue
        try:
            sess = ort.InferenceSession(str(tmp), so)
            out = sess.run(None, {"x": np.arange(1024, dtype=np.float32)})
            expect = np.arange(1024, dtype=np.float32) + 8.0
            ok = np.allclose(out[0], expect)
            print(f"  run OK, numerically correct: {ok}")
        except Exception as e:  # noqa: BLE001
            print(f"  session/run raised: {type(e).__name__}: {e}")

    if counters.is_file():
        print("\n--- counters ---")
        print(json.dumps(json.loads(counters.read_text()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
