"""``python -m onnxruntime_ep_vulkan`` — say what is installed and whether it works.

Two modes, and the split matters:

* default: describe the installation without touching ORT registration. Safe anywhere.
* ``--check``: actually register, build a four-element ``Add``, run it, and report whether
  the EP was selected. This is the positive control. A package that can only describe
  itself has never been seen in its working state.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import (
    EpVulkanError,
    assert_ep_selected,
    describe,
    providers,
    register_execution_provider_library,
)


def _trivial_add_model() -> bytes:
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["c"])],
        "trivial_add",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [4]),
        ],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


def _check() -> int:
    import numpy as np
    import onnxruntime as ort

    path = register_execution_provider_library()
    print(f"registered: {path}")

    devices = [d for d in ort.get_ep_devices() if d.ep_name == "VulkanExecutionProvider"]
    print(f"ep devices advertised: {len(devices)}")
    if not devices:
        # Not an error by itself: the EP is *designed* to advertise zero devices when there
        # is no Vulkan ICD, and a shader-less build does the same. This module cannot tell
        # those two apart from outside, and says so rather than guessing.
        print(
            "  note: zero devices. Either this machine has no Vulkan ICD, or this is a\n"
            "  shader-less build. This package cannot distinguish them; run\n"
            "  `epctl --probe-loader` from the repository to find out which."
        )

    sess = ort.InferenceSession(_trivial_add_model(), providers=providers())
    out = sess.run(
        None, {"a": np.ones(4, np.float32), "b": np.full(4, 2.0, np.float32)}
    )[0]
    print(f"session providers: {sess.get_providers()}")
    print(f"output: {out.tolist()} (expected [3.0, 3.0, 3.0, 3.0])")
    try:
        assert_ep_selected(sess)
    except EpVulkanError as exc:
        print(f"\nFAIL: {exc}")
        return 1
    if not np.allclose(out, 3.0):
        print("\nFAIL: the EP was selected but the numbers are wrong.")
        return 1
    print("\nOK: the Vulkan EP was selected for the session and produced correct output.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m onnxruntime_ep_vulkan")
    parser.add_argument(
        "--check",
        action="store_true",
        help="register the EP and run a trivial model on it (positive control)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.check:
        return _check()

    info = describe()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    for key, value in info.items():
        print(f"{key}: {value}")
    if info.get("library_path") is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
