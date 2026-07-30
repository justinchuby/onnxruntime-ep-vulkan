"""Why the "N device handle(s) were still live" warning fired on 2.09 GB, and what it meant.

The warning used to end in an open disjunction: *"ORT frees what it allocated, so this is either
a leak on our side or a tensor the session outlived."* It is neither. This probe is the control
that shows which.

`HandleRegistry` is **process-global per device** (`factory::REGISTRIES`), shared by every
allocator and data transfer for that device. `VulkanAllocator::release` read
`registry.stats().live_spans` and attributed the result to the allocator being released — but with
two sessions open on one device, the count includes the *other* session's live tensors. The first
release therefore reported ~322 handles and 2.09 GB that were not orphaned, were not leaked, and
were freed normally a moment later when the second session went away.

Run it two ways and compare the same number:

    PROBE_SESSIONS=1  ->  live at release 0        (single session: nothing else holds spans)
    PROBE_SESSIONS=2  ->  live at first release N  (the other session's tensors, still in use)

Neither run leaks. `alloc_frees_after_release` is 0 in both, and that is the number that would go
red if the benign reading were wrong: a Free arriving after a release would mean ORT still held a
pointer into a registry we had torn down.
"""

from __future__ import annotations

import gc
import os
import pathlib
import sys

import numpy as np
import onnxruntime as ort

EP_NAME = "VulkanExecutionProvider"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"

_MODEL = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models"
    r"\Microsoft\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
    r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
)

LAYERS = 32
SESSIONS = int(os.environ.get("PROBE_SESSIONS", "2"))


def build_feeds() -> dict[str, np.ndarray]:
    empty_kv = np.zeros((1, 32, 0, 96), dtype=np.float16)
    feeds: dict[str, np.ndarray] = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.array([[1]], dtype=np.int64),
    }
    for layer in range(LAYERS):
        feeds[f"past_key_values.{layer}.key"] = empty_kv
        feeds[f"past_key_values.{layer}.value"] = empty_kv
    return feeds


def make_session() -> ort.InferenceSession:
    lib = os.environ.get(EP_LIB_ENV)
    if not lib:
        raise SystemExit(f"set {EP_LIB_ENV} to the built plugin")
    ort.register_execution_provider_library(EP_NAME, str(lib))
    sess = ort.InferenceSession(str(_MODEL), providers=[EP_NAME, "CPUExecutionProvider"])
    # The gate that earned its place: registering under a name the session does not then request
    # makes ORT print "Unknown Provider Type ... Falling back to CPUExecutionProvider" and NOT
    # raise, so every measurement below would silently describe the CPU EP.
    if EP_NAME not in sess.get_providers():
        raise SystemExit(f"{EP_NAME} not in {sess.get_providers()} — refusing to measure")
    return sess


def main() -> int:
    feeds = build_feeds()
    print(f"opening {SESSIONS} concurrent session(s) on device "
          f"{os.environ.get('ONNXRUNTIME_EP_VULKAN_DEVICE', '?')}")

    sessions = []
    for i in range(SESSIONS):
        s = make_session()
        s.run(None, feeds)
        sessions.append(s)
        print(f"  session {i + 1} open and run once")

    # Release them one at a time. With SESSIONS=2 the FIRST release happens while the second
    # session's tensors are still live in the shared registry — which is exactly the state that
    # produced the 2.09 GB warning under pytest, where several model sessions overlap.
    for i in range(len(sessions)):
        print(f"  releasing session {i + 1} of {len(sessions)}")
        sessions[i] = None
        gc.collect()

    print("all sessions released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
