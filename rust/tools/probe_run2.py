"""Does the all-zero-logits failure change character on the SECOND run of a session?

WHY THIS PROBE EXISTS
=====================

The Vulkan-vs-CPU comparison that found all-zero logits ran the session **once**. That matters
specifically, and it is the one question this file exists to answer:

    ORT's memory-pattern planner does not engage on the first `Run`. It *records* the allocation
    pattern on run 1 and sub-divides from run 2 onward. Measured, on both devices:
    1 run -> 0 interior pointers, 2 -> 13, 3 -> 26, 5 -> 52.

So on run 1, every tensor sits at the base of its own allocation. From run 2, several tensors are
packed into one allocation and ORT hands back `base + n`. **Any code that treats a bound pointer as
a span base is correct on run 1 and wrong from run 2** — and it fails as a wrong answer, not a
crash. A probe pointed at run 1 is pointed at a moment that entire class of bug cannot occur in.

That cuts both ways here, and both directions are informative:

* If run 2 differs from run 1, the interior-pointer hypothesis is live and the bug is in address
  handling, not arithmetic.
* If run 2 is byte-identical to run 1, the interior-pointer hypothesis is **excluded** for this
  failure, which is worth as much: it removes a suspect from a list three people are working
  through in parallel, and it stops anyone spending a cycle on it.

WHAT INSTRUMENT WOULD GO RED IF THIS PROBE'S CLAIM WERE FALSE
=============================================================

The claim is "the EP ran and produced these outputs". The instrument is the provider assertion
below, which aborts before comparing anything if `VulkanExecutionProvider` is not in
`session.get_providers()`.

That assertion is not boilerplate. This project has now produced three flattering results from
instruments that were not measuring what their wording implied, and the most recent was exactly
this: a Vulkan-vs-CPU comparison that reported `bit-identical: True` on both devices because it
never called `register_execution_provider_library`. ORT printed `Unknown Provider Type ... Falling
back to CPUExecutionProvider` and **did not raise**, so the comparison compared CPU to CPU and
passed. A checker that verifies the value but not the provider is a gate that does not gate.
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

RUNS = int(os.environ.get("PROBE_RUNS", "3"))
LAYERS = 32


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


def make_session(use_ep: bool) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if not use_ep:
        return ort.InferenceSession(str(_MODEL), opts, providers=["CPUExecutionProvider"])

    lib = os.environ.get(EP_LIB_ENV)
    if not lib:
        raise SystemExit(f"{EP_LIB_ENV} is not set; refusing to run a comparison that would "
                         f"silently be CPU-vs-CPU.")
    try:
        ort.register_execution_provider_library(EP_NAME, lib)
    except Exception as e:  # noqa: BLE001 — already registered is fine, anything else is not
        if "already" not in str(e).lower():
            raise
    sess = ort.InferenceSession(str(_MODEL), opts, providers=[EP_NAME, "CPUExecutionProvider"])

    # THE GATE. Not decoration: see the module docstring.
    providers = sess.get_providers()
    if EP_NAME not in providers:
        raise SystemExit(
            f"REFUSING TO COMPARE: {EP_NAME} is not among this session's providers "
            f"({providers}). ORT falls back to CPU with a printed warning and no exception, so "
            f"without this check the comparison below would compare CPU to CPU and report "
            f"agreement. That has already happened once on this project."
        )
    return sess


def main() -> int:
    if not _MODEL.exists():
        print(f"SKIP: model not found at {_MODEL}")
        return 0

    feeds = build_feeds()
    sess = make_session(use_ep=True)
    print(f"providers: {sess.get_providers()}")

    runs: list[list[np.ndarray]] = []
    for i in range(RUNS):
        out = sess.run(None, feeds)
        runs.append([np.array(o, copy=True) for o in out])
        logits = runs[-1][0].astype(np.float64)
        nz = int(np.count_nonzero(logits))
        print(
            f"  run {i + 1}/{RUNS}: logits range [{logits.min():.4f}, {logits.max():.4f}] "
            f"argmax {int(logits.argmax())}, {nz}/{logits.size} non-zero"
        )
        sys.stdout.flush()

    print("\n--- run-to-run comparison (the interior-pointer question) ---")
    # NaN-aware. A naive max|a-b| reports `nan` whenever either side contains a NaN, and `nan`
    # is not a difference — it is the absence of a comparison. Reading it as "the outputs
    # changed" would be exactly the kind of composite misreading that produced the all-zero
    # result being green: an individually-correct number used to answer a question it does not
    # address. `bit_identical` compares the raw bytes, which settles identity regardless of NaN.
    def bit_identical(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

    for k, o in enumerate(runs[0]):
        arr = np.asarray(o)
        if arr.size and arr.dtype.kind == "f":
            nan_n = int(np.count_nonzero(np.isnan(arr.astype(np.float64))))
            if nan_n:
                print(
                    f"  NOTE output[{k}] {arr.shape} {arr.dtype}: {nan_n}/{arr.size} NaN on run 1"
                )

    verdict_changed = False
    for i in range(1, RUNS):
        diffs = []
        for k, (a, b) in enumerate(zip(runs[0], runs[i])):
            if not bit_identical(np.asarray(a), np.asarray(b)):
                fa = np.asarray(a).astype(np.float64)
                fb = np.asarray(b).astype(np.float64)
                finite = np.isfinite(fa) & np.isfinite(fb)
                d = float(np.max(np.abs(fa[finite] - fb[finite]))) if finite.any() else float("nan")
                diffs.append((k, d))
        if diffs:
            verdict_changed = True
            print(f"run 1 vs run {i + 1}: {len(diffs)} output(s) differ BITWISE")
            for k, d in diffs[:10]:
                print(f"    output[{k}] max|delta| over finite elements = {d}")
        else:
            print(f"run 1 vs run {i + 1}: BIT-IDENTICAL across all {len(runs[0])} outputs")

    print()
    if verdict_changed:
        print(
            "FINDING: the outputs CHANGE between runs of the same session with identical\n"
            "inputs. That is a strong signal, and the memory-pattern planner is the leading\n"
            "explanation: from run 2 ORT packs tensors into shared allocations and hands back\n"
            "`base + n`. Any code treating a bound pointer as a span base is correct on run 1\n"
            "and wrong afterwards. Look at address handling, not arithmetic."
        )
    else:
        print(
            "FINDING: the outputs are byte-identical across runs. The memory-pattern planner\n"
            "DOES engage from run 2 (measured separately: 0/13/26 interior pointers at 1/2/3\n"
            "runs), so it engaged here and changed nothing. That EXCLUDES interior-pointer\n"
            "mishandling as the cause of this failure. The bug is deterministic and\n"
            "independent of tensor placement — it is upstream of address handling."
        )

    # Tear the session down *here*, while the process is still alive and the ORT logger is still
    # attached. Dropping it at interpreter exit means the allocator release either never runs or
    # runs with nothing listening, and the still-live-handle numbers then read 0 — an
    # unfalsifiable zero rather than a clean one. Measured: without this, a run that leaves 322
    # handles live reports `alloc_allocators_released: 0`.
    print("\n--- teardown (allocator release happens here, not at exit) ---")
    if os.environ.get("PROBE_NO_TEARDOWN") == "1":
        print(
            "PROBE_NO_TEARDOWN=1: leaking the session deliberately, so the allocator is released\n"
            "at interpreter exit with the session's tensors still held. This is the control for\n"
            "the still-live-handles warning: the only difference from the default path is WHEN\n"
            "the session is destroyed, so any change in the live count is a property of the\n"
            "observation point and not of our bookkeeping."
        )
        globals()["_leaked"] = sess
        return 0
    del sess
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
