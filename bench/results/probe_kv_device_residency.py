"""Probe: can the KV cache stay on the device across `run()` calls?

The question this answers
-------------------------
Round 2 established that ORT allocates our fused-node outputs through this EP's device
allocator (195/195 device-resident), and that binding them EP-side returns zeros, because a
device-backed span's *host staging block* is authoritative and nothing writes it on that path
(`probe_bound_output_correctness.py` -> HOST_STAGING_IS_AUTHORITATIVE).  The finding then was
that removing the KV round-trip needs either an EP-owned cache ORT never sees, or **IOBinding
with device `OrtValue`s** — a caller-side change that was untested.

This tests the caller-side option, because if it works it makes the device side authoritative
and unlocks the Step 1c output bind that is already built.

What has to be true, and each part is asked separately
-------------------------------------------------------
1. **Addressable.**  A caller can obtain memory for *this* EP's device, not CUDA's.  ORT 1.28's
   Python binding hardcodes `device_type='gpu'` to CUDA — `ortvalue_from_shape_and_type(...,
   'gpu', 0, 0x10de)` fails with *"Can't allocate memory on the CUDA device using this package"*
   even with our vendor id.  The escape is `OrtEpDevice.memory_info(DEFAULT)` passed as
   `memory_info=`, which routes to the EP's own allocator.
2. **Shared.**  The allocation must land on the *same* `VkDevice` the kernels run on.  Ours
   warns loudly when it does not (`SPLIT-DEVICE`), and a split device means a dispatch cannot
   bind the buffer at all — so a run that looks bound but reads `SPLIT-DEVICE` is refused here.
3. **Round-trippable.**  The bound output must be feedable as the next run's input *without*
   passing through host memory.  This is the whole point; anything less is the same round trip
   with more steps.
4. **Correct.**  Against the CPU EP, not against itself.  A device path that returns the
   previous run's data — or zeros — agrees with itself perfectly.  `1.0` relative agreement on
   every element is the signature of two all-zero tensors, and that mistake has already been
   made once on this branch.

A note on `device_name()`
--------------------------
An `OrtValue` allocated through our `memory_info` reports `device_name() == 'cuda'`.  That is
the Python binding's naming table, not a statement about where the bytes are: the allocation
demonstrably went through this EP's allocator, because the EP's own second-device warning fired
on it.  The name is not used as evidence anywhere below.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
CASE = REPO / "evidence" / "cases" / "group_query_attention_f16.onnx"


def _lib() -> str:
    return os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )


def _ep_device(ort):
    for d in ort.get_ep_devices():
        if d.ep_name == "VulkanExecutionProvider":
            return d
    return None


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-3)))


def main() -> int:
    import onnxruntime as ort

    doc: dict = {"probe": "can the KV cache stay device-resident across run() calls?",
                 "ort_version": ort.__version__}

    ort.register_execution_provider_library("VulkanExecutionProvider", _lib())
    d = _ep_device(ort)
    if d is None:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = ["the Vulkan EP is not among ORT's EP devices; nothing below is about it"]
        print(json.dumps(doc, indent=2))
        return 2
    doc["ep_device"] = {k: d.ep_metadata.get(k) for k in
                        ("vulkan.device_name", "vulkan.device_index", "vulkan.vendor_id")}

    # ── Question 1: addressable? ──────────────────────────────────────────────
    #
    # The session is created FIRST, and that ordering is load-bearing rather than tidy.  Asking
    # the EP for an allocator before any session exists builds a standalone device context, and
    # the EP says so: `alloc_device_frame = SPLIT-DEVICE`, a second VkDevice that no compute
    # dispatch can bind.  Measured on the first run of this probe, which refused itself for it.
    # An arena inherits this hazard: a caller that touches the allocator before the session gets
    # a device the kernels never run on, and every byte it holds is unreachable.
    if not CASE.is_file():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"the GQA evidence case is missing at {CASE}"]
        print(json.dumps(doc, indent=2))
        return 2

    counters = HERE / "kv_device_residency_counters.json"
    counters.unlink(missing_ok=True)
    os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)

    sess = ort.InferenceSession(
        str(CASE), providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    cpu = ort.InferenceSession(str(CASE), providers=["CPUExecutionProvider"])
    doc["case"] = {"inputs": [i.name for i in sess.get_inputs()],
                   "outputs": [o.name for o in sess.get_outputs()]}

    q1: dict = {}
    try:
        ort.OrtValue.ortvalue_from_shape_and_type(
            [2, 3], np.float16, "gpu", 0, int(d.ep_metadata["vulkan.vendor_id"], 16)
        )
        q1["by_device_type_and_vendor"] = "OK"
    except Exception as e:  # noqa: BLE001
        q1["by_device_type_and_vendor"] = f"{type(e).__name__}: {str(e)[:160]}"

    mi = None
    try:
        mi = d.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
        ov = ort.OrtValue.ortvalue_from_shape_and_type([2, 3], np.float16, memory_info=mi)
        q1["by_ep_memory_info"] = "OK"
        q1["memory_info_name"] = mi.name
        q1["memory_info_device_id"] = mi.device_id
        # Recorded, and explicitly NOT used as evidence: the binding's naming table says
        # 'cuda' for any non-CPU device.  Where the bytes are is established by the EP's own
        # allocator having produced them, not by this string.
        q1["ortvalue_device_name_reported"] = ov.device_name()
        q1["ortvalue_ptr_nonzero"] = ov.data_ptr() != 0
    except Exception as e:  # noqa: BLE001
        q1["by_ep_memory_info"] = f"{type(e).__name__}: {str(e)[:200]}"
    doc["q1_addressable"] = q1

    if mi is None or q1.get("by_ep_memory_info") != "OK":
        doc["verdict"] = "CALLER_CANNOT_ADDRESS_EP_MEMORY"
        doc["why"] = [
            "no route from a Python caller to this EP's device memory was found, so IOBinding "
            "with device OrtValues is not available in this shape and the EP-owned arena is "
            "the only remaining mechanism"
        ]
        print(json.dumps(doc, indent=2))
        return 1

    feeds = _build_feeds(sess)

    binding = sess.io_binding()
    ref = cpu.run(None, feeds)
    for name, arr in feeds.items():
        binding.bind_cpu_input(name, arr)
    bound: dict[str, object] = {}
    q2: dict = {}
    for o in sess.get_outputs():
        shape = _SHAPES[o.name]
        try:
            ov = ort.OrtValue.ortvalue_from_shape_and_type(
                shape, _np_dtype(o.type), memory_info=mi
            )
            binding.bind_ortvalue_output(o.name, ov)
            bound[o.name] = ov
        except Exception as e:  # noqa: BLE001
            q2[o.name] = f"{type(e).__name__}: {str(e)[:160]}"
            binding.bind_output(o.name)
    doc["q2_bind_failures"] = q2

    try:
        sess.run_with_iobinding(binding)
        doc["q3_run_with_device_outputs"] = "OK"
    except Exception as e:  # noqa: BLE001
        doc["q3_run_with_device_outputs"] = f"{type(e).__name__}: {str(e)[:300]}"
        doc["verdict"] = "ORT_REFUSES_DEVICE_BOUND_OUTPUTS"
        print(json.dumps(doc, indent=2))
        return 1

    got = binding.copy_outputs_to_cpu()

    # The control, and the primary criterion.  `cpu` vs `bound` conflates two differences: the
    # writeback path (what this probe is about) and the EP's own fp16 arithmetic gap against
    # the CPU reference (which exists with or without binding and is not this change's to
    # answer).  `ep` vs `bound` is the same kernel on the same inputs with only the writeback
    # path differing, so anything but agreement to the digit is the binding changing the answer.
    # Round 2 scored a bound lane against `cpu` and passed an all-zero result for it.
    plain = sess.run(None, feeds)
    doc["q4_rel_vs_cpu"] = {o.name: _rel(np.asarray(got[i]), np.asarray(ref[i]))
                            for i, o in enumerate(sess.get_outputs())}
    doc["q4_rel_vs_unbound_ep"] = {o.name: _rel(np.asarray(got[i]), np.asarray(plain[i]))
                                   for i, o in enumerate(sess.get_outputs())}
    doc["q4_unbound_ep_rel_vs_cpu"] = {o.name: _rel(np.asarray(plain[i]), np.asarray(ref[i]))
                                       for i, o in enumerate(sess.get_outputs())}
    scores = doc["q4_rel_vs_unbound_ep"]

    # The degeneracy guard.  Two all-zero tensors agree perfectly, and a relative metric with a
    # denominator floor reports that agreement as a pass.  This branch has already shipped one
    # such pass and it took a nonzero count to catch it.
    nonzero = {o.name: int(np.count_nonzero(np.asarray(got[i])))
               for i, o in enumerate(sess.get_outputs())}
    doc["nonzero_elements_returned"] = nonzero

    if counters.is_file():
        c = json.loads(counters.read_text(encoding="utf-8"))
        doc["counters"] = {k: c.get(k) for k in
                           ("alloc_device_frame", "alloc_device_authoritative_spans",
                            "dispatches_executed", "compute_calls", "compute_failures",
                            "device_losses", "readback_bytes", "outputs_device_resident",
                            "outputs_host_resident", "outputs_device_bound")}

    why: list[str] = []
    frame = (doc.get("counters") or {}).get("alloc_device_frame")
    if frame == "SPLIT-DEVICE":
        doc["verdict"] = "ERROR(instrument)"
        why.append(
            "alloc_device_frame = SPLIT-DEVICE: the allocation landed on a second VkDevice, "
            "which a compute dispatch cannot bind. Every number here would describe a device "
            "the kernels did not run on"
        )
    elif any(v == 0 for v in nonzero.values()):
        doc["verdict"] = "DEVICE_BOUND_OUTPUTS_RETURN_NOTHING"
        why.append(
            f"outputs returned all zeros: {[k for k, v in nonzero.items() if v == 0]} — the "
            f"same host-staging-authoritative shape as the EP-side bind, now reached from the "
            f"caller side"
        )
    elif max(scores.values()) > 0.0:
        doc["verdict"] = "DEVICE_BOUND_OUTPUTS_DIFFER_FROM_THE_UNBOUND_EP"
        why.append(
            f"same kernel, same inputs, only the writeback path differs, yet the results move "
            f"by {max(scores.values()):.4g} — the binding changed the answer"
        )
    else:
        doc["verdict"] = "KV_CAN_STAY_DEVICE_RESIDENT"
        why.append(
            "outputs allocated in EP device memory and bound through IOBinding; the run "
            "succeeded and every output is bit-identical to the same session run unbound"
        )
        why.append(
            f"the EP's own gap against the CPU reference is unchanged by binding: "
            f"{doc['q4_unbound_ep_rel_vs_cpu']} unbound vs {doc['q4_rel_vs_cpu']} bound. That "
            f"gap is this case's fp16 arithmetic and is not this probe's question"
        )
        why.append(
            "ORDERING IS LOAD-BEARING: the session must exist before the allocator is asked "
            "for memory. Asking first yields alloc_device_frame = SPLIT-DEVICE — a second "
            "VkDevice no dispatch can bind. This probe refused itself for exactly that on its "
            "first run"
        )
    doc["why"] = why
    doc["ep_side_step1c_bind"] = os.environ.get("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", "0")
    lane = "epbind" if (doc["counters"] or {}).get("outputs_device_bound") else "callerbind"
    doc["lane"] = lane
    (HERE / f"kv_device_residency-{lane}.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8"
    )
    print(json.dumps(doc, indent=2))
    return 0 if doc["verdict"] == "KV_CAN_STAY_DEVICE_RESIDENT" else 1


_DT = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


def _np_dtype(t: str):
    return _DT[t]


#: The GQA evidence case's shapes, with B=1 and S=1 (decode) fixed.  `past` is 4 tokens and
#: `present` is 5 — the extents differ, which is exactly the relocation the arena exists to
#: remove, and it is why this case is usable as a stand-in for the Phi-3.5 KV shape.
_B, _S, _PAST, _TOTAL = 1, 1, 4, 5
_SHAPES = {
    "attn_out": [_B, _S, 256],
    "present_key": [_B, 2, _TOTAL, 32],
    "present_value": [_B, 2, _TOTAL, 32],
}


def _build_feeds(sess) -> dict:
    """Feeds for the GQA evidence case.

    Deterministic and non-degenerate on purpose: an all-zero or constant input would make a
    correct kernel and a broken one produce the same tensor, which is the failure mode this
    probe exists to avoid on the *output* side.
    """
    rng = np.random.default_rng(20260802)

    def f16(*shape):
        return rng.standard_normal(shape).astype(np.float16)

    return {
        "packed_qkv": f16(_B, _S, 384),
        "past_key": f16(_B, 2, _PAST, 32),
        "past_value": f16(_B, 2, _PAST, 32),
        "seqlens_k": np.array([_TOTAL - 1], dtype=np.int32),
        "total_seq": np.array(_TOTAL, dtype=np.int32),
        "cos_cache": f16(64, 8),
        "sin_cache": f16(64, 8),
    }


if __name__ == "__main__":
    raise SystemExit(main())
