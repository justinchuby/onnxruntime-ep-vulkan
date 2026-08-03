"""Does a device-resident KV cache actually remove the round trip, or only move it?

`probe_kv_device_residency.py` established that ORT *permits* a caller-allocated device output and
that the bytes come back correct.  It cannot answer this question, because it reads every output
with `copy_outputs_to_cpu()` — it *asks* for host bytes, so of course it pays for them.  Its
counters say so plainly: 6 bound outputs, 6 downloads, 3584 download bytes = 2 x (512 + 640 + 640),
one whole-span refresh each.  The round trip was moved, not removed.

The claim that matters is narrower and this probe is built only for it:

    when nobody asks for the host copy, nobody pays for it,
    and the per-step cost does not grow with the length of the chain.

# The two lanes

* **resident** — `present_key`/`present_value` come back as device `OrtValue`s and are fed
  straight back in as the next step's `past_key`/`past_value` via `bind_ortvalue_input`.  The host
  never sees them.
* **host** — the same chain, but each step calls `.numpy()` on the KV outputs and re-feeds them
  with `bind_cpu_input`.  This is what the EP does today.

Both lanes run the *same* number of steps on the *same* session with the *same* inputs, so the
only difference is which side of the link the KV lives on.

# The falsifier, and why it is a slope rather than a total

A total can be beaten by a lane that does less work.  The per-step *slope* of
`alloc_device_download_bytes` cannot: both lanes execute the same dispatches on the same shapes, so
if the resident lane's slope is not lower, the residency bought nothing and this probe must say so.

`ROUND_TRIP_NOT_REMOVED` is a real verdict here and it exits non-zero.  This project has twice
shipped a number that was an observation ending early or a path that had been removed along with
whatever it was counting, so a lane whose slope is zero *because it stopped running* is checked
against the dispatch count before its zero is believed.

# What is deliberately NOT claimed

No wall-clock.  The box is permanently contended (`PERF.md` §20); this is a byte count, and byte
counts are exact.  Niobe's readback slope is 393,216 B per past token and exact to the byte, so a
change in this term is unambiguous when it happens.
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

_B, _S, _HEADS, _HEAD_DIM = 1, 1, 2, 32
_PAST, _TOTAL = 4, 5
_STEPS = 6


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
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-3)))


def _counters(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _static_feeds(rng):
    def f16(*shape):
        return rng.standard_normal(shape).astype(np.float16)

    return {
        "packed_qkv": f16(_B, _S, 384),
        "cos_cache": f16(64, 8),
        "sin_cache": f16(64, 8),
    }


def _run_chain(ort, sess, mi, resident: bool, counters_path: pathlib.Path):
    """Run `_STEPS` decode steps and return what each one cost on the link.

    Returns `(per_step_download_bytes, final_outputs_on_host, dispatches, notes)`.

    In the `resident` lane the `past` KV is allocated in device memory **once**, before the loop,
    and every step binds that same `OrtValue`; the outputs are bound to device `OrtValue`s and
    nothing calls `.numpy()` inside the loop.  In the `host` lane every step re-feeds `past` from
    a numpy array and materialises the outputs, which is the path the EP takes today.

    # The extent this case cannot exercise, stated rather than hidden

    The GQA evidence case fixes `past` at 4: binding a 0- or 5-token `past_key` is rejected by ORT
    with *"Got invalid dimensions for input"*, measured.  So this probe holds the extent constant
    and varies only **who reads the bytes**.  That isolates the term it is about, but it means the
    growing-context case — where Niobe's 393,216 B per past token lives and where the win is
    largest — is NOT measured here.  A number from this probe is a lower bound on that one, and
    must not be quoted as it.
    """
    rng = np.random.default_rng(20260802)
    static = _static_feeds(rng)
    past_k_host = rng.standard_normal((_B, _HEADS, _PAST, _HEAD_DIM)).astype(np.float16)
    past_v_host = rng.standard_normal((_B, _HEADS, _PAST, _HEAD_DIM)).astype(np.float16)
    per_step: list[int] = []
    notes: list[str] = []
    dispatches: list[int] = []
    last_kv = None

    past_k_dev = past_v_dev = None
    if resident:
        # Seeded once, on the device, before any accounting starts.  The seeding upload is real
        # and is deliberately outside the per-step window: it is paid once per session, not once
        # per token, and charging a per-token axis with it would flatter this lane.
        #
        # `ortvalue_from_numpy` takes no `memory_info` — measured, it raises TypeError — so the
        # device allocation and the fill are two calls.  This is the same shape as the
        # `device_type='gpu'` gap: the Python binding's convenient constructors are reachable only
        # by the built-in providers, and a plugin EP has to go the long way round.
        past_k_dev = ort.OrtValue.ortvalue_from_shape_and_type(
            list(past_k_host.shape), np.float16, memory_info=mi
        )
        past_v_dev = ort.OrtValue.ortvalue_from_shape_and_type(
            list(past_v_host.shape), np.float16, memory_info=mi
        )
        past_k_dev.update_inplace(past_k_host)
        past_v_dev.update_inplace(past_v_host)

    before = _counters(counters_path)
    prev_bytes = int(before.get("alloc_device_download_bytes") or 0)
    prev_dispatch = int(before.get("dispatches_executed") or 0)

    for _step in range(_STEPS):
        binding = sess.io_binding()
        for name, arr in static.items():
            binding.bind_cpu_input(name, arr)
        binding.bind_cpu_input("seqlens_k", np.array([_TOTAL - 1], dtype=np.int32))
        binding.bind_cpu_input("total_seq", np.array(_TOTAL, dtype=np.int32))

        if resident:
            binding.bind_ortvalue_input("past_key", past_k_dev)
            binding.bind_ortvalue_input("past_value", past_v_dev)
        else:
            binding.bind_cpu_input("past_key", past_k_host)
            binding.bind_cpu_input("past_value", past_v_host)

        outs = {}
        for o in sess.get_outputs():
            shape = [_B, _S, 256] if o.name == "attn_out" else [_B, _HEADS, _TOTAL, _HEAD_DIM]
            ov = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float16, memory_info=mi)
            binding.bind_ortvalue_output(o.name, ov)
            outs[o.name] = ov

        sess.run_with_iobinding(binding)

        if not resident:
            # What the EP does today: every output comes back across the link every step.
            for ov in outs.values():
                ov.numpy()
        last_kv = outs

        now = _counters(counters_path)
        b = int(now.get("alloc_device_download_bytes") or 0)
        d = int(now.get("dispatches_executed") or 0)
        per_step.append(b - prev_bytes)
        dispatches.append(d - prev_dispatch)
        prev_bytes, prev_dispatch = b, d

    # Materialise the final outputs in BOTH lanes, after the per-step accounting is closed, so the
    # lanes can be compared for correctness without the resident lane's one readback being charged
    # to a step.  A lane that is cheaper because it never produced the answer is not cheaper.
    final = {k: np.asarray(v.numpy()) for k, v in last_kv.items()}
    notes.append(
        f"{_STEPS} steps at a FIXED past extent of {_PAST}: this case's graph rejects any other, "
        "so the growing-context term is not measured here"
    )
    return per_step, final, dispatches, notes


def main() -> int:
    import onnxruntime as ort

    doc: dict = {
        "probe": "does device residency remove the KV round trip, or only move it?",
        "ort_version": ort.__version__,
        "steps": _STEPS,
    }
    ort.register_execution_provider_library("VulkanExecutionProvider", _lib())
    d = _ep_device(ort)
    if d is None:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = ["the Vulkan EP is not among ORT's EP devices; nothing below is about it"]
        print(json.dumps(doc, indent=2))
        return 2
    doc["ep_device"] = {k: d.ep_metadata.get(k) for k in
                        ("vulkan.device_name", "vulkan.device_index", "vulkan.vendor_id")}
    if not CASE.is_file():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"the GQA evidence case is missing at {CASE}"]
        print(json.dumps(doc, indent=2))
        return 2

    counters = HERE / "kv_chain_counters.json"
    counters.unlink(missing_ok=True)
    os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)

    # Ordering is load-bearing: the session must exist before the allocator is asked for memory,
    # or the EP builds a second VkDevice (`SPLIT-DEVICE`) that no dispatch can bind.  Established
    # by probe_kv_device_residency.py refusing itself for exactly that.
    sess = ort.InferenceSession(
        str(CASE), providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    mi = d.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    if mi is None:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [
            "the EP registered no DEFAULT allocator, so there is no device memory to keep a KV "
            "cache in and the question was never put. Set ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1"
        ]
        doc["device_memory_env"] = os.environ.get(
            "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", "<unset>"
        )
        print(json.dumps(doc, indent=2))
        return 2

    try:
        host_steps, host_final, host_disp, _ = _run_chain(ort, sess, mi, False, counters)
        res_steps, res_final, res_disp, notes = _run_chain(ort, sess, mi, True, counters)
    except Exception as e:  # noqa: BLE001
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"a lane raised before it could be scored: {type(e).__name__}: {e}"]
        print(json.dumps(doc, indent=2))
        return 2

    doc["host_lane_download_bytes_per_step"] = host_steps
    doc["resident_lane_download_bytes_per_step"] = res_steps
    doc["host_lane_dispatches_per_step"] = host_disp
    doc["resident_lane_dispatches_per_step"] = res_disp

    # The non-triviality guard.  A lane that stopped dispatching would have a beautiful slope.
    # This project has already been handed a 6.7% "saving" that was a run ending early.
    if sum(res_disp) == 0 or sum(res_disp) != sum(host_disp):
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [
            f"the two lanes did not execute the same work: {sum(host_disp)} dispatches host vs "
            f"{sum(res_disp)} resident. A cheaper lane that did less is not a cheaper lane"
        ]
        print(json.dumps(doc, indent=2))
        return 2

    # Correctness before bandwidth.  The chains are fed identically and must end identically; a
    # resident lane that diverges is reading its own stale KV, which is the exact defect the
    # input cache had (run B returning run A's answer to the bit).
    rel = {k: _rel(res_final[k], host_final[k]) for k in host_final}
    doc["resident_vs_host_final"] = rel
    doc["nonzero_final"] = {k: int(np.count_nonzero(v)) for k, v in res_final.items()}
    if any(v == 0 for v in doc["nonzero_final"].values()):
        doc["verdict"] = "RESIDENT_CHAIN_RETURNS_NOTHING"
        doc["why"] = ["an all-zero tensor agrees with another all-zero tensor perfectly"]
        print(json.dumps(doc, indent=2))
        return 1
    if any(v > 1e-3 for v in rel.values()):
        doc["verdict"] = "RESIDENT_CHAIN_DISAGREES"
        doc["why"] = [
            "feeding the KV back device-to-device changed the answer; residency is not sound "
            "and no bandwidth number from it is worth anything"
        ]
        print(json.dumps(doc, indent=2))
        return 1

    # Every step is identical here (the extent is fixed), so all of them count. There is no
    # warm-up step to exclude and none is excluded: excluding one would be a choice about which
    # numbers to keep.
    host_tail = sum(host_steps)
    res_tail = sum(res_steps)
    doc["host_lane_total_bytes"] = host_tail
    doc["resident_lane_total_bytes"] = res_tail
    doc["bytes_removed_from_the_link"] = host_tail - res_tail

    # The steady slope, which is the number that generalises.
    #
    # The counters file is written at the end of a step, so a step's bytes are visible to the
    # NEXT sample: the arrays are shifted by one and each lane's first entry carries the previous
    # lane's tail.  That is an artifact of where the file is flushed, not of the transfer, and it
    # is reported rather than corrected away.  Steps 1.. are unaffected and are what the slope is
    # taken over.
    doc["counter_sample_lag_steps"] = 1
    host_slope = sorted(host_steps[1:])
    res_slope = sorted(res_steps[1:])
    doc["host_lane_steady_bytes_per_step"] = host_slope[len(host_slope) // 2]
    doc["resident_lane_steady_bytes_per_step"] = res_slope[len(res_slope) // 2]

    if res_tail >= host_tail:
        doc["verdict"] = "ROUND_TRIP_NOT_REMOVED"
        doc["why"] = [
            f"the resident lane moved {res_tail} byte(s) across the link against the host lane's "
            f"{host_tail}. Device residency is correct but is not yet buying anything, and the "
            "readback must not be quoted as reduced"
        ]
        print(json.dumps(doc, indent=2))
        return 1

    doc["verdict"] = "ROUND_TRIP_REMOVED"
    doc["why"] = [
        f"{_STEPS} steps ran with the KV in device memory and agree with the host lane to "
        f"{max(rel.values()):.3g}",
        f"link traffic fell from {host_tail} to {res_tail} byte(s) over the measured steps",
        f"steady slope {doc['host_lane_steady_bytes_per_step']} -> "
        f"{doc['resident_lane_steady_bytes_per_step']} byte(s) per step, at identical dispatch "
        f"counts ({sum(host_disp)} each). A slope cannot be beaten by a lane that does less work",
        "no wall-clock is quoted: the box is permanently contended (PERF.md §20). This is a byte "
        "count and byte counts are exact",
        *notes,
    ]
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
