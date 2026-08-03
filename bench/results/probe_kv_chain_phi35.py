"""Does the *real* Phi-3.5 graph decline the KV round trip across a growing decode chain?

`probe_kv_chain_readback.py` answered this on the GQA evidence case and returned
`ROUND_TRIP_REMOVED` — but that case fixes `past` at 4 (ORT rejects any other extent on it,
measured), so its number is a **lower bound at ctx 8192, never that number**.  It proves the
mechanism exists; it cannot prove the mechanism *fires* on a 355-node island with 64 KV outputs
bound by a real caller across a real decode chain.

This probe puts that question, and only that question:

    on the real graph, with `present.{0..31}.{key,value}` bound in this EP's device memory and
    fed straight back as the next step's `past_key_values.{0..31}.{key,value}`, does the
    per-step readback slope fall to zero as the context grows?

# The axis, and why it is a slope

Niobe measured the readback at **393,216 B per past token**, exact to the byte
(32 layers x 2 x 32 heads x 96 dim x 2 B).  In a chain the past grows by one token per step, so
the *shipping* path's per-step readback must rise by exactly 393,216 B per step.  That is a
slope, at identical dispatch counts, on the same session — a total could be beaten by a lane
that did less work; a slope at equal work cannot.

`ROUND_TRIP_NOT_REMOVED` is a real verdict here and it exits non-zero.  A resident lane whose
slope is zero *because it stopped running* is checked against `dispatches_executed`,
`compute_failures` and `device_losses` before its zero is believed — this project has already
been handed one "6.7% saving" that was an observation ending early.

# The three lanes

* **host**     — the shipping path.  `sess.run()`, KV comes back to numpy every step and is
                 re-fed as numpy.  This is what every caller gets today.
* **resident** — `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` + `ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS=1`.
                 The 64 `present.*` outputs are bound to device `OrtValue`s and rebound as the
                 next step's `past`.  Nothing calls `.numpy()` on them, ever.
* **cpu**      — the CPU EP, same chain, same seeds.  Correctness before bandwidth.

`logits` is deliberately left UNBOUND in the resident lane and is read every step (the caller
needs it to pick the next token).  It is a constant 32,064 x sizeof(dtype) per step and does not
grow with the context, so it appears in the intercept and not in the slope — and it doubles as a
liveness control: if the readback axis went to zero because the readback *counter* died rather
than because the transfer did, the logits term would have vanished too.

# What is deliberately NOT claimed

No wall-clock.  The box is permanently contended (`PERF.md` §20) and another team is on the GPU;
that is baseline, not a wait.  Counts, bytes and slopes only.  The device name is read off the
run (`vulkan.device_name` in the EP device metadata), never off the selector — `DEVICE=0` has
run on `1=NVIDIA` on this box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

ONNX_FILE = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
    r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
)
EP_NAME = "VulkanExecutionProvider"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

# From the model's declared shapes and genai_config.json, independent of every counter below.
LAYERS = 32
KV_HEADS = 32
HEAD_DIM = 96
F16 = 2
VOCAB = 32064
BYTES_PER_PAST_TOKEN = LAYERS * 2 * KV_HEADS * HEAD_DIM * F16  # 393,216 — Niobe's slope

SEED_PAST = 4
STEPS = 6


def _lib() -> str:
    return os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )


def _dll_hash() -> str:
    p = pathlib.Path(_lib())
    if not p.is_file():
        return "<absent>"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16].upper()


def _counters(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc.get("counters", doc)


# --------------------------------------------------------------------------- worker


def _worker(lane: str, steps: int, seed_past: int, out_path: pathlib.Path) -> int:
    import numpy as np
    import onnxruntime as ort

    if os.environ.get("PROBE_VERBOSE") == "1":
        # The EP's own log lines (`CopyTensors` release summary, bind decisions) are the only
        # record of WHICH side of the boundary a transfer came from. Off by default: a verbose
        # run is a different run.
        ort.set_default_logger_severity(1)

    rng = np.random.default_rng(20260803)
    doc: dict = {"lane": lane, "steps": steps, "seed_past": seed_past,
                 "ort_version": ort.__version__}

    counters_path = pathlib.Path(os.environ[COUNTERS_ENV]) if COUNTERS_ENV in os.environ else None
    if counters_path is not None:
        counters_path.unlink(missing_ok=True)

    ep_device = None
    if lane != "cpu":
        try:
            ort.register_execution_provider_library(EP_NAME, _lib())
        except Exception as exc:  # noqa: BLE001
            if "already registered" not in str(exc):
                raise
        for d in ort.get_ep_devices():
            if d.ep_name == EP_NAME:
                ep_device = d
                break
        if ep_device is None:
            doc["verdict"] = "ERROR(instrument)"
            doc["why"] = ["the Vulkan EP is not among ORT's EP devices"]
            out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            return 2
        # Read the device off the run, never off the selector: DEVICE=0 has run on 1=NVIDIA.
        doc["ep_device"] = {
            k: ep_device.ep_metadata.get(k)
            for k in ("vulkan.device_name", "vulkan.device_index", "vulkan.vendor_id")
        }

    providers = ["CPUExecutionProvider"] if lane == "cpu" else [EP_NAME, "CPUExecutionProvider"]
    so = ort.SessionOptions()
    sess = ort.InferenceSession(
        str(ONNX_FILE),
        so,
        providers=providers,
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    # ORT's silent CPU retry cannot be allowed to manufacture a success on an EP lane.
    if lane != "cpu" and EP_NAME not in sess.get_providers():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"{EP_NAME} absent from {sess.get_providers()}"]
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return 2

    logits_dtype = np.float16 if "float16" in sess.get_outputs()[0].type else np.float32
    doc["logits_dtype"] = str(np.dtype(logits_dtype))

    mi = None
    if lane in ("resident", "outbind", "outbind_noread", "outbind_readdev", "separating"):
        # Ordering is load-bearing: asking for the allocator before the session exists builds a
        # second VkDevice (SPLIT-DEVICE) that no dispatch can reach.
        mi = ep_device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
        if mi is None:
            doc["verdict"] = "ERROR(instrument)"
            doc["why"] = [
                "the EP registered no DEFAULT allocator, so there is no device memory to keep a "
                "KV cache in and the question was never put",
                f"ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY={os.environ.get('ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY', '<unset>')}",
            ]
            out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            return 2

    # The seed past. Identical bytes in every lane (same rng seed, same draw order), so the three
    # chains are the same chain and any divergence is the writeback path, not the input.
    seed_kv = [
        rng.standard_normal((1, KV_HEADS, seed_past, HEAD_DIM)).astype(np.float16) * 0.02
        for _ in range(LAYERS * 2)
    ]
    doc["seed_kv_sha256"] = hashlib.sha256(
        b"".join(a.tobytes() for a in seed_kv)
    ).hexdigest()[:16]

    past_host = {}
    for layer in range(LAYERS):
        past_host[f"past_key_values.{layer}.key"] = seed_kv[2 * layer]
        past_host[f"past_key_values.{layer}.value"] = seed_kv[2 * layer + 1]

    # Lane taxonomy. `resident` is the question; the other two exist to attribute whatever it
    # costs to a side of the boundary, because a per-step download total cannot tell an input-side
    # transfer from an output-side one when the two spans are the same size (and here they are
    # literally the same buffers).
    #   outbind — outputs bound to device OrtValues, `past` fed from HOST numpy (the constant
    #             seed). Any download it records is output-side by construction, because no
    #             input-side span is one of our handles at all.
    bind_outputs_lane = lane in ("resident", "outbind", "outbind_noread", "outbind_readdev",
                                 "separating")
    device_past_lane = lane in ("resident", "separating")
    # `outbind_noread` reads nothing at all, not even logits. It exists to settle whether an
    # observed transfer is caused by the caller's one legitimate read or by the runtime.
    # `outbind_readdev` reads logits out of a device OrtValue the caller owns, instead of letting
    # ORT copy it into a CPU-bound output. Same one read, different door.
    read_logits = lane != "outbind_noread"
    logits_via_device = lane == "outbind_readdev"

    past_dev = None
    if device_past_lane:
        past_dev = {}
        for name, arr in past_host.items():
            ov = ort.OrtValue.ortvalue_from_shape_and_type(
                list(arr.shape), np.float16, memory_info=mi
            )
            ov.update_inplace(arr)
            past_dev[name] = ov

    if lane == "separating":
        rc = _separating(sess, ort, np, mi, past_host, seed_past, doc, counters_path)
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return rc

    token = 1  # bos
    per_step: list[dict] = []
    logits_sig: list[dict] = []
    logits_full: list[np.ndarray] = []
    prev = _counters(counters_path) if counters_path else {}

    def _delta(now: dict, key: str) -> int:
        return int(now.get(key) or 0) - int(prev.get(key) or 0)

    for step in range(steps):
        # `outbind` does not chain: its `past` is the constant seed, so its extent is fixed and
        # its per-step cost is flat by construction. That is the point — it isolates the
        # output-side term, it is not a decode chain and is never quoted as one.
        past_len = seed_past + step if lane not in ("outbind", "outbind_noread", "outbind_readdev") else seed_past
        input_ids = np.array([[token]], dtype=np.int64)
        attn = np.ones((1, past_len + 1), dtype=np.int64)

        if lane in ("cpu", "host"):
            feeds = {"input_ids": input_ids, "attention_mask": attn}
            feeds.update(past_host)
            outs = sess.run(None, feeds)
            names = [o.name for o in sess.get_outputs()]
            got = dict(zip(names, outs))
            logits = np.asarray(got["logits"])
            past_host = {
                f"past_key_values.{layer}.{kind}": np.asarray(got[f"present.{layer}.{kind}"])
                for layer in range(LAYERS)
                for kind in ("key", "value")
            }
        else:
            binding = sess.io_binding()
            binding.bind_cpu_input("input_ids", input_ids)
            binding.bind_cpu_input("attention_mask", attn)
            if device_past_lane:
                for name, ov in past_dev.items():
                    binding.bind_ortvalue_input(name, ov)
            else:
                # The chain is deliberately broken in `outbind`: `past` is the constant seed, so
                # its extent is fixed and no input-side span is device-authoritative. The answer
                # is not a decode chain; the byte accounting is what this lane is for.
                for name, arr in past_host.items():
                    binding.bind_cpu_input(name, arr)
            present = {}
            if bind_outputs_lane:
                for layer in range(LAYERS):
                    for kind in ("key", "value"):
                        n = f"present.{layer}.{kind}"
                        ov = ort.OrtValue.ortvalue_from_shape_and_type(
                            [1, KV_HEADS, past_len + 1, HEAD_DIM], np.float16, memory_info=mi
                        )
                        binding.bind_ortvalue_output(n, ov)
                        present[n] = ov
                        if os.environ.get("PROBE_VERBOSE") == "1":
                            # Correlates this tensor's NAME with the address the EP logs at its
                            # bind and refresh sites. A byte total cannot tell "one tensor, 64
                            # times" from "64 tensors, once" when every KV tensor is the same
                            # size, and an address can.
                            print(f"PROBE_PTR {n} 0x{ov.data_ptr():x}", flush=True)
            if not bind_outputs_lane:
                for o in sess.get_outputs():
                    if o.name != "logits":
                        binding.bind_output(o.name, "cpu")
            # logits: the caller genuinely needs it, it is a constant per step, and it does not
            # grow with the context, so it lives in the intercept and not in the slope. It also
            # doubles as a liveness control: if the readback axis fell to zero because the counter
            # died rather than because the transfer did, this term would have vanished too.
            # The caller wants logits on the host and the KV on the device. Bind logits to a CPU
            # `OrtValue` this probe owns, so the value read back is unambiguously the one bound.
            #
            # The previous version bound logits with `bind_output("logits", "cpu")` and read
            # `binding.get_outputs()[out_names.index("logits")]`. `get_outputs()` returns outputs
            # in **binding order**, not session order, and logits is the model's output 0 while it
            # was bound last — so index 0 handed back `present.0.key`, and the probe read a KV
            # tensor as its logits. It showed up as an extra 30,720 B download per step and an
            # argmax that disagreed with the host lane at step 0 on identical inputs. The
            # disagreement is what caught it; the byte count alone looked like a runtime defect
            # and was mine.
            if read_logits and not logits_via_device:
                logits_ov = ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((1, 1, VOCAB), dtype=logits_dtype)
                )
                binding.bind_ortvalue_output("logits", logits_ov)
            else:
                logits_ov = ort.OrtValue.ortvalue_from_shape_and_type(
                    [1, 1, VOCAB], logits_dtype, memory_info=mi
                )
                binding.bind_ortvalue_output("logits", logits_ov)
            sess.run_with_iobinding(binding)
            if read_logits:
                logits = np.asarray(logits_ov.numpy())
            else:
                logits = np.zeros((1, 1, VOCAB), dtype=logits_dtype)
            if lane == "resident":
                # The chain: this step's present is the next step's past. Nothing is materialised.
                past_dev = {
                    f"past_key_values.{layer}.{kind}": present[f"present.{layer}.{kind}"]
                    for layer in range(LAYERS)
                    for kind in ("key", "value")
                }

        flat = np.asarray(logits, dtype=np.float64).reshape(-1)[-VOCAB:]
        logits_full.append(flat.copy())
        token = int(np.argmax(flat))
        logits_sig.append(
            {
                "step": step,
                "past_len": past_len,
                "argmax": token,
                "max": float(flat.max()),
                "sum": float(flat.sum()),
                "sha256": hashlib.sha256(np.asarray(logits).tobytes()).hexdigest()[:16],
            }
        )

        now = _counters(counters_path) if counters_path else {}
        per_step.append(
            {
                "step": step,
                "past_len": past_len,
                "readback_bytes": _delta(now, "session_staging_readback_bytes"),
                "readbacks": _delta(now, "session_staging_readbacks"),
                "download_bytes": _delta(now, "alloc_device_download_bytes"),
                "downloads": _delta(now, "alloc_device_downloads"),
                "upload_bytes": _delta(now, "session_staging_upload_bytes"),
                "dispatches": _delta(now, "dispatches_executed"),
            }
        )
        prev = now

    doc["per_step"] = per_step
    doc["logits"] = logits_sig
    # The full logits vectors, so the driver can compare lanes numerically instead of trusting
    # argmax. Argmax agreement is necessary and nowhere near sufficient: it is one index out of
    # 32064 and it survives a great deal of damage to the other 32063.
    np.save(str(out_path) + ".logits.npy", np.stack(logits_full).astype(np.float32))
    if counters_path:
        del sess  # the complete document lands at teardown
        final = _counters(counters_path)
        doc["final_counters"] = {
            k: final.get(k)
            for k in (
                "session_staging_readback_bytes",
                "session_staging_readbacks",
                "session_staging_upload_bytes",
                "alloc_device_download_bytes",
                "alloc_device_downloads",
                "alloc_device_upload_bytes",
                "alloc_device_frame",
                "alloc_device_authority_grants",
                "outputs_device_bound",
                "outputs_device_resident",
                "outputs_host_resident",
                "dispatches_executed",
                "compute_calls",
                "compute_failures",
                "device_losses",
            )
        }
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- separating cases


def _separating(sess, ort, np, mi, past_host, seed_past, doc, counters_path) -> int:
    """The cases that distinguish a correct residency implementation from a lucky one.

    Every one of these is a case the passing lane does not exercise. The chain runs one session,
    one inference at a time, a context that grows by one, and never touches a span while the GPU
    is on it — so the chain would keep passing under a residency scheme that is wrong in all four
    of these ways. That is the whole reason they are here.
    """
    import hashlib as _h

    LOGITS = [1, 1, VOCAB]
    dt = np.float16

    def _dev_past(arrs):
        out = {}
        for name, arr in arrs.items():
            ov = ort.OrtValue.ortvalue_from_shape_and_type(list(arr.shape), dt, memory_info=mi)
            ov.update_inplace(arr)
            out[name] = ov
        return out

    def _step(session, past, past_len, token, keep_present=True):
        b = session.io_binding()
        b.bind_cpu_input("input_ids", np.array([[token]], dtype=np.int64))
        b.bind_cpu_input("attention_mask", np.ones((1, past_len + 1), dtype=np.int64))
        for n, ov in past.items():
            b.bind_ortvalue_input(n, ov)
        present = {}
        for layer in range(LAYERS):
            for kind in ("key", "value"):
                n = f"present.{layer}.{kind}"
                ov = ort.OrtValue.ortvalue_from_shape_and_type(
                    [1, KV_HEADS, past_len + 1, HEAD_DIM], dt, memory_info=mi
                )
                b.bind_ortvalue_output(n, ov)
                if keep_present:
                    present[n] = ov
        lv = ort.OrtValue.ortvalue_from_numpy(np.zeros(LOGITS, dtype=dt))
        b.bind_ortvalue_output("logits", lv)
        session.run_with_iobinding(b)
        logits = np.asarray(lv.numpy(), dtype=np.float64).reshape(-1)[-VOCAB:]
        nxt = {
            f"past_key_values.{layer}.{kind}": present[f"present.{layer}.{kind}"]
            for layer in range(LAYERS)
            for kind in ("key", "value")
        } if keep_present else {}
        return logits, nxt

    def _sig(x):
        return {
            "argmax": int(np.argmax(x)),
            "sum": float(x.sum()),
            "sha256": _h.sha256(x.astype(np.float32).tobytes()).hexdigest()[:16],
            "nonzero_fraction": float(np.count_nonzero(x) / x.size),
        }

    cases: dict[str, dict] = {}

    # ── 1. A session outliving an inference, and outliving the OrtValues it wrote into. ────────
    # The device spans from run N are dropped before run N+1 is issued. If residency held a
    # borrowed pointer into a freed allocation, or if the allocator recycled a span whose
    # authority flag still said "device", this is where it shows.
    dev = _dev_past(past_host)
    a0, nxt = _step(sess, dev, seed_past, 1)
    del nxt
    import gc
    gc.collect()
    a1, _ = _step(sess, dev, seed_past, 1)
    cases["session_outlives_inference"] = {
        "what": "same session, same inputs, run twice with the first run's output OrtValues "
                "dropped and collected in between",
        "first": _sig(a0),
        "second": _sig(a1),
        "pass": bool(np.array_equal(a0, a1)) and np.count_nonzero(a0) > VOCAB // 2,
        "why": "a deterministic graph on identical inputs must return identical bits; a "
               "difference here is a span that was reused or freed under the session",
    }

    # ── 2. Two sessions on one device. ────────────────────────────────────────────────────────
    # Authority is tracked per span. If it were tracked per device, or if two sessions shared a
    # staging frame keyed by address, the second session would corrupt the first.
    sess2 = ort.InferenceSession(
        str(ONNX_FILE), ort.SessionOptions(),
        providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    mi2 = None
    for d in ort.get_ep_devices():
        if d.ep_name == EP_NAME:
            mi2 = d.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
            break
    dev_a = _dev_past(past_host)
    dev_b = {}
    for name, arr in past_host.items():
        ov = ort.OrtValue.ortvalue_from_shape_and_type(list(arr.shape), dt, memory_info=mi2)
        ov.update_inplace(arr)
        dev_b[name] = ov
    b0, _ = _step(sess2, dev_b, seed_past, 1)
    a2, _ = _step(sess, dev_a, seed_past, 1)
    b1, _ = _step(sess2, dev_b, seed_past, 1)
    cases["two_sessions_one_device"] = {
        "what": "a second session on the same device, interleaved run-for-run with the first",
        "session1": _sig(a2),
        "session2_before": _sig(b0),
        "session2_after": _sig(b1),
        "pass": bool(np.array_equal(b0, b1)) and bool(np.array_equal(a0, a2)),
        "why": "neither session may perturb the other's spans; both must still reproduce their "
               "own single-session answer bit for bit",
    }
    del sess2

    # ── 3. A context outgrowing its first allocation. ─────────────────────────────────────────
    # Each step's `present` is one token longer than the last, so every step allocates a span
    # larger than any span in the pool. A residency scheme that assumed a fixed extent, or that
    # cached a bind target by size, breaks here and nowhere in a fixed-extent probe.
    dev = _dev_past(past_host)
    grow = []
    tok = 1
    for i in range(6):
        lg, dev = _step(sess, dev, seed_past + i, tok)
        tok = int(np.argmax(lg))
        grow.append({"past_len": seed_past + i, **_sig(lg)})
    extents = [g["past_len"] for g in grow]
    cases["context_outgrows_first_allocation"] = {
        "what": f"the past grows {extents[0]} -> {extents[-1] + 1} tokens, a new and larger span "
                "allocated on every one of 6 steps",
        "steps": grow,
        "pass": all(g["nonzero_fraction"] > 0.5 for g in grow)
                and len({g["argmax"] for g in grow}) > 1,
        "why": "a growing extent must keep producing live, varying logits; a constant or dead "
               "output means the larger span was never actually read",
    }

    # ── 4. A readback taken while a dispatch is in flight. ────────────────────────────────────
    # Today this is safe because the fence is waited before Compute returns — but NOTHING
    # asserts it, so it is asserted here: issue the run, and read a device span the kernel wrote
    # at the first instant the API allows, with no synchronisation of our own.
    dev = _dev_past(past_host)
    b = sess.io_binding()
    b.bind_cpu_input("input_ids", np.array([[1]], dtype=np.int64))
    b.bind_cpu_input("attention_mask", np.ones((1, seed_past + 1), dtype=np.int64))
    for n, ov in dev.items():
        b.bind_ortvalue_input(n, ov)
    watch = ort.OrtValue.ortvalue_from_shape_and_type(
        [1, KV_HEADS, seed_past + 1, HEAD_DIM], dt, memory_info=mi
    )
    b.bind_ortvalue_output("present.0.key", watch)
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            n = f"present.{layer}.{kind}"
            if n == "present.0.key":
                continue
            b.bind_ortvalue_output(
                n,
                ort.OrtValue.ortvalue_from_shape_and_type(
                    [1, KV_HEADS, seed_past + 1, HEAD_DIM], dt, memory_info=mi
                ),
            )
    lv = ort.OrtValue.ortvalue_from_numpy(np.zeros(LOGITS, dtype=dt))
    b.bind_ortvalue_output("logits", lv)
    sess.run_with_iobinding(b)
    immediate = np.asarray(watch.numpy()).astype(np.float32).copy()
    settled = np.asarray(watch.numpy()).astype(np.float32).copy()
    cases["readback_during_dispatch"] = {
        "what": "present.0.key read at the first instant the API permits after run_with_iobinding "
                "returns, with no synchronisation by the caller, then read again",
        "immediate_sha256": _h.sha256(immediate.tobytes()).hexdigest()[:16],
        "settled_sha256": _h.sha256(settled.tobytes()).hexdigest()[:16],
        "immediate_nonzero_fraction": float(np.count_nonzero(immediate) / immediate.size),
        "pass": bool(np.array_equal(immediate, settled))
                and np.count_nonzero(immediate) > immediate.size // 2,
        "why": "the first read must already see the finished kernel's bytes. If it differs from "
               "the second read, or is mostly zero, the fence is not being waited and every "
               "byte count in this probe was taken against a race",
    }

    doc["separating_cases"] = cases
    failed = [k for k, v in cases.items() if not v["pass"]]
    doc["separating_failed"] = failed
    if counters_path:
        fc = _counters(counters_path)
        doc["final_counters"] = {
            k: fc.get(k) for k in ("dispatches_executed", "compute_calls", "compute_failures",
                                   "device_losses", "alloc_device_frame",
                                   "session_staging_readback_bytes",
                                   "alloc_device_download_bytes", "outputs_device_bound")
        }
    if failed:
        doc["verdict"] = "SEPARATING_CASE_FAILED"
        doc["why"] = [f"{k}: {cases[k]['why']}" for k in failed]
        return 1
    doc["verdict"] = "SEPARATING_CASES_PASS"
    return 0


# --------------------------------------------------------------------------- driver


def _run_lane(lane: str, steps: int, seed_past: int, scratch: pathlib.Path) -> dict:
    out = scratch / f"phi35_chain_{lane}.json"
    counters = scratch / f"phi35_chain_{lane}.counters.json"
    out.unlink(missing_ok=True)
    counters.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(counters)
    env["ONNXRUNTIME_VULKAN_EP_LIB"] = _lib()
    if lane in ("resident", "outbind", "outbind_noread", "outbind_readdev", "separating"):
        env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
        # `PROBE_LEAVE_BIND_DEFAULT=1` runs these lanes with `ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS`
        # *unset*, so they exercise the shipped default rather than an explicit request. Since
        # 2026-08-03 the default is ON; before that it was OFF and these lanes needed the `=1`.
        # Keeping both spellings runnable is the point: a lane that can only be entered by asking
        # for it cannot tell you what a user who asks for nothing gets.
        if os.environ.get("PROBE_LEAVE_BIND_DEFAULT") == "1":
            env.pop("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", None)
        else:
            env["ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS"] = "1"
    else:
        env.pop("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", None)
        env.pop("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", None)
    cmd = [
        sys.executable, str(pathlib.Path(__file__).resolve()),
        "--worker", "--lane", lane, "--steps", str(steps),
        "--seed-past", str(seed_past), "--out", str(out),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True)
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    doc["exit_code"] = proc.returncode
    doc["logits_npy"] = str(out) + ".logits.npy"
    if proc.returncode != 0:
        doc.setdefault("verdict", "ERROR(instrument)")
        doc["stderr_tail"] = (proc.stderr or b"").decode("utf-8", "replace")[-2000:]
    return doc


def _slope(points: list[tuple[int, int]]) -> float:
    """Least-squares slope of y against x. Two or more points required."""
    n = len(points)
    if n < 2:
        return float("nan")
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    if den == 0:
        return float("nan")
    return (n * sxy - sx * sy) / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--lane", default="host")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed-past", type=int, default=SEED_PAST)
    ap.add_argument("--out", default="")
    ap.add_argument("--lanes", default="host,resident,cpu")
    args = ap.parse_args()

    if args.worker:
        return _worker(args.lane, args.steps, args.seed_past, pathlib.Path(args.out))

    scratch = HERE
    doc: dict = {
        "probe": "does the real Phi-3.5 graph decline the KV round trip across a decode chain?",
        "model": str(ONNX_FILE),
        "dll_sha256_16": _dll_hash(),
        "steps": args.steps,
        "seed_past": args.seed_past,
        "bytes_per_past_token_declared": BYTES_PER_PAST_TOKEN,
    }
    if not ONNX_FILE.is_file():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"the model is missing at {ONNX_FILE}"]
        print(json.dumps(doc, indent=2))
        return 2

    lanes = {}
    for lane in args.lanes.split(","):
        lane = lane.strip()
        if lane:
            lanes[lane] = _run_lane(lane, args.steps, args.seed_past, scratch)
    doc["lanes"] = lanes

    rc = _score(doc, lanes)
    # `--out` names the top-level record in parent mode too. Without this the artifact path was a
    # constant, so any re-run on a different device or context length silently overwrote the
    # committed record of a run nobody re-took -- the file kept the name of the evidence and the
    # contents of the last invocation. Observed: a ctx-512 run on the UMA device replaced the
    # committed RTX 4060 seed_past=4 record.
    out_path = pathlib.Path(args.out) if args.out else (HERE / "phi35_kv_chain.json")
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps(doc, indent=2))
    return rc


def _link_bytes(step: dict) -> int:
    """Every byte that crossed the PCIe link on this step, from either accounting.

    `session_staging_readback_bytes` is the EP's own output readback; `alloc_device_download_bytes`
    is the device→host traffic a *reader* causes. They are separate counters with separate
    producers and summing them is the only honest total: a change that moves bytes from one to the
    other has moved the transfer, not removed it, and this project has shipped that mistake before.
    """
    return int(step.get("readback_bytes") or 0) + int(step.get("download_bytes") or 0)


def _compare_logits(a: dict, b: dict) -> dict | None:
    """Numeric agreement between two lanes' logits, with the degeneracy guard.

    A relative metric scores a perfect 1.0 when both sides are all zeros, and a "max relative
    difference" of 0.0 is exactly what a dead output looks like. So the nonzero count is reported
    alongside, and a lane whose logits are constant is called out rather than congratulated.
    """
    pa, pb = a.get("logits_npy"), b.get("logits_npy")
    if not (pa and pb and pathlib.Path(pa).is_file() and pathlib.Path(pb).is_file()):
        return None
    import numpy as np

    xa, xb = np.load(pa), np.load(pb)
    n = min(len(xa), len(xb))
    xa, xb = xa[:n].astype(np.float64), xb[:n].astype(np.float64)
    diff = np.abs(xa - xb)
    scale = np.maximum(np.abs(xa), np.abs(xb))
    rel = np.where(scale > 0, diff / np.maximum(scale, 1e-30), 0.0)
    return {
        "steps_compared": int(n),
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float(rel.max()),
        "mean_abs_diff": float(diff.mean()),
        "bitwise_identical": bool(np.array_equal(xa, xb)),
        "nonzero_fraction_a": float(np.count_nonzero(xa) / xa.size),
        "nonzero_fraction_b": float(np.count_nonzero(xb) / xb.size),
        "distinct_values_a": int(len(np.unique(xa))),
        "distinct_values_b": int(len(np.unique(xb))),
    }


def _score(doc: dict, lanes: dict) -> int:
    why: list[str] = []
    if "host" not in lanes:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = ["the `host` control lane was not run; there is nothing to compare against"]
        return 2

    # ── The separating cases are their own verdict: they are correctness, not bandwidth, and a
    # failure in any of them invalidates every byte count in the run. Checked first. ───────────
    if "separating" in lanes:
        sep = lanes["separating"]
        lanes = {k: v for k, v in lanes.items() if k != "separating"}
        doc["separating_cases"] = sep.get("separating_cases")
        doc["separating_verdict"] = sep.get("verdict")
        if sep.get("separating_failed") or sep.get("exit_code"):
            doc["verdict"] = "SEPARATING_CASE_FAILED"
            doc["why"] = sep.get("why") or [sep.get("stderr_tail", "")[-800:]]
            return 1
        why.append(
            "the four separating cases pass: a session outliving an inference (and outliving the "
            "OrtValues it wrote), two sessions on one device interleaved run-for-run, a context "
            "outgrowing its first allocation over 6 growing spans, and a readback taken at the "
            "first instant the API permits after the run returns. None of these is exercised by "
            "the chain itself, which is why they are here"
        )

    # ── Non-triviality. A lane whose slope is zero because it stopped running has a beautiful
    # slope. Checked before any byte is believed. ──────────────────────────────────────────────
    for name, ln in lanes.items():
        fc = ln.get("final_counters") or {}
        if ln.get("exit_code"):
            doc["verdict"] = "ERROR(instrument)"
            doc["why"] = [f"lane {name} exited {ln['exit_code']}", ln.get("stderr_tail", "")]
            return 2
        if name == "cpu":
            continue
        if fc.get("compute_failures") or fc.get("device_losses"):
            doc["verdict"] = "ERROR(instrument)"
            doc["why"] = [
                f"lane {name} recorded compute_failures={fc.get('compute_failures')} "
                f"device_losses={fc.get('device_losses')}; a run that ended early is not a "
                "cheaper run"
            ]
            return 2
        if not fc.get("dispatches_executed"):
            doc["verdict"] = "ERROR(instrument)"
            doc["why"] = [f"lane {name} executed 0 dispatches — nothing ran on the device"]
            return 2

    host = lanes["host"]

    # ── Correctness before bandwidth. Against the CPU EP first: a different kernel on different
    # arithmetic, so the criterion is agreement in the answer, not in the bits. ────────────────
    if "cpu" in lanes:
        cpu = lanes["cpu"]
        doc["host_vs_cpu_logits"] = [
            {
                "step": a["step"],
                "ep_argmax": a["argmax"],
                "cpu_argmax": c["argmax"],
                "agree": a["argmax"] == c["argmax"],
            }
            for a, c in zip(host["logits"], cpu["logits"])
        ]
        doc["host_vs_cpu_numeric"] = _compare_logits(host, cpu)
        if doc["host_vs_cpu_numeric"]:
            doc["host_vs_cpu_numeric"]["note"] = (
                "the criterion here is agreement in the ANSWER, not in the bits: the CPU EP runs "
                "different kernels with different accumulation, so bitwise equality is not the "
                "standard and max_rel_diff is dominated by logits near zero where a relative "
                "measure is meaningless. The token chain is the load-bearing comparison"
            )
        bad = [r["step"] for r in doc["host_vs_cpu_logits"] if not r["agree"]]
        nz = (doc["host_vs_cpu_numeric"] or {}).get("nonzero_fraction_a", 0.0)
        if nz < 0.5:
            doc["verdict"] = "ERROR(instrument)"
            doc["why"] = [
                f"the EP lane's logits are {nz:.3f} nonzero — a mostly-dead output agrees with "
                "anything on a relative metric, so no agreement number from this run is evidence"
            ]
            return 2
        if bad:
            doc["verdict"] = "EP_DISAGREES_WITH_CPU"
            doc["why"] = [
                f"the Vulkan EP and the CPU EP chose different tokens at steps {bad} from the "
                "same seed. Correctness comes before bandwidth; no byte count here is quotable"
            ]
            return 1

    if "resident" not in lanes:
        doc["verdict"] = "CORRECTNESS_ONLY"
        doc["why"] = [
            "the `resident` lane was not run, so nothing is claimed about the round trip. This "
            "run is a correctness check only"
        ]
        return 0

    res = lanes["resident"]
    hd = [int(s["dispatches"]) for s in host["per_step"]]
    rd = [int(s["dispatches"]) for s in res["per_step"]]
    doc["dispatches_per_step"] = {"host": hd, "resident": rd}
    if hd != rd:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [
            f"the lanes did not execute the same work: {hd} host vs {rd} resident. A slope is "
            "only a slope at equal work"
        ]
        return 2

    # ── Then the two writeback paths against each other. Same session, same kernel, same seed KV,
    # same token — the ONLY difference is where the outputs land. Anything but agreement to the
    # bit is the writeback path changing the answer, and here the criterion really is the bits. ─
    cmp_rows = []
    for a, b in zip(host["logits"], res["logits"]):
        cmp_rows.append(
            {
                "step": a["step"],
                "host_argmax": a["argmax"],
                "resident_argmax": b["argmax"],
                "agree": a["argmax"] == b["argmax"],
                "host_sha256": a["sha256"],
                "resident_sha256": b["sha256"],
                "bitwise_identical": a["sha256"] == b["sha256"],
            }
        )
    doc["resident_vs_host_logits"] = cmp_rows
    doc["resident_vs_host_numeric"] = _compare_logits(host, res)
    if not all(r["bitwise_identical"] for r in cmp_rows):
        doc["verdict"] = "RESIDENT_CHAIN_DISAGREES"
        doc["why"] = [
            "the two lanes are the same inference through two writeback paths — same session, "
            "same seed KV, same token chain — and they returned different logits at steps "
            + str([r["step"] for r in cmp_rows if not r["bitwise_identical"]])
            + ". No bandwidth number from this lane is worth anything until that is explained"
        ]
        return 1

    # ── The slope. Per past token, at equal work. ─────────────────────────────────────────────
    # The counters file is flushed on the dispatch path, so a transfer that happens *after* a
    # step's last dispatch lands in the next step's sample. Step 0 is therefore short by one
    # step's readout in every lane, and the slope is taken over steps 1.. rather than corrected.
    doc["counter_sample_lag_steps"] = 1
    host_pts = [(int(s["past_len"]), _link_bytes(s)) for s in host["per_step"][1:]]
    res_pts = [(int(s["past_len"]), _link_bytes(s)) for s in res["per_step"][1:]]
    doc["host_lane_link_bytes_per_step"] = [_link_bytes(s) for s in host["per_step"]]
    doc["resident_lane_link_bytes_per_step"] = [_link_bytes(s) for s in res["per_step"]]
    host_slope = _slope(host_pts)
    res_slope = _slope(res_pts)
    doc["host_lane_slope_bytes_per_past_token"] = host_slope
    doc["resident_lane_slope_bytes_per_past_token"] = res_slope
    doc["declared_slope_bytes_per_past_token"] = BYTES_PER_PAST_TOKEN
    doc["slope_points_after_lag"] = {"host": len(host_pts), "resident": len(res_pts)}

    # A slope that is not a number must not reach a comparison. `_slope` returns NaN on fewer
    # than two points, and after the one-step counter lag a `--steps 2` run has exactly one —
    # so `abs(nan - 393216) > 1` is **False**, the host-lane control passes, `nan >= nan` is
    # False, and the run certifies ROUND_TRIP_REMOVED while printing "reproduces Niobe's slope
    # to the byte: nan". Measured 2026-08-03 at `--seed-past 2048 --steps 2`; the per-step byte
    # counts in that run were correct and the verdict was reached without them.
    #
    # Every comparison in this scorer is `>` or `>=`, and every one of them is False against NaN.
    # A guard that silently answers "no objection" to each of a series of independent checks is
    # not a series of checks. Refuse first, compare after.
    if not all(map(math.isfinite, (host_slope, res_slope))):
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [
            f"a slope is not a number: host={host_slope}, resident={res_slope}, from "
            f"{len(host_pts)}/{len(res_pts)} usable points after the {doc['counter_sample_lag_steps']}"
            "-step counter lag. Two points are the minimum and every threshold below compares "
            "False against NaN, so this would have been certified rather than caught. Run at "
            "least 3 steps.",
            "the per-step byte counts in this run are still valid and are reported above; what "
            "is missing is the line through them",
        ]
        return 2

    for extra in ("outbind", "outbind_noread", "outbind_readdev"):
        if extra in lanes:
            doc.setdefault("isolation_lanes", {})[extra] = {
                "link_bytes_per_step": [_link_bytes(s) for s in lanes[extra]["per_step"]],
                "downloads_per_step": [s["downloads"] for s in lanes[extra]["per_step"]],
            }

    # The host lane is the control that says the instrument can see the term at all: it must
    # reproduce Niobe's 393,216 B per past token. A resident-lane zero against a host lane that
    # also reads zero would mean the counter died, not the transfer.
    if abs(host_slope - BYTES_PER_PAST_TOKEN) > 1:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [
            f"the shipping lane's slope is {host_slope} B per past token, not the declared "
            f"{BYTES_PER_PAST_TOKEN}. The axis this probe reads is not the axis the model's KV "
            "traffic is on, so a fall in it would mean nothing"
        ]
        return 2

    if res_slope >= host_slope:
        doc["verdict"] = "ROUND_TRIP_NOT_REMOVED"
        doc["why"] = [
            f"the resident lane's slope is {res_slope} B per past token against the shipping "
            f"lane's {host_slope}. Device residency is correct on the real graph and is not yet "
            "buying anything; the readback must not be quoted as reduced"
        ]
        return 1

    why.append(
        f"the shipping lane reproduces Niobe's slope to the byte: {host_slope} B per past token "
        f"(declared {BYTES_PER_PAST_TOKEN})"
    )
    if res_slope <= 0:
        why.append(
            "with the 64 KV outputs bound in this EP's device memory and fed straight back as "
            "the next step's past, the slope is flat: 0 B per past token. The lane's link traffic "
            f"is {doc['resident_lane_link_bytes_per_step'][-1]} B on every step regardless of "
            "context length, at identical dispatch counts "
            f"({sum(hd)} each). A ratio is not quoted against zero; the shape of the line is the "
            "finding — the shipping path has a slope and this one does not"
        )
    else:
        why.append(
            f"with the 64 KV outputs bound in this EP's device memory and fed straight back as "
            f"the next step's past, the slope is {res_slope} B per past token — "
            f"{host_slope / res_slope:.1f}x lower, at identical dispatch counts "
            f"({sum(hd)} each)"
        )
    why.append(
        "this is a real 355-node island with 64 KV outputs over a real decode chain, not the "
        "fixed-extent GQA case: the past grows by one token every step and the extent is "
        f"{host['per_step'][0]['past_len']} -> {host['per_step'][-1]['past_len']}"
    )
    why.append(
        "no wall-clock is quoted: the box is permanently contended (PERF.md §20) and another "
        "team is on the GPU. This is a byte count and byte counts are exact"
    )
    if res_slope > 0:
        why.append(
            f"the resident lane's slope is NOT zero: {res_slope} B per past token remains, and it "
            "is disclosed rather than rounded away — see the isolation lanes for what it is"
        )
    else:
        named = (
            f"the flat term that remains is {doc['resident_lane_link_bytes_per_step'][-1]} B per "
            "step and it is named, not rounded away: it is `logits`, 32064 x fp16 = 64,128 B, "
            "which the caller asked for"
        )
        if "outbind_noread" in lanes:
            nr = doc.get("isolation_lanes", {}).get("outbind_noread", {})
            named += (
                ". The `outbind_noread` lane reads nothing and pays "
                f"{max(nr.get('link_bytes_per_step', [0]))} B and "
                f"{max(nr.get('downloads_per_step', [0]))} downloads over the same steps, which "
                "is what proves the remaining term is the caller's request and not the runtime's"
            )
        else:
            named += (
                ". The `outbind_noread` lane, which is what would PROVE that term is the "
                "caller's request rather than the runtime's, was not run here"
            )
        why.append(named)
    why.append(
        "the grouping this fires under is NOT the general case: Phi-3.5-mini has 32 KV heads for "
        "32 query heads, Nq/Nkv = 1.00, the degenerate ratio where the grouping defect is "
        "invisible. It is 4x on Llama-3 8B. Nothing here is keyed on head counts, but the "
        "general grouping case is UNTESTED and this number must not be extrapolated to it"
    )
    doc["device_read_off_the_run"] = (host.get("ep_device") or {}).get("vulkan.device_name")
    why.append(
        "the device is read off the run, not off the selector: "
        f"{doc['device_read_off_the_run']} "
        f"(index {(host.get('ep_device') or {}).get('vulkan.device_index')}, vendor "
        f"{(host.get('ep_device') or {}).get('vulkan.vendor_id')}), DLL sha256[:16] "
        f"{doc.get('dll_sha256_16')}"
    )
    doc["verdict"] = "ROUND_TRIP_REMOVED"
    doc["why"] = why
    return 0


if __name__ == "__main__":
    sys.exit(main())
