"""Is the KV **arena** — `present` aliasing `past` in one fixed allocation — reachable at all?

The arena is the last of the three levers on the original KV ledger and the only path to
ctx 8192 on an 8 GB card: the shipping (growing) convention holds `past` (C tokens) and
`present` (C+1 tokens) alive at the same instant, so peak KV is `2 x 393,216 x C`, and
`2 x 393,216 x 8192 + 2.29 GB weights = 8.73 GB` does not fit.  One shared allocation is
`1 x 393,216 x 8192 + 2.29 GB = 5.51 GB`, which does.  That arithmetic was written down and
then measured last round: 6144 is the largest context this box has reached, resident lane only.

# The question this probe puts, and why it comes before any EP code

The EP already implements the shared-buffer convention (`shares_past_buffer` in
`attention.rs`, `present_len == past_stride` in `gqa_f16.comp`).  Nothing in the kernel has to
change to make `present` alias `past`.  What has never been established is whether the
*graph* will accept the input that convention requires:

    a `past_key_values.L.{key,value}` whose declared extent is the ARENA (fixed, large),
    while the true past length is carried by `attention_mask` alone.

If ORT refuses the bind, or if the model's own GQA reads its past length off the past
tensor's shape rather than off `seqlens_k`, then the arena is not an EP change at all — it is
a property of how the model was exported, and no amount of Vulkan work reaches it.  That is a
finding about the model, not a failure, and it is cheaper to learn here than after a week of
allocator work.

**Prediction, written before the first run** (2026-08-03, Switch):

  * `accepts` — ORT will accept the oversized `past` bind.  Symbolic-dimension binding does
    not cross-check `past_sequence_length` against the `attention_mask` length; they are
    different symbols and ORT resolves each from the value that carries it.
  * `honours` — I do **not** know which way this goes, and that is the whole point of the
    run.  ORT's CPU GQA supports `past_present_share_buffer`, but the Phi-3.5 export declares
    the growing convention, and a kernel written for the growing convention has no reason to
    consult `seqlens_k` for the *past* extent when the past tensor's own shape states it.
    If the logits from the arena-shaped run differ from the true-shaped run, the export does
    not support it.

The two are reported separately because they fail for different reasons and only the second
is about attention semantics.

# Method

One CPU-EP session, two runs, identical everything except the shape of `past`:

  * `tight`  — `past` at `[1, 32, P, 96]`, the real P tokens.  This is the shipping shape.
  * `arena`  — `past` at `[1, 32, A, 96]` for `A > P`, first P token slots byte-identical to
    `tight`, the tail filled with a value that is *not* zero (zeros are the one filler a
    buggy reader could consume without changing the answer — the arena must be able to
    poison a wrong read, or a pass proves nothing).

`attention_mask` is `[1, P+1]` in both, so `seqlens_k` is P in both.  The comparison is
bit-level on the whole logits vector, not argmax: argmax is one index of 32,064 and survives
a great deal of damage to the other 32,063.

No clock.  Counts, bytes, shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
EP_NAME = "VulkanExecutionProvider"

# ARCHIVAL: pinned to the exact Foundry cache layout this investigation measured against
# (the pre-2026-08-05 "...-cuda-gpu/cuda-int4-rtn-block-32/..." catalog revision, issue
# #11). Intentionally NOT auto-resolved against the live cache: a live resolver could
# silently pick a *different* cached revision than the one this result was measured
# against, which would misattribute a new run to an old artifact. Override PHI35_MODEL to
# replay against a different artifact explicitly (issue #19).
ONNX_FILE = pathlib.Path(
    os.environ.get(
        "PHI35_MODEL",
        r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
        r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
        r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    )
)

LAYERS = 32
KV_HEADS = 32
HEAD_DIM = 96
VOCAB = 32064
BYTES_PER_PAST_TOKEN = LAYERS * 2 * KV_HEADS * HEAD_DIM * 2  # 393,216

PREDICTION = {
    "accepts": "ACCEPT — symbolic dims are resolved per-value, not cross-checked",
    "honours": "UNKNOWN — this is the question; a growing-convention export has no reason "
               "to read the past extent off seqlens_k",
}


def _sig(a) -> dict:
    import numpy as np

    flat = np.asarray(a, dtype=np.float64).reshape(-1)[-VOCAB:]
    return {
        "argmax": int(flat.argmax()),
        "max": float(flat.max()),
        "sum": float(flat.sum()),
        "sha256": hashlib.sha256(np.asarray(a).tobytes()).hexdigest()[:16],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="graph", choices=("graph", "chain"))
    ap.add_argument("--past", type=int, default=4, help="true past token count P")
    ap.add_argument("--arena", type=int, default=16, help="arena extent A (A > P)")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lane", default="")
    ap.add_argument("--lanes", default="",
                    help="comma-separated subset of host,grow,arena. The capacity runs omit "
                         "`host`: a CPU-EP decode over an 8k past is a different experiment, "
                         "and correctness is established at small A where it is cheap.")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.mode == "chain":
        return chain_main(args)
    return graph_main(args)


def graph_main(args) -> int:
    import numpy as np
    import onnxruntime as ort

    P, A = args.past, args.arena
    if A <= P:
        print("arena must exceed past", file=sys.stderr)
        return 2

    doc: dict = {
        "probe": "kv_arena_graph_accepts",
        "ort_version": ort.__version__,
        "model": str(ONNX_FILE),
        "past": P,
        "arena": A,
        "prediction": PREDICTION,
        "bytes_per_past_token": BYTES_PER_PAST_TOKEN,
    }

    rng = np.random.default_rng(20260803)
    real = [
        (rng.standard_normal((1, KV_HEADS, P, HEAD_DIM)).astype(np.float16) * 0.02)
        for _ in range(LAYERS * 2)
    ]
    # The arena tail is poisoned, not zeroed: a reader that consumes it must change the answer,
    # otherwise "the answers agreed" is compatible with "the tail was read and happened not to
    # matter". 0.5 in f16 is exact and is ~25 sigma from the real KV distribution.
    arena = []
    for a in real:
        buf = np.full((1, KV_HEADS, A, HEAD_DIM), np.float16(0.5), dtype=np.float16)
        buf[:, :, :P, :] = a
        arena.append(buf)

    sess = ort.InferenceSession(
        str(ONNX_FILE),
        ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )

    input_ids = np.array([[1]], dtype=np.int64)
    attn = np.ones((1, P + 1), dtype=np.int64)

    def feeds(kv):
        f = {"input_ids": input_ids, "attention_mask": attn}
        for layer in range(LAYERS):
            f[f"past_key_values.{layer}.key"] = kv[2 * layer]
            f[f"past_key_values.{layer}.value"] = kv[2 * layer + 1]
        return f

    names = [o.name for o in sess.get_outputs()]

    lanes = {}
    for lane, kv in (("tight", real), ("arena", arena)):
        try:
            outs = sess.run(None, feeds(kv))
        except Exception as exc:  # noqa: BLE001
            lanes[lane] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
            continue
        got = dict(zip(names, outs))
        rec = {
            "logits": _sig(got["logits"]),
            "present_shape": list(np.asarray(got["present.0.key"]).shape),
            "past_shape": list(kv[0].shape),
        }
        rec["_full"] = np.asarray(got["logits"], dtype=np.float64).reshape(-1)[-VOCAB:]
        lanes[lane] = rec

    if "error" in lanes["arena"]:
        doc["accepts"] = "REFUSED"
        doc["why"] = [lanes["arena"]["error"]]
        doc["verdict"] = "ARENA_SHAPE_REFUSED_BY_GRAPH"
    else:
        doc["accepts"] = "ACCEPTED"
        t = lanes["tight"]["_full"]
        a = lanes["arena"]["_full"]
        max_abs = float(np.max(np.abs(t - a)))
        denom = np.maximum(np.abs(t), 1e-6)
        worst_rel = float(np.max(np.abs(t - a) / denom))
        doc["compare"] = {
            "max_abs": max_abs,
            "worst_rel": worst_rel,
            "bitwise_equal": bool(np.array_equal(t, a)),
            "argmax_tight": lanes["tight"]["logits"]["argmax"],
            "argmax_arena": lanes["arena"]["logits"]["argmax"],
        }
        # A degeneracy guard: two all-zero logits vectors agree perfectly and mean nothing.
        doc["degeneracy"] = {
            "nonzero_frac_tight": float(np.count_nonzero(t) / t.size),
            "distinct_tight": int(np.unique(t).size),
        }
        if doc["degeneracy"]["nonzero_frac_tight"] < 0.5 or doc["degeneracy"]["distinct_tight"] < 100:
            doc["verdict"] = "NOT_PERFORMED(degenerate logits)"
        elif doc["compare"]["bitwise_equal"]:
            doc["verdict"] = "ARENA_SHAPE_HONOURED_BITWISE"
        elif worst_rel < 1e-3:
            doc["verdict"] = "ARENA_SHAPE_HONOURED_APPROX"
        else:
            doc["verdict"] = "ARENA_SHAPE_ACCEPTED_BUT_NOT_HONOURED"

    for lane in lanes.values():
        lane.pop("_full", None)
    doc["lanes"] = lanes

    out = pathlib.Path(args.out or str(HERE / "kv_arena_graph_accepts.json"))
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k != "lanes"}, indent=2))
    print(f"[wrote] {out}")
    return 0 if doc.get("verdict", "").startswith("ARENA_SHAPE_HONOURED") else 1


# =========================================================================== chain
#
# The arena on the real graph, in this EP's device memory, against the shipping lane.
#
# Three lanes, all six steps, one seed:
#
#   host    — the shipping path: growing `past`, numpy in and numpy out. Peak KV is
#             `past` + `present` alive at once, which is the 2x that does not fit at 8192.
#   grow    — device-resident, growing convention (`KV_ARENA` unset). Last round's lane.
#   arena   — device-resident, `KV_ARENA=1`. One allocation per KV tensor for the whole
#             session; `present` IS `past`.
#
# The correctness control is read BEFORE the byte count, and it is not the logits alone: at
# the end of the chain every one of the 64 KV tensors is downloaded and its *valid* region
# compared against the host lane's, byte for byte. A fabricated residual and a fabricated
# correctness defect both came out of this probe family last round and neither survived
# comparing two lanes computing the same inference.


def _counters(path) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc.get("counters", doc)


def _chain_worker(args) -> int:
    import numpy as np
    import onnxruntime as ort

    lane, A, P, steps = args.lane, args.arena, args.past, args.steps
    out_path = pathlib.Path(args.out)
    doc: dict = {"lane": lane, "arena": A, "seed_past": P, "steps": steps,
                 "ort_version": ort.__version__}
    counters_path = pathlib.Path(os.environ[COUNTERS_ENV]) if COUNTERS_ENV in os.environ else None
    if counters_path is not None:
        counters_path.unlink(missing_ok=True)

    ep_device = None
    if lane != "host":
        try:
            ort.register_execution_provider_library(EP_NAME, str(
                os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB",
                               REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll")))
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
        # The device is read off the run, never off the selector.
        doc["ep_device"] = {
            k: ep_device.ep_metadata.get(k)
            for k in ("vulkan.device_name", "vulkan.device_index", "vulkan.vendor_id")
        }

    providers = ["CPUExecutionProvider"] if lane == "cpu" else [EP_NAME, "CPUExecutionProvider"]
    sess = ort.InferenceSession(
        str(ONNX_FILE), ort.SessionOptions(), providers=providers,
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    if lane not in ("host", "cpu") and EP_NAME not in sess.get_providers():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"{EP_NAME} absent from {sess.get_providers()}"]
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return 2

    logits_dtype = np.float16 if "float16" in sess.get_outputs()[0].type else np.float32
    mi = None
    if lane in ("grow", "arena", "growflag"):
        # Ordering is load-bearing: the allocator asked for before the session exists builds a
        # second VkDevice no dispatch can reach. `bind_target_for` refuses that frame, which is
        # a decline, not a wrong answer — but the run would be about nothing.
        mi = ep_device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
        if mi is None:
            doc["verdict"] = "ERROR(instrument)"
            doc["why"] = ["the EP registered no DEFAULT allocator"]
            out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            return 2

    rng = np.random.default_rng(20260803)
    seed_kv = [rng.standard_normal((1, KV_HEADS, P, HEAD_DIM)).astype(np.float16) * 0.02
               for _ in range(LAYERS * 2)]
    doc["seed_kv_sha256"] = hashlib.sha256(b"".join(a.tobytes() for a in seed_kv)).hexdigest()[:16]
    kv_names = [f"{layer}.{kind}" for layer in range(LAYERS) for kind in ("key", "value")]

    past_host = {f"past_key_values.{n}": seed_kv[i] for i, n in enumerate(kv_names)}

    # The arena. One allocation per KV tensor for the whole session, `A` tokens of stride and
    # one spare token of capacity so that the `present` extent ORT infers (`A + seq_len`) is
    # still inside the allocation it is pointed at. The spare token is never written: the EP
    # writes at stride A, `tok_pos < A`.
    arena_ov = {}
    if lane == "arena":
        for i, n in enumerate(kv_names):
            buf = np.full((1, KV_HEADS, A, HEAD_DIM), np.float16(0.5), dtype=np.float16)
            buf[:, :, :P, :] = seed_kv[i]
            ov = ort.OrtValue.ortvalue_from_shape_and_type(
                [1, KV_HEADS, A, HEAD_DIM], np.float16, memory_info=mi)
            ov.update_inplace(buf)
            arena_ov[n] = ov
    past_dev = {}
    if lane in ("grow", "growflag"):
        for i, n in enumerate(kv_names):
            ov = ort.OrtValue.ortvalue_from_shape_and_type(
                [1, KV_HEADS, P, HEAD_DIM], np.float16, memory_info=mi)
            ov.update_inplace(seed_kv[i])
            past_dev[n] = ov

    token = 1
    per_step: list[dict] = []
    logits_sig: list[dict] = []
    logits_full: list = []
    prev = _counters(counters_path)

    def _delta(now, key):
        return int(now.get(key) or 0) - int(prev.get(key) or 0)

    for step in range(steps):
        past_len = P + step
        input_ids = np.array([[token]], dtype=np.int64)
        attn = np.ones((1, past_len + 1), dtype=np.int64)

        if lane in ("host", "cpu", "plain", "plainflag"):
            # `plain`/`plainflag` are the **caller who binds nothing**: the EP is in the
            # provider list and every tensor crosses as numpy, so ORT allocates `present`
            # from the computed shape. If the arena's shape decision followed the flag rather
            # than the binding, `plainflag`'s last KV token would be stale and nothing would
            # say so. This pair is the only place that can tell.
            feeds = {"input_ids": input_ids, "attention_mask": attn}
            feeds.update(past_host)
            names = [o.name for o in sess.get_outputs()]
            got = dict(zip(names, sess.run(None, feeds)))
            logits = np.asarray(got["logits"])
            past_host = {f"past_key_values.{n}": np.asarray(got[f"present.{n}"])
                         for n in kv_names}
        else:
            binding = sess.io_binding()
            binding.bind_cpu_input("input_ids", input_ids)
            binding.bind_cpu_input("attention_mask", attn)
            present = {}
            for n in kv_names:
                if lane == "arena":
                    # The declaration: the same OrtValue in and out. Nothing else in this
                    # probe makes `present` alias `past` — the EP checks that ORT's output
                    # tensor really is the buffer the dispatch wrote and declines otherwise,
                    # so a caller that only *said* arena gets the shipping route and a
                    # correct answer, not a plausible one.
                    binding.bind_ortvalue_input(f"past_key_values.{n}", arena_ov[n])
                    binding.bind_ortvalue_output(f"present.{n}", arena_ov[n])
                else:
                    binding.bind_ortvalue_input(f"past_key_values.{n}", past_dev[n])
                    ov = ort.OrtValue.ortvalue_from_shape_and_type(
                        [1, KV_HEADS, past_len + 1, HEAD_DIM], np.float16, memory_info=mi)
                    binding.bind_ortvalue_output(f"present.{n}", ov)
                    present[n] = ov
            logits_ov = ort.OrtValue.ortvalue_from_numpy(
                np.zeros((1, 1, VOCAB), dtype=logits_dtype))
            binding.bind_ortvalue_output("logits", logits_ov)
            sess.run_with_iobinding(binding)
            logits = np.asarray(logits_ov.numpy())
            if lane in ("grow", "growflag"):
                past_dev = present

        flat = np.asarray(logits, dtype=np.float64).reshape(-1)[-VOCAB:]
        logits_full.append(flat.copy())
        token = int(np.argmax(flat))
        logits_sig.append({
            "step": step, "past_len": past_len, "argmax": token,
            "max": float(flat.max()), "sum": float(flat.sum()),
            "sha256": hashlib.sha256(np.asarray(logits).tobytes()).hexdigest()[:16],
        })
        now = _counters(counters_path)
        per_step.append({
            "step": step, "past_len": past_len,
            "readback_bytes": _delta(now, "session_staging_readback_bytes"),
            "download_bytes": _delta(now, "alloc_device_download_bytes"),
            "upload_bytes": _delta(now, "session_staging_upload_bytes"),
            "dispatches": _delta(now, "dispatches_executed"),
        })
        prev = now

    doc["per_step"] = per_step
    doc["logits"] = logits_sig
    np.save(str(out_path) + ".logits.npy", np.stack(logits_full).astype(np.float32))

    # ---- the 65th..128th outputs: the KV itself -----------------------------------------
    # Read AFTER every byte count above is closed, so the download this performs cannot be
    # charged to the lane it is checking. This is the check the byte counts are not allowed to
    # be read without.
    final_kv = {}
    src = arena_ov if lane == "arena" else (
        past_dev if lane in ("grow", "growflag") else past_host)
    for n in kv_names:
        v = src[f"past_key_values.{n}"] if lane in ("host", "cpu", "plain", "plainflag") else src[n]
        a = np.asarray(v.numpy() if hasattr(v, "numpy") else v)
        # The arena's valid region is [0, P + steps); the tail is capacity, not content.
        valid = a[:, :, : P + steps, :]
        final_kv[n] = hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest()[:16]
    doc["final_kv_sha256"] = final_kv
    np.savez_compressed(
        str(out_path) + ".kv.npz",
        **{n: np.ascontiguousarray(
            (np.asarray(src[f"past_key_values.{n}"]) if lane in ("host", "cpu", "plain", "plainflag")
             else np.asarray(src[n].numpy()))[:, :, : P + steps, :])
           for n in kv_names},
    )

    final = _counters(counters_path)
    doc["counters"] = {k: final.get(k) for k in (
        "kv_cache_convention", "alloc_device_frame", "outputs_device_bound",
        "outputs_device_resident", "outputs_host_resident",
        "session_staging_readback_bytes", "session_staging_upload_bytes",
        "alloc_device_download_bytes", "alloc_device_upload_bytes",
        "alloc_device_authority_grants", "dispatches_executed", "compute_calls",
        "compute_failures", "device_losses", "shaders_dispatched_digest",
        # Peak device memory is the only instrument that can see the arena: under
        # DEVICE_MEMORY=1 the KV cache never crosses the link, so the byte slope is blind
        # to the change the arena makes.
        "alloc_high_water_bytes", "session_device_high_water_bytes",
        "alloc_device_bytes_in_use", "session_device_bytes_in_use",
        "alloc_device_attach_failures", "alloc_device_attach_attempts",
        "claimed_nodes", "subject_changed_declines",
    )}
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 0


def _slope(points) -> float:
    n = len(points)
    if n < 2:
        return float("nan")
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    den = n * sxx - sx * sx
    return float("nan") if den == 0 else (n * sxy - sx * sy) / den


def chain_main(args) -> int:
    import numpy as np

    if args.worker:
        return _chain_worker(args)

    if args.past + args.steps > args.arena:
        # The arena capacity is a hard ceiling: a step past it is dropped by the kernel's
        # `tok_pos < present_len` guard rather than corrupting memory, but a dropped write is
        # a wrong answer, and a probe that overruns its own arena measures that instead.
        print(f"past({args.past}) + steps({args.steps}) exceeds arena({args.arena})",
              file=sys.stderr)
        return 2
    scratch = HERE / f"kv_arena_chain-A{args.arena}"
    scratch.mkdir(exist_ok=True)
    lanes = {}
    wanted = [l for l in ("host", "grow", "arena", "growflag", "plain", "plainflag")
              if not args.lanes or l in args.lanes.split(",")]
    for lane in wanted:
        out = scratch / f"{lane}.json"
        env = dict(os.environ)
        env[COUNTERS_ENV] = str(scratch / f"{lane}-counters.json")
        if lane == "host":
            env.pop("ONNXRUNTIME_EP_VULKAN_KV_ARENA", None)
        else:
            env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
            env["ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS"] = "1"
            # `growflag` and `plainflag` are the separating cases for **a caller who does not
            # honour the declaration**: the flag is on and the caller still uses the growing
            # convention. `growflag` pre-allocates a growing `present` (so ORT can catch a
            # shape mismatch); `plainflag` binds nothing at all, so ORT allocates from the
            # computed shape and only the bytes can say whether the last token was written.
            env["ONNXRUNTIME_EP_VULKAN_KV_ARENA"] = (
                "1" if lane in ("arena", "growflag", "plainflag") else "0")
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--mode", "chain",
               "--worker", "--lane", lane, "--arena", str(args.arena),
               "--past", str(args.past), "--steps", str(args.steps), "--out", str(out)]
        rc = subprocess.call(cmd, env=env)
        rec = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        rec["worker_rc"] = rc
        rec["logits_npy"] = str(out) + ".logits.npy"
        rec["kv_npz"] = str(out) + ".kv.npz"
        lanes[lane] = rec

    doc: dict = {"probe": "kv_arena_chain", "arena": args.arena, "past": args.past,
                 "steps": args.steps}
    # Correctness FIRST. A byte count cannot tell you it measured the wrong tensor.
    #
    # The reference is the **shipping Vulkan lane** (`grow`), not the CPU lane. `grow` and
    # `arena` share device, kernel, and inputs exactly, so bit-identity is the right bar and
    # any difference is the aliasing. `host` is a different arithmetic (CPU fp32 accumulation
    # against fp16 on device) and cannot be a bitwise reference — it is kept as an
    # argmax-agreement control so a lane that is bitwise self-consistent but wrong in both
    # copies still fails something.
    correctness = {}

    def _load(lane, key):
        p = pathlib.Path(lanes.get(lane, {}).get(key, ""))
        return np.load(str(p)) if p.is_file() else None

    ref_logits = _load("grow", "logits_npy")
    ref_kv = _load("grow", "kv_npz")
    host_logits = _load("host", "logits_npy")
    for lane in ("arena",):
        got = _load(lane, "logits_npy")
        if ref_logits is None or got is None:
            correctness[lane] = {"verdict": "NOT_PERFORMED"}
            continue
        rec = {
            "reference_lane": "grow",
            "logits_bitwise_equal": bool(np.array_equal(ref_logits, got)),
            "logits_max_abs": float(np.max(np.abs(ref_logits.astype(np.float64) -
                                                  got.astype(np.float64)))),
            "argmax_equal": [int(a) == int(b) for a, b in
                             zip(ref_logits.argmax(axis=1), got.argmax(axis=1))],
        }
        got_kv = _load(lane, "kv_npz")
        if ref_kv is not None and got_kv is not None:
            same = [n for n in ref_kv.files if n in got_kv.files
                    and np.array_equal(ref_kv[n], got_kv[n])]
            rec["kv_tensors_compared"] = len(ref_kv.files)
            rec["kv_tensors_bitwise_equal"] = len(same)
            rec["kv_tensors_differing"] = [n for n in ref_kv.files if n not in same][:8]
            nz = float(np.count_nonzero(ref_kv[ref_kv.files[0]]) /
                       ref_kv[ref_kv.files[0]].size)
            rec["kv_reference_nonzero_frac"] = nz
            if nz < 0.5:
                rec["verdict"] = "NOT_PERFORMED(degenerate reference KV)"
        rec.setdefault(
            "verdict",
            "BIT_IDENTICAL" if rec["logits_bitwise_equal"]
            and rec.get("kv_tensors_bitwise_equal") == rec.get("kv_tensors_compared")
            else "DIVERGENT")
        correctness[lane] = rec
    if host_logits is not None and ref_logits is not None:
        correctness["grow_vs_host_control"] = {
            "reference_lane": "host",
            "note": "CPU fp32 vs device fp16 — approximate by construction, argmax only",
            "logits_max_abs": float(np.max(np.abs(host_logits.astype(np.float64) -
                                                  ref_logits.astype(np.float64)))),
            "argmax_equal": [int(a) == int(b) for a, b in
                             zip(host_logits.argmax(axis=1), ref_logits.argmax(axis=1))],
        }
    plain_ref = _load("plain", "logits_npy")
    plain_flag = _load("plainflag", "logits_npy")
    if plain_ref is not None and plain_flag is not None:
        pk, pf = _load("plain", "kv_npz"), _load("plainflag", "kv_npz")
        same = ([n for n in pk.files if n in pf.files and np.array_equal(pk[n], pf[n])]
                if pk is not None and pf is not None else [])
        rec = {
            "reference_lane": "plain",
            "logits_bitwise_equal": bool(np.array_equal(plain_ref, plain_flag)),
            "logits_max_abs": float(np.max(np.abs(plain_ref.astype(np.float64) -
                                                  plain_flag.astype(np.float64)))),
            "kv_tensors_compared": len(pk.files) if pk is not None else 0,
            "kv_tensors_bitwise_equal": len(same),
        }
        rec["kv_cache_convention"] = (lanes.get("plainflag", {}).get("counters") or {}).get(
            "kv_cache_convention")
        rec["compute_failures"] = (lanes.get("plainflag", {}).get("counters") or {}).get(
            "compute_failures")
        rec["dispatches_executed"] = (lanes.get("plainflag", {}).get("counters") or {}).get(
            "dispatches_executed")
        if rec["logits_bitwise_equal"] and \
                rec["kv_tensors_bitwise_equal"] == rec["kv_tensors_compared"]:
            rec["verdict"] = "UNBOUND_CALLER_UNAFFECTED"
        elif rec["compute_failures"] and not rec["dispatches_executed"]:
            # The EP refused before dispatching anything and ORT re-ran the island on the CPU
            # EP. The lanes differ because one is CPU arithmetic and one is fp16 on device —
            # that is a *loud* refusal, which is the outcome this case is testing for. It is
            # not the same object as a silent wrong answer and must not share its name.
            rec["verdict"] = "UNBOUND_CALLER_REFUSED_LOUDLY(fell back to CPU EP)"
        else:
            rec["verdict"] = "UNBOUND_CALLER_CORRUPTED_BY_THE_FLAG"
        correctness["plainflag_vs_plain"] = rec
    if "growflag" in lanes:
        # A caller who pre-allocates a growing `present` while the flag is on. ORT itself
        # rejects the shape mismatch, so the failure is loud without the EP doing anything —
        # recorded because "ORT catches it" is a fact about ORT, not a property of this EP.
        correctness["growflag"] = {
            "reference_lane": "grow",
            "worker_rc": lanes["growflag"].get("worker_rc"),
            "verdict": ("GROWING_CALLER_REFUSED_LOUDLY(ORT shape check)"
                        if lanes["growflag"].get("worker_rc") not in (0, None)
                        else "GROWING_CALLER_RAN(the flag did not shorten present)"),
        }
    doc["correctness"] = correctness

    bytes_rec = {}
    for lane, rec in lanes.items():
        steps = rec.get("per_step") or []
        pts_dl = [(s["past_len"], s["download_bytes"] + s["readback_bytes"]) for s in steps]
        bytes_rec[lane] = {
            "per_step_link_bytes": [p[1] for p in pts_dl],
            "slope_bytes_per_past_token": _slope(pts_dl),
            "dispatches": [s["dispatches"] for s in steps],
            "kv_cache_convention": (rec.get("counters") or {}).get("kv_cache_convention"),
            # The arena's win is **peak device memory**, not link traffic: under
            # DEVICE_MEMORY=1 the KV cache never crosses the link either way, so the byte
            # slope is blind to the whole change. The high-water mark is the instrument that
            # can see it, and it is the one that decides whether ctx 8192 fits in 8 GB.
            "alloc_high_water_bytes": (rec.get("counters") or {}).get("alloc_high_water_bytes"),
            "session_device_high_water_bytes":
                (rec.get("counters") or {}).get("session_device_high_water_bytes"),
            "alloc_device_attach_failures":
                (rec.get("counters") or {}).get("alloc_device_attach_failures"),
            "compute_failures": (rec.get("counters") or {}).get("compute_failures"),
            "device_losses": (rec.get("counters") or {}).get("device_losses"),
            "ep_device": rec.get("ep_device"),
            "worker_rc": rec.get("worker_rc"),
        }
    doc["bytes"] = bytes_rec
    # The arena's own signature, and it is a *shape* claim, not a byte claim: the convention
    # the dispatch was built with, read off the effective push constants.
    conv = bytes_rec.get("arena", {}).get("kv_cache_convention")
    ok_corr = correctness.get("arena", {}).get("verdict") == "BIT_IDENTICAL"
    grow_rc = lanes.get("grow", {}).get("worker_rc")
    if lanes.get("arena", {}).get("worker_rc") != 0:
        doc["verdict"] = "ARENA_LANE_DID_NOT_RUN"
    elif conv != "SHARED":
        doc["verdict"] = f"ARENA_NOT_TAKEN(kv_cache_convention={conv})"
    elif grow_rc not in (0, None) or correctness.get("arena", {}).get(
            "verdict") == "NOT_PERFORMED":
        # The arena ran and the shipping lane did not. That is a **capacity** finding, not a
        # correctness one, and it must not be reported as either a pass or a divergence:
        # there is no reference to compare against because the reference could not be taken.
        doc["verdict"] = "ARENA_RAN_WHERE_GROWING_COULD_NOT(no bitwise reference this size)"
    elif not ok_corr:
        doc["verdict"] = "ARENA_TAKEN_BUT_DIVERGENT"
    else:
        doc["verdict"] = "ARENA_TAKEN_AND_BIT_IDENTICAL"

    out = pathlib.Path(args.out or str(HERE / f"kv_arena_chain-A{args.arena}.json"))
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(json.dumps(doc, indent=2, default=str))
    print(f"[wrote] {out}")
    return 0 if doc["verdict"].startswith(
        ("ARENA_TAKEN_AND_BIT_IDENTICAL", "ARENA_RAN_WHERE_GROWING_COULD_NOT")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
