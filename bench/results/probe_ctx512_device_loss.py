r"""Is the ctx-512 device loss a TDR, a kernel fault, or the box — and can it be caught?

WHAT HAPPENED
-------------
Both of Tank's ctx-512 measurement points, and both of mine, were truncated by

    vkWaitForFences failed: The logical device has been lost
      -> ORT re-runs on the CPU EP  ->  EXIT = 0

A run that dies partway and exits 0 does not look like a failure. It looks like a smaller
number: differencing a truncated pair produced an apparent 6.7% KV saving that was an
observation ending early.

THE THREE QUESTIONS, AND WHY THIS PROBE ANSWERS THEM SEPARATELY
---------------------------------------------------------------
1. WHOSE IS IT?  Already answered, and not by this probe: the pre-fix binary on `main` loses
   the device on the same point, and Tank's default-lane control produced identical text. It is
   not the input-cache change and it is not the device-memory provider.

2. IS IT A TIMEOUT?  A Windows TDR writes `Display` event 4101 ("stopped responding and has
   successfully recovered"). A GPU *fault* writes `nvlddmkm` events instead. This probe reads
   the System log around its own run window and reports which appeared. It does not infer a
   TDR from the *absence* of a fault, or the reverse — it reports what the log actually holds
   and says UNDETERMINED when neither appears.

3. WHICH KERNEL?  The Phi-3.5 island is 355 nodes, so "it faults at ctx 512" names no kernel.
   The isolation arm runs a SINGLE GroupQueryAttention node at the same geometry and past
   length, many times. GQA is the ctx-dependent kernel and it is the one changed twice today.
   If the standalone arm faults, it is ours and it is GQA. If it survives while the full model
   falls over, the fault is somewhere else in the island, and that is worth knowing before
   anyone rewrites the attention loop.

MAKING IT NOT EXIT 0
--------------------
`onnxruntime.InferenceSession.run` catches an EP failure, silently rebuilds the session on
CPU-only providers, retries, and returns a plausible answer. That is the whole reporting
defect: the EP reported the failure correctly, ORT handled it correctly, and the *harness*
turned both into a success. `sess.disable_fallback()` makes the failure raise. Every arm here
calls it, and the probe exits non-zero on any device loss.

The EP now also counts device losses separately (`device_losses` in the counters JSON) and
names the mechanism in the status text, so a run that swallowed the exception still leaves a
record that says the device went away.

USAGE
-----
    $env:VULKAN_SDK="C:\VulkanSDK\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\Bin;$env:PATH"
    $env:ONNXRUNTIME_VULKAN_EP_LIB="...\onnxruntime_vulkan_ep.dll"
    python bench\results\probe_ctx512_device_loss.py

No clock is read. Counts, exceptions and the Windows event log only.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
EP_NAME = "VulkanExecutionProvider"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
GQA_KEY = (
    "com.microsoft::GroupQueryAttention/1+/f16,f16,f16,i32,i32,f16,f16>f16,f16,f16"
    "/metadata/runtime-extent/past_key+past_value+cos_cache+sin_cache"
)

PHI = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
    r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
)

# Phi-3.5's own attention geometry, so the isolation arm is the same kernel shape the model
# runs and not a toy. 32 query heads, 32 KV heads (this model does no grouping at all),
# head_dim 96, full-width rotary.
B, S, NQ, NKV, D = 1, 1, 32, 32, 96
MAXSEQ = 8192
ROT = D


def build_gqa(path: pathlib.Path, past: int) -> None:
    total = past + S
    node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "packed_qkv", "", "",
            "past_key", "past_value",
            "seqlens_k", "total_seq", "cos_cache", "sin_cache",
        ],
        outputs=["attn_out", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=NQ,
        kv_num_heads=NKV,
        scale=1.0 / float(np.sqrt(D)),
        do_rotary=1,
        rotary_interleaved=0,
    )
    g = helper.make_graph(
        [node],
        "gqa_ctx_probe",
        [
            helper.make_tensor_value_info(
                "packed_qkv", TensorProto.FLOAT16, [B, S, (NQ + 2 * NKV) * D]
            ),
            helper.make_tensor_value_info("past_key", TensorProto.FLOAT16, [B, NKV, past, D]),
            helper.make_tensor_value_info("past_value", TensorProto.FLOAT16, [B, NKV, past, D]),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [B]),
            helper.make_tensor_value_info("total_seq", TensorProto.INT32, []),
            helper.make_tensor_value_info("cos_cache", TensorProto.FLOAT16, [MAXSEQ, ROT // 2]),
            helper.make_tensor_value_info("sin_cache", TensorProto.FLOAT16, [MAXSEQ, ROT // 2]),
        ],
        [
            helper.make_tensor_value_info("attn_out", TensorProto.FLOAT16, [B, S, NQ * D]),
            helper.make_tensor_value_info("present_key", TensorProto.FLOAT16, [B, NKV, total, D]),
            helper.make_tensor_value_info(
                "present_value", TensorProto.FLOAT16, [B, NKV, total, D]
            ),
        ],
    )
    m = helper.make_model(
        g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    m.ir_version = 10
    onnx.save(m, str(path))


def _session(model_path: str, **kw):
    """A session that will RAISE on EP failure rather than answering from the CPU EP.

    `disable_fallback()` is the whole point. Without it `run()` catches the EP error, rebuilds
    the session CPU-only, retries, and returns a plausible answer — which is how a lost device
    reaches the caller as exit status 0.
    """
    try:
        ort.register_execution_provider_library(EP_NAME, os.environ["ONNXRUNTIME_VULKAN_EP_LIB"])
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise
    sess = ort.InferenceSession(
        model_path, providers=[EP_NAME, "CPUExecutionProvider"], **kw
    )
    if EP_NAME not in sess.get_providers():
        raise SystemExit(f"ERROR(instrument): {EP_NAME} absent from {sess.get_providers()}")
    sess.disable_fallback()
    return sess


# --------------------------------------------------------------------------- arms


def arm_gqa(past: int, iters: int) -> int:
    """Isolation arm: one GQA node, Phi-3.5's geometry, `iters` decode steps at `past`."""
    with tempfile.TemporaryDirectory() as td:
        mp = pathlib.Path(td) / f"gqa_{past}.onnx"
        build_gqa(mp, past)
        sess = _session(str(mp))
        rng = np.random.default_rng(0)
        pos = np.arange(MAXSEQ, dtype=np.float32)[:, None] / 10000.0 ** (
            np.arange(ROT // 2, dtype=np.float32)[None, :] * 2.0 / ROT
        )
        feeds = {
            "packed_qkv": (
                rng.standard_normal((B, S, (NQ + 2 * NKV) * D)).astype(np.float16) * 0.1
            ),
            "past_key": (rng.standard_normal((B, NKV, past, D)) * 0.1).astype(np.float16),
            "past_value": (rng.standard_normal((B, NKV, past, D)) * 0.1).astype(np.float16),
            "seqlens_k": np.array([past + S - 1], dtype=np.int32),
            "total_seq": np.array(past + S, dtype=np.int32),
            "cos_cache": np.cos(pos).astype(np.float16),
            "sin_cache": np.sin(pos).astype(np.float16),
        }
        for i in range(iters):
            sess.run(None, feeds)
            if i % 25 == 0:
                print(f"    gqa past={past} iter {i}", flush=True)
    return 0


def arm_phi(past: int, iters: int) -> int:
    """Whole-model arm: the case Tank and I both saw die."""
    sess = _session(
        str(PHI),
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    feeds = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.ones((1, past + 1), dtype=np.int64),
    }
    kv = np.zeros((1, 32, past, 96), dtype=np.float16)
    for layer in range(32):
        feeds[f"past_key_values.{layer}.key"] = kv
        feeds[f"past_key_values.{layer}.value"] = kv
    for i in range(iters):
        sess.run(None, feeds)
        if i % 5 == 0:
            print(f"    phi past={past} iter {i}", flush=True)
    return 0


ARMS = {"gqa": arm_gqa, "phi": arm_phi}


# --------------------------------------------------------------------------- driver


def gpu_events(since: float) -> dict:
    """What the Windows System log recorded during our run window.

    Reports what is there. `Display` 4101 is the TDR signature; `nvlddmkm` 13/153 are driver
    fault signatures. Neither is inferred from the absence of the other.
    """
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since))
    ps = (
        "$s=[datetime]::Parse('" + iso + "');"
        "$e=Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$s} "
        "-ErrorAction SilentlyContinue | Where-Object { $_.Id -eq 4101 -or "
        "$_.ProviderName -eq 'nvlddmkm' };"
        "$e | ForEach-Object { \"$($_.ProviderName)/$($_.Id)\" } | Group-Object | "
        "ForEach-Object { \"$($_.Name)=$($_.Count)\" }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception:  # noqa: BLE001
        return {"state": "UNREADABLE"}
    counts: dict = {}
    for line in out.splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.rsplit("=", 1)
            try:
                counts[k] = int(v)
            except ValueError:
                pass
    tdr = counts.get("Display/4101", 0)
    faults = sum(v for k, v in counts.items() if k.startswith("nvlddmkm/"))
    if tdr and not faults:
        state = "TDR"
    elif faults and not tdr:
        state = "GPU_FAULT"
    elif faults and tdr:
        state = "BOTH"
    else:
        state = "UNDETERMINED"
    return {"state": state, "counts": counts}


def run_arm(arm: str, past: int, iters: int, scratch: pathlib.Path) -> dict:
    cfile = scratch / f"{arm}_p{past}_n{iters}.counters.json"
    cfile.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(cfile)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()),
         "--worker", "--arm", arm, "--past", str(past), "--iters", str(iters)],
        capture_output=True, text=True, env=env,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    lost_in_text = "device has been lost" in text or "DEVICE_LOST" in text
    c: dict = {}
    if cfile.is_file():
        raw = json.loads(cfile.read_text(encoding="utf-8"))
        c = raw.get("counters", raw)
    return {
        "arm": arm,
        "past_len": past,
        "iters": iters,
        "exit": proc.returncode,
        "device_lost_in_text": lost_in_text,
        "device_losses_counter": c.get("device_losses"),
        "compute_calls": c.get("compute_calls"),
        "compute_failures": c.get("compute_failures"),
        "dispatches_executed": c.get("dispatches_executed"),
        "claimed_nodes": c.get("claimed_nodes"),
        "events": gpu_events(t0),
    }


def is_trivial(p: dict) -> bool:
    """Did the EP execute anything at all in this arm?

    An arm whose node was declined runs entirely on the CPU EP and cannot lose the device.
    It reports NO_LOSS for the same reason a switched-off smoke detector reports no fire.
    The first run of this probe did exactly that: the ledger declined `gqa_f16` on a changed
    shader digest, the arm read `calls=0 dispatches=0`, and the verdict would have been
    "GQA is innocent" from a run GQA never entered.
    """
    return not (p.get("compute_calls") or 0) or not (p.get("dispatches_executed") or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", default="gqa", choices=sorted(ARMS))
    ap.add_argument("--past", type=int, default=512)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--plan",
        default="gqa:512:200,gqa:2048:100,phi:512:25",
        help="comma-separated arm:past:iters points",
    )
    args = ap.parse_args()

    if args.worker:
        return ARMS[args.arm](args.past, args.iters)

    scratch = ROOT / "bench" / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    points = []
    for spec in args.plan.split(","):
        arm, past, iters = spec.split(":")
        print(f"[ctx-loss] {arm} past={past} iters={iters} ...", flush=True)
        points.append(run_arm(arm, int(past), int(iters), scratch))

    print()
    print("=" * 78)
    print("CTX DEVICE LOSS: which arm, which mechanism, and does it exit non-zero")
    print("=" * 78)
    for p in points:
        print(
            f"  {p['arm']:4s} past={p['past_len']:>5}  iters={p['iters']:>4}  "
            f"exit={p['exit']:>3}  lost={p['device_lost_in_text']!s:5s}  "
            f"device_losses={p['device_losses_counter']}  "
            f"calls={p['compute_calls']}  dispatches={p['dispatches_executed']}  "
            f"events={p['events'].get('state')}"
        )

    lost = [p for p in points if p["device_lost_in_text"] or p["device_losses_counter"]]
    silent = [p for p in lost if p["exit"] == 0]
    trivial = [p for p in points if is_trivial(p) and not p["device_lost_in_text"]]

    print()
    if trivial:
        print(f"  ERROR(instrument): {len(trivial)} arm(s) executed NO EP work:")
        for p in trivial:
            print(
                f"    {p['arm']} past={p['past_len']}  calls={p['compute_calls']} "
                f"dispatches={p['dispatches_executed']} claimed_nodes={p['claimed_nodes']}"
            )
        print("  Those arms ran on the CPU EP. They cannot lose the device and their NO_LOSS")
        print("  is vacuous. Most likely cause: the proof ledger declines the form because the")
        print("  shader digest moved — re-prove with")
        print("    python rust/tools/gen_proof_ledger.py --reprove --append --model <case>")
        print("  and REBUILD (the DLL embeds the ledger), or arm with")
        print(f"    ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN={GQA_KEY}")

    print()
    if not lost:
        print("  NO_LOSS — no arm lost the device in this session. The defect is intermittent,")
        print("  so this is NOT evidence that it is fixed; it is evidence this pass missed it.")
    else:
        arms_lost = sorted({p["arm"] for p in lost})
        print(f"  LOSS in arm(s): {arms_lost}")
        if arms_lost == ["gqa"] or "gqa" in arms_lost:
            print("  A single GroupQueryAttention node reproduces it. The fault is ours and it")
            print("  is in the attention kernel, not in the 355-node island around it.")
        else:
            print("  The whole-model arm faults and the standalone GQA arm does not. GQA is not")
            print("  implicated by this reading; do not rewrite the attention loop on it.")
        states = {p["events"].get("state") for p in lost}
        print(f"  Windows event evidence: {states}")
        print("    TDR         = watchdog timeout (Display/4101)")
        print("    GPU_FAULT   = driver fault, no timeout event (nvlddmkm/13,153)")
        print("    UNDETERMINED= neither appeared; mechanism not established by this instrument")

    print()
    if silent:
        print(f"  REPORTING DEFECT LIVE: {len(silent)} arm(s) lost the device and exited 0.")
    elif lost:
        print("  REPORTING OK: every arm that lost the device exited non-zero.")

    rec = HERE / "ctx_device_loss.json"
    rec.write_text(
        json.dumps({"points": points, "trivial_arms": len(trivial)}, indent=2),
        encoding="utf-8",
    )
    print(f"\n  record: {rec}")

    # The exit status is the point of the exercise. A vacuous arm is an instrument failure
    # and must not be reported as a clean pass.
    if trivial:
        return 2
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
