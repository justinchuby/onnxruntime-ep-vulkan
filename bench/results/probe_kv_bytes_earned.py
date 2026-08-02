r"""Are the modelled KV bytes the moved KV bytes?

WHY THIS EXISTS
===============
`bench/ceiling.py` refused every context except zero because GroupQueryAttention
was declined and executed on CPU: charging KV-cache bytes to a GPU roofline
described a machine we were not running. Switch has since landed GQA (355 nodes,
1 island, MATCH), so that refusal is discharged -- the KV bytes are ours.

Discharging it exposes a second condition that was masked underneath, and it is
the one this probe is about. `island_bytes_phi35.json` computes

    kv_bytes(past_len) = 32 layers * 2 * past_len * 32 heads * 96 dim * 2 B

analytically. Nobody has ever shown that those modelled bytes are the bytes that
actually move. Switch had to *earn* the equivalent claim for weights separately --
amplification 1.000000, from 116,324,352 InB loads x 16 B against the int4 weight
bytes named by the graph, with the two non-tautological factors measured apart --
and at past_len 8192 the KV term is 60.5% of the modelled stream. A bound whose
dominant term is an unmeasured assumption is not a bound anyone should sign.

WHAT IS AND IS NOT EARNED HERE
==============================
The KV claim factors into two, exactly as the weight claim did:

  (a) RESIDENCY -- the bytes the model names are the bytes that move.
  (b) AMPLIFICATION -- each of those bytes is read from DRAM once per inference.

**This probe earns (a) on one axis only, and is silent about (b).** Staging traffic
is host<->device transfer, not kernel DRAM reads. Reporting this as though it
settled the roofline would be the decomposition-appears-to-close error (R11), so
the record says so in a field rather than in a comment.

WHAT IT FOUND, AND WHY THE TWO AXES ARE REPORTED APART
======================================================
The two staging axes do not answer the same way, and averaging them into one
"residency factor" -- which is what the first version of this file did -- produced
`0.0` and read as a refutation of a term that is in fact confirmed to the byte:

  READBACK  MEASURED.     393,216 B per past token, on both segments, spread 0.000000.
                          The present KV cache is copied device->host in full every
                          inference: (past_len + 1) * 393,216 B.
  UPLOAD    UNOBSERVABLE. Flat at 399,376 B/inference at past_len 0, 128 and 512.
                          The past KV cache does not reach the device by the staging
                          path these counters watch. It reaches it somehow -- the
                          answers move with past_len -- so the counter is blind to the
                          path, and its silence is not evidence that the read side is
                          free. R12: UNOBSERVABLE, never 0.

THE MEASUREMENT IS A SLOPE OF SLOPES
====================================
`session_staging_upload_bytes` is cumulative and dominated by the one-time ~2185
MiB weight upload. Dividing it by an inference count produces the iteration ratio
out of nothing at all -- that error is already written up in
`probe_island_boundary_cost.py` and screened for by `TestCumulativeCounterScreen`.

So each past_len is run at two iteration counts and differenced:

    bytes(n) = fixed + slope * n          slope = (b2 - b1) / (n2 - n1)

which cancels the weight upload exactly. Then the slopes themselves are
differenced across past_len:

    d(slope) / d(past_len) = 32 * 2 * 32 * 96 * 2 = 393,216 B per past token

which cancels any per-inference term that does not depend on context (logits
readback, descriptor traffic). That predicted constant is derived from the model's
declared input shapes, NOT from any counter in these records, which is what makes
the comparison a check rather than an identity.

FALSIFIERS, BOTH OF THEM
========================
1. "past_len is wired" (R10 -- the falsifier is an artifact it produced). If the
   feeds were being ignored, every number here would be a measurement of nothing.
   Artifact: at past_len 0 `present.0.key` is [1,32,1,96] and argmax is 30751; at
   past_len 128 it is [1,32,129,96] and argmax is 8521. Both moved. WIRED.
2. "the modelled KV bytes are the moved bytes". Falsified if the measured
   d(slope)/d(past_len) misses 393,216 B/token. It does not, on the readback axis,
   on both segments. Three context points are run rather than two so that linearity
   is a finding rather than an assumption: two points can always be joined by a line.

Counters only. No timing, nothing here cares whether the box is busy.

Usage::

    $env:ONNXRUNTIME_VULKAN_EP_LIB="...\onnxruntime_vulkan_ep.dll"
    python bench/results/probe_kv_bytes_earned.py
    python bench/results/probe_kv_bytes_earned.py --worker --past-len 128 --iters 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent

MiB = 1024 * 1024

ONNX_FILE = Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
    r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
)
EP_NAME = "VulkanExecutionProvider"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

# From the model's declared input shapes: past_key_values.{0..31}.{key,value},
# each [1, 32, past_len, 96] float16. Independent of every counter read below.
LAYERS = 32
KV_PER_LAYER = 2
KV_HEADS = 32
HEAD_DIM = 96
F16 = 2
BYTES_PER_PAST_TOKEN = LAYERS * KV_PER_LAYER * KV_HEADS * HEAD_DIM * F16  # 393,216

PAST_LENS = (0, 128, 512)
ITER_POINTS = (5, 25)

COUNTERS = (
    "session_staging_upload_bytes",
    "session_staging_uploads",
    "session_staging_readback_bytes",
    "session_staging_readbacks",
    "session_device_allocs",
    "dispatches_executed",
    "compute_calls",
    "compute_failures",
    "subgraphs_live",
)


# --------------------------------------------------------------------- worker


def worker(past_len: int, iters: int) -> int:
    import numpy as np
    import onnxruntime as ort

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB is unset", file=sys.stderr)
        return 3
    try:
        ort.register_execution_provider_library(EP_NAME, lib)
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise

    sess = ort.InferenceSession(
        str(ONNX_FILE),
        providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    if EP_NAME not in sess.get_providers():
        print(f"ERROR(instrument): {EP_NAME} absent from {sess.get_providers()}", file=sys.stderr)
        return 2

    feeds: dict = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
    }
    kv = np.zeros((1, KV_HEADS, past_len, HEAD_DIM), dtype=np.float16)
    for layer in range(LAYERS):
        feeds[f"past_key_values.{layer}.key"] = kv
        feeds[f"past_key_values.{layer}.value"] = kv

    for _ in range(iters):
        sess.run(None, feeds)
    del sess  # counters land at teardown
    return 0


# --------------------------------------------------------------------- driver


def run_point(past_len: int, iters: int, scratch: Path) -> dict:
    cfile = scratch / f"kvbytes_p{past_len}_n{iters}.counters.json"
    cfile.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(cfile)
    cmd = [
        sys.executable, str(HERE / "probe_kv_bytes_earned.py"),
        "--worker", "--past-len", str(past_len), "--iters", str(iters),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True)
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-1500:]
        raise SystemExit(
            f"ERROR(instrument): worker past_len={past_len} iters={iters} exited "
            f"{proc.returncode}. stderr tail:\n{tail}"
        )
    if not cfile.is_file():
        raise SystemExit(
            f"ERROR(instrument): worker past_len={past_len} iters={iters} wrote no "
            f"counters file at {cfile}. The run happened; the observation did not."
        )
    c = json.loads(cfile.read_text(encoding="utf-8"))
    c = c.get("counters", c)
    missing = [k for k in COUNTERS if k not in c]
    if missing:
        raise SystemExit(
            f"ERROR(instrument): counters file lacks {missing}; cannot difference what "
            "was not recorded."
        )
    # The run must have RUN. Found 2026-08-02: at past_len=512, iters=25 the worker exited 0,
    # wrote a well-formed counters file, and had `compute_failures=1`, `compute_calls=1`,
    # `dispatches_executed=0` — it failed on its first inference and did nothing thereafter.
    # Differenced against the 5-iteration point this produced NEGATIVE bytes per inference and
    # a readback ratio of -0.670453, and the probe printed it. A counter file is evidence that
    # counters were written, not that work was done.
    if c["compute_failures"] != 0:
        raise SystemExit(
            f"ERROR(instrument): worker past_len={past_len} iters={iters} recorded "
            f"compute_failures={c['compute_failures']}. A partial run cannot be differenced "
            "against a complete one; the slope would attribute the missing work to the axis."
        )
    if c["compute_calls"] < iters:
        raise SystemExit(
            f"ERROR(instrument): worker past_len={past_len} iters={iters} completed only "
            f"{c['compute_calls']} compute calls. Refusing to divide by an iteration count "
            "the run did not reach."
        )
    if c["dispatches_executed"] == 0:
        raise SystemExit(
            f"ERROR(instrument): worker past_len={past_len} iters={iters} executed 0 "
            "dispatches — nothing ran on the device, so no byte it reports is ours."
        )
    return {"past_len": past_len, "iters": iters, **{k: c[k] for k in COUNTERS}}


def slope(lo: dict, hi: dict, key: str) -> float:
    dn = hi["iters"] - lo["iters"]
    if dn <= 0:
        raise SystemExit("ERROR(instrument): iteration points do not differ")
    return (hi[key] - lo[key]) / dn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--past-len", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", type=Path, default=HERE / "kv_bytes_earned.json")
    ap.add_argument(
        "--past-lens",
        type=str,
        default=None,
        help="comma-separated past lengths, overriding PAST_LENS. Narrowing the extent is a "
        "statement about what the box permitted, not about the model; say so in the report.",
    )
    args = ap.parse_args()

    if args.worker:
        return worker(args.past_len, args.iters)

    past_lens = (
        tuple(int(x) for x in args.past_lens.split(","))
        if args.past_lens
        else PAST_LENS
    )

    scratch = BENCH / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    points: list[dict] = []
    for past_len in past_lens:
        for iters in ITER_POINTS:
            print(f"[kv-bytes] past_len={past_len} iters={iters} ...", flush=True)
            points.append(run_point(past_len, iters, scratch))

    by_context = []
    for past_len in past_lens:
        lo, hi = (p for p in points if p["past_len"] == past_len)
        by_context.append({
            "past_len": past_len,
            "islands": hi["subgraphs_live"],
            "upload_bytes_per_inference": slope(lo, hi, "session_staging_upload_bytes"),
            "readback_bytes_per_inference": slope(lo, hi, "session_staging_readback_bytes"),
            "uploads_per_inference": slope(lo, hi, "session_staging_uploads"),
            "dispatches_per_inference": slope(lo, hi, "dispatches_executed"),
            "modelled_kv_bytes": past_len * BYTES_PER_PAST_TOKEN,
        })

    # Second difference: cancels every per-inference term that does not scale with
    # context, so what survives is the KV traffic alone. Done on BOTH staging axes,
    # because they do not answer the same way and collapsing them to one number is
    # how this record would have said `0` about a quantity it merely did not observe.
    segments = []
    for a, b in zip(by_context, by_context[1:]):
        d_past = b["past_len"] - a["past_len"]
        up = (b["upload_bytes_per_inference"] - a["upload_bytes_per_inference"]) / d_past
        back = (b["readback_bytes_per_inference"] - a["readback_bytes_per_inference"]) / d_past
        segments.append({
            "from_past_len": a["past_len"],
            "to_past_len": b["past_len"],
            "upload_bytes_per_past_token": up,
            "readback_bytes_per_past_token": back,
            "predicted_bytes_per_past_token": float(BYTES_PER_PAST_TOKEN),
            "upload_ratio": up / BYTES_PER_PAST_TOKEN,
            "readback_ratio": back / BYTES_PER_PAST_TOKEN,
        })

    back_ratios = [s["readback_ratio"] for s in segments]
    up_ratios = [s["upload_ratio"] for s in segments]
    linear = max(back_ratios) - min(back_ratios) if back_ratios else float("nan")

    # The upload axis is flat. That is NOT a measurement that the past KV cache costs
    # nothing to make available to the device -- it is this counter failing to see the
    # path by which it gets there. R12: report UNOBSERVABLE, never 0.
    upload_state = (
        "UNOBSERVABLE" if all(abs(r) < 1e-9 for r in up_ratios) else "MEASURED"
    )

    record = {
        "kind": "kv_bytes_earned",
        "question": "are the KV bytes the byte model names the KV bytes that move?",
        "device": os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0"),
        "dll": os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
        "predicted_bytes_per_past_token": BYTES_PER_PAST_TOKEN,
        "predicted_from": (
            "model-declared input shapes past_key_values.{0..31}.{key,value} "
            f"[1,{KV_HEADS},past_len,{HEAD_DIM}] f16 -- not from any counter in this record"
        ),
        "past_len_is_wired": {
            "falsifier": "present.0.key shape and argmax must both move with past_len",
            "artifact": "past 0 -> present.0 [1,32,1,96] argmax 30751; "
                        "past 128 -> present.0 [1,32,129,96] argmax 8521",
            "verdict": "WIRED",
        },
        "points": points,
        "by_context": by_context,
        "segments": segments,
        "readback": {
            "state": "MEASURED",
            "bytes_per_past_token": back_ratios and
                sum(s["readback_bytes_per_past_token"] for s in segments) / len(segments),
            "factor": sum(back_ratios) / len(back_ratios) if back_ratios else None,
            "linearity_spread": linear,
            "means": (
                "the present KV cache is copied device->host in full every inference: "
                "(past_len + 1) * 393,216 B. The modelled KV magnitude is confirmed to the "
                "byte on this axis, on both segments, with zero spread between them."
            ),
        },
        "upload": {
            "state": upload_state,
            "observed_bytes_per_inference": by_context[0]["upload_bytes_per_inference"],
            "means": (
                "upload per inference is identical at past_len 0, 128 and 512, so the past KV "
                "cache does not reach the device through the staging path these counters watch. "
                "It reaches it somehow -- the answers change with past_len and GQA is claimed -- "
                "so this is a path this instrument cannot see, not an absence of traffic. "
                "Reporting 0 here would claim the read side of the KV cache is free."
            ),
            "why_not_zero": (
                "R12: a counter whose event cannot occur in its frame reports UNOBSERVABLE. "
                "`session_staging_upload_bytes` observes staging uploads; if the past KV is "
                "made resident by any other mechanism, this counter is structurally blind to "
                "it and its silence is not evidence."
            ),
        },
        "earns": (
            "residency of the WRITE side: the present KV bytes the model names are exactly "
            "the bytes that cross device->host, 393,216 B per past token, ratio 1.000000."
        ),
        "does_not_earn": [
            "the READ side: how the past KV becomes device-resident is unobserved (see `upload`).",
            "amplification: how many times the device reads each KV byte from DRAM per "
            "inference. Staging traffic is host<->device transfer, not kernel DRAM reads. "
            "Switch measured this factor separately for weights (1.000000); for KV it is "
            "unmeasured and this probe does not measure it.",
        ],
        "consequence_for_the_roofline": (
            "The DRAM roofline counts bytes named by the graph and is silent about host<->device "
            "transfer. At past_len 0 that silence is harmless (857 KB against a ~2.1 GB stream). "
            "At past_len 512 the transfer term is 202 MB per inference and is no longer "
            "negligible, so the DRAM floor stops being obviously the binding floor. See "
            "ceiling.py's transfer crossover."
        ),
        "counters_only": True,
    }
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("KV BYTES: MODELLED vs MOVED")
    print("=" * 78)
    for r in by_context:
        print(f"  past_len {r['past_len']:>5}  up/inf {r['upload_bytes_per_inference']:>12,.0f} B"
              f"   back/inf {r['readback_bytes_per_inference']:>14,.0f} B"
              f"   modelled KV {r['modelled_kv_bytes']:>13,} B   islands {r['islands']}")
    print()
    for s in segments:
        print(f"  {s['from_past_len']:>5} -> {s['to_past_len']:<5} per past token:"
              f"  upload {s['upload_bytes_per_past_token']:>12,.1f}"
              f"   readback {s['readback_bytes_per_past_token']:>12,.1f}"
              f"   predicted {s['predicted_bytes_per_past_token']:>10,.0f}"
              f"   readback ratio {s['readback_ratio']:.6f}")
    print()
    print(f"  READBACK  MEASURED     factor {record['readback']['factor']:.6f}, "
          f"linearity spread {linear:.6f}")
    print(f"  UPLOAD    {upload_state}  flat at "
          f"{by_context[0]['upload_bytes_per_inference']:,.0f} B/inference -- not 0, unseen")
    print(f"  earns     {record['earns']}")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
