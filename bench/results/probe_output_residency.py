"""Can an EP-owned KV arena exist at all? Ask ORT, not the headers.

THE QUESTION
------------
Tank established the seam and its absence in one sentence: `bind_target_for` is called on
**inputs only** (`vk/session.rs`), and the output path is an unconditional download of
`actual_output_byte_sizes.iter().sum()` into whatever pointer ORT hands back. So today no
configuration can decline the `present.*` readback.

The proposed fix is an output-side `bind_target_for`: when ORT's output tensor already lives
in this EP's device memory, dispatch straight into it and skip the round trip entirely.

That fix has a precondition which no amount of reading ORT's headers settles:

    Does ORT ever ask THIS EP's allocator for a fused node's OUTPUT buffer?

If it does, an arena is buildable and the output-side bind is the seam.
If it never does, ORT owns every output as host memory, the round trip is structural for
plugin EPs, and that is a finding for the ORT team rather than something to work around
quietly here.

HOW IT IS ANSWERED
------------------
`write_outputs_to_ort` already has to ask, because it must know whether the pointer ORT
returned is writable host memory or an opaque device handle. It calls
`transfer::host_backing_for`. Two counters now record which way that call went:

    outputs_device_resident   ORT allocated this output through the EP's device provider
    outputs_host_resident     ordinary host memory

This probe runs the SAME model twice — device allocator disarmed, then armed — and reads
those counters. The disarmed lane is the control: it must report 0 device-resident outputs,
because with no device provider registered there is nothing for ORT to allocate into. A lane
that reports device residency with the allocator OFF would mean the counter is measuring
something other than what it claims.

WHAT WOULD FALSIFY THE CONCLUSION
---------------------------------
* If the armed lane shows `outputs_device_resident > 0`, "ORT forbids it" is refuted and the
  arena is buildable. The number also says how many of the 65 outputs are reachable.
* If the armed lane shows 0 while `alloc_device_frame == SHARED` — i.e. the provider WAS
  registered and ORT still declined to use it for outputs — that is the structural finding.
* If `alloc_device_frame` is not `SHARED` in the armed lane, this probe proves NOTHING: the
  provider never took the offer, so ORT was never in a position to allocate into it. That is
  reported as ERROR(instrument), not as a negative result. An unarmed lane answering "no" is
  the switched-off smoke detector again.

Usage:
    python bench/results/probe_output_residency.py
    python bench/results/probe_output_residency.py --worker --arm armed
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
EP_NAME = "VulkanExecutionProvider"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

# ARCHIVAL: pinned to the exact Foundry cache layout this investigation measured against
# (the pre-2026-08-05 "...-cuda-gpu/cuda-int4-rtn-block-32/..." catalog revision, issue
# #11). Intentionally NOT auto-resolved against the live cache: a live resolver could
# silently pick a *different* cached revision than the one this result was measured
# against, which would misattribute a new run to an old artifact. Override PHI35_MODEL to
# replay against a different artifact explicitly (issue #19).
PHI = pathlib.Path(
    os.environ.get(
        "PHI35_MODEL",
        r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
        r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
        r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    )
)

LAYERS, KV_HEADS, HEAD_DIM = 32, 32, 96

# Result-identity contract (issue #19 follow-up, Morpheus review on PR #31): the resolved model
# path and its exact content hash are stamped into the output record below, computed lazily
# (only once the model has already been used successfully) so a PHI35_MODEL override or a
# stale/wrong cached file can never be silently absorbed into the evidence. Reuses the streaming
# SHA-256 helper `model_provenance.sha256_of` rather than a 23rd divergent hasher.
sys.path.insert(0, str(ROOT / "rust" / "tools"))
import model_provenance as _model_provenance  # noqa: E402


def _result_identity() -> dict:
    return {
        "onnx_file": str(PHI),
        "onnx_sha256": _model_provenance.sha256_of(PHI),
    }


def worker(past_len: int, iters: int) -> int:
    ort.register_execution_provider_library(EP_NAME, os.environ["ONNXRUNTIME_VULKAN_EP_LIB"])
    sess = ort.InferenceSession(
        str(PHI),
        providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    if EP_NAME not in sess.get_providers():
        raise SystemExit(f"ERROR(instrument): {EP_NAME} absent from {sess.get_providers()}")
    sess.disable_fallback()
    feeds = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
    }
    kv = np.zeros((1, KV_HEADS, past_len, HEAD_DIM), dtype=np.float16)
    for layer in range(LAYERS):
        feeds[f"past_key_values.{layer}.key"] = kv
        feeds[f"past_key_values.{layer}.value"] = kv
    for _ in range(iters):
        sess.run(None, feeds)
    return 0


def run_lane(arm: str, past_len: int, iters: int) -> dict:
    scratch = ROOT / "bench" / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    cfile = scratch / f"outres_{arm}_p{past_len}.counters.json"
    cfile.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(cfile)
    if arm == "armed":
        env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
    else:
        env.pop("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", None)
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker",
         "--arm", arm, "--past-len", str(past_len), "--iters", str(iters)],
        capture_output=True, text=True, env=env,
    )
    if not cfile.is_file():
        raise SystemExit(
            f"ERROR(instrument): lane {arm} wrote no counters file.\n"
            + (proc.stdout or "") + (proc.stderr or "")
        )
    raw = json.loads(cfile.read_text(encoding="utf-8"))
    c = raw.get("counters", raw)
    frame = None
    for key in ("alloc_device_frame", "device_frame"):
        if key in raw:
            frame = raw[key]
        if key in c:
            frame = c[key]
    return {
        "arm": arm,
        "exit": proc.returncode,
        "past_len": past_len,
        "iters": iters,
        "compute_calls": c.get("compute_calls"),
        "dispatches_executed": c.get("dispatches_executed"),
        "device_losses": c.get("device_losses"),
        "outputs_device_resident": c.get("outputs_device_resident"),
        "outputs_host_resident": c.get("outputs_host_resident"),
        "alloc_device_frame": frame,
        "raw_keys_with_frame": sorted(k for k in list(raw) + list(c) if "frame" in k),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", default="armed", choices=("armed", "default"))
    ap.add_argument("--past-len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        return worker(args.past_len, args.iters)

    lanes = [run_lane(a, args.past_len, args.iters) for a in ("default", "armed")]

    print()
    print("=" * 78)
    print("CAN ORT PLACE A FUSED-NODE OUTPUT IN THIS EP'S DEVICE MEMORY?")
    print("=" * 78)
    for p in lanes:
        total = (p["outputs_device_resident"] or 0) + (p["outputs_host_resident"] or 0)
        print(
            f"  {p['arm']:8s} exit={p['exit']}  calls={p['compute_calls']}  "
            f"dispatches={p['dispatches_executed']}  outputs_seen={total}  "
            f"device_resident={p['outputs_device_resident']}  "
            f"host_resident={p['outputs_host_resident']}  frame={p['alloc_device_frame']}"
        )

    default, armed = lanes

    verdict, detail = "UNDETERMINED", ""
    for p in lanes:
        if not (p["compute_calls"] or 0) or not (p["dispatches_executed"] or 0):
            verdict = "ERROR(instrument)"
            detail = (
                f"lane {p['arm']} executed no EP work (calls={p['compute_calls']}, "
                f"dispatches={p['dispatches_executed']}). A lane that never entered the EP "
                "cannot report where the EP's outputs live."
            )
        if p["device_losses"]:
            verdict = "ERROR(instrument)"
            detail = f"lane {p['arm']} lost the device; nothing in it is a measurement."
    if verdict != "ERROR(instrument)":
        if default["outputs_device_resident"]:
            verdict = "ERROR(instrument)"
            detail = (
                "the CONTROL lane reports device-resident outputs with the device allocator "
                "disarmed. No provider is registered there, so there is nothing for ORT to "
                "allocate into — the counter is measuring something other than residency."
            )
        elif armed["outputs_device_resident"]:
            verdict = "OUTPUTS_ARE_DEVICE_ALLOCATED"
            detail = (
                f"ORT allocated {armed['outputs_device_resident']} output tensor(s) through "
                "this EP's device provider, so an output-side `bind_target_for` has something "
                "to bind. That is NOT the same as the round trip being removable: whether the "
                "bound bytes reach the caller is a separate question, answered by "
                "probe_bound_output_correctness.py, and the answer measured on 2026-08-02 was "
                "HOST_STAGING_IS_AUTHORITATIVE — the bound lane returns all zeros."
            )
        elif armed["alloc_device_frame"] == "SHARED":
            verdict = "ORT_ROUND_TRIP_IS_STRUCTURAL"
            detail = (
                "the device provider WAS registered and sharing the engine's VkDevice "
                "(alloc_device_frame=SHARED), and ORT still allocated every fused-node output "
                "in host memory. There is nothing for an output-side bind to bind. The KV "
                "round trip is not a choice this EP is making; it is ORT's contract for "
                "plugin EPs, and it belongs in front of the ORT team."
            )
        else:
            verdict = "ERROR(instrument)"
            detail = (
                f"armed lane reports alloc_device_frame={armed['alloc_device_frame']!r}, not "
                "SHARED. The provider never took the offer, so ORT was never in a position to "
                "allocate an output into it. This lane proves nothing either way — it is a "
                "switched-off detector reporting no smoke."
            )

    print()
    print(f"  VERDICT: {verdict}")
    for line in detail.split(". "):
        if line.strip():
            print(f"    {line.strip().rstrip('.')}.")

    rec = HERE / "output_residency.json"
    rec.write_text(
        json.dumps(
            {**_result_identity(), "lanes": lanes, "verdict": verdict, "detail": detail},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  record: {rec}")
    return 0 if verdict in ("OUTPUTS_ARE_DEVICE_ALLOCATED", "ORT_ROUND_TRIP_IS_STRUCTURAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
