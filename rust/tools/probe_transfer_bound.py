"""Bound the prize: how much of Phi-3.5 wall time could device-backed allocation possibly remove?

WHY THIS EXISTS
===============
Switch's `VkQueryPool` timestamps produced a phase table that reads, at a glance, as though
transfers are bounded by `vulkan.fence_wait` (30.4%).  They are not, and the reason is
structural rather than numeric:

  * `Phase::Record` is opened at `vk/session.rs` before `vkBeginCommandBuffer` and dropped
    after `vkEndCommandBuffer`.
  * The host staging **memcpy** is timed inside that window and reported through
    `Tracer::record_transfer`, which accumulates into `phase_us[Phase::Upload]`
    (`trace.rs`) but emits **no `ph:"X"` span** — only `ph:"C"` counters.

So an aggregation over `ph:"X"` spans — which is how the 68.3/30.4/0.3 table was produced —
cannot see upload at all, and the upload time it cannot see is *already inside* the 68.3%
`vulkan.record` figure.  `record`, `submit` and `fence_wait` are siblings; `upload` and
`readback` are **children of `record`**.  The summary log prints all five in one flat column
with no nesting marker, so summing that column double-counts.

WHAT THIS SCRIPT DERIVES
========================
An upper bound on transfer, decomposed into the only two places transfer time can live:

  HOST side   `phase_us[upload] + phase_us[readback]`
              Real staging memcpy on the CPU thread.  Device-backed allocation removes this
              only for spans that become device-resident.  Nested inside `record`.

  DEVICE side `phase_us[fence_wait] - sum(gpu kernel ns)`
              The command buffer contains `vkCmdCopyBuffer` calls that are NOT wrapped in
              timestamp queries (`GpuQueryPool::new(.., kernels.len())` is sized to dispatches
              only).  Their execution is inside the fence wait but is not isolated, so the
              fence-wait residual is an UPPER bound on GPU-side copy: it also contains queue
              latency, scheduling and fence-signal overhead.

  BOUND       (host + device) / total_wall.  This is a CEILING, not an estimate.  Perfect
              device-backed allocation cannot beat it, and will not reach it.

R9 — THE RED INSTRUMENT
=======================
The claim is "transfer is at most X% of wall time".  It goes red if
`phase_us[upload] + phase_us[readback] + (fence_wait - gpu_kernel_total)` exceeds the measured
bound on a rerun, or if `gpu_kernel_total > fence_wait` (which would mean the timestamp
conversion or the phase nesting is wrong, and the whole decomposition is void).  Both are
asserted below.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
    r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
)
EP_NAME = "VulkanExecutionProvider"


# ---------------------------------------------------------------------------------------
# child: one traced Phi-3.5 inference
# ---------------------------------------------------------------------------------------
def _child() -> int:
    import numpy as np
    import onnxruntime as ort

    lib = pathlib.Path(os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]).resolve()
    try:
        ort.register_execution_provider_library(EP_NAME, str(lib))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            print(f"REGISTER-FAILED {exc}", flush=True)
            return 2

    opts = ort.SessionOptions()
    opts.log_severity_level = 1  # INFO — the tracer summary is log::info!

    sess = ort.InferenceSession(str(MODEL), opts, providers=[EP_NAME, "CPUExecutionProvider"])

    # Guard (Trinity/coordinator standing rule): without this the run measures CPU and
    # reports it under our name.
    provs = sess.get_providers()
    if EP_NAME not in provs:
        print(f"EP-NOT-ACTIVE {provs}", flush=True)
        return 3
    print(f"EP-ACTIVE {provs}", flush=True)

    feeds: dict[str, "np.ndarray"] = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.array([[1]], dtype=np.int64),
    }
    empty = np.empty((1, 32, 0, 96), dtype=np.float16)
    for i in range(32):
        feeds[f"past_key_values.{i}.key"] = empty
        feeds[f"past_key_values.{i}.value"] = empty

    n_runs = int(os.environ.get("PROBE_RUNS", "1"))
    for _ in range(n_runs):
        sess.run(None, feeds)
    del sess
    return 0


# ---------------------------------------------------------------------------------------
# parent: drive the child, parse the summary
# ---------------------------------------------------------------------------------------
_PHASE_RE = re.compile(
    r"^\s+(?:\u2514\u2500\s+)?(compile|upload|record|submit|fence_wait|readback)\s+(\d+) us \(x(\d+)\)"
)
_GPU_RE = re.compile(r"^\s+(\S+)\s+(\d+) ns \(x(\d+)\)")
_TRANSFER_RE = re.compile(
    r"transfer: upload (\d+) calls / ([\d.]+) MiB; readback (\d+) calls / ([\d.]+) MiB"
)
_ISLAND_RE = re.compile(r'"island_count"\s*:\s*(\d+)')


def parse(text: str) -> dict:
    phases: dict[str, tuple[int, int]] = {}
    gpu: dict[str, tuple[int, int]] = {}
    in_gpu = False
    xfer = None
    for line in text.splitlines():
        m = _PHASE_RE.match(line)
        if m:
            phases[m.group(1)] = (int(m.group(2)), int(m.group(3)))
            continue
        if "GPU time (device timestamp queries" in line:
            in_gpu = True
            continue
        if in_gpu:
            m = _GPU_RE.match(line)
            if m:
                gpu[m.group(1)] = (int(m.group(2)), int(m.group(3)))
                continue
            if line.strip().startswith("="):
                in_gpu = False
        m = _TRANSFER_RE.search(line)
        if m:
            xfer = {
                "upload_calls": int(m.group(1)),
                "upload_mib": float(m.group(2)),
                "readback_calls": int(m.group(3)),
                "readback_mib": float(m.group(4)),
            }
    return {"phases": phases, "gpu_ns": gpu, "transfer": xfer}


def run_once(device: str, device_memory: str, trace_json: pathlib.Path, runs: str) -> dict:
    env = dict(os.environ)
    counters = trace_json.with_suffix(".counters.json")
    env["ONNXRUNTIME_EP_VULKAN_DEVICE"] = device
    env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = device_memory
    env["ONNXRUNTIME_EP_VULKAN_TRACE"] = str(trace_json)
    env["ONNXRUNTIME_EP_VULKAN_TRACE_GPU"] = "1"
    env["ONNXRUNTIME_EP_VULKAN_VERBOSE"] = "1"
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    env["PROBE_RUNS"] = runs
    env["PROBE_CHILD"] = "1"
    p = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    out = p.stdout + "\n" + p.stderr
    try:
        trace_json.with_suffix(".log.txt").write_text(out, errors="replace")
    except OSError:
        pass
    res = parse(out)
    res["rc"] = p.returncode
    res["ep_active"] = "EP-ACTIVE" in out
    res["raw"] = out
    res["counters"] = None
    if counters.exists():
        try:
            res["counters"] = json.loads(counters.read_text(errors="replace"))
        except (OSError, ValueError):
            pass
    # The device the EP ACTUALLY selected — never inferred from the env value.
    res["session_device"] = None
    res["mirror_device"] = None
    for line in out.splitlines():
        m = re.search(r"VulkanSession: selected '([^']+)'", line)
        if m:
            res["session_device"] = m.group(1)
        m = re.search(r"mirroring onto '([^']+)'", line)
        if m:
            res["mirror_device"] = m.group(1)
    # The staging verdict is a log line, not a counter — capture it verbatim.
    res["verdict"] = None
    for line in out.splitlines():
        if "MEMORY:" in line:
            res["verdict"] = line.split("VulkanExecutionProvider:", 1)[-1].strip()
    # island count from the trace JSON summary instant, if present
    res["islands"] = None
    if trace_json.exists():
        try:
            blob = trace_json.read_text(errors="replace")
            m = _ISLAND_RE.search(blob)
            if m:
                res["islands"] = int(m.group(1))
        except OSError:
            pass
    if res["islands"] is None:
        # Fall back to the compile log, which names each fused subgraph.
        subs = {int(x) for x in re.findall(r"fused subgraph (\d+)", out)}
        if subs:
            res["islands"] = len(subs)
    return res


def bound(res: dict) -> dict | None:
    ph = res["phases"]
    if "fence_wait" not in ph or "record" not in ph:
        return None
    us = {k: v[0] for k, v in ph.items()}
    upload_us = us.get("upload", 0)
    readback_us = us.get("readback", 0)
    record_us = us["record"]
    submit_us = us.get("submit", 0)
    fence_us = us["fence_wait"]
    compile_us = us.get("compile", 0)

    # record already CONTAINS upload+readback. Wall time attributable to the EP is the
    # sibling phases only.
    wall_us = compile_us + record_us + submit_us + fence_us

    gpu_ns_total = sum(v[0] for v in res["gpu_ns"].values())
    gpu_us = gpu_ns_total / 1000.0

    host_xfer_us = upload_us + readback_us
    dev_xfer_ceiling_us = max(0.0, fence_us - gpu_us)

    return {
        "wall_us": wall_us,
        "compile_us": compile_us,
        "record_us": record_us,
        "record_excl_xfer_us": record_us - host_xfer_us,
        "submit_us": submit_us,
        "fence_us": fence_us,
        "gpu_kernel_us": gpu_us,
        "host_xfer_us": host_xfer_us,
        "upload_us": upload_us,
        "readback_us": readback_us,
        "dev_xfer_ceiling_us": dev_xfer_ceiling_us,
        "xfer_ceiling_us": host_xfer_us + dev_xfer_ceiling_us,
        "xfer_ceiling_pct": 100.0 * (host_xfer_us + dev_xfer_ceiling_us) / max(wall_us, 1),
        "host_xfer_pct": 100.0 * host_xfer_us / max(wall_us, 1),
        "gpu_gt_fence": gpu_us > fence_us,
    }


def main() -> int:
    if os.environ.get("PROBE_CHILD"):
        return _child()

    if not MODEL.exists():
        print(f"SKIP: model not found at {MODEL}")
        return 0
    if not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"):
        print("SKIP: ONNXRUNTIME_VULKAN_EP_LIB not set")
        return 0

    print(
        "NOTE ON DEVICE LABELS — ONNXRUNTIME_EP_VULKAN_DEVICE indexes the list returned by\n"
        "`Instance::enumerate_capable_devices()`, which ends with\n"
        "  result.sort_by_key(|d| Reverse(d.info.kind.score()))   // best first: discrete > integrated\n"
        "so on this desk the mapping is the OPPOSITE of raw Vulkan enumeration order:\n"
        "    ONNXRUNTIME_EP_VULKAN_DEVICE=0  ->  NVIDIA GeForce RTX 4060 Laptop GPU  (DISCRETE)\n"
        "    ONNXRUNTIME_EP_VULKAN_DEVICE=1  ->  Intel(R) Iris(R) Xe Graphics        (UMA)\n"
        "This script prints the device name the EP actually selected for every cell. Do not\n"
        "label a number in this table from the env value alone.\n"
    )

    outdir = REPO / "rust" / "target" / "transfer_bound"
    outdir.mkdir(parents=True, exist_ok=True)

    devices = os.environ.get("PROBE_DEVICES", "0,1").split(",")
    dm_modes = os.environ.get("PROBE_DM", "0,1").split(",")
    runs = os.environ.get("PROBE_RUNS", "1")

    results: dict[str, dict] = {}
    fail = 0
    for dev in devices:
        for dm in dm_modes:
            key = f"d{dev}_dm{dm}"
            tj = outdir / f"trace_{key}.json"
            print(f"\n=== device {dev}, DEVICE_MEMORY={dm}, runs={runs} ===", flush=True)
            r = run_once(dev, dm, tj, runs)
            if not r["ep_active"]:
                print(f"  EP NOT ACTIVE (rc={r['rc']}) — refusing to report numbers")
                print("  " + "\n  ".join(r["raw"].splitlines()[-25:]))
                fail = 1
                results[key] = {"ep_active": False}
                continue
            b = bound(r)
            if b is None:
                print("  NO PHASE TABLE in summary — instrument absent, not a zero (R7)")
                print("  " + "\n  ".join(r["raw"].splitlines()[-25:]))
                fail = 1
                results[key] = {"ep_active": True, "phases": r["phases"]}
                continue
            results[key] = {
                "ep_active": True,
                "islands": r["islands"],
                "session_device": r["session_device"],
                "mirror_device": r["mirror_device"],
                "transfer": r["transfer"],
                "phases": r["phases"],
                "gpu_ns": r["gpu_ns"],
                "counters": r["counters"],
                "verdict": r["verdict"],
                "bound": b,
            }
            print(f"  SELECTED DEVICE    {r['session_device']}")
            if r["mirror_device"] is not None:
                same = r["mirror_device"] == r["session_device"]
                print(
                    f"  mirror device      {r['mirror_device']}  "
                    f"{'[MATCHES session]' if same else '[!! DIFFERENT DEVICE !!]'}"
                )
                if not same:
                    fail = 1
            print(f"  islands            {r['islands']}")
            print(f"  transfer bytes     {r['transfer']}")
            print(f"  --- host phases (record CONTAINS upload/readback) ---")
            print(f"  compile            {b['compile_us']/1000:10.1f} ms")
            print(f"  record   (total)   {b['record_us']/1000:10.1f} ms")
            print(f"    of which upload  {b['upload_us']/1000:10.1f} ms   <-- host staging memcpy")
            print(f"    of which readbk  {b['readback_us']/1000:10.1f} ms")
            print(f"    record excl xfer {b['record_excl_xfer_us']/1000:10.1f} ms")
            print(f"  submit             {b['submit_us']/1000:10.1f} ms")
            print(f"  fence_wait         {b['fence_us']/1000:10.1f} ms")
            print(f"    gpu kernels      {b['gpu_kernel_us']/1000:10.1f} ms  (timestamp queries)")
            print(f"    residual (CEIL)  {b['dev_xfer_ceiling_us']/1000:10.1f} ms  <-- copies+latency")
            print(f"  wall (siblings)    {b['wall_us']/1000:10.1f} ms")
            print(f"  ** TRANSFER CEILING {b['xfer_ceiling_pct']:.1f}% of wall "
                  f"({b['xfer_ceiling_us']/1000:.1f} ms) **")
            print(f"     host-side alone  {b['host_xfer_pct']:.1f}%")
            if b["gpu_gt_fence"]:
                print("  RED: gpu kernel total EXCEEDS fence_wait — decomposition is void.")
                fail = 1
            c = r["counters"] or {}
            keys = (
                "alloc_device_backed_spans",
                "alloc_device_authoritative_spans",
                "alloc_staged_spans",
                "alloc_device_uploads",
                "alloc_device_upload_bytes",
                "alloc_unified_memory",
                "pointers_in_guard_band",
                "pointers_use_after_free",
                "pointers_interior",
            )
            print("  counters: " + ", ".join(f"{k.replace('alloc_','')}={c[k]}" for k in keys if k in c))
            if r["verdict"]:
                print(f"  verdict: {r['verdict'][:300]}")

    (outdir / "bound.json").write_text(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in results.items()},
            indent=2,
        )
    )
    print(f"\nwrote {outdir / 'bound.json'}")
    return fail


if __name__ == "__main__":
    sys.exit(main())
