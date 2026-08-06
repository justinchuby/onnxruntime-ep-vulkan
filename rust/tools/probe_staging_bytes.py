"""Session staging bytes, measured with NO tracing flag set.

# Why this exists

Persistent weight residency (the ~95% lever) is to be verified **on bytes, not wall time**:
bytes are deterministic, wall time swings 9.5x under contention. But until now the bytes were
recorded in two places and neither was usable for that check on a default run:

  * ``alloc_device_upload_bytes`` counts copies through the *allocator's* device-memory provider,
    which only exists when ``ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1``. It read **0** on a run whose
    ``cmd_upload`` phase was 15.2 seconds.
  * ``Tracer::record_transfer`` counted the session's staging copies -- and early-returned unless
    ``ONNXRUNTIME_EP_VULKAN_TRACE`` or verbose was set. The default run recorded nothing.

``counters::staging`` now counts staging unconditionally and it lands in the counters JSON every
run. This probe is the falsifier for that claim: it sets **only** the counters file, runs the
model 1/2/3 times, and prints the byte totals.

# What to read

If weights are re-uploaded every inference, ``session_staging_upload_bytes`` is linear in the run
count. **When residency lands the sweep goes flat, and that is the win** -- but read
``staging_sentence`` in the output before believing a zero: this counter cannot tell "resident"
from "the hook moved".

Env: ``ONNXRUNTIME_VULKAN_EP_LIB`` (required), ``ONNXRUNTIME_EP_VULKAN_DEVICE`` (default 0),
``PROBE_SWEEP`` (default ``1,2,3``), ``PROBE_DM`` (default unset = provider off).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

# Resolved by identity (variant name + execution provider), not by a hardcoded path: Foundry
# Local's own on-disk cache layout is versioned by its CLI's internal catalog revision (issue
# #11), and a hardcoded path silently goes stale when that happens with no code change on either
# side. See foundry_discovery.py for the full discovery contract (fail-loud, never guessed).
import foundry_discovery as _foundry_discovery  # noqa: E402

_PHI35_SPEC = _foundry_discovery.FoundryModelSpec(
    variant_name="Phi-3.5-mini-instruct-cuda-gpu",
    execution_provider="CUDAExecutionProvider",
    onnx_filename="phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    download_alias="phi-3.5-mini",
)
try:
    MODEL = _foundry_discovery.resolve_model_path(_PHI35_SPEC)
except _foundry_discovery.FoundryDiscoveryError as exc:
    raise SystemExit(f"ERROR(instrument): Phi-3.5 model not resolvable: {exc}")

EP_NAME = "VulkanExecutionProvider"
MIB = 1024.0 * 1024.0


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
    sess = ort.InferenceSession(str(MODEL), opts, providers=[EP_NAME, "CPUExecutionProvider"])

    # Standing rule: a run that does not assert the EP is active measures CPU and reports it
    # under our name. Two fabricated speedups on this project came from skipping this.
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

    for _ in range(int(os.environ.get("PROBE_RUNS", "1"))):
        sess.run(None, feeds)
    del sess
    return 0


def run_once(runs: int, outdir: pathlib.Path) -> dict:
    counters = outdir / f"staging_runs{runs}.counters.json"
    counters.unlink(missing_ok=True)

    env = dict(os.environ)
    env["PROBE_RUNS"] = str(runs)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    env["PROBE_CHILD"] = "1"
    # DELIBERATELY NOT SET: ONNXRUNTIME_EP_VULKAN_TRACE / _TRACE_GPU / verbose. The whole point
    # is that the byte accounting exists without them.
    env.pop("ONNXRUNTIME_EP_VULKAN_TRACE", None)
    env.pop("ONNXRUNTIME_EP_VULKAN_TRACE_GPU", None)
    if "PROBE_DM" in os.environ:
        env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = os.environ["PROBE_DM"]

    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve())],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    out = proc.stdout + proc.stderr
    if "EP-ACTIVE" not in out:
        print(out[-2000:])
        raise SystemExit(f"EP was not active for runs={runs}; refusing to report bytes")
    if not counters.exists():
        raise SystemExit(f"no counters file for runs={runs}")
    return json.loads(counters.read_text())


def main() -> int:
    if os.environ.get("PROBE_CHILD"):
        return _child()
    if not MODEL.exists():
        print(f"SKIP: model not found at {MODEL}")
        return 0
    if "ONNXRUNTIME_VULKAN_EP_LIB" not in os.environ:
        print("SKIP: ONNXRUNTIME_VULKAN_EP_LIB is not set")
        return 0

    outdir = REPO / "rust" / "target" / "staging_bytes"
    outdir.mkdir(parents=True, exist_ok=True)
    sweep = [int(x) for x in os.environ.get("PROBE_SWEEP", "1,2,3").split(",")]

    print(f"device={os.environ.get('ONNXRUNTIME_EP_VULKAN_DEVICE', '(default)')} "
          f"DEVICE_MEMORY={os.environ.get('PROBE_DM', '(unset)')}  no tracing flags set\n")
    print(f"{'runs':>5} {'session_staging_upload_MiB':>27} {'alloc_device_upload_MiB':>24}"
          f" {'uploads':>8} {'compute_calls':>14}")
    rows = []
    for runs in sweep:
        c = run_once(runs, outdir)
        rows.append((runs, c))
        print(
            f"{runs:>5} {c.get('session_staging_upload_bytes', 0) / MIB:>27.2f}"
            f" {c.get('alloc_device_upload_bytes', 0) / MIB:>24.2f}"
            f" {c.get('session_staging_uploads', 0):>8} {c.get('compute_calls', 0):>14}"
        )

    if len(rows) >= 2:
        base = rows[0][1].get("session_staging_upload_bytes", 0)
        print("\nper-run delta (bytes are deterministic; wall time is not):")
        for i in range(1, len(rows)):
            prev = rows[i - 1][1].get("session_staging_upload_bytes", 0)
            cur = rows[i][1].get("session_staging_upload_bytes", 0)
            ratio = (cur / base) if base else float("nan")
            print(
                f"  runs {rows[i - 1][0]} -> {rows[i][0]}: +{(cur - prev) / MIB:.2f} MiB"
                f"   cumulative x{ratio:.4f} of the 1-run figure"
            )
        print(
            "\nLINEAR means the whole weight set is re-staged every inference (the ~95% lever is\n"
            "unclaimed). FLAT means residency landed -- but a flat ZERO is ambiguous: read the\n"
            "staging sentence in the run's verdict, which says this counter cannot distinguish\n"
            "'resident' from 'record_transfer no longer brackets the copy'."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
