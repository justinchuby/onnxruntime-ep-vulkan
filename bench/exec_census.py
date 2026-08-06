"""Per-run execution census: how many node-executions land on each EP, and where.

Mouse, 2026-07-31.  The falsifier for "N nodes claimed" is not the claim log — it is ORT's
own profile.  This runs Phi-3.5 three times in one session and counts node executions by
provider, then names the CPU-side ops so the residual is attributable rather than a number.

Output → bench/results/exec_census-dev{N}.json.  Nothing is written to the repo root.

Usage::

    $env:ONNXRUNTIME_EP_VULKAN_DEVICE="0"; python bench/exec_census.py
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

import numpy as np
import onnxruntime as ort

_HERE = pathlib.Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_ROOT = _HERE.parent

sys.path.insert(0, str(_ROOT / "rust" / "tools"))
import foundry_discovery as _foundry_discovery  # noqa: E402

# Resolved by identity (variant name + execution provider), not by a hardcoded path: Foundry
# Local's own on-disk cache layout is versioned by its CLI's internal catalog revision (issue
# #11), and a hardcoded path silently goes stale when that happens with no code change on either
# side. See rust/tools/foundry_discovery.py for the full discovery contract (fail-loud, never
# guessed).
_PHI35_SPEC = _foundry_discovery.FoundryModelSpec(
    variant_name="Phi-3.5-mini-instruct-cuda-gpu",
    execution_provider="CUDAExecutionProvider",
    onnx_filename="phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    download_alias="phi-3.5-mini",
)
try:
    _ONNX_FILE = _foundry_discovery.resolve_model_path(_PHI35_SPEC)
except _foundry_discovery.FoundryDiscoveryError as exc:
    raise SystemExit(f"ERROR(instrument): Phi-3.5 model not resolvable: {exc}")

EP_NAME = "VulkanExecutionProvider"
EP_LIB = pathlib.Path(os.environ["ONNXRUNTIME_VULKAN_EP_LIB"])
N_RUNS = 3


def main() -> int:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    dev = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")

    try:
        ort.register_execution_provider_library(EP_NAME, str(EP_LIB))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise

    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = str(_RESULTS / f"exec_census_dev{dev}")

    sess = ort.InferenceSession(
        str(_ONNX_FILE),
        sess_options=opts,
        providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )

    # R9: assert the EP is actually in the session before comparing anything.  A census taken
    # against a session that fell back to CPU is a census of the CPU EP.
    providers = sess.get_providers()
    if EP_NAME not in providers:
        print(f"ERROR(instrument): {EP_NAME} not in session providers {providers}")
        return 2

    feeds: dict[str, np.ndarray] = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.array([[1]], dtype=np.int64),
    }
    empty_kv = np.empty((1, 32, 0, 96), dtype=np.float16)
    for layer in range(32):
        feeds[f"past_key_values.{layer}.key"] = empty_kv
        feeds[f"past_key_values.{layer}.value"] = empty_kv

    outs = [sess.run(None, feeds) for _ in range(N_RUNS)]
    prof = pathlib.Path(sess.end_profiling())

    events = json.loads(prof.read_text(encoding="utf-8"))
    by_provider: collections.Counter[str] = collections.Counter()
    cpu_ops: collections.Counter[str] = collections.Counter()
    cpu_nodes: collections.Counter[str] = collections.Counter()
    for e in events:
        if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time"):
            continue
        args = e.get("args", {})
        ep = args.get("provider", "?")
        by_provider[ep] += 1
        if ep == "CPUExecutionProvider":
            cpu_ops[args.get("op_name", "?")] += 1
            cpu_nodes[e["name"].removesuffix("_kernel_time")] += 1

    identical = all(
        np.array_equal(a, b) for r in outs[1:] for a, b in zip(outs[0], r, strict=True)
    )
    argmax = int(np.argmax(outs[0][0].reshape(-1)))

    result = {
        "device_selector": dev,
        "providers": providers,
        "runs": N_RUNS,
        "node_executions_by_provider": dict(by_provider),
        "cpu_ops_per_session": {k: v // N_RUNS for k, v in sorted(cpu_ops.items())},
        "cpu_nodes": sorted(cpu_nodes),
        "cross_run_identical_all_outputs": identical,
        "argmax_run0": argmax,
    }
    out_path = _RESULTS / f"exec_census-dev{dev}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    prof.unlink(missing_ok=True)

    print(json.dumps(result, indent=2))
    print(f"[exec_census] → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
