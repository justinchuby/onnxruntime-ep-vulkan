"""Probe: criterion 3(a) on a run that genuinely executes Phi-3.5, messenger armed.

Every clean validation reading this project holds was taken over `add_f32_dispatches_end_to_end`
— one Add, a handful of dispatches.  Since Switch's GQA fixes the EP claims a single large fused
island of the real model, which is where descriptor lifetime, barrier scope and aliasing defects
actually appear, and that frame has never been examined under the layer.

This PROBES.  It asserts nothing about what it will find, because an assertion written before
the reading is a test of the author's guess.  `test_validation_phi35.py` carries the assertions,
written against what this observed.

Usage: python probe_validation_phi35.py [--child]
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

# Resolved by identity through foundry_discovery (issue #11), never by a hardcoded cache
# path: Foundry Local's on-disk layout is versioned by its CLI's internal catalog revision
# (this machine's cache moved from "...-cuda-gpu" to "...-cuda-gpu-2" between when this
# constant was first written and 2026-08-05), and a hardcoded path goes stale silently when
# that happens.  PHI35_MODEL, if set, is still honored as an explicit direct override.
sys.path.insert(0, str(HERE))
from test_phi35 import _PHI35_SPEC, _foundry_discovery  # noqa: E402

MODEL_DISCOVERY_ERROR: str | None = None
_override = os.environ.get("PHI35_MODEL")
if _override:
    MODEL: pathlib.Path | None = pathlib.Path(_override)
else:
    try:
        MODEL = _foundry_discovery.resolve_model_path(_PHI35_SPEC)
    except _foundry_discovery.FoundryDiscoveryError as _exc:
        MODEL = None
        MODEL_DISCOVERY_ERROR = str(_exc)

# Result-identity contract (issue #19 follow-up, Morpheus review on PR #31): the resolved
# model path and its exact content hash are stamped into every JSON record this probe
# writes, computed lazily at write time -- never at import -- so a PHI35_MODEL override, or
# a stale/substituted file silently sitting at the resolved path, can never be absorbed
# invisibly into the evidence. Reuses the same streaming SHA-256 helper the bench/results/
# archival scripts reuse, rather than a divergent hasher. Unlike those scripts, MODEL can be
# None here (resolution can fail outright), so the failure is reported as data instead of
# raising out of a result-writer that must still land its record.
sys.path.insert(0, str(REPO / "rust" / "tools"))
import model_provenance as _model_provenance  # noqa: E402


def _result_identity() -> dict:
    if MODEL is None:
        return {
            "onnx_file": None,
            "onnx_sha256": None,
            "onnx_resolution_error": MODEL_DISCOVERY_ERROR,
        }
    try:
        return {"onnx_file": str(MODEL), "onnx_sha256": _model_provenance.sha256_of(MODEL)}
    except OSError as exc:
        return {"onnx_file": str(MODEL), "onnx_sha256": None, "onnx_hash_error": str(exc)}

#: Printed by the child the instant the last inference returns and before anything is torn
#: down.  Everything the layer said up to here was said while the instance, the device, every
#: descriptor set and every command buffer were live: that is the dispatch window.  Anything
#: after it is teardown, where the production device is leaked and messages are about object
#: lifetimes rather than the dispatch path (R12 — UNOBSERVABLE, never 0).
BOUNDARY = "[CRITERION3A-PHI35] last inference returned; dispatch window closes here"


def _child() -> int:
    import numpy as np  # noqa: F401  (the feeds builder owns the dtypes)
    import onnxruntime as ort

    if MODEL is None:
        raise SystemExit(
            f"[CRITERION3A-PHI35] Phi-3.5 model not resolvable: {MODEL_DISCOVERY_ERROR}"
        )

    ort.set_default_logger_severity(0)
    ort.register_execution_provider_library(
        "VulkanExecutionProvider", os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
    )
    so = ort.SessionOptions()
    so.log_severity_level = 0
    so.add_session_config_entry("ep.enable_validation", "1")
    sess = ort.InferenceSession(
        str(MODEL), so, providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    sys.path.insert(0, str(REPO / "tests" / "ops"))
    from test_phi35 import _build_phi35_feeds  # noqa: PLC0415

    sess.run(None, _build_phi35_feeds())
    sys.stdout.flush()
    sys.stderr.flush()
    print(BOUNDARY, flush=True)
    print(BOUNDARY, file=sys.stderr, flush=True)
    return 0


_VUID = re.compile(r"VUID-")
_VALIDATION_LINE = re.compile(r"\[Vulkan validation\]")


def split_frame(output: str) -> dict:
    """Split a child transcript into the dispatch window and the teardown window."""
    lines = output.splitlines()
    boundary = max(
        (i for i, ln in enumerate(lines) if BOUNDARY in ln), default=-1
    )
    in_frame = [ln for i, ln in enumerate(lines) if _VUID.search(ln) and i <= boundary]
    teardown = [ln for i, ln in enumerate(lines) if _VUID.search(ln) and i > boundary]
    messenger = [ln for i, ln in enumerate(lines) if _VALIDATION_LINE.search(ln) and i <= boundary]
    return {
        "boundary_seen": boundary >= 0,
        "in_frame_vuids": in_frame,
        "teardown_vuids": teardown,
        "messenger_lines_in_frame": messenger,
    }


def run_arm(arm: str = "clean", *, timeout: int = 5400) -> dict:
    """Run one Phi-3.5 arm under validation and return the reading.

    One definition of an arm, shared by this probe and `test_validation_phi35.py`: two
    builders would be two definitions of the case, and the arms could then differ for a
    reason nobody wrote down.
    """
    out = REPO / "bench" / "results"
    out.mkdir(parents=True, exist_ok=True)
    selector = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset")
    counters = out / f"validation_phi35_counters-dev{selector}-{arm}.json"
    counters.unlink(missing_ok=True)

    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    env["ONNXRUNTIME_EP_VULKAN_VALIDATE"] = "1"
    if arm == "liveness":
        # Best-practices messages arrive at WARNING severity through the EP's own messenger,
        # in this process and inside this frame.  They are the only way found to make the
        # callback speak during a healthy run: the EP's plant lives in a Rust test path that
        # the ORT session never takes, and our messenger subscribes to ERROR|WARNING only,
        # so a clean run is silent whether the callback is live or dead.
        env["VK_LAYER_ENABLES"] = "VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT"
    else:
        env.pop("VK_LAYER_ENABLES", None)
    r = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--child"],
        env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    output = (r.stdout or "") + "\n" + (r.stderr or "")
    frame = split_frame(output)

    doc = {
        "probe": "criterion 3(a) — validation over a real Phi-3.5 execution",
        "arm": arm,
        "device_selector": selector,
        "child_exit_code": r.returncode,
        "counters_written": counters.is_file(),
        "frame": {k: (v if not isinstance(v, list) else v[:20]) for k, v in frame.items()},
        "in_frame_vuid_count": len(frame["in_frame_vuids"]),
        "teardown_vuid_count": len(frame["teardown_vuids"]),
        "messenger_lines_in_frame_count": len(frame["messenger_lines_in_frame"]),
        "transcript_tail": output[-2500:],
    }
    if counters.is_file():
        c = json.loads(counters.read_text(encoding="utf-8"))
        doc["counters"] = {
            k: c.get(k) for k in (
                "claimed_nodes", "islands_offered", "viable_islands_retained",
                "ledger_gate", "ledger_hits", "unproven_declines", "device_losses",
                "dispatches_executed", "model_output_equivalence",
                # The device the run actually used, as the run reported it: the selector is
                # a request, and its number is not the allocator's enumeration index.
                "alloc_device_frame_session_devices",
            )
        }
    doc.update(_result_identity())
    (out / f"validation_phi35_probe-dev{selector}-{arm}.json").write_text(
        json.dumps({k: v for k, v in doc.items() if k != "transcript_tail"}, indent=2),
        encoding="utf-8",
    )
    return doc


def main() -> int:
    doc = run_arm(os.environ.get("PROBE_ARM", "clean"))
    tail = doc.pop("transcript_tail", "")
    print(json.dumps(doc, indent=2))
    print("\n--- child transcript tail ---\n" + tail)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        raise SystemExit(_child())
    raise SystemExit(main())
