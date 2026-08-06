"""Current-state reading of the real Phi-3.5 artifact: how many nodes does the EP claim?

Written to answer one question with an artifact rather than an assertion, because the claim
that the EP is at `0/363` has now been routed twice from a diagnostic taken against an older
build.  The frame of a result is the binary that produced it, so this records the ledger digest
and the DLL's mtime beside the count.

Counts only.  No duration is quoted: `gpu_steady_tail` detects foreign work through
non-stationarity alone, six agents share this machine, and no device-clock figure is certifiable
right now.  A claimed-node count needs no clock.
"""

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
# No PHI35_MODEL pre-resolver override here: this is a LIVE tool (issue #19), and a raw path
# consulted before the resolver would bypass the exact variant+execution-provider validation the
# resolver exists to enforce, silently accepting a different model/provider than the one this
# probe claims to measure. Archival bench/results/ scripts are the ones with the explicit
# override, because their job is to replay a specific historical artifact by design.
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

CHILD = "--child"

if len(sys.argv) > 1 and sys.argv[1] == CHILD:
    import onnxruntime as ort

    ort.register_execution_provider_library(
        "VulkanExecutionProvider", os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
    )
    so = ort.SessionOptions()
    so.log_severity_level = 2
    sess = ort.InferenceSession(
        str(MODEL), so, providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    if os.environ.get("PHI35_PROBE_RUN") == "1":
        # The dynamic re-translate happens at Compute time, not at session creation, so a
        # session-only probe never reaches it. One inference is enough to provoke it.
        import numpy as np

        sys.path.insert(0, str(REPO / "tests" / "ops"))
        from test_phi35 import _build_phi35_feeds  # noqa: PLC0415

        _ = np  # feeds builder owns the dtypes
        sess.run(None, _build_phi35_feeds())
    raise SystemExit(0)

out = REPO / "bench" / "results"
out.mkdir(parents=True, exist_ok=True)
counters = out / "phi35_claim_reading.json"
if counters.exists():
    counters.unlink()

env = dict(os.environ)
env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
r = subprocess.run(
    [sys.executable, str(pathlib.Path(__file__).resolve()), CHILD],
    env=env,
    capture_output=True,
    encoding="utf-8",
    errors="replace",
    timeout=3600,
)
if not counters.is_file():
    print("ERROR(instrument): no counters file was written; this run says nothing.")
    print(r.stdout[-3000:])
    print(r.stderr[-3000:])
    raise SystemExit(2)

c = json.loads(counters.read_text(encoding="utf-8"))
ledger = REPO / "evidence" / "proof_ledger.jsonl"
dll = pathlib.Path(env["ONNXRUNTIME_VULKAN_EP_LIB"])
reading = {
    "device_selector": env.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset"),
    "ledger_entries": sum(
        1 for ln in ledger.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()
    ),
    "dll_mtime": dll.stat().st_mtime,
    "claimed_nodes": c.get("claimed_nodes"),
    "ledger_hits": c.get("ledger_hits"),
    "unproven_declines": c.get("unproven_declines"),
    "unproven_forms_claimed": c.get("unproven_forms_claimed"),
    "islands_offered": c.get("islands_offered"),
    "viable_islands_retained": c.get("viable_islands_retained"),
    "ledger_gate": c.get("ledger_gate"),
    "claimed_form_evidence": c.get("claimed_form_evidence"),
}
(out / "phi35_claim_reading_summary.json").write_text(
    json.dumps(reading, indent=2), encoding="utf-8"
)
for k, v in reading.items():
    print(f"{k:26} {v}")
