#!/usr/bin/env python
"""Is a whole-graph CPU rebuild **observable**? Owner: Tank.

WHAT THIS IS AND IS NOT
-----------------------
Switch established the fault: at ctx 4096 the shipping lane (`DEVICE_MEMORY=1`, `KV_ARENA=0`)
fails `alloc_device failed for input buffer`, ORT rebuilds all 355 nodes on CPU, and the process
**exits 0**. He owns the allocation fix. This probe owns one question only:

    can anyone downstream of that run tell it apart from a normal one?

A decline that misdescribes itself was RAI-012. This is a decline with no message at all: correct
answers come back, the EP has stopped participating, and nothing in the exit code says so. If
nothing distinguishes it, that finding outranks the allocation failure, because the allocation
failure is a bug someone will fix and silence is a property the lane will keep.

Second question, same run: `session.disable_cpu_ep_fallback=1` exists and is wired — the
node-level arm is demonstrated by `probe_cpu_fallback_guard.py`. **Does it fire here?** A silent
whole-graph CPU rebuild that passes a session configured to forbid exactly that is a hole in the
flag, not a detail of the allocator.

ARMS (predictions written before the run)
-----------------------------------------
A ctx=1,    fallback allowed  -> runs on the EP.  dispatches_executed > 0.  THE SCREEN.
B ctx=4096, fallback allowed  -> Switch's fault.  Prediction: exit 0, dispatches_executed == 0.
C ctx=4096, fallback FORBIDDEN-> the question.    A session that forbids CPU fallback must not
                                 come back happy from a run that executed entirely on the CPU.
D ctx=1,    fallback FORBIDDEN, EP made to retain no island -> BLINDNESS CONTROL. If D refuses
                                 and C does not, the flag is wired and has a hole. If D also
                                 fails to refuse, this harness is not driving the flag and C
                                 says nothing.
E ctx=4096, fallback allowed, at the ORT DEFAULT severity -> the user-facing arm. A/B/C/D run
                                 at VERBOSE, where an emission nobody sees by default is still
                                 visible. RAI-013 is exactly the finding that "it was emitted"
                                 and "the user was told" are not the same claim.

A NOTE ON A NULL MANIPULATION THIS PROBE ALREADY WALKED INTO
------------------------------------------------------------
The first run of C and D named `CPUExecutionProvider` in the provider list *and* set
`disable_cpu_ep_fallback`. ORT refuses that combination outright — "Conflicting session
configuration: explicitly added the CPU EP" — before any graph is partitioned, and both arms
reported `session_created=False`. That refusal looks exactly like the guard firing on a CPU
fallback and is not one; quoting it would have been a manipulation that never reached the
mechanism. The forbidding arms name the Vulkan EP alone.

Every arm is screened on `dispatches_executed`, which is the screen that catches this fault:
an arm that executed nothing is not evidence about a lane, it is evidence about a lane's absence.

NO CLOCK. Counts, exit codes, and log text only.

USAGE
    $env:ONNXRUNTIME_VULKAN_EP_LIB=...\\onnxruntime_vulkan_ep.dll
    python rust/tools/probe_silent_cpu_rebuild.py [--ctx 4096]
"""

from __future__ import annotations

import argparse
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
OUT = REPO / "bench" / "results" / "_probe_silent_cpu_rebuild"

CHILD = "--child"


# --------------------------------------------------------------------------------------------
# child: one session, one inference, everything it can be asked written to disk
# --------------------------------------------------------------------------------------------
def _child() -> int:
    import numpy as np
    import onnxruntime as ort

    ctx = int(os.environ["PROBE_CTX"])
    forbid = os.environ.get("PROBE_FORBID_FALLBACK") == "1"
    severity = int(os.environ.get("PROBE_SEVERITY", "0"))

    ort.register_execution_provider_library(
        "VulkanExecutionProvider", os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
    )
    so = ort.SessionOptions()
    so.log_severity_level = severity
    # THE ARM THAT WAS INVALID THE FIRST TIME. Naming `CPUExecutionProvider` in the provider list
    # while `disable_cpu_ep_fallback` is set makes ORT refuse the session for a *configuration*
    # reason — "Conflicting session configuration: explicitly added the CPU EP" — before any
    # graph is partitioned. That refusal looks exactly like the guard firing and is not: it would
    # have been a null manipulation quoted as the measurement. The forbidding arms therefore name
    # the Vulkan EP alone.
    providers = ["VulkanExecutionProvider"] if forbid else [
        "VulkanExecutionProvider",
        "CPUExecutionProvider",
    ]
    if forbid:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    result: dict = {"ctx": ctx, "forbid_fallback": forbid, "providers": providers,
                    "log_severity_level": severity}
    try:
        sess = ort.InferenceSession(str(MODEL), so, providers=providers)
        result["session_created"] = True
        result["get_providers"] = sess.get_providers()
    except Exception as exc:  # noqa: BLE001 — the refusal is the measurement
        result["session_created"] = False
        result["session_error"] = f"{type(exc).__name__}: {exc}"
        pathlib.Path(os.environ["PROBE_RESULT"]).write_text(json.dumps(result), encoding="utf-8")
        return 0

    feeds: dict[str, np.ndarray] = {
        "input_ids": np.ones((1, ctx), dtype=np.int64),
        "attention_mask": np.ones((1, ctx), dtype=np.int64),
    }
    empty_kv = np.empty((1, 32, 0, 96), dtype=np.float16)
    for layer in range(32):
        feeds[f"past_key_values.{layer}.key"] = empty_kv
        feeds[f"past_key_values.{layer}.value"] = empty_kv
    try:
        outs = sess.run(None, feeds)
        result["ran"] = True
        result["logits_shape"] = list(outs[0].shape)
        result["logits_finite"] = bool(np.isfinite(outs[0]).all())
    except Exception as exc:  # noqa: BLE001
        result["ran"] = False
        result["run_error"] = f"{type(exc).__name__}: {exc}"

    pathlib.Path(os.environ["PROBE_RESULT"]).write_text(json.dumps(result), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------------------------
def _arm(name: str, ctx: int, forbid: bool, extra: dict[str, str], severity: int = 0) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    counters = OUT / f"{name}_counters.json"
    result = OUT / f"{name}_result.json"
    for p in (counters, result):
        if p.exists():
            p.unlink()

    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    env["PROBE_CTX"] = str(ctx)
    env["PROBE_RESULT"] = str(result)
    env["PROBE_FORBID_FALLBACK"] = "1" if forbid else "0"
    env["PROBE_SEVERITY"] = str(severity)
    # Pinned explicitly and recorded, both of them, in every arm. An unpinned lane variable is a
    # variable somebody else's default is setting.
    env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
    env["ONNXRUNTIME_EP_VULKAN_KV_ARENA"] = "0"
    env.update(extra)

    r = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), CHILD],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    rec: dict = {
        "arm": name,
        "ctx": ctx,
        "forbid_fallback": forbid,
        "log_severity_level": severity,
        "device_memory": env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"],
        "kv_arena": env["ONNXRUNTIME_EP_VULKAN_KV_ARENA"],
        "extra_env": extra,
        "exit_code": r.returncode,
    }
    rec.update(json.loads(result.read_text(encoding="utf-8")) if result.is_file() else
               {"child_wrote_nothing": True})
    if counters.is_file():
        c = json.loads(counters.read_text(encoding="utf-8"))
        rec["counters"] = {
            k: c.get(k)
            for k in (
                "compile_calls",
                "subgraphs_live",
                "subgraphs_stub",
                "compute_calls",
                "compute_failures",
                "dispatches_executed",
                "claimed_nodes",
                "islands_offered",
                "viable_islands_retained",
                "ledger_hits",
                "unproven_declines",
            )
        }
    else:
        rec["counters"] = None
    log = (r.stdout or "") + (r.stderr or "")
    (OUT / f"{name}_log.txt").write_text(log, encoding="utf-8")
    # ORT's own log goes out through a C++ sink that writes wide characters into this pipe, so a
    # UTF-8 read interleaves NULs. Strip them before matching, or every search for a token ORT
    # emitted answers "absent" and the probe reports blindness as silence.
    flat = log.replace("\x00", "")
    # What could a downstream reader see *without* reading the EP's private counters?
    rec["observable"] = {
        "exit_code": r.returncode,
        "log_bytes": len(log),
        "ep_broken_commitment_warn": "BROKEN COMMITMENT" in flat,
        "ep_allocator_error": "gpu-allocator failed to allocate" in flat,
        "ort_retry_on_cpu": "Falling back to" in flat and "retrying" in flat,
        "ort_ep_fail": "EP_FAIL" in flat,
    }
    rec["dispatches_executed"] = (rec.get("counters") or {}).get("dispatches_executed")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()

    if not MODEL.is_file():
        print(f"ERROR(instrument): model not found at {MODEL}")
        return 3
    if "ONNXRUNTIME_VULKAN_EP_LIB" not in os.environ:
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB is unset")
        return 3

    arms = [
        _arm("A_ctx1_fallback_allowed", 1, False, {}),
        _arm(f"B_ctx{args.ctx}_fallback_allowed", args.ctx, False, {}),
        _arm(f"C_ctx{args.ctx}_fallback_forbidden", args.ctx, True, {}),
        # D — blindness control. The EP is made to retain no island at all, so ORT places all
        # 355 nodes on the CPU EP at *partition* time rather than after a compute failure. Same
        # end state (nothing ran on the EP), different route in. If the flag refuses here and not
        # in C, the difference is the fault; if it refuses in neither, this harness is not
        # driving the flag and C is not evidence.
        _arm("D_ctx1_fallback_forbidden_no_island_retained", 1, True,
             {"ONNXRUNTIME_EP_VULKAN_PARTITION_MIN_NODES": "100000",
              "ONNXRUNTIME_EP_VULKAN_PARTITION_ANCHOR_EXEMPTION": "0"}),
        # E — THE USER-FACING ARM. Everything above runs at VERBOSE, where an emission nobody
        # sees by default is still visible; RAI-013 is precisely the finding that those two are
        # not the same claim. This one runs at the ORT default severity a user gets without
        # asking for anything.
        _arm(f"E_ctx{args.ctx}_fallback_allowed_default_severity", args.ctx, False, {}, severity=2),
    ]

    (OUT / "arms.json").write_text(json.dumps(arms, indent=2), encoding="utf-8")

    print(f"probe_silent_cpu_rebuild — model {MODEL.name}")
    print(f"  lane pinned: DEVICE_MEMORY=1 KV_ARENA=0 (recorded in every arm)")
    for a in arms:
        print(f"\n[{a['arm']}]")
        print(f"  exit_code            {a['exit_code']}")
        print(f"  providers            {a.get('providers')}  severity={a.get('log_severity_level')}")
        print(f"  session_created      {a.get('session_created')}")
        print(f"  ran                  {a.get('ran')}  {a.get('run_error', '')[:90]}")
        print(f"  session_error        {a.get('session_error', '-')[:120]}")
        print(f"  dispatches_executed  {a.get('dispatches_executed')}   <- THE SCREEN")
        print(f"  counters             {a.get('counters')}")
        print(f"  observable           {a['observable']}")
    print(f"\nartifacts: {OUT}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == CHILD:
        raise SystemExit(_child())
    raise SystemExit(main())
