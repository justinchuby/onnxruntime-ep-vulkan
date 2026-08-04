#!/usr/bin/env python
"""Does `session.disable_cpu_ep_fallback=1` catch a **run-time** re-execution on the CPU EP?
Owner: Tank.

WHY THIS EXISTS AND WHY `probe_silent_cpu_rebuild.py` COULD NOT ANSWER IT
------------------------------------------------------------------------
On Phi-3.5 at ctx 4096 the fault is a Compute-time allocator failure: the EP claims all 355
nodes, ORT calls Compute, the allocator refuses a 64 MiB intermediate, and ORT prints
*"Falling back to ['CPUExecutionProvider'] and retrying"* and **re-executes the fused node on
the CPU** — exit 0, correct logits, `dispatches_executed: 0`.

`disable_cpu_ep_fallback=1` was the obvious screen for that, and on Phi-3.5 it **refuses the
session before Compute is ever reached** — not because of the allocator, but because 5 nodes
were declined `[unproven]` at partition time and are assigned to the CPU EP. Arm D of that probe
provokes the same refusal with the same sentence by a completely different route (no island
retained, no allocator involvement, ctx 1). **The two arms are indistinguishable**, so that
refusal separates nothing, and on any graph with even one CPU-assigned node the flag can never
reach the run-time question at all.

This probe removes the confound: a graph the EP claims **entirely**, so a
`disable_cpu_ep_fallback` session can be created, plus the planted Compute-failure control
(`ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE`) to produce the failure on demand on working
hardware. Every run under that variable is marked `fault_injection: ACTIVE` in its counters, so
none of these can be quoted as a clean run.

ARMS (predictions written before the run)
-----------------------------------------
F allowed,  no injected failure  -> runs on the EP. dispatches_executed > 0. THE SCREEN, and the
                                    proof the case is fully claimed.
G forbidden, no injected failure -> session CREATES and runs. Non-vacuity: the flag is not
                                    refusing this graph for the partition-time reason, so
                                    anything arm H reports is about the run-time path.
H forbidden, failure INJECTED    -> THE QUESTION. If ORT falls back to the CPU EP and returns a
                                    correct answer with exit 0, a session that was configured to
                                    forbid exactly that got it anyway, and the flag has a hole.
I allowed,  failure INJECTED     -> the comparison arm: same failure, fallback permitted.

NO CLOCK. Exit codes, counters, and log text only.

USAGE
    $env:ONNXRUNTIME_VULKAN_EP_LIB=...\\onnxruntime_vulkan_ep.dll
    python rust/tools/probe_runtime_fallback_guard.py
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "bench" / "results" / "_probe_runtime_fallback_guard"
CASE = REPO / "evidence" / "cases" / "add_f32.onnx"
CHILD = "--child"


def _child() -> int:
    import numpy as np
    import onnxruntime as ort

    forbid = os.environ["PROBE_FORBID"] == "1"
    ort.register_execution_provider_library(
        "VulkanExecutionProvider", os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
    )
    so = ort.SessionOptions()
    so.log_severity_level = 0
    providers = ["VulkanExecutionProvider"] if forbid else [
        "VulkanExecutionProvider",
        "CPUExecutionProvider",
    ]
    if forbid:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    result: dict = {"forbid": forbid, "providers": providers}
    try:
        sess = ort.InferenceSession(os.environ["PROBE_CASE"], so, providers=providers)
        result["session_created"] = True
    except Exception as exc:  # noqa: BLE001
        result["session_created"] = False
        result["session_error"] = f"{type(exc).__name__}: {exc}"
        pathlib.Path(os.environ["PROBE_RESULT"]).write_text(json.dumps(result), encoding="utf-8")
        return 0

    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 4 for d in inp.shape]
        feeds[inp.name] = np.ones(shape, dtype=np.float32)
    try:
        outs = sess.run(None, feeds)
        result["ran"] = True
        result["output_correct"] = bool(np.allclose(outs[0], 2.0))
    except Exception as exc:  # noqa: BLE001
        result["ran"] = False
        result["run_error"] = f"{type(exc).__name__}: {exc}"
    pathlib.Path(os.environ["PROBE_RESULT"]).write_text(json.dumps(result), encoding="utf-8")
    return 0


def _arm(name: str, forbid: bool, inject: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    counters = OUT / f"{name}_counters.json"
    result = OUT / f"{name}_result.json"
    for p in (counters, result):
        if p.exists():
            p.unlink()
    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
    env["ONNXRUNTIME_EP_VULKAN_KV_ARENA"] = "0"
    env["PROBE_FORBID"] = "1" if forbid else "0"
    env["PROBE_CASE"] = str(CASE)
    env["PROBE_RESULT"] = str(result)
    if inject:
        env["ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE"] = "1"
    else:
        env.pop("ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE", None)

    r = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), CHILD],
        env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=1800,
    )
    rec: dict = {"arm": name, "forbid_fallback": forbid, "failure_injected": inject,
                 "device_memory": "1", "kv_arena": "0", "exit_code": r.returncode}
    rec.update(json.loads(result.read_text(encoding="utf-8")) if result.is_file()
               else {"child_wrote_nothing": True})
    c = json.loads(counters.read_text(encoding="utf-8")) if counters.is_file() else None
    rec["counters"] = None if c is None else {
        k: c.get(k) for k in (
            "compile_calls", "compute_calls", "compute_failures", "dispatches_executed",
            "claimed_nodes", "fault_injection", "compute_failures_injected",
        )
    }
    rec["dispatches_executed"] = (rec["counters"] or {}).get("dispatches_executed")
    log = (r.stdout or "") + (r.stderr or "")
    (OUT / f"{name}_log.txt").write_text(log, encoding="utf-8")
    # ORT's C++ sink writes wide characters into this pipe; a UTF-8 read interleaves NULs, and a
    # search that does not strip them reports every token ORT emitted as absent.
    flat = log.replace("\x00", "")
    rec["observable"] = {
        "ep_broken_commitment_warn": "BROKEN COMMITMENT" in flat,
        "ort_retry_on_cpu": "Falling back to" in flat and "retrying" in flat,
        "ort_ep_fail": "EP_FAIL" in flat,
    }
    return rec


def main() -> int:
    if "ONNXRUNTIME_VULKAN_EP_LIB" not in os.environ:
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB is unset")
        return 3
    if not CASE.is_file():
        print(f"ERROR(instrument): case not found: {CASE}")
        return 3

    arms = [
        _arm("F_allowed_clean", False, False),
        _arm("G_forbidden_clean", True, False),
        _arm("H_forbidden_failure_injected", True, True),
        _arm("I_allowed_failure_injected", False, True),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arms.json").write_text(json.dumps(arms, indent=2), encoding="utf-8")

    by = {a["arm"]: a for a in arms}
    print(f"probe_runtime_fallback_guard — case {CASE.name}, lane DEVICE_MEMORY=1 KV_ARENA=0")
    for a in arms:
        print(f"\n[{a['arm']}]")
        print(f"  exit_code            {a['exit_code']}")
        print(f"  session_created      {a.get('session_created')}  {a.get('session_error', '')[:100]}")
        print(f"  ran / correct        {a.get('ran')} / {a.get('output_correct')}  {a.get('run_error', '')[:80]}")
        print(f"  dispatches_executed  {a.get('dispatches_executed')}   <- THE SCREEN")
        print(f"  counters             {a.get('counters')}")
        print(f"  observable           {a['observable']}")

    print("\n--- READING ---")
    f = by["F_allowed_clean"]
    if not (f.get("dispatches_executed") or 0) > 0:
        print("ERROR(instrument): arm F executed nothing on the EP; the case is not fully "
              "claimed on this build and no arm below is evidence about the guard.")
        return 3
    g = by["G_forbidden_clean"]
    if not g.get("session_created"):
        print("ERROR(instrument): arm G's session was refused with no failure injected, so the "
              "flag is refusing this graph at partition time and arm H cannot reach the "
              "run-time path — the same confound Phi-3.5 has. "
              f"{g.get('session_error', '')[:200]}")
        return 3
    h = by["H_forbidden_failure_injected"]
    hole = bool(h.get("ran")) and h.get("exit_code") == 0 and (h.get("dispatches_executed") or 0) == 0
    print(
        "FINDING: a session with session.disable_cpu_ep_fallback=1 "
        + ("WAS re-executed on the CPU EP and returned success anyway — THE FLAG HAS A HOLE at "
           "the run-time fallback path; it is a partition-time screen only."
           if hole else
           "refused / did not silently fall back. The flag covers the run-time path.")
    )
    print(f"  arm H: session_created={h.get('session_created')} ran={h.get('ran')} "
          f"output_correct={h.get('output_correct')} exit={h.get('exit_code')} "
          f"dispatches={h.get('dispatches_executed')} "
          f"ort_retry_on_cpu={h['observable']['ort_retry_on_cpu']}")
    i = by["I_allowed_failure_injected"]
    print(f"  arm I (fallback allowed, same failure): ran={i.get('ran')} "
          f"exit={i.get('exit_code')} dispatches={i.get('dispatches_executed')} "
          f"ort_retry_on_cpu={i['observable']['ort_retry_on_cpu']}")
    print(f"  arms H and I differ: {h.get('ran') != i.get('ran') or h.get('exit_code') != i.get('exit_code')}")
    print(f"\nartifacts: {OUT}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == CHILD:
        raise SystemExit(_child())
    raise SystemExit(main())
