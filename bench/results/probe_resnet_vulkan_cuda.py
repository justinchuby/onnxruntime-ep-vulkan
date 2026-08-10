"""ResNet-50 on the Vulkan EP vs ORT's CUDA EP, with the CPU EP as reference (issue #122).

WHAT THIS IS
============
The driver for `bench/resnet.py`. It answers issue #122's three questions with one artifact:

* **the numbers** — steady-state median latency with dispersion, per batch size, per arm;
* **the comparison** — a paired, per-repeat, polarity-labelled ratio of the Vulkan arm against
  the CUDA arm, with the CPU EP alongside as correctness reference and timing context;
* **the gaps** — ORT's own per-node provider attribution for both GPU arms, the EP's dispatch
  counters, the static support census, and the list of op types that fall back.

THREE PASSES, NEVER ONE
=======================
1. **verify** — CPU reference, then each GPU arm, in one process, on the sessions that are then
   timed. A case that does not pass is not timed. (`argmax 0`: 161 dispatches, all-zero logits.)
2. **timed** — one subprocess per (repeat, arm), arms alternated per repeat. Tracing off, no
   profiler, no counters file. A wall time measured through an instrument is not the wall time
   without it. Subprocesses and not a loop because ORT's EP registration is process-global and
   the Vulkan counters file is written from a process-exit hook — and because a CUDA context and
   a Vulkan context co-resident in one process would each be measured against the other's
   allocations.
3. **diagnose** — one subprocess per arm with `enable_profiling` and the counters file set.
   This is where partition, fallback, transfer nodes and dispatch witnesses come from. It is
   never the pass whose wall clock is quoted.

WHAT IT REFUSES TO PUBLISH
==========================
`resnet.admissibility` recomputes the verdict from the recorded evidence on every read. If the
provenance, the equivalence gate, the dispatch witness, the CUDA execution witness, the device
identification, the quiescence verdict or the repeat count fails, the record is
`INDETERMINATE` and the ratios are marked not-quotable rather than quoted with a caveat.

GPU EXCLUSIVITY
===============
This box is shared and other agents measure on it. `--require-lock` refuses to start unless the
caller holds the desk's GPU lock file, so a benchmark cannot be started on top of somebody
else's benchmark by accident.

Usage::

    python bench/results/probe_resnet_vulkan_cuda.py --device 0 \
        --ep-lib rust/target/release/onnxruntime_vulkan_ep.dll \
        --batch 1,4,16 --repeats 5 --iters 10 --out resnet_vulkan_cuda.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1]
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import real_model as rm  # noqa: E402
import resnet as rn  # noqa: E402

COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
DEFAULT_BATCH = (1, 4, 16)


# --------------------------------------------------------------------------------------------
# Session plumbing
# --------------------------------------------------------------------------------------------


def _preload_cuda(ort) -> dict:
    """Load the CUDA/cuDNN DLLs the ORT GPU wheel needs, and record whether it worked.

    ORT >= 1.21 resolves CUDA from the `nvidia-*-cu12` pip packages through this call, and it
    must happen **before** the first session. Recorded rather than assumed: a CUDA arm that
    silently fell back to CPU because a DLL was missing would produce plausible numbers for
    the wrong thing, which is exactly the class of defect this lane's gates exist against.
    """
    fn = getattr(ort, "preload_dlls", None)
    if fn is None:
        return {"called": False, "reason": "onnxruntime has no preload_dlls (pre-1.21)"}
    try:
        fn()
        return {"called": True, "ok": True}
    except Exception as exc:
        return {"called": True, "ok": False, "reason": str(exc)}


def _register_vulkan(ort, lib: "str | None") -> dict:
    """Register the EP plugin library, and record which bytes were registered."""
    if not lib:
        return {"registered": False, "reason": f"{EP_LIB_ENV} unset and --ep-lib not given"}
    p = Path(lib)
    if not p.is_file():
        return {"registered": False, "reason": f"{p} does not exist"}
    try:
        ort.register_execution_provider_library(rn.EP_NAME, str(p.resolve()))
    except Exception as exc:
        if "already registered" not in str(exc):
            return {"registered": False, "reason": str(exc), "path": str(p.resolve())}
    return {"registered": True, "path": str(p.resolve()), "sha256": rm.sha256_file(p),
            "bytes": p.stat().st_size}


def _session(ort, path: Path, arm: rm.Arm, device_index: int, *, profiling: bool = False):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    # Pinned, never defaulted: an unpinned optimisation level makes two runs on two ORT builds
    # incomparable without anyone noticing, and it is the level that decides whether the 53
    # BatchNormalization nodes still exist when the EP is asked what it can claim.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if rn.EP_NAME in arm.providers:
        opts.add_session_config_entry("ep.device_index", str(device_index))
    if profiling:
        opts.enable_profiling = True
        opts.profile_file_prefix = f"prof_resnet_{arm.name}"
    providers = []
    provider_options = []
    for name in arm.providers:
        providers.append(name)
        if name == rn.CUDA_EP:
            # The CUDA device is pinned for the same reason the Vulkan device is: a machine can
            # have more than one, and a result that does not name which card it ran on is not a
            # result.
            provider_options.append({"device_id": str(device_index)})
        else:
            provider_options.append({})
    return ort.InferenceSession(str(path), opts, providers=providers,
                                provider_options=provider_options)


def _timed_build(ort, path: Path, arm: rm.Arm, device_index: int, **kw):
    t0 = time.perf_counter()
    sess = _session(ort, path, arm, device_index, **kw)
    return sess, (time.perf_counter() - t0) * 1000.0


def _sample(sess, feeds, iters: int, warmup: int) -> dict:
    """Warm up, discard and count, then time. First run is recorded, never averaged in."""
    t0 = time.perf_counter()
    sess.run(None, feeds)
    first_ms = (time.perf_counter() - t0) * 1000.0
    for _ in range(max(0, warmup - 1)):
        sess.run(None, feeds)
    xs = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, feeds)
        xs.append((time.perf_counter() - t) * 1000.0)
    return {"first_run_ms": first_ms, "samples_ms": xs, "warmups_discarded": max(1, warmup)}


# --------------------------------------------------------------------------------------------
# GPU device identity
# --------------------------------------------------------------------------------------------


def _nvidia_smi_query() -> dict:
    """CUDA-side device identity. `None` fields are absences, never guesses."""
    fields = ("name,uuid,pci.bus_id,driver_version,memory.total,"
              "utilization.gpu,memory.used,clocks.sm,clocks.max.sm")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    if out.returncode != 0:
        return {"available": False, "reason": (out.stderr or "").strip()[:400]}
    gpus = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        gpus.append({
            "name": parts[0], "uuid": parts[1], "pci_bus_id": parts[2],
            "driver_version": parts[3], "memory_total_mib": _int(parts[4]),
            "utilization_gpu_pct": _int(parts[5]), "memory_used_mib": _int(parts[6]),
            "sm_clock_mhz": _int(parts[7]), "sm_max_clock_mhz": _int(parts[8]),
        })
    return {"available": True, "gpus": gpus}


def _int(s):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def foreign_load(before: dict, after: dict) -> dict:
    """What else was on the GPU while this ran.

    §18 of `docs/PERF.md`: foreign GPU work moves this box's tail more than any kernel change
    measured so far. A run that does not disclose it is not disclosing its largest term. This
    is a *disclosure*, not a gate — the gate is `contention.quiescence`.
    """
    def _mem(rec):
        gs = (rec or {}).get("gpus") or []
        return gs[0].get("memory_used_mib") if gs else None

    def _util(rec):
        gs = (rec or {}).get("gpus") or []
        return gs[0].get("utilization_gpu_pct") if gs else None

    return {
        "provenance_class": "MEASUREMENT",
        "memory_used_mib_before": _mem(before),
        "memory_used_mib_after": _mem(after),
        "utilization_pct_before": _util(before),
        "utilization_pct_after": _util(after),
        "note": ("nvidia-smi samples taken immediately before and after the timed pass. "
                 "Non-zero memory before the run is another process's residency; this lane "
                 "does not have the desk to itself and says so."),
    }


# --------------------------------------------------------------------------------------------
# Verification — before anything is timed
# --------------------------------------------------------------------------------------------


def verify_case(ort, np, path: Path, case: rm.Case, feeds: dict, device_index: int,
                arms) -> dict:
    ref_sess = _session(ort, path, rn.CPU_ARM, device_index)
    reference = ref_sess.run(None, feeds)
    per_arm = {rn.CPU_ARM.name: {
        "verdict": rm.MATCH, "self": True,
        "note": "the reference arm is compared against itself by construction; recorded so "
                "the table has no hole"}}
    ok = True
    for arm in arms:
        if arm.name == rn.CPU_ARM.name:
            continue
        try:
            sess = _session(ort, path, arm, device_index)
            out = sess.run(None, feeds)
            v = rn.classify_case(case, out, reference, np)
            del sess
        except Exception as exc:
            v = {"verdict": rm.DIVERGENT, "reason": f"arm failed to run: {exc}"}
        per_arm[arm.name] = v
        ok = ok and v["verdict"] == rm.MATCH
    del ref_sess
    return {"gate": "PASS" if ok else "FAIL", "arms": per_arm}


# --------------------------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------------------------


def timed_worker(argv) -> int:
    """One arm, one batch, one repeat, in its own process. No profiler, no counters."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-timed", action="store_true")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ep-lib", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort

    arm = {x.name: x for x in rn.ARMS}[a.arm]
    preload = _preload_cuda(ort) if rn.CUDA_EP in arm.providers else {"called": False}
    reg = _register_vulkan(ort, a.ep_lib) if rn.EP_NAME in arm.providers else {
        "registered": False, "reason": "arm does not use the Vulkan EP"}
    model_rec = rm.resolve_model(rn.RESNET50)
    case = rn.resnet_cases([a.batch])[0]
    feeds = rn.resnet_feeds(case, np)
    prev = arm.apply_env(os.environ)
    try:
        sess, build_ms = _timed_build(ort, Path(model_rec["path"]), arm, a.device)
        got = _sample(sess, feeds, a.iters, a.warmup)
        providers = list(sess.get_providers())
        del sess
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    Path(a.out).write_text(json.dumps({
        "arm": a.arm, "batch": a.batch, "case": case.label,
        "feeds_sha256": rm.feeds_digest(feeds),
        "session_build_ms": build_ms,
        "ep_registration": reg,
        "cuda_preload": preload,
        "session_providers": providers,
        **got,
    }), encoding="utf-8")
    return 0


def _profile_provider_counts(sess, ort) -> "dict | None":
    """ORT's own per-node attribution. The MEASUREMENT of where the graph actually ran."""
    try:
        prof = sess.end_profiling()
    except Exception:
        return None
    try:
        events = json.loads(Path(prof).read_text(encoding="utf-8"))
    except Exception:
        return None
    finally:
        try:
            Path(prof).unlink()
        except OSError:
            pass
    counts: dict = {}
    node_us: dict = {}
    op_us: dict = {}
    op_counts: dict = {}
    fallback_ops: dict = {}
    transfer_us = 0
    transfer_nodes = 0
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        argsd = ev.get("args") or {}
        prov = argsd.get("provider")
        if not prov:
            continue
        dur = int(ev.get("dur") or 0)
        op = argsd.get("op_name") or "UNKNOWN"
        counts[prov] = counts.get(prov, 0) + 1
        node_us[prov] = node_us.get(prov, 0) + dur
        key = f"{prov}::{op}"
        op_us[key] = op_us.get(key, 0) + dur
        op_counts[key] = op_counts.get(key, 0) + 1
        if prov == rn.CPU_EP:
            fallback_ops[op] = fallback_ops.get(op, 0) + 1
        if op.startswith("Memcpy"):
            transfer_us += dur
            transfer_nodes += 1
    total_us = sum(node_us.values())
    ops = [{"provider_op": k, "microseconds": v, "nodes": op_counts[k],
            "share_of_node_time": (v / total_us) if total_us else None}
           for k, v in sorted(op_us.items(), key=lambda kv: -kv[1])]
    return {
        "provenance_class": "MEASUREMENT",
        "counts": counts,
        "microseconds": node_us,
        "by_op": ops,
        "node_microseconds_total": total_us,
        "cpu_fallback_op_types": dict(sorted(fallback_ops.items(), key=lambda kv: -kv[1])),
        "transfer_nodes": {
            "nodes": transfer_nodes,
            "microseconds": transfer_us,
            "share_of_node_time": (transfer_us / total_us) if total_us else None,
            "note": "ORT inserts MemcpyToHost/MemcpyFromHost at EP boundaries; their cost is "
                    "the price of partition fragmentation and is attributed separately here",
        },
    }


def diagnose_worker(argv) -> int:
    """One arm, one batch, profiled, with the counters file set. Never the timed pass."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-diagnose", action="store_true")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--ep-lib", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort

    arm = {x.name: x for x in rn.ARMS}[a.arm]
    preload = _preload_cuda(ort) if rn.CUDA_EP in arm.providers else {"called": False}
    reg = _register_vulkan(ort, a.ep_lib) if rn.EP_NAME in arm.providers else {
        "registered": False, "reason": "arm does not use the Vulkan EP"}
    model_rec = rm.resolve_model(rn.RESNET50)
    case = rn.resnet_cases([a.batch])[0]
    feeds = rn.resnet_feeds(case, np)
    rec = {"arm": a.arm, "batch": a.batch, "case": case.label, "ep_registration": reg,
           "cuda_preload": preload}
    prev = arm.apply_env(os.environ)
    try:
        sess = _session(ort, Path(model_rec["path"]), arm, a.device, profiling=True)
        for _ in range(max(1, a.iters)):
            sess.run(None, feeds)
        rec["inferences"] = max(1, a.iters)
        rec["profile"] = _profile_provider_counts(sess, ort)
        rec["session_providers"] = list(sess.get_providers())
        del sess
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    Path(a.out).write_text(json.dumps(rec), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def run_timed(args, cases, scratch: Path, ep_lib) -> dict:
    """Interleaved: repeat by repeat, arms alternated, one subprocess each."""
    raw = {c.label: {a.name: [] for a in rn.ARMS} for c in cases}
    builds = {c.label: {a.name: [] for a in rn.ARMS} for c in cases}
    firsts = {c.label: {a.name: [] for a in rn.ARMS} for c in cases}
    digests: dict = {}
    errors = []
    for rep in range(args.repeats):
        for arm in rm.arm_order(rn.ARMS, rep):
            for case in cases:
                tag = f"{arm.name}_b{case.m}_r{rep}"
                out_path = scratch / f"timed_{tag}.json"
                out_path.unlink(missing_ok=True)
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-timed",
                       "--arm", arm.name, "--batch", str(case.m),
                       "--device", str(args.device), "--iters", str(args.iters),
                       "--warmup", str(args.warmup), "--out", str(out_path)]
                if ep_lib:
                    cmd += ["--ep-lib", str(ep_lib)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if not out_path.exists():
                    errors.append({"tag": tag, "returncode": proc.returncode,
                                   "stderr": (proc.stderr or "")[-1500:]})
                    continue
                got = json.loads(out_path.read_text(encoding="utf-8"))
                raw[case.label][arm.name].extend(got["samples_ms"])
                builds[case.label][arm.name].append(got["session_build_ms"])
                firsts[case.label][arm.name].append(got["first_run_ms"])
                digests.setdefault(case.label, got["feeds_sha256"])
                if digests[case.label] != got["feeds_sha256"]:
                    errors.append({"tag": tag, "error": "two arms were fed different bytes",
                                   "expected": digests[case.label],
                                   "observed": got["feeds_sha256"]})
        print(f"[resnet] repeat {rep + 1}/{args.repeats} done", flush=True)
    return {"raw": raw, "builds": builds, "firsts": firsts, "digests": digests,
            "errors": errors}


def run_diagnostics(args, cases, scratch: Path, ep_lib) -> dict:
    runs = []
    for case in cases:
        for arm in rn.ARMS:
            tag = f"{arm.name}_b{case.m}"
            rec_path = scratch / f"diag_{tag}.json"
            counters = scratch / f"counters_{tag}.json"
            for p in (rec_path, counters):
                p.unlink(missing_ok=True)
            env = dict(os.environ)
            env[COUNTERS_ENV] = str(counters)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-diagnose",
                   "--arm", arm.name, "--batch", str(case.m), "--device", str(args.device),
                   "--iters", str(args.diag_iters), "--out", str(rec_path)]
            if ep_lib:
                cmd += ["--ep-lib", str(ep_lib)]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if rec_path.exists():
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
            else:
                rec = {"arm": arm.name, "batch": case.m,
                       "error": f"worker exit {proc.returncode}",
                       "stderr": (proc.stderr or "")[-1500:]}
            if counters.exists():
                try:
                    rec["counters"] = json.loads(counters.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    rec["counters"] = None
            rec["dispatch"] = rm.dispatch_diagnosis(rec.get("counters"),
                                                    rec.get("inferences") or 0)
            runs.append(rec)
            prov = ((rec.get("profile") or {}).get("counts")) or {}
            print(f"[diag] {arm.name} b{case.m}: providers={prov} "
                  f"dispatches={(rec.get('counters') or {}).get('dispatches_executed')}",
                  flush=True)
    return {"runs": runs}


def environment_record(args, ep_reg: dict, device, nvidia: dict, cuda_preload: dict) -> dict:
    import onnxruntime as ort

    import environment as env_mod

    ort_dll = Path(ort.__file__).resolve().parent / "capi" / "onnxruntime.dll"
    rec = {
        "onnxruntime": {
            "version": ort.__version__,
            "available_providers": list(ort.get_available_providers()),
            "library": {
                "path": str(ort_dll),
                "sha256": rm.sha256_file(ort_dll) if ort_dll.is_file() else None,
                "bytes": ort_dll.stat().st_size if ort_dll.is_file() else None,
            },
            "note": "ONE onnxruntime build serves all three arms; a CUDA-vs-Vulkan ratio taken "
                    "across two ORT versions would be a version comparison wearing an EP's name",
        },
        "ep_library": ep_reg,
        "cuda_preload": cuda_preload,
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(),
        "device": {
            "index": getattr(device, "index", None),
            "name": getattr(device, "name", None),
            "uuid": getattr(device, "uuid", None),
            "luid": getattr(device, "luid", None),
            "pci": getattr(device, "pci", None),
            "driver": getattr(device, "driver_version", None),
            "transfer_class": getattr(device, "transfer_class", None),
            "timestamp_period_ns": getattr(device, "timestamp_period", None),
            "max_compute_shared_memory": getattr(device, "max_compute_shared_memory", None),
            "subgroup_size": getattr(device, "subgroup_size", None),
        },
        "cuda_device": nvidia,
        "same_physical_device": {
            "claim": "the Vulkan arm and the CUDA arm must run on the SAME card or the ratio "
                     "is a comparison of two pieces of hardware",
            "vulkan_pci": getattr(device, "pci", None),
            "cuda_pci": ((nvidia.get("gpus") or [{}])[0]).get("pci_bus_id"),
        },
        "power_and_affinity": {
            "assumption": "stock Windows power plan, no CPU affinity mask, no GPU clock lock",
            "thread_affinity_set": False,
            "gpu_clocks_locked": False,
            "note": "this box is shared indefinitely (docs/PERF.md §20), so wall clock is "
                    "STEADY_UNCERTIFIED by default; arms are interleaved precisely because the "
                    "machine state cannot be assumed quiet",
        },
        "methodology": {
            "repeats": args.repeats,
            "iters_per_repeat": args.iters,
            "warmup_per_session": args.warmup,
            "warmups_discarded": True,
            "arm_order": "alternated per repeat (real_model.arm_order)",
            "process_per_arm_per_repeat": True,
            "tracing_during_timed_pass": False,
            "diagnostics_are_a_separate_pass": True,
        },
    }
    try:
        rec["build"] = env_mod.build_info()
    except Exception as exc:  # pragma: no cover - environment dependent
        rec["build"] = {"error": str(exc)}
    return rec


def _capabilities(epctl: "Path | None") -> dict:
    if not epctl or not Path(epctl).is_file():
        return {"error": f"epctl not found at {epctl}"}
    try:
        out = subprocess.run([str(epctl), "--dump-capabilities", "--json"],
                             capture_output=True, text=True, timeout=120)
        return json.loads(out.stdout)
    except Exception as exc:
        return {"error": str(exc)}


def _lock_held(lock_dir: Path, holder: str) -> dict:
    """Does this run hold the desk's GPU lock, and who else is in it?"""
    mine = lock_dir / f"{holder}.lock"
    others = []
    if lock_dir.is_dir():
        others = sorted(p.name for p in lock_dir.glob("*.lock") if p.name != mine.name)
    return {"lock_dir": str(lock_dir), "holder": holder, "held": mine.is_file(),
            "other_holders": others,
            "exclusive": mine.is_file() and not others}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker-timed" in argv:
        return timed_worker(argv)
    if "--worker-diagnose" in argv:
        return diagnose_worker(argv)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch", default=",".join(str(b) for b in DEFAULT_BATCH))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--diag-iters", type=int, default=3)
    ap.add_argument("--ep-lib", default=os.environ.get(EP_LIB_ENV))
    ap.add_argument("--epctl", default=None)
    ap.add_argument("--lock-dir", default=str(Path.home() / ".copilot" / "repos" / ".gpu-lock"))
    ap.add_argument("--lock-holder", default="niobe-11")
    ap.add_argument("--require-lock", action="store_true",
                    help="refuse to run unless this holder's lock file is the only one present")
    a = ap.parse_args(argv)

    # The lock is checked BEFORE anything is imported, let alone before a device is opened, so
    # that a refusal cannot itself be the thing that perturbs another lane's benchmark.
    lock = _lock_held(Path(a.lock_dir), a.lock_holder)
    if a.require_lock and not lock["exclusive"]:
        print(f"[resnet] REFUSED(instrument=gpu_lock_not_exclusive): {lock}", file=sys.stderr)
        return 3

    import numpy as np
    import onnxruntime as ort

    import contention
    import devices as device_mod

    scratch = Path(__file__).resolve().parent / "resnet-scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    facts, how = device_mod.probe()
    device = next((d for d in facts if d.index == a.device), None)
    if device is None:
        print(f"[resnet] no Vulkan device at index {a.device} (probe: {how})", file=sys.stderr)
        return 2

    ep_reg = _register_vulkan(ort, a.ep_lib)
    cuda_preload = _preload_cuda(ort)
    model_rec = rm.resolve_model(rn.RESNET50)
    print(f"[resnet] {model_rec['path']}\n[resnet]   sha256 {model_rec['sha256']} "
          f"({model_rec['provenance']})", flush=True)

    batch = [int(x) for x in a.batch.split(",") if x]
    cases = rn.resnet_cases(batch)

    # --- verify, before anything is timed -------------------------------------------------
    equivalence = {}
    timed_cases = []
    for case in cases:
        feeds = rn.resnet_feeds(case, np)
        v = verify_case(ort, np, Path(model_rec["path"]), case, feeds, a.device, rn.ARMS)
        v["feeds_sha256"] = rm.feeds_digest(feeds)
        equivalence[case.label] = v
        print(f"[resnet] verify {case.label}: {v['gate']}", flush=True)
        if v["gate"] == "PASS":
            timed_cases.append(case)

    # --- machine state --------------------------------------------------------------------
    nvidia_before = _nvidia_smi_query()
    window = contention.sample_now(seconds=4.0, interval=0.5)
    quiet = contention.quiescence(window, contention.occupancy_check())

    # --- timed ----------------------------------------------------------------------------
    timed = run_timed(a, timed_cases, scratch, a.ep_lib) if timed_cases else {
        "raw": {}, "builds": {}, "firsts": {}, "digests": {}, "errors": []}
    nvidia_after = _nvidia_smi_query()

    # --- diagnose (separate pass) ---------------------------------------------------------
    diag = run_diagnostics(a, cases, scratch, a.ep_lib)

    def _per_repeat(label, arm_name):
        xs = timed["raw"].get(label, {}).get(arm_name, [])
        k = a.iters
        return [sorted(xs[i * k:(i + 1) * k])[k // 2] for i in range(len(xs) // k)]

    rows = []
    for case in timed_cases:
        row = {
            "case": case.label,
            "batch": case.m,
            "feeds_sha256": timed["digests"].get(case.label),
            "arms": {},
        }
        for arm in rn.ARMS:
            xs = timed["raw"][case.label][arm.name]
            st = rm.latency_stats(xs)
            row["arms"][arm.name] = {
                "role": arm.role,
                "providers": list(arm.providers),
                "latency": {**st, "provenance_class": "MEASUREMENT"},
                "throughput": rm.throughput(case, st.get("median_ms")),
                "session_build_ms": rm.latency_stats(timed["builds"][case.label][arm.name]),
                "first_run_ms": rm.latency_stats(timed["firsts"][case.label][arm.name]),
                "per_repeat_median_ms": _per_repeat(case.label, arm.name),
                "samples_ms": [round(x, 4) for x in xs],
            }
        v = _per_repeat(case.label, rn.VULKAN_ARM.name)
        c = _per_repeat(case.label, rn.CUDA_ARM.name)
        p = _per_repeat(case.label, rn.CPU_ARM.name)
        row["vulkan_vs_cuda"] = rn.ratio_record(c, v, baseline="cuda", candidate="vulkan")
        row["vulkan_vs_cpu"] = rn.ratio_record(p, v, baseline="cpu", candidate="vulkan")
        row["cuda_vs_cpu"] = rn.ratio_record(p, c, baseline="cpu", candidate="cuda")
        rows.append(row)

    # --- witnesses ------------------------------------------------------------------------
    vulkan_dispatches = 0
    cuda_nodes = 0
    for r in diag["runs"]:
        if r.get("arm") == rn.VULKAN_ARM.name:
            vulkan_dispatches = max(
                vulkan_dispatches, int((r.get("counters") or {}).get("dispatches_executed") or 0))
        if r.get("arm") == rn.CUDA_ARM.name:
            cuda_nodes = max(
                cuda_nodes, int(((r.get("profile") or {}).get("counts") or {}).get(
                    rn.CUDA_EP, 0)))

    gate = "PASS" if timed_cases and all(
        equivalence[c.label]["gate"] == "PASS" for c in timed_cases) else "FAIL"
    epctl = Path(a.epctl) if a.epctl else (_ROOT / "rust" / "target" / "release" / "epctl.exe")
    caps = _capabilities(epctl)

    admis = rn.admissibility(
        provenance_ok=bool(model_rec.get("provenance_ok")),
        equivalence_gate=gate,
        vulkan_dispatched=vulkan_dispatches,
        cuda_ran=bool(cuda_nodes),
        quiescence=quiet,
        device_identified=bool(getattr(device, "uuid", None) or getattr(device, "pci", None)),
        repeats=a.repeats, iters=a.iters,
    )

    report = {
        "schema": rn.SCHEMA,
        "issue": 122,
        "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model_rec,
        "model_pin": rn.RESNET50_PIN,
        "model_url": rn.RESNET50_URL,
        "environment": environment_record(a, ep_reg, device, nvidia_before, cuda_preload),
        "gpu_lock": lock,
        "quiescence": quiet,
        "foreign_gpu_load": foreign_load(nvidia_before, nvidia_after),
        "equivalence": equivalence,
        "rows": rows,
        "diagnostics": diag,
        "static_support_census": rn.support_census(rn.RESNET50_OP_HISTOGRAM, caps),
        "admissibility": admis,
        "quotable": rn.quotable(admis),
        "generalisation_limit": rn.GENERALISATION_LIMIT,
        "worker_errors": timed["errors"],
        "PROVENANCE": {
            "latency_medians": "MEASUREMENT",
            "ratios": "MEASUREMENT",
            "provider_node_counts": "MEASUREMENT",
            "dispatches_executed": "MEASUREMENT",
            "static_support_census": "MODEL",
            "device_and_driver_identity": "SPECIFICATION",
            "model_pin": "SPECIFICATION",
        },
    }
    out_path = Path(a.out) if a.out else Path(__file__).with_name("resnet_vulkan_cuda.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(report)
    print(f"\n[resnet] wrote {out_path}")
    print(f"[resnet] admissibility: {admis['verdict']} "
          f"(failed: {admis['failed_checks'] or 'none'})")
    return 0 if admis["verdict"] == rn.ADMISSIBLE else 1


def _print_table(report: dict) -> None:
    dev = (report["environment"]["device"] or {}).get("name")
    print(f"\n  ResNet-50 v1-12 on {dev}")
    print(f"  {'batch':>5}  {'vulkan ms':>10}  {'cuda ms':>9}  {'cpu ms':>9}  "
          f"{'vk/cuda':>8}  {'vk/cpu':>7}")
    for r in report["rows"]:
        a = r["arms"]
        def med(name):
            return (a.get(name, {}).get("latency") or {}).get("median_ms")
        vk, cu, cp = med("vulkan"), med("cuda"), med("cpu")
        print(f"  {r['batch']:>5}  {vk if vk is None else round(vk, 3):>10}  "
              f"{cu if cu is None else round(cu, 3):>9}  "
              f"{cp if cp is None else round(cp, 3):>9}  "
              f"{round(r['vulkan_vs_cuda'].get('median') or 0, 3):>8}  "
              f"{round(r['vulkan_vs_cpu'].get('median') or 0, 3):>7}")
    print("  ratio polarity: > 1 means the candidate takes LONGER than the baseline")


if __name__ == "__main__":
    raise SystemExit(main())
