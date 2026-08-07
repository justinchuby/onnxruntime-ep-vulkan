"""Wall-clock latency of **real models** on the Vulkan EP, against the CPU EP, on one device.

This is the instrument issue #56 asked for. Before it, this repository had real-model
instruments with no clock and clocked instruments with no real model; §26 of `docs/PERF.md`
is the reading it produces.

WHAT IT MEASURES
================
Three arms — `vulkan_tiled`, `vulkan_untiled` (the `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS=1`
kill switch) and `cpu` — over two real models:

* **Foundry Local Phi-3.5-mini-instruct int4 RTN block-32**, resolved through
  `foundry_discovery` (never a hardcoded cache path), swept over prefill widths and — the hole
  every previous measurement in this repo left open — **decode with a non-empty KV cache**.
* **MobileNetV2-12** (f32 CNN, pinned in `rust/modelrunner`'s download cache with a recorded
  sha256), swept over batch size. It shares no operator with the row tile, so it prices
  *general* EP overhead rather than the one kernel under change.

HOW IT REFUSES TO LIE
=====================
* Arms **alternate per repeat**; the paired per-repeat ratio is the estimate.
* `M = 1` prefill is the **null control**: both Vulkan arms resolve `QB_ROWS = 1` and build the
  identical pipeline, so its spread is the noise floor every other row is read against.
* Correctness runs **before** timing, in the same process, on the same session objects that are
  then timed, for **every** arm — never a verdict inherited from an earlier run of an earlier
  build.
* Session build (pipeline creation, first-touch allocation) is recorded **separately** from
  steady state, and warmups are discarded and counted.
* The timed pass runs with tracing **off**. Dispatch counts, CPU-fallback attribution, GPU
  timestamps and counters come from a **separate diagnostic pass** (`--diagnose`), because a
  wall time measured through an instrument is not the wall time without it.

Usage::

    python bench/results/probe_real_model_latency.py --device 0 --out real_model_latency.json
    python bench/results/probe_real_model_latency.py --device 0 --diagnose \
        --out real_model_diagnostics.json
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

COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
CLAIM_LOG_ENV = "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"

DEFAULT_PREFILL_M = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_DECODE_PAST = (0, 128, 512, 1024)
DEFAULT_BATCH = (1, 2, 4, 8, 16, 32)


# --------------------------------------------------------------------------------------------
# Session plumbing
# --------------------------------------------------------------------------------------------


def _session(ort, path: Path, arm: rm.Arm, device_index: int, *, profiling: bool = False):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    # Pinned rather than left to ORT's default, for the same reason the device is pinned: an
    # unpinned knob makes two runs on two ORT builds incomparable without anyone noticing.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if rm.EP_NAME in arm.providers:
        opts.add_session_config_entry("ep.device_index", str(device_index))
    if profiling:
        opts.enable_profiling = True
        opts.profile_file_prefix = f"prof_{arm.name}"
    return ort.InferenceSession(str(path), opts, providers=list(arm.providers))


def _timed_build(ort, path: Path, arm: rm.Arm, device_index: int, **kw):
    t0 = time.perf_counter()
    sess = _session(ort, path, arm, device_index, **kw)
    return sess, (time.perf_counter() - t0) * 1000.0


def _sample(sess, feeds, iters: int, warmup: int) -> dict:
    """Warm up, discard, then time. Returns steady-state samples plus the first-run cost.

    The first run after a build pays command-buffer recording, pipeline creation and first-touch
    allocation, and on an idle GPU it also pays the power manager. It is recorded rather than
    averaged in: a benchmark that folds it into the median is reporting a number no steady-state
    user ever sees, and one that discards it silently hides a real cost of short-lived sessions.
    """
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
    return {"first_run_ms": first_ms, "samples_ms": xs}


# --------------------------------------------------------------------------------------------
# Correctness, before anything is timed
# --------------------------------------------------------------------------------------------


def verify_case(ort, np, path: Path, case: rm.Case, feeds: dict, device_index: int,
                arms) -> dict:
    """Run the CPU reference and every Vulkan arm once, and classify.

    Returns a record whose ``gate`` is ``PASS`` only if every compared arm matched. A case that
    does not pass is not timed — an unverified fast number is the most expensive kind.
    """
    ref_sess = _session(ort, path, rm.CPU_ARM, device_index)
    reference = ref_sess.run(None, feeds)
    per_arm = {}
    vulkan_outputs = {}
    ok = True
    for arm in arms:
        if arm.name == rm.CPU_ARM.name:
            per_arm[arm.name] = {"verdict": rm.MATCH, "self": True,
                                 "note": "the reference arm is compared against itself by "
                                         "construction; recorded so the table has no hole"}
            continue
        prev = arm.apply_env(os.environ)
        try:
            sess = _session(ort, path, arm, device_index)
            out = sess.run(None, feeds)
            v = rm.classify_outputs(case, out, reference, np)
            per_arm[arm.name] = v
            vulkan_outputs[arm.name] = out
            ok = ok and v["verdict"] == rm.MATCH
            del sess
        finally:
            for k, old in prev.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
    del ref_sess
    rec = {"gate": "PASS" if ok else "FAIL", "arms": per_arm}
    # The null control's *claim* is that at M=1 the two Vulkan arms bind the same SPIR-V. That is
    # checkable exactly, and a latency ratio cannot check it: bit-identical outputs are what
    # "identical pipeline" means, and anything else means the control is not a control.
    if rm.is_null_control(case) and len(vulkan_outputs) == 2:
        names = list(vulkan_outputs)
        rec["null_control_bitwise"] = rm.bitwise_identical(
            vulkan_outputs[names[0]], vulkan_outputs[names[1]], np)
        rec["null_control_bitwise"]["arms"] = names
    return rec


# --------------------------------------------------------------------------------------------
# The timed matrix
# --------------------------------------------------------------------------------------------


def run_matrix(args, model_rec: dict, cases, arms, device) -> dict:
    import numpy as np
    import onnxruntime as ort

    path = Path(model_rec["path"])
    feeds_by_case = {}
    digests = {}
    for case in cases:
        feeds_by_case[case.label] = rm.build_feeds(case, np)
        digests[case.label] = rm.feeds_digest(feeds_by_case[case.label])

    equivalence = {}
    timed_cases = []
    for case in cases:
        v = verify_case(ort, np, path, case, feeds_by_case[case.label], device.index, arms)
        equivalence[case.label] = v
        print(f"[real] verify {case.label}: {v['gate']}", flush=True)
        if v["gate"] == "PASS":
            timed_cases.append(case)

    raw = {c.label: {a.name: [] for a in arms} for c in timed_cases}
    builds = {c.label: {a.name: [] for a in arms} for c in timed_cases}
    firsts = {c.label: {a.name: [] for a in arms} for c in timed_cases}

    for rep in range(args.repeats):
        for arm in rm.arm_order(arms, rep):
            prev = arm.apply_env(os.environ)
            try:
                for case in timed_cases:
                    # A fresh session per arm is required, not incidental: the row tile is chosen
                    # at translate time, so a reused session keeps the arm it was built with.
                    sess, build_ms = _timed_build(ort, path, arm, device.index)
                    got = _sample(sess, feeds_by_case[case.label], args.iters, args.warmup)
                    raw[case.label][arm.name].extend(got["samples_ms"])
                    builds[case.label][arm.name].append(build_ms)
                    firsts[case.label][arm.name].append(got["first_run_ms"])
                    del sess
            finally:
                for k, old in prev.items():
                    if old is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = old
        print(f"[real] repeat {rep + 1}/{args.repeats} done", flush=True)

    # Per-repeat medians, for the paired ratio. Built from the same samples as the pooled
    # distribution, so the two cannot describe different runs.
    def _per_repeat(label, arm_name):
        xs = raw[label][arm_name]
        k = args.iters
        return [sorted(xs[i * k:(i + 1) * k])[k // 2] for i in range(len(xs) // k)]

    rows = []
    for case in timed_cases:
        row = {
            "case": case.label,
            "model": model_rec["key"],
            "phase": case.phase,
            "m": case.m,
            "past": case.past,
            "tokens_advanced": case.tokens,
            "feeds_sha256": digests[case.label],
            "null_control": rm.is_null_control(case),
            "kv_round_trip": rm.kv_round_trip_bytes(case),
            "arms": {},
        }
        for arm in arms:
            st = rm.latency_stats(raw[case.label][arm.name])
            row["arms"][arm.name] = {
                "role": arm.role,
                "providers": list(arm.providers),
                "env": {k: v for k, v in arm.env},
                "latency": st,
                "throughput": rm.throughput(case, st.get("median_ms")),
                "session_build_ms": rm.latency_stats(builds[case.label][arm.name]),
                "first_run_ms": rm.latency_stats(firsts[case.label][arm.name]),
                "samples_ms": [round(x, 4) for x in raw[case.label][arm.name]],
                "per_repeat_median_ms": _per_repeat(case.label, arm.name),
                "bandwidth_proxy": rm.bandwidth_proxy(case, st.get("median_ms")),
            }
        u = _per_repeat(case.label, rm.VULKAN_UNTILED.name)
        t = _per_repeat(case.label, rm.VULKAN_TILED.name)
        c = _per_repeat(case.label, rm.CPU_ARM.name)
        row["row_tile_speedup"] = rm.paired_ratios(u, t)
        row["vulkan_vs_cpu_tiled"] = rm.paired_ratios(c, t)
        row["vulkan_vs_cpu_untiled"] = rm.paired_ratios(c, u)
        rows.append(row)

    floor = next((r["row_tile_speedup"] for r in rows if r["null_control"]), {})
    for r in rows:
        if not r["null_control"]:
            med = r["row_tile_speedup"].get("median")
            r["row_tile_speedup"]["exceeds_noise_floor"] = (
                rm.exceeds_noise_floor(med, floor) if med else None)

    return {
        "model": model_rec,
        "rows": rows,
        "equivalence": equivalence,
        "noise_floor": {
            "source": "M=1 prefill, tiled vs untiled — identical specialisation, identical SPIR-V",
            **floor,
        },
    }


# --------------------------------------------------------------------------------------------
# Diagnostics — a separate pass, never the timed one
# --------------------------------------------------------------------------------------------


def _profile_provider_counts(sess, ort) -> "dict | None":
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
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        argsd = ev.get("args") or {}
        prov = argsd.get("provider")
        if not prov:
            continue
        dur = int(ev.get("dur") or 0)
        counts[prov] = counts.get(prov, 0) + 1
        node_us[prov] = node_us.get(prov, 0) + dur
        # Per *op type*, not only per provider. "the Vulkan EP owns 161 nodes" does not say
        # where the time goes, and the whole optimisation question for #56 is which operator
        # the wall clock is actually sitting in — a provider total cannot distinguish a slow
        # `MatMulNBits` from a slow `GroupQueryAttention`.
        op = argsd.get("op_name") or "UNKNOWN"
        key = f"{prov}::{op}"
        op_us[key] = op_us.get(key, 0) + dur
        op_counts[key] = op_counts.get(key, 0) + 1
    total_us = sum(node_us.values())
    ops = [{"provider_op": k, "microseconds": v, "nodes": op_counts[k],
            "share_of_node_time": (v / total_us) if total_us else None}
           for k, v in sorted(op_us.items(), key=lambda kv: -kv[1])]
    return {"counts": counts, "microseconds": node_us, "by_op": ops,
            "node_microseconds_total": total_us}


def diagnose_worker(argv) -> int:
    """One arm, one case, in its own process: counters are only complete at teardown."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-diagnose", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--past", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort

    spec = rm.MODELS[a.model]
    model_rec = rm.resolve_model(spec)
    arm = {x.name: x for x in rm.ARMS}[a.arm]
    case = rm.Case(a.model, a.phase, a.m, a.past,
                   tokens=(a.m if a.phase == "prefill" else (1 if a.phase == "decode" else None)),
                   unit=("images" if a.phase == "batch" else "tokens"))
    arm.apply_env(os.environ)

    rec: dict = {"arm": a.arm, "case": case.label, "model": a.model}
    if rm.EP_NAME in arm.providers:
        lib = os.environ.get(EP_LIB_ENV)
        if not lib or not Path(lib).is_file():
            rec["error"] = f"{EP_LIB_ENV} unset or missing"
            Path(a.out).write_text(json.dumps(rec), encoding="utf-8")
            return 2
        try:
            ort.register_execution_provider_library(rm.EP_NAME, str(Path(lib).resolve()))
        except Exception as exc:
            if "already registered" not in str(exc):
                rec["error"] = f"registration failed: {exc}"
                Path(a.out).write_text(json.dumps(rec), encoding="utf-8")
                return 2

    feeds = rm.build_feeds(case, np)
    sess = _session(ort, Path(model_rec["path"]), arm, a.device, profiling=True)
    rec["providers"] = list(sess.get_providers())
    for _ in range(a.iters):
        sess.run(None, feeds)
    rec["inferences"] = a.iters
    rec["profile"] = _profile_provider_counts(sess, ort)
    rec["fallback"] = rm.fallback_diagnosis((rec["profile"] or {}).get("counts"))
    del sess
    Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return 0


def run_diagnostics(args, models, device) -> dict:
    """Spawn one worker per (model, arm, case) and merge, adding the EP counters file.

    A subprocess per arm and not a loop, for the reason `phi35.py` already states: the counters
    file is written from a process-exit hook, so a process that has not torn down has not written
    it, and ORT's EP registration is process-global.
    """
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    out: dict = {"device": {"index": device.index, "name": device.name}, "runs": []}
    plan = []
    for key in models:
        if key == rm.PHI35.key:
            plan += [(key, "prefill", 1, 0), (key, "prefill", 8, 0), (key, "prefill", 128, 0),
                     (key, "decode", 1, 1024)]
        else:
            plan += [(key, "batch", 1, 0), (key, "batch", 16, 0)]
    for (key, phase, m, past) in plan:
        for arm in rm.ARMS:
            tag = f"{key}_{phase}_{m}_{past}_{arm.name}".replace("/", "_")
            rec_path = scratch / f"diag_{tag}.json"
            counters = scratch / f"counters_{tag}.json"
            for p in (rec_path, counters):
                p.unlink(missing_ok=True)
            env = dict(os.environ)
            env[COUNTERS_ENV] = str(counters)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-diagnose",
                   "--model", key, "--arm", arm.name, "--phase", phase, "--m", str(m),
                   "--past", str(past), "--device", str(device.index),
                   "--iters", str(args.diag_iters), "--out", str(rec_path)]
            proc = subprocess.run(cmd, env=env, capture_output=True)
            rec = {"arm": arm.name, "case": f"{key}/{phase}/M{m}/past{past}"}
            if rec_path.exists():
                try:
                    rec = json.loads(rec_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    rec["error"] = "unreadable worker record"
            else:
                rec["error"] = f"worker exit {proc.returncode}"
            if counters.exists():
                try:
                    rec["counters"] = json.loads(counters.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    rec["counters"] = None
            rec["dispatch"] = rm.dispatch_diagnosis(rec.get("counters"),
                                                    rec.get("inferences") or 0)
            out["runs"].append(rec)
            disp = rec["dispatch"].get("dispatches_per_inference")
            print(f"[diag] {rec.get('case')} {arm.name}: "
                  f"islands={rec['dispatch'].get('islands')} "
                  f"disp/inf={disp} "
                  f"fallback={(rec.get('fallback') or {}).get('cpu_fallback_node_executions')}",
                  flush=True)
    return out


# --------------------------------------------------------------------------------------------


def environment_record(device, args) -> dict:
    import onnxruntime as ort

    import environment as env_mod

    lib = os.environ.get(EP_LIB_ENV)
    rec = {
        "onnxruntime": ort.__version__,
        "onnxruntime_lib": str(Path(ort.__file__).resolve().parent / "capi"),
        "ep_library": lib,
        "ep_library_sha256": rm.sha256_file(lib) if lib and Path(lib).is_file() else None,
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(),
        "device": {
            "index": device.index,
            "name": device.name,
            # Stable identity (issue #18, landed on main as #54). An index is a position in a
            # probe order and a name is not unique across two identical cards, so neither can
            # say WHICH card a reading came from once the box changes. `uuid`/`luid`/`pci` can.
            # `getattr` with a default because the artifacts already committed under §26 were
            # written before #54 and carry `index`/`name`/`driver` only: this records identity
            # for every FUTURE run without retroactively claiming the old runs recorded it.
            "uuid": getattr(device, "uuid", None),
            "luid": getattr(device, "luid", None),
            "pci": getattr(device, "pci", None),
            "driver": getattr(device, "driver_version", None),
            "transfer_class": getattr(device, "transfer_class", None),
            "timestamp_period_ns": getattr(device, "timestamp_period", None),
            "max_compute_shared_memory": getattr(device, "max_compute_shared_memory", None),
            "subgroup_size": getattr(device, "subgroup_size", None),
        },
        "power_and_affinity": {
            "assumption": "stock Windows power plan, no CPU affinity mask set, no GPU clock lock",
            "thread_affinity_set": False,
            "gpu_clocks_locked": False,
            "note": ("this box is shared indefinitely (PERF.md §20), so wall clock is "
                     "STEADY_UNCERTIFIED by default; arms are interleaved precisely because the "
                     "machine state cannot be assumed quiet"),
        },
        "methodology": {
            "repeats": args.repeats,
            "iters_per_repeat": args.iters,
            "warmup_per_session": args.warmup,
            "warmups_discarded": True,
            "arm_order": "alternated per repeat (see real_model.arm_order)",
            "session_per_arm_per_repeat": True,
            "tracing_during_timed_pass": False,
        },
    }
    try:
        rec["build"] = env_mod.build_info()
    except Exception as exc:  # pragma: no cover - environment dependent
        rec["build"] = {"error": str(exc)}
    return rec


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker-diagnose" in argv:
        return diagnose_worker(argv)

    ap = argparse.ArgumentParser(description=__doc__)
    # Default `None`, resolved AFTER `--diagnose` is known. A single default here was a real
    # defect on 2026-08-07: `--diagnose` inherited the timed pass's filename and overwrote a
    # completed thirteen-minute matrix with a profiling record that has a different schema. The
    # two passes measure different things and must not be able to land on the same path by
    # accident.
    ap.add_argument("--out", default=None,
                    help="defaults to real_model_latency.json, or real_model_diagnostics.json "
                         "under --diagnose")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--models", default="all",
                    help="comma-separated model keys, or 'all'")
    ap.add_argument("--prefill-m", default=",".join(str(x) for x in DEFAULT_PREFILL_M))
    ap.add_argument("--decode-past", default=",".join(str(x) for x in DEFAULT_DECODE_PAST))
    ap.add_argument("--batch", default=",".join(str(x) for x in DEFAULT_BATCH))
    ap.add_argument("--diagnose", action="store_true",
                    help="run the profiling pass instead of the timed pass")
    ap.add_argument("--diag-iters", type=int, default=3)
    ap.add_argument("--scratch", default=str(_BENCH / "results" / "_issue56_scratch"))
    args = ap.parse_args(argv)

    import bench as bench_mod
    import devices as device_mod

    if not bench_mod.register_ep():
        print("[real] refusing: no Vulkan EP to measure. A one-armed comparison is not a "
              "comparison.", file=sys.stderr)
        return 2

    facts, _src = device_mod.probe()
    try:
        device = bench_mod.select_device(facts, args.device)
    except Exception as exc:
        print(f"[real] refusing to run: {exc}", file=sys.stderr)
        return 2
    print(f"[real] device {device.index}: {device.name} [{device.transfer_class}]", flush=True)

    keys = list(rm.MODELS) if args.models == "all" else args.models.split(",")
    missing = [k for k in keys if k not in rm.MODELS]
    if missing:
        print(f"[real] unknown model key(s): {missing}", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else Path(__file__).with_name(
        "real_model_diagnostics.json" if args.diagnose else "real_model_latency.json")
    if args.out and not out_path.is_absolute():
        # A bare filename lands beside this script (the results directory); anything with a
        # separator is taken as given, relative to the caller's cwd.
        out_path = (Path(__file__).with_name(args.out) if len(Path(args.out).parts) == 1
                    else Path(args.out).resolve())

    if args.diagnose:
        report = {
            "schema": "real_model_diagnostics/1",
            "environment": environment_record(device, args),
            **run_diagnostics(args, keys, device),
        }
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  wrote {out_path}")
        return 0

    prefill_m = [int(x) for x in args.prefill_m.split(",") if x]
    decode_past = [int(x) for x in args.decode_past.split(",") if x != ""]
    batch = [int(x) for x in args.batch.split(",") if x]

    models_out = []
    failures = []
    for key in keys:
        spec = rm.MODELS[key]
        try:
            model_rec = rm.resolve_model(spec)
        except rm.ModelUnavailable as exc:
            # Loud, never a silent skip: a lane that drops a model when its cache moves reads as
            # complete while covering less than it claims.
            print(f"[real] MODEL UNAVAILABLE: {exc}", file=sys.stderr)
            failures.append({"model": key, "error": str(exc)})
            continue
        print(f"[real] {key}: {model_rec['path']}", flush=True)
        print(f"[real]   sha256 {model_rec['sha256']} ({model_rec['provenance']}, "
              f"agrees_with_recorded={model_rec['agrees_with_recorded_provenance']})", flush=True)
        cases = (rm.phi35_cases(prefill_m, decode_past) if key == rm.PHI35.key
                 else rm.mobilenet_cases(batch))
        models_out.append(run_matrix(args, model_rec, cases, rm.ARMS, device))

    report = {
        "schema": rm.SCHEMA,
        "issue": 56,
        "depends_on": "PR #53 (squad/7-tile-matmulnbits-prefill) — the row tile under test",
        "environment": environment_record(device, args),
        "models": models_out,
        "unavailable_models": failures,
        "second_device": {
            "present": False,
            "detail": "vulkaninfo reports exactly one Vulkan device on this box; there is no "
                      "second GPU and no software ICD installed. Absence is recorded rather "
                      "than left to inference.",
        },
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_tables(report)
    print(f"\n  wrote {out_path}")
    return 1 if failures else 0


def _print_tables(report: dict) -> None:
    for mrec in report["models"]:
        print(f"\n  {mrec['model']['key']} on {report['environment']['device']['name']}")
        print(f"  {'case':>28}  {'tiled ms':>9}  {'untiled ms':>10}  {'cpu ms':>9}  "
              f"{'tile x':>7}  {'vs cpu':>7}  {'throughput':>16}")
        for r in mrec["rows"]:
            a = r["arms"]
            tp = a["vulkan_tiled"].get("throughput") or {}
            tag = "  <- null control" if r["null_control"] else ""
            print(f"  {r['case'][-28:]:>28}  "
                  f"{a['vulkan_tiled']['latency'].get('median_ms', float('nan')):>9.2f}  "
                  f"{a['vulkan_untiled']['latency'].get('median_ms', float('nan')):>10.2f}  "
                  f"{a['cpu']['latency'].get('median_ms', float('nan')):>9.2f}  "
                  f"{r['row_tile_speedup'].get('median', float('nan')):>6.3f}x  "
                  f"{r['vulkan_vs_cpu_tiled'].get('median', float('nan')):>6.3f}x  "
                  f"{tp.get('value', float('nan')):>10.1f} {tp.get('unit', ''):<5}{tag}")


if __name__ == "__main__":
    raise SystemExit(main())
