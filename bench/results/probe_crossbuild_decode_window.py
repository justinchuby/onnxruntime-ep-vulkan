"""Where is decode slower, and by how much — the same two builds, swept across KV length (#96).

WHY THIS EXISTS ALONGSIDE `probe_crossbuild_gqa_landing.py`
===========================================================
PR #95 measured ten workloads across two builds and found exactly one row that got worse:
Phi-3.5 decode at ``past = 128``, ~14%, in all three whole-process repeats. It measured two decode
points — 128 and 1024 — and 1024 showed nothing. Two points cannot tell a *window* from a *point*
from a *trend*, and issue #96's hypothesis 3 is precisely that there is a window.

This instrument is an independent re-implementation, by a different author, of a *compatible*
protocol: same model and provenance resolution (`bench/real_model.py`), same one-process-per-
(workload, arm, repeat) rule, same equivalence-and-witness gating, same ratio convention. It shares
no driver, gate, pairing or verdict code with PR #95's probe, on purpose — a reproduction that
imports the thing it is reproducing reproduces its bugs too.

What is different, and why:

* **Seven Phi-3.5 points instead of two**: ``past`` 0 (prefill M=1), 32, 64, 128, 256, 512, 1024.
  That is the sweep #96 asks for, and it is what locates an edge if there is one.
* **The band's source is a workload with no GQA node in it at all.** PR #95 took its band from
  prefill ``M = 1``, which resolves to ``local = 1`` — but so does *every* decode point in this
  sweep, so that row is a member of the treatment group, not a control. MobileNetV2 contains no
  `GroupQueryAttention`; the landed change cannot reach it; its spread is drift and nothing else.
* **The arm-identity check is an admissibility rule, not a report.** A record whose
  ``ep_library_sha256`` is not its arm's declared digest, or whose ``gqa_f16`` witness key is not
  the key its arm must produce, is refused before it can contribute a ratio.

WHAT IT DOES NOT DO
===================
It does not name a mechanism. ``--attribution`` localises a difference to a device kernel or to a
named host phase, using the EP's own pre-existing tracer, and that is a separate section of the
artifact with its own caveats. Neither pass may be quoted as evidence for a shader change; see
`bench/results/prereg_crossbuild_decode_window.md` §9.

Usage
-----
::

    # 1. hash the pre-registration (it must already exist)
    # 2. sweep, holding the machine's GPU lock for the whole run
    python bench/results/probe_crossbuild_decode_window.py \
        --baseline-lib  <c96e7d9 build>/onnxruntime_vulkan_ep.dll \
        --candidate-lib <85fbda2 build>/onnxruntime_vulkan_ep.dll \
        --out bench/results/crossbuild_decode_window.json

    # 3. per-kernel / per-phase attribution at three lengths, tracer on
    python bench/results/probe_crossbuild_decode_window.py --attribution ... --out ...
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1]
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "python"), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import real_model as rm  # noqa: E402

EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"
TRACE_GPU_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU"

#: Fixed by the pre-registration. Changing any of these invalidates the artifact's prereg digest.
REPEATS = 3
WARMUPS = 5
ITERS = 20
#: The band's floor. The applied band is `max(BAND_FLOOR, drift-control half-range)`; the rule is
#: declared before the first timed iteration and the number is only ever *resolved* by the run.
BAND_FLOOR = 0.05

#: `(key, phase, m, past, role)`, measured in this order. `role` decides what the record is FOR:
#: `control-no-gqa` sets the band, `treatment` is what the band is applied to.
WORKLOADS = (
    (rm.MOBILENETV2.key, "batch", 1, 0, "control-no-gqa"),
    (rm.MOBILENETV2.key, "batch", 16, 0, "control-no-gqa"),
    (rm.PHI35.key, "prefill", 1, 0, "treatment"),
    (rm.PHI35.key, "decode", 1, 32, "treatment"),
    (rm.PHI35.key, "decode", 1, 64, "treatment"),
    (rm.PHI35.key, "decode", 1, 128, "treatment"),
    (rm.PHI35.key, "decode", 1, 256, "treatment"),
    (rm.PHI35.key, "decode", 1, 512, "treatment"),
    (rm.PHI35.key, "decode", 1, 1024, "treatment"),
)

#: KV lengths the attribution pass runs at. A subset, because tracing is not free and attribution
#: is a *localisation*, not a headline.
ATTRIBUTION_PAST = (64, 128, 256)

#: The witness key each arm must produce on a workload that has a `GroupQueryAttention` node.
#: The candidate's dispatch passes `vec![local]` and the baseline's `vec![]`, so the keys the EP
#: writes at pipeline-creation time are `gqa_f16:1` and `gqa_f16:` — different strings, and the
#: only place in this run where the two builds are distinguished by something the *device* saw.
EXPECTED_GQA_KEY = {"candidate": "gqa_f16:1", "baseline": "gqa_f16:"}

#: Host phases the EP's tracer emits. `nested_in` children are excluded from the sum by the
#: aggregator, not by this list — the list is only the vocabulary.
HOST_PHASES = ("compile", "prepack", "record", "desc_alloc", "pipeline_lookup",
               "cmd_upload", "upload", "submit", "fence_wait", "readback")


def workload_label(key: str, phase: str, m: int, past: int) -> str:
    if phase == "batch":
        return f"{key}/batch/N{m}"
    return f"{key}/{phase}/M{m}/past{past}"


def make_case(key: str, phase: str, m: int, past: int) -> rm.Case:
    if phase == "batch":
        return rm.Case(key, "batch", m, 0, tokens=None, unit="images")
    return rm.Case(key, phase, m, past, tokens=m if phase == "prefill" else 1)


def has_gqa(key: str) -> bool:
    """Does this model contain a `GroupQueryAttention` node the EP will claim?

    Phi-3.5 does — 32 of them, one per layer. MobileNetV2 is a convnet and has none, which is what
    makes it a no-treatment control rather than a second subject.
    """
    return key == rm.PHI35.key


def outputs_digest(outputs, np) -> str:
    """sha256 over every output's dtype, shape and exact bytes, in order.

    Bytes, not values: two arrays that compare equal under a tolerance are not the same outputs,
    and this digest is the thing that says the arms produced *identical* results rather than
    acceptable ones.
    """
    h = hashlib.sha256()
    for o in outputs:
        a = np.ascontiguousarray(o)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# Exclusive GPU: wait, never kill
# --------------------------------------------------------------------------------------------


def default_lock_path() -> Path:
    """The machine's existing GPU lock, deliberately outside every worktree.

    Deliberately not created inside the repository: a lock that lives in a worktree is not a lock
    between worktrees, which is the only contention that matters here.
    """
    env = os.environ.get("ONNXRUNTIME_EP_VULKAN_GPU_LOCK")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "gpu-exclusive.lock"


class GpuLock:
    """An OS byte-range lock on a file outside every worktree.

    Policy is **wait, never kill**. If another process holds the device, this blocks and records
    how long it blocked. Nothing is ever terminated, no process list is acted on, and the record
    says so in a field a reviewer can read.
    """

    def __init__(self, path: Path, poll_s: float = 5.0):
        self.path = Path(path)
        self.poll_s = poll_s
        self._fh = None
        self.record: dict = {"state": "UNACQUIRED", "killed_anything": False,
                             "policy": "wait, never kill"}

    def acquire(self) -> dict:
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        t0 = time.time()
        waits = 0
        while True:
            try:
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                waits += 1
                if waits == 1:
                    print(f"[lock] held by another process; waiting (never killing)", flush=True)
                time.sleep(self.poll_s)
        self.record = {
            "state": "ACQUIRED",
            "lock_path": str(self.path),
            "waited_seconds": round(time.time() - t0, 1),
            "poll_seconds": self.poll_s,
            "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "acquired_at_monotonic": time.perf_counter(),
            "killed_anything": False,
            "policy": "wait, never kill",
            "mechanism": "msvcrt.locking LK_NBLCK, 1 byte at offset 0",
        }
        return self.record

    def release(self) -> dict:
        import msvcrt
        if self._fh is not None:
            try:
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._fh.close()
            self._fh = None
        held = None
        if self.record.get("acquired_at_monotonic") is not None:
            held = round(time.perf_counter() - self.record["acquired_at_monotonic"], 1)
        self.record["state"] = "RELEASED"
        self.record["held_seconds"] = held
        self.record["released_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.record.pop("acquired_at_monotonic", None)
        return self.record


# Anything that looks like a filesystem location, whatever key it arrives under.
_ABS_PATH = re.compile(r"([A-Za-z]:[\\/]|\\\\[^\\]+\\|/home/|/Users/|/tmp/)")
# Fields that identify *who else* was on the box. The policy is wait-never-kill, so this
# instrument has no reason to hold a holder's pid or command line, let alone publish one.
_HOLDER_FIELDS = ("holder_pid", "holder_cmdline", "holder", "holder_name", "owner", "user")


def sanitize_lock(record: dict) -> dict:
    """Publish *that* the lock was held, not where an operator's home directory is.

    Key-agnostic on purpose. An earlier version scrubbed the one key it knew about (`lock_path`)
    and would have published any other path-shaped value untouched; the guard suite plants a
    record carrying a home directory under a different key and a holder command line, because a
    screen that only knows today's field names is not a screen.
    """
    out = {}
    for k, v in record.items():
        if k == "acquired_at_monotonic" or k in _HOLDER_FIELDS:
            continue
        if isinstance(v, str) and _ABS_PATH.search(v):
            out[k] = Path(v).name
            out["path_note"] = ("basenames only; an absolute path is a fact about this "
                                "operator's machine, not about the measurement")
            continue
        out[k] = v
    return out


def stamp_released(out_path: Path, lock: "GpuLock") -> None:
    """Re-embed the *terminal* lock record, and nothing else.

    The artifact is written while the run is still holding the device, so the only lock state that
    exists at that moment is `ACQUIRED`. This replaces it with the released record — held duration,
    release time — and asserts that no measurement field moved while doing so.
    """
    if not out_path.is_file():
        return
    art = json.loads(out_path.read_text(encoding="utf-8"))
    before = json.dumps({k: v for k, v in art.items() if k != "exclusivity"}, sort_keys=True)
    art["exclusivity"] = sanitize_lock(lock.record)
    after = json.dumps({k: v for k, v in art.items() if k != "exclusivity"}, sort_keys=True)
    assert before == after, "stamping the lock record must not touch any measurement field"
    out_path.write_text(json.dumps(art, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------------
# The worker: one process, one (workload, arm, repeat)
# --------------------------------------------------------------------------------------------


def _session(ort, path: Path, providers, device_index: int):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if rm.EP_NAME in providers:
        opts.add_session_config_entry("ep.device_index", str(device_index))
    return ort.InferenceSession(str(path), opts, providers=list(providers))


def worker(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--past", type=int, default=0)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUPS)
    ap.add_argument("--traced", action="store_true",
                    help="attribution mode: the tracer is on and no headline number is produced")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rec: dict = {
        "workload": workload_label(a.model, a.phase, a.m, a.past),
        "arm": a.arm, "repeat": a.repeat, "model_key": a.model,
        "phase": a.phase, "m": a.m, "past": a.past,
        "traced": bool(a.traced),
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_monotonic": time.perf_counter(),
    }

    def bail(reason: str, **extra) -> int:
        """A refusal. `speed` is never written and is popped if it somehow exists."""
        rec["admissible"] = False
        rec["refusal"] = {"reason": reason, **extra}
        rec.pop("speed", None)
        rec["finished_monotonic"] = time.perf_counter()
        Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return 3

    import numpy as np
    import onnxruntime as ort

    lib = os.environ.get(EP_LIB_ENV)
    if not lib or not Path(lib).is_file():
        return bail(f"{EP_LIB_ENV} unset or not a file", lib=lib)
    rec["ep_library_sha256"] = rm.sha256_file(lib)
    rec["ep_library_bytes"] = Path(lib).stat().st_size
    try:
        ort.register_execution_provider_library(rm.EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            return bail(f"EP registration failed: {exc}")

    try:
        spec = rm.MODELS[a.model]
        model_rec = rm.resolve_model(spec)
    except Exception as exc:  # noqa: BLE001
        return bail(f"model unresolvable: {exc}")
    rec["model"] = {k: v for k, v in model_rec.items() if k != "path"}

    case = make_case(a.model, a.phase, a.m, a.past)
    feeds = rm.build_feeds(case, np)
    rec["feeds_sha256"] = rm.feeds_digest(feeds)
    path = Path(model_rec["path"])

    # -- correctness first, in this process, against this process's own CPU reference ----------
    try:
        ref_sess = _session(ort, path, (rm.CPU_EP,), a.device)
        t0 = time.perf_counter()
        reference = ref_sess.run(None, feeds)
        rec["cpu_reference_ms"] = (time.perf_counter() - t0) * 1000.0
        del ref_sess
    except Exception as exc:  # noqa: BLE001
        return bail(f"CPU reference failed: {exc}")

    try:
        t0 = time.perf_counter()
        sess = _session(ort, path, (rm.EP_NAME, rm.CPU_EP), a.device)
        rec["session_build_ms"] = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        return bail(f"Vulkan session build failed: {exc}")
    rec["providers"] = list(sess.get_providers())
    if rm.EP_NAME not in rec["providers"]:
        return bail("Vulkan EP absent from the session's provider list")

    runs = 0
    try:
        t0 = time.perf_counter()
        got = sess.run(None, feeds)
        runs += 1
        rec["first_run_ms"] = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        return bail(f"Vulkan first run failed: {exc}")

    rec["outputs_sha256"] = outputs_digest(got, np)
    if not rec["outputs_sha256"]:
        return bail("empty outputs digest")
    equivalence = rm.classify_outputs(case, got, reference, np)
    rec["equivalence"] = equivalence
    if equivalence.get("verdict") != rm.MATCH:
        return bail("equivalence not MATCH", verdict=equivalence.get("verdict"))

    for _ in range(a.warmup):
        sess.run(None, feeds)
        runs += 1
    samples = []
    for _ in range(a.iters):
        t = time.perf_counter()
        sess.run(None, feeds)
        samples.append((time.perf_counter() - t) * 1000.0)
        runs += 1

    # Taken again after the timed pass: an arm that agreed on its verification run and drifted
    # under repetition would otherwise keep a verdict its timed iterations never earned.
    post = sess.run(None, feeds)
    runs += 1
    rec["outputs_sha256_post_timing"] = outputs_digest(post, np)
    if rec["outputs_sha256_post_timing"] != rec["outputs_sha256"]:
        return bail("outputs digest changed across the timed pass")
    del sess

    rec["inference_calls"] = runs
    if a.traced:
        # An attribution process publishes no headline: its latency carries the tracer's own cost.
        rec["traced_samples_ms"] = [round(x, 4) for x in samples]
    else:
        rec["speed"] = {
            "samples_ms": [round(x, 4) for x in samples],
            **rm.latency_stats(samples),
            "throughput": rm.throughput(case, statistics.median(samples)),
        }
    rec["admissible"] = True
    rec["refusal"] = None
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["finished_monotonic"] = time.perf_counter()
    Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------------------------
# Witness and admissibility — pure functions over a written record
# --------------------------------------------------------------------------------------------


def gqa_witness(counters: "dict | None") -> dict:
    """The `gqa_f16` pipeline key the EP actually built, out of its own counters file.

    `pipeline_variants` is written by `vk::session` from the **resolved** specialisation vector at
    `vkCreateComputePipelines` time, so it is a statement about what the device was handed. It is
    the only field in this run that can tell the two builds apart from the inside.
    """
    out = {"present": False, "gqa_keys": [], "local_size": None, "all_variants": []}
    if not counters:
        return out
    out["present"] = True
    variants = list(counters.get("pipeline_variants") or [])
    out["all_variants"] = variants
    keys = [v for v in variants if v.split(":")[0] == "gqa_f16"]
    out["gqa_keys"] = keys
    sizes = []
    for k in keys:
        tail = k.split(":", 1)[1]
        sizes.append(int(tail.split(",")[0]) if tail else None)
    out["local_size"] = sizes[0] if len(sizes) == 1 else (sizes or None)
    for field in ("dispatches_executed", "compute_calls", "compute_failures",
                  "claimed_nodes", "running_device_names", "shaders_dispatched"):
        out[field] = counters.get(field)
    return out


def admissibility_gate(rec: dict, expected_lib_sha: "dict[str, str] | None" = None) -> dict:
    """Re-derive admissibility from the written record, and strip `speed` from anything refused.

    Pure on purpose: the worker already refuses in-process, but the worker is the thing being
    trusted. This pass reads only the record, so a mutated record cannot reach a timing — which is
    exactly the surface `test_crossbuild_decode_window.py` drives.
    """
    def refuse(reason: str, **extra) -> dict:
        rec["admissible"] = False
        rec["refusal"] = {"reason": reason, **extra}
        rec.pop("speed", None)
        return rec

    if not rec.get("admissible"):
        rec.pop("speed", None)
        return rec

    eq = rec.get("equivalence")
    if not eq or not eq.get("verdict"):
        return refuse("no output equivalence verdict on this (arm, repeat)")
    if eq.get("verdict") != rm.MATCH:
        return refuse("equivalence not MATCH", verdict=eq.get("verdict"))

    if not rec.get("outputs_sha256"):
        return refuse("missing or empty outputs digest")
    if rec.get("outputs_sha256_post_timing") != rec.get("outputs_sha256"):
        return refuse("outputs digest changed across the timed pass")

    model = rec.get("model") or {}
    if not model.get("sha256"):
        return refuse("no model digest recorded for this process")
    recorded = model.get("recorded_sha256")
    if recorded and model["sha256"] != recorded:
        return refuse("model digest does not match the recorded provenance pin",
                      measured=model["sha256"], recorded=recorded)
    if model.get("agrees_with_recorded_provenance") is False:
        return refuse("model provenance disagreement reported by the resolver")

    w = rec.get("path_witness") or {}
    if not w.get("present"):
        return refuse("no path witness: the EP wrote no counters file")
    if w.get("compute_failures"):
        return refuse("compute_failures > 0", compute_failures=w.get("compute_failures"))
    if not w.get("dispatches_executed"):
        return refuse("no dispatch executed on the device")

    arm = rec.get("arm")
    if expected_lib_sha:
        want = expected_lib_sha.get(arm)
        if want and rec.get("ep_library_sha256") != want:
            return refuse("EP library digest is not this arm's declared build",
                          measured=rec.get("ep_library_sha256"), declared=want)

    keys = list(w.get("gqa_keys") or [])
    if has_gqa(rec.get("model_key", "")):
        want_key = EXPECTED_GQA_KEY.get(arm)
        if keys != [want_key]:
            return refuse("gqa_f16 witness key is not the one this arm must produce",
                          measured=keys, declared=want_key)
    elif keys:
        return refuse("a no-GQA control built a gqa_f16 pipeline", measured=keys)

    return rec


# --------------------------------------------------------------------------------------------
# Driving one process
# --------------------------------------------------------------------------------------------


def child_env(args, arm: str, counters_path: Path, trace_path: "Path | None") -> dict:
    """Production defaults on both arms, with exactly two variables set by this driver.

    Every inherited `ONNXRUNTIME_EP_VULKAN_*` is stripped, so an operator's shell cannot become
    part of the result — including the GQA local-size override, which would make the two arms
    agree by construction and is the single most dangerous variable in this experiment.
    """
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("ONNXRUNTIME_EP_VULKAN_"):
            env.pop(k)
    env[EP_LIB_ENV] = args.candidate_lib if arm == "candidate" else args.baseline_lib
    env[COUNTERS_ENV] = str(counters_path)
    if trace_path is not None:
        env[TRACE_ENV] = str(trace_path)
        env[TRACE_GPU_ENV] = "1"
    return env


def run_one(args, key, phase, m, past, arm, repeat, scratch: Path, *,
            traced: bool = False) -> dict:
    label = workload_label(key, phase, m, past)
    tag = f"{label}_{arm}_r{repeat}{'_traced' if traced else ''}"
    tag = tag.replace("/", "_").replace(".", "_")
    out_path = scratch / f"rec_{tag}.json"
    counters_path = scratch / f"counters_{tag}.json"
    trace_path = (scratch / f"trace_{tag}.json") if traced else None
    for p in (out_path, counters_path, trace_path):
        if p is not None:
            p.unlink(missing_ok=True)

    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--model", key, "--phase", phase, "--m", str(m), "--past", str(past),
           "--arm", arm, "--repeat", str(repeat), "--device", str(args.device),
           "--iters", str(args.iters), "--warmup", str(args.warmup), "--out", str(out_path)]
    if traced:
        cmd.append("--traced")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=child_env(args, arm, counters_path, trace_path),
                          capture_output=True, text=True)
    wall = time.perf_counter() - t0

    if out_path.exists():
        rec = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        rec = {"workload": label, "arm": arm, "repeat": repeat, "model_key": key,
               "phase": phase, "m": m, "past": past, "admissible": False,
               "refusal": {"reason": f"worker wrote no record (exit {proc.returncode})",
                           "stderr": (proc.stderr or "")[-1500:]}}
    rec["worker_exit"] = proc.returncode
    rec["worker_wall_s"] = round(wall, 3)
    rec["traced"] = traced

    counters = None
    if counters_path.exists():
        try:
            counters = json.loads(counters_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            counters = None
    rec["path_witness"] = gqa_witness(counters)
    if traced and trace_path is not None and trace_path.is_file():
        rec["trace"] = trace_breakdown(trace_path, rec.get("inference_calls") or 0)

    admissibility_gate(rec, expected_lib_sha=args._lib_sha)
    if not rec.get("admissible") and "trace" in rec:
        # A device-clock breakdown is a timing, and the equivalence gate does not distinguish
        # between kinds of timing: a process whose outputs did not agree with the CPU reference
        # may not have executed the graph this experiment is about, so none of its numbers —
        # host OR device — may be quoted. Kept as a count so the refusal stays legible.
        spans = sum((rec["trace"].get("gpu_span_counts") or {}).values())
        rec["trace"] = {"withheld": "record refused; no timing is published for it",
                        "gpu_spans_seen": spans}
    assert rec.get("admissible") or "speed" not in rec, \
        "a refused record must not carry speed fields"
    assert rec.get("admissible") or "traced_samples_ms" not in rec, \
        "a refused record must not carry traced samples"
    return rec


# --------------------------------------------------------------------------------------------
# Attribution: the EP's own tracer, per kernel and per host phase
# --------------------------------------------------------------------------------------------


def kernel_key(name: str) -> str:
    """`vulkan.gpu.gqa_f16` -> `gqa_f16`.

    The tracer's GPU span names are fully qualified. Summarising under the qualified name and
    then looking a kernel up under the bare one returns `None`, and a difference built on
    `x or 0` turns that `None` into a **zero delta** — a missing measurement wearing the
    appearance of a measured absence of difference. The first attribution pass of this
    investigation published exactly that, and this function plus
    `test_attribution_delta_never_reports_a_false_zero_for_a_present_kernel` is the repair.
    """
    return name.split("vulkan.gpu.", 1)[-1]


def trace_breakdown(trace_path: Path, inference_calls: int) -> dict:
    """Per-inference device and host breakdown out of the EP's Chrome trace.

    Two rules, both of which are the difference between a breakdown and a double-count:

    * `cat == "gpu"` spans are the **only** GPU time here. They come from `VkQueryPool`
      timestamp queries on the device's own clock. Host wall time around a submit is not GPU time
      and is never treated as such.
    * A host phase span carrying a `nested_in` arg is a *child* of another phase (the staging
      upload inside `record`, the pipeline lookup inside `record`). Summing it with its parent
      counts it twice, so children are reported separately and excluded from the phase total.
      `rust/src/trace.rs:830` emits that arg for **every** phase and writes the literal string
      `"none"` for a sibling, so a truthiness test on it classifies the whole top level as
      nested — this function tests the *value*, and `test_crossbuild_decode_window.py` plants
      that exact mutation.

    Every `*_per_inference` figure is a **mean** over `inference_calls`, not a median: the
    process's first run carries session build and pipeline compilation, so the mean sits above
    the timed median and the two must not be differenced.
    """
    d = json.loads(trace_path.read_text(encoding="utf-8"))
    events = d["traceEvents"] if isinstance(d, dict) else d
    gpu: collections.Counter = collections.Counter()
    gpu_n: collections.Counter = collections.Counter()
    host: collections.Counter = collections.Counter()
    host_n: collections.Counter = collections.Counter()
    nested: collections.Counter = collections.Counter()
    nested_n: collections.Counter = collections.Counter()
    parents: dict = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "?")
        dur = float(e.get("dur", 0) or 0)
        cat = e.get("cat")
        if cat == "gpu":
            gpu[kernel_key(name)] += dur
            gpu_n[kernel_key(name)] += 1
        elif cat == "ep.phase":
            short = name.split("vulkan.", 1)[-1]
            parent = (e.get("args") or {}).get("nested_in")
            if parent in (None, "", "none"):
                host[short] += dur
                host_n[short] += 1
            else:
                nested[short] += dur
                nested_n[short] += 1
                parents[short] = parent
    n = max(1, int(inference_calls or 0))
    return {
        "inference_calls": inference_calls,
        "gpu_us_per_inference": {k: round(v / n, 2)
                                 for k, v in sorted(gpu.items(), key=lambda kv: -kv[1])},
        "gpu_span_counts": dict(gpu_n),
        "gpu_total_us_per_inference": round(sum(gpu.values()) / n, 2),
        "host_phase_us_per_inference": {k: round(v / n, 2)
                                        for k, v in sorted(host.items(), key=lambda kv: -kv[1])},
        "host_phase_counts": dict(host_n),
        "host_sibling_total_us_per_inference": round(sum(host.values()) / n, 2),
        "host_nested_us_per_inference": {k: round(v / n, 2) for k, v in nested.items()},
        "host_nested_counts": dict(nested_n),
        "host_nested_parents": parents,
        "note": ("gpu_* is device-clock timestamp-query time; host_phase_* is host wall time and "
                 "is NOT additive with it. nested spans are excluded from host_phase_* to avoid "
                 "double-counting their parent. Every per_inference figure is a MEAN over "
                 "inference_calls and includes the first run's compile, so it sits above the "
                 "timed median and may not be differenced against it."),
    }


# --------------------------------------------------------------------------------------------
# Pairing and verdicts
# --------------------------------------------------------------------------------------------


def paired(records, label) -> dict:
    """Per-repeat baseline/candidate ratios, with the refusal accounting that gates them.

    Ratio is `baseline / candidate`, so `> 1` means the candidate is faster. Paired within a
    repeat, because the two arms of a repeat share whatever the machine was doing.
    """
    cand = {r["repeat"]: r for r in records
            if r["workload"] == label and r["arm"] == "candidate"}
    base = {r["repeat"]: r for r in records
            if r["workload"] == label and r["arm"] == "baseline"}
    refusals = [{"arm": r["arm"], "repeat": r["repeat"], "refusal": r.get("refusal")}
                for r in records if r["workload"] == label and not r.get("admissible")]
    out: dict = {"workload": label, "refusals": refusals, "repeats_paired": 0,
                 "arm_identity": "OK"}

    # Two arms that the production path cannot tell apart are not two arms. On a GQA workload the
    # witness keys must differ; on a control (no GQA pipeline on either arm) the library digests
    # must differ, which is the only distinction available there.
    for rep in sorted(set(cand) & set(base)):
        c, b = cand[rep], base[rep]
        ck = (c.get("path_witness") or {}).get("gqa_keys") or []
        bk = (b.get("path_witness") or {}).get("gqa_keys") or []
        same_lib = c.get("ep_library_sha256") == b.get("ep_library_sha256")
        if same_lib:
            out["arm_identity"] = "IDENTICAL-LIBRARY"
        elif ck and ck == bk:
            out["arm_identity"] = "IDENTICAL-WITNESS"
    if out["arm_identity"] != "OK":
        out["verdict"] = "REFUSED"
        out["refused_because"] = out["arm_identity"]
        return out

    rows = []
    for rep in sorted(set(cand) & set(base)):
        c, b = cand[rep], base[rep]
        if not (c.get("admissible") and b.get("admissible")):
            continue
        cm = (c.get("speed") or {}).get("median_ms")
        bm = (b.get("speed") or {}).get("median_ms")
        if not cm or not bm:
            continue
        rows.append({"repeat": rep, "candidate_ms": cm, "baseline_ms": bm,
                     "ratio": bm / cm,
                     "candidate_rsd": (c.get("speed") or {}).get("rsd"),
                     "baseline_rsd": (b.get("speed") or {}).get("rsd")})
    out["per_repeat"] = rows
    out["repeats_paired"] = len(rows)
    if not rows:
        out["verdict"] = "REFUSED"
        out["refused_because"] = "no paired admissible repeat"
        return out
    ratios = [r["ratio"] for r in rows]
    out["ratio_median"] = statistics.median(ratios)
    out["ratio_min"] = min(ratios)
    out["ratio_max"] = max(ratios)
    out["ratio_half_range"] = (max(ratios) - min(ratios)) / 2.0
    out["candidate_median_ms"] = statistics.median([r["candidate_ms"] for r in rows])
    out["baseline_median_ms"] = statistics.median([r["baseline_ms"] for r in rows])
    return out


def verdict_for(p: dict, band: float, repeats_required: int = REPEATS) -> str:
    """The pre-registered rule, applied. Nothing here reads a number before deciding."""
    if p.get("verdict") == "REFUSED":
        return "REFUSED"
    if p.get("repeats_paired", 0) < repeats_required:
        return "REFUSED"
    lo, hi, med = p["ratio_min"], p["ratio_max"], p["ratio_median"]
    if lo > 1 + band:
        return "FASTER"
    if hi < 1 - band:
        return "SLOWER"
    if abs(med - 1.0) > band:
        return "INDETERMINATE"
    if lo >= 1 - 2 * band and hi <= 1 + 2 * band:
        return "NEUTRAL"
    return "INDETERMINATE"


def drift_envelope(pairs: list) -> dict:
    """The no-GQA controls' own per-repeat ratio spread, and the band it implies.

    This is the number the band rule resolves against, and it is measured on workloads the landed
    change provably cannot reach. A treatment row whose worst repeat sits inside this envelope is
    disclosed as `INSIDE-DRIFT` even when the rule gives it a verdict.
    """
    ratios: list = []
    half_ranges: list = []
    for p in pairs:
        if p.get("role") != "control-no-gqa" or not p.get("per_repeat"):
            continue
        rs = [r["ratio"] for r in p["per_repeat"]]
        ratios.extend(rs)
        half_ranges.append((max(rs) - min(rs)) / 2.0)
    if not ratios:
        return {"n": 0, "band": BAND_FLOOR, "source": "floor only — no control produced a ratio"}
    return {
        "n": len(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "half_range_max": max(half_ranges),
        "band": max(BAND_FLOOR, max(half_ranges)),
        "source": "max per-workload half-range over the no-GQA drift controls",
    }


def window_claim(pairs: list) -> dict:
    """Only a shape the sweep actually measured, with its edges named.

    A single slow length surrounded by not-slow lengths is reported as a **point**, not a window
    with interpolated edges; a window needs an interior and a not-slow length on each side of it.
    """
    treat = [p for p in pairs if p.get("role") == "treatment" and p.get("past") is not None]
    treat.sort(key=lambda p: p["past"])
    slow = [p["past"] for p in treat if p.get("verdict") == "SLOWER"]
    if not slow:
        return {"claim": "NO-SLOW-LENGTH",
                "detail": "no measured KV length is SLOWER under the pre-registered rule",
                "lengths_measured": [p["past"] for p in treat]}
    lengths = [p["past"] for p in treat]
    lo_edge = next((lengths[i - 1] for i, x in enumerate(lengths)
                    if x == min(slow) and i > 0), None)
    hi_edge = next((lengths[i + 1] for i, x in enumerate(lengths)
                    if x == max(slow) and i + 1 < len(lengths)), None)
    if len(slow) == 1:
        return {"claim": "SINGLE-SLOW-POINT", "slow_lengths": slow,
                "not_slow_below": lo_edge, "not_slow_above": hi_edge,
                "lengths_measured": lengths}
    contiguous = lengths[lengths.index(min(slow)):lengths.index(max(slow)) + 1] == sorted(slow)
    return {"claim": "WINDOW" if contiguous else "NON-CONTIGUOUS-SLOW-SET",
            "slow_lengths": slow, "not_slow_below": lo_edge, "not_slow_above": hi_edge,
            "lengths_measured": lengths}


# --------------------------------------------------------------------------------------------


def environment_record(args) -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "repeats": REPEATS, "warmups": args.warmup, "iters": args.iters,
        "arm_order_rule": "candidate first on even repeats, baseline first on odd",
        "one_process_per": "(workload, arm, repeat)",
        "env_hygiene": ("every inherited ONNXRUNTIME_EP_VULKAN_* stripped from every child; "
                        "only the counters-file path (and, in the attribution pass, the trace "
                        "path) is set by this driver"),
        "assumption": "stock Windows power plan, no CPU affinity mask, no GPU clock lock",
    }


def library_identity(path: str) -> dict:
    p = Path(path)
    return {"path_basename": p.name, "sha256": rm.sha256_file(p), "bytes": p.stat().st_size}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return worker(argv)
    if "--resummarize" in argv:
        return _resummarize_main(argv)

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-lib", required=True)
    ap.add_argument("--candidate-lib", required=True)
    ap.add_argument("--baseline-commit", default=None)
    ap.add_argument("--candidate-commit", default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUPS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=str(_BENCH / "results" / "_decode_window_scratch"))
    ap.add_argument("--prereg", default=str(_BENCH / "results"
                                            / "prereg_crossbuild_decode_window.md"))
    ap.add_argument("--lock", default=None)
    ap.add_argument("--attribution", action="store_true",
                    help="run the tracer pass instead of the sweep")
    ap.add_argument("--only", default=None, help="comma-separated workload substrings")
    args = ap.parse_args(argv)

    for name in ("baseline_lib", "candidate_lib"):
        if not Path(getattr(args, name)).is_file():
            print(f"[decode-window] {name} is not a file: {getattr(args, name)}", file=sys.stderr)
            return 2
    libs = {"baseline": library_identity(args.baseline_lib),
            "candidate": library_identity(args.candidate_lib)}
    if libs["baseline"]["sha256"] == libs["candidate"]["sha256"]:
        print("[decode-window] refusing: both arms point at the same library", file=sys.stderr)
        return 2
    args._lib_sha = {k: v["sha256"] for k, v in libs.items()}

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    lock = GpuLock(Path(args.lock) if args.lock else default_lock_path())
    lock.acquire()
    print(f"[lock] {lock.record['state']} after {lock.record['waited_seconds']}s wait", flush=True)
    try:
        if args.attribution:
            return _attribution_main(args, libs, scratch, lock)
        return _sweep_main(args, libs, scratch, lock)
    finally:
        lock.release()
        stamp_released(Path(args.out), lock)
        print(f"[lock] RELEASED after {lock.record.get('held_seconds')}s held "
              f"(nothing was killed)", flush=True)


def _wanted(args):
    if not args.only:
        return WORKLOADS
    want = [s.strip() for s in args.only.split(",") if s.strip()]
    return tuple(w for w in WORKLOADS
                 if any(s in workload_label(w[0], w[1], w[2], w[3]) for s in want))


def resummarize_sweep(records: list, repeats: int) -> dict:
    """Rebuild a sweep artifact's `workloads`, `band` and `window` from its own `records`.

    The workload list is taken from the records rather than from the `WORKLOADS` constant, so
    this stays a function of the stored data and does not silently re-read today's source.
    """
    seen = {}
    for r in records:
        label = r.get("workload")
        if label is not None and label not in seen:
            seen[label] = {"role": r.get("role"),
                           "model_key": r.get("model_key"),
                           "past": r.get("past")}
    pairs = []
    for label, meta in seen.items():
        p = paired(records, label)
        p["role"] = meta["role"]
        p["past"] = meta["past"] if has_gqa(meta["model_key"] or "") else None
        p["model_key"] = meta["model_key"]
        pairs.append(p)

    drift = drift_envelope(pairs)
    band = drift["band"]
    for p in pairs:
        p["verdict"] = verdict_for(p, band, repeats_required=repeats)
        if p.get("per_repeat") and drift.get("n"):
            p["inside_drift_envelope"] = bool(
                drift["min"] <= p["ratio_min"] <= drift["max"]
                and drift["min"] <= p["ratio_max"] <= drift["max"])
            p["worst_repeat_ratio"] = p["ratio_min"]
    return {
        "band": {"floor": BAND_FLOOR, "rule": "max(0.05, max no-GQA control half-range)",
                 "applied": band, "drift_envelope": drift},
        "workloads": pairs,
        "window": window_claim(pairs),
    }


def artifact_kind(art: dict) -> str:
    """`sweep` or `attribution`, decided by what the artifact publishes, not by its filename."""
    if "workloads" in art and "window" in art:
        return "sweep"
    if "summary" in art:
        return "attribution"
    return "unknown"


def _resummarize_main(argv) -> int:
    """Rebuild an artifact's derived section from its own `records`. No device, no lock.

    This exists so a reviewer with neither this GPU nor this model can check the analysis: every
    figure in `workloads`/`band`/`window` (sweep) and in `summary` (attribution) is a function of
    `records`, and `--resummarize --check` asserts it. The measurement is never re-run and never
    changes; only the derivation does. The artifact kind is detected from its contents, so the
    same flag works on both artifacts and cannot report a spurious mismatch by re-deriving the
    wrong section.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--resummarize", required=True, help="artifact to re-derive")
    ap.add_argument("--check", action="store_true",
                    help="compare against the stored derivation and fail on any difference")
    ap.add_argument("--write", action="store_true", help="write the re-derived section back")
    a = ap.parse_args(argv)

    path = Path(a.resummarize)
    art = json.loads(path.read_text(encoding="utf-8"))
    kind = artifact_kind(art)
    if kind == "unknown":
        print(f"  {path.name} publishes neither `workloads`+`window` nor `summary`; "
              f"there is nothing to re-derive", file=sys.stderr)
        return 2
    repeats = max((r.get("repeat", 0) for r in art["records"]), default=-1) + 1

    if kind == "attribution":
        fresh = {"summary": summarize_attribution(art["records"], repeats)}
    else:
        fresh = resummarize_sweep(art["records"], repeats)
    stored = {k: art.get(k) for k in fresh}
    same = json.dumps(fresh, sort_keys=True) == json.dumps(stored, sort_keys=True)
    differing = [k for k in fresh
                 if json.dumps(fresh[k], sort_keys=True) != json.dumps(stored[k], sort_keys=True)]
    print(f"  kind={kind} records={len(art['records'])} repeats={repeats} "
          f"re-derived={'+'.join(fresh)} "
          f"reproduces={'YES' if same else 'NO'}")
    if not same:
        print(f"  differing sections: {', '.join(differing)}")
    if a.write and not same:
        art.update(fresh)
        art["resummarized_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        path.write_text(json.dumps(art, indent=2), encoding="utf-8")
        print(f"  -> rewrote {path}")
        return 0
    if a.check and not same:
        print("  the stored derivation is not a function of the stored records", file=sys.stderr)
        return 1
    return 0


def _sweep_main(args, libs, scratch: Path, lock: GpuLock) -> int:
    records = []
    t_start = time.time()
    for rep in range(args.repeats):
        arms = ("candidate", "baseline") if rep % 2 == 0 else ("baseline", "candidate")
        for (key, phase, m, past, role) in _wanted(args):
            for arm in arms:
                rec = run_one(args, key, phase, m, past, arm, rep, scratch)
                rec["role"] = role
                records.append(rec)
                med = (rec.get("speed") or {}).get("median_ms")
                print(f"[decode-window] r{rep} {rec['workload']:44s} {arm:9s} "
                      f"{'REFUSED: ' + str((rec.get('refusal') or {}).get('reason')) if not rec.get('admissible') else f'{med:9.2f} ms'}",
                      flush=True)

    derived = resummarize_sweep(records, args.repeats)
    pairs = derived["workloads"]
    band = derived["band"]["applied"]
    drift = derived["band"]["drift_envelope"]

    artifact = {
        "instrument": Path(__file__).name,
        "question": ("Is there a KV-length window in which the compiled 85fbda2 library is "
                     "slower at Phi-3.5 decode than the compiled c96e7d9 library?"),
        "issue": 96,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(time.time() - t_start, 1),
        "arms": {
            "baseline": {**libs["baseline"], "commit": args.baseline_commit},
            "candidate": {**libs["candidate"], "commit": args.candidate_commit},
        },
        "environment": environment_record(args),
        "ratio_convention": "baseline_median_ms / candidate_median_ms; > 1 means candidate faster",
        **derived,
        "counts": {
            "records": len(records),
            "admissible": sum(1 for r in records if r.get("admissible")),
            "refused": sum(1 for r in records if not r.get("admissible")),
        },
        "records": [{k: v for k, v in r.items() if k != "model"} for r in records],
        "model_identities": _model_identities(records),
        "exclusivity": sanitize_lock(lock.record),
    }
    if Path(args.prereg).is_file():
        artifact["preregistration"] = {
            "path": Path(args.prereg).name,
            "sha256": rm.sha256_file(args.prereg),
            "bytes": Path(args.prereg).stat().st_size,
        }
    Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"\n  band applied: {band:.4f}  (drift envelope "
          f"{drift.get('min', float('nan')):.3f}..{drift.get('max', float('nan')):.3f})")
    for p in pairs:
        if p.get("per_repeat"):
            rs = " / ".join(f"{r['ratio']:.3f}" for r in p["per_repeat"])
            print(f"  {p['workload']:44s} {p['ratio_median']:.3f}x  [{rs}]  {p['verdict']}")
        else:
            print(f"  {p['workload']:44s} {p['verdict']}")
    print(f"  window: {artifact['window']['claim']}")
    print(f"\n  -> {args.out}")
    return 0


def _model_identities(records) -> dict:
    out = {}
    for r in records:
        m = r.get("model") or {}
        if m.get("key") and m.get("sha256"):
            out[m["key"]] = {k: m.get(k) for k in
                             ("key", "sha256", "bytes", "provenance", "resolver",
                              "recorded_sha256", "agrees_with_recorded_provenance",
                              "weights_bytes")}
    return out


def attribution_delta(cand: dict, base: dict) -> dict:
    """candidate - baseline, per kernel and per host phase, with absence kept distinguishable.

    A key that only one arm produced yields `null`, not `0`. Zero is a measured statement that
    two arms agreed; `null` is the statement that one of them never supplied the number, and the
    two must never render the same on the face of an artifact.
    """
    def diff(x, y):
        return None if x is None or y is None else round(x - y, 2)

    out: dict = {"gpu_total": diff(cand.get("gpu_total_us"), base.get("gpu_total_us"))}
    for k in sorted(set(cand["gpu_us"]) | set(base["gpu_us"])):
        out[f"gpu_{k}"] = diff(cand["gpu_us"].get(k), base["gpu_us"].get(k))
    for k in sorted(set(cand["host_phase_us"]) | set(base["host_phase_us"])):
        out[f"host_{k}"] = diff(cand["host_phase_us"].get(k), base["host_phase_us"].get(k))
    for k in sorted(set(cand["host_nested_us"]) | set(base["host_nested_us"])):
        out[f"host_nested_{k}"] = diff(cand["host_nested_us"].get(k), base["host_nested_us"].get(k))
    out["host_sibling_total"] = diff(cand.get("host_sibling_total_us"),
                                     base.get("host_sibling_total_us"))
    missing = [k for k, v in out.items() if v is None]
    if missing:
        out["not_differenced"] = missing
    return out


# Kernels the `c96e7d9..85fbda2` diff provably cannot reach: the compiled delta is
# `gqa_f16.comp` and `ops/attention.rs`, so every other pipeline is byte-identical on both arms
# and its device time is a *control* for whatever the machine was doing during the pass.
UNTOUCHED_KERNELS = ("q_gemv_matmul_nbits_f16", "skip_simplified_layer_norm_f16",
                     "simplified_layer_norm_f16", "ew_binary_mul_f16",
                     "ew_unary_sigmoid_f16", "gather_f16")


def kernel_ratios(cand: dict, base: dict) -> dict:
    """candidate/baseline per kernel, with the untouched kernels marked as what they are.

    The question this answers is the one a single kernel's delta cannot: did `gqa_f16` move
    *relative to kernels the change cannot have touched*, or did the whole device move together?
    A `gqa_f16` ratio of 1.05 next to a `q_gemv_matmul_nbits_f16` ratio of 1.12 is not evidence
    that `gqa_f16` regressed; it is evidence that the pass drifted and `gqa_f16` drifted less.
    """
    out: dict = {"per_kernel": {}, "untouched_kernels_present": []}
    for k in sorted(set(cand) | set(base)):
        c, b = cand.get(k), base.get(k)
        if not c or not b:
            out["per_kernel"][k] = None
            continue
        out["per_kernel"][k] = round(c / b, 4)
        if k in UNTOUCHED_KERNELS:
            out["untouched_kernels_present"].append(k)
    controls = [out["per_kernel"][k] for k in out["untouched_kernels_present"]
                if out["per_kernel"].get(k)]
    gqa = out["per_kernel"].get("gqa_f16")
    if controls:
        out["untouched_ratio_min"] = min(controls)
        out["untouched_ratio_max"] = max(controls)
        out["untouched_ratio_median"] = statistics.median(controls)
        if gqa is not None:
            out["gqa_f16_ratio"] = gqa
            out["gqa_outside_untouched_spread"] = not (out["untouched_ratio_min"] <= gqa
                                                       <= out["untouched_ratio_max"])
            out["reading"] = (
                "gqa_f16's ratio sits INSIDE the spread of kernels the change cannot touch, so "
                "this pass does not localise a device-time difference to gqa_f16"
                if not out["gqa_outside_untouched_spread"] else
                "gqa_f16's ratio sits OUTSIDE the spread of kernels the change cannot touch — a "
                "localisation candidate, not a conclusion, on this number of repeats")
    return out


def summarize_attribution(records: list, repeats: int) -> list:
    """Pure re-derivation of the attribution summary from the written records.

    Split out from the run so `--resummarize` can rebuild it without touching the device: an
    analysis that can only be produced by re-measuring cannot be checked by a reviewer, and two
    bugs in this file's first two passes (a kernel-key mismatch that published a false zero, and
    a `nested_in: "none"` string that emptied the host top level) were both analysis bugs that a
    re-derivation would have caught without a GPU.
    """
    summary = []
    for past in ATTRIBUTION_PAST:
        row: dict = {"past": past}
        for arm in ("candidate", "baseline"):
            rs = [r for r in records if r["past"] == past and r["arm"] == arm
                  and r.get("admissible") and (r.get("trace") or {}).get("gpu_us_per_inference")]
            if not rs:
                row[arm] = None
                continue

            def med(fn, rs=rs):
                """Median across repeats. `None` when nothing supplied it — a different state
                from zero, which must not collapse into it."""
                vals = [v for v in (fn(r["trace"]) for r in rs) if v is not None]
                return statistics.median(vals) if vals else None

            kernels = sorted({k for r in rs for k in r["trace"]["gpu_us_per_inference"]})
            phases = sorted({k for r in rs for k in r["trace"]["host_phase_us_per_inference"]})
            nested = sorted({k for r in rs for k in r["trace"]["host_nested_us_per_inference"]})
            traced_medians = [statistics.median(r["traced_samples_ms"]) for r in rs
                              if r.get("traced_samples_ms")]
            row[arm] = {
                "n": len(rs),
                "gpu_total_us": med(lambda t: t["gpu_total_us_per_inference"]),
                "gpu_us": {k: med(lambda t, k=k: t["gpu_us_per_inference"].get(k))
                           for k in kernels},
                "host_phase_us": {k: med(lambda t, k=k: t["host_phase_us_per_inference"].get(k))
                                  for k in phases},
                "host_nested_us": {k: med(lambda t, k=k: t["host_nested_us_per_inference"].get(k))
                                   for k in nested},
                "host_sibling_total_us": med(
                    lambda t: t.get("host_sibling_total_us_per_inference")),
                "traced_median_ms": (statistics.median(traced_medians)
                                     if traced_medians else None),
            }
        c, b = row.get("candidate"), row.get("baseline")
        if c and b:
            row["delta_us"] = attribution_delta(c, b)
            row["kernel_ratios"] = kernel_ratios(c["gpu_us"], b["gpu_us"])
            row["gqa_f16_per_repeat"] = per_repeat_kernel(records, past, "gqa_f16", repeats)
            row["control_kernel_per_repeat"] = per_repeat_kernel(
                records, past, "q_gemv_matmul_nbits_f16", repeats)
            for key in ("gqa_f16_per_repeat", "control_kernel_per_repeat"):
                rows = row[key]
                if rows:
                    signs = {1 if p["delta_us"] > 0 else (-1 if p["delta_us"] < 0 else 0)
                             for p in rows}
                    row[f"{key}_sign_consistent"] = len(signs) == 1
            row["per_repeat_note"] = (
                "device-clock time per inference, one traced process per repeat. Three repeats "
                "support a DIRECTION at best, never a magnitude. `control_kernel_per_repeat` is "
                "`q_gemv_matmul_nbits_f16`, which the c96e7d9..85fbda2 diff does not touch: if "
                "its direction matches gqa_f16's, the pass moved as a whole and neither "
                "direction is attributable to the change.")
            row["delta_us_note"] = (
                "`delta_us` differences the two arms' MEDIANS across repeats. It is NOT a median "
                "of paired differences and the two are not interchangeable — the paired figure "
                "is in `gqa_f16_per_repeat` and `control_kernel_per_repeat`.")
        summary.append(row)
    return summary


def per_repeat_kernel(records: list, past: int, kernel: str, repeats: int) -> list:
    out = []
    for rep in range(repeats):
        def pick(arm, rep=rep):
            return next((r for r in records
                         if r.get("past") == past and r.get("arm") == arm
                         and r.get("repeat") == rep and r.get("admissible")
                         and (r.get("trace") or {}).get("gpu_us_per_inference")), None)
        cr, br = pick("candidate"), pick("baseline")
        if not (cr and br):
            continue
        cg = cr["trace"]["gpu_us_per_inference"].get(kernel)
        bg = br["trace"]["gpu_us_per_inference"].get(kernel)
        if cg is None or bg is None:
            continue
        out.append({"repeat": rep, "kernel": kernel, "candidate_us": cg, "baseline_us": bg,
                    "delta_us": round(cg - bg, 2), "ratio": round(cg / bg, 4)})
    return out


def _attribution_main(args, libs, scratch: Path, lock: GpuLock) -> int:
    records = []
    for rep in range(args.repeats):
        arms = ("candidate", "baseline") if rep % 2 == 0 else ("baseline", "candidate")
        for past in ATTRIBUTION_PAST:
            for arm in arms:
                rec = run_one(args, rm.PHI35.key, "decode", 1, past, arm, rep, scratch,
                              traced=True)
                rec["role"] = "attribution"
                records.append(rec)
                tr = rec.get("trace") or {}
                print(f"[attrib] r{rep} past={past:5d} {arm:9s} "
                      f"gpu={tr.get('gpu_total_us_per_inference')}us "
                      f"gqa={(tr.get('gpu_us_per_inference') or {}).get('gqa_f16')}us "                      f"{'' if rec.get('admissible') else 'REFUSED ' + str((rec.get('refusal') or {}).get('reason'))}",
                      flush=True)

    summary = summarize_attribution(records, args.repeats)

    artifact = {
        "instrument": Path(__file__).name,
        "pass": "attribution",
        "issue": 96,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caveat": ("The tracer is on in every process here, so no latency in this section is a "
                   "wall-clock result and none may be quoted as one. It answers WHERE a "
                   "difference sits, never WHETHER there is one."),
        "instrument_provenance": ("ONNXRUNTIME_EP_VULKAN_TRACE + _TRACE_GPU, the EP's own tracer, "
                                  "present unchanged in both trees (the c96e7d9..85fbda2 compiled "
                                  "delta is gqa_f16.comp and ops/attention.rs only). No PR #94 "
                                  "instrumentation is used."),
        "arms": {"baseline": {**libs["baseline"], "commit": args.baseline_commit},
                 "candidate": {**libs["candidate"], "commit": args.candidate_commit}},
        "summary": summary,
        "records": [{k: v for k, v in r.items() if k != "model"} for r in records],
        "counts": {"records": len(records),
                   "admissible": sum(1 for r in records if r.get("admissible"))},
        "exclusivity": sanitize_lock(lock.record),
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
