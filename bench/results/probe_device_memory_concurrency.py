r"""Gate 3: do CONCURRENT sessions separate the two lanes?

The third of the four gates named at the end of Session 46n, and the only one predicted (in
`bench/results/device-memory-flip-gates-prediction.md`, written before this file ran) to be a
real blocker to flipping `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` on by default.

WHY THIS GATE IS DIFFERENT FROM THE ONE ALREADY CLOSED
------------------------------------------------------
`probe_device_memory_hazards.py`'s `two_sessions` lane builds two sessions and INTERLEAVES them
on one thread. That is a lifetime question: can a span carved by one session survive the other's
allocator release. It is not a concurrency question, and it cannot be — one thread cannot race
itself. Everything the device-memory path touches that a session does not own is process-global:

  * `HandleRegistry`  (allocator.rs) — one span arena, one free list, for the whole process
  * `tally::`          — every `alloc_*` counter, moved on alloc, on free, and on the residency
                         screen that now runs inside `free`
  * the provider map   — `ensure_registered`, one provider per device index, process-wide

A flag that ships OFF means none of that is exercised by a default run. Flipping it makes every
ORT user's second thread a caller of this code. So the question this probe asks is narrow:

    Two sessions inferring SIMULTANEOUSLY on two threads, under the resident lane.
    Do the bytes change, and do the counters still add up?

THE THREE LANES, AND WHAT EACH IS FOR
--------------------------------------
  ship_serial   `DEVICE_MEMORY` unset, two sessions, one thread, alternating.
                THE ORACLE. Every byte every other lane produces is compared against this one.
  ship_threads  `DEVICE_MEMORY` unset, two sessions, TWO THREADS.
                THE DISCRIMINATOR, and the lane the first cut of this probe did not have. Without
                it, a failure in `res_threads` is attributable to the flag by assumption rather
                than by measurement. If this lane fails the same way, concurrency is broken in the
                path that SHIPS TODAY, and gate 3 is a project defect rather than a reason to keep
                `DEVICE_MEMORY` off — the same shape as the refuted five-form story: withholding
                the thing and not withholding it must produce different outcomes, or the thing was
                never what bound.
  res_serial    `DEVICE_MEMORY=1`, two sessions, one thread, alternating.
                Isolates "the flag" from "the threads": if this lane already differs from the
                oracle, threading is not the cause and the finding is somewhere else.
  res_threads   `DEVICE_MEMORY=1`, two sessions, TWO THREADS, barrier-released so their Compute
                calls actually overlap rather than merely being issued from two threads.

`res_threads` minus `res_serial` is the concurrency effect. `res_serial` minus `ship_serial` is
the flag effect. A single armed-threaded lane compared against the shipping path would have
conflated the two, and the conflation is the exact shape of the causal story I got wrong last
round: a difference attributed to the thing I changed, when a discriminator was available and
unrun.

CORRECTNESS IS READ BEFORE ANY COUNTER
---------------------------------------
Standing rule, minted when a byte count in `probe_kv_chain_phi35.py` was measuring the wrong
tensor and only a bitwise comparison against an identical computation caught it. All 65 outputs
of both sessions at every step are SHA-256'd and compared name by name. An aggregate is not a
comparison.

THE OVERLAP IS MEASURED, NOT ASSUMED
-------------------------------------
Two threads that never happen to be inside `Compute` at the same moment prove nothing about
concurrency, and a `NO_SEPARATION` from such a run is the smoke detector that reported no fire
because it was switched off. Each worker thread increments a shared counter on entry to its
`run` and decrements on exit; the probe records the maximum concurrent depth observed. A lane
whose `max_concurrent_depth < 2` is reported `ERROR(instrument)` and its verdict is withheld.
No clock is read to establish this — it is a counter, sampled by the threads themselves.

NO CLOCK ANYWHERE
-----------------
Niobe owns timing this round. Counts, byte figures, digests, verdicts.

USAGE
-----
    $env:VULKAN_SDK="C:\VulkanSDK\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\Bin;$env:PATH"
    $env:ONNXRUNTIME_VULKAN_EP_LIB="...\rust\target\release\onnxruntime_vulkan_ep.dll"
    python bench\results\probe_device_memory_concurrency.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

# ARCHIVAL: pinned to the exact Foundry cache layout this investigation measured against
# (the pre-2026-08-05 "...-cuda-gpu/cuda-int4-rtn-block-32/..." catalog revision, issue
# #11). Intentionally NOT auto-resolved against the live cache: a live resolver could
# silently pick a *different* cached revision than the one this result was measured
# against, which would misattribute a new run to an old artifact. Override PHI35_MODEL to
# replay against a different artifact explicitly (issue #19).
ONNX_FILE = pathlib.Path(
    os.environ.get(
        "PHI35_MODEL",
        r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
        r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
        r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    )
)

# Result-identity contract (issue #19 follow-up, Morpheus review on PR #31): the resolved
# model path and its exact content hash are stamped into every output record below,
# computed lazily (only once the model has already been used successfully) so a
# PHI35_MODEL override or a stale/wrong cached file can never be silently absorbed into the
# evidence. Reuses the streaming SHA-256 helper `model_provenance.sha256_of` rather than a
# 23rd divergent hasher.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "rust" / "tools"))
import model_provenance as _model_provenance  # noqa: E402


def _result_identity() -> dict:
    return {
        "onnx_file": str(ONNX_FILE),
        "onnx_sha256": _model_provenance.sha256_of(ONNX_FILE),
    }
EP_NAME = "VulkanExecutionProvider"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
ENV_DEVICE_MEMORY = "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"

LAYERS = 32
KV_HEADS = 32
HEAD_DIM = 96
VOCAB = 32064
SEED_PAST = 4
STEPS = 2
SESSIONS = 2
ROUNDS = 1  # overridden by --rounds; a race needs repetition, not a single observation

LANES = ("ship_serial", "ship_threads", "res_serial", "res_threads")

COUNTER_KEYS = (
    "compute_calls",
    "compute_failures",
    "device_losses",
    "queue_submit_contentions",
    "dispatches_executed",
    "broken_commitments",
    "outputs_bind_attempted",
    "outputs_bind_declined",
    "outputs_device_bound",
    "alloc_allocations",
    "alloc_frees",
    "alloc_allocators_live",
    "alloc_allocators_released",
    "alloc_device_backed_spans",
    "alloc_staged_spans",
    "alloc_device_attach_attempts",
    "alloc_device_attach_failures",
    "alloc_failed_lookups",
    "alloc_frees_after_release",
    "alloc_live_at_release_spans",
    "alloc_live_at_release_bytes",
    "alloc_high_water_bytes",
    "alloc_device_frame",
    "alloc_device_frame_device",
    "alloc_device_frames_declared",
    "alloc_device_residency_evaluations",
    "session_staging_readback_bytes",
    "session_staging_upload_bytes",
)


def _lib() -> str:
    return os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )


def _counters(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc.get("counters", doc)


# --------------------------------------------------------------------------- worker


def _digest(arr) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _seed(np, salt: int):
    """One seeded KV block per session.

    The two sessions are given DIFFERENT seeds on purpose. Identical inputs would make a
    cross-session span mix-up invisible: session A reading session B's buffer would return the
    right answer. A distinct seed per session means any crossing shows up as a digest that
    belongs to the other session, which is a far more specific finding than "a number moved".
    """
    rng = np.random.default_rng(20260803 + salt)
    past = {}
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            past[f"past_key_values.{layer}.{kind}"] = (
                rng.standard_normal((1, KV_HEADS, SEED_PAST, HEAD_DIM)).astype(np.float16) * 0.02
            )
    return past


def _feeds(np, past, step, salt: int):
    past_len = int(past["past_key_values.0.key"].shape[2])
    feeds = {
        "input_ids": np.array([[1 + step + 7 * salt]], dtype=np.int64),
        "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
    }
    feeds.update(past)
    return feeds


def _make_session(ort):
    so = ort.SessionOptions()
    return ort.InferenceSession(
        str(ONNX_FILE),
        so,
        providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )


class _Depth:
    """Max observed concurrent depth inside `run`, sampled by the running threads.

    Not a clock. A counter incremented on entry and decremented on exit under one lock; the
    maximum it reaches is the number of threads that were simultaneously inside ORT's `Run`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cur = 0
        self.max = 0

    def __enter__(self):
        with self._lock:
            self.cur += 1
            self.max = max(self.max, self.cur)
        return self

    def __exit__(self, *_exc):
        with self._lock:
            self.cur -= 1
        return False


def _worker(lane: str, out_path: pathlib.Path) -> int:
    import numpy as np
    import onnxruntime as ort

    doc: dict = {
        "lane": lane,
        "ort_version": ort.__version__,
        "device_memory_env": os.environ.get(ENV_DEVICE_MEMORY),
        "kv_arena_env": os.environ.get("ONNXRUNTIME_EP_VULKAN_KV_ARENA"),
    }
    counters_path = pathlib.Path(os.environ[COUNTERS_ENV]) if COUNTERS_ENV in os.environ else None
    if counters_path is not None:
        counters_path.unlink(missing_ok=True)

    def bail(why: str) -> int:
        doc["verdict"] = "ERROR(instrument)"
        doc.setdefault("why", []).append(why)
        out_path.write_text(json.dumps({**doc, **_result_identity()}, indent=2), encoding="utf-8")
        return 2

    try:
        ort.register_execution_provider_library(EP_NAME, _lib())
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise
    ep_device = next((d for d in ort.get_ep_devices() if d.ep_name == EP_NAME), None)
    if ep_device is None:
        return bail("the Vulkan EP is not among ORT's EP devices")
    doc["ep_device"] = {
        k: ep_device.ep_metadata.get(k)
        for k in ("vulkan.device_name", "vulkan.device_index", "vulkan.vendor_id")
    }

    sessions = [_make_session(ort) for _ in range(SESSIONS)]
    for i, s in enumerate(sessions):
        if EP_NAME not in s.get_providers():
            return bail(f"session {i}: {EP_NAME} absent from {s.get_providers()}")
    names = [o.name for o in sessions[0].get_outputs()]
    doc["output_count"] = len(names)

    depth = _Depth()
    digests: dict[str, str] = {}
    argmaxes: dict[str, int] = {}
    errors: list[str] = []
    lock = threading.Lock()

    def one_session(sid: int, rnd: int) -> None:
        past = _seed(np, sid)
        try:
            for step in range(STEPS):
                feeds = _feeds(np, past, step, sid)
                with depth:
                    outs = sessions[sid].run(None, feeds)
                got = dict(zip(names, outs))
                with lock:
                    for n in names:
                        digests[f"r{rnd}/s{sid}/step{step}/{n}"] = _digest(np.asarray(got[n]))
                    flat = np.asarray(got["logits"], dtype=np.float64).reshape(-1)[-VOCAB:]
                    argmaxes[f"r{rnd}/s{sid}/step{step}"] = int(np.argmax(flat))
                past = {
                    f"past_key_values.{layer}.{kind}": np.asarray(got[f"present.{layer}.{kind}"])
                    for layer in range(LAYERS)
                    for kind in ("key", "value")
                }
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"round {rnd} session {sid}: {type(exc).__name__}: {exc}"[:600])

    rounds = int(os.environ.get("_CONC_ROUNDS", ROUNDS))
    doc["rounds"] = rounds
    for rnd in range(rounds):
        if lane.endswith("_threads"):
            # Released together so the two Compute calls actually overlap. A barrier, not a
            # sleep: nothing here reads a clock.
            gate = threading.Barrier(SESSIONS)

            def threaded(sid: int, rnd: int = rnd) -> None:
                gate.wait()
                one_session(sid, rnd)

            threads = [threading.Thread(target=threaded, args=(i,)) for i in range(SESSIONS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for sid in range(SESSIONS):
                one_session(sid, rnd)

    doc["digests"] = digests
    doc["argmaxes"] = argmaxes
    doc["errors"] = errors
    doc["max_concurrent_depth"] = depth.max
    doc["final_counters"] = {k: _counters(counters_path).get(k) for k in COUNTER_KEYS}
    doc["verdict"] = "LANE_RAN" if not errors else "LANE_RAISED"
    out_path.write_text(json.dumps({**doc, **_result_identity()}, indent=2), encoding="utf-8")
    return 0 if not errors else 3


# --------------------------------------------------------------------------- driver


def _run_lane(lane: str, rounds: int) -> dict:
    out = HERE / f"devmem_concurrency_{lane}.json"
    counters = HERE / f"devmem_concurrency_{lane}.counters.json"
    out.unlink(missing_ok=True)
    counters.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(counters)
    env["_CONC_ROUNDS"] = str(rounds)
    if lane.startswith("res_"):
        env[ENV_DEVICE_MEMORY] = "1"
    else:
        env.pop(ENV_DEVICE_MEMORY, None)
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", "--lane", lane],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    doc["exit"] = proc.returncode
    doc["stderr_tail"] = (proc.stderr or "")[-1500:]
    return doc


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--lane", default="ship_serial", choices=LANES)
    ap.add_argument("--record", default="device_memory_concurrency.json")
    ap.add_argument(
        "--rounds", type=int, default=1,
        help="rounds of (SESSIONS x STEPS) inferences per lane. A race is a RATE, not an event: "
             "one clean round is not a clearance and one dirty round is not a rate.",
    )
    args = ap.parse_args()

    if args.worker:
        return _worker(args.lane, HERE / f"devmem_concurrency_{args.lane}.json")

    lanes = {}
    for lane in LANES:
        print(f"[concurrency] {lane} x{args.rounds} ...", flush=True)
        lanes[lane] = _run_lane(lane, args.rounds)

    oracle = lanes["ship_serial"]
    report: dict = {
        "schema": 1,
        "gate": "3 of 4 — concurrent sessions",
        "ep_lib": os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
        "device": oracle.get("ep_device"),
        "lanes": lanes,
    }

    print()
    print("=" * 78)
    print("GATE 3: CONCURRENT SESSIONS UNDER THE RESIDENT LANE")
    print("=" * 78)

    findings: list[str] = []
    # ---- instrument validity, read before anything else -------------------------------------
    if oracle.get("verdict") != "LANE_RAN" or not oracle.get("digests"):
        report["verdict"] = "ERROR(instrument): the oracle lane did not produce digests"
        print("  " + report["verdict"])
        (HERE / args.record).write_text(json.dumps({**report, **_result_identity()}, indent=2), encoding="utf-8")
        return 2
    depth = lanes["res_threads"].get("max_concurrent_depth")
    if lanes["res_threads"].get("verdict") == "LANE_RAN" and (depth or 0) < SESSIONS:
        findings.append(
            f"ERROR(instrument): res_threads reached max_concurrent_depth={depth}, so its "
            f"{SESSIONS} sessions were never simultaneously inside Run. A NO_SEPARATION from "
            "this lane says nothing about concurrency."
        )
    # The queue submit lock's only falsifier. A threaded lane that never contended either was not
    # concurrent at the submit boundary or is not taking the lock; either way a clean result from
    # it is not evidence the serialization is in force.
    for ln in ("ship_threads", "res_threads"):
        c = (lanes[ln].get("final_counters") or {}).get("queue_submit_contentions")
        if c is not None and c == 0 and (lanes[ln].get("max_concurrent_depth") or 0) >= SESSIONS:
            findings.append(
                f"ERROR(instrument): {ln} ran {SESSIONS}-deep but queue_submit_contentions=0 — "
                "the queue submit lock was never contended, so this run does not witness it."
            )

    # ---- correctness, name by name, before any counter ---------------------------------------
    per_lane: dict = {}
    for lane in LANES:
        d = lanes[lane]
        if lane == "ship_serial":
            continue
        mismatched = sorted(
            k for k, v in d.get("digests", {}).items() if oracle["digests"].get(k) != v
        )
        missing = sorted(set(oracle["digests"]) - set(d.get("digests", {})))
        per_lane[lane] = {
            "outputs_compared": len(oracle["digests"]),
            "mismatched": mismatched,
            "mismatched_count": len(mismatched),
            "missing": missing,
            "argmax_agrees": d.get("argmaxes") == oracle.get("argmaxes"),
            "errors": d.get("errors", []),
            "max_concurrent_depth": d.get("max_concurrent_depth"),
        }
        # A digest that belongs to the OTHER session is a specific, nameable defect.
        crossed = []
        for k, v in d.get("digests", {}).items():
            if oracle["digests"].get(k) == v:
                continue
            twin = k.replace("s0/", "s@/").replace("s1/", "s0/").replace("s@/", "s1/")
            if oracle["digests"].get(twin) == v:
                crossed.append(k)
        per_lane[lane]["cross_session_digests"] = sorted(crossed)
        # A race is a RATE. Count the ROUNDS that were dirty, not the digests: one bad round
        # dirties 65 digests and would read as 65 failures if counted naively.
        dirty_rounds = sorted({k.split("/", 1)[0] for k in mismatched + missing})
        err_rounds = sorted({e.split()[1] for e in d.get("errors", []) if e.startswith("round ")})
        per_lane[lane]["dirty_rounds"] = dirty_rounds
        per_lane[lane]["rounds"] = d.get("rounds")
        per_lane[lane]["failure_rate"] = (
            f"{len(set(dirty_rounds) | {f'r{r}' for r in err_rounds})}/{d.get('rounds')}"
        )
    report["comparison"] = per_lane

    print(f"  oracle: ship_serial, {len(oracle['digests'])} digests "
          f"({SESSIONS} sessions x {STEPS} steps x {oracle.get('output_count')} outputs)")
    for lane, r in per_lane.items():
        print(
            f"  {lane:12s} exit={lanes[lane].get('exit')}  depth={r['max_concurrent_depth']}  "
            f"dirty_rounds={r['failure_rate']}  "
            f"mismatched={r['mismatched_count']}/{r['outputs_compared']}  "
            f"argmax_agrees={r['argmax_agrees']}  crossed={len(r['cross_session_digests'])}  "
            f"errors={len(r['errors'])}"
        )
        for e in r["errors"]:
            print(f"      RAISED: {e}")

    # ---- counters, read second ----------------------------------------------------------------
    print()
    print("  counters (lane -> value):")
    watch = (
        "compute_calls", "compute_failures", "device_losses", "dispatches_executed",
        "queue_submit_contentions",
        "broken_commitments", "outputs_device_bound", "outputs_bind_declined",
        "alloc_allocations", "alloc_frees", "alloc_failed_lookups",
        "alloc_frees_after_release", "alloc_live_at_release_spans",
        "alloc_device_frame", "alloc_device_frames_declared",
        "session_staging_readback_bytes",
    )
    for key in watch:
        cells = "  ".join(f"{ln}={lanes[ln].get('final_counters', {}).get(key)}" for ln in LANES)
        print(f"    {key:34s} {cells}")

    thr = lanes["res_threads"].get("final_counters", {}) or {}
    ser = lanes["res_serial"].get("final_counters", {}) or {}
    for key in ("alloc_failed_lookups", "alloc_frees_after_release", "compute_failures",
                "device_losses", "broken_commitments", "alloc_live_at_release_spans"):
        if (thr.get(key) or 0) > 0:
            findings.append(f"{key} = {thr[key]} in res_threads (serial lane: {ser.get(key)})")
    # Equal work: two threads must submit the same dispatches as two serial sessions.
    if thr.get("dispatches_executed") != ser.get("dispatches_executed"):
        findings.append(
            f"unequal work: res_threads dispatches_executed={thr.get('dispatches_executed')} "
            f"vs res_serial {ser.get('dispatches_executed')}"
        )
    # NOT screened: alloc_allocations vs alloc_frees. The first cut of this probe asserted they
    # must be equal and reported a "span leak" in res_threads at 1231/325 — while res_serial,
    # a lane with no finding at all, read 1296/390. Spans held by live sessions at the moment
    # the counters are dumped are not leaks, so the equality was never the invariant; my own
    # screen manufactured a defect out of a run that had a real one two lines above it. The
    # honest comparison is the RATIO against the serial lane at the same step count, and even
    # that is not decidable while a lane raises partway through. Recorded, not screened.
    report["span_accounting"] = {
        ln: {
            "alloc_allocations": (lanes[ln].get("final_counters") or {}).get("alloc_allocations"),
            "alloc_frees": (lanes[ln].get("final_counters") or {}).get("alloc_frees"),
        }
        for ln in LANES
    }

    # ---- the discriminator: is this the flag, or the project? --------------------------------
    def broke(ln: str) -> bool:
        return bool(lanes[ln].get("errors")) or bool(per_lane.get(ln, {}).get("mismatched_count"))

    ship_threads_broke = broke("ship_threads")
    res_threads_broke = broke("res_threads")
    report["discriminator"] = {
        "ship_threads_broke": ship_threads_broke,
        "res_threads_broke": res_threads_broke,
        "reading": (
            "BOTH lanes break under threads — concurrency is broken in the path that ships "
            "today, so this gate is a PROJECT defect and NOT a reason to keep DEVICE_MEMORY off"
            if ship_threads_broke and res_threads_broke
            else "only the RESIDENT lane breaks under threads — the flag is what binds, and "
            "this gate blocks the flip"
            if res_threads_broke and not ship_threads_broke
            else "only the SHIPPING lane breaks under threads — unexpected; the resident path "
            "is the safer one under concurrency"
            if ship_threads_broke and not res_threads_broke
            else "neither lane breaks under threads"
        ),
    }

    separated = any(
        per_lane[ln]["mismatched_count"] or per_lane[ln]["errors"] for ln in per_lane
    )
    print()
    if separated:
        verdict = "CONCURRENCY_SEPARATES"
    elif findings:
        verdict = "NO_BYTE_SEPARATION_WITH_COUNTER_FINDINGS"
    else:
        verdict = "NO_LANE_SEPARATES_UNDER_CONCURRENCY"
    report["verdict"] = verdict
    report["findings"] = findings
    print(f"  VERDICT: {verdict}")
    print(f"  DISCRIMINATOR: {report['discriminator']['reading']}")
    for f in findings:
        print(f"    FINDING: {f}")
    if not findings and not separated:
        print("    Every output of both sessions is byte-identical to the shipping path, the")
        print("    two threads were simultaneously inside Run, and no allocator counter moved")
        print("    into a state that only a race can produce.")

    rec = HERE / args.record
    rec.write_text(json.dumps({**report, **_result_identity()}, indent=2), encoding="utf-8")
    print(f"\n  record: {rec}")
    return 0 if verdict == "NO_LANE_SEPARATES_UNDER_CONCURRENCY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
