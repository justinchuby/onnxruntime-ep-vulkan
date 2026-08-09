"""A/A calibration for the issue #96 decode comparison: what does *no build change at all* cost?

Two cross-build runs have disagreed about Phi-3.5 decode at `past = 128`. One measured the
candidate ~14% slower; the other measured parity. Neither run could say how much of its own
number was the build, because neither ever measured the same binary against itself.

That is what this probe does, and all it does. It allocates one binary to *both* sides of a
comparison and runs the identical protocol:

* **`aa-candidate`** — the `85fbda2` library against the `85fbda2` library.
* **`aa-baseline`** — the `c96e7d9` library against the `c96e7d9` library.

Any difference either arm shows is not a build difference. It is what this desk does to two
identical things run one after the other, and it is the only defensible scale for deciding
whether the treatment ratio is a real effect. The previous revision imported that scale from a
28 ms CNN and applied it to a 137 ms LLM decode; this one measures it at the workload it grades.

**Disjointness is the point.** Every process here is its own OS process with its own PID, its own
session, and a wall-clock span that does not overlap any other. The A/A records never share a
process with the treatment records — they cannot, because the treatment records were produced by
a different run entirely and are reused byte-identically. `decode_window_evidence.calibration`
enforces that: an A/A arm sharing a PID with a treatment record produces *no band*, and with no
band every verdict is INDETERMINATE. The check is not decorative.

**What this probe does not do.** It does not re-measure the treatment. It takes no position on
whether the regression is real. It has no planted effect in it: the planted positive that proves
the detector can fire lives in the test suite as a transformation of these records, because a
planted number is not a measurement and must never be persisted as one.

Run:

    python bench/results/probe_decode_aa_calibration.py \
        --candidate-lib <path-to-85fbda2 dll> --baseline-lib <path-to-c96e7d9 dll> \
        --reuse-records bench/results/crossbuild_decode_window_records.json \
        --out bench/results/decode_window_evidence.json

    python bench/results/probe_decode_aa_calibration.py \
        --resummarize bench/results/decode_window_evidence.json --check

Exit: 0 on success, 2 on a refused summary (identity disagreement), 3 on a failed `--check`,
5 on a public-path violation, 4 if the GPU lock could not be taken.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1]
_ROOT = _BENCH.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import decode_window_evidence as dwe  # noqa: E402
import real_model as rm  # noqa: E402

EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

#: Repeats per side. Matches the two runs being reconciled, so the A/A band is measured under the
#: same repeat count whose unanimity rule it feeds. A band from 10 repeats grading a 3-repeat rule
#: would be a different protocol wearing the same name.
REPEATS = 3

#: Discarded before timing: first-touch page faults, pipeline creation, GPU clock ramp.
WARMUPS = 5

#: Timed iterations per process. The median of these is the record's `median_ms`.
ITERS = 20

#: The workload issue #96 reports. The A/A arms run here and nowhere else — a calibration taken
#: at a workload other than the one it grades is the defect that got the last revision rejected.
TREATMENT_PAST = 128

#: The A/A allocations. Both sides of each are the *same* binary; the two side names exist only
#: so the pairing code has something to put on the left and the right.
ALLOCATIONS = (
    {"name": "aa-candidate", "lib": "candidate", "left": "candidate_a", "right": "candidate_b"},
    {"name": "aa-baseline", "lib": "baseline", "left": "baseline_a", "right": "baseline_b"},
)

_ABS_PATH = re.compile(r"([A-Za-z]:[\\/]|\\\\[^\\]+\\|/home/|/Users/|/tmp/)")


# --------------------------------------------------------------------------------------------
# Public paths
# --------------------------------------------------------------------------------------------


class PublicPathError(RuntimeError):
    """An artifact tried to publish a path that is a fact about this desk, not the measurement."""


def assert_public(blob) -> None:
    """Refuse to write any absolute filesystem path into a committed artifact.

    Applied to the whole document, not to a list of fields, because the field that leaks is
    always the one nobody remembered to list.
    """
    text = json.dumps(blob)
    hit = _ABS_PATH.search(text)
    if hit:
        start = max(0, hit.start() - 40)
        raise PublicPathError(f"absolute path in artifact near: ...{text[start:hit.end() + 40]}...")


def workload_label(past: int, allocation: str) -> str:
    """The A/A workload label.

    Deliberately *not* parseable as a decode length: `decode_window_evidence.past_of` returns
    None for it, so an A/A row can never be mistaken for a measurement of KV length 128 and can
    never enter the window rule. The calibration grades; it is not graded.
    """
    return f"{rm.PHI35.key}/decode/M1/past{past}/{allocation}"


# --------------------------------------------------------------------------------------------
# GPU exclusivity
# --------------------------------------------------------------------------------------------


class GpuLock:
    """Advisory, session-local, never kills anything.

    This is a cooperation mechanism between this repository's own probes, and the artifact says
    so in those words. It is *not* device exclusivity: nothing here can stop a browser compositor
    or another user's process from using the GPU, and an artifact that called this exclusivity
    would be claiming a guarantee the operating system did not give it.
    """

    def __init__(self, path: Path, poll_seconds: float = 5.0, timeout_seconds: float = 1800.0):
        self.path = path
        self.poll = poll_seconds
        self.timeout = timeout_seconds
        self._fh = None
        self.waited = 0.0
        self.acquired_at = None
        self._t0 = None

    def __enter__(self):
        import msvcrt

        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout
        while True:
            self._fh = open(self.path, "a+b")
            try:
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                self._fh.close()
                self._fh = None
                if time.time() >= deadline:
                    raise TimeoutError(f"GPU lock still held after {self.timeout}s")
                time.sleep(self.poll)
                self.waited += self.poll
        self.acquired_at = _dt.datetime.now().isoformat(timespec="seconds")
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import msvcrt

        if self._fh is not None:
            try:
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self._fh.close()
                self._fh = None
        return False

    def record(self) -> dict:
        return {
            "mechanism": "msvcrt.locking LK_NBLCK, 1 byte at offset 0",
            "scope": "advisory, between this repository's own probes only",
            "not_a_claim": (
                "this is not device exclusivity; nothing here prevents another process on this "
                "machine from using the GPU, and concurrent GPU users are disclosed separately"
            ),
            "policy": "wait, never kill",
            "poll_seconds": self.poll,
            "waited_seconds": round(self.waited, 1),
            "acquired_at": self.acquired_at,
            "held_seconds": round(time.perf_counter() - self._t0, 1) if self._t0 else None,
            "path_note": "session-local, outside every worktree; the absolute path is a fact "
                         "about this operator's machine and is not published",
        }


# --------------------------------------------------------------------------------------------
# Concurrent GPU disclosure — gate 5
# --------------------------------------------------------------------------------------------


def gpu_tenants() -> dict:
    """Who else was on the device, by process name and count, at the moment of the snapshot.

    Names only, never command lines or user names — a command line on a developer desk is a
    private path. On a run whose central number is a drift band, what else was on the GPU is the
    first thing a reader needs and the last thing the previous revision recorded.
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process | Where-Object { $_.Path } | ForEach-Object { $_.ProcessName }) -join ','"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:  # noqa: BLE001 — disclosure must never fail the run
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if out.returncode != 0:
        return {"available": False, "reason": f"snapshot exited {out.returncode}"}
    names = sorted({n.strip() for n in out.stdout.split(",") if n.strip()})
    interesting = sorted(
        n for n in names
        if any(k in n.lower() for k in ("chrome", "msedge", "firefox", "code", "nvidia", "dwm",
                                        "teams", "obs", "blender", "python", "steam"))
    )
    return {
        "available": True,
        "total_processes_with_a_path": len(names),
        "gpu_capable_by_name": interesting,
        "note": (
            "process names only; this is a name-based screen of what was running, not a "
            "device-level enumeration of who held a Vulkan queue"
        ),
    }


# --------------------------------------------------------------------------------------------
# Worker — one process, one (allocation side, repeat)
# --------------------------------------------------------------------------------------------


def _digest_outputs(outputs, np) -> str:
    """Order-sensitive digest over every output tensor, dtype and shape included."""
    h = hashlib.sha256()
    for tensor in outputs:
        arr = np.ascontiguousarray(tensor)
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def worker(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--past", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--allocation", required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmups", type=int, default=WARMUPS)
    ap.add_argument("--iters", type=int, default=ITERS)
    args = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort

    started_at = _dt.datetime.now().isoformat(timespec="seconds")
    started_mono = time.perf_counter()

    record: dict = {
        "workload": workload_label(args.past, args.allocation),
        "arm": args.arm,
        "allocation": args.allocation,
        "role": "aa",
        "repeat": args.repeat,
        "past": args.past,
        "m": 1,
        "phase": "decode",
        "pid": os.getpid(),
        "started_at": started_at,
        "model_key": rm.PHI35.key,
    }

    lib = Path(args.lib).resolve()
    blob = lib.read_bytes()
    record["ep_library_sha256"] = hashlib.sha256(blob).hexdigest()
    record["ep_library_bytes"] = len(blob)

    try:
        resolved = rm.resolve_model(rm.PHI35)
        model_path = Path(resolved["path"])
        record["model_sha256"] = resolved["sha256"]
        record["model_bytes"] = int(resolved["bytes"])
        record["model_weights_bytes"] = int(resolved["weights_bytes"])

        case = rm.Case(rm.PHI35.key, "decode", 1, args.past, tokens=1)
        feeds = rm.phi35_feeds(case, np)
        record["feeds_sha256"] = rm.feeds_digest(feeds)

        ort.register_execution_provider_library(rm.EP_NAME, str(lib))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("ep.device_index", "0")

        t0 = time.perf_counter()
        sess = ort.InferenceSession(
            str(model_path), opts, providers=[rm.EP_NAME, "CPUExecutionProvider"]
        )
        record["session_build_ms"] = (time.perf_counter() - t0) * 1000.0
        record["providers"] = list(sess.get_providers())
        record["ort_version"] = ort.__version__

        t0 = time.perf_counter()
        first = sess.run(None, feeds)
        record["first_run_ms"] = (time.perf_counter() - t0) * 1000.0
        record["outputs_sha256"] = _digest_outputs(first, np)

        for _ in range(args.warmups - 1):
            sess.run(None, feeds)

        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            outputs = sess.run(None, feeds)
            samples.append((time.perf_counter() - t0) * 1000.0)
        record["outputs_sha256_post_timing"] = _digest_outputs(outputs, np)
        record["inference_calls"] = 1 + (args.warmups - 1) + args.iters

        # Every repeat gets its own CPU reference and its own all-output comparison. Gate 5: a
        # timing whose numerical correctness was established in some other process is a timing
        # about a different session than the one that produced it.
        cpu_opts = ort.SessionOptions()
        cpu_opts.log_severity_level = 3
        cpu_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_sess = ort.InferenceSession(
            str(model_path), cpu_opts, providers=["CPUExecutionProvider"]
        )
        t0 = time.perf_counter()
        reference = cpu_sess.run(None, feeds)
        record["cpu_reference_ms"] = (time.perf_counter() - t0) * 1000.0
        record["equivalence"] = rm.classify_outputs(case, outputs, reference, np)

        stats = rm.latency_stats(samples)
        stats["samples_ms"] = [round(s, 4) for s in samples]
        record["speed"] = stats

        counters_path = os.environ.get(COUNTERS_ENV)
        witness = {"present": False}
        if counters_path and Path(counters_path).exists():
            try:
                counters = json.loads(Path(counters_path).read_text(encoding="utf-8"))
                variants = counters.get("pipeline_variants") or []
                witness = {
                    "present": True,
                    "gqa_keys": sorted(v for v in variants if v.startswith("gqa_")),
                    "all_variants": sorted(variants),
                    "dispatches_executed": counters.get("dispatches_executed"),
                    "compute_calls": counters.get("compute_calls"),
                    "compute_failures": counters.get("compute_failures"),
                    "running_device_names": counters.get("running_device_names"),
                }
            except Exception as exc:  # noqa: BLE001
                witness = {"present": False, "reason": f"{type(exc).__name__}: {exc}"}
        record["path_witness"] = witness

        equivalent = record["equivalence"].get("verdict") == rm.MATCH
        record["admissible"] = bool(equivalent)
        if not equivalent:
            # Structural, not cosmetic: a record that failed equivalence loses its timing
            # outright so that nothing downstream can read a speed off it by accident.
            record["refusal"] = {"reason": "EP-vs-CPU equivalence is not MATCH"}
            record.pop("speed", None)
    except Exception as exc:  # noqa: BLE001
        record["admissible"] = False
        record["refusal"] = {"reason": f"{type(exc).__name__}: {exc}"}
        record.pop("speed", None)

    record["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    record["worker_wall_s"] = round(time.perf_counter() - started_mono, 3)
    record["started_monotonic"] = started_mono
    record["finished_monotonic"] = time.perf_counter()
    Path(args.out).write_text(json.dumps(record), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def _child_env(counters_path: Path) -> dict:
    """A clean environment for every child.

    Every inherited `ONNXRUNTIME_EP_VULKAN_*` is stripped. A stray switch left in the operator's
    shell is a treatment nobody recorded, and both arms of an A/A would inherit it, which is
    exactly the case where it would be invisible.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("ONNXRUNTIME_EP_VULKAN_")}
    env[COUNTERS_ENV] = str(counters_path)
    return env


def _run_side(args, allocation: dict, side_arm: str, lib: Path, repeat: int, scratch: Path) -> dict:
    tag = f"{allocation['name']}-{side_arm}-r{repeat}"
    out = scratch / f"rec-{tag}.json"
    counters = scratch / f"counters-{tag}.json"
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--past", str(args.past), "--arm", side_arm, "--allocation", allocation["name"],
        "--repeat", str(repeat), "--lib", str(lib), "--out", str(out),
        "--warmups", str(args.warmups), "--iters", str(args.iters),
    ]
    proc = subprocess.run(cmd, env=_child_env(counters), capture_output=True, text=True)
    if not out.exists():
        return {
            "workload": workload_label(args.past, allocation["name"]),
            "arm": side_arm, "allocation": allocation["name"], "role": "aa", "repeat": repeat,
            "admissible": False,
            "refusal": {"reason": f"worker exited {proc.returncode} without writing a record"},
            "worker_exit": proc.returncode,
            "worker_stderr_tail": proc.stderr[-400:],
        }
    record = json.loads(out.read_text(encoding="utf-8"))
    record["worker_exit"] = proc.returncode
    return record


def _sweep(args, libs: dict, scratch: Path) -> list:
    records: list = []
    for repeat in range(args.repeats):
        for allocation in ALLOCATIONS:
            lib = libs[allocation["lib"]]
            sides = [allocation["left"], allocation["right"]]
            # Flip which side goes first on odd repeats. In an A/A the two sides are the same
            # binary, so any systematic first-vs-second effect is pure order, and alternating is
            # what stops it from being read as a difference between the sides.
            if repeat % 2:
                sides.reverse()
            for side_arm in sides:
                print(f"  {allocation['name']} {side_arm} repeat {repeat} ...", flush=True)
                record = _run_side(args, allocation, side_arm, lib, repeat, scratch)
                status = dwe.classify_record(record)["status"]
                speed = record.get("speed", {}).get("median_ms")
                print(f"    -> {status}"
                      + (f" median {speed:.2f} ms" if speed else "")
                      + (f" ({record.get('refusal', {}).get('reason')})" if status == dwe.REFUSED else ""),
                      flush=True)
                records.append(record)
    return records


def _library_identity(path: Path, commit: str) -> dict:
    blob = path.read_bytes()
    return {
        "path_basename": path.name,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "commit": commit,
    }


def _resummarize(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resummarize", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help="rewrite the artifact's summary block from its own records; the records "
                         "are never touched, so this can correct a summarizer without re-measuring")
    args = ap.parse_args(argv)

    path = Path(args.resummarize)
    artifact = json.loads(path.read_text(encoding="utf-8"))

    if args.write:
        artifact["summary"] = dwe.summarize(
            artifact.get("records", []),
            arms=artifact.get("arms", {}),
            repeats_required=artifact.get("environment", {}).get("repeats", REPEATS),
            aa_allocations=artifact.get("aa_allocations", ()),
            reference_effect=artifact.get("reference_effect"),
        )
        artifact["summary_written_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        assert_public(artifact)
        path.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
        print(f"rewrote the summary of {path.name} from its own {len(artifact.get('records', []))} records")

    result = dwe.check(artifact)
    print(f"reproduces={'YES' if result['reproduces'] else 'NO'}")
    for difference in result["differences"][:40]:
        print(f"  {difference}")
    if len(result["differences"]) > 40:
        print(f"  ... and {len(result['differences']) - 40} more")
    if args.check and not result["reproduces"]:
        return 3
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        argv.remove("--worker")
        return worker(argv)
    if "--resummarize" in argv:
        return _resummarize(argv)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-lib", required=True)
    ap.add_argument("--baseline-lib", required=True)
    ap.add_argument("--candidate-commit", default="85fbda29a92e0e99c3895be8b13664d4ee670c50")
    ap.add_argument("--baseline-commit", default="c96e7d94ff706d26ee6a1bd9bb084c0ade426820")
    ap.add_argument("--reuse-records", required=True,
                    help="JSON artifact whose records are reused byte-identically as the "
                         "treatment arm; this probe never re-times the treatment")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--past", type=int, default=TREATMENT_PAST)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--warmups", type=int, default=WARMUPS)
    ap.add_argument("--iters", type=int, default=ITERS)
    args = ap.parse_args(argv)

    libs = {"candidate": Path(args.candidate_lib).resolve(),
            "baseline": Path(args.baseline_lib).resolve()}
    arms = {
        "candidate": _library_identity(libs["candidate"], args.candidate_commit),
        "baseline": _library_identity(libs["baseline"], args.baseline_commit),
    }
    for allocation in ALLOCATIONS:
        source = arms[allocation["lib"]]
        for side in (allocation["left"], allocation["right"]):
            arms[side] = dict(source, allocated_as=allocation["name"], same_binary_as=allocation["lib"])

    reused_doc = json.loads(Path(args.reuse_records).read_text(encoding="utf-8"))
    reused = reused_doc["records"] if isinstance(reused_doc, dict) else reused_doc

    scratch = Path(args.scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    started_at = _dt.datetime.now().isoformat(timespec="seconds")
    tenants_before = gpu_tenants()

    try:
        with GpuLock(Path(args.lock).resolve()) as lock:
            print(f"A/A calibration at past={args.past}, {args.repeats} repeats", flush=True)
            fresh = _sweep(args, libs, scratch)
            exclusivity = lock.record()
    except TimeoutError as exc:
        print(f"could not take the GPU lock: {exc}", file=sys.stderr)
        return 4

    records = list(reused) + fresh
    aa_allocations = [
        {"workload": workload_label(args.past, a["name"]), "left": a["left"], "right": a["right"]}
        for a in ALLOCATIONS
    ]
    summary = dwe.summarize(
        records, arms=arms, repeats_required=args.repeats,
        aa_allocations=aa_allocations, reference_effect=0.859,
    )

    artifact = {
        "schema": dwe.SCHEMA,
        "instrument": Path(__file__).name,
        "issue": dwe.ISSUE,
        "question": (
            "How much does this desk move an identical binary against itself at Phi-3.5 decode "
            "past=128, and does the issue #96 comparison exceed that?"
        ),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "arms": arms,
        "aa_allocations": aa_allocations,
        "reference_effect": 0.859,
        "reference_effect_note": (
            "the ratio previously reported for this workload; carried so the summary can state "
            "how often this protocol would have detected it, not as a target"
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "onnxruntime": _ort_version(),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "iters": args.iters,
            "one_process_per": "(allocation, side, repeat)",
            "side_order_rule": "left side first on even repeats, right side first on odd",
            "env_hygiene": "every inherited ONNXRUNTIME_EP_VULKAN_* stripped from every child; "
                           "only the counters-file path is set by this driver",
            "assumption": "stock Windows power plan, no CPU affinity mask, no GPU clock lock",
        },
        "device": _device_identity(fresh),
        "concurrent_gpu_users": tenants_before,
        "exclusivity": exclusivity,
        "records_provenance": {
            "treatment": {
                "reused_from": Path(args.reuse_records).name,
                "count": len(reused),
                "reused_byte_identically": True,
                "note": "not re-timed by this probe; independently revalidated before reuse",
            },
            "calibration": {"measured_here": len(fresh)},
        },
        "summary": summary,
        "records": records,
    }
    assert_public(artifact)
    Path(args.out).write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {summary['counts']}")
    print(f"band: {summary['band']['band']} ({summary['band'].get('reason') or summary['band']['rule']})")
    print(f"window: {summary['window']['claim']}")
    return 2 if summary["refuses"] else 0


def _ort_version() -> "str | None":
    try:
        import onnxruntime

        return onnxruntime.__version__
    except Exception:  # noqa: BLE001
        return None


def _device_identity(records) -> dict:
    """Device and driver, read off the records the run itself produced."""
    names = sorted({
        r.get("path_witness", {}).get("running_device_names")
        for r in records
        if isinstance(r.get("path_witness"), dict) and r["path_witness"].get("running_device_names")
    })
    return {
        "running_device_names": names,
        "source": "reported by the EP in each measuring process, not typed into this file",
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublicPathError as exc:
        print(f"refusing to write artifact: {exc}", file=sys.stderr)
        raise SystemExit(5) from exc
