r"""Does arming the device-memory provider remove the host round-trip for KV?

THE QUESTION, AND WHY IT IS NOT A CODE READING
==============================================
Niobe measured, on counters alone, that the present KV cache is copied device->host
in full every inference: **393,216 B per past token, ratio 1.000000 on both
segments, linearity spread 0.000000** (`kv_bytes_earned.json`). Past 2048 tokens the
implied crossover rates exceed any link that physically exists on this box, so at
real context lengths the inference is bound by host<->device KV transfer.

The coordinator's hypothesis: `past_key_values.*` and `present.*` are graph inputs
and outputs, ORT owns those allocations, and unless the EP offers a device-memory
allocator for them they live in host memory and every token pays a round-trip --
which is `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY`, whose default is OFF.

If that is right, arming the flag moves the readback slope. If it is wrong, the
slope does not move and the round-trip is ORT's contract rather than our
configuration.

WHY THIS PROBE DOES NOT COMPUTE THE SLOPE ITSELF
================================================
It runs **Niobe's** `probe_kv_bytes_earned.py` unchanged, once per lane, and
compares the two records it produces. The second-difference arithmetic that cancels
the one-time weight upload is hers and is not re-implemented here: a falsifier that
depends on my arithmetic to disagree with her number is not independent of me. What
this file adds is the lane axis and the frame disclosure, nothing else.

WHAT IT RECORDS BESIDES THE SLOPE (the §6.5 caution)
====================================================
`alloc_device_frame` reads OFF in the default lane and SHARED only when armed. Any
number here is meaningless without the lane it was taken in, so every lane record
carries `alloc_device_frame`, `alloc_device_frame_device`,
`alloc_device_buffer_binds`, `alloc_device_authoritative_spans` and
`alloc_device_residency_evaluations` read straight out of the per-point counters
files Niobe's driver leaves in `bench/_scratch`.

`alloc_device_buffer_binds` is the load-bearing one. It counts the engine binding
one of the allocator's device buffers instead of staging its own copy. If it is 0
in the armed lane then no allocation this EP served was ever bound by a dispatch,
and the round-trip cannot have been removed no matter what the allocator did.

Counts only. No clock anywhere in this file; contention cannot touch a byte count.

Usage::

    $env:VULKAN_SDK="C:\VulkanSDK\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\Bin;$env:PATH"
    $env:ONNXRUNTIME_VULKAN_EP_LIB="...\rust\target\release\onnxruntime_vulkan_ep.dll"
    python bench/results/probe_device_memory_kv.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
NIOBE_PROBE = HERE / "probe_kv_bytes_earned.py"
SCRATCH = BENCH / "_scratch"

ENV_DEVICE_MEMORY = "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"

FRAME_KEYS = (
    "alloc_device_frame",
    "alloc_device_frame_device",
    "alloc_device_frame_allocator_index",
    "alloc_device_frame_session_devices",
    "alloc_device_buffer_binds",
    "alloc_device_authoritative_spans",
    "alloc_device_residency_evaluations",
    "alloc_device_backed_spans",
    "alloc_staged_spans",
    "alloc_allocations",
)

LANES = (
    ("default", None),
    ("armed", "1"),
)


def dll_sha256() -> dict:
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib or not Path(lib).is_file():
        raise SystemExit(
            "ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB is unset or missing. Both lanes must "
            "run the same binary and the record must say which one."
        )
    h = hashlib.sha256(Path(lib).read_bytes()).hexdigest()
    return {"path": lib, "sha256": h}


def run_lane(name: str, value: str | None, keep: Path) -> dict:
    """Run Niobe's probe once with the flag set as *value*; return her record + frame keys."""
    for stale in SCRATCH.glob("kvbytes_p*_n*.counters.json"):
        stale.unlink()
    keep.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.pop(ENV_DEVICE_MEMORY, None)
    if value is not None:
        env[ENV_DEVICE_MEMORY] = value

    out = keep / f"kv_bytes_earned-{name}.json"
    cmd = [sys.executable, str(NIOBE_PROBE), "--out", str(out)]
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True)
    if proc.returncode != 0 or not out.is_file():
        tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
        raise SystemExit(
            f"ERROR(instrument): lane {name!r} exited {proc.returncode} and produced "
            f"{'no' if not out.is_file() else 'a'} record. The run happened; the observation "
            f"did not. stderr tail:\n{tail}"
        )

    record = json.loads(out.read_text(encoding="utf-8"))

    frames = []
    for cfile in sorted(SCRATCH.glob("kvbytes_p*_n*.counters.json")):
        c = json.loads(cfile.read_text(encoding="utf-8"))
        c = c.get("counters", c)
        frames.append({"point": cfile.stem, **{k: c.get(k, "<absent>") for k in FRAME_KEYS}})
        shutil.copy2(cfile, keep / f"{name}-{cfile.name}")

    return {"lane": name, ENV_DEVICE_MEMORY: value or "<unset>", "record": record, "frames": frames}


def segments_of(lane: dict) -> list[dict]:
    return lane["record"].get("segments", [])


def point_validity(record: dict) -> tuple[dict[int, bool], list[dict]]:
    """Which context points are admissible, and why the others are not.

    A point is admissible only if the counters describe a COMPLETE run of the loop the
    driver asked for: one EP compute call per iteration, and one readback per upload. A
    snapshot taken with an upload outstanding (`uploads == readbacks + 1`) is an inference
    in flight, so the byte totals are a prefix of the run and differencing them measures
    where the observation stopped rather than what the run did.

    This is R13, not fastidiousness: a truncated snapshot is `ERROR(instrument)` and an
    instrument error is never a detection. Averaging it in would have produced a slope
    that looked like a KV saving and was an observation ending early.
    """
    ok: dict[int, bool] = {}
    notes: list[dict] = []
    for p in record.get("points", []):
        complete = (
            p.get("compute_calls") == p["iters"]
            and p.get("session_staging_uploads") == p["iters"]
            and p.get("session_staging_readbacks") == p["iters"]
        )
        past = p["past_len"]
        ok[past] = ok.get(past, True) and complete
        if not complete:
            notes.append({
                "past_len": past,
                "iters": p["iters"],
                "compute_calls": p.get("compute_calls"),
                "session_staging_uploads": p.get("session_staging_uploads"),
                "session_staging_readbacks": p.get("session_staging_readbacks"),
                "classification": "ERROR(instrument): truncated snapshot, not a measurement",
            })
    return ok, notes


def admissible_segments(lane: dict) -> tuple[list[dict], list[dict]]:
    """Segments both of whose endpoints ran to completion, plus the rejection notes."""
    ok, notes = point_validity(lane["record"])
    good = [
        s for s in segments_of(lane)
        if ok.get(s["from_past_len"], False) and ok.get(s["to_past_len"], False)
    ]
    return good, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "device_memory_kv_lanes.json")
    ap.add_argument("--keep", type=Path, default=HERE / "device_memory_kv_lanes")
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="re-derive the verdict from the kept lane records without running anything; "
        "the analysis is the instrument under change, not the run",
    )
    args = ap.parse_args()

    if not NIOBE_PROBE.is_file():
        raise SystemExit(f"ERROR(instrument): {NIOBE_PROBE} is absent; nothing to run.")

    if args.reuse:
        dll = {"path": "<reused>", "sha256": "<reused>"}
        lanes = []
        for name, value in LANES:
            rec = args.keep / f"kv_bytes_earned-{name}.json"
            if not rec.is_file():
                raise SystemExit(f"ERROR(instrument): --reuse but {rec} is absent.")
            frames = []
            for cfile in sorted(args.keep.glob(f"{name}-kvbytes_p*_n*.counters.json")):
                c = json.loads(cfile.read_text(encoding="utf-8"))
                c = c.get("counters", c)
                frames.append(
                    {"point": cfile.stem, **{k: c.get(k, "<absent>") for k in FRAME_KEYS}}
                )
            lanes.append({
                "lane": name,
                ENV_DEVICE_MEMORY: value or "<unset>",
                "record": json.loads(rec.read_text(encoding="utf-8")),
                "frames": frames,
            })
    else:
        dll = dll_sha256()
        lanes = []
        for name, value in LANES:
            print(
                f"[devmem-kv] lane={name} {ENV_DEVICE_MEMORY}={value or '<unset>'} ...",
                flush=True,
            )
            lanes.append(run_lane(name, value, args.keep))

        dll_after = dll_sha256()
        if dll_after["sha256"] != dll["sha256"]:
            raise SystemExit(
                "ERROR(instrument): the DLL changed between lanes; the comparison is between "
                "two different binaries and says nothing about the flag."
            )

    by_lane = {}
    rejected = {}
    for lane in lanes:
        segs, notes = admissible_segments(lane)
        rejected[lane["lane"]] = notes
        by_lane[lane["lane"]] = {
            ENV_DEVICE_MEMORY: lane[ENV_DEVICE_MEMORY],
            "admissible_segments": [(s["from_past_len"], s["to_past_len"]) for s in segs],
            "readback_bytes_per_past_token": [s["readback_bytes_per_past_token"] for s in segs],
            "readback_ratio": [s["readback_ratio"] for s in segs],
            "upload_bytes_per_past_token": [s["upload_bytes_per_past_token"] for s in segs],
            "upload_bytes_per_inference": [
                c["upload_bytes_per_inference"] for c in lane["record"]["by_context"]
            ],
            "readback_bytes_per_inference": [
                c["readback_bytes_per_inference"] for c in lane["record"]["by_context"]
            ],
            "session_staging_upload_bytes_total": [
                p["session_staging_upload_bytes"] for p in lane["record"]["points"]
            ],
            "upload_state": lane["record"].get("upload", {}).get("state", "<absent>"),
            "frames": lane["frames"],
        }

    a = by_lane.get("default", {})
    b = by_lane.get("armed", {})
    shared = [s for s in a.get("admissible_segments", []) if s in b.get("admissible_segments", [])]
    moved = None
    if shared:
        ai = {tuple(s): v for s, v in zip(a["admissible_segments"], a["readback_bytes_per_past_token"])}
        bi = {tuple(s): v for s, v in zip(b["admissible_segments"], b["readback_bytes_per_past_token"])}
        moved = [bi[tuple(s)] - ai[tuple(s)] for s in shared]

    binds = sorted(
        {f["alloc_device_buffer_binds"] for f in by_lane.get("armed", {}).get("frames", [])}
    )
    frames_seen = sorted(
        {str(f["alloc_device_frame"]) for f in by_lane.get("armed", {}).get("frames", [])}
    )

    verdict = "UNDECIDED"
    why = ""
    if moved is not None and shared:
        if all(abs(d) < 1.0 for d in moved):
            verdict = "UNCHANGED"
            why = (
                "On every segment admissible in BOTH lanes the readback slope is byte-identical "
                "with the device-memory provider armed. Arming the allocator does NOT remove the "
                "host round-trip for present/past KV. The mechanism agrees: "
                "vk::host_device_memory::bind_target_for is called on INPUTS only "
                "(vk/session.rs step 1a); the readback is an unconditional sum over "
                "actual_output_byte_sizes, with no output-side bind to decline it."
            )
        else:
            verdict = "MOVED"
            why = (
                "The readback slope differs between lanes on a segment admissible in both, so the "
                "flag is load-bearing for KV residency."
            )
    elif not shared:
        verdict = "ERROR(instrument)"
        why = (
            "No context segment ran to completion in both lanes, so there is nothing to compare. "
            "The runs happened; the observations did not."
        )

    doc = {
        "kind": "device_memory_kv_lanes",
        "question": "does arming ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY remove the host round-trip "
        "for past_key_values/present?",
        "arithmetic_owner": "bench/results/probe_kv_bytes_earned.py (Niobe) -- this file runs it "
        "once per lane and compares; it does not recompute the slope",
        "dll": dll,
        "device": os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "<unset>"),
        "predicted_bytes_per_past_token": 393216,
        "lanes": by_lane,
        "rejected_points": rejected,
        "admissible_segments_in_both_lanes": shared,
        "readback_slope_delta_armed_minus_default": moved,
        "armed_lane_alloc_device_buffer_binds": binds,
        "armed_lane_alloc_device_frame": frames_seen,
        "verdict": verdict,
        "why": why,
        "no_clock": "counts only; no duration is quoted anywhere in this record",
    }
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in (
        "verdict", "readback_slope_delta_armed_minus_default",
        "armed_lane_alloc_device_buffer_binds", "armed_lane_alloc_device_frame")}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
