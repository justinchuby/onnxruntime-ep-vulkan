"""The index-space falsifier: run BOTH selectors and require the allocator index to move.

Why a probe and not a code reading
-----------------------------------
`6ad67ba` changed the allocator to adopt the session's device by identity rather than by index,
and the unit tests cover the symmetry. That is not enough, and the reason is on the record: the
coordinator verified §6.5 as closed on selector 0 by reading a type transition, and Tank showed
the closure was a coincidence of two index spaces that happened to agree on this box.

    selector 0:  alloc_device_frame_allocator_index = '1'   offered = '1=NVIDIA'  -> SHARED
    selector 1:  alloc_device_frame_allocator_index = '1'   offered = '0=Intel'   -> SPLIT-DEVICE

The allocator asked for factory index 1 on **both** selectors. It did not follow the selector at
all; selector 0 merely agreed with it. So a probe that runs one selector and finds `SHARED`
establishes nothing, and R10 asks for an artifact whose *content varies with its input*.

This module runs both selectors and applies one criterion:

    the allocator's factory device index must DIFFER between the two selectors,
    and must equal the session's offered index in each.

A run in which the allocator index is the same string on both selectors is `SAME_INDEX_BOTH_ARMS`
and is a **failure**, even if both arms report `SHARED` -- because that is the pre-fix state
observed from one side. The two-arm requirement is the whole point: it is what makes the pass
unforgeable by a coincidence.

Contention
----------
Nothing here is a timing measurement, so `machine_quiescence` does not apply. These are state
observations from the counters artifact and are admissible on a contended box.

R13
---
A selector that fails to start a session, or a counters artifact that never appears, is
`ERROR(instrument)` and is never a finding about the index space.

Usage::

    python bench/results/probe_indexspace.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEC65 = HERE / "probe_sec65.py"

KEYS = (
    "alloc_device_frame",
    "alloc_device_frame_allocator_index",
    "alloc_device_frame_device",
    "alloc_device_frame_session_devices",
    "alloc_device_frame_sides",
    "alloc_device_buffer_binds",
    "alloc_device_backed_spans",
    "alloc_device_authoritative_spans",
    "alloc_device_residency_evaluations",
)


def run_selector(sel: int, out: Path) -> dict:
    """One selector, one process, three sequential sessions, device-backed allocation on."""
    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(sel)
    env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(out)
    if out.exists():
        out.unlink()
    p = subprocess.run([sys.executable, str(SEC65)], env=env, capture_output=True,
                       text=True, errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"selector {sel} exited {p.returncode}: {p.stderr.strip()[-600:]}")
    if not out.exists():
        raise RuntimeError(f"selector {sel} produced no counters artifact at {out}")
    data = json.loads(out.read_text(encoding="utf-8"))
    return {k: data.get(k, "<absent>") for k in KEYS}


def offered_index(session_devices: object) -> "str | None":
    """`'0=Intel(R) Iris(R) Xe Graphics'` -> `'0'`. The session's own index space."""
    if not isinstance(session_devices, str) or "=" not in session_devices:
        return None
    return session_devices.split("=", 1)[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=HERE / "indexspace.json")
    args = ap.parse_args()

    report: dict = {"instrument": "probe_indexspace", "arms": {}}
    try:
        for sel in (0, 1):
            report["arms"][str(sel)] = run_selector(sel, HERE / f"_indexspace_sel{sel}.json")
    except Exception as e:
        report["verdict"] = "ERROR(instrument)"
        report["reason"] = str(e)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"ERROR(instrument): {e}", file=sys.stderr)
        return 3

    a0, a1 = report["arms"]["0"], report["arms"]["1"]
    i0 = str(a0["alloc_device_frame_allocator_index"])
    i1 = str(a1["alloc_device_frame_allocator_index"])
    o0 = offered_index(a0["alloc_device_frame_session_devices"])
    o1 = offered_index(a1["alloc_device_frame_session_devices"])

    checks = {
        "allocator_index_varies_with_selector": i0 != i1,
        "allocator_follows_session_on_selector_0": i0 == o0,
        "allocator_follows_session_on_selector_1": i1 == o1,
        "frame_shared_on_selector_0": a0["alloc_device_frame"] == "SHARED",
        "frame_shared_on_selector_1": a1["alloc_device_frame"] == "SHARED",
        "binds_left_zero_on_selector_0": isinstance(a0["alloc_device_buffer_binds"], int)
        and a0["alloc_device_buffer_binds"] > 0,
        "binds_left_zero_on_selector_1": isinstance(a1["alloc_device_buffer_binds"], int)
        and a1["alloc_device_buffer_binds"] > 0,
    }
    report["checks"] = checks
    report["allocator_indices"] = {"selector_0": i0, "selector_1": i1}
    report["offered_indices"] = {"selector_0": o0, "selector_1": o1}

    if not checks["allocator_index_varies_with_selector"]:
        report["verdict"] = "SAME_INDEX_BOTH_ARMS"
        report["detail"] = (
            f"the allocator asked for factory index {i0!r} on both selectors. Whatever either arm "
            "reports for alloc_device_frame, the allocator is not following the session; a "
            "SHARED on one arm is the two spaces coinciding, which is the pre-fix state."
        )
    elif all(checks.values()):
        report["verdict"] = "ONE_INDEX_SPACE"
        report["detail"] = (
            f"the allocator's factory index moved {i0!r} -> {i1!r} as the selector moved 0 -> 1, "
            f"matching the session's offered index on each arm ({o0!r}, {o1!r}). The artifact's "
            "content varies with its input, so this is not a coincidence of two spaces agreeing "
            "on this box: swapping the GPUs swaps both indices together. Both arms SHARED, and "
            "alloc_device_buffer_binds is non-zero on both, so the engine binds its own device "
            "buffers on the same device the session runs on."
        )
    else:
        report["verdict"] = "FAILED"
        report["detail"] = "; ".join(k for k, v in checks.items() if not v)

    args.json.write_text(json.dumps(report, indent=2))
    print("== index space, both selectors ==")
    for sel in ("0", "1"):
        a = report["arms"][sel]
        print(f"  selector {sel}: frame={a['alloc_device_frame']:<12s} "
              f"allocator_index={a['alloc_device_frame_allocator_index']!r:5s} "
              f"offered={a['alloc_device_frame_session_devices']!r}")
        print(f"    {'':10s} binds={a['alloc_device_buffer_binds']} "
              f"backed_spans={a['alloc_device_backed_spans']} "
              f"authoritative_spans={a['alloc_device_authoritative_spans']}")
    for k, v in checks.items():
        print(f"  [{'ok ' if v else 'FAIL'}] {k}")
    print(f"verdict: {report['verdict']}")
    print(report["detail"])
    print(f"wrote {args.json}")
    return 0 if report["verdict"] == "ONE_INDEX_SPACE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
