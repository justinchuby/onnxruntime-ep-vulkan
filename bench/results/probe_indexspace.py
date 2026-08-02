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

R11 in a selection rather than in a name (2026-08-01, from a Morpheus ruling)
-----------------------------------------------------------------------------
This extract is built from a `KEYS` list, and the first version of that list took
`alloc_device_backed_spans` and `alloc_device_authoritative_spans` but **not**
`alloc_staged_spans` or `alloc_device_authoritative_ceiling` — both of which existed in the source
artifact and were simply not selected.

The consequence: `alloc_device_authoritative_spans = 0` was printed with no way to tell a
**measured** zero from a **pinned** one. The coordinator could not interpret it, and that was not
ignorance — *the artifact genuinely did not contain the information*, and a reader who had
interpreted it confidently would have been guessing. Every field printed was individually true.

**A probe can mislead through what it omits while every field it prints is correct.** R11's shape
usually shows up in a *name* that claims more than the value supports; here it showed up in a
**selection**. So the rule this file now follows:

    a counter whose value is only interpretable against a companion key is not admissible
    without that companion key on the face of the same output.

The three companions now carried, and what each one settles:

* `alloc_staged_spans` and `alloc_device_authoritative_ceiling` — the ceiling is
  `backed - staged`, so a `0` authoritative count against a `0` ceiling is **the only correct
  value**, not a pin. `alloc_device_residency_evaluations` (already present) settles the other
  half: the question was *asked*, n times, and answered no. `UNOBSERVABLE` would be a stronger and
  **false** claim — R12 is for a question that cannot be asked in this frame, and this one was.
* `alloc_allocations` — `alloc_device_backed_spans = 9` is a bare numerator. 9 of 9 spans and 9 of
  900 are different findings and the extract could not distinguish them.

The arithmetic is printed on the face of the output (`ceiling = backed - staged`) so that it does
not have to be reconstructed by a reader who already knows to look. Reporting more does not change
what is measured: the verdict criterion is untouched.

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
    "alloc_allocations",
    "alloc_staged_spans",
    "alloc_device_backed_spans",
    "alloc_device_authoritative_ceiling",
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


def span_accounting(arm: dict) -> dict:
    """Say, on the face of the output, what kind of zero `authoritative_spans` is.

    This reports; it does not judge. The verdict criterion is the two-arm index test and nothing
    here feeds it — an accounting note that could withhold `ONE_INDEX_SPACE` would be a different
    instrument wearing this one's name.

    Three states, and they are distinguished by *evidence carried in the extract* rather than by
    the reader's background knowledge:

    * ``MEASURED_ZERO_AT_A_ZERO_CEILING`` — the ceiling is `backed - staged = 0`, so every
      device-backed span still has a host staging block and is a mirror. Zero is the only correct
      value; it is not a pin, and per R12 it is also **not** `UNOBSERVABLE`.
    * ``MEASURED_ZERO_BELOW_A_NONZERO_CEILING`` — spans *could* have been authoritative and none
      was. This is the interesting one and the extract must never hide it inside the same `0`.
    * ``NOT_A_NUMBER`` — the counter is the string `UNOBSERVABLE` or `UNWIRED`; the type
      discipline has already answered and no arithmetic applies.
    """
    auth = arm.get("alloc_device_authoritative_spans")
    backed = arm.get("alloc_device_backed_spans")
    staged = arm.get("alloc_staged_spans")
    ceiling = arm.get("alloc_device_authoritative_ceiling")
    evals = arm.get("alloc_device_residency_evaluations")

    note: dict = {
        "authoritative_spans": auth,
        "ceiling": ceiling,
        "backed_minus_staged": (backed - staged) if _ints(backed, staged) else None,
        "residency_evaluations": evals,
    }
    if not isinstance(auth, int):
        note["state"] = "NOT_A_NUMBER"
        note["detail"] = (
            f"alloc_device_authoritative_spans is {auth!r}, a string state and not a count. "
            "The type is the answer; no arithmetic applies."
        )
        return note

    note["ceiling_arithmetic_holds"] = (
        _ints(ceiling, backed, staged) and ceiling == backed - staged
    )
    if _ints(evals) and evals == 0:
        note["state"] = "UNWIRED_ZERO"
        note["detail"] = (
            "nobody ever asked: alloc_device_residency_evaluations is 0, so this zero counts "
            "nothing that was screened."
        )
    elif _ints(ceiling) and ceiling == 0:
        note["state"] = "MEASURED_ZERO_AT_A_ZERO_CEILING"
        note["detail"] = (
            f"ceiling = backed {backed} - staged {staged} = 0, so every device-backed span also "
            f"has a host staging block and is a mirror. 0 is the only correct value, and "
            f"{evals} residency evaluation(s) asked the question and answered no. A measured "
            "zero, not a pin, and not UNOBSERVABLE."
        )
    else:
        note["state"] = "MEASURED_ZERO_BELOW_A_NONZERO_CEILING" if auth == 0 else "MEASURED"
        note["detail"] = (
            f"ceiling = backed {backed} - staged {staged} = {ceiling}; authoritative = {auth} "
            f"over {evals} residency evaluation(s)."
        )
    return note


def _ints(*vals: object) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in vals)


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
    report["span_accounting"] = {sel: span_accounting(report["arms"][sel]) for sel in ("0", "1")}

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
        acc = report["span_accounting"][sel]
        print(f"  selector {sel}: frame={a['alloc_device_frame']:<12s} "
              f"allocator_index={a['alloc_device_frame_allocator_index']!r:5s} "
              f"offered={a['alloc_device_frame_session_devices']!r}")
        print(f"    {'':10s} binds={a['alloc_device_buffer_binds']} "
              f"allocations={a['alloc_allocations']} "
              f"staged_spans={a['alloc_staged_spans']} "
              f"backed_spans={a['alloc_device_backed_spans']}")
        print(f"    {'':10s} ceiling={a['alloc_device_authoritative_ceiling']} "
              f"(= backed {a['alloc_device_backed_spans']} - staged {a['alloc_staged_spans']})  "
              f"authoritative_spans={a['alloc_device_authoritative_spans']} "
              f"over {a['alloc_device_residency_evaluations']} residency evaluation(s)")
        print(f"    {'':10s} -> {acc['state']}: {acc['detail']}")
        if acc.get("ceiling_arithmetic_holds") is False:
            print(f"    {'':10s} !! ceiling does not equal backed - staged; the artifact "
                  "disagrees with itself and neither figure should be quoted")
    for k, v in checks.items():
        print(f"  [{'ok ' if v else 'FAIL'}] {k}")
    print(f"verdict: {report['verdict']}")
    print(report["detail"])
    print(f"wrote {args.json}")
    return 0 if report["verdict"] == "ONE_INDEX_SPACE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
