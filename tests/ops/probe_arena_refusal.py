"""Which of the five things that can unbind an aliased output is unbinding it on Intel.

THE HOLE THIS FILLS
===================
`bench/results/gqa_grouping_arena_dev1.json`: the KV-arena lane is refused on the Intel
Iris Xe, **identically at G=1 and G=4**, so it is not a grouping finding — it is the one
device-shaped hole in the grouping result. The refusal text names a symptom (`present` is
an alias of `past` but is not bound to that input's device buffer) and not a cause.

`rust/src/vk/session.rs`'s sweep fires when `bound_outputs[i] != bound_inputs[in_idx]`,
and **five different upstream failures reach it**:

  A. `ort_output_ptr` returned None                    -- ORT gave no output pointer
  B. `bind_target_for(out_ptr)` returned None          -- the output is not our device span
  C. `bound_inputs[in_idx]` is None                    -- the INPUT was never device-bound
  D. both bound, different VkBuffers                   -- ORT did not honour the alias
  E. `mark_device_authoritative` refused               -- bound, then unbound again

A, B and E are output-side; C is input-side; D means the alias declaration was a fiction.
**They have completely different owners**, and the record currently distinguishes none of
them, which is why this has sat open. This probe separates them from counters the run
emits, so whoever picks it up starts from a cause.

HOW IT SEPARATES THEM WITHOUT NEW EP CODE
=========================================
`outputs_bind_attempted` / `outputs_bind_declined` / `outputs_device_bound` bracket the
output side, and `alloc_device_attach_attempts` / `alloc_device_attach_failures` /
`alloc_device_attach_unavailable` / `alloc_failed_lookups` bracket the span lookups that
back B and C. A control arm on the SAME process shape that is known to bind (the growing
lane, which is GREEN on this device) supplies the contrast: a counter that reads the same
in both arms did not cause the difference.

**Read this as a narrowing, not as a diagnosis.** If the counters cannot separate the
five, the probe says so and names the one-line EP change that would -- that is a better
hand-off than a guess.

USAGE
=====
    $env:ONNXRUNTIME_VULKAN_EP_LIB = "rust/target/release/onnxruntime_vulkan_ep.dll"
    $env:ONNXRUNTIME_EP_VULKAN_DEVICE = "1"
    $env:ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY = "1"
    $env:ONNXRUNTIME_EP_VULKAN_KV_ARENA = "1"
    python tests/ops/probe_arena_refusal.py

    python tests/ops/probe_arena_refusal.py --selftest    # no GPU
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RESULTS_DIR = _REPO_ROOT / "bench" / "results"

#: The counters that bracket each candidate cause.  Named here so the mapping from a
#: reading to a cause is data a reader can check, not a paragraph they must trust.
CAUSE_EVIDENCE = {
    "A_no_ort_output_ptr": ["outputs_bind_attempted", "outputs_bind_declined"],
    "B_output_not_our_span": ["alloc_failed_lookups", "alloc_device_attach_failures"],
    "C_input_never_device_bound": [
        "alloc_device_authority_grants",
        "alloc_device_backed_spans",
        "alloc_device_attach_unavailable",
    ],
    "D_alias_not_honoured": ["outputs_device_bound", "alloc_device_buffer_binds"],
    "E_authority_refused": ["alloc_device_authoritative_spans", "outputs_bind_declined"],
    # Found by measurement, not enumerated in advance.  Measured 2026-08-04 on the Iris Xe
    # as the whole of the "Intel refuses the arena" finding.  NOT a split VkDevice:
    # `alloc_device_frame` reads SHARED in BOTH polarities.  What differs is the index
    # ORT keys its allocator to.
    "F_allocator_index_mismatch": [
        "alloc_device_frame",
        "alloc_device_frame_allocator_index",
        "alloc_device_frame_session_devices",
        "outputs_bind_declined",
        "alloc_device_authority_grants",
    ],
}


def _session_device_index(sides: str) -> str | None:
    """The index the session's own device list gives the device it ran on.

    `alloc_device_frame_session_devices` is "0=Intel(R) Iris(R) Xe Graphics; 1=NVIDIA..."
    -- when the factory advertises one device that list has one entry and its index is the
    only one the session can mean.
    """
    entries = [e.strip() for e in str(sides or "").split(";") if "=" in e]
    if len(entries) != 1:
        return None
    return entries[0].split("=", 1)[0].strip()


def classify(arena: dict, control: dict, arena_verdict: str | None = None) -> dict:
    """Narrow the five causes from two counter readings.  Never asserts one alone."""
    def g(d: dict, k: str, default=0):
        v = d.get(k)
        return default if v is None else v

    out: dict = {"ruled_out": [], "consistent_with": [], "notes": []}

    # A classification of a run that did not refuse is a classification of nothing.  This
    # arm exists because the first real run of this probe came back GREEN and the
    # classifier happily named a cause for it.
    if arena_verdict is not None and "refused" not in str(arena_verdict):
        out["consistent_with"] = ["NOT_REFUSED"]
        out["separated"] = True
        out["notes"].append(
            f"the arena arm returned {arena_verdict!r}, so there is no refusal to "
            "attribute; the five causes are not in play"
        )
        return out

    # F first, because it explains the refusal without any of A-E being a device property.
    # The first version of this classifier looked for `alloc_device_frame = SPLIT-DEVICE`
    # here, on the strength of the EP's own §6.5 warning predicting one.  The paired,
    # one-arm-per-process reading says the frame is SHARED in BOTH polarities -- the EP
    # honours the explicit device request, so only one VkDevice ever runs.  What actually
    # differs is the index ORT keys its allocator to, and the SPLIT-DEVICE label would
    # have been a wrong cause attached to a right headline.
    alloc_idx = str(g(arena, "alloc_device_frame_allocator_index", "") or "")
    sess_idx = _session_device_index(g(arena, "alloc_device_frame_session_devices", ""))
    declined = g(arena, "outputs_bind_declined")
    if alloc_idx and sess_idx is not None and alloc_idx != sess_idx and declined > 0:
        out["consistent_with"] = ["F_allocator_index_mismatch"]
        out["separated"] = True
        out["notes"].append(
            f"alloc_device_frame_allocator_index={alloc_idx!r} but the session's own "
            f"device list puts the device it ran on at index {sess_idx!r}, and "
            f"{declined} output bind(s) were declined. ORT keys its allocator to the "
            "device IT bound; device-memory OrtValues then carry an allocator identity "
            "the session's spans are not registered under, `bind_target_for` cannot "
            "resolve the `present` pointers, and the alias sweep refuses -- correctly. "
            "Set ONNXRUNTIME_EP_VULKAN_DEVICE before the EP library is registered so the "
            "factory advertises only that device. This is harness configuration, not a "
            "device capability, and `alloc_device_frame` is SHARED either way."
        )
        return out

    attempted = g(arena, "outputs_bind_attempted")
    bound = g(arena, "outputs_device_bound")

    if attempted == 0:
        out["consistent_with"].append("BIND_OUTPUTS path never ran (not one of A-E)")
        out["notes"].append(
            "outputs_bind_attempted == 0: the whole bind block was skipped, so the sweep "
            "refused an output that was never offered a bind. That is the "
            "`BIND_OUTPUTS=0` arm named in the session.rs comment, not a device story."
        )
        return out

    out["notes"].append(
        f"bind attempted={attempted} declined={declined} bound={bound}"
    )
    if bound > 0 and declined == 0:
        out["ruled_out"].append("A_no_ort_output_ptr")
        out["notes"].append(
            "every output offered a bind took one, so ORT hands out pointers and "
            "bind_target_for resolves them; an output-side failure is refuted"
        )
    elif declined > 0:
        out["consistent_with"].append("A_no_ort_output_ptr")
        out["notes"].append(
            f"{declined} of {attempted} output binds were DECLINED. `bound > 0` does not "
            "refute A -- attn_out binding while present_key/present_value are declined is "
            "exactly A for the two outputs the alias is about. The first version of this "
            "classifier ruled A out on `bound > 0` and would have mis-attributed the "
            "measured Intel refusal to D."
        )
    if g(arena, "alloc_failed_lookups") > g(control, "alloc_failed_lookups"):
        out["consistent_with"].append("B_output_not_our_span")
    else:
        out["ruled_out"].append("B_output_not_our_span")

    if g(arena, "alloc_device_attach_unavailable") > 0:
        out["consistent_with"].append("C_input_never_device_bound")
    if g(arena, "alloc_device_attach_failures") > 0:
        out["consistent_with"].append("C_input_never_device_bound")

    if g(arena, "alloc_device_authority_grants") == 0 and attempted > 0:
        out["consistent_with"].append("E_authority_refused")
    else:
        out["ruled_out"].append("E_authority_refused")

    if not out["consistent_with"]:
        out["consistent_with"].append("D_alias_not_honoured")
        out["notes"].append(
            "every other cause is ruled out by a counter, which leaves ORT handing the "
            "EP a different VkBuffer for `present` than for `past` despite the alias "
            "declaration -- the UMA/bind_target_for identity question"
        )
    out["consistent_with"] = sorted(set(out["consistent_with"]))
    out["ruled_out"] = sorted(set(out["ruled_out"]) - set(out["consistent_with"]))
    out["separated"] = len(out["consistent_with"]) == 1
    if not out["separated"]:
        out["hand_off"] = (
            "the counters do not separate these. The one-line EP change that would: in "
            "rust/src/vk/session.rs the sweep at `if !same` already knows which of "
            "bound_outputs[i] / bound_inputs[in_idx] is None -- record that discriminant "
            "in the bail! message (out=None / in=None / both-Some-differ) and the five "
            "collapse to one reading."
        )
    return out


def _read_counters(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("counters", d)


def _selftest() -> int:
    # A: bind path never ran
    r = classify({"outputs_bind_attempted": 0}, {}, "UNMEASURED(arena_refused)")
    assert "BIND_OUTPUTS path never ran (not one of A-E)" in r["consistent_with"], r

    # A green arena must classify NOTHING
    rg = classify({"outputs_bind_attempted": 3}, {}, "AGREE")
    assert rg["consistent_with"] == ["NOT_REFUSED"], rg

    # F: the measured Intel reading, verbatim from
    # bench/results/arena_refusal-dev1-noenv-arena.json (one arm, one process, fresh
    # counters file, ONNXRUNTIME_EP_VULKAN_DEVICE unset).
    measured_noenv = {
        "outputs_bind_attempted": 3,
        "outputs_bind_declined": 2,
        "outputs_device_bound": 1,
        "alloc_device_authority_grants": 1,
        "alloc_device_frame": "SHARED",
        "alloc_device_frame_allocator_index": "1",
        "alloc_device_frame_session_devices": "0=Intel(R) Iris(R) Xe Graphics",
    }
    rf = classify(measured_noenv, {}, "UNMEASURED(arena_refused)")
    assert rf["consistent_with"] == ["F_allocator_index_mismatch"], rf
    assert rf["separated"] is True

    # and its paired polarity: same arm, same device, env pinned -> AGREE, and the
    # discriminant counter moves to match. If this arm ever classified a cause it would
    # mean the classifier names causes for runs that worked.
    measured_pinned = dict(
        measured_noenv,
        outputs_bind_declined=0,
        outputs_device_bound=3,
        alloc_device_authority_grants=3,
        alloc_device_frame_allocator_index="0",
    )
    rp = classify(measured_pinned, {}, "AGREE")
    assert rp["consistent_with"] == ["NOT_REFUSED"], rp

    # A SHARED frame is NOT enough to rule F out, and a SPLIT frame is not what F is.
    assert measured_noenv["alloc_device_frame"] == "SHARED"

    # D: everything bound and granted, nothing failed -> the alias itself
    arena = {
        "outputs_bind_attempted": 3,
        "outputs_bind_declined": 0,
        "outputs_device_bound": 3,
        "alloc_failed_lookups": 0,
        "alloc_device_attach_unavailable": 0,
        "alloc_device_attach_failures": 0,
        "alloc_device_authority_grants": 1,
        "alloc_device_frame": "SHARED",
        "alloc_device_frame_allocator_index": "0",
        "alloc_device_frame_session_devices": "0=Intel(R) Iris(R) Xe Graphics",
    }
    r = classify(arena, {"alloc_failed_lookups": 0}, "UNMEASURED(arena_refused)")
    assert r["consistent_with"] == ["D_alias_not_honoured"], r
    assert r["separated"] is True, r

    # A declined bind on a matched allocator index is cause A, not D: `bound > 0` must
    # never on its own refute an output-side failure.
    ra = classify(dict(arena, outputs_bind_declined=2, outputs_device_bound=1),
                  {"alloc_failed_lookups": 0}, "UNMEASURED(arena_refused)")
    assert "A_no_ort_output_ptr" in ra["consistent_with"], ra
    assert "A_no_ort_output_ptr" not in ra["ruled_out"], ra

    # C: an attach was unavailable -> input side, and NOT separated from D alone
    arena_c = dict(arena, alloc_device_attach_unavailable=2)
    r2 = classify(arena_c, {"alloc_failed_lookups": 0}, "UNMEASURED(arena_refused)")
    assert "C_input_never_device_bound" in r2["consistent_with"], r2

    # The two readings must reach DIFFERENT conclusions or the classifier separates
    # nothing and every run would report whatever it reports first.
    assert r["consistent_with"] != r2["consistent_with"]

    # and an unseparated verdict must hand off rather than guess
    arena_amb = dict(arena, alloc_device_attach_unavailable=1,
                     alloc_device_authority_grants=0)
    r3 = classify(arena_amb, {"alloc_failed_lookups": 0}, "UNMEASURED(arena_refused)")
    assert r3["separated"] is False and "hand_off" in r3, r3
    print("SELFTEST PASS: 8 arms, the classifier separates and refuses to guess")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="session device index. Independent of ONNXRUNTIME_EP_VULKAN_DEVICE on "
             "purpose: the whole point of the no-env arm is to drive the session to a "
             "device the EP factory was never told to restrict itself to.",
    )
    ap.add_argument("--tag", default=None, help="suffix for the record filename")
    ap.add_argument(
        "--arm",
        choices=["arena", "control", "both"],
        default="both",
        help="run ONE arm per process. The counters file is process-cumulative and is "
             "written by the EP at its own cadence, so a file produced by a process that "
             "ran two arms cannot attribute a single counter to either of them. "
             "`--arm arena` plus `--arm control` in two processes with two counters "
             "files is the only configuration whose readings are attributable.",
    )
    args = ap.parse_args()
    if args.selftest:
        return _selftest()

    import onnxruntime as ort  # noqa: PLC0415

    import _models as m  # noqa: PLC0415

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB unset; refusing to run")
        return 2
    try:
        ort.register_execution_provider_library(m.EP_NAME, str(pathlib.Path(lib).resolve()))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            print(f"ERROR(instrument): {exc}")
            return 2

    import probe_gqa_grouping as g  # noqa: PLC0415

    env_dev = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE")
    dev_index = args.device_index if args.device_index is not None else int(env_dev or "0")
    g.DEVICE_INDEX = int(dev_index)
    counters_path = os.environ.get("ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE")
    if not counters_path:
        print("ERROR(instrument): ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE unset; the cause "
              "is read off counters and without them this probe reads nothing")
        return 2

    # Both arms are taken verbatim from probe_gqa_grouping.ARMS rather than written out
    # here.  The first version of this probe hand-wrote a "control" with past_stride=17,
    # which produced static input extents, was never claimed by the EP, and therefore
    # measured nothing at all -- trap #1 from my own round-37 notes, walked into by the
    # person who wrote the note.
    by_name = {a["name"]: a for a in g.ARMS}
    arena_arm = dict(by_name["g1_decode_arena"])
    control_arm = dict(by_name["g1_decode_growing"])

    arena_rec: dict = {}
    control_rec: dict = {}
    arena_counters: dict = {}
    control_counters: dict = {}
    if args.arm in ("arena", "both"):
        arena_rec = g.run_arena_arm(arena_arm, repeats=1)
        arena_counters = _read_counters(counters_path)
    if args.arm in ("control", "both"):
        control_rec = g.run_arm(control_arm, repeats=1)
        control_counters = _read_counters(counters_path)

    measured = arena_counters if args.arm != "control" else control_counters
    device_name = (
        arena_rec.get("device_name")
        or control_rec.get("device_name")
        or measured.get("alloc_device_frame_device")
        or "UNOBSERVABLE"
    )
    rec = {
        "probe": "arena_refusal",
        "arm_run": args.arm,
        "counters_attributable": args.arm != "both",
        "device_selector_requested": dev_index,
        "ep_vulkan_device_env": env_dev,
        "device_name_from_run": device_name,
        "running_device_names": measured.get("running_device_names"),
        "dispatches_executed": measured.get("dispatches_executed"),
        "claimed_nodes": measured.get("claimed_nodes"),
        "arena_verdict": arena_rec.get("verdict"),
        "arena_run_error": arena_rec.get("run_error"),
        "control_verdict": control_rec.get("verdict"),
        "counters_arena": {
            k: arena_counters.get(k)
            for ks in CAUSE_EVIDENCE.values()
            for k in ks
        },
        "counters_after_control": {
            k: control_counters.get(k)
            for ks in CAUSE_EVIDENCE.values()
            for k in ks
        },
        "alloc_device_frame_sides": measured.get("alloc_device_frame_sides"),
        "cause_evidence_map": CAUSE_EVIDENCE,
        "classification": classify(
            arena_counters, control_counters, arena_rec.get("verdict")
        )
        if args.arm != "control"
        else {"consistent_with": ["NOT_APPLICABLE(control-only process)"]},
    }
    tag = args.tag or f"dev{dev_index}"
    out = _RESULTS_DIR / f"arena_refusal-{tag}.json"
    out.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")

    print(f"device (off the run): {device_name}")
    print(f"running_device_names: {rec['running_device_names']}")
    print(f"dispatches_executed:  {rec['dispatches_executed']}")
    print(f"arena verdict:   {rec['arena_verdict']}")
    print(f"control verdict: {rec['control_verdict']}")
    print(f"frame: {rec['alloc_device_frame_sides']}")
    print(json.dumps(rec["counters_arena"], indent=1, sort_keys=True))
    print(json.dumps(rec["classification"], indent=1, sort_keys=True))
    print(f"record: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
