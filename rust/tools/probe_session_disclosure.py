#!/usr/bin/env python
"""Two-arm control for the §8.9.7 session-creation disclosure (RAI-009).

WHAT IS BEING FALSIFIED
-----------------------
A user creating a session against a build that would claim ops whose correctness is ``UNMEASURED``
or known ``DIVERGENT`` must be told **at session creation**, through ORT's own logging sink — not
left to discover it from a wrong answer later.

Two arms, both mandatory:

* **A (must WARN).** A form with a kernel and deliberately no proof (`mul_f16_unproven`, Mouse's
  planted case) is enabled through ``ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN``. The session-creation
  WARN must appear on ORT's sink and must name that form.
* **B (must be silent).** A proven form (`add_f32`, in the shipped ledger) is claimed with no
  escape hatch set. No session-creation WARN may appear — **and** the run must be shown to have
  claimed something, otherwise the silence is the silence of a session that claimed nothing.

Arm B is the deliverable, not arm A. A WARN that cannot be shown *not* to fire on a good run is a
printed opinion, not a detector.

EVIDENCE CLASS: PLANTED
-----------------------
Under Link's PLANTED/OBSERVED axis, arm A is **PLANTED** and is recorded as such. No production
build of this repository has ever claimed an unproven form — it requires the operator to name a
key in an environment variable. What this control demonstrates is the *warning path*, not the
frequency of the condition. Arm B is likewise PLANTED in its model, though its claim set is the
ordinary production one.

WHY THE PROOF KEY IS LEARNED, NEVER PASTED
------------------------------------------
Arm A runs the model **twice**: once with ``ONNXRUNTIME_EP_VULKAN_CLAIM_LOG`` to read the proof
key the claim gate itself computed, then once with that key in the escape hatch. A hardcoded key
that drifts out of the registry's canonical form makes the plant silently inert — the hatch
matches nothing, the form is declined, no WARN is due, and the arm goes green having tested
nothing. R10: the artifact's content must vary with its input.

EXIT CODES (R13)
----------------
``0`` PASS · ``1`` FAIL(condition) · ``4`` ERROR(instrument). An instrument error is never a
detection, and the three are never collapsed into two.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
LIB = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
RESULTS = REPO / "bench" / "results"

ENV_HATCH = "ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN"
ENV_CLAIM_LOG = "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"
ENV_COUNTERS = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

WARN_MARKER = "§8.9.7"
UNPROVEN_MARKER = "NOT established"
ORT_DECORATION = re.compile(r"\[[VIWEF]:onnxruntime:")

sys.path.insert(0, str(REPO / "rust" / "tools"))
# Imported, never re-implemented: a second decoder is a second thing that can drift, and the
# defect it would reintroduce (a delivered WARN reported as absent) has already happened twice.
from probe_broken_commitment import decode_both  # noqa: E402


# ---------------------------------------------------------------------------------------------
# child: one session, one arm
# ---------------------------------------------------------------------------------------------
def child(device_index: int, case: str, counters_path: pathlib.Path) -> int:
    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, str(REPO / "rust" / "tools"))
    import ledger_case_models as cases

    out_dir = REPO / "rust" / "target" / "probe_session_disclosure"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = cases.build(case, out_dir)

    os.environ[ENV_COUNTERS] = str(counters_path)
    ort.set_default_logger_severity(2)  # WARNING and above
    ort.register_execution_provider_library("VulkanExecutionProvider", str(LIB))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "VulkanExecutionProvider"]
    if device_index >= len(devices):
        print(f"CHILD-ERROR: device {device_index} not advertised ({len(devices)} available)")
        return 4

    so = ort.SessionOptions()
    so.log_severity_level = 2
    so.add_provider_for_devices([devices[device_index]], {})
    sess = ort.InferenceSession(str(model), so)
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 4 for d in i.shape]
        dtype = np.float16 if "float16" in i.type else np.float32
        feeds[i.name] = np.ones(shape, dtype=dtype)
    sess.run(None, feeds)
    print("CHILD-OK")
    return 0


# ---------------------------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------------------------
def run_child(device_index: int, case: str, tag: str, extra_env: dict) -> tuple[str, dict, int]:
    counters = RESULTS / f"session-disclosure-dev{device_index}-{tag}.json"
    counters.parent.mkdir(parents=True, exist_ok=True)
    counters.unlink(missing_ok=True)
    env = dict(os.environ)
    env.pop(ENV_HATCH, None)
    env.pop(ENV_CLAIM_LOG, None)
    env.update(extra_env)
    proc = subprocess.run(
        [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--child",
            "--device",
            str(device_index),
            "--case",
            case,
            "--counters",
            str(counters),
        ],
        capture_output=True,
        env=env,
    )
    log = decode_both(proc.stdout or b"") + "\n" + decode_both(proc.stderr or b"")
    doc = json.loads(counters.read_text()) if counters.is_file() else {}
    return log, doc, proc.returncode


def sink_disclosure_warns(log: str) -> list[str]:
    """Session-creation WARNs that reached **ORT's own sink**, not our private stderr line."""
    return [
        ln
        for ln in log.splitlines()
        if ORT_DECORATION.search(ln) and WARN_MARKER in ln and UNPROVEN_MARKER in ln
    ]


def readable(lines: list[str]) -> list[str]:
    """Trim each captured line to the part a human can read.

    ``decode_both`` deliberately reads the same bytes four ways, so a captured line carries the
    mojibake of the three readings that did not apply. Keeping that in the artifact makes the
    artifact unreadable, and an unreadable artifact is one nobody checks. The trim starts at ORT's
    own decoration, which is the earliest point the line is known to be legible.
    """
    out = []
    for ln in lines:
        m = ORT_DECORATION.search(ln)
        out.append((ln[m.start() :] if m else ln).replace("\x1b[m", "").strip())
    return out


def learn_proof_key(device_index: int, case: str) -> tuple[str | None, str]:
    """Pass 1: read the proof key the claim gate itself computed for the planted form."""
    claim_log = REPO / "rust" / "target" / f"session-disclosure-claims-dev{device_index}.jsonl"
    claim_log.parent.mkdir(parents=True, exist_ok=True)
    claim_log.unlink(missing_ok=True)
    log, _doc, _rc = run_child(
        device_index, case, "keylearn", {ENV_CLAIM_LOG: str(claim_log)}
    )
    if not claim_log.is_file():
        return None, log
    keys = []
    for line in claim_log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("proof_key") and rec.get("op", "").split("::")[-1] == "Mul":
            keys.append(rec["proof_key"])
    return (keys[0] if keys else None), log


def judge(device_index: int) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    report: dict = {"device": device_index, "evidence_class": "PLANTED"}

    # ---- ARM A: a claimed UNMEASURED form must WARN at session creation ----------------------
    key, keylog = learn_proof_key(device_index, "mul_f16_unproven")
    report["learned_proof_key"] = key
    if not key:
        return (
            "ERROR",
            [
                "ERROR(instrument): the claim log produced no proof key for the planted Mul "
                "form, so the escape hatch cannot be aimed and arm A would test nothing. "
                f"Claim-log pass tail: {keylog[-600:]!r}"
            ],
            report,
        )

    a_log, a_doc, a_rc = run_child(
        device_index, "mul_f16_unproven", "armA", {ENV_HATCH: key}
    )
    report["arm_a"] = {
        "returncode": a_rc,
        "claimed_form_evidence": a_doc.get("claimed_form_evidence"),
        "session_disclosure_channel": a_doc.get("session_disclosure_channel"),
        "claimed_forms_unmeasured": a_doc.get("claimed_forms_unmeasured"),
        "claimed_nodes": a_doc.get("claimed_nodes"),
    }
    if a_rc == 4:
        return "ERROR", [f"ERROR(instrument): arm A child could not run: {a_log[-600:]!r}"], report

    a_warns = sink_disclosure_warns(a_log)
    report["arm_a"]["ort_sink_warns"] = readable(a_warns[:2])
    if a_doc.get("claimed_nodes", 0) in (0, None):
        failures.append(
            "arm A claimed no nodes, so no disclosure was due and its WARN (or absence) "
            "measures nothing — the escape hatch did not take. "
            f"Learned key: {key!r}"
        )
    elif not a_warns:
        failures.append(
            "arm A claimed an UNMEASURED form and NO session-creation WARN reached ORT's sink. "
            f"counters say claimed_form_evidence={a_doc.get('claimed_form_evidence')!r}, "
            f"session_disclosure_channel={a_doc.get('session_disclosure_channel')!r}"
        )
    elif key not in "\n".join(a_warns).replace("\x00", ""):
        failures.append(
            "arm A warned but did not name the form it is warning about; a warning a reader "
            f"cannot act on is not a disclosure. WARN: {a_warns[0]!r}"
        )
    if a_doc and a_doc.get("claimed_form_evidence") not in ("UNMEASURED-PRESENT", None):
        failures.append(
            "arm A's counters token does not report the unmeasured form: "
            f"claimed_form_evidence={a_doc.get('claimed_form_evidence')!r}"
        )

    # ---- ARM B: a proven claim set must be SILENT --------------------------------------------
    b_log, b_doc, b_rc = run_child(device_index, "add_f32", "armB", {})
    report["arm_b"] = {
        "returncode": b_rc,
        "claimed_form_evidence": b_doc.get("claimed_form_evidence"),
        "session_disclosure_channel": b_doc.get("session_disclosure_channel"),
        "claimed_nodes": b_doc.get("claimed_nodes"),
        "claimed_forms_proven": b_doc.get("claimed_forms_proven"),
    }
    if b_rc == 4:
        return "ERROR", [f"ERROR(instrument): arm B child could not run: {b_log[-600:]!r}"], report

    b_warns = sink_disclosure_warns(b_log)
    report["arm_b"]["ort_sink_warns"] = readable(b_warns[:2])
    # Non-vacuity FIRST. Without this the arm passes for a reason unrelated to what it tests.
    if not b_doc.get("claimed_forms_proven"):
        failures.append(
            "arm B's silence is vacuous: the run claimed no proven form "
            f"(claimed_forms_proven={b_doc.get('claimed_forms_proven')!r}, "
            f"claimed_nodes={b_doc.get('claimed_nodes')!r}), so 'it did not warn' says nothing "
            "about whether it can."
        )
    elif b_warns:
        failures.append(
            "arm B: a session-creation WARN fired on a fully proven claim set, so this WARN "
            f"cannot be shown NOT to fire and is not a detector. WARN: {b_warns[0]!r}"
        )
    if b_doc.get("claimed_form_evidence") not in ("ALL-PROVEN", None):
        failures.append(
            "arm B's counters token is not ALL-PROVEN on a proven claim set: "
            f"{b_doc.get('claimed_form_evidence')!r}"
        )

    # ---- R13 blindness control: the witness must be able to see a WARN at all ----------------
    if not a_warns and not b_warns:
        return (
            "ERROR",
            [
                "ERROR(instrument): neither arm saw a single line on ORT's sink, so this probe "
                "cannot distinguish 'no WARN was emitted' from 'this witness cannot read the "
                "channel'. That is an instrument error, never a detection."
            ],
            report,
        )

    return ("FAIL" if failures else "PASS"), failures, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--case", default="add_f32")
    ap.add_argument("--counters", type=pathlib.Path)
    ap.add_argument("--devices", default="0,1")
    args = ap.parse_args()

    if args.child:
        return child(args.device, args.case, args.counters)

    if not LIB.is_file():
        print(f"ERROR(instrument): EP library not built: {LIB}")
        return 4

    overall, out = "PASS", []
    for dev in [int(d) for d in args.devices.split(",") if d.strip()]:
        verdict, failures, report = judge(dev)
        report["verdict"] = verdict
        report["failures"] = failures
        out.append(report)
        print(f"device {dev}: {verdict}")
        for f in failures:
            print(f"  - {f}")
        if verdict == "ERROR":
            overall = "ERROR"
        elif verdict == "FAIL" and overall != "ERROR":
            overall = "FAIL"

    RESULTS.mkdir(parents=True, exist_ok=True)
    artifact = RESULTS / "session-disclosure-control.json"
    artifact.write_text(
        json.dumps(
            {
                "control": "session-creation disclosure (DESIGN.md §8.9.7 / RAI-009)",
                "evidence_class": "PLANTED",
                "evidence_class_note": (
                    "Arm A's UNMEASURED form is one I planted; it is reachable only by naming a "
                    "key in ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN. No production build has ever "
                    "produced this condition. What is demonstrated is the warning path."
                ),
                "verdict": overall,
                "devices": out,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{overall} — {artifact}")
    return {"PASS": 0, "FAIL": 1, "ERROR": 4}[overall]


if __name__ == "__main__":
    sys.exit(main())
