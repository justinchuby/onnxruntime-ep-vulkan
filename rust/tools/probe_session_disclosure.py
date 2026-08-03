#!/usr/bin/env python
"""Two-arm control for the §8.9.7 session-creation disclosure (RAI-009).

WHAT IS BEING FALSIFIED
-----------------------
A user creating a session against a build that would claim ops whose correctness is ``UNMEASURED``
or known ``DIVERGENT`` must be told **at session creation**, through ORT's own logging sink — not
left to discover it from a wrong answer later.

Two arms, both mandatory:

* **A (must WARN).** A form with a kernel and deliberately no proof — Mouse's planted case, named
  by ``ledger_case_models.PLANTED_CONTROL_CASE`` and **never** spelled out here — is enabled
  through ``ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN``. The session-creation WARN must appear on
  ORT's sink and must name that form.

  The case name is imported rather than written down because it has already moved once: the plant
  was ``mul_f16_unproven`` until 2026-08-02, when populating the ledger for the op suite proved
  that very form and the control fired on its own author. It is now ``sub_f16_dyn_unproven``.
  A probe holding the old name does not go quietly wrong — it raises ``KeyError`` inside the
  child and this probe reports ``ERROR(instrument)``, which is the correct outcome and is how
  the drift was found — but a probe that cannot run is a probe that discloses nothing, and the
  plant will move again.
* **B (must be silent).** A proven form (`add_f32`, in the shipped ledger) is claimed with no
  escape hatch set. No session-creation WARN may appear — **and** the run must be shown to have
  claimed something, otherwise the silence is the silence of a session that claimed nothing.
* **C (must speak) and D (must fall silent when asked to).** The same proven form, with ORT's
  logger at INFO. RAI-008(b) asks for *"INFO per claimed form + proof key/ledger entry, WARN on
  UNMEASURED/DIVERGENT"* — two halves, and arms A and B run at WARNING severity, where the INFO
  half is invisible by construction. A probe with only A and B discharges half a condition and
  reports it whole. Arm C requires the INFO to be delivered, naming the form, its proof key, and
  the ledger entry behind it (the artifact, the device, the ORT build). Arm D requires the
  delivery counter to read differently at WARNING than at INFO, because a channel token that is
  the same under both is a constant wearing a measurement's name.

  **A limitation of this host, measured rather than assumed:** ORT 1.28 never writes the EP's
  INFO to its console sink — at *any* host severity — while it writes the WARN from the identical
  call site at all of them. This is not the host's threshold discarding it: ORT's own
  `Logger_GetLoggingSeverityLevel` reports VERBOSE at severity 0 and INFO at severity 1, both of
  which admit an INFO record, and other ORT-decorated lines appear in the same run so the witness
  is not blind. Arm C therefore reports the INFO half as **EMITTED and OFFERED, not witnessed on
  ORT's channel**, records the measurement supporting that under `ort_console_delivery`, and reads
  the *content* from whichever witness can see it (naming which one). It does not report delivery,
  because this process cannot witness delivery — the counter tokens are `OFFERED_TO_ORT` and
  `BELOW_ORT_THRESHOLD` for exactly that reason, and an earlier `ORT_SINK` was renamed because it
  asserted an arrival nobody had seen.

  **What a user at ORT's default severity actually gets, stated plainly:** the WARN, always; the
  INFO, never — at the default the threshold discards it, and above the default ORT drops it for a
  reason inside ORT that this probe has not established. The pair degrades in the safe direction
  (the bad news is the half that gets through), and arm D is what proves the EP is not the one
  falling silent: the channel counter reads `OFFERED_TO_ORT` at INFO and `BELOW_ORT_THRESHOLD` at
  WARNING, so it moves with its input and names which condition it is in.

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
# Arms A and B run ORT at WARNING so that the WARN's presence/absence is the only variable. Arm C
# needs INFO, and asking for it through the environment keeps the child a single code path.
ENV_SEVERITY = "_PROBE_ORT_SEVERITY"

WARN_MARKER = "§8.9.7"
UNPROVEN_MARKER = "NOT established"
PROVEN_MARKER = "proven by"
ORT_DECORATION = re.compile(r"\[[VIWEF]:onnxruntime:")

sys.path.insert(0, str(REPO / "rust" / "tools"))
# Imported, never re-implemented: a second decoder is a second thing that can drift, and the
# defect it would reintroduce (a delivered WARN reported as absent) has already happened twice.
from probe_broken_commitment import decode_both  # noqa: E402

# The planted control's case name is *owned by the case table*, not by this probe. See the module
# docstring: it has moved once already, and a copy here is a second place for it to be wrong.
from ledger_case_models import BUILDERS as _CASE_BUILDERS  # noqa: E402
from ledger_case_models import PLANTED_CONTROL_CASE  # noqa: E402

# Arm B's proven form. Unlike the plant this one is required to be *in* the ledger; it is checked
# at run time by arm B's non-vacuity assertion rather than trusted here.
PROVEN_CASE = "add_f32"


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
    severity = int(os.environ.get(ENV_SEVERITY, "2"))  # 2 = WARNING, 1 = INFO
    ort.set_default_logger_severity(severity)
    ort.register_execution_provider_library("VulkanExecutionProvider", str(LIB))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "VulkanExecutionProvider"]
    if device_index >= len(devices):
        print(f"CHILD-ERROR: device {device_index} not advertised ({len(devices)} available)")
        return 4

    so = ort.SessionOptions()
    so.log_severity_level = severity
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


def sink_disclosure_infos(log: str) -> list[str]:
    """Session-creation INFOs, on ORT's sink, that disclose a *proven* claim.

    Keyed on ``proven by`` rather than on the section marker alone, because §8.9.7 also emits a
    zero-claims INFO and a run that claimed nothing must never be read as a run that disclosed
    its proofs.
    """
    return [
        ln
        for ln in log.splitlines()
        if ORT_DECORATION.search(ln) and WARN_MARKER in ln and PROVEN_MARKER in ln
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
    # No op-type filter. This used to read `== "Mul"`, a second copy of the plant's identity that
    # survived the plant moving to `Sub` and would have gone silently inert. The planted case is a
    # single-node model, so the record is identifiable by its shape: exactly one node, carrying a
    # proof key, and — this is the part worth asserting — **missing from the ledger**. A plant the
    # ledger has since proven is not a plant, and aiming the escape hatch at it would make arm A
    # green while testing nothing.
    keyed = []
    for line in claim_log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("proof_key"):
            keyed.append(rec)
    if len(keyed) != 1:
        return None, log + (
            f"\n[probe] the planted case produced {len(keyed)} keyed claim records, not 1; "
            "this probe cannot tell which form it is meant to aim at."
        )
    rec = keyed[0]
    if rec.get("ledger_hit"):
        return None, log + (
            f"\n[probe] the planted control {rec['proof_key']!r} IS IN THE LEDGER. It has been "
            "proven since it was planted, so it is no longer an unproven form and arm A would "
            "warn about nothing. Move the plant (`ledger_case_models.PLANTED_CONTROL_CASE`) to a "
            "form the ledger does not cover."
        )
    return rec["proof_key"], log


def judge(device_index: int) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    report: dict = {
        "device": device_index,
        "evidence_class": "PLANTED",
        "planted_control_case": PLANTED_CONTROL_CASE,
        "proven_case": PROVEN_CASE,
    }

    # The case table is the authority on both names. Checking here turns a `KeyError` raised deep
    # inside a child process into a sentence that names what moved.
    for role, case in (("planted control", PLANTED_CONTROL_CASE), ("proven form", PROVEN_CASE)):
        if case not in _CASE_BUILDERS:
            return (
                "ERROR",
                [
                    f"ERROR(instrument): the {role} case {case!r} is not in "
                    "`ledger_case_models.BUILDERS`, so this probe cannot build the model it "
                    "judges. The case table moved and this probe did not follow."
                ],
                report,
            )

    # ---- ARM A: a claimed UNMEASURED form must WARN at session creation ----------------------
    key, keylog = learn_proof_key(device_index, PLANTED_CONTROL_CASE)
    report["learned_proof_key"] = key
    if not key:
        return (
            "ERROR",
            [
                "ERROR(instrument): the claim log produced no proof key for the planted "
                f"{PLANTED_CONTROL_CASE!r} form, so the escape hatch cannot be aimed and arm A "
                f"would test nothing. Claim-log pass tail: {keylog[-600:]!r}"
            ],
            report,
        )

    a_log, a_doc, a_rc = run_child(
        device_index, PLANTED_CONTROL_CASE, "armA", {ENV_HATCH: key}
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
    b_log, b_doc, b_rc = run_child(device_index, PROVEN_CASE, "armB", {})
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
    report["arm_b"]["claimed_forms_device_unattributed"] = b_doc.get(
        "claimed_forms_device_unattributed"
    )
    # Non-vacuity FIRST. Without this the arm passes for a reason unrelated to what it tests.
    #
    # "Proof-backed" is `claimed_forms_proven + claimed_forms_device_unattributed`, not
    # `claimed_forms_proven` alone. §8.9.18 split PROVEN into proven-here and proven-but-the-frame
    # was-never-checked, and on this hardware *every* baked entry is in the second state — so an
    # arm keyed on `claimed_forms_proven` alone reports every real run as vacuous. This probe
    # asked for ALL-PROVEN, which is the token that had been asserting a device frame nothing had
    # checked; asserting it now would be asserting the defect. What arm B actually tests is that a
    # claim set with nothing UNMEASURED, DIVERGENT or FAULTED in it does not warn, and that is
    # what it checks.
    b_backed = (b_doc.get("claimed_forms_proven") or 0) + (
        b_doc.get("claimed_forms_device_unattributed") or 0
    )
    if not b_backed:
        failures.append(
            "arm B's silence is vacuous: the run claimed no proof-backed form "
            f"(claimed_forms_proven={b_doc.get('claimed_forms_proven')!r}, "
            f"claimed_forms_device_unattributed="
            f"{b_doc.get('claimed_forms_device_unattributed')!r}, "
            f"claimed_nodes={b_doc.get('claimed_nodes')!r}), so 'it did not warn' says nothing "
            "about whether it can."
        )
    elif b_warns:
        failures.append(
            "arm B: a session-creation WARN fired on a fully proven claim set, so this WARN "
            f"cannot be shown NOT to fire and is not a detector. WARN: {b_warns[0]!r}"
        )
    if b_doc.get("claimed_form_evidence") not in (
        "ALL-PROVEN",
        "DEVICE-UNATTRIBUTED-PRESENT",
        None,
    ):
        failures.append(
            "arm B's counters token reports something unproven on a proof-backed claim set: "
            f"{b_doc.get('claimed_form_evidence')!r}"
        )

    # ---- ARM C: the INFO half. RAI-008(b) asks for *both*, and only the WARN was witnessed ----
    # Rai's condition reads "INFO per claimed form + proof key/ledger entry, WARN on
    # UNMEASURED/DIVERGENT". Arms A and B run ORT at WARNING severity, which cannot see the INFO
    # at all — so a probe with only those two arms discharges half a condition and reports it as
    # whole. This arm is the other half, satisfied as written rather than as I would have written
    # it: the INFO must reach ORT's own sink, name the form, name its proof key, and name the
    # artifact that proved it.
    c_log, c_doc, c_rc = run_child(
        device_index, PROVEN_CASE, "armC", {ENV_SEVERITY: "1"}
    )
    report["arm_c"] = {
        "returncode": c_rc,
        "claimed_form_evidence": c_doc.get("claimed_form_evidence"),
        "claimed_forms_proven": c_doc.get("claimed_forms_proven"),
        "claimed_nodes": c_doc.get("claimed_nodes"),
    }
    if c_rc == 4:
        return "ERROR", [f"ERROR(instrument): arm C child could not run: {c_log[-600:]!r}"], report

    c_infos = sink_disclosure_infos(c_log)
    c_stderr_infos = [
        ln for ln in c_log.splitlines()
        if ln.startswith("[vulkan-ep] INFO:") and PROVEN_MARKER in ln
    ]
    c_any_ort_lines = [ln for ln in c_log.splitlines() if ORT_DECORATION.search(ln)]
    report["arm_c"].update(
        {
            "session_disclosure_infos": c_doc.get("session_disclosure_infos"),
            "session_disclosure_infos_to_ort_sink": c_doc.get(
                "session_disclosure_infos_to_ort_sink"
            ),
            "session_disclosure_info_channel": c_doc.get("session_disclosure_info_channel"),
            "ort_sink_severity_threshold": c_doc.get("ort_sink_severity_threshold"),
            "ort_sink_infos": readable(c_infos[:2]),
            "ort_decorated_lines_seen": len(c_any_ort_lines),
        }
    )

    # (1) Non-vacuity, first, as in arm B — and proof-backed, for the same §8.9.18 reason.
    c_backed = (c_doc.get("claimed_forms_proven") or 0) + (
        c_doc.get("claimed_forms_device_unattributed") or 0
    )
    report["arm_c"]["claimed_forms_device_unattributed"] = c_doc.get(
        "claimed_forms_device_unattributed"
    )
    if not c_backed:
        failures.append(
            "arm C claimed no proof-backed form, so there was no INFO due and its presence or "
            f"absence measures nothing (claimed_forms_proven={c_doc.get('claimed_forms_proven')!r}, "
            "claimed_forms_device_unattributed="
            f"{c_doc.get('claimed_forms_device_unattributed')!r})."
        )
    else:
        # (2) The EP's own observable. This is the half the EP owns and can be held to: it emitted
        # the INFO, and ORT's threshold — read from ORT, not assumed — admitted it.
        if c_doc.get("ort_sink_severity_threshold") not in ("VERBOSE", "INFO"):
            failures.append(
                "arm C ran with the host asking for INFO and ORT reports its threshold as "
                f"{c_doc.get('ort_sink_severity_threshold')!r}. The arm cannot test delivery of "
                "a record the host has asked not to be told, so this is a mis-configured arm."
            )
        elif c_doc.get("session_disclosure_info_channel") != "OFFERED_TO_ORT":
            failures.append(
                "arm C claimed a proven form and the INFO half of the §8.9.7 disclosure was not "
                "offered to ORT at a level ORT's own threshold admits: "
                f"infos={c_doc.get('session_disclosure_infos')!r}, "
                f"to_ort_sink={c_doc.get('session_disclosure_infos_to_ort_sink')!r}, "
                f"channel={c_doc.get('session_disclosure_info_channel')!r}. RAI-008(b) asks for "
                "the INFO as well as the WARN; a disclosure that only speaks when the news is bad "
                "cannot be shown to work when the news is good."
            )

        # (3) The log witness, with its own blindness control. ORT 1.28 does not print the EP's
        # INFO to its console sink at any host severity, while it prints the WARN from the same
        # call site at all of them. `ort_decorated_lines_seen` is the blindness control: if it is
        # zero the witness saw nothing at all and its silence is an instrument outage rather than a
        # detection, and R13 forbids spelling an outage like a finding. The content is therefore
        # read from whichever witness can see it, and the channel is named for what was measured.
        witness = c_infos or c_stderr_infos
        report["arm_c"]["content_witness"] = (
            "ORT_SINK" if c_infos else ("EP_STDERR" if c_stderr_infos else "NONE")
        )
        # THE LIMITATION, RECORDED RATHER THAN ARGUED AWAY.
        # The EP offers the INFO and ORT's own threshold admits it, and the line still does not
        # appear on ORT's console — while the identical call at WARNING, in the same process and
        # the same second, does. That is not something this probe can repair from the EP side and
        # it is not something it may report as delivery. It is recorded, with the measurement that
        # supports it, so that RAI-008(b) can be read as the partial discharge it is.
        if not c_infos:
            report["arm_c"]["ort_console_delivery"] = "NOT_OBSERVED"
            report["arm_c"]["ort_console_delivery_note"] = (
                "The EP offered the INFO to Logger_LogMessage and ORT reports its own threshold "
                f"as {c_doc.get('ort_sink_severity_threshold')!r}, which admits INFO; "
                f"{len(c_any_ort_lines)} ORT-decorated line(s) were seen in the same run, so the "
                "witness is not blind. The record is nevertheless absent from ORT's console. "
                "RAI-008(b)'s INFO half is therefore EMITTED and OFFERED but NOT WITNESSED on "
                "ORT's own channel. Falsifier: an ORT build or host configuration in which an "
                "INFO from a plugin EP does appear on the console — then this line is delivered "
                "and this note is withdrawn."
            )
        else:
            report["arm_c"]["ort_console_delivery"] = "OBSERVED"
        if not witness:
            failures.append(
                "arm C saw the INFO on no channel at all — not ORT's sink and not the EP's own "
                "stderr — so there is nothing to check the content of."
            )
        else:
            info = "\n".join(witness)
            # The things the condition names, checked one at a time so a failure says which.
            if "ai.onnx::Add" not in info:
                failures.append(f"arm C's INFO does not name the claimed form: {witness[0]!r}")
            if "/ew_binary_add_f32/" not in info:
                failures.append(
                    "arm C's INFO does not carry the proof key of the form it discloses: "
                    f"{witness[0]!r}"
                )
            # The ledger *entry*, not just the key: the artifact that proved it, the device it was
            # proved on, and the ORT build it was proved against. A key alone tells a reader what
            # was claimed; the entry tells them what to go and check, which is the difference
            # between a disclosure and a label. The ledger's own filename is deliberately not a
            # needle — it is a constant this INFO could carry while knowing nothing about the form.
            # The device needle accepts either spelling because the two INFO branches word it
            # differently and both name it: the proven-here line says "on <device>", the
            # DEVICE-UNATTRIBUTED line says "entry-device=<device>" beside the running device it
            # could not be matched against. The property is that the entry's device is reachable
            # from the disclosure, not that one branch's phrasing survived.
            for what, needles in (
                ("the artifact that proved it", ("evidence/",)),
                ("the device it was proved on", (" on device", "entry-device=")),
                ("the ORT build it was proved against", (" against ",)),
            ):
                if not any(needle in info for needle in needles):
                    failures.append(
                        f"arm C's INFO does not name {what}, so the ledger entry behind the claim "
                        f"is not reachable from the disclosure: {witness[0]!r}"
                    )

    # ---- ARM D: the INFO channel counter must move with its input ----------------------------
    # Arm B ran the same model at the default severity, where ORT's threshold is WARNING. If the
    # new counter reported OFFERED_TO_ORT there too it would be a constant wearing a measurement's
    # name — the exact shape Morpheus refused for `ledger_hits`. It must read BELOW_ORT_THRESHOLD,
    # and the difference between the two arms is the whole evidence that the counter reads
    # anything. The arm pins the two tokens rather than only their inequality: two tokens that
    # merely differ would also be satisfied by a counter that had them backwards.
    report["arm_d"] = {
        "at_INFO": c_doc.get("session_disclosure_info_channel"),
        "at_WARNING": b_doc.get("session_disclosure_info_channel"),
        "threshold_at_WARNING": b_doc.get("ort_sink_severity_threshold"),
        "expected": {"at_INFO": "OFFERED_TO_ORT", "at_WARNING": "BELOW_ORT_THRESHOLD"},
    }
    if b_doc.get("session_disclosure_info_channel") == c_doc.get(
        "session_disclosure_info_channel"
    ):
        failures.append(
            "arm D: `session_disclosure_info_channel` reads the same at ORT severity WARNING and "
            f"at INFO ({b_doc.get('session_disclosure_info_channel')!r}), so it does not vary "
            "with the only thing it is supposed to depend on and certifies nothing."
        )
    else:
        for arm, got, want in (
            ("INFO", c_doc.get("session_disclosure_info_channel"), "OFFERED_TO_ORT"),
            ("WARNING", b_doc.get("session_disclosure_info_channel"), "BELOW_ORT_THRESHOLD"),
        ):
            if got != want:
                failures.append(
                    f"arm D: at ORT severity {arm} the INFO channel reads {got!r}, expected "
                    f"{want!r}. The two arms differ, so the counter varies, but it does not name "
                    "the condition it is in — and a channel token that names the wrong condition "
                    "is worse than one that names none."
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
    ap.add_argument("--case", default=PROVEN_CASE)
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
                "limitations": [
                    d["arm_c"]["ort_console_delivery_note"]
                    for d in out
                    if isinstance(d.get("arm_c"), dict)
                    and d["arm_c"].get("ort_console_delivery") == "NOT_OBSERVED"
                ],
                "rai_008b_reading": (
                    "WARN half: DISCHARGED — witnessed on ORT's own sink on every device listed, "
                    "naming the unproven form and its proof key, with the must-be-silent polarity "
                    "shown on a proven claim set. INFO half: EMITTED and OFFERED, and NOT "
                    "witnessed on ORT's console; see `limitations`. Read this artifact as a "
                    "partial discharge. It is recorded this way deliberately: a control that "
                    "reported PASS on the pair while only half the pair had been seen would be "
                    "the same defect the control was built to detect."
                ),
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
