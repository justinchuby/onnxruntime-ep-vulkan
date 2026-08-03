"""Probe: does every dispatch write every byte of the push-constant range it declares?

Trinity found this in frame on both devices, 2026-08-02, at WARNING severity through the EP's
own messenger:

    vkCmdDispatch(): Pipeline uses a push constant range with offset 0 and size 128,
                     but 104 bytes were never set with vkCmdPushConstants

also at 88, 72, 36, 20 and 4 bytes.  It is not a VUID, so criterion 3(a) is unaffected.  It is
still a defect: **unwritten push-constant bytes are undefined, not zero.**  Nothing misbehaves
today only because no shader reads past the block it declares — a property of the shaders, not
of the API contract, and worth nothing the moment a shader grows a field.

WHAT MAKES A ZERO HERE A MEASUREMENT
------------------------------------
The same trap Trinity documented for the VUID count applies with more force here, because the
thing being counted is a warning and warnings are what a silent messenger is silent about.  A
bare `push_constant_lines: 0` is `UNOBSERVABLE`: it is equally consistent with "every byte is
written" and with "the callback is dead", and it is what a run with validation switched off
prints too.

Two independent liveness conditions, both required before a zero is reported as a finding:

  1. **The messenger spoke.**  `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT`
     puts WARNING-severity `BestPractices-` lines on the EP's own messenger, in this process and
     inside this frame (Trinity's technique).  Zero of them means the callback never fired and
     the reading is void.
  2. **The probe has been shown to fire.**  `--sensitivity` records the same reading against a
     build known to have the defect.  A detector never observed in its positive state is a
     detector with no demonstrated positive state.  The stored sensitivity record is compared
     against the current reading by `verdict()`, and a run whose sensitivity record is missing
     reports `UNPROVEN_DETECTOR` rather than a pass.

And the run must be non-trivial: an arm that dispatched nothing cannot have failed to write a
push constant.  `dispatches_executed == 0` is `ERROR(instrument)`, never a pass — the same guard
that caught the ledger-declined GQA arm.

Usage
-----
    python probe_push_constants_written.py --sensitivity   # record the pre-fix positive
    python probe_push_constants_written.py                 # measure this build
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tests" / "ops"))

from probe_validation_phi35 import BOUNDARY  # noqa: E402

PROBE_CHILD = REPO / "tests" / "ops" / "probe_validation_phi35.py"

#: The layer's own wording.  Matched on the invariant half of the sentence rather than on a
#: byte count, because the counts are exactly what changes between builds.
PUSH_NOT_SET = re.compile(r"never set with vkCmdPushConstants", re.IGNORECASE)

#: How many bytes each such line says were never set.  Kept because the *set* of shortfalls is
#: the shape of the defect: 104/88/72/36/20/4 are 128 minus the six distinct pack sizes.
SHORTFALL = re.compile(r"but (\d+) bytes were never set", re.IGNORECASE)

SENSITIVITY = HERE / "push_constants_sensitivity.json"


def _dll_hash() -> str:
    import hashlib  # noqa: PLC0415

    dll = pathlib.Path(
        os.environ.get(
            "ONNXRUNTIME_VULKAN_EP_LIB",
            str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
        )
    )
    if not dll.is_file():
        return "MISSING"
    return hashlib.sha256(dll.read_bytes()).hexdigest()[:16].upper()


def measure(timeout: int = 5400) -> dict:
    """One liveness-armed Phi-3.5 run, scored for unwritten push-constant bytes.

    The *case* is Trinity's child (`probe_validation_phi35.py --child`) and the *frame* is her
    boundary marker, both imported rather than re-implemented: two builders would be two
    definitions of the arm, and the readings could then differ for a reason nobody wrote down.
    What is not reused is her `run_arm`, which truncates the transcript to its last 2500
    characters and its line lists to 20 entries — fine for a count expected to be 0, useless for
    counting 300-odd warnings scattered through a frame.
    """
    selector = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset")
    counters = HERE / f"push_constants_counters-dev{selector}.json"
    counters.unlink(missing_ok=True)

    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    env["ONNXRUNTIME_EP_VULKAN_VALIDATE"] = "1"
    # Trinity's liveness technique: best-practices messages ride the EP's own messenger at
    # WARNING severity, in-process and in-frame, so a healthy run stops being silent.
    env["VK_LAYER_ENABLES"] = "VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT"

    r = subprocess.run(
        [sys.executable, str(PROBE_CHILD), "--child"],
        env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    output = (r.stdout or "") + "\n" + (r.stderr or "")
    lines = output.splitlines()
    boundary = max((i for i, ln in enumerate(lines) if BOUNDARY in ln), default=-1)
    in_frame = lines[: boundary + 1] if boundary >= 0 else []

    hits = [ln for ln in in_frame if PUSH_NOT_SET.search(ln)]
    after = [ln for ln in lines[boundary + 1:] if PUSH_NOT_SET.search(ln)]
    messenger = [ln for ln in in_frame if "[Vulkan validation]" in ln]
    vuids = [ln for ln in in_frame if "VUID-" in ln]
    shortfalls = sorted({int(m.group(1)) for ln in hits if (m := SHORTFALL.search(ln))})

    c = json.loads(counters.read_text(encoding="utf-8")) if counters.is_file() else {}
    (HERE / f"push_constants_transcript-dev{selector}.txt").write_text(
        output, encoding="utf-8"
    )
    return {
        "dll_sha256_16": _dll_hash(),
        "device_selector_requested": selector,
        # The selector is a request, not an identity (Trinity, 2026-08-02): read the device off
        # the run.  `=0` runs on the device the enumerator calls 1.
        "device_reported_by_run": c.get("alloc_device_frame_session_devices"),
        "child_exit_code": r.returncode,
        "boundary_seen": boundary >= 0,
        "dispatches_executed": c.get("dispatches_executed"),
        "claimed_nodes": c.get("claimed_nodes"),
        "device_losses": c.get("device_losses"),
        "messenger_lines_in_frame": len(messenger),
        "in_frame_vuid_count": len(vuids),
        "push_constant_lines": len(hits),
        "push_constant_lines_teardown": len(after),
        "shortfall_bytes_observed": shortfalls,
        "sample_lines": [ln.strip()[:200] for ln in (hits[:6] or messenger[:3])],
    }


def verdict(now: dict, sens: dict | None) -> tuple[str, int, list[str]]:
    """Classify a reading.  Refusals outrank findings; findings outrank passes."""
    why: list[str] = []

    if now["child_exit_code"] != 0:
        return "ERROR(instrument)", 2, [
            f"child exited {now['child_exit_code']} — an observation that ended early is not a "
            f"reading; a run that dies partway and exits nonzero at least says so"
        ]
    if not now.get("dispatches_executed"):
        return "ERROR(instrument)", 2, [
            "dispatches_executed is 0 or absent — nothing was dispatched, so nothing could have "
            "failed to write a push constant. This is the ledger-decline shape: the EP ran on "
            "the CPU and the arm would report a clean sweep it never took"
        ]
    if now.get("device_losses"):
        return "ERROR(instrument)", 2, [
            f"device_losses={now['device_losses']} — the frame was truncated by a lost device"
        ]
    if now["messenger_lines_in_frame"] == 0:
        return "UNOBSERVABLE", 2, [
            "0 messenger lines in frame with best-practices armed: the callback never spoke, so "
            "a 0 push-constant count is consistent with the layer being absent. Not a reading"
        ]
    why.append(
        f"messenger alive: {now['messenger_lines_in_frame']} in-frame lines with "
        f"best-practices armed"
    )

    if now["push_constant_lines"] > 0:
        return "UNWRITTEN_PUSH_CONSTANT_BYTES", 1, why + [
            f"{now['push_constant_lines']} dispatches left declared push-constant bytes unset; "
            f"shortfalls seen: {now['shortfall_bytes_observed']}"
        ]

    if sens is None:
        return "UNPROVEN_DETECTOR", 2, why + [
            "0 push-constant lines, but no sensitivity record exists: this probe has never been "
            "observed in its positive state, so its zero is unearned. Run --sensitivity against "
            "a build with the defect"
        ]
    if sens.get("push_constant_lines", 0) <= 0:
        return "UNPROVEN_DETECTOR", 2, why + [
            "the stored sensitivity record itself reads 0 — it records a build in which the "
            "detector also did not fire, which proves nothing"
        ]
    if sens.get("dll_sha256_16") == now.get("dll_sha256_16"):
        return "ERROR(instrument)", 2, why + [
            f"the sensitivity record was taken on this very binary ({now['dll_sha256_16']}) — "
            f"it cannot be both the positive control and the subject"
        ]
    why.append(
        f"detector proven: it reported {sens['push_constant_lines']} lines "
        f"(shortfalls {sens.get('shortfall_bytes_observed')}) on build "
        f"{sens.get('dll_sha256_16')}, and 0 on {now['dll_sha256_16']}"
    )
    return "PUSH_CONSTANTS_FULLY_WRITTEN", 0, why


def main(argv: list[str]) -> int:
    if "--sensitivity" in argv:
        doc = measure()
        doc["role"] = "sensitivity — a build expected to exhibit the defect"
        SENSITIVITY.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(json.dumps(doc, indent=2))
        if doc["push_constant_lines"] == 0:
            print(
                "\nSENSITIVITY DID NOT FIRE — this build shows no unwritten push constants, so "
                "it cannot serve as the positive control.",
                file=sys.stderr,
            )
            return 1
        print(f"\nrecorded -> {SENSITIVITY}")
        return 0

    sens = None
    if SENSITIVITY.is_file():
        sens = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    now = measure()
    v, code, why = verdict(now, sens)
    now["verdict"] = v
    now["why"] = why
    now["sensitivity_record"] = (
        {k: sens.get(k) for k in ("dll_sha256_16", "push_constant_lines",
                                  "shortfall_bytes_observed")}
        if sens else None
    )
    (HERE / "push_constants_written.json").write_text(
        json.dumps(now, indent=2), encoding="utf-8"
    )
    print(json.dumps(now, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
