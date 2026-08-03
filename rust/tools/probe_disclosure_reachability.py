#!/usr/bin/env python
"""What a user is actually told at session creation, measured rather than derived (RAI-013).

WHAT IS BEING FALSIFIED
-----------------------
"The user was told" is a claim about a *channel*, not about an emission. The §8.9.7 disclosure's
INFO half is emitted correctly and the artifact reports ``session_disclosure_infos: 1 /
..._to_ort_sink: 0 / BELOW_ORT_THRESHOLD`` — honest, and a statement that the record was refused by
ORT's threshold. It says nothing about whether the user saw it, because ORT's sink is not the only
thing attached to a running process.

This probe measures the bytes a console receives from the shipping path, under the settings a user
who sets nothing gets. Nothing here is inferred from a severity comparison.

ARMS
----
* **ARM 1 — PURE.** Every ``ONNXRUNTIME_EP_VULKAN_*`` and probe variable is stripped from the
  child's environment, ORT's severity is left at whatever ORT defaults to (the child does not call
  ``set_default_logger_severity``), and the session is created the ordinary way. Its stdout and
  stderr are captured separately and classified. **No counters file is requested**, because the
  counters file is itself an environment variable and this arm's whole claim is that it set none.
  What this arm can say is exactly what a console would render.
* **ARM 2 — INSTRUMENTED.** Identical, plus ``ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE`` so the same
  run's internal account can be read and lined up against arm 1's console. The one added variable
  is named in the artifact. If arm 1 and arm 2 disagree about what appeared on the console, the
  instrument changed the measurement and the artifact says so.
* **ARM 3 — ESCALATION (PLANTED).** The same shipping path with
  ``ONNXRUNTIME_EP_VULKAN_FORCE_STDERR_FAILURE`` armed, which makes every write to this process's
  stderr report failure without writing — the host this box cannot otherwise produce (a Windows
  GUI process, a service with no console, a redirect to a full disk). With ORT's default threshold
  refusing INFO *and* stderr refusing the bytes, no quiet channel is left, and the disclosure must
  arrive at WARNING on ORT's own sink. This is the arm that makes the repair falsifiable: without
  it, "the escalation fires when nothing else carries the record" is a sentence, not a run. The
  run is marked ``"stderr_fault_injection": "ACTIVE"`` in its own artifact, so a refused write can
  never be quoted as a suffered one.

  Under Link's PLANTED/OBSERVED axis this arm is PLANTED and arms 1 and 2 are OBSERVED.

CLASSIFICATION
--------------
Every captured line lands in exactly one bucket:

* ``ort`` — carries ORT's own decoration ``[X:onnxruntime:...]``: this is ORT's sink.
* ``ep_stderr`` — carries this crate's ``[vulkan-ep] LEVEL:`` prefix: this is the EP's own
  unconditional stderr line.
* ``other`` — anything else (the child's own prints, Python warnings).

A disclosure is **reachable** in a bucket when a line in that bucket contains the §8.9.7 marker and
the content that makes it a disclosure. Buckets are counted separately and never summed into one
"was it seen" boolean: "ORT dropped it and stderr carried it" and "both carried it" are different
findings about the channel.

EXIT CODES (R13)
----------------
``0`` PASS · ``1`` FAIL(condition) · ``4`` ERROR(instrument).
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

ENV_COUNTERS = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
ENV_STDERR_FAULT = "ONNXRUNTIME_EP_VULKAN_FORCE_STDERR_FAILURE"
SECTION_MARKER = "8.9.7"
PROVEN_MARKER = "proven by"
UNATTRIBUTED_MARKER = "UNATTRIBUTED"
ESCALATION_MARKER = "delivery escalation"
ORT_DECORATION = re.compile(r"\[[VIWEF]:onnxruntime:")
EP_STDERR = re.compile(r"\[vulkan-ep\] (ERROR|WARN|INFO|DEBUG|TRACE):")

sys.path.insert(0, str(REPO / "rust" / "tools"))
from probe_broken_commitment import decode_both  # noqa: E402

PROVEN_CASE = "add_f32"


def child(device_index: int, case: str) -> int:
    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, str(REPO / "rust" / "tools"))
    import ledger_case_models as cases

    out_dir = REPO / "rust" / "target" / "probe_disclosure_reachability"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = cases.build(case, out_dir)

    # Deliberately no `set_default_logger_severity` and no `log_severity_level`: this arm is about
    # the settings a user who sets nothing gets, and setting either would make it about mine.
    ort.register_execution_provider_library("VulkanExecutionProvider", str(LIB))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "VulkanExecutionProvider"]
    if device_index >= len(devices):
        print(f"CHILD-ERROR: device {device_index} not advertised ({len(devices)} available)")
        return 4
    so = ort.SessionOptions()
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


def scrubbed_env() -> dict:
    """The environment of a user who set nothing this project owns."""
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("ONNXRUNTIME_EP_VULKAN_") or k.startswith("_PROBE_"):
            env.pop(k)
    env.pop("RUST_LOG", None)
    env.pop("ORT_LOGGING_LEVEL", None)
    return env


def run_child(device_index: int, tag: str, extra_env: dict) -> dict:
    env = scrubbed_env()
    env.update(extra_env)
    proc = subprocess.run(
        [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--child",
            "--device",
            str(device_index),
            "--case",
            PROVEN_CASE,
        ],
        capture_output=True,
        env=env,
    )
    return {
        "tag": tag,
        "returncode": proc.returncode,
        "stdout": decode_both(proc.stdout or b""),
        "stderr": decode_both(proc.stderr or b""),
        "env_added": sorted(extra_env),
    }


def classify(text: str) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"ort": [], "ep_stderr": [], "other": []}
    for ln in text.splitlines():
        if not ln.strip():
            continue
        if ORT_DECORATION.search(ln):
            buckets["ort"].append(ln)
        elif EP_STDERR.search(ln):
            buckets["ep_stderr"].append(ln)
        else:
            buckets["other"].append(ln)
    return buckets


def trim(lines: list[str], limit: int = 4) -> list[str]:
    out = []
    for ln in lines[:limit]:
        m = ORT_DECORATION.search(ln) or EP_STDERR.search(ln)
        out.append((ln[m.start():] if m else ln).replace("\x1b[m", "").strip()[:400])
    return out


def disclosure_lines(lines: list[str]) -> list[str]:
    """Lines that are a §8.9.7 disclosure of what this session claimed and what proved it.

    Matched on ASCII substrings of the message itself, never on the section marker: ``§`` is
    multi-byte, ``decode_both`` deliberately reads the same bytes four ways, and the first draft of
    this function keyed on ``"§8.9.7"`` — which the DEVICE-UNATTRIBUTED branch does not carry (it
    is filed under ``[§10.0.1 R12]``). That draft reported **zero disclosure lines on every
    channel** and printed "The user was not told" on a run whose console had the whole disclosure
    on it. A probe that asserts the defect is not a measurement, and this one nearly was.
    """
    return [
        ln
        for ln in lines
        if "session claims" in ln
        and (PROVEN_MARKER in ln or "all work runs on the CPU EP" in ln)
    ]


def escalation_lines(lines: list[str]) -> list[str]:
    """The disclosure re-emitted at WARNING because no quiet channel would carry it."""
    return [ln for ln in lines if ESCALATION_MARKER in ln]


def measure(device_index: int) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    report: dict = {"device": device_index, "ort_version": None}
    try:
        import onnxruntime as ort

        report["ort_version"] = ort.__version__
    except Exception:  # pragma: no cover - instrument
        pass

    pure = run_child(device_index, "pure", {})
    counters_path = RESULTS / f"disclosure-reachability-dev{device_index}-counters.json"
    counters_path.unlink(missing_ok=True)
    inst = run_child(device_index, "instrumented", {ENV_COUNTERS: str(counters_path)})
    doc = json.loads(counters_path.read_text()) if counters_path.is_file() else {}

    esc_path = RESULTS / f"disclosure-reachability-dev{device_index}-escalation.json"
    esc_path.unlink(missing_ok=True)
    esc = run_child(
        device_index,
        "escalation",
        {ENV_COUNTERS: str(esc_path), ENV_STDERR_FAULT: "1"},
    )
    esc_doc = json.loads(esc_path.read_text()) if esc_path.is_file() else {}

    for arm in (pure, inst, esc):
        if arm["returncode"] == 4 or "CHILD-ERROR" in arm["stdout"]:
            return (
                "ERROR",
                [f"ERROR(instrument): arm {arm['tag']} could not run: {arm['stderr'][-600:]!r}"],
                report,
            )
        if arm["returncode"] != 0:
            return (
                "ERROR",
                [
                    f"ERROR(instrument): arm {arm['tag']} exited {arm['returncode']}: "
                    f"{arm['stderr'][-600:]!r}"
                ],
                report,
            )

    for arm in (pure, inst, esc):
        console = classify(arm["stdout"] + "\n" + arm["stderr"])
        seen = {k: disclosure_lines(v) for k, v in console.items()}
        report[arm["tag"]] = {
            "env_added": arm["env_added"],
            "lines_total": {k: len(v) for k, v in console.items()},
            "disclosure_lines": {k: len(v) for k, v in seen.items()},
            "escalation_lines": {k: len(escalation_lines(v)) for k, v in console.items()},
            "ort_disclosure_sample": trim(seen["ort"]),
            "ep_stderr_disclosure_sample": trim(seen["ep_stderr"]),
            "ort_lines_sample": trim(console["ort"]),
            "ep_stderr_all": trim(console["ep_stderr"], limit=20),
        }

    keys = (
        "session_disclosures",
        "session_disclosure_infos",
        "session_disclosure_infos_to_ort_sink",
        "session_disclosure_info_channel",
        "session_disclosure_infos_to_stderr",
        "session_disclosure_infos_reached_user",
        "session_disclosure_info_escalations",
        "session_disclosure_stderr_failures",
        "session_disclosure_info_reach",
        "stderr_fault_injection",
        "ort_sink_severity_threshold",
        "claimed_form_evidence",
        "claimed_forms_proven",
        "claimed_forms_device_unattributed",
    )
    report["counters"] = {k: doc.get(k) for k in keys}
    report["counters_escalation_arm"] = {k: esc_doc.get(k) for k in keys}

    # ---- The measurement, stated as conditions ----------------------------------------------
    # (1) Non-vacuity. A run that claimed nothing tells us nothing about a disclosure channel.
    backed = (report["counters"].get("claimed_forms_proven") or 0) + (
        report["counters"].get("claimed_forms_device_unattributed") or 0
    )
    if not backed:
        failures.append(
            "the instrumented arm claimed no proof-backed form, so no proven-forms INFO was due "
            "and its presence or absence on any channel measures nothing."
        )
        return "FAIL", failures, report

    # (2) Blindness control. If no ORT-decorated line was captured at all, this witness cannot see
    # ORT's sink and its silence about the INFO is not evidence about the INFO.
    if report["pure"]["lines_total"]["ort"] == 0:
        failures.append(
            "ERROR-SHAPED CONDITION: the pure arm captured zero ORT-decorated lines, so the "
            "witness is blind to ORT's sink and cannot report on delivery there."
        )

    # (3) The claim under test: with nothing set, does the disclosure reach the console at all?
    pure_seen = report["pure"]["disclosure_lines"]
    if pure_seen["ort"] == 0 and pure_seen["ep_stderr"] == 0:
        failures.append(
            "with no environment variable set and ORT at its default severity, no §8.9.7 "
            "disclosure of what proved the claimed forms reached the console on any channel. "
            "The user was not told."
        )

    # (4) The instrument must not be the thing that made it visible.
    if report["pure"]["disclosure_lines"] != report["instrumented"]["disclosure_lines"]:
        failures.append(
            "the pure and instrumented arms disagree about what reached the console "
            f"({report['pure']['disclosure_lines']} vs "
            f"{report['instrumented']['disclosure_lines']}), so the counters variable changed the "
            "measurement and the instrumented arm's account is not about the shipping path."
        )

    # (5) The counter must report the reachability it was measured to have. This is the half that
    #     `to_ort_sink: 0` could never say: on this host ORT's threshold refuses the INFO and the
    #     console prints it, and an artifact that reports only the first is telling the truth to
    #     nobody.
    reach = report["counters"].get("session_disclosure_info_reach")
    console_had_it = pure_seen["ort"] + pure_seen["ep_stderr"] > 0
    if console_had_it and reach != "REACHED_USER":
        failures.append(
            f"the console carried the disclosure and the artifact reports reach={reach!r}. The "
            "counter must be able to say the user was reachable, or it is measuring the channel "
            "nobody was on."
        )
    if not console_had_it and reach == "REACHED_USER":
        failures.append(
            "the artifact claims the disclosure reached the user and no line carrying it was "
            "captured on any channel — a delivery nobody witnessed."
        )
    if report["counters"].get("session_disclosure_stderr_failures"):
        failures.append(
            "the shipping path could not write a disclosure to stderr on a run where nothing was "
            f"planted: {report['counters'].get('session_disclosure_stderr_failures')!r} refused "
            "write(s). That is a real reachability defect, not an instrument error."
        )

    # ---- ARM 3: the escalation, on the shipping binary ---------------------------------------
    esc_report = report["escalation"]
    esc_counters = report["counters_escalation_arm"]
    if esc_counters.get("stderr_fault_injection") != "ACTIVE":
        failures.append(
            "ERROR-SHAPED CONDITION: the escalation arm's artifact does not report the planted "
            f"stderr failure as ACTIVE ({esc_counters.get('stderr_fault_injection')!r}), so this "
            "arm did not run the condition it names and its result cannot be read either way."
        )
    else:
        # Liveness of the plant: with the fault armed, the EP's own stderr lines must be *gone*.
        # Without this the arm would pass on a build where the injection did nothing and the
        # escalation fired for some other reason.
        if esc_report["lines_total"]["ep_stderr"] != 0:
            failures.append(
                "the planted stderr failure is armed and this crate still wrote "
                f"{esc_report['lines_total']['ep_stderr']} line(s) to stderr, so the arm is not "
                "in the state it claims to be testing."
            )
        # Blindness control: ORT's sink must still be capturable, or the arm cannot see the
        # escalation whether or not it fired.
        if esc_report["lines_total"]["ort"] == 0:
            failures.append(
                "ERROR-SHAPED CONDITION: the escalation arm captured no ORT-decorated line at "
                "all, so it is blind to the channel the escalation travels on."
            )
        elif esc_report["escalation_lines"]["ort"] == 0:
            failures.append(
                "no quiet channel could carry the §8.9.7 disclosure and nothing was re-emitted on "
                "ORT's sink at WARNING. A user on a host with no console is told nothing at all."
            )
        if esc_counters.get("session_disclosure_info_reach") != "ESCALATED_TO_WARNING":
            failures.append(
                "the escalation arm's reach token reads "
                f"{esc_counters.get('session_disclosure_info_reach')!r}, expected "
                "'ESCALATED_TO_WARNING'. The token must name which of the four states the run was "
                "in; reading REACHED_USER here would credit a quiet channel that refused, and "
                "UNREACHABLE would deny a delivery that happened."
            )
        if not esc_counters.get("session_disclosure_stderr_failures"):
            failures.append(
                "the escalation arm refused every stderr write and counted none of them. A "
                "channel that can fail and does not count its failures is a statement about the "
                "branch that ran, not about the channel."
            )
        # R10: the artifact must vary with its input. If the escalation arm's tokens read the same
        # as the default arm's, this probe is reporting a constant.
        if esc_counters.get("session_disclosure_info_reach") == report["counters"].get(
            "session_disclosure_info_reach"
        ):
            failures.append(
                "the reach token reads the same with the console alive and with it dead "
                f"({reach!r}), so it does not vary with its input."
            )

    return ("FAIL" if failures else "PASS"), failures, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--case", default=PROVEN_CASE)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()
    if args.child:
        return child(args.device, args.case)

    if not LIB.is_file():
        print(f"ERROR(instrument): {LIB} not built")
        return 4

    verdict, failures, report = measure(args.device)
    report["verdict"] = verdict
    report["failures"] = failures
    out = args.out or RESULTS / f"disclosure-reachability-dev{args.device}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nVERDICT: {verdict}  ->  {out}")
    return {"PASS": 0, "FAIL": 1, "ERROR": 4}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
