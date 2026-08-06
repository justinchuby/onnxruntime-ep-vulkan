"""M0 criteria 4 and 5 — the WITNESSED artifact, both polarities in one lane.

WHY THIS FILE EXISTS
====================
DESIGN.md §10, criteria 4 and 5, have read **"Partially met — mechanism landed, artifact
not seen"** since 2026-07-30.  The paired positive controls landed in
``test_claim_diagnostics.py``; what was missing is R10's falsifier itself:

  > The falsifier for "X is wired" is an observation of an artifact X produced, whose
  > content varies with X's input.

So this file produces the artifact.  One run, one binary per criterion-4 row, two binaries
for criterion 5, and a JSON record in ``bench/results/`` whose fields **differ between the
two polarities**.  A file in which the two rows are identical is a failing file.

THE TRAP THIS FILE IS BUILT AROUND — LINK'S ICD PROBE, 2026-08-01
=================================================================
The Windows lane's ICD-removal negative control matched the substring
``passed the §7.2 capability gate`` to decide whether suppression had taken.
``engine::loader_probe_report()`` prints that phrase on **every** run — the line is
``"{n} device(s) passed the §7.2 capability gate."`` and ``n`` is ``0`` when suppression
*did* work.  The control therefore short-circuited on every input it ever saw.  It had
never once fired, while the lane reported "the gate cannot fail".

A detector that fires on every input is not a detector, it is a constant.  The
consequence for this file is a rule:

  **The negative control does not get to assert that it fired.  It has to show it.**

Concretely: the criterion-4 negative row is only admissible when
``ci/check_icd_suppression.classify`` — Link's, not a second copy — returns
``suppressed`` for the report the *suppressed* child produced AND returns
``icd_suppression_ineffective`` for the report the *unsuppressed* child produced, in the
same run.  Two different tokens out of one classifier on two inputs is the R10 falsifier;
one token twice is a constant, whichever token it is.

R13 — THREE OUTCOMES, AND THE ONE CRITERION 4 EXPLICITLY ASKS FOR
==================================================================
The M0 table added this to criterion 4 on 2026-07-31:

  > the two outcomes must be spelled differently from the control's own outage, since
  > "advertised zero devices" and "the probe threw" are the same word in a summary line.

So every row carries a ``row_state`` of ``OBSERVED`` / ``ERROR(instrument)``, and a child
that crashed, timed out, or never printed its record is an **outage**: it produces
``InstrumentError``, never a zero device count.  Zero devices is a reading; no reading is
not a zero.

WHAT IS NOT CLAIMED HERE
========================
No duration.  §10.0 obligation 8: a device-clock figure is quotable only with a
device-state record, and this file records none.  Everything it quotes is a **count**, a
**token** or an **exact byte comparison** — quantities the machine's load cannot perturb
(§10.0.4).  The reader is not handed a count and left to supply a clock.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import _shaderless
import _verdict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "bench" / "results"

#: Marker the child prints its record behind.  A line, not a file, so a child that died
#: before writing leaves no half-written artifact to be misread as a reading.
RECORD_MARKER = "__CRITERION_WITNESS__"

ROW_OBSERVED = "OBSERVED"
ROW_ERROR = "ERROR(instrument)"

#: The line Link's broken probe matched.  Recorded per row as a **field**, never as a
#: gate: on a runner whose loader still enumerates an ICD with zero capable devices it is
#: printed in both polarities, which is the entire content of his finding.
GATE_LINE_PHRASE = "passed the §7.2 capability gate"

#: Criterion 4 requires the plugin to *log a warning*, not merely to stay quiet.
#: `factory.rs:258`.  It is a discriminator as well as a clause: the shader-full binary
#: with a working ICD never prints it.
ZERO_DEVICE_WARNING = "advertising zero devices"


def _load_link_classifier():
    """Import ``ci/check_icd_suppression.py``.  Link's file, used, not re-implemented.

    §10.0.1 R10 sub-rule: one mechanism, not two.  A second copy of this classification in
    ``tests/`` would drift from the one CI runs and the drift would be invisible.
    """
    path = REPO / "ci" / "check_icd_suppression.py"
    if not path.is_file():
        raise _verdict.InstrumentError(
            "[criterion 4 instrument failure] ERROR(instrument): "
            f"{path} does not exist, so there is no way to tell whether the ICD-removal "
            "negative control fired.  Without it a suppressed run and an unsuppressed run "
            "are the same reading and the row asserts nothing."
        )
    spec = importlib.util.spec_from_file_location("check_icd_suppression", path)
    if spec is None or spec.loader is None:
        raise _verdict.InstrumentError(
            f"[criterion 4 instrument failure] ERROR(instrument): cannot import {path}"
        )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_icd_suppression"] = mod
    spec.loader.exec_module(mod)
    return mod


def _epctl_for(lib: Path) -> Path:
    name = "epctl.exe" if sys.platform == "win32" else "epctl"
    return lib.parent / name


def _device_selector() -> str:
    return os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "")


def _run_row(*, row: str, lib: Path, extra_env: "dict[str, str]", quiet_seconds: float) -> dict:
    """Run one polarity in a child process and return its record.

    Raises ``InstrumentError`` if the child did not reach an observation.  It never
    returns a record with a fabricated zero.
    """
    env = dict(os.environ)
    env.update(extra_env)
    env["ONNXRUNTIME_VULKAN_EP_LIB"] = str(lib)
    # The child must not recurse into pytest collection.
    env["ONNXRUNTIME_EP_VULKAN_WITNESS_CHILD"] = "1"

    # ORT writes its profile into the process CWD and, on Windows, may still hold the file
    # open when the child exits — so the child runs in a scratch directory that is deleted
    # with it.  Probe output never lands in the repo (housekeeping ruling, 2026-08-01).
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix=f"witness_{row}_") as scratch:
        result = _verdict.run_subprocess_checked(
            [sys.executable, str(Path(__file__).resolve()), "--row", row, "--lib", str(lib)],
            what=f"criterion witness row {row!r}",
            quiet_seconds=quiet_seconds,
            env=env,
            cwd=scratch,
        )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    for line in combined.splitlines():
        if line.startswith(RECORD_MARKER):
            try:
                record = json.loads(line[len(RECORD_MARKER):])
            except ValueError as exc:
                raise _verdict.InstrumentError(
                    f"[criterion witness instrument failure] ERROR(instrument): row {row!r} "
                    f"printed an unparseable record: {exc}\nline: {line[:600]}"
                ) from exc
            record["row_state"] = ROW_OBSERVED
            record["child_exit_code"] = result.returncode
            # The EP's own warnings are emitted by Rust `log::warn!` through ORT's native
            # sink, which writes to the OS-level stderr of the child.  `redirect_stderr`
            # inside the child cannot see them — it only rebinds `sys.stderr`.  The first
            # cut of this file read the child's Python-level buffer, found it empty, and
            # correctly reported ERROR(instrument) rather than "the guard has stopped
            # reporting": the reason string was there, in a stream nobody was reading.
            record["child_stderr"] = (result.stderr or "")
            record["child_stdout"] = "\n".join(
                line for line in (result.stdout or "").splitlines()
                if not line.startswith(RECORD_MARKER)
            )
            return record

    raise _verdict.InstrumentError(
        f"[criterion witness instrument failure] ERROR(instrument): row {row!r} produced no "
        f"record (child exited {result.returncode}).  A child that crashed before reading "
        "the device count and a child that read zero devices are the SAME WORD in a "
        "summary line — DESIGN.md §10 criterion 4, amended 2026-07-31 — so this is an "
        "outage and asserts nothing about the EP.\n"
        f"stdout tail:\n{(result.stdout or '')[-1500:]}\n"
        f"stderr tail:\n{(result.stderr or '')[-1500:]}"
    )


def _write_artifact(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    selector = _device_selector() or "unset"
    path = RESULTS / f"{name}-dev{selector}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[WITNESS] wrote {path}", file=sys.stderr)
    return path


# ===========================================================================
# CRITERION 4 — no ICD -> zero devices, session on CPU; ICD -> non-zero, claims
# ===========================================================================


@pytest.mark.skipif(
    os.environ.get("ONNXRUNTIME_EP_VULKAN_WITNESS_CHILD") == "1",
    reason="child process: must not spawn another child",
)
@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set — no binary to witness",
)
def test_criterion4_icd_polarity_witness() -> None:
    """The criterion-4 artifact: one binary, two ICD worlds, one file.

    Closes the "artifact not seen" half of the row.  What the file has to show, and what
    each assertion below is for:

    * the **negative control actually fired** — Link's classifier returns ``suppressed``
      on the suppressed report.  Without this the row is his ``$probe -match`` bug;
    * the classifier returns a **different** token on the unsuppressed report.  A
      classifier that returned ``suppressed`` for both would satisfy the first assertion
      and detect nothing;
    * ``ep_devices_advertised`` is ``0`` with no ICD and ``> 0`` with one, from the same
      binary in the same run;
    * with no ICD the session still runs and its output is **bit-identical** to the
      expected values — criterion 4's "session runs on CPU" half;
    * ``vulkan_node_events`` is ``0`` with no ICD.  This is ORT's own profile, not ours.
    """
    link = _load_link_classifier()
    lib = Path(os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]).resolve()

    nonexistent = str(REPO / "does_not_exist" / "no_such_icd.json")
    # Two ways to take the ICD away, tried in order.  They are neutralised by different
    # things, and which one takes is itself the reading.
    #
    #   driver_search_path — VK_ICD_FILENAMES / VK_DRIVER_FILES / VK_ADD_DRIVER_FILES.
    #     The Vulkan loader documents all three as "ignored when running a Vulkan
    #     application with elevated privileges" (PLATFORMS.md §7.4.1).  On an elevated
    #     runner this arm cannot take, and the failure mode is not a bad answer but a
    #     control that never reached its subject.
    #   loader_driver_filter — VK_LOADER_DRIVERS_DISABLE, a filter rather than a search
    #     path (loader 1.3.234+).  The loader's own table attaches no elevation caveat to
    #     it: it can only remove a driver, never name one for the loader to load.
    #
    # Verified unelevated on a GPU box, 2026-08-04: the first arm takes there and this
    # test passes, so the Windows-lane red does not reproduce on a real device.  Whether
    # the second arm is what rescues an ELEVATED runner is not read here — that box could
    # not be elevated non-interactively — so the mechanism that took is RECORDED in the
    # artifact rather than predicted.
    suppression_arms = [
        (
            "driver_search_path",
            {
                "VK_ICD_FILENAMES": nonexistent,
                "VK_DRIVER_FILES": nonexistent,
                "VK_ADD_DRIVER_FILES": "",
            },
        ),
        (
            "loader_driver_filter",
            {
                "VK_ICD_FILENAMES": nonexistent,
                "VK_DRIVER_FILES": nonexistent,
                "VK_ADD_DRIVER_FILES": "",
                "VK_LOADER_DRIVERS_DISABLE": "*",
            },
        ),
    ]

    records: dict[str, dict] = {}
    suppression_attempts: list[dict] = []

    record = _run_row(row="icd_present", lib=lib, extra_env={}, quiet_seconds=90)
    records["icd_present"] = record

    for arm_name, extra in suppression_arms:
        record = _run_row(row="icd_suppressed", lib=lib, extra_env=extra, quiet_seconds=90)
        record["suppression_mechanism"] = arm_name
        state = link.classify(record["loader_probe_report"])["state"]
        suppression_attempts.append({"mechanism": arm_name, "state": state})
        records["icd_suppressed"] = record
        if state == link.STATE_SUPPRESSED:
            break

    for row, record in records.items():
        verdict = link.classify(record["loader_probe_report"])
        record["icd_suppression_state"] = verdict["state"]
        record["icd_suppression_detail"] = verdict.get("detail", "")
        record["devices_passing_gate"] = verdict.get("devices_passing_gate")
        record["gate_line_present"] = GATE_LINE_PHRASE in record["loader_probe_report"]
        # Criterion 4's third clause — "logs a warning" — read from the child's OS-level
        # stderr, which is where the EP's Rust `log::warn!` actually lands.
        child_text = record["child_stderr"] + record["child_stdout"]
        record["zero_device_warning_emitted"] = ZERO_DEVICE_WARNING in child_text
        # The report is large and the record is the artifact; keep the discriminating
        # tail, not the whole probe dump.
        record["loader_probe_report"] = record["loader_probe_report"][-1200:]
        record["child_stderr"] = record["child_stderr"][-2000:]
        record["child_stdout"] = record["child_stdout"][-800:]
        record["ep_log"] = record["ep_log"][-800:]
        records[row] = record

    neg = records["icd_suppressed"]
    pos = records["icd_present"]

    payload = {
        "criterion": 4,
        "criterion_text": (
            "A machine with no Vulkan ICD loads the plugin, advertises zero devices, logs "
            "a warning, and the session still runs on CPU."
        ),
        "device_selector": _device_selector() or "unset",
        "library": str(lib),
        "library_sha_prefix": _sha_prefix(lib),
        "negative_control_fired": neg["icd_suppression_state"] == link.STATE_SUPPRESSED,
        "suppression_mechanism": neg.get("suppression_mechanism"),
        "suppression_attempts": suppression_attempts,
        "classifier": "ci/check_icd_suppression.py::classify (Link's; not re-implemented)",
        "rows": records,
        "no_duration_quoted": (
            "§10.0 obligation 8: this artifact contains no device-clock or wall-clock "
            "figure. Every field is a count, a token or an exact comparison (§10.0.4)."
        ),
    }
    path = _write_artifact("criterion4_icd_witness", payload)

    # ── 1. The negative control fired.  Link's lesson, first, because everything below
    #       it is vacuous if this is false. ───────────────────────────────────────────
    if neg["icd_suppression_state"] != link.STATE_SUPPRESSED:
        raise _verdict.InstrumentError(
            "[criterion 4 instrument failure] ERROR(instrument): the ICD-removal negative "
            f"control did not fire — classifier says {neg['icd_suppression_state']!r}.\n"
            f"{neg['icd_suppression_detail']}\n"
            "attempts: "
            + "; ".join(f"{a['mechanism']}={a['state']}" for a in suppression_attempts)
            + "\n"
            "The control did not reach its observation, so THIS IS NOT A CRITERION-4 "
            "FAILURE and it is not a pass either (R13).  On Windows the usual cause is "
            "PLATFORMS.md §7.4.1: the LunarG loader ignores VK_DRIVER_FILES / "
            "VK_ICD_FILENAMES in elevated processes.  Run the lane unelevated, or "
            "unregister the ICD from HKLM\\SOFTWARE\\Khronos\\Vulkan\\Drivers.\n"
            "If `loader_driver_filter` is among the attempts above and also failed, then "
            "elevation is NOT the explanation: VK_LOADER_DRIVERS_DISABLE is a filter, not "
            "a search path, and the loader's own table attaches no elevation caveat to "
            "it.  Read the attempt list before reaching for §7.4.1.\n"
            f"artifact: {path}"
        )

    # ── 2. And it is not a constant.  Same classifier, other input, other token. ─────
    assert pos["icd_suppression_state"] != neg["icd_suppression_state"], (
        "Criterion 4 FAILS as a witness: the suppression classifier returned "
        f"{neg['icd_suppression_state']!r} for BOTH polarities.  A detector that fires on "
        "every input is not a detector, it is a constant — the defect Link found in the "
        "Windows lane's own ICD probe on 2026-08-01, reproduced here.  R10's falsifier is "
        "an artifact whose content VARIES with its input.\n"
        f"artifact: {path}"
    )

    # ── 3. Zero devices with no ICD; non-zero with one.  Same binary, same run. ──────
    assert neg["ep_devices_advertised"] == 0, (
        "Criterion 4 negative FAILS: with no usable ICD the EP advertised "
        f"{neg['ep_devices_advertised']} device(s).  §2.3 and §7.8: probe_devices() must "
        "return an empty list rather than present itself as a working EP.\n"
        f"artifact: {path}"
    )
    assert pos["ep_devices_advertised"] > 0, (
        "Criterion 4 positive FAILS: with the ICD present the EP advertised zero devices, "
        "so the pair has no polarity — an EP that always advertises zero passes the "
        "negative perfectly.\n"
        f"artifact: {path}"
    )

    # ── 4. The session still ran, and it ran correctly, on CPU. ─────────────────────
    assert neg["output_exact_match"] is True, (
        "Criterion 4 negative FAILS: with no ICD the session did not produce the exact "
        f"expected output.  observed={neg.get('output')} expected={neg.get('expected')}\n"
        f"artifact: {path}"
    )
    assert neg["vulkan_node_events"] == 0, (
        "Criterion 4 negative FAILS: ORT's own profile shows "
        f"{neg['vulkan_node_events']} VulkanExecutionProvider node event(s) on a machine "
        "with no usable ICD.  The EP claimed work it cannot have executed.\n"
        f"artifact: {path}"
    )

    # ── 5. "logs a warning" is a clause of the criterion, not a nicety — and it is a
    #       discriminator: the working binary must NOT print it. ─────────────────────
    assert neg["zero_device_warning_emitted"] is True, (
        "Criterion 4 negative FAILS: the plugin advertised zero devices and said nothing "
        f"about it.  The criterion reads 'advertises zero devices, LOGS A WARNING, and the "
        f"session still runs on CPU'.  A silent zero and a broken probe are the same "
        f"reading to the user who has to debug it.\n"
        f"child stderr tail:\n{neg['child_stderr'][-800:]}\nartifact: {path}"
    )
    assert pos["zero_device_warning_emitted"] is False, (
        "Criterion 4 FAILS as a witness: the ICD-present binary also logged the "
        "zero-device warning, so the warning is printed unconditionally and discriminates "
        "nothing.\n"
        f"artifact: {path}"
    )
    assert pos["vulkan_node_events"] > 0, (
        "Criterion 4 positive FAILS: with the ICD present ORT's profile shows zero "
        "VulkanExecutionProvider node events, so 'ran on CPU' is not distinguishable from "
        "'ran on CPU because the EP never works'.\n"
        f"artifact: {path}"
    )

    print(
        f"\n[CRITERION 4 WITNESS] {path.name}\n"
        f"  icd_present    : suppression={pos['icd_suppression_state']} "
        f"devices={pos['ep_devices_advertised']} vulkan_node_events={pos['vulkan_node_events']} "
        f"gate_line_present={pos['gate_line_present']} warn={pos['zero_device_warning_emitted']}\n"
        f"  icd_suppressed : suppression={neg['icd_suppression_state']} "
        f"devices={neg['ep_devices_advertised']} vulkan_node_events={neg['vulkan_node_events']} "
        f"gate_line_present={neg['gate_line_present']} warn={neg['zero_device_warning_emitted']} "
        f"output_exact_match=True\n"
        "  negative control FIRED (classifier returned two different tokens on two inputs).\n"
        f"  NOTE: gate_line_present is {pos['gate_line_present']}/{neg['gate_line_present']} "
        "— the phrase Link's probe matched is NOT the discriminator here either.",
        file=sys.stderr,
    )


# ===========================================================================
# CRITERION 5 — shader-less build advertises zero devices and claims nothing
# ===========================================================================


@pytest.mark.skipif(
    os.environ.get("ONNXRUNTIME_EP_VULKAN_WITNESS_CHILD") == "1",
    reason="child process: must not spawn another child",
)
@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set — no binary to witness",
)
def test_criterion5_shaderless_polarity_witness() -> None:
    """The criterion-5 artifact: two binaries from one source tree, one file.

    The discriminator is the **reason string**, which criterion 5 names explicitly and
    which R13 requires on the failing path: ``built without shaders``.  The shader-full
    binary never prints it, and a build that failed for an unrelated reason never gets far
    enough to print it — which is the "distinguishable from a build that failed for an
    unrelated reason" clause, discharged by the artifact rather than by argument.

    Building the shader-less binary is an instrument step: every way it can fail raises
    ``InstrumentError`` (see ``_shaderless.py``), because a binary that was not built has
    said nothing about criterion 5.
    """
    full_lib = Path(os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]).resolve()
    sl_lib, _sl_epctl = _shaderless.build()

    rows = {
        "shaders_compiled": full_lib,
        "shaderless": sl_lib,
    }
    records: dict[str, dict] = {}
    for row, lib in rows.items():
        record = _run_row(row=row, lib=lib, extra_env={}, quiet_seconds=90)
        text = (
            record["ep_log"]
            + record["child_stderr"]
            + record["child_stdout"]
            + record["loader_probe_report"]
        )
        record["shaderless_reason_emitted"] = _shaderless.SHADERLESS_REASON in text
        record["loader_probe_report"] = record["loader_probe_report"][-800:]
        record["ep_log"] = record["ep_log"][-1500:]
        record["child_stderr"] = record["child_stderr"][-2000:]
        record["child_stdout"] = record["child_stdout"][-800:]
        records[row] = record

    neg = records["shaderless"]
    pos = records["shaders_compiled"]

    payload = {
        "criterion": 5,
        "criterion_text": (
            "A shader-less build advertises zero devices and claims nothing, with a "
            "'built without shaders' reason (§7.8 condition 3)."
        ),
        "device_selector": _device_selector() or "unset",
        "shader_full_library": str(full_lib),
        "shader_full_sha_prefix": _sha_prefix(full_lib),
        "shaderless_library": str(sl_lib),
        "shaderless_sha_prefix": _sha_prefix(sl_lib),
        "negative_control_fired": neg["shaderless_reason_emitted"],
        "allow_missing_glslc_route": (
            "UNREACHABLE on this host: build.rs::installed_sdk_glslc() scans C:\\VulkanSDK "
            "unconditionally (build.rs:175-198) and honours no environment variable, so "
            "ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1 (build.rs:247) is never consulted "
            "where the SDK is installed. The shader-less artifact here is built from an "
            "empty shader source set, which reaches build.rs's other write_shader_modules"
            "(out_dir, &[]) return. OWED TO TANK: make the env var a hard override checked "
            "BEFORE the search, so this lane can be entered deliberately."
        ),
        "rows": records,
        "no_duration_quoted": (
            "§10.0 obligation 8: no device-clock or wall-clock figure appears in this "
            "artifact."
        ),
    }
    path = _write_artifact("criterion5_shaderless_witness", payload)

    # ── 1. The negative control fired — the artifact said WHY it claims nothing. ────
    if not neg["shaderless_reason_emitted"]:
        raise _verdict.InstrumentError(
            "[criterion 5 instrument failure] ERROR(instrument): the shader-less binary "
            f"never emitted the reason string {_shaderless.SHADERLESS_REASON!r}.  Either "
            "the build is not actually shader-less — in which case the negative control "
            "has no subject — or §7.8 condition 3's guard has stopped reporting on the "
            "failing path.  Those are different findings with different owners and this "
            "check cannot tell them apart, so it reports neither.\n"
            f"ep_log tail:\n{neg['ep_log']}\nartifact: {path}"
        )

    # ── 2. And the reason is not printed unconditionally. ──────────────────────────
    assert not pos["shaderless_reason_emitted"], (
        "Criterion 5 FAILS as a witness: the SHADER-FULL binary also emitted "
        f"{_shaderless.SHADERLESS_REASON!r}.  A reason string printed on every run "
        "discriminates nothing — Link's ICD probe, exactly.\n"
        f"artifact: {path}"
    )

    # ── 3. Zero devices, and non-zero from the same source tree with shaders. ──────
    assert neg["ep_devices_advertised"] == 0, (
        "Criterion 5 FAILS: the shader-less build advertised "
        f"{neg['ep_devices_advertised']} device(s).  §7.8 condition 3: a shader-less "
        "artifact must not present itself as a working EP; it must never load, claim, and "
        "then fail at pipeline creation.\n"
        f"artifact: {path}"
    )
    assert pos["ep_devices_advertised"] > 0, (
        "Criterion 5 positive FAILS: the shader-full binary advertised zero devices, so "
        "the pair has no polarity.\n"
        f"artifact: {path}"
    )

    # ── 4. Claims nothing, and the session still produces the right answer. ────────
    assert neg["vulkan_node_events"] == 0, (
        "Criterion 5 FAILS: the shader-less build claimed "
        f"{neg['vulkan_node_events']} node(s).  It has no pipelines to run them with.\n"
        f"artifact: {path}"
    )
    assert neg["output_exact_match"] is True, (
        "Criterion 5 FAILS: the shader-less build did not fall back cleanly — the session "
        f"output is wrong.  observed={neg.get('output')} expected={neg.get('expected')}\n"
        f"artifact: {path}"
    )
    assert pos["vulkan_node_events"] > 0, (
        "Criterion 5 positive FAILS: the shader-full binary claimed nothing either, so "
        "'claims nothing' says nothing about the shaders.\n"
        f"artifact: {path}"
    )

    print(
        f"\n[CRITERION 5 WITNESS] {path.name}\n"
        f"  shaders_compiled : devices={pos['ep_devices_advertised']} "
        f"vulkan_node_events={pos['vulkan_node_events']} "
        f"reason_emitted={pos['shaderless_reason_emitted']}\n"
        f"  shaderless       : devices={neg['ep_devices_advertised']} "
        f"vulkan_node_events={neg['vulkan_node_events']} "
        f"reason_emitted={neg['shaderless_reason_emitted']} output_exact_match=True\n"
        "  negative control FIRED (reason string present in one polarity, absent in the other).",
        file=sys.stderr,
    )


def _sha_prefix(path: Path) -> str:
    import hashlib

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "UNREADABLE"


# ===========================================================================
# THE CHILD
# ===========================================================================
# One process per row.  It must be a child because (a) `VK_DRIVER_FILES` is read by the
# Vulkan loader at instance creation and this suite has already loaded one, and (b) ORT
# registers an EP library process-wide by name, so two libraries under one name cannot
# coexist.  Both are properties of the world, not workarounds.


def _child_main(row: str, lib_path: str) -> int:
    import contextlib
    import io

    import numpy as np
    import onnxruntime as ort
    from onnx_ir import DataType as DT

    import _models as m

    lib = Path(lib_path)
    record: dict = {"row": row, "library": str(lib)}

    epctl = _epctl_for(lib)
    if epctl.is_file():
        # encoding="utf-8" is not a style choice: `instance.rs` prints the gate-verdict
        # line through a literal "§" (U+00A7, 2 UTF-8 bytes: 0xC2 0xA7), and Python's
        # `text=True` with no explicit encoding decodes child stdout with
        # `locale.getpreferredencoding()` — cp1252 on an English Windows runner, which
        # turns those two bytes into "Â§" and breaks every regex in
        # `ci/check_icd_suppression.py` that looks for "§7.2 capability gate", on BOTH
        # polarities (see run 31094738484, job 92593900456: both suppression arms came
        # back `probe_report_unreadable`, not because suppression failed but because the
        # healthy report was silently mis-decoded). `errors="replace"` keeps this an
        # instrument-error rather than a hard crash on any other undecodable byte.
        probe = subprocess.run(
            [str(epctl), "--probe-loader"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=180,
        )
        record["loader_probe_report"] = (probe.stdout or "") + (probe.stderr or "")
        record["epctl_exit_code"] = probe.returncode
    else:
        record["loader_probe_report"] = ""
        record["epctl_exit_code"] = None

    log = io.StringIO()
    devices = 0
    node_events = 0
    output: list = []
    expected = [4.0, 6.0]
    exact = False

    with contextlib.redirect_stderr(log):
        try:
            ort.register_execution_provider_library(m.EP_NAME, str(lib))
        except Exception as exc:  # noqa: BLE001
            record["register_error"] = f"{type(exc).__name__}: {exc}"
        devices = len([d for d in ort.get_ep_devices() if d.ep_name == m.EP_NAME])

        model = m.make_model(
            "Add",
            [m.tensor("a", DT.FLOAT, [2]), m.tensor("b", DT.FLOAT, [2])],
            [m.tensor("out", DT.FLOAT, [2])],
        )
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.enable_profiling = True
        opts.profile_file_prefix = f"_witness_{row}"
        sess = ort.InferenceSession(
            model, opts, providers=[m.EP_NAME, "CPUExecutionProvider"]
        )
        result = sess.run(None, {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            "b": np.array([3.0, 4.0], dtype=np.float32),
        })
        profile_path = sess.end_profiling()
        output = [float(v) for v in result[0]]
        # Exact, not approximate: an Add of exactly representable floats has one right
        # answer and a tolerance here would hide a wrong one (§10.0.4 — the unperturbable
        # quantity is the claim of record).
        exact = bool(np.array_equal(result[0], np.array(expected, dtype=np.float32)))

        try:
            with open(profile_path) as fh:
                events = json.load(fh)
            node_events = sum(
                1 for e in events
                if e.get("cat") == "Node"
                and isinstance(e.get("args"), dict)
                and e["args"].get("provider") == m.EP_NAME
            )
        finally:
            with contextlib.suppress(OSError):
                os.remove(profile_path)

    record["ep_devices_advertised"] = devices
    record["vulkan_node_events"] = node_events
    record["output"] = output
    record["expected"] = expected
    record["output_exact_match"] = exact
    record["ep_log"] = log.getvalue()
    record["env_vk_driver_files"] = os.environ.get("VK_DRIVER_FILES", "")
    record["env_vk_icd_filenames"] = os.environ.get("VK_ICD_FILENAMES", "")

    print(RECORD_MARKER + json.dumps(record))
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--row", required=True)
    parser.add_argument("--lib", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(_child_main(args.row, args.lib))
