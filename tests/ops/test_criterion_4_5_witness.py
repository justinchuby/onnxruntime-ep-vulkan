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
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest

import _registry_suppression
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
# THE SUPPRESSION ARMS, AND THE ONE THING THAT MUST NEVER HAPPEN TO THEM
# ===========================================================================
# An arm is a *pair*: something to arm (an environment the child must carry, or a
# registry entry to flip) and a child run made while it is armed.  PR #45's rejected
# revision (Morpheus, 2026-08-06, head e2b7e00) broke that pair: it built the arm's
# environment, wrapped it in `contextlib.nullcontext(arm)`, entered the context WITHOUT
# binding its result, and then spawned the child with a hard-coded `extra_env={}`.  Both
# env-var arms became dead code — they ran a child in an unmodified world and reported
# `icd_suppression_ineffective` because nothing had been suppressed, not because the
# loader had refused the variables.  The reading looked like evidence and was not.
#
# The rule this section encodes: **entering an arm must hand back the environment that
# arm intends, and that value — not a literal — is what reaches the child.**  Every arm
# yields a `dict[str, str]`, including the registry arms, whose intended environment is
# genuinely empty and says so explicitly.  A control that fails the moment an arm's
# environment stops reaching the child lives at the bottom of this file.


def _format_attempts(attempts: "list[dict]") -> str:
    """One line per arm: what it was, what it delivered, and what came back.

    ``env_applied`` is printed because an arm that delivered nothing did not lose a
    comparison — it never entered one (Morpheus's Blocker 2, PR #45).  A reader of a CI
    log must be able to tell those apart without downloading the artifact.
    """
    parts = []
    for a in attempts:
        env = a.get("env_applied")
        if env is None:
            delivered = "env=n/a"
        elif env:
            delivered = "env=" + ",".join(env)
        else:
            delivered = "env=<none: this arm does not act through the environment>"
        detail = f" ({a['detail']})" if a.get("detail") else ""
        parts.append(f"{a['mechanism']}={a['state']} [{delivered}]{detail}")
    return "; ".join(parts)


def _arm_intended_env(arm: object) -> "dict[str, str]":
    """The environment overrides *arm* intends the child to carry.

    Env-var arms intend their own dict; registry arms intend nothing, because they act on
    HKLM rather than on the child's environment.  Stated in one place so "this arm has no
    environment" and "this arm's environment was dropped" are never the same expression.
    """
    return dict(arm) if isinstance(arm, dict) else {}


@contextlib.contextmanager
def _arm_context(arm: object) -> "Iterator[dict[str, str]]":
    """Arm one suppression mechanism and yield the environment the child must carry.

    ``arm`` is either the environment dict itself, or a zero-argument callable returning a
    context manager that disables a registry entry and restores it (verified) on exit —
    see ``_registry_suppression.py``.  Both shapes yield a ``dict[str, str]``, so the
    caller binds one type-stable value and cannot accidentally spawn a child outside the
    arm it believes it is testing.

    A registry arm raises ``RegistryMechanismUnavailable`` on entry (non-Windows, missing
    key, no write access); the caller catches it and moves to the next arm.
    """
    if isinstance(arm, dict):
        yield dict(arm)
        return
    with arm():
        yield {}


def _icd_suppression_arms() -> "list[tuple[str, object]]":
    """The three ways to take the ICD away, in the order they are tried.

    Module level, and the single source of truth: the regression control at the bottom of
    this file reads *these* definitions rather than re-declaring them, so a control cannot
    agree with a harness that never passed the arms on.

    They are neutralised by different things, and which one takes is itself the reading.

      driver_search_path — VK_ICD_FILENAMES / VK_DRIVER_FILES / VK_ADD_DRIVER_FILES.
        The Vulkan loader documents all three as "ignored when running a Vulkan
        application with elevated privileges" (PLATFORMS.md §7.4.1).  On an elevated
        runner this arm cannot take, and the failure mode is not a bad answer but a
        control that never reached its subject.
      loader_driver_filter — VK_LOADER_DRIVERS_DISABLE, documented as a filter rather
        than a search path (loader 1.3.234+) with no elevation caveat in the loader's
        *markdown* table.  PROVEN WRONG on 2026-08-06 by reading the loader's own source
        (`loader/loader_environment.c`, `parse_generic_filter_environment_var` →
        `loader_secure_getenv`): this arm goes through the identical `is_high_integrity()`
        gate as the search-path vars, so on a High-integrity runner it is dropped too.
        Kept here (a) because it is free to try, (b) because it still takes on a
        non-elevated dev box, and (c) because `registry_disable` beating an arm that was
        never armed would prove nothing — a competitor has to be in the race.
      registry_disable — flips the ICD's own registered value under
        HKLM\\SOFTWARE\\Khronos\\Vulkan\\Drivers to 1 (disabled). This key is scanned "the
        same way regardless of elevation" (LoaderDriverInterface.md), so it is not gated
        by `is_high_integrity()` at all — it is the mechanism this project's own CI step
        used to REGISTER the ICD in the first place, reused in reverse. This is the arm
        expected to take on the GitHub-hosted Windows runner.

    Verified unelevated on a GPU box, 2026-08-04: the first arm takes there and the
    criterion-4 test passes, so the Windows-lane red does not reproduce on a real device.

    WHAT IS *NOT* CLAIMED HERE.  PR #45's earlier revision carried an in-code comment
    saying the GitHub-hosted Windows runner had been observed reporting
    `icd_suppression_ineffective` for the first two arms while `registry_disable` took.
    That comment has been removed: on that revision both env arms spawned their child with
    no environment override at all (Morpheus's Blocker 1), so the runner never ran them —
    `registry_disable` won by walkover, and a walkover is not a comparison.  Which arm
    takes on the runner is RECORDED by this test, per run, in
    `bench/results/criterion4_icd_witness-dev*.json` — every attempt carries the
    environment that actually reached its child (`env_applied`) and the VK_* variables the
    child actually observed (`env_observed_in_child`) — and is read from there, not
    predicted from here.

    WHAT *HAS* NOW BEEN OBSERVED, with both competitors demonstrably in the race:
    run 31140073862, job 92747912924 (Windows, head 26061ef).  `high_integrity_process`
    was `true`; `driver_search_path` and `loader_driver_filter` each delivered their
    variables and the child echoed them back
    (`VK_DRIVER_FILES=D:\\a\\...\\does_not_exist\\no_such_icd.json`,
    `VK_LOADER_DRIVERS_DISABLE=*`), and each still reported
    `icd_suppression_ineffective`; `registry_disable` reported `suppressed`.  That is a
    comparison, not a walkover.  The Linux twin (job 92747912988) recorded
    `driver_search_path` delivering the same variables and reporting `suppressed`.
    """
    nonexistent = str(REPO / "does_not_exist" / "no_such_icd.json")
    return [
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
        ("registry_disable", _registry_suppression.suppress_icd_registry),
    ]


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
    # Morpheus's Blocker 2 (PR #45): a High-integrity process is exactly the condition
    # under which every env-var arm below is silently dropped by the loader (this file's
    # own root-cause comment, `is_high_integrity()`/`loader_secure_getenv`), so an env
    # arm's apparent success there would be indistinguishable from `registry_disable`'s —
    # except that only one of them is actually trustworthy under those conditions. Stamped
    # into the payload and checked below once `suppression_mechanism` is known.
    high_integrity_process = _registry_suppression.is_high_integrity()

    # The arms themselves live at module level (`_icd_suppression_arms`), single-sourced
    # so the regression control at the bottom of this file reads the same definitions.
    suppression_arms = _icd_suppression_arms()

    records: dict[str, dict] = {}
    suppression_attempts: list[dict] = []
    winning_record: "dict | None" = None
    last_attempted_record: "dict | None" = None

    record = _run_row(row="icd_present", lib=lib, extra_env={}, quiet_seconds=90)
    records["icd_present"] = record

    for arm_name, arm in suppression_arms:
        intended_env = _arm_intended_env(arm)
        try:
            # `as arm_env` is load-bearing: the arm's environment is *bound* and passed
            # to the child.  PR #45's rejected head entered this context without binding
            # and passed `extra_env={}`, which made both env arms dead code.
            with _arm_context(arm) as arm_env:
                if arm_env != intended_env:
                    raise _verdict.InstrumentError(
                        "[criterion 4 instrument failure] ERROR(instrument): arm "
                        f"{arm_name!r} was entered with environment "
                        f"{sorted(arm_env)} but intends {sorted(intended_env)}. The arm's "
                        "environment was dropped or rewritten before the child was "
                        "spawned, so whatever the child reports is a reading of an "
                        "unmodified world, not of this mechanism."
                    )
                record = _run_row(
                    row="icd_suppressed", lib=lib, extra_env=arm_env, quiet_seconds=90
                )
        except _registry_suppression.RegistryMechanismUnavailable as exc:
            suppression_attempts.append(
                {
                    "mechanism": arm_name,
                    "state": "mechanism_unavailable",
                    "env_applied": None,
                    "env_intended": sorted(intended_env),
                    "detail": str(exc),
                }
            )
            continue
        # The arm was armed, the child ran under it, and the child reports which VK_*
        # variables it actually saw.  That last part is what makes a *comparison*
        # possible: an arm whose variables never arrived did not lose to
        # `registry_disable`, it never entered the race (Morpheus's Blocker 2).
        observed_vk_env = record.get("child_vk_env") or {}
        undelivered = sorted(k for k, v in arm_env.items() if observed_vk_env.get(k) != v)
        if undelivered:
            raise _verdict.InstrumentError(
                "[criterion 4 instrument failure] ERROR(instrument): arm "
                f"{arm_name!r} set {sorted(arm_env)} for its child, but the child did not "
                f"observe {undelivered} in its own environment. The arm never reached its "
                "subject, so its verdict — and any comparison against another arm — is "
                "not a reading.\n"
                f"child VK_* environment: {observed_vk_env}"
            )
        record["suppression_mechanism"] = arm_name
        record["arm_env_applied"] = sorted(arm_env)
        state = link.classify(record["loader_probe_report"])["state"]
        suppression_attempts.append(
            {
                "mechanism": arm_name,
                "state": state,
                # Recorded per attempt so the artifact itself falsifies the dead-arm
                # regression: an env arm with `env_applied: []` never ran its mechanism.
                "env_applied": sorted(arm_env),
                "env_observed_in_child": {k: observed_vk_env.get(k) for k in sorted(arm_env)},
            }
        )
        last_attempted_record = record
        if state == link.STATE_SUPPRESSED and winning_record is None:
            # First arm to actually suppress becomes the witness row.  The loop does NOT
            # stop here: every remaining arm is still run, because "registry_disable beat
            # the env arms" is only a claim about mechanisms if the env arms were run at
            # all (Morpheus, PR #45).  Stopping at the first winner is what left the
            # earlier revision with a single attempt and nothing to compare it to.
            winning_record = record

    if winning_record is not None:
        records["icd_suppressed"] = winning_record
    elif last_attempted_record is not None:
        # No arm suppressed anything.  Keep the last child that actually ran so check 1
        # below reports a fired-but-negative control rather than an instrument outage.
        records["icd_suppressed"] = last_attempted_record

    if "icd_suppressed" not in records:
        # Every arm was unavailable on this machine (e.g. non-Windows and no HKLM to
        # fall back to) before a single child even ran — an instrument outage, not a
        # reading, and must not be conflated with "the control fired and failed".
        raise _verdict.InstrumentError(
            "[criterion 4 instrument failure] ERROR(instrument): every ICD-suppression "
            "mechanism was unavailable on this machine — no child was even run for the "
            "suppressed row.\n"
            "attempts: " + _format_attempts(suppression_attempts)
        )

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
        "high_integrity_process": high_integrity_process,
        "comparative_basis": (
            "Each attempt records `env_applied` — the environment overrides that actually "
            "reached that arm's child — and `env_observed_in_child`, the values the child "
            "read back out of its own environment. An arm whose `env_applied` is empty "
            "while its intended environment was not is a DEAD ARM, and a mechanism that "
            "'won' against dead arms won by walkover, which is not a comparison "
            "(Morpheus, PR #45 rejection at head e2b7e00). A registry arm's `env_applied` "
            "is legitimately empty: it acts on HKLM, not on the environment. Every arm is "
            "attempted on every run — the sweep does not stop at the first winner — so "
            "`suppression_attempts` is the full field, and `suppression_mechanism` names "
            "the first arm in it that actually suppressed the ICD."
        ),
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
            "attempts: " + _format_attempts(suppression_attempts) + "\n"
            "The control did not reach its observation, so THIS IS NOT A CRITERION-4 "
            "FAILURE and it is not a pass either (R13). PROVEN root cause (2026-08-06, "
            "reading loader/loader_environment.c upstream, not assumed): every VK_* env "
            "var this file tries — VK_ICD_FILENAMES/VK_DRIVER_FILES *and* the filter var "
            "VK_LOADER_DRIVERS_DISABLE — is read through loader_secure_getenv, which "
            "returns NULL whenever the calling process token is High integrity. This is "
            "not a loader-age issue: real CI (job 92670932473) reports loader version "
            "1.3.301, well past the 1.3.234 filter-var floor. If `registry_disable` is "
            "ALSO in the attempts above and failed, the HKLM key this project's own CI "
            "step writes to (HKLM\\SOFTWARE\\Khronos\\Vulkan\\Drivers) was not writable "
            "or not found — see its `mechanism_unavailable` detail.\n"
            f"artifact: {path}"
        )

    # ── 1.5. On a High-integrity process, only `registry_disable` is trustworthy
    #        evidence — Morpheus's Blocker 2 (PR #45). Every env-var arm above is read
    #        through `loader_secure_getenv`, which this file's own root-cause comment
    #        proves returns NULL whenever `is_high_integrity()` is true; an env arm that
    #        nonetheless reported STATE_SUPPRESSED there would be a false positive this
    #        project cannot distinguish from a real one without this check. ───────────
    if high_integrity_process and neg.get("suppression_mechanism") != "registry_disable":
        raise _verdict.InstrumentError(
            "[criterion 4 instrument failure] ERROR(instrument): this process is "
            "High integrity (is_high_integrity() == True), so every VK_* env var arm "
            "is silently dropped by the loader's own loader_secure_getenv gate — yet the "
            f"arm that reported suppressed was {neg.get('suppression_mechanism')!r}, not "
            "'registry_disable'. That is not trustworthy evidence: either the "
            "high-integrity detection is wrong, or an env arm coincidentally looked "
            "suppressed for an unrelated reason. Only 'registry_disable' is proven not "
            "to be gated by process integrity (LoaderDriverInterface.md, 'regardless of "
            "elevation').\n"
            "attempts: " + _format_attempts(suppression_attempts)
            + f"\nartifact: {path}"
        )

    # ── 1.6. A win over arms that never ran is a walkover, not a comparison.
    #        Morpheus's Blocker 2 (PR #45): at head e2b7e00 both env arms spawned their
    #        child with `extra_env={}`, so `registry_disable` was the only mechanism that
    #        ever acted, and the artifact's "the other two were ineffective" reading was
    #        an artifact of the harness. This check makes that state impossible to record
    #        as evidence: an env arm that ran with an empty environment is a dead arm. ──
    if neg.get("suppression_mechanism") == "registry_disable":
        walkovers = [
            a["mechanism"]
            for a in suppression_attempts
            if a["mechanism"] != "registry_disable" and a.get("env_applied") == []
        ]
        if walkovers:
            raise _verdict.InstrumentError(
                "[criterion 4 instrument failure] ERROR(instrument): `registry_disable` "
                f"reported suppressed, but the competing arm(s) {walkovers} ran with an "
                "EMPTY environment — they were never armed. A mechanism that beats arms "
                "that never acted has beaten nobody, and this artifact must not be read "
                "as a comparison between suppression mechanisms.\n"
                "attempts: " + _format_attempts(suppression_attempts)
                + f"\nartifact: {path}"
            )


    # ── 1.7. Every declared arm was actually attempted.  With the sweep no longer
    #        stopping at the first winner, a future `break` (or an exception swallowed
    #        mid-loop) would quietly shrink the comparison back to one arm, which is the
    #        shape the PR #45 evidence had.  The artifact must name every mechanism. ──
    attempted = [a["mechanism"] for a in suppression_attempts]
    declared = [name for name, _ in suppression_arms]
    if attempted != declared:
        raise _verdict.InstrumentError(
            "[criterion 4 instrument failure] ERROR(instrument): the suppression sweep "
            f"recorded {attempted} but this file declares {declared}. Every arm is run on "
            "every platform so that the winner's margin is over mechanisms that actually "
            "acted; a short sweep is not a comparison.\n"
            "attempts: " + _format_attempts(suppression_attempts)
            + f"\nartifact: {path}"
        )

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
        # Mechanism and process integrity go into the LOG, not only into the artifact:
        # Morpheus's non-blocking finding 3 (PR #45) — `bench/results/` is not uploaded by
        # every lane, so a stamp that exists only there is not retrievable from CI.
        f"  mechanism      : {neg.get('suppression_mechanism')!r} "
        f"(high_integrity_process={high_integrity_process})\n"
        f"  arms           : {_format_attempts(suppression_attempts)}\n"
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
# PLANTED REGRESSION CONTROL — an arm that never reaches its child is a dead arm
# ===========================================================================
# PR #45 was rejected at head `e2b7e00` for exactly one defect (Morpheus, 2026-08-06):
# the suppression loop built each arm's environment, entered its context WITHOUT binding
# the result, and spawned the child with a hard-coded `extra_env={}`.  Both env-var arms
# became dead code, and the `icd_suppression_ineffective` they then reported was read as
# a comparison in which `registry_disable` beat them.  It had beaten nothing.
#
# These controls exist so that regression cannot recur silently.  They run on EVERY
# platform, need no GPU, no EP library and no registry access — the property under test
# is this file's own control flow — so they are never skipped and their absence from a
# lane summary is itself visible.  Each one fails on the rejected revision:
#
#   * `test_every_icd_arm_delivers_its_own_environment_to_its_child` drives the REAL
#     `test_criterion4_icd_polarity_witness` body with only the child-spawn seam, the
#     classifier and the registry mechanism stubbed, and asserts the exact environment
#     each arm intends is the environment its child receives.
#   * `test_the_dead_arm_defect_is_detected_when_planted` plants the defect itself — an
#     arm context that yields nothing, `e2b7e00`'s behaviour — and requires the witness
#     to report ERROR(instrument).  A control that cannot fail is a constant (R10), so
#     this control is falsified here rather than trusted.
#   * `test_run_row_delivers_extra_env_into_the_spawned_process_environment` closes the
#     other half of the seam: that `_run_row` puts those overrides into the environment
#     the child process is actually spawned with.


class _PlantedClassifier:
    """Stand-in for ``ci/check_icd_suppression.py`` with a token the planted rows carry.

    Deliberately NOT a copy of Link's classifier: these controls are about this file's
    plumbing, and a real classifier here would make them depend on a real loader report.
    """

    STATE_SUPPRESSED = "suppressed"
    STATE_INEFFECTIVE = "icd_suppression_ineffective"

    @staticmethod
    def classify(report: str) -> dict:
        if "PLANTED_SUPPRESSED=1" in report:
            return {
                "state": _PlantedClassifier.STATE_SUPPRESSED,
                "detail": "planted",
                "devices_passing_gate": 0,
            }
        return {
            "state": _PlantedClassifier.STATE_INEFFECTIVE,
            "detail": "planted",
            "devices_passing_gate": 1,
        }


def _planted_record(*, row: str, suppressed: bool, vk_env: "dict[str, str]") -> dict:
    """A child record with every field the criterion-4 body reads, and no real child."""
    return {
        "row": row,
        "library": "planted",
        "child_vk_env": dict(vk_env),
        "loader_probe_report": (
            f"PLANTED_SUPPRESSED={'1' if suppressed else '0'} {GATE_LINE_PHRASE}"
        ),
        "epctl_exit_code": 3 if suppressed else 0,
        "ep_devices_advertised": 0 if suppressed else 1,
        "vulkan_node_events": 0 if suppressed else 1,
        "output_exact_match": True,
        "ep_log": "planted",
        "child_stderr": ZERO_DEVICE_WARNING if suppressed else "",
        "child_stdout": "",
        "row_state": ROW_OBSERVED,
        "child_exit_code": 0,
    }


def _plant_criterion4_seams(
    monkeypatch,
    tmp_path,
    *,
    winning_arm: str = "registry_disable",
    high_integrity: bool = True,
) -> dict:
    """Stub the child spawn, the classifier and the registry arm; record what happens.

    Everything else — arm selection, context entry, the checks, the artifact — is the
    real code under test.  The returned dict is the observation: every ``_run_row`` call
    with the environment it was handed, and the registry arm's arm/restore counts.

    ``winning_arm`` chooses which planted world is being modelled: ``registry_disable``
    is the Windows shape (env arms are delivered and lose), any env arm name is the
    Linux shape (the first arm suppresses and the sweep must still run the rest).
    """
    state = {"calls": [], "registry_entered": 0, "registry_exited": 0, "registry_armed": False}
    winning_env = _arm_intended_env(dict(_icd_suppression_arms())[winning_arm])

    def fake_run_row(*, row: str, lib: Path, extra_env: "dict[str, str]", quiet_seconds: float):
        state["calls"].append({"row": row, "extra_env": dict(extra_env)})
        # The planted world: exactly one mechanism actually suppresses anything. Every
        # other arm is armed and delivered — it simply loses, which is the whole point:
        # losing and never running are different states and must read differently.
        if row != "icd_suppressed":
            suppressed = False
        elif winning_arm == "registry_disable":
            suppressed = state["registry_armed"]
        else:
            suppressed = extra_env == winning_env
        return _planted_record(row=row, suppressed=suppressed, vk_env=extra_env)

    @contextlib.contextmanager
    def fake_suppress_icd_registry():
        state["registry_entered"] += 1
        state["registry_armed"] = True
        try:
            yield ["planted-icd-manifest.json"]
        finally:
            state["registry_armed"] = False
            state["registry_exited"] += 1

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_run_row", fake_run_row)
    monkeypatch.setattr(module, "_load_link_classifier", lambda: _PlantedClassifier)
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(
        _registry_suppression, "suppress_icd_registry", fake_suppress_icd_registry
    )
    monkeypatch.setattr(_registry_suppression, "is_high_integrity", lambda: high_integrity)
    monkeypatch.setenv("ONNXRUNTIME_VULKAN_EP_LIB", str(tmp_path / "planted_lib.bin"))
    monkeypatch.delenv("ONNXRUNTIME_EP_VULKAN_WITNESS_CHILD", raising=False)
    return state


def test_every_env_arm_declares_a_non_empty_environment() -> None:
    """The arms themselves must still be arms: an empty override dict suppresses nothing.

    Cheap, but it is the other way the same regression can arrive — not by dropping the
    environment on the way to the child, but by emptying it at the source.
    """
    arms = _icd_suppression_arms()
    assert [name for name, _ in arms] == [
        "driver_search_path",
        "loader_driver_filter",
        "registry_disable",
    ], "the arm roster changed; the criterion-4 evidence names these mechanisms"

    env_arms = {name: arm for name, arm in arms if isinstance(arm, dict)}
    assert set(env_arms) == {"driver_search_path", "loader_driver_filter"}
    for name, env in env_arms.items():
        assert env, f"arm {name!r} declares an EMPTY environment — it cannot suppress an ICD"
        assert all(k.startswith("VK_") for k in env), f"arm {name!r} sets a non-Vulkan variable"
    assert "VK_ICD_FILENAMES" in env_arms["driver_search_path"]
    assert "VK_DRIVER_FILES" in env_arms["driver_search_path"]
    assert env_arms["loader_driver_filter"]["VK_LOADER_DRIVERS_DISABLE"] == "*"


def test_every_icd_arm_delivers_its_own_environment_to_its_child(monkeypatch, tmp_path) -> None:
    """THE control for Morpheus's Blocker 1: each arm's environment reaches its child.

    Runs the real criterion-4 body to completion.  Under the rejected `e2b7e00` revision
    both env arms would appear here with ``extra_env == {}``.
    """
    state = _plant_criterion4_seams(monkeypatch, tmp_path)

    test_criterion4_icd_polarity_witness()

    calls = state["calls"]
    assert calls[0] == {"row": "icd_present", "extra_env": {}}, (
        "the positive row must run in an unmodified world"
    )
    suppressed_calls = [c for c in calls if c["row"] == "icd_suppressed"]
    assert len(suppressed_calls) == 3, (
        "every declared arm must be run — the sweep does not stop early; "
        f"got {len(suppressed_calls)}"
    )

    arms = _icd_suppression_arms()
    for (arm_name, arm), call in zip(arms, suppressed_calls):
        intended = _arm_intended_env(arm)
        assert call["extra_env"] == intended, (
            f"arm {arm_name!r} spawned its child with {sorted(call['extra_env'])} but "
            f"intends {sorted(intended)} — this is exactly the PR #45 dead-arm regression"
        )
    assert suppressed_calls[0]["extra_env"]["VK_ICD_FILENAMES"]
    assert suppressed_calls[1]["extra_env"]["VK_LOADER_DRIVERS_DISABLE"] == "*"
    assert suppressed_calls[2]["extra_env"] == {}, (
        "the registry arm acts on HKLM, not on the environment"
    )

    # Armed before, restored after — the registry arm's context was entered exactly once
    # and left exactly once, around its child run.
    assert state["registry_entered"] == 1
    assert state["registry_exited"] == 1
    assert state["registry_armed"] is False

    written = list(tmp_path.glob("criterion4_icd_witness-dev*.json"))
    assert len(written) == 1, f"expected exactly one criterion-4 artifact, got {written}"
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["suppression_mechanism"] == "registry_disable"
    assert payload["high_integrity_process"] is True
    attempts = {a["mechanism"]: a for a in payload["suppression_attempts"]}
    assert set(attempts) == {"driver_search_path", "loader_driver_filter", "registry_disable"}
    arm_env_by_name = {name: _arm_intended_env(arm) for name, arm in arms}
    for name in ("driver_search_path", "loader_driver_filter"):
        assert attempts[name]["env_applied"], (
            f"the artifact records arm {name!r} as having delivered no environment — "
            "a reader would be right to call `registry_disable`'s win a walkover"
        )
        assert attempts[name]["env_applied"] == sorted(arm_env_by_name[name])
        assert attempts[name]["state"] == _PlantedClassifier.STATE_INEFFECTIVE
        assert attempts[name]["env_observed_in_child"] == arm_env_by_name[name], (
            f"arm {name!r}'s child did not observe the variables the arm set, so the arm "
            "never reached its subject and its verdict is not a reading"
        )
    assert attempts["registry_disable"]["env_applied"] == []
    assert attempts["registry_disable"]["state"] == _PlantedClassifier.STATE_SUPPRESSED


def test_the_sweep_does_not_stop_at_the_first_arm_that_wins(monkeypatch, tmp_path) -> None:
    """The Linux shape: arm 1 suppresses, and the remaining arms are STILL run.

    A sweep that stops at its first winner records one attempt and calls it a comparison.
    That is how PR #45's evidence came to rest on a single mechanism, and it is why this
    control plants the opposite world from the one above: here ``driver_search_path``
    wins, and the artifact must still name every mechanism that was offered.
    """
    state = _plant_criterion4_seams(
        monkeypatch, tmp_path, winning_arm="driver_search_path", high_integrity=False
    )

    test_criterion4_icd_polarity_witness()

    suppressed_calls = [c for c in state["calls"] if c["row"] == "icd_suppressed"]
    assert len(suppressed_calls) == 3, (
        "the first arm suppressed the ICD and the sweep stopped — the remaining "
        f"mechanisms were never compared against it; got {len(suppressed_calls)} attempt(s)"
    )
    assert state["registry_entered"] == 1 and state["registry_exited"] == 1

    payload = json.loads(
        next(iter(tmp_path.glob("criterion4_icd_witness-dev*.json"))).read_text(encoding="utf-8")
    )
    assert payload["suppression_mechanism"] == "driver_search_path", (
        "the witness row must be the arm that actually suppressed the ICD"
    )
    assert payload["high_integrity_process"] is False
    attempts = {a["mechanism"]: a for a in payload["suppression_attempts"]}
    assert set(attempts) == {"driver_search_path", "loader_driver_filter", "registry_disable"}
    assert attempts["driver_search_path"]["state"] == _PlantedClassifier.STATE_SUPPRESSED
    for name in ("driver_search_path", "loader_driver_filter"):
        assert attempts[name]["env_applied"], f"arm {name!r} delivered no environment"
        assert attempts[name]["env_observed_in_child"] == _arm_intended_env(
            dict(_icd_suppression_arms())[name]
        )


def test_the_dead_arm_defect_is_detected_when_planted(monkeypatch, tmp_path) -> None:
    """Falsify the control itself: plant `e2b7e00`'s defect and require an outage.

    A control that cannot be made to fail is a constant, not a control (R10, and Link's
    ICD-probe lesson at the top of this file).  Here the arm context is replaced by one
    that yields no environment for any arm — precisely what
    ``with contextlib.nullcontext(arm):`` plus ``extra_env={}`` did — and the witness must
    report ERROR(instrument) rather than produce a reading.
    """
    _plant_criterion4_seams(monkeypatch, tmp_path)

    @contextlib.contextmanager
    def dead_arm_context(arm: object):
        if isinstance(arm, dict):
            yield {}
            return
        with arm():
            yield {}

    monkeypatch.setattr(sys.modules[__name__], "_arm_context", dead_arm_context)

    with pytest.raises(_verdict.InstrumentError) as excinfo:
        test_criterion4_icd_polarity_witness()
    message = str(excinfo.value)
    assert "driver_search_path" in message
    assert "dropped or rewritten" in message, (
        "the dead-arm defect must be reported as an instrument outage naming the drop; "
        f"got: {message[:400]}"
    )


def test_run_row_delivers_extra_env_into_the_spawned_process_environment(
    monkeypatch, tmp_path
) -> None:
    """The other half of the seam: ``extra_env`` becomes the child process's environment.

    ``_run_row`` is stubbed out in the control above, so this is where its own contract is
    checked — against the exact call it makes to spawn the child.
    """
    captured: dict = {}

    class _Result:
        returncode = 0
        stdout = RECORD_MARKER + json.dumps({"row": "icd_suppressed", "child_vk_env": {}})
        stderr = ""

    def fake_run_subprocess_checked(cmd, *, what, quiet_seconds, **kwargs):
        captured["env"] = dict(kwargs["env"])
        captured["cmd"] = list(cmd)
        return _Result()

    monkeypatch.setattr(_verdict, "run_subprocess_checked", fake_run_subprocess_checked)

    overrides = {
        "VK_ICD_FILENAMES": str(tmp_path / "no_such_icd.json"),
        "VK_ADD_DRIVER_FILES": "",
        "VK_LOADER_DRIVERS_DISABLE": "*",
    }
    record = _run_row(
        row="icd_suppressed", lib=tmp_path / "planted_lib.bin", extra_env=overrides,
        quiet_seconds=1,
    )

    for name, value in overrides.items():
        assert captured["env"][name] == value, (
            f"{name} did not reach the spawned child's environment — an arm that sets it "
            "would be suppressing nothing"
        )
    assert captured["env"]["ONNXRUNTIME_EP_VULKAN_WITNESS_CHILD"] == "1"
    assert record["row_state"] == ROW_OBSERVED


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
    # The VK_* environment THIS PROCESS actually received, read from inside the child.
    # The parent uses it to prove an arm reached its subject: an arm whose variables are
    # absent here never suppressed anything, whatever its child then reported (Morpheus's
    # Blocker 1/2, PR #45 — both env arms were spawned with no overrides at all).
    record["child_vk_env"] = {
        name: value for name, value in sorted(os.environ.items()) if name.startswith("VK_")
    }

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
