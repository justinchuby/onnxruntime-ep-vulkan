"""Run a child command with the Vulkan ICD actually taken away, and say which mechanism did it.

WHY THIS EXISTS
    The Windows lane's ICD-removal negative control used to arm itself by pointing
    ``VK_ICD_FILENAMES`` / ``VK_DRIVER_FILES`` at a nonexistent manifest and then asking
    ``ci/check_icd_suppression.py`` whether that had taken.  On the GitHub-hosted Windows
    runner the answer is always no: the runner is a High-integrity process and the LunarG
    loader reads those variables through ``loader_secure_getenv``, which returns NULL
    there (PLATFORMS.md §7.4.1).  The step therefore printed
    ``ERROR(instrument=icd_suppression_ineffective)`` and exited 0 on every single run.
    It executed, it was green, and it asserted NOTHING — a control that never reaches its
    subject is not a control, and a green that cannot go red is not a signal (R13).

    Issue #1 / PR #45 established, with both env-var competitors demonstrably in the race
    on the runner itself, that a third mechanism is *not* gated by process integrity:
    disabling the ICD's registered value under ``HKLM\\SOFTWARE\\Khronos\\Vulkan\\Drivers``,
    which the loader scans "the same way regardless of elevation".  That mechanism has one
    implementation with verified-restore semantics and two-polarity tests
    (``tests/ops/_registry_suppression.py``); this script uses it rather than
    re-implementing it, so there is exactly one thing to review and exactly one thing to
    break.

WHAT IT CLAIMS AND WHAT IT DOES NOT
    Claims: when it exits 0, the loader-probe it ran *inside* the arm reported no device
    passing the §7.2 capability gate, so the child command that followed ran in a process
    with no usable ICD.  The child's own exit status and output are the caller's to judge.

    Does not claim: that the child asserted anything, that the arm is the only one that
    could have worked, or that the arm would work on another host.  Every arm attempted is
    recorded with the environment that actually reached its probe, so a mechanism that
    "won" over arms that never ran is visible as a walkover rather than a comparison
    (Morpheus, PR #45 rejection at head e2b7e00).

TERMINAL STATES (R13)
    0  ICD-SUPPRESSION-ARM: PASS — an arm took, the child ran; its code is on the stamp
       line ``[ICD-SUPPRESSED] mechanism=<name> child_exit=<n>`` and in --record-out.
    2  usage
    4  ICD-SUPPRESSION-ARM: ERROR(instrument=no_mechanism_suppressed_the_icd) — every
       declared arm was attempted and none removed the ICD.  The child did NOT run.  This
       is not a detection and not a pass.

USAGE
    python ci/with_icd_suppressed.py --probe rust/target/release/epctl.exe \
        --probe-report bench/results/ci-lane/icd-probe-report.txt \
        --record-out bench/results/ci-lane/icd-suppression.json \
        -- python ci/gate_chain_fp32.py --verdict-out ... --workdir ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parent
for _p in (str(CI_DIR), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_icd_suppression  # noqa: E402  (path set above)

from tests.ops import _registry_suppression  # noqa: E402  (path set above)

LABEL = "ICD-SUPPRESSION-ARM"

EXIT_PASS = 0
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4


def declared_arms(nonexistent_icd: str) -> "list[tuple[str, object]]":
    """The mechanisms tried, in order, each paired with the way it is armed.

    An arm is either a dict of environment overrides or a zero-argument callable returning
    a context manager.  Both are entered through :func:`arm_context`, which yields the
    environment the child must carry, so no arm can be entered without its environment
    being bound — the defect Morpheus rejected PR #45's earlier revision for.

    The two env arms are kept even though they cannot take on an elevated runner: they are
    free to try, they DO take on an unelevated host (so this script works on a dev box and
    on Linux), and a registry arm that beat competitors which were never armed would have
    proved nothing.
    """
    return [
        (
            "driver_search_path",
            {
                "VK_ICD_FILENAMES": nonexistent_icd,
                "VK_DRIVER_FILES": nonexistent_icd,
                "VK_ADD_DRIVER_FILES": "",
            },
        ),
        (
            "loader_driver_filter",
            {
                "VK_ICD_FILENAMES": nonexistent_icd,
                "VK_DRIVER_FILES": nonexistent_icd,
                "VK_ADD_DRIVER_FILES": "",
                "VK_LOADER_DRIVERS_DISABLE": "*",
            },
        ),
        ("registry_disable", _registry_suppression.suppress_icd_registry),
    ]


@contextlib.contextmanager
def arm_context(arm: object):
    """Arm one mechanism and yield the environment overrides the child must carry.

    Registry arms yield an empty dict because they act on HKLM, not on the environment;
    that is different from "this arm's environment was dropped", and the two are never the
    same expression here.  ``RegistryMechanismUnavailable`` is raised at ENTER time on a
    non-Windows host or without write access, so an unavailable mechanism can never be
    mistaken for one that ran and failed.
    """
    if isinstance(arm, dict):
        yield dict(arm)
        return
    with arm():
        yield {}


def probe_state(
    probe: "list[str]", report_path: Path, env_overrides: "dict[str, str]"
) -> "tuple[dict, str]":
    """Run the loader probe inside the current arm and classify what it saw.

    A probe that cannot be launched at all is an instrument state, never a suppression:
    "the binary is missing" and "the ICD is gone" produce the same silence, and only one
    of them means the control reached its subject.
    """
    child_env = dict(os.environ)
    child_env.update(env_overrides)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [*probe, "--probe-loader"],
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        report_path.write_text(f"probe could not be launched: {exc}\n", encoding="utf-8")
        return (
            {
                "state": check_icd_suppression.STATE_UNREADABLE,
                "devices_passing_gate": None,
                "epctl_exit_code": None,
                "detail": f"The loader probe {probe!r} could not be launched ({exc}). "
                "No observation was made, so nothing about the ICD is known.",
            },
            "",
        )
    text = (completed.stdout or "") + (completed.stderr or "")
    report_path.write_text(text, encoding="utf-8")
    result = check_icd_suppression.classify(text)
    result["epctl_exit_code"] = completed.returncode
    # The same two-witness rule the standalone screen applies (R13 obligation 3): a text
    # parse and an exit code that disagree is an instrument state, and picking the
    # convenient one is how a lane learns to lie. Reused, not re-derived.
    expected = check_icd_suppression.EXPECTED_EPCTL_EXIT.get(result["state"])
    if expected is not None and completed.returncode not in expected:
        result["witnesses_disagree"] = (
            f"text parse says {result['state']} (epctl should have exited one of "
            f"{list(expected)}) but epctl exited {completed.returncode}"
        )
        result["state"] = check_icd_suppression.STATE_UNREADABLE
        result["detail"] = (
            "The two witnesses disagree: " + result["witnesses_disagree"] + "."
        )
    return result, text


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="run a child command with the Vulkan ICD removed by whichever "
        "mechanism this host allows"
    )
    p.add_argument(
        "--probe",
        required=True,
        nargs="+",
        help="argv of the loader probe; `--probe-loader` is appended to it",
    )
    p.add_argument("--probe-report", required=True, help="where to write the probe output")
    p.add_argument("--record-out", default="", help="where to write the JSON record")
    p.add_argument(
        "--nonexistent-icd",
        default="",
        help="path the env arms point at; must not exist. Defaults to a path under the "
        "repo that is never created.",
    )
    p.add_argument(
        "--arms",
        default="",
        help="comma-separated subset of arms to declare, in order. Default (and the only "
        "value the lane uses) is every arm. Narrowing is recorded in the JSON as "
        "`arms_declared` so a comparison cannot be quietly reduced to a walkover.",
    )
    p.add_argument("child", nargs=argparse.REMAINDER, help="-- then the command to run")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    child = list(args.child)
    if child and child[0] == "--":
        child = child[1:]
    if not child:
        print(f"{LABEL}: usage — no child command after `--`.", flush=True)
        return EXIT_USAGE

    nonexistent = args.nonexistent_icd or str(REPO / "does_not_exist" / "no_such_icd.json")
    report_path = Path(args.probe_report)
    record = {
        "high_integrity_process": _registry_suppression.is_high_integrity(),
        "suppression_attempts": [],
        "suppression_mechanism": None,
        "child_exit_code": None,
        "comparative_basis": (
            "Every declared arm is attempted until one removes the ICD, and each attempt "
            "records the environment that actually reached its probe (`env_applied`). A "
            "registry arm's `env_applied` is legitimately empty: it acts on HKLM, not on "
            "the environment. An arm recorded as `mechanism_unavailable` was never in the "
            "race and cannot be counted as one this host beat."
        ),
    }

    arms = declared_arms(nonexistent)
    if args.arms:
        wanted = [n.strip() for n in args.arms.split(",") if n.strip()]
        by_name = dict(arms)
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            print(f"{LABEL}: usage — unknown arm(s) {unknown}; known: {list(by_name)}.", flush=True)
            return EXIT_USAGE
        arms = [(n, by_name[n]) for n in wanted]
    record["arms_declared"] = [n for n, _ in arms]
    winner = None
    for name, arm in arms:
        try:
            with arm_context(arm) as arm_env:
                intended = dict(arm) if isinstance(arm, dict) else {}
                if intended and not arm_env:
                    # The exact defect that got PR #45 rejected: an arm entered without
                    # its environment bound spawns an unmodified child and then reports on
                    # a mechanism that was never applied.
                    record["suppression_attempts"].append(
                        {
                            "mechanism": name,
                            "state": "arm_environment_dropped",
                            "env_applied": {},
                            "detail": "This arm intended environment overrides and none "
                            "reached its probe. Nothing was measured.",
                        }
                    )
                    continue
                result, _text = probe_state(args.probe, report_path, arm_env)
                attempt = {
                    "mechanism": name,
                    "state": result["state"],
                    "env_applied": dict(arm_env),
                    "epctl_exit_code": result.get("epctl_exit_code"),
                    "detail": result.get("detail", ""),
                }
                record["suppression_attempts"].append(attempt)
                if result["state"] != check_icd_suppression.STATE_SUPPRESSED:
                    continue

                # Still inside the arm: the child must run under the same suppression the
                # probe just verified, not after it has been restored.
                record["suppression_mechanism"] = name
                winner = attempt
                print(
                    f"{LABEL}: PASS — `{name}` removed the ICD "
                    f"({result.get('evidence', 'probe reported no capable device')}). "
                    "Running the child under it.",
                    flush=True,
                )
                completed = subprocess.run(
                    child, env={**os.environ, **arm_env}
                )  # streams inherit: the caller reads the child's own output
                record["child_exit_code"] = completed.returncode
                break
        except _registry_suppression.RegistryMechanismUnavailable as exc:
            record["suppression_attempts"].append(
                {
                    "mechanism": name,
                    "state": "mechanism_unavailable",
                    "env_applied": {},
                    "detail": f"{exc}",
                }
            )

    attempted = [a["mechanism"] for a in record["suppression_attempts"]]
    declared = [n for n, _ in arms]
    if winner is None and attempted != declared:
        record["instrument_fault"] = (
            f"declared arms {declared} but attempted {attempted}: an arm that was never "
            "tried cannot be reported as one that did not work."
        )

    if args.record_out:
        out = Path(args.record_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    if winner is None:
        print(f"{LABEL}: ERROR(instrument=no_mechanism_suppressed_the_icd)", flush=True)
        for attempt in record["suppression_attempts"]:
            print(f"  {attempt['mechanism']}: {attempt['state']} — {attempt['detail']}", flush=True)
        print(
            f"{LABEL}: every declared arm was attempted and the ICD is still present, so "
            "the child was NOT run. Per DESIGN.md §10.0.1 R13 this is not a detection and "
            "not a pass.",
            flush=True,
        )
        return EXIT_ERROR_INSTRUMENT

    print(
        f"[ICD-SUPPRESSED] mechanism={record['suppression_mechanism']} "
        f"child_exit={record['child_exit_code']}",
        flush=True,
    )
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
