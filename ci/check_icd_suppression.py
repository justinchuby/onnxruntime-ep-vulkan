#!/usr/bin/env python3
"""``check_icd_suppression`` — did the ICD-removal negative control actually fire?

The problem, which is specific to Windows and permanent
--------------------------------------------------------

``PLATFORMS.md`` §7.4.1: the LunarG loader **silently ignores**
``VK_DRIVER_FILES`` / ``VK_ICD_FILENAMES`` / ``VK_ADD_DRIVER_FILES`` in elevated
processes, and GitHub's Windows runners are elevated (``runneradmin``, Administrators,
UAC disabled) — which is exactly why that lane registers lavapipe in the registry instead
of using the env var.  So the Windows lane's ICD-removal negative control may never have
fired on any runner it ever ran on.  And when a control does not fire, the step it guards
*passes*, and the lane reports "the gate cannot fail" — blaming the gate for a control
that never reached its observation.  That is a real detection's costume on an instrument
outage and it routes the finding to the wrong owner (R13).

Why this is a script and not four lines of PowerShell
------------------------------------------------------

Because the four lines of PowerShell were wrong, and nothing could tell.  They tested::

    if ($probe -match 'passed the §7\\.2 capability gate') { ...ineffective...; exit 0 }

``engine::loader_probe_report()`` emits that phrase on **every** run — the line is
``"{n_passed} device(s) passed the §7.2 capability gate."`` and ``n_passed`` is ``0`` when
the suppression *did* take.  The match therefore succeeded unconditionally: the control
short-circuited to ``exit 0`` on every run, including the runs where it would have worked.
**A detector that fires on every input is not a detector, it is a constant** — the same
defect ``bench/results/probe_gpustate.py`` found in its own ancestry check and named in
exactly those words.

The count is the discriminator, so the count is what this parses.  Being a script, it has
two-polarity tests (``ci/test_lane_checks.py``) that feed it both reports and assert the
**token**, and it writes a record whose content varies with its input — which is R10's
falsifier for "X is wired", and the thing a ``::warning`` annotation is not.

Terminal states, R13
--------------------

    0  ICD-SUPPRESSION: PASS                         the ICD is gone; run the control
    4  ICD-SUPPRESSION: ERROR(instrument=icd_suppression_ineffective)
                                                     the ICD is still there; the control
                                                     cannot reach its observation, and
                                                     this asserts NOTHING about the gate
    4  ICD-SUPPRESSION: ERROR(instrument=probe_report_unreadable)
                                                     the report was reworded; guessing
                                                     would green-light a broken runner or
                                                     condemn a working one

There is no ``FAIL(condition=...)`` here on purpose.  This check has no condition of its
own to detect — it only decides whether a *different* check is able to run.  Every
non-``PASS`` outcome is an instrument state.

USAGE
    epctl --probe-loader > probe.txt ; python ci/check_icd_suppression.py probe.txt \\
        [--record-out bench/results/ci-lane/icd-suppression.json] [--exit-code N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

LABEL = "ICD-SUPPRESSION"

#: The contract between ``engine.rs`` (Switch's) and this file, expressed as prose because
#: that is what the report is. ``epctl`` reads the same phrase; when it moves, both move.
GATE_VERDICT_RE = re.compile(
    r"(?P<n>\d+)\s+device\(s\)\s+passed the §7\.2 capability gate", re.UNICODE
)

#: The other way a successful suppression shows up, and the one that surprised me.
#: When the ICD really is gone, ``vkCreateInstance`` returns ``ERROR_INCOMPATIBLE_DRIVER``
#: and the report **never reaches** the "N device(s) passed" line at all. Measured on my
#: desk, both polarities, with the real binary:
#:
#:     loader healthy   -> "2 device(s) passed the §7.2 capability gate."      epctl exit 0
#:     ICD suppressed   -> "FAIL: vkCreateInstance returned ERROR_INCOMPATIBLE_DRIVER."
#:                         ...and no gate line at all                          epctl exit 3
#:
#: A first draft of this file classified that second report as
#: ``probe_report_unreadable`` — which would have short-circuited the negative control on
#: every run, in exactly the same way as the substring test it replaced, for exactly the
#: same reason: I reasoned about the instrument instead of running it. It is recorded here
#: because the lesson is more durable than the regex.
NO_ICD_RE = re.compile(
    r"ERROR_INCOMPATIBLE_DRIVER|the loader found no usable ICD", re.IGNORECASE
)

STATE_SUPPRESSED = "suppressed"
STATE_INEFFECTIVE = "icd_suppression_ineffective"
STATE_UNREADABLE = "probe_report_unreadable"

#: The epctl exit codes each state predicts. A second witness with a different failure
#: mode (R13 obligation 3): a text parse can be fooled by a rewording, an exit code cannot,
#: and an exit code cannot say *why*, so neither is sufficient alone.
EXPECTED_EPCTL_EXIT = {
    STATE_SUPPRESSED: (1, 3),
    STATE_INEFFECTIVE: (0,),
}


def classify(report: str) -> dict:
    """Decide, from the loader-probe report alone, whether the ICD is actually gone.

    Pure function so both polarities can be tested without a Vulkan loader, on a lane that
    has no GPU.  A check that can only be exercised on the machine it is meant to protect
    is a check nobody exercises.
    """
    m = GATE_VERDICT_RE.search(report)
    if m is None:
        if NO_ICD_RE.search(report):
            return {
                "state": STATE_SUPPRESSED,
                "devices_passing_gate": 0,
                "evidence": "vkCreateInstance failed with no usable ICD",
                "detail": "The loader found no usable ICD and instance creation failed, "
                "so the report never reaches the capability-gate line. The ICD really "
                "is absent from this process: the negative control has a subject and "
                "its result is readable.",
            }
        return {
            "state": STATE_UNREADABLE,
            "devices_passing_gate": None,
            "detail": "The loader-probe report contains neither the capability-gate line "
            "nor an instance-creation failure. It cannot tell 'no capable device' from "
            "'the report was reworded', and guessing would either green-light a broken "
            "runner or condemn a working one.",
        }
    n = int(m.group("n"))
    if n == 0:
        return {
            "state": STATE_SUPPRESSED,
            "devices_passing_gate": 0,
            "evidence": "0 devices passed the capability gate",
            "detail": "0 devices passed the §7.2 capability gate: nothing in this process "
            "can execute Vulkan work, so the negative control has a subject and its "
            "result is readable.",
        }
    return {
        "state": STATE_INEFFECTIVE,
        "devices_passing_gate": n,
        "evidence": f"{n} devices passed the capability gate",
        "detail": f"{n} device(s) still passed the §7.2 capability gate with "
        "VK_DRIVER_FILES/VK_ICD_FILENAMES pointed at a nonexistent ICD. The LunarG "
        "loader ignores those variables in elevated processes (PLATFORMS.md §7.4.1) "
        "and GitHub's Windows runners are elevated. The control did not reach its "
        "observation, so it asserts NOTHING about the gate — in particular it is not "
        "evidence that the gate cannot fail. This lane's falsifier is the "
        "loader-independent decline-probe control, which is mandatory and cannot be "
        "skipped.",
    }


def parse_args(argv):
    p = argparse.ArgumentParser(description="did the ICD-removal control actually fire?")
    p.add_argument("report", help="file containing `epctl --probe-loader` output, or - for stdin")
    p.add_argument(
        "--record-out",
        default="",
        help="write the classification as JSON. R10's falsifier for 'this probe is wired' "
        "is an artifact it produced whose content varies with its input, and a "
        "::warning annotation is not one.",
    )
    p.add_argument(
        "--exit-code",
        type=int,
        default=None,
        help="epctl's own exit code, recorded as a second witness with a different "
        "failure mode from the text parse",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.report == "-":
        text = sys.stdin.read()
    else:
        path = Path(args.report)
        if not path.exists():
            result = {
                "state": STATE_UNREADABLE,
                "devices_passing_gate": None,
                "detail": f"No loader-probe report at {path}. The probe step did not run "
                "or did not write its output; absence of the report is an instrument "
                "state, never a clean suppression.",
            }
            text = ""
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            result = classify(text)
    if args.report == "-":
        result = classify(text)

    result["epctl_exit_code"] = args.exit_code
    # A second witness with a different failure mode (R13 obligation 3): epctl returns 1
    # when it read "0 device(s) passed". A text parse and an exit code disagreeing is
    # itself a finding, and it is reported rather than silently resolved in either
    # direction.
    if args.exit_code is not None:
        expected = EXPECTED_EPCTL_EXIT.get(result["state"])
        if expected is not None and args.exit_code not in expected:
            result["witnesses_disagree"] = (
                f"text parse says {result['state']} (epctl should have exited one of "
                f"{list(expected)}) but epctl exited {args.exit_code}"
            )
            result["state"] = STATE_UNREADABLE
            result["detail"] = (
                "The two witnesses disagree: " + result["witnesses_disagree"] + ". "
                "Two readers of one report that disagree is an instrument state, and "
                "picking the convenient one is how a lane learns to lie."
            )

    if args.record_out:
        out = Path(args.record_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if result["state"] == STATE_SUPPRESSED:
        print(f"{LABEL}: PASS", flush=True)
        print(result["detail"], flush=True)
        return EXIT_PASS

    print(f"{LABEL}: ERROR(instrument={result['state']})", flush=True)
    print(result["detail"], flush=True)
    print(
        f"{LABEL}: the check did not reach its observation. Per DESIGN.md §10.0.1 R13 "
        "this is NOT a detection and NOT a pass.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


if __name__ == "__main__":
    raise SystemExit(main())
