#!/usr/bin/env python3
"""``check_device_state`` — a lane may not publish a duration without a device-state record.

The finding this exists for
---------------------------

``gpu_steady_tail()`` is a variance test over a suffix, so it cannot see a bias.  A board
held at its 210 MHz idle clock produces a series that is perfectly steady about a wrong
mean, and **the wrong number carried the better RSD** (0.8035% at 10.99x wrong, against
0.8098% for the correct figure).  A low clock does not raise RSD; it lowers it.  R9
amendment 5 demotes that check from gate to precondition, and §10.0 obligation 8 puts a
**device-state record over the statistic's own suffix** in its place.

Why it is in ``ci/`` when no CI lane quotes a timing figure
-----------------------------------------------------------

Because that is true today by luck and this makes it true by construction.  I grepped
``ci/*.py`` for ``tenancy``, ``SOLE_TENANT``, ``sm_clock``, ``STEADY_UNCERTIFIED`` and
``gpu_steady`` and got zero matches: the lanes prove the EP *executed* and that the
verdict is attributed, and say nothing about the device state that produced any timing.
Survivable while no lane publishes a duration — and a lane that adds one tomorrow would
inherit the whole finding silently.  This step means it cannot: the first lane step that
writes a millisecond into an artifact turns this red on the same run.

The three terminal states, and why the third is the important one
------------------------------------------------------------------

R13, with obligation 8 amendments 2 and 3 deciding which is which:

* ``PASS`` — the lane published no timing figure, **or** every one it published carries a
  certified companion.
* ``FAIL(condition=STEADY_UNCERTIFIED)`` — a figure was published and its record is
  missing or incomplete, on a host that could have produced one.
* ``ERROR(instrument=device_state_producer_absent)`` — a figure was published on a host
  with **no device-state producer at all**.  This is the case the obligation was tightened
  for.  The cheapest way to satisfy obligation 8 as first worded is to take the
  measurement where the requirement is vacuous, and *a CI runner with no GPU telemetry is
  that loophole at industrial scale*.  Absence of telemetry is an instrument error here.
  It is never a pass, and it is never ``SOLE_TENANT``.

Note what this means for these lanes specifically, because it is not a temporary state:
**all three CI lanes run lavapipe, and a CPU renderer has no device clock to record.**
``device_state.lavapipe_note()`` is the written answer; the short form is that a lane on a
software rasteriser can never certify a device-clock figure, and no better probe changes
that.  So this check's ``PASS`` on those lanes is always the *first* kind — "published
nothing timed" — and that is the honest shape of it.

USAGE
    python ci/check_device_state.py --scan bench/results/ci-lane [--summary <file>]
                                    [--explain]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import device_state as ds  # noqa: E402

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

LABEL = "DEVICE-STATE"


def report_pass(detail: str) -> int:
    print(f"{LABEL}: PASS", flush=True)
    print(detail, flush=True)
    return EXIT_PASS


def report_fail(condition: str, detail: str) -> int:
    print(f"{LABEL}: FAIL(condition={condition})", flush=True)
    print(detail, flush=True)
    return EXIT_FAIL_CONDITION


def report_instrument_error(instrument: str, detail: str) -> int:
    print(f"{LABEL}: ERROR(instrument={instrument})", flush=True)
    print(detail, flush=True)
    print(
        f"{LABEL}: the check did not reach its observation. Per DESIGN.md §10.0.1 R13 "
        "this is NOT a detection and NOT a pass.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


def parse_args(argv):
    p = argparse.ArgumentParser(description="obligation-8 publication guard for a CI lane")
    p.add_argument(
        "--scan",
        action="append",
        default=[],
        help="file or directory of lane evidence to scan (repeatable)",
    )
    p.add_argument(
        "--summary",
        default="",
        help="optional free-text job summary or log; scanned by a second witness with a "
        "different failure mode (a JSON parser cannot see a duration in prose)",
    )
    p.add_argument(
        "--lane-marker",
        default="",
        help="path to a marker file the lane writes when it first produces evidence. Its "
        "purpose is to tell 'the lane published no duration' apart from 'the lane died "
        "before it could publish anything'. Both are non-passes, but only the first is a "
        "finding about device state; the second is a finding about an earlier step, and "
        "this guard must not add a second red to a lane that has already gone red "
        "somewhere else.",
    )
    p.add_argument(
        "--explain",
        action="store_true",
        help="print the producer registry and the lavapipe ruling, then exit 0",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.explain:
        print(json.dumps(ds.PRODUCERS, indent=2))
        print()
        print(ds.lavapipe_note())
        return EXIT_PASS

    if not args.scan:
        print(__doc__, flush=True)
        return EXIT_USAGE

    roots = [Path(s) for s in args.scan]
    missing = [r for r in roots if not r.exists()]
    if len(missing) == len(roots):
        # Every path absent. That is not "the lane published nothing" — it is this check
        # having no subject, and a check with no subject reports UNOBSERVABLE (R12), not
        # a clean lane.
        #
        # But there are two ways to have no subject, and they deserve different exit
        # codes even though neither is a pass. Observed on the 2026-08-01 main run: the
        # Linux lane died at Clippy, never reached the step that creates the evidence
        # directory, and this guard — correctly running under `always()` — reported
        # ERROR(instrument=lane_evidence_absent) and failed the job a second time. The
        # report was true and the second red was noise: no figure can be published by a
        # run that never got far enough to publish one, so the risk this guard exists to
        # stop was not present.
        #
        # The marker distinguishes them, and it can only ever be written by the lane's own
        # evidence-producing step, so "marker present" implies "the lane reached the point
        # where publishing was possible". That direction is the one that matters: the
        # marker cannot be absent on a run that did publish.
        if args.lane_marker and not Path(args.lane_marker).exists():
            print("DEVICE-STATE: ERROR(instrument=lane_did_not_reach_evidence)", flush=True)
            print(
                f"The lane marker {args.lane_marker} was never written, so the lane failed "
                "before it produced any evidence at all.",
                flush=True,
            )
            print(
                "This is NOT a pass and must not be read as one. It is also not a "
                "device-state finding: there is no figure to certify because there was no "
                "run to take one from. The lane is already red for the reason that "
                "stopped it, and this guard declines to add a second red on top of a "
                "first one it did not cause.",
                flush=True,
            )
            print(
                "::warning title=Device-state guard had no subject::"
                "The lane did not reach evidence production, so obligation 8 was neither "
                "satisfied nor violated. Fix the earlier failure and re-read this step.",
                flush=True,
            )
            return EXIT_PASS
        return report_instrument_error(
            "lane_evidence_absent",
            "None of the scanned paths exist: "
            + ", ".join(str(r) for r in roots)
            + ".\nThis guard cannot distinguish 'the lane published no duration' from "
            "'the lane produced no evidence at all', and the second is not a pass."
            + (
                f"\nThe lane marker {args.lane_marker} IS present, so the lane did reach "
                "evidence production and then produced none — which is an instrument "
                "failure, not an empty run."
                if args.lane_marker
                else ""
            ),
        )

    entries = ds.scan_paths([r for r in roots if r.exists()])
    unparseable = [e for e in entries if e.get("error")]
    if unparseable:
        first = unparseable[0]
        return report_instrument_error(
            "lane_evidence_unparseable",
            # R13 second clause: quote the failure text, never the failure count.
            f"{first['file']}: {first['error']}\n"
            "A document this guard cannot read is a document whose durations it cannot "
            "see, which is indistinguishable from a document with none.",
        )

    offenders = []
    certified = []
    carried = []
    for e in entries:
        durs = e.get("durations") or []
        if not durs:
            continue
        if e.get("instrument_dump"):
            carried.append((e, durs))
            continue
        result = ds.certify(e.get("companion"))
        if result["state"] == ds.CERTIFIED:
            certified.append((e, durs, result))
        else:
            offenders.append((e, durs, result))

    host = ds.host_producer_status()

    # Carried-not-claimed figures are printed on every run, pass or fail. A guard that
    # stays quiet about the numbers it decided not to act on is a guard whose scope
    # nobody can audit (R10: the falsifier is an artifact whose content varies with its
    # input, and this is that artifact).
    for e, durs in carried:
        quoted = ", ".join(f"{d['path']}={d['value']}" for d in durs[:6])
        print(
            f"{LABEL}: {ds.STEADY_UNCERTIFIED} (carried, not claimed) {e['file']}\n"
            f"    {quoted}{' …' if len(durs) > 6 else ''}\n"
            f"    instrument dump: {e['instrument_dump']}\n"
            f"    These are not quotable and this lane quotes none of them. If one is "
            f"ever quoted it becomes a lane claim and this guard requires its companion.",
            flush=True,
        )

    summary_hits = []
    if args.summary:
        sp = Path(args.summary)
        if sp.exists():
            summary_hits = ds.find_summary_durations(
                sp.read_text(encoding="utf-8", errors="replace")
            )

    if not offenders and not summary_hits:
        n_files = len(entries)
        if certified:
            lines = [
                f"{len(certified)} document(s) published a timing figure and every one "
                "carries a certified device-state record (§10.0 obligation 8)."
            ]
            for e, durs, result in certified:
                lines.append(
                    f"  {e['file']}: {len(durs)} figure(s); tenancy={result['tenancy']}; "
                    f"clock {result['clock_min_mhz']}/{result['clock_median_mhz']}/"
                    f"{result['clock_max_mhz']} MHz against a board maximum of "
                    f"{result['board_max_mhz']} MHz"
                )
            detail = "\n".join(lines)
        else:
            detail = (
                f"No lane-authored timing figure is published in any of {n_files} lane "
                f"artifact(s), so there is nothing for obligation 8 to qualify"
                + (
                    f" ({len(carried)} instrument dump(s) carry unquotable figures, "
                    "listed above)."
                    if carried
                    else "."
                )
                + "\nThis is the honest shape of a PASS on a lavapipe lane: a CPU renderer "
                "has no device clock, so a certified device-clock figure is not merely "
                "absent here, it is unavailable in principle "
                "(ci/device_state.py :: lavapipe_note).\n"
                f"Host producer status: {host['status']}; "
                f"available producers: {host['available'] or 'none'}."
            )
        return report_pass(detail)

    # Something published a duration without a usable record. Which of the two failing
    # states applies is decided by whether this host could have produced one at all.
    lines = []
    for e, durs, result in offenders:
        quoted = ", ".join(f"{d['path']}={d['value']}" for d in durs[:6])
        lines.append(f"  {e['file']}\n    figures: {quoted}\n    {result.get('detail', '')}")
    if summary_hits:
        lines.append(
            "  <job summary>\n    figures: "
            + ", ".join(repr(h) for h in summary_hits[:6])
            + "\n    A figure in prose is published exactly as much as one in JSON."
        )
    body = "\n".join(lines)

    # An unusable record and a missing one are different findings with different owners.
    # A record that says the probe failed routes to whoever owns the probe; a record that
    # is simply absent routes to whoever published the figure. Fold them together and the
    # finding goes to the wrong desk (R13).
    broken = next((r for _, _, r in offenders if r.get("state") == "ERROR"), None)
    if broken is not None:
        return report_instrument_error(
            broken.get("instrument", "device_state_record_unusable"), body
        )

    if not host["available"]:
        return report_instrument_error(
            "device_state_producer_absent",
            f"{body}\n\n"
            f"This host ({host['platform']}) has no device-state producer: "
            f"{host['available'] or 'none'} available.\n"
            "Obligation 8 amendment 2: the absence of the companion is NEVER a waiver. "
            "The cheapest way to satisfy the obligation as first worded is to measure on "
            "a platform with no telemetry, where the requirement is vacuous — and a CI "
            "runner with no GPU telemetry is that loophole at scale. So a duration "
            "published here is ERROR(instrument), not a pass and not SOLE_TENANT.\n"
            "Fix: publish this figure from a host with a producer, or do not publish it.",
        )

    return report_fail(
        ds.STEADY_UNCERTIFIED,
        f"{body}\n\n"
        f"This host has a producer available ({host['available']}), so the record could "
        "have been taken and was not. §10.0 obligation 8: absent a device-state record "
        "covering the statistic's own suffix, the figure is STEADY_UNCERTIFIED — not a "
        "failed measurement, a measurement with no reading yet.\n"
        f"Fix: wrap the measured command in the producer and splice its summary into the "
        f"artifact under '{ds.COMPANION_KEY}'.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
