#!/usr/bin/env python3
"""Gate the committed Phi-3.5 evidence artifact. One authority, run in the host-free lane.

WHY THIS EXISTS
===============
The Phi-3.5 real-model numbers in ``docs/PERF.md`` are the only numbers in this repository
that a reader is likely to quote at someone outside it. They are also the numbers that cost
the most to reproduce: a 2.3 GB weight file, two builds of the EP from two different commits,
and a GPU. That combination is exactly the shape in which a claim rots quietly — nobody
re-derives it, so nobody notices when the artifact underneath it stops saying what the prose
says it says.

So the artifact is frozen and digest-sealed, and **this** is the only thing in the repository
allowed to decide whether it may be published. Not the harness that produced it (a producer
that grades its own output grades it kindly), not the prose (prose cannot be run), and not a
second copy of the rules living in a test. The rules live in one function,
``bench/phi_evidence.py::evidence_gate``; this file is a lane-facing wrapper that runs it and translates
its answer into this repository's terminal-state vocabulary. ``ci/negative_control_phi_evidence.py``
then attacks that one function directly, once per condition it can report.

WHY IT IS HOST-FREE
===================
It reads a committed JSON file and recomputes a SHA-256 over it. No Vulkan, no ONNX Runtime,
no model, no GPU. That is deliberate: a gate that only runs where the measurement runs is a
gate that runs on one workstation, and the failure it is meant to catch — the artifact and the
documentation drifting apart — happens in a pull request, not on the workstation.

WHAT IT CANNOT DO
=================
It cannot tell you the measurement was *right*. It reads what was recorded. If the harness
mismeasured, every field here is consistent and wrong together, and this screen says PASS. What
it can do is refuse to let a recorded measurement be published as something other than what it
recorded — a hardware reading filed as software, a decode non-result filed as a win, a band
derived from the data it is judging, a headline widened past the one model that was measured.

TERMINAL STATES (§10.0.1 R13)
=============================
PASS / FAIL(condition=...) / ERROR(instrument=...), exits 0 / 1 / 4. An instrument error is
never a detection: an artifact that is absent or unparseable is an outage of this screen, and
the lane must not read it as evidence that the claims are sound.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "bench"))

from check_tick_conversions import (  # noqa: E402
    EXIT_ERROR_INSTRUMENT,
    EXIT_FAIL_CONDITION,
    EXIT_PASS,
)

TAG = "PHI-EVIDENCE"

DEFAULT_ARTIFACT = REPO_ROOT / "bench" / "results" / "phi35_evidence_v4.json"


def report_pass(detail: str) -> int:
    print(f"{TAG}: PASS — {detail}", flush=True)
    return EXIT_PASS


def report_fail(condition: str, detail: str) -> int:
    print(f"{TAG}: FAIL(condition={condition})", flush=True)
    print(detail, flush=True)
    print(
        f"{TAG}: this is a finding about what the committed artifact claims. The gate "
        "reached its observation; the offending field is quoted above.",
        flush=True,
    )
    return EXIT_FAIL_CONDITION


def report_instrument_error(instrument: str, detail: str) -> int:
    print(f"{TAG}: ERROR(instrument={instrument})", flush=True)
    print(detail, flush=True)
    print(
        f"{TAG}: the screen did not reach its observation, so this is NOT a detection "
        "(DESIGN.md §10.0.1 R13). Do not read it as admissible evidence and do not read it "
        "as a violation.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


def screen(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    ap.add_argument("--json", metavar="PATH",
                    help="also write the verdict as JSON, for a downstream reader")
    args = ap.parse_args(argv)

    try:
        import phi_evidence as pe
    except Exception as exc:  # pragma: no cover - import failure is an outage
        return report_instrument_error(
            "gate_unimportable",
            f"bench/phi_evidence.py could not be imported: {exc!r}. The gate is that module; "
            f"without it this screen has no rules to apply and must not invent any.",
        )

    try:
        record = pe.load_frozen(args.artifact)
    except pe.FrozenArtifactMissing as exc:
        return report_instrument_error(
            "artifact_absent",
            f"{exc}\n"
            f"  The evidence artifact is not in the tree. That is an outage of this screen, "
            f"not a clean bill of health for the documentation that cites it.",
        )
    except pe.FrozenArtifactUnreadable as exc:
        return report_instrument_error(
            "artifact_unparseable",
            f"{exc}\n"
            f"  The bytes exist and are not JSON. Nothing was ruled on.",
        )

    verdict = pe.evidence_gate(record)
    if args.json:
        Path(args.json).write_text(
            json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if verdict["verdict"] == pe.ERROR:
        return report_instrument_error(
            str(verdict.get("condition") or "gate_error"), str(verdict.get("detail")))
    if verdict["verdict"] != pe.PASS:
        return report_fail(str(verdict.get("condition")), f"  {verdict.get('detail')}")

    subjects = [v.get("subject") for v in record.get("verdicts") or []]
    print(f"{TAG}: artifact {Path(args.artifact).as_posix().split('/')[-1]} "
          f"content_sha256={record['identity']['content_sha256']}", flush=True)
    for v in record.get("verdicts") or []:
        pe_pt = v.get("point_estimate")
        print(f"{TAG}:   {v.get('subject')}: {v.get('verdict')} "
              f"(point estimate {pe_pt:.4f}x, floor {v.get('floor'):.4f}x)"
              if pe_pt else f"{TAG}:   {v.get('subject')}: {v.get('verdict')}", flush=True)
    print(f"{TAG}:   decode conclusion: "
          f"{record['claim_limits']['decode_conclusion']}; "
          f"{len(record.get('decode_observations') or [])} observation(s) carried, none "
          f"superseding another", flush=True)
    print(f"{TAG}:   CUDA comparison: {record['claim_limits']['cuda_comparison']}; "
          f"closes_issue_69={record['claim_limits']['closes_issue_69']}", flush=True)
    return report_pass(
        f"{len(subjects)} verdict subject(s) admissible; {verdict['detail']}")


if __name__ == "__main__":
    try:
        sys.exit(screen())
    except Exception as exc:  # pragma: no cover - last-resort outage translation
        print(f"{TAG}: ERROR(instrument=screen_raised) {exc!r}", flush=True)
        sys.exit(EXIT_ERROR_INSTRUMENT)
