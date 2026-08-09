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
``bench/phi_evidence.py::evidence_gate``; this file is a lane-facing wrapper that calls
``gate_artifact`` — exact bytes and exact byte length first, then that one function on the
parsed content — and translates its answer into this repository's terminal-state vocabulary.
``ci/negative_control_phi_evidence.py`` then attacks both directly, once per condition they
can report.

The byte check runs before the parse on purpose. A parse normalises: a CRLF-translated copy of
the artifact re-serialises to the same content digest as the original, so a content digest
alone cannot tell the two apart. The sidecar seal binds the file to its exact bytes *and* its
exact length, hashed as read, so a line-ending rewrite is refused rather than waved through.

Nothing printed below the verdict is read out of the record directly. It all comes from
``admissible_output``, which on a refusal withholds every timing, ratio, separation, speedup,
band edge and lower bound and scrubs numeric leaves from what remains. A refused row is not
allowed to leave a success-shaped number in this log for someone to quote.

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

    verdict = pe.gate_artifact(args.artifact)

    # Everything printed from here down comes out of `admissible_output`, never out of the
    # record directly. On a refusal that function withholds every timing, ratio, separation,
    # speedup, band edge and lower bound, so a refused row cannot be quoted out of this log.
    try:
        record = pe.load_frozen(args.artifact)
    except (pe.FrozenArtifactMissing, pe.FrozenArtifactUnreadable):
        record = None
    published = pe.admissible_output(record, verdict) if record is not None else {
        "verdict": verdict["verdict"], "condition": verdict.get("condition"),
        "detail": str(verdict.get("detail")), "subjects": [],
    }

    if args.json:
        Path(args.json).write_text(
            json.dumps(published, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if verdict["verdict"] == pe.ERROR:
        return report_instrument_error(
            str(verdict.get("condition") or "gate_error"), str(published.get("detail")))
    if verdict["verdict"] != pe.PASS:
        for row in published.get("subjects") or []:
            print(f"{TAG}:   {row.get('subject')}: {row.get('status')} — "
                  f"{row.get('admissible')}", flush=True)
        return report_fail(str(verdict.get("condition")), f"  {published.get('detail')}")

    sealed = verdict.get("frozen_bytes") or {}
    print(f"{TAG}: artifact {Path(args.artifact).name} "
          f"exact_bytes_sha256={sealed.get('sha256_of_exact_bytes')} "
          f"byte_length={sealed.get('byte_length')} "
          f"content_sha256={published.get('content_sha256')}", flush=True)
    subjects = published.get("subjects") or []
    for v in subjects:
        pe_pt = v.get("point_estimate")
        scope = v.get("band_scope") or {}
        band = (f"read against the committed {scope.get('source')} band "
                f"[{scope.get('lo')}, {scope.get('hi')}] and against no other band")
        print(f"{TAG}:   {v.get('subject')}: {v.get('verdict')} "
              f"(point estimate {pe_pt:.4f}x, floor {v.get('floor'):.4f}x; {band})"
              if pe_pt else f"{TAG}:   {v.get('subject')}: {v.get('verdict')}", flush=True)
    limits = published.get("claim_limits") or {}
    observations = published.get("decode_observations") or []
    print(f"{TAG}:   decode conclusion: {limits.get('decode_conclusion')}; "
          f"{len(observations)} observation(s) carried, none superseding another", flush=True)
    for obs in observations:
        interval = obs.get("interval") or {}
        extra = ""
        if interval.get("lo") is not None and interval.get("hi") is not None:
            extra += f" interval [{interval['lo']}, {interval['hi']}]"
        if obs.get("power") is not None:
            extra += f" power {obs['power']}"
        extra += (" raw samples held here" if obs.get("raw_samples_held_here")
                  else " raw samples NOT held here")
        print(f"{TAG}:     {obs.get('id')}: {obs.get('point_estimate')}x "
              f"{obs.get('verdict')}{extra}", flush=True)
    print(f"{TAG}:   CUDA comparison: {limits.get('cuda_comparison')}; "
          f"closes_issue_69={limits.get('closes_issue_69')}", flush=True)
    return report_pass(
        f"{len(subjects)} verdict subject(s) admissible; {published.get('detail')}")


if __name__ == "__main__":
    try:
        sys.exit(screen())
    except Exception as exc:  # pragma: no cover - last-resort outage translation
        print(f"{TAG}: ERROR(instrument=screen_raised) {exc!r}", flush=True)
        sys.exit(EXIT_ERROR_INSTRUMENT)
