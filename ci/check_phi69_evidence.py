#!/usr/bin/env python3
"""Lane guard for the #69 v5 Phi-3.5 evidence posture.

This is host-free: it imports the publication gate and rules on a frozen evidence record. It
has two jobs.

1. **Assert the standing posture.** No admissible timing artifact exists for #69 while this box
   is contended (docs/PERF.md §20/§28), so with no record the honest verdict is
   ``INDETERMINATE`` and the lane is green precisely *because* nothing is quoted. This is not a
   blocked task; a refusal is the instrument working.

2. **Rule on a record if one is supplied** (``--record path.json``). It runs the single
   publication authority `bench/phi69_evidence.py::publish`. The guard is red if a record
   claims ``timing_admissible`` (this box cannot certify one), if any correctness/structural
   condition fails, or if a suppressed timing number leaks into the published output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))

import phi69_evidence as pe  # noqa: E402

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 4


def _check_record(path: Path) -> int:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"PHI69-EVIDENCE: ERROR -- cannot read {path}: {exc}")
        return EXIT_ERROR
    pub = pe.publish(record)
    for line in pe._report(pub):
        print(line)
    if pub["timing_admissible"]:
        print(
            "PHI69-EVIDENCE: FAIL -- record claims timing is admissible, but this box cannot "
            "certify a wall-clock figure (docs/PERF.md section 20)."
        )
        return EXIT_FAIL
    # Structural + correctness witnesses must still hold even under a timing refusal.
    if not pub["correctness"]["all_output_equivalence"]:
        print("PHI69-EVIDENCE: FAIL -- all-output correctness did not hold.")
        return EXIT_FAIL
    # F3: derive the forbidden values from the record itself rather than hard-coding literals,
    # so the check embeds no measurement and catches a leak of any suppressed timing float.
    leaked = pe._residual_timing_leak(record, pub)
    if leaked:
        print(
            f"PHI69-EVIDENCE: FAIL -- {len(leaked)} suppressed timing number(s) leaked into "
            f"the output."
        )
        return EXIT_FAIL
    print("PHI69-EVIDENCE: PASS (INDETERMINATE timing; witnesses intact; nothing quoted).")
    return EXIT_PASS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", type=Path, default=None,
                    help="path to a phi69-evidence/v5 JSON record to rule on")
    ap.add_argument("--check", action="store_true",
                    help="assert the standing posture (default when no --record is given)")
    args = ap.parse_args(argv)

    if args.record is not None:
        return _check_record(args.record)

    # No record: affirm the standing INDETERMINATE posture. F5: a real registry-consistency
    # check -- every admissibility condition must be a known gate condition, and the registry
    # must be non-empty -- replaces a branch that compared a set to itself and could never fire.
    unknown = set(pe.TIMING_ADMISSIBILITY) - set(pe.GATE_CONDITIONS)
    if not pe.GATE_CONDITIONS or unknown:
        print(f"PHI69-EVIDENCE: ERROR -- gate registry inconsistent (unknown: {sorted(unknown)}).")
        return EXIT_ERROR
    print(
        "PHI69-EVIDENCE: INDETERMINATE -- no admissible timing artifact for issue #69 while the "
        "isolation gate is unavailable on this box (docs/PERF.md section 20/28). No timing, ratio "
        "or speedup is published, and no output-equivalence result is claimed absent a record; "
        "the gate enforces the correctness and structural requirements a record must satisfy."
    )
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
