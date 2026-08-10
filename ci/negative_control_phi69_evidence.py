#!/usr/bin/env python3
"""Negative control for the #69 v5 publication gate (`bench/phi69_evidence.py`).

A gate that has only ever been seen passing is one step from a gate that cannot fail. This
mounts **at least one attack per condition** in `GATE_CONDITIONS` and fails loudly if any
condition is left unattacked, so the battery cannot rot silently as conditions are added. It
runs host-free: no GPU, no ORT session, no clock — the gate is pure over a frozen record.

Arms are labelled by provenance, following the house convention:

* ``OBSERVED`` — the arm reflects the real state of this box, not a value constructed to make
                 the gate fire. Two conditions are OBSERVED here: this machine's quiescence is
                 ``CONTENDED`` and no device-state companion exists (docs/PERF.md §20).
* ``PLANTED``  — text written to exercise a path. Proves the path is wired; proves nothing
                 about whether it is reached in the wild.
"""

from __future__ import annotations

import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))

import phi69_evidence as pe  # noqa: E402

EXIT_PASS = 0
EXIT_UNATTACKED = 1
EXIT_ATTACK_SURVIVED = 2

OBSERVED = "OBSERVED"
PLANTED = "PLANTED"


def _rows(verdict: str = "MATCH") -> list:
    return [{"index": i, "verdict": verdict} for i in range(pe.EXPECTED_OUTPUTS)]


def _admissible() -> dict:
    text = "line one\nline two\n"
    # Synthetic placeholder timings (repeated-digit sentinels); they trace to no measurement and
    # match no recorded artifact. Clean-room by construction.
    return {
        "schema": "phi69-evidence/v5",
        "subjects": [
            {"name": "prefill/M128", "pooled": False, "arms": [
                {"arm": "head", "aggregate": "MATCH", "per_output": _rows(),
                 "median_ms": 1111.0, "point_ratio": 1.111},
                {"arm": "pre72", "aggregate": "MATCH", "per_output": _rows(), "median_ms": 2222.0},
            ]},
            {"name": "decode/M1/past128", "pooled": False, "timing_verdict": pe.INDETERMINATE,
             "preserved_observations": list(pe.REQUIRED_DECODE_OBSERVATIONS), "arms": [
                {"arm": "head", "aggregate": "MATCH", "per_output": _rows(), "median_ms": 333.0},
                {"arm": "pre72", "aggregate": "MATCH", "per_output": _rows(), "median_ms": 444.0},
            ]},
        ],
        "raw_runs": [
            {"arm": "head", "source_commit": "a" * 40, "dll_sha256": "d1",
             "build_recipe_sha256": "r1", "worktree_dirty": False},
            {"arm": "pre72", "source_commit": "b" * 40, "dll_sha256": "d2",
             "build_recipe_sha256": "r2", "worktree_dirty": False},
        ],
        "device": {"uuid": f"uuid:{pe.PINNED_DEVICE_UUID}", "luid": "0x00-0x1234",
                   "pci_bus_id": "0000:01:00.0", "driver_version": "573.44",
                   "device_type": "discrete-gpu", "driver_name": "NVIDIA RTX A1000"},
        "model": {"graph_sha256": "g" * 64, "weights_sha256": "w" * 64,
                  "provenance": {"foundry_variant_id": "Phi-3.5-mini-instruct-generic-gpu"}},
        "quiescence": {"verdict": pe.QUIET},
        "device_state": {"verdict": pe.COMPANION_QUOTABLE},
        "feeds_digest_by_subject": {"cal/A": "feed-d1", "prefill/M128": "feed-d2"},
        "calibration_subjects": ["cal/A"],
        "verdict_subjects": ["prefill/M128"],
        "provenance": {"base_delta_claim":
                       "the delta changed bench tooling, docs and CI; no Rust or shader source changed"},
        "refusals": [],
        "content_digests": [{"name": "proof_ledger", "source_text": text,
                             "recorded": pe._lf_digest(text)}],
        "within_arm_dispersion": {"role": "diagnostic", "moved_a_verdict": False},
        "isolation": {"language": "cooperative process exclusion only; no GPU lock"},
        "headline_scope": {"model": "one", "prefill_family": "one", "adapter": "one",
                           "box": "one", "cuda_comparison": "NONE", "closes_issue_69": False},
        "uncertainty": {"aa_band": [0.90, 1.05], "power_boost_qualification":
                        "the A/A band does not establish quiet, boost or concurrent-GPU state"},
        "witnesses": {"islands": 33, "dispatches": 40},
    }


# (condition, provenance, mutator)
ATTACKS = [
    ("record_wellformed", PLANTED, lambda r: r.__setitem__("schema", "phi_evidence/v4")),
    ("immutable_run_binding", PLANTED, lambda r: r["raw_runs"][0].__setitem__("source_commit", "HEAD")),
    ("device_identity_immutable", PLANTED, lambda r: r["device"].__setitem__("uuid", "uuid:" + "0" * 32)),
    ("model_identity_provenance", PLANTED, lambda r: r["model"].__setitem__("provenance", {})),
    ("quiescence_quiet", OBSERVED, lambda r: r["quiescence"].__setitem__("verdict", pe.CONTENDED)),
    ("device_state_companion", OBSERVED, lambda r: r.__setitem__("device_state", None)),
    ("per_output_integrity", PLANTED, lambda r: r["subjects"][0]["arms"][0].__setitem__("per_output", _rows()[:1])),
    ("all_output_equivalence", PLANTED, lambda r: r["subjects"][0]["arms"][0]["per_output"][7].__setitem__("verdict", "DIVERGENT")),
    ("calibration_content_disjoint", PLANTED, lambda r: r["feeds_digest_by_subject"].__setitem__("cal/A", "feed-d2")),
    ("decode_p128_separate", PLANTED, lambda r: r["subjects"][1].__setitem__("timing_verdict", "IMPROVEMENT")),
    ("provenance_claim_accurate", PLANTED, lambda r: r["provenance"].__setitem__("base_delta_claim", "CI-only, touches no bench source")),
    ("refusal_output_sanitized", PLANTED, lambda r: r.__setitem__("refusals", [{"detail": r"C:\Users\secret-user\thing"}])),
    ("digests_platform_stable", PLANTED, lambda r: r["content_digests"][0].__setitem__("recorded", "deadbeef")),
    ("no_dispersion_promotion", PLANTED, lambda r: r["within_arm_dispersion"].__setitem__("moved_a_verdict", True)),
    ("no_dispersion_promotion", PLANTED, lambda r: r.__delitem__("within_arm_dispersion")),
    ("isolation_language_cooperative", PLANTED, lambda r: r["isolation"].__setitem__("language", "exclusive GPU ownership")),
    ("headline_scope_not_widened", PLANTED, lambda r: r["headline_scope"].__setitem__("model", "all")),
    ("uncertainty_qualified", PLANTED, lambda r: r["uncertainty"].__setitem__("power_boost_qualification", "")),
]


def main(argv=None) -> int:
    # 1. The battery must attack every condition; an unattacked condition can rot.
    attacked = {name for name, _, _ in ATTACKS}
    missing = set(pe.GATE_CONDITIONS) - attacked
    if missing:
        print(f"NEGATIVE CONTROL FAILED: conditions with no attack arm: {sorted(missing)}")
        return EXIT_UNATTACKED

    # 2. The accept polarity must actually publish, or the fixture is broken.
    baseline = pe.publish(_admissible())
    if not baseline["timing_admissible"]:
        print("NEGATIVE CONTROL FAILED: admissible fixture did not publish a QUOTABLE verdict")
        print(baseline["refusals"])
        return EXIT_ATTACK_SURVIVED

    # 3. Each attack must be caught by its own condition.
    n_observed = n_planted = 0
    for name, provenance, mutate in ATTACKS:
        record = _admissible()
        mutate(record)
        result = pe.GATE_CONDITIONS[name](record)
        ok = result[0]
        if ok:
            print(f"  [{provenance}] {name}: ATTACK SURVIVED -- condition still passed")
            return EXIT_ATTACK_SURVIVED
        pub = pe.publish(record)
        if not any(rf["condition"] == name for rf in pub["refusals"]):
            print(f"  [{provenance}] {name}: caught by condition but absent from publish() refusals")
            return EXIT_ATTACK_SURVIVED
        # On refusal, no suppressed timing number may leak into the published subjects. The
        # banned set is derived from the record under test (F3) -- no hard-coded literals -- and
        # is value-based, so a number smuggled under an innocuous key is caught too (F2).
        leaked = pe._residual_timing_leak(record, pub)
        if leaked:
            print(f"  [{provenance}] {name}: {len(leaked)} timing number(s) leaked past suppression")
            return EXIT_ATTACK_SURVIVED
        n_observed += provenance == OBSERVED
        n_planted += provenance == PLANTED
        print(f"  [{provenance}] {name}: caught -- {result[1]}")

    print(
        f"NEGATIVE CONTROL PASSED: {len(ATTACKS)} arms "
        f"({n_observed} OBSERVED / {n_planted} PLANTED), "
        f"all {len(pe.GATE_CONDITIONS)} conditions attacked and caught."
    )
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
