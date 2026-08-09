#!/usr/bin/env python3
"""Attack `bench/phi_evidence.py::evidence_gate` once per condition it claims to be able to report.

WHY THIS EXISTS
===============
``ci/check_phi_evidence.py`` is green. That is not, by itself, information: a screen that
cannot go red is green about everything, and the most expensive way to learn that is to find
out after the claim has been quoted somewhere. This file is the falsification arm. It takes the
real committed artifact, breaks exactly one thing about it, and requires the gate to convict on
exactly the condition that thing belongs to.

The list of conditions is not written here. It is ``phi_evidence.GATE_CONDITIONS``, walked at
run time, and a token with no arm in this file is itself a failure. Adding a condition to the
gate and forgetting to attack it is therefore not possible quietly.

WHAT EACH ARM IS
================
``PLANTED``  — a defect this file introduced on purpose, to see whether the gate notices.
``OBSERVED`` — a property of the real, unmutated artifact, asserted live.

Both matter. A gate that convicts everything is as useless as one that convicts nothing, so the
healthy arm (the real artifact, untouched, must PASS) is a control in the same sense the broken
ones are.

WHY IT MUTATES THE RECORD AND RE-FREEZES
========================================
Most arms re-seal the mutated record, because otherwise every single one of them would be
caught by the digest check and would prove nothing about the condition it is aimed at. The
digest arm is the exception and does *not* re-seal: that is the whole point of it.

TERMINAL STATES (§10.0.1 R13)
=============================
PASS / FAIL / ERROR, exits 0 / 1 / 4. An arm that could not be constructed is an outage of this
control, not a detection about the gate.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))

import phi_evidence as pe  # noqa: E402

ARTIFACT = REPO_ROOT / "bench" / "results" / "phi35_evidence_v4.json"

RESULTS: "list[tuple[str, str, str, bool, str]]" = []


def record(kind: str, condition: str, name: str, ok: bool, detail: str) -> None:
    RESULTS.append((kind, condition, name, ok, detail))
    print(f"  [{'ok ' if ok else 'RED'}] {kind:8s} {condition:36s} {name}")
    if not ok:
        print(f"           {detail}")


def _reseal(record_: dict) -> dict:
    """Re-stamp the digest so an arm is judged on its content, not on its seal."""
    record_.setdefault("identity", {}).pop("content_sha256", None)
    record_["identity"]["content_sha256"] = pe.content_digest(record_)
    return record_


def _convicts(mutated: dict, condition: str) -> "tuple[bool, str]":
    verdict = pe.evidence_gate(mutated)
    ok = verdict["verdict"] == pe.FAIL and verdict.get("condition") == condition
    return ok, (f"gate said {verdict['verdict']}"
                f"({verdict.get('condition')}): {str(verdict.get('detail'))[:160]}")


def _arm(base: dict, condition: str, name: str, mutate, *, reseal: bool = True) -> None:
    mutated = copy.deepcopy(base)
    try:
        mutate(mutated)
    except Exception as exc:  # pragma: no cover - a broken arm is an outage
        record("PLANTED", condition, name, False, f"arm could not be built: {exc!r}")
        return
    if reseal:
        _reseal(mutated)
    ok, detail = _convicts(mutated, condition)
    record("PLANTED", condition, name, ok, detail)


# --------------------------------------------------------------------------------------------
# The mutations, one per condition token.
# --------------------------------------------------------------------------------------------


def _m_schema(r):
    r["schema"] = "phi35_frozen_evidence/999"


def _m_digest(r):
    # No reseal. A single flipped digit in a published ratio, and nothing else.
    r["verdicts"][0]["point_estimate"] = (r["verdicts"][0]["point_estimate"] or 1.0) + 1.0


def _m_device(r):
    r["environment"]["device"].pop("uuid", None)


def _m_impl_label(r):
    # The exact rejection this gate was built for: a discrete-GPU reading filed as a software
    # rasteriser's, and the same mistake in reverse.
    r["environment"]["device"]["driver_name"] = "llvmpipe (LLVM 17, 256 bits)"


def _m_isolation(r):
    r["isolation"]["what_it_does"] = (
        "the harness takes exclusive GPU ownership for the duration of a point")


def _m_calibration_disjoint(r):
    r["calibration"]["subjects"].append(r["verdicts"][0]["subject"])


def _m_calibration_same_bytes(r):
    """Two calibration subjects under different names feeding byte-identical tensors.

    This is not hypothetical: the first sweep taken for this artifact used `decode/M1/past0`
    and `prefill/M1/past0`, which build the same feeds, and the band silently counted one
    measurement twice. The gate now convicts on it, which is why the shipped case set does not
    contain it.
    """
    digests = r["environment"]["feeds_digest_by_subject"]
    cal = r["calibration"]["subjects"]
    digests[cal[1]] = digests[cal[0]]


def _m_calibration_borrows_a_verdict_input(r):
    digests = r["environment"]["feeds_digest_by_subject"]
    digests[r["calibration"]["subjects"][0]] = digests[r["verdicts"][0]["subject"]]


def _m_band_self_derived(r):
    r["calibration"]["band"]["derived_from"] = [r["verdicts"][0]["subject"]]


def _m_equivalence(r):
    # The "output 0 is the model" mistake: compare the logits and call the KV blocks checked.
    case = r["equivalence"][0]
    case["outputs_compared"] = 1
    for arm in case["arms"]:
        if not arm.get("self"):
            arm["outputs_compared"] = 1
            arm["per_output"] = arm["per_output"][:1]


def _m_witness(r):
    w = r["equivalence"][0]["production_witness"]
    w["vulkan_node_executions"] = 0
    w["provider_requested_only"] = True


def _m_headline(r):
    r["headline"]["models"] = [pe.HEADLINE_MODEL, "resnet50"]


def _m_claim_cuda(r):
    r["claim_limits"]["cuda_comparison"] = "PARITY"


def _m_claim_closes(r):
    r["claim_limits"]["closes_issue_69"] = True


def _m_claim_decode(r):
    r["claim_limits"]["decode_conclusion"] = pe.IMPROVEMENT


def _m_decode_dropped(r):
    kept = [o for o in r["decode_observations"] if o.get("id") != "prior-point-estimate"]
    r["decode_observations"] = kept


def _m_decode_uncertainty_stripped(r):
    for o in r["decode_observations"]:
        if o.get("point_estimate") == 0.9651:
            o["interval"] = None
            o["power"] = None


def _m_decode_superseded(r):
    r["decode_observations"][0]["superseded_by"] = r["decode_observations"][1]["id"]


def _m_dispersion(r):
    v = r["verdicts"][0]
    v["basis"] = "within-arm-dispersion"
    v["dispersion"]["role"] = "authoritative"


def _m_ledger_absent(r):
    r["proof_ledger"].pop("file_sha256", None)


def _m_ledger_unenforced(r):
    enf = r["equivalence"][0]["runtime_enforcement"]
    enf["claimed_without_ledger_hit"] = 3


def _m_private_path(r):
    r["environment"]["software"]["ep_library_path"] = \
        r"C:\Users\somebody\repos\ep\target\release\onnxruntime_vulkan_ep.dll"


def _m_verdict_disagrees(r):
    for v in r["verdicts"]:
        if v["verdict"] == pe.INDETERMINATE:
            v["verdict"] = pe.IMPROVEMENT
            return
    r["verdicts"][0]["verdict"] = pe.REGRESSION


ARMS = (
    ("schema_unknown", "an artifact written to a schema this gate has never read", _m_schema,
     True),
    ("identity_digest_mismatch", "one published ratio edited after freezing", _m_digest, False),
    ("device_identity_incomplete", "the adapter UUID removed", _m_device, True),
    ("vulkan_implementation_mislabelled",
     "a hardware reading whose driver reads llvmpipe", _m_impl_label, True),
    ("isolation_overclaimed", "prose claiming exclusive GPU ownership", _m_isolation, True),
    ("calibration_not_disjoint",
     "a verdict subject smuggled into the calibration set", _m_calibration_disjoint, True),
    ("calibration_not_disjoint",
     "two calibration subjects that feed byte-identical inputs under different names",
     _m_calibration_same_bytes, True),
    ("calibration_not_disjoint",
     "a calibration subject fed the same input as a verdict subject",
     _m_calibration_borrows_a_verdict_input, True),
    ("band_self_derived", "the band re-derived from the verdict's own data",
     _m_band_self_derived, True),
    ("equivalence_incomplete", "only output 0 compared, the 64 KV blocks unchecked",
     _m_equivalence, True),
    ("production_path_unwitnessed", "the EP requested but executing nothing", _m_witness, True),
    ("headline_scope_widened", "a second model added to a one-model headline", _m_headline,
     True),
    ("claim_limit_violated", "a CUDA parity claim with no CUDA measurement", _m_claim_cuda,
     True),
    ("claim_limit_violated", "the artifact declaring issue #69 closed", _m_claim_closes, True),
    ("claim_limit_violated", "the decode conclusion promoted to a win", _m_claim_decode, True),
    ("decode_observation_dropped", "one of the two independent decode observations deleted",
     _m_decode_dropped, True),
    ("decode_observation_dropped",
     "the surviving observation's interval and power stripped off it",
     _m_decode_uncertainty_stripped, True),
    ("decode_observation_dropped", "one decode observation declared superseded by the other",
     _m_decode_superseded, True),
    ("dispersion_promoted", "a verdict rebased onto within-arm dispersion", _m_dispersion,
     True),
    ("proof_ledger_absent", "the ledger identity removed", _m_ledger_absent, True),
    ("proof_ledger_absent", "claimed nodes with no ledger entry behind them",
     _m_ledger_unenforced, True),
    ("private_path_disclosed", "a home-directory path in a committed field", _m_private_path,
     True),
    ("verdict_disagrees_with_classifier", "an INDETERMINATE subject relabelled a win",
     _m_verdict_disagrees, True),
)


def main() -> int:
    try:
        base = pe.load_frozen(ARTIFACT)
    except (pe.FrozenArtifactMissing, pe.FrozenArtifactUnreadable) as exc:
        print(f"NEGATIVE-CONTROL: ERROR(instrument=artifact_unusable) {exc}")
        print("  No arm was run. This is an outage of the control, not a finding.")
        return 4

    print("healthy arm — the real artifact, untouched")
    healthy = pe.evidence_gate(copy.deepcopy(base))
    record("OBSERVED", "-", "the committed artifact is admissible as it stands",
           healthy["verdict"] == pe.PASS, json.dumps(healthy)[:200])

    print("\nthe classifier is symmetric, structurally, not by assertion")
    band = base["calibration"]["band"]
    up = next((v for v in base["verdicts"] if v["verdict"] == pe.IMPROVEMENT), None)
    if up:
        forward = pe.classify_ratio(up["series"], band)
        mirrored = pe.classify_ratio(pe.mirror_series(up["series"]), pe.mirror_band(band))
        record("OBSERVED", "-",
               "mirroring a measured IMPROVEMENT yields exactly REGRESSION",
               forward["verdict"] == pe.IMPROVEMENT and mirrored["verdict"] == pe.REGRESSION,
               f"forward={forward['verdict']} mirrored={mirrored['verdict']}")
    flat = pe.classify_ratio({"ratios": [1.0, 1.0, 1.0]}, band)
    record("OBSERVED", "-", "a subject that did not move is INDETERMINATE in both directions",
           flat["verdict"] == pe.INDETERMINATE, f"got {flat['verdict']}")

    print("\ndispersion is diagnostic: it cannot move a verdict")
    widened = copy.deepcopy(base)
    for v in widened["verdicts"]:
        d = v.get("dispersion") or {}
        if d:
            d["rsd"] = 9.99
            d["stdev_ms"] = 1e6
    _reseal(widened)
    after = pe.evidence_gate(widened)
    record("OBSERVED", "-",
           "a hundredfold within-arm spread changes no verdict and no admissibility",
           after["verdict"] == pe.PASS
           and [v["verdict"] for v in widened["verdicts"]]
           == [v["verdict"] for v in base["verdicts"]],
           f"gate={after['verdict']}({after.get('condition')})")

    print("\nplanted defects — one per condition the gate claims to report")
    for condition, name, mutate, reseal in ARMS:
        _arm(base, condition, name, mutate, reseal=reseal)

    print("\ncoverage of the gate's own condition list")
    attacked = {c for c, *_ in ARMS}
    uncovered = [c for c in pe.GATE_CONDITIONS if c not in attacked]
    record("OBSERVED", "-",
           "every token in phi_evidence.GATE_CONDITIONS has at least one arm here",
           not uncovered, f"unattacked: {uncovered}")
    unknown = sorted(attacked - set(pe.GATE_CONDITIONS))
    record("OBSERVED", "-", "no arm aims at a condition the gate cannot report",
           not unknown, f"unknown tokens: {unknown}")

    passed = sum(1 for *_x, ok, _d in RESULTS if ok)
    total = len(RESULTS)
    kinds: dict[str, int] = {}
    for kind, *_ in RESULTS:
        kinds[kind] = kinds.get(kind, 0) + 1
    print("")
    print(f"{passed}/{total} arms behaved as declared  "
          f"({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})")
    if passed != total:
        print("NEGATIVE-CONTROL: FAIL — an arm did not behave as declared. Either the gate "
              "cannot detect what it says it detects, or this control is aimed wrong. Both "
              "are findings.")
        return 1
    print("NEGATIVE-CONTROL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
