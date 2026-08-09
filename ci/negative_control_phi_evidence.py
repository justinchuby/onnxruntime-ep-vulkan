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


# ---- per-record provenance: one arm per field the reviewer named ---------------------------


def _first_run(r) -> dict:
    return r["raw"]["runs"][0]


def _drop_provenance_field(field):
    def mutate(r):
        _first_run(r)["provenance"].pop(field, None)
    return mutate


def _m_prov_library_digest_edited(r):
    _first_run(r)["provenance"]["ep_library_sha256"] = "0" * 64


def _m_prov_library_not_mapped(r):
    _first_run(r)["provenance"]["ep_library_loaded_in_process"]["found"] = False


def _m_prov_resolver_disagrees(r):
    _first_run(r)["provenance"]["model_resolver"]["agrees_with_recorded_provenance"] = False


def _m_prov_resolver_agreement_unknown(r):
    _first_run(r)["provenance"]["model_resolver"]["agrees_with_recorded_provenance"] = None


def _m_prov_weight_digest_edited(r):
    _first_run(r)["provenance"]["external_weights"]["files"][0]["sha256"] = "b" * 64


def _m_prov_weights_unscanned(r):
    _first_run(r)["provenance"]["external_weights"]["scanned"] = False


def _m_prov_device_name_swapped(r):
    _first_run(r)["provenance"]["device_name"] = "llvmpipe (LLVM 17, 256 bits)"


def _m_prov_shader_count_zero(r):
    _first_run(r)["provenance"]["shaders"]["count"] = 0


def _m_prov_shader_digest_blanked(r):
    _first_run(r)["provenance"]["shaders"]["digest"] = ""


def _m_prov_dispatches_zero(r):
    _first_run(r)["provenance"]["dispatches_executed"] = 0


def _m_prov_dispatches_stringified(r):
    _first_run(r)["provenance"]["dispatches_executed"] = "many"


def _m_prov_agreement_flag_flipped(r):
    pairs = _first_run(r)["provenance"]["provenance_agreement"]["pairs"]
    pairs[0]["agree"] = not pairs[0]["agree"]


def _m_prov_agreement_side_edited(r):
    pairs = _first_run(r)["provenance"]["provenance_agreement"]["pairs"]
    for p in pairs:
        if p["field"] == "device_name":
            p["right"] = "some other adapter"


def _m_prov_agreement_verdict_reworded(r):
    _first_run(r)["provenance"]["provenance_agreement"]["verdict"] = "PROBABLY"


def _m_prov_agreement_pair_removed(r):
    prov = _first_run(r)["provenance"]["provenance_agreement"]
    prov["pairs"] = [p for p in prov["pairs"] if p["field"] != "ep_library_loaded"]


def _m_prov_agreement_type_broken(r):
    pairs = _first_run(r)["provenance"]["provenance_agreement"]["pairs"]
    pairs[0]["agree"] = "yes"


# ---- the loader contract, the band scope, the reconciliation, the ledger's reach ------------


def _m_loader_claims_validation(r):
    r["identity"]["loader_contract"]["load_frozen"]["validates_content_digest"] = True


def _m_loader_claims_stripping(r):
    r["identity"]["loader_contract"]["load_frozen"]["strips_superseded_blocks"] = True


def _m_loader_claims_refusal(r):
    r["identity"]["loader_contract"]["load_frozen"]["refuses_tampered_content"] = True


def _m_band_scope_removed(r):
    r["verdicts"][0].pop("band_scope", None)


def _m_band_scope_declared_universal(r):
    r["verdicts"][0]["band_scope"]["band_independent"] = True


def _m_band_scope_edges_edited(r):
    r["verdicts"][0]["band_scope"]["hi"] = 1.5


def _m_alternative_band_misreported(r):
    """The M64 shape exactly: a subject reported as indeterminate under a band that classifies it.

    A +/-3% band is narrower than the committed one, and a subject whose every repeat sits above
    1.5 is FASTER under it. Recording INDETERMINATE there is the claim that the subject is
    indeterminate whatever band you pick, which is a claim about bands nobody measured.
    """
    for v in r["verdicts"]:
        for alt in v.get("alternative_bands") or []:
            if alt["name"] == "hypothetical-3pc":
                alt["verdict"] = pe.INDETERMINATE
                return


def _m_alternative_bands_removed(r):
    r["verdicts"][0].pop("alternative_bands", None)


def _m_reconciliation_removed(r):
    r.pop("decode_observations_reconciliation", None)


def _m_reconciliation_drops_an_observation(r):
    rec = r["decode_observations_reconciliation"]
    rec["observation_ids"] = rec["observation_ids"][:-1]


def _m_reconciliation_drops_a_point_estimate(r):
    rec = r["decode_observations_reconciliation"]
    rec["point_estimates"] = [x for x in rec["point_estimates"] if round(x, 4) != 0.859]


def _m_reconciliation_concludes_a_win(r):
    r["decode_observations_reconciliation"]["conclusion"] = pe.IMPROVEMENT


def _m_reconciliation_arbitrates(r):
    r["decode_observations_reconciliation"]["arbitrated"] = True


def _m_ledger_called_diagnostic(r):
    r["proof_ledger"]["production_reachability"]["diagnostic_only"] = True


def _m_ledger_consumer_dropped(r):
    reach = r["proof_ledger"]["production_reachability"]
    reach["consumers"] = [c for c in reach["consumers"] if c["role"] != "pipeline-audit"]


def _m_ledger_consumer_unsourced(r):
    r["proof_ledger"]["production_reachability"]["consumers"][0].pop("symbol", None)


def _m_refused_row_keeps_its_numbers(r):
    v = r["verdicts"][0]
    v["status"] = "REFUSED"


def _m_refused_row_keeps_a_win(r):
    v = r["verdicts"][0]
    for key in list(v):
        if key in pe.RESULT_SHAPED_KEYS:
            v.pop(key)
    v["status"] = "REFUSED"
    v["verdict"] = pe.IMPROVEMENT


def _m_discarded_list_removed(r):
    r["raw"].pop("discarded_runs", None)


def _m_discard_rule_removed(r):
    r["raw"].pop("discard_rule", None)


def _m_discard_rule_is_a_timing_rule(r):
    r["raw"]["discard_rule"] = "attempts slower than the arm's median were re-run"


def _m_discarded_entry_hides_its_samples(r):
    r["raw"]["discarded_runs"] = [{
        "phase": "prefill", "m": 128, "past": 0, "reason": "did not run on Vulkan",
        "providers_reported": ["CPUExecutionProvider"], "dispatches_executed": 0,
        "criterion": "structural", "samples_ms": [],
    }]


def _m_discarded_entry_unattributed(r):
    r["raw"]["discarded_runs"] = [{"reason": "did not run on Vulkan", "samples_ms": [1.0]}]


def _m_run_reports_only_cpu(r):
    for run in r["raw"]["runs"]:
        if run.get("arm") == pe.ARM_BASELINE:
            run["providers_reported"] = ["CPUExecutionProvider"]
            break


def _m_run_reports_no_providers(r):
    for run in r["raw"]["runs"]:
        if run.get("arm") == pe.ARM_HEAD:
            run["providers_reported"] = []
            break


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
    ("loader_contract_misdescribed",
     "the record claiming load_frozen validates the digest it does not validate",
     _m_loader_claims_validation, True),
    ("loader_contract_misdescribed",
     "the record claiming the loader strips superseded blocks, which it does not",
     _m_loader_claims_stripping, True),
    ("loader_contract_misdescribed",
     "the record claiming the loader refuses tampered content, which it does not",
     _m_loader_claims_refusal, True),
    ("record_provenance_incomplete", "the per-record EP library digest deleted",
     _drop_provenance_field("ep_library_sha256"), True),
    ("record_provenance_incomplete", "the in-process module witness deleted",
     _drop_provenance_field("ep_library_loaded_in_process"), True),
    ("record_provenance_incomplete", "the model resolver block deleted",
     _drop_provenance_field("model_resolver"), True),
    ("record_provenance_incomplete", "the external-weight identity deleted",
     _drop_provenance_field("external_weights"), True),
    ("record_provenance_incomplete", "the device name deleted",
     _drop_provenance_field("device_name"), True),
    ("record_provenance_incomplete", "the dispatched-shader identity deleted",
     _drop_provenance_field("shaders"), True),
    ("record_provenance_incomplete", "the dispatch count deleted",
     _drop_provenance_field("dispatches_executed"), True),
    ("record_provenance_incomplete", "the provenance-agreement block deleted",
     _drop_provenance_field("provenance_agreement"), True),
    ("record_provenance_incomplete",
     "the provenance-agreement flag replaced by a string, so its type no longer decides anything",
     _m_prov_agreement_type_broken, True),
    ("record_provenance_incomplete", "one required agreement pair removed",
     _m_prov_agreement_pair_removed, True),
    ("record_provenance_incomplete",
     "the model provenance agreement recorded as unknown rather than agreed",
     _m_prov_resolver_agreement_unknown, True),
    ("record_provenance_incomplete", "the external-weight scan marked incomplete",
     _m_prov_weights_unscanned, True),
    ("record_provenance_incomplete", "a shader digest blanked", _m_prov_shader_digest_blanked,
     True),
    ("record_provenance_incomplete", "the shader count zeroed", _m_prov_shader_count_zero, True),
    ("record_provenance_incomplete", "the dispatch count zeroed", _m_prov_dispatches_zero, True),
    ("record_provenance_incomplete", "the dispatch count replaced by a word",
     _m_prov_dispatches_stringified, True),
    ("record_provenance_disagrees", "the per-record library digest edited",
     _m_prov_library_digest_edited, True),
    ("record_provenance_disagrees", "the library the timing process actually mapped removed",
     _m_prov_library_not_mapped, True),
    ("record_provenance_disagrees", "the model recorded as disagreeing with its provenance",
     _m_prov_resolver_disagrees, True),
    ("record_provenance_disagrees", "an external weight digest edited",
     _m_prov_weight_digest_edited, True),
    ("record_provenance_disagrees", "the device the EP ran on swapped for a software rasteriser",
     _m_prov_device_name_swapped, True),
    ("record_provenance_disagrees", "an agreement flag flipped without its pair",
     _m_prov_agreement_flag_flipped, True),
    ("record_provenance_disagrees", "one side of an agreement pair edited",
     _m_prov_agreement_side_edited, True),
    ("record_provenance_disagrees", "the agreement verdict reworded into a hedge",
     _m_prov_agreement_verdict_reworded, True),
    ("verdict_band_unscoped", "a verdict with no band named", _m_band_scope_removed, True),
    ("verdict_band_unscoped", "a verdict declared true of every band",
     _m_band_scope_declared_universal, True),
    ("verdict_band_unscoped", "a verdict scoped to a band the record did not commit",
     _m_band_scope_edges_edited, True),
    ("verdict_band_unscoped",
     "a subject reported indeterminate under a band that classifies it FASTER",
     _m_alternative_band_misreported, True),
    ("verdict_band_unscoped", "the alternative-band readings removed altogether",
     _m_alternative_bands_removed, True),
    ("decode_observations_unreconciled", "the reconciliation removed", _m_reconciliation_removed,
     True),
    ("decode_observations_unreconciled", "an observation left out of the reconciliation",
     _m_reconciliation_drops_an_observation, True),
    ("decode_observations_unreconciled", "the 0.859x estimate dropped from the reconciled text",
     _m_reconciliation_drops_a_point_estimate, True),
    ("decode_observations_unreconciled", "the reconciliation concluding a decode win",
     _m_reconciliation_concludes_a_win, True),
    ("decode_observations_unreconciled", "the reconciliation claiming to have arbitrated",
     _m_reconciliation_arbitrates, True),
    ("proof_ledger_reachability_understated", "the compiled ledger called diagnostic-only",
     _m_ledger_called_diagnostic, True),
    ("proof_ledger_reachability_understated", "the pipeline-audit consumer dropped",
     _m_ledger_consumer_dropped, True),
    ("proof_ledger_reachability_understated", "a consumer named with no symbol to check",
     _m_ledger_consumer_unsourced, True),
    ("refused_row_leaks_results", "a refused row still carrying its timing and ratio",
     _m_refused_row_keeps_its_numbers, True),
    ("refused_row_leaks_results", "a refused row still carrying a success-shaped verdict",
     _m_refused_row_keeps_a_win, True),
    ("discarded_runs_undisclosed", "the list of re-run attempts removed altogether",
     _m_discarded_list_removed, True),
    ("discarded_runs_undisclosed", "the grounds for re-running an attempt left unstated",
     _m_discard_rule_removed, True),
    ("discarded_runs_undisclosed", "a discard rule that selects on the timing it produced",
     _m_discard_rule_is_a_timing_rule, True),
    ("discarded_runs_undisclosed", "a discarded attempt whose samples are withheld",
     _m_discarded_entry_hides_its_samples, True),
    ("discarded_runs_undisclosed", "a discarded attempt that does not say which case it was",
     _m_discarded_entry_unattributed, True),
    ("record_provenance_disagrees", "a baseline timing whose session reported only the CPU EP",
     _m_run_reports_only_cpu, True),
    ("record_provenance_incomplete", "a timing that reports no providers at all",
     _m_run_reports_no_providers, True),
)

#: Arms that must be run against the *bytes* rather than the parsed record.
#:
#: A parse normalises. That is exactly why these exist: a CRLF translation of a frozen artifact
#: produces a record that parses identically and hashes identically at the content level, and
#: only a check taken over the exact bytes can see it. Each of these writes a copy of the real
#: artifact and its seal into a scratch directory, transforms the copy, and requires
#: `gate_artifact` to refuse.


def _b_unsealed(art: Path) -> None:
    pe.seal_path_for(art).unlink()


def _b_crlf(art: Path) -> None:
    art.write_bytes(art.read_bytes().replace(b"\n", b"\r\n"))


def _b_trailing_newline_stripped(art: Path) -> None:
    art.write_bytes(art.read_bytes().rstrip(b"\n"))


def _b_one_byte_flipped(art: Path) -> None:
    raw = bytearray(art.read_bytes())
    for i, ch in enumerate(raw):
        if chr(ch).isdigit() and i > 100:
            raw[i] = ord("7") if chr(ch) != "7" else ord("3")
            break
    art.write_bytes(bytes(raw))


def _b_seal_says_nothing(art: Path) -> None:
    pe.seal_path_for(art).write_text('{"artifact": "x"}\n', encoding="utf-8")


BYTE_ARMS = (
    ("frozen_bytes_unsealed", "the byte seal deleted, leaving nothing binding the bytes",
     _b_unsealed),
    ("frozen_bytes_unsealed", "a seal that declares no digest and no length",
     _b_seal_says_nothing),
    ("frozen_bytes_length_mismatch",
     "the artifact translated to CRLF, which parses identically and is a different file",
     _b_crlf),
    ("frozen_bytes_length_mismatch", "the trailing newline stripped", _b_trailing_newline_stripped),
    ("frozen_bytes_mismatch", "one digit changed in place, leaving the length untouched",
     _b_one_byte_flipped),
)


def _byte_arm(condition: str, name: str, transform) -> None:
    scratch = REPO_ROOT / "bench" / "scratch" / "phi-evidence-negative-control"
    scratch.mkdir(parents=True, exist_ok=True)
    copy_path = scratch / ARTIFACT.name
    try:
        copy_path.write_bytes(ARTIFACT.read_bytes())
        pe.seal_path_for(copy_path).write_bytes(pe.seal_path_for(ARTIFACT).read_bytes())
        healthy = pe.gate_artifact(copy_path)
        if healthy["verdict"] != pe.PASS:
            record("PLANTED", condition, name, False,
                   f"the untransformed copy was already {healthy['verdict']}"
                   f"({healthy.get('condition')}): the arm proves nothing")
            return
        transform(copy_path)
        verdict = pe.gate_artifact(copy_path)
        ok = verdict["verdict"] == pe.FAIL and verdict.get("condition") == condition
        record("PLANTED", condition, name, ok,
               f"gate_artifact said {verdict['verdict']}({verdict.get('condition')}): "
               f"{str(verdict.get('detail'))[:160]}")
    except Exception as exc:  # pragma: no cover - a broken arm is an outage
        record("PLANTED", condition, name, False, f"arm could not be built: {exc!r}")
    finally:
        copy_path.unlink(missing_ok=True)
        pe.seal_path_for(copy_path).unlink(missing_ok=True)


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

    print("\nplanted defects in the bytes themselves — a parse would normalise these away")
    for condition, name, transform in BYTE_ARMS:
        _byte_arm(condition, name, transform)

    print("\nwhat a refusal is allowed to say")
    refused = copy.deepcopy(base)
    refused["claim_limits"]["closes_issue_69"] = True
    _reseal(refused)
    refused_verdict = pe.evidence_gate(refused)
    published = pe.admissible_output(refused, refused_verdict)
    leaked = []

    def _walk(node, path="$"):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            leaked.append(f"{path}={node!r}")

    _walk(published)
    record("OBSERVED", "-",
           "a refused record publishes no timing, ratio, band, floor or separation",
           refused_verdict["verdict"] == pe.FAIL and not leaked,
           f"gate={refused_verdict['verdict']} leaked={leaked[:6]}")
    record("OBSERVED", "-",
           "a refused row keeps its subject and the fact of refusal, and nothing else",
           all(set(row) == {"subject", "status", "admissible", "withheld"}
               for row in published.get("subjects") or []),
           json.dumps(published.get("subjects", [])[:1])[:200])

    print("\nwhat an admissible record must still carry")
    admissible = pe.admissible_output(copy.deepcopy(base), healthy)
    carried = {round(float(o["point_estimate"]), 4)
               for o in (admissible.get("decode_observations") or [])
               if o.get("point_estimate") is not None}
    record("OBSERVED", "-",
           "both independent decode observations survive into the published output",
           {0.859, 0.9651} <= carried, f"carried {sorted(carried)}")
    record("OBSERVED", "-",
           "the published output carries the reconciliation and concludes INCONCLUSIVE",
           (admissible.get("decode_reconciliation") or {}).get("conclusion") == pe.INCONCLUSIVE,
           json.dumps(admissible.get("decode_reconciliation", {}))[:200])

    print("\nwhich layer does what, exercised rather than asserted")
    tampered = copy.deepcopy(base)
    tampered["decode_observations"][0]["superseded_by"] = tampered["decode_observations"][1]["id"]
    _reseal(tampered)
    scratch = REPO_ROOT / "bench" / "scratch" / "phi-evidence-negative-control"
    scratch.mkdir(parents=True, exist_ok=True)
    probe = scratch / "loader-probe.json"
    probe.write_bytes((json.dumps(tampered, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    reloaded = pe.load_frozen(probe)
    record("OBSERVED", "-",
           "load_frozen strips nothing: a superseded block comes back exactly as written",
           reloaded["decode_observations"][0].get("superseded_by")
           == tampered["decode_observations"][1]["id"],
           "the loader returned the block it was given")
    record("OBSERVED", "-",
           "the layer that refuses a supersession is the gate, not the loader",
           pe.evidence_gate(reloaded).get("condition") == "decode_observation_dropped",
           json.dumps(pe.evidence_gate(reloaded))[:200])
    probe.unlink(missing_ok=True)

    print("\ncoverage of the gate's own condition list")
    attacked = {c for c, *_ in ARMS} | {c for c, *_ in BYTE_ARMS}
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
