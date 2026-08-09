#!/usr/bin/env python3
"""Two-polarity, behavioural tests for `bench/phi_evidence.py`.

Every test here drives the real function and observes what it *does*. None of them reads the
module's source text, and none of them asserts that a docstring contains a word. That
distinction is the point: the defect this instrument exists to prevent is documentation that
describes behaviour the code does not have, and a test that greps prose cannot tell the two
apart — it would agree with the prose in exactly the case where the prose is wrong.

Host-free by construction: no GPU, no Vulkan, no ONNX Runtime, no 2.3 GB weight file. Records
are synthesised in-process, except for the handful of tests that deliberately read the real
committed artifact.

    python -m pytest bench/test_phi_evidence.py -q
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent
REPO_ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))

import phi_evidence as pe  # noqa: E402
from _polarity import convicts  # noqa: E402

ARTIFACT = BENCH / "results" / "phi35_evidence_v4.json"


@pytest.fixture(scope="module")
def real() -> dict:
    """The committed artifact. Read once; every test that mutates it takes a deep copy."""
    return pe.load_frozen(ARTIFACT)


def _reseal(record: dict) -> dict:
    record.setdefault("identity", {}).pop("content_sha256", None)
    record["identity"]["content_sha256"] = pe.content_digest(record)
    return record


def _mutate(real: dict, fn, *, reseal: bool = True) -> dict:
    out = copy.deepcopy(real)
    fn(out)
    return _reseal(out) if reseal else out


# ============================================================================================
# load_frozen: what it does, proved by doing it
# ============================================================================================


def test_load_frozen_does_not_validate_the_digest(tmp_path):
    """The contract, exercised rather than described.

    `load_frozen` reads bytes and parses JSON. It is *not* a validator, and this test proves
    that by handing it a record whose content no longer matches its own seal and requiring it
    to return that record without complaint. The complaint is `verify_frozen_identity`'s job,
    and the second half of this test requires that one to fire on the same bytes.

    Both halves are needed. The first alone would be satisfied by a function that validates
    nothing anywhere; the second alone would be satisfied by a loader that validates and a
    verifier that is never reached.
    """
    path = tmp_path / "a.json"
    pe.freeze({"schema": pe.SCHEMA, "headline": {"models": [pe.HEADLINE_MODEL]}}, path)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["headline"]["models"] = ["something-else-entirely"]
    path.write_text(json.dumps(tampered), encoding="utf-8")

    loaded = pe.load_frozen(path)
    assert loaded["headline"]["models"] == ["something-else-entirely"], (
        "load_frozen refused or altered a record whose bytes were edited after freezing; its "
        "documented contract is that it does neither"
    )

    identity = pe.verify_frozen_identity(loaded)
    assert identity["verdict"] == "DIVERGENT", identity
    assert identity["recorded"] != identity["recomputed"]


def test_load_frozen_does_not_validate_the_schema(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"schema": "not-a-schema-anyone-knows"}), encoding="utf-8")
    assert pe.load_frozen(path)["schema"] == "not-a-schema-anyone-knows"
    verdict = pe.evidence_gate(pe.load_frozen(path))
    assert verdict["verdict"] == pe.FAIL and verdict["condition"] == "schema_unknown"


def test_load_frozen_distinguishes_absent_from_unparseable(tmp_path):
    """Two different states, two different exceptions, neither of them a verdict."""
    with pytest.raises(pe.FrozenArtifactMissing):
        pe.load_frozen(tmp_path / "nothing-here.json")

    bad = tmp_path / "c.json"
    bad.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(pe.FrozenArtifactUnreadable):
        pe.load_frozen(bad)


def test_verify_frozen_identity_reports_unmeasured_rather_than_guessing():
    """A record that was never frozen is its own state, not a mismatch and not a match."""
    out = pe.verify_frozen_identity({"schema": pe.SCHEMA})
    assert out["verdict"] == "UNMEASURED"
    assert out["recorded"] is None


def test_freeze_is_stable_under_key_order_and_whitespace(tmp_path):
    a = {"schema": pe.SCHEMA, "z": 1, "a": {"n": [1, 2], "m": "x"}}
    b = {"a": {"m": "x", "n": [1, 2]}, "schema": pe.SCHEMA, "z": 1}
    assert pe.content_digest(a) == pe.content_digest(b)

    p = tmp_path / "d.json"
    digest = pe.freeze(dict(a), p)
    reloaded = pe.load_frozen(p)
    assert pe.verify_frozen_identity(reloaded)["verdict"] == pe.MATCH
    assert reloaded["identity"]["content_sha256"] == digest


def test_freeze_seals_a_failing_record_just_as_readily(tmp_path):
    """Freezing is a write, not an endorsement.

    If freezing refused inadmissible records, the only artifacts that could ever exist would be
    the flattering ones, and a measurement that came out badly would have nowhere to live. The
    gate is what refuses; freeze records.
    """
    p = tmp_path / "e.json"
    pe.freeze({"schema": pe.SCHEMA, "claim_limits": {"cuda_comparison": "PARITY"}}, p)
    assert pe.verify_frozen_identity(pe.load_frozen(p))["verdict"] == pe.MATCH
    assert pe.evidence_gate(pe.load_frozen(p))["verdict"] == pe.FAIL


# ============================================================================================
# The classifier
# ============================================================================================


def test_the_classifier_is_symmetric_under_reciprocal():
    """Mirror the world and the verdict mirrors with it, over randomised inputs.

    Asymmetry is the classic shape of a benchmark that only knows how to find good news: a
    threshold applied to `ratio > x` with no matching `ratio < 1/x` treats a 20% win as a
    result and a 20% loss as noise. Here the two clauses are one clause reciprocated, and this
    test is the live proof over 400 random series rather than a claim about the source.
    """
    rng = random.Random(0x69C0DE)
    band = {"lo": 0.94, "hi": 1.06, "source": "calibration"}
    mirrored_band = pe.mirror_band(band)
    seen = {pe.IMPROVEMENT: 0, pe.REGRESSION: 0, pe.INDETERMINATE: 0}
    for _ in range(400):
        series = {"ratios": [round(rng.uniform(0.5, 2.0), 6) for _ in range(3)]}
        forward = pe.classify_ratio(series, band)
        mirrored = pe.classify_ratio(pe.mirror_series(series), mirrored_band)
        expected = {pe.IMPROVEMENT: pe.REGRESSION,
                    pe.REGRESSION: pe.IMPROVEMENT,
                    pe.INDETERMINATE: pe.INDETERMINATE}[forward["verdict"]]
        assert mirrored["verdict"] == expected, (series, forward, mirrored)
        seen[forward["verdict"]] += 1
    assert all(seen.values()), (
        f"the sweep never produced one of the three verdicts ({seen}); a symmetry test that "
        f"only ever sees one outcome has not tested symmetry"
    )


def test_one_lucky_repeat_does_not_make_an_improvement():
    """Every repeat must clear the band, not the median of them."""
    band = {"lo": 0.95, "hi": 1.05, "source": "calibration"}
    lucky = pe.classify_ratio({"ratios": [1.60, 1.60, 1.01]}, band)
    assert lucky["verdict"] == pe.INDETERMINATE, lucky
    assert lucky["point_estimate"] > 1.05, "the median really is above the band; that is the point"

    honest = pe.classify_ratio({"ratios": [1.20, 1.18, 1.22]}, band)
    assert honest["verdict"] == pe.IMPROVEMENT, honest


def test_a_classifier_with_no_band_refuses_rather_than_defaults():
    out = pe.classify_ratio({"ratios": [3.0, 3.0, 3.0]}, {})
    assert out["verdict"] == pe.INDETERMINATE, (
        "with no calibration band a 3x reading was classified anyway; an uncalibrated ratio is "
        "not a result, however large"
    )


def test_within_arm_dispersion_declares_itself_diagnostic():
    out = pe.within_arm_dispersion([10.0, 10.5, 9.8, 30.0])
    assert out["role"] == "diagnostic"
    assert out["n"] == 4 and out["max_ms"] == 30.0


# ============================================================================================
# The gate, one rejection blocker at a time, against the real artifact
# ============================================================================================


def test_the_committed_artifact_is_admissible(real):
    verdict = pe.evidence_gate(copy.deepcopy(real))
    assert verdict["verdict"] == pe.PASS, verdict


def test_the_gate_errors_rather_than_raises_on_a_non_record():
    out = pe.evidence_gate(["not", "a", "mapping"])
    assert out["verdict"] == pe.ERROR, out
    assert out["condition"] == "record_not_a_mapping"


def test_a_hardware_reading_may_not_be_filed_as_lavapipe(real):
    """Rejection blocker 2, in both directions.

    The rejected revision labelled an RTX A1000 measurement as lavapipe. A gate that only
    caught that direction would still let a lavapipe measurement be published as hardware,
    which is the same error pointed the other way and the more flattering of the two.
    """
    def as_lavapipe(r):
        r["environment"]["device"]["driver_name"] = "lavapipe"
    out = pe.evidence_gate(_mutate(real, as_lavapipe))
    assert out["condition"] == "vulkan_implementation_mislabelled", out

    def software_dressed_as_hardware(r):
        r["environment"]["device"]["name"] = "llvmpipe (LLVM 17, 256 bits)"
        r["environment"]["device"]["driver_name"] = "llvmpipe"
        r["environment"]["device"]["device_type"] = "cpu"
    out = pe.evidence_gate(_mutate(real, software_dressed_as_hardware))
    assert out["condition"] == "vulkan_implementation_mislabelled", out

    def discrete_gpu_called_software(r):
        r["environment"]["device"]["implementation_type"] = "software"
    out = pe.evidence_gate(_mutate(real, discrete_gpu_called_software))
    assert out["condition"] == "vulkan_implementation_mislabelled", out


def test_the_real_artifact_records_a_hardware_adapter_with_full_identity(real):
    device = real["environment"]["device"]
    assert device["implementation_type"] == "hardware"
    assert device["device_type"] == "discrete-gpu"
    for field in ("name", "uuid", "luid", "pci", "driver_name", "driver_version",
                  "vulkan_api_version"):
        assert device.get(field), f"{field} is absent from the recorded adapter identity"
    haystack = f"{device['name']} {device['driver_name']}".lower()
    assert not any(s in haystack for s in pe._SOFTWARE_DRIVERS), device


def test_locking_language_may_not_claim_the_gpu(real):
    for text in (
        "the harness takes exclusive GPU ownership while a point is timed",
        "no other process can use the device during a measurement",
        "the runner is the sole owner of the GPU",
    ):
        out = pe.evidence_gate(_mutate(real, lambda r, t=text: r["isolation"].__setitem__("clocks", t)))
        assert out["condition"] == "isolation_overclaimed", (text, out)

    assert real["isolation"]["mode"] == "cooperative-harness-exclusion"
    assert "cannot exclude a process that does not cooperate" in \
        real["isolation"]["what_it_cannot_do"]


def test_calibration_subjects_must_be_disjoint_from_verdict_subjects(real):
    assert not (set(real["calibration"]["subjects"])
                & {v["subject"] for v in real["verdicts"]})

    def overlap(r):
        r["calibration"]["subjects"].append(r["verdicts"][-1]["subject"])
    assert pe.evidence_gate(_mutate(real, overlap))["condition"] == "calibration_not_disjoint"

    def emptied(r):
        r["calibration"]["subjects"] = []
    assert pe.evidence_gate(_mutate(real, emptied))["condition"] == "calibration_not_disjoint"


def test_subjects_are_disjoint_by_input_and_not_merely_by_label(real):
    """A different name is not a different subject.

    Found live: the first sweep taken for this artifact used `decode/M1/past0` alongside
    `prefill/M1/past0`. Both build `input_ids` of length 1 with an empty KV cache, so they feed
    byte-identical tensors, and the calibration band counted one measurement twice under two
    names. The shipped case set uses `decode/M1/past16` instead, and the gate now refuses the
    shape that hid it.
    """
    digests = real["environment"]["feeds_digest_by_subject"]
    subjects = set(real["calibration"]["subjects"]) | {v["subject"] for v in real["verdicts"]}
    assert subjects <= set(digests), "a subject was measured with no recorded input digest"
    assert len({digests[s] for s in subjects}) == len(subjects), (
        "two subjects feed byte-identical inputs: " +
        repr(sorted((s, digests[s][:12]) for s in subjects))
    )

    def collapse_within_calibration(r):
        d = r["environment"]["feeds_digest_by_subject"]
        cal = r["calibration"]["subjects"]
        d[cal[1]] = d[cal[0]]
    out = pe.evidence_gate(_mutate(real, collapse_within_calibration))
    assert out["condition"] == "calibration_not_disjoint", out
    assert "byte-identical" in out["detail"]

    def borrow_a_verdict_input(r):
        d = r["environment"]["feeds_digest_by_subject"]
        d[r["calibration"]["subjects"][0]] = d[r["verdicts"][0]["subject"]]
    out = pe.evidence_gate(_mutate(real, borrow_a_verdict_input))
    assert out["condition"] == "calibration_not_disjoint", out

    def digest_missing(r):
        r["environment"]["feeds_digest_by_subject"].pop(r["calibration"]["subjects"][0])
    out = pe.evidence_gate(_mutate(real, digest_missing))
    assert out["condition"] == "calibration_not_disjoint", out


def test_the_band_may_not_be_derived_from_the_data_it_judges(real):
    band = real["calibration"]["band"]
    assert band["source"] == "calibration"
    assert set(band["derived_from"]) <= set(real["calibration"]["subjects"])

    def circular(r):
        r["calibration"]["band"]["derived_from"] = [v["subject"] for v in r["verdicts"]]
    assert pe.evidence_gate(_mutate(real, circular))["condition"] == "band_self_derived"

    def unsourced(r):
        r["calibration"]["band"]["source"] = "assumed"
    assert pe.evidence_gate(_mutate(real, unsourced))["condition"] == "band_self_derived"


def test_every_model_output_is_compared_not_just_output_zero(real):
    for case in real["equivalence"]:
        assert case["outputs_total"] == pe.PHI35_OUTPUT_COUNT
        assert case["outputs_compared"] == pe.PHI35_OUTPUT_COUNT
        compared = [a for a in case["arms"] if not a.get("self")]
        assert compared, case["subject"]
        for arm in compared:
            assert len(arm["per_output"]) == pe.PHI35_OUTPUT_COUNT
            assert {row["kind"] for row in arm["per_output"]} == {"logits", "kv_block"}
            assert all(row["verdict"] == pe.MATCH for row in arm["per_output"])

    def logits_only(r):
        case = r["equivalence"][0]
        case["outputs_compared"] = 1
        for arm in case["arms"]:
            if not arm.get("self"):
                arm["per_output"] = arm["per_output"][:1]
    assert pe.evidence_gate(_mutate(real, logits_only))["condition"] == "equivalence_incomplete"


def test_a_reference_compared_only_against_itself_is_not_evidence(real):
    def self_only(r):
        case = r["equivalence"][0]
        case["arms"] = [a for a in case["arms"] if a.get("self")]
    assert pe.evidence_gate(_mutate(real, self_only))["condition"] == "equivalence_incomplete"


def test_the_production_path_must_be_witnessed_by_the_runtime_not_the_ep(real):
    for case in real["equivalence"]:
        witness = case["production_witness"]
        assert witness["source"] == "onnxruntime-profile"
        assert witness["vulkan_node_executions"] > 0
        assert witness["provider_requested_only"] is False

    def requested_but_idle(r):
        w = r["equivalence"][0]["production_witness"]
        w["vulkan_node_executions"] = 0
        w["provider_requested_only"] = True
    assert pe.evidence_gate(_mutate(real, requested_but_idle))["condition"] == \
        "production_path_unwitnessed"

    def self_reported(r):
        r["equivalence"][0]["production_witness"]["source"] = "the-ep-said-so"
    assert pe.evidence_gate(_mutate(real, self_reported))["condition"] == "production_path_unwitnessed"


def test_the_headline_stays_on_one_model(real):
    assert real["headline"]["models"] == [pe.HEADLINE_MODEL]
    assert real["headline"]["generalises"] is False

    def widened(r):
        r["headline"]["models"] = [pe.HEADLINE_MODEL, "bert-squad-12"]
    assert pe.evidence_gate(_mutate(real, widened))["condition"] == "headline_scope_widened"

    def generalised(r):
        r["headline"]["generalises"] = True
    assert pe.evidence_gate(_mutate(real, generalised))["condition"] == "headline_scope_widened"


def test_no_cuda_claim_and_no_closure_of_issue_69(real):
    assert real["claim_limits"]["cuda_comparison"] == "NONE"
    assert real["claim_limits"]["closes_issue_69"] is False

    for field, value in (("cuda_comparison", "FASTER"),
                         ("cuda_comparison", "PARITY"),
                         ("closes_issue_69", True),
                         ("decode_conclusion", pe.IMPROVEMENT)):
        out = pe.evidence_gate(_mutate(real, lambda r, f=field, v=value:
                              r["claim_limits"].__setitem__(f, v)))
        assert out["condition"] == "claim_limit_violated", (field, value, out)


def test_both_independent_decode_observations_survive(real):
    """Rejection blocker 1: two disagreeing observations, neither resolving the other."""
    carried = {round(o["point_estimate"], 4): o
               for o in real["decode_observations"] if o.get("independent")}
    assert 0.859 in carried
    assert 0.9651 in carried
    assert carried[0.9651]["interval"] == {"level": 0.95, "lo": 0.820, "hi": 1.136}
    assert carried[0.9651]["power"] == 0.346
    assert real["claim_limits"]["decode_conclusion"] == pe.INCONCLUSIVE
    for obs in real["decode_observations"]:
        assert not obs.get("supersedes") and not obs.get("superseded_by")
        assert obs["verdict"] != pe.IMPROVEMENT

    for target in (0.859, 0.9651):
        def drop(r, t=target):
            r["decode_observations"] = [o for o in r["decode_observations"]
                                        if round(o.get("point_estimate") or 0, 4) != t]
        assert pe.evidence_gate(_mutate(real, drop))["condition"] == "decode_observation_dropped", target


def test_a_third_observation_cannot_stand_in_for_a_deleted_one(real):
    """The count is not the requirement; the two named observations are.

    This is the hole a plain "at least two" check leaves: delete one of the two originals, and
    the branch's own fresher measurement silently takes its place, leaving a green gate and a
    record that no longer disagrees with itself.
    """
    def swap(r):
        r["decode_observations"] = [
            o for o in r["decode_observations"] if round(o.get("point_estimate") or 0, 4) != 0.859
        ] + [{"id": "invented", "independent": True, "point_estimate": 1.4,
              "interval": None, "power": None, "verdict": pe.INCONCLUSIVE}]
    out = pe.evidence_gate(_mutate(real, swap))
    assert out["condition"] == "decode_observation_dropped", out
    assert "0.859" in out["detail"]


def test_a_supersession_claim_between_the_observations_is_refused(real):
    def supersede(r):
        r["decode_observations"][1]["supersedes"] = r["decode_observations"][0]["id"]
    assert pe.evidence_gate(_mutate(real, supersede))["condition"] == "decode_observation_dropped"


def test_dispersion_cannot_change_a_verdict(real):
    """Offline within-arm spread is diagnostic; the gate refuses to let it decide."""
    def wild(r):
        for v in r["verdicts"]:
            v["dispersion"]["rsd"] = 12.5
            v["dispersion"]["stdev_ms"] = 1e9
    widened = _mutate(real, wild)
    assert pe.evidence_gate(widened)["verdict"] == pe.PASS
    assert [v["verdict"] for v in widened["verdicts"]] == \
        [v["verdict"] for v in real["verdicts"]]

    def promoted(r):
        r["verdicts"][0]["basis"] = "within-arm-dispersion"
    assert pe.evidence_gate(_mutate(real, promoted))["condition"] == "dispersion_promoted"

    def relabelled(r):
        r["verdicts"][0]["dispersion"]["role"] = "authoritative"
    assert pe.evidence_gate(_mutate(real, relabelled))["condition"] == "dispersion_promoted"


def test_the_proof_ledger_is_part_of_the_semantic_delta(real):
    ledger = real["proof_ledger"]
    assert ledger["file_sha256"]
    assert ledger["entries_total"] == ledger["self_declared_entry_count"]
    for case in real["equivalence"]:
        enforcement = case["runtime_enforcement"]
        assert enforcement["present"] is True
        assert enforcement["claimed"] > 0
        assert enforcement["claimed_without_ledger_hit"] == 0

    for mutate, why in (
        (lambda r: r["proof_ledger"].pop("file_sha256"), "ledger identity removed"),
        (lambda r: r["equivalence"][0]["runtime_enforcement"].__setitem__("present", False),
         "no claim log for a timed configuration"),
        (lambda r: r["equivalence"][0]["runtime_enforcement"].__setitem__(
            "claimed_without_ledger_hit", 1), "a claimed node with no proof behind it"),
    ):
        assert pe.evidence_gate(_mutate(real, mutate))["condition"] == "proof_ledger_absent", why


def test_the_ledger_digest_in_the_artifact_is_the_ledger_in_the_tree(real):
    """The recorded digest is checkable, so it is checked."""
    ledger = REPO_ROOT / "evidence" / "proof_ledger.jsonl"
    assert pe._sha256_file(ledger) == real["proof_ledger"]["file_sha256"], (
        "the artifact's recorded proof-ledger digest no longer matches evidence/"
        "proof_ledger.jsonl. Either the ledger moved under the evidence, or the evidence is "
        "stale; both need a re-measurement, not an edit."
    )


def test_no_private_path_reaches_the_committed_artifact(real):
    text = ARTIFACT.read_text(encoding="utf-8")
    hit = pe._PRIVATE_PATH_RE.search(text)
    assert hit is None, f"the committed artifact carries a home-directory path: {hit!r}"

    def leak(r):
        r["environment"]["software"]["scratch_dir"] = "/home/someone/bench/scratch"
    assert pe.evidence_gate(_mutate(real, leak))["condition"] == "private_path_disclosed"


def test_redaction_replaces_a_home_path_rather_than_hiding_the_field():
    assert pe._redact(r"C:\Users\alice\repos\x\out.npz").startswith("<home>")
    assert pe._redact("/home/bob/x") == "<home>/x"
    assert pe._redact("bench/results/x.json") == "bench/results/x.json"


def test_a_recorded_verdict_must_be_the_classifiers_own_answer(real):
    band = real["calibration"]["band"]
    for v in real["verdicts"]:
        assert pe.classify_ratio(v["series"], band)["verdict"] == v["verdict"], v["subject"]

    def relabel(r):
        for v in r["verdicts"]:
            if v["verdict"] == pe.INDETERMINATE:
                v["verdict"] = pe.IMPROVEMENT
                return
    assert pe.evidence_gate(_mutate(real, relabel))["condition"] == \
        "verdict_disagrees_with_classifier"


def test_an_edited_artifact_is_caught_even_when_every_claim_still_reads_well(real):
    """Immutable identity: the seal is over content, so content edits are visible."""
    def quiet_edit(r):
        r["verdicts"][0]["point_estimate"] = 9.9
    assert pe.evidence_gate(_mutate(real, quiet_edit, reseal=False))["condition"] == \
        "identity_digest_mismatch"


# ============================================================================================
# Instrument contracts: every public function's refusal, watched rather than described
#
# R13 keeps two findings apart. A FAIL is a statement about the measurement; an
# EvidenceInstrumentError is a statement about the instrument's input — a band whose edges are
# inverted, a latency series containing a negative number, a "record" that is not a mapping.
# Returning INDETERMINATE for those would file an instrument fault as a quiet null result,
# which is exactly how a broken harness reads as an unremarkable one.
#
# These are also the reject polarity the instrument census scores: the gate and the classifier
# through `bench/_polarity.py::convicts` (they refuse by returning a token), everything else
# through `pytest.raises`.
# ============================================================================================


def test_the_gate_convicts_on_the_grounds_it_names(real):
    """Reject polarity for the gate, declared through the enforcing helper.

    `convicts` goes red both when the verdict is not a refusal AND when the condition differs
    from the one named here — so a gate that convicted on every mutation, which is the failure
    mode a bare `verdict == FAIL` check would certify as health, cannot pass through it.
    """
    def lie(record):
        record["environment"]["device"]["driver_name"] = "llvmpipe (LLVM 17.0.6, 256 bits)"

    assert convicts(pe.evidence_gate(_mutate(real, lie)),
                    condition="vulkan_implementation_mislabelled") == \
        "vulkan_implementation_mislabelled"


def test_the_classifier_refuses_rather_than_rounding_towards_a_win():
    """Reject polarity for the classifier: a series straddling the band is INDETERMINATE."""
    band = {"lo": 0.9, "hi": 1.1}
    straddling = {"ratios": [1.5, 1.05, 1.6], "n": 3}
    assert convicts(pe.classify_ratio(straddling, band)) == pe.INDETERMINATE


def test_identity_verification_refuses_an_edited_record(real):
    """Reject polarity for the identity check: DIVERGENT and UNMEASURED are both refusals."""
    edited = copy.deepcopy(real)
    edited["headline"]["note"] = "edited after freezing"
    assert convicts(pe.verify_frozen_identity(edited)) == "DIVERGENT"
    never_frozen = {"identity": {}}
    assert convicts(pe.verify_frozen_identity(never_frozen)) == "UNMEASURED"


def test_a_non_record_is_an_instrument_error_not_a_verdict(tmp_path):
    """One call per instrument, by name: a `parametrize` list references but never calls."""
    record = {"identity": {}, "headline": {"model": "x"}}
    assert pe.canonical_bytes(record).startswith(b"{")
    assert len(pe.content_digest(record)) == 64
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.canonical_bytes(["not", "a", "mapping"])
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.content_digest(["not", "a", "mapping"])
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.verify_frozen_identity(["not", "a", "mapping"])
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.mirror_series(["not", "a", "mapping"])
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.mirror_band(["not", "a", "mapping"])


def test_the_band_takes_a_sequence_of_series_and_says_so():
    """A single series handed in where a list was expected would pool its keys as subjects."""
    band = pe.calibration_band([{"ratios": [1.0, 1.02], "subject": "a"},
                                {"ratios": [0.98], "subject": "b"}])
    assert band["n"] == 3 and band["lo"] == 0.98 and band["hi"] == 1.02
    assert pe.calibration_band([])["n"] == 0
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.calibration_band({"ratios": [1.0], "subject": "a"})
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.calibration_band([["ratios", 1.0]])


def test_freeze_refuses_a_non_record(tmp_path):
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.freeze(["not", "a", "record"], tmp_path / "x.json")


def test_unpaired_repeat_counts_are_an_instrument_error():
    """Pairing the common prefix would report a series shorter than the run that produced it."""
    assert pe.paired_ratio_series([2.0, 2.0], [1.0, 1.0])["n"] == 2
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.paired_ratio_series([2.0, 2.0, 2.0], [1.0, 1.0])


def test_an_inverted_band_is_an_instrument_error_not_a_run_of_null_results():
    """Every subject would read INDETERMINATE, and a broken band would look like a quiet run."""
    series = {"ratios": [1.4, 1.5, 1.6], "n": 3}
    assert pe.classify_ratio(series, {"lo": 0.9, "hi": 1.1})["verdict"] == pe.IMPROVEMENT
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.classify_ratio(series, {"lo": 1.1, "hi": 0.9})
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.mirror_band({"lo": 1.1, "hi": 0.9})


def test_a_non_positive_ratio_cannot_be_mirrored():
    assert pe.mirror_series({"ratios": [2.0]})["ratios"] == [0.5]
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.mirror_series({"ratios": [2.0, 0.0]})


def test_a_non_positive_latency_is_a_timer_fault_not_a_steady_arm():
    """Folded into a dispersion figure it reads as an unusually steady arm, which is worse."""
    assert pe.within_arm_dispersion([10.0, 11.0, 12.0])["n"] == 3
    for bad in ([10.0, 0.0], [10.0, -1.0], [10.0, float("inf")], [10.0, float("nan")]):
        with pytest.raises(pe.EvidenceInstrumentError):
            pe.within_arm_dispersion(bad)


def test_a_subject_label_is_enforced_not_trusted():
    """A malformed label does not produce a wrong verdict; it invents a subject."""
    assert pe.subject_label("prefill", 128, 0).endswith("/prefill/M128/past0")
    for bad in (("warmup", 128, 0), ("prefill", 0, 0), ("prefill", 128, -1), ("prefill", True, 0)):
        with pytest.raises(pe.EvidenceInstrumentError):
            pe.subject_label(*bad)


# ============================================================================================
# The measurement plan itself
# ============================================================================================


def test_verdict_and_calibration_cases_cannot_overlap_by_construction():
    assert not (set(pe.VERDICT_CASES) & set(pe.CALIBRATION_CASES))
    labels_v = {pe.subject_label(*c) for c in pe.VERDICT_CASES}
    labels_c = {pe.subject_label(*c) for c in pe.CALIBRATION_CASES}
    assert not (labels_v & labels_c)


def test_the_gate_condition_list_is_the_gate_and_has_no_orphans():
    assert len(set(pe.GATE_CONDITIONS)) == len(pe.GATE_CONDITIONS)
    source = (BENCH / "phi_evidence.py").read_text(encoding="utf-8")
    for token in pe.GATE_CONDITIONS:
        assert source.count(f'"{token}"') >= 2, (
            f"{token} is declared in GATE_CONDITIONS but never returned by evidence_gate(); a condition "
            f"the gate cannot report is a promise the lane cannot keep"
        )


# ============================================================================================
# Documentation agrees with the artifact, and the artifact is what was measured
# ============================================================================================


def test_docs_perf_quotes_the_artifact_it_cites(real):
    """Every ratio §27 publishes is re-derived here from the frozen record.

    Prose and artifact drifting apart is the specific failure this whole instrument exists to
    prevent, and it is not prevented by a human promising to keep them in step.
    """
    perf = (REPO_ROOT / "docs" / "PERF.md").read_text(encoding="utf-8")
    section = perf[perf.index("## 27."):]

    assert real["identity"]["content_sha256"] in section, (
        "§27 does not name the exact artifact digest it is describing"
    )
    for v in real["verdicts"]:
        point = f"{v['point_estimate']:.3f}x"
        assert point in section, (
            f"{v['subject']} is published in the artifact at {point}, which does not appear in "
            f"docs/PERF.md §27"
        )
    assert "INCONCLUSIVE" in section
    assert "0.859x" in section and "0.9651x" in section
    assert "lavapipe" in section, (
        "§27 no longer distinguishes hardware Vulkan from software lavapipe; that distinction "
        "is one of the rejection blockers this section answers"
    )
    for overclaim in ("Closes #69", "beats CUDA", "faster than CUDA", "CUDA parity"):
        assert overclaim not in section, f"§27 now overclaims: {overclaim!r}"



# ============================================================================================
# The byte seal: exact bytes and exact length, checked before anything parses
# ============================================================================================


def _sealed_copy(tmp_path) -> Path:
    """A byte-for-byte copy of the committed artifact and its seal, in a scratch directory."""
    dst = tmp_path / ARTIFACT.name
    dst.write_bytes(ARTIFACT.read_bytes())
    pe.seal_path_for(dst).write_bytes(pe.seal_path_for(ARTIFACT).read_bytes())
    return dst


def test_the_committed_artifact_matches_its_own_seal():
    verdict = pe.verify_frozen_bytes(ARTIFACT)
    assert verdict["verdict"] == pe.MATCH, verdict
    assert verdict["observed"]["byte_length"] == verdict["declared"]["byte_length"]


def test_a_crlf_translation_is_refused_even_though_it_parses_identically(tmp_path):
    """The whole reason the seal exists, exercised rather than described.

    A line-ending rewrite is the transformation a Windows checkout performs by default. It does
    not change what the record *says*: parse both copies and they are equal, so the content
    digest over the re-serialised record is identical and sees nothing. The bytes are a
    different file, and the seal is taken over the bytes.

    Both halves are asserted here, because the first alone would be satisfied by a gate that
    refuses everything and the second alone by a gate that refuses nothing.
    """
    copy_path = _sealed_copy(tmp_path)
    assert pe.gate_artifact(copy_path)["verdict"] == pe.PASS

    original = copy_path.read_bytes()
    copy_path.write_bytes(original.replace(b"\n", b"\r\n"))
    assert copy_path.read_bytes() != original, "the fixture did not actually translate anything"

    # The content is untouched: this is what a digest over the parsed record would see.
    assert json.loads(copy_path.read_text(encoding="utf-8")) == json.loads(
        original.decode("utf-8")), "the fixture changed the content, so it proves nothing"
    assert pe.verify_frozen_identity(pe.load_frozen(copy_path))["verdict"] == pe.MATCH, (
        "the content digest noticed the line endings; then this test is not testing what it "
        "says it is"
    )

    verdict = pe.gate_artifact(copy_path)
    assert verdict["verdict"] == pe.FAIL
    assert verdict["condition"] == "frozen_bytes_length_mismatch", verdict


def test_a_padded_or_truncated_artifact_is_reported_as_a_length(tmp_path):
    copy_path = _sealed_copy(tmp_path)
    copy_path.write_bytes(copy_path.read_bytes() + b" ")
    verdict = pe.gate_artifact(copy_path)
    assert verdict["condition"] == "frozen_bytes_length_mismatch", verdict


def test_a_same_length_edit_is_reported_as_a_digest(tmp_path):
    copy_path = _sealed_copy(tmp_path)
    raw = bytearray(copy_path.read_bytes())
    for i, ch in enumerate(raw):
        if i > 200 and chr(ch).isdigit():
            raw[i] = ord("7") if chr(ch) != "7" else ord("3")
            break
    copy_path.write_bytes(bytes(raw))
    verdict = pe.gate_artifact(copy_path)
    assert len(bytes(raw)) == pe.seal_bytes(ARTIFACT)["byte_length"]
    assert verdict["condition"] == "frozen_bytes_mismatch", verdict


def test_an_unsealed_artifact_is_refused_rather_than_trusted(tmp_path):
    copy_path = _sealed_copy(tmp_path)
    pe.seal_path_for(copy_path).unlink()
    verdict = pe.gate_artifact(copy_path)
    assert verdict["condition"] == "frozen_bytes_unsealed", verdict


def test_the_bytes_are_checked_before_the_parse(tmp_path):
    """A file that is neither the sealed bytes nor JSON reports the byte problem, not the parse.

    Ordering is the claim. If the parse ran first this would come back as an unreadable
    artifact, which is an instrument outage, and an outage is not a detection.
    """
    copy_path = _sealed_copy(tmp_path)
    copy_path.write_bytes(b"{ not json at all")
    verdict = pe.gate_artifact(copy_path)
    assert verdict["verdict"] == pe.FAIL
    assert verdict["condition"] == "frozen_bytes_length_mismatch", verdict


def test_seal_bytes_normalises_nothing(tmp_path):
    lf = tmp_path / "lf.json"
    lf.write_bytes(b'{"a": 1}\n')
    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(b'{"a": 1}\r\n')
    assert pe.seal_bytes(lf)["sha256_of_exact_bytes"] != pe.seal_bytes(crlf)[
        "sha256_of_exact_bytes"]
    assert pe.seal_bytes(lf)["byte_length"] != pe.seal_bytes(crlf)["byte_length"]


def test_an_absent_artifact_is_an_outage_not_a_finding(tmp_path):
    verdict = pe.gate_artifact(tmp_path / "nothing.json")
    assert verdict["verdict"] == pe.ERROR and verdict["condition"] == "artifact_absent"


# ============================================================================================
# The loader contract: what each layer does, proved by making each layer do it
# ============================================================================================


def test_load_frozen_does_not_strip_a_superseded_block_and_the_gate_is_what_refuses_it(tmp_path):
    """Three separate claims, because the reviewer's objection was about which layer acts.

    1. `load_frozen` returns a record carrying a supersession claim, unaltered: it strips
       nothing. A loader that quietly removed the block would return a record that passes, and
       nobody would learn that the block had been written.
    2. The block survives the round trip byte-identically.
    3. The gate is the layer that refuses it.
    """
    path = tmp_path / "superseded.json"
    record = copy.deepcopy(pe.load_frozen(ARTIFACT))
    record["decode_observations"][0]["superseded_by"] = \
        record["decode_observations"][1]["id"]
    pe.freeze(record, path)

    loaded = pe.load_frozen(path)
    assert "superseded_by" in loaded["decode_observations"][0], (
        "load_frozen stripped a superseded block. It does not strip; the gate refuses. "
        "Documenting one layer's behaviour as another's is the defect this test exists for."
    )
    assert loaded["decode_observations"][0]["superseded_by"] == \
        record["decode_observations"][1]["id"]

    verdict = pe.evidence_gate(loaded)
    assert verdict["verdict"] == pe.FAIL
    assert verdict["condition"] == "decode_observation_dropped", verdict


def test_the_loader_contract_is_recorded_and_the_gate_recomputes_it(real):
    assert real["identity"]["loader_contract"] == pe.LOADER_CONTRACT
    edited = _mutate(real, lambda r: r["identity"]["loader_contract"]["load_frozen"].update(
        {"validates_content_digest": True}))
    verdict = pe.evidence_gate(edited)
    assert verdict["condition"] == "loader_contract_misdescribed", verdict


def test_a_contract_claiming_the_loader_strips_is_refused(real):
    edited = _mutate(real, lambda r: r["identity"]["loader_contract"]["load_frozen"].update(
        {"strips_superseded_blocks": True}))
    assert pe.evidence_gate(edited)["condition"] == "loader_contract_misdescribed"


def test_every_layer_named_in_the_contract_exists_and_is_callable():
    for name in pe.LOADER_CONTRACT:
        assert callable(getattr(pe, name, None)), (
            f"the loader contract describes {name}, which is not a function in this module; a "
            f"contract about a layer that does not exist cannot be checked by anybody"
        )


# ============================================================================================
# Per-record provenance: every named field fails closed, deleted or mutated
# ============================================================================================


def _first_run(record: dict) -> dict:
    return record["raw"]["runs"][0]


@pytest.mark.parametrize("field", pe.REQUIRED_RECORD_PROVENANCE)
def test_deleting_any_required_provenance_field_makes_the_gate_refuse(real, field):
    """One case per field the reviewer named. Deletion is fatal, individually, for each."""
    edited = _mutate(real, lambda r: _first_run(r)["provenance"].pop(field, None))
    verdict = pe.evidence_gate(edited)
    assert verdict["verdict"] == pe.FAIL, (field, verdict)
    assert verdict["condition"] == "record_provenance_incomplete", (field, verdict)


@pytest.mark.parametrize("field", pe.REQUIRED_RECORD_PROVENANCE)
def test_replacing_any_required_provenance_field_with_the_wrong_type_is_fatal(real, field):
    edited = _mutate(real, lambda r: _first_run(r)["provenance"].__setitem__(field, "yes"))
    verdict = pe.evidence_gate(edited)
    assert verdict["verdict"] == pe.FAIL, (field, verdict)
    assert verdict["condition"] in ("record_provenance_incomplete",
                                    "record_provenance_disagrees"), (field, verdict)


def test_a_provenance_block_missing_entirely_is_fatal(real):
    edited = _mutate(real, lambda r: _first_run(r).pop("provenance"))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_incomplete"


def test_the_library_a_record_says_it_timed_must_be_the_one_the_process_had_mapped(real):
    edited = _mutate(real, lambda r: _first_run(r)["provenance"][
        "ep_library_loaded_in_process"].__setitem__("sha256", "0" * 64))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_disagrees"


def test_a_record_whose_library_was_never_mapped_is_fatal(real):
    edited = _mutate(real, lambda r: _first_run(r)["provenance"][
        "ep_library_loaded_in_process"].__setitem__("found", False))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_disagrees"


def test_a_baseline_record_carrying_the_head_library_is_fatal(real):
    head = real["environment"]["software"]["ep_library_sha256"]

    def swap(r):
        for run in r["raw"]["runs"]:
            if run["arm"] == pe.ARM_BASELINE:
                run["provenance"]["ep_library_sha256"] = head
                run["provenance"]["ep_library_loaded_in_process"]["sha256"] = head
                return
    edited = _mutate(real, swap)
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_disagrees"


def test_a_resolver_that_does_not_agree_may_not_be_published(real):
    edited = _mutate(real, lambda r: _first_run(r)["provenance"]["model_resolver"].__setitem__(
        "agrees_with_recorded_provenance", False))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_disagrees"


def test_an_agreement_flag_that_is_not_a_boolean_is_not_agreement(real):
    """'Unknown' and 'agrees' are different states, and a benchmark may not publish under the
    first while looking like the second. A truthy string is the exact shape of that mistake."""
    edited = _mutate(real, lambda r: _first_run(r)["provenance"]["model_resolver"].__setitem__(
        "agrees_with_recorded_provenance", "true"))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_incomplete"


def test_external_weight_metadata_may_not_be_emptied(real):
    edited = _mutate(real, lambda r: _first_run(r)["provenance"]["external_weights"].__setitem__(
        "files", []))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_incomplete"


def test_external_weight_metadata_may_not_disagree_with_the_artifact(real):
    edited = _mutate(real, lambda r: _first_run(r)["provenance"]["external_weights"]["files"][0]
                     .__setitem__("sha256", "1" * 64))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_disagrees"


def test_a_record_taken_on_another_device_may_not_be_filed_here(real):
    edited = _mutate(real, lambda r: _first_run(r)["provenance"].__setitem__(
        "device_name", "llvmpipe (LLVM 17.0.6, 256 bits)"))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_disagrees"


def test_a_record_that_dispatched_no_shader_did_not_run_on_the_gpu(real):
    edited = _mutate(real, lambda r: _first_run(r)["provenance"]["shaders"].__setitem__(
        "count", 0))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_incomplete"


def test_a_record_that_executed_no_dispatch_timed_something_else(real):
    edited = _mutate(real, lambda r: _first_run(r)["provenance"].__setitem__(
        "dispatches_executed", 0))
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_incomplete"


def test_a_vulkan_arm_whose_session_reported_only_the_cpu_is_refused(real):
    """The defect this instrument actually caught on this desk, kept as a test.

    One baseline repeat came back with the EP unregistered and ran on the CPU provider at less
    than half the median of the repeats either side of it. Nothing in the timing said so; the
    provider list and the dispatch count both did.
    """
    edited = _mutate(real, lambda r: _first_run(r).__setitem__(
        "providers_reported", ["CPUExecutionProvider"]))
    verdict = pe.evidence_gate(edited)
    assert verdict["condition"] == "record_provenance_disagrees", verdict


@pytest.mark.parametrize("field", pe.REQUIRED_AGREEMENT_PAIRS)
def test_deleting_an_agreement_pair_is_fatal(real, field):
    def drop(r):
        agreement = _first_run(r)["provenance"]["provenance_agreement"]
        agreement["pairs"] = [p for p in agreement["pairs"] if p.get("field") != field]
    edited = _mutate(real, drop)
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_incomplete", field


@pytest.mark.parametrize("field", pe.REQUIRED_AGREEMENT_PAIRS)
def test_an_agreement_pair_that_does_not_recompute_is_fatal(real, field):
    """The flag is not trusted: the gate recomputes the pair from its own two sides.

    That is what makes all three mutations fatal — editing a side, flipping the flag, or
    swapping the rule out for one that would make a disagreement read as agreement.
    """
    def edit(r):
        for p in _first_run(r)["provenance"]["provenance_agreement"]["pairs"]:
            if p.get("field") == field:
                p["right"] = "something else entirely"
    edited = _mutate(real, edit)
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_disagrees", field


def test_an_agreement_verdict_may_not_be_asserted_over_a_failing_pair(real):
    def edit(r):
        pairs = _first_run(r)["provenance"]["provenance_agreement"]["pairs"]
        pairs[0]["agree"] = False
    edited = _mutate(real, edit)
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_disagrees"


def test_an_unknown_agreement_rule_cannot_be_recomputed_and_is_fatal(real):
    def edit(r):
        pairs = _first_run(r)["provenance"]["provenance_agreement"]["pairs"]
        pairs[0]["rule"] = "vibes"
    edited = _mutate(real, edit)
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_incomplete"


def test_the_equivalence_arms_carry_the_same_provenance_burden(real):
    """The equivalence pass is where the production-path witness comes from, so its records
    are provenance-bearing too. A gate that screened only the timing rows would leave the
    witness unattributed."""
    def edit(r):
        for case in r["equivalence"]:
            for arm in case["arms"]:
                if arm.get("provenance"):
                    arm["provenance"].pop("device_name")
                    return
        pytest.skip("no equivalence arm carries provenance")
    edited = _mutate(real, edit)
    assert pe.evidence_gate(edited)["condition"] == "record_provenance_incomplete"


def test_check_record_provenance_errors_rather_than_verdicts_on_a_non_record():
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.check_record_provenance("not a run", {})


# ============================================================================================
# What was thrown away, and on what grounds
# ============================================================================================


def test_the_artifact_discloses_every_attempt_it_refused(real):
    raw = real["raw"]
    assert isinstance(raw["discarded_runs"], list), (
        "an artifact that re-ran anything must say so; 'nothing was discarded' is an empty list"
    )
    assert "structural" in raw["discard_rule"].lower()
    for entry in raw["discarded_runs"]:
        assert entry["samples_ms"], "a discarded attempt withholding its samples is not disclosed"
        assert entry["dispatches_executed"] == 0 or \
            "VulkanExecutionProvider" not in (entry["providers_reported"] or []), entry


def test_removing_the_discard_disclosure_makes_the_gate_refuse(real):
    edited = _mutate(real, lambda r: r["raw"].pop("discarded_runs"))
    assert pe.evidence_gate(edited)["condition"] == "discarded_runs_undisclosed"


def test_a_discard_rule_that_selects_on_timing_is_refused(real):
    edited = _mutate(real, lambda r: r["raw"].__setitem__(
        "discard_rule", "attempts slower than the arm median were re-run"))
    verdict = pe.evidence_gate(edited)
    assert verdict["condition"] == "discarded_runs_undisclosed", verdict


def test_the_discard_predicate_never_reads_a_timing():
    """The rule itself, exercised on synthetic records rather than read out of its docstring."""
    healthy = {"providers_reported": ["VulkanExecutionProvider", "CPUExecutionProvider"],
               "provenance": {"dispatches_executed": 2840}, "samples_ms": [9999.0]}
    assert pe._vulkan_actually_ran(healthy) is None, (
        "a run that registered the EP and dispatched was refused for being slow"
    )
    fell_back = dict(healthy, providers_reported=["CPUExecutionProvider"], samples_ms=[1.0])
    assert pe._vulkan_actually_ran(fell_back) is not None
    no_dispatch = {"providers_reported": ["VulkanExecutionProvider"],
                   "provenance": {"dispatches_executed": 0}}
    assert pe._vulkan_actually_ran(no_dispatch) is not None


# ============================================================================================
# A refused row publishes nothing that looks like a result
# ============================================================================================


def test_a_refused_row_keeps_its_name_and_loses_its_numbers():
    row = {"subject": "phi/prefill/M128/past0", "verdict": pe.IMPROVEMENT,
           "point_estimate": 2.077, "floor": 1.9, "series": {"ratios": [2.0, 2.1]},
           "head_median_ms": 1449.6, "baseline_median_ms": 3020.6,
           "dispersion": {"iqr": 3.0}}
    out = pe.sanitise_refused_row(row)
    assert set(out) == {"subject", "status", "admissible", "withheld"}
    assert out["subject"] == row["subject"]
    assert out["status"] == "REFUSED"
    assert "point_estimate" in out["withheld"] and "verdict" in out["withheld"]
    assert "2.077" not in json.dumps(out) and "1449" not in json.dumps(out)


def test_a_refusal_publishes_no_number_anywhere_in_its_payload(real):
    """Every numeric leaf, not merely the ones a summary happens to print."""
    edited = _mutate(real, lambda r: r["claim_limits"].__setitem__("closes_issue_69", True))
    verdict = pe.evidence_gate(edited)
    assert verdict["verdict"] == pe.FAIL

    published = pe.admissible_output(edited, verdict)
    leaves = []

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            leaves.append(node)
    walk(published)
    assert leaves == [], f"a refused record published numbers: {leaves[:8]}"
    assert "decode_observations" not in published
    assert "claim_limits" not in published


def test_a_refusal_carries_the_subject_names_but_no_verdicts(real):
    edited = _mutate(real, lambda r: r["claim_limits"].__setitem__("closes_issue_69", True))
    published = pe.admissible_output(edited)
    assert published["subjects"], "a refusal that names no subject is not a report"
    for row in published["subjects"]:
        assert set(row) == {"subject", "status", "admissible", "withheld"}
        assert row["status"] == "REFUSED"


def test_a_gate_that_refuses_still_says_which_condition_it_refused_on(real):
    edited = _mutate(real, lambda r: r["claim_limits"].__setitem__("closes_issue_69", True))
    published = pe.admissible_output(edited)
    assert published["condition"] == "claim_limit_violated"
    assert published["verdict"] == pe.FAIL


def test_a_passing_record_publishes_its_numbers(real):
    """The other polarity. A sanitiser that withheld everything always would pass the tests
    above and be useless."""
    published = pe.admissible_output(real)
    assert published["verdict"] == pe.PASS
    assert published["subjects"] and published["subjects"][0]["point_estimate"] > 0
    assert len(published["decode_observations"]) >= len(pe.REQUIRED_DECODE_OBSERVATIONS)
    estimates = {round(o["point_estimate"], 4) for o in published["decode_observations"]}
    assert {round(o["point_estimate"], 4) for o in pe.REQUIRED_DECODE_OBSERVATIONS} <= estimates


def test_the_gate_refuses_a_record_whose_own_refused_row_still_shows_a_result(real):
    def edit(r):
        r["verdicts"][0]["status"] = "REFUSED"
    edited = _mutate(real, edit)
    assert pe.evidence_gate(edited)["condition"] == "refused_row_leaks_results"


def test_sanitise_refused_row_errors_on_a_non_row():
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.sanitise_refused_row(["not", "a", "row"])


# ============================================================================================
# Band scope: a verdict is a statement about a subject read against a band
# ============================================================================================


def test_every_verdict_names_the_band_it_was_read_against(real):
    band = real["calibration"]["band"]
    for v in real["verdicts"]:
        scope = v["band_scope"]
        assert scope["lo"] == pytest.approx(band["lo"])
        assert scope["hi"] == pytest.approx(band["hi"])
        assert scope["source"] == "calibration"
        assert scope["band_independent"] is False, (
            "a verdict that declares itself band-independent is claiming more than a single "
            "calibration sitting can support"
        )


def test_a_verdict_stripped_of_its_band_scope_is_refused(real):
    edited = _mutate(real, lambda r: r["verdicts"][0].pop("band_scope"))
    assert pe.evidence_gate(edited)["condition"] == "verdict_band_unscoped"


def test_an_indeterminate_subject_carries_its_reading_under_other_bands(real):
    """M64's classification depends on the band, and the artifact has to say so.

    A 3% band classifies it FASTER. Reporting it as indeterminate full stop would be a claim
    about every band anyone might commit, which this evidence cannot support. The gate
    recomputes each alternative reading from the same series, so the alternatives cannot be
    asserted either.
    """
    by_subject = {v["subject"]: v for v in real["verdicts"]}
    indeterminate = [v for v in real["verdicts"] if v["verdict"] == pe.INDETERMINATE]
    if not indeterminate:
        pytest.skip("no subject came back indeterminate in this sweep")
    for v in indeterminate:
        readings = {a["name"]: a for a in v["alternative_bands"]}
        assert {b["name"] for b in pe.ALTERNATIVE_BANDS} <= set(readings)
        for name, reading in readings.items():
            recomputed = pe.classify_ratio(
                v["series"], {"lo": reading["lo"], "hi": reading["hi"]})
            assert reading["verdict"] == recomputed["verdict"], (v["subject"], name)
    assert by_subject  # the table was not empty


def test_an_alternative_band_reading_that_was_asserted_rather_than_computed_is_refused(real):
    def edit(r):
        for v in r["verdicts"]:
            if v.get("alternative_bands"):
                v["alternative_bands"][0]["verdict"] = pe.INDETERMINATE
                return
        pytest.skip("no alternative band readings recorded")
    edited = _mutate(real, edit)
    verdict = pe.evidence_gate(edited)
    assert verdict["condition"] == "verdict_band_unscoped", verdict


def test_removing_the_alternative_readings_is_refused(real):
    edited = _mutate(real, lambda r: [v.pop("alternative_bands", None) for v in r["verdicts"]])
    assert pe.evidence_gate(edited)["condition"] == "verdict_band_unscoped"


# ============================================================================================
# Both decode observations, reconciled and neither superseding the other
# ============================================================================================


def test_the_reconciliation_names_both_observations_and_concludes_inconclusive(real):
    rec = real["decode_observations_reconciliation"]
    text = json.dumps(rec)
    assert "0.859" in text and "0.9651" in text
    assert "0.820" in text and "1.136" in text and "0.346" in text
    assert rec["conclusion"] == pe.INCONCLUSIVE
    assert rec["arbitrated"] is False, (
        "the reconciliation claims to have arbitrated between the two observations; it has not, "
        "and saying it did would make one of them superseding by implication"
    )
    assert set(rec["observation_ids"]) >= {o["id"] for o in real["decode_observations"]
                                           if o.get("id")}


def test_removing_the_reconciliation_is_refused(real):
    edited = _mutate(real, lambda r: r.pop("decode_observations_reconciliation"))
    assert pe.evidence_gate(edited)["condition"] == "decode_observations_unreconciled"


def test_a_reconciliation_that_concludes_a_win_is_refused(real):
    edited = _mutate(real, lambda r: r["decode_observations_reconciliation"].__setitem__(
        "conclusion", pe.IMPROVEMENT))
    assert pe.evidence_gate(edited)["condition"] == "decode_observations_unreconciled"


def test_a_reconciliation_that_leaves_one_observation_out_is_refused(real):
    def edit(r):
        rec = r["decode_observations_reconciliation"]
        rec["observation_ids"] = rec["observation_ids"][:1]
        rec["point_estimates"] = rec["point_estimates"][:1]
    edited = _mutate(real, edit)
    assert pe.evidence_gate(edited)["condition"] == "decode_observations_unreconciled"


def test_both_observations_reach_the_published_output(real):
    published = pe.admissible_output(real)
    text = json.dumps(published)
    assert "0.859" in text and "0.9651" in text
    assert published["decode_reconciliation"]["conclusion"] == pe.INCONCLUSIVE


# ============================================================================================
# The compiled proof ledger is reachable from production, and the artifact says where
# ============================================================================================


def test_the_artifact_names_every_production_consumer_of_the_ledger(real):
    recorded = real["proof_ledger"]["production_reachability"]
    assert recorded["diagnostic_only"] is False
    roles = {c["role"] for c in recorded["consumers"]}
    assert roles == {c["role"] for c in pe.PROOF_LEDGER_CONSUMERS}


def test_every_named_ledger_consumer_is_a_symbol_that_exists_in_the_tree(real):
    """The rows are checkable, and this checks them. A citation nobody follows is prose."""
    for consumer in real["proof_ledger"]["production_reachability"]["consumers"]:
        source = (REPO_ROOT / consumer["file"]).read_text(encoding="utf-8", errors="replace")
        for symbol in consumer["symbol"].split(" / "):
            bare = symbol.split("::")[-1].strip()
            assert f"fn {bare}" in source, (
                f"{consumer['file']} does not define {bare}, which the artifact cites as a "
                f"production consumer of the proof ledger"
            )


def test_calling_the_ledger_diagnostic_only_is_refused(real):
    edited = _mutate(real, lambda r: r["proof_ledger"]["production_reachability"].__setitem__(
        "diagnostic_only", True))
    assert pe.evidence_gate(edited)["condition"] == "proof_ledger_reachability_understated"


def test_dropping_a_production_consumer_is_refused(real):
    def edit(r):
        reach = r["proof_ledger"]["production_reachability"]
        reach["consumers"] = reach["consumers"][:-1]
    edited = _mutate(real, edit)
    assert pe.evidence_gate(edited)["condition"] == "proof_ledger_reachability_understated"


# ============================================================================================
# Two-polarity screening for the layers added this revision
#
# Every function below is watched refusing and watched accepting, in that order, so that a
# mutant that refused everything and a mutant that certified everything are both red. The
# census reads these; they are written to be read by a person first.
# ============================================================================================


def test_seal_path_for_refuses_an_unnamed_artifact_and_names_a_real_one(tmp_path):
    with pytest.raises(pe.EvidenceInstrumentError):
        pe.seal_path_for("")
    assert pe.seal_path_for(tmp_path / "a.json").name == "a.json.seal.json"


def test_seal_bytes_refuses_an_absent_file_and_measures_a_present_one(tmp_path):
    with pytest.raises(pe.FrozenArtifactMissing):
        pe.seal_bytes(tmp_path / "absent.json")
    present = tmp_path / "p.json"
    present.write_bytes(b"abc")
    assert pe.seal_bytes(present)["byte_length"] == 3


def test_write_seal_refuses_an_absent_artifact_and_seals_a_present_one(tmp_path):
    with pytest.raises(pe.FrozenArtifactMissing):
        pe.write_seal(tmp_path / "absent.json")
    present = tmp_path / "q.json"
    present.write_bytes(b'{"a": 1}\n')
    seal = pe.write_seal(present)
    assert seal["byte_length"] == 9
    assert json.loads(pe.seal_path_for(present).read_text(encoding="utf-8")) == seal


def test_verify_frozen_bytes_convicts_a_translated_copy_and_clears_an_untouched_one(tmp_path):
    copy_path = _sealed_copy(tmp_path)
    assert pe.verify_frozen_bytes(copy_path)["verdict"] == pe.MATCH
    copy_path.write_bytes(copy_path.read_bytes().replace(b"\n", b"\r\n"))
    convicts(pe.verify_frozen_bytes(copy_path),
             condition="frozen_bytes_length_mismatch",
             because="a CRLF copy is a different file and the seal is over the bytes")


def test_gate_artifact_convicts_edited_bytes_and_passes_the_committed_ones(tmp_path):
    assert pe.gate_artifact(ARTIFACT)["verdict"] == pe.PASS
    copy_path = _sealed_copy(tmp_path)
    copy_path.write_bytes(copy_path.read_bytes() + b"\n")
    convicts(pe.gate_artifact(copy_path), condition="frozen_bytes_length_mismatch",
             because="one byte appended to a sealed artifact")


def test_check_record_provenance_convicts_a_stripped_record_and_clears_a_whole_one(real):
    env = real["environment"]
    run = copy.deepcopy(real["raw"]["runs"][0])
    assert pe.check_record_provenance(run, env)["verdict"] == pe.PASS
    run["provenance"].pop("dispatches_executed")
    convicts(pe.check_record_provenance(run, env),
             condition="record_provenance_incomplete",
             because="a timed Vulkan run that executed no dispatch timed something else")


def test_admissible_output_convicts_a_refused_record_and_publishes_a_sound_one(real):
    published = pe.admissible_output(real)
    assert published["verdict"] == pe.PASS and published["subjects"]
    refused = _mutate(real, lambda r: r["claim_limits"].__setitem__("closes_issue_69", True))
    convicts(pe.admissible_output(refused), condition="claim_limit_violated",
             because="an artifact that claims to close issue #69 publishes nothing")
