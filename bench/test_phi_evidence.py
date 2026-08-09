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
