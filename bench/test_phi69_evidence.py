"""Two-polarity tests for the #69 v5 publication gate (`bench/phi69_evidence.py`).

Every condition in `GATE_CONDITIONS` is exercised in both polarities: the admissible fixture
passes it, and a single targeted mutation makes it fail and forces `publish` onto the refusal
path. The point is the same one `bench/_polarity.py` makes — a screen that only ever sees the
accept polarity has never watched the instrument disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
import phi69_evidence as pe  # noqa: E402
import _polarity as pol  # noqa: E402  (the enforcing value-polarity helpers)


def _rows(verdict: str = "MATCH") -> list:
    return [{"index": i, "verdict": verdict} for i in range(pe.EXPECTED_OUTPUTS)]


# Synthetic placeholder timings authored for these fixtures. They are deliberately
# repeated-digit sentinels (1111.0, 2222.0, 333.0, 444.0) and a synthetic 1.111 ratio: they
# trace to no measurement, match no recorded artifact, and are only ever exercised on the accept
# polarity of a host-free fixture -- the standing posture publishes nothing. Clean-room by
# construction: no value here is copied from any prior revision's material.
_SYNTH_PREFILL_HEAD_MS = 1111.0
_SYNTH_PREFILL_HEAD_RATIO = 1.111
_SYNTH_PREFILL_PRE_MS = 2222.0
_SYNTH_DECODE_HEAD_MS = 333.0
_SYNTH_DECODE_PRE_MS = 444.0


def _prefill_subject(name: str) -> dict:
    return {
        "name": name,
        "pooled": False,
        "arms": [
            {"arm": "head", "aggregate": "MATCH", "per_output": _rows(),
             "median_ms": _SYNTH_PREFILL_HEAD_MS, "point_ratio": _SYNTH_PREFILL_HEAD_RATIO},
            {"arm": "pre72", "aggregate": "MATCH", "per_output": _rows(),
             "median_ms": _SYNTH_PREFILL_PRE_MS},
        ],
    }


def _admissible() -> dict:
    """A synthetic record that satisfies every condition — the accept polarity."""
    text = "line one\nline two\n"
    return {
        "schema": "phi69-evidence/v5",
        "subjects": [
            _prefill_subject("prefill/M128"),
            {
                "name": "decode/M1/past128",
                "pooled": False,
                "timing_verdict": pe.INDETERMINATE,
                "preserved_observations": list(pe.REQUIRED_DECODE_OBSERVATIONS),
                "arms": [
                    {"arm": "head", "aggregate": "MATCH", "per_output": _rows(),
                     "median_ms": _SYNTH_DECODE_HEAD_MS},
                    {"arm": "pre72", "aggregate": "MATCH", "per_output": _rows(),
                     "median_ms": _SYNTH_DECODE_PRE_MS},
                ],
            },
        ],
        "raw_runs": [
            {"arm": "head", "source_commit": "a" * 40, "dll_sha256": "d1",
             "build_recipe_sha256": "r1", "worktree_dirty": False},
            {"arm": "pre72", "source_commit": "b" * 40, "dll_sha256": "d2",
             "build_recipe_sha256": "r2", "worktree_dirty": False},
        ],
        "device": {
            "uuid": f"uuid:{pe.PINNED_DEVICE_UUID}", "luid": "0x00-0x1234",
            "pci_bus_id": "0000:01:00.0", "driver_version": "573.44",
            "device_type": "discrete-gpu", "driver_name": "NVIDIA RTX A1000",
        },
        "model": {
            "graph_sha256": "g" * 64, "weights_sha256": "w" * 64,
            "provenance": {"foundry_variant_id": "Phi-3.5-mini-instruct-generic-gpu"},
        },
        "quiescence": {"verdict": pe.QUIET},
        "device_state": {"verdict": pe.COMPANION_QUOTABLE},
        "feeds_digest_by_subject": {"cal/A": "feed-d1", "prefill/M128": "feed-d2"},
        "calibration_subjects": ["cal/A"],
        "verdict_subjects": ["prefill/M128"],
        "provenance": {
            "base_delta_claim": (
                "the delta to the PR base changed bench tooling, docs and CI; no Rust or "
                "shader source changed"
            ),
        },
        "refusals": [],
        "content_digests": [
            {"name": "proof_ledger", "source_text": text, "recorded": pe._lf_digest(text)},
        ],
        "within_arm_dispersion": {"role": "diagnostic", "moved_a_verdict": False},
        "isolation": {"language": "cooperative harness and process exclusion only; no GPU lock"},
        "headline_scope": {
            "model": "one", "prefill_family": "one", "adapter": "one", "box": "one",
            "cuda_comparison": "NONE", "closes_issue_69": False,
        },
        "uncertainty": {
            "aa_band": [0.90, 1.05],
            "power_boost_qualification": (
                "the A/A band measures observed noise and does not establish quiet, boost or "
                "concurrent-GPU state"
            ),
        },
        "witnesses": {"islands": 33, "dispatches": 40},
    }


# --- accept polarity ------------------------------------------------------- #
def test_admissible_record_publishes_quotable():
    pub = pe.publish(_admissible())
    assert pub["timing_admissible"] is True
    assert pub["timing_verdict"] == "QUOTABLE"
    assert pub["refusals"] == []


def test_all_conditions_pass_on_admissible():
    conditions = pe._evaluate(_admissible())
    failing = {k: v["reason"] for k, v in conditions.items() if not v["ok"]}
    assert failing == {}, failing


# --- reject polarity: one mutation per condition --------------------------- #
# Each mutator breaks exactly one named condition.
def _m_record_wellformed(r): r["schema"] = "phi_evidence/v4"
def _m_immutable_run_binding(r): r["raw_runs"][0]["source_commit"] = "HEAD"
def _m_device_identity_immutable(r): r["device"]["uuid"] = "uuid:" + "0" * 32
def _m_model_identity_provenance(r): r["model"]["provenance"] = {}
def _m_quiescence_quiet(r): r["quiescence"]["verdict"] = pe.CONTENDED
def _m_device_state_companion(r): r["device_state"] = None
def _m_per_output_integrity(r): r["subjects"][0]["arms"][0]["per_output"] = _rows()[:1]
def _m_all_output_equivalence(r): r["subjects"][0]["arms"][0]["per_output"][7]["verdict"] = "DIVERGENT"
def _m_calibration_content_disjoint(r): r["feeds_digest_by_subject"]["cal/A"] = "feed-d2"
def _m_decode_p128_separate(r): r["subjects"][1]["timing_verdict"] = "IMPROVEMENT"
def _m_provenance_claim_accurate(r): r["provenance"]["base_delta_claim"] = "CI-only, touches no bench source"
def _m_refusal_output_sanitized(r): r["refusals"] = [{"detail": r"C:\Users\secret-user\thing"}]
def _m_digests_platform_stable(r): r["content_digests"][0]["recorded"] = "deadbeef"
def _m_no_dispersion_promotion(r): r["within_arm_dispersion"]["moved_a_verdict"] = True
def _m_isolation_language_cooperative(r): r["isolation"]["language"] = "exclusive GPU ownership"
def _m_headline_scope_not_widened(r): r["headline_scope"]["model"] = "all"
def _m_uncertainty_qualified(r): r["uncertainty"]["power_boost_qualification"] = ""


MUTATORS = {
    "record_wellformed": _m_record_wellformed,
    "immutable_run_binding": _m_immutable_run_binding,
    "device_identity_immutable": _m_device_identity_immutable,
    "model_identity_provenance": _m_model_identity_provenance,
    "quiescence_quiet": _m_quiescence_quiet,
    "device_state_companion": _m_device_state_companion,
    "per_output_integrity": _m_per_output_integrity,
    "all_output_equivalence": _m_all_output_equivalence,
    "calibration_content_disjoint": _m_calibration_content_disjoint,
    "decode_p128_separate": _m_decode_p128_separate,
    "provenance_claim_accurate": _m_provenance_claim_accurate,
    "refusal_output_sanitized": _m_refusal_output_sanitized,
    "digests_platform_stable": _m_digests_platform_stable,
    "no_dispersion_promotion": _m_no_dispersion_promotion,
    "isolation_language_cooperative": _m_isolation_language_cooperative,
    "headline_scope_not_widened": _m_headline_scope_not_widened,
    "uncertainty_qualified": _m_uncertainty_qualified,
}


def test_every_condition_has_a_mutator():
    assert set(MUTATORS) == set(pe.GATE_CONDITIONS), (
        "a gate condition has no negative-polarity mutator: "
        f"{set(pe.GATE_CONDITIONS) ^ set(MUTATORS)}"
    )


@pytest.mark.parametrize("condition", sorted(pe.GATE_CONDITIONS))
def test_mutation_fails_exactly_its_condition(condition):
    record = _admissible()
    MUTATORS[condition](record)
    conditions = pe._evaluate(record)
    assert conditions[condition]["ok"] is False, f"{condition} did not fail under its mutation"


def test_each_condition_callable_directly_both_polarities():
    # The instrument census reads the test AST for direct by-name calls. Exercise every
    # condition by name, in both polarities, so each is INVOKED (not left as drift) and so a
    # reader sees the accept and reject sides side by side.
    good = _admissible()
    assert pe.record_wellformed(good)[0] is True
    assert pe.immutable_run_binding(good)[0] is True
    assert pe.device_identity_immutable(good)[0] is True
    assert pe.model_identity_provenance(good)[0] is True
    assert pe.quiescence_quiet(good)[0] is True
    assert pe.device_state_companion(good)[0] is True
    assert pe.per_output_integrity(good)[0] is True
    assert pe.all_output_equivalence(good)[0] is True
    assert pe.calibration_content_disjoint(good)[0] is True
    assert pe.decode_p128_separate(good)[0] is True
    assert pe.provenance_claim_accurate(good)[0] is True
    assert pe.refusal_output_sanitized(good)[0] is True
    assert pe.digests_platform_stable(good)[0] is True
    assert pe.no_dispersion_promotion(good)[0] is True
    assert pe.isolation_language_cooperative(good)[0] is True
    assert pe.headline_scope_not_widened(good)[0] is True
    assert pe.uncertainty_qualified(good)[0] is True

    bad = _admissible()
    _m_record_wellformed(bad)
    assert pol.denies(pe.record_wellformed(bad))
    bad = _admissible(); _m_immutable_run_binding(bad)
    assert pol.denies(pe.immutable_run_binding(bad))
    bad = _admissible(); _m_device_identity_immutable(bad)
    assert pol.denies(pe.device_identity_immutable(bad))
    bad = _admissible(); _m_model_identity_provenance(bad)
    assert pol.denies(pe.model_identity_provenance(bad))
    bad = _admissible(); _m_quiescence_quiet(bad)
    assert pol.denies(pe.quiescence_quiet(bad))
    bad = _admissible(); _m_device_state_companion(bad)
    assert pol.denies(pe.device_state_companion(bad))
    bad = _admissible(); _m_per_output_integrity(bad)
    assert pol.denies(pe.per_output_integrity(bad))
    bad = _admissible(); _m_all_output_equivalence(bad)
    assert pol.denies(pe.all_output_equivalence(bad))
    bad = _admissible(); _m_calibration_content_disjoint(bad)
    assert pol.denies(pe.calibration_content_disjoint(bad))
    bad = _admissible(); _m_decode_p128_separate(bad)
    assert pol.denies(pe.decode_p128_separate(bad))
    bad = _admissible(); _m_provenance_claim_accurate(bad)
    assert pol.denies(pe.provenance_claim_accurate(bad))
    bad = _admissible(); _m_refusal_output_sanitized(bad)
    assert pol.denies(pe.refusal_output_sanitized(bad))
    bad = _admissible(); _m_digests_platform_stable(bad)
    assert pol.denies(pe.digests_platform_stable(bad))
    bad = _admissible(); _m_no_dispersion_promotion(bad)
    assert pol.denies(pe.no_dispersion_promotion(bad))
    bad = _admissible(); _m_isolation_language_cooperative(bad)
    assert pol.denies(pe.isolation_language_cooperative(bad))
    bad = _admissible(); _m_headline_scope_not_widened(bad)
    assert pol.denies(pe.headline_scope_not_widened(bad))
    bad = _admissible(); _m_uncertainty_qualified(bad)
    assert pol.denies(pe.uncertainty_qualified(bad))


def test_publish_withholds_on_inadmissible_record():
    # The reject polarity of the publication authority itself: an inadmissible record must
    # come back withheld (no quotable verdict, refusal reasons carried). `pol.suppresses`
    # raises unless that contract holds, so a `publish` that leaked a number cannot pass.
    bad = _admissible()
    _m_quiescence_quiet(bad)  # the standing state of this box
    withheld = pol.suppresses(pe.publish(bad))
    assert withheld["timing_verdict"] == pe.INDETERMINATE
    # accept polarity: the admissible fixture publishes a quotable verdict.
    assert pe.publish(_admissible())["timing_verdict"] == "QUOTABLE"


def test_suppress_timings_and_evaluate_directly():
    assert pe._suppress_timings({"median_ms": 5.0, "keep": 1}) == {
        "median_ms": pe.SUPPRESSED, "keep": 1}
    assert isinstance(pe._evaluate(_admissible()), dict)


@pytest.mark.parametrize("condition", sorted(pe.GATE_CONDITIONS))
def test_any_broken_condition_refuses_timing(condition):
    record = _admissible()
    MUTATORS[condition](record)
    pub = pe.publish(record)
    # A broken timing-admissibility condition must refuse; the rest still cannot publish a
    # quotable number because the whole record is then internally inconsistent, but at minimum
    # the broken condition appears in refusals.
    assert any(r["condition"] == condition for r in pub["refusals"])
    if condition in pe.TIMING_ADMISSIBILITY:
        assert pub["timing_admissible"] is False
        assert pub["timing_verdict"] == pe.INDETERMINATE


# --- suppression behaviour ------------------------------------------------- #
def test_refusal_suppresses_every_timing_number():
    record = _admissible()
    _m_quiescence_quiet(record)  # the real state of this box
    pub = pe.publish(record)
    assert pub["timing_admissible"] is False
    # F3: no hard-coded literals -- the leak detector derives the banned set from the record.
    assert pe._residual_timing_leak(record, pub) == []
    assert pe.SUPPRESSED in repr(pub["subjects"])


def test_residual_timing_leak_both_polarities():
    # Accept polarity: a properly suppressed publication leaks nothing.
    clean = _admissible()
    _m_quiescence_quiet(clean)
    pub = pe.publish(clean)
    assert pe._residual_timing_leak(clean, pub) == []
    # Reject polarity: a publication that still carries an input timing float is a leak the
    # detector must report (this is the alarm firing, not staying silent).
    leaked_pub = dict(pub)
    leaked_pub["subjects"] = [{"name": "prefill/M128", "arms": [{"median_ms": _SYNTH_PREFILL_HEAD_MS}]}]
    leak = pe._residual_timing_leak(clean, leaked_pub)
    assert leak == [_SYNTH_PREFILL_HEAD_MS], "the leak detector failed to flag a surviving timing"


def test_value_based_suppression_catches_timing_under_innocuous_key():
    # F2: a timing float copied under a non-timing key name must not survive suppression.
    record = _admissible()
    _m_quiescence_quiet(record)
    # Smuggle the head prefill median under a key the name-based screen would ignore.
    record["subjects"][0]["arms"][0]["note_value"] = _SYNTH_PREFILL_HEAD_MS
    pub = pe.publish(record)
    assert pe._residual_timing_leak(record, pub) == [], "a timing float leaked under a plain key"
    flat = repr(pub["subjects"])
    assert str(_SYNTH_PREFILL_HEAD_MS) not in flat


def test_prefill_becomes_steady_uncertified_on_refusal():
    record = _admissible()
    _m_device_state_companion(record)
    pub = pe.publish(record)
    prefill = next(s for s in pub["subjects"] if s["name"].startswith("prefill"))
    assert prefill["timing_verdict"] == pe.STEADY_UNCERTIFIED


def test_decode_stays_indeterminate_and_separate():
    record = _admissible()
    _m_quiescence_quiet(record)
    pub = pe.publish(record)
    decode = next(s for s in pub["subjects"] if s["name"] == "decode/M1/past128")
    assert decode["timing_verdict"] == pe.INDETERMINATE
    assert decode["pooled"] is False


def test_report_refusal_lists_reasons_without_private_paths():
    record = _admissible()
    _m_quiescence_quiet(record)
    lines = pe._report(pe.publish(record))
    joined = "\n".join(lines)
    assert "REFUSED" in joined
    assert "quiescence_quiet" in joined
    assert "C:\\Users" not in joined and "/home/" not in joined


# --- drift guard: our verdict names must equal the source modules ---------- #
def test_verdict_names_track_source():
    contention = pytest.importorskip("contention")
    companion = pytest.importorskip("device_companion")
    assert (pe.QUIET, pe.CONTENDED, pe.UNMEASURED) == (
        contention.QUIET, contention.CONTENDED, contention.UNMEASURED)
    assert (pe.COMPANION_QUOTABLE, pe.COMPANION_WITHHELD, pe.COMPANION_UNCERTIFIED) == (
        companion.QUOTABLE, companion.WITHHELD, companion.UNCERTIFIED)


def test_lf_digest_is_crlf_stable():
    assert pe._lf_digest("a\r\nb\r\n") == pe._lf_digest("a\nb\n")


def test_all_printed_strings_are_ascii():
    # The reviewer flagged a cp1252/UTF-8 console failure from a non-ASCII banner. Every reason
    # and every report line the gate prints must survive a strict-ASCII stdout.
    record = _admissible()
    _m_quiescence_quiet(record)
    pub = pe.publish(record)
    for line in pe._report(pub):
        line.encode("ascii")
    for r in pub["refusals"]:
        r["reason"].encode("ascii")
    pe.SUPPRESSED.encode("ascii")
    for name in pe.GATE_CONDITIONS:
        pe.GATE_CONDITIONS[name](_admissible())[1].encode("ascii")


# --- F4: anti-overclaim conditions must gate publication ------------------- #
_ANTI_OVERCLAIM = (
    "headline_scope_not_widened",
    "no_dispersion_promotion",
    "isolation_language_cooperative",
    "provenance_claim_accurate",
    "decode_p128_separate",
    "calibration_content_disjoint",
    "refusal_output_sanitized",
    "record_wellformed",
)


def test_timing_admissibility_is_the_whole_registry():
    # F4/F5: a number may be published only when every condition passes; there is no
    # timing-only subset that could admit a figure while an anti-overclaim condition refuses.
    assert set(pe.TIMING_ADMISSIBILITY) == set(pe.GATE_CONDITIONS)
    assert not (set(pe.TIMING_ADMISSIBILITY) - set(pe.GATE_CONDITIONS))


@pytest.mark.parametrize("condition", _ANTI_OVERCLAIM)
def test_anti_overclaim_condition_blocks_quotable(condition):
    # Negative arm: a record that would otherwise be quotable but fails an anti-overclaim
    # condition must NOT publish a number -- it must refuse.
    record = _admissible()
    MUTATORS[condition](record)
    pub = pe.publish(record)
    assert pub["timing_admissible"] is False, f"{condition} did not block publication"
    assert pub["timing_verdict"] == pe.INDETERMINATE
    assert pol.suppresses(pub)  # raises unless the figure was actually withheld


def test_anti_overclaim_positive_still_publishes():
    # Positive arm: with no anti-overclaim condition broken, the clean fixture still publishes.
    assert pe.publish(_admissible())["timing_admissible"] is True


# --- F6: sanitization scope covers the whole published surface ------------- #
def test_sanitization_covers_non_refusal_fields():
    # Negative arm: a private path in a field OTHER than `refusals` (here a device string) is
    # caught; the v4-era scan looked only at `refusals` and would have missed it.
    record = _admissible()
    record["device"]["driver_name"] = r"NVIDIA at C:\Users\secret-user\driver"
    assert pol.denies(pe.refusal_output_sanitized(record))
    # Positive arm: the clean fixture carries no private path anywhere.
    assert pe.refusal_output_sanitized(_admissible())[0] is True


# --- F7: the 65-output correctness witness is only claimed when earned ------ #
def test_recordless_publish_claims_no_correctness_witness():
    # Negative arm: with no subjects there is no correctness result, so the witness is not
    # established and nothing asserts all-output equivalence.
    pub = pe.publish({"schema": "phi69-evidence/v5", "subjects": [], "raw_runs": []})
    assert pub["correctness"]["established"] is False
    assert pub["correctness"]["all_output_equivalence"] is False
    assert pub["correctness"]["outputs_verified"] == 0


def test_truncated_outputs_do_not_establish_correctness():
    # Negative arm: present rows all MATCH, but fewer than 65 -> the witness is NOT established.
    record = _admissible()
    _m_per_output_integrity(record)  # truncates one arm's per_output to a single row
    pub = pe.publish(record)
    assert pub["correctness"]["established"] is False


def test_full_record_establishes_correctness_witness():
    # Positive arm: a full 65-output record with every row MATCH establishes the witness.
    pub = pe.publish(_admissible())
    assert pub["correctness"]["established"] is True
    assert pub["correctness"]["outputs_expected"] == pe.EXPECTED_OUTPUTS


# --- F8: no_dispersion_promotion must not pass vacuously ------------------- #
def test_no_dispersion_absence_is_not_vacuous():
    # Negative arm: an absent dispersion block must FAIL (it used to pass vacuously).
    record = _admissible()
    del record["within_arm_dispersion"]
    assert pol.denies(pe.no_dispersion_promotion(record))
    # Also an implicit/empty block cannot certify anything.
    record2 = _admissible()
    record2["within_arm_dispersion"] = {}
    assert pol.denies(pe.no_dispersion_promotion(record2))
    # Positive arm: a present, diagnostic, non-promoting block passes.
    assert pe.no_dispersion_promotion(_admissible())[0] is True
