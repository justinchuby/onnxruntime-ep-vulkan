"""The shipped gates, watched to disagree — by mutating the shipped source, not a copy of it.

WHAT THIS FILE IS FOR
=====================
PR #95 was rejected on B1: its headline mutation test called the probe's ``verdict_for``,
asserted the raw verdict still read ``FASTER``, and then **re-implemented the witness gate in
the test body** and asserted its own copy fired.  Deleting the shipped gate left 37 crossbuild
tests, 104 bench tests, 14 census tests and the instrument audit all green.

So this suite is built on one rule: **no assertion here re-implements anything.**  Every
property below is computed by calling :mod:`bench.crossbuild_summary` and nothing else, and
every gate is then *deleted from that module's source* and the same property is required to
fail.  ``_MUTATIONS`` is the matrix; ``test_every_mutation_breaks_the_property_it_should`` is
the proof.  If somebody deletes a gate, they do not have to remember to update a test — the
test is already watching that exact line.

The mutants are compiled in memory from the real source text.  Nothing is written to disk, and
the shipped module object is never patched, so a mutation cannot leak into another test.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import crossbuild_summary as xb  # noqa: E402

SOURCE_TEXT = pathlib.Path(xb.__file__).read_text(encoding="utf-8")

M128 = "phi-3.5-mini-instruct-cuda-int4-rtn-block-32/prefill/M128/past0"
M32 = "phi-3.5-mini-instruct-cuda-int4-rtn-block-32/prefill/M32/past0"
M1 = "phi-3.5-mini-instruct-cuda-int4-rtn-block-32/prefill/M1/past0"
P128 = "phi-3.5-mini-instruct-cuda-int4-rtn-block-32/decode/M1/past128"
MOBILENET = "mobilenetv2-12/batch/N1"


@pytest.fixture(scope="module")
def frozen():
    return xb.load_frozen()


def _records(frozen, workload):
    return copy.deepcopy([r for r in frozen["records"] if r["workload"] == workload])


def _mutant(*replacements):
    """Compile a copy of the shipped module with one gate removed.

    Anchors must be unique in the source: an anchor that matches twice would silently mutate
    something other than the thing named, which is the failure mode this whole file exists to
    prevent.
    """
    src = SOURCE_TEXT
    for old, new in replacements:
        assert src.count(old) == 1, f"mutation anchor is not unique: {old!r}"
        src = src.replace(old, new)
    mod = types.ModuleType("crossbuild_summary_mutant")
    mod.__file__ = xb.__file__
    exec(compile(src, xb.__file__, "exec"), mod.__dict__)  # noqa: S102 - deliberate
    return mod


# ---------------------------------------------------------------------------------------
# THE PROPERTIES. Each one is a question asked of a module, answered only by calling it.
# ---------------------------------------------------------------------------------------
def _identical_witness_refuses_faster(mod, frozen):
    """The M=128 row, its witnesses forced identical, must lose its FASTER verdict."""
    recs = _records(frozen, M128)
    for r in recs:
        r["path_witness"]["gqa_keys"] = ["gqa_f16:64"]
    row = mod.pair_repeats(recs, M128, models=frozen["models"])
    graded = mod.gated_verdict(row, 0.153416)
    return graded["raw_verdict"] == "FASTER" and graded["verdict"] == "REFUSED"


def _missing_equivalence_refuses(mod, frozen):
    r = _records(frozen, M128)[0]
    r.pop("equivalence")
    return bool(mod.record_refusals(r, models=frozen["models"]))


def _empty_equivalence_refuses(mod, frozen):
    r = _records(frozen, M128)[0]
    r["equivalence"] = {}
    return bool(mod.record_refusals(r, models=frozen["models"]))


def _divergent_equivalence_refuses(mod, frozen):
    r = _records(frozen, M128)[0]
    r["equivalence"]["verdict"] = "DIVERGENT"
    return bool(mod.record_refusals(r, models=frozen["models"]))


def _wrong_model_digest_refuses(mod, frozen):
    r = _records(frozen, M128)[0]
    r["model_sha256"] = "0" * 64
    return bool(mod.record_refusals(r, models=frozen["models"]))


def _wrong_artifact_model_digest_refuses(mod, frozen):
    r = _records(frozen, M128)[0]
    models = copy.deepcopy(frozen["models"])
    models[r["model_key"]]["sha256"] = "1" * 64
    return bool(mod.record_refusals(r, models=models))


def _refusal_retaining_speed_raises(mod, frozen):
    r = _records(frozen, M128)[0]
    r["refusal"] = {"why": "planted"}
    try:
        mod.record_refusals(r, models=frozen["models"])
    except mod.AdmissibilityError:
        return True
    return False


def _inadmissible_retaining_speed_raises(mod, frozen):
    r = _records(frozen, M128)[0]
    r["admissible"] = False
    try:
        mod.record_refusals(r, models=frozen["models"])
    except mod.AdmissibilityError:
        return True
    return False


def _borrowed_witness_refuses(mod, frozen):
    """A witness lifted from MobileNetV2 carries MobileNetV2's claimed-node count."""
    recs = _records(frozen, M128)
    donor = [r for r in frozen["records"] if r["workload"] == MOBILENET][0]
    recs[0]["path_witness"] = copy.deepcopy(donor["path_witness"])
    row = mod.pair_repeats(recs, M128, models=frozen["models"])
    return bool(row["refusals"]) and mod.gated_verdict(row, 0.153416)["verdict"] == "REFUSED"


def _dropped_repeat_refuses(mod, frozen):
    recs = _records(frozen, M128)
    for r in recs:
        if r["repeat"] == 2:
            r["equivalence"]["verdict"] = "DIVERGENT"
    row = mod.pair_repeats(recs, M128, models=frozen["models"])
    return (
        row["repeats_paired"] == 2
        and mod.gated_verdict(row, 0.153416)["verdict"] == "REFUSED"
    )


def _missing_repeat_refuses(mod, frozen):
    """A repeat that never happened leaves no refusal behind — only the completeness gate.

    This is deliberately a different shape from ``_dropped_repeat_refuses``: there the repeat
    was refused and said so, here it is simply absent, and absence is the case in which a
    two-repeat verdict would otherwise be published as though three had been run.
    """
    recs = [r for r in _records(frozen, M128) if r["repeat"] != 2]
    row = mod.pair_repeats(recs, M128, models=frozen["models"])
    return (
        row["repeats_paired"] == 2
        and not row["refusals"]
        and mod.gated_verdict(row, 0.153416)["verdict"] == "REFUSED"
    )


def _divergent_outputs_refuse(mod, frozen):
    recs = _records(frozen, M128)
    for r in recs:
        if r["arm"] == "baseline":
            r["outputs_sha256"] = "2" * 64
            r["outputs_sha256_post_timing"] = "2" * 64
    row = mod.pair_repeats(recs, M128, models=frozen["models"])
    return mod.gated_verdict(row, 0.153416)["verdict"] == "REFUSED"


def _calibration_is_never_graded(mod, frozen):
    summary = mod.summarize(frozen)
    calib = [r for r in summary["rows"] if r["witness_class"] == mod.NO_GQA]
    return bool(calib) and all(r["verdict"] == "CALIBRATION" for r in calib)


def _band_excludes_every_subject(mod, frozen):
    """B2: no row that the band grades may have contributed a ratio to the band."""
    summary = mod.summarize(frozen)
    calibrating = set(summary["band"]["calibration_workloads"])
    graded = {r["workload"] for r in summary["rows"] if r["role"] == "subject"}
    return bool(calibrating) and bool(graded) and not (calibrating & graded)


def _null_control_can_still_be_claimed(mod, frozen):
    """B2's other half: the null control's verdict must be able to MOVE.

    Niobe's repro: a null control reading 1.80x / 1.90x / 2.00x -- an 'improvement' of 90% on
    a workload where both builds dispatch the same geometry -- came back NEUTRAL under PR #95's
    band in every possible universe.  Here it must come back FASTER.
    """
    recs = _records(frozen, M1)
    for r in recs:
        if r["arm"] == "baseline":
            factor = {0: 1.80, 1: 1.90, 2: 2.00}[r["repeat"]]
            r["speed"]["samples_ms"] = [s * factor for s in r["speed"]["samples_ms"]]
    doc = copy.deepcopy(frozen)
    doc["records"] = [r for r in doc["records"] if r["workload"] != M1] + recs
    summary = mod.summarize(doc)
    row = [r for r in summary["rows"] if r["workload"] == M1][0]
    return row["verdict"] == "FASTER"


def _frozen_digest_is_enforced(mod, frozen):
    try:
        mod.load_frozen(expect_sha256="f" * 64)
    except mod.ProvenanceError:
        return True
    return False


def _superseded_blocks_are_withheld(mod, frozen):
    doc = mod.load_frozen()
    return all(block not in doc for block in mod.SUPERSEDED_BLOCKS)


# --- the offline within-arm (A/A surrogate) diagnostic ---------------------------------
def _row(mod, frozen, workload):
    return [r for r in mod.summarize(frozen)["rows"] if r["workload"] == workload][0]


def _p128_is_provisional_descriptive(mod, frozen):
    """0.859x is a provisional descriptive ratio, bound to issue #96, and not a SLOWER call."""
    row = _row(mod, frozen, P128)
    status = row["descriptive_status"]
    return (
        row["verdict"] == "INDETERMINATE"
        and status["status"] == mod.PROVISIONAL_DESCRIPTIVE
        and status["quotable_as"] == "description_only"
        and "#96" in (status["until"] or "")
        and "A/A" in (status["until"] or "")
    )


def _p128_is_not_separated_from_within_arm_drift(mod, frozen):
    """The reason it stays provisional: its 13.7% sits inside this host's own A/A envelope."""
    sep = _row(mod, frozen, P128)["separation_from_drift"]
    return sep["class"] == mod.NOT_SEPARATED and sep["separation_ratio"] < 1.0


def _m128_is_separated_from_within_arm_drift(mod, frozen):
    """The other side of the same instrument: M=128 clears the envelope by more than 10x."""
    sep = _row(mod, frozen, M128)["separation_from_drift"]
    return sep["class"] == mod.SEPARATED and sep["separation_ratio"] > 10.0


def _within_arm_envelope_uses_the_larger_surrogate(mod, frozen):
    """The published envelope is the conservative one, and at P128 that choice bites."""
    rows = mod.summarize(frozen)["rows"]
    if not all(
        r["within_arm"]["within_arm_envelope"]
        == max(r["within_arm"]["across_repeat_envelope"], r["within_arm"]["split_half_envelope"])
        for r in rows
    ):
        return False
    wa = [r for r in rows if r["workload"] == P128][0]["within_arm"]
    return wa["split_half_envelope"] > wa["across_repeat_envelope"]


def _zero_envelope_is_refused(mod, frozen):
    """An envelope of zero would make every effect infinitely separated. It must not render."""
    row = _row(mod, frozen, P128)
    try:
        mod.separation_from_drift(row, {"within_arm_envelope": 0.0})
    except mod.SchemaError:
        return True
    return False


def _within_arm_is_a_diagnostic_not_a_gate(mod, frozen):
    """The dispersion may never move a verdict: the band decides, this only describes."""
    summary = mod.summarize(frozen)
    band = summary["band"]["applied"]
    for row in summary["rows"]:
        bare = mod.pair_repeats(frozen["records"], row["workload"], models=frozen["models"])
        if mod.gated_verdict(bare, band)["verdict"] != row["verdict"]:
            return False
    return True


def _m32_is_quotable_only_as_a_floor(mod, frozen):
    """M=32's magnitude is inside its own arm's drift, so the headline quotes the floor."""
    claims = {c["workload"]: c for c in mod.summarize(frozen)["headline"]["claims"]}
    m32 = claims.get(M32)
    return bool(m32) and m32["quotable_as"] == "floor_only" and m32["quote"] == "at least 1.278x"


def _headline_refuses_more_than_one_model(mod, frozen):
    """A claim that silently spread to a second model must not render as one sentence."""
    rows = mod.summarize(frozen)["rows"]
    for row in rows:
        if row["workload"] == MOBILENET:
            row["verdict"] = "FASTER"
    try:
        mod.headline(rows)
    except mod.SchemaError:
        return True
    return False


PROPERTIES = {
    "identical_witness_refuses_faster": _identical_witness_refuses_faster,
    "missing_equivalence_refuses": _missing_equivalence_refuses,
    "empty_equivalence_refuses": _empty_equivalence_refuses,
    "divergent_equivalence_refuses": _divergent_equivalence_refuses,
    "wrong_model_digest_refuses": _wrong_model_digest_refuses,
    "wrong_artifact_model_digest_refuses": _wrong_artifact_model_digest_refuses,
    "refusal_retaining_speed_raises": _refusal_retaining_speed_raises,
    "inadmissible_retaining_speed_raises": _inadmissible_retaining_speed_raises,
    "borrowed_witness_refuses": _borrowed_witness_refuses,
    "dropped_repeat_refuses": _dropped_repeat_refuses,
    "missing_repeat_refuses": _missing_repeat_refuses,
    "divergent_outputs_refuse": _divergent_outputs_refuse,
    "calibration_is_never_graded": _calibration_is_never_graded,
    "band_excludes_every_subject": _band_excludes_every_subject,
    "null_control_can_still_be_claimed": _null_control_can_still_be_claimed,
    "frozen_digest_is_enforced": _frozen_digest_is_enforced,
    "superseded_blocks_are_withheld": _superseded_blocks_are_withheld,
    "p128_is_provisional_descriptive": _p128_is_provisional_descriptive,
    "p128_is_not_separated_from_within_arm_drift": _p128_is_not_separated_from_within_arm_drift,
    "m128_is_separated_from_within_arm_drift": _m128_is_separated_from_within_arm_drift,
    "within_arm_envelope_uses_the_larger_surrogate": _within_arm_envelope_uses_the_larger_surrogate,
    "zero_envelope_is_refused": _zero_envelope_is_refused,
    "within_arm_is_a_diagnostic_not_a_gate": _within_arm_is_a_diagnostic_not_a_gate,
    "m32_is_quotable_only_as_a_floor": _m32_is_quotable_only_as_a_floor,
    "headline_refuses_more_than_one_model": _headline_refuses_more_than_one_model,
}


# ---------------------------------------------------------------------------------------
# THE MUTATION MATRIX. Each row: delete one shipped gate, name the properties that must die.
# ---------------------------------------------------------------------------------------
_MUTATIONS = {
    "delete_witness_gate": (
        [
            (
                'elif row.get("witness_class") == NOT_DISTINGUISHED and raw in ("FASTER", "SLOWER"):',
                "elif False:",
            )
        ],
        ["identical_witness_refuses_faster"],
    ),
    "bypass_all_gates_by_publishing_the_raw_verdict": (
        [('        "verdict": verdict,\n        "role": role,', '        "verdict": raw,\n        "role": role,')],
        [
            "identical_witness_refuses_faster",
            "borrowed_witness_refuses",
            "dropped_repeat_refuses",
            "divergent_outputs_refuse",
            "calibration_is_never_graded",
        ],
    ),
    "delete_the_admissibility_gate": (
        [("    return why\n\n\ndef witness_class", "    return []\n\n\ndef witness_class")],
        [
            "missing_equivalence_refuses",
            "empty_equivalence_refuses",
            "divergent_equivalence_refuses",
            "wrong_model_digest_refuses",
            "wrong_artifact_model_digest_refuses",
            "dropped_repeat_refuses",
        ],
    ),
    "delete_the_equivalence_premise": (
        [
            (
                '    equivalence = record.get("equivalence")',
                "    equivalence = {'verdict': 'MATCH'}\n    _ignored = record.get('equivalence')",
            )
        ],
        [
            "missing_equivalence_refuses",
            "empty_equivalence_refuses",
            "divergent_equivalence_refuses",
        ],
    ),
    "delete_the_model_digest_premise": (
        [
            (
                '            if digest is not None and digest != pin["sha256"]:',
                "            if False:",
            )
        ],
        ["wrong_model_digest_refuses", "wrong_artifact_model_digest_refuses"],
    ),
    "let_a_withdrawn_record_keep_its_timing": (
        [
            (
                '    if speed_present and record.get("refusal") is not None:',
                "    if False:",
            ),
            (
                '    if speed_present and record.get("admissible") is False:',
                "    if False:",
            ),
        ],
        ["refusal_retaining_speed_raises", "inadmissible_retaining_speed_raises"],
    ),
    "delete_the_borrowed_witness_screen": (
        [('    if len(islands.get("claimed_nodes", {None})) > 1:', "    if False:")],
        ["borrowed_witness_refuses"],
    ),
    "delete_the_completeness_gate": (
        [('    if row.get("repeats_paired") != EXPECTED_REPEATS:', "    if False:")],
        ["missing_repeat_refuses"],
    ),
    "delete_the_cross_arm_output_gate": (
        [('    if not row.get("cross_arm_bitwise_identical"):', "    if False:")],
        ["divergent_outputs_refuse"],
    ),
    "grade_the_calibration_controls_too": (
        [
            (
                '    role = "calibration" if row.get("witness_class") == NO_GQA else "subject"',
                '    role = "subject"',
            )
        ],
        ["calibration_is_never_graded", "band_excludes_every_subject"],
    ),
    "restore_b2_by_calibrating_on_the_subject": (
        [
            (
                '        if r.get("witness_class") == NO_GQA\n',
                '        if r.get("witness_class") is not None\n',
            ),
            (
                '    role = "calibration" if row.get("witness_class") == NO_GQA else "subject"',
                '    role = "subject"',
            ),
        ],
        ["band_excludes_every_subject", "null_control_can_still_be_claimed"],
    ),
    "delete_the_frozen_digest_check": (
        [
            (
                "    if expect_sha256 is not None and got != expect_sha256:",
                "    if False:",
            )
        ],
        ["frozen_digest_is_enforced"],
    ),
    "hand_out_the_superseded_pr95_derivation": (
        [
            (
                "    out = {k: v for k, v in doc.items() if k not in SUPERSEDED_BLOCKS}",
                "    out = dict(doc)",
            )
        ],
        ["superseded_blocks_are_withheld"],
    ),
    # --- the offline within-arm diagnostic ---------------------------------------------
    "promote_the_provisional_ratio_to_a_claim": (
        [
            (
                '    if verdict == "INDETERMINATE" and direction in ("FASTER_SIDE", "SLOWER_SIDE"):',
                "    if False:",
            )
        ],
        ["p128_is_provisional_descriptive"],
    ),
    "shrink_the_within_arm_envelope_to_nothing": (
        [
            (
                '        "within_arm_envelope": max(across_env, split_env),',
                '        "within_arm_envelope": max(across_env, split_env) / 1000.0,',
            )
        ],
        [
            "p128_is_not_separated_from_within_arm_drift",
            "within_arm_envelope_uses_the_larger_surrogate",
        ],
    ),
    "drop_the_split_half_surrogate": (
        [
            (
                '        "within_arm_envelope": max(across_env, split_env),',
                '        "within_arm_envelope": across_env,',
            )
        ],
        ["within_arm_envelope_uses_the_larger_surrogate"],
    ),
    "delete_the_zero_envelope_guard": (
        [
            (
                "    if not isinstance(envelope, (int, float)) or envelope <= 0:",
                "    if False:",
            )
        ],
        ["zero_envelope_is_refused"],
    ),
    "let_the_diagnostic_overturn_a_verdict": (
        [
            (
                '        row["descriptive_status"] = descriptive_status('
                'row, row["separation_from_drift"])',
                '        row["descriptive_status"] = descriptive_status('
                'row, row["separation_from_drift"])\n'
                '        if row["separation_from_drift"]["class"] == NOT_SEPARATED:\n'
                '            row["verdict"] = "REFUSED"',
            )
        ],
        ["within_arm_is_a_diagnostic_not_a_gate"],
    ),
    "quote_the_median_where_only_the_floor_is_supported": (
        [('        floor = r["descriptive_status"]["quotable_as"] == "floor_only"', "        floor = False")],
        ["m32_is_quotable_only_as_a_floor"],
    ),
    "let_the_headline_span_more_than_one_model": (
        [("    if len(models_claimed) != 1:", "    if False:")],
        ["headline_refuses_more_than_one_model"],
    ),
    "make_separation_unreachable": (
        [("SEPARATION_STRONG = 2.0", "SEPARATION_STRONG = 1e9")],
        ["m128_is_separated_from_within_arm_drift"],
    ),
}


def test_every_property_holds_on_the_shipped_module(frozen):
    """Baseline: the module as shipped satisfies every property, computed by calling it."""
    failed = [name for name, probe in PROPERTIES.items() if not probe(xb, frozen)]
    assert not failed, f"the shipped module fails its own properties: {failed}"


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_every_mutation_breaks_the_property_it_should(mutation, frozen):
    """Delete a shipped gate; the named properties must go red. This is B1, mechanised.

    The point of the *pair* of tests is that neither alone is evidence.  A property that holds
    on the shipped module might hold on any module (that is exactly what PR #95's suite could
    not distinguish); a property that fails on a mutant might fail on everything.  Both
    together say the property is load-bearing on the shipped line.
    """
    replacements, must_break = _MUTATIONS[mutation]
    mutant = _mutant(*replacements)
    mutant.FROZEN_PATH = xb.FROZEN_PATH
    survived = []
    for name in must_break:
        try:
            still_true = PROPERTIES[name](mutant, frozen)
        except Exception:  # a mutant that explodes has also lost the property
            still_true = False
        if still_true:
            survived.append(name)
    assert not survived, (
        f"mutation {mutation!r} deleted a shipped gate and these properties survived: "
        f"{survived}. A guard nothing can falsify is not a guard."
    )


def test_the_mutation_matrix_covers_every_property():
    """No property may sit in the suite without a mutation that kills it."""
    covered = {name for _, names in _MUTATIONS.values() for name in names}
    assert set(PROPERTIES) == covered, (
        f"properties with no mutation: {sorted(set(PROPERTIES) - covered)}; "
        f"mutations naming an unknown property: {sorted(covered - set(PROPERTIES))}"
    )


# ---------------------------------------------------------------------------------------
# TWO-POLARITY COVERAGE for the census (rust/tools/audit_instruments.py, bench domain).
#
# Every module-public function of `crossbuild_summary.py` is an instrument by the bench
# screen's structural rule. Each one is called here bare (accept polarity) and inside
# `pytest.raises` (reject polarity), so none of them enters the baseline as `unfalsified`.
# These are deliberately small: the *evidence* that the gates work is the mutation battery
# above, and this section claims nothing more than that each entry point has been watched to
# refuse as well as to answer.
# ---------------------------------------------------------------------------------------
def test_load_frozen_reads_the_pin_and_refuses_a_substitute(frozen, tmp_path_factory):
    doc = xb.load_frozen()
    assert len(doc["records"]) == 60
    assert doc["_frozen"]["sha256"] == xb.FROZEN_SHA256
    with pytest.raises(xb.ProvenanceError):
        xb.load_frozen(expect_sha256="0" * 64)


def test_load_frozen_refuses_a_path_that_is_not_the_evidence():
    with pytest.raises(xb.ProvenanceError):
        xb.load_frozen(pathlib.Path(__file__), expect_sha256=None)


def test_record_refusals_answers_and_refuses(frozen):
    good = _records(frozen, M128)[0]
    assert xb.record_refusals(good, models=frozen["models"]) == []
    with pytest.raises(xb.SchemaError):
        xb.record_refusals({"not": "a record"})


def test_witness_class_answers_and_refuses(frozen):
    cand = [r for r in frozen["records"] if r["workload"] == M128 and r["arm"] == "candidate"][0]
    base = [r for r in frozen["records"] if r["workload"] == M128 and r["arm"] == "baseline"][0]
    mob = [r for r in frozen["records"] if r["workload"] == MOBILENET][0]
    assert xb.witness_class(cand["path_witness"], base["path_witness"]) == xb.DISTINGUISHED
    assert xb.witness_class(mob["path_witness"], mob["path_witness"]) == xb.NO_GQA
    assert xb.witness_class(cand["path_witness"], cand["path_witness"]) == xb.NOT_DISTINGUISHED
    with pytest.raises(xb.SchemaError):
        xb.witness_class(cand["path_witness"], {"present": False})


def test_pair_repeats_answers_and_refuses(frozen):
    row = xb.pair_repeats(frozen["records"], M128, models=frozen["models"])
    assert row["repeats_paired"] == 3
    with pytest.raises(xb.SchemaError):
        xb.pair_repeats(frozen["records"], "no/such/workload")


def test_calibration_band_answers_and_refuses(frozen):
    rows = [
        xb.pair_repeats(frozen["records"], w, models=frozen["models"])
        for w in sorted({r["workload"] for r in frozen["records"]})
    ]
    band = xb.calibration_band(rows)
    assert band["n_calibration_ratios"] == 12
    with pytest.raises(xb.SchemaError):
        xb.calibration_band([r for r in rows if r["witness_class"] != xb.NO_GQA])


def test_raw_verdict_answers_and_refuses():
    assert xb.raw_verdict([2.0, 2.1, 2.2], 0.15) == "FASTER"
    assert xb.raw_verdict([0.5, 0.6, 0.55], 0.15) == "SLOWER"
    assert xb.raw_verdict([1.0, 1.01, 0.99], 0.15) == "NEUTRAL"
    assert xb.raw_verdict([1.0, 2.0, 0.5], 0.15) == "INDETERMINATE"
    with pytest.raises(xb.SchemaError):
        xb.raw_verdict([], 0.15)


def test_raw_verdict_refuses_a_nonpositive_band():
    assert xb.raw_verdict([1.5, 1.6, 1.7], 0.05) == "FASTER"
    with pytest.raises(xb.SchemaError):
        xb.raw_verdict([1.5, 1.6, 1.7], 0.0)


def test_gated_verdict_answers_and_refuses(frozen):
    row = xb.pair_repeats(frozen["records"], M128, models=frozen["models"])
    assert xb.gated_verdict(row, 0.153416)["verdict"] == "FASTER"
    with pytest.raises(xb.SchemaError):
        xb.gated_verdict({"workload": M128}, 0.15)


def test_summarize_answers_and_refuses(frozen):
    summary = xb.summarize(frozen)
    assert summary["counts"]["records"] == 60
    with pytest.raises(xb.SchemaError):
        xb.summarize({"records": []})


def test_summarize_refuses_an_artifact_still_carrying_pr95s_derivation(frozen):
    poisoned = dict(frozen)
    poisoned["band"] = {"applied": 0.0510292125170773}
    with pytest.raises(xb.ProvenanceError):
        xb.summarize(poisoned)


def test_sensitivity_answers_and_refuses(frozen):
    summary = xb.summarize(frozen)
    swept = xb.sensitivity(summary["rows"])
    assert len(swept["bands"]) == len(xb.SENSITIVITY_BANDS)
    with pytest.raises(xb.SchemaError):
        xb.sensitivity(summary["rows"], bands=[])


def test_markdown_table_answers_and_refuses(frozen):
    summary = xb.summarize(frozen)
    table = xb.markdown_table(summary)
    assert table.count("\n") == len(summary["rows"]) + 1
    with pytest.raises(xb.SchemaError):
        xb.markdown_table({"schema": "something/else"})


def test_dispatch_grid_claim_answers_and_refuses(frozen):
    row = xb.pair_repeats(frozen["records"], P128, models=frozen["models"])
    claim = xb.dispatch_grid_claim(row)
    assert claim["grids_equal_across_arms"] is True
    assert claim["inferred_grid"] == [32, 1, 1]
    with pytest.raises(xb.SchemaError):
        xb.dispatch_grid_claim({"workload": P128})


# ---------------------------------------------------------------------------------------
# THE RULE ITSELF
# ---------------------------------------------------------------------------------------
def test_the_decision_rule_is_symmetric():
    """PR #95 required every repeat for FASTER but only the median for SLOWER.

    An asymmetric rule makes a regression cheaper to claim than an improvement, which is the
    wrong way round for a change whose author wants the improvement.
    """
    band = 0.10
    mirrored = [1 / r for r in (1.30, 1.05, 1.40)]
    assert xb.raw_verdict([1.30, 1.05, 1.40], band) == "INDETERMINATE"
    assert xb.raw_verdict(mirrored, band) == "INDETERMINATE"


def test_the_band_floor_is_data_independent():
    """The floor is a constant. If the calibration envelope collapses, 5% still binds."""
    rows = [
        {
            "workload": "synthetic/control",
            "witness_class": xb.NO_GQA,
            "ratios": [1.0, 1.0, 1.0],
            "refusals": [],
            "repeats_paired": 3,
            "cross_arm_bitwise_identical": True,
        }
    ]
    band = xb.calibration_band(rows)
    assert band["applied"] == xb.BAND_FLOOR
    assert band["floor_binds"] is True


def test_the_band_provenance_says_post_hoc_out_loud():
    """No pre-registration is claimed for this band, because none exists."""
    assert "POST-HOC" in xb.BAND_PROVENANCE
    assert "no externally timestamped pre-timing rule" in xb.BAND_PROVENANCE
    lowered = xb.BAND_PROVENANCE.lower()
    assert "pre-registered" not in lowered and "preregistered" not in lowered
