"""Locks the third-arm driver `bench/results/probe_pr97_third_arm.py`, GPU-free.

The PR #97 arm renames the decode kernel (`gqa_f16` -> `gqa_decode_f16`) and specialises it with a
KV-parallel factor `W` the *host* picks from cache capacity. That moves two things a reviewer
cannot check by eye into code:

  * `kv_parallel()` — a reimplementation of PR #97's host-side rule, written from its source rather
    than imported from it, so the predeclared witness table is an independent prediction and not a
    restatement. `test_kv_parallel_matches_independent_recomputation` recomputes it a third way,
    from the definition in the frozen pre-registration, and demands all three agree.
  * `gqa_witness_multi()` — a *widening* of the frozen extractor, which matched `gqa_f16` by exact
    name and therefore saw nothing at all on a PR #97 build. Widening an extractor that feeds an
    admissibility gate is precisely the kind of change that can turn a refusal into a pass, so the
    tests below assert what it must still refuse: the wrong `W`, both kernels at once, and no GQA
    kernel at all.

Nothing here opens a device, loads a model, or reads a DLL.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(RESULTS))

pr97 = pytest.importorskip("probe_pr97_third_arm")
base = pytest.importorskip("probe_crossbuild_decode_window")


# --------------------------------------------------------------------------------------------
# The host-side W rule.
# --------------------------------------------------------------------------------------------

def _independent_kv_parallel(bound: int) -> int:
    """Recomputed from the prose definition in `prereg_pr97_third_arm.md` §3, not from the driver.

    "the largest power of two W <= 16 for which bound // W >= 32; 1 if none qualifies."
    """
    best = 1
    for w in (1, 2, 4, 8, 16):
        if bound // w >= 32:
            best = w
    return best


@pytest.mark.parametrize("past,expected", [
    (0, 1), (32, 1), (64, 2), (128, 4), (256, 8), (512, 16), (1024, 16),
])
def test_kv_parallel_matches_independent_recomputation(past, expected):
    """The predeclared W table, the driver, and a third derivation must all agree."""
    bound = past + 1
    assert pr97.kv_parallel(bound) == expected
    assert _independent_kv_parallel(bound) == expected
    w = pr97.expected_witness(pr97.PR97_COMMIT, "phi-3.5-mini-instruct-cuda-int4-rtn-block-32",
                              "decode", 1, past)
    assert w == f"gqa_decode_f16:{expected}"


def test_kv_parallel_never_exceeds_the_shader_maximum():
    for bound in range(1, 20000):
        assert 1 <= pr97.kv_parallel(bound) <= 16
        assert pr97.kv_parallel(bound) & (pr97.kv_parallel(bound) - 1) == 0


def test_kv_parallel_is_monotone_in_bound():
    prev = 1
    for bound in range(1, 20000):
        w = pr97.kv_parallel(bound)
        assert w >= prev
        prev = w


def test_past32_is_the_serial_internal_control():
    """W == 1 at past=32 is what makes that length a control *by PR #97's own construction*."""
    assert pr97.kv_parallel(33) == 1


# --------------------------------------------------------------------------------------------
# The widened witness extractor.
# --------------------------------------------------------------------------------------------

def _counters(*variants):
    return {"pipeline_variants": list(variants), "dispatches_executed": 1, "compute_calls": 1,
            "compute_failures": 0, "claimed_nodes": 1, "running_device_names": ["dev"],
            "shaders_dispatched": len(variants)}


def test_widened_extractor_sees_the_renamed_kernel():
    out = pr97.gqa_witness_multi(_counters("gqa_decode_f16:4", "q_gemv_matmul_nbits_f16:64"))
    assert out["gqa_keys"] == ["gqa_decode_f16:4"]
    assert out["kv_parallel"] == 4


def test_widened_extractor_still_sees_the_original_kernel():
    out = pr97.gqa_witness_multi(_counters("gqa_f16:1", "q_gemv_matmul_nbits_f16:64"))
    assert out["gqa_keys"] == ["gqa_f16:1"]
    assert out["kv_parallel"] is None


def test_frozen_extractor_is_blind_to_the_renamed_kernel():
    """Documents *why* the driver exists. If this ever fails, the frozen file was edited."""
    out = base.gqa_witness(_counters("gqa_decode_f16:4"))
    assert out["gqa_keys"] == []


def test_widening_does_not_recurse_when_installed():
    """Regression: the widened extractor delegated through the name it had itself replaced.

    `WidenedWitness` rebinds `base.gqa_witness`, so a late-bound delegate called itself until the
    stack ran out. It failed loudly rather than silently, but it failed *after* acquiring the GPU
    lock, which is the expensive place to discover it.
    """
    with pr97.WidenedWitness():
        assert base.gqa_witness is pr97.gqa_witness_multi
        out = base.gqa_witness(_counters("gqa_decode_f16:8"))
    assert out["gqa_keys"] == ["gqa_decode_f16:8"]
    assert out["kv_parallel"] == 8


def test_widened_witness_restores_the_frozen_extractor():
    saved = base.gqa_witness
    with pr97.WidenedWitness():
        pass
    assert base.gqa_witness is saved
    assert base.gqa_witness(_counters("gqa_decode_f16:4"))["gqa_keys"] == []


def test_widened_witness_refuses_to_nest():
    with pr97.WidenedWitness():
        with pytest.raises(RuntimeError):
            with pr97.WidenedWitness():
                pass
    assert base.gqa_witness is not pr97.gqa_witness_multi


# --------------------------------------------------------------------------------------------
# What the widened extractor must still refuse. Widening feeds the gate; these are the leaks.
# --------------------------------------------------------------------------------------------

def _gate(variants, expected_key):
    rec = {"admissible": True, "phase": "decode", "workload": "w", "arm": "candidate",
           "speed": {"median_ms": 1.0}}
    rec["path_witness"] = pr97.gqa_witness_multi(_counters(*variants))
    ok = rec["path_witness"]["gqa_keys"] == [expected_key]
    return ok


def test_gate_refuses_the_wrong_kv_parallel_factor():
    assert not _gate(["gqa_decode_f16:2"], "gqa_decode_f16:4")


def test_gate_refuses_when_both_gqa_kernels_ran():
    """A build that dispatched both would otherwise look like a valid arm."""
    out = pr97.gqa_witness_multi(_counters("gqa_f16:1", "gqa_decode_f16:4"))
    assert out["gqa_keys"] == ["gqa_f16:1", "gqa_decode_f16:4"]
    assert out["kv_parallel"] is None
    assert not _gate(["gqa_f16:1", "gqa_decode_f16:4"], "gqa_decode_f16:4")


def test_gate_refuses_when_no_gqa_kernel_ran():
    assert not _gate(["q_gemv_matmul_nbits_f16:64"], "gqa_decode_f16:4")


def test_gate_refuses_the_old_kernel_on_the_new_arm():
    """The PR #97 DLL failing to take its own path must not pass as the PR #97 arm."""
    assert not _gate(["gqa_f16:1"], "gqa_decode_f16:4")


def test_widened_extractor_on_absent_counters_is_not_present():
    out = pr97.gqa_witness_multi(None)
    assert out["present"] is False
    assert out["gqa_keys"] == []


# --------------------------------------------------------------------------------------------
# Kernel attribution across a rename.
# --------------------------------------------------------------------------------------------

def _shares(gqa_c, gqa_b, ctrl_c, ctrl_b):
    return pr97.kernel_share({"gqa_decode_f16": gqa_c, "q_gemv_matmul_nbits_f16": ctrl_c},
                             {"gqa_f16": gqa_b, "q_gemv_matmul_nbits_f16": ctrl_b})


def test_kernel_share_reports_both_ratio_conventions_as_reciprocals():
    """Regression: the band test compared a baseline/candidate ratio to a candidate/baseline spread.

    It gave the right verdict on a 1.7x effect and would have given the wrong one on a small
    effect, which is the case where a verdict actually matters.
    """
    k = _shares(20.0, 40.0, 10.0, 10.0)
    assert k["us_ratio_same_workload"] == pytest.approx(2.0)
    assert k["gqa_ratio_cand_over_base"] == pytest.approx(0.5)
    assert k["us_ratio_same_workload"] == pytest.approx(1.0 / k["gqa_ratio_cand_over_base"])


def test_kernel_moved_is_judged_in_the_control_convention():
    """A kernel that moved exactly like the untouched controls must NOT read as moved."""
    k = _shares(10.5, 10.0, 10.5, 10.0)
    assert k["gqa_ratio_cand_over_base"] == pytest.approx(1.05)
    assert pr97._kernel_moved_beyond_controls(k) is False


def test_kernel_moved_detects_a_real_localised_speedup():
    k = _shares(23489.85, 40004.67, 10617.0, 10000.0)
    assert pr97._kernel_moved_beyond_controls(k) is True


def test_a_kernel_inside_the_control_spread_is_not_moved_either_way():
    """cand/base = 1.055 is inside [1.05, 1.06]; its reciprocal 0.948 is not.

    Under the old mixed comparison this row reported KERNEL-ONLY for a kernel that moved exactly
    as much as kernels the change cannot touch. This is the guard that fails on the old code.
    """
    k = pr97.kernel_share(
        {"gqa_decode_f16": 10.55, "q_gemv_matmul_nbits_f16": 10.5, "gather_f16": 10.6},
        {"gqa_f16": 10.0, "q_gemv_matmul_nbits_f16": 10.0, "gather_f16": 10.0})
    assert k["untouched_ratio_min"] == pytest.approx(1.05)
    assert k["untouched_ratio_max"] == pytest.approx(1.06)
    assert pr97._kernel_moved_beyond_controls(k) is False


def test_kernel_share_needs_both_arms():
    assert pr97.kernel_share({"gqa_decode_f16": 5.0}, {"q_gemv_matmul_nbits_f16": 5.0})["baseline"] is None
    assert pr97._kernel_moved_beyond_controls(
        pr97.kernel_share({"gqa_decode_f16": 5.0}, {"q_gemv_matmul_nbits_f16": 5.0})) is None


def test_verdict_table_cannot_award_resolves_p128_regression():
    """Q0 was declared unreachable before timing, because the sweep found no p128 regression."""
    derived = {"workloads": [
        {"workload": "w", "role": "treatment", "past": 128, "verdict": "FASTER",
         "ratio_median": 1.9},
    ]}
    rows = pr97.verdict_rows(derived, None)
    assert rows[0]["addendum_verdict"] == "WHOLE-MODEL-FASTER"
    assert all("RESOLVES-P128-REGRESSION" != r["addendum_verdict"] for r in rows)


def test_refused_row_never_carries_a_ratio():
    derived = {"workloads": [
        {"workload": "w", "role": "treatment", "past": 256, "verdict": "REFUSED",
         "ratio_median": None},
    ]}
    row = pr97.verdict_rows(derived, None)[0]
    assert row["addendum_verdict"] == "INCOMPARABLE"
    assert row["ratio_median"] is None


# --------------------------------------------------------------------------------------------
# The kill-switch bit-identity pass.
# --------------------------------------------------------------------------------------------

ks = pytest.importorskip("probe_pr97_killswitch")


def _ksrec(past, config, digest, admissible=True, eq="MATCH", witness=None):
    return {"past": past, "config": config, "outputs_sha256": digest,
            "admissible": admissible, "equivalence": {"verdict": eq},
            "measured_gqa_witness": witness or ["gqa_decode_f16:1"],
            "refusal": None if admissible else {"reason": "equivalence not MATCH"}}


def test_killswitch_reports_bit_identical_when_digests_agree():
    rows = ks.compare([_ksrec(512, "default", "aa"), _ksrec(512, "forced1", "aa")])
    assert rows[0]["verdict"] == "BIT-IDENTICAL"


def test_killswitch_reports_bits_differ_and_keeps_the_equivalence_verdict():
    """"Different bits" and "different answer" are separate findings; only the second is a bug."""
    rows = ks.compare([_ksrec(512, "default", "aa"), _ksrec(512, "forced1", "bb")])
    assert rows[0]["verdict"] == "BITS-DIFFER"
    assert rows[0]["equivalence"] == {"default": "MATCH", "forced1": "MATCH"}


def test_killswitch_refuses_a_claim_when_one_side_is_inadmissible():
    rows = ks.compare([_ksrec(64, "default", "aa"),
                       _ksrec(64, "forced1", None, admissible=False)])
    assert rows[0]["verdict"] == "UNTESTED"
    assert "digests" not in rows[0]


def test_killswitch_refuses_a_claim_when_a_configuration_is_unstable():
    """A configuration that cannot reproduce its own digest cannot be compared to another one."""
    rows = ks.compare([_ksrec(512, "default", "aa"), _ksrec(512, "default", "cc"),
                       _ksrec(512, "forced1", "aa"), _ksrec(512, "forced1", "aa")])
    assert rows[0]["verdict"] == "NON-DETERMINISTIC"


def test_killswitch_marks_past32_as_the_positive_control():
    rows = ks.compare([_ksrec(32, "default", "aa"), _ksrec(32, "forced1", "aa")])
    assert rows[0]["is_positive_control"] is True
    assert rows[0]["host_rule_W"] == 1


def test_forced_kv_parallel_sets_exactly_one_variable_after_the_scrub():
    """The override must be applied *after* the frozen scrub, or a shell could supply it."""
    saved = base.child_env
    seen = {}

    def fake(args, arm, counters_path, trace_path):
        return {"PATH": "x"}

    base.child_env = fake
    try:
        with ks.ForcedKvParallel("1"):
            seen = base.child_env(None, "candidate", None, None)
    finally:
        base.child_env = saved
    assert seen[ks.KILL_SWITCH] == "1"
    assert base.child_env is saved


def test_forced_kv_parallel_none_sets_nothing():
    saved = base.child_env
    base.child_env = lambda *a: {"PATH": "x"}
    try:
        with ks.ForcedKvParallel(None):
            seen = base.child_env(None, "candidate", None, None)
    finally:
        base.child_env = saved
    assert ks.KILL_SWITCH not in seen


# --------------------------------------------------------------------------------------------
# Static SPIR-V sizing of the new module.
# --------------------------------------------------------------------------------------------

spv = pytest.importorskip("spirv_gqa_decode_pr97")

_DIS = """
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%c16 = OpConstant %uint 16
%c128 = OpConstant %uint 128
%arr128 = OpTypeArray %float %c128
%arr16x128 = OpTypeArray %arr128 %c16
%ptr_wg = OpTypePointer Workgroup %arr16x128
%ptr_fn = OpTypePointer Function %arr128
%acc = OpVariable %ptr_wg Workgroup
%tmp = OpVariable %ptr_fn Function
"""


def test_storage_bytes_sizes_a_nested_array():
    wg = spv.storage_bytes(_DIS, "Workgroup")
    assert wg["total_bytes"] == 16 * 128 * 4
    assert wg["variable_count"] == 1


def test_storage_bytes_separates_storage_classes():
    assert spv.storage_bytes(_DIS, "Function")["total_bytes"] == 128 * 4


def test_glslc_flags_match_the_shipped_build():
    """These are read off rust/build.rs; a static analysis compiled differently proves nothing."""
    build_rs = (pathlib.Path(__file__).resolve().parents[1] / "rust" / "build.rs")
    if not build_rs.is_file():
        pytest.skip("rust/build.rs not present")
    text = build_rs.read_text(encoding="utf-8", errors="replace")
    for flag in spv.GLSLC_FLAGS:
        assert flag in text, f"{flag} is not what rust/build.rs passes to glslc"


def test_w_values_cover_every_factor_the_host_rule_can_pick():
    reachable = {pr97.kv_parallel(b) for b in range(1, 20000)}
    assert reachable <= set(spv.W_VALUES)


# --------------------------------------------------------------------------------------------
# A screen over the committed artifacts. Skipped when they are absent.
# --------------------------------------------------------------------------------------------

def _artifact(name):
    p = RESULTS / name
    if not p.is_file():
        pytest.skip(f"{name} not present")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["pr97_vs_candidate.json", "pr97_vs_baseline.json"])
def test_published_sweep_resummarizes_exactly(name):
    """The published summary must be a function of the published records and nothing else."""
    art = _artifact(name)
    fresh = pr97.resummarize(art)
    for k, v in fresh.items():
        assert json.dumps(art.get(k), sort_keys=True) == json.dumps(v, sort_keys=True), k


@pytest.mark.parametrize("name", ["pr97_vs_candidate.json", "pr97_vs_baseline.json"])
def test_published_sweep_has_no_speed_on_a_refused_record(name):
    for r in _artifact(name)["records"]:
        if not r.get("admissible"):
            assert "speed" not in r or not r["speed"], r.get("workload")


@pytest.mark.parametrize("name", ["pr97_vs_candidate.json", "pr97_vs_baseline.json"])
def test_published_sweep_witnesses_all_agree(name):
    art = _artifact(name)
    assert art["witness_audit"]["all_agree"] is True
    assert art["witness_audit"]["disagreements"] == []


@pytest.mark.parametrize("name", ["pr97_vs_candidate.json", "pr97_vs_baseline.json"])
def test_published_sweep_arms_are_distinct_libraries(name):
    arms = _artifact(name)["arms"]
    assert arms["baseline"]["sha256"] != arms["candidate"]["sha256"]


@pytest.mark.parametrize("name", ["pr97_vs_candidate.json", "pr97_vs_baseline.json"])
def test_published_sweep_never_claims_to_resolve_a_regression(name):
    """Q0 was declared unreachable before timing; no artifact may award it afterwards."""
    art = _artifact(name)
    assert "RESOLVES-P128-REGRESSION" not in json.dumps(art["addendum_verdicts"])


def test_published_attribution_resummarizes_exactly():
    art = _artifact("pr97_attribution.json")
    assert json.dumps(art["per_length"], sort_keys=True) == json.dumps(
        pr97.resummarize(art)["per_length"], sort_keys=True)


def test_published_attribution_publishes_no_wall_clock():
    art = _artifact("pr97_attribution.json")
    assert "caveat" in art
    for r in art["records"]:
        assert "speed" not in r or r.get("admissible")


def test_published_attribution_compares_ratios_in_one_convention():
    """The defect this guard exists for: a b/c ratio tested against a c/b spread."""
    for past, k in _artifact("pr97_attribution.json")["per_length"].items():
        if k.get("refused"):
            continue
        assert k["gqa_ratio_cand_over_base"] == pytest.approx(
            1.0 / k["us_ratio_same_workload"], rel=1e-3)
        assert "untouched_ratio_convention" in k


def test_published_killswitch_never_reports_speed():
    art = _artifact("pr97_killswitch.json")
    assert "no_latency_published" in art
    for r in art["records"]:
        assert "speed" not in r


def test_published_killswitch_kill_switch_demonstrably_took_effect():
    """A BITS-DIFFER verdict is only meaningful if the override actually changed W."""
    for row in _artifact("pr97_killswitch.json")["rows"]:
        if row["verdict"] in ("BIT-IDENTICAL", "BITS-DIFFER"):
            assert row["witnesses"]["forced1"] == ["gqa_decode_f16:1"]


def test_published_spirv_states_its_static_limit():
    art = _artifact("spirv_gqa_decode_pr97.json")
    joined = " ".join(art["limits"]).lower()
    assert "machine code" in joined
    assert art["decode_unspecialised"]["spirv_val"] == "PASS"


def test_no_artifact_claims_cuda_measurement():
    """The model filename contains 'cuda'; no artifact may state a CUDA measurement."""
    for name in ("pr97_vs_candidate.json", "pr97_vs_baseline.json", "pr97_attribution.json",
                 "pr97_killswitch.json", "spirv_gqa_decode_pr97.json"):
        p = RESULTS / name
        if not p.is_file():
            continue
        for m in re.finditer(r"[^\"]*cuda[^\"]*", p.read_text(encoding="utf-8"), re.I):
            assert "int4-rtn-block-32" in m.group(0), f"{name}: {m.group(0)[:120]}"


# --------------------------------------------------------------------------------------------
# Corrections applied after PR #99 was rejected. Each guard below locks a defect that review
# found in the Phase 1 evidence and that this third arm had inherited verbatim.
# --------------------------------------------------------------------------------------------


def _sweeps():
    for name in ("pr97_vs_candidate.json", "pr97_vs_baseline.json"):
        p = RESULTS / name
        if p.is_file():
            yield name, json.loads(p.read_text(encoding="utf-8"))


def test_frozen_window_claim_is_fail_open_over_refused_lengths():
    """The defect itself: `lengths_measured` counts lengths that produced no speed figure."""
    for name, doc in _sweeps():
        declared = set(doc["window"]["lengths_measured"])
        entitled = set(doc["honest_window"]["lengths_with_admissible_records"])
        assert entitled < declared, f"{name}: expected the frozen claim to overstate coverage"
        assert doc["window"]["claim"] == "NO-SLOW-LENGTH"


def test_honest_window_excludes_every_zero_admissible_length():
    for name, doc in _sweeps():
        hw = doc["honest_window"]
        for length in hw["lengths_with_zero_admissible_records"]:
            assert length not in hw["corrected_claim_covers"], f"{name}: past={length}"
            assert hw["records_per_length"][str(length)]["admissible"] == 0


def test_honest_window_covers_exactly_the_lengths_with_data():
    for name, doc in _sweeps():
        hw = doc["honest_window"]
        recomputed = sorted(
            p for p in {r["past"] for r in doc["records"] if r.get("past") is not None}
            if any(r.get("past") == p and r.get("admissible") is True for r in doc["records"])
        )
        assert hw["corrected_claim_covers"] == recomputed, name
        assert hw["lengths_with_zero_admissible_records"] == [32, 64, 256], name


def test_a_sweep_that_measured_nothing_cannot_claim_no_slow_length():
    """The reductio: strip every admissible record and the corrected claim must collapse."""
    probe = pytest.importorskip("probe_pr97_third_arm")
    records = [{"past": 128, "admissible": False}, {"past": 512, "admissible": False}]
    hw = probe.honest_window(records, {"claim": "NO-SLOW-LENGTH",
                                       "lengths_measured": [128, 512]})
    assert hw["corrected_claim"] == "NO-LENGTH-MEASURED"
    assert hw["corrected_claim_covers"] == []


def test_p128_confidence_interval_does_not_exclude_parity_in_either_pairing():
    """The honest limit of the headline: past=128 is not established at 95% with n=3."""
    for name, doc in _sweeps():
        row = next(r for r in doc["ratio_ci95"] if r["past"] == 128)
        assert row["n_paired"] == 3, name
        assert row["ci95_low"] < 1.0 < row["ci95_high"], name
        assert row["ci95_excludes_parity"] is False, name


def test_long_context_speedup_excludes_parity_in_both_pairings():
    """What survives review: past=512 and past=1024 are interval-supported, not band-supported."""
    for name, doc in _sweeps():
        for length in (512, 1024):
            row = next(r for r in doc["ratio_ci95"] if r["past"] == length)
            assert row["ci95_excludes_parity"] is True, f"{name}: past={length}"
            assert row["ci95_low"] > 1.0, f"{name}: past={length}"


def test_confidence_intervals_recompute_from_the_published_records():
    import math
    import statistics
    probe = pytest.importorskip("probe_pr97_third_arm")
    for name, doc in _sweeps():
        fresh = probe.ratio_ci95({"workloads": doc["workloads"]}, doc["records"])
        assert json.dumps(fresh, sort_keys=True) == json.dumps(doc["ratio_ci95"], sort_keys=True), name
        row = next(r for r in fresh if r["past"] == 512)
        assert math.isfinite(row["geomean_ratio"]) and row["n_paired"] == 3
        assert statistics.fmean([row["ci95_low"], row["ci95_high"]]) > 1.0


def test_band_sensitivity_is_published_and_no_verdict_flips_at_the_floor():
    for name, doc in _sweeps():
        bs = doc["band_sensitivity"]
        assert bs["band_floor"] == 0.05
        assert bs["band_applied"] > bs["band_floor"], name
        assert bs["any_flip"] is False and bs["verdicts_that_flip"] == [], name


def test_indeterminate_rows_are_never_described_as_inside_the_band():
    """`INDETERMINATE` means the median is OUTSIDE the band; the old `why` said the opposite."""
    for name, doc in _sweeps():
        for row in doc["addendum_verdicts"]:
            if row["sweep_verdict"] != "INDETERMINATE":
                continue
            assert row["addendum_verdict"] == "INDETERMINATE", name
            assert "inside the band" not in row.get("why", ""), f"{name}: {row['workload']}"
            assert "OUTSIDE the band" in row.get("why", ""), f"{name}: {row['workload']}"


def test_indeterminate_is_not_laundered_into_neutral():
    for name, doc in _sweeps():
        for row in doc["addendum_verdicts"]:
            if row["sweep_verdict"] == "INDETERMINATE":
                assert row["addendum_verdict"] != "NEUTRAL", f"{name}: {row['workload']}"


def test_arm1_to_arm3_p128_is_indeterminate_not_faster():
    doc = json.loads((RESULTS / "pr97_vs_baseline.json").read_text(encoding="utf-8"))
    row = next(r for r in doc["addendum_verdicts"] if r["past"] == 128)
    assert row["sweep_verdict"] == "INDETERMINATE"
    assert row["addendum_verdict"] == "INDETERMINATE"


def test_samples_binding_covers_every_timed_sample():
    for name, doc in _sweeps():
        sb = doc["samples_binding"]
        counted = sum(len((r.get("speed") or {}).get("samples_ms") or []) for r in doc["records"])
        assert sb["samples"] == counted > 0, name
        assert sb["records"] == len(doc["records"]), name


def test_samples_binding_changes_when_any_sample_changes():
    """F4: the check read only `median_ms`, so scaled or deleted samples still reproduced."""
    probe = pytest.importorskip("probe_pr97_third_arm")
    base_records = [{"workload": "w", "arm": "a", "repeat": 0,
                     "speed": {"samples_ms": [1.0, 2.0, 3.0]}}]
    ref = probe.samples_binding(base_records)["sha256"]

    scaled = [{"workload": "w", "arm": "a", "repeat": 0,
               "speed": {"samples_ms": [10.0, 20.0, 30.0]}}]
    dropped = [{"workload": "w", "arm": "a", "repeat": 0,
                "speed": {"samples_ms": [1.0, 2.0]}}]
    absent = [{"workload": "w", "arm": "a", "repeat": 0, "speed": {}}]
    nudged = [{"workload": "w", "arm": "a", "repeat": 0,
               "speed": {"samples_ms": [1.0 + 1e-12, 2.0, 3.0]}}]

    for tag, recs in (("scaled", scaled), ("dropped", dropped),
                      ("absent", absent), ("nudged", nudged)):
        assert probe.samples_binding(recs)["sha256"] != ref, tag


def test_resummarize_republishes_the_samples_binding():
    for name, doc in _sweeps():
        probe = pytest.importorskip("probe_pr97_third_arm")
        fresh = probe.resummarize(doc)
        assert fresh["samples_binding"] == doc["samples_binding"], name


def test_attribution_artifact_is_also_sample_bound():
    p = RESULTS / "pr97_attribution.json"
    if not p.is_file():
        pytest.skip("attribution artifact absent")
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["samples_binding"]["records"] == len(doc["records"])


def test_pr97_compiled_delta_is_not_limited_to_shader_and_attention():
    """F3: `evidence/proof_ledger.jsonl` is `include_str!`d, so it is part of the compiled delta."""
    root = pathlib.Path(__file__).resolve().parents[1]
    registry = root / "rust" / "src" / "registry.rs"
    if not registry.is_file():
        pytest.skip("registry.rs absent")
    assert 'include_str!("../../evidence/proof_ledger.jsonl")' in registry.read_text(
        encoding="utf-8"), "the ledger is no longer compiled in; revisit the delta claim"
    for name, _ in _sweeps():
        blob = (RESULTS / name).read_text(encoding="utf-8")
        for bad in ("compiled delta is gqa_f16.comp and attention.rs only",
                    "the only files that reach the binary"):
            assert bad not in blob, f"{name}: {bad}"


def test_docs_do_not_restate_the_overstated_length_coverage():
    perf = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "PERF.md")
    body = perf.read_text(encoding="utf-8")
    section = body[body.index("## 29."):] if "## 29." in body else body
    for bad in ("no measured KV length is SLOWER at 0, 32, 64, 128, 256, 512, 1024",
                "all seven lengths", "every declared length was measured"):
        assert bad not in section, bad


def test_guard_count_in_docs_matches_this_file():
    """F5: the published guard count must equal the number of `def test_` here."""
    here = pathlib.Path(__file__).read_text(encoding="utf-8")
    n = len(re.findall(r"^def test_", here, re.M))
    perf = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "PERF.md")
    body = perf.read_text(encoding="utf-8")
    section = body[body.index("## 29."):] if "## 29." in body else body
    claims = {int(m) for m in re.findall(r"(\d+)\s+GPU-free guards", section)}
    assert claims, "PERF section 29 must state a guard count"
    assert claims == {n}, f"docs claim {claims}, file defines {n}"
