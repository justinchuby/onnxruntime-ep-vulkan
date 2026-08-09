"""Locks the analysis half of `bench/results/probe_crossbuild_decode_window.py`, GPU-free.

Issue #96 asks whether the compiled `85fbda2` library is slower than the compiled `c96e7d9`
library at Phi-3.5 decode. The probe answers it by timing, and a timing instrument's most
dangerous failure is not a wrong number — it is a number that *should never have been published*
surviving into the artifact. So the surface this file drives is the gate, not the clock:

  * `admissibility_gate` re-derives admissibility from a **written record**, so every mutation
    below is a record a compromised or buggy worker could plausibly have written. Each one must
    refuse, and — separately asserted, because it is the part that would leak — must carry **no
    `speed` field** afterwards.
  * `paired` must refuse a workload whose two arms the production path cannot tell apart. A run
    that pointed both arms at the same DLL, or that let `ONNXRUNTIME_EP_VULKAN_GQA_LOCAL_SIZE`
    into the children, would produce exactly that and would otherwise report a beautiful 1.00x.
  * `verdict_for` must not call a direction on a spread that includes the band, and
    `window_claim` must not interpolate a *window* out of one slow length.

Nothing here opens a device, loads a model, or reads a DLL. It is safe on a machine with no GPU
and it is safe to run while the GPU lock is held by somebody else, which is the point: the guards
have to be runnable by a reviewer who cannot reproduce the measurement.

The last section is a screen over the committed artifacts, skipped when they are absent.
"""
from __future__ import annotations

import copy
import json
import pathlib
import re
import sys

import pytest

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(RESULTS))

probe = pytest.importorskip("probe_crossbuild_decode_window")


# --------------------------------------------------------------------------------- fixtures
def _witness(arm: str, *, gqa: bool = True) -> dict:
    keys = [probe.EXPECTED_GQA_KEY[arm]] if gqa else []
    return {
        "present": True,
        "gqa_keys": keys,
        "all_variants": keys + ["q_gemv_matmul_nbits_f16:"],
        "dispatches_executed": 2130,
        "compute_calls": 24,
        "compute_failures": 0,
        "claimed_nodes": 1,
        "running_device_names": ["NVIDIA RTX A1000 Laptop GPU"],
    }


def _record(arm: str = "candidate", *, repeat: int = 0, median_ms: float = 137.0,
            model_key: str | None = None, gqa: bool = True) -> dict:
    """A record that must pass the gate. Every mutation test starts from a copy of this."""
    key = model_key if model_key is not None else probe.rm.PHI35.key
    return {
        "workload": probe.workload_label(key, "decode", 1, 128),
        "model_key": key,
        "arm": arm,
        "repeat": repeat,
        "admissible": True,
        "refusal": None,
        "ep_library_sha256": "aa" if arm == "candidate" else "bb",
        "ep_library_bytes": 2_563_072 if arm == "candidate" else 2_560_000,
        "outputs_sha256": "deadbeef",
        "outputs_sha256_post_timing": "deadbeef",
        "equivalence": {"verdict": probe.rm.MATCH, "primary": {"verdict": probe.rm.MATCH}},
        "model": {"key": key, "sha256": "cafe", "recorded_sha256": "cafe",
                  "agrees_with_recorded_provenance": True},
        "path_witness": _witness(arm, gqa=gqa),
        "inference_calls": 27,
        "speed": {"median_ms": median_ms, "rsd": 0.7, "samples_ms": [median_ms] * 20},
    }


def _gated(rec: dict, **kw) -> dict:
    return probe.admissibility_gate(copy.deepcopy(rec), **kw)


def _assert_refused(out: dict, fragment: str) -> None:
    assert out["admissible"] is False
    assert fragment in out["refusal"]["reason"], out["refusal"]
    # The whole point of the refusal: a refused record may not carry a speed field.
    assert "speed" not in out, "a refused record retained its timing"


# ------------------------------------------------------------------- the gate lets good in
def test_a_clean_record_is_admissible_on_both_arms():
    for arm in ("candidate", "baseline"):
        out = _gated(_record(arm))
        assert out["admissible"] is True, out.get("refusal")
        assert out["speed"]["median_ms"] == 137.0


def test_the_gate_is_pure_and_does_not_mutate_its_caller_s_dict():
    rec = _record("candidate")
    rec["equivalence"] = None
    before = json.dumps(rec, sort_keys=True)
    probe.admissibility_gate(copy.deepcopy(rec))
    assert json.dumps(rec, sort_keys=True) == before


# ------------------------------------------------------------- equivalence mutations (#96 §2)
def test_missing_equivalence_verdict_refuses_and_drops_the_timing():
    rec = _record()
    del rec["equivalence"]
    _assert_refused(_gated(rec), "no output equivalence verdict")


def test_empty_equivalence_dict_refuses():
    rec = _record()
    rec["equivalence"] = {}
    _assert_refused(_gated(rec), "no output equivalence verdict")


def test_equivalence_present_but_verdictless_refuses():
    rec = _record()
    rec["equivalence"] = {"primary": {"max_abs": 0.0}}
    _assert_refused(_gated(rec), "no output equivalence verdict")


def test_divergent_equivalence_refuses_and_names_the_verdict():
    rec = _record()
    rec["equivalence"] = {"verdict": probe.rm.DIVERGENT, "secondary_divergent": 3}
    out = _gated(rec)
    _assert_refused(out, "equivalence not MATCH")
    assert out["refusal"]["verdict"] == probe.rm.DIVERGENT


def test_a_record_already_marked_inadmissible_still_loses_its_speed():
    """The exact leak: a worker that bailed but wrote a `speed` block anyway."""
    rec = _record()
    rec["admissible"] = False
    rec["refusal"] = {"reason": "equivalence not MATCH"}
    out = _gated(rec)
    assert "speed" not in out
    assert out["refusal"]["reason"] == "equivalence not MATCH"


# ------------------------------------------------------------------------- digest mutations
def test_missing_outputs_digest_refuses():
    rec = _record()
    rec["outputs_sha256"] = ""
    _assert_refused(_gated(rec), "missing or empty outputs digest")


def test_outputs_that_drifted_across_the_timed_pass_refuse():
    rec = _record()
    rec["outputs_sha256_post_timing"] = "0ther"
    _assert_refused(_gated(rec), "outputs digest changed across the timed pass")


def test_absent_post_timing_digest_refuses_rather_than_passing_by_omission():
    rec = _record()
    del rec["outputs_sha256_post_timing"]
    _assert_refused(_gated(rec), "outputs digest changed across the timed pass")


# ---------------------------------------------------------------------- provenance mutations
def test_model_digest_that_disagrees_with_the_pin_refuses():
    rec = _record()
    rec["model"]["sha256"] = "not-the-pinned-graph"
    out = _gated(rec)
    _assert_refused(out, "model digest does not match the recorded provenance")
    assert out["refusal"]["recorded"] == "cafe"


def test_missing_model_digest_refuses():
    rec = _record()
    rec["model"]["sha256"] = None
    _assert_refused(_gated(rec), "no model digest recorded")


def test_resolver_reported_provenance_disagreement_refuses():
    rec = _record()
    rec["model"]["agrees_with_recorded_provenance"] = False
    _assert_refused(_gated(rec), "model provenance disagreement")


# ------------------------------------------------------------------ production-path witness
def test_absent_counters_file_refuses_because_nothing_witnesses_the_ep():
    rec = _record()
    rec["path_witness"] = {"present": False}
    _assert_refused(_gated(rec), "no path witness")


def test_zero_dispatches_refuses_even_with_a_counters_file():
    rec = _record()
    rec["path_witness"]["dispatches_executed"] = 0
    _assert_refused(_gated(rec), "no dispatch executed on the device")


def test_any_compute_failure_refuses():
    rec = _record()
    rec["path_witness"]["compute_failures"] = 1
    _assert_refused(_gated(rec), "compute_failures > 0")


@pytest.mark.parametrize("arm,wrong", [("candidate", "gqa_f16:"), ("baseline", "gqa_f16:1")])
def test_an_arm_wearing_the_other_arm_s_witness_key_refuses(arm, wrong):
    """The strongest single check in the suite.

    `pipeline_variants` is written by the EP from the *resolved* specialisation vector at
    `vkCreateComputePipelines`. If the candidate reports `gqa_f16:` it did not run the candidate
    module; if the baseline reports `gqa_f16:1` the driver's environment leaked a spec constant
    into it. Either way the two arms are not the two builds and no ratio means anything.
    """
    rec = _record(arm)
    rec["path_witness"]["gqa_keys"] = [wrong]
    out = _gated(rec)
    _assert_refused(out, "gqa_f16 witness key is not the one this arm must produce")
    assert out["refusal"]["declared"] == probe.EXPECTED_GQA_KEY[arm]


def test_a_gqa_workload_with_no_gqa_pipeline_at_all_refuses():
    rec = _record("candidate")
    rec["path_witness"]["gqa_keys"] = []
    _assert_refused(_gated(rec), "gqa_f16 witness key is not the one")


def test_two_gqa_variants_in_one_process_refuse():
    """A second variant means the size changed mid-run; the record describes two experiments."""
    rec = _record("candidate")
    rec["path_witness"]["gqa_keys"] = ["gqa_f16:1", "gqa_f16:16"]
    _assert_refused(_gated(rec), "gqa_f16 witness key is not the one")


def test_a_no_gqa_control_that_built_a_gqa_pipeline_refuses():
    """MobileNetV2 has no `GroupQueryAttention` node. If it built one, it is not the control."""
    rec = _record("candidate", model_key=probe.rm.MOBILENETV2.key, gqa=False)
    rec["path_witness"]["gqa_keys"] = ["gqa_f16:1"]
    _assert_refused(_gated(rec), "a no-GQA control built a gqa_f16 pipeline")


def test_the_no_gqa_control_passes_when_it_builds_no_gqa_pipeline():
    rec = _record("candidate", model_key=probe.rm.MOBILENETV2.key, gqa=False)
    assert _gated(rec)["admissible"] is True


# ------------------------------------------------------------------------- library identity
def test_a_library_digest_that_is_not_this_arm_s_declared_build_refuses():
    rec = _record("candidate")
    _assert_refused(_gated(rec, expected_lib_sha={"candidate": "zz", "baseline": "bb"}),
                    "EP library digest is not this arm's declared build")


def test_the_declared_library_digest_passes():
    rec = _record("candidate")
    assert _gated(rec, expected_lib_sha={"candidate": "aa", "baseline": "bb"})["admissible"]


# --------------------------------------------------------------------------- witness parsing
def test_gqa_witness_reads_the_resolved_local_size_out_of_the_variant_key():
    w = probe.gqa_witness({"pipeline_variants": ["gqa_f16:1", "softmax_f16:"],
                           "dispatches_executed": 7})
    assert w["present"] and w["gqa_keys"] == ["gqa_f16:1"] and w["local_size"] == 1
    assert w["dispatches_executed"] == 7


def test_gqa_witness_reads_the_baseline_s_empty_spec_vector_as_a_missing_size():
    w = probe.gqa_witness({"pipeline_variants": ["gqa_f16:"]})
    assert w["gqa_keys"] == ["gqa_f16:"] and w["local_size"] is None


def test_gqa_witness_of_no_counters_file_is_not_present():
    assert probe.gqa_witness(None) == {"present": False, "gqa_keys": [],
                                       "local_size": None, "all_variants": []}


def test_gqa_witness_does_not_match_a_different_kernel_by_prefix():
    w = probe.gqa_witness({"pipeline_variants": ["gqa_f16_v2:1"]})
    assert w["gqa_keys"] == []


# -------------------------------------------------------------------------- pairing refusals
def _pair_records(*, cand_ms, base_ms, cand_lib="aa", base_lib="bb",
                  cand_keys=None, base_keys=None, admissible=(True, True)):
    recs = []
    for rep, (cm, bm) in enumerate(zip(cand_ms, base_ms)):
        c = _record("candidate", repeat=rep, median_ms=cm)
        b = _record("baseline", repeat=rep, median_ms=bm)
        c["ep_library_sha256"], b["ep_library_sha256"] = cand_lib, base_lib
        if cand_keys is not None:
            c["path_witness"]["gqa_keys"] = cand_keys
        if base_keys is not None:
            b["path_witness"]["gqa_keys"] = base_keys
        if not admissible[0]:
            c["admissible"] = False
            c["refusal"] = {"reason": "equivalence not MATCH"}
            c.pop("speed")
        if not admissible[1]:
            b["admissible"] = False
            b["refusal"] = {"reason": "equivalence not MATCH"}
            b.pop("speed")
        recs += [c, b]
    return recs, recs[0]["workload"]


def test_pairing_computes_baseline_over_candidate_so_above_one_is_candidate_faster():
    recs, label = _pair_records(cand_ms=[100.0], base_ms=[110.0])
    p = probe.paired(recs, label)
    assert p["per_repeat"][0]["ratio"] == pytest.approx(1.1)
    assert p["repeats_paired"] == 1


def test_both_arms_on_the_same_library_are_refused_as_one_arm():
    recs, label = _pair_records(cand_ms=[100.0] * 3, base_ms=[110.0] * 3,
                                cand_lib="same", base_lib="same")
    p = probe.paired(recs, label)
    assert p["verdict"] == "REFUSED" and p["refused_because"] == "IDENTICAL-LIBRARY"
    assert "ratio_median" not in p


def test_two_arms_with_the_same_witness_key_are_refused_as_one_arm():
    """What an inherited `ONNXRUNTIME_EP_VULKAN_GQA_LOCAL_SIZE=1` would produce."""
    recs, label = _pair_records(cand_ms=[100.0] * 3, base_ms=[110.0] * 3,
                                cand_keys=["gqa_f16:1"], base_keys=["gqa_f16:1"])
    p = probe.paired(recs, label)
    assert p["verdict"] == "REFUSED" and p["refused_because"] == "IDENTICAL-WITNESS"
    assert "ratio_median" not in p


def test_a_refused_arm_is_not_paired_and_leaves_the_repeat_out():
    recs, label = _pair_records(cand_ms=[100.0], base_ms=[110.0], admissible=(False, True))
    p = probe.paired(recs, label)
    assert p["repeats_paired"] == 0 and p["verdict"] == "REFUSED"
    assert p["refusals"] and p["refusals"][0]["arm"] == "candidate"


def test_refusals_are_reported_even_when_other_repeats_paired():
    recs_ok, label = _pair_records(cand_ms=[100.0, 100.0], base_ms=[110.0, 110.0])
    bad = _record("candidate", repeat=2)
    bad["admissible"] = False
    bad["refusal"] = {"reason": "no dispatch executed on the device"}
    bad.pop("speed")
    p = probe.paired(recs_ok + [bad, _record("baseline", repeat=2)], label)
    assert p["repeats_paired"] == 2
    assert [r["arm"] for r in p["refusals"]] == ["candidate"]


# ----------------------------------------------------------------------------- verdict rule
def _p(ratios):
    import statistics
    return {"per_repeat": [{"repeat": i, "ratio": r} for i, r in enumerate(ratios)],
            "repeats_paired": len(ratios), "ratio_median": statistics.median(ratios),
            "ratio_min": min(ratios), "ratio_max": max(ratios)}


def test_faster_requires_every_repeat_outside_the_band():
    assert probe.verdict_for(_p([1.20, 1.18, 1.22]), 0.05, 3) == "FASTER"
    assert probe.verdict_for(_p([1.20, 1.18, 1.04]), 0.05, 3) != "FASTER"


def test_slower_requires_every_repeat_outside_the_band():
    assert probe.verdict_for(_p([0.86, 0.88, 0.84]), 0.05, 3) == "SLOWER"
    assert probe.verdict_for(_p([0.86, 0.88, 0.99]), 0.05, 3) != "SLOWER"


def test_a_14_percent_regression_is_only_called_when_every_repeat_shows_it():
    """#96's claim, expressed as the rule that would have had to fire."""
    assert probe.verdict_for(_p([1 / 1.14, 1 / 1.15, 1 / 1.13]), 0.05, 3) == "SLOWER"


def test_a_tight_spread_around_one_is_neutral():
    assert probe.verdict_for(_p([1.001, 0.998, 1.004]), 0.05, 3) == "NEUTRAL"


def test_a_wide_spread_around_one_is_indeterminate_not_neutral():
    assert probe.verdict_for(_p([1.30, 0.70, 1.00]), 0.05, 3) == "INDETERMINATE"


def test_too_few_paired_repeats_is_refused_however_clean_the_ratios():
    assert probe.verdict_for(_p([1.40, 1.42]), 0.05, 3) == "REFUSED"


def test_an_already_refused_pair_stays_refused():
    assert probe.verdict_for({"verdict": "REFUSED"}, 0.05, 3) == "REFUSED"


def test_a_wider_band_swallows_a_direction_the_floor_would_have_called():
    assert probe.verdict_for(_p([1.08, 1.07, 1.09]), 0.05, 3) == "FASTER"
    assert probe.verdict_for(_p([1.08, 1.07, 1.09]), 0.12, 3) == "NEUTRAL"


# ------------------------------------------------------------------------------ drift band
def test_the_band_is_the_worst_control_half_range_when_that_exceeds_the_floor():
    pairs = [{"role": "control-no-gqa",
              "per_repeat": [{"ratio": 0.90}, {"ratio": 1.10}]},
             {"role": "control-no-gqa", "per_repeat": [{"ratio": 1.00}, {"ratio": 1.02}]},
             {"role": "treatment", "per_repeat": [{"ratio": 5.0}]}]
    d = probe.drift_envelope(pairs)
    assert d["band"] == pytest.approx(0.10) and d["n"] == 4
    assert d["min"] == 0.90 and d["max"] == 1.10


def test_the_floor_holds_when_the_controls_are_quiet():
    pairs = [{"role": "control-no-gqa", "per_repeat": [{"ratio": 1.00}, {"ratio": 1.01}]}]
    assert probe.drift_envelope(pairs)["band"] == probe.BAND_FLOOR


def test_a_treatment_row_can_never_widen_the_band():
    quiet = [{"role": "control-no-gqa", "per_repeat": [{"ratio": 1.00}, {"ratio": 1.01}]}]
    noisy = quiet + [{"role": "treatment", "per_repeat": [{"ratio": 0.5}, {"ratio": 2.0}]}]
    assert probe.drift_envelope(noisy)["band"] == probe.drift_envelope(quiet)["band"]


def test_with_no_control_the_band_falls_back_to_the_floor_and_says_so():
    d = probe.drift_envelope([{"role": "treatment", "per_repeat": [{"ratio": 1.0}]}])
    assert d["n"] == 0 and d["band"] == probe.BAND_FLOOR and "floor only" in d["source"]


# ----------------------------------------------------------------------------- window claim
def _w(entries):
    return [{"role": "treatment", "past": past, "verdict": v} for past, v in entries]


def test_no_slow_length_is_reported_as_such_with_the_lengths_it_looked_at():
    c = probe.window_claim(_w([(32, "NEUTRAL"), (128, "NEUTRAL"), (512, "NEUTRAL")]))
    assert c["claim"] == "NO-SLOW-LENGTH" and c["lengths_measured"] == [32, 128, 512]


def test_one_slow_length_is_a_point_not_a_window():
    c = probe.window_claim(_w([(64, "NEUTRAL"), (128, "SLOWER"), (256, "NEUTRAL")]))
    assert c["claim"] == "SINGLE-SLOW-POINT"
    assert c["not_slow_below"] == 64 and c["not_slow_above"] == 256


def test_contiguous_slow_lengths_with_named_edges_are_a_window():
    c = probe.window_claim(_w([(32, "NEUTRAL"), (64, "SLOWER"), (128, "SLOWER"),
                               (256, "NEUTRAL")]))
    assert c["claim"] == "WINDOW" and c["slow_lengths"] == [64, 128]
    assert c["not_slow_below"] == 32 and c["not_slow_above"] == 256


def test_a_gap_in_the_slow_set_is_not_called_a_window():
    c = probe.window_claim(_w([(32, "SLOWER"), (64, "NEUTRAL"), (128, "SLOWER")]))
    assert c["claim"] == "NON-CONTIGUOUS-SLOW-SET"


def test_a_slow_length_at_the_edge_of_the_sweep_has_no_edge_beyond_it():
    c = probe.window_claim(_w([(512, "SLOWER"), (1024, "NEUTRAL")]))
    assert c["claim"] == "SINGLE-SLOW-POINT" and c["not_slow_below"] is None


def test_controls_never_enter_the_window_claim():
    pairs = _w([(128, "NEUTRAL")]) + [{"role": "control-no-gqa", "past": None,
                                      "verdict": "SLOWER"}]
    assert probe.window_claim(pairs)["claim"] == "NO-SLOW-LENGTH"


# ------------------------------------------------------------------- trace breakdown mutants
def _trace(tmp_path, events):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return p


def _phase(name, dur, nested_in="none"):
    return {"ph": "X", "cat": "ep.phase", "name": f"vulkan.{name}", "dur": dur,
            "args": {"nested_in": nested_in}}


def _gpu(name, dur):
    return {"ph": "X", "cat": "gpu", "name": f"vulkan.gpu.{name}", "dur": dur}


def test_a_sibling_phase_is_not_classified_as_nested_by_the_literal_string_none(tmp_path):
    """`rust/src/trace.rs:830` writes `nested_in: "none"` for a top-level phase.

    A truthiness test on that arg puts `record`, `submit` and `fence_wait` — the entire host
    top level — into the nested bucket and leaves the sibling total empty. The first attribution
    pass of this investigation did exactly that, and this is the mutant that catches it.
    """
    b = probe.trace_breakdown(_trace(tmp_path, [
        _phase("record", 100.0), _phase("fence_wait", 200.0), _phase("submit", 5.0),
        _phase("cmd_upload", 60.0, nested_in="record"),
        _phase("desc_alloc", 10.0, nested_in="record"),
    ]), inference_calls=1)
    assert set(b["host_phase_us_per_inference"]) == {"record", "fence_wait", "submit"}
    assert set(b["host_nested_us_per_inference"]) == {"cmd_upload", "desc_alloc"}
    assert b["host_sibling_total_us_per_inference"] == pytest.approx(305.0)
    assert b["host_nested_parents"]["cmd_upload"] == "record"


def test_a_child_phase_is_never_added_into_the_sibling_total(tmp_path):
    b = probe.trace_breakdown(_trace(tmp_path, [
        _phase("record", 100.0), _phase("upload", 96.0, nested_in="record"),
    ]), inference_calls=1)
    assert b["host_sibling_total_us_per_inference"] == pytest.approx(100.0)


def test_gpu_time_comes_only_from_gpu_category_spans(tmp_path):
    b = probe.trace_breakdown(_trace(tmp_path, [
        _gpu("gqa_f16", 400.0), _phase("fence_wait", 900.0),
    ]), inference_calls=1)
    assert b["gpu_total_us_per_inference"] == pytest.approx(400.0)


def test_kernel_names_are_reduced_to_the_key_the_delta_looks_up(tmp_path):
    b = probe.trace_breakdown(_trace(tmp_path, [_gpu("gqa_f16", 40.0)]), inference_calls=1)
    assert list(b["gpu_us_per_inference"]) == ["gqa_f16"]
    assert probe.kernel_key("vulkan.gpu.q_gemv_matmul_nbits_f16") == "q_gemv_matmul_nbits_f16"


def test_per_inference_figures_divide_by_the_inference_count(tmp_path):
    b = probe.trace_breakdown(_trace(tmp_path, [_gpu("gqa_f16", 400.0)]), inference_calls=4)
    assert b["gpu_us_per_inference"]["gqa_f16"] == pytest.approx(100.0)


# --------------------------------------------------------------------- attribution deltas
def _arm(gqa, total, *, record=100.0, sibling=305.0, extra=None):
    return {"gpu_total_us": total, "gpu_us": {"gqa_f16": gqa, **(extra or {})},
            "host_phase_us": {"record": record}, "host_nested_us": {},
            "host_sibling_total_us": sibling}


def test_attribution_delta_never_reports_a_false_zero_for_a_present_kernel():
    """The published-artifact bug this file exists to make impossible.

    Summarising under `vulkan.gpu.gqa_f16` and differencing under `gqa_f16` returned `None` on
    both sides, and `x or 0` rendered it as `0` — a kernel that moved by 1.4 ms published as
    'no difference'.
    """
    d = probe.attribution_delta(_arm(41045.22, 64765.81), _arm(39637.30, 62033.22))
    assert d["gpu_gqa_f16"] == pytest.approx(1407.92)
    assert d["gpu_gqa_f16"] != 0


def test_a_kernel_only_one_arm_produced_is_null_not_zero():
    d = probe.attribution_delta(_arm(10.0, 20.0, extra={"only_candidate": 5.0}),
                                _arm(10.0, 20.0))
    assert d["gpu_only_candidate"] is None
    assert "gpu_only_candidate" in d["not_differenced"]


def test_two_arms_that_genuinely_agree_report_zero_and_declare_nothing_missing():
    d = probe.attribution_delta(_arm(10.0, 20.0), _arm(10.0, 20.0))
    assert d["gpu_gqa_f16"] == 0 and "not_differenced" not in d


def test_the_host_sibling_total_is_differenced_too():
    d = probe.attribution_delta(_arm(10.0, 20.0, sibling=400.0),
                                _arm(10.0, 20.0, sibling=305.0))
    assert d["host_sibling_total"] == pytest.approx(95.0)


# ------------------------------------------------------------------ control-kernel reading
def test_a_whole_device_drift_is_not_read_as_a_gqa_localisation():
    """The reading that keeps this investigation honest.

    If `q_gemv_matmul_nbits_f16` — a kernel the `c96e7d9..85fbda2` diff does not touch — moved
    further than `gqa_f16` in the same direction, the pass drifted as a whole. Calling that a
    GQA regression is the correlation-as-causation error the issue asks to avoid.
    """
    r = probe.kernel_ratios({"gqa_f16": 105.5, "q_gemv_matmul_nbits_f16": 112.5,
                             "gather_f16": 92.8},
                            {"gqa_f16": 100.0, "q_gemv_matmul_nbits_f16": 100.0,
                             "gather_f16": 100.0})
    assert r["gqa_f16_ratio"] == pytest.approx(1.055)
    assert r["gqa_outside_untouched_spread"] is False
    assert "does not localise" in r["reading"]


def test_a_gqa_only_move_is_reported_as_a_localisation_candidate():
    r = probe.kernel_ratios({"gqa_f16": 140.0, "q_gemv_matmul_nbits_f16": 100.5,
                             "gather_f16": 99.7},
                            {"gqa_f16": 100.0, "q_gemv_matmul_nbits_f16": 100.0,
                             "gather_f16": 100.0})
    assert r["gqa_outside_untouched_spread"] is True
    assert "localisation candidate, not a conclusion" in r["reading"]


def test_a_kernel_present_on_one_arm_only_gets_no_ratio():
    r = probe.kernel_ratios({"gqa_f16": 10.0, "ghost_f16": 1.0}, {"gqa_f16": 10.0})
    assert r["per_kernel"]["ghost_f16"] is None


def test_the_untouched_set_names_kernels_the_compiled_diff_cannot_reach():
    """`git diff c96e7d9 85fbda2` touches `gqa_f16.comp` and `ops/attention.rs` and nothing else
    that is compiled, so no other pipeline's SPIR-V can differ between the arms."""
    assert "gqa_f16" not in probe.UNTOUCHED_KERNELS
    assert "q_gemv_matmul_nbits_f16" in probe.UNTOUCHED_KERNELS


# --------------------------------------------------------------- attribution re-derivation
def _traced_record(arm, repeat, past, gqa_us, ctl_us, *, admissible=True):
    r = _record(arm, repeat=repeat)
    r["workload"] = probe.workload_label(probe.rm.PHI35.key, "decode", 1, past)
    r["past"] = past
    r["traced"] = True
    r.pop("speed")
    r["traced_samples_ms"] = [130.0, 131.0, 132.0]
    r["trace"] = {
        "inference_calls": 27,
        "gpu_us_per_inference": {"gqa_f16": gqa_us, "q_gemv_matmul_nbits_f16": ctl_us},
        "gpu_total_us_per_inference": gqa_us + ctl_us,
        "host_phase_us_per_inference": {"record": 60000.0, "fence_wait": 75000.0},
        "host_nested_us_per_inference": {"cmd_upload": 50000.0},
        "host_sibling_total_us_per_inference": 135000.0,
    }
    if not admissible:
        r["admissible"] = False
        r["refusal"] = {"reason": "equivalence not MATCH"}
        r["trace"] = {"withheld": "record refused; no timing is published for it"}
        r.pop("traced_samples_ms")
    return r


def test_the_attribution_summary_is_a_function_of_the_records():
    recs = []
    for rep, (cg, bg) in enumerate([(40003.0, 38707.0), (41170.0, 40900.0), (41202.0, 39027.0)]):
        recs.append(_traced_record("candidate", rep, 128, cg, 22000.0))
        recs.append(_traced_record("baseline", rep, 128, bg, 20000.0))
    rows = probe.summarize_attribution(recs, 3)
    row = next(r for r in rows if r["past"] == 128)
    assert row["candidate"]["n"] == 3 and row["baseline"]["n"] == 3
    assert len(row["gqa_f16_per_repeat"]) == 3
    assert row["gqa_f16_per_repeat_sign_consistent"] is True
    # `delta_us` differences the two arms' MEDIANS; the paired rows difference within a repeat.
    # They are different statistics and the artifact says so.
    assert row["delta_us"]["gpu_gqa_f16"] == pytest.approx(41170.0 - 39027.0)
    assert [p["delta_us"] for p in row["gqa_f16_per_repeat"]] == \
        [pytest.approx(40003.0 - 38707.0), pytest.approx(41170.0 - 40900.0),
         pytest.approx(41202.0 - 39027.0)]
    # Re-derivation is deterministic: the same records must produce the same summary.
    assert json.dumps(probe.summarize_attribution(recs, 3)) == json.dumps(rows)


def test_a_refused_traced_record_never_enters_the_attribution_summary():
    recs = [_traced_record("candidate", 0, 64, 22890.0, 20000.0, admissible=False),
            _traced_record("baseline", 0, 64, 19040.0, 20000.0, admissible=False)]
    row = next(r for r in probe.summarize_attribution(recs, 1) if r["past"] == 64)
    assert row["candidate"] is None and row["baseline"] is None
    assert "delta_us" not in row


def test_a_length_with_only_one_admissible_arm_is_not_differenced():
    recs = [_traced_record("candidate", 0, 128, 40000.0, 20000.0),
            _traced_record("baseline", 0, 128, 39000.0, 20000.0, admissible=False)]
    row = next(r for r in probe.summarize_attribution(recs, 1) if r["past"] == 128)
    assert row["baseline"] is None and "delta_us" not in row


# ----------------------------------------------------------------------------- lock hygiene
def test_the_lock_record_publishes_no_absolute_path_and_no_holder_identity():
    s = probe.sanitize_lock({"lock_path": r"C:\Users\someone\AppData\Local\Temp\gpu-exclusive.lock",
                             "path": "/home/someone/gpu-exclusive.lock",
                             "waited_seconds": 12.5, "acquired_at": "2026-08-08T23:00:00",
                             "acquired_at_monotonic": 91234.5,
                             "holder_pid": 4242, "holder_cmdline": "python secret.py",
                             "killed_anything": False, "policy": "wait, never kill"})
    blob = json.dumps(s)
    assert "someone" not in blob and "secret.py" not in blob
    assert not _ABSOLUTE.search(blob), blob
    assert s["lock_path"] == "gpu-exclusive.lock" and s["path"] == "gpu-exclusive.lock"
    assert s["waited_seconds"] == 12.5 and s["killed_anything"] is False
    assert "acquired_at_monotonic" not in s


def test_the_lock_record_keeps_the_never_killed_statement_intact():
    s = probe.sanitize_lock({"state": "RELEASED", "held_seconds": 1021.2,
                             "killed_anything": False, "policy": "wait, never kill"})
    assert s == {"state": "RELEASED", "held_seconds": 1021.2,
                 "killed_anything": False, "policy": "wait, never kill"}


# ------------------------------------------------------- screens over the committed artifacts
ARTIFACTS = ("crossbuild_decode_window.json", "crossbuild_decode_attribution.json",
             "spirv_gqa_crossbuild.json")
RECORD_ARTIFACTS = ("crossbuild_decode_window.json", "crossbuild_decode_attribution.json")
_ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]|/home/|/Users/|\\\\[A-Za-z0-9_.-]+\\")


def _artifact(name):
    p = RESULTS / name
    if not p.is_file():
        pytest.skip(f"{name} not committed")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ARTIFACTS)
def test_no_committed_artifact_leaks_an_absolute_path_or_a_user_name(name):
    blob = (RESULTS / name).read_text(encoding="utf-8") if (RESULTS / name).is_file() \
        else pytest.skip(f"{name} not committed")
    hits = sorted(set(_ABSOLUTE.findall(blob)))
    assert not hits, f"{name} publishes filesystem paths: {hits}"
    assert "justinchu" not in blob.lower()


@pytest.mark.parametrize("name", ARTIFACTS)
def test_no_committed_artifact_claims_a_cuda_measurement(name):
    blob = (RESULTS / name).read_text(encoding="utf-8") if (RESULTS / name).is_file() \
        else pytest.skip(f"{name} not committed")
    # `cuda` may only appear inside the model's own name, which is a file on disk, not a claim.
    for m in re.finditer(r"cuda", blob, re.I):
        ctx = blob[max(0, m.start() - 60):m.end() + 60]
        assert "int4-rtn-block-32" in ctx or "phi-3.5" in ctx.lower(), \
            f"{name} mentions CUDA outside the model name: {ctx!r}"


@pytest.mark.parametrize("name", RECORD_ARTIFACTS)
def test_every_refused_record_in_a_committed_artifact_carries_no_speed(name):
    art = _artifact(name)
    leaks = [r["workload"] for r in art["records"]
             if not r.get("admissible") and ("speed" in r or "throughput" in r)]
    assert not leaks, f"refused records with timings: {leaks}"


@pytest.mark.parametrize("name", RECORD_ARTIFACTS)
def test_every_admissible_record_names_its_arm_s_witness_key(name):
    art = _artifact(name)
    for r in art["records"]:
        if not r.get("admissible") or not probe.has_gqa(r.get("model_key", "")):
            continue
        assert (r["path_witness"]["gqa_keys"]
                == [probe.EXPECTED_GQA_KEY[r["arm"]]]), r["workload"]


def test_the_sweep_ran_exactly_the_processes_its_protocol_declares():
    art = _artifact("crossbuild_decode_window.json")
    env = art["environment"]
    expected = len(art["workloads"]) * 2 * env["repeats"]
    assert art["counts"]["records"] == expected
    assert art["counts"]["admissible"] + art["counts"]["refused"] == expected


def test_the_published_ratios_recompute_from_the_published_per_repeat_rows():
    import statistics
    art = _artifact("crossbuild_decode_window.json")
    checked = 0
    for p in art["workloads"]:
        for row in p.get("per_repeat") or []:
            assert row["ratio"] == pytest.approx(row["baseline_ms"] / row["candidate_ms"])
            checked += 1
        if p.get("per_repeat"):
            rs = [r["ratio"] for r in p["per_repeat"]]
            assert p["ratio_median"] == pytest.approx(statistics.median(rs))
            assert p["ratio_min"] == pytest.approx(min(rs))
            assert p["ratio_max"] == pytest.approx(max(rs))
    assert checked, "no per-repeat ratio in the artifact to recompute"


def test_every_published_verdict_reapplies_from_the_published_band():
    art = _artifact("crossbuild_decode_window.json")
    band = art["band"]["applied"]
    repeats = art["environment"]["repeats"]
    for p in art["workloads"]:
        assert p["verdict"] == probe.verdict_for(p, band, repeats), p["workload"]


def test_the_published_band_reapplies_from_the_published_controls():
    art = _artifact("crossbuild_decode_window.json")
    assert art["band"]["applied"] == pytest.approx(
        probe.drift_envelope(art["workloads"])["band"])


def test_the_published_window_claim_reapplies_from_the_published_verdicts():
    art = _artifact("crossbuild_decode_window.json")
    assert probe.window_claim(art["workloads"])["claim"] == art["window"]["claim"]


def test_the_two_arms_are_two_different_libraries_of_the_declared_sizes():
    art = _artifact("crossbuild_decode_window.json")
    b, c = art["arms"]["baseline"], art["arms"]["candidate"]
    assert b["sha256"] != c["sha256"]
    assert b["commit"].startswith("c96e7d9") and c["commit"].startswith("85fbda2")


def test_the_artifact_carries_the_digest_of_the_preregistration_it_was_run_under():
    art = _artifact("crossbuild_decode_window.json")
    pre = art.get("preregistration")
    assert pre and len(pre["sha256"]) == 64
    on_disk = RESULTS / pre["path"]
    if on_disk.is_file():
        assert probe.rm.sha256_file(on_disk) == pre["sha256"], \
            "the preregistration was edited after the sweep ran"


# -------------------------------------------------- screens over the attribution artifact
def test_the_published_attribution_summary_re_derives_from_its_own_records():
    """The whole analysis, checked without a GPU. Both bugs this pass shipped and fixed —
    a kernel-key mismatch that published a false zero, and a `nested_in: "none"` string that
    emptied the host top level — were analysis bugs this check would have caught."""
    art = _artifact("crossbuild_decode_attribution.json")
    repeats = max(r.get("repeat", 0) for r in art["records"]) + 1
    fresh = probe.summarize_attribution(art["records"], repeats)
    assert json.dumps(fresh, sort_keys=True) == json.dumps(art["summary"], sort_keys=True)


def test_the_published_sweep_derivation_re_derives_from_its_own_records():
    """The sweep's band, every verdict and the window claim are functions of the 54 stored
    records and of nothing else — checkable by a reviewer with neither this GPU nor this model."""
    art = _artifact("crossbuild_decode_window.json")
    repeats = max(r.get("repeat", 0) for r in art["records"]) + 1
    fresh = probe.resummarize_sweep(art["records"], repeats)
    for section in ("band", "workloads", "window"):
        assert json.dumps(fresh[section], sort_keys=True) \
            == json.dumps(art[section], sort_keys=True), section


def test_the_sweep_re_derivation_reads_the_workloads_from_the_records():
    """The defect this forbids: re-deriving against today's `WORKLOADS` constant instead of
    against the stored records, which would make the check pass for an artifact that measured
    a different set of workloads than the source now declares."""
    art = _artifact("crossbuild_decode_window.json")
    kept = [r for r in art["records"] if "past1024" not in r["workload"]]
    fresh = probe.resummarize_sweep(kept, 3)
    labels = {p["workload"] for p in fresh["workloads"]}
    assert not any("past1024" in lbl for lbl in labels), \
        "a workload absent from the records must not reappear from the source constant"


def test_the_artifact_kind_is_decided_by_content_not_by_filename():
    """`--resummarize` must not re-derive the wrong section and report a spurious mismatch:
    that is exactly what an earlier revision did to the sweep artifact."""
    assert probe.artifact_kind(_artifact("crossbuild_decode_window.json")) == "sweep"
    assert probe.artifact_kind(_artifact("crossbuild_decode_attribution.json")) == "attribution"
    assert probe.artifact_kind({"records": []}) == "unknown"


def test_no_refused_attribution_record_publishes_a_device_timing():
    art = _artifact("crossbuild_decode_attribution.json")
    for r in art["records"]:
        if r.get("admissible"):
            continue
        assert "speed" not in r and "traced_samples_ms" not in r, r["workload"]
        tr = r.get("trace") or {}
        assert "gpu_us_per_inference" not in tr, \
            f"{r['workload']} {r['arm']} publishes device time on a refused record"
        assert tr.get("withheld"), r["workload"]


def test_the_attribution_artifact_says_it_is_not_a_wall_clock_result():
    art = _artifact("crossbuild_decode_attribution.json")
    assert "wall-clock" in art["caveat"]
    assert "PR #94" in art["instrument_provenance"] or "No PR #94" in art["instrument_provenance"]


def test_every_attribution_delta_that_is_null_is_declared_rather_than_shown_as_zero():
    art = _artifact("crossbuild_decode_attribution.json")
    for row in art["summary"]:
        d = row.get("delta_us")
        if not d:
            continue
        nulls = sorted(k for k, v in d.items() if v is None and k != "not_differenced")
        assert nulls == sorted(d.get("not_differenced", [])), row["past"]


# ------------------------------------------------------ screens over the SPIR-V artifact
def test_the_spirv_artifact_states_what_it_cannot_show():
    art = _artifact("spirv_gqa_crossbuild.json")
    text = art["what_this_cannot_show"]
    assert "driver" in text.lower() and "SASS" in text
    assert "pipeline_executable_properties" in text


def test_the_spirv_modules_differ_only_in_how_the_workgroup_size_is_declared():
    art = _artifact("spirv_gqa_crossbuild.json")
    changed = art["module_diff"]["spirv_diff_changed_lines"]
    allowed = re.compile(r"Bound:|SpecId|WorkgroupSize|OpConstantComposite|OpSpecConstant")
    stray = [ln for ln in changed if not allowed.search(ln)]
    assert not stray, f"the module diff is not confined to the workgroup-size declaration: {stray}"
    assert art["findings"]["body_instructions_differ"] is False


def test_the_spirv_artifact_records_that_no_instruction_reads_the_workgroup_size():
    """The module-level half of H1: a specialisation constant cannot stop a fold that does not
    exist. Nothing in the body consumes `gl_WorkGroupSize`, so no loop bound depends on it."""
    art = _artifact("spirv_gqa_crossbuild.json")
    assert art["findings"]["workgroup_size_consumed_by_body"] is False
    assert art["findings"]["workgroup_size_consumer_lines"] == []


def test_the_spirv_artifact_records_that_neither_module_uses_shared_memory():
    art = _artifact("spirv_gqa_crossbuild.json")
    assert art["findings"]["shared_memory_present"] is False
    for arm in ("baseline", "candidate"):
        assert not any(k.endswith("Workgroup")
                       for k in art["arms"][arm]["storage_classes"]), arm


def test_both_spirv_modules_validate():
    art = _artifact("spirv_gqa_crossbuild.json")
    assert art["arms"]["baseline"]["spirv_val"] == "PASS"
    assert art["arms"]["candidate"]["spirv_val"] == "PASS"
    assert art["candidate_frozen_to_1"]["spirv_val"] == "PASS"


def test_the_opcode_histogram_delta_matches_the_two_published_histograms():
    art = _artifact("spirv_gqa_crossbuild.json")
    b = art["arms"]["baseline"]["opcode_histogram"]
    c = art["arms"]["candidate"]["opcode_histogram"]
    recomputed = {k: c.get(k, 0) - b.get(k, 0) for k in set(b) | set(c)
                  if c.get(k, 0) != b.get(k, 0)}
    assert recomputed == art["module_diff"]["opcode_histogram_delta"]
    assert (sum(c.values()) - sum(b.values())
            == art["module_diff"]["instruction_count_delta"])
