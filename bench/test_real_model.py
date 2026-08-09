"""Self-tests for `bench/real_model.py` — no GPU, no EP, no model file.

Each test is named for the **plausible but wrong** reading it prevents, in the style of
`bench/test_plausible_but_wrong.py`. The point of this file is that the parts of the #56
instrument that can be wrong without a device — provenance, feeds, throughput arithmetic,
pairing, verdicts — are checked on every `pytest bench` run, not only on the one machine that
has an RTX A1000 and a Foundry cache.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import real_model as rm  # noqa: E402


# ---------------------------------------------------------------------------
# Model identity and provenance
# ---------------------------------------------------------------------------

def test_no_model_spec_carries_a_literal_path():
    """A literal cache path is a guess about a version (#11: the cache moved under us)."""
    for spec in rm.MODELS.values():
        blob = repr(spec)
        assert ".foundry" not in blob
        assert ":\\" not in blob and ":/" not in blob


def test_phi35_is_resolved_through_foundry_not_a_filename():
    assert rm.PHI35.resolver == "foundry"
    assert rm.PHI35.variant_name and rm.PHI35.execution_provider


def test_missing_model_raises_rather_than_skipping(tmp_path, monkeypatch):
    """#56 asks for numbers on real models; a lane that silently skips one reads as complete."""
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    with pytest.raises(rm.ModelUnavailable):
        rm.resolve_model(rm.MOBILENETV2)


def test_repo_cache_dir_honours_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    assert rm.repo_cache_dir() == tmp_path


def test_recorded_sha256_comes_from_another_tools_artifact(tmp_path):
    """A hash this module both writes and checks proves nothing, so it is read from elsewhere."""
    (tmp_path / "rust-model-runner").mkdir()
    (tmp_path / rm.MOBILENETV2.recorded_provenance).write_text('{"onnx_sha256": "abc"}')
    assert rm.recorded_sha256(rm.MOBILENETV2, tmp_path) == "abc"


def test_external_weights_are_hashed_not_just_the_graph(tmp_path):
    """The Foundry Phi-3.5 `.onnx` is 26 MB and its weights are 2.29 GB in a sibling `.data`
    file. Hashing the graph alone identifies the topology and says nothing about the numbers the
    benchmark multiplies: replace the blob and the recorded hash does not move."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    blob = tmp_path / "weights.bin"
    payload = np.arange(16, dtype=np.float32)
    blob.write_bytes(payload.tobytes())
    t = numpy_helper.from_array(payload.reshape(4, 4), name="W")
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    t.external_data.extend([
        onnx.StringStringEntryProto(key="location", value="weights.bin"),
        onnx.StringStringEntryProto(key="offset", value="0"),
        onnx.StringStringEntryProto(key="length", value=str(payload.nbytes)),
    ])
    graph = helper.make_graph([], "g", [], [], initializer=[t])
    model = helper.make_model(graph)
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert [f["location"] for f in rec["files"]] == ["weights.bin"]
    assert rec["files"][0]["sha256"] == rm.sha256_file(blob)
    assert rec["files"][0]["bytes"] == payload.nbytes


def test_a_graph_with_no_external_data_says_so_rather_than_looking_unscanned(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), name="W")
    model = helper.make_model(helper.make_graph([], "g", [], [], initializer=[t]))
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())
    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert rec["complete"] is True
    assert rec["files"] == []
    # #78: the reason must name the places it looked, not just assert an absence. "no external
    # data" from a walk that only read `graph.initializer` was the pre-#78 false negative.
    assert "covers every weight byte" in rec["reason"]
    for place in ("subgraph", "sparse initializer", "function body", "training_info"):
        assert place in rec["reason"], place


def test_a_missing_weight_blob_is_reported_not_silently_skipped(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), name="W")
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    t.external_data.extend([onnx.StringStringEntryProto(key="location", value="gone.bin")])
    model = helper.make_model(helper.make_graph([], "g", [], [], initializer=[t]))
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())
    rec = rm.external_data_provenance(mpath)
    assert rec["files"][0]["missing"] is True
    assert rec["files"][0]["sha256"] is None


def test_recorded_sha256_absent_is_none_not_empty_string(tmp_path):
    """``None`` (no record) and ``""`` (a record of nothing) are different states."""
    assert rm.recorded_sha256(rm.MOBILENETV2, tmp_path) is None


def test_resolve_model_reports_provenance_disagreement_rather_than_crashing(tmp_path,
                                                                           monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / rm.MOBILENETV2.cache_filename).write_bytes(b"not really an onnx file")
    (tmp_path / "rust-model-runner").mkdir()
    (tmp_path / rm.MOBILENETV2.recorded_provenance).write_text(
        '{"onnx_sha256": "0000000000000000"}')
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(models))
    rec = rm.resolve_model(rm.MOBILENETV2, results_dir=tmp_path)
    assert rec["agrees_with_recorded_provenance"] is False
    assert rec["sha256"] != rec["recorded_sha256"]


# ---------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------

def test_decode_cases_have_a_non_empty_kv_cache():
    """The single largest hole #56 names: every earlier measurement ran at past == 0."""
    cases = rm.phi35_cases([1, 2], [0, 128, 512])
    decode = [c for c in cases if c.phase == "decode"]
    assert decode, "there must be decode cases at all"
    assert any(c.past > 0 for c in decode), "at least one decode case must carry a real cache"


def test_prefill_feeds_have_an_empty_cache_and_decode_feeds_do_not():
    pre = rm.phi35_feeds(rm.Case(rm.PHI35.key, "prefill", 4, 0, tokens=4), np)
    dec = rm.phi35_feeds(rm.Case(rm.PHI35.key, "decode", 1, 7, tokens=1), np)
    assert pre["past_key_values.0.key"].shape == (1, 32, 0, 96)
    assert dec["past_key_values.0.key"].shape == (1, 32, 7, 96)
    assert dec["past_key_values.31.value"].shape == (1, 32, 7, 96)


def test_attention_mask_covers_past_plus_new_tokens():
    """GQA derives `seqlens_k` from the mask; a mask of length `m` at past>0 is rejected by ORT
    with `seqlens_k[0] = N is out of range`, which is exactly how the modelrunner's generic
    input generator fails on this model."""
    dec = rm.phi35_feeds(rm.Case(rm.PHI35.key, "decode", 1, 31, tokens=1), np)
    assert dec["attention_mask"].shape == (1, 32)
    pre = rm.phi35_feeds(rm.Case(rm.PHI35.key, "prefill", 8, 0, tokens=8), np)
    assert pre["attention_mask"].shape == (1, 8)


def test_feeds_are_byte_identical_across_calls():
    """Two arms must be fed the same bytes or the comparison is between two different inputs."""
    case = rm.Case(rm.PHI35.key, "decode", 1, 5, tokens=1)
    first = rm.feeds_digest(rm.phi35_feeds(case, np))
    second = rm.feeds_digest(rm.phi35_feeds(case, np))
    assert first == second


def test_feeds_digest_notices_a_renamed_key():
    """Hashing values alone would call a feed dict with the right values under the wrong keys
    identical to the correct one."""
    a = {"x": np.ones(4, dtype=np.float32)}
    b = {"y": np.ones(4, dtype=np.float32)}
    assert rm.feeds_digest(a) != rm.feeds_digest(b)


def test_feeds_digest_notices_a_changed_dtype_at_equal_values():
    a = {"x": np.ones(4, dtype=np.float32)}
    b = {"x": np.ones(4, dtype=np.float16)}
    assert rm.feeds_digest(a) != rm.feeds_digest(b)


def test_kv_round_trip_is_the_measured_slope_not_a_guess():
    assert rm.PHI35_BYTES_PER_PAST_TOKEN == 393216
    kv = rm.kv_round_trip_bytes(rm.Case(rm.PHI35.key, "decode", 1, 1024, tokens=1))
    assert kv["past_upload_bytes"] == 1024 * 393216
    assert kv["present_readback_bytes"] == 1025 * 393216


def test_kv_round_trip_is_none_for_a_model_without_a_cache():
    assert rm.kv_round_trip_bytes(rm.Case(rm.MOBILENETV2.key, "batch", 8, 0)) is None


# ---------------------------------------------------------------------------
# Arms and ordering
# ---------------------------------------------------------------------------

def test_the_two_vulkan_arms_differ_by_exactly_one_variable():
    tiled = dict(rm.VULKAN_TILED.env)
    untiled = dict(rm.VULKAN_UNTILED.env)
    assert set(tiled) == set(untiled) == {rm.ROWS_ENV}
    assert tiled[rm.ROWS_ENV] != untiled[rm.ROWS_ENV]
    assert untiled[rm.ROWS_ENV] == "1", "the kill switch pins the tile back to one row"


def test_arm_order_alternates():
    """A fixed order produced a spurious 0.905x at M=1 on identical SPIR-V (§25.4)."""
    a = rm.arm_order(rm.ARMS, 0)
    b = rm.arm_order(rm.ARMS, 1)
    assert [x.name for x in a] == list(reversed([x.name for x in b]))


def test_arm_env_is_restorable():
    environ = {rm.ROWS_ENV: "9"}
    prev = rm.VULKAN_UNTILED.apply_env(environ)
    assert environ[rm.ROWS_ENV] == "1"
    assert prev[rm.ROWS_ENV] == "9"


def test_m1_is_the_null_control_and_m4_is_not():
    assert rm.is_null_control(rm.Case(rm.PHI35.key, "prefill", 1, 0, tokens=1))
    assert rm.is_null_control(rm.Case(rm.PHI35.key, "decode", 1, 512, tokens=1))
    assert not rm.is_null_control(rm.Case(rm.PHI35.key, "prefill", 4, 0, tokens=4))


# ---------------------------------------------------------------------------
# Statistics and throughput
# ---------------------------------------------------------------------------

def test_latency_stats_reports_a_distribution_not_a_number():
    st = rm.latency_stats([10, 11, 12, 13, 100])
    for key in ("median_ms", "min_ms", "max_ms", "p05_ms", "p95_ms", "mad_ms", "rsd"):
        assert key in st
    assert st["median_ms"] == 12
    assert st["max_ms"] == 100, "an outlier must remain visible, not be smoothed away"


def test_latency_stats_of_nothing_is_not_zero():
    """An empty sample is `n = 0`, not a median of 0 — a fabricated fast number."""
    st = rm.latency_stats([])
    assert st == {"n": 0}


def test_prefill_throughput_divides_by_tokens_consumed():
    case = rm.Case(rm.PHI35.key, "prefill", 8, 0, tokens=8)
    tp = rm.throughput(case, 100.0)
    assert tp["value"] == pytest.approx(80.0)
    assert tp["unit"] == "tokens/s"


def test_decode_throughput_does_not_divide_by_the_cache_length():
    """Dividing a decode step by `past` manufactures a throughput that *grows* as the model
    slows down — the number getting better as the thing gets worse."""
    short = rm.Case(rm.PHI35.key, "decode", 1, 128, tokens=1)
    long = rm.Case(rm.PHI35.key, "decode", 1, 2048, tokens=1)
    assert rm.throughput(short, 50.0)["value"] == pytest.approx(20.0)
    assert rm.throughput(long, 50.0)["value"] == pytest.approx(20.0)


def test_batch_throughput_is_images_not_tokens():
    case = rm.Case(rm.MOBILENETV2.key, "batch", 16, 0, tokens=None, unit="images")
    tp = rm.throughput(case, 160.0)
    assert tp["unit"] == "images/s"
    assert tp["value"] == pytest.approx(100.0)


def test_throughput_of_a_zero_latency_is_none_not_infinity():
    assert rm.throughput(rm.Case(rm.PHI35.key, "prefill", 1, 0, tokens=1), 0.0) is None


def test_paired_ratio_survives_one_displaced_repeat():
    """The median of per-repeat ratios, not the ratio of pooled medians: a repeat where one arm
    was displaced by a co-tenant must not move the estimate."""
    a = [10.0, 10.0, 10.0, 90.0]
    b = [5.0, 5.0, 5.0, 45.0]
    assert rm.paired_ratios(a, b)["median"] == pytest.approx(2.0)


def test_paired_ratio_of_nothing_is_n_zero():
    assert rm.paired_ratios([], [])["n"] == 0


def test_noise_floor_refuses_to_call_a_sub_floor_ratio_a_speedup():
    floor = {"n": 3, "median": 1.0, "min": 0.94, "max": 1.15}
    assert rm.exceeds_noise_floor(1.05, floor) is False
    assert rm.exceeds_noise_floor(1.62, floor) is True


def test_missing_noise_floor_is_none_not_false():
    """"No control to read this against" is a different state from "not significant"."""
    assert rm.exceeds_noise_floor(1.62, {}) is None
    assert rm.exceeds_noise_floor(1.62, {"n": 0}) is None


# ---------------------------------------------------------------------------
# Equivalence verdicts
# ---------------------------------------------------------------------------

def _logits(argmax_at: int, n: int = 64, scale: float = 1.0):
    x = np.linspace(0.0, 0.1, n).astype(np.float32) * scale
    x[argmax_at] = 5.0 * scale
    return x.reshape(1, 1, n)


def test_identical_logits_match():
    v = rm.classify_logits(_logits(3), _logits(3), np)
    assert v["verdict"] == rm.MATCH


def test_moved_argmax_is_divergent_even_within_tolerance():
    a, b = _logits(3), _logits(3)
    b[0, 0, 3] = 0.0
    b[0, 0, 4] = 5.0
    v = rm.classify_logits(a, b, np)
    assert v["verdict"] == rm.DIVERGENT


def test_all_zero_reference_is_divergent_not_a_perfect_match():
    """The `argmax 0` defect: 161 nodes dispatched, `compute_failures: 0`, and both tensors
    all-zero, which any elementwise comparison calls a perfect agreement."""
    z = np.zeros((1, 1, 64), dtype=np.float32)
    v = rm.classify_logits(z, z, np)
    assert v["verdict"] == rm.DIVERGENT
    assert v["reference_all_zero"] is True


def test_shape_mismatch_is_divergent_not_an_exception():
    v = rm.classify_logits(np.zeros((1, 1, 8)), np.zeros((1, 1, 9)), np)
    assert v["verdict"] == rm.DIVERGENT
    assert v["reason"] == "shape mismatch"


def test_logit_budget_is_a_fraction_of_scale_not_a_constant():
    """A constant absolute budget silently tightens as the logit scale grows with the cache:
    0.5 is 3.8% of scale at past=0 and 2.1% at past=1024, so the same kernel gets a stricter
    exam for a longer sequence. The budget must track the reference's own magnitude."""
    small = _logits(3, scale=1.0)
    large = _logits(3, scale=10.0)
    v_small = rm.classify_logits(small, small, np)
    v_large = rm.classify_logits(large, large, np)
    assert v_large["abs_budget"] == pytest.approx(10.0 * v_small["abs_budget"])


def test_a_logit_error_above_the_scale_fraction_is_divergent():
    ref = _logits(3, scale=1.0)
    cand = ref.copy()
    cand[0, 0, 10] += 1.01 * rm.PHI35_LOGIT_SCALE_FRACTION * float(np.abs(ref).max())
    v = rm.classify_logits(cand, ref, np)
    assert v["verdict"] == rm.DIVERGENT
    assert v["max_abs"] > v["abs_budget"]


def test_a_logit_error_below_the_scale_fraction_is_a_match():
    ref = _logits(3, scale=1.0)
    cand = ref.copy()
    cand[0, 0, 63] += 0.5 * rm.PHI35_LOGIT_SCALE_FRACTION * float(np.abs(ref).max())
    v = rm.classify_logits(cand, ref, np)
    assert v["verdict"] == rm.MATCH


def test_logits_are_gated_on_the_distribution_they_induce():
    """argmax and top-10 can both agree while the sampler's distribution moves — scale the whole
    vector and the ranking is untouched but the probabilities are not. A logit-space bound alone
    would call that MATCH, and what a decoder emits is a distribution, not a ranking."""
    ref = _logits(3, scale=1.0)
    cand = (ref * 1.5).astype(np.float32)
    v = rm.classify_logits(cand, ref, np, scale_fraction=10.0)  # absolute clause disabled
    assert v["argmax_candidate"] == v["argmax_reference"]
    assert v["topk_overlap"] == v["top_k"]
    assert v["max_prob_delta"] > rm.PHI35_MAX_PROB_DELTA
    assert v["verdict"] == rm.DIVERGENT


def test_one_element_a_third_above_the_floor_is_not_a_defect():
    """MEASURED: at past=128 the Vulkan arms put exactly one element of 396,288 at 1.33x the
    fp16 floor. Failing the whole run on that is an instrument that cannot distinguish an
    accumulation tail from a wrong cache; passing anything at all is a fudge. The band does
    both — see the gross-ceiling and marginal-fraction controls below."""
    rng = np.random.default_rng(5)
    ref = rng.standard_normal((1, 32, 129, 96)).astype(np.float16).astype(np.float64)
    cand = ref.copy()
    idx = np.unravel_index(np.argmax(np.abs(ref)), ref.shape)
    floor = rm.KV_ULP_BUDGET * rm.FP16_EPS * np.abs(ref).max() + rm.KV_REL_TOL * abs(ref[idx])
    cand[idx] = ref[idx] + 1.33 * floor
    v = rm.classify_activation(cand, ref, np)
    assert v["elements_outside_tolerance"] == 1
    assert v["elements_gross"] == 0
    assert v["verdict"] == rm.MATCH


def test_a_gross_element_fails_however_few_there_are():
    """The ceiling is what keeps the marginal band from being a fudge: one element far out is
    DIVERGENT even though one element in 400,000 is a vanishing fraction."""
    rng = np.random.default_rng(6)
    ref = rng.standard_normal((1, 32, 129, 96)).astype(np.float16).astype(np.float64)
    cand = ref.copy()
    idx = np.unravel_index(np.argmax(np.abs(ref)), ref.shape)
    floor = rm.KV_ULP_BUDGET * rm.FP16_EPS * np.abs(ref).max() + rm.KV_REL_TOL * abs(ref[idx])
    cand[idx] = ref[idx] + (rm.KV_GROSS_MULTIPLE + 1.0) * floor
    v = rm.classify_activation(cand, ref, np)
    assert v["elements_gross"] == 1
    assert v["marginal_fraction_observed"] < rm.KV_MARGINAL_FRACTION
    assert v["verdict"] == rm.DIVERGENT


def test_a_systematic_small_error_fails_on_its_population():
    """A bias that puts every element just above the floor is the failure mode a per-element
    ceiling alone would miss: no single element is gross, and the tensor is still wrong."""
    rng = np.random.default_rng(7)
    ref = rng.standard_normal((1, 32, 129, 96)).astype(np.float16).astype(np.float64)
    floor = rm.KV_ULP_BUDGET * rm.FP16_EPS * np.abs(ref).max() + rm.KV_REL_TOL * np.abs(ref)
    cand = ref + 1.5 * floor
    v = rm.classify_activation(cand, ref, np)
    assert v["elements_gross"] == 0
    assert v["marginal_fraction_observed"] > rm.KV_MARGINAL_FRACTION
    assert v["verdict"] == rm.DIVERGENT


def test_bitwise_control_calls_identical_outputs_identical():
    a = [np.arange(12, dtype=np.float16).reshape(3, 4)]
    assert rm.bitwise_identical(a, [x.copy() for x in a], np)["identical"] is True


def test_bitwise_control_catches_a_one_bit_difference():
    """At M=1 the tiled and untiled arms are claimed to bind the same SPIR-V. If that claim is
    false the outputs will differ somewhere, and a *tolerance* would hide it — this is the one
    comparison in the harness that must be exact."""
    a = [np.arange(12, dtype=np.float16).reshape(3, 4)]
    b = [a[0].copy()]
    b[0][1, 1] = np.nextafter(b[0][1, 1], np.float16(1000.0))
    v = rm.bitwise_identical(a, b, np)
    assert v["identical"] is False
    assert v["first_difference"] == "output[0]"


def test_bitwise_control_does_not_crash_on_mismatched_counts():
    v = rm.bitwise_identical([np.zeros(4)], [np.zeros(4), np.zeros(4)], np)
    assert v["identical"] is False
    assert v["first_difference"] == "output count"


def test_classify_outputs_catches_a_wrong_kv_cache_behind_correct_logits():
    """A decode step whose logits agree but whose `present` cache is wrong produces a correct
    first token and a wrong sequence. Comparing output 0 alone would call it MATCH."""
    case = rm.Case(rm.PHI35.key, "decode", 1, 4, tokens=1)
    logits = _logits(3)
    rng = np.random.default_rng(11)
    good_kv = rng.standard_normal((1, 32, 5, 96)).astype(np.float16)
    bad_kv = good_kv.copy()
    bad_kv[0, 0, 0, 0] = np.float16(7.0)
    ok = rm.classify_outputs(case, [logits, good_kv], [logits, good_kv], np)
    bad = rm.classify_outputs(case, [logits, bad_kv], [logits, good_kv], np)
    assert ok["verdict"] == rm.MATCH
    assert bad["verdict"] == rm.DIVERGENT
    assert bad["secondary_divergent"] == 1


def test_activation_gate_is_elementwise_not_an_aggregate_or():
    """An aggregate `max_abs <= floor OR max_rel_signal <= tol` has a hole a planted error walks
    through: put the error on an element *below* the signal threshold and the relative clause
    excludes it, so the OR passes on a tensor with a 7.0 error in it. That is not hypothetical —
    it is what the first version of this gate did to `test_classify_outputs_catches_a_wrong_kv
    _cache_behind_correct_logits`."""
    rng = np.random.default_rng(11)
    ref = rng.standard_normal((1, 32, 5, 96)).astype(np.float16).astype(np.float64)
    idx = np.unravel_index(np.argmin(np.abs(ref)), ref.shape)  # a below-threshold element
    cand = ref.copy()
    cand[idx] = 7.0
    v = rm.classify_activation(cand, ref, np)
    assert v["verdict"] == rm.DIVERGENT
    assert v["elements_outside_tolerance"] == 1
    assert v["max_rel_signal"] == 0.0, "the excluded element proves the aggregate OR would pass"


def test_one_fp16_ulp_of_scale_is_not_a_divergence():
    """The gate this replaced called every Vulkan arm DIVERGENT at `max_rel = 2.73` on a tensor
    whose largest absolute error was one fp16 ULP — a cancellation meter read as an accuracy
    meter. That failure is the reason `classify_activation` exists."""
    rng = np.random.default_rng(5)
    ref = (rng.standard_normal((1, 32, 4, 96)) * 8.0).astype(np.float16)
    cand = ref.astype(np.float64).copy()
    # One ULP at magnitude 16 is 0.015625 — exactly the residual the real run produced.
    cand[0, 0, 0, 0] += 0.015625
    # ...and a near-zero element perturbed by the same absolute amount, which is what drives the
    # unrestricted relative figure to the hundreds.
    idx = np.unravel_index(np.argmin(np.abs(ref)), ref.shape)
    cand[idx] += 0.015625
    v = rm.classify_activation(cand, ref, np)
    assert v["verdict"] == rm.MATCH
    assert v["max_rel_all"] > 1.0, "the unrestricted relative figure is still huge — and unused"
    assert v["max_rel_signal"] < 0.05


def test_activation_gate_scales_with_the_tensors_own_magnitude():
    """A constant absolute tolerance is absurdly tight at magnitude 30 and absurdly loose at
    magnitude 1e-4. The budget is ULPs of the reference's own scale, so both are caught."""
    small = np.full((4, 4), 1e-3, dtype=np.float64)
    cand_small = small.copy()
    cand_small[0, 0] = 1e-1
    assert rm.classify_activation(cand_small, small, np)["verdict"] == rm.DIVERGENT
    big = np.full((4, 4), 30.0, dtype=np.float64)
    cand_big = big.copy()
    cand_big[0, 0] = 30.0 + 0.015625
    assert rm.classify_activation(cand_big, big, np)["verdict"] == rm.MATCH


def test_activation_gate_refuses_an_all_zero_reference():
    z = np.zeros((2, 3), dtype=np.float64)
    v = rm.classify_activation(z, z, np)
    assert v["verdict"] == rm.DIVERGENT


def test_empty_activation_match_is_labelled_as_vacuous():
    """An empty `present` is a real state; a vacuous MATCH must not read as a checked one."""
    e = np.zeros((1, 32, 0, 96), dtype=np.float16)
    v = rm.classify_activation(e, e, np)
    assert v["verdict"] == rm.MATCH and v["empty"] is True


def test_classify_outputs_notices_a_missing_output():
    case = rm.Case(rm.PHI35.key, "decode", 1, 4, tokens=1)
    v = rm.classify_outputs(case, [_logits(3)], [_logits(3), np.ones((1, 2))], np)
    assert v["verdict"] == rm.DIVERGENT


def test_classifier_output_is_gated_on_the_predicted_label():
    """A tolerance that passes while the predicted class moves is not a useful tolerance."""
    ref = np.zeros((2, 1000), dtype=np.float32)
    ref[:, 7] = 1.0
    cand = ref.copy()
    cand[1, 7] = 0.0
    cand[1, 8] = 1.0
    assert rm.classify_tensor(ref, ref, np)["verdict"] == rm.MATCH
    assert rm.classify_tensor(cand, ref, np)["verdict"] == rm.DIVERGENT


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_dispatch_diagnosis_divides_by_inferences_not_by_repeats():
    """A cumulative counter divided by the wrong denominator is how this project once read a
    1.78x "improvement" that was really an iteration ratio."""
    d = rm.dispatch_diagnosis({"dispatches_executed": 1000, "subgraphs_live": 33}, 10)
    assert d["dispatches_per_inference"] == pytest.approx(100.0)


def test_dispatch_diagnosis_with_no_counters_is_none_not_zero():
    d = rm.dispatch_diagnosis(None, 10)
    assert d["dispatches_per_inference"] is None
    assert d["islands"] is None


def test_fallback_diagnosis_separates_claimed_from_executed():
    f = rm.fallback_diagnosis({rm.EP_NAME: 3, rm.CPU_EP: 97})
    assert f["cpu_fallback_node_executions"] == 97
    assert f["vulkan_share"] == pytest.approx(0.03)


def test_fallback_diagnosis_of_an_absent_profile_is_not_a_perfect_score():
    f = rm.fallback_diagnosis(None)
    assert f["vulkan_share"] is None
    assert f["total_node_executions"] == 0


def test_bandwidth_proxy_says_what_it_assumes():
    p = rm.bandwidth_proxy(rm.Case(rm.PHI35.key, "decode", 1, 1024, tokens=1), 500.0)
    assert p["implied_gib_per_s"] > 0
    assert "device-resident" in p["assumes"]


def test_bandwidth_proxy_is_none_without_a_kv_cache():
    assert rm.bandwidth_proxy(rm.Case(rm.MOBILENETV2.key, "batch", 1, 0), 10.0) is None


# ---------------------------------------------------------------------------
# The probe's own output paths — a regression control
# ---------------------------------------------------------------------------


def _probe_module():
    """Import `probe_real_model_latency` without running it. It imports ORT lazily, so this is
    safe on a machine with no EP."""
    import importlib.util

    path = _BENCH / "results" / "probe_real_model_latency.py"
    spec = importlib.util.spec_from_file_location("probe_real_model_latency", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_two_passes_do_not_default_to_the_same_file():
    """REGRESSION (2026-08-07): `--diagnose` inherited the timed pass's default `--out` and
    overwrote a completed matrix with a profiling record carrying a different schema. The two
    passes measure different things; landing on one path by accident destroyed evidence that
    took thirteen minutes to produce and would have been reported as a matrix.

    Asserted on the parser's own defaults rather than by running the probe, because the failure
    was in argument resolution and that is the thing that must stay fixed.
    """
    src = (_BENCH / "results" / "probe_real_model_latency.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--out", default=None' in src, (
        "--out must not carry a pass-independent default; that is what caused the overwrite"
    )
    assert '"real_model_diagnostics.json" if args.diagnose else "real_model_latency.json"' in src


def test_the_diagnostics_pass_names_its_own_schema():
    """A file whose name says `latency` and whose schema says `diagnostics` is how the overwrite
    stayed invisible for one command. The schema string is the second, independent witness."""
    src = (_BENCH / "results" / "probe_real_model_latency.py").read_text(encoding="utf-8")
    assert '"schema": "real_model_diagnostics/1"' in src
    assert '"schema": rm.SCHEMA' in src
    assert rm.SCHEMA != "real_model_diagnostics/1"


# ---------------------------------------------------------------------------
# MiniLM, pinned identity only - issue #78
# ---------------------------------------------------------------------------
#
# The defect these arms exist against, stated once: `resolve_model` used to decide a model's
# identity by asking whether a file with the expected *name* existed in a cache directory. Any
# same-named MiniLM export - a third-party re-export, a quantised variant, a partially written
# download - satisfied that question, and the resolver then reported it as MiniLM with a
# `sha256` computed from whatever it found. Nothing in the record disagreed, because the record
# was derived from the file rather than compared against a pin.


def _minilm_cache(tmp_path, monkeypatch, blob: bytes):
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    (tmp_path / rm.MINILM.cache_filename).write_bytes(blob)
    return tmp_path


def _minilm_sidecar_dir(tmp_path, digest: str):
    results = tmp_path / "results"
    (results / "pinned-bytes").mkdir(parents=True, exist_ok=True)
    (results / rm.MINILM.recorded_provenance).write_text(
        json.dumps({"onnx_sha256": digest}), encoding="utf-8"
    )
    return results


def _tiny_onnx_bytes():
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="/enc/Identity")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model.SerializeToString()


def _pinned_spec_for(blob: bytes):
    """A MINILM spec whose pin names *these* bytes, so the happy path can be driven offline."""
    import dataclasses as _dc

    pin = dict(rm.MINILM_PIN)
    pin["sha256"] = hashlib.sha256(blob).hexdigest()
    pin["pinned_bytes"] = len(blob)
    return _dc.replace(rm.MINILM, pin=pin)


def test_minilm_is_pinned_to_an_immutable_revision_not_a_branch():
    """`main` is a name for whatever was pushed last; a 40-hex commit is a name for bytes."""
    assert rm.MINILM_PIN["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert len(rm.MINILM_PIN["revision"]) == 40
    for movable in ("main", "master", "refs/heads", "latest", "HEAD"):
        assert movable not in rm.MINILM_PIN["revision"]
    assert rm.MINILM_PIN["repo"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert rm.MINILM_PIN["file"] == "onnx/model.onnx"
    assert len(rm.MINILM_PIN["sha256"]) == 64
    assert rm.MINILM_PIN["pinned_bytes"] == 90405214


def test_the_minilm_spec_carries_no_literal_local_path():
    blob = repr(rm.MINILM)
    assert ":\\" not in blob and ":/" not in blob
    assert "Users" not in blob and "home" not in blob


def test_minilm_is_absent_from_the_timed_matrix_so_no_speed_claim_can_appear():
    """#78 asks for identity. A latency number smuggled in beside it is an unreviewed claim."""
    assert rm.MINILM.key not in rm.MODELS
    assert rm.MINILM.key in rm.PROVENANCE_ONLY
    assert rm.MINILM.key in rm.ALL_MODELS
    assert set(rm.MODELS) & set(rm.PROVENANCE_ONLY) == set()


def test_the_probe_iterates_the_timed_matrix_not_every_known_model():
    """The structural reason the previous test stays true when someone adds `--models all`."""
    src = (_BENCH / "results" / "probe_real_model_latency.py").read_text(encoding="utf-8")
    assert "rm.ALL_MODELS" not in src
    assert "rm.PROVENANCE_ONLY" not in src
    assert "rm.MODELS" in src


def test_an_absent_pinned_model_is_unavailable_not_an_unverified_pass(tmp_path, monkeypatch):
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    with pytest.raises(rm.ModelUnavailable) as exc:
        rm.resolve_model(rm.MINILM)
    assert "absent" in str(exc.value)


def test_a_same_named_file_that_is_not_the_pinned_bytes_is_refused(tmp_path, monkeypatch):
    """THE ISSUE. The filename was the whole check; this is the file that exploited it."""
    blob = _tiny_onnx_bytes()
    _minilm_cache(tmp_path, monkeypatch, blob)
    results = _minilm_sidecar_dir(tmp_path, hashlib.sha256(blob).hexdigest())
    with pytest.raises(rm.ModelUnavailable) as exc:
        rm.resolve_model(rm.MINILM, results_dir=results)
    msg = str(exc.value)
    assert "REFUSED(instrument=provenance_mismatch)" in msg
    assert "not the bytes pinned" in msg


def test_a_truncated_download_is_refused_rather_than_hashed_and_believed(
        tmp_path, monkeypatch):
    blob = _tiny_onnx_bytes()
    spec = _pinned_spec_for(blob)
    _minilm_cache(tmp_path, monkeypatch, blob[:-4])
    results = _minilm_sidecar_dir(tmp_path, spec.pin["sha256"])
    with pytest.raises(rm.ModelUnavailable):
        rm.resolve_model(spec, results_dir=results)


def test_the_pinned_bytes_resolve_and_the_record_says_so(tmp_path, monkeypatch):
    blob = _tiny_onnx_bytes()
    spec = _pinned_spec_for(blob)
    _minilm_cache(tmp_path, monkeypatch, blob)
    results = _minilm_sidecar_dir(tmp_path, spec.pin["sha256"])
    rec = rm.resolve_model(spec, results_dir=results)
    assert rec["provenance_ok"] is True
    assert rec["sha256"] == spec.pin["sha256"]
    assert rec["bytes"] == len(blob)
    assert rec["provenance"] == "pinned-immutable"
    assert rec["agrees_with_recorded_provenance"] is True


def test_a_missing_sidecar_cannot_produce_a_verified_resolution(tmp_path, monkeypatch):
    """#78: `agrees_with_recorded_provenance: None` was reported and then ignored."""
    blob = _tiny_onnx_bytes()
    spec = _pinned_spec_for(blob)
    _minilm_cache(tmp_path, monkeypatch, blob)
    empty = tmp_path / "results"
    (empty / "pinned-bytes").mkdir(parents=True)
    with pytest.raises(rm.ModelUnavailable) as exc:
        rm.resolve_model(spec, results_dir=empty)
    assert "REFUSED(instrument=" in str(exc.value)


def test_a_sidecar_that_disagrees_with_the_bytes_is_refused(tmp_path, monkeypatch):
    """Two witnesses that disagree is not one witness that agrees."""
    blob = _tiny_onnx_bytes()
    spec = _pinned_spec_for(blob)
    _minilm_cache(tmp_path, monkeypatch, blob)
    results = _minilm_sidecar_dir(tmp_path, "0" * 64)
    with pytest.raises(rm.ModelUnavailable):
        rm.resolve_model(spec, results_dir=results)


def test_the_public_record_of_a_resolution_carries_no_local_path(tmp_path, monkeypatch):
    blob = _tiny_onnx_bytes()
    spec = _pinned_spec_for(blob)
    _minilm_cache(tmp_path, monkeypatch, blob)
    results = _minilm_sidecar_dir(tmp_path, spec.pin["sha256"])
    rec = rm.resolve_model(spec, results_dir=results)
    public = json.dumps(rec["public_provenance"])
    assert str(tmp_path) not in public
    assert "AppData" not in public and "Users" not in public
    assert "huggingface.co" in public
    # ... while the caller-facing record still knows where the file is, because the caller has
    # to open it. The separation is the point: one field is published, the other is not.
    assert rec["path"].startswith(str(tmp_path))


def test_there_is_no_fallback_to_another_model_when_minilm_is_refused(tmp_path, monkeypatch):
    """A resolver that quietly returns MobileNet would publish MobileNet numbers as MiniLM."""
    blob = _tiny_onnx_bytes()
    _minilm_cache(tmp_path, monkeypatch, blob)
    (tmp_path / rm.MOBILENETV2.cache_filename).write_bytes(blob)
    results = _minilm_sidecar_dir(tmp_path, hashlib.sha256(blob).hexdigest())
    with pytest.raises(rm.ModelUnavailable):
        rm.resolve_model(rm.MINILM, results_dir=results)


def test_resolving_a_pinned_model_never_imports_onnxruntime(tmp_path, monkeypatch):
    """A check that already handed the bytes to the runtime is not deciding whether to."""
    import subprocess

    blob = _tiny_onnx_bytes()
    spec = _pinned_spec_for(blob)
    cache = _minilm_cache(tmp_path, monkeypatch, blob)
    results = _minilm_sidecar_dir(tmp_path, spec.pin["sha256"])
    script = tmp_path / "probe_import.py"
    script.write_text(
        "import sys, os, json, dataclasses\n"
        f"sys.path.insert(0, {str(_BENCH)!r})\n"
        f"os.environ[{rm.REPO_CACHE_ENV!r}] = {str(cache)!r}\n"
        "import real_model as rm\n"
        f"pin = json.loads(open({str(tmp_path / 'pin.json')!r}, encoding='utf-8').read())\n"
        "spec = dataclasses.replace(rm.MINILM, pin=pin)\n"
        f"rec = rm.resolve_model(spec, results_dir={str(results)!r})\n"
        "assert rec['provenance_ok'] is True\n"
        "print(json.dumps(sorted(m for m in sys.modules if 'onnxruntime' in m)))\n",
        encoding="utf-8",
    )
    (tmp_path / "pin.json").write_text(json.dumps(spec.pin), encoding="utf-8")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_minilm_feeds_are_int64_token_ids_not_float_activations():
    """A BERT encoder fed float32 raises at session-run time, long after the record said OK."""
    cases = rm.minilm_cases([8, 16])
    assert [c.m for c in cases] == [8, 16]
    for case in cases:
        feeds = rm.minilm_feeds(case, np)
        assert set(feeds) == {"input_ids", "attention_mask", "token_type_ids"}
        for name, arr in feeds.items():
            assert arr.dtype == np.int64, (name, arr.dtype)
            assert arr.shape == (1, case.m), (name, arr.shape)
        assert feeds["input_ids"][0][0] == 101
        assert feeds["input_ids"][0][-1] == 102
        assert (feeds["attention_mask"] == 1).all()
        assert (feeds["token_type_ids"] == 0).all()
        assert feeds["input_ids"].max() < rm.MINILM_VOCAB


def test_minilm_feeds_are_deterministic_across_calls():
    """Two runs that fed different tokens are two runs that are not comparable."""
    case = rm.minilm_cases([16])[0]
    a, b = rm.minilm_feeds(case, np), rm.minilm_feeds(case, np)
    for name in a:
        assert (a[name] == b[name]).all(), name


def test_build_feeds_dispatches_minilm_without_a_mobilenet_shaped_guess():
    case = rm.minilm_cases([8])[0]
    feeds = rm.build_feeds(case, np)
    assert set(feeds) == {"input_ids", "attention_mask", "token_type_ids"}


@pytest.mark.parametrize("lengths,label", [
    ([], "empty list - an empty matrix reads as a completed one"),
    ([0], "zero positions"),
    ([1], "one position cannot carry [CLS]...[SEP]"),
    ([-8], "negative"),
    ([16, 16], "the same length twice inflates any total over the list"),
    (["16"], "a string that int() would have happily coerced"),
    ([16.0], "a float that int() would have happily truncated"),
    ([True], "a bool, which is an int to isinstance and a mistake to a reader"),
    ([None], "None"),
    (16, "a bare int rather than an iterable of them"),
    ("16", "a string, which is iterable and would have yielded characters"),
])
def test_minilm_cases_refuses_a_shape_nobody_chose(lengths, label):
    with pytest.raises(ValueError):
        rm.minilm_cases(lengths)


def test_minilm_feeds_refuses_a_case_that_is_not_this_model_or_this_phase():
    """A feed built for the wrong model is wrong in a way the session will not catch."""
    with pytest.raises(ValueError):
        rm.minilm_feeds(rm.Case(rm.MOBILENETV2.key, "batch", 1, 0), np)
    with pytest.raises(ValueError):
        rm.minilm_feeds(rm.Case(rm.MINILM.key, "decode", 1, 128), np)
    with pytest.raises(ValueError):
        rm.minilm_feeds(rm.Case(rm.MINILM.key, "prefill", 16, 128), np)
    with pytest.raises(ValueError):
        rm.minilm_feeds(rm.Case(rm.MINILM.key, "prefill", 1, 0), np)
    with pytest.raises(ValueError):
        rm.minilm_feeds(object(), np)


# ---------------------------------------------------------------------------
# Cross-reader agreement - there must not be two provenance opinions in this repo
# ---------------------------------------------------------------------------

def test_the_bench_and_rust_provenance_readers_agree_on_the_same_file(tmp_path):
    """`rust/tools/model_provenance.py` already enforces size+digest for the differential-test
    subjects. `bench/pinned_bytes.py` enforces size+digest+sidecar+traversal for the pinned
    bench subject. Where their remits overlap - "do these bytes match this digest and size" -
    they must never disagree, or the repository holds two provenance opinions and a reader has
    to guess which one a green lane came from.
    """
    import importlib.util

    spec_path = _BENCH.parent / "rust" / "tools" / "model_provenance.py"
    spec = importlib.util.spec_from_file_location("_rust_model_provenance", spec_path)
    rust_mp = importlib.util.module_from_spec(spec)
    # Registered before exec because its `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, which does not exist for an unregistered module.
    sys.modules[spec.name] = rust_mp
    spec.loader.exec_module(rust_mp)

    sys.path.insert(0, str(_BENCH))
    import pinned_bytes as pb

    blob = _tiny_onnx_bytes()
    path = tmp_path / "m.onnx"
    path.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    side = tmp_path / "side.json"
    side.write_text(json.dumps({"onnx_sha256": digest}), encoding="utf-8")

    def pin_for(sha, nbytes):
        return {**rm.MINILM_PIN, "sha256": sha, "pinned_bytes": nbytes}

    cases = [
        (digest, len(blob), True),
        ("0" * 64, len(blob), False),
        (digest, len(blob) + 1, False),
    ]
    for sha, nbytes, expect_ok in cases:
        entry = rust_mp.ModelProvenance(
            name="x", url="https://example.invalid/m.onnx", sha256=sha, bytes=nbytes,
            fetched="2026-08-09", why="cross-reader agreement control",
        )
        rust_ok = True
        try:
            rust_mp.verify_file(path, entry)
        except rust_mp.ProvenanceMismatch:
            rust_ok = False

        bench_ok = True
        try:
            pb.check_pinned_bytes(path, pin_for(sha, nbytes), sidecar=side)
        except pb.ProvenanceError:
            bench_ok = False

        assert rust_ok is expect_ok, (sha, nbytes)
        assert bench_ok is rust_ok, (
            f"the two provenance readers disagree about {sha[:8]}/{nbytes}: "
            f"rust={rust_ok} bench={bench_ok}"
        )


def test_minilm_is_deliberately_not_in_the_rust_download_contract():
    """Stated so a reader does not read its absence as an oversight.

    `bench/results/model_provenance.json` is the *download* contract for differential-testing
    subjects: `rust/modelrunner` fetches from its `url` and `tests/ops/` runs CPU-vs-Vulkan
    agreement over them. MiniLM has no agreement lane here and is not fetched by that runner -
    it is resolved from a HuggingFace revision by `bench/real_model.py`. Adding it would make
    the runner and `ci/check_verification_subjects.py` claim a subject with no verification
    behind it, which is the exact species of false completeness #78 is about.
    """
    contract = json.loads(
        (_BENCH / "results" / "model_provenance.json").read_text(encoding="utf-8")
    )
    names = {m["name"] for m in contract["models"]}
    assert rm.MINILM.key not in names
    # ...and the pin still has a home that is read programmatically, so this is a routing
    # decision and not a gap: the bench resolver reads MINILM_PIN on every resolution.
    assert rm.MINILM.pin is rm.MINILM_PIN


# ---------------------------------------------------------------------------
# The real cached bytes, when this machine has them
# ---------------------------------------------------------------------------

def test_the_real_pinned_minilm_verifies_when_it_is_present():
    """Not a skip-shaped pass: when the file is here, this is the whole gate end to end."""
    path = rm.repo_cache_dir() / rm.MINILM.cache_filename
    if not path.is_file():
        pytest.skip(f"the pinned MiniLM is not in this machine's model cache ({path.name})")
    rec = rm.resolve_model(rm.MINILM)
    assert rec["provenance_ok"] is True
    assert rec["sha256"] == rm.MINILM_PIN["sha256"]
    assert rec["bytes"] == rm.MINILM_PIN["pinned_bytes"]
    assert rec["external_data"]["scanned"] is True
    assert rec["external_data"]["files"] == []


def test_the_shipped_sidecar_records_the_pinned_digest_and_size():
    """The committed witness must name the same bytes the code pins, or one of them is stale."""
    side = json.loads(
        (_BENCH / "results" / rm.MINILM.recorded_provenance).read_text(encoding="utf-8")
    )
    assert side["onnx_sha256"] == rm.MINILM_PIN["sha256"]
    assert side["onnx_bytes"] == rm.MINILM_PIN["pinned_bytes"]
    assert side["pin"]["revision"] == rm.MINILM_PIN["revision"]
    assert side["pin"]["repo"] == rm.MINILM_PIN["repo"]
    assert side["pin"]["file"] == rm.MINILM_PIN["file"]
    assert rm.MINILM_PIN["revision"] in side["pin"]["url"]
    assert side["external_data"]["declared_files"] == 0


def test_external_data_provenance_reaches_a_declaration_hidden_in_a_subgraph(tmp_path):
    """Kills `edp-partial-walk-again`.

    This is the pre-#78 defect itself, asserted against the shipped adapter rather than
    against the module it delegates to: a model whose only external weight is declared
    inside an `If` branch used to be reported as `scanned: true, files: []`, which reads
    exactly like "this model has no external weights".
    """
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    def ext(name):
        t = helper.make_tensor(name, TensorProto.FLOAT, [1], [0.0])
        t.ClearField("float_data")
        t.data_location = TensorProto.EXTERNAL
        kv = t.external_data.add()
        kv.key, kv.value = "location", "w.bin"
        return t

    def sub(inits=()):
        return helper.make_graph(
            [helper.make_node("Identity", ["x"], ["sy"])], "sub", [],
            [helper.make_tensor_value_info("sy", TensorProto.FLOAT, [1])],
            initializer=list(inits))

    node = helper.make_node("If", ["x"], ["y"], then_branch=sub([ext("hidden")]),
                            else_branch=sub())
    graph = helper.make_graph(
        [node], "g", [helper.make_tensor_value_info("x", TensorProto.BOOL, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())
    (tmp_path / "w.bin").write_bytes(b"weights that are not in the .onnx")

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert rec["complete"] is True
    assert [f["location"] for f in rec["files"]] == ["w.bin"], rec
    assert rec["files"][0]["bytes"] == len(b"weights that are not in the .onnx")
    assert rec["files"][0]["sha256"]


def test_external_data_provenance_reports_a_missing_blob_rather_than_an_empty_list(tmp_path):
    """`files: []` and "the declared weights are not on this disk" must not look alike."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    t = helper.make_tensor("w", TensorProto.FLOAT, [1], [0.0])
    t.ClearField("float_data")
    t.data_location = TensorProto.EXTERNAL
    kv = t.external_data.add()
    kv.key, kv.value = "location", "absent.bin"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])], initializer=[t])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["files"] == [{"location": "absent.bin", "bytes": 0, "sha256": None,
                             "missing": True}]


def test_external_data_provenance_refuses_an_unsafe_declaration_rather_than_ignoring_it(
        tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    t = helper.make_tensor("w", TensorProto.FLOAT, [1], [0.0])
    t.ClearField("float_data")
    t.data_location = TensorProto.EXTERNAL
    kv = t.external_data.add()
    kv.key, kv.value = "location", "../../escape.bin"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])], initializer=[t])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is False
    assert rec["complete"] is False
    assert "REFUSED(instrument=external_unsafe)" in rec["reason"]
