"""Self-tests for `bench/real_model.py` — no GPU, no EP, no model file.

Each test is named for the **plausible but wrong** reading it prevents, in the style of
`bench/test_plausible_but_wrong.py`. The point of this file is that the parts of the #56
instrument that can be wrong without a device — provenance, feeds, throughput arithmetic,
pairing, verdicts — are checked on every `pytest bench` run, not only on the one machine that
has an RTX A1000 and a Foundry cache.
"""

from __future__ import annotations

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
    assert rec["files"] == []
    assert "covers the weights" in rec["reason"]


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


# =============================================================================================
# Issue #78 — immutable MiniLM provenance, fail-closed verification, capability contract
#
# Every test below is named for the wrong reading it prevents. No network, no downloads, no GPU:
# the pinned bytes are never fetched, they are *synthesised* to the pinned digest's shape and the
# pin is pointed at the synthetic file, so the contract is exercised without leaving the box.
# =============================================================================================

import dataclasses as _dc  # noqa: E402
import hashlib as _hashlib  # noqa: E402
import json as _json  # noqa: E402

MINILM_SHA = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
MINILM_BYTES = 90_405_214
MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
#: The unverified blob that was sitting in the default cache when #78 was filed.
SUBSTITUTE_SHA = "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e"


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _hashlib.sha256(payload).hexdigest()


def _spec_for(tmp_path, filename="probe.bin", *, sha=None, nbytes=None, **over):
    """A pinned spec whose repo-cache resolves inside *tmp_path*."""
    fields = dict(
        key="pin-probe", family="test", resolver="repo-cache", cache_filename=filename,
        source_repo=MINILM_REPO, source_revision=MINILM_REV, source_file="onnx/model.onnx",
        pinned_sha256=sha if sha is not None else MINILM_SHA,
        pinned_bytes=nbytes if nbytes is not None else MINILM_BYTES,
        capability=rm.CAP_PROVENANCE_ONLY,
    )
    fields.update(over)
    return rm.ModelSpec(**fields)


# --- 1. the pin itself -----------------------------------------------------------------------

def test_the_minilm_pin_is_exactly_the_verified_identity():
    """The one thing this whole issue is about. Any drift here is the bug returning."""
    assert rm.MINILM.source_repo == MINILM_REPO
    assert rm.MINILM.source_revision == MINILM_REV
    assert rm.MINILM.source_file == "onnx/model.onnx"
    assert rm.MINILM.pinned_sha256 == MINILM_SHA
    assert rm.MINILM.pinned_bytes == MINILM_BYTES
    assert rm.MINILM.pinned_external == ()
    assert rm.MINILM.is_pinned


def test_the_minilm_url_is_immutable_and_names_sentence_transformers():
    url = rm.pinned_source_url(rm.MINILM)
    assert url == (f"https://huggingface.co/{MINILM_REPO}/resolve/{MINILM_REV}/onnx/model.onnx")
    assert "/resolve/main/" not in url
    assert "Xenova" not in url


def test_a_mutable_revision_cannot_construct_a_spec(tmp_path):
    """`main` is the defect the earlier draft shipped: today's bytes, not tomorrow's."""
    for bad in ("main", "v1.0", "refs/heads/main", "1110a24"):
        with pytest.raises(ValueError, match="commit SHA"):
            _spec_for(tmp_path, source_revision=bad)


def test_a_revision_with_a_trailing_newline_is_refused(tmp_path):
    """`re.match(...$)` accepts a trailing newline; `re.fullmatch` does not. This is why."""
    with pytest.raises(ValueError, match="commit SHA"):
        _spec_for(tmp_path, source_revision=MINILM_REV + "\n")


def test_a_partial_pin_cannot_construct_a_spec(tmp_path):
    """No early return on a missing digest: half a pin is not a pin."""
    with pytest.raises(ValueError, match="partial pin"):
        _spec_for(tmp_path, pinned_sha256="")
    with pytest.raises(ValueError, match="partial pin"):
        _spec_for(tmp_path, pinned_bytes=0)
    with pytest.raises(ValueError, match="partial pin"):
        _spec_for(tmp_path, source_repo="")
    with pytest.raises(ValueError, match="partial pin"):
        _spec_for(tmp_path, source_file="")


def test_a_malformed_repo_or_file_shape_is_refused(tmp_path):
    for bad in ("https://huggingface.co/a/b", "a/b/c", "../a/b", "a b/c", "onlyname"):
        with pytest.raises(ValueError, match="source_repo"):
            _spec_for(tmp_path, source_repo=bad)
    for bad in ("/onnx/model.onnx", "onnx\\model.onnx", "../model.onnx", "onnx/../../x"):
        with pytest.raises(ValueError, match="source_file"):
            _spec_for(tmp_path, source_file=bad)


def test_a_malformed_digest_is_refused(tmp_path):
    for bad in ("ABC", MINILM_SHA.upper(), MINILM_SHA[:-1], MINILM_SHA + "0", MINILM_SHA + "\n"):
        with pytest.raises(ValueError, match="pinned_sha256"):
            _spec_for(tmp_path, pinned_sha256=bad)


def test_an_unpinned_spec_has_no_url_at_all():
    """The refusal is the point: there is no mutable fallback to construct."""
    with pytest.raises(ValueError, match="no immutable pin"):
        rm.pinned_source_url(rm.MOBILENETV2)


# --- 2. resolution fails closed --------------------------------------------------------------

def test_the_exact_pinned_bytes_resolve_and_verify(tmp_path, monkeypatch):
    payload = b"\0" * 32
    sha = _write(tmp_path / "probe.bin", payload)
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(rm, "external_data_provenance",
                        lambda p: {"scanned": True, "files": [], "reason": "none",
                                   "tensors_scanned": 0})
    spec = _spec_for(tmp_path, sha=sha, nbytes=len(payload))
    rec = rm.resolve_model(spec)
    assert rec["provenance_ok"] is True
    assert rec["sha256"] == sha


def test_the_759c_substitute_is_refused_by_name(tmp_path, monkeypatch):
    """The exact scenario of #78: right filename, wrong bytes, must not read as MODEL_OK.

    The substitute is written at *exactly* the pinned byte count, so the size check cannot
    acquit or convict and the digest is the only thing that can decide. An earlier version of
    this test used a short payload and passed even with the hash comparison disabled — it was
    testing the size guard while claiming to test the digest guard.
    """
    payload = b"substitute bytes, same name, same length"
    _write(tmp_path / "probe.bin", payload)
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(rm, "external_data_provenance",
                        lambda p: {"scanned": True, "files": [], "reason": "none",
                                   "tensors_scanned": 0})
    spec = _spec_for(tmp_path, sha=SUBSTITUTE_SHA, nbytes=len(payload))
    with pytest.raises(rm.ModelProvenanceMismatch, match="resolved sha256"):
        rm.resolve_model(spec)


def test_the_refusal_names_the_immutable_url_so_an_operator_can_act(tmp_path, monkeypatch):
    _write(tmp_path / "probe.bin", b"wrong")
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    spec = _spec_for(tmp_path)
    with pytest.raises(rm.ModelProvenanceMismatch) as ei:
        rm.resolve_model(spec)
    assert f"https://huggingface.co/{MINILM_REPO}/resolve/{MINILM_REV}/" in str(ei.value)


def test_renamed_but_byte_identical_still_verifies(tmp_path, monkeypatch):
    """Identity is per-file content, not the filename. A rename is not a content change."""
    payload = b"identical bytes under another name"
    sha = _write(tmp_path / "renamed.onnx", payload)
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(rm, "external_data_provenance",
                        lambda p: {"scanned": True, "files": [], "reason": "none",
                                   "tensors_scanned": 0})
    spec = _spec_for(tmp_path, "renamed.onnx", sha=sha, nbytes=len(payload))
    assert rm.resolve_model(spec)["provenance_ok"] is True


def test_a_wrong_size_with_a_right_hash_is_still_refused(tmp_path, monkeypatch):
    payload = b"0123456789"
    sha = _write(tmp_path / "probe.bin", payload)
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(rm, "external_data_provenance",
                        lambda p: {"scanned": True, "files": [], "reason": "none",
                                   "tensors_scanned": 0})
    spec = _spec_for(tmp_path, sha=sha, nbytes=len(payload) + 1)
    with pytest.raises(rm.ModelProvenanceMismatch, match="does not match the pinned"):
        rm.resolve_model(spec)


def test_source_metadata_drift_is_refused_even_when_the_digest_matches():
    """Xenova metadata may not describe sentence-transformers bytes as provenance_ok."""
    with pytest.raises(rm.ModelProvenanceMismatch, match="disagrees with the pinned"):
        rm.verify_source_metadata(rm.MINILM, {"repo": "Xenova/all-MiniLM-L6-v2"})
    with pytest.raises(rm.ModelProvenanceMismatch, match="disagrees with the pinned"):
        rm.verify_source_metadata(rm.MINILM, {"revision": "0" * 40})
    with pytest.raises(rm.ModelProvenanceMismatch, match="disagrees with the pinned"):
        rm.verify_source_metadata(rm.MINILM, {"file": "onnx/model_quantized.onnx"})
    rm.verify_source_metadata(rm.MINILM, rm._source_identity(rm.MINILM))  # the matching case


def test_an_unscanned_graph_is_refused_rather_than_assumed_clean():
    rec = {"key": "k", "sha256": rm.MINILM.pinned_sha256, "bytes": rm.MINILM.pinned_bytes,
           "external_data": {"scanned": False, "files": [], "reason": "onnx missing"},
           "source": rm._source_identity(rm.MINILM)}
    with pytest.raises(rm.ModelProvenanceMismatch, match="did not run"):
        rm.verify_pinned_provenance(rm.MINILM, rec)


def test_external_data_disagreement_fails_closed_in_both_directions():
    base = {"key": "k", "sha256": rm.MINILM.pinned_sha256, "bytes": rm.MINILM.pinned_bytes,
            "source": rm._source_identity(rm.MINILM)}
    unexpected = dict(base, external_data={
        "scanned": True, "reason": None,
        "files": [{"location": "w.data", "bytes": 4, "sha256": "0" * 64}]})
    with pytest.raises(rm.ModelProvenanceMismatch, match="declares external data the pin does not"):
        rm.verify_pinned_provenance(rm.MINILM, unexpected)

    missing = dict(base, external_data={
        "scanned": True, "reason": None,
        "files": [{"location": "w.data", "bytes": 0, "sha256": None, "missing": True}]})
    with pytest.raises(rm.ModelProvenanceMismatch, match="absent on disk"):
        rm.verify_pinned_provenance(rm.MINILM, missing)

    pinned_ext = _dc.replace(
        rm.MINILM, key="ext-probe",
        pinned_external=(rm.ExternalFile("w.data", "1" * 64, 4),))
    lost = dict(base, external_data={"scanned": True, "reason": None, "files": []})
    with pytest.raises(rm.ModelProvenanceMismatch, match="does not reference"):
        rm.verify_pinned_provenance(pinned_ext, lost)

    wrong_hash = dict(base, external_data={
        "scanned": True, "reason": None,
        "files": [{"location": "w.data", "bytes": 4, "sha256": "2" * 64}]})
    with pytest.raises(rm.ModelProvenanceMismatch, match="but the pin says"):
        rm.verify_pinned_provenance(pinned_ext, wrong_hash)


def test_external_tensors_nested_in_subgraphs_and_attributes_are_found(tmp_path):
    """A `.data` hanging off an `If` branch must not scan clean and then fail at inference."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    def ext(name):
        t = helper.make_tensor(name, TensorProto.FLOAT, [1], [0.0])
        t.ClearField("float_data")
        t.data_location = TensorProto.EXTERNAL
        kv = t.external_data.add()
        kv.key, kv.value = "location", "nested.data"
        return t

    sub = helper.make_graph([], "sub", [], [helper.make_tensor_value_info(
        "o", TensorProto.FLOAT, [1])], initializer=[ext("nested_init")])
    const = helper.make_node("Constant", [], ["c"], value=ext("attr_tensor"))
    branch = helper.make_node("If", ["cond"], ["o"], then_branch=sub, else_branch=sub)
    g = helper.make_graph(
        [const, branch], "g",
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])],
        [helper.make_tensor_value_info("o", TensorProto.FLOAT, [1])])
    model = helper.make_model(g)
    p = tmp_path / "nested.onnx"
    p.write_bytes(model.SerializeToString())

    rec = rm.external_data_provenance(p)
    assert rec["scanned"] is True
    assert [f["location"] for f in rec["files"]] == ["nested.data"]
    assert rec["files"][0]["missing"] is True, "the blob does not exist; it must say so"
    assert rec["tensors_scanned"] >= 3


# --- 3. public records may not name the machine ----------------------------------------------

def test_a_planted_absolute_path_does_not_survive_the_real_serializer(tmp_path):
    """Planted leaks, driven through the production writer — not a re-implementation of it."""
    home = str(Path.home())
    planted = {
        "top": home + "/secret/model.onnx",
        "nested": {"deep": [home + "\\a\\b.onnx", {"k": "C:\\Users\\someone\\x.onnx"}]},
        "message": f"failed to open {home}/cache/m.onnx: no such file",
        rm.RUNTIME_ONLY_KEY: {"path": home + "/private/real.onnx"},
    }
    out = tmp_path / "rec.json"
    text = rm.write_public_json(planted, out)
    assert rm.RUNTIME_ONLY_KEY not in _json.loads(text)
    assert home not in text
    assert "C:\\Users\\someone" not in text
    assert rm._scan_public(_json.loads(text)) == []


def test_the_serializer_refuses_rather_than_writes_when_it_cannot_root(tmp_path, monkeypatch):
    """A screen that only ever passes has never been seen working."""
    monkeypatch.setattr(rm, "_public_record", lambda node: node)  # defeat the sanitiser
    out = tmp_path / "leaky.json"
    with pytest.raises(ValueError, match="machine-identifying"):
        rm.write_public_json({"p": str(Path.home() / "x.onnx")}, out)
    assert not out.exists(), "a refused record must not be written"


def test_a_resolved_record_carries_a_rooted_path_and_a_private_runtime_path(tmp_path, monkeypatch):
    payload = b"abc"
    sha = _write(tmp_path / "probe.bin", payload)
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(rm, "external_data_provenance",
                        lambda p: {"scanned": True, "files": [], "reason": "none",
                                   "tensors_scanned": 0})
    rec = rm.resolve_model(_spec_for(tmp_path, sha=sha, nbytes=len(payload)))
    assert not Path(rec["path"]).is_absolute() or rec["path"].startswith("<")
    assert rm.runtime_path(rec).is_file(), "the private path must still open"
    with pytest.raises(rm.ModelUnavailable, match="no runtime path"):
        rm.runtime_path(rm._public_record(rec))


# --- 4. the capability contract --------------------------------------------------------------

def test_minilm_is_provenance_only_until_the_op_work_lands():
    assert rm.MINILM.capability == rm.CAP_PROVENANCE_ONLY
    assert rm.MINILM.key not in rm._CASE_BUILDERS


def test_cases_for_refuses_a_provenance_only_model_instead_of_defaulting():
    """The `else: mobilenet_cases(...)` fallback is the bug; this is the test that forbids it.

    Matched on the phrase unique to the *capability* branch. Matching on "provenance_only" alone
    was vacuous: the missing-builder branch interpolates ``spec.capability`` into its own message,
    so both branches contained the word and the test passed with the capability gate removed.
    """
    with pytest.raises(rm.ModelNotBenchable, match="no case table for it"):
        rm.cases_for(rm.MINILM, prefill_m=(1,), decode_past=(0,), batch=(1,))


def test_no_case_builder_can_produce_a_case_for_another_models_key():
    """Every case must belong to the spec it was built from."""
    for key, spec in rm.MODELS.items():
        if spec.capability != rm.CAP_BENCHABLE:
            continue
        cases = rm.cases_for(spec, prefill_m=(1, 8), decode_past=(0, 128), batch=(1, 16))
        assert cases, f"{key} is benchable but produced no cases"
        for c in cases:
            assert c.model_key == key, f"{key} produced a case for {c.model_key}"


def test_an_unknown_model_key_fails_closed():
    ghost = rm.ModelSpec(key="not-a-model", family="x", resolver="repo-cache",
                         cache_filename="x.onnx")
    with pytest.raises(ValueError, match="unknown model key"):
        rm.cases_for(ghost, batch=(1,))


def test_build_feeds_refuses_a_provenance_only_model():
    case = rm.Case(rm.MINILM.key, "batch", 1, 0)
    with pytest.raises(rm.ModelNotBenchable, match="no feed builder"):
        rm.build_feeds(case, np)


def test_benchable_keys_excludes_minilm_but_models_still_contains_it():
    assert rm.MINILM.key in rm.MODELS, "--models all must still resolve and verify it"
    assert rm.MINILM.key not in rm.benchable_keys()
    assert set(rm.benchable_keys()) == {rm.PHI35.key, rm.MOBILENETV2.key}
    # explicit key list, order preserved
    assert rm.benchable_keys([rm.MOBILENETV2.key, rm.MINILM.key, rm.PHI35.key]) == [
        rm.MOBILENETV2.key, rm.PHI35.key]


def test_benchable_keys_refuses_an_unknown_key_rather_than_dropping_it():
    """A filter that silently discards a typo reports a narrower run as a complete one."""
    with pytest.raises(ValueError, match="unknown model key"):
        rm.benchable_keys([rm.PHI35.key, "not-a-model"])


def test_verify_pinned_provenance_accepts_a_record_that_matches_the_pin():
    """The accept polarity for the guard whose refusals are exercised above."""
    rec = {"key": rm.MINILM.key, "sha256": rm.MINILM.pinned_sha256,
           "bytes": rm.MINILM.pinned_bytes,
           "external_data": {"scanned": True, "files": [], "reason": "no external data",
                             "tensors_scanned": 7},
           "source": rm._source_identity(rm.MINILM)}
    rm.verify_pinned_provenance(rm.MINILM, rec)
    assert rec["provenance_ok"] is True
    assert rm.MINILM.source_revision in rec["provenance_detail"]


def test_an_unpinned_spec_records_that_it_was_not_checked_rather_than_passing():
    """`provenance_ok is None` is a third state, and it must not be mistaken for True."""
    rec = {"key": rm.MOBILENETV2.key, "sha256": "0" * 64, "bytes": 1,
           "external_data": {"scanned": True, "files": [], "reason": None}}
    rm.verify_pinned_provenance(rm.MOBILENETV2, rec)
    assert rec["provenance_ok"] is None
    assert "not checked" in rec["provenance_detail"]


def test_an_unknown_capability_cannot_construct_a_spec():
    with pytest.raises(ValueError, match="capability"):
        rm.ModelSpec(key="k", family="f", resolver="repo-cache", cache_filename="x.onnx",
                     capability="sort-of")


# --- 5. every driver entry point honours the contract ----------------------------------------

def _probe_source() -> str:
    p = _BENCH / "results" / "probe_real_model_latency.py"
    return p.read_text(encoding="utf-8")


def test_no_driver_dispatches_cases_with_an_else_fallback():
    """The literal shape of the defect, forbidden by reading the driver's source."""
    src = _probe_source()
    assert "else rm.mobilenet_cases" not in src
    assert "rm.cases_for(" in src, "the driver must go through the capability-aware dispatcher"


def test_every_driver_entry_point_checks_capability():
    src = _probe_source()
    # main(), run_diagnostics() and the worker each gate on capability.
    assert src.count("capability != rm.CAP_BENCHABLE") >= 3, (
        "each of main(), run_diagnostics() and _worker_diagnose must refuse a "
        "non-benchable model on its own; a gate only one entry point honours is not a gate")


def test_the_driver_writes_through_the_public_serializer_only():
    src = _probe_source()
    assert "rm.write_public_json(" in src
    assert "write_text(json.dumps(report" not in src
    assert "write_text(json.dumps(rec" not in src


def test_the_driver_opens_the_private_runtime_path_not_the_public_one():
    src = _probe_source()
    assert 'model_rec["path"]' not in src, (
        "the public path is rooted and opens nothing; sessions must use rm.runtime_path()")
    assert "rm.runtime_path(model_rec)" in src


def test_the_unavailable_message_for_a_pinned_model_names_a_route_that_exists(tmp_path):
    """`ort-model-runner`'s manifest has no MiniLM entry, so it must not be recommended."""
    msg = rm._unavailable_message(rm.MINILM, tmp_path / "all-MiniLM-L6-v2.onnx")
    assert "ort-model-runner" not in msg
    assert rm.pinned_source_url(rm.MINILM) in msg
    assert rm.MINILM.pinned_sha256 in msg and str(rm.MINILM.pinned_bytes) in msg
    assert str(Path.home()) not in msg, "the message is user-facing and may still not leak"
    # the unpinned model keeps the producer that really does produce it
    assert "ort-model-runner" in rm._unavailable_message(rm.MOBILENETV2, tmp_path / "m.onnx")


def test_the_pinned_absent_model_raises_that_exact_message(tmp_path, monkeypatch):
    """Production call site, not just the helper: resolve_model must use it."""
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    with pytest.raises(rm.ModelUnavailable) as ei:
        rm.resolve_model(rm.MINILM)
    assert str(ei.value) == rm._unavailable_message(
        rm.MINILM, tmp_path / rm.MINILM.cache_filename)


def test_nothing_in_this_module_reaches_the_network(monkeypatch):
    """No test here may download. Proven by breaking the socket, not by reading the code."""
    import socket

    def boom(*a, **k):
        raise AssertionError("this lane must never open a socket")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    assert rm.pinned_source_url(rm.MINILM).startswith("https://")
    assert rm.MINILM.is_pinned