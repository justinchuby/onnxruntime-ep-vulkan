"""§8.9.24(4) model-scale answer: the gates on `probe_criterion10_chain.py`.

The probe answers which side is further from the true value of the WHOLE model. This file
is what makes that answer quotable:

  * the liveness bar must go red on every way a layer can contribute nothing -- because
    "a layer-at-a-time pass that silently produces nothing for a layer would report a
    clean chain", and the round before this one documented the failure mode where a
    **wrong residual looks sound** (`np.spacing` returning `inf` at fp16's maximum);
  * the seam check must go red on a chain reseeded from an EP tap, which is round 36's
    arm F and would decide the answer by construction;
  * the emitted records must carry all of it, and deliberate corruptions of those records
    must be caught rather than read past;
  * `assert_record_proposes_no_motion` must hold on every emitted record, because
    §8.9.24(4) permits this answer and forbids what follows from it.

No GPU and no model are needed here. The records are the artifacts of runs that had both.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench" / "results"))

import probe_criterion10_chain as chain  # noqa: E402
from probe_criterion10_side import (  # noqa: E402
    MotionInRecordError,
    assert_record_proposes_no_motion,
)

RECORDS = sorted((REPO / "bench" / "results").glob("criterion10_chain-dev*.json"))


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _records():
    return [(p, _load(p)) for p in RECORDS]


# ==========================================================================================
# The probe's own selftest, as a test
# ==========================================================================================
def test_probe_selftest_passes():
    assert chain._selftest() == 0


# ==========================================================================================
# The liveness bar -- the load-bearing part
# ==========================================================================================
def _layers(n=4, **over):
    ls = [chain._fake_layer(i) for i in range(n)]
    if over:
        ls[-1] = chain._fake_layer(n - 1, **over)
    return ls


def test_liveness_bar_passes_a_live_chain():
    chain.assert_every_layer_live(_layers(4), expect=4)


@pytest.mark.parametrize(
    "kill",
    [
        {"nodes_evaluated": 10},
        {"residual_changed": False},
        {"delta_is_nonzero": False},
        {"delta_out_l2": 0.0},
        {"all_finite": False},
        {"weight_witnesses": {"qkv": {"rows_dequantised": 0, "rows_expected": 9216,
                                      "weight_is_live": False}}},
        {"weight_witnesses": {"qkv": {"rows_dequantised": 4096, "rows_expected": 9216,
                                      "weight_is_live": True}}},
        {"local_liveness": {"cpu": {"live": False, "median_ulp": 900.0},
                            "vulkan": {"live": True, "median_ulp": 0.0}}},
    ],
    ids=[
        "short-node-count", "residual-unchanged", "zero-delta", "zero-delta-norm",
        "non-finite", "dead-weight", "partial-dequant", "local-bar-red",
    ],
)
def test_liveness_bar_refuses_a_layer_that_contributed_nothing(kill):
    with pytest.raises(chain.DeadLayerError):
        chain.assert_every_layer_live(_layers(4, **kill), expect=4)


def test_liveness_bar_refuses_an_absent_layer():
    """The specific failure a layer-at-a-time pass has that a dense one does not: a layer
    that never appears sums 0 into the residual stream and every layer after it still
    produces plausible numbers."""
    with pytest.raises(chain.DeadLayerError, match="absent from the record"):
        chain.assert_every_layer_live(_layers(4)[:3], expect=4)


def test_dead_weight_raises_inside_the_arithmetic_not_at_the_gate():
    """Defence in depth: the gate is the backstop, but a weight that dequantises to zero
    must stop the layer where it happens, not be summed as a clean 0 and audited later."""
    packed = np.full((4, 1, 16), 0x88, dtype=np.uint8)  # every nibble 8 -> value 0
    scales = np.ones((4, 1), dtype=np.float32)
    with pytest.raises(chain.DeadLayerError):
        chain.matmulnbits_f64([np.ones(32)], packed, scales, n=4, k=32, block_size=32, bits=4)


# ==========================================================================================
# The seam -- where arm F would come back
# ==========================================================================================
def test_seam_check_refuses_a_chain_reseeded_from_an_ep_tap():
    with pytest.raises(chain.ChainReseededError, match="arm F"):
        chain.assert_chain_never_reseeded(_layers(4), {"vulkan": {2: "digest2"}})


def test_seam_check_passes_a_chain_that_matches_neither_side():
    chain.assert_chain_never_reseeded(_layers(4), {"cpu": {0: "x"}, "vulkan": {1: "y"}})


def test_position_zero_attention_does_not_consult_q():
    """The chain's attention step claims the softmax over one key is exactly 1, so the
    output is the V slice and Q is irrelevant. If that ever stops holding, every layer of
    the chain is wrong and the local liveness bar is the only thing that would notice."""
    rng = np.random.default_rng(11)
    qkv = rng.normal(size=3 * 4 * 8)
    args = dict(heads=4, kv_heads=4, do_rotary=0, interleaved=0)
    a, k, v = chain.gqa_position0_f64(qkv, np.zeros(0), np.zeros(0), **args)
    qkv2 = qkv.copy()
    qkv2[: 4 * 8] *= -7.0
    a2, k2, v2 = chain.gqa_position0_f64(qkv2, np.zeros(0), np.zeros(0), **args)
    assert np.array_equal(a, a2) and np.array_equal(k, k2) and np.array_equal(v, v2)
    assert np.array_equal(a, qkv[2 * 4 * 8 :])


# ==========================================================================================
# The aggregation defect this probe's first real run exposed in its own code
# ==========================================================================================
def test_a_decisive_reference_variant_cannot_speak_for_a_conflicted_one():
    """Found by reading the first real run, not by a test: f64's discriminators conflicted
    on output 64 and f16r said "cpu", and the record printed direction='cpu' with
    variants_agree_on_direction=True. That is the single-discriminator defect one level up.
    """
    assert chain.direction_across_variants({"f64": None, "f16r": "cpu"})["direction"] is None
    assert chain.direction_across_variants({"f64": "vulkan", "f16r": "cpu"})["direction"] is None
    assert chain.direction_across_variants({"f64": "cpu", "f16r": "cpu"})["direction"] == "cpu"


# ==========================================================================================
# The emitted records
# ==========================================================================================
@pytest.mark.skipif(not RECORDS, reason="no criterion10_chain record on this tree")
@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_chain_is_complete_and_every_layer_is_live(path):
    rec = _load(path)
    c = rec["arm_chain"]
    assert c["status"] == "MEASURED", c.get("gate_errors")
    assert c["layers_run"] == chain.N_LAYERS
    assert c["nodes_evaluated_total"] == c["nodes_expected_total"] == chain.NODES_TOTAL == 355
    chain.assert_every_layer_live(c["per_layer"], expect=chain.N_LAYERS)
    assert c["seam_check"].startswith("PASS")
    assert c["embedding"]["reference_embedding_matches_the_initialiser_row"] is True
    assert c["rotary_caches"]["cos_bit_equal_between_eps"] is True
    assert c["rotary_caches"]["sin_bit_equal_between_eps"] is True


@pytest.mark.skipif(not RECORDS, reason="no criterion10_chain record on this tree")
@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_is_screened_on_dispatches_and_claims(path):
    """Switch's ctx-4096 finding: a silent CPU rebuild exits 0, executes zero dispatches
    and reads as a clean EP run. Every arm names its device off the run."""
    rec = _load(path)
    for arm in ("arm_untapped", "arm_tapped"):
        a = rec[arm]
        assert a["status"] == "MEASURED", a
        assert a["dispatch_screen"] == "PASS", a
        assert str(a["attribution"]).startswith("ATTRIBUTED"), a
        assert a["dispatches_executed_this_arm"] > 0
        assert a["device_name"] != "UNOBSERVABLE"


@pytest.mark.skipif(not RECORDS, reason="no criterion10_chain record on this tree")
@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_the_verdict_is_computed_on_the_untapped_run(path):
    """131 extra graph outputs could have changed what the EPs executed. The answer is
    computed on the graph criterion 10 actually runs; the taps only feed the liveness bar,
    and the record has to show the two runs agreed bit for bit on the three outputs."""
    rec = _load(path)
    perturb = rec["arm_tapped"]["tapping_is_non_perturbative"]
    assert set(perturb) == {"logits", "present.31.key", "present.31.value"}
    for nm, v in perturb.items():
        assert v["cpu_tapped_equals_untapped"], nm
        assert v["vulkan_tapped_equals_untapped"], nm


@pytest.mark.skipif(not RECORDS, reason="no criterion10_chain record on this tree")
@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_every_answer_reports_both_reference_variants_and_all_discriminators(path):
    rec = _load(path)
    ans = rec["the_answer"]
    assert set(ans) == {
        "output_0_logits", "output_63_present.31.key", "output_64_present.31.value"
    }
    for label, a in ans.items():
        assert a["status"] == "MEASURED", label
        assert set(a["by_reference_variant"]) == {"f64", "f16r"}, label
        for v, pv in a["by_reference_variant"].items():
            assert set(pv["verdict_by_discriminator"]) == set(
                __import__("probe_criterion10_side").DISCRIMINATORS
            ), (label, v)
        # a direction may only be quoted when BOTH variants are unanimous and agree
        if a["direction"] is not None:
            assert a["variants_agree_on_direction"] is True
            assert a["variants_without_a_direction"] == []
            for pv in a["by_reference_variant"].values():
                assert pv["unanimous_direction"] == a["direction"]


@pytest.mark.skipif(not RECORDS, reason="no criterion10_chain record on this tree")
@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_proposes_no_motion(path):
    assert_record_proposes_no_motion(_load(path))


@pytest.mark.skipif(not RECORDS, reason="no criterion10_chain record on this tree")
def test_the_no_motion_gate_can_still_refuse_this_record():
    """A gate that has never gone red witnesses nothing. §8.9.24(4)'s risk is specific: an
    argument in the direction of loosening, quoting a measured distance as an allowance."""
    rec = _load(RECORDS[0])
    for key in ("proposed_atol", "tolerance_budget", "should_loosen"):
        bad = copy.deepcopy(rec)
        bad["the_answer"][key] = 1.0
        with pytest.raises(MotionInRecordError):
            assert_record_proposes_no_motion(bad)


@pytest.mark.skipif(len(RECORDS) < 2, reason="need two devices")
def test_the_two_devices_agree_on_every_direction_they_state():
    """Device name is off the run, not off the selector. A direction that holds on one
    device and reverses on the other is a device finding, not a which-side finding."""
    recs = [_load(p) for p in RECORDS]
    names = {r["arm_untapped"]["device_name"] for r in recs}
    assert len(names) == len(recs), f"two records, one device: {names}"
    for label in recs[0]["the_answer"]:
        dirs = {r["arm_untapped"]["device_name"]: r["the_answer"][label]["direction"]
                for r in recs}
        stated = {d for d in dirs.values() if d is not None}
        assert len(stated) <= 1, f"{label}: devices disagree on direction: {dirs}"


@pytest.mark.skipif(not RECORDS, reason="no criterion10_chain record on this tree")
@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_corrupting_a_record_is_caught_rather_than_read_past(path):
    """Nine deliberate corruptions. The point of the gate is that a record which LOOKS
    clean is not trusted on that basis."""
    rec = _load(path)
    per = rec["arm_chain"]["per_layer"]
    corruptions = [
        ("layer 17 vanishes", lambda r: r["arm_chain"]["per_layer"].pop(17)),
        ("layer 5 evaluated 10 nodes",
         lambda r: r["arm_chain"]["per_layer"][5].update(nodes_evaluated=10)),
        ("layer 9 residual unchanged",
         lambda r: r["arm_chain"]["per_layer"][9].update(residual_changed=False)),
        ("layer 0 delta all zero",
         lambda r: r["arm_chain"]["per_layer"][0].update(delta_is_nonzero=False)),
        ("layer 31 non-finite",
         lambda r: r["arm_chain"]["per_layer"][31].update(all_finite=False)),
        ("layer 12 delta norm zero",
         lambda r: r["arm_chain"]["per_layer"][12].update(delta_out_l2=0.0)),
        ("layer 3 qkv weight dead",
         lambda r: r["arm_chain"]["per_layer"][3]["weight_witnesses"]["qkv"].update(
             weight_is_live=False)),
        ("layer 22 down weight half dequantised",
         lambda r: r["arm_chain"]["per_layer"][22]["weight_witnesses"]["down"].update(
             rows_dequantised=1)),
        ("layer 8 local bar red on the vulkan side",
         lambda r: r["arm_chain"]["per_layer"][8]["local_liveness"]["vulkan"].update(
             live=False, median_ulp=1e6)),
    ]
    assert len(per) == chain.N_LAYERS
    caught = 0
    for _why, mutate in corruptions:
        bad = copy.deepcopy(rec)
        mutate(bad)
        try:
            chain.assert_every_layer_live(bad["arm_chain"]["per_layer"], expect=chain.N_LAYERS)
        except chain.DeadLayerError:
            caught += 1
    assert caught == len(corruptions), f"only {caught}/{len(corruptions)} corruptions caught"
