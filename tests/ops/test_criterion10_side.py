"""Gate on the §8.9.24(4) answer: which side is wrong on criterion 10's outputs 0, 63, 64.

§8.9.24(4) makes this question blocking for everyone else, so the answer has to be
defended by something that runs without a GPU and fails loudly when the artifact stops
supporting it. This module reads the records written by
`bench/results/probe_criterion10_side.py` and asserts what may be read off them.

Two things it is careful NOT to do.

It does not re-derive the answer. Every number here comes off a record produced by a run
on named hardware; a test that recomputed the verdict from stored inputs would be
asserting agreement between two copies of the same arithmetic and would pass on a device
that had never been opened.

And it does not pass when the records are absent. A missing artifact SKIPs with the
command to regenerate it. A gate that silently passes on no evidence is the shape of
defect this project has been bitten by three times (§8.9.22, §8.9.24, and Switch's ctx-4096
silent CPU rebuild, which exits 0 and reads as a clean EP run).

THE MOVER IS NOT THE MEASURER: nothing here proposes what follows from the answer, and
`test_neither_record_proposes_a_motion` enforces that on the artifacts mechanically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "bench" / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RESULTS))

DEVICES = ("dev0", "dev1")

#: The three outputs criterion 10 fails on, and the arm that answers for each.
ORACLE_ARMS = (
    "arm_b_present31_key",
    "arm_b_present31_value",
    "arm_c_isolated_qkv_proj31",
    "arm_d_isolated_lm_head",
)


def _record(tag: str) -> dict:
    p = RESULTS / f"criterion10_side-{tag}.json"
    if not p.exists():
        pytest.skip(
            f"{p.name} absent -- regenerate with:\n"
            f"  $env:ONNXRUNTIME_VULKAN_EP_LIB=<path to onnxruntime_vulkan_ep.dll>\n"
            f"  python bench/results/probe_criterion10_side.py "
            f"--device {tag[-1]} --out bench/results/{p.name}\n"
            "(this SKIPs rather than passes: a gate that passes on no evidence is not a gate)"
        )
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module", params=DEVICES)
def rec(request) -> dict:
    return _record(request.param)


# ==========================================================================================
# The run happened, on hardware, on the EP
# ==========================================================================================
@pytest.mark.parametrize("arm", ORACLE_ARMS)
def test_every_oracle_arm_ran_on_the_ep(rec, arm):
    """An oracle arm that ORT quietly rebuilt on the CPU EP is CPU-vs-CPU.

    Its residual against a float64 reference would then be a statement about ORT's CPU
    kernel twice over, and its `which_is_further_from_true` would be structurally
    "neither" -- a clean-looking null that means nothing. Switch established that such a
    run **exits 0**. Attribution is a counters delta, not a return code.
    """
    a = rec[arm]
    assert a["status"] == "MEASURED", f"{arm}: {a.get('status')} -- {a.get('error')}"
    assert a["attribution"] == "ATTRIBUTED", (
        f"{arm} claimed no nodes on this session: its residual is CPU-vs-CPU and its "
        f"verdict is an artifact of the harness, not a measurement"
    )
    assert a["dispatch_screen"] == "PASS", (
        f"{arm}: {a.get('dispatches_executed_this_arm')} dispatches this arm"
    )
    assert a["dispatches_executed_this_arm"] > 0, (
        f"{arm} executed zero EP dispatches of its own. The process-cumulative counter "
        f"({a.get('dispatches_executed_process_cumulative')}) would still be nonzero from "
        f"earlier arms, which is exactly why the screen is a delta"
    )


def test_the_device_is_named_off_the_run_not_off_the_selector(rec):
    """A selector ordinal names no hardware -- it is a request, not an identity.

    Both records in this pair prove the point on their face: `--device 0` ran on allocator
    index 1 and `--device 1` ran on allocator index 0. A gate that trusted the selector
    would have both device names wrong and would not notice.
    """
    name = rec["arm_a_in_situ"]["device_name"]
    assert name and name != "UNKNOWN", "no device name was read off the run"
    assert "=" in name, f"expected '<index>=<vendor string>', got {name!r}"
    idx, vendor = name.split("=", 1)
    assert idx.isdigit()
    assert len(vendor) > 4 and not vendor.isdigit(), (
        f"{vendor!r} is not a vendor string; a number here means the selector was echoed "
        f"back instead of the device being read off the run"
    )
    assert rec["device_selector_requested"] is not None
    assert "a selector is a request" in rec["selector_caveat"], (
        "the record must say in its own body that the selector is a request, not an "
        "identity -- a caveat that lives only in a report is a caveat that gets dropped"
    )
    assert rec["arm_a_in_situ"]["device_name_source"].endswith("off the run")


# ==========================================================================================
# The reference is alive, and is an oracle
# ==========================================================================================
@pytest.mark.parametrize("arm", ORACLE_ARMS)
def test_the_reference_is_live(rec, arm):
    """A dead reference agrees with everything.

    If the float64 reference were all zeros, or constant, or accidentally the same array as
    one of the outputs, every arm would report bit-exactness and the probe would look like
    a triumph. `reference_liveness` is the arm that makes a null answer distinguishable
    from a working one.
    """
    live = rec[arm]["reference_liveness"]
    assert live["live"] is True, f"{arm}: dead reference -- {live}"


def test_the_position_zero_rotary_form_is_read_off_the_run(rec):
    """A PREDICTION IS NOT A READING -- and this is the specimen.

    The first draft of this probe reasoned that at sequence position 0 the rotation is by
    angle 0, so `cos_cache[0] == 1`, so `present.31.key` is the K slice of the packed QKV
    **verbatim**. That reasoning is sound and its conclusion is FALSE: Phi-3.5 folds a
    long-rope attention factor into the cache, and `cos_cache[0]` is uniformly
    1.1904296875. Because the sine is zero the rotation still degenerates to a single
    multiply -- ONE correctly-rounded operation -- so the reference is still envelope-free
    and the conclusion that mattered survived. The stated reason for it did not.

    Had the cache carried a nonzero sine, the same unmeasured assumption would have
    reported a rotation defect as a copy defect. So this asserts the form was MEASURED and
    that `reference_is_exact` follows from the measured form rather than being declared.
    """
    a = rec["arm_b_present31_key"]
    form = a["rope_form_at_position_0"]
    assert form in ("IDENTITY", "PURE_SCALE", "ROTATION")
    assert "READ OFF THE RUN" in a["rope_check_source"]

    # exactness is a consequence of the form, never an assertion beside it
    assert a["reference_is_exact"] is (form in ("IDENTITY", "PURE_SCALE")), (
        f"reference_is_exact={a['reference_is_exact']} does not follow from form={form}"
    )
    # the identity flag and the form must not be able to drift apart
    assert a["rope_is_identity_at_position_0"] is (form == "IDENTITY")
    assert a["rope_sin_is_zero_at_position_0"] is (form in ("IDENTITY", "PURE_SCALE"))

    if form == "ROTATION":
        pytest.fail(
            "the position-0 reference now carries a rounding envelope; arms B are no "
            "longer oracles and the §8.9.24(4) answer for outputs 63/64 must be rebuilt"
        )


def test_the_model_is_the_pure_scale_case_and_not_the_predicted_copy(rec):
    """Pin the reading that refuted the prediction, so a regression cannot quietly restore it.

    If this ever reads IDENTITY again it means either the model changed or the cache tap
    broke; either way the recorded refutation would no longer be about this artifact.
    """
    a = rec["arm_b_present31_key"]
    assert a["rope_form_at_position_0"] == "PURE_SCALE"
    assert a["rope_is_identity_at_position_0"] is False
    assert a["rope_cos_is_constant_at_position_0"] is True
    assert a["rope_scale_at_position_0"] == pytest.approx(1.1904296875, abs=0.0)


# ==========================================================================================
# The answer
# ==========================================================================================
def test_outputs_63_and_64_carry_no_locally_made_error(rec):
    """The §8.9.24(4) answer for outputs 63 and 64.

    Fed IDENTICAL bytes and measured against an envelope-free float64 reference built from
    those same bytes, BOTH EPs are bit-exact. Neither side is wrong here. Whatever criterion
    10 sees on outputs 63 and 64 was not made at this node.
    """
    for arm in ("arm_b_present31_key", "arm_b_present31_value"):
        a = rec[arm]
        assert a["vulkan_ep_vs_float64"]["bit_exact"] is True, f"{arm}: Vulkan EP is not exact"
        assert a["cpu_ep_vs_float64"]["bit_exact"] is True, f"{arm}: ORT's CPU EP is not exact"
        assert a["which_is_further_from_true"] == "neither (equal)"
        assert a["both_bit_exact_against_the_reference"] is True


def test_the_divergence_on_63_and_64_is_inherited_as_an_identity(rec):
    """"Inherited" must be an identity, not a story.

    Arms B say a residual on 63/64 cannot have been MADE at the GQA node. That is an
    argument from two bit-exact rows. Arm E turns it into a reconstruction: each EP's own
    in-situ `present.31.{key,value}` is recomputed in float64 from that EP's own packed-QKV
    tap and must come back BIT-EQUAL. When it does, outputs 63 and 64 are a deterministic
    function of the QKV tensor on both sides and their divergence is exactly the divergence
    already in that tensor -- with no locally-made component left to attribute.

    Arm E consults no reference and so can never support a verdict; it can only bound how
    arms B and C are allowed to be read.
    """
    e = rec["arm_e_inheritance_identity"]
    assert e["status"] == "MEASURED", e.get("error")
    assert e["class"] == "CONSTRAINT"
    assert e["both_sides_reconstruct_bit_exactly"] is True, e["per_side"]
    for side in ("vulkan", "cpu"):
        s = e["per_side"][side]
        assert s["key_elements_mismatched"] == 0
        assert s["value_elements_mismatched"] == 0


def test_output_0_the_vulkan_ep_is_the_closer_side(rec):
    """The §8.9.24(4) answer for output 0, and it runs against this project's default.

    At the `lm_head` MatMulNBits, fed identical bytes, ORT's CPU EP is FURTHER from the
    true float64 value than the Vulkan EP -- unanimously, on every discriminator that
    separates the two sides at all. This is the second node where "the Vulkan EP is wrong
    until proven otherwise" reads backwards (the first was round 36's final RMSNorm).

    Nothing follows from this here. It is not a licence to move a tolerance, it is not a
    defect filed against ORT, and it is not a statement about output 0 as criterion 10 sees
    it -- see `test_the_record_states_its_own_limit`.
    """
    a = rec["arm_d_isolated_lm_head"]
    assert a["which_is_further_from_true"] == "cpu"
    assert a["unanimous_direction"] == "cpu"
    assert a["discriminators_conflict"] is False
    assert a["vulkan_ep_vs_float64"]["max_ulp_diff"] < a["cpu_ep_vs_float64"]["max_ulp_diff"]


def test_the_round_36_disagreement_on_lm_head_is_a_statistic_not_a_result(rec):
    """Round 36 read `neither (equal)` here; this round reads `cpu`. Both are true.

    Round 36 discriminated on `median_ulp_diff`. On this tensor the median is 0 on BOTH
    sides -- 32053 of 32064 elements are bit-exact on the Vulkan side -- so the median
    could not have separated them whatever the answer was, and "neither (equal)" off two
    zeros is a null result wearing the clothes of a measurement.

    The record must carry BOTH readings, or a change of statistic reads as a result that
    changed between rounds. That is the R11 "one max swapped for another" shape and this
    project has published it once already.
    """
    by = rec["arm_d_isolated_lm_head"]["verdict_by_discriminator"]
    assert by["median_ulp_diff"]["verdict"] == "neither (equal)", (
        "the round-36 reading is no longer reproducible off this record, so the "
        "reconciliation in this docstring is no longer supported by the artifact"
    )
    assert by["median_ulp_diff"]["vulkan_value"] == 0.0
    assert by["median_ulp_diff"]["cpu_value"] == 0.0
    assert by["max_ulp_diff"]["verdict"] == "cpu"
    assert "median_ulp_diff" in rec["arm_d_isolated_lm_head"]["discriminators_silent"]


def test_the_qkv_projection_arm_is_a_conflict_and_is_not_resolved(rec):
    """Arm C has NO answer, and the record must say so rather than pick one.

    On the isolated layer-31 `qkv_proj` MatMulNBits the discriminators split: ORT's CPU EP
    differs from the float64 value in 18 elements against the Vulkan EP's 2, while the
    Vulkan EP's single worst element is ~32x further in absolute terms. More elements
    slightly wrong, or fewer elements badly wrong -- there is no fact of the matter about
    which is "further from true" without a further choice, and making that choice inside
    a verdict field would be a criterion motion dressed as a measurement.

    The single-discriminator version of this probe reported "neither (equal)" for this arm
    and hid the split entirely. That is the whole reason `verdict_by_discriminator` exists.
    """
    a = rec["arm_c_isolated_qkv_proj31"]
    assert a["discriminators_conflict"] is True
    assert a["unanimous_direction"] is None, (
        "a split must not be resolved into a direction"
    )
    assert "conflict_note" in a
    by = a["verdict_by_discriminator"]
    assert {by["elements_differing"]["verdict"], by["max_abs_diff"]["verdict"]} == {
        "cpu",
        "vulkan",
    }


def test_no_arm_reports_the_vulkan_ep_as_the_sole_wrong_side(rec):
    """The finding, stated once as a whole, over all three of criterion 10's failures.

    On no arm in this record is the Vulkan EP unanimously the side further from the true
    float64 value. Two arms are bit-exact on both sides, one is a genuine split, and one
    puts ORT's CPU EP further. If this ever fails it is a real result and must be reported,
    not repaired.
    """
    offenders = [
        arm for arm in ORACLE_ARMS if rec[arm].get("unanimous_direction") == "vulkan"
    ]
    assert not offenders, (
        f"{offenders} now put the Vulkan EP unanimously further from the true value; "
        f"this is a finding, not a test failure to be fixed"
    )


# ==========================================================================================
# What the record is not allowed to be
# ==========================================================================================
def test_the_in_situ_arm_is_never_an_oracle(rec):
    """Arm A compares the two EPs to EACH OTHER on their own in-situ inputs.

    Round 36's discarded arm F laid a float64 reference built from the CPU's tapped inputs
    against Vulkan's in-situ output and reported `which_is_further_from_true: "vulkan"` --
    BY CONSTRUCTION. Isolation means identical inputs on both sides or it means nothing.
    Arm A therefore must carry no verdict field at all, and must say what it is.
    """
    a = rec["arm_a_in_situ"]
    assert a["class"] == "INHERITANCE"
    assert "which_is_further_from_true" not in a
    assert "by construction" in a["not_an_oracle_answer"]
    for st in a["per_tensor"].values():
        assert "which_is_further_from_true" not in st


def test_neither_record_proposes_a_motion(rec):
    """§8.9.24(4) forbids a tolerance, unit, predicate or verdict motion in this round.

    Morpheus's warning was specific: if the reference is wrong the remedy is a different
    oracle, and the argument would be made in the direction of LOOSENING using the
    reference's own error as the budget. Arm D -- ORT's CPU EP being 11 ULP from the true
    value at the `lm_head` -- is exactly the number that argument would want. So the guard
    runs over the finished artifact as a gate and not merely inside the probe.
    """
    from probe_criterion10_side import MotionInRecordError, assert_record_proposes_no_motion

    try:
        assert_record_proposes_no_motion(rec)
    except MotionInRecordError as exc:  # pragma: no cover
        pytest.fail(f"the record proposes a motion §8.9.24(4) forbids: {exc}")


def test_the_no_motion_guard_can_still_refuse():
    """A guard that cannot go red witnesses nothing.

    `test_neither_record_proposes_a_motion` passing is only evidence if the guard would
    have failed on a record that did propose something.
    """
    from probe_criterion10_side import MotionInRecordError, assert_record_proposes_no_motion

    with pytest.raises(MotionInRecordError):
        assert_record_proposes_no_motion({"arm_d": {"proposed_atol": 2e-3}})


def test_the_record_states_its_own_limit(rec):
    """Every answer above is per-hop with identical inputs. The in-situ question is OPEN.

    "Which side is wrong on output 0 as criterion 10 sees it" is NOT answered by arm D: at
    model scale the two EPs reach the `lm_head` with different hidden states, and comparing
    either against a float64 reference built from one of them is round 36's arm F again.
    The record must carry that limit and what it would take to lift it, in its own body --
    a limit stated only in a report is a limit that gets dropped on the next quotation.
    """
    lim = rec["what_this_does_not_answer"]
    assert lim["the_in_situ_question"]
    assert lim["what_it_would_take"]
    assert rec["not_done_here"]


def test_the_tolerances_have_not_moved(rec):
    """Three rounds of declining, and §8.9.24(4) now forbids it outright.

    Read from the source of truth rather than the record, so that a probe reporting the
    old numbers while the comparator used new ones cannot pass.

    The two constants are pinned separately and to their OWN values. The first draft of
    this gate pinned both to fp16's `rtol=0.02, atol=1e-3` -- fp32's are an order of
    magnitude tighter -- and failed. That failure is worth keeping in the record: I asserted
    a number I had not read, in a test whose entire purpose is to stop numbers from moving
    unread. Criterion 10 runs the fp16 path; the fp32 entry is pinned because a drift there
    would move a gate nobody is currently watching.
    """
    import _models as m

    expected = {
        "MATMULNBITS_FP16": {"rtol": 0.02, "atol": 0.001},
        "MATMULNBITS_FP32": {"rtol": 0.001, "atol": 0.0001},
    }
    for name, want in expected.items():
        tol = getattr(m, name)
        got = {
            "rtol": tol["rtol"] if isinstance(tol, dict) else tol.rtol,
            "atol": tol["atol"] if isinstance(tol, dict) else tol.atol,
        }
        assert got == want, f"{name} moved: {got} != {want}"


# ==========================================================================================
# Cross-device
# ==========================================================================================
def test_both_devices_reach_the_same_answer():
    """A verdict that holds on one vendor and not the other is a device finding, not an answer.

    Both records must exist for this to mean anything, so it SKIPs rather than passing when
    either is missing.
    """
    recs = {tag: _record(tag) for tag in DEVICES}
    names = {tag: r["arm_a_in_situ"]["device_name"] for tag, r in recs.items()}
    assert len(set(names.values())) == 2, (
        f"both records name the same device {names}; one of them did not run where it "
        f"was asked to and the pair proves nothing about cross-vendor agreement"
    )
    for arm in ORACLE_ARMS:
        verdicts = {tag: r[arm]["which_is_further_from_true"] for tag, r in recs.items()}
        assert len(set(verdicts.values())) == 1, f"{arm} splits by device: {verdicts} on {names}"
        unanimous = {tag: r[arm].get("unanimous_direction") for tag, r in recs.items()}
        assert len(set(map(str, unanimous.values()))) == 1, (
            f"{arm} unanimity splits by device: {unanimous} on {names}"
        )


def test_the_isolated_arms_agree_bit_for_bit_across_vendors():
    """An unexpected reading, asserted so that losing it is visible.

    On every isolated arm the residual distributions are IDENTICAL between an NVIDIA RTX
    4060 and an Intel Iris Xe -- not merely the same verdict, the same numbers. These
    kernels are producing bit-identical results on two unrelated GPUs. That is worth
    pinning: if it ever stops being true, a per-vendor numerical difference has appeared
    somewhere and this is the cheapest place it would show.
    """
    recs = {tag: _record(tag) for tag in DEVICES}
    for arm in ORACLE_ARMS:
        for side in ("vulkan_ep_vs_float64", "cpu_ep_vs_float64"):
            vals = {
                tag: {
                    k: r[arm][side][k]
                    for k in ("max_ulp_diff", "median_ulp_diff", "elements_differing",
                              "max_abs_diff", "bit_exact")
                }
                for tag, r in recs.items()
            }
            a, b = (vals[t] for t in DEVICES)
            assert a == b, f"{arm}/{side} differs across vendors: {vals}"


def test_the_in_situ_residuals_do_not_agree_across_vendors_and_that_is_expected():
    """The negative control for the arm above.

    If the in-situ residuals were ALSO identical across two different GPUs, the likeliest
    explanation would not be remarkable agreement -- it would be that one record was copied
    from the other, or that neither run reached the device. The isolated arms agreeing is
    only informative because the in-situ ones do not.
    """
    recs = {tag: _record(tag) for tag in DEVICES}
    counts = {
        tag: r["arm_a_in_situ"]["per_tensor"]["logits (output 0)"]["elements_differing"]
        for tag, r in recs.items()
    }
    assert len(set(counts.values())) == 2, (
        f"the two devices report identical in-situ divergence {counts}; either the records "
        f"are duplicates or neither run reached the hardware"
    )


# ==========================================================================================
# §8.9.24(3): the fence on ULP-at-scale applies to this record too
# ==========================================================================================
@pytest.mark.parametrize("arm", ORACLE_ARMS)
def test_every_at_scale_figure_carries_its_companions(rec, arm):
    """I own `ULP-at-scale` and it is fenced, not withdrawn.

    Any row reporting a residual in ULP-at-scale must also report the allowance in the same
    unit and the failing set on the element basis. The refuted finding quoted an at-scale
    residual against a step borrowed from the tensor maximum; the companions are what make
    that impossible to do again without it being visible on the same row.
    """
    import _models as m

    for side in ("vulkan_ep_vs_float64", "cpu_ep_vs_float64", "vk_vs_cpu"):
        m.assert_ulp_at_scale_row_is_complete(rec[arm][side], where=f"{arm}/{side}")


def test_an_undefined_scale_reports_absence_and_never_a_zero(rec):
    """A residual over an undefined denominator is an absence, and a 0 there would acquit.

    This is the same failure direction as the `np.spacing` overflow found this round: an
    instrument that reports 0 where it means "I could not measure" produces a clean row
    that nobody re-derives.
    """
    for arm in ORACLE_ARMS:
        for side in ("vulkan_ep_vs_float64", "cpu_ep_vs_float64", "vk_vs_cpu"):
            st = rec[arm][side]
            if not st["one_ulp_at_scale"]:
                assert st["max_ulp_at_scale_diff"] is None, (
                    f"{arm}/{side} reports an at-scale residual over an undefined scale"
                )
