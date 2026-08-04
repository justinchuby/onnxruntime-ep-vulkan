"""Criterion 10's re-score under §8.9.22, as a gate rather than a one-off probe.

A probe run once is a story; the record it wrote is only a claim if something checks it.
These arms read `bench/results/criterion10_rescore_8922-dev{0,1}.json` and assert the
properties the re-score is being quoted for. They are device-free: they read artifacts.

WHAT THEY DEFEND
================
1. **Per-output verdicts, never an aggregate.** 65 entries, each with its own verdict.
2. **Two numbers, never one** (§8.9.22 ruling 1) — every output carries the ruled residual
   AND the subnormal population, on every entry, not only on the failing ones.
3. **The subnormal population is published**, including where it is zero. A zero that is
   absent from the record is indistinguishable from a quantity nobody measured.
4. **The verdict predicate is `np.allclose` and reads no ULP statistic.** This is what
   makes the re-score honest: the repaired statistic could not have flipped the criterion,
   so a pass would not have been bought by the repair — and neither was the failure.
5. **The device is named off the run**, and the lane refuses a run with zero EP dispatches.

If the artifacts are absent the arms are SKIPPED with the command that regenerates them --
never silently green.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _models as m  # noqa: E402

_RESULTS = pathlib.Path(__file__).resolve().parents[2] / "bench" / "results"
_DEVICES = ("0", "1")

_REGEN = (
    "regenerate with: $env:ONNXRUNTIME_VULKAN_EP_LIB=...; "
    "$env:ONNXRUNTIME_EP_VULKAN_DEVICE={dev}; "
    "$env:ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE=...; "
    "python tests/ops/probe_criterion10_rescore.py"
)


def _record(dev: str) -> dict:
    p = _RESULTS / f"criterion10_rescore_8922-dev{dev}.json"
    if not p.exists():
        pytest.skip(f"{p.name} absent; {_REGEN.format(dev=dev)}")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("dev", _DEVICES)
def test_every_output_carries_its_own_verdict(dev: str) -> None:
    rec = _record(dev)
    per = rec["per_output"]
    assert len(per) == rec["outputs_total"] == 65, len(per)
    assert {p["index"] for p in per} == set(range(65))
    for p in per:
        assert p["verdict"] in {"WITHIN_TOLERANCE", "OUTSIDE_TOLERANCE", "DEGENERATE"}, p


@pytest.mark.parametrize("dev", _DEVICES)
def test_two_numbers_never_one_on_every_output(dev: str) -> None:
    rec = _record(dev)
    for p in rec["per_output"]:
        report = p["ruled_observable_report"]
        assert "ULP on the normal domain" in report, p
        assert "subnormal references" in report, p
        assert "published not dropped" in report, p
        # The population is a key in its own right, including when it is zero.
        assert p["subnormal_reference_elements"] is not None
        assert p["subnormal_reference_fraction"] is not None
        assert p["normal_domain_verdict"] in {"MEASURED", "ERROR(instrument=empty_normal_domain)"}


@pytest.mark.parametrize("dev", _DEVICES)
def test_the_verdict_predicate_reads_no_ulp_statistic(dev: str) -> None:
    """The load-bearing arm.

    If this ever fails, the re-score's central sentence -- *the repaired statistic could
    not have flipped this criterion* -- has stopped being true, and the whole result needs
    re-deriving before it is quoted again.
    """
    rec = _record(dev)
    for p in rec["per_output"]:
        pred = p["verdict_predicate"]
        assert "np.allclose" in pred, p
        assert "no ULP statistic on any basis participates" in pred, p


@pytest.mark.parametrize("dev", _DEVICES)
def test_the_ruled_statistic_did_not_move_the_failing_outputs(dev: str) -> None:
    """The measured answer to "was DIVERGENT an artifact of the element basis".

    On every failing output the ruled residual EQUALS the element-basis max, because the
    subnormal population on those tensors is 0, 0 and 2 of thousands. §8.9.22's mechanism
    is real on the artifact that motivated it and **absent here**. Asserted so that if a
    future run makes it present, this goes red and the conclusion is re-derived rather
    than inherited.
    """
    rec = _record(dev)
    assert rec["failing_indices"] == [0, 63, 64], rec["failing_indices"]
    for i in rec["failing_indices"]:
        p = rec["per_output"][i]
        assert p["ruled_equals_element_basis"] is True, p
        assert p["subnormal_reference_elements"] <= 2, p


@pytest.mark.parametrize("dev", _DEVICES)
def test_the_two_failure_mechanisms_are_distinguished(dev: str) -> None:
    """The re-score's readings, pinned -- AND the reading of them that §8.9.24 refuted.

    What the at-scale statistic says is still true of the at-scale statistic: the logits
    fail on ~19% of elements with residuals several steps wide at the tensor's scale, while
    layer 31's key and value fail on a handful of elements whose residual is at or below
    ONE step *of the tensor maximum*. Those numbers are not withdrawn.

    What I concluded from them was wrong, and §8.9.24 needed no run to show it. I wrote
    that 63 and 64 "fail within one representable step" and therefore that the gate was
    unsatisfiable. Two errors, compounding:

    * I quoted `atol` alone. `np.allclose` tests ``|a-b| <= atol + rtol*|b|`` -- a two-term
      sum, and I quoted one term of it.
    * I divided by the spacing at the TENSOR MAXIMUM while the predicate evaluates PER
      ELEMENT. The step I called "one representable step" was borrowed from a value
      roughly 500x larger than the elements that were failing.

    On the element basis, at each failing element's own magnitude, they do not fail within
    one step -- they fail by hundreds of steps against an allowance of tens. This test
    asserts BOTH readings on the same row, because the fence in §8.9.24(3) exists precisely
    so that the at-scale figure can never again appear without the arithmetic that bounds it.

    Reported. Not acted on: `atol` and `rtol` are untouched, and §8.9.24(4) forbids a
    motion outright. THE MOVER IS NOT THE MEASURER.
    """
    rec = _record(dev)
    logits = rec["per_output"][0]["failing_element_census"]
    assert logits["failing_elements"] > 1000, logits
    assert logits["failing_max_ulp_at_scale"] > 1.0, logits
    assert logits["failing_residual_within_one_ulp_at_scale"] < logits["failing_elements"] / 10

    for i in (63, 64):
        c = rec["per_output"][i]["failing_element_census"]
        assert c["failing_elements"] < 50, c

        # (a) the at-scale reading, unchanged and still true OF THE AT-SCALE UNIT
        assert c["failing_max_ulp_at_scale"] <= 1.0, c
        assert c["failing_residual_within_one_ulp_at_scale"] == c["failing_elements"], c

        # (b) the reading that refutes the conclusion I drew from (a). On the element
        #     basis the same residuals are hundreds of steps wide.
        assert c["failing_ulp_element_basis_max"] > 20.0, (
            f"output {i}: the element-basis residual no longer exceeds the general bound; "
            f"the §8.9.24 inversion is no longer demonstrated by this artifact -- {c}"
        )
        assert c["failing_ulp_element_basis_max"] > c["allowance_in_ulps_element_basis_min"], (
            f"output {i} would now be INSIDE its allowance on the element basis, which "
            f"would mean the gate no longer fails it -- a result, not a test to repair"
        )

        # (c) and the allowance is never the sub-step quantity the refuted finding used
        assert c["allowance_in_ulps_element_basis_min"] >= c[
            "satisfiability_bound_element_basis"
        ], c


@pytest.mark.parametrize("dev", _DEVICES)
def test_the_general_bound_holds_on_every_row(dev: str) -> None:
    """`allowance/ulp(b) >= rtol*2**10 = 20.48`, independent of magnitude.

    For normal fp16, ``ulp(b) <= |b| * 2**-10``, so the `rtol` term alone buys at least
    `rtol * 1024` representable steps at ANY magnitude -- minimum 20.48 attained at
    |b| = 32768, and subnormals get >= 16,777 steps from `atol` alone. There is no
    magnitude at which this predicate is unsatisfiable within one step, which is why the
    finding needed no run to refute.

    Asserting it here rather than only in the unit lane means the claim is checked against
    real measured tensors, not only against a swept range.
    """
    rec = _record(dev)
    for i in (0, 63, 64):
        c = rec["per_output"][i]["failing_element_census"]
        assert c["satisfiability_bound_element_basis"] == pytest.approx(0.02 * 1024)
        assert c["allowance_in_ulps_element_basis_min"] >= 20.48


@pytest.mark.parametrize("dev", _DEVICES)
def test_every_at_scale_figure_carries_its_companions(dev: str) -> None:
    """§8.9.24(3): ULP-at-scale is fenced, not withdrawn, and I am the owner of the fence.

    A row may report a residual in ULP-at-scale only if it also reports the allowance in
    the same unit and the failing set on the element basis. Both halves of the refuted
    finding are then visible on the row that would make it, so the next instance is caught
    where it is written rather than three rounds later.

    `assert_ulp_at_scale_row_is_complete` RAISES; a warning would be read past.
    """
    rec = _record(dev)
    for p in rec["per_output"]:
        m.assert_ulp_at_scale_row_is_complete(
            p, where=f"dev{dev} per_output[{p['index']}]"
        )
        if "failing_element_census" in p:
            m.assert_ulp_at_scale_row_is_complete(
                p["failing_element_census"], where=f"dev{dev} census[{p['index']}]"
            )


@pytest.mark.parametrize("dev", _DEVICES)
def test_the_device_is_named_off_the_run_and_dispatches_were_screened(dev: str) -> None:
    rec = _record(dev)
    name = rec["device_name_from_run"]
    assert name and name != "UNOBSERVABLE", rec
    assert any(v in name for v in ("NVIDIA", "Intel", "AMD", "llvmpipe")), name
    # A selector is a request, not an identity: the two must be recorded separately.
    assert rec["device_selector_requested"] == dev
    assert rec["dispatch_screen"] == "PASS", rec
    assert (rec["dispatches_executed"] or 0) > 0, rec


def test_both_devices_reach_the_same_per_output_verdicts() -> None:
    """Vendor independence of the verdict, asserted rather than assumed.

    Note what this does NOT say: the two devices' *numbers* differ (the logits' failing
    element count is 6056 on NVIDIA and 5932 on Intel). Only the per-output verdicts are
    claimed identical.
    """
    recs = [_record(d) for d in _DEVICES]
    v0 = [p["verdict"] for p in recs[0]["per_output"]]
    v1 = [p["verdict"] for p in recs[1]["per_output"]]
    assert v0 == v1
    assert recs[0]["failing_indices"] == recs[1]["failing_indices"] == [0, 63, 64]
