"""The mutation grid for issue #96: every gate, deleted or relaxed, has to turn this file red.

Two revisions of this investigation were rejected, and both times the defect was the same shape:
a claim that survived the absence of the evidence it was about. A length with no admissible
timing counted as a length measured and found not slow. A `--check` that re-derived a summary
without ever reading the raw samples it was derived from. A band imported from a workload five
times quieter than the one it graded.

So the tests here are not written as "does the happy path work". They are written as *"here is
the specific way this could go wrong; break it that way and watch a test fail."* Each one takes
the shipped production code in `bench/decode_window_evidence.py`, feeds it a record set that has
been damaged in one exact way, and asserts on the damage. None of them re-implements the rule
they are checking — a test that computes the answer a second way is testing its own arithmetic,
and when the two implementations drift it is the test that gets edited.

The grid, in the order the gates are numbered in the revision brief:

* **absent is absent** — refused records may not become timings, "not slow"s, edges, neighbours,
  or window support, and a refused record must not still be carrying a speed block;
* **a window needs its edges** — with `past = 128`'s named neighbours refused, no claim survives;
  with every treatment record refused, no claim survives either, and the two failures are
  reported differently because they mean different things;
* **no arm defines and judges itself** — an A/A arm sharing a process with the treatment yields
  no band, and no band yields no verdict;
* **the detector can fire** — a planted effect, pushed through the shipped verdict function over
  the shipped records, comes back SLOWER; if it did not, none of the negatives would mean
  anything;
* **`--check` binds the raw** — scale the samples, delete the samples, strip a witness, or make a
  library digest disagree with its arm, and reproduction fails;
* **the claims match the artifact** — the numbers in `docs/PERF.md` and `docs/DESIGN.md` are the
  numbers in the artifact, the historical 0.859 is still recorded, and no document says the
  regression failed to reproduce.

Every test is GPU-free. They read the committed artifact and never measure anything.
"""

from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import decode_window_evidence as dwe  # noqa: E402
from _polarity import withholds  # noqa: E402

ARTIFACT_PATH = _BENCH / "results" / "decode_window_evidence.json"
REUSED_PATH = _BENCH / "results" / "crossbuild_decode_window_records.json"
PROBE_PATH = _BENCH / "results" / "probe_decode_aa_calibration.py"
PERF_PATH = _ROOT / "docs" / "PERF.md"
DESIGN_PATH = _ROOT / "docs" / "DESIGN.md"

TREATMENT_WORKLOAD = "phi-3.5-mini-instruct-cuda-int4-rtn-block-32/decode/M1/past128"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def mutable(artifact) -> dict:
    """A private deep copy, so a mutation in one test cannot leak into the next."""
    return copy.deepcopy(artifact)


def _resummarize(art: dict) -> dict:
    return dwe.summarize(
        art["records"],
        arms=art["arms"],
        repeats_required=art["environment"]["repeats"],
        aa_allocations=art["aa_allocations"],
        reference_effect=art.get("reference_effect"),
    )


def _row(summary: dict, past: int) -> dict:
    return next(r for r in summary["rows"] if r["past"] == past)


# =============================================================================================
# Gate 1 — absent is absent
# =============================================================================================


class TestARefusedRecordIsAbsentAndNotEvidence:
    """A record with no admissible timing is the absence of evidence, not evidence of absence."""

    def test_the_artifact_has_refusals_so_this_grid_is_not_vacuous(self, artifact):
        """If every record were accepted, every test below would pass by having nothing to test."""
        counts = artifact["summary"]["counts"]
        assert counts["refused"] > 0, "no refused records: the absent-is-absent gate is untested"
        assert counts["accepted"] + counts["refused"] == counts["records"]

    def test_a_refused_record_carries_no_timing_at_all(self, artifact):
        """Structural, not cosmetic. A refused record that kept its `speed` block is a timing
        waiting to be read by the next piece of code that forgets to check the status."""
        for record in artifact["records"]:
            if dwe.classify_record(record)["status"] == dwe.REFUSED:
                assert "speed" not in record or record["speed"] is None, (
                    f"refused record {record.get('workload')} {record.get('arm')} "
                    f"repeat {record.get('repeat')} still carries a speed block"
                )

    def test_refused_lengths_are_not_in_the_measured_set(self, artifact):
        """The three lengths whose equivalence failed are absent from `measured_lengths`.

        This is the exact defect that got the previous revision rejected: it published
        `lengths_measured: [0, 32, 64, 128, 256, 512, 1024]` from a run that had admissible
        pairs at four of those and none at three."""
        window = artifact["summary"]["window"]
        for length in (32, 64, 256):
            assert length not in window["measured_lengths"]
            assert length in window["unmeasured_lengths"]

    def test_prefill_is_not_a_decode_length(self, artifact):
        """`past = 0` is prefill. Admitting it to the decode universe is how a decode window
        acquires a neighbour from a different phase."""
        assert 0 not in dwe.DECODE_LENGTH_UNIVERSE
        assert dwe.past_of("model/prefill/M1/past0") is None
        assert dwe.past_of("model/decode/M1/past128") == 128
        assert 0 not in artifact["summary"]["window"]["measured_lengths"]

    @pytest.mark.parametrize("witness", dwe.REQUIRED_WITNESSES)
    def test_dropping_any_single_witness_refuses_the_record(self, mutable, witness):
        """Mutation: delete one witness field from one accepted record. It must stop counting.

        Parameterised over every required witness because "we check the important ones" is how a
        field stops being checked."""
        record = next(r for r in mutable["records"] if dwe.classify_record(r)["status"] == dwe.ACCEPTED)
        before = dwe.classify_record(record)["status"]
        record.pop(witness)
        after = dwe.classify_record(record)
        assert before == dwe.ACCEPTED
        assert after["status"] == dwe.REFUSED, f"dropping {witness} left the record accepted"
        assert any(witness in reason for reason in after["reasons"])

    def test_a_non_finite_median_refuses(self, mutable):
        record = next(r for r in mutable["records"] if dwe.classify_record(r)["status"] == dwe.ACCEPTED)
        record["speed"]["median_ms"] = float("nan")
        assert dwe.classify_record(record)["status"] == dwe.REFUSED

    def test_a_median_that_does_not_match_its_own_samples_refuses(self, mutable):
        """The published median has to come from the published samples. Otherwise a summary can
        be reproduced from records whose raw data says something else entirely."""
        record = next(r for r in mutable["records"] if dwe.classify_record(r)["status"] == dwe.ACCEPTED)
        record["speed"]["median_ms"] = record["speed"]["median_ms"] * 1.5
        result = dwe.classify_record(record)
        assert result["status"] == dwe.REFUSED
        assert any("does not re-derive" in reason for reason in result["reasons"])

    def test_a_divergent_secondary_output_refuses_even_when_the_primary_matched(self, mutable):
        """Gate 5 asks for equivalence over *all* outputs. A decode step whose logits agree and
        whose KV cache does not produces a correct first token and a wrong sequence."""
        record = next(r for r in mutable["records"] if dwe.classify_record(r)["status"] == dwe.ACCEPTED)
        record["equivalence"] = {"verdict": "MATCH", "primary": {"verdict": "MATCH"},
                                 "worst_secondary": {"verdict": "DIVERGENT"}}
        result = dwe.classify_record(record)
        assert result["status"] == dwe.REFUSED
        assert any("equivalence" in reason for reason in result["reasons"])


# =============================================================================================
# Gate 2 — a window needs its edges
# =============================================================================================


class TestAWindowClaimNeedsItsNamedNeighbours:
    """No claim about a window survives the absence of the lengths that would bound it."""

    def test_the_published_window_is_indeterminate_and_names_the_absent_sides(self, artifact):
        window = artifact["summary"]["window"]
        assert window["claim"] == dwe.INDETERMINATE
        assert window["predecessor"] == 64
        assert window["successor"] == 256
        assert set(window["missing_sides"]) == {"predecessor", "successor"}
        assert "64" in window["reason"] and "256" in window["reason"]

    def test_mutation_refuse_every_treatment_record_and_no_claim_survives(self, mutable):
        """The sweep that measured *nothing* must not publish the sweep that measured everything.

        Previous revision, verified at its own head: refusing every treatment record still
        produced `NO-SLOW-LENGTH` with seven lengths listed as measured."""
        for record in mutable["records"]:
            if record.get("role") != "aa":
                record["admissible"] = False
                record["refusal"] = {"reason": "mutation: refused"}
                record.pop("speed", None)
        summary = _resummarize(mutable)
        assert summary["window"]["claim"] == dwe.INDETERMINATE
        assert summary["window"]["measured_lengths"] == []
        assert summary["accepted_decode_lengths"] == []
        assert all(row["verdict"] == dwe.INDETERMINATE for row in summary["rows"])

    def test_mutation_refuse_only_the_p128_neighbours_and_no_claim_survives(self, mutable):
        """Both neighbours are already refused in the real data, so this mutation makes the
        *counterfactual*: give 64 and 256 back, and the claim becomes makeable."""
        summary = _resummarize(mutable)
        assert summary["window"]["claim"] == dwe.INDETERMINATE
        assert summary["window"]["missing_sides"] == ["predecessor", "successor"]

        # Hand the rule a synthetic verdict map in which the neighbours exist and are decisive.
        with_neighbours = dwe.window_verdict(
            {64: dwe.NEUTRAL, 128: dwe.SLOWER, 256: dwe.NEUTRAL, 512: dwe.NEUTRAL}
        )
        assert with_neighbours["claim"] == "SINGLE-SLOW-POINT"
        # Take either neighbour away again and the same rule refuses to name an edge.
        without = dwe.window_verdict({128: dwe.SLOWER, 256: dwe.NEUTRAL, 512: dwe.NEUTRAL})
        assert without["claim"] == dwe.INDETERMINATE
        assert without["missing_sides"] == ["predecessor"]

    def test_a_measured_but_undecided_neighbour_is_reported_differently_from_an_absent_one(self):
        """"We measured it and cannot resolve it" and "we have nothing" are different facts and
        call for different work. Collapsing them is how an unmeasured length becomes a null."""
        absent = dwe.window_verdict({128: dwe.SLOWER, 256: dwe.NEUTRAL})
        undecided = dwe.window_verdict(
            {64: dwe.INDETERMINATE, 128: dwe.SLOWER, 256: dwe.NEUTRAL}
        )
        assert absent["claim"] == undecided["claim"] == dwe.INDETERMINATE
        assert absent["missing_sides"] == ["predecessor"]
        assert undecided["missing_sides"] == []
        assert undecided["undecided_sides"] == ["predecessor"]
        assert "did not resolve" in undecided["reason"]

    def test_a_window_is_only_named_when_a_measured_neighbour_is_also_slow(self):
        window = dwe.window_verdict({64: dwe.SLOWER, 128: dwe.SLOWER, 256: dwe.NEUTRAL})
        assert window["claim"] == "WINDOW"
        assert window["slow_lengths"] == [64, 128]

    def test_a_target_outside_the_pre_registered_universe_cannot_be_claimed(self):
        """A length nobody pre-registered has no defined neighbours, so it has no window."""
        result = dwe.window_verdict({100: dwe.SLOWER}, target=100)
        assert result["claim"] == dwe.INDETERMINATE
        assert "not in the pre-registered ordered universe" in result["reason"]


# =============================================================================================
# Gate 3 — no arm defines and judges itself
# =============================================================================================


class TestCalibrationIsDisjointFromWhatItGrades:
    """The band comes from A/A arms that share no process with the treatment. Or there is no band."""

    def test_both_aa_arms_are_present_and_are_same_binary_allocations(self, artifact):
        arms = artifact["summary"]["aa_arms"]
        assert len(arms) == 2
        names = {a["workload"].rsplit("/", 1)[-1] for a in arms}
        assert names == {"aa-candidate", "aa-baseline"}
        for arm in arms:
            left = artifact["arms"][arm["left"]]
            right = artifact["arms"][arm["right"]]
            assert left["sha256"] == right["sha256"], (
                f"{arm['workload']} is not an A/A: its two sides are different binaries"
            )
            assert arm["n_pairs"] == artifact["environment"]["repeats"]

    def test_the_aa_arms_are_measured_at_the_workload_they_grade(self, artifact):
        """The previous revision calibrated on a 28 ms CNN and applied the band to a 137 ms LLM
        decode. Both A/A arms here run at `decode past=128`, the treatment workload."""
        for arm in artifact["summary"]["aa_arms"]:
            assert "/decode/M1/past128/" in arm["workload"]

    def test_no_aa_process_is_also_a_treatment_process(self, artifact):
        """Disjointness is asserted on process *identity*, not on the bare PID. Windows recycles
        PIDs and these two runs are days apart, so the same integer really does appear on both
        sides; `(pid, started_at)` is what actually distinguishes one process from another."""
        aa_keys, treatment_keys = set(), set()
        for record in artifact["records"]:
            if dwe.classify_record(record)["status"] != dwe.ACCEPTED:
                continue
            bucket = aa_keys if record.get("role") == "aa" else treatment_keys
            bucket.add(dwe._process_key(record))
        assert aa_keys and treatment_keys
        assert not (aa_keys & treatment_keys)

    def test_a_recycled_pid_alone_is_not_treated_as_contamination(self, artifact):
        """The guard has to be tight enough to catch a shared process and loose enough not to
        cry contamination over PID reuse. This artifact genuinely contains such reuse, so this is
        a live case rather than a hypothetical one."""
        aa_pids, treatment_pids = set(), set()
        for record in artifact["records"]:
            if dwe.classify_record(record)["status"] != dwe.ACCEPTED:
                continue
            (aa_pids if record.get("role") == "aa" else treatment_pids).add(record["pid"])
        assert aa_pids & treatment_pids, "expected recycled PIDs across the two sessions"
        assert artifact["summary"]["band"]["band"] is not None

    def test_an_aa_row_can_never_be_read_as_a_decode_length(self, artifact):
        """The A/A workload labels are deliberately unparseable as a KV length, so a calibration
        row cannot wander into the window rule as a measurement of 128."""
        for arm in artifact["summary"]["aa_arms"]:
            assert dwe.past_of(arm["workload"]) is None

    def test_mutation_contaminate_the_aa_arm_and_the_band_disappears(self, mutable):
        """Make an A/A record claim to be the same process as a treatment record: the calibration
        is no longer independent, so there is no band, so nothing can be graded."""
        treatment = next(
            r for r in mutable["records"]
            if r.get("role") != "aa" and dwe.classify_record(r)["status"] == dwe.ACCEPTED
        )
        aa_record = next(r for r in mutable["records"] if r.get("role") == "aa")
        aa_record["pid"] = treatment["pid"]
        aa_record["started_at"] = treatment["started_at"]
        summary = _resummarize(mutable)
        assert summary["band"]["band"] is None
        assert "shares a process" in summary["band"]["reason"]
        assert summary["band"]["contaminated"]
        assert all(row["verdict"] == dwe.INDETERMINATE for row in summary["rows"])

    def test_mutation_delete_the_aa_arms_and_nothing_can_be_graded(self, mutable):
        """Without a calibration there is no scale, and without a scale a ratio is a number, not
        a verdict. This is why the A/A arm is load-bearing rather than decorative."""
        mutable["records"] = [r for r in mutable["records"] if r.get("role") != "aa"]
        summary = _resummarize(mutable)
        assert summary["band"]["band"] is None
        assert summary["band"]["reason"] == "no admissible A/A pair to calibrate on"
        for row in summary["rows"]:
            assert row["verdict"] == dwe.INDETERMINATE
            assert row["reason"] == "no band: calibration produced none"

    def test_the_band_never_falls_below_its_floor(self, artifact):
        assert artifact["summary"]["band"]["band"] >= dwe.BAND_FLOOR
        assert dwe.calibration([{"workload": "w", "left": "a", "right": "b", "n_pairs": 3,
                                 "ratios": [1.0, 1.0, 1.0], "half_range": 0.0,
                                 "per_repeat": []}])["band"] == dwe.BAND_FLOOR


# =============================================================================================
# Gate 3 — the planted positive
# =============================================================================================


class TestTheDetectorCanActuallyFire:
    """A verdict rule that cannot return SLOWER makes every NEUTRAL it returns meaningless."""

    def test_a_planted_slowdown_on_the_real_records_is_detected(self, mutable):
        """Plant a 25% slowdown into the candidate side of every `past = 128` record and push it
        through the *shipped* verdict function. It must come back SLOWER.

        The planted numbers are never written to an artifact. A planted effect is a transformation
        of a measurement, not a measurement, and persisting one would put a fabricated timing into
        a file whose whole purpose is that its timings are not fabricated."""
        planted = 1.25
        for record in mutable["records"]:
            if record.get("workload") == TREATMENT_WORKLOAD and record.get("arm") == "candidate":
                if dwe.classify_record(record)["status"] == dwe.ACCEPTED:
                    record["speed"]["samples_ms"] = [s * planted for s in record["speed"]["samples_ms"]]
                    record["speed"]["median_ms"] *= planted
        summary = _resummarize(mutable)
        row = _row(summary, 128)
        assert row["verdict"] == dwe.SLOWER, (
            f"a planted {planted:.0%} slowdown was not detected: {row['ratios']}"
        )
        assert summary["window"]["claim"] == dwe.INDETERMINATE, (
            "the planted effect must still not produce a window claim — its neighbours are absent"
        )

    def test_a_planted_speedup_is_detected_in_the_other_direction(self, mutable):
        for record in mutable["records"]:
            if record.get("workload") == TREATMENT_WORKLOAD and record.get("arm") == "candidate":
                if dwe.classify_record(record)["status"] == dwe.ACCEPTED:
                    record["speed"]["samples_ms"] = [s / 1.25 for s in record["speed"]["samples_ms"]]
                    record["speed"]["median_ms"] /= 1.25
        assert _row(_resummarize(mutable), 128)["verdict"] == dwe.FASTER

    def test_the_unanimity_rule_refuses_a_split_decision(self):
        """One repeat outside the band and two inside is INDETERMINATE, not SLOWER. This is the
        rule that makes the protocol conservative — and :func:`power_at` is what prices it."""
        pair = {"ratios": [0.80, 1.00, 1.01]}
        graded = dwe.verdict_for(pair, 0.05, repeats_required=3)
        assert graded["verdict"] == dwe.INDETERMINATE
        assert graded["reason"] == "repeats straddle the band"

    def test_too_few_accepted_repeats_is_indeterminate_not_a_verdict(self):
        graded = dwe.verdict_for({"ratios": [0.5, 0.5]}, 0.05, repeats_required=3)
        assert graded["verdict"] == dwe.INDETERMINATE
        assert "2 accepted repeat(s), 3 required" in graded["reason"]


# =============================================================================================
# Gate 4 — the conclusion is INCONCLUSIVE, and both observations survive
# =============================================================================================


class TestThePublishedConclusionIsInconclusive:
    """`past = 128` is undecided. Not reproduced, not refuted, not explained."""

    def test_p128_is_indeterminate_in_the_artifact(self, artifact):
        row = _row(artifact["summary"], 128)
        assert row["verdict"] == dwe.INDETERMINATE
        assert row["reason"] == "repeats straddle the band"

    def test_the_p128_interval_contains_both_parity_and_the_reported_effect(self, artifact):
        """This is the whole argument for INCONCLUSIVE, and it is arithmetic, not judgement: an
        interval containing both hypotheses distinguishes neither."""
        interval = _row(artifact["summary"], 128)["interval"]
        assert interval["low"] < 1.0 < interval["high"]
        assert interval["low"] < 0.859 < interval["high"]

    def test_the_protocol_reports_how_much_it_could_have_missed(self, artifact):
        """A null with unreported power is not a null. Below 0.8, a non-detection is the expected
        outcome under *both* hypotheses and cannot be read as evidence for either."""
        power = _row(artifact["summary"], 128)["power_at_reference_effect"]
        assert power["true_ratio"] == 0.859
        assert 0.0 < power["power"] < 0.8

    def test_the_historical_ratio_is_recorded_and_not_rounded_away(self, artifact):
        assert artifact["reference_effect"] == 0.859
        assert "not as a target" in artifact["reference_effect_note"]

    def test_power_is_monotone_in_the_band_and_in_the_effect(self):
        """Sanity on the power model itself: a wider band and a smaller true effect both make a
        unanimous SLOWER harder. If this inverted, the number would be reassuring nonsense."""
        sd = 0.0655
        assert dwe.power_at(0.859, sd, 0.11, 3) < dwe.power_at(0.859, sd, 0.05, 3)
        assert dwe.power_at(0.90, sd, 0.075, 3) < dwe.power_at(0.80, sd, 0.075, 3)

    def test_an_interval_is_not_invented_from_a_single_observation(self):
        assert dwe.log_ratio_interval([1.0])["low"] is None
        assert dwe.log_ratio_interval([])["geometric_mean"] is None


# =============================================================================================
# Gate 5 — identity
# =============================================================================================


class TestIdentityIsPinnedAndDisagreementRefuses:
    """A timing whose binary cannot be named is not evidence about a binary."""

    def test_every_declared_arm_carries_a_digest_size_and_commit(self, artifact):
        for name, arm in artifact["arms"].items():
            assert re.fullmatch(r"[0-9a-f]{64}", arm["sha256"]), name
            assert isinstance(arm["bytes"], int) and arm["bytes"] > 0, name
            assert re.fullmatch(r"[0-9a-f]{40}", arm["commit"]), name

    def test_the_published_identity_agrees_with_every_accepted_record(self, artifact):
        assert artifact["summary"]["identity"]["agrees"]
        assert artifact["summary"]["identity"]["disagreements"] == []
        assert artifact["summary"]["refuses"] is False

    def test_mutation_change_one_records_digest_and_the_summary_refuses(self, mutable):
        record = next(r for r in mutable["records"] if dwe.classify_record(r)["status"] == dwe.ACCEPTED)
        record["ep_library_sha256"] = "0" * 64
        summary = _resummarize(mutable)
        assert summary["refuses"] is True
        assert not summary["identity"]["agrees"]

    def test_the_model_and_its_external_weights_are_pinned_on_every_measured_record(self, artifact):
        """26 MB of graph and 2.29 GB of external weights. Hashing only the `.onnx` would pin the
        structure and leave the weights — almost all of the bytes — unwitnessed."""
        aa = [r for r in artifact["records"] if r.get("role") == "aa"]
        assert aa
        for record in aa:
            assert re.fullmatch(r"[0-9a-f]{64}", record["model_sha256"])
            assert record["model_weights_bytes"] == 2291238912

    def test_the_ort_version_is_recorded_in_the_artifact_not_only_in_prose(self, artifact):
        """The previous revision put the device in a document and left the machine-readable
        artifact without an ONNX Runtime version anywhere. The artifact is what gets cited."""
        assert artifact["environment"]["onnxruntime"]
        assert artifact["device"]["running_device_names"] == ["NVIDIA RTX A1000"]

    def test_concurrent_gpu_users_are_disclosed(self, artifact):
        """On a run whose central number is a drift band, what else was on the GPU is the first
        thing a reader needs."""
        tenants = artifact["concurrent_gpu_users"]
        assert tenants["available"] is True
        assert isinstance(tenants["gpu_capable_by_name"], list)

    def test_the_lock_is_described_as_advisory_and_never_as_exclusivity(self, artifact):
        """An advisory file lock between this repo's own probes is not device exclusivity, and an
        artifact that called it exclusivity would be claiming a guarantee the OS did not give."""
        exclusivity = artifact["exclusivity"]
        assert "advisory" in exclusivity["scope"]
        assert "not device exclusivity" in exclusivity["not_a_claim"]
        assert exclusivity["policy"] == "wait, never kill"

    def test_the_ordering_rule_is_recorded(self, artifact):
        assert "even repeats" in artifact["environment"]["side_order_rule"]
        assert "ONNXRUNTIME_EP_VULKAN_" in artifact["environment"]["env_hygiene"]

    def test_no_absolute_path_is_published_anywhere_in_the_artifact(self, artifact):
        from importlib import util

        spec = util.spec_from_file_location("_probe_aa", PROBE_PATH)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.assert_public(artifact)  # raises PublicPathError on any absolute path

    def test_the_public_path_guard_is_not_vacuous(self, artifact):
        from importlib import util

        spec = util.spec_from_file_location("_probe_aa2", PROBE_PATH)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(module.PublicPathError):
            module.assert_public({"leak": r"C:\Users\someone\secret\model.onnx"})


# =============================================================================================
# Gate 7 — --resummarize --check binds the raw
# =============================================================================================


class TestCheckBindsThePublishedRawSamples:
    """`--check` must fail on the mutations it exists to catch. Verified one at a time."""

    def test_the_committed_artifact_reproduces_from_its_own_records(self, artifact):
        result = dwe.check(artifact)
        assert result["reproduces"], result["differences"][:5]

    def test_mutation_scale_every_raw_sample_and_reproduction_fails(self, mutable):
        """Verified against the previous revision at its own head: multiplying every
        `samples_ms` array by 10 left `--check` at rc 0, because the summarizer only ever read
        `median_ms`. Here the median has to re-derive from the samples, so it cannot."""
        for record in mutable["records"]:
            speed = record.get("speed")
            if isinstance(speed, dict) and speed.get("samples_ms"):
                speed["samples_ms"] = [s * 10 for s in speed["samples_ms"]]
        assert not dwe.check(mutable)["reproduces"]

    def test_mutation_delete_every_raw_sample_and_reproduction_fails(self, mutable):
        for record in mutable["records"]:
            speed = record.get("speed")
            if isinstance(speed, dict):
                speed.pop("samples_ms", None)
        assert not dwe.check(mutable)["reproduces"]

    def test_mutation_strip_a_witness_and_reproduction_fails(self, mutable):
        record = next(r for r in mutable["records"] if dwe.classify_record(r)["status"] == dwe.ACCEPTED)
        record.pop("path_witness")
        assert not dwe.check(mutable)["reproduces"]

    def test_mutation_refuse_a_length_and_reproduction_fails(self, mutable):
        for record in mutable["records"]:
            if record.get("workload") == TREATMENT_WORKLOAD:
                record["admissible"] = False
                record["refusal"] = {"reason": "mutation"}
                record.pop("speed", None)
        assert not dwe.check(mutable)["reproduces"]

    def test_mutation_contaminate_the_aa_arm_and_reproduction_fails(self, mutable):
        treatment = next(
            r for r in mutable["records"]
            if r.get("role") != "aa" and dwe.classify_record(r)["status"] == dwe.ACCEPTED
        )
        aa_record = next(r for r in mutable["records"] if r.get("role") == "aa")
        aa_record["pid"] = treatment["pid"]
        aa_record["started_at"] = treatment["started_at"]
        assert not dwe.check(mutable)["reproduces"]

    def test_mutation_make_an_identity_disagree_and_reproduction_fails(self, mutable):
        mutable["arms"]["candidate"]["sha256"] = "1" * 64
        assert not dwe.check(mutable)["reproduces"]

    def test_an_artifact_with_no_summary_does_not_silently_pass(self):
        assert not dwe.check({"records": []})["reproduces"]

    def test_the_shipped_check_command_exits_zero_on_the_committed_artifact(self):
        """The production entry point, not just the library behind it."""
        result = subprocess.run(
            [sys.executable, str(PROBE_PATH), "--resummarize", str(ARTIFACT_PATH), "--check"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "reproduces=YES" in result.stdout


# =============================================================================================
# Gate 1/6 — provenance of the reused records
# =============================================================================================


class TestTheReusedRecordsAreWhatTheySayTheyAre:
    """54 records were carried over rather than re-measured. That has to be checkable."""

    def test_the_reused_file_declares_its_source_and_hashes(self, artifact):
        reused = json.loads(REUSED_PATH.read_text(encoding="utf-8"))
        provenance = reused["provenance"]
        assert len(reused["records"]) == 54
        assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_artifact_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", provenance["records_blob_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", provenance["records_blob_sha256_in_source"])
        assert "cf6e3f5442f2bf590405e82b35b1e48974079498" in provenance["measured_by"]

    def test_the_reused_records_blob_hashes_to_its_declared_digest(self):
        """The records were lifted out as a literal slice, re-emitted with LF terminators because
        the source document is CRLF. Re-slice the committed file the same way and the digest has
        to come back — otherwise "byte-identical" is a word, not a property. The source-side
        digest is published alongside so the CRLF original can be checked too."""
        import hashlib

        text = REUSED_PATH.read_text(encoding="utf-8")
        start = text.index('"records": [') + len('"records": ') 
        depth, end = 0, None
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        blob = text[start:end]
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        declared = json.loads(text)["provenance"]["records_blob_sha256"]
        assert digest == declared

    def test_the_derived_blocks_of_the_rejected_artifact_were_not_carried_over(self):
        """What was rejected was the summarizer, not the records. Importing the old band, its
        verdicts or its window claim would import the rejection with them."""
        reused = json.loads(REUSED_PATH.read_text(encoding="utf-8"))
        assert set(reused) == {"schema", "what", "why_here", "provenance", "records"}
        assert "NO-SLOW-LENGTH" not in REUSED_PATH.read_text(encoding="utf-8")

    def test_every_reused_record_is_present_unchanged_in_the_evidence_artifact(self, artifact):
        reused = json.loads(REUSED_PATH.read_text(encoding="utf-8"))["records"]
        carried = [r for r in artifact["records"] if r.get("role") != "aa"]
        assert carried == reused

    def test_the_treatment_records_were_not_re_timed_by_this_probe(self, artifact):
        provenance = artifact["records_provenance"]
        assert provenance["treatment"]["count"] == 54
        assert provenance["treatment"]["reused_byte_identically"] is True
        assert provenance["calibration"]["measured_here"] == 12


# =============================================================================================
# Gate 6 — the compiled delta, enumerated
# =============================================================================================


class TestTheCompiledDeltaIsEnumeratedInFull:
    """"The shader and nothing else" was false in both previous revisions."""

    def test_the_proof_ledger_is_a_compiled_input(self):
        """`evidence/proof_ledger.jsonl` is `include_str!`'d into the crate, so it is baked into
        the binary. It differs between the two arms. Any enumeration of the compiled delta that
        stops at `rust/` is therefore incomplete, and the two DLLs differ by more than a shader."""
        registry = (_ROOT / "rust" / "src" / "registry.rs").read_text(encoding="utf-8")
        assert 'include_str!("../../evidence/proof_ledger.jsonl")' in registry

    def test_the_documents_name_all_three_compiled_inputs(self):
        perf = PERF_PATH.read_text(encoding="utf-8")
        section = perf[perf.index("## 27."):]
        for compiled_input in ("gqa_f16.comp", "attention.rs", "proof_ledger.jsonl"):
            assert compiled_input in section, f"{compiled_input} is not named in PERF §27"

    def test_no_isolated_shader_causality_is_claimed(self):
        """With three compiled inputs differing, no timing difference can be attributed to the
        shader alone. The documents must not say it can."""
        perf = PERF_PATH.read_text(encoding="utf-8")
        section = perf[perf.index("## 27."):]
        for forbidden in (
            "caused by the shader",
            "the shader is responsible",
            "attributable to gqa_f16.comp alone",
        ):
            assert forbidden not in section


# =============================================================================================
# Gate 8 — the documents say what the artifact says
# =============================================================================================


def _perf_section() -> str:
    perf = PERF_PATH.read_text(encoding="utf-8")
    return perf[perf.index("## 27."):]


#: The subsection this change adds to DESIGN §8.13. Forbidden-phrase checks are scoped to it
#: rather than to the whole document: `docs/DESIGN.md` already says "does not reproduce" twice at
#: base, about a counters extract and about lavapipe, and a guard that reddens on unrelated
#: pre-existing prose would be a guard nobody could keep.
DESIGN_MARKER = "#### Whether that decode geometry costs anything — still open (issue #96)"


def _design_section() -> str:
    design = DESIGN_PATH.read_text(encoding="utf-8")
    start = design.index(DESIGN_MARKER)
    end = design.index("\n## 9.", start)
    return design[start:end]


class TestTheDocumentsMatchTheArtifact:
    """Every figure in prose is a figure in the artifact, and the prose preserves the uncertainty."""

    def test_perf_reports_p128_as_inconclusive(self):
        section = _perf_section()
        assert "INCONCLUSIVE" in section

    def test_no_document_claims_the_regression_failed_to_reproduce(self):
        """"Does not reproduce" is the exact wording the review rejected: an underpowered null is
        a non-replication at best, and this run cannot even claim that."""
        for name, text in (("PERF §27", _perf_section()), ("DESIGN §8.13", _design_section())):
            for forbidden in ("does not reproduce", "did not reproduce", "no regression",
                              "NO-SLOW-LENGTH"):
                assert forbidden.lower() not in text.lower(), f"{name} claims: {forbidden}"

    def test_both_observations_are_preserved_in_both_documents(self):
        """A design doc that keeps only the reading that agrees with it is how an open question
        gets closed by attrition."""
        for path in (PERF_PATH, DESIGN_PATH):
            text = path.read_text(encoding="utf-8")
            assert "0.859" in text, f"{path.name} has lost the earlier observation"
        assert "0.9651" in _perf_section()
        assert "0.9651" in _design_section()

    def test_perf_quotes_the_bands_actual_value(self, artifact):
        assert f"{artifact['summary']['band']['band']:.4f}"[:6] in _perf_section()

    def test_perf_quotes_the_interval_and_the_power(self, artifact):
        section = _perf_section()
        interval = _row(artifact["summary"], 128)["interval"]
        assert f"{interval['low']:.3f}" in section
        assert f"{interval['high']:.3f}" in section
        power = _row(artifact["summary"], 128)["power_at_reference_effect"]["power"]
        assert f"{power:.2f}" in section or f"{power * 100:.0f}%" in section

    def test_perf_states_the_unmeasured_lengths_exactly(self, artifact):
        section = _perf_section()
        for length in artifact["summary"]["window"]["unmeasured_lengths"]:
            assert str(length) in section
        assert "no admissible pair" in section

    def test_design_does_not_overstate_the_coverage(self):
        """The previous revision's DESIGN said "across 6 KV lengths ... no measured decode length
        is SLOWER", from a run with admissible pairs at three of them."""
        text = DESIGN_PATH.read_text(encoding="utf-8")
        assert "6 KV lengths" not in text
        assert "three of the six" in text or "3 of the 6" in text

    def test_the_documents_do_not_claim_a_causal_explanation_for_the_disagreement(self):
        """Two runs disagree and nobody knows why. Naming a cause — thermal state, a background
        process, a protocol difference — without measuring it would be the same failure that put
        an unmeasured length in the measured set."""
        for name, section in (("PERF §27", _perf_section()), ("DESIGN §8.13", _design_section())):
            for forbidden in ("explained by thermal", "caused by background", "because the machine was",
                              "due to contention from"):
                assert forbidden not in section, f"{name} claims: {forbidden}"

    def test_the_issue_is_not_declared_closed(self):
        section = _perf_section()
        assert "remains open" in section
        assert "closed" not in section.split("remains open")[0][-400:]

    def test_perf_discloses_that_the_calibration_postdates_the_treatment(self):
        """The A/A arms were measured on a later day than the records they grade. That is a real
        limitation — a band from one desk-state applied to another — and hiding it would make the
        band look stronger than it is."""
        section = _perf_section()
        assert "later session" in section or "a different session" in section


class TestTheProseCountsAreCorrect:
    """The previous revision's PERF said 96 tests; the file had 92 and collected 99."""

    def test_the_record_counts_in_prose_match_the_artifact(self, artifact):
        section = _perf_section()
        counts = artifact["summary"]["counts"]
        assert str(counts["records"]) in section
        assert str(counts["accepted"]) in section
        assert str(counts["refused"]) in section

    def test_no_test_total_is_asserted_in_the_committed_documents(self):
        """A test count in prose is stale the moment a test is added, and it was stale twice.
        The count belongs in a report, derived by running the suite, not in a document."""
        for name, section in (("PERF §27", _perf_section()), ("DESIGN §8.13", _design_section())):
            assert not re.search(r"\b\d+\s+tests?,\s+GPU-free", section), name


# =============================================================================================
# Two-polarity coverage for every public instrument in the summarizer
# =============================================================================================
#
# `rust/tools/audit_instruments.py` scores an instrument `unfalsified` until it has watched that
# instrument DISAGREE — nothing having ever seen it refuse, a broken one and a working one look
# the same. These functions are TOTAL: they withhold a verdict instead of raising, so the
# `pytest.raises` model is blind to them and `bench/_polarity.py::withholds` is the enforcing
# assertion that makes the refusal observable. It raises when the thing inside it reached a
# verdict anyway, and again when it withheld one without saying why.
#
# The tests below are not a lexical duplication of the grid above. Each one puts the shipped
# function on the exact input where it is supposed to decline, so an implementation that declined
# to decline goes red here.


class TestEveryInstrumentCanBeWatchedRefusing:
    """The reject polarity, one instrument at a time."""

    def test_classify_refuses_a_record_with_no_samples(self, artifact):
        record = copy.deepcopy(next(r for r in artifact["records"] if r.get("speed")))
        record["speed"]["samples_ms"] = []
        why = withholds(dwe.classify_record(record), "status", because="a timing with no samples")
        assert "samples" in why.lower()

    def test_classify_accepts_the_records_the_artifact_says_it_accepts(self, artifact):
        """The accept polarity of the same instrument, on the same shipped code path."""
        accepted_here = [r for r in artifact["records"] if dwe.classify_record(r)["status"] == dwe.ACCEPTED]
        assert len(accepted_here) == artifact["summary"]["counts"]["accepted"]

    def test_accepted_withholds_every_record_when_every_record_is_refused(self, artifact):
        refused = [r for r in artifact["records"] if dwe.classify_record(r)["status"] == dwe.REFUSED]
        assert refused
        withholds(dwe.accepted(refused), because="a record set with nothing admissible in it")
        assert dwe.accepted(artifact["records"])

    def test_status_counts_withholds_the_accepted_count_when_nothing_is_admissible(self, artifact):
        refused = [r for r in artifact["records"] if dwe.classify_record(r)["status"] == dwe.REFUSED]
        withholds(dwe.status_counts(refused), "accepted", because="every record refused")
        assert dwe.status_counts(artifact["records"])["accepted"] > 0

    def test_paired_withholds_a_ratio_when_one_side_is_missing(self, artifact):
        one_sided = [
            r for r in artifact["records"]
            if r.get("workload") == TREATMENT_WORKLOAD and r.get("arm") == "candidate"
        ]
        assert one_sided
        why = withholds(dwe.paired(one_sided, TREATMENT_WORKLOAD), "n_pairs",
                        because="no repeat has a record on both sides")
        assert dwe.paired(artifact["records"], TREATMENT_WORKLOAD)["n_pairs"] == 3
        assert why == ""

    def test_past_of_withholds_a_length_for_a_label_that_is_not_a_decode_row(self):
        withholds(dwe.past_of(f"{TREATMENT_WORKLOAD}/aa-candidate"), because="an A/A label")
        withholds(dwe.past_of("mobilenetv2-12/batch/N1"), because="a batch label")
        assert dwe.past_of(TREATMENT_WORKLOAD) == 128

    def test_calibration_withholds_a_band_with_no_aa_arm(self, artifact):
        why = withholds(dwe.calibration([]), "band", because="nothing to calibrate on")
        assert "no admissible A/A pair" in why
        arms = [dwe.paired(artifact["records"], a["workload"], left=a["left"], right=a["right"])
                for a in artifact["aa_allocations"]]
        assert dwe.calibration(arms)["band"] is not None

    def test_verdict_for_withholds_a_verdict_with_no_band(self, artifact):
        pair = dwe.paired(artifact["records"], TREATMENT_WORKLOAD)
        why = withholds(dwe.verdict_for(pair, None, repeats_required=3), "verdict",
                        because="a comparison with no scale to read it against")
        assert "no band" in why
        assert dwe.verdict_for(pair, 0.0748, repeats_required=3)["verdict"] == dwe.INDETERMINATE

    def test_log_ratio_interval_withholds_an_interval_from_one_observation(self):
        why = withholds(dwe.log_ratio_interval([0.96]), "low",
                        because="a single observation has no interval")
        assert "at least two" in why
        assert dwe.log_ratio_interval([0.96, 1.02, 0.89])["low"] is not None

    def test_power_at_withholds_a_number_when_there_is_no_dispersion_to_speak_of(self):
        withholds(dwe.power_at(0.859, 0.0, 0.0748, 3), because="zero dispersion is not a model")
        withholds(dwe.power_at(0.859, 0.05, 0.0748, 0), because="no repeats to be unanimous over")
        assert 0.0 < dwe.power_at(0.859, 0.0655, 0.0748, 3) < 1.0

    def test_window_verdict_withholds_a_claim_when_a_neighbour_is_missing(self):
        why = withholds(dwe.window_verdict({128: dwe.NEUTRAL, 512: dwe.NEUTRAL}), "claim",
                        because="the target's predecessor was never measured")
        assert "64" in why
        full = dwe.window_verdict({64: dwe.NEUTRAL, 128: dwe.NEUTRAL, 256: dwe.NEUTRAL})
        assert full["claim"] != dwe.INDETERMINATE

    def test_identity_agreement_withholds_agreement_when_an_arm_carries_two_binaries(self, artifact):
        records = copy.deepcopy(artifact["records"])
        target = next(r for r in records if r.get("arm") == "candidate")
        target["ep_library_sha256"] = "9" * 64
        why = withholds(dwe.identity_agreement(records, artifact["arms"]), "agrees",
                        because="one arm observed under two different binaries")
        assert "candidate" in why
        assert dwe.identity_agreement(artifact["records"], artifact["arms"])["agrees"] is True

    def test_summarize_withholds_the_band_when_the_calibration_is_gone(self, mutable):
        mutable["records"] = [r for r in mutable["records"] if r.get("role") != "aa"]
        why = withholds(
            dwe.summarize(
                mutable["records"],
                arms=mutable["arms"],
                repeats_required=mutable["environment"]["repeats"],
                aa_allocations=mutable["aa_allocations"],
                reference_effect=mutable.get("reference_effect"),
            ),
            "band.band",
            because="the A/A arms were deleted, so there is no scale to grade against",
        )
        assert "no admissible A/A pair" in why
        assert _resummarize(json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")))["band"]["band"]

    def test_summarize_withholds_the_window_when_the_neighbours_are_gone(self, mutable):
        withholds(
            dwe.summarize(
                mutable["records"],
                arms=mutable["arms"],
                repeats_required=mutable["environment"]["repeats"],
                aa_allocations=mutable["aa_allocations"],
                reference_effect=mutable.get("reference_effect"),
            ),
            "window.claim",
            because="64 and 256 were never measured, so no window claim is available",
        )

    def test_check_withholds_reproduction_when_the_samples_are_scaled(self, mutable):
        for record in mutable["records"]:
            speed = record.get("speed")
            if speed:
                speed["samples_ms"] = [s * 1.5 for s in speed["samples_ms"]]
        why = withholds(dwe.check(mutable), "reproduces", because="the raw samples were scaled")
        assert why
        assert dwe.check(json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")))["reproduces"]


class TestTheWithholdsAssertionIsNotAnAnnotation:
    """`withholds` earns its polarity credit the way `pytest.raises` does — by failing the test
    when the thing inside it did not withhold. If it credited a marker it would be the Guard D
    shape with the sign flipped, which is the defect `bench/_polarity.py` exists to avoid."""

    def test_it_rejects_an_instrument_that_reached_a_verdict(self):
        from _polarity import PolarityError

        with pytest.raises(PolarityError):
            withholds({"band": 0.0748}, "band", because="a band that exists is not a refusal")

    def test_it_rejects_a_refusal_that_gives_no_reason(self):
        from _polarity import PolarityError

        with pytest.raises(PolarityError):
            withholds({"band": None}, "band", because="a silent refusal")

    def test_it_rejects_a_field_that_is_not_there(self):
        from _polarity import PolarityError

        with pytest.raises(PolarityError):
            withholds({"reason": "x"}, "band", because="an absent field is not a withheld verdict")

    def test_it_requires_the_refusal_case_to_be_named(self):
        from _polarity import PolarityError

        with pytest.raises(PolarityError):
            withholds({"band": None, "reason": "x"}, "band", because="   ")
