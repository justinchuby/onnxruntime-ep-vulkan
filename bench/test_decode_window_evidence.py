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
  regression failed to reproduce;
* **the prose reads the artifact in the direction the code computes** — the ratio convention is
  `baseline / candidate`, it is stated that way in both documents, the inverted phrasings are
  grepped for by name, the session separation is quoted to the second from the records, the
  calibration's protocol delta is disclosed with both call counts, the recycled PID named in prose
  is the one the records actually share, and the power figure is published as an upper bound.

Every test is GPU-free. They read the committed artifact and never measure anything.
"""

from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import sys
from datetime import datetime
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
        PIDs and these two runs are hours apart, so the same integer really does appear on both
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
        section = _perf_section()
        for compiled_input in ("gqa_f16.comp", "attention.rs", "proof_ledger.jsonl"):
            assert compiled_input in section, f"{compiled_input} is not named in PERF §27"

    def test_no_isolated_shader_causality_is_claimed(self):
        """With three compiled inputs differing, no timing difference can be attributed to the
        shader alone. The documents must not say it can."""
        section = _perf_section()
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
    """§27 and only §27.

    The slice is **bounded at the next top-level heading**. It used to run to the end of the
    file, which was the same text while §27 was last; it stopped being the same text when
    issue #90 added §28 (the KV-parallel decode module) directly below. An unbounded slice
    grades a *different* change's prose against *this* artifact — every guard below would have
    been screening §28's sentences, and the forbidden-phrase checks would have been the first
    to fire on prose that was never about this measurement.
    """
    perf = PERF_PATH.read_text(encoding="utf-8")
    start = perf.index("## 27.")
    nxt = perf.find("\n## ", start + 1)
    return perf[start:] if nxt == -1 else perf[start:nxt]


#: The subsection this change adds to DESIGN §8.13. Forbidden-phrase checks are scoped to it
#: rather than to the whole document: `docs/DESIGN.md` already says "does not reproduce" twice at
#: base, about a counters extract and about lavapipe, and a guard that reddens on unrelated
#: pre-existing prose would be a guard nobody could keep.
DESIGN_MARKER = "#### Whether that decode geometry costs anything — still open (issue #96)"


def _design_section() -> str:
    """The #96 subsection, bounded at the next heading of any level — for the same reason
    `_perf_section` is bounded: issue #90 added §8.14 immediately below it, and `\\n## 9.` is no
    longer the first thing that follows."""
    design = DESIGN_PATH.read_text(encoding="utf-8")
    start = design.index(DESIGN_MARKER)
    ends = [i for i in (design.find("\n### ", start), design.find("\n## ", start)) if i != -1]
    return design[start:min(ends)] if ends else design[start:]


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
        """The A/A arms were measured hours after the records they grade, in a separate session.
        That is a real limitation — a band from one desk-state applied to another — and hiding it
        would make the band look stronger than it is. The exact spans are held by
        `TestTheSessionSeparationIsQuotedFromTheRecords`; this only checks the disclosure exists."""
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
# Gate 9 — the prose reads the artifact in the direction the code computes
# =============================================================================================
#
# Four claims were rejected on this file's previous head, and every one of them was a prose
# statement that the shipped code contradicted: the ratio convention was printed upside down, the
# two sessions were called "days apart" when they are hours apart, the calibration was called "the
# same protocol" when it issued two fewer inference calls per record, and the admissibility
# function was called `classify()` after it had been renamed. None of them could be caught by a
# test that only read the artifact, because the artifact was right and the sentence about it was
# wrong. So these tests read the documents and the module's own docstrings, and derive every
# figure they check from the records rather than restating it.


def _sections() -> tuple:
    return (("PERF §27", _perf_section()), ("DESIGN §8.13", _design_section()))


def _shared_pid_events(artifact) -> list:
    """Start times of every accepted record whose PID appears on both sides of the session
    boundary. The PID-recycling argument is exactly the claim that these timestamps are far
    apart, so the span between the outermost two is a figure the documents may quote."""
    treatment, calibration = _by_role(artifact)
    left, right = dwe.accepted(treatment), dwe.accepted(calibration)
    shared = {r["pid"] for r in left} & {r["pid"] for r in right}
    return [_dt(r["started_at"]) for r in left + right if r["pid"] in shared]


# The two bands a power figure in this section can belong to, and the power each one yields against
# the 0.859 reference effect. They are written down here so that a document quoting one power at the
# other band, or dropping the band entirely, is a test failure rather than a reading comprehension
# exercise. Every one of them is re-derived from `power_at` before it is compared to the prose.
SELECTED_BAND = 0.0748      # max(0.05, widest A/A half-range) from this artifact's calibration
SELECTED_POWER = 0.6621
OLDER_BAND_FLOOR = 0.1106   # the CNN-derived floor a previous revision used
OLDER_POWER = 0.346


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _span(seconds: float) -> str:
    """`4h 45m 56s`. The one rendering the documents are allowed to use, so a drifted figure is a
    string mismatch rather than a judgement call."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def _by_role(artifact: dict) -> tuple:
    treatment = [r for r in artifact["records"] if r.get("role") != "aa"]
    calibration = [r for r in artifact["records"] if r.get("role") == "aa"]
    assert treatment and calibration
    return treatment, calibration


def _ended(records: list) -> list:
    """Refused records are written without a `finished_at` — the worker never reached the end of
    the protocol. A span derived from them would be derived from a key that is not there, so the
    ends of a session are taken from the records that actually have one."""
    ended = [r for r in records if r.get("finished_at")]
    assert ended, "no record in this set carries a finish time"
    return ended


class TestTheRatioConventionIsTheOneTheCodeImplements:
    """0.9651 is a 3.5% move and the convention is the only thing that says which way."""

    def test_the_constant_names_baseline_over_candidate(self):
        assert dwe.RATIO_CONVENTION.startswith("baseline_median_ms / candidate_median_ms")
        assert "> 1 means the candidate is faster" in dwe.RATIO_CONVENTION

    def test_paired_puts_the_baseline_on_the_left_of_the_division(self, artifact):
        """Structural, not numerical: the left side of the ratio is the record whose arm is
        `baseline`. Swap the operands in `paired()` and this goes red without any number moving."""
        pair = dwe.paired(artifact["records"], TREATMENT_WORKLOAD)
        assert (pair["left"], pair["right"]) == ("baseline", "candidate")
        for row in pair["per_repeat"]:
            baseline = next(
                r for r in artifact["records"]
                if r.get("workload") == TREATMENT_WORKLOAD
                and r.get("arm") == "baseline"
                and r.get("repeat") == row["repeat"]
            )
            assert row["left_median_ms"] == baseline["speed"]["median_ms"]
            assert row["ratio"] == row["left_median_ms"] / row["right_median_ms"]

    def test_a_candidate_that_takes_longer_gives_a_ratio_below_one(self, mutable):
        """The direction, end to end, through the shipped functions: double the candidate's
        timings and the ratio must fall *below* 1 and the verdict must be SLOWER. Under the
        inverted convention the prose used to state, this same input would read as a speedup."""
        for record in mutable["records"]:
            if record.get("workload") == TREATMENT_WORKLOAD and record.get("arm") == "candidate":
                if dwe.classify_record(record)["status"] == dwe.ACCEPTED:
                    record["speed"]["samples_ms"] = [s * 2 for s in record["speed"]["samples_ms"]]
                    record["speed"]["median_ms"] *= 2
        pair = dwe.paired(mutable["records"], TREATMENT_WORKLOAD)
        assert all(r < 1.0 for r in pair["ratios"]), pair["ratios"]
        assert dwe.verdict_for(pair, 0.05, repeats_required=3)["verdict"] == dwe.SLOWER

    def test_a_candidate_that_takes_less_time_gives_a_ratio_above_one(self, mutable):
        for record in mutable["records"]:
            if record.get("workload") == TREATMENT_WORKLOAD and record.get("arm") == "candidate":
                if dwe.classify_record(record)["status"] == dwe.ACCEPTED:
                    record["speed"]["samples_ms"] = [s / 2 for s in record["speed"]["samples_ms"]]
                    record["speed"]["median_ms"] /= 2
        pair = dwe.paired(mutable["records"], TREATMENT_WORKLOAD)
        assert all(r > 1.0 for r in pair["ratios"]), pair["ratios"]
        assert dwe.verdict_for(pair, 0.05, repeats_required=3)["verdict"] == dwe.FASTER

    def test_the_artifact_publishes_the_convention_the_module_defines(self, artifact):
        assert artifact["summary"]["ratio_convention"] == dwe.RATIO_CONVENTION
        for pair in artifact["summary"]["aa_arms"]:
            assert pair["ratio_convention"] == dwe.RATIO_CONVENTION

    def test_the_documents_state_the_ratio_convention_the_code_implements(self):
        """Named in `docs/PERF.md` §27.3 by this exact test name, so the document says which test
        holds it and the test says which document it holds."""
        for name, section in _sections():
            assert "baseline_median_ms / candidate_median_ms" in section \
                or "baseline_median_ms ÷ candidate_median_ms" in section, \
                f"{name} never states the ratio convention"
            assert "above 1 the candidate is faster" in section.lower(), \
                f"{name} omits the direction"

    def test_no_document_inverts_the_convention(self):
        """The inverted phrasings are enumerated in the module, not here, so the guard and the
        thing it guards cannot drift apart."""
        for name, section in _sections():
            for phrase in dwe.INVERTED_CONVENTION_PHRASES:
                assert phrase.lower() not in section.lower(), f"{name} inverts the ratio: {phrase}"

    def test_every_direction_claim_in_the_documents_runs_the_right_way(self):
        """The phrase list can only catch inversions somebody thought to enumerate, and §27 states
        the direction in more than one place — so a document that fixes one sentence and leaves
        another inverted passes a presence check. This finds *every* sentence that ties a side of
        1.0 to a direction, however it is worded, and checks each one individually."""
        expected = {("above", "faster"), ("below", "slower")}
        for name, section in _sections():
            claims = [(m.group(1).lower(), m.group(2).lower())
                      for m in dwe.CONVENTION_DIRECTION_CLAIM.finditer(section)]
            assert claims, f"{name} never ties a side of 1.0 to a direction"
            for side, direction in claims:
                side = {"over": "above", "under": "below"}.get(side, side)
                assert (side, direction) in expected, (
                    f"{name} says {side} 1.0 the candidate is {direction}; the summarizer divides "
                    f"{dwe.RATIO_CONVENTION}, so that is backwards"
                )

    def test_no_document_hands_the_headline_row_a_directional_verdict(self, artifact):
        """The tables are now checked cell by cell, but the headline claim is a sentence, and a
        sentence can promote `past = 128` to SLOWER or FASTER while every table stays honest. Two
        rules, both sentence-scoped: no sentence may pair the headline row with a directional
        verdict, and no sentence may name a directional verdict as the answer, verdict, result or
        conclusion. Sentences that merely define the vocabulary ("a row is SLOWER or FASTER only
        if …") name no row and reach no conclusion, so they are untouched."""
        assert _row(artifact["summary"], 128)["verdict"] == dwe.INDETERMINATE
        directional = re.compile(r"\b(SLOWER|FASTER)\b")
        attribution = re.compile(
            r"\b(answer|verdict|conclusion|result)\b[^.\n]{0,120}?\b(SLOWER|FASTER)\b", re.I)
        for name, section in _sections():
            for sentence in re.split(r"(?<=[.!?])\s+", section):
                if re.search(r"past\s*=?\s*128", sentence) and directional.search(sentence):
                    raise AssertionError(
                        f"{name} gives the headline row a directional verdict: {sentence.strip()}"
                    )
                found = attribution.search(sentence)
                assert not found, (
                    f"{name} states a directional verdict as the finding: {sentence.strip()}"
                )

    def test_no_document_claims_a_speedup(self, artifact):
        """This artifact contains no row that clears the band upward in every repeat, so no
        speedup is available to claim, and the revision may not imply one."""
        assert not any(row["verdict"] == dwe.FASTER for row in artifact["summary"]["rows"])
        for name, section in _sections():
            for phrase in ("is a speedup", "shows a speedup", "a measured speedup",
                           "the candidate was faster", "the candidate ran faster",
                           "faster after the change"):
                assert phrase.lower() not in section.lower(), f"{name} claims: {phrase}"


class TestTheCalibrationProtocolDeltaIsDisclosed:
    """The band came from an arm that did not run the treatment's protocol. Say so, or go red."""

    def test_the_two_arms_did_not_issue_the_same_number_of_inference_calls(self, artifact):
        treatment, calibration = _by_role(artifact)
        assert {r["inference_calls"] for r in dwe.accepted(treatment)} == {27}
        assert {r["inference_calls"] for r in dwe.accepted(calibration)} == {25}
        assert {r["speed"]["n"] for r in dwe.accepted(treatment + calibration)} == {20}

    def test_protocol_delta_reports_the_mismatch_and_names_it(self, artifact):
        delta = dwe.protocol_delta(artifact["records"])
        assert delta["matched"] is False
        assert delta["differences"]
        assert "inference_calls" in delta["differences"][0]
        assert "not of the treatment session's noise" in delta["reason"]

    def test_protocol_delta_agrees_when_the_two_arms_agree(self, mutable):
        """The accept polarity. Give the calibration the treatment's call count and the instrument
        must stop objecting — otherwise it objects to everything and its objection means nothing."""
        for record in mutable["records"]:
            if record.get("role") == "aa":
                record["inference_calls"] = 27
        delta = dwe.protocol_delta(mutable["records"])
        assert delta["matched"] is True
        assert delta["differences"] == []

    def test_the_documents_disclose_both_call_counts(self):
        section = _perf_section()
        assert "inference call" in section
        assert "25" in section and "27" in section

    def test_no_document_claims_the_calibration_ran_the_same_protocol(self):
        for name, section in _sections():
            for phrase in ("runs the same protocol", "ran the same protocol",
                           "of the same protocol", "same protocol as the records",
                           "under the same protocol as the treatment"):
                assert phrase.lower() not in section.lower(), f"{name} claims: {phrase}"

    def test_no_document_calls_the_band_a_bound_on_the_treatment_noise(self):
        """A "lower bound on the noise" is a directional claim, and nothing measures the
        direction. Either bound is a smoothed version of "not a measurement of it at all"."""
        for name, section in _sections():
            for phrase in ("lower bound on the noise", "upper bound on the noise",
                           "band is a lower bound", "band is an upper bound"):
                assert phrase.lower() not in section.lower(), f"{name} claims: {phrase}"

    def test_perf_states_the_consequence_for_the_neutral_rows(self):
        """A NEUTRAL row graded against this band is not "no difference at this length"."""
        section = _perf_section()
        assert "no difference wider than a band" in section


class TestTheSessionSeparationIsQuotedFromTheRecords:
    """"Days apart" was false. The replacement is arithmetic on the records, to the second."""

    def test_the_sessions_are_hours_apart_not_days(self, artifact):
        treatment, calibration = _by_role(artifact)
        gap = (min(_dt(r["started_at"]) for r in calibration)
               - max(_dt(r["finished_at"]) for r in _ended(treatment))).total_seconds()
        assert 0 < gap < 24 * 3600, gap

    def test_perf_quotes_every_separation_it_states(self, artifact):
        """Each span is re-derived here and looked up as a literal string in the document, so a
        figure that drifts from the records fails rather than ages."""
        treatment, calibration = _by_role(artifact)
        t_start = min(_dt(r["started_at"]) for r in treatment)
        t_end = max(_dt(r["finished_at"]) for r in _ended(treatment))
        a_start = min(_dt(r["started_at"]) for r in calibration)
        a_end = max(_dt(r["finished_at"]) for r in _ended(calibration))
        p128 = [r for r in treatment if r.get("workload") == TREATMENT_WORKLOAD]
        section = _perf_section()
        for label, seconds in (
            ("treatment end → calibration start", (a_start - t_end).total_seconds()),
            ("first record → last record", (a_end - t_start).total_seconds()),
            ("last past128 record → calibration start",
             (a_start - max(_dt(r["finished_at"]) for r in _ended(p128))).total_seconds()),
        ):
            assert _span(seconds) in section, f"PERF §27 does not state {label} ({_span(seconds)})"

    def test_the_documents_do_not_call_the_two_sessions_days_apart(self):
        for name, section in _sections():
            for phrase in ("days apart", "different days", "on a different day",
                           "a different day", "the next day", "the following day", "overnight"):
                assert phrase.lower() not in section.lower(), f"{name} claims: {phrase}"

    def test_no_document_measures_the_separation_in_days(self):
        """The phrase list above is a list of the exact wordings that were wrong. This catches the
        claim itself: any counted quantity of days. "the same day", "that day" and "day-scale
        wording" — which the retraction needs — are not counted quantities and are left alone."""
        counted_days = re.compile(
            r"\b(?:\d+|one|two|three|four|five|several|a few|a couple of)\s+days?\b", re.I)
        for name, section in _sections():
            found = counted_days.findall(section) or counted_days.search(section)
            assert not found, f"{name} measures the session separation in days: {found}"

    def test_perf_invents_no_separation_that_the_records_do_not_support(self, artifact):
        """The mirror of the test above it. That one demands the true spans be present; this one
        demands that nothing *else* span-shaped is present, so a fourth figure cannot be added by
        hand and go unchecked because the three required ones are still there."""
        treatment, calibration = _by_role(artifact)
        t_start = min(_dt(r["started_at"]) for r in treatment)
        t_end = max(_dt(r["finished_at"]) for r in _ended(treatment))
        a_start = min(_dt(r["started_at"]) for r in calibration)
        a_end = max(_dt(r["finished_at"]) for r in _ended(calibration))
        p128 = [r for r in treatment if r.get("workload") == TREATMENT_WORKLOAD]
        shared = _shared_pid_events(artifact)
        derivable = {
            _span((a_start - t_end).total_seconds()),
            _span((a_end - t_start).total_seconds()),
            _span((a_start - max(_dt(r["finished_at"]) for r in _ended(p128))).total_seconds()),
            _span(abs((max(shared) - min(shared)).total_seconds())),
        }
        for name, section in _sections():
            printed = set(re.findall(r"\d+h \d+m \d+s", section))
            assert printed <= derivable, (
                f"{name} states a separation that does not re-derive from the records: "
                f"{sorted(printed - derivable)}"
            )

    def test_the_headline_row_and_the_calibration_ran_on_the_same_date(self, artifact):
        """The "different day" wording was not merely imprecise for the `past = 128` row — it was
        false. Every record on both sides of that row's comparison, and every A/A record, carries
        the same calendar date."""
        treatment, calibration = _by_role(artifact)
        dates = {_dt(r["started_at"]).date()
                 for r in calibration + [t for t in treatment if t.get("workload") == TREATMENT_WORKLOAD]}
        assert len(dates) == 1, sorted(map(str, dates))

    def test_perf_states_how_many_treatment_records_predate_the_calibration_date(self, artifact):
        """Six records did start on the previous calendar day, and dropping that on the floor
        would be the same sin in the other direction."""
        treatment, calibration = _by_role(artifact)
        aa_date = {_dt(r["started_at"]).date() for r in calibration}.pop()
        earlier = [r for r in treatment if _dt(r["started_at"]).date() < aa_date]
        assert f"{len(earlier)} of the {len(treatment)}" in _perf_section()


class TestThePidRecyclingEvidenceNamesTheRealProcess:
    """The recycling argument stands. The PID the prose named did not."""

    def _shared(self, artifact) -> set:
        treatment, calibration = _by_role(artifact)
        return ({r["pid"] for r in dwe.accepted(treatment)}
                & {r["pid"] for r in dwe.accepted(calibration)})

    def test_the_artifact_really_does_recycle_a_pid(self, artifact):
        assert self._shared(artifact), "no recycled PID: the whole argument would be hypothetical"

    def test_the_recycled_pid_this_module_names_is_the_one_in_the_artifact(self, artifact):
        """Named by this exact test name in `_process_key`'s docstring. Every PID that docstring
        cites has to be a PID the records actually share."""
        cited = {int(p) for p in re.findall(r"PID \*\*(\d+)\*\*", dwe._process_key.__doc__)}
        assert cited, "the docstring names no PID at all"
        assert cited <= self._shared(artifact), (cited, self._shared(artifact))

    def test_every_pid_the_document_calls_recycled_is_recycled(self, artifact):
        cited = {int(p) for p in re.findall(r"PID\*{0,2}\s+\*{0,2}(\d{3,6})", _perf_section())}
        assert cited, "PERF §27 names no recycled PID"
        assert cited <= self._shared(artifact), (cited, self._shared(artifact))


class TestThePublishedTablesAgreeWithTheArtifactCellByCell:
    """The tables are where a wrong number hides best: every figure in them also appears in the
    prose, so a presence check on the prose passes while the table says something else. These
    parse the markdown and check each cell against the records it claims to report."""

    @staticmethod
    def _table(section: str, header_fragment: str) -> list:
        """The rows of the first markdown table whose header contains `header_fragment`."""
        rows, in_table = [], False
        for line in section.splitlines():
            stripped = line.strip()
            if not in_table:
                if stripped.startswith("|") and header_fragment in stripped:
                    in_table = True
                continue
            if not stripped.startswith("|"):
                break
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append(cells)
        assert rows, f"no table found with header fragment {header_fragment!r}"
        return rows

    @staticmethod
    def _num(cell: str) -> float:
        return float(re.sub(r"[^\d.\-]", "", cell))

    def test_the_protocol_table_reports_the_call_counts_the_records_carry(self, artifact):
        """Blocker 3's disclosure. If this table ever says the two arms issued the same number of
        inference calls, the retraction it exists to carry has been undone."""
        treatment, calibration = _by_role(artifact)
        actual = {}
        for label, records in (("treatment", treatment), ("A/A calibration", calibration)):
            ended = _ended(dwe.accepted(records))
            actual[label] = (
                len(ended),
                {r["speed"]["n"] for r in ended},
                {r["inference_calls"] for r in ended},
            )
        assert actual["treatment"][2] != actual["A/A calibration"][2], (
            "the arms agree on call count, so this whole disclosure would be wrong"
        )

        rows = self._table(_perf_section(), "inference calls")
        seen = set()
        for cells in rows:
            label = next(k for k in actual if cells[0].startswith(k))
            seen.add(label)
            count, timed, calls = actual[label]
            assert f"({count} accepted)" in cells[0], f"{label}: record count drifted from {count}"
            assert self._num(cells[1]) == float(next(iter(timed))), f"{label}: timed iterations"
            assert self._num(cells[2]) == float(next(iter(calls))), (
                f"{label}: the table says {cells[2]} inference calls, the records say {calls}"
            )
            assert self._num(cells[3]) == self._num(cells[2]) - self._num(cells[1]), (
                f"{label}: untimed calls do not equal inference calls minus timed iterations"
            )
        assert seen == set(actual), f"the protocol table omits an arm: {set(actual) - seen}"

    def test_the_verdict_table_reports_the_summary_the_artifact_published(self, artifact):
        """Blockers 1 and the INDETERMINATE requirement, cell by cell: ratios, geometric mean,
        interval, verdict and power all re-read from the summary. A reciprocated geo-mean or a
        verdict promoted past the one the summarizer reached fails here even if the surrounding
        prose still quotes the right numbers somewhere."""
        summary = artifact["summary"]
        by_workload = {row["workload"]: row for row in summary["rows"]}
        rows = self._table(_perf_section(), "geo-mean")
        assert len(rows) == len(by_workload), (len(rows), len(by_workload))

        for cells in rows:
            tail = cells[0].strip("`* ").lstrip("…")
            row = next(r for w, r in by_workload.items() if w.endswith(tail))
            printed = [self._num(x) for x in cells[1].split("/")]
            assert printed == [round(x, 4) for x in row["ratios"]], (row["workload"], printed)
            assert self._num(cells[2]) == round(row["geometric_mean"], 4), (
                f"{row['workload']}: table geo-mean {cells[2]} is not the published "
                f"{row['geometric_mean']:.4f} — note that its reciprocal is "
                f"{1 / row['geometric_mean']:.4f}, which is what an inverted convention prints"
            )
            low, high = (self._num(x) for x in cells[3].strip("*[] ").split(","))
            assert (low, high) == (round(row["interval"]["low"], 3),
                                   round(row["interval"]["high"], 3)), row["workload"]

            verdict = cells[4].strip("* ")
            expected = row["verdict"]
            if verdict != expected:
                assert verdict == "INCONCLUSIVE" and expected == dwe.INDETERMINATE, (
                    f"{row['workload']}: the table says {verdict}, the artifact says {expected}"
                )
            assert verdict != dwe.FASTER, f"{row['workload']}: the table claims a speedup"

            assert "≤" in cells[5], f"{row['workload']}: power stated without its inequality"
            power = dwe.power_at(0.859, row["interval"]["log_sd"], summary["band"]["band"], 3)
            printed_power = self._num(cells[5])
            assert printed_power >= power - 5e-5, (
                f"{row['workload']}: the table claims power ≤ {printed_power}, but power_at the "
                f"published band is {power:.6f}. A ceiling rounded down is a claim the artifact "
                f"does not support: round these up."
            )
            assert printed_power - power < 0.01 + 5e-5, (
                f"{row['workload']}: the table's ≤ {printed_power} is looser than the "
                f"{power:.6f} the artifact supports by more than one printed digit"
            )


class TestThePowerFigureIsPublishedAsAnUpperBound:
    """A power figure computed against a band that is not known to be wide enough is a ceiling."""

    def test_the_power_figure_is_an_upper_bound(self, artifact):
        """The property the qualifier rests on: `power_at` decreases as the band widens. So a band
        that may be too narrow yields a power that may be too high, and never one too low."""
        row = _row(artifact["summary"], 128)
        sd = row["interval"]["log_sd"]
        band = artifact["summary"]["band"]["band"]
        wider = [dwe.power_at(0.859, sd, band * k, 3) for k in (1.0, 1.25, 1.5, 2.0)]
        assert wider == sorted(wider, reverse=True), wider
        assert dwe.power_at(0.859, sd, band * 0.5, 3) > wider[0]

    def test_the_documents_qualify_the_power_figure_wherever_they_state_it(self, artifact):
        """Every occurrence of the figure in prose has to carry its qualifier within the same
        breath — a bare `0.66` reads as a point estimate."""
        power = _row(artifact["summary"], 128)["power_at_reference_effect"]["power"]
        printed = f"{power:.2f}"
        for name, section in _sections():
            hits = [m.start() for m in re.finditer(re.escape(printed), section)]
            assert hits, f"{name} does not state the power figure at all"
            for at in hits:
                window = section[max(0, at - 60):at]
                assert "≤" in window or "at most" in window, (
                    f"{name} states {printed} without its upper-bound qualifier"
                )
            assert "upper bound" in section, f"{name} never says which way the bound runs"

    def test_the_power_figure_is_pinned_to_the_band_it_was_computed_at(self, artifact):
        """A power number without its band is not a number. The section quotes two: 0.6621 at the
        selected 0.0748 band, and 0.346 at the 0.1106 CNN-derived floor a previous revision used.
        Both are re-derived here, so neither can drift from the band it is printed beside, and the
        older figure cannot be quietly restated as if it belonged to this run's band."""
        row = _row(artifact["summary"], 128)
        sd = row["interval"]["log_sd"]
        selected_band = artifact["summary"]["band"]["band"]
        assert round(selected_band, 4) == SELECTED_BAND, selected_band

        selected = dwe.power_at(0.859, sd, selected_band, 3)
        assert round(selected, 4) == SELECTED_POWER, selected
        assert selected == pytest.approx(
            row["power_at_reference_effect"]["power"], rel=1e-12
        ), "the artifact's own power field is not power_at of the artifact's own band"

        older = dwe.power_at(0.859, sd, OLDER_BAND_FLOOR, 3)
        assert round(older, 3) == OLDER_POWER, older
        assert older < selected, "the wider band must not be the more powerful one"

        section = _perf_section()
        blocks = re.split(r"\n\s*\n", section)
        for figure, band in ((f"{SELECTED_POWER:.4f}", f"{SELECTED_BAND:.4f}"),
                             (f"{OLDER_POWER:.3f}", f"{OLDER_BAND_FLOOR:.4f}")):
            stating = [b for b in blocks if figure in b]
            assert stating, f"§27 does not state the power figure {figure}"
            for block in stating:
                assert band in block, (
                    f"§27 states {figure} in a paragraph or table that never names the band "
                    f"{band} it was computed at"
                )


class TestTheProseNamesTheShippedApi:
    """`classify` was renamed `classify_record`; a document naming the old one describes code that
    does not exist, and the rename was made precisely so the name could not be borrowed."""

    def test_the_summarizer_exposes_classify_record_and_not_classify(self):
        assert callable(dwe.classify_record)
        assert not hasattr(dwe, "classify")

    def test_no_document_names_the_function_that_no_longer_exists(self):
        for name, section in _sections():
            assert "classify()" not in section, f"{name} names the pre-rename API"
            assert not re.search(r"`classify`", section), f"{name} names the pre-rename API"
        assert "classify_record" in _perf_section()


class TestBothObservationsSurviveUnpooled:
    """Two runs disagree. Averaging them would manufacture a third number that nobody measured."""

    def test_the_two_observations_are_reported_separately(self):
        section = _perf_section()
        assert "0.859" in section and "0.9651" in section
        assert not re.search(r"\bpooled\b", section, re.I), "the two runs were pooled"
        for forbidden in ("combined estimate", "averaged across the two runs",
                          "weighted mean of the two"):
            assert forbidden.lower() not in section.lower(), forbidden

    def test_the_headline_verdict_is_still_the_one_that_declines(self, artifact):
        assert _row(artifact["summary"], 128)["verdict"] == dwe.INDETERMINATE
        assert artifact["summary"]["window"]["claim"] == dwe.INDETERMINATE
        assert "INCONCLUSIVE" in _perf_section()


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

    def test_protocol_delta_withholds_a_match_when_the_arms_disagree(self, artifact, mutable):
        why = withholds(dwe.protocol_delta(artifact["records"]), "matched",
                        because="the calibration issued 25 inference calls and the treatment 27")
        assert "protocol" in why
        for record in mutable["records"]:
            if record.get("role") == "aa":
                record["inference_calls"] = 27
        assert dwe.protocol_delta(mutable["records"])["matched"] is True

    def test_protocol_delta_withholds_a_verdict_when_one_side_is_empty(self, artifact):
        """Nothing to compare is not agreement. A comparison with one arm missing has to decline
        rather than report a match by vacuity."""
        treatment = [r for r in artifact["records"] if r.get("role") != "aa"]
        why = withholds(dwe.protocol_delta(treatment), "matched",
                        because="there is no calibration arm to compare the treatment against")
        assert "nothing to compare" in why

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
