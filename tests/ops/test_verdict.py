"""Falsification suite for the `model_output_equivalence` record — §10.0 third amendment.

R9: *evidence scales only with falsifying instruments, not agreeing ones.*  A test file
that constructs a correct verdict and observes that it is correct is agreement.  The
claims this file exists to falsify are stated as attempts:

  - **"MATCH is unrepresentable at a zero own-provider count."**  Every route to a MATCH
    token is tried against an attribution that says the EP ran nothing — the derivation,
    the literal, the fabricated attribution, the series — and each must fail to produce
    one.  A single success voids the amendment.
  - **"UNATTRIBUTED is not DIVERGENT."**  Both reds are produced from the same comparison
    outcome and must differ in token, in explanation and in what they permit.
  - **"The attribution comes from an instrument we do not own."**  The constructor is
    attacked with a literal, a dict, a hand-built object and a stale file.
  - **"An instrument error never counts as a detection."**  The three terminal states are
    produced in one run and must be three distinct exception types.

Everything here runs without a GPU, without the EP library and without a model: these are
properties of the mechanism, so they must be observable when the hardware is not.  They
are also immune to the 9.5x contention swing that makes timing-based lanes unreliable on
this machine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import _verdict as v
import _models as m

ARTIFACT = "phi35-onnxruntime-genai-2026-07-30/model.onnx#sha256:deadbeef"


# ---------------------------------------------------------------------------
# Profile builders — the only legitimate source of an attribution
# ---------------------------------------------------------------------------

def _node(provider: str, op: str = "SomeOp") -> dict:
    return {"cat": "Node", "name": op, "ph": "X", "ts": 0, "dur": 1,
            "args": {"provider": provider, "op_name": op}}


def _write(events: list[dict], path: Path) -> Path:
    path.write_text(json.dumps(events), encoding="utf-8")
    return path


def _attr_ran(tmp_path: Path, islands: int = 1, cpu_nodes: int = 10) -> v.ExecutionAttribution:
    """An attribution from a profile in which this EP executed *islands* islands."""
    events = [_node(v.EP_NAME, f"VulkanExecutionProvider_abc_{i}") for i in range(islands)]
    events += [_node("CPUExecutionProvider", f"Gather_{i}") for i in range(cpu_nodes)]
    return v.ExecutionAttribution.from_profile(_write(events, tmp_path / "ran.json"))


def _attr_fellback(tmp_path: Path) -> v.ExecutionAttribution:
    """The specimen: ORT fell back inside run(); only CPU node events exist."""
    events = [_node("CPUExecutionProvider", f"Op_{i}") for i in range(30)]
    return v.ExecutionAttribution.from_profile(_write(events, tmp_path / "fellback.json"))


# ===========================================================================
# Claim 1 — MATCH is unrepresentable at a zero own-provider count
# ===========================================================================

def test_agreeing_comparison_at_zero_own_count_is_unattributed_not_match(tmp_path: Path) -> None:
    """The specimen, reproduced: a correct comparison that agreed, with the EP absent.

    On 2026-07-30 this exact situation produced ``MATCH``.  Every gate in the lane passed.
    The comparison was wired, invoked, correctly named and arithmetically correct — and
    certified a run in which this EP contributed zero nodes.
    """
    verdict = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_AGREE,
        attribution=_attr_fellback(tmp_path),
        artifact=ARTIFACT,
        device_index="0",
    )
    assert verdict.verdict == v.VERDICT_UNATTRIBUTED
    assert verdict.verdict != v.VERDICT_MATCH
    assert not verdict.permits_report, "UNATTRIBUTED must void the triple and the ratio"
    assert verdict.comparison == v.COMPARISON_AGREE, (
        "the comparison outcome must survive in the record: the arithmetic was correct, "
        "it was just about a different world"
    )


def test_no_argument_spells_match(tmp_path: Path) -> None:
    """A caller may not pass a verdict.  ``MATCH`` is not an input to any constructor."""
    attribution = _attr_fellback(tmp_path)
    for token in v.VERDICTS:
        with pytest.raises(ValueError) as exc:
            v.EquivalenceVerdict.from_comparison(
                comparison=token, attribution=attribution, artifact=ARTIFACT
            )
        assert "verdict" in str(exc.value).lower()


@pytest.mark.parametrize(
    "fake",
    [
        pytest.param(None, id="none"),
        pytest.param({"VulkanExecutionProvider": 1}, id="the-hardcoded-executed_by"),
        pytest.param("VulkanExecutionProvider", id="a-string"),
        pytest.param(1, id="an-int"),
    ],
)
def test_fabricated_attribution_is_rejected(fake) -> None:
    """§10.0's named cheapest cheat: ``executed_by: {"VulkanExecutionProvider": 1}``.

    Clause 3 closes it structurally — the argument must be an attribution object, and the
    only way to obtain one is to parse a profile file.
    """
    with pytest.raises(TypeError) as exc:
        v.EquivalenceVerdict.from_comparison(
            comparison=v.COMPARISON_AGREE, attribution=fake, artifact=ARTIFACT
        )
    assert "ExecutionAttribution" in str(exc.value)


def test_attribution_constructor_is_private() -> None:
    """A hand-built attribution is refused: the value must be one a mechanism computed."""
    with pytest.raises(TypeError) as exc:
        v.ExecutionAttribution(
            executed_by={v.EP_NAME: 999},
            node_events=999,
            source="ort_profile",
            profile_path="fake",
            profile_digest="sha256:0",
            profile_mtime_ns=0,
        )
    assert "private" in str(exc.value)


def test_unmeasured_cannot_become_match(tmp_path: Path) -> None:
    """The no-attribution constructor produces UNMEASURED and has no route to MATCH."""
    verdict = v.EquivalenceVerdict.unmeasured(reason="no oracle run", artifact=ARTIFACT)
    assert verdict.verdict == v.VERDICT_UNMEASURED
    assert verdict.executed_by == {}
    assert verdict.to_record()["attribution_source"] == v.ATTRIBUTION_SOURCE_NONE
    assert not verdict.permits_report


def test_verdict_property_cannot_be_assigned(tmp_path: Path) -> None:
    """Even with a legitimate attribution, the token is derived and not settable."""
    verdict = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_DISAGREE,
        attribution=_attr_ran(tmp_path),
        artifact=ARTIFACT,
    )
    assert verdict.verdict == v.VERDICT_DIVERGENT
    with pytest.raises(AttributeError):
        verdict.verdict = v.VERDICT_MATCH  # type: ignore[misc]


def test_positive_polarity_match_is_reachable_when_earned(tmp_path: Path) -> None:
    """Paired control (R9).

    Without this, every test above would also pass against a constructor that returned
    ``UNATTRIBUTED`` unconditionally — a mechanism that can never say ``MATCH`` is not a
    stricter gate, it is a broken one.
    """
    verdict = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_AGREE,
        attribution=_attr_ran(tmp_path, islands=1, cpu_nodes=10),
        artifact=ARTIFACT,
        device_index="1",
        device_name="Intel(R) Iris(R) Xe Graphics",
    )
    assert verdict.verdict == v.VERDICT_MATCH
    assert verdict.permits_report
    record = verdict.to_record()
    assert record["executed_by"] == {v.EP_NAME: 1, "CPUExecutionProvider": 10}
    assert record["attribution_source"] == v.ATTRIBUTION_SOURCE_PROFILE
    assert record["attribution_witnesses"]["profile_digest"].startswith("sha256:")


# ===========================================================================
# Claim 2 — UNATTRIBUTED is not DIVERGENT
# ===========================================================================

def test_unattributed_and_divergent_are_different_reds(tmp_path: Path) -> None:
    """Same comparison outcome, different attribution, and the two reds must not merge.

    ``DIVERGENT`` says *our kernels compute the wrong answer*; ``UNATTRIBUTED`` says *our
    kernels did not run*.  Different owners, different fixes, different next questions.
    A lane that prints one red for both is a lane with R13's defect.
    """
    divergent = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_DISAGREE,
        attribution=_attr_ran(tmp_path),
        artifact=ARTIFACT,
    )
    unattributed = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_DISAGREE,
        attribution=_attr_fellback(tmp_path),
        artifact=ARTIFACT,
    )
    assert divergent.verdict == v.VERDICT_DIVERGENT
    assert unattributed.verdict == v.VERDICT_UNATTRIBUTED
    assert divergent.verdict != unattributed.verdict
    assert divergent.explain() != unattributed.explain()
    assert "did not run" in unattributed.explain()
    assert "wrong answer" in divergent.explain()
    # Both void the triple — being distinct is not being softer.
    assert not divergent.permits_report and not unattributed.permits_report


def test_a_disagreement_with_no_attribution_is_not_a_kernel_bug(tmp_path: Path) -> None:
    """A CPU-vs-CPU disagreement is a statement about the oracle, not about our kernels.

    It must not be recorded as ``DIVERGENT``, and the disagreement must not be lost: the
    comparison outcome stays on the record.
    """
    verdict = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_DISAGREE,
        attribution=_attr_fellback(tmp_path),
        artifact=ARTIFACT,
    )
    assert verdict.verdict == v.VERDICT_UNATTRIBUTED
    assert verdict.to_record()["comparison"] == v.COMPARISON_DISAGREE


# ===========================================================================
# Claim 3 — two witnesses, and disagreement is red (clause 2)
# ===========================================================================

def test_split_frame_when_witnesses_disagree(tmp_path: Path) -> None:
    """Profile says the EP ran; our own counter says zero dispatches.  Nothing is reportable."""
    attribution = _attr_ran(tmp_path).with_counters_witness(0)
    assert attribution.split_frame
    verdict = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_AGREE, attribution=attribution, artifact=ARTIFACT
    )
    assert verdict.verdict == v.VERDICT_SPLIT_FRAME
    assert not verdict.permits_report


def test_split_frame_in_the_other_direction(tmp_path: Path) -> None:
    """Our counter says we dispatched; ORT's profile says we executed nothing."""
    attribution = _attr_fellback(tmp_path).with_counters_witness(354)
    assert attribution.split_frame
    verdict = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_AGREE, attribution=attribution, artifact=ARTIFACT
    )
    assert verdict.verdict == v.VERDICT_SPLIT_FRAME


def test_witnesses_of_different_magnitude_agree(tmp_path: Path) -> None:
    """1 fused island and 354 dispatches is agreement, not a split frame.

    The witnesses count different things: presence is what must agree, not magnitude.
    Without this control the split-frame check would fire on every healthy run — and a
    check that fires on everything is the same as one that fires on nothing.
    """
    attribution = _attr_ran(tmp_path, islands=1).with_counters_witness(354)
    assert not attribution.split_frame
    assert v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_AGREE, attribution=attribution, artifact=ARTIFACT
    ).verdict == v.VERDICT_MATCH


def test_missing_second_witness_is_not_an_agreeing_one(tmp_path: Path) -> None:
    attribution = _attr_ran(tmp_path).with_counters_witness(None)
    assert not attribution.split_frame
    assert attribution.witnesses["counters_dispatches_executed"] is None


# ===========================================================================
# Claim 4 — R13: three terminal states, three exception types
# ===========================================================================

def test_missing_profile_is_an_instrument_error_not_a_detection(tmp_path: Path) -> None:
    with pytest.raises(v.InstrumentError) as exc:
        v.ExecutionAttribution.from_profile(tmp_path / "nope.json")
    assert "instrument failure" in str(exc.value)
    assert not isinstance(exc.value, AssertionError)


def test_corrupt_profile_is_an_instrument_error(tmp_path: Path) -> None:
    p = tmp_path / "corrupt.json"
    p.write_text("{ not json !!!", encoding="utf-8")
    with pytest.raises(v.InstrumentError):
        v.ExecutionAttribution.from_profile(p)
    assert not p.exists(), "the trace must be deleted even on a parse failure"


def test_non_list_profile_is_an_instrument_error(tmp_path: Path) -> None:
    p = tmp_path / "object.json"
    p.write_text('{"traceEvents": []}', encoding="utf-8")
    with pytest.raises(v.InstrumentError):
        v.ExecutionAttribution.from_profile(p)


def test_the_three_states_are_three_types(tmp_path: Path) -> None:
    """PASS, FAIL(condition) and ERROR(instrument) produced in one test, all distinguishable.

    On 2026-07-31 nobody could make this distinction unaided: ``NameError`` and a correct
    fallback detection both presented as "the gate went red".
    """
    # PASS
    assert _attr_ran(tmp_path).assert_executed() == 1
    # FAIL(condition) — an AssertionError that states what it observed
    with pytest.raises(AssertionError) as fail:
        _attr_fellback(tmp_path).assert_executed()
    assert "CPUExecutionProvider" in str(fail.value), (
        "R13 obligation 2: a guard must state what it observed even when it fails"
    )
    assert "fused-island" in str(fail.value)
    # ERROR(instrument)
    with pytest.raises(v.InstrumentError) as err:
        v.ExecutionAttribution.from_profile(tmp_path / "absent.json")
    assert not isinstance(err.value, AssertionError)
    assert isinstance(err.value, RuntimeError), (
        "InstrumentError must remain a RuntimeError so pre-existing except-clauses hold"
    )


def test_instrument_error_carries_no_observation(tmp_path: Path) -> None:
    """The distinguishing feature of an outage is the ABSENCE of an observation."""
    with pytest.raises(v.InstrumentError) as exc:
        v.ExecutionAttribution.from_profile(tmp_path / "absent.json")
    assert exc.value.observed is None


# ===========================================================================
# Claim 5 — the record travels with the number it qualifies
# ===========================================================================

_COUNTERS_STUB = """{
  "abi_version": 2,
  "dispatches_executed": 354,
  "claimed_nodes": 353,
  "model_output_equivalence": "UNMEASURED"
}
"""


def test_writer_emits_token_and_record(tmp_path: Path) -> None:
    counters = tmp_path / "counters.json"
    counters.write_text(_COUNTERS_STUB, encoding="utf-8")

    verdict = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_AGREE,
        attribution=_attr_ran(tmp_path).with_counters_witness(354),
        artifact=ARTIFACT,
        device_index="0",
        device_name="NVIDIA GeForce RTX 4060 Laptop GPU",
    )
    token = v.write_equivalence_record(counters, verdict)
    assert token == v.VERDICT_MATCH

    doc = json.loads(counters.read_text(encoding="utf-8"))
    assert doc["model_output_equivalence"] == "MATCH", (
        "the token must stay a STRING: rust/src/counters.rs::extract_equivalence and "
        "epctl --check-counters parse it as one"
    )
    assert doc["dispatches_executed"] == 354, "the writer must not disturb other counters"
    record = doc[v.EQUIVALENCE_RECORD_KEY]
    assert record["executed_by"][v.EP_NAME] == 1
    assert record["device_name"].startswith("NVIDIA")
    assert record["own_provider_count_means"].startswith("fused-island")


def test_writer_rewrites_rather_than_accumulates(tmp_path: Path) -> None:
    counters = tmp_path / "counters.json"
    counters.write_text(_COUNTERS_STUB, encoding="utf-8")
    first = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_AGREE, attribution=_attr_ran(tmp_path), artifact=ARTIFACT
    )
    second = v.EquivalenceVerdict.from_comparison(
        comparison=v.COMPARISON_AGREE, attribution=_attr_fellback(tmp_path), artifact=ARTIFACT
    )
    v.write_equivalence_record(counters, first)
    v.write_equivalence_record(counters, second)
    doc = json.loads(counters.read_text(encoding="utf-8"))
    assert doc["model_output_equivalence"] == v.VERDICT_UNATTRIBUTED
    assert doc[v.EQUIVALENCE_RECORD_KEY]["verdict"] == v.VERDICT_UNATTRIBUTED
    assert v.read_equivalence_record(counters)["executed_by"] == {"CPUExecutionProvider": 30}


def test_writer_refuses_a_literal(tmp_path: Path) -> None:
    counters = tmp_path / "counters.json"
    counters.write_text(_COUNTERS_STUB, encoding="utf-8")
    with pytest.raises(TypeError) as exc:
        v.write_equivalence_record(counters, "MATCH")  # type: ignore[arg-type]
    assert "may not pass a literal" in str(exc.value)
    assert json.loads(counters.read_text(encoding="utf-8"))["model_output_equivalence"] == "UNMEASURED"


def test_writer_on_missing_counters_is_an_instrument_error(tmp_path: Path) -> None:
    verdict = v.EquivalenceVerdict.unmeasured(reason="x")
    with pytest.raises(v.InstrumentError):
        v.write_equivalence_record(tmp_path / "absent.json", verdict)


def test_read_counters_dispatches_missing_file_is_none(tmp_path: Path) -> None:
    assert v.read_counters_dispatches(tmp_path / "absent.json") is None
    assert v.read_counters_dispatches(None) is None


# ===========================================================================
# Claim 6 — the criterion-10 series, and the number it counts
# ===========================================================================

def test_three_attributed_agreeing_runs_close_criterion_10(tmp_path: Path) -> None:
    series = v.AttributedRunSeries.from_runs(
        comparisons=[v.COMPARISON_AGREE] * 3,
        attribution=_attr_ran(tmp_path, islands=3, cpu_nodes=30),
        artifact=ARTIFACT,
        device_index="0",
    )
    assert series.verdict == v.VERDICT_MATCH
    assert series.own_count == 3
    assert series.islands_per_run == 1
    assert series.uniformly_attributed
    series.assert_closes_criterion_10()


def test_series_record_denies_the_graph_node_reading(tmp_path: Path) -> None:
    """R11 — the definition must travel with the number, or the next reader supplies one.

    ``3`` beside a 363-node model reads as a catastrophe unless the record says what it
    counts.  This is the trap Guard D already flagged, checked one level up.
    """
    series = v.AttributedRunSeries.from_runs(
        comparisons=[v.COMPARISON_AGREE] * 3,
        attribution=_attr_ran(tmp_path, islands=3),
        artifact=ARTIFACT,
    )
    text = series.describe()
    assert "not graph nodes" in text.lower() or "NOT graph nodes" in text
    # NOT a literal. This assertion used to pin "353" and would have gone red the moment
    # Mouse claimed two more ops — punishing the EP for doing more work, which is the
    # naming trap eating its own guard. The durable property is that the description
    # gives the reader a fusion ratio whose graph-node figure dwarfs the island count.
    ratio = re.search(r"(\d+) of (\d+) graph nodes", text)
    assert ratio is not None, (
        "the description must give the reader the real fusion ratio in the form "
        f"'<claimed> of <total> graph nodes', or the bare {series.own_count} is unreadable.\n"
        f"{text}"
    )
    claimed, total = int(ratio.group(1)), int(ratio.group(2))
    assert claimed <= total and claimed > 100 * series.own_count, (
        "the ratio must show the island count is orders of magnitude smaller than the node "
        f"count it stands for; got {claimed} of {total} beside {series.own_count} islands"
    )
    assert series.to_record()["series"]["counts_what"].startswith("fused-island")


def test_series_with_zero_attribution_is_unattributed_not_match(tmp_path: Path) -> None:
    """Three agreeing runs of a session in which the EP never executed: still not MATCH."""
    series = v.AttributedRunSeries.from_runs(
        comparisons=[v.COMPARISON_AGREE] * 3,
        attribution=_attr_fellback(tmp_path),
        artifact=ARTIFACT,
    )
    assert series.verdict == v.VERDICT_UNATTRIBUTED
    with pytest.raises(AssertionError) as exc:
        series.assert_closes_criterion_10()
    assert "UNATTRIBUTED" in str(exc.value)


def test_series_rejects_a_dropped_run(tmp_path: Path) -> None:
    """Two islands across three runs: one run fell back and the series must not close."""
    series = v.AttributedRunSeries.from_runs(
        comparisons=[v.COMPARISON_AGREE] * 3,
        attribution=_attr_ran(tmp_path, islands=2),
        artifact=ARTIFACT,
    )
    assert not series.uniformly_attributed
    with pytest.raises(AssertionError) as exc:
        series.assert_closes_criterion_10()
    assert "not uniform" in str(exc.value)


def test_series_rejects_too_few_runs(tmp_path: Path) -> None:
    series = v.AttributedRunSeries.from_runs(
        comparisons=[v.COMPARISON_AGREE] * 2,
        attribution=_attr_ran(tmp_path, islands=2),
        artifact=ARTIFACT,
    )
    with pytest.raises(AssertionError) as exc:
        series.assert_closes_criterion_10(required_runs=3)
    assert "needs 3" in str(exc.value)


def test_series_rejects_one_disagreeing_run(tmp_path: Path) -> None:
    series = v.AttributedRunSeries.from_runs(
        comparisons=[v.COMPARISON_AGREE, v.COMPARISON_DISAGREE, v.COMPARISON_AGREE],
        attribution=_attr_ran(tmp_path, islands=3),
        artifact=ARTIFACT,
    )
    assert series.verdict == v.VERDICT_DIVERGENT
    with pytest.raises(AssertionError):
        series.assert_closes_criterion_10()


def test_series_requires_a_real_attribution() -> None:
    with pytest.raises(TypeError):
        v.AttributedRunSeries.from_runs(
            comparisons=[v.COMPARISON_AGREE] * 3,
            attribution={"VulkanExecutionProvider": 3},  # type: ignore[arg-type]
            artifact=ARTIFACT,
        )


# ===========================================================================
# Claim 7 — the second witness with a different failure mode (R13 obligation 3)
# ===========================================================================

def test_fatal_log_scan_finds_the_line() -> None:
    captured = (
        "2026-07-31 21:00:00 [I] some benign line\n"
        "[E:onnxruntime:, sequential_executor.cc:516] Non-zero status code returned. "
        "EP_FAIL ... Falling back to CPUExecutionProvider.\n"
        "more output\n"
    )
    hits = v.find_fatal_log_lines(captured)
    assert len(hits) == 1
    assert "Falling back" in hits[0]


def test_fatal_log_scan_is_total() -> None:
    """A grep cannot NameError — that is the entire reason it is the second witness."""
    assert v.find_fatal_log_lines("") == []
    assert v.find_fatal_log_lines("nothing to see") == []
    assert v.find_fatal_log_lines("\x00\ufffd binary junk \n") == []


def test_attribution_describe_never_prints_a_bare_count(tmp_path: Path) -> None:
    text = _attr_ran(tmp_path, islands=1).describe()
    assert "NOT 1 graph node" in text
    zero = _attr_fellback(tmp_path).describe()
    assert "ran NOTHING" in zero and "CPUExecutionProvider" in zero


# ---------------------------------------------------------------------------
# write_unmeasured_verdict — the bail-out recorder
#
# This one is a WRITER, not a guard, and it swallows by design (R13: an
# instrument error in the recorder must not erase the finding already on its
# way up the stack). So it has no reject polarity in the census's sense and it
# is baselined with a hand note rather than pretended into SCREENED.
#
# What CAN be falsified is its content and its silence, and both are below.
# ---------------------------------------------------------------------------


def test_unmeasured_writer_records_unmeasured_and_cannot_record_match(
    tmp_path: Path,
) -> None:
    """Accept polarity: it writes, and what it writes is UNMEASURED with a reason."""
    counters = tmp_path / "counters.json"
    counters.write_text(json.dumps({"dispatches_executed": 0}), encoding="utf-8")

    m.write_unmeasured_verdict(counters, "Guard A: the EP is not in get_providers()")

    doc = json.loads(counters.read_text(encoding="utf-8"))
    assert doc[v.EQUIVALENCE_KEY] == v.VERDICT_UNMEASURED
    record = doc[v.EQUIVALENCE_RECORD_KEY]
    assert record["verdict"] == v.VERDICT_UNMEASURED
    assert "Guard A" in record["detail"], "the reason is the finding; it must survive"
    # There is no argument by which this function writes MATCH. Not a policy: a shape.
    assert v.VERDICT_MATCH not in json.dumps(doc)


def test_unmeasured_writer_swallows_a_broken_path_but_the_writer_under_it_does_not(
    tmp_path: Path,
) -> None:
    """The deliberate-swallow polarity, with the control that proves it IS a swallow.

    If ``write_unmeasured_verdict`` were simply a no-op, the first assertion would
    pass for the wrong reason. The second shows the writer underneath genuinely
    raises on the same input, so the silence upstairs is a caught exception and not
    an absent implementation.
    """
    unwritable = tmp_path / "no-such-dir" / "counters.json"

    m.write_unmeasured_verdict(unwritable, "bail-out")  # must not raise
    assert not unwritable.exists()

    with pytest.raises(Exception):
        v.write_equivalence_record(
            unwritable, v.EquivalenceVerdict.unmeasured(reason="bail-out")
        )


def test_unmeasured_writer_declines_a_missing_counters_path(tmp_path: Path) -> None:
    """No path, no artifact, no exception — the env var is simply not always set."""
    m.write_unmeasured_verdict(None, "bail-out")
    m.write_unmeasured_verdict("", "bail-out")
    assert list(tmp_path.iterdir()) == []

