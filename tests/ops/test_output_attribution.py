"""Per-output attribution — the fifth costume, and its falsifiers.

Morpheus's discharge condition (c) for criterion 10 is a non-triviality guard on both
sides: *"64 pairs of zeros satisfy (a) perfectly, which is ``0.0 == 0.0`` in a fourth
costume."*  The guard tests each side for constancy.

There is a fifth costume it cannot reach.  On 2026-08-02 the criterion-10 record on the
real Phi-3.5 artifact read::

    oracle_outputs_compared              = 65
    oracle_outputs_degenerate            = 0
    oracle_outputs_within_tolerance      = 65
    oracle_max_abs_diff_over_all_outputs = 0.0
    verdict                              = UNATTRIBUTED

Every oracle field perfect, agreement exact, nothing degenerate — because the EP claimed
zero of 363 nodes and ``vk_out`` *was* ``cpu_out``.  Not zeros.  Not constants.  **Two
sides from one computation.**  A degeneracy guard cannot see that: both sides are real,
varying, input-dependent values.  Only the attribution requirement stood between that
record and a claim that all 65 outputs matched.

Today it stands because attribution is zero.  The concern this module answers is what
happens as Mouse fills the proof ledger five forms at a time:

    The EP claims some nodes.  Attribution says yes.  ``MATCH`` becomes representable.
    But the nodes producing outputs 1..64 may still be declining, so those 64
    comparisons are CPU-against-CPU while the verdict says the EP ran.

Same vacuity, through an attribution check that has already said yes — and it does not
look like a defect.  It looks like partial acceleration with a clean oracle, which is
precisely what a filling ledger is supposed to look like.

**What the data supports.**  Probed on real hardware, both selectors, 2026-08-02
(``bench/results/per_output_attribution_probe.json``): ORT's trace names every
CPU-executed node with its *graph node name*, and delivers each fused island as a single
event named ``VulkanExecutionProvider_VulkanExecutionProvider_<hash>_0_0`` naming no
constituent.  So the derivation available is a **complement**, and it is sound in exactly
one direction:

  * ``CPU-ONLY`` — every node upstream of this output carries an explicit other-provider
    event.  Sound: an optimiser can delete an event, it cannot invent one.
  * ``EP-COVERED`` — some ancestor carries no other-provider event, so it was absorbed
    into a fused island *or eliminated before execution*.  Not sound, and therefore used
    only to **withhold** ``MATCH``, never to grant it.

Every test here is GPU-free, deterministic and contention-immune: synthesised topologies
and synthesised profile events.  No wall-clock assertion, no threshold, no subprocess.
The hardware arm lives in ``test_output_attribution_hw.py``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _verdict  # noqa: E402
from _verdict import (  # noqa: E402
    EP_NAME,
    OUTPUT_CPU_ONLY,
    OUTPUT_EP_COVERED,
    OUTPUT_UNOBSERVABLE,
    InstrumentError,
    OutputAttribution,
)

CPU = "CPUExecutionProvider"


# ---------------------------------------------------------------------------
# Synthesised material.  Two independent branches to two outputs — a shared
# ancestor would make every output downstream of every node and the question
# unanswerable by topology, which is itself worth saying out loud.
# ---------------------------------------------------------------------------

TWO_BRANCH_TOPOLOGY = {
    "outputs": ["out_a", "out_b"],
    "producer": {"t1": "add_a", "out_a": "mul_a", "t2": "sin_b", "out_b": "cos_b"},
    "node_inputs": {
        "add_a": ["x", "y"],
        "mul_a": ["t1", "y"],
        "sin_b": ["x"],
        "cos_b": ["t2"],
    },
}


def _profile(**node_to_provider: str) -> list[dict]:
    """A minimal ORT trace: one ``Node`` event per named node, with the suffix ORT adds."""
    return [
        {"cat": "Node", "name": f"{node}_kernel_time", "args": {"provider": provider}}
        for node, provider in node_to_provider.items()
    ]


def _coverage(**node_to_provider: str) -> OutputAttribution:
    return OutputAttribution.from_topology(
        topology=TWO_BRANCH_TOPOLOGY,
        node_providers=_verdict.node_providers(_profile(**node_to_provider)),
    )


# ---------------------------------------------------------------------------
# The reading itself
# ---------------------------------------------------------------------------


def test_an_output_whose_every_ancestor_ran_elsewhere_is_cpu_only() -> None:
    """The sound direction.  This is the label that refuses."""
    cov = _coverage(add_a=CPU, mul_a=CPU, sin_b=CPU, cos_b=CPU)
    assert cov.per_output == {"out_a": OUTPUT_CPU_ONLY, "out_b": OUTPUT_CPU_ONLY}
    assert cov.cpu_only_count == 2
    assert not cov.any_output_reaches_ep


def test_an_output_with_an_absorbed_ancestor_is_ep_covered() -> None:
    """A fused island names no constituent, so its nodes are absent from the trace."""
    cov = _coverage(sin_b=CPU, cos_b=CPU)  # add_a / mul_a absorbed: no events
    assert cov.per_output == {"out_a": OUTPUT_EP_COVERED, "out_b": OUTPUT_CPU_ONLY}
    assert cov.ep_covered_count == 1
    assert cov.partial, "one branch claimed and one declined is the state of interest"


def test_the_partial_state_is_named_and_not_folded_into_either_extreme() -> None:
    """``partial`` is the intermediate-ledger reading; it is neither all nor nothing."""
    assert _coverage(sin_b=CPU, cos_b=CPU).partial
    assert not _coverage(add_a=CPU, mul_a=CPU, sin_b=CPU, cos_b=CPU).partial  # all CPU
    assert not _coverage(sin_b=EP_NAME, cos_b=EP_NAME).partial  # all covered


def test_an_output_no_node_produces_is_unobservable_never_covered() -> None:
    """R12: a question that cannot be put in this frame gets its own token.

    Defaulting a pass-through output to ``EP-COVERED`` would be a pass by absence, which
    is the shape this whole module exists to make unrepresentable.
    """
    cov = OutputAttribution.from_topology(
        topology={
            "outputs": ["passthrough"],
            "producer": {},
            "node_inputs": {},
        },
        node_providers={"anything": CPU},
    )
    assert cov.per_output == {"passthrough": OUTPUT_UNOBSERVABLE}
    assert "no node in this graph produces this output" in cov.reason_for("passthrough")


def test_every_label_carries_a_reason_a_reader_can_act_on() -> None:
    cov = _coverage(sin_b=CPU, cos_b=CPU)
    assert "carries an explicit other-provider event" in cov.reason_for("out_b")
    assert "withholds MATCH, it does not grant it" in cov.reason_for("out_a")


# ---------------------------------------------------------------------------
# The second source, and why it is allowed to speak (2026-08-02, Phi-3.5)
# ---------------------------------------------------------------------------
# The trace complement alone labelled 65/65 Phi-3.5 outputs EP-COVERED at an own-count of
# ZERO — ORT's own graph optimisers delete node events wholesale, so "carries no
# other-provider event" is nearly uninformative on a real model.  Our claim log carries
# graph node names and lives inside the frame whose existence is in question, which
# disqualifies it from granting anything.  It is consulted only where it accuses us.


def test_the_claim_log_can_only_ever_withhold_never_grant() -> None:
    """Adding the second source moves outputs toward CPU-ONLY, never away from it."""
    trace_only = _coverage(sin_b=CPU, cos_b=CPU)
    assert trace_only.per_output["out_a"] == OUTPUT_EP_COVERED

    partial_claim = OutputAttribution.from_topology(
        topology=TWO_BRANCH_TOPOLOGY,
        node_providers=_verdict.node_providers(_profile(sin_b=CPU, cos_b=CPU)),
        claimed_nodes={"add_a"},  # mul_a absent from the log: we did not claim it
    )
    assert partial_claim.per_output["out_a"] == OUTPUT_EP_COVERED, (
        "one claimed ancestor is enough for the weaker label"
    )
    assert partial_claim.cpu_only_count >= trace_only.cpu_only_count, (
        "the second source may only add refusals"
    )


def test_an_output_whose_ancestors_we_declined_is_cpu_only_even_if_the_trace_forgot_them() -> None:
    """The Phi-3.5 shape: ORT ate the events, and we still know we did not claim them."""
    cov = OutputAttribution.from_topology(
        topology=TWO_BRANCH_TOPOLOGY,
        node_providers=_verdict.node_providers(_profile(sin_b=CPU)),  # cos_b event gone
        claimed_nodes={"add_a", "mul_a"},
    )
    assert cov.per_output == {"out_a": OUTPUT_EP_COVERED, "out_b": OUTPUT_CPU_ONLY}, (
        "without the claim log out_b would read EP-COVERED on a missing event alone"
    )
    assert cov.claim_log_join == 2


def test_a_claim_log_that_joins_nothing_is_an_instrument_error_not_a_red() -> None:
    """ORT renames nodes before GetCapability; a broken join would fail everything.

    R13, and the corollary that a result confirming a prediction deserves more scrutiny:
    "every output is CPU-ONLY" is exactly what this work went looking for, so the state
    in which the instrument produces it *by being broken* has to be a different token.
    """
    with pytest.raises(InstrumentError) as exc:
        OutputAttribution.from_topology(
            topology=TWO_BRANCH_TOPOLOGY,
            node_providers=_verdict.node_providers(_profile(sin_b=CPU, cos_b=CPU)),
            claimed_nodes={"some_name_ort_rewrote"},
        )
    assert "the join is broken" in str(exc.value)
    assert "manufactured red" in str(exc.value)


def test_the_absence_of_a_claim_log_is_recorded_not_read_as_claimed_nothing() -> None:
    cov = _coverage(sin_b=CPU, cos_b=CPU)
    assert cov.claim_log_join == -1
    assert "no claim log" in cov.claim_log_state
    assert cov.to_record()["claim_log_join"] == _verdict.OUTPUT_COVERAGE_NOT_COMPUTED


# ---------------------------------------------------------------------------
# The instrument's own falsifier (R9, R10)
# ---------------------------------------------------------------------------


def test_a_cpu_only_output_that_disagrees_refutes_this_instrument() -> None:
    """The two sides of a ``CPU-ONLY`` output are one computation; they must agree.

    If the oracle says one disagreed, the labelling is wrong — and that is a finding
    about *this* instrument, never about the EP.  Without this the coverage reading would
    be unfalsifiable, which is R10's own shape one level up.
    """
    cov = _coverage(sin_b=CPU, cos_b=CPU)
    assert cov.refuted_by(["out_b"]) == ["out_b"]
    assert cov.refuted_by(["out_a"]) == [], "an EP-covered disagreement is a real finding"
    assert cov.refuted_by([]) == []


def test_a_trace_that_named_nothing_is_an_instrument_error_not_full_coverage() -> None:
    """R13: the reading was never reached.  ``UNOBSERVABLE`` everywhere would read as one."""
    with pytest.raises(InstrumentError) as exc:
        OutputAttribution.from_topology(
            topology=TWO_BRANCH_TOPOLOGY, node_providers={}
        )
    assert "named zero nodes" in str(exc.value)
    assert "it is an outage (R13)" in str(exc.value)


def test_a_malformed_topology_is_an_instrument_error() -> None:
    with pytest.raises(InstrumentError) as exc:
        OutputAttribution.from_topology(
            topology={"outputs": ["a"]}, node_providers={"n": CPU}
        )
    assert "instrument outage" in str(exc.value)


def test_coverage_cannot_be_fabricated() -> None:
    """R10 amendment 1: a reading is what a mechanism computed, not what an author typed."""
    with pytest.raises(TypeError) as exc:
        OutputAttribution(
            per_output={"out_a": OUTPUT_EP_COVERED},
            reasons={},
            ep_name=EP_NAME,
            node_provider_count=1,
        )
    assert "is private" in str(exc.value)


def test_profile_suffixes_are_stripped_back_to_graph_node_names() -> None:
    assert _verdict.strip_profile_suffix("claimed_add_kernel_time") == "claimed_add"
    assert _verdict.strip_profile_suffix("n_fence_before") == "n"
    assert _verdict.strip_profile_suffix("plain") == "plain"


# ---------------------------------------------------------------------------
# Composition with the verdict — the whole point
# ---------------------------------------------------------------------------


def _attribution(own: int, *, other: int = 30) -> _verdict.ExecutionAttribution:
    """A parsed attribution, built the only way there is: from a trace."""
    events = [
        {"cat": "Node", "name": f"fused_{i}", "args": {"provider": EP_NAME}}
        for i in range(own)
    ] + [
        {"cat": "Node", "name": f"cpu_{i}", "args": {"provider": CPU}}
        for i in range(other)
    ]
    return _verdict.ExecutionAttribution(
        _token=_verdict._PARSED,
        executed_by=_verdict.tally_providers(events),
        node_events=len(events),
        source=_verdict.ATTRIBUTION_SOURCE_PROFILE,
        profile_path="<synthesised>",
        profile_digest="sha256:0000000000000000",
        profile_mtime_ns=0,
    )


def _verdict_for(att: _verdict.ExecutionAttribution) -> _verdict.EquivalenceVerdict:
    return _verdict.EquivalenceVerdict.from_comparison(
        comparison=_verdict.COMPARISON_AGREE,
        attribution=att,
        artifact="synthetic",
    )


def test_the_fifth_costume_is_refused_a_positive_own_count_no_output_reaches_it() -> None:
    """**The specimen.**  The EP ran; nothing it ran reaches a compared output.

    Before this module the verdict was ``MATCH``: agreement plus a positive own-count.
    Every comparison was our-CPU against ORT's-CPU and nothing in the record said so.
    """
    att = _attribution(own=1).with_output_coverage(
        _coverage(add_a=CPU, mul_a=CPU, sin_b=CPU, cos_b=CPU)
    )
    assert att.own_count == 1, "the session-scope reading is unchanged and still positive"
    assert not att.attributed, "the output-scope reading is what refuses"
    v = _verdict_for(att)
    assert v.verdict == _verdict.VERDICT_UNATTRIBUTED
    assert v.verdict != _verdict.VERDICT_DIVERGENT
    assert not v.permits_report


def test_that_refusal_explains_itself_as_the_new_cause_not_the_old_one() -> None:
    """``UNATTRIBUTED`` now has two causes and they route to different owners."""
    ran_nothing = _verdict_for(_attribution(own=0)).explain()
    assert "this EP did not run" in ran_nothing
    assert "Owner: whoever owns the run-time fallback (Switch)" in ran_nothing

    ran_but_unreached = _verdict_for(
        _attribution(own=1).with_output_coverage(
            _coverage(add_a=CPU, mul_a=CPU, sin_b=CPU, cos_b=CPU)
        )
    ).explain()
    assert "not one graph output is downstream of anything it executed" in ran_but_unreached
    assert "no degeneracy guard can see" in ran_but_unreached
    assert "Owner: whoever owns the claim predicates" in ran_but_unreached


def test_a_partially_claiming_session_can_still_match_and_the_record_scopes_it() -> None:
    """Refusing MATCH for every partial session would block the criterion forever.

    Phi-3.5 declines 8 nodes; a rule of "all outputs or nothing" never closes.  What the
    record must not do is quote 65 agreements as 65 pieces of evidence when 64 of them
    were vacuous — so the counts travel with the verdict.
    """
    att = _attribution(own=1).with_output_coverage(_coverage(sin_b=CPU, cos_b=CPU))
    v = _verdict_for(att)
    assert v.verdict == _verdict.VERDICT_MATCH
    rec = v.to_record()
    assert rec["outputs_reaching_this_ep"] == 1
    assert rec["outputs_cpu_only"] == 1
    assert rec["output_coverage"]["partial"] is True
    assert rec["output_coverage"]["per_output"]["out_b"] == OUTPUT_CPU_ONLY


def test_an_uncomputed_coverage_is_recorded_as_uncomputed_and_never_as_clearance() -> None:
    """R12 again: an instrument that did not run is not a green reading.

    ``reaches_compared_outputs`` is vacuously true without coverage — this property may
    only ever *withhold* MATCH on a positive reading, never grant it on a missing one —
    so the record has to say which of the two a reader is looking at.
    """
    att = _attribution(own=1)
    assert att.output_coverage is None
    assert att.coverage_state == _verdict.OUTPUT_COVERAGE_NOT_COMPUTED
    rec = _verdict_for(att).to_record()
    assert rec["output_coverage"] == _verdict.OUTPUT_COVERAGE_NOT_COMPUTED
    assert rec["outputs_cpu_only"] == _verdict.OUTPUT_COVERAGE_NOT_COMPUTED


def test_coverage_survives_the_counters_witness_copy_and_vice_versa() -> None:
    """The two ``with_*`` builders each preserve the other's reading.

    A copy-constructor that dropped the coverage would silently restore the old verdict,
    and nothing downstream would look different.
    """
    cov = _coverage(add_a=CPU, mul_a=CPU, sin_b=CPU, cos_b=CPU)
    a = _attribution(own=1).with_output_coverage(cov).with_counters_witness(7, reason="")
    assert a.output_coverage is not None and not a.attributed
    assert a.counters_dispatches == 7

    b = _attribution(own=1).with_counters_witness(7, reason="").with_output_coverage(cov)
    assert b.counters_dispatches == 7 and not b.attributed
    assert b.witness_agreement == _verdict.WITNESS_AGREEMENT_AGREE


def test_split_frame_still_outranks_the_new_refusal() -> None:
    """Precedence: two witnesses disagreeing about presence is the more urgent red."""
    att = (
        _attribution(own=1)
        .with_output_coverage(_coverage(add_a=CPU, mul_a=CPU, sin_b=CPU, cos_b=CPU))
        .with_counters_witness(0, reason="")
    )
    assert _verdict_for(att).verdict == _verdict.VERDICT_SPLIT_FRAME


def test_coverage_must_be_the_real_class_not_a_lookalike() -> None:
    class Faked:
        any_output_reaches_ep = True

    with pytest.raises(TypeError):
        _attribution(own=1).with_output_coverage(Faked())  # type: ignore[arg-type]


def test_the_vocabulary_gained_no_sixth_verdict_token() -> None:
    """Link's ``ci/check_verdict.py`` and Niobe's ``bench/admissible.py`` consume these.

    The fifth costume is a new *cause* of ``UNATTRIBUTED``, not a new token: a comparison
    that could not be attributed to this EP is exactly what ``UNATTRIBUTED`` already
    means.  Inventing a sixth would have made every consumer's exhaustive branch wrong.
    """
    assert _verdict.VERDICTS == (
        "MATCH",
        "DIVERGENT",
        "UNMEASURED",
        "UNATTRIBUTED",
        "SPLIT-FRAME",
    )
    assert _verdict.OUTPUT_COVERAGE_TOKENS == (
        OUTPUT_EP_COVERED,
        OUTPUT_CPU_ONLY,
        OUTPUT_UNOBSERVABLE,
    )
    assert set(_verdict.OUTPUT_COVERAGE_TOKENS).isdisjoint(_verdict.VERDICTS)
