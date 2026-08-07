"""Equality guards between `rust/src/trace.rs` and the Python tables that mirror it.

Three modules carry a copy of the EP's span vocabulary — `bench/phases.py`,
`bench/cuda_profile.py` and `trace.rs`'s own module-header table. They drifted, and the drift
produced a committed profile that shipped ``verdict: "GPU_TIME_MEASURED"`` alongside
``refusals: ["trace.rs and cuda_profile.py disagree about the phase tree"]``:

* ``trace.rs`` gained ``Phase::BindCheck``;
* ``phases.py``'s ``HOST_PHASES`` did not, so ``phase_spans()`` silently dropped all fourteen
  of its spans — the module of record for ``docs/PERF.md``'s phase table could not see the
  phase at all;
* ``cuda_profile.py``'s ``SIBLING_PHASES`` did not either, so its cross-check fired a refusal
  against the very change that introduced it, on every traced run;
* and it could not simply be *added* to either, because it was the first sibling phase
  structurally outside ``vulkan.subgraph`` and the containment contract does not admit one.

Nothing in the tree failed. 81/81 bench tests and 19/19 `trace::` Rust tests passed with the
disagreement sitting inside a committed artifact. These tests are the thing that should have
failed.

The guards, one per direction of drift:

1. the phase set is identical in all three places, checked **bidirectionally** — every ordered
   pair, with the direction named in the failure, because a phase missing downstream is silently
   dropped while a phase missing upstream is a row that can never appear;
2. the **tier** of every phase is identical in all three places (a tier is a parent, so a
   disagreement means a phase checked against a span that does not bound it);
3. no phase lives outside the tier-0 bucketing anchor, and session scope is admissible only for
   the two phases ORT invokes from ``Compile()``;
4. sibling/nested classification is identical, and the tier tables partition the phase set;
5. every emitted name is declared in the header table;
6. no emitted name is a prefix of another **within its ``cat``** (the Python restatement of
   ``trace.rs::no_trace_name_is_a_prefix_of_another_within_its_category``), with the single
   cross-category exemption held to exactly one pair and proved non-vacuous;
7. the structural spans are not in anyone's phase list.

They are tests rather than an import-time derivation on purpose. `phases.py` and
`cuda_profile.py` must keep reducing traces written by a *different* checkout; a reduction that
read the current `trace.rs` could not read last month's artifact. The declarations stay where
they are and the disagreement is loud here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cuda_profile as cp  # noqa: E402
import phases  # noqa: E402
import trace_vocabulary as tv  # noqa: E402


def test_the_rust_source_is_actually_parsed():
    """A guard that cannot read its source is not a guard.

    Every assertion below compares a Python table against this parse. If the parse silently
    returned an empty tuple, all of them would compare a table against nothing and pass — which
    is the exact failure mode `counters_abi.py` raises rather than returns for.
    """
    assert tv.TRACE_RS.is_file(), f"{tv.TRACE_RS} not found"
    names = tv.phase_names()
    assert len(names) >= 8, f"implausibly few phases parsed: {names}"
    assert "record" in names and "fence_wait" in names


def test_host_phases_equals_the_rust_phase_enum():
    """`phases.HOST_PHASES` == `Phase::as_str`'s arms.

    Set equality, not sequence: `HOST_PHASES` is in reporting order and `Phase::ALL` is in
    declaration order, and neither ordering is load-bearing. Membership is.
    """
    assert set(phases.HOST_PHASES) == set(tv.phase_names()), (
        f"phases.py and trace.rs disagree about which phases exist.\n"
        f"  only in phases.py : {sorted(set(phases.HOST_PHASES) - set(tv.phase_names()))}\n"
        f"  only in trace.rs  : {sorted(set(tv.phase_names()) - set(phases.HOST_PHASES))}\n"
        f"A phase in trace.rs and not here is dropped by phase_spans(); a phase here and not "
        f"in trace.rs is a row that can never appear.")
    assert len(phases.HOST_PHASES) == len(set(phases.HOST_PHASES))


def test_sub_record_phases_equals_the_rust_nesting():
    assert set(phases.SUB_RECORD_PHASES) | {"upload", "readback"} == set(tv.nested_phases()), (
        f"phases.py's nesting table and Phase::nested_in disagree: "
        f"SUB_RECORD_PHASES={phases.SUB_RECORD_PHASES}, trace.rs nested={tv.nested_phases()}")


def test_cuda_profile_sibling_and_nested_tables_equal_the_rust_source():
    """The table that produced the shipped refusal.

    `SIBLING_PHASES` omitted `bind_check`, so `phase_breakdown` classified every one of its
    spans as "this module says nested, the span says sibling" and emitted a refusal into a
    document whose verdict read GPU_TIME_MEASURED.
    """
    assert set(cp.SIBLING_PHASES) == set(tv.sibling_phases()), (
        f"cuda_profile.SIBLING_PHASES != trace.rs siblings.\n"
        f"  only in cuda_profile: {sorted(set(cp.SIBLING_PHASES) - set(tv.sibling_phases()))}\n"
        f"  only in trace.rs    : {sorted(set(tv.sibling_phases()) - set(cp.SIBLING_PHASES))}\n"
        f"Every traced run will emit a phase-tree refusal until these agree.")
    assert set(cp.NESTED_PHASES) == set(tv.nested_phases()), (
        f"cuda_profile.NESTED_PHASES != trace.rs nested phases: "
        f"{sorted(set(cp.NESTED_PHASES) ^ set(tv.nested_phases()))}")


#: The three vocabularies, as ``(name, phase set)``. Every unordered pair is compared in **both**
#: directions below.
_VOCABULARIES = (
    ("trace.rs", lambda: set(tv.phase_names())),
    ("bench/phases.py", lambda: set(phases.HOST_PHASES)),
    ("bench/cuda_profile.py", lambda: set(cp.PHASE_TIER)),
)


@pytest.mark.parametrize("left,right", [(a, b) for a in _VOCABULARIES for b in _VOCABULARIES
                                        if a[0] != b[0]],
                         ids=lambda v: v[0])
def test_the_phase_vocabulary_agrees_in_both_directions(left, right):
    """Bidirectional equality across every ordered pair of the three modules.

    Ordered pairs, not unordered, and the direction is named in the message. A single
    ``set(a) == set(b)`` assertion is *logically* bidirectional but its failure output is a
    symmetric difference, and the two directions mean different things operationally:

    * a phase in ``trace.rs`` and not in ``phases.py`` is **silently dropped** — the module of
      record for ``docs/PERF.md``'s phase table cannot see a phase that exists, which is what
      happened to ``bind_check``;
    * a phase in ``phases.py`` and not in ``trace.rs`` is a **row that can never appear** — a
      table entry describing an instrument that was removed.

    Both are defects. Reading a symmetric difference and working out which one you have is a
    step that gets skipped.
    """
    (lname, lget), (rname, rget) = left, right
    missing = sorted(lget() - rget())
    assert not missing, (
        f"DIRECTION {lname} -> {rname}: {missing} are declared in {lname} and absent from "
        f"{rname}. A phase present upstream and missing downstream is dropped by the "
        f"downstream reduction without a word; it does not refuse, it just reports a smaller "
        f"number. Add it to {rname} (with its tier) or remove it from {lname}.")


@pytest.mark.parametrize("phase", sorted(set(tv.phase_names())))
def test_the_tier_of_every_phase_agrees_across_all_three_modules(phase):
    """Morpheus's tier tree, asserted once per phase, in all three places.

    `trace.rs` derives the tier from `Phase::containment`; the two Python modules declare it.
    A tier is a *parent*, so a disagreement here is a phase checked against a span that does not
    bound it — which is precisely the refusal this harness emitted on every traced run when
    `bind_check` (tier 1) was checked against `vulkan.subgraph` (also tier 1, beside it).
    """
    rust = tv.tiers()[phase]
    assert phases.PHASE_TIER.get(phase, "absent") == rust, (
        f"phases.PHASE_TIER[{phase!r}] = {phases.PHASE_TIER.get(phase, 'absent')!r} but "
        f"trace.rs derives tier {rust!r} from Phase::containment")
    assert cp.PHASE_TIER.get(phase, "absent") == rust, (
        f"cuda_profile.PHASE_TIER[{phase!r}] = {cp.PHASE_TIER.get(phase, 'absent')!r} but "
        f"trace.rs derives tier {rust!r} from Phase::containment")


def test_no_phase_lives_outside_the_bucketing_anchor():
    """The ruling, as an assertion rather than as prose.

    Every phase is either inside the tier-0 anchor or explicitly session-scope, and session
    scope is admissible for exactly the two phases ORT invokes from ``Compile()``. The pair of
    claims is what makes this a rule: adding a phase to the session-scope set to silence a
    failure would have to be a lie about when ORT calls it, and the third assertion below is
    what catches that lie.
    """
    tiers = tv.tiers()
    session = set(tv.session_scope_phases())
    for phase, tier in tiers.items():
        assert tier is not None or phase in session, (
            f"{phase!r} has no tier and is not declared session-scope, so nothing says which "
            f"span must contain it")
    assert session == {"compile", "prepack"}, (
        f"session scope is admissible only for the phases ORT invokes outside Compute(); "
        f"{sorted(session - {'compile', 'prepack'})} claim it and run inside Compute")
    assert set(phases.SESSION_SCOPE_PHASES) == session
    assert set(cp.SESSION_PHASES) == session
    # And every call-scope phase names a parent span that exists.
    parents = tv.parent_spans()
    known = set(tv.structural_spans().values()) | {f"vulkan.{p}" for p in tv.phase_names()}
    for phase, tier in tiers.items():
        if tier is None:
            assert parents[phase] is None
            continue
        assert parents[phase] in known, (
            f"{phase!r} declares parent span {parents[phase]!r}, which nothing emits")


def test_the_tier_partition_is_exhaustive_and_disjoint():
    """Every phase is in exactly one tier, and the tier tables partition the phase set."""
    all_phases = set(tv.phase_names())
    buckets = [set(cp.SESSION_PHASES), set(cp.TIER1_PHASES), set(cp.TIER2_PHASES),
               set(cp.NESTED_PHASES)]
    for i, a in enumerate(buckets):
        for b in buckets[i + 1:]:
            assert not (a & b), f"{sorted(a & b)} is in two tiers at once"
    assert set().union(*buckets) == all_phases, (
        f"tier tables do not cover the phase set: "
        f"{sorted(all_phases ^ set().union(*buckets))}")
    assert set(cp.TIER1_PHASES) == {"bind_check"}, (
        "tier 1 is `bind_check` beside the `vulkan.subgraph` dispatch region")
    assert set(cp.TIER2_PHASES) == {"record", "submit", "fence_wait"}


def test_the_two_python_tables_partition_the_phase_set():
    """No phase may be in both lists, and none may be in neither."""
    sib, nest = set(cp.SIBLING_PHASES), set(cp.NESTED_PHASES)
    assert not (sib & nest), f"{sorted(sib & nest)} is classified both ways"
    assert sib | nest == set(phases.HOST_PHASES)


def test_structural_spans_are_not_phases_anywhere():
    """A bracket summed with the things it brackets counts the same microseconds twice.

    That is `phase_containment`'s ERROR arm, and it is why `vulkan.compute_call` is a
    structural `cat == "ep"` span and not a `Phase`.
    """
    structural = set(tv.structural_spans().values())
    assert structural == set(phases.STRUCTURAL_SPANS), (
        f"phases.STRUCTURAL_SPANS != trace.rs structural spans: "
        f"{sorted(structural ^ set(phases.STRUCTURAL_SPANS))}")
    assert phases.SUBGRAPH in structural
    for name in structural:
        assert name not in {f"vulkan.{p}" for p in phases.HOST_PHASES}
        assert name.removeprefix("vulkan.") not in cp.SIBLING_PHASES
        assert name.removeprefix("vulkan.") not in cp.NESTED_PHASES
    assert cp.COMPUTE_CALL_SPAN in structural
    assert cp.SUBGRAPH_SPAN in structural


def test_every_emitted_name_is_declared_in_the_header_table():
    """`phases.py` cites that table as "the artifact declares its own structure"."""
    undeclared = sorted(tv.all_emitted_names() - tv.declared_span_names())
    assert not undeclared, (
        f"{undeclared} are emitted by trace.rs and absent from its module-header span "
        f"vocabulary table, which bench/phases.py reads as the declaration of what the "
        f"artifact contains.")


def test_no_emitted_name_is_a_prefix_of_another_within_its_category():
    """The Python restatement of
    `trace.rs::no_trace_name_is_a_prefix_of_another_within_its_category`.

    Two independent implementations of one invariant. The Rust one fails `cargo test`; this
    one fails the bench suite, and the matchers that would be fooled live on this side.

    Scoped by ``cat``, because Morpheus's ruling orders the instant named
    ``vulkan.record_path`` and ``vulkan.record`` is a phase — global prefix-freedom is
    **unsatisfiable given the ordered name**. See :data:`trace_vocabulary.KNOWN_CROSS_CAT_PREFIXES`
    and D12; flagged for Morpheus.
    """
    assert tv.prefix_collisions() == [], (
        f"a `startswith` matcher cannot tell these apart: {tv.prefix_collisions()}")


def test_the_cross_category_prefix_exemption_is_narrow_and_not_vacuous():
    """The one place the prefix rule bends, held to exactly one pair.

    Written as its own test so the exemption cannot grow quietly: adding a second entry to
    ``KNOWN_CROSS_CAT_PREFIXES`` fails here, and the failure is the review conversation.

    It is also asserted **non-vacuous** — the pair it names must actually be a prefix pair. An
    exemption for a collision that no longer exists is a licence sitting in the codebase waiting
    for a name that revives it.
    """
    assert len(tv.KNOWN_CROSS_CAT_PREFIXES) == 1, (
        f"the cross-category prefix exemption must stay at one pair; it is now "
        f"{sorted(tv.KNOWN_CROSS_CAT_PREFIXES)}. Each entry is a name that a matcher can only "
        f"separate by reading `cat` first — prove every consumer does before adding one.")
    for short, long in tv.KNOWN_CROSS_CAT_PREFIXES:
        assert long.startswith(short) and long != short, (
            f"{long!r} does not start with {short!r}; this exemption grants nothing and is "
            f"a licence waiting for a future name to collide into")
    assert ("vulkan.record", "vulkan.record_path") in tv.KNOWN_CROSS_CAT_PREFIXES
    # The pair really is cross-category, which is the entire justification.
    cats = dict(tv.categorised_names())
    assert cats["vulkan.record"] == tv.CAT_PHASE
    assert cats["vulkan.record_path"] == tv.CAT_RECORD_PATH
    # A same-category collision is still fatal, exemption or not.
    assert tv.prefix_collisions({"vulkan.record", "vulkan.record_path"}) == [
        ("vulkan.record", "vulkan.record_path")], (
        "passing an explicit name set drops the category, and the strict rule must apply to it")


def test_the_record_path_instants_do_not_collide_with_a_structural_span():
    """The specific pair that motivated the invariant.

    `record_path()` emitted `vulkan.compute[REPLAY]`; the proposed whole-`Compute` span was
    `vulkan.compute`. `cuda_profile.compute_calls` matched subgraph spans with `startswith`,
    so the prefix matcher the anchor fix required would have captured both. The instant is now
    `vulkan.record_path`, which shares no prefix with either structural span.
    """
    prefix = tv.record_path_instant_prefix()
    assert prefix == "vulkan.record_path", (
        "Morpheus's ruling names this instant; a rename here changes the artifact vocabulary")
    for span in tv.structural_spans().values():
        assert not prefix.startswith(span) and not span.startswith(prefix), (
            f"{prefix!r} and {span!r} share a prefix")


def test_the_anchor_matcher_is_exact_and_rejects_the_instants():
    """Behavioural, not textual: feed the reduction both vocabularies and check it separates.

    A grep for `startswith` in this file would pass the day someone reintroduces one
    somewhere else. This asks the function.
    """
    events = [
        {"name": cp.COMPUTE_CALL_SPAN, "cat": "ep", "ph": "X", "ts": 0, "dur": 1000},
        {"name": cp.SUBGRAPH_SPAN, "cat": "ep", "ph": "X", "ts": 100, "dur": 500},
        # The instants, under the old name and the ruled one.
        {"name": "vulkan.compute[REPLAY]", "cat": "ep.path", "ph": "i", "ts": 50,
         "args": {"path": "REPLAY"}},
        {"name": "vulkan.record_path[REPLAY]", "cat": "ep.path", "ph": "i", "ts": 60,
         "args": {"path": "REPLAY"}},
        # A same-prefix span that must not be mistaken for the anchor.
        {"name": cp.COMPUTE_CALL_SPAN + "_extra", "cat": "ep", "ph": "X", "ts": 0, "dur": 9_000},
    ]
    calls = cp.compute_calls(events)
    assert len(calls) == 1, "exactly one anchor span is present"
    assert calls[0]["dur"] == 1000, "the same-prefix decoy span was matched instead"
    assert cp.record_paths(events) == {"REPLAY": 2}, "instants are read by cat, not by name"


def test_the_record_phase_and_the_record_path_instant_do_not_contaminate_each_other():
    """Behavioural proof that the cross-category prefix exemption is safe.

    ``vulkan.record`` (a tier-2 phase) is a literal prefix of ``vulkan.record_path`` (the
    record-path instant). The exemption rests entirely on the claim that no consumer sees both
    in one candidate set, because each selects on ``cat`` first. That claim is checked here by
    running the two consumers over a trace containing both, rather than by asserting it in prose.

    If a future matcher goes back to ``startswith`` over an unfiltered event list, the instant's
    duration-less event lands in the phase total and the phase's span lands in the record-path
    histogram — and this goes red before either number is quoted.
    """
    events = [
        {"name": "vulkan.record", "cat": "ep.phase", "ph": "X", "ts": 0, "dur": 5_000,
         "args": {"nested_in": "none", "tier": "2"}},
        {"name": "vulkan.record_path[REPLAY]", "cat": "ep.path", "ph": "i", "ts": 10,
         "args": {"path": "REPLAY"}},
        {"name": "vulkan.record_path[FIRST_RECORD]", "cat": "ep.path", "ph": "i", "ts": 20,
         "args": {"path": "FIRST_RECORD"}},
    ]
    breakdown = cp.phase_breakdown(events)
    assert breakdown["siblings_us"] == {"record": {"us": 5_000, "count": 1}}, (
        "the record-path instants were swept into the phase total")
    assert breakdown["phase_tree_disagreements"] == []
    assert cp.record_paths(events) == {"REPLAY": 1, "FIRST_RECORD": 1}, (
        "the `vulkan.record` phase span was swept into the record-path histogram")


@pytest.mark.parametrize("phase", sorted(set(tv.phase_names())))
def test_each_rust_phase_is_classified_by_both_python_modules(phase):
    """Per-phase so the failure names the phase rather than a set difference."""
    assert phase in phases.HOST_PHASES, f"trace.rs emits `vulkan.{phase}`; phases.py drops it"
    assert phase in set(cp.SIBLING_PHASES) | set(cp.NESTED_PHASES), (
        f"trace.rs emits `vulkan.{phase}`; cuda_profile.py classifies it as neither sibling "
        f"nor nested, so every traced run will refuse")
