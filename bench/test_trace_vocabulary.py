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

1. the phase set is identical in all three places;
2. sibling/nested classification is identical;
3. every emitted name is declared in the header table;
4. no emitted name is a prefix of another (the Python restatement of
   ``trace.rs::no_trace_name_is_a_prefix_of_another``);
5. the structural spans are not in anyone's phase list.

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


def test_no_emitted_name_is_a_prefix_of_another():
    """The Python restatement of `trace.rs::no_trace_name_is_a_prefix_of_another`.

    Two independent implementations of one invariant. The Rust one fails `cargo test`; this
    one fails the bench suite, and the matchers that would be fooled live on this side.
    """
    assert tv.prefix_collisions() == [], (
        f"a `startswith` matcher cannot tell these apart: {tv.prefix_collisions()}")


def test_the_record_path_instants_do_not_collide_with_the_compute_span():
    """The specific pair that motivated the invariant.

    `record_path()` emitted `vulkan.compute[REPLAY]`; the proposed whole-`Compute` span was
    `vulkan.compute`. `cuda_profile.compute_calls` matched subgraph spans with `startswith`,
    so the prefix matcher the anchor fix required would have captured both.
    """
    prefix = tv.record_path_instant_prefix()
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
        # The instants, under both the old and the new name.
        {"name": "vulkan.compute[REPLAY]", "cat": "ep.path", "ph": "i", "ts": 50,
         "args": {"path": "REPLAY"}},
        {"name": "vulkan.path[REPLAY]", "cat": "ep.path", "ph": "i", "ts": 60,
         "args": {"path": "REPLAY"}},
        # A same-prefix span that must not be mistaken for the anchor.
        {"name": cp.COMPUTE_CALL_SPAN + "_extra", "cat": "ep", "ph": "X", "ts": 0, "dur": 9_000},
    ]
    calls = cp.compute_calls(events)
    assert len(calls) == 1, "exactly one anchor span is present"
    assert calls[0]["dur"] == 1000, "the same-prefix decoy span was matched instead"
    assert cp.record_paths(events) == {"REPLAY": 2}, "instants are read by cat, not by name"


@pytest.mark.parametrize("phase", sorted(set(tv.phase_names())))
def test_each_rust_phase_is_classified_by_both_python_modules(phase):
    """Per-phase so the failure names the phase rather than a set difference."""
    assert phase in phases.HOST_PHASES, f"trace.rs emits `vulkan.{phase}`; phases.py drops it"
    assert phase in set(cp.SIBLING_PHASES) | set(cp.NESTED_PHASES), (
        f"trace.rs emits `vulkan.{phase}`; cuda_profile.py classifies it as neither sibling "
        f"nor nested, so every traced run will refuse")
