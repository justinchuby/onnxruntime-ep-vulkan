"""The **one** derivation of the trace vocabulary from `rust/src/trace.rs`.

Three Python modules carry a hand-written copy of the EP's phase list:

* ``bench/phases.py`` — ``HOST_PHASES`` / ``SUB_RECORD_PHASES``, the module of record for
  ``docs/PERF.md``'s phase table;
* ``bench/cuda_profile.py`` — ``SIBLING_PHASES`` / ``NESTED_PHASES``, the gap attribution;
* this repository's docs, which quote both.

They drifted, and the drift was silent in one direction and self-contradictory in the other.
`Phase::BindCheck` was added to ``trace.rs`` and to neither Python table. ``phases.py`` dropped
all fourteen of its spans on the floor — the module of record could not see the phase at all —
while ``cuda_profile.py``'s cross-check fired a refusal on every traced run and the refusal was
shipped inside a committed profile whose verdict read ``GPU_TIME_MEASURED``.

That is the same defect class as `a52024f`/`898a2ba` in `VulkanEpCounters`: **a mirror that
co-exists with its source is not a check, it is a second source.** `rust/tools/counters_abi.py`
solved it by deriving the field list from `counters.rs` at import and letting the disagreement be
loud. This module does the same for the span vocabulary.

Unlike `counters_abi.py`, the tables in `phases.py` and `cuda_profile.py` are **not** replaced by
this parse. They are declarations, and they must keep working against a stored trace produced by
a different checkout — a reduction that imports the current `trace.rs` cannot read last month's
artifact. So this module is consumed by *tests* (`bench/test_trace_vocabulary.py`), which is where
a disagreement should be loud, and never on the reduction path.

What is parsed, and from where:

* ``phase_names()`` — the ``Phase::X => "tag"`` arms of ``Phase::as_str()``;
* ``nested_phases()`` / ``sibling_phases()`` — the arms of ``Phase::nested_in()``, whose ``match``
  is exhaustive with no ``_`` arm precisely so a new phase cannot default to "sibling";
* ``structural_spans()`` — ``SPAN_COMPUTE_CALL`` / ``SPAN_SUBGRAPH``, the ``cat == "ep"`` brackets
  that are **not** phases and are never summed;
* ``declared_span_names()`` — every ``vulkan.*`` name appearing in the module-header vocabulary
  table, which `phases.py` cites as "the artifact declares its own structure".

Every function raises :class:`TraceVocabularyUnreadable` rather than returning a plausible subset.
An empty phase list compares equal to nothing and would turn every guard below into a pass.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRACE_RS = REPO / "rust" / "src" / "trace.rs"

#: Where the module header stops. Everything above this is prose, including the vocabulary table.
_HEADER_END = "use std::cell::Cell;"

_AS_STR_RE = re.compile(r"pub fn as_str\(self\) -> &'static str \{\s*match self \{(.*?)\n\s*\}",
                        re.S)
_ARM_RE = re.compile(r"Phase::(\w+)\s*=>\s*\"(\w+)\"")
_NESTED_RE = re.compile(r"pub fn nested_in\(self\) -> Option<Phase> \{\s*match self \{(.*?)\n\s*\}",
                        re.S)
_CONST_RE = r'pub const {name}: &str = "([^"]+)";'


class TraceVocabularyUnreadable(RuntimeError):
    """`trace.rs` could not be parsed, so no vocabulary claim may be made from it.

    R13: an instrument error is not a detection. Returning ``()`` here would make every equality
    test in `bench/test_trace_vocabulary.py` compare a table against nothing and pass.
    """


def source(path: "str | Path | None" = None) -> str:
    p = Path(path) if path else TRACE_RS
    if not p.is_file():
        raise TraceVocabularyUnreadable(f"{p} does not exist")
    return p.read_text("utf-8", errors="replace")


def header(src: "str | None" = None) -> str:
    """The `//!` module header, which contains the span-vocabulary table."""
    src = source() if src is None else src
    end = src.find(_HEADER_END)
    if end < 0:
        raise TraceVocabularyUnreadable(
            f"could not find {_HEADER_END!r} in trace.rs, so the module header — which holds the "
            f"span vocabulary table — cannot be delimited")
    return src[:end]


def phase_names(src: "str | None" = None) -> "tuple[str, ...]":
    """Every phase tag, in declaration order, from ``Phase::as_str()``."""
    src = source() if src is None else src
    body = _AS_STR_RE.search(src)
    if body is None:
        raise TraceVocabularyUnreadable(
            "could not find `Phase::as_str`'s match block in trace.rs. Refusing to guess the "
            "phase vocabulary: a plausible wrong one is the defect this module exists to remove.")
    names = tuple(tag for _variant, tag in _ARM_RE.findall(body.group(1)))
    if len(names) < 5 or len(set(names)) != len(names):
        raise TraceVocabularyUnreadable(
            f"parsed {len(names)} phase tag(s) from Phase::as_str and they are not distinct: "
            f"{names}")
    return names


def nested_phases(src: "str | None" = None) -> "tuple[str, ...]":
    """Phase tags with a parent — the ones that must **never** enter a sibling total.

    Read from ``Phase::nested_in()``: an arm mapping to ``Some(Phase::Parent)`` is nested, an arm
    mapping to ``None`` is a sibling. The ``match`` has no ``_`` arm, so this is total.
    """
    src = source() if src is None else src
    body = _NESTED_RE.search(src)
    if body is None:
        raise TraceVocabularyUnreadable(
            "could not find `Phase::nested_in`'s match block in trace.rs")
    tag = dict(zip((v for v, _ in _ARM_RE.findall(_variant_block(src))),
                   phase_names(src)))
    out: list[str] = []
    for line in body.group(1).splitlines():
        if line.lstrip().startswith("//"):
            continue
        arrow = line.find("=>")
        if arrow < 0:
            continue
        head, rhs = line[:arrow], line[arrow + 2:]
        if "Some(Phase::" not in rhs:
            continue
        for variant in re.findall(r"Phase::(\w+)", head):
            if variant in tag:
                out.append(tag[variant])
    # Multi-line arms: the variants may sit on their own lines above the `=>`.
    if not out:
        raise TraceVocabularyUnreadable(
            "parsed no nested phases from Phase::nested_in; `upload` and `readback` are nested by "
            "construction, so zero means the parse is wrong, not that the tree is flat")
    return tuple(dict.fromkeys(out))


def _variant_block(src: str) -> str:
    body = _AS_STR_RE.search(src)
    if body is None:
        raise TraceVocabularyUnreadable("could not find `Phase::as_str`'s match block in trace.rs")
    return body.group(1)


def sibling_phases(src: "str | None" = None) -> "tuple[str, ...]":
    """Phase tags that may be summed: every phase that is not nested, in declaration order."""
    src = source() if src is None else src
    nested = set(nested_phases(src))
    return tuple(p for p in phase_names(src) if p not in nested)


def _const(name: str, src: "str | None" = None) -> str:
    src = source() if src is None else src
    m = re.search(_CONST_RE.format(name=name), src)
    if m is None:
        raise TraceVocabularyUnreadable(f"no `pub const {name}` in trace.rs")
    return m.group(1)


def structural_spans(src: "str | None" = None) -> "dict[str, str]":
    """The ``cat == "ep"`` brackets, which are **not** phases and are never summed.

    ``compute_call`` brackets the whole ORT ``Compute`` callback and contains ``subgraph``, which
    opens inside ``dispatch_ort``. Neither may appear in ``HOST_PHASES`` or in any sibling total:
    adding a bracket to the sum of the things it brackets is the `phase_containment` ERROR arm.
    """
    src = source() if src is None else src
    return {
        "compute_call": _const("SPAN_COMPUTE_CALL", src),
        "subgraph": _const("SPAN_SUBGRAPH", src),
    }


def record_path_instant_prefix(src: "str | None" = None) -> str:
    """Name prefix of the ``cat == "ep.path"`` instants."""
    return _const("INSTANT_RECORD_PATH", source() if src is None else src)


def declared_span_names(src: "str | None" = None) -> "frozenset[str]":
    """Every ``vulkan.*`` name written into the module-header vocabulary table."""
    return frozenset(re.findall(r"`(vulkan\.[A-Za-z0-9_.]*)", header(src)))


def all_emitted_names(src: "str | None" = None) -> "frozenset[str]":
    """Every ``vulkan.*`` span/instant name this checkout's `trace.rs` can emit."""
    src = source() if src is None else src
    out = set(structural_spans(src).values())
    out.add(record_path_instant_prefix(src))
    out.update(f"vulkan.{p}" for p in phase_names(src))
    return frozenset(out)


def prefix_collisions(names: "set[str] | frozenset[str] | None" = None,
                      src: "str | None" = None) -> "list[tuple[str, str]]":
    """Pairs ``(short, long)`` where a ``startswith`` matcher cannot tell two names apart.

    The Python restatement of `trace.rs::no_trace_name_is_a_prefix_of_another`. Two independent
    implementations of one invariant is the cross-check, not duplication: the Rust one fails
    `cargo test`, this one fails the bench suite, and a reduction here matches on names the Rust
    side never sees this file.
    """
    names = set(all_emitted_names(src) if names is None else names)
    return sorted((a, b) for a in names for b in names if a != b and b.startswith(a))


def main() -> int:
    print(f"trace.rs           : {TRACE_RS}")
    print(f"phases             : {', '.join(phase_names())}")
    print(f"  siblings         : {', '.join(sibling_phases())}")
    print(f"  nested           : {', '.join(nested_phases())}")
    print(f"structural spans   : {structural_spans()}")
    print(f"record-path instant: {record_path_instant_prefix()}[...]")
    undeclared = sorted(all_emitted_names() - declared_span_names())
    collisions = prefix_collisions()
    if undeclared:
        print(f"\nFAIL(undeclared): {', '.join(undeclared)} are emitted but absent from the "
              f"span vocabulary table in trace.rs's module header.")
    if collisions:
        print("\nFAIL(prefix collision):")
        for a, b in collisions:
            print(f"    {b!r} starts with {a!r}")
    if not undeclared and not collisions:
        print("\nPASS(every emitted name is declared, and no name prefixes another)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
