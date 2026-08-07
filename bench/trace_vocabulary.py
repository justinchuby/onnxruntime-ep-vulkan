"""The **one** derivation of the trace vocabulary from `rust/src/trace.rs`.

Three Python modules carry a hand-written copy of the EP's phase list:

* ``bench/phases.py`` — ``HOST_PHASES`` / ``SUB_RECORD_PHASES``, the module of record for
  ``docs/PERF.md``'s phase table;
* ``bench/cuda_profile.py`` — the tier tables behind the gap attribution;
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
* ``containment()`` — the arms of ``Phase::containment()``, whose ``match`` is exhaustive with no
  ``_`` arm precisely so a new phase cannot default into a tier it does not belong to;
* ``tiers()`` / ``nested_phases()`` / ``sibling_phases()`` — **derived** from ``containment()``,
  exactly as the Rust side derives them, so the two cannot disagree about the tree;
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

#: The tier the bucketing anchor itself sits at. Phases are numbered below it.
ANCHOR_TIER = 0

#: Trace ``cat`` for each kind of name. A ``startswith`` matcher can only confuse two names that
#: share a category, because a consumer selects candidates by ``cat`` before it matches by name.
CAT_STRUCTURAL = "ep"
CAT_PHASE = "ep.phase"
CAT_RECORD_PATH = "ep.path"

_AS_STR_RE = re.compile(r"pub fn as_str\(self\) -> &'static str \{\s*match self \{(.*?)\n\s*\}",
                        re.S)
_ARM_RE = re.compile(r"Phase::(\w+)\s*=>\s*\"(\w+)\"")
_CONTAINMENT_ARM_RE = re.compile(
    r"((?:Phase::\w+\s*\|\s*)*Phase::\w+)\s*=>\s*\{?\s*"
    r"Containment::(\w+)(?:\(\s*Phase::(\w+)\s*\))?",
    re.S,
)
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


def _fn_body(src: str, signature: str) -> str:
    r"""The brace-matched body of the first `fn` whose signature contains ``signature``.

    Brace-matched rather than regex-terminated on purpose. `Phase::containment`'s last arm wraps
    its right-hand side in a block, and a ``(.*?)\n\s*\}`` pattern stops at that inner close —
    silently returning a *prefix* of the match, which parses as "those phases are undeclared"
    rather than as a parse failure. A short read that looks like an answer is the failure mode
    this whole module exists to prevent.
    """
    i = src.find(signature)
    if i < 0:
        raise TraceVocabularyUnreadable(f"could not find `{signature}` in trace.rs")
    open_brace = src.find("{", i)
    if open_brace < 0:
        raise TraceVocabularyUnreadable(f"`{signature}` has no body in trace.rs")
    depth = 0
    for j in range(open_brace, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[open_brace + 1:j]
    raise TraceVocabularyUnreadable(f"unbalanced braces after `{signature}` in trace.rs")


def _strip_comments(body: str) -> str:
    return "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("//"))


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


def _variant_to_tag(src: str) -> "dict[str, str]":
    body = _AS_STR_RE.search(src)
    if body is None:
        raise TraceVocabularyUnreadable("could not find `Phase::as_str`'s match block in trace.rs")
    return {variant: tag for variant, tag in _ARM_RE.findall(body.group(1))}


def containment(src: "str | None" = None) -> "dict[str, tuple[str, str | None]]":
    """``{tag: (kind, parent_tag_or_None)}`` from ``Phase::containment()``.

    ``kind`` is one of ``SessionScope``, ``Anchor``, ``DispatchRegion``, ``Phase``. This is the
    single declaration of the tree; the tier numbers below are derived from it, on both sides of
    the language boundary, so neither side can hold a tier its containment contradicts.
    """
    src = source() if src is None else src
    body = _strip_comments(_fn_body(src, "pub fn containment(self) -> Containment"))
    tag = _variant_to_tag(src)
    out: "dict[str, tuple[str, str | None]]" = {}
    for heads, kind, parent in _CONTAINMENT_ARM_RE.findall(body):
        parent_tag = tag.get(parent) if parent else None
        if parent and parent_tag is None:
            raise TraceVocabularyUnreadable(
                f"`Containment::Phase(Phase::{parent})` names a variant that `Phase::as_str` does "
                f"not declare")
        for variant in re.findall(r"Phase::(\w+)", heads):
            if variant in tag:
                out[tag[variant]] = (kind, parent_tag)
    missing = [p for p in phase_names(src) if p not in out]
    if missing:
        raise TraceVocabularyUnreadable(
            f"`Phase::containment` did not classify {missing}. The Rust match is exhaustive with "
            f"no `_` arm, so a phase missing here is a parse failure, not a flat tree — and "
            f"treating it as 'no parent' is how a nested phase gets summed with its own parent.")
    return out


def tiers(src: "str | None" = None) -> "dict[str, int | None]":
    """``{tag: tier}`` — 1, 2, 3, … under the anchor, or ``None`` for session scope.

    Derived from :func:`containment` by the same rule `Phase::tier` uses: the anchor's direct
    children are tier 1, the dispatch region's are tier 2, and a phase inside a phase is one
    deeper than its parent.
    """
    src = source() if src is None else src
    tree = containment(src)
    base = {"SessionScope": None, "Anchor": 1, "DispatchRegion": 2}
    out: "dict[str, int | None]" = {}

    def resolve(tag: str, seen: "frozenset[str]" = frozenset()) -> "int | None":
        if tag in out:
            return out[tag]
        if tag in seen:
            raise TraceVocabularyUnreadable(
                f"containment cycle through {tag!r}; a phase cannot contain its own ancestor")
        kind, parent = tree[tag]
        if kind in base:
            out[tag] = base[kind]
        elif kind == "Phase":
            assert parent is not None
            pt = resolve(parent, seen | {tag})
            out[tag] = None if pt is None else pt + 1
        else:
            raise TraceVocabularyUnreadable(
                f"unknown Containment variant {kind!r} for phase {tag!r}. Refusing to guess a "
                f"tier: a wrong tier is a wrong parent, and a wrong parent is a wrong total.")
        return out[tag]

    for tag in phase_names(src):
        resolve(tag)
    return out


def parent_spans(src: "str | None" = None) -> "dict[str, str | None]":
    """``{tag: the span name this phase must lie inside}``, ``None`` for session scope."""
    src = source() if src is None else src
    spans = structural_spans(src)
    out: "dict[str, str | None]" = {}
    for tag, (kind, parent) in containment(src).items():
        if kind == "SessionScope":
            out[tag] = None
        elif kind == "Anchor":
            out[tag] = spans["compute_call"]
        elif kind == "DispatchRegion":
            out[tag] = spans["subgraph"]
        else:
            out[tag] = f"vulkan.{parent}"
    return out


def session_scope_phases(src: "str | None" = None) -> "tuple[str, ...]":
    """Phases ORT invokes outside `Compute()`, so outside every anchor span.

    Morpheus's ruling is that no phase may live outside the bucketing anchor. These are the
    phases for which that is a fact about ORT's callback structure rather than a defect: they run
    in `Compile()`, before any `Compute` span exists. They are named here so a checker can assert
    something **positive** about them — that they lie outside every anchor — instead of skipping
    them, and so that a phase which runs inside `Compute` and claims session scope still fails.
    """
    src = source() if src is None else src
    t = tiers(src)
    return tuple(p for p in phase_names(src) if t[p] is None)


def phases_in_tier(tier: int, src: "str | None" = None) -> "tuple[str, ...]":
    """Phase tags at ``tier``, in declaration order. Only these may be summed together."""
    src = source() if src is None else src
    t = tiers(src)
    return tuple(p for p in phase_names(src) if t[p] == tier)


def nested_phases(src: "str | None" = None) -> "tuple[str, ...]":
    """Phase tags contained in another **phase** — the ones that must never enter its total."""
    src = source() if src is None else src
    tree = containment(src)
    return tuple(p for p in phase_names(src) if tree[p][0] == "Phase")


def sibling_phases(src: "str | None" = None) -> "tuple[str, ...]":
    """Phase tags no other phase contains: their wall times are disjoint.

    Disjoint is not the same as summable-against-one-parent. `bind_check` (tier 1) and `record`
    (tier 2) are both here and answer to different spans; use :func:`phases_in_tier` to decompose
    a span.
    """
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

    ``compute_call`` is tier 0, the bucketing anchor: it brackets the whole ORT ``Compute``
    callback and every phase is checked against it. ``subgraph`` is tier 1, the dispatch region,
    which opens inside ``dispatch_ort``. Neither may appear in ``HOST_PHASES`` or in any phase
    total: adding a bracket to the sum of the things it brackets is the `phase_containment` ERROR
    arm.
    """
    src = source() if src is None else src
    return {
        "compute_call": _const("SPAN_COMPUTE_CALL", src),
        "subgraph": _const("SPAN_SUBGRAPH", src),
    }


def anchor_span(src: "str | None" = None) -> str:
    """The tier-0 bucketing anchor. Every call-scope phase must lie inside one of these."""
    return structural_spans(src)["compute_call"]


def record_path_instant_prefix(src: "str | None" = None) -> str:
    """Name prefix of the ``cat == "ep.path"`` instants."""
    return _const("INSTANT_RECORD_PATH", source() if src is None else src)


def declared_span_names(src: "str | None" = None) -> "frozenset[str]":
    """Every ``vulkan.*`` name written into the module-header vocabulary table."""
    return frozenset(re.findall(r"`(vulkan\.[A-Za-z0-9_.]*)", header(src)))


def categorised_names(src: "str | None" = None) -> "tuple[tuple[str, str], ...]":
    """``(name, cat)`` for every name this checkout emits, as a matcher would see it.

    The ``cat`` matters: a consumer selects candidate events by ``cat`` and only then matches by
    name, so two names in different categories are never in the same candidate set and cannot
    shadow each other. :func:`prefix_collisions` uses that to scope the prefix rule to the hazard
    rather than to the whole vocabulary.
    """
    src = source() if src is None else src
    spans = structural_spans(src)
    out = [(spans["compute_call"], CAT_STRUCTURAL), (spans["subgraph"], CAT_STRUCTURAL),
           (record_path_instant_prefix(src), CAT_RECORD_PATH)]
    out.extend((f"vulkan.{p}", CAT_PHASE) for p in phase_names(src))
    return tuple(out)


def all_emitted_names(src: "str | None" = None) -> "frozenset[str]":
    """Every ``vulkan.*`` span/instant name this checkout's `trace.rs` can emit."""
    return frozenset(n for n, _cat in categorised_names(src))


#: Cross-category ``(prefix, longer)`` pairs that are permitted to exist.
#:
#: Morpheus's ruling orders the record-path instant named ``vulkan.record_path``. ``vulkan.record``
#: is a phase and is a literal prefix of it, so **global prefix-freedom is unsatisfiable given the
#: ordered name**. The rule is therefore enforced absolutely *within* a ``cat`` — which is where
#: the hazard is, because that is the set a matcher chooses from — and cross-category pairs must
#: be exactly these, so a new one cannot appear quietly.
#:
#: This list does not weaken the within-category rule; :func:`prefix_collisions` never consults it
#: for a same-``cat`` pair. Flagged for Morpheus as D12.
KNOWN_CROSS_CAT_PREFIXES: "frozenset[tuple[str, str]]" = frozenset({
    ("vulkan.record", "vulkan.record_path"),
})


def prefix_collisions(names: "set[str] | frozenset[str] | None" = None,
                      src: "str | None" = None) -> "list[tuple[str, str]]":
    """Pairs ``(short, long)`` where a ``startswith`` matcher cannot tell two names apart.

    The Python restatement of
    `trace.rs::no_trace_name_is_a_prefix_of_another_within_its_category`. Two independent
    implementations of one invariant is the cross-check, not duplication: the Rust one fails
    `cargo test`, this one fails the bench suite, and a reduction here matches on names the Rust
    side never sees this file.

    Passing an explicit ``names`` set drops the category information, so the rule is applied
    globally to it — that is the strict reading, and it is what a caller checking an ad-hoc list
    should get.
    """
    if names is not None:
        s = set(names)
        return sorted((a, b) for a in s for b in s if a != b and b.startswith(a))
    pairs = categorised_names(src)
    out: "list[tuple[str, str]]" = []
    for a, cat_a in pairs:
        for b, cat_b in pairs:
            if a == b or not b.startswith(a):
                continue
            if cat_a == cat_b or (a, b) not in KNOWN_CROSS_CAT_PREFIXES:
                out.append((a, b))
    return sorted(set(out))


def main() -> int:
    print(f"trace.rs           : {TRACE_RS}")
    print(f"anchor (tier 0)    : {anchor_span()}")
    print(f"phases             : {', '.join(phase_names())}")
    for tier in (1, 2, 3):
        members = phases_in_tier(tier)
        if members:
            print(f"  tier {tier}           : {', '.join(members)}")
    if session_scope_phases():
        print(f"  session scope    : {', '.join(session_scope_phases())}")
    print(f"  nested in a phase: {', '.join(nested_phases())}")
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
        print("\nPASS(every emitted name is declared and tiered, and no name prefixes another "
              "within its category)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
