r"""Guards that `DESIGN.md`'s §5 spine is intact and every live `§5.5` citation has a target.

WHY THIS EXISTS
---------------
A previous revision of the issue-#73 work inserted a new `#### 5.4.2` ruling by replacing the
line that held `### 5.5 \`Compile\` — plan build and prepacking`. Nothing failed. The heading
simply disappeared, four citations elsewhere in the tree began pointing at nothing, and §5.5's
normative content — the five-step `Compile` contract — was silently absorbed into §5.4 with no
heading of its own.

That is the same defect class this repository keeps finding in its own numbers: **a reference
that decays without failing** (`DESIGN.md` §5.4.1(a)'s citation ruling). A section number is a
reference. Deleting its target does not error; it points at something else.

WHAT IT ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
Three things, each with a distinct falsifier:

1. The exact §5.5 heading text exists, and the first non-blank line after it is `Compile`'s step
   list. This is the specific deletion that happened, pinned in the specific form it happened in.
2. The §5 heading spine appears in ascending order with no duplicates. This catches a heading
   that survives but is moved, renumbered, or accidentally duplicated by a merge.
3. Every `§5.5` citation in the tree resolves to a section that exists **in the document the
   citation names**.

Point 3 is scoped tightly on purpose. This repository has a large number of pre-existing
citations to sections that do not exist, in both `DESIGN.md` and `OP_COVERAGE.md`; a guard that
swept all of them would be red on arrival and would be turned off within a day, which is worse
than not having it. `DESIGN.md` §5.4.1(a) already ruled the general form — *existing citations
are not swept; new ones follow the convention, and any citation found stale is converted rather
than repaired.* This guard covers the citations that the #73 change put at risk, and says so.

It also does not assert that a citation's *claim* matches its target's content. One live §5.5
citation (`rust/src/ops/common/claim.rs`, "compose-before-bespoke") points at a rule that lives
in `OP_COVERAGE.md` §5.6, not §5.5. That is a pre-existing mis-citation, it is out of scope for
issue #73, and asserting a title match here would make this guard fail for a reason it was not
written to detect.

Run::

    pytest tests/ops/test_design_sections_resolve.py -v --no-header
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
DESIGN = REPO / "docs" / "DESIGN.md"
OP_COVERAGE = REPO / "docs" / "OP_COVERAGE.md"

#: The heading whose deletion is the falsifier this file was written for, verbatim.
SECTION_5_5 = "### 5.5 `Compile` — plan build and prepacking"

#: The first line of §5.5's body. If the heading is deleted, this line survives and attaches to
#: whatever precedes it, which is exactly how the deletion stayed invisible.
SECTION_5_5_FIRST_LINE = "For each fused subgraph, in order:"

#: The §5 spine, in the order `DESIGN.md` must present it. `5.4.1(a)` is a section in its own
#: right and is listed as one.
DESIGN_5_SPINE = ("5.1", "5.2", "5.3", "5.4", "5.4.1", "5.4.1(a)", "5.4.2", "5.5", "5.6")

#: Every `§5.5` citation live in the tree, with the document each one refers to. Attribution is
#: explicit because "§5.5" alone is ambiguous — the two documents both have one and they are
#: different sections. A citation added without a row here fails `test_every_5_5_citation_is_attributed`.
CITATIONS_TO_5_5: dict[tuple[str, str], pathlib.Path] = {
    ("docs/DESIGN.md", "§5.5 step 2 makes it an invariant violation"): DESIGN,
    ("docs/OP_COVERAGE.md", "`DESIGN.md` §5.5/§6.3"): DESIGN,
    ("docs/OP_COVERAGE.md", "**`build.rs` consumes `src/ops/shader_variants.txt`** — §5.5"): OP_COVERAGE,
    ("rust/src/registry.rs", "(`DESIGN.md` §5.5 step 2)"): DESIGN,
    ("rust/src/ops/common/claim.rs", "`OP_COVERAGE.md` §5.5's compose-before-bespoke rule"): OP_COVERAGE,
}

#: Files to sweep for `§5.5` occurrences. Kept to the committed source and docs; build outputs and
#: the virtualenv are not part of the corpus.
SWEEP_GLOBS = ("docs/*.md", "rust/src/**/*.rs", "rust/tools/*.py", "tests/**/*.py", "ci/*.py")


def _lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _headings(path: pathlib.Path, prefix: str) -> list[tuple[int, str, str]]:
    """(line number, number, title) for every ATX heading whose number starts with `prefix`.

    The letter suffix is part of the number: `5.4.1(a)` is a distinct section from `5.4.1`, and
    folding them together would make the spine look duplicated.
    """
    out = []
    for i, line in enumerate(_lines(path), start=1):
        m = re.match(r"^#{2,6}\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)\s+(.*)$", line)
        if m:
            number = m.group(1)
            bare = number.split("(", 1)[0]
            if bare == prefix or bare.startswith(prefix + "."):
                out.append((i, number, m.group(2)))
    return out


def test_design_5_5_heading_exists_verbatim():
    """D1: the exact heading, present exactly once."""
    lines = _lines(DESIGN)
    hits = [i for i, line in enumerate(lines, start=1) if line.strip() == SECTION_5_5]
    assert hits, (
        f"{DESIGN.name} has no {SECTION_5_5!r}. This heading was silently deleted once already "
        f"by an edit that replaced it with a new subsection; §5.5 is normative (it is the "
        f"`Compile` contract) and is cited from three other places."
    )
    assert len(hits) == 1, f"{SECTION_5_5!r} appears {len(hits)} times at lines {hits}"


def test_design_5_5_heading_immediately_precedes_its_own_body():
    """The heading must sit on top of §5.5's content, not merely exist somewhere."""
    lines = _lines(DESIGN)
    idx = next(i for i, line in enumerate(lines) if line.strip() == SECTION_5_5)
    tail = [line.strip() for line in lines[idx + 1 :] if line.strip()]
    assert tail, "§5.5 is the last line of the file"
    assert tail[0] == SECTION_5_5_FIRST_LINE, (
        f"the first content under §5.5 is {tail[0]!r}, expected {SECTION_5_5_FIRST_LINE!r}. "
        f"If the step list has moved out from under its heading, §5.5 is orphaned even though "
        f"the heading still exists."
    )


def test_the_compile_step_list_is_not_orphaned():
    """The inverse direction: the body must not appear without the heading above it.

    Checked separately from the test above because the two failures mean different things — one
    says the heading moved, this one says the body did.
    """
    lines = _lines(DESIGN)
    hits = [i for i, line in enumerate(lines) if line.strip() == SECTION_5_5_FIRST_LINE]
    assert len(hits) == 1, f"{SECTION_5_5_FIRST_LINE!r} appears {len(hits)} times"
    prior = [line.strip() for line in lines[: hits[0]] if line.strip()]
    assert prior[-1] == SECTION_5_5, (
        f"the `Compile` step list is preceded by {prior[-1]!r}, not by its own heading. A "
        f"normative section with no heading cannot be cited and does not appear in a table of "
        f"contents."
    )


def test_design_section_5_spine_is_complete_and_ordered():
    """Every §5 heading appears exactly once, in ascending order, with none missing."""
    found = _headings(DESIGN, "5")
    numbers = [n for _, n, _ in found]
    for want in DESIGN_5_SPINE:
        assert numbers.count(want) == 1, (
            f"§{want} appears {numbers.count(want)} times in {DESIGN.name}; expected exactly once. "
            f"Found spine: {numbers}"
        )
    spine_positions = [numbers.index(want) for want in DESIGN_5_SPINE]
    assert spine_positions == sorted(spine_positions), (
        f"the §5 headings are out of order: {numbers}"
    )


def test_the_new_5_4_2_ruling_sits_between_5_4_1_and_5_5():
    """Issue #73's ruling is a subsection of §5.4 and must not have displaced §5.5."""
    numbers = [n for _, n, _ in _headings(DESIGN, "5")]
    assert numbers.index("5.4.1") < numbers.index("5.4.2") < numbers.index("5.5")


def _sweep_5_5_citations() -> list[tuple[str, int, str]]:
    out = []
    for pattern in SWEEP_GLOBS:
        for path in sorted(REPO.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(REPO).as_posix()
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                # `§5.5` but not `§5.5.1`, and not this file's own catalogue.
                if re.search(r"§5\.5(?!\d|\.\d)", line) and rel != "tests/ops/test_design_sections_resolve.py":
                    out.append((rel, i, line.strip()))
    return out


def test_every_5_5_citation_is_attributed_and_resolves():
    """Each live `§5.5` citation names a document, and that document has a §5.5."""
    for rel, lineno, line in _sweep_5_5_citations():
        match = [target for (src, frag), target in CITATIONS_TO_5_5.items() if src == rel and frag in line]
        assert match, (
            f"{rel}:{lineno} cites §5.5 but has no row in CITATIONS_TO_5_5:\n  {line}\n"
            f"Add one naming which document's §5.5 it means. `§5.5` is ambiguous — DESIGN.md and "
            f"OP_COVERAGE.md both have one and they are unrelated sections."
        )
        target = match[0]
        numbers = [n for _, n, _ in _headings(target, "5")]
        assert "5.5" in numbers, (
            f"{rel}:{lineno} cites §5.5 of {target.name}, which has no §5.5 heading. "
            f"Found: {numbers}"
        )


def test_the_catalogue_has_no_dead_rows():
    """Every row in CITATIONS_TO_5_5 corresponds to a citation that is actually in the tree.

    Without this, a citation could be deleted and its row would keep the catalogue looking
    complete — the catalogue would document a corpus that no longer exists.
    """
    swept = _sweep_5_5_citations()
    for src, frag in CITATIONS_TO_5_5:
        assert any(rel == src and frag in line for rel, _, line in swept), (
            f"CITATIONS_TO_5_5 has a row for {src} / {frag!r} but no such line is in the tree"
        )
