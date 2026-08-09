"""Every anchor-related number in the tree is re-derived here from a committed artifact.

WHY THIS EXISTS
---------------
Issue #73's fix moves a modelled anchor count (193 → 161) and withdraws an undERIVED one
(`225 anchors`). Both of those are exactly the kind of figure this project keeps having to
retract: a number that reads as measured, sits next to numbers that are, and has no artifact
behind it.

**No artifact in this repository carries an anchor count.** `PartitionStats` has no anchor
field. The `anchors` key in `bench/results/island_counterfactual_bert*.json` is a *list of op
names*, not a count. So every anchor count in the tree is a recomputation, and the only honest
form for one is a recomputation that runs.

This file is that recomputation. It reads `bench/results/_claim_log_phi35_r15_after.jsonl` —
the claim log of the run every Phi-3.5 figure in `DESIGN.md` and `OP_COVERAGE.md` is about —
counts what is actually in it, and asserts that the constants in `partition.rs` agree.

It also asserts the *absence* of the withdrawn claims, because a retraction that leaves the
original sentence in a second file is not a retraction.

WHAT BINDS TO WHAT
------------------
Every count below names its field and its collection explicitly:

  * `363` lines in the file                      → the offered-node census of that run
  * `355` of them with `claimed == true`         → `Island::…::nodes`
  * `161` claimed lines with `op ==
    "com.microsoft::MatMulNBits"`                → `Island::…::anchors` (MODEL)
  * `32` claimed lines with `op ==
    "GroupQueryAttention"`                       → the share withdrawn from `anchors`

Run::

    pytest tests/ops/test_anchor_claims_are_witnessed.py -v --no-header
"""

from __future__ import annotations

import collections
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
PARTITION_RS = REPO / "rust" / "src" / "ops" / "partition.rs"
DESIGN = REPO / "docs" / "DESIGN.md"
OP_COVERAGE = REPO / "docs" / "OP_COVERAGE.md"

#: The claim log the Phi-3.5 island constants are about. Named, not globbed: a different run's
#: log would give different counts and the test would still be green, which is the failure mode
#: this whole file exists to prevent.
CLAIM_LOG = REPO / "bench" / "results" / "_claim_log_phi35_r15_after.jsonl"

#: The op whose nodes carry a resident weight at a designated site on this graph, and therefore
#: the only op on it that can anchor.
ANCHORING_OP = "com.microsoft::MatMulNBits"


def _claim_log_rows() -> list[dict]:
    assert CLAIM_LOG.is_file(), (
        f"{CLAIM_LOG.relative_to(REPO).as_posix()} is missing. Every anchor count in the tree is "
        f"derived from it; without it they are unwitnessed and this test must not pass."
    )
    rows = []
    for line in CLAIM_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _island_anchor_constants() -> dict[str, int]:
    """`anchors:` literals from every `Island` const in `partition.rs`, keyed by const name."""
    text = PARTITION_RS.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(
        r"pub const (\w+): Island = Island \{(.*?)\};", text, flags=re.DOTALL
    ):
        name, body = m.group(1), m.group(2)
        a = re.search(r"anchors:\s*(\d+)", body)
        assert a, f"{name} has no `anchors:` field"
        out[name] = int(a.group(1))
    return out


def test_the_claim_log_census_is_what_the_constants_say_it_is():
    """363 offered, 355 claimed — the two counts `Island::…::nodes` is bound to."""
    rows = _claim_log_rows()
    assert len(rows) == 363, f"claim log has {len(rows)} lines, expected 363"
    claimed = [r for r in rows if r.get("claimed") is True]
    assert len(claimed) == 355, f"{len(claimed)} claimed lines, expected 355"

    nodes = set(re.findall(r"nodes:\s*(\d+)", PARTITION_RS.read_text(encoding="utf-8")))
    assert "355" in nodes, (
        "no `Island` constant carries `nodes: 355`, but the claim log says 355 nodes were claimed"
    )


def test_the_modelled_anchor_count_is_the_matmulnbits_count_and_nothing_else():
    """161, derived, and bound to the exact field and collection it comes from."""
    claimed = [r for r in _claim_log_rows() if r.get("claimed") is True]
    histogram = collections.Counter(r["op"] for r in claimed)

    matmulnbits = histogram[ANCHORING_OP]
    assert matmulnbits == 161, (
        f"{matmulnbits} claimed lines carry op == {ANCHORING_OP!r}, expected 161. Every anchor "
        f"count in the tree is this number; if it has moved, they all move with it."
    )

    for name, value in _island_anchor_constants().items():
        assert value == matmulnbits, (
            f"partition.rs::{name} declares anchors: {value}, but the claim log it cites yields "
            f"{matmulnbits}. The constant is a MODEL and must equal its own derivation."
        )


def test_the_withdrawn_gqa_share_is_the_difference_between_the_old_number_and_the_new():
    """193 − 161 = 32, and the 32 is a real population, not a rounding.

    Stated so that the retraction is checkable rather than asserted: the old constant was not
    wrong by an unknown amount, it was wrong by exactly the GroupQueryAttention nodes, which are
    still on the graph and still claimed — they simply do not anchor, because
    `GroupQueryAttention` designates no weight site (DESIGN.md §5.4.2).
    """
    claimed = [r for r in _claim_log_rows() if r.get("claimed") is True]
    histogram = collections.Counter(r["op"] for r in claimed)
    gqa = histogram["GroupQueryAttention"] + histogram["com.microsoft::GroupQueryAttention"]
    assert gqa == 32, f"{gqa} claimed GroupQueryAttention lines, expected 32"
    assert histogram[ANCHORING_OP] + gqa == 193, (
        "the retired name-only anchor count was 161 MatMulNBits + 32 GQA = 193; if that no longer "
        "reconstructs, the withdrawal note in partition.rs is describing a different graph"
    )


def test_the_claim_log_histogram_is_fully_accounted_for():
    """Nothing else on this graph is in a heavy family, so nothing else could have anchored.

    Without this, `anchors == 161` would be consistent with a graph containing some other
    weight-bearing heavy op that nobody counted.
    """
    claimed = [r for r in _claim_log_rows() if r.get("claimed") is True]
    histogram = collections.Counter(r["op"] for r in claimed)
    assert sum(histogram.values()) == 355

    text = PARTITION_RS.read_text(encoding="utf-8")
    body = text.split("pub const HEAVY_OP_FAMILIES: &[&str] = &[", 1)[1].split("];", 1)[0]
    families = set(re.findall(r'"([^"]+)"', body))
    bare = {f.rsplit("::", 1)[-1] for f in families}

    heavy_on_this_graph = {
        op: n for op, n in histogram.items() if op in families or op.rsplit("::", 1)[-1] in bare
    }
    assert heavy_on_this_graph == {ANCHORING_OP: 161, "com.microsoft::GroupQueryAttention": 32}, (
        f"the heavy-family population of this graph is {heavy_on_this_graph}, which is not the "
        f"161 MatMulNBits + 32 GQA the anchor model assumes"
    )


def test_the_withdrawn_claims_are_absent_from_the_tree():
    """A retraction that leaves the original sentence standing elsewhere is not a retraction."""
    stale = {
        "225 anchors": "an anchor count with no artifact behind it; PartitionStats has no such field",
        "anchors: 193": "the retired name-only anchor model, superseded by 161",
    }
    # A withdrawal note is allowed to quote the phrase it withdraws; a live claim is not. The
    # marker must appear within this many lines above the quotation, so that a quotation cannot
    # drift away from the note that disowns it and become a live claim again by accident.
    marker_window = 12
    markers = ("withdrawn", "retired", "superseded", "previously read", "no longer")
    corpus = list(REPO.glob("docs/*.md")) + list(REPO.glob("rust/src/**/*.rs"))
    for path in corpus:
        lines = path.read_text(encoding="utf-8").splitlines()
        rel = path.relative_to(REPO).as_posix()
        for phrase, why in stale.items():
            for i, line in enumerate(lines, start=1):
                if phrase not in line:
                    continue
                window = " ".join(lines[max(0, i - 1 - marker_window) : i]).lower()
                assert any(m in window for m in markers), (
                    f"{rel}:{i} carries {phrase!r} with no withdrawal marker in the "
                    f"{marker_window} lines above it — {why}\n  {line.strip()}"
                )


def test_no_document_says_the_anchor_count_is_read_rather_than_modelled():
    """The specific false framing that produced `225 anchors`, forbidden by construction."""
    for path in (DESIGN, OP_COVERAGE, PARTITION_RS):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            # Markdown emphasis is stripped so a disclaimer cannot be defeated by a `**not**`.
            low = re.sub(r"[*`_]", "", line).lower()
            if "anchor" in low and "partitionstats" in low:
                assert (
                    "no anchor" in low
                    or "has no" in low
                    or "not among" in low
                    or "is not" in low
                ), (
                    f"{path.name}:{i} associates an anchor count with PartitionStats, which has "
                    f"no anchor field:\n  {line.strip()}"
                )
