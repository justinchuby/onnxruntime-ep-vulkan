"""Locks `docs/PERF.md` §26 and the shader header against the artifacts they cite.

`bench/test_real_model.py` checks the *instrument*. This file checks the *publication*, which is a
different failure mode and the one that got PR #64 rejected. Every defect below was found in a
published draft of §26, and every one of them was internally plausible:

* a ratio cell copied from the column next to it (`decode past = 512`: `0.42×`, the `vk/cpu`
  value, printed as the row-tile speedup — the artifact says `1.0153`);
* a **mean** published under the word "median" and then rounded down, so a 17% arm asymmetry
  read as 14%;
* by-kernel rows that were self-consistent and appear in no committed artifact;
* a citation naming a file that does not carry the field being quoted;
* correctness gates described as bitwise when the code applies a three-band tolerance, and as
  top-5 when the code applies an elementwise tolerance;
* two conventions for "ratio" mixed inside one table, one column each.

Nothing here restates a constant. Every expected value is re-derived from `bench/results/*.json`
or from the source file that implements the rule, so a test that passes is evidence the document
agrees with the measurement rather than evidence that two copies of the same typo agree.

No GPU, no EP, no model file.
"""

from __future__ import annotations

import ast
import json
import re
import statistics
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
_REPO = _BENCH.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import real_model as rm  # noqa: E402

RESULTS = _BENCH / "results"
PERF = _REPO / "docs" / "PERF.md"
DESIGN = _REPO / "docs" / "DESIGN.md"
SHADER = _REPO / "rust" / "shaders" / "glsl" / "gqa_f16.comp"
ATTENTION_RS = _REPO / "rust" / "src" / "ops" / "attention.rs"

#: The four latency matrices §26 quotes, by the subsection that quotes each.
LATENCY_ARTIFACTS = {
    "26.3": "real_model_latency_before_gqa.json",
    "26.6": "real_model_latency.json",
    "26.9": "real_model_latency_postmerge.json",
    "26.10": "real_model_latency_on_main.json",
}
#: The only committed artifact carrying a per-kernel GPU breakdown.
PER_KERNEL_ARTIFACT = "real_model_gqa_local_size.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        pytest.skip(f"{name} is not committed in this tree")
    return json.loads(path.read_text(encoding="utf-8"))


def _perf() -> str:
    if not PERF.is_file():
        pytest.skip("docs/PERF.md is absent")
    return PERF.read_text(encoding="utf-8")


def _section(text: str, number: str) -> str:
    """The body of `### {number} ...` up to the next `### 26.x` heading.

    Sliced by heading rather than by line number so renumbering or inserting prose above does not
    silently move which text a test reads — a test that reads the wrong section passes for the
    wrong reason.
    """
    starts = [m for m in re.finditer(r"^### (26\.\d+)\b", text, re.M)]
    for i, m in enumerate(starts):
        if m.group(1) == number:
            end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            return text[m.start():end]
    raise AssertionError(f"docs/PERF.md has no section {number}")


def _phi(doc: dict) -> dict:
    for model in doc["models"]:
        key = model["model"]
        name = key["key"] if isinstance(key, dict) else key
        if "phi" in name.lower():
            return model
    raise AssertionError("no phi-3.5 model block in the artifact")


def _rows(doc: dict) -> dict[str, dict]:
    """Rows by their trailing `phase/M{m}/past{n}` key, with the model prefix stripped.

    The artifact's `case` is `{model key}/prefill/M1/past0`; the document names the case alone.
    """
    out = {}
    for row in _phi(doc)["rows"]:
        out["/".join(row["case"].split("/")[-3:])] = row
    return out


#: `| prefill M=1 | 27.52 | ... |` -> ("prefill M=1", ["27.52", ...])
_TABLE_ROW = re.compile(r"^\|\s*(?P<case>[^|]+?)\s*\|(?P<rest>.*)\|\s*$", re.M)


def _table(section: str, *header_cells: str) -> str:
    """The one markdown table in `section` whose header row contains all of `header_cells`.

    §26.6 holds two tables that both have `prefill M=8` rows and different widths, so selecting
    rows by their label alone reads cells out of whichever table came first. Selecting the table
    by its header makes the test say which table it means.
    """
    blocks, current = [], []
    for line in section.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    matches = [
        "\n".join(block)
        for block in blocks
        if len(block) >= 2 and all(cell in block[0] for cell in header_cells)
    ]
    assert len(matches) == 1, (
        f"expected exactly one table whose header holds {header_cells}, found {len(matches)}"
    )
    return matches[0]


def _table_rows(section: str, case_pattern: str) -> list[tuple[str, list[str]]]:
    out = []
    for m in _TABLE_ROW.finditer(section):
        case = m.group("case").strip().strip("*")
        if not re.match(case_pattern, case):
            continue
        cells = [c.strip() for c in m.group("rest").split("|")]
        out.append((case, cells))
    return out


def _num(cell: str) -> float:
    """The number in a table cell, with markdown emphasis and the `×` suffix stripped."""
    bare = cell.replace("*", "").replace("×", "").replace("`", "").strip()
    return float(bare)


def _case_key(label: str) -> str:
    """`prefill M=1` -> the artifact's `case` key. Derived, not tabulated."""
    label = label.replace("(null control)", "").strip()
    m = re.match(r"prefill M=(\d+)$", label)
    if m:
        return f"prefill/M{m.group(1)}/past0"
    m = re.match(r"decode past=(\d+)$", label)
    if m:
        return f"decode/M1/past{m.group(1)}"
    raise AssertionError(f"unrecognised case label {label!r}")


# ---------------------------------------------------------------------------
# §26.3 — the table cells are the artifact's own fields
# ---------------------------------------------------------------------------

def test_26_3_latency_cells_are_the_artifact_medians():
    """The three ms columns are `latency.median_ms`, not `per_repeat_median_ms`.

    The two differ — at `M = 1` tiled they are 27.52 and 27.83 — and both are in the artifact, so
    quoting the wrong one produces a number a reader can find and still cannot reproduce.
    """
    rows = _rows(_load(LATENCY_ARTIFACTS["26.3"]))
    wrong = []
    for case, cells in _table_rows(_section(_perf(), "26.3"), r"(prefill M=|decode past=)"):
        row = rows[_case_key(case)]
        for column, arm in ((0, "vulkan_tiled"), (1, "vulkan_untiled"), (2, "cpu")):
            want = round(row["arms"][arm]["latency"]["median_ms"], 2)
            got = _num(cells[column])
            if abs(got - want) > 0.005:
                wrong.append(f"{case} {arm}: table {got}, artifact median_ms {want}")
    assert not wrong, "\n".join(wrong)


def test_26_3_ratio_cells_are_their_own_artifact_fields():
    """`row-tile` is `row_tile_speedup.median`; `vk/cpu` is `vulkan_vs_cpu_tiled.median`."""
    rows = _rows(_load(LATENCY_ARTIFACTS["26.3"]))
    wrong = []
    for case, cells in _table_rows(_section(_perf(), "26.3"), r"(prefill M=|decode past=)"):
        row = rows[_case_key(case)]
        for column, field in ((3, "row_tile_speedup"), (4, "vulkan_vs_cpu_tiled")):
            want = round(row[field]["median"], 2 if field == "vulkan_vs_cpu_tiled" else 3)
            got = _num(cells[column])
            if abs(got - want) > 0.006:
                wrong.append(f"{case} {field}: table {got}, artifact median {want}")
    assert not wrong, "\n".join(wrong)


def test_no_ratio_cell_is_its_neighbours_value():
    """THE B1 CONTROL. `row-tile` and `vk/cpu` are adjacent, unrelated and were swapped once.

    Checked in BOTH directions and for every row, because "the number is in the artifact" is
    exactly what made the defect survive review: `0.42×` was a real measurement of a real thing,
    printed one column to the left of where it belonged.
    """
    rows = _rows(_load(LATENCY_ARTIFACTS["26.3"]))
    wrong = []
    for case, cells in _table_rows(_section(_perf(), "26.3"), r"(prefill M=|decode past=)"):
        row = rows[_case_key(case)]
        row_tile = row["row_tile_speedup"]["median"]
        vk_cpu = row["vulkan_vs_cpu_tiled"]["median"]
        if abs(row_tile - vk_cpu) < 0.02:
            continue  # the two fields genuinely coincide here; a swap is unobservable
        if abs(_num(cells[3]) - vk_cpu) < 0.006:
            wrong.append(f"{case}: the row-tile cell holds the vk/cpu value {vk_cpu:.4f}")
        if abs(_num(cells[4]) - row_tile) < 0.006:
            wrong.append(f"{case}: the vk/cpu cell holds the row-tile value {row_tile:.4f}")
    assert not wrong, "\n".join(wrong)


def test_26_3_does_not_bold_a_ratio_inside_the_null_control():
    """Bold is a claim. A ratio the document itself calls unreadable may not be emphasised.

    `decode past = 512`'s row-tile cell was bold at `0.42×`; at its true `1.015×` it sits inside
    the null control's width, which §26.3 says makes it not a reading at all.
    """
    rows = _rows(_load(LATENCY_ARTIFACTS["26.3"]))
    floor = _phi(_load(LATENCY_ARTIFACTS["26.3"]))["noise_floor"]["ratios"]
    lo, hi = min(floor), max(floor)
    bold = []
    for case, cells in _table_rows(_section(_perf(), "26.3"), r"(prefill M=|decode past=)"):
        ratio = rows[_case_key(case)]["row_tile_speedup"]["median"]
        if "**" in cells[3] and lo <= ratio <= hi:
            bold.append(f"{case}: {cells[3]} is inside the null control {lo:.3f}-{hi:.3f}")
    assert not bold, "\n".join(bold)


# ---------------------------------------------------------------------------
# §26.6 — one convention, stated and applied
# ---------------------------------------------------------------------------

def test_26_6_is_ratios_of_medians_throughout():
    """THE MIXED-CONVENTION CONTROL.

    `before` and `after` are separate sessions, so the artifact's paired ratios do not exist
    across them. §26.6 therefore declares ratio-of-medians and must use it in EVERY cell — the
    rejected draft took its `vk/cpu before` column from §26.3's paired field and computed its
    `after` column as a ratio of medians, which is two statistics in one column with no marking.
    """
    section = _section(_perf(), "26.6")
    assert "ratios of medians" in section, (
        "§26.6 no longer states its convention; an unstated convention is the defect"
    )
    table = _table(section, "before", "after", "gain", "vk/cpu")
    before, after = _rows(_load(LATENCY_ARTIFACTS["26.3"])), _rows(_load(LATENCY_ARTIFACTS["26.6"]))
    wrong = []
    for case, cells in _table_rows(table, r"(prefill M=|decode past=)"):
        key = _case_key(case)
        b, a = before[key], after[key]
        b_tiled = b["arms"]["vulkan_tiled"]["latency"]["median_ms"]
        a_tiled = a["arms"]["vulkan_tiled"]["latency"]["median_ms"]
        want_gain = b_tiled / a_tiled
        if abs(_num(cells[2]) - round(want_gain, 3)) > 0.0006:
            wrong.append(f"{case} gain: table {cells[2]}, ratio-of-medians {want_gain:.4f}")
        got_before, got_after = (x.strip() for x in cells[3].split("→"))
        for got, doc in ((got_before, b), (got_after, a)):
            want = (
                doc["arms"]["cpu"]["latency"]["median_ms"]
                / doc["arms"]["vulkan_tiled"]["latency"]["median_ms"]
            )
            if abs(_num(got) - round(want, 2)) > 0.006:
                wrong.append(f"{case} vk/cpu: table {got}, ratio-of-medians {want:.4f}")
    assert not wrong, "\n".join(wrong)


def test_26_6_vk_cpu_before_is_not_26_3s_paired_value():
    """The specific leak the previous draft had, asserted directly rather than by arithmetic."""
    before = _rows(_load(LATENCY_ARTIFACTS["26.3"]))
    table = _table(_section(_perf(), "26.6"), "before", "after", "gain", "vk/cpu")
    wrong = []
    for case, cells in _table_rows(table, r"(prefill M=|decode past=)"):
        row = before[_case_key(case)]
        paired = row["vulkan_vs_cpu_tiled"]["median"]
        unpaired = (
            row["arms"]["cpu"]["latency"]["median_ms"]
            / row["arms"]["vulkan_tiled"]["latency"]["median_ms"]
        )
        if abs(round(paired, 2) - round(unpaired, 2)) < 0.005:
            continue  # indistinguishable at two decimals here
        got = _num(cells[3].split("→")[0])
        if abs(got - round(paired, 2)) < 0.005:
            wrong.append(
                f"{case}: §26.6 'before' is {got}, which is §26.3's PAIRED "
                f"{paired:.4f}, not the ratio of medians {unpaired:.4f}"
            )
    assert not wrong, "\n".join(wrong)


# ---------------------------------------------------------------------------
# A mean may not be labelled a median
# ---------------------------------------------------------------------------

def _null_control_ratios(name: str) -> list[float]:
    """Per-repeat `M = 1` tiled ÷ untiled, recomputed from the artifact's own repeat medians."""
    row = _rows(_load(name))["prefill/M1/past0"]
    tiled = row["arms"]["vulkan_tiled"]["per_repeat_median_ms"]
    untiled = row["arms"]["vulkan_untiled"]["per_repeat_median_ms"]
    return [t / u for t, u in zip(tiled, untiled)]


def test_the_null_control_median_is_a_median():
    """THE B5 CONTROL. 1.143 is the MEAN of [1.16739, 1.22372, 1.03830]; the median is 1.16739.

    Both numbers are real and both are in this document, which is why the defect survived: the
    rejected draft printed the mean, called it the median, and rounded it to "~14%" where the
    median rounds to ~17%.
    """
    ratios = _null_control_ratios(LATENCY_ARTIFACTS["26.10"])
    median, mean = statistics.median(ratios), statistics.fmean(ratios)
    assert abs(median - 1.16739) < 5e-5 and abs(mean - 1.14313) < 5e-5, (ratios, median, mean)

    text = _perf()
    for m in re.finditer(r"median[^.\n]{0,40}?(\d\.\d{3})", text):
        value = float(m.group(1))
        if abs(value - round(mean, 3)) < 5e-4 and abs(value - round(median, 3)) > 5e-4:
            raise AssertionError(
                f"docs/PERF.md calls {value} a median at offset {m.start()}; that is the MEAN of "
                f"the {LATENCY_ARTIFACTS['26.10']} null control. Its median is {median:.5f}."
            )


def test_no_null_control_claim_rounds_the_asymmetry_to_fourteen_percent():
    """The prose form of the same error. ~14% is the mean's rounding; the median's is ~17%."""
    ratios = _null_control_ratios(LATENCY_ARTIFACTS["26.10"])
    pct = round((statistics.median(ratios) - 1) * 100)
    assert pct == 17, ratios
    for path in (PERF, DESIGN):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"~?\s?(\d{1,2})%\s+(?:separation|asymmetry|apart)", text):
            assert int(m.group(1)) == pct, (
                f"{path.name} calls the null-control asymmetry {m.group(1)}%; the median of "
                f"{[round(r, 5) for r in ratios]} is {pct}%"
            )


def test_the_null_control_band_is_one_orientation_at_a_time():
    """"0.79 - 1.26" was one endpoint from each orientation: a band that exists in neither.

    Scoped to the bands the document itself says are taken *across all four runs* — a per-run
    noise floor is a different, legitimate band, and a test that conflated them would be the same
    class of error it exists to catch.
    """
    ratios = [r for name in LATENCY_ARTIFACTS.values() for r in _null_control_ratios(name)]
    fwd = (min(ratios), max(ratios))
    rev = (min(1 / r for r in ratios), max(1 / r for r in ratios))
    assert abs(fwd[0] - 0.79102) < 5e-5 and abs(fwd[1] - 1.22372) < 5e-5, fwd
    assert abs(rev[0] - 0.81718) < 5e-5 and abs(rev[1] - 1.26419) < 5e-5, rev
    text = _perf()
    checked = 0
    for m in re.finditer(r"(\d\.\d{2,3})\s*[-–]\s*(\d\.\d{2,3})", text):
        window = text[max(0, m.start() - 300):m.end() + 300]
        if not re.search(r"(all )?four runs|across the four", window):
            continue
        if re.search(r"earlier draft quoted|exists in neither", window):
            continue  # the withdrawal names the band it withdraws, which is the point of it
        lo, hi = float(m.group(1)), float(m.group(2))
        if not (0.7 < lo < 1.0 and 1.1 < hi < 1.4):
            continue
        checked += 1
        assert (abs(lo - fwd[0]) < 0.006 and abs(hi - fwd[1]) < 0.006) or (
            abs(lo - rev[0]) < 0.006 and abs(hi - rev[1]) < 0.006
        ), (
            f"docs/PERF.md publishes the four-run null-control band {lo}-{hi}, which is neither "
            f"{fwd[0]:.5f}-{fwd[1]:.5f} (tiled/untiled) nor {rev[0]:.5f}-{rev[1]:.5f} "
            "(row_tile_speedup orientation). Mixing the endpoints invents a band."
        )
    assert checked, "no four-run null-control band found in docs/PERF.md; this test is vacuous"


# ---------------------------------------------------------------------------
# Per-kernel rows must exist in the artifact
# ---------------------------------------------------------------------------

def _by_kernel(case: str, local_size: int) -> tuple[dict[str, float], float]:
    doc = _load(PER_KERNEL_ARTIFACT)
    for entry in doc["timing"]:
        if entry["case"] != case:
            continue
        for point in entry["points"]:
            if point["local_size"] == local_size:
                return point["by_kernel_us"], point["total_us"]
    raise AssertionError(f"{PER_KERNEL_ARTIFACT} has no {case} at local_size={local_size}")


def _rule_local_size(total: int) -> int:
    """`ops::attention::gqa_local_size_with`, re-read from the Rust rather than restated."""
    src = ATTENTION_RS.read_text(encoding="utf-8")
    cap = int(re.search(r"GQA_MAX_LOCAL_SIZE: u32 = (\d+)", src).group(1))
    minimum = int(re.search(r"GQA_MIN_GROUPS: u32 = (\d+)", src).group(1))
    local = 1
    while local * 2 <= cap and total // (local * 2) >= minimum:
        local *= 2
    return local


def test_every_published_by_kernel_row_exists_in_the_artifact():
    """THE ORPHAN CONTROL. A by-kernel ms figure in no artifact is an orphan, however plausible.

    The rejected draft carried a `2911.21 / 1884.19 / 1020.11` triple that balanced perfectly and
    came from nowhere committed. Internal consistency is not provenance.
    """
    tables = [
        _table(_section(_perf(), "26.4"), "GPU total", "gqa_f16"),
        _table(_section(_perf(), "26.6"), "GPU total", "gqa_f16"),
    ]
    doc = _load(PER_KERNEL_ARTIFACT)
    known = set()
    for entry in doc["timing"]:
        for point in entry["points"]:
            known.add(round(point["total_us"] / 1000, 2))
            for value in point["by_kernel_us"].values():
                known.add(round(value / 1000, 2))
    orphans = []
    for table in tables:
        for case, cells in _table_rows(table, r"(prefill M=|decode past=)"):
            for cell in cells:
                for token in re.findall(r"\d+\.\d\d", cell):
                    if float(token) not in known:
                        orphans.append(
                            f"{case}: {token} ms appears in no {PER_KERNEL_ARTIFACT} point"
                        )
    assert not orphans, "\n".join(sorted(set(orphans)))


def test_the_per_kernel_rows_use_the_size_the_rule_picks():
    """§26.6's `after` column claims "the sizes the rule picks"; the rule is in the Rust.

    Re-evaluated from `GQA_MIN_GROUPS` and `GQA_MAX_LOCAL_SIZE` rather than restated, so a change
    to the dispatch rule that nobody re-measured shows up here as a document that quotes a point
    the rule no longer selects.
    """
    doc = _load(PER_KERNEL_ARTIFACT)
    table = _table(_section(_perf(), "26.6"), "GPU total", "gqa_f16")
    wrong = []
    for entry in doc["timing"]:
        if not entry["case"].startswith("prefill/M"):
            continue
        m = int(entry["case"].split("/M")[1].split("/")[0])
        row = re.search(rf"^\|\s*prefill M={m}\s*\|.*$", table, re.M)
        if not row:
            continue  # this size is not one of the rows §26.6 prints per-kernel
        local = _rule_local_size(entry["invocations"])
        by_kernel, total = _by_kernel(entry["case"], local)
        gqa = round(by_kernel["vulkan.gpu.gqa_f16"] / 1000, 2)
        total_ms = round(total / 1000, 2)
        for want, what in ((total_ms, "GPU total after"), (gqa, f"gqa_f16 after (local={local})")):
            if f"{want:.2f}" not in row.group(0):
                wrong.append(
                    f"prefill M={m}: the rule picks local_size={local}, whose {what} is "
                    f"{want:.2f} ms; §26.6's row is {row.group(0).strip()}"
                )
    assert not wrong, "\n".join(wrong)


def test_the_gqa_share_at_m128_is_the_artifact_share():
    """64.6%, from the one field that holds it. 64.7% was published and is not this number.

    A withdrawal may name the number it withdraws — that is what makes it a withdrawal — so an
    occurrence is an offence only when nothing near it says the number is withdrawn.
    """
    by_kernel, total = _by_kernel("prefill/M128/past0", 1)
    share = 100 * by_kernel["vulkan.gpu.gqa_f16"] / total
    assert abs(share - 64.60) < 0.05, share
    for path in (PERF, DESIGN, SHADER):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"64\.7%", text):
            window = text[max(0, m.start() - 400):m.end() + 400]
            assert re.search(r"withdraw|not 64\.7%|both are withdrawn|and quoted 64\.7%", window), (
                f"{path.name} at offset {m.start()} publishes 64.7% as a live figure; the "
                f"artifact share is {share:.2f}%"
            )


# ---------------------------------------------------------------------------
# Citations must name the file that holds the field
# ---------------------------------------------------------------------------

def test_real_model_diagnostics_carries_no_per_kernel_gpu_field():
    """The premise of the next test, asserted rather than assumed."""
    doc = _load("real_model_diagnostics.json")
    blob = json.dumps(doc)
    assert "by_kernel_us" not in blob, (
        "real_model_diagnostics.json now DOES carry per-kernel GPU time; the citation rule "
        "below was written when it did not, and must be re-derived rather than relaxed"
    )
    assert "vulkan.gpu." not in blob


def test_nothing_cites_the_diagnostics_file_for_per_kernel_gpu_time():
    """THE B2 CONTROL. The shader header cited a file with no such field, and quoted 64.7%.

    The trigger is the *field* (`by_kernel_us`) or the *claim form* ("% of all GPU time"), not the
    loose phrase "per-kernel" — §26.1's provenance table legitimately names the file and the words
    in adjacent cells, and a screen that fired on that would train readers to ignore it.
    """
    offenders = []
    for path in (PERF, DESIGN, SHADER):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"real_model_diagnostics(?:_before_gqa)?\.json", text):
            window = text[max(0, m.start() - 250):m.end() + 250]
            if not re.search(r"by_kernel_us|% of all GPU time", window):
                continue
            if re.search(r"\*\*not\*\* the source|carries no per-kernel|contains no per-kernel",
                         window):
                continue  # the withdrawal itself, which must stay legible
            offenders.append(f"{path.name} at offset {m.start()}")
    assert not offenders, (
        "these cite real_model_diagnostics.json for a per-kernel GPU share it does not hold: "
        + ", ".join(offenders)
    )


def test_the_shader_header_cites_the_artifact_that_holds_the_number():
    """Positive form: the header must name the file, the field and the point selector."""
    header = SHADER.read_text(encoding="utf-8")[:4000]
    for needle in (
        "bench/results/real_model_gqa_local_size.json",
        "by_kernel_us",
        "local_size == 1",
        "64.6%",
    ):
        assert needle in header, f"gqa_f16.comp's header no longer states {needle!r}"


def test_no_shader_comment_premises_a_local_size_of_one():
    """The size is a specialisation constant; a comment that assumes 1 is a stale premise.

    Both surviving instances were correctness arguments — "the atomics are never contended" and
    "they are separate workgroups" — so a reader who trusted them would conclude that neither the
    atomics nor the read-only recompute were load-bearing. Both are.
    """
    text = SHADER.read_text(encoding="utf-8")
    bad = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("//"):
            continue
        if re.search(r"local_size\s*=\s*1\b|local_size=1\b", stripped):
            if "may not assume" in stripped or "not knowable" in stripped:
                continue
            bad.append(f"{i}: {stripped}")
    assert not bad, "gqa_f16.comp comments still premise a workgroup size of 1:\n" + "\n".join(bad)


# ---------------------------------------------------------------------------
# The documented gates must be the gates the code applies
# ---------------------------------------------------------------------------

def test_the_kv_activation_gate_is_three_band_and_not_bitwise():
    """THE B4 CONTROL, part one. KV activations were published as bitwise; they are not.

    `classify_activation` applies a ULP-and-relative floor, a marginal band bounded by a fraction
    of elements, and a gross band that is fatal. Calling that "bitwise" claims a stronger result
    than was measured, and a bitwise gate would fail the very run being published.
    """
    assert rm.KV_ULP_BUDGET > 0 and 0 < rm.KV_REL_TOL < 1
    assert rm.KV_GROSS_MULTIPLE > 1 and 0 < rm.KV_MARGINAL_FRACTION < 1
    body = _function_source((_BENCH / "real_model.py").read_text(encoding="utf-8"),
                            "classify_activation")
    assert "KV_ULP_BUDGET" in body and "KV_GROSS_MULTIPLE" in body, (
        "classify_activation no longer applies the three-band rule this test describes"
    )
    text = PERF.read_text(encoding="utf-8")
    for m in re.finditer(r"bitwise", text, re.I):
        window = text[max(0, m.start() - 150):m.end() + 150]
        if not re.search(r"KV activation|present_key|present_value|classify_activation", window):
            continue
        assert re.search(r"not bitwise|rather than bitwise|is \*\*not\*\* bitwise|no bitwise",
                         window), (
            f"docs/PERF.md at offset {m.start()} describes a KV activation gate as bitwise; "
            "bench/real_model.py applies a three-band tolerance"
        )


def test_the_logits_gate_reports_relative_error_and_does_not_bound_it():
    """THE B4 CONTROL, part two. `max_rel` is 374 in the published run; no gate could pass it."""
    src = (_BENCH / "real_model.py").read_text(encoding="utf-8")
    body = _function_source(src, "classify_logits")
    assert "max_rel" in body, "classify_logits no longer computes a relative error at all"
    for line in body.splitlines():
        if "max_rel" in line and re.search(r"(<=|<|>|>=)", line) and "note" not in line:
            raise AssertionError(
                "classify_logits now GATES on max_rel; docs/PERF.md §26.2 says it is reported "
                f"only. Re-derive the prose before relaxing this test. Offending line: {line!r}"
            )
    section = _section(_perf(), "26.2")
    assert "reported" in section and "max_rel" in section


def test_the_mobilenet_gate_is_elementwise_tolerance_not_top_k():
    """THE B4 CONTROL, part three. Published as argmax+top-5; the code has neither top-5 nor k."""
    src = (_BENCH / "real_model.py").read_text(encoding="utf-8")
    body = _function_source(src, "classify_tensor")
    assert "MOBILENET_ATOL" in body and "MOBILENET_RTOL" in body
    assert "top" not in body.lower().replace("topology", ""), (
        "classify_tensor now consults a top-k; §26.2 says it does not"
    )
    section = _section(_perf(), "26.2")
    for m in re.finditer(r"top-5", section):
        window = section[max(0, m.start() - 200):m.end() + 200]
        assert re.search(r"not a top-5|never was|no top-5", window), (
            f"§26.2 at offset {m.start()} describes a top-5 gate MobileNetV2 never had"
        )


def test_the_documented_gates_are_the_gates_the_code_applies():
    """Every constant §26.2 quotes must be the module's live value."""
    section = _section(_perf(), "26.2")
    for name in (
        "PHI35_TOP_K",
        "PHI35_LOGIT_SCALE_FRACTION",
        "PHI35_MAX_PROB_DELTA",
        "MOBILENET_ATOL",
        "MOBILENET_RTOL",
        "KV_ULP_BUDGET",
        "KV_REL_TOL",
        "KV_GROSS_MULTIPLE",
        "KV_MARGINAL_FRACTION",
    ):
        value = getattr(rm, name)
        assert name in section, f"§26.2 does not name {name}, which it gates on"
        rendered = repr(value) if not isinstance(value, float) else None
        candidates = {str(value)}
        if rendered:
            candidates.add(rendered)
        if isinstance(value, float):
            candidates |= {f"{value:g}", f"{value}", f"{value:.0e}".replace("e-0", "e-")}
        assert any(c in section for c in candidates), (
            f"§26.2 names {name} but not its value {value!r} (looked for {sorted(candidates)})"
        )


def _function_source(module_src: str, name: str) -> str:
    tree = ast.parse(module_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(module_src, node) or ""
    raise AssertionError(f"bench/real_model.py has no function {name}")


# ---------------------------------------------------------------------------
# Bandwidth, hashes, test counts
# ---------------------------------------------------------------------------

def test_the_bandwidth_span_is_the_seven_differential_points():
    """THE B6 CONTROL. The published span and median must be the artifact's own points."""
    doc = _load("real_model_diagnostics_before_gqa.json")
    points = _differential_bandwidths(doc)
    assert len(points) == 7, points
    lo, hi = min(points), max(points)
    median = statistics.median(sorted(points))
    assert 199 < lo < 201 and 244 < hi < 245, (lo, hi)
    assert abs(median - 226.8) < 0.5, median
    text = _perf()
    for m in re.finditer(r"(\d{3})\s*[-–]\s*(\d{3})\s*GB/s", text):
        assert (int(m.group(1)), int(m.group(2))) == (round(lo / 5) * 5, round(hi / 5) * 5), (
            f"docs/PERF.md publishes {m.group(0)}; the seven differential points span "
            f"{lo:.1f}-{hi:.1f} GB/s"
        )
    for m in re.finditer(r"median\s*~?\s*(\d{3})\b[^\n]{0,12}GB/s", text):
        assert abs(int(m.group(1)) - median) < 1.5, (
            f"docs/PERF.md publishes a bandwidth median of {m.group(1)}; the sorted median of "
            f"{[round(p, 1) for p in points]} is {median:.1f}"
        )


def _differential_bandwidths(doc: dict) -> list[float]:
    found = []
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if "bandwidth" in key and "gb_s" in key and isinstance(value, (int, float)):
                    found.append(float(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(doc)
    if not found:
        pytest.skip("this artifact carries no differential bandwidth points")
    return found


def test_26_1_lists_every_ep_library_hash_the_artifacts_carry():
    """No single-hash implication: §26 spans several builds and must say which is which."""
    hashes = {}
    for section, name in LATENCY_ARTIFACTS.items():
        doc = _load(name)
        sha = _find_ep_sha(doc)
        if sha:
            hashes.setdefault(sha, []).append(name)
    sha = _find_ep_sha(_load(PER_KERNEL_ARTIFACT))
    if sha:
        hashes.setdefault(sha, []).append(PER_KERNEL_ARTIFACT)
    assert len(hashes) > 1, "this test is vacuous unless the artifacts really span builds"
    section = _section(_perf(), "26.1")
    missing = [h[:8] for h in hashes if h[:8] not in section and h not in section]
    assert not missing, (
        "§26.1 does not name these EP builds, so it implies one binary produced everything: "
        + ", ".join(sorted(missing))
    )


def _find_ep_sha(doc: dict) -> str | None:
    found = []
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "ep_library_sha256" and isinstance(value, str):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(doc)
    return found[0] if found else None


def _attention_tests() -> dict[str, str]:
    """Every `#[test] fn` in `rust/src/ops/attention.rs`, name -> body, by brace matching."""
    src = ATTENTION_RS.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"#\[test\]\s*(?:#\[[^\]]*\]\s*)*fn\s+(\w+)\s*\(\s*\)\s*\{", src):
        start = m.end() - 1
        depth = 0
        end = start
        for i in range(start, len(src)):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        out[m.group(1)] = src[start:end + 1]
    return out


def _dispatch_rule_tests() -> list[str]:
    """The tests that lock the GQA workgroup-size rule, SELECTED BY WHAT THEY TOUCH.

    Not a hand-kept list and not "the ones added by this branch" — a branch-relative count cannot
    be recomputed after a squash, which is the same defect `rewitness/3` exists to fix. A test
    belongs to this set iff its body names the rule's constants, its function, or the
    specialisation constant it produces. That is a property of the tree, so it survives any
    landing shape.
    """
    return [
        name
        for name, body in _attention_tests().items()
        if re.search(r"local_size|LOCAL_SIZE|MIN_GROUPS|spec_constants", body)
    ]


def test_this_section_names_every_gqa_dispatch_test_that_exists():
    """THE TEST-COUNT CONTROL. The PR body said 8; the tree has 10, and had for two commits.

    Derived from the Rust, so the count cannot drift: adding a rule test without listing it fails
    here, and listing one that does not exist fails here too.
    """
    names = _dispatch_rule_tests()
    assert names, "no GQA dispatch-rule tests found; the rule or the selector moved"
    section = _section(_perf(), "26.8")
    missing = [n for n in names if n not in section]
    assert not missing, "§26.8 does not name these ops::attention tests: " + ", ".join(missing)
    spelled = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    counts = re.findall(r"\*\*(\w+)\*\*\s+`ops::attention` unit tests", section)
    assert counts, "§26.8 no longer states how many ops::attention unit tests lock the rule"
    for word in counts:
        got = spelled.get(word.lower()) or int(word)
        assert got == len(names), (
            f"§26.8 says {word} ops::attention unit tests; rust/src/ops/attention.rs has "
            f"{len(names)} that touch the dispatch rule: {sorted(names)}"
        )


def test_this_file_is_counted_where_it_is_cited():
    """§26.8 quotes a size for this module; a stale size is the same defect one level up."""
    section = _section(_perf(), "26.8")
    m = re.search(r"`bench/test_perf_claims\.py`\s*\(\*\*(\d+)\*\* GPU-free tests\)", section)
    assert m, "§26.8 no longer states this module's test count"
    mine = len([n for n in globals() if n.startswith("test_")])
    assert int(m.group(1)) == mine, (
        f"§26.8 says {m.group(1)} tests in bench/test_perf_claims.py; it defines {mine}"
    )
