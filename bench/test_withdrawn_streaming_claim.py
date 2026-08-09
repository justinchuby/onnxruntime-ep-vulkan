#!/usr/bin/env python3
"""Guards for the §26.4 withdrawal (issue #81).

A withdrawal is a claim like any other, and the ways it rots are specific: the withdrawn figures
get quietly re-published somewhere else in the document; the correction loses its "MODEL" label and
starts reading as a measurement; the byte figure drifts away from the artifact field it was derived
from; or somebody attaches a number to the `(8, 4)` tile, which has never been run.

Each of those is a separate test here, and each is written so that it fails when the *document*
moves — not when an unrelated file does. The tests were checked against deliberate mutations:
re-publishing the withdrawn share, re-typing the byte figure as a literal, dropping the MODEL
label, and quoting a time for `(8, 4)` each turn one of them red.

Deliberately NOT asserted here: that the withdrawn numbers are absent from the document. They must
be *present*, inside the withdrawal, or a reader cannot tell what was withdrawn. What is asserted
is that they never appear outside it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "docs" / "PERF.md"


def _perf() -> str:
    return PERF.read_text(encoding="utf-8")


def _section(text: str, number: str) -> str:
    """The body of `### <number>` up to the next heading at the same or a shallower depth."""
    start = re.search(rf"^#+\s*{re.escape(number)}\b", text, re.M)
    assert start, f"docs/PERF.md has no section {number}"
    depth = len(re.match(r"#+", text[start.start():]).group(0))
    rest = text[start.end():]
    nxt = re.search(rf"^#{{1,{depth}}}\s+\d", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


#: The exact strings the withdrawal is about. Each may appear ONLY inside a withdrawal context.
WITHDRAWN = ("200–245", "median **227**", "~323 ms", "~11%", "ceil(M/4)", "2.291 GB")


class TestTheWithdrawnFiguresStayWithdrawn:
    @pytest.mark.parametrize("figure", WITHDRAWN)
    def test_each_withdrawn_figure_appears_only_beside_the_word_withdrawn(self, figure):
        text = _perf()
        for m in re.finditer(re.escape(figure), text):
            window = text[max(0, m.start() - 1400): m.end() + 600]
            assert re.search(r"withdraw", window, re.I), (
                f"docs/PERF.md offset {m.start()} uses {figure!r} outside any withdrawal context; "
                "these figures were retracted by issue #81 and may only be quoted as retracted"
            )

    def test_the_withdrawal_is_actually_in_the_section_that_published_them(self):
        assert "WITHDRAWN" in _section(_perf(), "26.4")

    def test_the_withdrawal_names_both_defects_not_just_the_divisor(self):
        """Fixing only the divisor would invite a rescaled republication of the same numbers."""
        section = _section(_perf(), "26.4")
        assert "ceil(M/2)" in section
        assert "1,861,189,632" in section
        assert "2.291" in section

    def test_no_replacement_bandwidth_is_published_anywhere(self):
        """Any GB/s figure in §26.4 or §26.4.1 must sit inside the withdrawal."""
        for number in ("26.4", "26.4.1"):
            section = _section(_perf(), number)
            for m in re.finditer(r"\bGB/s\b", section):
                window = section[max(0, m.start() - 1400): m.end() + 200]
                assert re.search(r"withdraw", window, re.I), (
                    f"§{number} publishes a GB/s figure outside the withdrawal; issue #81 "
                    "withdrew the reading and published no replacement"
                )


class TestTheSurvivingFiguresAreBoundToArtifactFields:
    def test_the_witnessed_pass_counts_are_the_probes_own(self):
        """`ceil(M/2)` is not an inference from the constants; it is a trace field."""
        doc = json.loads(
            (ROOT / "bench" / "results" / "weight_reread_phi35.json").read_text(encoding="utf-8")
        )
        rows = [
            r for r in doc["by_shape_prefill"]
            if r.get("N") == "ALL SHAPES"
        ]
        seen = {r["m_total"]: r["tiled_amplification"] for r in rows}
        assert seen == {2: 1.0, 4: 2.0, 5: 3.0}, seen
        for m, passes in seen.items():
            assert passes == -(-m // 2), (m, passes)
        section = _section(_perf(), "26.4.1")
        assert "1.0, 2.0, 3.0" in section
        assert "[2, 4, 5]" in section

    def test_the_byte_figure_is_the_artifacts_denominator_times_the_model_pass_count(self):
        doc = json.loads(
            (ROOT / "bench" / "results" / "weight_reread_phi35.json").read_text(encoding="utf-8")
        )
        weight_bytes = doc["denominator"]["int4_weight_bytes_from_graph"]
        assert weight_bytes == 1861189632, weight_bytes
        assert doc["denominator"]["matmulnbits_nodes"] == 161
        passes = -(-128 // 2)
        assert passes == 64
        total = passes * weight_bytes
        assert total == 119116136448, total
        section = _section(_perf(), "26.4.1")
        assert f"{total:,}" in section, f"§26.4.1 does not publish {total:,} B"
        assert "119.1" in section

    def test_the_extended_pass_counts_are_labelled_MODEL_not_measured(self):
        section = _section(_perf(), "26.4.1")
        for m in re.finditer(r"\bM = 128\b", section):
            window = section[max(0, m.start() - 700): m.end() + 700]
            assert "MODEL" in window, (
                "§26.4.1 quotes an M = 128 figure with no MODEL label nearby; the probe recorded "
                "M in {2, 4, 5} and nothing wider"
            )


class TestTheUnrunTileIsNotQuoted:
    def test_the_eight_by_four_tile_is_named_UNMEASURED(self):
        section = _section(_perf(), "26.4.1")
        assert "UNMEASURED" in section
        assert "(8, 4)" in section

    def test_no_time_or_ratio_is_attached_to_it(self):
        """Nothing in this tree has run `(8, 4)`. A number beside it would be invented."""
        text = _perf()
        for m in re.finditer(r"\(8,\s*4\)|GEMV_TILE=8,4", text):
            window = text[m.end(): m.end() + 400]
            offending = re.search(r"\b\d+(\.\d+)?\s*(ms|GB/s|×|x speedup)\b", window)
            assert offending is None, (
                f"docs/PERF.md attaches {offending.group(0)!r} to the (8,4) tile, which has never "
                "been run"
            )


class TestTheRustTestsArtifactClaimIsTheArtifacts:
    """`rust/tests/gemv_tile_request.rs` cites two committed diagnostics artifacts by record count
    and by pipeline key. Rust cannot read those JSONs, so nothing on that side stops the sentence
    from drifting. This is the guard that does.

    Checked against mutations: changing the record count, adding an `M32`/`M64` case to the claim,
    or altering a quoted key each turns one of these red.
    """

    ARTIFACTS = ("real_model_diagnostics.json", "real_model_diagnostics_before_gqa.json")

    def _doc(self, name: str) -> dict:
        return json.loads(
            (ROOT / "bench" / "results" / name).read_text(encoding="utf-8")
        )

    def _rust(self) -> str:
        return (
            ROOT / "rust" / "tests" / "gemv_tile_request.rs"
        ).read_text(encoding="utf-8")

    @pytest.mark.parametrize("name", ARTIFACTS)
    def test_the_record_count_is_the_artifacts_own(self, name):
        count = len(self._doc(name)["runs"])
        assert count == 18, (name, count)
        assert f"{count} run" in self._rust(), (
            f"rust/tests/gemv_tile_request.rs does not say '{count} run records' but "
            f"{name} has {count}"
        )

    @pytest.mark.parametrize("name", ARTIFACTS)
    def test_the_quoted_keys_are_recorded_where_the_comment_says_they_are(self, name):
        wanted = {
            "prefill/M8/past0": "q_gemv_matmul_nbits_f16:32,4,32,0,16,1,2",
            "prefill/M128/past0": "q_gemv_matmul_nbits_f16:32,4,32,0,16,1,2",
            "prefill/M1/past0": "q_gemv_matmul_nbits_f16:32,4,32,0,16,1,1",
            "decode/M1/past1024": "q_gemv_matmul_nbits_f16:32,4,32,0,16,1,1",
        }
        runs = [r for r in self._doc(name)["runs"] if r.get("arm") == "vulkan_tiled"]
        for suffix, key in wanted.items():
            matching = [r for r in runs if r.get("case", "").endswith("/" + suffix)]
            assert len(matching) == 1, (name, suffix, [r.get("case") for r in matching])
            variants = matching[0].get("counters", {}).get("pipeline_variants") or []
            assert key in variants, (name, suffix, variants)

    def test_the_comment_claims_no_case_the_artifacts_do_not_carry(self):
        """The specific rot this prevents: quoting an `M32` or `M64` pipeline that was never run."""
        cases = {
            r["case"].split("/", 1)[1]
            for name in self.ARTIFACTS
            for r in self._doc(name)["runs"]
            if "/" in r.get("case", "")
        }
        rust = self._rust()
        for m in re.finditer(r"prefill/M(\d+)", rust):
            assert f"prefill/M{m.group(1)}/past0" in cases, (
                f"rust/tests/gemv_tile_request.rs cites prefill/M{m.group(1)}, which no committed "
                f"diagnostics record carries. Recorded prefill cases: {sorted(cases)}"
            )


class TestTheSharedMemoryArithmeticCannotRevert:
    def test_the_document_distinguishes_the_spend_from_the_floor(self):
        """`shared float red[2048]` is 8 KiB. 16 KiB is Vulkan's guaranteed floor, not the spend.

        Bound to the shader's own constant so the sentence cannot drift from the code.
        """
        shader = (
            ROOT / "rust" / "shaders" / "glsl" / "templates" / "q_gemv.comp"
        ).read_text(encoding="utf-8")
        assert "shared float red[QB_RED_WORDS];" in shader
        words = int(re.search(r"const GEMV_RED_WORDS: u32 = (\d+);", (
            ROOT / "rust" / "src" / "ops" / "quant.rs"
        ).read_text(encoding="utf-8")).group(1))
        assert words == 2048
        assert words * 4 == 8 * 1024

        section = _section(_perf(), "26.4.2")
        assert "8 KiB" in section
        assert "16 KiB" in section
        window = section[section.index("8 KiB"): section.index("8 KiB") + 400]
        assert "floor" in window, (
            "§26.4.2 must say which of the two numbers is the floor; conflating them is the "
            "error issue #81 corrected"
        )
