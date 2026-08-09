"""Locks `docs/PERF.md` §27 and the two committed cross-build JSONs against the raw records.

`bench/test_crossbuild_summary.py` checks the **instrument** — that every gate in
`bench/crossbuild_summary.py` is load-bearing, by mutating that module's own source and demanding
the mutants go red. This file checks the **publication**, which is a different failure mode and
the one that got PR #95 rejected: a table can be internally consistent, arithmetically correct,
and still say something the records do not support.

Every defect screened for below was real, in a published draft of the artifact this supersedes:

* a band computed from a null control's own three ratios and then applied *to that control*, so
  its `NEUTRAL` verdict held in every possible universe and measured nothing;
* the *widest deviation* of a set published under the name "half-range" (5.10% against a true
  half-range of 4.45%);
* "pre-registered" said of a rule whose only timestamp is the artifact that embeds it;
* "exclusive GPU access" said of a cooperative `msvcrt` byte-range lock, in a record that itself
  lists 26 GPU application entries running at the time;
* a two-commit binary delta described as one shader change, with a third compiled input
  (`evidence/proof_ledger.jsonl`, `include_str!`'d) not enumerated at all;
* "155 tests" cited beside a file that has 14;
* `[32, 1, 1]` printed as if it were a field of the artifact, when no artifact field holds a grid;
* "real models" plural in a title whose evidence is Phi-3.5 prefill.

Nothing here restates a constant. Every expected number is recomputed from the frozen raw records
**through the shipped summarizer** — the same functions the CLI and the artifact use — so a
passing test is evidence that the document agrees with the measurement rather than evidence that
two copies of the same claim agree with each other.

No GPU, no EP, no model file, no network.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
_REPO = _BENCH.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import crossbuild_summary as xb  # noqa: E402

PERF = _REPO / "docs" / "PERF.md"
DESIGN = _REPO / "docs" / "DESIGN.md"
SUMMARY_PATH = _BENCH / "results" / "real_model_crossbuild_gqa_landing_v2.json"

BASELINE_COMMIT = "c96e7d94ff706d26ee6a1bd9bb084c0ade426820"
CANDIDATE_COMMIT = "85fbda29a92e0e99c3895be8b13664d4ee670c50"

#: Everything that reaches the cdylib. `evidence/proof_ledger.jsonl` is in here because
#: `rust/src/registry.rs` `include_str!`s it — the input PR #95's compiled-delta claim omitted,
#: and the reason attribution here is bounded rather than isolated.
COMPILED_EXACT = {
    "rust/build.rs",
    "rust/Cargo.toml",
    "rust/Cargo.lock",
    "rust/wrapper_ort.h",
    "rust/src/ops/shader_variants.txt",
    "evidence/proof_ledger.jsonl",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _perf() -> str:
    if not PERF.is_file():
        pytest.skip("docs/PERF.md is absent")
    return PERF.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """`text` with every whitespace run collapsed to one space.

    Every prose assertion below goes through this. A phrase that a hard wrap happens to split
    across two lines is still the phrase the document makes; a test that fails on the wrap is
    testing the formatter, and the temptation it creates is to reflow the sentence rather than
    to keep it true.
    """
    return re.sub(r"\s+", " ", text)


def _section(text: str, number: str) -> str:
    """The body of `### 27.x ...` up to the next `### 27.y`, or all of `## 27` for "27".

    Sliced by heading rather than by line number: a test that reads the wrong section passes for
    the wrong reason, and §27 sits at the end of a file that is still growing.
    """
    if number == "27":
        m = re.search(r"^## 27\.", text, re.M)
        if not m:
            raise AssertionError("docs/PERF.md has no section 27")
        nxt = re.search(r"^## 28\.", text[m.end():], re.M)
        return text[m.start(): m.end() + nxt.start()] if nxt else text[m.start():]
    starts = list(re.finditer(r"^### (27\.\d+)\b", text, re.M))
    for i, m in enumerate(starts):
        if m.group(1) == number:
            end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            return text[m.start(): end]
    raise AssertionError(f"docs/PERF.md has no section {number}")


def _table(section: str, *header_cells: str) -> "list[list[str]]":
    """Rows of the one table in `section` whose header holds every one of `header_cells`.

    Selecting the table by its header rather than by position is what stops a test reading cells
    out of whichever table came first — §27 carries eight of them.
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
    hits = [b for b in blocks if len(b) >= 2 and all(c in b[0] for c in header_cells)]
    assert len(hits) == 1, (
        f"expected exactly one table whose header holds {header_cells}, found {len(hits)}"
    )
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in hits[0][2:]
    ]


@pytest.fixture(scope="module")
def summary():
    """The shipped summarizer's own output, recomputed here from the frozen raw records."""
    return xb.summarize()


@pytest.fixture(scope="module")
def perf():
    return _perf()


# ---------------------------------------------------------------------------
# Provenance: the records were reused, not measured here
# ---------------------------------------------------------------------------

def test_the_frozen_file_is_the_pr95_bytes_and_the_document_says_whose_they_are(perf):
    """The digest §27.1 publishes must be the digest of the file on disk, and named as reused.

    A rebuild that quietly re-measured, or quietly edited, would still produce a plausible table.
    The only thing distinguishing "these are PR #95's records" from "these are some records" is
    that the shipped loader recomputes the digest and refuses on a mismatch — and that the
    published digest is that same one.
    """
    doc = xb.load_frozen()
    assert doc["schema"] == xb.SCHEMA
    body = _flat(_section(perf, "27.1"))
    assert xb.FROZEN_SHA256 in body, "§27.1 does not publish the digest the loader enforces"
    assert str(xb.SOURCE_PR) in body and xb.SOURCE_HEAD in body
    low = body.lower()
    assert "not re-measured" in low, "§27.1 must say the timings were not re-measured"
    assert "reused" in low, "§27.1 must say the records are reused"
    assert not re.search(r"\bwe (?:re-?)?ran\b|\bI measured\b|\bwe measured\b", body, re.I)
    for m in re.finditer(r"measured (?:here|on this branch)", low):
        prefix = low[max(0, m.start() - 40): m.start()]
        assert "not " in prefix or "never " in prefix, (
            "§27.1 claims the timings were measured here: "
            f"...{body[max(0, m.start() - 80): m.end() + 40]}..."
        )


def test_the_superseded_pr95_derivation_is_withheld_not_inherited():
    """`band`, `workloads` and `preregistration` must not be readable out of the frozen file.

    Withholding them is not tidiness. PR #95's `preregistration` block is text embedded in the
    artifact at finalize time, so its digest binds the text to itself and not to a point in time;
    inheriting it would re-import a claim of precedence that never existed.
    """
    doc = xb.load_frozen()
    for block in xb.SUPERSEDED_BLOCKS:
        assert block not in doc, f"{block!r} survived into the loaded artifact"
    raw = json.loads(xb.FROZEN_PATH.read_text(encoding="utf-8"))
    assert any(b in raw for b in xb.SUPERSEDED_BLOCKS), (
        "the frozen file no longer carries the superseded blocks, so this test proves nothing "
        "about the loader — the raw bytes were edited, which is the thing not to do"
    )


def test_no_pre_registration_is_claimed_anywhere(perf):
    """The word may appear only near the sentence that denies one exists."""
    body = _section(perf, "27")
    for m in re.finditer(r"pre-?regist\w*", body, re.I):
        window = body[max(0, m.start() - 400): m.end() + 400].lower()
        assert any(
            phrase in window
            for phrase in ("post-hoc", "no externally timestamped", "not claimed", "superseded")
        ), f"§27 uses {m.group(0)!r} without denying that one exists"
    assert xb.BAND_PROVENANCE.startswith("POST-HOC")


def test_half_range_is_not_used_for_a_maximum_deviation(perf):
    """PR #95's `null_control_half_range` was max|r−1|, which is not a half-range.

    The number is kept — it is a real property of the null control — but it may not wear a name
    that means something else. §27 is allowed to say the word only where it is naming PR #95's
    mislabel or giving the true half-range; anywhere else the old name has come back.
    """
    body = _flat(_section(perf, "27"))
    ratios = [1.051029213150834, 0.9619337960499632, 1.0380703034465]
    max_dev = max(abs(r - 1.0) for r in ratios)
    true_half_range = (max(ratios) - min(ratios)) / 2.0
    assert round(max_dev, 6) == 0.051029
    assert round(true_half_range, 6) == 0.044548
    assert max_dev != pytest.approx(true_half_range)

    occurrences = list(re.finditer(r"half-range", body, re.I))
    assert occurrences, "§27 no longer explains the mislabel it exists to correct"
    for m in occurrences:
        window = body[max(0, m.start() - 400): m.end() + 400].lower()
        assert any(
            phrase in window
            for phrase in ("mislabel", "misnames", "pr #95", "true half-range", "superseded")
        ), (
            "§27 uses 'half-range' outside the passage that corrects it: "
            f"...{body[max(0, m.start() - 80): m.end() + 80]}..."
        )
    assert f"{max_dev * 100:.2f}%" in body or f"{max_dev:.6f}" in body


# ---------------------------------------------------------------------------
# The band, and the circularity it had to lose
# ---------------------------------------------------------------------------

def test_the_band_is_derived_only_from_workloads_that_are_never_graded(summary):
    """B2, as an invariant rather than as prose: calibration ∩ subjects = ∅."""
    band = summary["band"]
    calib = set(band["calibration_workloads"])
    graded = {r["workload"] for r in summary["rows"] if r["role"] == "subject"}
    assert calib and graded
    assert calib.isdisjoint(graded)
    for row in summary["rows"]:
        if row["workload"] in calib:
            assert row["verdict"] == "CALIBRATION"
            assert row["witness_class"] == xb.NO_GQA
        else:
            assert row["witness_class"] != xb.NO_GQA
    envelope = max(
        abs(x - 1.0) for r in summary["rows"] if r["workload"] in calib for x in r["ratios"]
    )
    assert band["calibration_envelope"] == pytest.approx(envelope)
    assert band["applied"] == pytest.approx(max(xb.BAND_FLOOR, envelope))


def test_the_null_control_cannot_reach_a_speed_verdict_under_its_own_envelope(summary):
    """PR #95's specific defect: the M=1 null control's band came from the null control.

    Under the corrected band the control's widest deviation sits strictly *inside* the band, so
    `FASTER`/`SLOWER` are unreachable for it — and, crucially, that is now a **consequence** of
    controls measured elsewhere rather than an identity that would hold for any numbers at all.
    """
    band = summary["band"]["applied"]
    null_rows = [r for r in summary["rows"] if r["workload"].endswith("prefill/M1/past0")]
    assert len(null_rows) == 1
    ratios = null_rows[0]["ratios"]
    own_envelope = max(abs(r - 1.0) for r in ratios)
    assert own_envelope < band, "the null control could still define its own verdict"
    assert xb.raw_verdict(ratios, band) == "NEUTRAL"
    assert xb.raw_verdict(ratios, own_envelope) == "NEUTRAL"
    assert xb.raw_verdict([1.0 + 5 * own_envelope] * 3, 5 * own_envelope) == "NEUTRAL", (
        "a band taken from a set always neutralises that set — which is why this one is not"
    )


def test_the_document_publishes_the_applied_band_and_never_the_superseded_one(perf, summary):
    body = _section(perf, "27.3")
    assert f"{summary['band']['applied']:.6f}" in body
    if "5.10%" in body:
        idx = body.index("5.10%")
        window = body[max(0, idx - 300): idx + 300].lower()
        assert "superseded" in window or "pr #95" in window, (
            "§27.3 quotes 5.10% without marking it as PR #95's superseded value"
        )
    assert "the floor does not bind" in body.lower()
    assert summary["band"]["floor_binds"] is False


def test_the_decision_rule_is_symmetric(summary):
    """PR #95 required every repeat for FASTER but only the median for SLOWER.

    That makes a regression cheaper to claim than an improvement, which is a thumb on the scale
    in the direction of finding something. Mirroring a ratio set about 1 must mirror its verdict.
    """
    band = summary["band"]["applied"]
    faster = [1.0 + band + 0.01] * 3
    slower = [1.0 - band - 0.01] * 3
    assert xb.raw_verdict(faster, band) == "FASTER"
    assert xb.raw_verdict(slower, band) == "SLOWER"
    assert xb.raw_verdict([1.0 + band + 0.01, 1.0, 1.0 - band - 0.01], band) == "INDETERMINATE"
    assert xb.raw_verdict([1.0, 1.0, 1.0], band) == "NEUTRAL"
    for ratios in ([1.0 + band + 0.2] * 3, [1.0 + band + 0.01, 1.0 + band + 0.9, 1.0]):
        mirrored = [2.0 - r for r in ratios]
        got, back = xb.raw_verdict(ratios, band), xb.raw_verdict(mirrored, band)
        assert {"FASTER": "SLOWER", "SLOWER": "FASTER"}.get(got, got) == back, (
            f"{ratios} gives {got} but its mirror {mirrored} gives {back}"
        )
    with pytest.raises(xb.SchemaError):
        xb.raw_verdict([], band)


# ---------------------------------------------------------------------------
# The published tables, recomputed
# ---------------------------------------------------------------------------

def test_every_published_verdict_cell_is_what_the_shipped_gate_returns(perf, summary):
    """The §27.4 table, cell by cell, against `markdown_table(summarize())`.

    The whole table is regenerated and compared row by row, so a hand-edited millisecond, a
    re-ordered row or an invented workload all fail.
    """
    published = _table(_section(perf, "27.4"), "workload", "ratio (median)", "verdict")
    rendered = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in xb.markdown_table(summary).splitlines()[2:]
    ]
    assert len(published) == len(rendered) == len(summary["rows"]) == 10
    for pub, ren in zip(published, rendered):
        assert pub == ren, f"§27.4 row disagrees with the summarizer:\n  doc  {pub}\n  code {ren}"


def test_every_published_millisecond_comes_back_out_of_the_raw_samples(summary):
    """`summarize()` is not trusted either: its medians are re-derived from `speed.samples_ms`.

    `speed.median_ms` in the frozen file carries full precision while `samples_ms` is rounded to
    four decimals, so the comparison is at 5e-4 ms rather than exact — stating the tolerance is
    the point, because an exact comparison here would have to be wrong about one of the two.
    """
    doc = xb.load_frozen()
    by_cell = {(r["workload"], r["arm"], r["repeat"]): r for r in doc["records"]}
    assert len(by_cell) == 60
    assert doc["environment"]["methodology"]["warmup_per_session"] == xb.EXPECTED_WARMUPS
    keys = ("candidate_median_ms", "baseline_median_ms")
    for row in summary["rows"]:
        for rep in row["per_repeat"]:
            for arm, key in zip(("candidate", "baseline"), keys):
                rec = by_cell[(row["workload"], arm, rep["repeat"])]
                samples = rec["speed"]["samples_ms"]
                assert len(samples) == xb.EXPECTED_ITERS, "20 timed iterations were promised"
                assert rec["speed"]["n"] == xb.EXPECTED_ITERS
                middle = sorted(samples)[len(samples) // 2 - 1: len(samples) // 2 + 1]
                assert rep[key] == pytest.approx(sum(middle) / 2.0, abs=5e-4)
            assert rep["ratio"] == pytest.approx(
                rep["baseline_median_ms"] / rep["candidate_median_ms"]
            )


def test_the_sensitivity_table_is_the_shipped_sweep(perf, summary):
    """§27.5's grid must be `sensitivity()`'s, at the bands the document's columns name."""
    applied = summary["band"]["applied"]
    sweep = xb.sensitivity(summary["rows"], list(xb.SENSITIVITY_BANDS) + [applied])
    published = _table(_section(perf, "27.5"), "subject", "5%", "30%")
    label = {
        "prefill M=128": "prefill/M128/past0",
        "prefill M=32": "prefill/M32/past0",
        "prefill M=64": "prefill/M64/past0",
        "prefill M=1 (null)": "prefill/M1/past0",
        "decode past=128": "decode/M1/past128",
        "decode past=1024": "decode/M1/past1024",
    }
    columns = ["0.0500", "0.0510", "0.1000", "0.1500", f"{applied:.4f}", "0.2000",
               "0.2500", "0.3000"]
    assert len(published) == 6
    for cells in published:
        workload = next(w for w in sweep["by_workload"] if w.endswith(label[cells[0]]))
        at = sweep["by_workload"][workload]["at_band"]
        got = [c.replace("*", "").strip() for c in cells[1:]]
        assert len(got) == len(columns), f"§27.5 {cells[0]!r} has {len(got)} band columns"
        for column, have in zip(columns, got):
            want = at[column]
            assert have in (want, want[:5]), (
                f"§27.5 {cells[0]!r} at band {column}: document says {have!r}, sweep says {want!r}"
            )


def test_the_counts_line_matches_the_rows(perf, summary):
    counts = summary["counts"]
    assert counts["records"] == 60
    assert counts["admissible_records"] == 60
    assert counts["calibration"] + counts["subjects"] == counts["workloads"] == 10
    assert sum(counts["verdicts"].values()) == 10
    body = _section(perf, "27.4")
    for verdict, n in counts["verdicts"].items():
        if n:
            assert re.search(rf"{verdict}\s*\**\s*{n}\b", body), (
                f"§27.4 does not publish {verdict} {n}"
            )


# ---------------------------------------------------------------------------
# Admissibility of the raw records, checked directly rather than through the summary
# ---------------------------------------------------------------------------

def test_no_two_timed_passes_were_in_flight_at_once():
    """Process non-overlap, from the recorded spans — the claim the lock does *not* make.

    This is the strong form of the exclusivity evidence and it does not depend on the lock at
    all: 60 records, near-distinct PIDs, and no pair of `[started_at, finished_at]` intervals
    intersecting. The spans are ISO-8601 local timestamps at one-second resolution, so the
    comparison is `end <= start` on parsed datetimes and a same-second boundary is not an
    overlap — which is the honest reading of a second-resolution clock, and is stated in §27.6.
    """
    doc = xb.load_frozen()
    spans = sorted(
        (
            datetime.fromisoformat(r["started_at"]),
            datetime.fromisoformat(r["finished_at"]),
            r["pid"],
            f"{r['workload']}/{r['arm']}/r{r['repeat']}",
        )
        for r in doc["records"]
    )
    assert len(spans) == 60
    for (s0, e0, p0, w0), (s1, e1, p1, w1) in zip(spans, spans[1:]):
        assert e0 >= s0, f"{w0} finished before it started"
        assert e0 <= s1, f"overlapping timed spans: pid {p0} ({w0}) and pid {p1} ({w1})"
    assert len({s[2] for s in spans}) >= 58, "fewer distinct PIDs than the artifact claims"


def test_every_record_carries_a_cpu_oracle_that_agreed():
    """The oracle is per record, in the measuring process, and its verdict is MATCH.

    `cpu_reference_outputs_sha256` is the thing the Vulkan output was compared against, and
    `providers` naming the CPU EP beside the Vulkan one is what says the comparison happened in
    this session rather than being copied from somewhere else.
    """
    doc = xb.load_frozen()
    for r in doc["records"]:
        cell = f"{r['workload']}/{r['arm']}/r{r['repeat']}"
        eq = r.get("equivalence")
        assert eq, f"{cell} has no equivalence block"
        assert eq["verdict"] == "MATCH", cell
        assert r.get("cpu_reference_outputs_sha256"), f"{cell} has no CPU-EP reference digest"
        assert r.get("cpu_reference_ms") is not None, cell
        assert "CPUExecutionProvider" in r["providers"], cell
        assert not eq.get("secondary_divergent"), cell


def test_every_record_binds_its_arm_to_the_library_that_arm_declares():
    doc = xb.load_frozen()
    roles = set()
    for r in doc["records"]:
        arm = doc["arms"][r["arm"]]
        assert r["ep_library_sha256"] == arm["ep_library_sha256"], r["workload"]
        assert r["ep_library_role"] == r["arm"]
        roles.add(r["ep_library_sha256"])
    assert len(roles) == 2, "the two arms did not load two different libraries"
    assert doc["arms"]["candidate"]["commit"].startswith(CANDIDATE_COMMIT[:7])
    assert doc["arms"]["baseline"]["commit"].startswith(BASELINE_COMMIT[:7])


def test_the_outputs_did_not_move_under_the_timer():
    """The digest taken after the timed pass must equal the one taken before it, on all 60.

    This is the pre/post pair the artifact actually records. It is an *output* digest, not a
    pipeline digest: it says the twenty timed iterations computed the same numbers the
    equivalence check had already compared to the CPU EP, so nothing was recompiled, retuned or
    fallen back mid-run in a way that changed the result. It does not by itself prove no
    pipeline was rebuilt, and §27.1 does not say it does.
    """
    doc = xb.load_frozen()
    for r in doc["records"]:
        cell = f"{r['workload']}/{r['arm']}/r{r['repeat']}"
        w = r["path_witness"]
        assert w["present"] is True, cell
        assert w["compute_failures"] == 0, cell
        assert r["outputs_sha256_post_timing"] == r["outputs_sha256"], cell
        islands = w["viable_islands_retained"]
        expected = 27 * islands if islands else 0
        assert w["compute_calls"] == expected, (
            f"{cell}: {w['compute_calls']} compute calls over {islands} islands, expected "
            f"{expected} = (1 first run + {xb.EXPECTED_WARMUPS} warmups + {xb.EXPECTED_ITERS} "
            f"timed + 1 post-timing verification) x islands"
        )


def test_the_two_arms_computed_the_same_outputs_in_every_paired_repeat(summary):
    for row in summary["rows"]:
        assert row["cross_arm_bitwise_identical"] is True, row["workload"]
        assert row["repeats_paired"] == 3
        assert not row["refusals"]


# ---------------------------------------------------------------------------
# Language: what the lock proves, and what it does not
# ---------------------------------------------------------------------------

def test_the_lock_is_never_described_as_exclusive_gpu_ownership(perf, summary):
    """F3. The record itself lists 26 GPU application entries; "exclusive access" is not available.

    `msvcrt.locking(..., LK_NBLCK, ...)` is advisory. It excludes other *cooperating harness
    processes*. Saying more than that would be the artifact contradicting its own contents.
    """
    lang = summary["exclusivity_language"]
    proves = " ".join(lang["proves"]).lower()
    assert "cooperates by taking the same lock" in proves
    assert not re.search(r"exclusive (?:gpu|device) (?:access|ownership)", proves)
    denied = " ".join(lang["does_not_prove"]).lower()
    assert "advisory" in denied and "exclusive ownership of the gpu" in denied
    excl = summary["exclusivity"]
    assert excl["no_process_was_killed"] is True
    assert excl["state"] == "RELEASED"

    low = _section(perf, "27.6").lower()
    assert "advisory and cooperative" in low
    for phrase in ("exclusive gpu access", "sole use of the gpu",
                   "exclusive ownership of the device", "the gpu to itself"):
        assert phrase not in low, f"§27.6 claims {phrase!r}"


def test_the_published_gpu_application_count_is_the_recorded_one(perf, summary):
    """26 is not a round number chosen for effect; it is `len(...)` of a recorded list.

    And it is *entries*, not processes: four of them carry an `xN` multiplicity, so the same
    record says 26 and 30 about two different things. Publishing one number without the other is
    how "26" would quietly become a claim about how busy the device was.
    """

    def _expanded(entries):
        total = 0
        for e in entries:
            m = re.search(r"x(\d+)$", e.strip())
            total += int(m.group(1)) if m else 1
        return total

    excl = summary["exclusivity"]
    for key in ("gpu_compute_apps_at_acquire", "gpu_compute_apps_at_release"):
        assert len(excl[key]) == 26
        assert _expanded(excl[key]) == 30
    body = _section(perf, "27.6")
    assert "26 GPU compute" in body and "application entries" in body
    assert "30 processes" in body


# ---------------------------------------------------------------------------
# The compiled delta
# ---------------------------------------------------------------------------

def _git(*args) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO), *args],
            check=True, capture_output=True, text=True, timeout=180,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git unavailable or the commits are not in this clone: {exc}")


def test_the_enumerated_compiled_delta_is_the_whole_compiled_delta(summary):
    """F4. Every path that reaches the `.dll` and differs must appear in `differing`.

    The rule is applied to `git diff --name-status` rather than to a hand list: a fourth compiled
    input landing between these two commits fails this test rather than being quietly absent from
    the document, which is exactly what happened to `evidence/proof_ledger.jsonl`.
    """
    names = [
        line.split("\t", 1)[1].strip()
        for line in _git("diff", "--name-status", BASELINE_COMMIT, CANDIDATE_COMMIT).splitlines()
        if "\t" in line
    ]
    assert names, "the two commits have no diff at all, which cannot be right"

    included = _git("grep", "-n", r"include_str!\|include_bytes!", CANDIDATE_COMMIT, "--", "rust/")
    assert "evidence/proof_ledger.jsonl" in included, (
        "the ledger is no longer include_str!'d; the compiled-input list is out of date"
    )

    def compiled(path: str) -> bool:
        if path.startswith("rust/src/") and path.endswith(".rs"):
            return True
        if path.startswith("rust/shaders/"):
            return True
        return path in COMPILED_EXACT

    differing = {d["path"] for d in summary["compiled_input_delta"]["differing"]}
    assert {n for n in names if compiled(n)} == differing, (
        "the enumerated compiled delta is not the compiled delta git reports"
    )
    assert differing == {
        "rust/shaders/glsl/gqa_f16.comp",
        "rust/src/ops/attention.rs",
        "evidence/proof_ledger.jsonl",
    }


def test_the_shader_delta_really_is_one_non_comment_line():
    """The claim "one line of GLSL" is checked against the diff, not asserted."""
    diff = _git("diff", "-U0", BASELINE_COMMIT, CANDIDATE_COMMIT, "--",
                "rust/shaders/glsl/gqa_f16.comp")
    changed = [
        ln[1:].strip()
        for ln in diff.splitlines()
        if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    ]
    code = [ln for ln in changed if ln and not ln.startswith("//")]
    assert len(code) == 2, f"expected one changed line (a - and a +), got {code}"
    assert "local_size_x_id" in code[1] and "local_size_x_id" not in code[0]


def test_attribution_is_stated_as_bounded_and_not_as_isolated(perf, summary):
    """F4's language half. Two binaries that differ by 3,072 bytes did not differ in nothing."""
    delta = summary["compiled_input_delta"]
    assert delta["attribution"].startswith("BOUNDED, not isolated")
    ledger = next(d for d in delta["differing"] if d["path"].endswith("proof_ledger.jsonl"))
    assert ledger["on_the_timed_path"] is False
    assert sum(1 for d in delta["differing"] if d["on_the_timed_path"]) == 2

    body = _flat(_section(perf, "27.7"))
    low = body.lower()
    assert "bounded attribution, not isolation" in low
    assert "reachable from a timed inference" in low
    assert "is not a path proof" in low
    arms = summary["arms"]
    assert arms["candidate"]["ep_library_bytes"] - arms["baseline"]["ep_library_bytes"] == 3072
    assert "3,072 bytes" in body


def test_the_two_libraries_are_not_offered_as_the_path_evidence(summary):
    """A binary hash distinguishes two files. It does not say which code ran."""
    assert (
        summary["arms"]["candidate"]["ep_library_sha256"]
        != summary["arms"]["baseline"]["ep_library_sha256"]
    )
    for row in summary["rows"]:
        if row["role"] != "subject":
            continue
        assert row["witness_class"] == xb.DISTINGUISHED
        assert row["gqa_keys"]["candidate"] != row["gqa_keys"]["baseline"]


# ---------------------------------------------------------------------------
# Path and grid witnesses
# ---------------------------------------------------------------------------

def test_the_grid_is_labelled_inference_wherever_it_is_not_witnessed(perf, summary):
    """F7/req 9. No artifact field holds a grid, so `[32, 1, 1]` may appear only as inference."""
    inferred = 0
    for row in summary["rows"]:
        claim = xb.dispatch_grid_claim(row)
        assert "witnessed" in claim
        if claim["inferred_grid"] is not None:
            inferred += 1
            assert claim["inferred_grid"] == [32, 1, 1]
            assert claim["inference_inputs"], "an inference with no stated inputs is a claim"
            assert "without pinning it" in claim["inferred_not_witnessed"]
            assert claim["grids_equal_across_arms"] is True
    assert inferred == 3, "the three local==1 Phi workloads are the only inferred grids"

    records = json.dumps(json.loads(xb.FROZEN_PATH.read_text(encoding="utf-8"))["records"])
    for key in ('"grid"', '"dispatch_grid"', '"spec_const'):
        assert key not in records, (
            f"the frozen records now carry {key}; the inference language is stale"
        )
    assert '"local_size"' in records, (
        "the local size is a witnessed field and §27.8 says so; if it has gone, the section's "
        "distinction between what is recorded and what is inferred no longer holds"
    )

    body = _flat(_section(perf, "27.8"))
    idx = body.find("[32, 1, 1]")
    assert idx != -1, "§27.8 no longer discusses the inferred grid"
    window = body[max(0, idx - 400): idx + 400].lower()
    assert "infer" in window and "not an artifact field" in window


def test_the_witness_keys_published_are_the_keys_in_the_records(perf, summary):
    published = _table(_section(perf, "27.8"), "candidate key", "baseline key")
    by_workload = {
        row["workload"]: (
            sorted({k for g in row["gqa_keys"]["candidate"] for k in g}),
            sorted({k for g in row["gqa_keys"]["baseline"] for k in g}),
        )
        for row in summary["rows"]
    }
    seen = 0
    for cells in published:
        if "no `gqa_f16` pipeline" in cells[1]:
            for workload, (cand, base) in by_workload.items():
                if workload.startswith(("mobilenetv2", "all-MiniLM")):
                    assert cand == [] and base == []
                    seen += 1
            continue
        key = cells[1].strip("`")
        matches = [w for w, (cand, _) in by_workload.items() if cand == [key]]
        assert matches, f"§27.8 publishes candidate key {key!r}; no record carries it"
        for workload in matches:
            assert by_workload[workload][1] == ["gqa_f16:"], (
                "the baseline key is not the empty-spec key the section describes"
            )
            seen += 1
    assert seen == len(by_workload), "§27.8's table does not account for every workload"


def test_identical_witnesses_across_arms_remove_a_speed_verdict_in_production(summary):
    """B1 restated at the publication level: the gate is the shipped one, called here.

    `bench/test_crossbuild_summary.py` proves this by deleting the gate from the module source.
    This asserts the same property through the same shipped function on a real row, so the two
    files fail together rather than one covering for the other.
    """
    row = dict(next(r for r in summary["rows"] if r["raw_verdict"] == "FASTER"))
    row["witness_class"] = xb.NOT_DISTINGUISHED
    verdict = xb.gated_verdict(row, summary["band"]["applied"])
    assert verdict["verdict"] == "REFUSED"
    assert any("witness" in reason for reason in verdict["gate_reasons"])
    assert verdict["raw_verdict"] == "FASTER", "the raw reading is preserved, not erased"


# ---------------------------------------------------------------------------
# Scope of the claim
# ---------------------------------------------------------------------------

def test_the_claim_is_phi35_prefill_and_says_so(perf):
    """F8/req 8. "Real models" plural is not what the records support."""
    body = _section(perf, "27")
    head = body[: body.index("### 27.1")]
    low = head.lower()
    assert "phi-3.5 prefill" in low
    assert 'not "real models" plural' in low
    assert "no cuda number appears" in low


def test_no_cuda_or_foreign_execution_provider_number_is_published(perf, summary):
    """Issue #69's title names CUDA. This evidence does not address it, and must not appear to."""
    body = _section(perf, "27")
    for m in re.finditer(r"cuda", body, re.I):
        window = body[max(0, m.start() - 220): m.end() + 220].lower()
        if "cuda-int4" in window:
            continue  # the Foundry model file's own name contains "cuda"
        assert any(
            phrase in window
            for phrase in ("no cuda", "does not address", "not a cuda", "names cuda")
        ), f"§27 mentions CUDA without disclaiming it: …{window[:200]}…"

    blob = json.dumps(summary).lower()
    assert "cudaexecutionprovider" not in blob
    assert "dmlexecutionprovider" not in blob
    assert summary["not_a_claim_about"].startswith("any other execution provider")


def test_the_non_gqa_models_are_presented_as_controls_and_never_as_wins(perf, summary):
    body = _section(perf, "27.4")
    assert "MobileNetV2 and MiniLM are controls, not wins" in body
    for row in summary["rows"]:
        if row["workload"].startswith(("mobilenetv2", "all-MiniLM")):
            assert row["verdict"] == "CALIBRATION"
            assert row["ratios"], "the raw ratios are preserved, not discarded"


def test_p128_is_descriptive_and_m64_is_not_claimed(perf, summary):
    """Req 3's last clause: narrow honestly where the corrected envelope says to."""
    rows = {r["workload"].split("/", 1)[1]: r for r in summary["rows"]}
    p128 = rows["decode/M1/past128"]
    assert p128["ratio_median"] == pytest.approx(0.859, abs=5e-4)
    assert p128["verdict"] == "INDETERMINATE"
    assert all(r < 1.0 for r in p128["ratios"]), "all three repeats are below unity"
    assert 1.0 - max(p128["ratios"]) < summary["band"]["applied"], (
        "SLOWER would survive the applied band; the document's reasoning is stale"
    )
    m64 = rows["prefill/M64/past0"]
    assert m64["verdict"] == "INDETERMINATE"
    sweep = xb.sensitivity(summary["rows"])
    assert set(sweep["by_workload"][m64["workload"]]["at_band"].values()) == {"INDETERMINATE"}

    body = _section(perf, "27.4")
    assert "descriptive" in body.lower()
    assert "issue #96" in body


def test_m128_and_m32_survive_the_correction(summary):
    rows = {r["workload"].split("/", 1)[1]: r for r in summary["rows"]}
    m128, m32 = rows["prefill/M128/past0"], rows["prefill/M32/past0"]
    assert m128["verdict"] == "FASTER"
    assert m128["ratio_median"] == pytest.approx(2.077, abs=5e-4)
    assert min(m128["ratios"]) - 1.0 > 1.0, "M=128's break-even band is under 100%"
    assert m32["verdict"] == "FASTER"
    assert min(m32["ratios"]) == pytest.approx(1.278, abs=5e-4)
    assert min(m32["ratios"]) > 1.0 + summary["band"]["applied"]


# ---------------------------------------------------------------------------
# Counts, models, and what a public artifact may contain
# ---------------------------------------------------------------------------

def test_the_targeted_census_file_has_the_number_of_tests_the_document_says(perf):
    """F2/req 7. PR #95 cited 155 beside a file that has 14.

    Counted from the file's own AST rather than from a remembered run, and the document is
    checked for the wrong number as well as the right one — a corrected count sitting beside an
    uncorrected sentence is the defect half-fixed.
    """
    path = _REPO / "tests" / "ops" / "test_harness_census.py"
    if not path.is_file():
        pytest.skip("tests/ops/test_harness_census.py is absent")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    n = sum(
        1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )
    assert n == 14
    body = _flat(_section(perf, "27.9"))
    assert "**14** tests" in body
    for m in re.finditer(r"\b155\b", body):
        window = body[max(0, m.start() - 250): m.end() + 250].lower()
        assert any(p in window for p in ("pr #95", "superseded", "corrected", "wrong")), (
            "§27.9 carries 155 outside the sentence that corrects it: "
            f"...{body[max(0, m.start() - 80): m.end() + 80]}..."
        )
    for cells in _table(_section(perf, "27.9"), "command", "result"):
        assert "155" not in " ".join(cells), f"§27.9's counts table still publishes 155: {cells}"


def test_every_count_in_the_counts_table_names_a_command(perf):
    rows = _table(_section(perf, "27.9"), "command", "result")
    assert len(rows) >= 4
    for cells in rows:
        assert cells[0].startswith("`") and cells[0].endswith("`"), (
            f"§27.9 publishes a result with no command: {cells}"
        )


def test_minilm_is_pinned_independently_of_any_pull_request(summary):
    """Req 10. Repo, revision, file, digest, size — and no dependence on PR #83.

    Independence is checked structurally rather than by grepping for a number: the pin has to
    carry everything needed to re-fetch the file, and `bench/real_model.py` on this branch must
    still carry no MiniLM entry, so there is nothing for the control to have borrowed.
    """
    pin = xb.MODEL_PINS["all-MiniLM-L6-v2-onnx"]
    assert pin["sha256"] == "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
    assert pin["bytes"] == 90405214
    assert pin["repo"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert pin["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert pin["file"] == "onnx/model.onnx"
    assert re.fullmatch(r"[0-9a-f]{40}", pin["revision"])
    model = summary["models"]["all-MiniLM-L6-v2-onnx"]
    assert model["sha256"] == pin["sha256"] and model["bytes"] == pin["bytes"]

    real_model = _REPO / "bench" / "real_model.py"
    if real_model.is_file():
        assert "MiniLM" not in real_model.read_text(encoding="utf-8"), (
            "bench/real_model.py has grown a MiniLM entry; this control's pin is no longer the "
            "independent one and the claim of independence from PR #83 must be re-checked"
        )
    blob = json.dumps(summary) + (xb.__doc__ or "")
    assert "pull/83" not in blob, "the published artifact links to the rejected branch"
    for m in re.finditer(r"#83", blob):
        window = blob[max(0, m.start() - 300): m.end() + 60].lower()
        assert "nothing about this control depends on" in window, (
            "#83 appears in the published artifact outside the sentence that denies any "
            f"dependence on it: ...{blob[max(0, m.start() - 120): m.end() + 40]}..."
        )


def test_a_record_whose_model_digest_disagrees_with_the_pin_is_refused():
    """The pin is load-bearing: a model swapped under the same key must stop the timing.

    Records carry `model_key`, not a digest, so the identity binds through the artifact's own
    `models` block. Corrupting that block is therefore the mutation that matters — if the gate
    read the digest from nowhere, every model would be the pinned one by default.
    """
    doc = xb.load_frozen()
    rec = next(r for r in doc["records"] if r["model_key"] == "all-MiniLM-L6-v2-onnx")
    assert not xb.record_refusals(rec, models=doc["models"])

    swapped = json.loads(json.dumps(doc["models"]))
    swapped["all-MiniLM-L6-v2-onnx"]["sha256"] = "0" * 64
    why = xb.record_refusals(rec, models=swapped)
    assert any(w.startswith("model_digest_wrong") for w in why), why

    resized = json.loads(json.dumps(doc["models"]))
    resized["all-MiniLM-L6-v2-onnx"]["bytes"] = 1
    assert any(
        w.startswith("model_bytes_wrong") for w in xb.record_refusals(rec, models=resized)
    )

    unknown = json.loads(json.dumps(rec))
    unknown["model_key"] = "some-model-nobody-pinned"
    assert any(
        w.startswith("model_unpinned") for w in xb.record_refusals(unknown, models=doc["models"])
    )


def test_neither_published_json_carries_a_private_path_or_a_credential():
    """Req 10's second half. A public artifact is read by people who are not on this box."""
    patterns = [
        (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]", re.I), "an absolute Windows user path"),
        (re.compile(r"/home/[A-Za-z0-9._-]+/"), "an absolute POSIX home path"),
        (re.compile(r"[A-Za-z]:[\\/]+\.copilot", re.I), "an agent working directory"),
        (re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"), "a GitHub token"),
        (re.compile(r"(?i)\b(?:api[_-]?key|password|secret|bearer)\b\s*[\"']?\s*[:=]"),
         "a credential"),
        (re.compile(r"https://[^\s\"]*:[^\s\"@/]*@"), "a URL with inline credentials"),
    ]
    for path in (xb.FROZEN_PATH, SUMMARY_PATH):
        if not path.is_file():
            pytest.skip(f"{path.name} is not committed in this tree")
        text = path.read_text(encoding="utf-8")
        for pattern, what in patterns:
            m = pattern.search(text)
            assert m is None, f"{path.name} contains {what}: {m.group(0)[:60]!r}"


def test_the_committed_summary_is_what_the_summarizer_produces_today(summary):
    """The artifact is derived, so it must be reproducible: `--finalize` is not a hand edit."""
    if not SUMMARY_PATH.is_file():
        pytest.skip("the derived summary is not committed in this tree")
    committed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert committed["schema"] == xb.SUMMARY_SCHEMA
    assert committed == json.loads(json.dumps(summary)), (
        "bench/results/real_model_crossbuild_gqa_landing_v2.json is stale — regenerate it with "
        "`python bench/crossbuild_summary.py --finalize`"
    )


def test_the_design_note_no_longer_calls_the_specialised_pipeline_the_same_pipeline():
    """§8.13 said "the same pipeline behaviour as before issue #56"; the cache key differs."""
    if not DESIGN.is_file():
        pytest.skip("docs/DESIGN.md is absent")
    text = DESIGN.read_text(encoding="utf-8")
    assert "the same geometry, the same grid, the same pipeline behaviour" not in text
    idx = text.find("### 8.13")
    assert idx != -1
    body = text[idx: idx + 20000]
    assert "`gqa_f16:1`" in body and "derived, not measured" in body
    assert "issue #96" in body
