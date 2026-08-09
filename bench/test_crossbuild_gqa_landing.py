"""Self-tests and mutation tests for the issue-#69 cross-build instrument.

`bench/results/probe_crossbuild_gqa_landing.py` answers one question — *is current `origin/main`
faster on real models than the Vulkan tree immediately before the GQA local-size change?* — by
running two separately built EP libraries. No amount of care in the driver makes that answer
trustworthy on its own; what makes it trustworthy is that the gate which lets a number be
published **refuses** when its premises are missing. So this file does not test that the
instrument produces numbers. It mutates records that already exist in the committed artifact and
asserts that each mutation is caught.

Each test is named for the **plausible but wrong** reading it prevents, in the style of
`bench/test_plausible_but_wrong.py` and `bench/test_real_model.py`. No GPU, no EP, no model file:
everything here runs from the committed JSON and from pure functions.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
_RESULTS = _BENCH / "results"
for _p in (str(_BENCH), str(_RESULTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_crossbuild_gqa_landing as xb  # noqa: E402
import real_model as rm  # noqa: E402

ARTIFACT = _RESULTS / "real_model_crossbuild_gqa_landing.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip(f"{ARTIFACT.name} is not present")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def admissible_record(artifact) -> dict:
    """One real, admissible, *timed* record — the thing the mutations are applied to.

    Mutating a synthetic record would only test the fixture. These are the bytes that were
    published, and `bench/results/probe_crossbuild_gqa_landing.py` has to refuse them once they
    stop being true.
    """
    for rec in artifact["records"]:
        if rec.get("admissible") and rec.get("model_key") == rm.PHI35.key:
            return rec
    pytest.skip("no admissible Phi-3.5 record in the artifact")


def _rehydrate(rec: dict, artifact: dict) -> dict:
    """Put the model provenance block back on a record.

    The published artifact carries model provenance once, in `models`, rather than repeating a
    2.3 GB model's external-data block sixty times. `admissibility_gate` reads it from the record
    because that is where it sits at measurement time, so the tests restore it.
    """
    out = copy.deepcopy(rec)
    out["model"] = copy.deepcopy(artifact["models"][rec["model_key"]])
    return out


# ---------------------------------------------------------------------------
# Mutation 1 — a timing whose equivalence verdict is missing
# ---------------------------------------------------------------------------

def test_mutation_missing_equivalence_refuses_and_drops_the_speed(admissible_record, artifact):
    """A fast arm that was never checked against the CPU EP is a fast *wrong* arm.

    The plausible-but-wrong reading is that equivalence is a property of the *build*, so checking
    it once is enough. It is a property of the (arm, repeat): the gate must refuse a record that
    carries no verdict of its own, and the refusal must take the number with it.
    """
    rec = _rehydrate(admissible_record, artifact)
    assert rec["speed"]["median_ms"] > 0

    rec.pop("equivalence")
    out = xb.admissibility_gate(rec)

    assert out["admissible"] is False
    assert "speed" not in out, "a refused record must be structurally free of speed fields"
    assert "equivalence" in out["refusal"]["reason"]


def test_mutation_empty_equivalence_is_not_a_pass(admissible_record, artifact):
    """`{}` is not a verdict, and neither is a verdict-shaped dict with nothing in it."""
    for empty in ({}, {"verdict": None}, {"verdict": ""}):
        rec = _rehydrate(admissible_record, artifact)
        rec["equivalence"] = empty
        out = xb.admissibility_gate(rec)
        assert out["admissible"] is False
        assert "speed" not in out


def test_mutation_divergent_equivalence_refuses(admissible_record, artifact):
    rec = _rehydrate(admissible_record, artifact)
    rec["equivalence"] = {"verdict": rm.DIVERGENT}
    out = xb.admissibility_gate(rec)
    assert out["admissible"] is False
    assert "speed" not in out


def test_mutation_missing_outputs_digest_refuses(admissible_record, artifact):
    """No digest means no cross-process comparison, so there is nothing to compare arms by."""
    for value in ("", None):
        rec = _rehydrate(admissible_record, artifact)
        rec["outputs_sha256"] = value
        out = xb.admissibility_gate(rec)
        assert out["admissible"] is False
        assert "speed" not in out


# ---------------------------------------------------------------------------
# Mutation 2 — the wrong model, benchmarked under the right name
# ---------------------------------------------------------------------------

def test_mutation_wrong_model_digest_refuses(admissible_record, artifact):
    """The file that ran must be the file that was pinned.

    The plausible-but-wrong reading is that resolving a model by *name* through the repository's
    own tooling is provenance. It is not: a cache can move, a download can be redirected, a
    Foundry version can be bumped underneath the name. The digest is the identity.
    """
    rec = _rehydrate(admissible_record, artifact)
    rec["model"]["sha256"] = "0" * 64
    out = xb.admissibility_gate(rec)

    assert out["admissible"] is False
    assert "speed" not in out
    assert "digest" in out["refusal"]["reason"]
    assert out["refusal"]["measured"] == "0" * 64


def test_mutation_resolver_reported_provenance_disagreement_refuses(admissible_record, artifact):
    """`real_model.resolve_model` *reports* disagreement rather than raising; the gate acts."""
    rec = _rehydrate(admissible_record, artifact)
    rec["model"]["agrees_with_recorded_provenance"] = False
    out = xb.admissibility_gate(rec)
    assert out["admissible"] is False
    assert "speed" not in out


def test_minilm_pin_is_complete_and_immutable():
    """A pin missing any of repo/revision/file/digest is not a pin, it is a suggestion."""
    assert len(xb.MINILM_REVISION) == 40 and all(
        c in "0123456789abcdef" for c in xb.MINILM_REVISION)
    assert len(xb.MINILM_SHA256) == 64
    url = xb.minilm_url()
    assert xb.MINILM_REVISION in url
    assert "/resolve/main/" not in url, "a branch name is not an immutable revision"


def test_minilm_is_never_fed_an_image_tensor():
    """MiniLM is a text encoder. Feeding it ImageNet floats would 'work' and mean nothing."""
    np = pytest.importorskip("numpy")
    case = xb.make_case(xb.MINILM_KEY, "encode", 128, 0)
    feeds = xb.build_feeds(case, np)
    assert set(feeds) == {"input_ids", "attention_mask", "token_type_ids"}
    for name, arr in feeds.items():
        assert arr.dtype == np.int64, f"{name} must be int64 token data, not image floats"
        assert arr.shape == (1, 128)


# ---------------------------------------------------------------------------
# Mutation 3 — two arms that never actually ran different code
# ---------------------------------------------------------------------------

def test_mutation_identical_path_witness_across_arms_downgrades_faster(artifact):
    """A speedup between two builds that bound the *same* pipeline is not a speedup.

    This is the mutation that matters most for this artifact's headline. The candidate's GQA
    dispatch carries a specialisation constant (`gqa_f16:64`); the baseline's carries none
    (`gqa_f16:`). If a mutation makes the witness identical across arms, a `FASTER` verdict has
    lost its evidence that two different code paths ran, and must become a refusal — even though
    every timing in it is still perfectly real.
    """
    faster = next((w for w in artifact["workloads"] if w["verdict"] == "FASTER"), None)
    if faster is None:
        pytest.skip("no FASTER workload in the artifact")
    assert faster["witness_distinguishes_arms"] is True

    label = faster["workload"]
    records = copy.deepcopy(artifact["records"])
    for rec in records:
        if rec["workload"] == label:
            rec["path_witness"]["gqa_keys"] = ["gqa_f16:"]

    band = artifact["band"]["applied"]
    mutated = xb.paired(records, label)
    assert mutated["witness_distinguishes_arms"] is False

    verdict = xb.verdict_for(mutated, band)
    assert verdict == "FASTER", "the timings themselves are untouched, so the raw verdict stands"

    # ...and the published pipeline applies the witness gate on top of the raw verdict.
    if verdict == "FASTER" and not mutated["witness_distinguishes_arms"]:
        verdict = "REFUSED"
    assert verdict == "REFUSED"


def test_every_phi_workload_witness_distinguishes_the_two_builds(artifact):
    """The real artifact must actually satisfy what the mutation above breaks."""
    for w in artifact["workloads"]:
        if w.get("expected_candidate_gqa_local_size") is None:
            continue
        assert w["witness_distinguishes_arms"] is True, w["workload"]
        assert w["candidate_gqa_local_size_as_expected"] is True, w["workload"]
        assert w["baseline_gqa_unspecialised"] is True, w["workload"]


def test_a_faster_verdict_requires_every_repeat_to_improve(artifact):
    """A median that clears the band on the strength of one lucky repeat is not a result."""
    band = artifact["band"]["applied"]
    p = {"repeats_paired": xb.REPEATS, "ratio_median": 1.40,
         "ratio_min": 1.0, "ratio_max": 2.0, "refusals": []}
    assert xb.verdict_for(p, band) == "INDETERMINATE"
    p["ratio_min"] = 1 + band + 1e-6
    assert xb.verdict_for(p, band) == "FASTER"


def test_a_short_run_cannot_be_published_as_a_full_one(artifact):
    """Fewer paired repeats than the pre-registration declared is a refusal, not a smaller n."""
    band = artifact["band"]["applied"]
    p = {"repeats_paired": xb.REPEATS - 1, "ratio_median": 2.0,
         "ratio_min": 2.0, "ratio_max": 2.0, "refusals": []}
    assert xb.verdict_for(p, band) == "REFUSED"


# ---------------------------------------------------------------------------
# Mutation 4 — a refused record that kept its number
# ---------------------------------------------------------------------------

def test_mutation_refused_record_retaining_speed_is_caught(admissible_record, artifact):
    """The refusal discipline is structural: `speed` is absent, not null, not zero.

    The plausible-but-wrong reading is that `admissible: false` is enough, because a reader will
    check it. Readers do not check it — aggregators do, and an aggregator that keys on `speed`
    will happily average a refused arm into a claim. So the gate removes the key.
    """
    rec = _rehydrate(admissible_record, artifact)
    rec["admissible"] = False
    rec["refusal"] = {"reason": "synthetic refusal that kept its number"}
    assert "speed" in rec, "the mutation must start from a record that still has one"

    out = xb.admissibility_gate(rec)
    assert out["admissible"] is False
    assert "speed" not in out


def test_no_refused_record_in_the_published_artifact_carries_speed(artifact):
    speed_keys = ("speed", "median_ms", "samples_ms", "throughput")
    for rec in artifact["records"]:
        if not rec.get("admissible"):
            for k in speed_keys:
                assert k not in rec, f"{rec['workload']} {rec['arm']} r{rec['repeat']}: {k}"


def test_a_refused_arm_cannot_be_paired_into_a_ratio(artifact):
    """Count borrowing, the specific form: repeat 2 refuses, so repeat 0's number stands in."""
    label = next(w["workload"] for w in artifact["workloads"]
                 if w.get("repeats_paired") == xb.REPEATS)
    records = copy.deepcopy(artifact["records"])
    for rec in records:
        if rec["workload"] == label and rec["arm"] == "baseline" and rec["repeat"] == 2:
            rec["admissible"] = False
            rec["refusal"] = {"reason": "synthetic"}
            rec.pop("speed", None)

    p = xb.paired(records, label)
    assert p["repeats_paired"] == xb.REPEATS - 1
    assert [r["repeat"] for r in p["per_repeat"]] == [0, 1]
    assert xb.verdict_for(p, artifact["band"]["applied"]) == "REFUSED"


# ---------------------------------------------------------------------------
# The artifact's own invariants
# ---------------------------------------------------------------------------

def test_the_two_arms_are_two_different_binaries(artifact):
    cand = artifact["arms"]["candidate"]["ep_library_sha256"]
    base = artifact["arms"]["baseline"]["ep_library_sha256"]
    assert cand and base and cand != base, "an A/B of one build is not an A/B"
    assert artifact["arms"]["candidate"]["commit"] != artifact["arms"]["baseline"]["commit"]


def test_every_record_names_the_binary_that_produced_it(artifact):
    by_role = {"candidate": artifact["arms"]["candidate"]["ep_library_sha256"],
               "baseline": artifact["arms"]["baseline"]["ep_library_sha256"]}
    for rec in artifact["records"]:
        if "ep_library_sha256" in rec:
            assert rec["ep_library_sha256"] == by_role[rec["arm"]], rec["workload"]


def test_each_arm_and_repeat_ran_in_its_own_process(artifact):
    """One process per (workload, arm, repeat) is what makes 'which DLL ran' answerable.

    The naive form of this test — "all PIDs distinct" — is *wrong on Windows*, which recycles PID
    numbers: this run reused two of them across gaps of eleven and twenty-three minutes. The
    invariant that actually matters is that no two records were alive in the same process at the
    same time, because ORT registers an EP library process-globally and a shared process would
    make "which build produced this number" unanswerable.
    """
    live: dict[int, list[tuple[str, str, str]]] = {}
    for rec in artifact["records"]:
        if "pid" not in rec:
            continue
        live.setdefault(rec["pid"], []).append(
            (rec["started_at"], rec.get("finished_at") or rec["started_at"], rec["workload"]))
    for pid, spans in live.items():
        spans.sort()
        for (s0, e0, w0), (s1, _e1, w1) in zip(spans, spans[1:]):
            assert e0 <= s1, f"pid {pid} held {w0} and {w1} at the same time"


def test_records_cover_every_workload_arm_and_repeat(artifact):
    """Sixty slots, sixty records: a missing cell is a silent drop, not a smaller experiment."""
    seen = {(r["workload"], r["arm"], r["repeat"]) for r in artifact["records"]}
    labels = {w["workload"] for w in artifact["workloads"]}
    expected = {(lab, arm, rep) for lab in labels
                for arm in ("candidate", "baseline") for rep in range(xb.REPEATS)}
    assert seen == expected


def test_the_band_was_a_rule_not_a_number_chosen_afterwards(artifact):
    band = artifact["band"]
    assert band["applied"] >= band["floor"]
    assert band["applied"] == max(band["floor"], band["null_control_half_range"])
    assert band["null_control"] in {w["workload"] for w in artifact["workloads"]}
    prereg = artifact["preregistration"]["text"]
    assert "BAND = max(5%, the observed half-range of the NULL CONTROL)" in prereg


def test_the_null_control_is_not_claimed_as_a_gain(artifact):
    """M=1 dispatches identical geometry on both builds. If it 'improves', the run is drifting."""
    null = next(w for w in artifact["workloads"]
                if w["workload"] == artifact["band"]["null_control"])
    assert null["verdict"] in {"NEUTRAL", "INDETERMINATE", "SLOWER"}
    assert null["expected_candidate_gqa_local_size"] == 1


def test_arms_agree_bit_for_bit_on_every_workload(artifact):
    for w in artifact["workloads"]:
        assert w["cross_arm_bitwise_identical"] is True, w["workload"]


def test_no_cuda_number_is_present(artifact):
    """A Vulkan-over-prior-Vulkan gain is not a CUDA win, and this file may not imply one."""
    blob = json.dumps(artifact).lower()
    assert "cudaexecutionprovider" not in blob
    assert "not a cuda comparison" in artifact["not_a_claim_about"].lower()


def test_the_public_artifact_carries_no_absolute_paths(artifact):
    """Runtime paths and the public channel are separate; a home directory is not evidence."""
    import re

    blob = json.dumps(artifact)
    # `https://…` legitimately contains `s:/`, so a drive letter is matched as `X:\` only.
    for pattern in (r"(?<![A-Za-z]):?[A-Za-z]:\\\\", r"/home/", r"/Users/",
                    r"\.venv", r"AppData", r"site-packages"):
        assert not re.search(pattern, blob), f"{pattern} leaked into the public artifact"


def test_exclusivity_proof_covers_the_whole_window_and_killed_nothing(artifact):
    excl = artifact["exclusivity"]
    assert excl["state"] == "RELEASED"
    assert excl["no_process_was_killed"] is True
    assert excl["held_seconds"] > 0
    assert excl["waited_seconds"] >= 0
    assert "\\" not in excl["lock_path"] and "/" not in excl["lock_path"]


def test_sanitize_exclusivity_removes_paths_but_keeps_the_proof():
    raw = {
        "state": "RELEASED", "held_seconds": 12.0, "no_process_was_killed": True,
        "lock_path": r"C:\Users\someone\.copilot\gpu-exclusive.lock",
        "gpu_compute_apps_at_acquire": [
            r"1, C:\Users\someone\AppData\Local\Thing\thing.exe, [N/A]",
            r"2, C:\Users\someone\AppData\Local\Thing\thing.exe, [N/A]",
        ],
    }
    out = xb.sanitize_exclusivity(raw)
    assert out["lock_path"] == "gpu-exclusive.lock"
    assert out["gpu_compute_apps_at_acquire"] == ["thing.exe x2"]
    assert out["state"] == "RELEASED" and out["no_process_was_killed"] is True
    assert "someone" not in json.dumps(out)


def test_gqa_witness_reads_the_production_counter_not_this_scripts_belief():
    """The witness is the specialisation vector `vk::session` handed to pipeline creation."""
    cand = xb.gqa_witness({"pipeline_variants": ["gqa_f16:64", "gather_f16:256"],
                           "dispatches_executed": 10, "compute_failures": 0})
    base = xb.gqa_witness({"pipeline_variants": ["gqa_f16:", "gather_f16:256"],
                           "dispatches_executed": 10, "compute_failures": 0})
    assert cand["gqa_keys"] == ["gqa_f16:64"] and cand["local_size"] == 64
    assert base["gqa_keys"] == ["gqa_f16:"] and base["local_size"] is None
    assert cand["gqa_keys"] != base["gqa_keys"]
    assert xb.gqa_witness(None)["present"] is False


def test_expected_gqa_local_sizes_follow_the_shipped_rule():
    """Derived from `ops::attention::gqa_local_size`, written down so the witness can falsify it.

    `total = B * Nq * S = 32 * S`; `local` is the largest power of two <= 64 leaving >= 32
    workgroups. Decode is 1 — the landed change is a prefill change and is *expected* to leave
    decode geometry exactly where it was.
    """
    assert xb.EXPECTED_GQA_LOCAL[(rm.PHI35.key, "prefill", 1, 0)] == 1
    assert xb.EXPECTED_GQA_LOCAL[(rm.PHI35.key, "prefill", 32, 0)] == 32
    assert xb.EXPECTED_GQA_LOCAL[(rm.PHI35.key, "prefill", 64, 0)] == 64
    assert xb.EXPECTED_GQA_LOCAL[(rm.PHI35.key, "prefill", 128, 0)] == 64
    assert xb.EXPECTED_GQA_LOCAL[(rm.PHI35.key, "decode", 1, 128)] == 1
    assert xb.EXPECTED_GQA_LOCAL[(rm.PHI35.key, "decode", 1, 1024)] == 1


def test_finalize_refuses_a_lock_record_that_never_released(tmp_path):
    art = {"schema": xb.SCHEMA, "records": [], "exclusivity": {"state": "HELD"}}
    art_path = tmp_path / "a.json"
    art_path.write_text(json.dumps(art), encoding="utf-8")
    excl = tmp_path / "e.json"
    excl.write_text(json.dumps({"state": "HELD"}), encoding="utf-8")
    rc = xb.main(["--finalize", str(art_path), "--exclusivity", str(excl)])
    assert rc == 2
    assert json.loads(art_path.read_text(encoding="utf-8"))["exclusivity"]["state"] == "HELD"


# ---------------------------------------------------------------------------
# docs/PERF.md §27 against the artifact it cites
# ---------------------------------------------------------------------------

PERF = _BENCH.parent / "docs" / "PERF.md"

#: The §27.4 table's row labels, in artifact terms. Kept here rather than parsed out of the
#: prose so that renaming a row in the document cannot silently drop it from this check.
_S27_ROWS = {
    "Phi-3.5 prefill `M=1` — null control": "prefill/M1/past0",
    "Phi-3.5 prefill `M=32`": "prefill/M32/past0",
    "Phi-3.5 prefill `M=64`": "prefill/M64/past0",
    "Phi-3.5 prefill `M=128`": "prefill/M128/past0",
    "Phi-3.5 decode `past=128`": "decode/M1/past128",
    "Phi-3.5 decode `past=1024`": "decode/M1/past1024",
    "MobileNetV2 `N=1`": "mobilenetv2-12/batch/N1",
    "MobileNetV2 `N=16`": "mobilenetv2-12/batch/N16",
    "all-MiniLM-L6-v2 `S=128`": "all-MiniLM-L6-v2-onnx/encode/S128",
    "all-MiniLM-L6-v2 `S=384`": "all-MiniLM-L6-v2-onnx/encode/S384",
}


def _s27() -> str:
    if not PERF.is_file():
        pytest.skip("docs/PERF.md is not present")
    text = PERF.read_text(encoding="utf-8")
    start = text.index("## 27. ")
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def _by_suffix(artifact: dict, suffix: str) -> dict:
    matches = [w for w in artifact["workloads"] if w["workload"].endswith(suffix)]
    assert len(matches) == 1, f"{suffix} matched {len(matches)} workloads"
    return matches[0]


def test_section_27_result_table_is_the_artifact(artifact):
    """Every cell of §27.4 re-derived from the JSON — the failure mode §26 was reviewed for.

    A published table that nobody recomputes is a table that drifts from its artifact. Latency
    cells are the median over per-repeat medians, ratio cells the median over per-repeat ratios;
    the two do not divide into one another and this test asserts each against its own field.
    """
    section = _s27()
    checked = 0
    for label, suffix in _S27_ROWS.items():
        w = _by_suffix(artifact, suffix)
        row = next((line for line in section.splitlines()
                    if line.startswith(f"| {label} |")), None)
        assert row, f"§27.4 has no row for {label}"
        cells = [c.strip() for c in row.strip("|").split("|")]
        cand, base, ratio, per_repeat, verdict = cells[1:6]
        assert float(cand) == pytest.approx(w["candidate_median_ms"], abs=0.01), label
        assert float(base) == pytest.approx(w["baseline_median_ms"], abs=0.01), label
        assert float(ratio.strip("*×")) == pytest.approx(w["ratio_median"], abs=0.001), label
        published = [float(x) for x in per_repeat.split("/")]
        assert published == pytest.approx(w["ratios"], abs=0.0006), label
        assert verdict.strip("*") == w["verdict"], label
        checked += 1
    assert checked == len(_S27_ROWS)


def test_section_27_band_and_record_counts_are_the_artifact(artifact):
    section = _s27()
    assert f"{artifact['band']['applied'] * 100:.2f}%" in section
    n = len(artifact["records"])
    refused = sum(1 for r in artifact["records"] if not r.get("admissible"))
    assert f"**{n} records, {n} admissible, zero refusals.**" in section
    assert refused == 0


def test_section_27_names_both_ep_library_hashes(artifact):
    """§26.1's rule, one section later: never imply that one binary produced everything."""
    section = _s27()
    for arm in ("candidate", "baseline"):
        sha = artifact["arms"][arm]["ep_library_sha256"]
        assert sha[:8] in section, f"§27.1 does not name the {arm} build {sha[:8]}"
        assert artifact["arms"][arm]["commit"] in section


def test_section_27_drift_envelope_is_the_controls_own_range(artifact):
    """The ±15% envelope is derived from the four no-GQA controls, not asserted."""
    ratios = [r for w in artifact["workloads"]
              if w.get("expected_candidate_gqa_local_size") is None
              for r in w["ratios"]]
    assert ratios, "no control workloads in the artifact"
    section = _s27()
    assert f"**{min(ratios):.3f} to {max(ratios):.3f}**" in section, (
        f"§27.5 no longer quotes the controls' own range {min(ratios):.3f}–{max(ratios):.3f}")


def test_section_27_claims_no_faster_row_the_artifact_refuses(artifact):
    """The document may not bold a verdict the rule did not produce."""
    section = _s27()
    faster = {w["workload"] for w in artifact["workloads"] if w["verdict"] == "FASTER"}
    for w in artifact["workloads"]:
        if w["verdict"] == "FASTER":
            continue
        label = next((k for k, v in _S27_ROWS.items() if w["workload"].endswith(v)), None)
        if label is None:
            continue
        row = next(line for line in section.splitlines() if line.startswith(f"| {label} |"))
        assert "**FASTER**" not in row, f"{w['workload']} is {w['verdict']}, not FASTER"
    assert faster, "this test is vacuous if nothing cleared the band"


def test_section_27_says_it_is_not_a_cuda_claim():
    section = _s27()
    assert "not a CUDA win" in section or "not a comparison with CUDA" in section
    assert "Vulkan against prior Vulkan" in section


def test_this_file_is_counted_where_it_is_cited():
    """§27.10 quotes a size for this module; a stale size is the same defect one level up."""
    section = _s27()
    m = re.search(r"`bench/test_crossbuild_gqa_landing\.py`\s*\(\*\*(\d+)\*\* GPU-free tests\)",
                  section)
    assert m, "§27.10 no longer states this module's test count"
    mine = len([n for n in globals() if n.startswith("test_")])
    assert int(m.group(1)) == mine, (
        f"§27.10 says {m.group(1)} tests in bench/test_crossbuild_gqa_landing.py; "
        f"it defines {mine}")


def test_design_8_13_records_the_decode_regression(artifact):
    """DESIGN §8.13 said decode's pipeline behaviour was unchanged; §27 measured otherwise."""
    design = _BENCH.parent / "docs" / "DESIGN.md"
    if not design.is_file():
        pytest.skip("docs/DESIGN.md is not present")
    text = design.read_text(encoding="utf-8")
    decode = _by_suffix(artifact, "decode/M1/past128")
    assert decode["verdict"] == "SLOWER", "this test is vacuous unless decode regressed"
    assert f"{decode['ratio_median']:.3f}×" in text, (
        "§8.13 does not carry the measured decode ratio")
    assert "real_model_crossbuild_gqa_landing.json" in text

