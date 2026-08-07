"""Results must not outlive the code they measured.

Issue #69's replacement branch sits on top of PR #72, which changed what the Vulkan
attention path *does*.  Every Vulkan number measured before that change describes a
different program.  The instruction that produced this file was blunt about it: do not
carry pre-#72 baseline rankings forward as current results.

A prose disclaimer would not survive the next person in a hurry, so the rule is
mechanical here.  Each committed suite record pins the digests of the sources that
decide attention behaviour.  If the tree's copy of those files no longer matches what a
record was measured against, that record is stale *by construction* and this module
says so by name.

The screen is deliberately one-directional.  It cannot tell you a result is still
correct -- only that it cannot possibly be, because the code moved underneath it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import cuda_competition as cc  # noqa: E402
from _polarity import omits, records  # noqa: E402

CUDA69 = HERE / "results" / "_cuda69"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def committed_suites() -> "list[Path]":
    """Every committed record under _cuda69 that publishes Vulkan timings.

    Deliberately not keyed on one filename or one schema.  The pre-GQA artifacts this
    branch replaced were called ``baseline_fixed.*``; a screen that named them would
    have gone quiet the moment they were renamed, which is the failure mode it exists
    to prevent.
    """
    out = []
    for p in sorted(CUDA69.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        schema = str(d.get("schema", ""))
        if schema.startswith(("cuda_competition/", "cuda_profile/")):
            out.append(p)
    return out


def _superseded(d: dict) -> bool:
    return bool(d.get("superseded_by"))


# --- the instrument itself, falsified -------------------------------------------


def test_ep_provenance_pins_the_sources_that_decide_attention():
    prov = cc.ep_provenance()
    assert set(prov["pinned_sources"]) == set(cc.EP_PINNED_SOURCES)
    for rel, digest in prov["pinned_sources"].items():
        assert digest == _sha256(REPO / rel), rel


def test_ep_provenance_records_the_binary_that_actually_ran(monkeypatch, tmp_path):
    lib = tmp_path / "fake_ep.dll"
    lib.write_bytes(b"not really a dll, but it is the bytes that were loaded")
    monkeypatch.setenv(cc.EP_LIB_ENV, str(lib))
    prov = cc.ep_provenance()
    assert prov["lib_sha256"] == _sha256(lib)
    assert prov["lib_bytes"] == lib.stat().st_size


def test_ep_provenance_omits_a_library_digest_it_could_not_take(monkeypatch):
    """An absent EP is reported as absent, never as a digest of nothing."""
    monkeypatch.delenv(cc.EP_LIB_ENV, raising=False)
    why = omits(cc.ep_provenance(), "lib_sha256",
                reason_field="lib_unavailable_because")
    assert cc.EP_LIB_ENV in why


def test_ep_provenance_records_the_digest_when_the_library_is_present(monkeypatch, tmp_path):
    lib = tmp_path / "ep.dll"
    lib.write_bytes(b"\x00\x01\x02")
    monkeypatch.setenv(cc.EP_LIB_ENV, str(lib))
    records(cc.ep_provenance(), "lib_sha256", _sha256(lib))


def test_ep_provenance_notices_when_a_pinned_source_is_dirty(monkeypatch):
    """The narrow dirtiness question must actually consult the pinned set."""
    seen: "list[list[str]]" = []
    real = subprocess.run

    def fake(cmd, *a, **kw):
        if isinstance(cmd, list) and "--porcelain" in cmd and "--" in cmd:
            seen.append(cmd[cmd.index("--") + 1:])
        return real(cmd, *a, **kw)

    monkeypatch.setattr(cc.subprocess, "run", fake)
    cc.ep_provenance()
    assert seen and set(seen[0]) == set(cc.EP_PINNED_SOURCES)


def test_a_digest_change_is_detected(tmp_path):
    """The comparison is not vacuous: perturb a byte and it must disagree."""
    a = tmp_path / "a"
    a.write_bytes(b"kernel source v1")
    first = _sha256(a)
    a.write_bytes(b"kernel source v2")
    assert _sha256(a) != first


# --- the screen over committed evidence -----------------------------------------


def test_there_is_at_least_one_committed_suite_to_screen():
    """Guards the screen below from passing by finding nothing."""
    assert committed_suites(), (
        f"no cuda_competition record under {CUDA69}; the staleness screen would "
        f"be vacuous")


@pytest.mark.parametrize("suite", committed_suites(), ids=lambda p: p.name)
def test_committed_suite_records_the_ep_build_it_measured(suite: Path):
    d = json.loads(suite.read_text(encoding="utf-8"))
    prov = d.get("ep_provenance")
    assert prov, (
        f"{suite.name} publishes Vulkan rankings but records nothing about which EP "
        f"build produced them. Re-run it with the current harness; an unattributed "
        f"ranking cannot be checked for staleness and must not be committed.")
    if _superseded(d):
        # A retired record may legitimately have no digests -- it predates the field.
        # What it may NOT do is stay silent about that, or it is indistinguishable from
        # a current record whose provenance was dropped.
        assert prov.get("lib_unavailable_because") or prov.get("pinned_sources"), (
            f"{suite.name} is superseded and carries no provenance and no explanation "
            f"for its absence")
        return
    assert prov.get("pinned_sources"), f"{suite.name}: ep_provenance has no pinned_sources"


@pytest.mark.parametrize("suite", committed_suites(), ids=lambda p: p.name)
def test_committed_suite_is_not_stale_against_this_tree(suite: Path):
    """The load-bearing one: committed numbers must describe the code that is here.

    A record that has *declared itself superseded* is exempt, because it is no longer
    claiming to describe this tree.  That exemption is not a loophole: the tests below
    require the superseding record to exist and require the label to be honest.
    """
    d = json.loads(suite.read_text(encoding="utf-8"))
    if _superseded(d):
        pytest.skip(f"{suite.name} declares itself superseded by {d['superseded_by']}")
    prov = d.get("ep_provenance") or {}
    drifted = []
    for rel, recorded in (prov.get("pinned_sources") or {}).items():
        current = _sha256(REPO / rel) if (REPO / rel).is_file() else None
        if recorded != current:
            drifted.append(f"  {rel}\n    measured against {recorded}\n    tree has     {current}")
    assert not drifted, (
        f"{suite.name} was measured against a different Vulkan attention path than the "
        f"one in this tree, so its rankings are stale:\n" + "\n".join(drifted) +
        "\n\nRe-run the suite against the current build. Editing the recorded digest "
        "would be forging provenance, and relabelling the file would leave the stale "
        "numbers in place under a new name -- neither is the remedy.")


# --- 'superseded' is a state with obligations, not an escape hatch ----------------

def test_a_superseded_record_names_a_successor_that_exists():
    """Otherwise `superseded_by` is a way to silence the screen with a string."""
    for suite in committed_suites():
        d = json.loads(suite.read_text(encoding="utf-8"))
        if not _superseded(d):
            continue
        successor = CUDA69 / d["superseded_by"]
        assert successor.exists(), (
            f"{suite.name} claims to be superseded by {d['superseded_by']}, which is not "
            f"committed. A record cannot be retired in favour of nothing.")
        assert successor.resolve() != suite.resolve(), (
            f"{suite.name} names itself as its own successor")


def test_a_superseded_record_says_why():
    for suite in committed_suites():
        d = json.loads(suite.read_text(encoding="utf-8"))
        if not _superseded(d):
            continue
        why = d.get("superseded_because")
        assert isinstance(why, str) and why.strip(), (
            f"{suite.name} is marked superseded with no reason; a reader cannot tell an "
            f"obsolete measurement from a mislabelled one")


def test_the_successor_is_not_itself_superseded():
    """A chain of retired records leaves no current result at all."""
    current = []
    for suite in committed_suites():
        d = json.loads(suite.read_text(encoding="utf-8"))
        if str(d.get("schema", "")).startswith("cuda_competition/") and not _superseded(d):
            current.append(suite.name)
    assert current, (
        "every committed competition suite is marked superseded, so the tree publishes no "
        "current ranking at all. Retiring the last one is not a way to become compliant.")


def test_a_superseded_record_is_marked_in_its_prose_too(): 
    """The JSON label is invisible to anyone reading the report."""
    for suite in committed_suites():
        d = json.loads(suite.read_text(encoding="utf-8"))
        if not _superseded(d):
            continue
        for ext in (".md", ".log"):
            side = suite.with_suffix(ext)
            if not side.exists():
                continue
            head = side.read_text(encoding="utf-8", errors="replace")[:600].upper()
            assert "SUPERSEDED" in head, (
                f"{side.name} carries retired numbers with no warning in the first screen "
                f"of text; the JSON marker does not reach a human reading the report")


@pytest.mark.parametrize("suite", committed_suites(), ids=lambda p: p.name)
def test_committed_suite_was_measured_on_clean_pinned_sources(suite: Path):
    d = json.loads(suite.read_text(encoding="utf-8"))
    if _superseded(d):
        pytest.skip(f"{suite.name} is superseded; it makes no claim about this tree")
    prov = d.get("ep_provenance") or {}
    assert prov.get("pinned_sources_dirty") is not True, (
        f"{suite.name} was measured while the pinned attention sources had uncommitted "
        f"edits, so its git_commit does not identify the code that ran.")


@pytest.mark.parametrize("suite", committed_suites(), ids=lambda p: p.name)
def test_every_committed_vulkan_row_declares_its_counter_scope(suite: Path):
    """B1, re-asserted against whatever record is current, not against one filename.

    Two shapes carry Vulkan timings: the competition suite, which holds a ``results``
    list, and the profile reduction, which *is* a single Vulkan record at top level.
    The second shape is the one that slipped through when this screen was first
    written, so both are named here.
    """
    d = json.loads(suite.read_text(encoding="utf-8"))
    missing = [r.get("workload") for r in d.get("results", [])
               if r.get("arm") == "vulkan" and not r.get("counters_scope")]
    if d.get("arm") == "vulkan" and not d.get("counters_scope"):
        missing.append(f"{suite.stem} (top-level reduction)")
    assert not missing, f"{suite.name}: vulkan rows without counters_scope: {missing}"


# --- an inflated measurement may exist, but never as a ranking --------------------

#: The one committed record allowed to hold counter-inflated Vulkan timings, because
#: demonstrating the inflation is its entire purpose.
INFLATED_AB = "counters_ab_inflated.json"


def test_only_the_declared_ab_artifact_carries_inflated_timings():
    """B1's root cause, screened rather than remembered.

    The rejected branch committed counter-inflated Vulkan rankings.  Deleting them is not
    a durable fix -- the next person to run the harness with the dump left on can commit
    the same thing again.  So the rule is positional: exactly one file may hold inflated
    rows, it is named here, and it is a methodology artifact rather than a ranking.
    """
    offenders = []
    for suite in committed_suites():
        d = json.loads(suite.read_text(encoding="utf-8"))
        rows = d.get("results") or ([d] if d.get("arm") else [])
        inflated = [r.get("workload") for r in rows
                    if r.get("arm") == "vulkan"
                    and r.get("counters_scope") == cc.COUNTERS_SCOPE_ALL_RUNS]
        if inflated and suite.name != INFLATED_AB:
            offenders.append(f"{suite.name}: {inflated}")
    assert not offenders, (
        "counter-inflated Vulkan timings are committed outside the declared A/B "
        f"artifact ({INFLATED_AB}):\n  " + "\n  ".join(offenders) +
        "\nThese are not rankings and must not be committed as if they were.")


def test_the_ab_artifact_actually_demonstrates_the_inflation():
    """The A/B must contain the thing it is committed to show, or it is dead weight."""
    ab = CUDA69 / INFLATED_AB
    assert ab.exists(), f"{INFLATED_AB} is cited as the backing for the inflation claim"
    d = json.loads(ab.read_text(encoding="utf-8"))
    rows = [r for r in d["results"]
            if r.get("arm") == "vulkan"
            and r.get("counters_scope") == cc.COUNTERS_SCOPE_ALL_RUNS]
    assert rows, f"{INFLATED_AB} holds no inflated Vulkan row; it demonstrates nothing"


def test_the_documented_inflation_figures_come_out_of_the_artifacts():
    """Neither side of the quoted A/B may be a remembered number."""
    inflated = json.loads((CUDA69 / INFLATED_AB).read_text(encoding="utf-8"))
    hot = next(r["median_ms"] for r in inflated["results"]
               if r["arm"] == "vulkan" and r["workload"] == "prefill_1")

    clean_files = [p for p in committed_suites() if p.name != INFLATED_AB]
    clean = None
    for p in clean_files:
        d = json.loads(p.read_text(encoding="utf-8"))
        if _superseded(d):
            # The A/B is quoted as a fact about the *current* build. Reading its clean
            # side out of a retired record would compare two different programs.
            continue
        for r in d.get("results") or []:
            if (r.get("arm") == "vulkan" and r.get("workload") == "prefill_1"
                    and r.get("counters_scope") == cc.COUNTERS_SCOPE_FIRST_RUN):
                clean = r["median_ms"]
    assert clean is not None, "no committed clean prefill_1 vulkan row to compare against"

    src = (HERE / "cuda_competition.py").read_text(encoding="utf-8")
    assert f"{hot:.3f}" in src, (
        f"cuda_competition.py quotes an inflated figure that is not the artifact's "
        f"{hot:.3f} ms")
    assert f"{clean:.3f}" in src, (
        f"cuda_competition.py quotes a clean figure that is not the artifact's "
        f"{clean:.3f} ms")
    assert hot > clean, "the A/B does not show inflation"
