#!/usr/bin/env python3
"""The falsifier for ci/check_census_completeness.py.

A completeness check that has only ever been observed passing reads as coverage and
asserts nothing — which is the exact defect the screen it guards was written to find, so
shipping it without this file would be self-refuting.

Each arm below mutates a SCRATCH COPY of the tree, the map or the census artifacts, runs
the screen against the copy, and requires the screen to report the specific condition that
mutation was designed to produce.  Nothing in the repository is modified.

The arm that matters most is `counter_added`: a mechanism-shaped surface appears in
production Rust, the census is not told, and the independent whole names it.  That is the
arm which proves the denominator is not derived from the numerator — without it, "50
surfaces, 12 mechanisms" would be a sentence, not a measurement.

The outage arms matter for the opposite reason.  A control that cannot tell "the screen
found nothing" from "the screen could not run" would certify a broken screen as a clean
tree, and the screen's whole point is that those are different facts (§10.0.1 R13).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI = REPO_ROOT / "ci"
SCREEN = CI / "check_census_completeness.py"
RUST_SRC = REPO_ROOT / "rust" / "src"
MAP = CI / "census_surface_map.json"
ARTIFACTS = REPO_ROOT / "bench" / "results"
SCRATCH = REPO_ROOT / "bench" / "results" / "census-completeness-control"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_ERROR_INSTRUMENT = 4

TAG = "CENSUS-EXTENT-CONTROL"


class AnchorMissing(RuntimeError):
    pass


def _anchor_replace(path: Path, anchor: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if anchor not in text:
        raise AnchorMissing(f"{path.name}: anchor not found: {anchor!r}")
    path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")


def _fresh_scratch() -> Path:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    return SCRATCH


def _copy_tree(dest_name: str) -> Path:
    dest = SCRATCH / dest_name
    shutil.copytree(RUST_SRC, dest)
    return dest


def _copy_artifacts(dest_name: str) -> Path:
    dest = SCRATCH / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    for path in ARTIFACTS.glob("wiring_census-*.json"):
        shutil.copy2(path, dest / path.name)
    return dest


def _copy_map(dest_name: str) -> Path:
    dest = SCRATCH / dest_name
    shutil.copy2(MAP, dest)
    return dest


def _run(src: Path, mapping: Path, artifacts: Path | None) -> tuple[int, str]:
    argv = [sys.executable, str(SCREEN), "--rust-src", str(src), "--map", str(mapping)]
    if artifacts is None:
        argv.append("--no-artifacts")
    else:
        argv += ["--artifacts", str(artifacts)]
    # Complete encoding pair, the same one run_check() in ci/test_lane_checks.py:54
    # already documents: pinning only the PARENT-side decode (the previous
    # `text=True, encoding="utf-8"`) is not enough, because the CHILD screen picks its
    # own stdout encoding from locale.getpreferredencoding() -- cp1252 on a default
    # Windows shell -- and this screen prints em-dashes and section signs in its report.
    # That lesson was written down for run_check and never carried here, so on Windows
    # the child's bytes failed to decode inside subprocess's reader thread, the
    # exception surfaced only as a PytestUnhandledThreadException-style warning, `out`
    # came back empty, and ALL TWELVE arms reported `arm_did_not_fire` -- including
    # "a new EP env switch appears and nobody tells the census", the arm that exists to
    # catch exactly the ONNXRUNTIME_EP_VULKAN_RANK_INFERENCE regression this file's
    # sibling map entry corrects (#47). A control that cannot read its subject reports
    # "no detection" for a mutation the screen DID name: an outage wearing a finding's
    # clothes. PYTHONIOENCODING pins the child's encode side; decoding the bytes here
    # with errors="replace" pins ours and guarantees an unexpected byte degrades to a
    # visible substitution rather than to silence.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(argv, capture_output=True, env=env)
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, out + err


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def arm_baseline() -> tuple[str, bool, str]:
    code, out = _run(RUST_SRC, MAP, ARTIFACTS)
    ok = code == EXIT_PASS
    return ("baseline: the real tree", ok, f"exit={code} (want {EXIT_PASS})")


def arm_counter_added() -> tuple[str, bool, str]:
    src = _copy_tree("src-counter-added")
    _anchor_replace(
        src / "counters.rs",
        "    pub abi_version: u32,",
        "    pub abi_version: u32,\n    /// planted by the negative control\n"
        "    pub shader_cache_evictions: u64,",
    )
    code, out = _run(src, MAP, ARTIFACTS)
    ok = code == EXIT_FAIL_CONDITION and "shader_cache_evictions" in out
    return (
        "a new C ABI counter appears in production Rust and nobody tells the census",
        ok,
        f"exit={code} (want {EXIT_FAIL_CONDITION}); named the surface: "
        f"{'shader_cache_evictions' in out}",
    )


def arm_phase_added() -> tuple[str, bool, str]:
    src = _copy_tree("src-phase-added")
    _anchor_replace(
        src / "trace.rs",
        "    Readback,",
        "    Readback,\n    /// planted by the negative control\n    ShaderJit,",
    )
    code, out = _run(src, MAP, ARTIFACTS)
    ok = code == EXIT_FAIL_CONDITION and "ShaderJit" in out
    return (
        "a new trace phase appears and nobody tells the census",
        ok,
        f"exit={code} (want {EXIT_FAIL_CONDITION}); named the surface: {'ShaderJit' in out}",
    )


def arm_env_added() -> tuple[str, bool, str]:
    src = _copy_tree("src-env-added")
    _anchor_replace(
        src / "allocator.rs",
        'pub const ENV_RESERVATION_MIB: &str = "ONNXRUNTIME_EP_VULKAN_VA_RESERVE_MIB";',
        'pub const ENV_RESERVATION_MIB: &str = "ONNXRUNTIME_EP_VULKAN_VA_RESERVE_MIB";\n'
        'pub const ENV_PLANTED_BY_CONTROL: &str = "ONNXRUNTIME_EP_VULKAN_PLANTED_SWITCH";',
    )
    code, out = _run(src, MAP, ARTIFACTS)
    ok = code == EXIT_FAIL_CONDITION and "ONNXRUNTIME_EP_VULKAN_PLANTED_SWITCH" in out
    return (
        "a new EP env switch appears and nobody tells the census",
        ok,
        f"exit={code} (want {EXIT_FAIL_CONDITION}); named the surface: "
        f"{'ONNXRUNTIME_EP_VULKAN_PLANTED_SWITCH' in out}",
    )


def arm_map_rot() -> tuple[str, bool, str]:
    mapping = _copy_map("map-rot.json")
    doc = json.loads(mapping.read_text(encoding="utf-8"))
    doc["surfaces"].append(
        {
            "kind": "counter",
            "id": "counter_that_was_deleted",
            "disposition": "censused",
            "mechanism": "partitioner",
            "reason": "planted by the negative control",
            "owner": "Link",
        }
    )
    mapping.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    code, out = _run(RUST_SRC, mapping, ARTIFACTS)
    ok = code == EXIT_FAIL_CONDITION and "counter_that_was_deleted" in out
    return (
        "the map claims coverage of a surface that no longer exists",
        ok,
        f"exit={code} (want {EXIT_FAIL_CONDITION}); named the entry: "
        f"{'counter_that_was_deleted' in out}",
    )


def arm_mechanism_dropped() -> tuple[str, bool, str]:
    """The census stops enumerating a mechanism whose surfaces are still instrumented."""
    arts = _copy_artifacts("artifacts-mechanism-dropped")
    touched = 0
    for path in arts.glob("wiring_census-*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("observations", {}).pop("ledger_lookup", None) is not None:
            touched += 1
            path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    if not touched:
        raise AnchorMissing(
            "no census artifact enumerated 'ledger_lookup' — the mutation this arm "
            "depends on could not be performed"
        )
    code, out = _run(RUST_SRC, MAP, arts)
    ok = code == EXIT_FAIL_CONDITION and "ledger_lookup" in out
    return (
        "the census drops a mechanism whose surfaces are still instrumented",
        ok,
        f"exit={code} (want {EXIT_FAIL_CONDITION}); named the mechanism: "
        f"{'ledger_lookup' in out}",
    )


def arm_unclaimed_name() -> tuple[str, bool, str]:
    arts = _copy_artifacts("artifacts-new-mechanism")
    for path in arts.glob("wiring_census-*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.setdefault("observations", {})["shader_cache_guard"] = "WIRED"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    code, out = _run(RUST_SRC, MAP, arts)
    ok = code == EXIT_FAIL_CONDITION and "shader_cache_guard" in out
    return (
        "a mechanism joins the census with no claim about what its NAME asserts",
        ok,
        f"exit={code} (want {EXIT_FAIL_CONDITION}); named the mechanism: "
        f"{'shader_cache_guard' in out}",
    )


def arm_name_claim_contradicted() -> tuple[str, bool, str]:
    mapping = _copy_map("map-name-claimed.json")
    doc = json.loads(mapping.read_text(encoding="utf-8"))
    hit = False
    for entry in doc.get("mechanism_names", []):
        if entry["mechanism"] == "gpu_tracer":
            entry["name_verified"] = True
            hit = True
    if not hit:
        raise AnchorMissing("no mechanism_names entry for 'gpu_tracer' to flip")
    mapping.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    code, out = _run(RUST_SRC, mapping, ARTIFACTS)
    ok = code == EXIT_FAIL_CONDITION and "gpu_tracer" in out
    return (
        "somebody records a name as verified while every recorded arm reads the same",
        ok,
        f"exit={code} (want {EXIT_FAIL_CONDITION}); named the mechanism: "
        f"{'gpu_tracer' in out}",
    )


def arm_map_missing() -> tuple[str, bool, str]:
    missing = SCRATCH / "no-such-map.json"
    code, out = _run(RUST_SRC, missing, ARTIFACTS)
    ok = code == EXIT_ERROR_INSTRUMENT and "surface_map_unavailable" in out
    return (
        "the surface map is absent — an outage, not a clean tree and not a gap",
        ok,
        f"exit={code} (want {EXIT_ERROR_INSTRUMENT})",
    )


def arm_artifacts_missing() -> tuple[str, bool, str]:
    empty = SCRATCH / "artifacts-empty"
    empty.mkdir(parents=True, exist_ok=True)
    code, out = _run(RUST_SRC, MAP, empty)
    ok = code == EXIT_ERROR_INSTRUMENT and "census_artifacts_unavailable" in out
    return (
        "the census produced no artifact — reported as an outage rather than as twelve "
        "mechanisms with no coverage",
        ok,
        f"exit={code} (want {EXIT_ERROR_INSTRUMENT})",
    )


def arm_extractor_blinded() -> tuple[str, bool, str]:
    """The single most dangerous failure: a denominator that silently shrinks."""
    src = _copy_tree("src-extractor-blinded")
    _anchor_replace(
        src / "counters.rs",
        "pub struct VulkanEpCounters {",
        "pub struct VulkanEpCountersRenamedByControl {",
    )
    code, out = _run(src, MAP, ARTIFACTS)
    ok = code == EXIT_ERROR_INSTRUMENT and "extractor_found_nothing" in out
    return (
        "the counter extractor's anchor moves, so the whole would silently shrink",
        ok,
        f"exit={code} (want {EXIT_ERROR_INSTRUMENT})",
    )


def arm_stale_absence_claim() -> tuple[str, bool, str]:
    """Issue #58, replayed: a map entry claims a real production symbol does not exist.

    Reconstructs the exact shape of the defect issue #58 found in
    ``ONNXRUNTIME_EP_VULKAN_GEMV_PACKED``'s entry — a `reason` asserting no artifact
    records the pipeline variant, when `counters::record_pipeline_variant` already does
    and did before the entry was last read. The check must catch this from the tree, not
    from the map re-describing itself.
    """
    mapping = _copy_map("map-stale-absence.json")
    doc = json.loads(mapping.read_text(encoding="utf-8"))
    hit = False
    for entry in doc.get("surfaces", []):
        if entry["id"] == "ONNXRUNTIME_EP_VULKAN_GEMV_PACKED":
            entry["absence_claims"] = ["record_pipeline_variant"]
            hit = True
    if not hit:
        raise AnchorMissing("no GEMV_PACKED entry in the surface map to plant the claim on")
    mapping.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    code, out = _run(RUST_SRC, mapping, ARTIFACTS)
    ok = (
        code == EXIT_FAIL_CONDITION
        and "record_pipeline_variant" in out
        and "stale_absence_claim" in out
    )
    return (
        "issue #58 replayed: a map entry claims 'record_pipeline_variant' does not exist",
        ok,
        f"exit={code} (want {EXIT_FAIL_CONDITION}); named the symbol: "
        f"{'record_pipeline_variant' in out}; named the condition: "
        f"{'stale_absence_claim' in out}",
    )


def arm_absence_claim_true() -> tuple[str, bool, str]:
    """The counter-control: a claim naming a symbol that really is absent must not fire.

    Without this arm, `arm_stale_absence_claim` alone cannot distinguish a check that
    correctly reads the tree from one that fails every `absence_claims` entry regardless
    of content — the same reasoning the frame-witness arm's REPLAYED/PLANTED split uses.
    `gemv_packed_dispatches` is the counter this project's own tests once asked for and
    never built (see tests/ops/test_wiring_census.py's prior comment, corrected 2026-08-07)
    — a genuinely absent symbol on this tree today.
    """
    mapping = _copy_map("map-absence-claim-true.json")
    doc = json.loads(mapping.read_text(encoding="utf-8"))
    hit = False
    for entry in doc.get("surfaces", []):
        if entry["id"] == "ONNXRUNTIME_EP_VULKAN_GEMV_PACKED":
            entry["absence_claims"] = ["gemv_packed_dispatches"]
            hit = True
    if not hit:
        raise AnchorMissing("no GEMV_PACKED entry in the surface map to plant the claim on")
    mapping.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    code, out = _run(RUST_SRC, mapping, ARTIFACTS)
    ok = code == EXIT_PASS and "stale_absence_claim" not in out
    return (
        "counter-control: a claim naming a genuinely absent symbol must not be flagged",
        ok,
        f"exit={code} (want {EXIT_PASS}); stale_absence_claim raised: "
        f"{'stale_absence_claim' in out}",
    )


def arm_empty_tree() -> tuple[str, bool, str]:
    empty = SCRATCH / "src-empty"
    empty.mkdir(parents=True, exist_ok=True)
    code, out = _run(empty, MAP, ARTIFACTS)
    ok = code == EXIT_ERROR_INSTRUMENT
    return (
        "an empty source tree — every census is complete over an empty whole",
        ok,
        f"exit={code} (want {EXIT_ERROR_INSTRUMENT})",
    )


ARMS = [
    arm_baseline,
    arm_counter_added,
    arm_phase_added,
    arm_env_added,
    arm_map_rot,
    arm_mechanism_dropped,
    arm_unclaimed_name,
    arm_name_claim_contradicted,
    arm_stale_absence_claim,
    arm_absence_claim_true,
    arm_map_missing,
    arm_artifacts_missing,
    arm_extractor_blinded,
    arm_empty_tree,
]


def main() -> int:
    _fresh_scratch()
    results = []
    try:
        for arm in ARMS:
            label, ok, detail = arm()
            print(f"[{'ok' if ok else 'BAD'}] {label} -> {detail}", flush=True)
            results.append((label, ok))
    except AnchorMissing as exc:
        print(f"{TAG}: ERROR(instrument=anchor_not_found)", flush=True)
        print(str(exc), flush=True)
        print(
            f"{TAG}: the control could not perform the mutation it was going to test, so "
            "it reached no observation. This is NOT a pass for the screen and NOT a "
            "finding against it (§10.0.1 R13).",
            flush=True,
        )
        return EXIT_ERROR_INSTRUMENT
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)

    bad = [label for label, ok in results if not ok]
    if bad:
        print(f"\n{TAG}: FAIL(condition=arm_did_not_fire)", flush=True)
        for label in bad:
            print(f"  {label}", flush=True)
        print(
            f"{TAG}: the screen did not report the mutation above. Until it does, its "
            "PASS on the real tree means nothing.",
            flush=True,
        )
        return EXIT_FAIL_CONDITION

    print(
        f"\n{TAG}: PASS — the screen names a counter, a trace phase and an env switch "
        "planted in production Rust that the census does not know about; names a "
        "mechanism the census stopped enumerating; refuses a name recorded as verified "
        "against arms that never varied; catches a replayed issue-#58 stale absence "
        "claim against the real tree while leaving a genuinely-absent-symbol claim "
        "unflagged; and reports a missing map, a missing artifact set, a moved extractor "
        "anchor and an empty tree as instrument outages rather than as coverage.",
        flush=True,
    )
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
