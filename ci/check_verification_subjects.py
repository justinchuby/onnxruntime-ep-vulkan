#!/usr/bin/env python3
"""``check_verification_subjects`` — does each check in this tree verify something other than itself?

THE DEFECT CLASS
================
`gen_proof_ledger.py --check` compared the proof ledger against **the proof ledger**: header
digest against body, declared count against lines present, every field against a shape. All of
those can be true of a ledger that describes a binary nobody built. On 2026-08-03 Switch edited a
GQA shader, did not regenerate, the EP declined all 32 GroupQueryAttention nodes for the whole
run — and `--check` printed `PASS: 103 entr(ies)` throughout. That verdict was quoted as merge
evidence at least six times in one day.

The verdict was not false. It was about the wrong subject. It established that the file is
internally consistent; it was read as establishing that the file describes the artifact about to
ship, which is the only thing anybody reads a proof ledger for. **Not broken — resolves anyway.**

So the question generalises, and a fix to one tool does not answer it: *how many other checks in
this tree ask a question whose only possible answer is the one they give?*

WHY THIS IS A SCRIPT AND NOT A PARAGRAPH
========================================
Link's standard from the lane sweep: report the count checked and the count found, so the sweep
is falsifiable. A prose sweep cannot be re-run, cannot be wrong in a way anybody notices, and
silently stops covering the tree the moment somebody adds a check.

This enumerates check entry points **from the tree** and requires every one of them to carry a
classification here. A new check with no row is a FAIL — not because the new check is wrong, but
because nobody has said what its subject is, and that is the state this whole exercise is about.

THE THREE VERDICTS
==================
``ARTIFACT``  the check reads a built binary, a device, or a run that happened. Its subject can
              disagree with it.
``EXTERNAL``  the check compares two independently-produced things — source against a recorded
              baseline, a workflow against an inventory, a log against a declared register. Its
              subject can disagree with it.
``SELF``      the check's only source of truth is the thing it is checking. It can report a
              condition of the file's internal form and nothing else. **This is the defect.**

A `SELF` row is not automatically a bug — a digest-vs-body check is a real and useful thing. It
is a bug when the verdict is *quoted* for something the check cannot see, which is a property of
the verdict text, so every `SELF` row must carry a `verdict_text` naming its own limit.

Usage:  python ci/check_verification_subjects.py [--check] [--list]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------------
# The classification. One row per check entry point; `verdict` is one of ARTIFACT/EXTERNAL/SELF.
# --------------------------------------------------------------------------------------------
TABLE: dict[str, dict[str, str]] = {
    # ---- rust/tools: the four tools with a literal `--check` mode ----------------------------
    "rust/tools/gen_proof_ledger.py --check": {
        "verdict": "ARTIFACT",
        "subject": "evidence/proof_ledger.jsonl",
        "oracle": "the built EP — OrtEpVulkanGetShaderSubject per entry, plus "
                  "OrtEpVulkanGetLedgerIdentity for the baked-vs-disk copy",
        "note": "WAS SELF until §8.9.21. This is the row the sweep exists because of.",
    },
    "rust/tools/counters_abi.py --check": {
        "verdict": "ARTIFACT",
        "subject": "the counters ABI as declared in rust/src/counters.rs",
        "oracle": "the built DLL's OrtEpVulkanGetCountersLayout manifest",
        "note": "already right; the model the ledger check was rebuilt on.",
    },
    "rust/tools/audit_instruments.py --check": {
        "verdict": "EXTERNAL",
        "subject": "the source tree's instrument population",
        "oracle": "ci/instrument_census.json, a baseline written in a different commit",
        "note": "source-level by nature — 'has this instrument a production caller' is a fact "
                "about source, not about a binary. The baseline is the other side.",
    },
    "rust/tools/audit_counter_test_lock.py --check": {
        "verdict": "EXTERNAL",
        "subject": "#[test] fns that touch a process-global or a contended env var",
        "oracle": "the measured contention set derived independently from the same tree",
        "note": "two derivations over one tree, not one derivation over itself: the set of "
                "tests that touch a global is computed separately from the set that locks.",
    },
    # ---- the EP's own CLI --------------------------------------------------------------------
    "epctl --check-counters": {
        "verdict": "ARTIFACT",
        "subject": "a claim about what a run did",
        "oracle": "the loaded EP's live counters",
    },
    # ---- ci/: the lane checks ----------------------------------------------------------------
    "ci/check_build_precondition.py": {
        "verdict": "EXTERNAL",
        "subject": "a lane's reported result",
        "oracle": "whether the build step actually built (BUILD_SKIPPED and friends)",
    },
    "ci/check_census_completeness.py": {
        "verdict": "EXTERNAL",
        "subject": "the wiring census output",
        "oracle": "criterion 12's three further requirements, declared outside the census",
    },
    "ci/check_readme_usage.py": {
        "verdict": "ARTIFACT",
        "subject": "the import statements in README.md's fenced python blocks",
        "oracle": "the tree itself, plus the declared dependency manifests",
        "note": (
            "The two sides were produced by different people for different reasons: the "
            "README is written by whoever ships a feature, the package layout by whoever "
            "builds it. That is exactly the pairing that drifted -- the README documented "
            "`import onnxruntime_ep_vulkan` for months while no such package existed."
        ),
    },
    "ci/check_device_loss.py": {
        "verdict": "ARTIFACT",
        "subject": "a run's log",
        "oracle": "the device-loss lines the driver emitted during that run",
    },
    "ci/check_device_state.py": {
        "verdict": "ARTIFACT",
        "subject": "a published duration",
        "oracle": "the device-state record sampled from the board",
    },
    "ci/check_fatal_log.py": {
        "verdict": "ARTIFACT",
        "subject": "a lane's exit status",
        "oracle": "known-fatal lines in the run's own log",
    },
    "ci/check_flake_witness.py": {
        "verdict": "ARTIFACT",
        "subject": "a red-then-green sequence",
        "oracle": "the failing test names recovered from the two runs' logs",
    },
    "ci/check_icd_suppression.py": {
        "verdict": "ARTIFACT",
        "subject": "an ICD-removal negative control",
        "oracle": "whether the loader actually honoured the suppression on this host",
    },
    "ci/check_lane_inventory.py": {
        "verdict": "EXTERNAL",
        "subject": "ci/lane_inventory.py's declared lanes",
        "oracle": ".github/workflows — every gate step must be classified",
    },
    "ci/check_ledger_portability.py": {
        "verdict": "ARTIFACT",
        "subject": "a cross-platform run's claim",
        "oracle": "the run's own ledger faults and decline counts",
    },
    "ci/check_ledger_census.py": {
        "verdict": "ARTIFACT",
        "subject": "evidence/proof_ledger.jsonl's key set over time",
        "oracle": "git history of the same file, taken with --full-history",
    },
    "ci/check_artifact_frame.py": {
        "verdict": "ARTIFACT",
        "subject": "a committed reading (bench/results/.../*) and the frame it carries",
        "oracle": "git log over the source paths the frame names, since the commit it names",
        "note": "the frame is written by the producer and read by the screen; the oracle is git, not the artifact. Rai found the case it exists for: a log committed three hours BEFORE the fix it was cited as evidence for.",
    },
    "ci/check_open_reds.py": {        "verdict": "ARTIFACT",
        "subject": "ci/open_reds.json's accepted reds",
        "oracle": "the guards themselves, executed",
    },
    "ci/check_run_disturbance.py": {
        "verdict": "ARTIFACT",
        "subject": "a timing figure",
        "oracle": "the repetitions' disagreement in the run that produced it",
    },
    "ci/check_suite_productivity.py": {
        "verdict": "ARTIFACT",
        "subject": "a green step",
        "oracle": "how many assertions that step actually executed",
    },
    "ci/check_tautological_assertions.py": {
        "verdict": "EXTERNAL",
        "subject": "assertion source text",
        "oracle": "the structural rule that an assertion's two sides must differ",
        "note": "the closest sibling of the ledger defect, one level down: an assertion whose "
                "two sides are the same text is a SELF check in miniature.",
    },
    "ci/check_tick_conversions.py": {
        "verdict": "EXTERNAL",
        "subject": "tick-to-nanosecond conversion sites in source",
        "oracle": "timestampPeriod's real per-device values",
    },
    "ci/check_verdict.py": {
        "verdict": "ARTIFACT",
        "subject": "the criterion-10 lane decision",
        "oracle": "ci/gate_chain_fp32.py's verdict record, produced by a separate process",
    },
    "ci/check_vocabulary.py": {
        "verdict": "EXTERNAL",
        "subject": "every ci/ check's verdict vocabulary",
        "oracle": "tests/ops/_verdict.py, the single definition",
    },
    "ci/check_verification_subjects.py": {
        "verdict": "EXTERNAL",
        "subject": "this table",
        "oracle": "the check entry points enumerated from the tree by `discovered()`",
        "note": "classified deliberately rather than exempted. A sweep that excused itself "
                "from its own rule would be the defect it screens for, one level up: the "
                "table's oracle must be the tree, or the sweep is a list agreeing with a list.",
    },
}


def discovered() -> list[str]:
    """Enumerate check entry points **from the tree**, not from the table above."""
    found: list[str] = []
    for path in sorted((REPO / "rust" / "tools").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        # A file that *dispatches* on `--check`, not one that merely passes the string to a
        # subprocess. The looser rule pulled in probe_ledger_subject_check.py, whose only
        # relationship to `--check` is that it runs one — a caller is not an entry point.
        if re.search(r'add_argument\(\s*"--check"|"--check" (?:not )?in (?:sys\.argv|argv)', text):
            found.append(f"rust/tools/{path.name} --check")
    for path in sorted((REPO / "ci").glob("check_*.py")):
        found.append(f"ci/{path.name}")
    epctl = REPO / "rust" / "src" / "bin" / "epctl.rs"
    if epctl.is_file():
        for m in re.finditer(r'"(--check-[a-z-]+)"', epctl.read_text(encoding="utf-8")):
            found.append(f"epctl {m.group(1)}")
    return sorted(set(found))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    found = discovered()
    unclassified = [f for f in found if f not in TABLE]
    stale = [k for k in TABLE if k not in found]
    self_rows = [k for k, v in TABLE.items() if v["verdict"] == "SELF"]

    by_verdict = {"ARTIFACT": 0, "EXTERNAL": 0, "SELF": 0}
    for k in found:
        if k in TABLE:
            by_verdict[TABLE[k]["verdict"]] += 1

    if args.list or not args.check:
        for k in found:
            row = TABLE.get(k)
            if row is None:
                print(f"  {'UNCLASSIFIED':<12} {k}")
                continue
            print(f"  {row['verdict']:<12} {k}")
            print(f"               subject: {row['subject']}")
            print(f"               oracle:  {row['oracle']}")
            if row.get("note"):
                print(f"               note:    {row['note']}")

    print(
        f"\nCHECKED {len(found)} check entry point(s) discovered from the tree; "
        f"{len(TABLE)} classified.\n"
        f"FOUND   {by_verdict['SELF']} that verify an artifact against itself; "
        f"{by_verdict['ARTIFACT']} verify against a binary/device/run, "
        f"{by_verdict['EXTERNAL']} against an independently-produced second thing."
    )

    failures = []
    if unclassified:
        failures.append(
            f"{len(unclassified)} check(s) exist in the tree with no row here — nobody has said "
            f"what their subject is: {unclassified}"
        )
    if stale:
        failures.append(
            f"{len(stale)} row(s) name a check that no longer exists; the sweep is describing a "
            f"tree that is gone: {stale}"
        )
    if self_rows:
        failures.append(
            f"{len(self_rows)} check(s) verify an artifact against itself and must say so in "
            f"their verdict text: {self_rows}"
        )
    if failures:
        print("\nFAIL(condition=UNCLASSIFIED_CHECK):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS — every check in the tree is classified, and none verifies only itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
