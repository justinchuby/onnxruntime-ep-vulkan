#!/usr/bin/env python3
"""Publish the lane inventory as an artifact, and refuse to let it rot.

Two jobs, and the second is the one that matters.

**Publish.** Emit `lane-inventory.json` and a human-readable summary so the difference
between `operational` and `green` is visible to a reader of the run rather than only to a
reader of this repository. The artifact's content varies with the inventory's content,
which is the R10 property the YAML itself cannot have.

**Refuse to rot.** The inventory is hand-maintained, and a hand-maintained list of "what
each lane runs" drifts the moment someone adds a step. So this checker cross-references the
inventory against the workflow YAML in one direction only: **every gate-like step in the
lane must have an inventory entry.** A step nobody has classified is exactly the state the
ICD control lived in for weeks.

The reverse direction is deliberately NOT checked. An inventory entry with no matching step
might be a stale entry, or might be a step that was silently deleted — and those want
opposite responses. Guessing between them from YAML is the R10 mistake in miniature.

Exit codes follow R13:

* ``0`` — PASS.
* ``1`` — ``FAIL(condition=...)``. The inventory is unsound or a step is unclassified.
* ``4`` — ``ERROR(instrument=...)``. This checker could not do its job.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lane_inventory as inv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 4

#: Steps that are provisioning, not gating. A step that installs a toolchain has nothing to
#: falsify; requiring an inventory entry for it would fill the inventory with noise and make
#: the entries that matter harder to see.
#:
#: This is a closed, code-level list with a reason per entry and no command-line escape, for
#: the same reason `INSTRUMENT_DUMPS` in `device_state.py` is: a runtime exclusion switch is
#: a waiver with a flag.
PROVISIONING_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^Install ", "installs a dependency; nothing to falsify"),
    (r"^Provision ", "fetches a dependency; nothing to falsify"),
    (r"^Set up ", "configures the runner; nothing to falsify"),
    (r"^Add .* repository$", "adds a package source; nothing to falsify"),
    (r"^Checkout", "fetches the tree; nothing to falsify"),
    (r"^Upload ", "publishes artifacts; its failure is a transport failure"),
    (r"^Download ", "fetches artifacts; its failure is a transport failure"),
    (r"^Build ", "compiles; a compile failure is self-announcing"),
    (r"^Verify GLSL compiler", "a presence probe for a tool, reported as ERROR(instrument)"),
    (r"^Probe Vulkan loader", "produces the report that other checks read; not itself a gate"),
    (r"^Check formatting", "formatting; classified in the separate `format` job"),
)


def is_provisioning(step_name: str) -> tuple[bool, str | None]:
    for pattern, reason in PROVISIONING_PATTERNS:
        if re.search(pattern, step_name):
            return True, reason
    return False, None


#: Maps a workflow step name onto the inventory entry that classifies it. Substring match on
#: a distinctive fragment, because step names carry expression syntax and section numbers
#: that change more often than the thing the step does.
STEP_TO_CHECK: tuple[tuple[str, str], ...] = (
    # ORDER IS LOAD-BEARING: first fragment that substrings the step name wins.
    # The productivity floors must precede the lints they guard, or
    # "Layering lint asserted something (productivity floor, libtest)" silently
    # maps to build.layering_lint and the floor becomes invisible to the
    # inventory while still appearing classified. A wrong mapping reads exactly
    # like a right one; that is why these four are first.
    ("Flake-witness negative control", "hostfree.flake_witness_negative_control"),
    ("Flake witness", "device.flake_witness"),
    # Same ordering hazard one row down: "Open-reds negative control" substrings nothing
    # of "Open reds", but "Open reds" would substring neither — they are disjoint by
    # luck, not by design, so the control goes first anyway and this comment records why.
    ("Open-reds negative control", "hostfree.open_reds_negative_control"),
    ("Open reds", "hostfree.open_reds"),
    ("Verification-subject screen", "hostfree.verification_subjects"),
    ("Layering lint asserted something", "build.layering_lint_productivity"),
    ("Portability lint asserted something", "build.portability_lint_productivity"),
    ("Integration targets asserted something", "build.integration_targets_productivity"),
    ("Build-precondition negative control", "hostfree.build_precondition_negative_control"),
    ("Build-precondition screen", "hostfree.build_precondition"),
    ("Two-polarity tests", "hostfree.lane_check_tests"),
    ("Two-polarity suite asserted something", "hostfree.lane_check_productivity"),
    ("Suite-productivity negative control", "hostfree.suite_productivity_negative_control"),
    ("Rust unit tests asserted something", "build.rust_unit_tests_productivity"),
    ("Test-lock auditor", "hostfree.test_lock_auditor"),
    ("Contention-gate failure extractor", "hostfree.contention_gate_extractor"),
    ("Contention gate", "build.contention_gate"),
    ("Op-correctness step asserted something", "device.op_correctness_productivity"),
    ("No-ICD fallback step asserted something", "device.no_icd_step_productivity"),
    ("Tautological-assertion screen", "hostfree.tautological_assertions"),
    ("Tick-conversion screen negative control", "hostfree.tick_screen_negative_control"),
    ("Tick-conversion screen", "hostfree.tick_conversion_screen"),
    ("Device-loss screen negative control", "hostfree.device_loss_negative_control"),
    ("Device-loss screen", "hostfree.device_loss_screen"),
    ("Census extent negative control", "hostfree.census_completeness_negative_control"),
    ("Census extent and independent whole", "hostfree.census_completeness"),
    ("Verdict vocabulary preflight", "hostfree.verdict_vocabulary"),
    ("cargo test --lib", "build.rust_unit_tests"),
    ("Rust unit tests", "build.rust_unit_tests"),
    ("Portability lint", "build.portability_lint"),
    ("Layering lint", "build.layering_lint"),
    ("Portability lint", "build.portability_lint"),
    ("Remaining integration targets", "build.integration_targets"),
    ("Lane inventory", "SELF"),
    ("Compile all targets", "build.compile_all_targets"),
    ("Clippy", "build.clippy"),
    ("op-correctness", "device.op_correctness"),
    ("Criterion 10", "device.criterion10_gate"),
    ("Known-fatal log line", "device.fatal_log_line"),
    ("Device-loss screen on this lane's own evidence", "device.device_loss_screen"),
    ("Proof-ledger portability negative control", "device.ledger_portability"),
    ("Proof-ledger portability screen", "device.ledger_portability"),
    ("no ICD", "device.icd_negative_control"),
    ("no-ICD", "device.icd_negative_control"),
    ("ICD suppression", "device.icd_negative_control"),
    ("declined artifact", "device.criterion10_gate"),
    ("device-state record", "device.device_state_guard"),
    ("Lane inventory", "SELF"),
)

STEP_NAME_RE = re.compile(r"^\s*-\s+name:\s*(.+?)\s*$")


def workflow_step_names_from_text(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        m = STEP_NAME_RE.match(line)
        if m:
            names.append(m.group(1).strip().strip("\"'"))
    return names


def workflow_step_names(path: Path) -> list[str]:
    """Step names from a workflow file, without a YAML parser.

    A parser would be better and is not available on every runner without an install step;
    the names are a flat regex over `- name:` lines, which is sufficient because this
    checker only ever asks whether a name is classified, never what the step does.
    """
    return workflow_step_names_from_text(path.read_text(encoding="utf-8"))


class UnionUnavailable(Exception):
    """The reference side could not be read. An outage, never a pass."""


def repo_root_for(path: Path) -> Path:
    """The git root containing `path`.

    Derived from the file, not from this script's location: the checker must be able to
    classify a workflow in any tree, and hardcoding its own repo root is how a check
    quietly starts reading the wrong file — found by the two-branch test on 2026-08-02.
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        cwd=str(path.parent),
    )
    if proc.returncode != 0:
        raise UnionUnavailable(
            f"{path} is not inside a git repository, so there is no other side to read: "
            f"{proc.stderr.strip() or 'no output'}"
        )
    return Path(proc.stdout.strip())


def step_names_at_ref(ref: str, rel_path: str, repo: Path) -> list[str]:
    """Step names for `rel_path` as it exists at `ref`.

    Deliberately NOT a merge. A three-way merge can conflict, and a conflict is a
    different conversation; the question here is only "does a step exist on either side
    that nobody has classified", and the union of the two name lists answers it without
    needing the merge to succeed. It is also correct in the case that actually bit us,
    where the two sides touched different regions of the file and merged cleanly.
    """
    proc = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    if proc.returncode != 0:
        raise UnionUnavailable(
            f"`git show {ref}:{rel_path}` failed: {proc.stderr.strip() or 'no output'}"
        )
    return workflow_step_names_from_text(proc.stdout)


def classify_names(names: list[str]) -> list[str]:
    """Names that are neither provisioning nor mapped to a known inventory entry."""
    known_ids = {c.id for c in inv.CHECKS} | {"SELF"}
    out = []
    for name in names:
        prov, _ = is_provisioning(name)
        if prov:
            continue
        hit = None
        for fragment, check_id in STEP_TO_CHECK:
            if fragment.lower() in name.lower():
                hit = check_id
                break
        if hit is None:
            out.append(name)
        elif hit not in known_ids:
            out.append(f"{name}  (mapped to unknown inventory id {hit!r})")
    return out


def unclassified_steps(path: Path) -> list[str]:
    return classify_names(workflow_step_names(path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workflow",
        action="append",
        default=[],
        help="workflow file whose gate steps must all be classified (repeatable)",
    )
    ap.add_argument("--out", help="write lane-inventory.json here")
    ap.add_argument("--summary", help="append a human summary to this file ($GITHUB_STEP_SUMMARY)")
    ap.add_argument("--render", action="store_true", help="print the inventory and exit 0")
    ap.add_argument(
        "--union-with",
        default="",
        help=(
            "also classify the step names of each --workflow as it exists at this git ref "
            "(e.g. origin/main). Catches the step that is classified on neither side of a "
            "merge that neither author could have seen alone."
        ),
    )
    ap.add_argument(
        "--union-required",
        action="store_true",
        help=(
            "treat an unreadable reference side as ERROR(instrument) instead of a warning. "
            "Use in CI, where a missing ref means the check silently degraded to the "
            "branch-only view it exists to replace."
        ),
    )
    args = ap.parse_args(argv)

    if args.render:
        print(inv.render())
        return EXIT_PASS

    problems = inv.validate()

    doc = inv.as_dict()
    if args.out:
        try:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"ERROR(instrument=inventory_artifact_unwritable): {e}")
            return EXIT_ERROR

    unclassified: dict[str, list[str]] = {}
    union_note = ""
    for wf in args.workflow:
        p = Path(wf)
        if not p.is_file():
            print(f"ERROR(instrument=workflow_absent): {wf} does not exist")
            return EXIT_ERROR
        try:
            names = workflow_step_names(p)
        except OSError as e:
            print(f"ERROR(instrument=workflow_unreadable): {wf}: {e}")
            return EXIT_ERROR
        side = "this tree"
        if args.union_with:
            try:
                root = repo_root_for(p.resolve())
                rel_path = p.resolve().relative_to(root).as_posix()
                other = step_names_at_ref(args.union_with, rel_path, root)
            except (UnionUnavailable, ValueError) as e:
                msg = (
                    f"the reference side `{args.union_with}` could not be read, so this "
                    f"run classified only the branch's own view of {wf} — which is exactly "
                    f"the blind spot --union-with exists to close. {e}"
                )
                if args.union_required:
                    print(f"ERROR(instrument=union_reference_unreadable): {msg}")
                    return EXIT_ERROR
                print(f"::warning title=union side unread::{msg}")
                union_note = f" (union with {args.union_with} UNAVAILABLE — branch view only)"
            else:
                only_there = [n for n in other if n not in names]
                names = names + only_there
                side = f"union of this tree and {args.union_with}"
                union_note = (
                    f" (union with {args.union_with}: {len(only_there)} step(s) present "
                    f"only there)"
                )
        bad = classify_names(names)
        if bad:
            unclassified[f"{wf} [{side}]"] = bad

    lines: list[str] = ["### Lane inventory — operational vs green", ""]
    for lane in inv.LANES:
        cls, why = inv.lane_classification(lane)
        lines.append(f"* **{lane}** — `{cls}`")
        lines.append(f"  * {why}")
        lines.append(f"  * {inv.falsifier_census(lane)[2]}")
    lines.append("")
    if args.workflow:
        lines.append(f"Workflow steps classified from: {side if args.workflow else 'n/a'}{union_note}")
        lines.append("")
    lines.append("Blind spots no lane catches:")
    for b in inv.BLIND_SPOTS:
        sub = b.substitute_status
        lines.append(f"* `{b.id}` — substitute: `{sub}`")
    summary = "\n".join(lines)
    print(summary)

    if args.summary:
        try:
            with open(args.summary, "a", encoding="utf-8") as fh:
                fh.write(summary + "\n")
        except OSError as e:
            print(f"ERROR(instrument=summary_unwritable): {e}")
            return EXIT_ERROR

    if problems:
        print("")
        print("FAIL(condition=inventory_unsound)")
        for p in problems:
            print(f"  - {p}")
        return EXIT_FAIL

    if unclassified:
        print("")
        print("FAIL(condition=unclassified_lane_step)")
        print(
            "  A lane step nobody has classified is the state the ICD negative control "
            "lived in: running, reported, and never once falsified."
        )
        for wf, names in unclassified.items():
            for n in names:
                print(f"  - {wf}: {n!r}")
        print(
            "  Add an entry to ci/lane_inventory.py (with its `misses` column filled in), "
            "or a STEP_TO_CHECK mapping if an existing entry covers it."
        )
        return EXIT_FAIL

    print("")
    print("PASS — every gate step in the checked workflows is classified.")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
