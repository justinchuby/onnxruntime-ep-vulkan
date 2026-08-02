#!/usr/bin/env python3
"""Criterion 12, the three things the census line does not supply.

WHAT THIS IS FOR
================

DESIGN.md §10 row 12 asks for a wiring census PLUS three further things.  The census
itself — ``tests/ops/test_wiring_census.py``, Trinity's — answers *did this mechanism
run*.  Morpheus's ruling is that this is not the same question as the criterion:

  > The census answers whether a mechanism ran; a criterion answers whether a claim is
  > false-able.  A census line can never discharge a criterion.

So this screen does not census anything.  It reads what the census produced and asks the
three questions the census cannot ask about itself, because the answer to each of them
would have to come from outside the census's own enumeration:

  (1) EXTENT.  For each mechanism, how much of that mechanism's surface did the
      observation touch?  The census emits one line per mechanism with no statement of
      what fraction of the mechanism the line covers.  "I looked and found nothing" and
      "I did not look there" are different facts (§10.0.1 R12), and a census that cannot
      tell them apart reports the second as the first.

  (2) THE DECOMPOSITION IDENTITY, AGAINST AN INDEPENDENT WHOLE.  The census says twelve
      mechanisms.  Twelve out of what?  If the denominator is derived from the same list
      that produces the numerator then ``12/12`` is true by construction, the check can
      never fail, and it reads as coverage while asserting nothing (§10.0.1 R11 — a
      decomposition that appears to close is the hardest kind of wrong).  This is exactly
      the shape Morpheus refused to let criterion 11 close on, where a ledger generated
      from the claim table makes ``ledger_hits == proven_key_lookups`` forever.

      The whole this screen uses comes from production Rust source that the census does
      not write and cannot edit: the C ABI counter fields (Tank's ``counters.rs``), the
      trace phases (``trace.rs``, Niobe's), and the EP environment switches (Switch's and
      Mouse's, spread across ``rust/src``).  Numerator and denominator have different
      authors in different files in a different language.  Adding a mechanism to the
      Rust tree therefore grows the denominator whether or not anyone tells the census,
      and that is the arm that makes this check able to fail.

  (3) NAME AGAINST CONTENT.  Tank's vocabulary already has the terminal state — MISNAMED —
      and the standing specimen is ``Phase::Record``: wired, invoked, correct,
      input-varying, and wrong by 50x in what it was called.  A census that verifies a
      mechanism ran and never asks whether its name describes what it did will certify
      that specimen.  The decidable form of the question is R10 turned on the census's own
      output: an observation whose text is byte-identical across every arm the census was
      run in has demonstrated only that the mechanism is present.  Presence does not
      distinguish a name from a wrong name.

WHAT THIS SCREEN DOES NOT CLAIM
===============================

  * It does not close row 12.  Trinity owns row 12's tally as she owns criterion 11's,
    and Morpheus's ruling on criterion 11 is explicit that supplying the artifact and
    closing the row must not be the same act.  This file produces evidence and a JSON
    record; the row is hers to move.

  * It does not decide that a mechanism is misnamed.  It decides that a mechanism's name
    is UNVERIFIED BY OBSERVATION — that nothing the census recorded would have differed
    had the name been wrong.  ``Phase::Record`` was correct code; the screen that should
    have caught it would not have called it broken either.

  * Its whole is the *instrumented* surface of the EP, not the EP.  A mechanism that
    exists in Rust and touches no counter, no trace phase and no env switch is invisible
    to this screen, and the report says so with a count rather than by omission.

  * For host-side mechanisms (``layering_lint``, ``instrument_census``,
    ``device_state_guard``) the independent whole has no surface at all, so their extent
    is reported UNOBSERVABLE, never 0/0.  A ratio of zero over zero presented as coverage
    is the identity defect this screen exists to refuse.

TERMINAL STATES (§10.0.1 R13)
=============================

PASS / FAIL(condition=...) / ERROR(instrument=...), exits 0 / 1 / 4.  An instrument error
is never a detection.  Findings quote the source text and the observation text; this file
never reports a finding count in place of a finding.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_tick_conversions import (  # noqa: E402
    EXIT_ERROR_INSTRUMENT,
    EXIT_FAIL_CONDITION,
    EXIT_PASS,
    EXIT_USAGE,
    strip_comments_and_strings,
    test_line_span,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUST_SRC = REPO_ROOT / "rust" / "src"
DEFAULT_MAP = Path(__file__).resolve().parent / "census_surface_map.json"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "bench" / "results"
ARTIFACT_GLOB = "wiring_census-*.json"

TAG = "CENSUS-EXTENT"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_pass(detail: str) -> int:
    print(f"{TAG}: PASS — {detail}", flush=True)
    return EXIT_PASS


def report_fail(condition: str, detail: str) -> int:
    print(f"{TAG}: FAIL(condition={condition})", flush=True)
    print(detail, flush=True)
    print(
        f"{TAG}: this is a finding about the census's coverage of an independently "
        "enumerated whole. The screen reached its observation and the source lines it "
        "read are quoted above.",
        flush=True,
    )
    return EXIT_FAIL_CONDITION


def report_instrument_error(instrument: str, detail: str) -> int:
    print(f"{TAG}: ERROR(instrument={instrument})", flush=True)
    print(detail, flush=True)
    print(
        f"{TAG}: the screen did not reach its observation, so this is NOT a detection "
        "(DESIGN.md §10.0.1 R13). Do not read it as a census gap and do not read it as "
        "coverage.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


# ---------------------------------------------------------------------------
# The independent whole.
#
# Three extractors over production Rust.  Each returns {id: (relpath, lineno, text)}.
# Every one of them is deliberately dumb: a regex over source text the census does not
# write.  A clever extractor that shared an abstraction with the census would reintroduce
# the coupling this whole file exists to break.
# ---------------------------------------------------------------------------

_COUNTER_STRUCT = re.compile(r"pub\s+struct\s+VulkanEpCounters\s*\{")
_COUNTER_FIELD = re.compile(r"^\s*pub\s+([a-z_][a-z0-9_]*)\s*:\s*[A-Za-z0-9_:<>\[\] ]+,")
_PHASE_ENUM = re.compile(r"pub\s+enum\s+Phase\s*\{")
_PHASE_VARIANT = re.compile(r"^\s*([A-Z][A-Za-z0-9]*)\s*(?:=\s*[^,]+)?,")
_ENV_NAME = re.compile(r"ONNXRUNTIME_EP_VULKAN_[A-Z0-9_]+")


def _rust_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.rs") if p.is_file())


def _block_lines(lines: list[str], start_idx: int) -> tuple[int, int]:
    """Line index range (inclusive, exclusive) of the brace block opening at start_idx."""
    depth = 0
    i = start_idx
    while i < len(lines):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth <= 0 and i > start_idx:
            return start_idx + 1, i
        if depth <= 0 and i == start_idx and lines[i].count("}"):
            return start_idx + 1, i
        i += 1
    return start_idx + 1, len(lines)


def extract_counters(root: Path) -> dict[str, tuple[str, int, str]]:
    path = root / "counters.rs"
    if not path.is_file():
        return {}
    code = strip_comments_and_strings(path.read_text(encoding="utf-8", errors="replace"))
    lines = code.splitlines()
    found: dict[str, tuple[str, int, str]] = {}
    for idx, line in enumerate(lines):
        if not _COUNTER_STRUCT.search(line):
            continue
        lo, hi = _block_lines(lines, idx)
        for j in range(lo, hi):
            mo = _COUNTER_FIELD.match(lines[j])
            if mo:
                found[mo.group(1)] = (
                    str(path.relative_to(root.parent.parent)).replace("\\", "/"),
                    j + 1,
                    lines[j].strip(),
                )
    return found


def extract_trace_phases(root: Path) -> dict[str, tuple[str, int, str]]:
    path = root / "trace.rs"
    if not path.is_file():
        return {}
    code = strip_comments_and_strings(path.read_text(encoding="utf-8", errors="replace"))
    lines = code.splitlines()
    found: dict[str, tuple[str, int, str]] = {}
    for idx, line in enumerate(lines):
        if not _PHASE_ENUM.search(line):
            continue
        lo, hi = _block_lines(lines, idx)
        for j in range(lo, hi):
            mo = _PHASE_VARIANT.match(lines[j])
            if mo:
                found[mo.group(1)] = (
                    str(path.relative_to(root.parent.parent)).replace("\\", "/"),
                    j + 1,
                    lines[j].strip(),
                )
    return found


def extract_env_switches(root: Path) -> tuple[dict[str, tuple[str, int, str]], dict]:
    """Every EP env switch named in production code, plus the frame this scan held out.

    Env names live inside string literals, so this scan does NOT strip strings — but it
    still holds out ``#[cfg(test)]`` bodies, and reports how many lines that removed.  A
    screen that does not publish what it declined to read is a screen whose silence a
    reader will mistake for a clean result (R12).
    """
    code_sites: dict[str, tuple[str, int, str]] = {}
    prose_sites: dict[str, tuple[str, int, str]] = {}
    held_out = 0
    scanned = 0
    for path in _rust_files(root):
        raw = path.read_text(encoding="utf-8", errors="replace")
        code = strip_comments_and_strings(raw)
        skip = test_line_span(code)
        code_lines = code.splitlines()
        rel = str(path.relative_to(root.parent.parent)).replace("\\", "/")
        for idx, line in enumerate(raw.splitlines(), start=1):
            if idx in skip:
                held_out += 1
                continue
            scanned += 1
            names = _ENV_NAME.findall(line)
            if not names:
                continue
            # A doc comment naming a switch is prose about the switch, not a site that
            # reads it.  Both are recorded; the code site wins, because a finding that
            # quotes a comment is a finding a reviewer will discount.
            is_code = bool(code_lines[idx - 1].strip()) if idx - 1 < len(code_lines) else False
            table = code_sites if is_code else prose_sites
            for name in names:
                table.setdefault(name, (rel, idx, line.strip()))
    found = dict(prose_sites)
    found.update(code_sites)
    prose_only = sorted(set(prose_sites) - set(code_sites))
    return found, {
        "production_lines_scanned": scanned,
        "cfg_test_lines_held_out": held_out,
        "env_switches_named_only_in_prose": prose_only,
    }


def build_whole(root: Path) -> tuple[list[dict], dict]:
    counters = extract_counters(root)
    phases = extract_trace_phases(root)
    envs, env_frame = extract_env_switches(root)
    surfaces: list[dict] = []
    for kind, table in (("counter", counters), ("trace_phase", phases), ("env_switch", envs)):
        for ident, (rel, lineno, text) in sorted(table.items()):
            surfaces.append(
                {"kind": kind, "id": ident, "file": rel, "line": lineno, "text": text}
            )
    frame = {
        "counter_fields": len(counters),
        "trace_phases": len(phases),
        "env_switches": len(envs),
        **env_frame,
    }
    return surfaces, frame


# ---------------------------------------------------------------------------
# The map.  A surface is covered only by an explicit, owned, reasoned entry — the same
# discipline as ci/tick_conversion_allowlist.json.  Growing the census's claimed coverage
# then requires a visible edit to this file in the same diff as the code that grew it.
#
# Four dispositions, and the second one is the one that makes the map honest:
#
#   censused        a census mechanism observes this surface.
#   uncensused      the surface is instrumented and NO census mechanism observes it.
#                   This is a standing gap, recorded with an owner.  It is reported as a
#                   gap, loudly, and it does not fail the screen: a surface nobody has
#                   mapped and a surface everybody has agreed is unobserved are different
#                   states, and collapsing them would make the screen permanently red and
#                   therefore unreadable.
#   out_of_frame    the surface cannot be observed from inside the census's frame — it is
#                   the census's own transport or its own lane parameter.
#   not_a_mechanism ABI header fields and similar: not a thing that can run.
# ---------------------------------------------------------------------------

DISPOSITIONS = {"censused", "uncensused", "out_of_frame", "not_a_mechanism"}


def load_map(path: Path) -> tuple[dict, str | None]:
    if not path.is_file():
        return {}, f"surface map not found at {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {}, f"surface map at {path} is not readable JSON: {exc}"
    if not isinstance(data, dict) or "surfaces" not in data:
        return {}, f"surface map at {path} has no 'surfaces' list"
    for entry in data["surfaces"]:
        for field in ("kind", "id", "disposition", "reason", "owner"):
            if field not in entry:
                return {}, f"surface map entry {entry!r} is missing '{field}'"
        if entry["disposition"] not in DISPOSITIONS:
            return {}, (
                f"surface map entry {entry['kind']}:{entry['id']} has unknown "
                f"disposition {entry['disposition']!r}"
            )
        if entry["disposition"] == "censused" and not entry.get("mechanism"):
            return {}, (
                f"surface map entry {entry['kind']}:{entry['id']} is 'censused' but "
                "names no mechanism"
            )
    for entry in data.get("mechanism_names", []):
        for field in ("mechanism", "name_asserts", "discriminator", "name_verified"):
            if field not in entry:
                return {}, f"mechanism_names entry {entry!r} is missing '{field}'"
    return data, None


# ---------------------------------------------------------------------------
# The census's own output.  Read from the artifact, never from the test source: the
# artifact is what the census produced, and R10 is explicit that the falsifier for "X is
# wired" is an artifact X produced, never a reading of X's code.
# ---------------------------------------------------------------------------


def load_census_artifacts(directory: Path) -> tuple[list[dict], str | None]:
    if not directory.is_dir():
        return [], f"census artifact directory {directory} does not exist"
    out = []
    for path in sorted(directory.glob(ARTIFACT_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return [], f"census artifact {path.name} is not readable JSON: {exc}"
        if "observations" not in data:
            return [], f"census artifact {path.name} carries no 'observations' object"
        out.append({"name": path.name, "observations": data["observations"]})
    if not out:
        return [], f"no {ARTIFACT_GLOB} artifacts under {directory}"
    return out, None


# ---------------------------------------------------------------------------
# (1) EXTENT and (2) THE WHOLE
# ---------------------------------------------------------------------------


def coverage(surfaces: list[dict], mapping: dict, mechanisms: set[str]) -> dict:
    by_key = {(e["kind"], e["id"]): e for e in mapping.get("surfaces", [])}
    seen = {(s["kind"], s["id"]) for s in surfaces}

    unmapped = [s for s in surfaces if (s["kind"], s["id"]) not in by_key]
    rot = [e for k, e in by_key.items() if k not in seen]

    per_mech: dict[str, list[dict]] = defaultdict(list)
    out_of_frame: list[dict] = []
    not_mechanism: list[dict] = []
    uncensused: list[dict] = []
    for s in surfaces:
        entry = by_key.get((s["kind"], s["id"]))
        if entry is None:
            continue
        if entry["disposition"] == "censused":
            per_mech[entry["mechanism"]].append({**s, **{"reason": entry["reason"]}})
        elif entry["disposition"] == "uncensused":
            uncensused.append({**s, **{"reason": entry["reason"], "owner": entry["owner"]}})
        elif entry["disposition"] == "out_of_frame":
            out_of_frame.append({**s, **{"reason": entry["reason"]}})
        else:
            not_mechanism.append({**s, **{"reason": entry["reason"]}})

    orphan_mechanisms = sorted(set(per_mech) - mechanisms)
    return {
        "unmapped": unmapped,
        "rot": rot,
        "per_mechanism": per_mech,
        "uncensused": uncensused,
        "out_of_frame": out_of_frame,
        "not_a_mechanism": not_mechanism,
        "orphan_mechanisms": orphan_mechanisms,
    }


def extent_table(
    per_mech: dict[str, list[dict]], mechanisms: list[str], artifacts: list[dict]
) -> list[dict]:
    """How much of each mechanism's independently enumerated surface the census names.

    'Names' is deliberately the weakest possible test — the identifier appears somewhere
    in the observation text.  A weaker numerator makes the resulting fractions an upper
    bound on coverage, which is the safe direction: a mechanism reported here at 1/6 is at
    most 1/6, and one reported at 6/6 has only been shown to mention six identifiers.
    """
    rows = []
    for mech in mechanisms:
        surfaces = per_mech.get(mech, [])
        texts = [
            str(a["observations"].get(mech, "")) for a in artifacts if mech in a["observations"]
        ]
        blob = " || ".join(texts)
        if not surfaces:
            rows.append(
                {
                    "mechanism": mech,
                    "extent": "UNOBSERVABLE",
                    "named": 0,
                    "surfaces": 0,
                    "why": (
                        "the independent whole enumerates no counter, trace phase or env "
                        "switch for this mechanism — its denominator would have to be "
                        "self-supplied, and 0/0 presented as coverage is the identity "
                        "defect this screen refuses"
                    ),
                    "unnamed": [],
                }
            )
            continue
        named = [s for s in surfaces if s["id"] in blob]
        rows.append(
            {
                "mechanism": mech,
                "extent": f"{len(named)}/{len(surfaces)}",
                "named": len(named),
                "surfaces": len(surfaces),
                "why": "",
                "unnamed": sorted(s["id"] for s in surfaces if s["id"] not in blob),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# (3) NAME AGAINST CONTENT
# ---------------------------------------------------------------------------

NAME_VARIES = "VARIES"
NAME_INVARIANT = "INVARIANT"
NAME_UNOBSERVABLE = "UNOBSERVABLE"


def name_content(mechanisms: list[str], artifacts: list[dict]) -> list[dict]:
    """Classify each mechanism's observation by whether its CONTENT responded to input.

    Three states, and the third is the point.  With fewer than two arms a mechanism is not
    'invariant' — it is unmeasured, and calling that invariant would be reporting 0 where
    the event could not occur (R12).
    """
    rows = []
    for mech in mechanisms:
        values = []
        missing = []
        for art in artifacts:
            if mech in art["observations"]:
                values.append((art["name"], str(art["observations"][mech])))
            else:
                missing.append(art["name"])
        distinct = sorted({v for _, v in values})
        if len(values) < 2:
            state = NAME_UNOBSERVABLE
        elif len(distinct) > 1:
            state = NAME_VARIES
        else:
            state = NAME_INVARIANT
        rows.append(
            {
                "mechanism": mech,
                "state": state,
                "arms": len(values),
                "distinct_observations": len(distinct),
                "not_enumerated_in": missing,
                "specimen": distinct[0] if distinct else "",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rust-src", type=Path, default=DEFAULT_RUST_SRC)
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP)
    ap.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    ap.add_argument(
        "--no-artifacts",
        action="store_true",
        help=(
            "narrow to the source-only rules and print the extent and name sections as "
            "UNOBSERVABLE by frame. Explicit because a check that silently narrows when "
            "its input is missing returns to the view it was written to replace."
        ),
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.rust_src.is_dir():
        return report_instrument_error(
            "rust_src_missing", f"--rust-src {args.rust_src} is not a directory."
        )

    surfaces, frame = build_whole(args.rust_src)
    if not surfaces:
        return report_instrument_error(
            "empty_whole",
            f"Enumerated zero instrumented surfaces under {args.rust_src}.\n"
            "A whole of size zero makes every coverage fraction 0/0 and every census "
            "complete by construction. That is an outage, not a clean tree.",
        )
    for label, count in (
        ("counter fields", frame["counter_fields"]),
        ("trace phases", frame["trace_phases"]),
        ("env switches", frame["env_switches"]),
    ):
        if count == 0:
            return report_instrument_error(
                "extractor_found_nothing",
                f"The {label} extractor returned nothing. Either the declaration it keys "
                "on moved, or the file is gone. Both make the denominator smaller than "
                "the truth, which inflates coverage silently — so it is reported as an "
                "outage rather than as a smaller whole.",
            )

    mapping, map_err = load_map(args.map)
    if map_err:
        return report_instrument_error(
            "surface_map_unavailable",
            f"{map_err}\n"
            "Without the map this screen cannot tell a surface the census deliberately "
            "excludes from one nobody has looked at, so it declines to report either.",
        )

    if args.no_artifacts:
        artifacts: list[dict] = []
        art_err = None
    else:
        artifacts, art_err = load_census_artifacts(args.artifacts)
        if art_err:
            return report_instrument_error(
                "census_artifacts_unavailable",
                f"{art_err}\n"
                "The census's own output is the only admissible evidence of what the "
                "census observed (R10: an artifact X produced, never a reading of X's "
                "code). Pass --no-artifacts to run the source-only rules deliberately; "
                "this screen will not do it silently.",
            )

    mechanisms: list[str] = []
    for art in artifacts:
        for mech in art["observations"]:
            if mech not in mechanisms:
                mechanisms.append(mech)

    cov = coverage(surfaces, mapping, set(mechanisms) if artifacts else set(mapping_mechs(mapping)))
    findings: list[str] = []

    if cov["unmapped"]:
        lines = [
            "Surfaces present in the independent whole with no entry in the surface map.",
            "Each is a mechanism-shaped thing the census has not been told about; the",
            "census's own list cannot report these because it does not enumerate them.",
            "",
        ]
        for s in cov["unmapped"]:
            lines.append(f"  {s['file']}:{s['line']}  [{s['kind']}] {s['id']}")
            lines.append(f"      {s['text']}")
        findings.append(("unmapped_surface", "\n".join(lines)))

    if cov["rot"]:
        lines = [
            "Surface map entries whose surface no longer exists in the Rust tree.",
            "A map that outlives its subject reports coverage of something that is gone.",
            "",
        ]
        for e in cov["rot"]:
            lines.append(
                f"  [{e['kind']}] {e['id']} — mapped to "
                f"{e.get('mechanism') or e['disposition']} (owner {e['owner']})"
            )
        findings.append(("surface_map_rot", "\n".join(lines)))

    if artifacts and cov["orphan_mechanisms"]:
        lines = [
            "The surface map routes a real surface to a mechanism that no census artifact",
            "enumerates. The surface is instrumented and nothing reports on it.",
            "",
        ]
        for mech in cov["orphan_mechanisms"]:
            ids = ", ".join(sorted(s["id"] for s in cov["per_mechanism"][mech]))
            lines.append(f"  mechanism '{mech}' <- surfaces: {ids}")
        lines.append("")
        lines.append(f"  census artifacts read: {', '.join(a['name'] for a in artifacts)}")
        findings.append(("mechanism_not_enumerated", "\n".join(lines)))

    names = {e["mechanism"]: e for e in mapping.get("mechanism_names", [])}
    rows_name = name_content(mechanisms, artifacts) if artifacts else []

    unclaimed = [m for m in mechanisms if m not in names]
    if unclaimed:
        lines = [
            "The census enumerates mechanisms whose NAME has no recorded claim.",
            "'What does this name assert, and what observation would differ if the name",
            "were wrong' has no answer on record for:",
            "",
        ]
        for mech in unclaimed:
            lines.append(f"  {mech}")
        lines.append("")
        lines.append(
            "  The standing specimen is Phase::Record — wired, invoked, correct, "
            "input-varying, and wrong by 50x in what it was called. A census with no "
            "name claim certifies that specimen."
        )
        findings.append(("unclaimed_mechanism_name", "\n".join(lines)))

    contradicted = []
    for row in rows_name:
        entry = names.get(row["mechanism"])
        if entry and entry["name_verified"] and row["state"] != NAME_VARIES:
            contradicted.append((row, entry))
    if contradicted:
        lines = [
            "A mechanism's name is recorded as verified, but nothing the census observed",
            "would have differed had the name been wrong.",
            "",
        ]
        for row, entry in contradicted:
            lines.append(
                f"  {row['mechanism']}: state={row['state']} over {row['arms']} arm(s), "
                f"{row['distinct_observations']} distinct observation(s)"
            )
            lines.append(f"      claim: {entry['name_asserts']}")
            lines.append(f"      discriminator on record: {entry['discriminator']}")
            lines.append(f"      observation: {row['specimen']}")
        findings.append(("name_claim_contradicted", "\n".join(lines)))

    rows_extent = (
        extent_table(cov["per_mechanism"], mechanisms, artifacts) if artifacts else []
    )

    record = {
        "criterion": "12",
        "produced_by": "ci/check_census_completeness.py (Link)",
        "closes_row": False,
        "closes_row_why": (
            "Trinity owns row 12's tally. Supplying the artifact and closing the row must "
            "not be the same act (Morpheus, criterion 11 ruling)."
        ),
        "independent_whole": {
            "source": "production Rust the census does not write: counters.rs fields, "
            "trace.rs Phase variants, ONNXRUNTIME_EP_VULKAN_* switches under rust/src",
            "size": len(surfaces),
            **frame,
        },
        "census_mechanisms": mechanisms,
        "census_artifacts": [a["name"] for a in artifacts],
        "extent": rows_extent,
        "name_content": rows_name,
        "uncensused_surfaces": [
            {"kind": s["kind"], "id": s["id"], "file": s["file"], "line": s["line"],
             "reason": s["reason"], "owner": s["owner"]}
            for s in cov["uncensused"]
        ],
        "out_of_frame": [
            {"kind": s["kind"], "id": s["id"], "reason": s["reason"]}
            for s in cov["out_of_frame"]
        ],
        "not_a_mechanism": [
            {"kind": s["kind"], "id": s["id"], "reason": s["reason"]}
            for s in cov["not_a_mechanism"]
        ],
        "findings": [c for c, _ in findings],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print_report(surfaces, frame, cov, mechanisms, rows_extent, rows_name, artifacts, args)

    if findings:
        condition = findings[0][0]
        detail = "\n\n".join(text for _, text in findings)
        if len(findings) > 1:
            detail += "\n\nConditions in this run: " + ", ".join(c for c, _ in findings)
        return report_fail(condition, detail)

    covered = sum(len(v) for v in cov["per_mechanism"].values())
    return report_pass(
        f"{len(surfaces)} instrumented surfaces enumerated from production Rust; "
        f"{covered} routed to a censused mechanism, {len(cov['uncensused'])} recorded as "
        f"instrumented-but-uncensused gaps, {len(cov['out_of_frame'])} out of frame, "
        f"{len(cov['not_a_mechanism'])} not mechanisms; every mechanism the census "
        "enumerates has a name claim on record. Coverage of the whole is stated above and "
        "is NOT a verdict on row 12 — see the gap list, which a PASS here does not close."
    )


def mapping_mechs(mapping: dict) -> set[str]:
    return {
        e["mechanism"]
        for e in mapping.get("surfaces", [])
        if e["disposition"] == "censused" and e.get("mechanism")
    }


def print_report(surfaces, frame, cov, mechanisms, rows_extent, rows_name, artifacts, args):
    print(f"\n{TAG}: the independent whole — what the census's twelve is twelve OF.")
    print(
        f"  {len(surfaces)} instrumented surfaces from production Rust under {args.rust_src}: "
        f"{frame['counter_fields']} C ABI counter fields, {frame['trace_phases']} trace "
        f"phases, {frame['env_switches']} EP env switches."
    )
    print(
        f"  frame: {frame['production_lines_scanned']} production lines read, "
        f"{frame['cfg_test_lines_held_out']} lines held out as #[cfg(test)] "
        "(UNOBSERVABLE by frame, not zero findings)."
    )
    print(
        "  The census does not write any of these files. A mechanism added to the Rust "
        "tree grows this denominator whether or not the census is told."
    )
    prose_only = frame.get("env_switches_named_only_in_prose") or []
    if prose_only:
        print(
            f"  {len(prose_only)} env switch(es) appear only in prose, never in a line of "
            f"production code: {', '.join(prose_only)}."
        )

    print(
        f"\n{TAG}: (2) THE WHOLE — {len(surfaces)} surfaces against the census's "
        f"{len(mechanisms) if mechanisms else 'unknown number of'} mechanisms."
    )
    covered = sum(len(v) for v in cov["per_mechanism"].values())
    print(
        f"  {covered} censused, {len(cov['uncensused'])} instrumented and observed by NO "
        f"census mechanism, {len(cov['out_of_frame'])} out of the census's frame, "
        f"{len(cov['not_a_mechanism'])} not mechanisms."
    )
    for s in cov["uncensused"]:
        print(
            f"    GAP  {s['file']}:{s['line']}  [{s['kind']}] {s['id']} "
            f"(owner {s['owner']}) — {s['reason']}"
        )
    print(
        "  Numerator and denominator have different authors, in different files, in a "
        "different language. This is the property that lets the count be wrong."
    )

    if not artifacts:
        print(
            f"\n{TAG}: extent and name-content are UNOBSERVABLE in this frame "
            "(--no-artifacts). The census produced no output for this run to read, and "
            "this screen will not infer coverage from source alone."
        )
        return

    print(f"\n{TAG}: (1) EXTENT — how much of each mechanism's surface the observation names.")
    for row in rows_extent:
        if row["extent"] == "UNOBSERVABLE":
            print(f"  {row['mechanism']:<26} UNOBSERVABLE — {row['why']}")
        else:
            tail = (
                f"  not named: {', '.join(row['unnamed'])}" if row["unnamed"] else ""
            )
            print(f"  {row['mechanism']:<26} {row['extent']}{tail}")
    print(
        "  An extent is an UPPER bound: the numerator counts identifiers the observation "
        "mentions, which is the weakest evidence of coverage that can be checked at all."
    )

    print(
        f"\n{TAG}: (3) NAME AGAINST CONTENT — did the observation's content respond to "
        f"input, across {len(artifacts)} census arm(s)?"
    )
    for row in rows_name:
        note = ""
        if row["state"] == NAME_INVARIANT:
            note = " — presence only; nothing here would differ if the name were wrong"
        elif row["state"] == NAME_UNOBSERVABLE:
            note = " — fewer than two arms; not invariant, unmeasured (R12)"
        print(
            f"  {row['mechanism']:<26} {row['state']:<13} "
            f"{row['distinct_observations']} distinct over {row['arms']} arm(s){note}"
        )
    invariant = [r["mechanism"] for r in rows_name if r["state"] == NAME_INVARIANT]
    unmeasured = [r["mechanism"] for r in rows_name if r["state"] == NAME_UNOBSERVABLE]
    print(
        f"  {len(invariant)} mechanism(s) certified on presence alone; "
        f"{len(unmeasured)} with too few arms to say. Neither is a claim that the name is "
        "wrong — Phase::Record was correct code."
    )
    print(
        "  The arms are the census runs that exist, not arms designed to vary each "
        "mechanism. INVARIANT therefore means 'no arm on record distinguished it', which "
        "is weaker than 'no arm could' and stronger than nothing: a name whose content no "
        "recorded run has ever moved is a name certified by presence."
    )


def main_guarded(argv=None) -> int:
    try:
        return main(argv)
    except SystemExit as exc:  # argparse
        return int(exc.code or EXIT_USAGE)
    except Exception as exc:  # noqa: BLE001
        return report_instrument_error(
            "screen_raised",
            f"{type(exc).__name__}: {exc}\n"
            "The screen crashed before reaching its observation.",
        )


if __name__ == "__main__":
    sys.exit(main_guarded())
