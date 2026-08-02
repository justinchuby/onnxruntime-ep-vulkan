#!/usr/bin/env python3
"""``check_tick_conversions`` — the source-level screen for the 52x defect class.

WHY THIS EXISTS, AND WHY NEITHER OF OUR TWO INSTRUMENT FAMILIES CAN REPLACE IT
=============================================================================
A device tick is not a nanosecond.  ``VkPhysicalDeviceLimits::timestampPeriod`` is 1.0 on
NVIDIA and on lavapipe, and **52.0833 on Intel Iris Xe**; parts in the field report up to
~83.  A code path that treats a raw tick as a nanosecond under-reports GPU time by that
factor.  `rust/src/trace.rs` owns the one correct conversion
(``GpuTimestampCalibration::ticks_to_ns`` / ``ticks_to_axis_us``), which applies the
period, masks to ``timestampValidBits`` and recovers a single counter wrap.

The residual this file attacks was named in my own report and not papered over:

    the unit tests prove the conversion is correct *where called*; they cannot prove
    every call site calls it.

That is the whole defect class, one layer up, and **neither of our instrument families can
see it**:

  * **Unit tests** close "is the arithmetic right".  They say nothing about "does every
    path use it", because a path that skips the conversion never appears in a test of the
    conversion.
  * **A device lane cannot close it either**, for the reason that makes the class nasty:
    on every device our CI can reach — NVIDIA, lavapipe, SwiftShader — ``timestampPeriod``
    is **1.0**, so the conversion is the *identity* and a path that skips it produces
    numerically identical output.  The bug is invisible on every runner we have and
    appears only on hardware we do not run in CI.

A defect class that is invisible to both arms of the evidence system is exactly the kind
R9 says to attack with a new instrument rather than with more of the old ones.  This is
that instrument.  It is **static**: it decides the question from the source text, where
the period's value is irrelevant, so it works on a runner with no GPU at all.

WHAT IT DECIDES, AND WHAT IT DOES NOT PRETEND TO
================================================
A lane script cannot do sound Rust semantic analysis, and writing one that *looked* like
it could would be R11 exactly — a decomposition that appears to close is the hardest kind
of wrong.  So this screen is deliberately **conservative in the direction that matters**:
it over-reports, and every over-report must be either fixed or entered in
``ci/tick_conversion_allowlist.json`` **with a recorded reason and an owner**.  That is
the design, not a limitation of it: adding a bypass then requires a visible edit to an
allowlist in the same diff, which is a thing a reviewer can see.

Three rules, each decidable from the text:

  **R-A — the arithmetic monopoly.**  Tick-valued and ``timestamp_period_ns``-valued
  expressions may take part in arithmetic (``+ - * /``, float casts, duration
  constructors) **only inside the sanctioned converters**.  Everywhere else a tick may be
  moved, stored, compared and logged, but not scaled.  A by-hand ``ticks * period`` is a
  bypass *even though it uses the period*, because it skips the mask and the wrap
  recovery — on Intel that is a 36-bit counter being read as if it were 64.

  **R-B — the single producer.**  Raw ticks enter the program in exactly one place:
  ``TimestampPool::read_results``.  This rule asserts that call site is unique and that
  its enclosing function also constructs a ``GpuTimestampCalibration``.  This is the arm
  that actually addresses *"does every path use it"*: a second reader of raw ticks
  anywhere in the tree turns the screen red and has to justify itself.  It is a
  structural invariant, so it is decidable, unlike a general flow analysis.

  **R-C — allowlist integrity.**  Every allowlist entry must still match a live site.  An
  entry whose site has moved or vanished is reported, so the allowlist cannot rot into a
  blanket that silently covers code nobody wrote it for.

TERMINAL STATES (R13)
=====================
    0  TICK-SCREEN: PASS
    1  TICK-SCREEN: FAIL(condition=...)     a finding about the source
    4  TICK-SCREEN: ERROR(instrument=...)   a finding about this screen, about nothing else

Every failure quotes the **line it read**, never a count on its own.

FRAME (R12)
===========
The frame is ``rust/src/**/*.rs``, production code.  Two things are outside it and are
reported as ``UNOBSERVABLE`` with their counts rather than contributing zero findings:
``#[cfg(test)]`` modules (a test may construct ticks freely; it ships nothing), and
non-Rust consumers of tick data such as ``bench/*.py``, which read already-converted
values out of a trace and never see a tick.

This screen imports nothing from ``tests/ops/_verdict.py``: it emits no *verdict* about
the EP, only a finding about source text, so it survives a vocabulary outage that takes
every other lane check down (see ``ci/check_vocabulary.py``).

USAGE
    python ci/check_tick_conversions.py [--root <dir>] [--json <out>] [--github-summary]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = Path(__file__).resolve().parent / "tick_conversion_allowlist.json"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

# The sanctioned conversion entry points. A line that calls one of these is converting,
# not bypassing. Named here once; the allowlist keys on function names, not on these.
CONVERTERS = ("ticks_to_ns", "ticks_to_axis_us", "mask_ticks", ".mask(")

# Tokens that denote a device tick or the scale factor that turns one into a duration.
# Kept apart because they license different things: a *tick* may be moved but never
# scaled, whereas the *period* is the scale factor itself and is legitimately carried
# from the driver to the calibration under its own name.
TICK_VALUE = re.compile(r"\b\w*ticks\w*\b")
PERIOD = re.compile(r"\btimestamp_period_ns\b|\btimestampPeriod\b")
TICK_TOKEN = re.compile(TICK_VALUE.pattern + "|" + PERIOD.pattern)

# Arithmetic that would turn a tick into a duration, or a duration into a wrong one.
# `->` and `=>` are excluded: a function signature that takes ticks is not a site that
# scales them, and a screen that says otherwise trains its readers to skim it.
ARITH = re.compile(
    r"(?<![-+*/=!<>])[-+*/](?![-+*/=>])"  # binary +,-,*,/ that is not ++/--/+=/==/->
    r"|\bas\s+f(?:32|64)\b"
    r"|\bas\s+i128\b"
    r"|Duration::from_(?:nanos|micros|millis|secs)"
)

# A binding or field whose NAME claims a duration. Feeding one of these from a tick
# expression on the same line is the defect stated in its purest form.
DURATION_NAME = re.compile(r"\b\w*_(?:ns|us|ms|secs|seconds|micros|nanos|duration)\b")

# `let <name> =` / `let Some(<name>) =` where <name> carries no tick token: the rename
# that would otherwise let a tick escape a name-based screen.
REBIND = re.compile(
    r"\blet\s+(?:mut\s+)?(?:Some\()?(?!\w*ticks)([A-Za-z_][A-Za-z0-9_]*)\)?\s*(?::[^=]*)?="
)

PRODUCER = "read_results"
CALIBRATION_TYPE = "GpuTimestampCalibration"


# ---------------------------------------------------------------------------
# Reporting — three tokens, and nothing else in this file prints a terminal state.
# ---------------------------------------------------------------------------


def report_pass(detail: str) -> int:
    print(f"TICK-SCREEN: PASS — {detail}", flush=True)
    return EXIT_PASS


def report_fail(condition: str, detail: str) -> int:
    print(f"TICK-SCREEN: FAIL(condition={condition})", flush=True)
    print(detail, flush=True)
    print(
        "TICK-SCREEN: this is a finding about the source. The screen reached its "
        "observation and the lines it read are quoted above.",
        flush=True,
    )
    return EXIT_FAIL_CONDITION


def report_instrument_error(instrument: str, detail: str) -> int:
    print(f"TICK-SCREEN: ERROR(instrument={instrument})", flush=True)
    print(detail, flush=True)
    print(
        "TICK-SCREEN: the screen did not reach its observation, so this is NOT a "
        "detection (DESIGN.md §10.0.1 R13). Do not route it as a source bug and do not "
        "read it as a clean tree.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


# ---------------------------------------------------------------------------
# Lexing. Comments and string literals are not code; a screen that reads them
# reports on prose, and the prose here is full of the word "ticks".
# ---------------------------------------------------------------------------


def strip_comments_and_strings(text: str) -> str:
    """Blank out comments and string literals, preserving line structure exactly.

    Line structure is preserved (blanks, not deletions) because every finding this screen
    reports quotes a file and a line number, and a screen whose line numbers are off by a
    comment block is a screen nobody will trust twice.
    """
    out = []
    i, n = 0, len(text)
    depth_block = 0
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if depth_block:
            if ch == "/" and nxt == "*":
                depth_block += 1
                out.append("  ")
                i += 2
                continue
            if ch == "*" and nxt == "/":
                depth_block -= 1
                out.append("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if ch == "/" and nxt == "*":
            depth_block = 1
            out.append("  ")
            i += 2
            continue
        if ch == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "r" and nxt in ('"', "#"):
            # Raw string: r"..." or r#"..."#
            j = i + 1
            hashes = 0
            while j < n and text[j] == "#":
                hashes += 1
                j += 1
            if j < n and text[j] == '"':
                closer = '"' + "#" * hashes
                end = text.find(closer, j + 1)
                end = n if end == -1 else end + len(closer)
                for k in range(i, end):
                    out.append("\n" if text[k] == "\n" else " ")
                i = end
                continue
        if ch == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, n)):
                out.append("\n" if text[k] == "\n" else " ")
            i = j
            continue
        if ch == "'" and i + 2 < n and (text[i + 2] == "'" or text[i + 1] == "\\"):
            # char literal, not a lifetime
            j = text.find("'", i + 1)
            j = n if j == -1 else j + 1
            for k in range(i, j):
                out.append(" ")
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


TEST_ATTR = re.compile(r"#\[cfg\(test\)\]")


def test_line_span(code: str) -> set[int]:
    """1-based line numbers inside `#[cfg(test)] mod ... { ... }` blocks.

    Reported, not silently dropped: R12 says a frame's exclusions are stated, so the
    summary prints how many lines were held out and why.
    """
    lines = code.splitlines()
    excluded: set[int] = set()
    i = 0
    while i < len(lines):
        if TEST_ATTR.search(lines[i]):
            depth = 0
            started = False
            j = i
            while j < len(lines):
                depth += lines[j].count("{") - lines[j].count("}")
                if "{" in lines[j]:
                    started = True
                excluded.add(j + 1)
                if started and depth <= 0:
                    break
                j += 1
            i = j + 1
            continue
        i += 1
    return excluded


FN_DECL = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")


def enclosing_functions(code: str) -> dict[int, str]:
    """Map 1-based line number → nearest enclosing `fn` name (best effort, brace-counted).

    Best effort is stated rather than hidden: the function name is used for *allowlist
    keying and for the human reading the report*, and every allowlist entry is
    additionally pinned to the text of the line it covers, so a mis-attributed function
    name cannot widen an exemption.
    """
    result: dict[int, str] = {}
    stack: list[tuple[str, int]] = []  # (fn name, brace depth at which it opened)
    depth = 0
    pending: str | None = None
    for lineno, line in enumerate(code.splitlines(), start=1):
        m = FN_DECL.search(line)
        if m:
            pending = m.group(1)
        opens = line.count("{")
        closes = line.count("}")
        if pending and opens:
            stack.append((pending, depth))
            pending = None
        depth += opens - closes
        while stack and depth <= stack[-1][1]:
            stack.pop()
        result[lineno] = stack[-1][0] if stack else "<module>"
    return result


# ---------------------------------------------------------------------------
# The allowlist
# ---------------------------------------------------------------------------


def load_allowlist(path: Path) -> tuple[list[dict], str | None]:
    if not path.is_file():
        return [], f"allowlist not found at {path}"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"allowlist at {path} could not be read: {exc!r}"
    entries = doc.get("sanctioned_sites")
    if not isinstance(entries, list):
        return [], f"allowlist at {path} has no 'sanctioned_sites' list"
    for e in entries:
        for field in ("file", "function", "contains", "reason", "owner"):
            if field not in e:
                return [], (
                    f"allowlist entry {e!r} is missing '{field}'. Every exemption carries "
                    "a recorded reason and an owner, or it is not an exemption."
                )
    return entries, None


def entry_matches(entry: dict, rel: str, fn: str, line: str) -> bool:
    return (
        entry["file"] == rel
        and entry["function"] == fn
        and entry["contains"] in " ".join(line.split())
    )


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


def classify_line(line: str) -> str | None:
    """Return a bypass kind for a tick-bearing line, or None if it is inert/converted."""
    squashed = " ".join(line.split())
    if any(c in squashed for c in CONVERTERS):
        return None
    if ARITH.search(squashed):
        return "tick_arithmetic_outside_the_converter"
    if DURATION_NAME.search(squashed) and TICK_VALUE.search(squashed):
        # A duration-named binding fed from a *tick*-named value on the same line, with no
        # converter call. This is the defect in its purest form — `let ns = end_ticks;` —
        # and it is deliberately keyed on TICK_VALUE rather than on the period: carrying
        # `timestamp_period_ns` from the driver limits into the calibration under its own
        # name is a move, not a scale, and flagging it would fill the allowlist with
        # entries that teach reviewers the file is noise.
        if "=" in squashed or ":" in squashed:
            return "duration_named_value_fed_from_a_tick_without_a_converter"
    m = REBIND.search(squashed)
    if m and TICK_VALUE.search(squashed[m.end() :]):
        # Laundering by rename. This screen decides from names, so the way to defeat it is
        # `let raw = end_ticks;` followed by arithmetic on `raw`, which carries no tick
        # token and is invisible to the rule above. Rebinding a tick to a non-tick name is
        # therefore itself the finding, caught at the boundary where it is still visible.
        # Inside the sanctioned converters the rebinding is exactly what conversion looks
        # like, which is why those functions are on the allowlist rather than special-cased
        # in the rules.
        return "tick_rebound_to_a_non_tick_name"
    return None


def scan(root: Path) -> tuple[list[dict], dict, str | None]:
    src = root / "rust" / "src"
    if not src.is_dir():
        return [], {}, f"{src} is not a directory; there is nothing to screen"
    findings: list[dict] = []
    stats = {
        "files": 0,
        "tick_lines": 0,
        "converted_or_inert": 0,
        "excluded_test_lines": 0,
        "producer_sites": [],
        "converter_call_sites": [],
    }
    for path in sorted(src.rglob("*.rs")):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return [], stats, f"could not read {path}: {exc!r}"
        stats["files"] += 1
        code = strip_comments_and_strings(raw)
        excluded = test_line_span(code)
        stats["excluded_test_lines"] += len(excluded)
        fns = enclosing_functions(code)
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(code.splitlines(), start=1):
            # Skip only the *declaration* of the producer, not any line that happens to
            # also contain the word `fn`. A one-line `fn other() { p.read_results(); }`
            # is a second reader and must not hide behind its own signature — this
            # exact evasion was found by the screen's own negative control.
            if f"{PRODUCER}(" in line and not re.search(rf"\bfn\s+{PRODUCER}\b", line):
                stats["producer_sites"].append(
                    {"file": rel, "line": lineno, "function": fns.get(lineno, "?")}
                )
            if any(c in line for c in CONVERTERS) and not re.search(
                r"\bfn\s+(?:ticks_to_ns|ticks_to_axis_us|mask_ticks|mask)\b", line
            ):
                stats["converter_call_sites"].append(
                    {"file": rel, "line": lineno, "function": fns.get(lineno, "?")}
                )
            if lineno in excluded:
                continue
            if not TICK_TOKEN.search(line):
                continue
            stats["tick_lines"] += 1
            kind = classify_line(line)
            if kind is None:
                stats["converted_or_inert"] += 1
                continue
            findings.append(
                {
                    "file": rel,
                    "line": lineno,
                    "function": fns.get(lineno, "?"),
                    "kind": kind,
                    "text": " ".join(raw.splitlines()[lineno - 1].split()),
                    "code": " ".join(line.split()),
                }
            )
    return findings, stats, None


def check_single_producer(stats: dict, root: Path) -> list[str]:
    """R-B. Raw ticks enter the program in one place, and that place builds a calibration."""
    problems: list[str] = []
    sites = stats["producer_sites"]
    if not sites:
        problems.append(
            f"No call site of `{PRODUCER}(` was found anywhere in rust/src. Either the "
            "producer was renamed — in which case this rule is now screening a function "
            "that no longer exists and is asserting nothing (R12: UNOBSERVABLE, not a "
            "pass) — or raw ticks now enter the program somewhere this screen cannot see."
        )
        return problems
    if len(sites) > 1:
        listed = "\n".join(
            f"    {s['file']}:{s['line']} in fn {s['function']}" for s in sites
        )
        problems.append(
            f"`{PRODUCER}(` is called from {len(sites)} sites. Raw, unmasked ticks are "
            "supposed to enter the program exactly once, so that exactly one place is "
            "responsible for converting them:\n"
            f"{listed}\n"
            "  Each additional reader is a path that must convert on its own, and a path "
            "that forgets is invisible on every device CI can reach, because "
            "timestampPeriod is 1.0 on all of them."
        )
        return problems
    site = sites[0]
    path = root / site["file"]
    try:
        code = strip_comments_and_strings(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        problems.append(f"could not re-read {site['file']}: {exc!r}")
        return problems
    fns = enclosing_functions(code)
    body = [
        line
        for lineno, line in enumerate(code.splitlines(), start=1)
        if fns.get(lineno) == site["function"]
    ]
    if not any(CALIBRATION_TYPE in line for line in body):
        problems.append(
            f"{site['file']}:{site['line']} reads raw ticks in fn {site['function']}, and "
            f"that function does not construct a `{CALIBRATION_TYPE}`. The one place that "
            "reads unmasked ticks is the one place that must attach the period and the "
            "valid-bit mask to them; if it does not, nothing downstream can."
        )
    return problems


def check_allowlist_integrity(entries: list[dict], root: Path) -> list[str]:
    """R-C. An exemption whose site no longer exists is a blanket, not an exemption."""
    problems: list[str] = []
    cache: dict[str, list[str]] = {}
    for e in entries:
        path = root / e["file"]
        if e["file"] not in cache:
            if not path.is_file():
                problems.append(
                    f"allowlist entry for {e['file']} (fn {e['function']}): file does not "
                    "exist. The exemption is covering nothing and must be removed."
                )
                cache[e["file"]] = []
                continue
            cache[e["file"]] = strip_comments_and_strings(
                path.read_text(encoding="utf-8", errors="replace")
            ).splitlines()
        squashed = [" ".join(line.split()) for line in cache[e["file"]]]
        if not any(e["contains"] in line for line in squashed):
            problems.append(
                f"allowlist entry for {e['file']} fn {e['function']} pins the text "
                f"{e['contains']!r}, which no longer appears in that file. An exemption "
                "that has lost its site does not expire quietly: it becomes a blanket "
                "over whatever moved into its place. Re-pin it or delete it.\n"
                f"      reason on record: {e['reason']}\n"
                f"      owner: {e['owner']}"
            )
    return problems


def _summary(text: str, enabled: bool, title: str, error: bool) -> None:
    if not enabled:
        return
    one_line = " ".join(text.split())[:900]
    print(f"::{'error' if error else 'notice'} title={title}::{one_line}", flush=True)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"### Tick-conversion screen — {title}\n\n{text}\n\n")
    except OSError:
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="static screen for tick→duration conversions")
    p.add_argument("--root", default=str(REPO_ROOT))
    p.add_argument("--allowlist", default=str(ALLOWLIST_PATH))
    p.add_argument("--json", default="", help="write the full inventory here")
    p.add_argument("--github-summary", action="store_true")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    root = Path(args.root).resolve()
    entries, alerr = load_allowlist(Path(args.allowlist).resolve())
    if alerr:
        return report_instrument_error(
            "allowlist_unreadable",
            f"{alerr}\nThe screen refuses to run without it: with no allowlist every "
            "sanctioned converter body reports as a bypass, and a screen that is red for "
            "everything is a screen nobody reads.",
        )

    findings, stats, scanerr = scan(root)
    if scanerr:
        return report_instrument_error("source_tree_unreadable", scanerr)
    if stats["files"] == 0 or stats["tick_lines"] == 0:
        return report_instrument_error(
            "no_tick_sites_found",
            f"Scanned {stats['files']} .rs files under {root}/rust/src and found no line "
            "mentioning a tick or timestampPeriod at all. That is not a clean tree, it is "
            "a screen pointed at the wrong place or a renamed vocabulary — R12: report "
            "UNOBSERVABLE, never 0.",
        )

    unexplained = []
    used_entries = set()
    for f in findings:
        hit = None
        for idx, e in enumerate(entries):
            if entry_matches(e, f["file"], f["function"], f["code"]):
                hit = idx
                break
        if hit is None:
            unexplained.append(f)
        else:
            used_entries.add(hit)

    producer_problems = check_single_producer(stats, root)
    allowlist_problems = check_allowlist_integrity(entries, root)

    inventory = {
        "root": str(root),
        "files_scanned": stats["files"],
        "tick_bearing_lines": stats["tick_lines"],
        "converted_or_inert": stats["converted_or_inert"],
        "excluded_test_lines": stats["excluded_test_lines"],
        "sanctioned_sites": len(entries),
        "sanctioned_sites_matched": len(used_entries),
        "producer_sites": stats["producer_sites"],
        "converter_call_sites": stats["converter_call_sites"],
        "unexplained": unexplained,
    }
    if args.json:
        try:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(
                json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            print(f"TICK-SCREEN: could not write {args.json}: {exc!r}", flush=True)

    # R13 obligation 2: state the observation whether or not the check passes.
    print(
        f"TICK-SCREEN: frame — {stats['files']} .rs files, {stats['tick_lines']} "
        f"tick-bearing production lines, {stats['excluded_test_lines']} lines held out "
        "as #[cfg(test)] (UNOBSERVABLE by frame, not zero findings). "
        f"{len(entries)} sanctioned sites on record, {len(used_entries)} of them matched. "
        f"Producer `{PRODUCER}(` call sites: {len(stats['producer_sites'])}. "
        f"Converter call sites: {len(stats['converter_call_sites'])}.",
        flush=True,
    )

    if unexplained:
        quoted = "\n".join(
            f"  {f['file']}:{f['line']}  (fn {f['function']})\n"
            f"      {f['text']}\n"
            f"      → {f['kind']}"
            for f in unexplained
        )
        detail = (
            "Tick-valued arithmetic outside the sanctioned converters:\n\n"
            f"{quoted}\n\n"
            "Each of these either converts through "
            "`GpuTimestampCalibration::ticks_to_ns` / `ticks_to_axis_us`, or is entered "
            f"in {Path(args.allowlist).name} with a reason and an owner. There is no "
            "third option, and that is deliberate: this defect is invisible on every "
            "device in CI, because timestampPeriod is 1.0 on NVIDIA, on lavapipe and on "
            "SwiftShader, so the conversion is the identity there and a path that skips "
            "it produces byte-identical output. The screen is the only arm that can see "
            "it."
        )
        _summary(detail, args.github_summary, "tick conversion bypassed", True)
        return report_fail("tick_conversion_bypassed", detail)

    if producer_problems:
        detail = "\n".join(f"  - {p}" for p in producer_problems)
        _summary(detail, args.github_summary, "raw tick producer not unique", True)
        return report_fail("raw_tick_producer_not_unique", detail)

    if allowlist_problems:
        detail = "\n".join(f"  - {p}" for p in allowlist_problems)
        _summary(detail, args.github_summary, "allowlist has lost its site", True)
        return report_fail("allowlist_entry_without_a_site", detail)

    detail = (
        f"{stats['tick_lines']} tick-bearing production lines across {stats['files']} "
        f"files; every one either calls a sanctioned converter, moves a tick without "
        f"scaling it, or is one of {len(used_entries)} sites recorded in "
        f"{Path(args.allowlist).name} with a reason and an owner. Raw ticks enter the "
        f"program at exactly one site ({stats['producer_sites'][0]['file']}:"
        f"{stats['producer_sites'][0]['line']}) and that function builds the "
        f"{CALIBRATION_TYPE}.\n"
        "  What this claims: no source path in rust/src scales a device tick without the "
        "period and the valid-bit mask.\n"
        "  What it does not claim: that the conversion is arithmetically correct — that "
        "is the unit tests' job (trace.rs) and this screen would pass over a wrong "
        "formula inside a sanctioned converter. The two arms are complementary and "
        "neither is the other."
    )
    _summary(detail, args.github_summary, "no tick conversion bypassed", False)
    return report_pass(detail)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
