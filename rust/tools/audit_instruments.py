"""Instrument census: is every reporting mechanism in the production call graph?

R10 (Morpheus, §10.0.1): a mechanism that exists in the source tree and not in the call
graph is indistinguishable from one that was never written, and review cannot tell them
apart. This is the screen that tells them apart.

# The six states, ordered by how late the failure is discoverable

    absent       no listener exists at all
    uninvoked    exists, never called from production code (tests do not count)
    unfalsified  called, and nothing has ever observed it produce BOTH answers, so a
                 guard that always passes, always crashes, or has inverted polarity is
                 indistinguishable from a working one
    unreachable  called, but its output goes where nothing reads it — including a guard
                 that raises before it reads its input, which is invoked and never in a
                 position to report
    out-of-frame wired, invoked, correct — and the event it counts *cannot occur* in the
                 frame it observes, so its honest report is `UNOBSERVABLE`, never `0` (R12)
    misnamed     wired, invoked, correct — and its name misdescribes its content (R11)

Two are decidable from text and are automated here: **uninvoked** (Rust and harness) and
**unfalsified** (harness, from the test AST). `absent`, `unreachable`, `out-of-frame` and
`misnamed` need a spec, a reader, a frame and a promise respectively; they live in
`instrument_census.json` as a hand census.

# RULING (Tank, 2026-08-01): R13 does NOT add a seventh state. It is the reporting layer.

The question was whether Guard D — which raised `NameError` before reading a single
profiling event, went red, and was reported as working — is a new census state.

It is not, and the reason is worth stating because the classification is load-bearing.
Ask what the state is a property OF. All six above are properties of **an instrument's
position in the system**: does it exist, is it called, has it been seen to discriminate,
does anything read it, can its event occur, does its name match. Guard D's own position
is already covered: invoked, never in a position to report = `unreachable`, the state
whose whole definition is "ran, produced nothing observable".

What R13 names is a property of **the channel the verdict travels down**: pytest's
summary line has a two-token alphabet (`PASS`, `FAILED`) and was carrying three states.
That confusion is not specific to `unreachable`. An `out-of-frame` counter quoted through
a two-token gate is misread exactly as badly; so is a `misnamed` one. R13 applies to every
row of this census at once, which is precisely what makes it a different axis and not a
seventh row. A state that applies to all states is not a state.

The consequence is mechanical rather than taxonomic, and it binds this script:

  **This census reports three terminal tokens, never two.** `PASS` (exit 0), `FAIL(drift)`
  (exit 1) — the condition it exists to detect — and `ERROR(instrument)` (exit 2), in which
  the census did not reach its observation. A traceback and a drift are different findings;
  before this, both left through the same door as "non-zero exit". `subprocess.TimeoutExpired`
  in a caller of this script is `ERROR(instrument)`, never a detection, and a lane that
  records one has not run the census whatever else it reports.

# Why it compares against a checked-in baseline instead of just printing

A census that prints a list is read once. `--check` compares against
`instrument_census.json` and exits 1 when a new uninvoked instrument appears or a known
one gets wired, so the list cannot rot silently. That is the same reason
`alloc_device_authoritative_spans` has a ceiling counter beside it: a number nobody is
forced to look at is a number nobody looks at.

# Known limits, stated because a screen that hides its blind spots is worse than none

  * Textual. Trait-object and function-pointer dispatch are invisible to it.
  * A name shared with an unrelated function elsewhere reads as "wired" (false negative).
    Mitigated by reporting bare-name references separately from call-shaped ones, so
    `.map(f)`-style uses are visible rather than being scored as dead.
  * It cannot see whether a wired instrument reports the *right thing* — `Phase::Record`
    passes this screen cleanly and was wrong by a factor of fifty.

Usage:
    python rust/tools/audit_instruments.py            # print the screen
    python rust/tools/audit_instruments.py --check    # 0 PASS / 1 FAIL(drift) / 2 ERROR
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
REPO = HERE.parents[2]
TESTS = REPO / "tests"
BASELINE = HERE.parent / "instrument_census.json"

# Files whose `pub fn`s are the instruments under audit. Everything the EP emits about
# itself is produced from one of these.
#
# `vk/host_device_memory.rs` is here for one function: `offer_shared_device`, the §6.5 seam.
# It is UNWIRED by construction until Switch calls it, and the whole point of R10 is that a
# seam nobody calls is indistinguishable from one that was never written. Screening it means
# the day it acquires a caller, this baseline goes red and somebody has to look.
INSTRUMENT_FILES = [
    "counters.rs",
    "trace.rs",
    "ops/claim_log.rs",
    "allocator.rs",
    "transfer.rs",
    "vk/host_device_memory.rs",
]

FN = re.compile(r"^\s*pub(?:\(\w+\))? fn ([a-z_][a-z0-9_]*)\s*[(<]")

# Comments must be stripped before counting references, or a doc comment mentioning an
# instrument makes it look wired. Found the hard way: writing "`Tracer::record_path()` has
# no production caller" in a comment removed `record_path` from this screen's own output.
# An instrument that a mention of its own deadness marks as alive is not an instrument.
RAW_STRING = re.compile(r'r(#*)"')


def strip_comments(text: str) -> str:
    """Remove comments AND string literals in a single left-to-right pass.

    A single pass is not fussiness. The first cut stripped comments with a regex and then
    strings with another, and a `//` inside a string literal left an unterminated quote that
    swallowed hundreds of lines of real code — silently reclassifying six WIRED counters as
    uninvoked. A screen that mis-scores in the *dead* direction is worse than no screen: it
    sends someone to wire something that is already wired.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
        elif c == "r" and (m := RAW_STRING.match(text, i)):
            close = '"' + m.group(1)
            j = text.find(close, m.end())
            i = n if j < 0 else j + len(close)
        elif c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
        elif c == "'" and i + 2 < n and (text[i + 2] == "'" or text.startswith("\\", i + 1)):
            # A char literal, not a lifetime: lifetimes are never closed by a quote.
            j = text.find("'", i + 1 + (2 if text[i + 1] == "\\" else 0))
            i = i + 1 if j < 0 else j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


# `*_for_test` helpers are test scaffolding by name and contract; they are supposed to
# have no production caller, and flagging them buries the real findings.
EXEMPT = re.compile(r"(^|_)(test|for_test)(_|$)|_for_test$|^test_")


def split_tests(text: str) -> tuple[str, str]:
    """Return (production, test) halves, split at the `#[cfg(test)] mod tests` marker.

    Comments are stripped from both halves: a reference inside a comment is prose, not a
    call graph edge.
    """
    m = re.search(r"^#\[cfg\(test\)\]\s*\nmod tests", text, re.M)
    if not m:
        return strip_comments(text), ""
    return strip_comments(text[: m.start()]), strip_comments(text[m.start() :])


def survey() -> list[dict]:
    bodies = {
        f: split_tests(f.read_text(encoding="utf-8", errors="replace"))
        for f in sorted(SRC.rglob("*.rs"))
    }

    rows: list[dict] = []
    for rel in INSTRUMENT_FILES:
        path = SRC / rel
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = FN.match(line)
            if not m:
                continue
            name = m.group(1)
            if EXEMPT.search(name):
                continue
            call = re.compile(rf"\b{re.escape(name)}\s*\(")
            bare = re.compile(rf"\b{re.escape(name)}\b")
            prod_calls = prod_refs = test_refs = 0
            for f, (p, t) in bodies.items():
                own = f == path
                prod_calls += len(call.findall(p)) - (1 if own else 0)
                prod_refs += len(bare.findall(p)) - (1 if own else 0)
                test_refs += len(bare.findall(t))
            # A definition line matches its own `name(`; the subtraction above removes it.
            if prod_calls <= 0 and prod_refs <= 0:
                state = "uninvoked"
            elif prod_calls <= 0:
                # Referenced but never called: passed as a function value (`.map(f)`).
                state = "wired-by-reference"
            else:
                state = "wired"
            rows.append(
                {
                    "file": rel,
                    "line": line_no,
                    "fn": name,
                    "prod_calls": max(prod_calls, 0),
                    "prod_refs": max(prod_refs, 0),
                    "test_refs": test_refs,
                    "state": state,
                }
            )
    return rows


def uninvoked(rows: list[dict]) -> list[str]:
    return sorted(f"{r['file']}::{r['fn']}" for r in rows if r["state"] == "uninvoked")


# ===========================================================================
# HARNESS DOMAIN (tests/) — added by Trinity, 2026-07-31
# ===========================================================================
#
# WHY THIS IS IN TANK'S FILE AND NOT A SECOND SCRIPT
# --------------------------------------------------
# A census whose answer depends on which of two censuses you ran is not a census.
# The harness lives in a different language and has a different call graph, but the
# question is identical — "is this instrument in the call graph, and has anything ever
# observed it produce a varying artifact?" — so it gets the same baseline file, the same
# `--check` drift semantics and the same five-state vocabulary.  One census, two domains.
#
# WHY `uninvoked` ALONE WOULD NOT HAVE CAUGHT GUARD D
# ---------------------------------------------------
# `assert_vulkan_executed_runtime` HAD four production callers from the day it landed.
# The Rust screen's question would have answered "wired" and been right.  It raised
# `NameError` at its first statement for its entire life and never read a profiling event,
# because every one of those callers sat behind a GPU gate and, when they finally ran, the
# crash was read as the guard firing.  So the harness domain needs one more machine-checkable
# state, later in Tank's discoverability ordering than `uninvoked`:
#
#     unfalsified  called, but no always-on test has observed it in BOTH polarities, so a
#                  guard that always passes, always crashes, or has inverted polarity is
#                  indistinguishable from a working one.
#
# It is decided from the test AST: an instrument is screened iff some test that is NOT
# GPU-gated calls it inside `pytest.raises(...)` (reject polarity) AND some non-gated test
# calls it outside one (accept polarity).  Both are required: a reject-only suite certifies
# a guard that rejects everything, and an accept-only suite certifies a guard that never
# rejects anything.  Guard D had neither and would have been red from the first commit.
#
# The blind spot, stated: this cannot see whether the polarity test's INPUT actually varies
# the thing under test (`test_guard_d.py` earns that by mutation, not by this screen), and
# a guard whose falsifier needs real hardware cannot be screened here at all — those are
# listed by hand under `hand.harness_notes` with the reason they are unscreenable.

# Files whose module-level functions are the harness instruments under audit.
HARNESS_INSTRUMENT_FILES = ["ops/_models.py"]

# A harness instrument is a function that renders a verdict: it either raises on a bad
# world or returns a number a gate reads.  Helpers that only build models or run sessions
# are not instruments and are excluded by name.
HARNESS_FN = re.compile(r"^(assert_|count_|check$|check_|require_|verify_|expect_)|_verdict$")

# Decorators / fixtures that mean "this test does not run in the always-on lane".
HARNESS_GATE = re.compile(r"require_vulkan|skipif|\bskip\b|xfail|slow|gpu|require_model")


def _harness_instruments(tests_root=None, files=None) -> dict[str, str]:
    """Return {fn_name: "file::fn"} for every harness instrument."""
    import ast as _ast

    tests_root = TESTS if tests_root is None else Path(tests_root)
    files = HARNESS_INSTRUMENT_FILES if files is None else files
    out: dict[str, str] = {}
    for rel in files:
        path = tests_root / rel
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if HARNESS_FN.search(node.name):
                    out[node.name] = f"tests/{rel}::{node.name}"
    return out


def _is_gated(fn) -> bool:
    """True if *fn* (an ast.FunctionDef) is skipped/gated out of the always-on lane."""
    import ast as _ast

    for dec in fn.decorator_list:
        if HARNESS_GATE.search(_ast.dump(dec)):
            return True
    for arg in fn.args.args:
        if HARNESS_GATE.search(arg.arg):
            return True
    return False


def harness_survey(tests_root=None, files=None) -> list[dict]:
    """Screen every harness instrument for callers and for two-polarity coverage.

    *tests_root* and *files* are parameters rather than constants so this screen can be
    pointed at a synthetic tree and watched to disagree — see
    ``tests/ops/test_harness_census.py``.  A screen that has only ever been run against the
    real repository, where it happens to print a plausible answer, is precisely the Guard D
    shape it exists to catch.
    """
    import ast as _ast

    tests_root = TESTS if tests_root is None else Path(tests_root)
    files = HARNESS_INSTRUMENT_FILES if files is None else files
    names = _harness_instruments(tests_root, files)
    stats = {n: {"calls": 0, "reject": 0, "accept": 0} for n in names}

    owner_files = {tests_root / rel for rel in files}
    # Calls from inside the owner module count as callers (an instrument invoked at import
    # time, like the Q/DQ oracle probe, is wired) but can never supply a polarity: polarity
    # is a property of a test that was written to watch it disagree.
    for path in sorted(owner_files):
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, _ast.Attribute)
                else func.id
                if isinstance(func, _ast.Name)
                else None
            )
            if name in stats:
                stats[name]["calls"] += 1

    for path in sorted(tests_root.rglob("*.py")):
        if path in owner_files:
            continue
        try:
            tree = _ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for fn in [n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)]:
            gated = _is_gated(fn)
            # Map every node in this function to whether it sits inside `pytest.raises`.
            raising: set[int] = set()
            for node in _ast.walk(fn):
                if isinstance(node, _ast.With):
                    if any(
                        "raises" in _ast.dump(item.context_expr) for item in node.items
                    ):
                        for inner in _ast.walk(node):
                            raising.add(id(inner))
            for node in _ast.walk(fn):
                if not isinstance(node, _ast.Call):
                    continue
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, _ast.Attribute)
                    else func.id
                    if isinstance(func, _ast.Name)
                    else None
                )
                if name not in stats:
                    continue
                stats[name]["calls"] += 1
                if gated:
                    continue
                if id(node) in raising:
                    stats[name]["reject"] += 1
                else:
                    stats[name]["accept"] += 1

    rows: list[dict] = []
    for name, qual in sorted(names.items()):
        s = stats[name]
        if s["calls"] == 0:
            state = "uninvoked"
        elif s["reject"] and s["accept"]:
            state = "screened"
        else:
            state = "unfalsified"
        rows.append({"id": qual, "fn": name, "state": state, **s})
    return rows


def harness_report(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Print the harness screen; return (uninvoked, unfalsified) id lists."""
    print()
    print("HARNESS INSTRUMENT SCREEN (tests/ — a guard nothing falsifies is not a guard)")
    print(f"  scanned {len(rows)} harness instrument fn(s) in {HARNESS_INSTRUMENT_FILES}")
    print()
    un = sorted(r["id"] for r in rows if r["state"] == "uninvoked")
    nf = sorted(r["id"] for r in rows if r["state"] == "unfalsified")
    for r in rows:
        if r["state"] == "screened":
            continue
        label = "UNINVOKED  " if r["state"] == "uninvoked" else "UNFALSIFIED"
        print(
            f"  {label} {r['id']:<58} calls={r['calls']} "
            f"reject_polarity={r['reject']} accept_polarity={r['accept']}"
        )
    scr = [r for r in rows if r["state"] == "screened"]
    print()
    for r in scr:
        print(
            f"  SCREENED   {r['id']:<58} calls={r['calls']} "
            f"reject_polarity={r['reject']} accept_polarity={r['accept']}"
        )
    print()
    print("  UNFALSIFIED is not a bug report; it is the absence of one. It says only that")
    print("  nothing in the always-on lane has ever watched this instrument disagree, so a")
    print("  broken one and a working one would look the same. Guard D lived here for its")
    print("  whole life while the Rust screen's question ('has it got a caller?') said WIRED.")
    return un, nf


def self_test() -> int:
    """The stripper gets its own falsifier, because its failure mode is silent.

    Each case is one way the first cut was wrong. `--self-test` runs before `--check` in the
    main path so a broken stripper cannot report a clean census.
    """
    cases = [
        ('let u = "http://x"; foo();', "foo("),  # `//` inside a string is not a comment
        ('bar(); // baz()\nfoo();', "baz("),  # a call named in a comment is not a call
        ('let s = "baz()"; foo();', "baz("),  # ...nor one named in a string
        ('let s = r#"baz() "# ; foo();', "baz("),  # ...nor in a raw string
        ("let c = '\\''; foo();", "foo("),  # escaped char literal must not eat the rest
        ("fn f<'a>(x: &'a str) { foo(); }", "foo("),  # a lifetime is not a char literal
        ("/* baz() /* nested */ */ foo();", "baz("),  # nested block comments
    ]
    bad = 0
    for src, needle in cases:
        stripped = strip_comments(src)
        present = needle in stripped
        # The needle is expected present only in the two cases whose needle is `foo(`.
        want = needle == "foo("
        if present != want:
            bad += 1
            print(f"  SELF-TEST FAIL: {src!r} -> {stripped!r} ({needle} present={present})")
        elif want and "foo(" not in stripped:
            bad += 1
    print(f"  stripper self-test: {len(cases) - bad}/{len(cases)} cases pass")
    return 1 if bad else 0


class CensusInstrumentError(RuntimeError):
    """The census did not reach its observation. R13: never a detection."""


def main(argv: list[str]) -> int:
    if self_test():
        # ERROR(instrument), not FAIL: a broken stripper means the census never reached its
        # observation. Raising rather than returning 1 keeps that distinction out of the
        # caller's hands — see `main_guarded`.
        raise CensusInstrumentError(
            "the comment/string stripper is broken; census not run and nothing was screened"
        )
    rows = survey()
    found = uninvoked(rows)

    print("INSTRUMENT WIRING SCREEN (uninvoked = no production caller; tests do not count)")
    print(f"  scanned {len(rows)} public instrument fn(s) across {len(INSTRUMENT_FILES)} file(s)")
    print()
    if not found:
        print("  no uninvoked instruments")
    for r in rows:
        if r["state"] != "uninvoked":
            continue
        print(
            f"  UNINVOKED  {r['file']}:{r['line']:<5} {r['fn']:<24}"
            f" prod_refs={r['prod_refs']} test_refs={r['test_refs']}"
        )
    print()
    ambiguous = [r for r in rows if r["state"] == "wired-by-reference"]
    if ambiguous:
        print("  REFERENCED BUT NEVER CALL-SHAPED (`.map(f)`, or a same-named local/param elsewhere —")
        print("  this class needs a human; `claim_log::record` lives here because `logging.rs` has a")
        print("  parameter named `record`, which is not a call to it):")
        for r in ambiguous:
            print(
                f"  AMBIGUOUS  {r['file']}:{r['line']:<5} {r['fn']:<24}"
                f" prod_refs={r['prod_refs']} test_refs={r['test_refs']}"
            )
        print()
    print("  NOTE: this screen cannot see whether a WIRED instrument reports the right thing.")
    print("  Phase::Record passed it cleanly while 96% of its time was a memcpy nested inside it.")

    h_rows = harness_survey()
    h_uninvoked, h_unfalsified = harness_report(h_rows)

    if "--write-baseline" in argv:
        base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        base["uninvoked"] = found
        base["ambiguous"] = sorted(f"{r['file']}::{r['fn']}" for r in ambiguous)
        base["harness_uninvoked"] = h_uninvoked
        base["harness_unfalsified"] = h_unfalsified
        BASELINE.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {BASELINE} ({len(found)} uninvoked, {len(h_unfalsified)} unfalsified)")
        return 0

    if "--check" not in argv:
        return 0

    if not BASELINE.exists():
        raise CensusInstrumentError(
            f"no baseline at {BASELINE}; the comparison input is missing, so no drift was "
            "observed either way"
        )
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = sorted(base["uninvoked"])
    new = [x for x in found if x not in expected]
    gone = [x for x in expected if x not in found]
    if new:
        print(f"\nFAIL: {len(new)} NEW uninvoked instrument(s):", file=sys.stderr)
        for x in new:
            print(f"  + {x}", file=sys.stderr)
    if gone:
        print(
            f"\nFAIL: {len(gone)} instrument(s) got wired — good news, update the baseline:",
            file=sys.stderr,
        )
        for x in gone:
            print(f"  - {x}", file=sys.stderr)

    # Harness domain. Drift is checked in BOTH directions for the same reason as above:
    # a newly unfalsified guard is a hole, and a newly screened one must be recorded or the
    # baseline slowly stops meaning anything.
    h_bad = False
    for key, current in (
        ("harness_uninvoked", h_uninvoked),
        ("harness_unfalsified", h_unfalsified),
    ):
        if key not in base:
            print(f"\nFAIL: baseline has no `{key}`; run --write-baseline.", file=sys.stderr)
            h_bad = True
            continue
        exp = sorted(base[key])
        added = [x for x in current if x not in exp]
        removed = [x for x in exp if x not in current]
        if added:
            h_bad = True
            print(f"\nFAIL: {len(added)} NEW {key[8:]} harness instrument(s):", file=sys.stderr)
            for x in added:
                print(f"  + {x}", file=sys.stderr)
            print(
                "  A harness instrument with no two-polarity self-test is the Guard D shape.\n"
                "  Give it one in the always-on lane (see tests/ops/test_guard_d.py), or add it\n"
                "  to the baseline WITH a hand note in `hand.harness_notes` saying why it cannot\n"
                "  be falsified without hardware.",
                file=sys.stderr,
            )
        if removed:
            h_bad = True
            print(
                f"\nFAIL: {len(removed)} harness instrument(s) left `{key[8:]}` — "
                "good news, update the baseline:",
                file=sys.stderr,
            )
            for x in removed:
                print(f"  - {x}", file=sys.stderr)

    if new or gone or h_bad:
        print("\nCENSUS VERDICT: FAIL(drift)")
        return 1
    print(f"\nOK: uninvoked set matches the baseline ({len(found)} known).")
    print(
        f"OK: harness screen matches the baseline "
        f"({len(h_uninvoked)} uninvoked, {len(h_unfalsified)} unfalsified)."
    )
    print("\nCENSUS VERDICT: PASS")
    return 0


# §10.0.1 R13 — three terminal tokens, never two.
#
# Before this wrapper, a drift and a traceback both left through the same door: "non-zero
# exit". That is the Guard D shape applied to the census itself — a mechanism whose outage
# is spelled the same way as its finding — and it would have been the harder specimen,
# because the census is the thing everyone else's evidence rests on.
#
# The token is printed on its own line AND encoded in the exit code, because a caller that
# reads only one of the two must still get three states:
#
#     0  PASS               the census ran and the baseline matches
#     1  FAIL(drift)        the census ran and found the condition it exists to detect
#     2  ERROR(instrument)  the census did not reach its observation. NEVER a detection.
#
# `subprocess.TimeoutExpired` in a caller of this script is ERROR(instrument) and belongs
# in the same bucket as an exception here: it is a lane failure of a different kind, and a
# lane that records one has not run the census, whatever else it reports.
EXIT_PASS = 0
EXIT_FAIL_DRIFT = 1
EXIT_ERROR_INSTRUMENT = 2


def main_guarded(argv: list[str]) -> int:
    """Run `main` and translate any escape into `ERROR(instrument)` rather than a FAIL."""
    try:
        rc = main(argv)
    except Exception as exc:  # noqa: BLE001 — the whole point is to catch everything
        import traceback

        traceback.print_exc()
        print(
            f"\nCENSUS VERDICT: ERROR(instrument) — {type(exc).__name__}: {exc}\n"
            "  The census did NOT reach its observation, so it detected nothing. Quote this "
            "text, not an exit code and not a failure count (R13).",
            file=sys.stderr,
        )
        return EXIT_ERROR_INSTRUMENT
    if rc not in (EXIT_PASS, EXIT_FAIL_DRIFT):
        print(
            f"\nCENSUS VERDICT: ERROR(instrument) — main() returned {rc}, which is not one of "
            "the two states it is allowed to return.",
            file=sys.stderr,
        )
        return EXIT_ERROR_INSTRUMENT
    return rc


if __name__ == "__main__":
    sys.exit(main_guarded(sys.argv[1:]))
