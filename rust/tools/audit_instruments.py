"""Instrument census: is every reporting mechanism in the production call graph?

R10 (Morpheus, §10.0.1): a mechanism that exists in the source tree and not in the call
graph is indistinguishable from one that was never written, and review cannot tell them
apart. This is the screen that tells them apart.

# The four states

    absent       no listener exists at all
    uninvoked    exists, never called from production code (tests do not count)
    unreachable  called, but its output goes where nothing reads it
    misnamed     wired, invoked, correct — and its name misdescribes its content

Only **uninvoked** is decidable from text, so only that one is automated here. `absent`
needs a spec, `unreachable` needs to know who reads which artifact, and `misnamed` needs
to know what the name promises. Those three are in `.squad/decisions/inbox/` as a hand
census; this script keeps the one machine-checkable column honest and, crucially, makes
the census *fail* when it drifts rather than quietly ageing.

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
    python rust/tools/audit_instruments.py --check    # exit 1 on drift
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
BASELINE = HERE.parent / "instrument_census.json"

# Files whose `pub fn`s are the instruments under audit. Everything the EP emits about
# itself is produced from one of these.
INSTRUMENT_FILES = ["counters.rs", "trace.rs", "ops/claim_log.rs", "allocator.rs", "transfer.rs"]

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


def main(argv: list[str]) -> int:
    if self_test():
        print("FAIL: the comment/string stripper is broken; census not run.", file=sys.stderr)
        return 1
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

    if "--write-baseline" in argv:
        base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        base["uninvoked"] = found
        base["ambiguous"] = sorted(f"{r['file']}::{r['fn']}" for r in ambiguous)
        BASELINE.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {BASELINE} ({len(found)} uninvoked)")
        return 0

    if "--check" not in argv:
        return 0

    if not BASELINE.exists():
        print(f"\nFAIL: no baseline at {BASELINE}", file=sys.stderr)
        return 1
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
    if new or gone:
        return 1
    print(f"\nOK: uninvoked set matches the baseline ({len(found)} known).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
