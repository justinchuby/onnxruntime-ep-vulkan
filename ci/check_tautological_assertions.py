#!/usr/bin/env python3
"""``check_tautological_assertions`` — assertions whose two sides are the same text.

READ THIS FIRST: WHAT THIS DOES NOT CATCH
=========================================
This screen exists because I wrote two assertions in one day that could only succeed, and
the coordinator observed — correctly — that the version to worry about is the one nobody
looks at, so a mechanical screen is worth more than either individual fix.

**Neither of those two assertions is detectable by this screen, and it is important that
that is the first thing written down rather than the last.**

  * ``test_localise_inherits_the_level_blindness_hole`` compared two *different
    expressions* that both happened to evaluate to exactly ``0.0``.  Textually the two
    sides differ.  Only running it reveals that its reading cannot move.
  * The ``fn_addr_eq`` tests asserted a predicate true several times and never once
    asserted it false.  Every individual assertion there is fine; the *set* has no
    negative polarity.  That is a property of a test function, not of a line.

So the class this screen covers is **strictly, and substantially, smaller** than "an
assertion that cannot fail".  It covers exactly one mechanical form: the two compared
sides are the same source text, or are both literal constants.  A ``PASS`` here means that
form is absent.  It does not mean the suite's assertions can fail, and the output says so
on every path — because a detection with a false description is worse than no description
(Tank, D-T85: prose added to make a counter honest became the thing asserting a
falsehood).

WHAT THE CENSUS SAYS, AND WHY THE SCREEN SHIPPED ANYWAY
=======================================================
Run over the tree at the time of writing, across Rust and Python sources:

    1,056 comparison assertions scanned (rs=614, py=442), **0 detections**

A screen with no catches is a screen with no evidence that it works (R9: evidence scales
with falsifying instruments, not agreeing ones).  Two consequences, both acted on:

  1. Its falsifier is planted, not observed.  ``ci/test_lane_checks.py`` runs it over
     deliberately-written tautologies and asserts each is caught, so a refactor that
     neuters the scanner fails too.  That is the permanent form of "plant a violation and
     check CI goes red", and it is the *only* reason this screen is known to be a screen.
  2. Its claim is scoped to **regression**, not discovery.  It found nothing; it exists so
     that the form cannot arrive later unnoticed.  That is a smaller claim than the one
     the screen's name suggests, which is why the name is not the definition (R11).

If a future run reports detections, that is new information.  If it keeps reporting zero,
that is the expected state and not evidence of health.

THREE DEFECTS THIS SCREEN HAD BEFORE IT HAD A CATCH
===================================================
Recorded because a screen nobody has seen fail is exactly the thing this screen is about,
and its own development produced three instances of the failure it hunts:

  1. **It reported ``PASS`` over a language it had not read.**  A leading-whitespace bug in
     ``PY_ASSERT`` meant 89 Python files yielded **zero** assertions, and the total was
     non-zero only because Rust carried it.  The remedy is the ``language_scanned_nothing``
     outage below: coverage is now asserted **per language**, because a total that another
     language paid for is not coverage.
  2. **Blanking string literals invented three false positives.**  ``frame["a"] ==
     frame["b"]`` blanks to a term compared to itself.  Three of the scanner's first four
     detections were this, and all three were correct code.  See ``_placeholder``.
  3. **Polarity was missing, so an idiom was reported as a defect.**  ``assert
     empty.median != empty.median`` is the NaN test in ``bench/test_harness.py``.  See
     ``_classify``: the hazard is *passing without reading the subject*, not sameness.

Defect 2 and defect 3 together are the point.  Four confident detections, four wrong.
An unscoped screen does not merely miss things; it asserts things.

THE TWO DETECTORS
=================
``IDENTICAL_OPERANDS``
    The two compared sides normalise to the same source text, **under equality only**.
    ``assert_eq!(x, x)``, ``assert a.b() == a.b()``.  Reads its subject once and compares
    it to itself, and therefore always passes.  Under *inequality* identical operands
    either always fail — safe, because a permanently red assertion is fixed on its first
    run — or are a deliberate NaN probe, so that polarity is not reported.

``BOTH_LITERAL``
    Both sides are literal constants: ``assert_eq!(0.0, 0.0)``, ``assert 1 == 1``.  Reads
    no subject at all.  Reported at **both** polarities, because ``assert_ne!(1, 2)``
    also passes without touching the code under test.

Comments and string literals are neutralised before scanning in both languages.  That is
required in the other direction too: ``rust/tests/layering.rs`` contains
``assert_eq!(1, 1)`` *inside a string*, as fixture text for the layering lint's own
planted-violation test.  Neutralising has to hide an assertion *inside* a string without
making two assertions *about* different strings look identical — hence ``_placeholder``
rather than blanking.  Both directions are covered by tests.

VOCABULARY
==========
Same three-way vocabulary and exit codes as ``ci/check_device_state.py``:

    0  ASSERTIONS: PASS(scanned=<n>, files=<n>)
    1  ASSERTIONS: FAIL(tautological_assertions=<n>)
    4  ASSERTIONS: ERROR(instrument=<token>)

``ERROR`` is reserved for the screen being unable to look: an unreadable tree, no source
files at all, no assertions at all, or a **language present in the tree that yielded
none**.  A screen that examined nothing must not report the same word as a screen that
examined everything and found it clean.

USAGE
=====
    python ci/check_tautological_assertions.py                # scan the repository
    python ci/check_tautological_assertions.py --root <dir>   # scan somewhere else
    python ci/check_tautological_assertions.py --verbose      # list every detection
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_ERROR_INSTRUMENT = 4

RUST_MACRO = re.compile(r"\b(?:debug_)?assert_(?:eq|ne)!\s*\(")
PY_ASSERT = re.compile(r"(?:^\s*|[;:]\s*)assert\s+(?P<body>.+)$")

#: A literal constant in either language.  Deliberately conservative: an identifier that
#: merely *looks* constant (``MAX``, ``EPSILON``) is not matched, because a named constant
#: on both sides is usually a legitimate cross-check of two different names.
LITERAL = re.compile(
    r"""^(?:
        [-+]?[0-9][0-9_]*(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?
            (?:f32|f64|u8|u16|u32|u64|usize|i8|i16|i32|i64|isize)?
      | 0x[0-9a-fA-F_]+(?:u\d+|i\d+|usize|isize)?
      | true | false | True | False | None
      | \[\] | \{\} | \(\)
    )$""",
    re.VERBOSE,
)

SKIP_DIRS = {"target", ".git", "node_modules", "__pycache__", ".venv", "venv", "build"}


def _placeholder(content: str) -> str:
    """A same-length, newline-preserving stand-in that keeps *distinct* strings distinct.

    Blanking string literals to spaces was the obvious implementation and it was wrong in a
    way that produced confident false positives::

        assert frame["dispatched_devices"] == frame["capable_devices"]

    blanks to ``frame[                    ] == frame[                 ]`` — which after
    whitespace normalisation is a term compared to itself.  Three of the scanner's first
    four "detections" were this, and all three were correct code.  Stripping has to make
    an assertion *inside* a string invisible without making two assertions *about*
    different strings identical.

    So the content is replaced by a repetition of its own digest: same length, same
    newlines, no parseable syntax, and different literals stay different.
    """
    digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
    keep = "".join("\n" if ch == "\n" else "" for ch in content)
    filler = (digest * (len(content) // len(digest) + 1))[: len(content) - len(keep)]
    out, i = [], 0
    for ch in content:
        if ch == "\n":
            out.append("\n")
        else:
            out.append(filler[i])
            i += 1
    return "".join(out)


@dataclass(frozen=True)
class Detection:
    kind: str
    path: str
    line: int
    text: str

    def render(self) -> str:
        return f"  {self.kind:16s} {self.path}:{self.line}\n      {self.text}"


def strip_rust(src: str) -> str:
    """Blank out comments, string literals and char literals, preserving line structure.

    Preserving newlines matters: line numbers in a detection have to point at the real
    line, and a stripper that collapsed the text would report a number that reads like
    evidence and is wrong.
    """
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            depth = 1
            out.append("  ")
            i += 2
            while i < n and depth:
                if src.startswith("/*", i):
                    depth += 1
                    out.append("  ")
                    i += 2
                elif src.startswith("*/", i):
                    depth -= 1
                    out.append("  ")
                    i += 2
                else:
                    out.append("\n" if src[i] == "\n" else " ")
                    i += 1
            continue
        if c == "r" and i + 1 < n and src[i + 1] in "#\"":
            j = i + 1
            hashes = 0
            while j < n and src[j] == "#":
                hashes += 1
                j += 1
            if j < n and src[j] == '"':
                closing = '"' + "#" * hashes
                end = src.find(closing, j + 1)
                end = n if end < 0 else end + len(closing)
                out.append(_placeholder(src[i:end]))
                i = end
                continue
        if c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(_placeholder(src[i:j]))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def strip_python(src: str) -> str:
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "#":
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c in "\"'":
            triple = src[i : i + 3]
            if triple in ('"""', "'''"):
                end = src.find(triple, i + 3)
                end = n if end < 0 else end + 3
            else:
                j = i + 1
                while j < n:
                    if src[j] == "\\":
                        j += 2
                        continue
                    if src[j] == c or src[j] == "\n":
                        j += 1
                        break
                    j += 1
                end = j
            out.append(_placeholder(src[i:end]))
            i = end
            continue
        out.append(c)
        i += 1
    return "".join(out)


def split_top_level(text: str) -> list[str]:
    """Split a macro argument list on top-level commas, stopping at the closing paren."""
    args, cur, depth = [], "", 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                args.append(cur)
                return args
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(cur)
            cur = ""
            continue
        cur += ch
    args.append(cur)
    return args


def normalise(expr: str) -> str:
    return re.sub(r"\s+", "", expr)


def _classify(a: str, b: str, equality: bool) -> str | None:
    """Name the hazard, which is **passing without reading the subject** — not sameness.

    That definition is what makes polarity matter, and it took a real detection to see it.
    ``assert empty.median != empty.median`` is the idiomatic NaN test in
    ``bench/test_harness.py``: identical operands, and correct code.  Identical operands
    under *inequality* either always fail (safe: a permanently red assertion gets fixed on
    its first run) or are a deliberate NaN probe.  Neither is the hazard.  Under
    *equality* they always pass having read one term and compared it to itself, which is.

    Two literals are the hazard at **either** polarity: ``assert_eq!(0.0, 0.0)`` and
    ``assert_ne!(1, 2)`` both pass without touching the code under test.
    """
    if not a or not b:
        return None
    if a == b:
        return "IDENTICAL_OPERANDS" if equality else None
    if LITERAL.match(a) and LITERAL.match(b):
        return "BOTH_LITERAL"
    return None


def scan_rust(display: str, src: str) -> tuple[list[Detection], int]:
    stripped = strip_rust(src)
    raw_lines = src.splitlines()
    found: list[Detection] = []
    scanned = 0
    for m in RUST_MACRO.finditer(stripped):
        scanned += 1
        args = split_top_level(stripped[m.end() :])
        if len(args) < 2:
            continue
        kind = _classify(
            normalise(args[0]), normalise(args[1]), equality="assert_eq!" in m.group(0)
        )
        if kind:
            lineno = stripped.count("\n", 0, m.start()) + 1
            text = raw_lines[lineno - 1].strip() if lineno <= len(raw_lines) else ""
            found.append(Detection(kind, display, lineno, text[:160]))
    return found, scanned


def scan_python(display: str, src: str) -> tuple[list[Detection], int]:
    stripped = strip_python(src)
    raw_lines = src.splitlines()
    found: list[Detection] = []
    scanned = 0
    for lineno, line in enumerate(stripped.splitlines(), 1):
        m = PY_ASSERT.search(line)
        if not m:
            continue
        body = split_top_level(m.group("body") + ")")[0]
        parts = re.split(r"(?<![=!<>])(==|!=)(?!=)", body, maxsplit=1)
        if len(parts) != 3:
            continue
        scanned += 1
        kind = _classify(normalise(parts[0]), normalise(parts[2]), equality=parts[1] == "==")
        if kind:
            text = raw_lines[lineno - 1].strip() if lineno <= len(raw_lines) else ""
            found.append(Detection(kind, display, lineno, text[:160]))
    return found, scanned


def iter_sources(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_dir() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix == ".rs":
            yield path, scan_rust
        elif path.suffix == ".py":
            yield path, scan_python


def run(root: Path) -> tuple[int, str]:
    detections: list[Detection] = []
    unreadable: list[str] = []
    scanned = {".rs": 0, ".py": 0}
    files = {".rs": 0, ".py": 0}

    for path, scanner in iter_sources(root):
        try:
            src = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError) as exc:
            unreadable.append(f"{path}: {exc!r}")
            continue
        files[path.suffix] += 1
        found, count = scanner(str(path.relative_to(root)).replace("\\", "/"), src)
        detections.extend(found)
        scanned[path.suffix] += count

    scope = (
        "This screen covers ONE form: the two compared sides are the same source text, or\n"
        "are both literal constants. It does NOT detect an assertion whose two sides are\n"
        "different expressions that always evaluate equal, and it does NOT detect a test\n"
        "whose assertions are all one polarity. Both of those have occurred in this\n"
        "repository and both are outside this screen's reach."
    )

    if unreadable:
        body = "\n".join(f"  {u}" for u in unreadable)
        return EXIT_ERROR_INSTRUMENT, (
            f"ASSERTIONS: ERROR(instrument=source_unreadable)\n{body}\n\n{scope}"
        )

    total_files = files[".rs"] + files[".py"]
    total_scanned = scanned[".rs"] + scanned[".py"]

    if total_files == 0:
        return EXIT_ERROR_INSTRUMENT, (
            f"ASSERTIONS: ERROR(instrument=no_sources_found)\n"
            f"  no .rs or .py files under {root}\n\n{scope}"
        )

    # Per-language, not just in total.  This exact hole shipped for one revision of this
    # file: a leading-whitespace bug in PY_ASSERT meant 89 Python files contributed zero
    # assertions, and the screen reported PASS over a language it had not read.  A total
    # that is non-zero because the *other* language carried it is not coverage.
    blind = [ext for ext in (".rs", ".py") if files[ext] and not scanned[ext]]
    if blind:
        detail = "\n".join(
            f"  {files[ext]} {ext} files yielded 0 comparison assertions" for ext in blind
        )
        return EXIT_ERROR_INSTRUMENT, (
            f"ASSERTIONS: ERROR(instrument=language_scanned_nothing)\n{detail}\n"
            f"  A language present in the tree and invisible to the scanner is an\n"
            f"  instrument outage, not a clean result.\n\n{scope}"
        )

    if total_scanned == 0:
        return EXIT_ERROR_INSTRUMENT, (
            f"ASSERTIONS: ERROR(instrument=no_assertions_scanned)\n"
            f"  {total_files} source files contained no comparison assertion. A screen\n"
            f"  that examined nothing must not report the same word as one that examined\n"
            f"  everything and found it clean.\n\n{scope}"
        )
    if detections:
        body = "\n".join(d.render() for d in detections)
        return EXIT_FAIL_CONDITION, (
            f"ASSERTIONS: FAIL(tautological_assertions={len(detections)})\n"
            f"{body}\n\n"
            f"  scanned {total_scanned} comparison assertions "
            f"(rs={scanned['.rs']}, py={scanned['.py']}) across {total_files} files\n\n{scope}"
        )
    return EXIT_PASS, (
        f"ASSERTIONS: PASS(scanned={total_scanned}, files={total_files})\n"
        f"  rs={scanned['.rs']} assertions in {files['.rs']} files; "
        f"py={scanned['.py']} in {files['.py']} files.\n"
        f"  No assertion compares a term to itself or two literals.\n\n{scope}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verbose", action="store_true", help="reserved; output is already full")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(
            f"ASSERTIONS: ERROR(instrument=root_not_a_directory)\n  {root}",
            file=sys.stderr,
        )
        return EXIT_ERROR_INSTRUMENT

    code, message = run(root)
    print(message, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
