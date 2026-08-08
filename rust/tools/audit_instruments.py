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

# THE FRAME (2026-08-02) — three domains, and the scope is declared rather than implied

This screen has three domains: the Rust instruments (`rust/src`), the harness instruments
(`tests/`), and the bench instruments (`bench/`). The third was added after Niobe found
that `bench/` — the whole certification apparatus, including the SM-clock record that was
the only instrument to refuse a device-clock series reading STEADY at RSD 0.0717% and
20.18x wrong — had never been in any census's frame, while this script printed
`CENSUS VERDICT: PASS`.

The defect was not the scope. It was the silence: a reader could not tell a PASS over the
tree from a PASS over two thirds of it. So the frame is now printed on every path
(`frame_report`), every top-level source directory carries an IN/OUT decision with a
reason (`FRAME_DIRS`), every `bench/*.py` is either screened or held out with a reason
(`BENCH_HELD_OUT`), and anything declared neither way is `FAIL(drift)`. R12 turned on this
script: "I did not look there" is a different fact from "I looked and found nothing", and
a census that cannot say which is reporting the second when it means the first.

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

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
REPO = HERE.parents[2]
TESTS = REPO / "tests"
BENCH = REPO / "bench"
BASELINE = HERE.parent / "instrument_census.json"

# ---------------------------------------------------------------------------
# SUMMARY COUNTS ARE DERIVED, NEVER TYPED — IN THE BASELINE *AND* IN THE SOURCE.
#
# Found on review of the issue #69 branch: `instrument_census.json` carried a sentence
# quoting a bench row count beside a MACHINE-GENERATED `bench_unfalsified` array that had
# long since grown past it. Nothing was wrong with the array. The defect was the second
# copy — a number a human typed once, sitting next to the collection it claims to
# summarise, with no mechanism that makes the two disagree out loud. That is a stale figure
# quoted as a finding, which is the exact defect class this branch exists to remove, applied
# to the census itself.
#
# So every summary number the baseline publishes lives in `counts`, is DERIVED from the
# arrays at `--write-baseline` time, and is RE-DERIVED and compared on every `--check`.
# Editing a count by hand is drift, and drift is red.
#
# FOUND AGAIN, ONE LAYER OUT, AND THIS IS THE PART THAT MATTERS.  The first cut of that
# repair screened the GENERATED DOCUMENT and nothing else. The stale figures it was written
# to remove survived, verbatim, in the `#` comments of THIS FILE and in the prose of the
# always-on test module — where they read exactly like findings, and where `--check` and
# every census test stayed green. A screen whose corpus is one file cannot see the copy of
# the number that lives beside it.
#
# The corpus is therefore the census's whole surface: the baseline, this script's comments
# and docstrings, and the test module's prose (:data:`SOURCE_CLAIM_CORPUS`), and the
# candidate scan below fails on any in-frame file that starts making census count claims
# without being declared. Every claim is bound to the collection its own sentence names —
# `bind`, never "does some derived number equal this integer?" — because this census
# derives enough numbers that a wrong sentence can nearly always find a witness among them.
COUNTED_ARRAYS = (
    "uninvoked",
    "ambiguous",
    "harness_uninvoked",
    "harness_unfalsified",
    "bench_uninvoked",
    "bench_unfalsified",
)

#: The quotable shapes: "<n> rows", "<n> bench rows", "<n> UNFALSIFIED rows". Deliberately
#: narrow — it screens claims about THIS census's rows, not every integer in the file. The
#: shapes are spelled without an example number for the reason this block exists.
#:
#: ``(?<![\w.])`` and not ``(?<![\d.])``: the looser guard read the ``13`` of ``R13 applies
#: to every row of this census`` as a row count, and a screen that convicts its own prose
#: for containing a rule name gets switched off.
#:
#: The label between the number and `rows` is CAPTURED, not skipped: see
#: :func:`_bound_count_key`. A number that is a real count of something else is not a
#: witness for the collection a sentence actually names.
ROW_CLAIM = re.compile(r"(?<![\w.])(\d[\d,]*)\s+((?:[A-Za-z_/.`-]+\s+){0,3})rows?\b")

#: The same claim with the noun ``rows`` left out: "<n> bench unfalsified", "<n>
#: uninvoked". This is the shape the review found in this file's own docstring — an
#: example spelled ``"<n> bench unfalsified "`` — which the row-shaped screen could not
#: see because the word it keys on was not in it. A count of a labelled collection is a
#: claim about that collection whether or not the sentence spells the noun.
ARRAY_CLAIM = re.compile(
    r"(?<![\w.])(\d[\d,]*)\s+((?:[A-Za-z_`.-]+\s+){0,2}"
    r"(?:uninvoked|unfalsified|ambiguous))(?![\w-])",
    re.IGNORECASE,
)

#: A row claim scoped to A SET OF MODULES: "<n> rows across these four modules",
#: "<n> rows in the four modules above", "<n> rows over these modules".
#:
#: WHY THIS NEEDED ITS OWN SHAPE.  A module-scoped claim passed the screen because its
#: number was a real derived count — of ONE of the modules named, not of the set. A screen
#: that asks only "does some derived count equal this integer?" cannot convict a sentence
#: whose number belongs to a different collection than its subject, and this census derives
#: a per-module count for every bench module for it to borrow from. So a module-scoped
#: claim is bound to the sum over the modules ITS OWN SCOPE NAMES, and the cardinality word
#: is bound to how many it names. The specimens are in :data:`STALE_SPECIMENS`, not here.
MODULE_SET_CLAIM = re.compile(
    r"(?<![\w.])(\d[\d,]*)\s+(?:[A-Za-z_/.`-]+\s+){0,3}rows?\s+"
    r"(?:across|in|over|for|among|between)\s+"
    r"(?:all\s+|both\s+)?(?:these|those|the)\s+"
    r"(?:([A-Za-z]+|\d+)\s+)?modules?\b",
    re.IGNORECASE,
)

#: The arithmetic form of the same claim: "the four modules … sum to <n>", "the sum over
#: these modules … is <n>". Its own shape because the sentence that shipped stale said the
#: sum rather than the rows, and a screen keyed on ``rows`` read straight past it.
MODULE_SUM_CLAIM = re.compile(
    r"(?:(?:these|those|the)\s+(?:(?P<card_a>[A-Za-z]+|\d+)\s+)?modules?\b"
    r".{0,160}?\bsums?\s+to\s+(?P<sum_a>\d[\d,]*)"
    r"|\bsum\s+(?:over|of|across)\s+(?:all\s+)?(?:these|those|the)\s+"
    r"(?:(?P<card_b>[A-Za-z]+|\d+)\s+)?modules?\b.{0,160}?\bis\s+(?P<sum_b>\d[\d,]*))",
    re.IGNORECASE | re.DOTALL,
)

#: Cardinality words a module-set claim may spell its module count with.
CARDINALS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

#: A module path as the census spells one, so a claim can be bound to the modules named
#: beside it rather than to whatever modules a reader assumes.
MODULE_NAME = re.compile(r"\b((?:bench|rust|tests|ci)/[A-Za-z0-9_./-]+\.py)\b")

#: Words that bind a plain row claim to one named collection. A claim that says which
#: collection it is counting is checked against THAT collection; a bare "<n> rows" with no
#: label still falls back to "some derived count equals it", because there is nothing in
#: the sentence to bind it to.
_ROW_DOMAINS = ("harness", "bench")
_ROW_STATES = ("uninvoked", "unfalsified", "ambiguous")

#: Words that make a labelled claim a DELTA rather than a total. `1 NEW uninvoked
#: instrument(s)` is this census's own drift line — a count of what changed in one run —
#: and `counts.uninvoked` is the size of the collection. Binding one to the other is the
#: borrowing this screen exists to stop, with the sign reversed: it would convict a
#: correct sentence for disagreeing with a collection it was never about. This census
#: derives totals and not deltas, so a delta-qualified claim is OUT OF SCOPE and is said
#: to be, rather than silently bound to the nearest number of the right shape.
_DELTA_WORDS = frozenset({"new", "fewer", "more", "additional", "extra", "remaining"})

#: Sentinel returned by :func:`_bound_count_key` for a delta-qualified claim.
_DELTA = "<delta>"

#: The qualified names of this census's own collections. A file that mentions one of
#: these is talking about THIS census; the bare state words (`uninvoked`, `ambiguous`)
#: are ordinary English in half the tree and are not evidence of subject.
_CENSUS_COLLECTION_NAMES = tuple(n for n in COUNTED_ARRAYS if "_" in n)

# ---------------------------------------------------------------------------
# THE STALE SENTENCES, AS DATA — SO NO PROSE HAS TO SPELL ONE AGAIN.
#
# Naming a defect means quoting it, and quoting a stale count means typing that number
# into a file this screen reads. The review's finding was exactly that: the sentences
# describing the defect had become fresh instances of it.
#
# So the specimens live here once, as strings, and nowhere else. Prose refers to them by
# KEY (`STALE_SPECIMENS["module_sum"]`) and never by digit; the tests plant them by
# reading this table rather than by retyping it, which is why the always-on test module
# contains no census numeral at all.
#
# `specimen_offenders` is the witness that keeps this table from becoming the loophole:
# every entry must STILL be convicted by the live screen, evaluated in the most favourable
# scope there is — the very module set the original entry named. A specimen that stops
# being stale is red, because it is then a live claim wearing a historical label.
STALE_SPECIMENS: "dict[str, str]" = {
    "bare_row": "85 bench rows read `unfalsified`",
    "labelled_row": "162 bench unfalsified rows",
    "labelled_array": "162 bench unfalsified",
    "module_set": "18 rows across these four modules",
    "module_sum": "the four modules the entry names sum to 43",
}


def _bound_count_key(label: str) -> "str | None":
    """The one count in ``counts`` a labelled row claim is about.

    ``STALE_SPECIMENS["labelled_array"]``'s label → ``bench_unfalsified``;
    ``"`counts.uninvoked` "`` → ``uninvoked``; ``"warm "`` → ``None`` (unbound, judged
    against every derived number); ``"NEW uninvoked "`` → :data:`_DELTA` (a count of what
    changed in one run, which this census does not derive at all). Domains and states are
    read separately so ``counts.bench_unfalsified``, ``bench-unfalsified`` and
    ``bench UNFALSIFIED`` all bind to the same collection — the point is the collection,
    not the punctuation.
    """
    words = {w for w in re.split(r"[^A-Za-z]+", label.lower()) if w}
    if words & _DELTA_WORDS:
        return _DELTA
    state = next((s for s in _ROW_STATES if s in words), None)
    if state is None:
        return None
    domain = next((d for d in _ROW_DOMAINS if d in words), None)
    key = f"{domain}_{state}" if domain else state
    return key if key in COUNTED_ARRAYS else None


def baseline_counts(base: dict) -> dict:
    """Every summary number the census publishes, derived from the arrays it summarises."""
    counts: dict = {k: len(base.get(k, [])) for k in COUNTED_ARRAYS}
    by_module: dict[str, int] = {}
    for row in base.get("bench_unfalsified", []):
        module = row.split("::")[0]
        by_module[module] = by_module.get(module, 0) + 1
    counts["bench_unfalsified_by_module"] = dict(sorted(by_module.items()))
    return counts


def derived_numbers(counts: dict) -> "set[int]":
    """The integers prose is allowed to quote as a row count."""
    out = {v for v in counts.values() if isinstance(v, int)}
    out |= set(counts.get("bench_unfalsified_by_module", {}).values())
    return out


def _strings(node, path="") -> "list[tuple[str, str]]":
    if isinstance(node, str):
        return [(path, node)]
    if isinstance(node, dict):
        return [x for k, v in node.items() for x in _strings(v, f"{path}.{k}" if path else k)]
    if isinstance(node, list):
        return [x for i, v in enumerate(node) for x in _strings(v, f"{path}[{i}]")]
    return []


def _entries(node, path=""):
    """Every dict in ``node``, with its key path. A hand entry is the scope a claim binds in."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _entries(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _entries(v, f"{path}[{i}]")


def entry_modules(entry: dict) -> "list[str]":
    """The modules a census entry declares itself to be about, in declaration order.

    Read from ``instrument`` when the entry has one, because that is the field that says
    what the entry IS about; anything else in the entry may name a module in passing (the
    test file that screens it, a module it is contrasted with) and binding to those would
    let the subject of a claim drift with its prose.
    """
    subject = entry.get("instrument")
    if not isinstance(subject, str):
        return []
    seen: "list[str]" = []
    for name in MODULE_NAME.findall(subject):
        if name not in seen:
            seen.append(name)
    return seen


def module_set_claims(base: dict, counts: dict) -> "list[str]":
    """Module-scoped row claims that the named modules' own rows do not add up to."""
    offenders: "list[str]" = []
    for path, entry in _entries(base):
        if path == "counts" or path.startswith("counts."):
            continue
        modules = entry_modules(entry)
        for key, value in entry.items():
            if not isinstance(value, str):
                continue
            where = f"{path}.{key}" if path else key
            offenders += _module_scope_claims(value, modules, counts, where)
    return offenders


def _module_scope_claims(text: str, modules: "list[str]", counts: dict,
                         where: str) -> "list[str]":
    """The module-scoped claims in *text*, bound to *modules*. Also returns their spans."""
    return [msg for msg, _span in _module_scope_claims_spanned(text, modules, counts, where)]


def _module_scope_claims_spanned(text: str, modules: "list[str]", counts: dict,
                                 where: str) -> "list[tuple[str, tuple[int, int]]]":
    by_module = counts.get("bench_unfalsified_by_module", {})
    out: "list[tuple[str, tuple[int, int]]]" = []
    for rx, take in ((MODULE_SET_CLAIM, "set"), (MODULE_SUM_CLAIM, "sum")):
        for m in rx.finditer(text):
            if take == "set":
                raw, spelled = m.group(1), m.group(2)
            else:
                raw = m.group("sum_a") or m.group("sum_b")
                spelled = m.group("card_a") or m.group("card_b")
            n = int(str(raw).replace(",", ""))
            span = (m.start(), m.end())
            quote = " ".join(m.group(0).split())
            if not modules:
                out.append((
                    f"{where}: {quote!r} — nothing in the scope of this claim names a "
                    f"module, so there is no module set for it to be a count of. A "
                    f"module-scoped figure with no module set is unwitnessable by "
                    f"construction", span))
                continue
            total = sum(by_module.get(mod, 0) for mod in modules)
            if n != total:
                out.append((
                    f"{where}: {quote!r} — the rows of {modules} sum to {total}, not {n}. "
                    f"A module-set claim is bound to the modules its own scope names, so "
                    f"another collection's count cannot witness it", span))
            spelled = (spelled or "").lower()
            if spelled:
                want = CARDINALS.get(spelled)
                if want is None and spelled.isdigit():
                    want = int(spelled)
                if want is not None and want != len(modules):
                    out.append((
                        f"{where}: {quote!r} — this scope names {len(modules)} module(s), "
                        f"not {want}", span))
    return out


def claim_offenders(text: str, modules: "list[str]", counts: dict,
                    where: str = "<text>") -> "list[str]":
    """Every numeric census claim in *text* that its own subject does not witness.

    THE ONE BINDING RULE, USED BY EVERY CORPUS.  Three bindings, tightest first:

    1. A module-scoped claim (:data:`MODULE_SET_CLAIM`, :data:`MODULE_SUM_CLAIM`) must
       equal the sum over the modules *its own scope* names, and its cardinality word must
       equal how many that is. A scope naming none is refused rather than waved through.
    2. A labelled claim (:data:`ROW_CLAIM`, :data:`ARRAY_CLAIM`) must equal the collection
       its label names — ``counts[key]``, not "any number this census derives".
    3. Only a claim with neither — a bare ``<n> rows`` with nothing in the sentence to bind
       it to — falls back to "some derived count equals it".

    Tightest-first is the point. A number bound by (1) is not re-judged by (2) or (3), so a
    module-set figure cannot be acquitted by a per-module count that happens to match it,
    and a labelled figure cannot be acquitted by an adjacent field's number.
    """
    allowed = derived_numbers(counts)
    offenders: "list[str]" = []
    consumed: "list[tuple[int, int]]" = []
    for msg, span in _module_scope_claims_spanned(text, modules, counts, where):
        offenders.append(msg)
        consumed.append(span)
    for m in MODULE_SET_CLAIM.finditer(text):
        consumed.append((m.start(), m.end()))
    for m in MODULE_SUM_CLAIM.finditer(text):
        consumed.append((m.start(), m.end()))

    def _already_bound(start: int) -> bool:
        return any(a <= start < b for a, b in consumed)

    for rx in (ROW_CLAIM, ARRAY_CLAIM):
        for m in rx.finditer(text):
            if _already_bound(m.start()):
                continue
            consumed.append((m.start(), m.end()))
            n = int(m.group(1).replace(",", ""))
            quote = " ".join(m.group(0).split())
            key = _bound_count_key(m.group(2) or "")
            if key is _DELTA or key == _DELTA:
                continue  # a count of what changed in one run; not a collection size
            if key is not None:
                if n != counts.get(key):
                    offenders.append(
                        f"{where}: {quote!r} — `counts.{key}` is {counts.get(key)}, not "
                        f"{n}. A claim that names a collection is checked against that "
                        f"collection, not against every number this census derives")
            elif n not in allowed:
                offenders.append(f"{where}: {quote!r} — no derived count equals {n}")
    return offenders


def prose_row_claims(base: dict, counts: dict) -> "list[str]":
    """Prose census claims in ``base`` that the collections they name do not witness.

    Entry-scoped: a claim binds to the modules the entry it lives in declares, because the
    entry is the unit a reader reads it in. See :func:`claim_offenders` for the rule.
    """
    scoped = {k: v for k, v in base.items() if k != "counts"}
    offenders: "list[str]" = []
    for path, entry in _entries(scoped):
        modules = entry_modules(entry)
        for key, value in entry.items():
            if not isinstance(value, str):
                continue
            offenders += claim_offenders(value, modules, counts,
                                         f"{path}.{key}" if path else key)
    return offenders


# ---------------------------------------------------------------------------
# THE SOURCE CORPUS. Comments, docstrings and test prose are claims too.
#
# The screen above reads the generated document. The stale figures it was written to
# remove were, at the same moment, sitting in this file's own `#` comments and in the
# always-on test module's prose — described there as the defect, spelled there as a fresh
# instance of it, and invisible because the corpus was one file.
#
# A comment is where a reader looks to find out what a mechanism means, so a count typed
# into one is quoted exactly as readily as a count in the artifact. These files are
# therefore screened with the SAME binding rule, over their comments, docstrings and
# string constants — the surfaces a number can be written on without the interpreter ever
# reading it back.
SOURCE_CLAIM_CORPUS: "dict[str, str]" = {
    "rust/tools/audit_instruments.py": (
        "this screen's own comments and docstrings. It explains the defect, so it is the "
        "one file most likely to quote a stale figure while describing one."
    ),
    "tests/ops/test_harness_census.py": (
        "the always-on test module for this census. Its prose states what each arm is "
        "for, in numbers, which is what makes it prose a reader trusts."
    ),
}

#: Where the census's numbers may legitimately appear as digits in a screened source file:
#: nowhere. A source file can always DERIVE the figure — `f\"{counts['bench_unfalsified']}\"`
#: — so the only hand-typed numerals left are the historical specimens, and those live in
#: :data:`STALE_SPECIMENS` where :func:`specimen_offenders` proves they are still stale.
#: The assignment's own source span is exempted structurally, by parsing this file, rather
#: than by a comment marker a future edit could copy onto a live claim.
_SPECIMEN_TABLE = "STALE_SPECIMENS"


def _specimen_span(text: str) -> "tuple[int, int]":
    """Character span of the :data:`STALE_SPECIMENS` assignment in *text*, or ``(-1, -1)``."""
    try:
        tree = ast.parse(text)
    except SyntaxError:  # pragma: no cover - a corpus file that will not parse
        return (-1, -1)
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    for node in tree.body:
        targets = getattr(node, "targets", []) or ([node.target] if
                                                   isinstance(node, ast.AnnAssign) else [])
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if _SPECIMEN_TABLE in names and node.end_lineno is not None:
            return (starts[node.lineno - 1], starts[node.end_lineno])
    return (-1, -1)


def source_prose(text: str) -> "list[tuple[int, str]]":
    """Every comment block and string constant in Python *text*, as ``(line, block)``.

    Comment runs are joined into one block so a claim and the module names it is about can
    be bound together across the line wrap that separates them — the specimen the review
    found spanned three comment lines, and a line-at-a-time screen would have read the
    number and the module names as unrelated.

    F-strings contribute only their literal parts, which is the whole point: a figure
    written as ``f"{counts['bench_unfalsified']} rows"`` has no digits in the source and
    cannot go stale, and this screen must not stand in the way of the fix it is asking for.
    """
    blocks: "list[tuple[int, str]]" = []
    pending: "list[str]" = []
    start = 0
    prev = -2
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):  # pragma: no cover
        toks = []
    for tok in toks:
        if tok.type != tokenize.COMMENT:
            continue
        line = tok.start[0]
        if line != prev + 1 and pending:
            blocks.append((start, " ".join(pending)))
            pending = []
        if not pending:
            start = line
        pending.append(tok.string.lstrip("#: ").rstrip())
        prev = line
    if pending:
        blocks.append((start, " ".join(pending)))

    try:
        tree = ast.parse(text)
    except SyntaxError:  # pragma: no cover
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                blocks.append((node.lineno, " ".join(node.value.split())))
    return blocks


def source_claim_offenders(text: str, counts: dict, name: str = "<source>",
                           *, exempt_specimen_table: bool = False) -> "list[str]":
    """Census count claims in one source file's comments, docstrings and string constants.

    Same binding rule as the baseline's prose (:func:`claim_offenders`); the scope a claim
    binds in is the comment block or the string it lives in, because that is the unit a
    reader reads.
    """
    skip = _specimen_span(text) if exempt_specimen_table else (-1, -1)
    exempt = set()
    if skip != (-1, -1):
        head = text[:skip[0]].count("\n") + 1
        tail = text[:skip[1]].count("\n") + 1
        exempt = set(range(head, tail + 1))
    offenders: "list[str]" = []
    for line, block in source_prose(text):
        if line in exempt:
            continue
        seen: "list[str]" = []
        for mod in MODULE_NAME.findall(block):
            if mod not in seen:
                seen.append(mod)
        offenders += claim_offenders(block, seen, counts, f"{name}:{line}")
    return offenders


def specimen_offenders(base: dict, counts: dict) -> "list[str]":
    """Every entry of :data:`STALE_SPECIMENS` that the live census would NOT convict.

    The witness that keeps the specimen table from becoming the loophole in the screen it
    exempts. Each specimen is judged in the most favourable scope available — the very
    module set the census entry it came from names — so a specimen that survives here has
    stopped being stale and is a live claim wearing a historical label.

    Both polarities in one arm: the table cannot quietly hold a true statement, and the
    screen cannot quietly stop firing on the sentences it was built for.
    """
    modules = four_module_scope(base, counts)
    out: "list[str]" = []
    for key, text in STALE_SPECIMENS.items():
        if not claim_offenders(text, modules, counts, f"STALE_SPECIMENS[{key!r}]"):
            out.append(
                f"STALE_SPECIMENS[{key!r}] = {text!r} is no longer stale: the live screen "
                f"acquits it against the census as it stands (modules {modules}). A "
                f"specimen that has become true is a claim, not a historical quotation, "
                f"and prose citing it now cites a figure nothing derives")
    return out


def four_module_scope(base: dict, counts: dict) -> "list[str]":
    """The bench modules of the census entry the module-set specimens came from.

    Read from the baseline rather than listed here, so the specimen witness is judged
    against the module set the census actually declares and not against one this file
    remembers.
    """
    by_module = counts.get("bench_unfalsified_by_module", {})
    best: "list[str]" = []
    for _path, entry in _entries(base):
        modules = entry_modules(entry)
        if len(modules) > len(best) and all(m in by_module for m in modules):
            best = modules
    return best


def claim_corpus_candidates(repo=None) -> "list[str]":
    """In-frame files that make a census count claim, as repo-relative posix paths.

    The arm that stops :data:`SOURCE_CLAIM_CORPUS` from silently going out of date. A file
    qualifies when it makes a claim THIS census could witness: a module-scoped shape (which
    is census-specific by construction), or a claim whose label binds to one of this
    census's collections in a file that names one of them by its qualified name.

    The qualified names matter. ``uninvoked`` and ``ambiguous`` are ordinary English in
    half this tree, and a candidate rule keyed on the bare words nominated every design
    document in ``docs/`` — a screen that demands a declaration for every file that uses a
    common word is the reject-all failure mode, and it gets switched off within a week.
    """
    root = REPO if repo is None else Path(repo)
    found: "list[str]" = []
    for sub in ("rust/tools", "tests", "bench", "ci", "docs"):
        base = root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in (".py", ".json", ".md"):
                continue
            if set(path.relative_to(root).parts) & FRAME_IGNORE_DIRS:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:  # pragma: no cover
                continue
            if MODULE_SET_CLAIM.search(text) or MODULE_SUM_CLAIM.search(text):
                found.append(path.relative_to(root).as_posix())
                continue
            if not any(name in text for name in _CENSUS_COLLECTION_NAMES):
                continue
            bound = False
            for rx in (ROW_CLAIM, ARRAY_CLAIM):
                for m in rx.finditer(text):
                    key = _bound_count_key(m.group(2) or "")
                    if key is not None and key != _DELTA:
                        bound = True
                        break
                if bound:
                    break
            if bound:
                found.append(path.relative_to(root).as_posix())
    return found


def corpus_claim_offenders(counts: dict, repo=None) -> "list[str]":
    """Run the source screen over every declared file, and check the corpus is complete."""
    root = REPO if repo is None else Path(repo)
    self_rel = HERE.relative_to(REPO).as_posix()
    offenders: "list[str]" = []
    for rel in SOURCE_CLAIM_CORPUS:
        path = root / rel
        if not path.is_file():
            raise CensusInstrumentError(
                f"{rel} is declared in SOURCE_CLAIM_CORPUS and is not in the tree; the "
                f"claim screen did not read it, so it observed nothing there")
        offenders += source_claim_offenders(
            path.read_text(encoding="utf-8"), counts, rel,
            exempt_specimen_table=rel == self_rel)
    declared = set(SOURCE_CLAIM_CORPUS) | {
        BASELINE.relative_to(root).as_posix() if BASELINE.is_relative_to(root)
        else BASELINE.name}
    for rel in claim_corpus_candidates(root):
        if rel not in declared:
            offenders.append(
                f"{rel}: makes a census count claim and is not declared in "
                f"SOURCE_CLAIM_CORPUS. Add it with a reason, or remove the claim — a "
                f"corpus that does not grow with the claims is how this screen went "
                f"blind the first time")
    return offenders


# ---------------------------------------------------------------------------
# THE FRAME. Declared here, printed on every path, and checked.
#
# Found 2026-08-02 by Niobe: this screen scanned `rust/src` and `tests/ops` and had never
# scanned `bench/` — so `CENSUS VERDICT: PASS` was true of what it scanned and read as
# true of the tree. That is this file's own `misnamed` state turned on itself, and the
# defect was not the scope: it was the silence. A reader could not tell whether `bench/`
# was screened and clean or never looked at.
#
# So scope is now DECLARED rather than implied. Every top-level directory that contains
# source is either screened or held out WITH A REASON, and a directory that is neither is
# `FAIL(drift)` — the census refuses to render a verdict over a tree it cannot account for.
# That is the arm that makes this declaration able to fail; a comment saying "we also scan
# bench" would not have been.
FRAME_DIRS: dict[str, str] = {
    "rust": "IN FRAME — the Rust instrument screen (INSTRUMENT_FILES under rust/src).",
    "tests": "IN FRAME — the harness screen (HARNESS_INSTRUMENT_FILES under tests/).",
    "bench": "IN FRAME — the bench screen (BENCH_INSTRUMENT_FILES under bench/).",
    "ci": (
        "OUT OF FRAME — lane gates, Link's. Every file here is `check_*`/`gate_*` invoked "
        "by name from a workflow, so `uninvoked` is decided by ci/lane_inventory.py "
        "against the workflow files, not by a call-graph screen. Two censuses over one "
        "tree is the failure this file exists to prevent; if that inventory stops "
        "running, this line is the wrong answer and should be moved to IN FRAME."
    ),
    "docs": "OUT OF FRAME — prose. Contains no executable instrument.",
    "evidence": "OUT OF FRAME — recorded artifacts. Data, not mechanism.",
    "third_party": "OUT OF FRAME — vendored. Not ours to screen or to fix.",
    "python": (
        "OUT OF FRAME — the pip-installable registration shim "
        "(python/src/onnxruntime_ep_vulkan). It carries real claim-truth guards "
        "(assert_ep_selected, verify_provenance) but has no tests/ directory of its own "
        "and no domain question defined for it yet, so 'not scanned' is the honest answer "
        "rather than a silent one. Candidate for a fourth census domain the day it grows "
        "an always-on self-test suite; until then this line is what stops a reader from "
        "reading FRAME PASS as 'python/ was looked at'."
    ),
    ".github": "OUT OF FRAME — workflow YAML; see ci/lane_inventory.py.",
    ".squad": "OUT OF FRAME — team state and prose records, not shipped mechanism.",
    ".copilot": "OUT OF FRAME — agent configuration, not shipped mechanism.",
}

# Directory names never counted as source-bearing anywhere in the tree.
FRAME_IGNORE_DIRS = {
    "target",
    "node_modules",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".venv",
    "venv",
    "results",
}

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
        src_lines = path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(src_lines, 1):
            m = FN.match(line)
            if not m:
                continue
            name = m.group(1)
            if EXEMPT.search(name):
                continue
            # A `#[cfg(test)]` item is not production code, wherever it sits in the file.
            #
            # The split above is positional — everything before `#[cfg(test)] mod tests` is
            # "production" — so a test-only helper declared beside the thing it exercises reads
            # as an instrument with no production caller, forever. `allocator.rs::
            # clear_session_devices` sat in the baseline on exactly that footing: it is
            # `#[cfg(test)]`, it is documented "**Tests only**", it has three test callers, and
            # the screen could not see any of that. Scoring it `uninvoked` is a false positive
            # by construction, and a screen that mis-scores in the dead direction is the one
            # thing this file says it must not do.
            if any(
                "#[cfg(test)]" in src_lines[i]
                for i in range(max(0, line_no - 6), line_no - 1)
                if not src_lines[i].lstrip().startswith("///")
            ):
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
#
# `ops/_verdict.py` added 2026-08-01 (Trinity).  It was written after this screen and the
# screen did not know about it, so the census read "9 harness instruments" while fifteen
# more — including every guard the §10.0 third amendment and R13 rest on — sat outside its
# frame.  A census whose frame excludes the newest instruments reports a number about a
# world it has not surveyed, which is the shape this whole file exists to catch (R12).
#
# `ops/conftest.py` added 2026-08-01 (Trinity, on Tank's open item).  It is the file that
# decides, for EVERY test in the suite, whether a red is `PASS` / `FAIL(condition)` /
# `ERROR(instrument)` — the R13 channel itself — and it sat outside the frame while the
# census reported on the instruments whose verdicts travel down it.  A screen that surveys
# the speakers and not the microphone is `out-of-frame` in its own vocabulary.
HARNESS_INSTRUMENT_FILES = [
    "ops/_models.py",
    "ops/_verdict.py",
    "ops/conftest.py",
    "ops/_watchdog.py",
]

# A harness instrument is a function that renders a verdict: it either raises on a bad
# world or returns a number a gate reads.  Helpers that only build models or run sessions
# are not instruments and are excluded by name.
#
# `classify_` added with `ops/_verdict.py`: an R13 classifier that maps (exit code, text)
# onto PASS/FAIL/ERROR renders a verdict as surely as an `assert_` does — it just returns
# the token instead of raising it.  See `hand.harness_notes`: the raise-based polarity
# model below cannot screen a total function, and saying so is the point.
#
# The optional leading `_` added 2026-08-01 (Trinity) with `ops/conftest.py`.  Without it
# the screen could see `require_vulkan` in that file and nothing else — not
# `_classify_failure`, which decides the R13 token for every test in the suite, and not
# `_assert_oracle_versions`, which decides whether the oracle is admissible at all.  A
# module-private name is private to Python, not to the call graph; hiding the two most
# load-bearing instruments in the harness behind an underscore convention is exactly the
# "reports a number about a world it has not surveyed" failure one line up.
#
# `ops/_watchdog.py` added 2026-08-01 (Trinity).  It decides whether a census step that has
# not returned is a hang or a slow machine — the distinction that kept
# `test_wiring_census.py` out of the suite — and `assert_alive` is the guard that stops a
# dead watchdog from reading as "nothing to report".  An instrument that adjudicates other
# instruments' silence must itself be inside the frame.
HARNESS_FN = re.compile(
    r"^_?(assert_|count_|check$|check_|require_|verify_|expect_|classify_)|_verdict$"
)

# Decorators / fixtures that mean "this test does not run in the always-on lane".
HARNESS_GATE = re.compile(r"require_vulkan|skipif|\bskip\b|xfail|slow|gpu|require_model")

# ---------------------------------------------------------------------------
# BENCH DOMAIN (added 2026-08-02, Tank, on Niobe's finding).
#
# WHY IT GETS ITS OWN SELECTION RULE, AND WHY THAT IS THE FINDING RATHER THAN A DETAIL.
#
# The first cut of this extension reused `HARNESS_FN` — the `assert_`/`check_`/`require_`
# name vocabulary — and it under-selected so badly that the extension would have
# reproduced, inside the fix, the exact state it was written to remove. The specimen:
#
#     bench/phases.py: 37 top-level functions, 0 selected by HARNESS_FN.
#
# That file holds `gpu_steady_tail`, `decomposition_identity`, `phase_containment`,
# `trace_matches_counters`, `valid_bits_applied` and `red_flags` — the machinery that
# decides whether a published figure is admissible at all. A name screen would have
# printed "bench/ scanned" over a module in which it saw nothing. `absent`, dressed as
# coverage.
#
# So the bench rule is structural, not lexical: **every module-public top-level function
# of a declared instrument module is an instrument**, the same rule the Rust screen uses
# (`pub fn`). `main` is the one exclusion — it is the CLI entry, not a verdict. A
# structural rule cannot be defeated by an author who names things differently from the
# author the vocabulary was read off, and the vocabularies here genuinely differ: Niobe
# writes `certify`/`grade`/`quiescence`, Trinity writes `assert_`/`require_`.
#
# Consequence, stated up front so it is not read as a regression: this admits most of
# bench/ as rows, most of them `unfalsified` (the current tally is derived into
# `instrument_census.json`'s `counts`, and is deliberately not restated here — a figure
# typed into a comment beside the collection it summarises goes stale in silence, which is
# what the self-count arm above exists to stop). `unfalsified` is the honest state of an
# instrument nothing has watched disagree. It was always the state of these; the census
# simply could not say so, because it had never looked.
BENCH_FN = re.compile(r"^(?!main$)(?!_)")

# Screened modules: those that render a verdict about a measurement.
BENCH_INSTRUMENT_FILES = [
    "device_companion.py",
    "phases.py",
    "admissible.py",
    "contention.py",
    "run_disturbance.py",
    "timestamp_audit.py",
    "win_gpu_counters.py",
    "devices.py",
    "stats.py",
    "portability.py",
    # Arrived with main d9a9c0c. Both render a verdict rather than record a number:
    # `ceiling.py` has an explicit refusal state, and `clock_log.window` returns
    # UNOBSERVABLE for a window with no samples — the R12 distinction this census exists
    # to keep, so it is screened rather than treated as capture.
    "ceiling.py",
    "clock_log.py",
    # Arrived with the issue #69 CUDA competition harness. All four render a verdict about
    # a measurement rather than record one, which is the line this list draws:
    #   `cuda_competition.py`  -- ADMISSIBLE / SPLIT_FRAME / INSTRUMENT_ERROR per arm, plus
    #                             the numeric-equivalence verdict across arms.
    #   `cuda_profile.py`      -- GPU_TIME_MEASURED / GPU_TIME_UNAVAILABLE / TRACE_ABSENT,
    #                             and refuses rather than sums when the phase tree disagrees.
    #   `cuda_probe.py`        -- decides whether a node partition actually ran on the EP it
    #                             names, which is a verdict about what was measured.
    #   `bench_models.py`      -- MODEL_OK / MODEL_ABSENT / MODEL_DIGEST_MISMATCH. The digest
    #                             mismatch is explicitly "a finding, not a re-pin".
    "cuda_competition.py",
    "cuda_profile.py",
    "cuda_probe.py",
    "bench_models.py",
    # Arrived with issue #56 (Niobe, the real-model harness). Screened rather than held
    # out because its `classify_*`/`bitwise_identical` functions decide whether two arms
    # of a benchmark agree, and `dispatch_diagnosis`/`fallback_diagnosis` decide whether a
    # run is admissible as evidence about device utilisation. Those are verdicts about a
    # measurement, which is this list's criterion.
    "real_model.py",
]

# Every other `bench/*.py`, with the reason it is not screened. A file in `bench/` that
# appears in neither list is `FAIL(drift)`: the census will not silently decide for you
# whether a new module is an instrument. This is the file-level form of the directory
# declaration above, and it exists for the same reason — the gap Niobe found was one
# unlisted directory, and one unlisted file is the same defect one level down.
BENCH_HELD_OUT: dict[str, str] = {
    "bench.py": "runner — orchestrates a run, renders no verdict.",
    "cases.py": "case table — data.",
    "compare.py": "presentation of two runs; the verdicts it prints come from stats.py.",
    "environment.py": "capture — records the world, judges nothing.",
    "exec_census.py": "runner for a census defined elsewhere.",
    "island_attribution.py": "attribution arithmetic; its verdict lives in phases.py.",
    "phi35.py": "model construction and output classification for one model.",
    "producers.py": "builds the inputs a measurement runs on.",
    "transfer_calibration.py": "calibration data producer.",
    # Arrived with main 5317bf0 and were the frame arm's first live catch: two files in
    # `bench/` that this census had never been told about. It refused to render a verdict
    # over them, which is exactly the behaviour the arm was written for.
    "spirv_simt.py": (
        "capture — a SPIR-V interpreter. Its own docstring draws the line: it 'reports the "
        "multiset of words the module loaded from a binding and lets the caller divide', "
        "and the amplification verdict is the caller's. It raises `InstrumentError` when it "
        "cannot execute, which is a refusal to measure, not a verdict about a measurement."
    ),
    "test_weight_reread.py": "test module — a caller, screened as polarity, not as an instrument.",
    "test_contention.py": "test module — a caller, screened as polarity, not as an instrument.",
    "test_device_state.py": "test module.",
    "test_harness.py": "test module.",
    "test_import_isolation.py": "test module.",
    "test_island_boundary_cost.py": "test module.",
    "test_marginal_tail_withholds.py": "test module.",
    # Arrived with PR #72 (the gqa_f16 workgroup-size change) and reproduced the same
    # defect class as fa5f514/98d5bf3 below: a new bench/ file lands, and nothing in this
    # dict has been told about it. `test_perf_claims.py` checks the *publication*
    # (docs/PERF.md section 26 and the shader header) against the artifacts it cites -- it
    # renders no verdict about a measurement itself, it is a caller of `real_model.py` and
    # a reader of committed JSON/markdown, so it belongs here rather than in
    # BENCH_INSTRUMENT_FILES.
    "test_perf_claims.py": "test module — a caller, screened as polarity, not as an instrument.",
    "test_phases.py": "test module.",
    "test_plausible_but_wrong.py": "test module.",
    "test_run_disturbance.py": "test module.",
    "test_tenancy_signature.py": "test module.",
    "test_win_gpu_counters.py": "test module.",
    # Arrived with main 98d5bf3 (Niobe, the paired-ratio soundness instrument). Declared
    # here by link rather than left to drift: the frame arm caught it on the first run
    # after the merge, which is the arm working. Handed back to Niobe if she means it as
    # an instrument rather than a caller — moving it to BENCH_INSTRUMENT_FILES is a
    # one-line change and this comment is the record that nobody decided it silently.
    "test_paired_ratio.py": "test module — a caller, screened as polarity, not as an instrument.",
    "test_ceiling.py": "test module — a caller, screened as polarity, not as an instrument.",
    "test_real_model.py": "test module — a caller, screened as polarity, not as an instrument.",
    # Arrived with Switch's fa5f514 and was the frame arm's second live catch. Worth naming
    # what it cost before it was declared: the frame arm runs BEFORE the uninvoked census, so
    # `audit_instruments --check` failed on the frame and never printed
    # `1 NEW uninvoked instrument(s)` — the exact string ci/open_reds.json holds as the
    # signature of Mouse's accepted red. The register then reported `signature_changed`
    # rather than ACCOUNTED, which is the arm behaving correctly: the acceptance was granted
    # for one red and a different red had taken its place.
    "test_kv_write_redundancy.py": (
        "test module — a caller, screened as polarity, not as an instrument."
    ),
    # Arrived 2026-08-06 (Tank) closing the `identify_by_uuid` unfalsified finding honestly
    # rather than by hand note. `_polarity.py` is the VALUE-polarity source the screen now
    # reads (see VALUE_REJECT_FN): it renders no verdict about a measurement, it enforces
    # the refusal contract of an instrument that returns one instead of raising it — the
    # same relationship `pytest.raises` has to a guard that raises, and pytest is not an
    # instrument either. Its own two polarities are screened in test_devices_identity.py,
    # which is where the mutation battery for identify_by_uuid also lives.
    "_polarity.py": (
        "polarity assertion helpers for TOTAL instruments — the screen's second polarity "
        "source, not a verdict about a measurement. Two-polarity tested in "
        "test_devices_identity.py."
    ),
    "test_devices_identity.py": (
        "test module — a caller, screened as polarity, not as an instrument. Carries the "
        "five planted mutants that earn identify_by_uuid its `screened` state."
    ),
    # Arrived with the issue #69 CUDA competition harness (2026-08-07, Tank). Declared by
    # hand rather than left to drift, and each with the reason it is NOT an instrument.
    "cuda_workloads.py": (
        "workload table — data. It builds feeds and digests them so two arms can be shown "
        "the same bytes; it renders no verdict about any measurement taken on them."
    ),
    "trace_vocabulary.py": (
        "parser — reads the span vocabulary declared in rust/src/trace.rs so bench/ can be "
        "checked against it instead of trusted. `prefix_collisions` reports a list; the "
        "verdict that a collision is fatal is asserted by bench/test_trace_vocabulary.py "
        "and by the Rust test `no_trace_name_is_a_prefix_of_another`, in both languages."
    ),
    "public_paths.py": (
        "provenance sanitiser — it renders a verdict about a PAYLOAD (does this artifact "
        "name a machine?), not about a measurement, which is the line this list draws. It "
        "refuses in TWO shapes, and saying only the first is what made the previous "
        "version of this note wrong: `dump_public_json`/`assert_public`/`write_public_text` "
        "raise `PathLeak`, and `contained_child`/`resolve_public_path` are TOTAL — they "
        "return `None` for a handle they will not stand behind, because their callers "
        "record the refusal beside the absent file rather than propagating an exception. "
        "Both shapes are screened in bench/test_public_paths.py: the raising ones through "
        "the `sanitise=False` path that proves the refusal fires, and the total ones "
        "through a specimen table of malformed handles asserted to be refused AND a "
        "legitimate contained handle asserted to resolve. `contained_child`'s totality is "
        "asserted as such — `test_contained_child_never_raises_whatever_it_is_handed` "
        "feeds it every specimen plus a non-path object, because a resolver that raises is "
        "one whose `is None` callers are bypassed rather than told no."
    ),
    "gen_public_path_legacy.py": (
        "generator for the checked-in legacy ratchet bench/public_path_legacy.json — the "
        "same relationship rust/tools/instrument_census.json's generator has to this file. "
        "It surveys; the ratchet's verdict is asserted in bench/test_public_paths.py."
    ),
    "test_cuda_competition.py": "test module — a caller, screened as polarity, not as an instrument.",
    "test_cuda_profile.py": "test module — a caller, screened as polarity, not as an instrument.",
    "test_trace_vocabulary.py": "test module — a caller, screened as polarity, not as an instrument.",
    "test_public_paths.py": (
        "test module — a caller, screened as polarity, not as an instrument. Carries both "
        "polarities of the provenance sanitiser and the repository-wide leak ratchet."
    ),
    "test_result_staleness.py": (
        "test module — a caller, screened as polarity, not as an instrument. Carries both "
        "polarities of `cuda_competition.ep_provenance` and the screen that stops a "
        "committed result outliving the EP build it measured."
    ),
}


def _harness_instruments(tests_root=None, files=None, fn_re=None, prefix="tests") -> dict[str, dict]:
    """Return ``{"<prefix>/<rel>::<fn>": {...}}`` for every harness instrument.

    KEYED BY FILE AND FUNCTION, NEVER BY FUNCTION ALONE.  The first version of this
    returned ``{fn_name: qualified_id}``, and a dict keyed on a bare name silently keeps
    the LAST module that defines it.  Eight names collide across the screened bench
    modules — ``attribute``, ``audit``, ``describe``, ``load``, ``probe``, ``render``,
    ``sha256_file``, ``summarise`` — so ``phases.py::attribute``, ``devices.py::probe`` and
    ``win_gpu_counters.py::summarise`` were not merely mis-scored: they were absent from
    the census, which printed ``PASS`` over a list they were never in.  A screen whose
    frame is decided by dict-insertion order is the ``out-of-frame`` state applied to
    itself.
    """
    import ast as _ast

    tests_root = TESTS if tests_root is None else Path(tests_root)
    files = HARNESS_INSTRUMENT_FILES if files is None else files
    fn_re = HARNESS_FN if fn_re is None else fn_re
    out: dict[str, dict] = {}
    for rel in files:
        path = tests_root / rel
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if fn_re.search(node.name):
                    qual = f"{prefix}/{rel}::{node.name}"
                    out[qual] = {"id": qual, "fn": node.name, "rel": rel,
                                 "module": Path(rel).stem, "path": path.resolve()}
    return out


def _fixture_instruments(tests_root=None, files=None, fn_re=None, prefix="tests") -> set[str]:
    """Return the subset of harness instruments that are pytest fixtures, by qualified id.

    A fixture is invoked by **parameter name**, never by a call expression, so the
    call-shaped caller model that screens every other instrument scores one as
    ``uninvoked`` no matter how many tests depend on it.  ``require_vulkan`` is depended on
    by most of this suite and read ``UNINVOKED calls=0`` the first time ``ops/conftest.py``
    entered the screen's frame — a false positive of exactly the kind this file warns about
    under "Known limits", and one that would have been mistaken for a finding.

    Decided from the decorator list, so it needs no import of pytest and works on a
    synthetic tree.
    """
    import ast as _ast

    tests_root = TESTS if tests_root is None else Path(tests_root)
    files = HARNESS_INSTRUMENT_FILES if files is None else files
    fn_re = HARNESS_FN if fn_re is None else fn_re
    out: set[str] = set()
    for rel in files:
        path = tests_root / rel
        if not path.is_file():
            continue
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            if not fn_re.search(node.name):
                continue
            for dec in node.decorator_list:
                if "fixture" in _ast.dump(dec):
                    out.add(f"{prefix}/{rel}::{node.name}")
                    break
    return out


# ---------------------------------------------------------------------------
# VALUE POLARITY FOR TOTAL INSTRUMENTS (added 2026-08-06, Tank, on the
# `bench/devices.py::identify_by_uuid` finding).
#
# The `pytest.raises` model above is blind to an instrument that RETURNS its refusal
# instead of raising it.  `identify_by_uuid` is the specimen: it returns `(device, why)`
# and never raises, because its caller `device_identity_check` prints the `why` on the
# refusal path.  The totality is deliberate.
#
# Three ways to close such a finding, and only the third is honest:
#   1. force an exception contract on the instrument so this screen can see it — changing
#      the subject to fit the instrument, and making production worse to make the screen
#      greener;
#   2. baseline it with a hand note — converting an open question into a permanent one;
#   3. give the screen a second polarity source that OBSERVES as strongly as
#      `pytest.raises` does.
#
# `pytest.raises` earns its credit by failing the test when the thing inside it does not
# raise.  So the second source is held to the same standard: a call to the instrument that
# appears as an argument to `bench/_polarity.py::refuses(...)` is reject polarity, and one
# that appears as an argument to `selects(...)` is accept polarity — and BOTH of those
# helpers raise `PolarityError` at run time when the contract they name is not honoured.
# They are assertions, not annotations.  A mutant instrument cannot pass through either.
#
# Crediting a bare marker would be the Guard D shape with the sign flipped, and this file
# says so about itself two hundred lines up; the enforcement is what makes this not that.
#
# The blind spot, unchanged and restated: neither model can see whether the test's INPUT
# actually varies the thing under test.  That is earned by mutation —
# `bench/test_devices_identity.py` for this instrument, `tests/ops/test_guard_d.py` for
# the harness domain — and it is not claimed by this screen.
# A verdict gate is total in a second shape: it takes a record and returns the same record
# with its green verdict either withheld or left standing, because the refusal has to travel
# with the numbers it disqualifies.  `withholds(...)`/`publishes(...)` name those polarities
# and enforce them at run time exactly as `refuses`/`selects` do — a gate wired to nothing
# cannot pass `withholds`, and a gate that fires on everything cannot pass `publishes`.
VALUE_REJECT_FN = frozenset({"refuses", "withholds", "omits"})
VALUE_ACCEPT_FN = frozenset({"selects", "publishes", "records"})


def _polarity_wrapped(fn, wrapper_names: "frozenset[str]") -> "set[int]":
    """ids of every ast node nested inside a call to one of *wrapper_names*."""
    import ast as _ast

    marked: set[int] = set()
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
        if name not in wrapper_names:
            continue
        for inner in _ast.walk(node):
            if inner is not node:
                marked.add(id(inner))
    return marked


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


def _dotted(node) -> "str | None":
    """``self._mod`` / ``mod`` / ``a.b.c`` as a string; anything else is None."""
    import ast as _ast

    parts: list[str] = []
    while isinstance(node, _ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, _ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _module_bindings(tree, self_module: str) -> dict:
    """What every name in one file is bound to, as far as its own imports declare it.

    Returns ``{"aliases": {local: module}, "names": {local: (module, original_fn)},
    "stars": [module], "self": self_module, "shadowed": {name}}``.

    This is deliberately a *declaration* reader and not an import graph: it believes what
    the file says about itself and nothing else.  `import cuda_workloads as cw` binds
    ``cw`` to the module ``cuda_workloads``; `from real_model import build_feeds` binds the
    bare name; `from x import *` binds nothing this screen can name, and a module-level
    ``def``/assignment of the same name shadows the import.  Everything else is unresolved,
    and unresolved is not credit.

    ONE INFERENCE BEYOND THE IMPORT STATEMENTS, AND WHY IT IS NOT A SLIPPERY SLOPE.  A
    module that may be absent is imported inside an accessor
    (``def _windows_module(): import win_gpu_counters; return win_gpu_counters``) and used
    through the value it returns — ``mod = _windows_module()``, then ``mod.observe(...)``.
    That is a real call to a real instrument, and refusing to see it would have printed
    ``UNINVOKED win_gpu_counters.py::observe``: a *fabricated detection*, which this file
    already records (``instrument_census.json``, the ``require_vulkan`` row) as worse than
    a missing row, because someone will act on it.  So an assignment from a function whose
    every ``return`` is a module this file imported binds its target — including
    ``self._mod`` — to that module.  Nothing else infers anything.
    """
    import ast as _ast

    aliases: dict[str, str] = {}
    names: dict[str, tuple[str, str]] = {}
    stars: list[str] = []
    shadowed: set[str] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for a in node.names:
                mod = a.name.split(".")[-1]
                aliases[a.asname or a.name.split(".")[0]] = mod
                if a.asname is None and "." not in a.name:
                    aliases[a.name] = mod
        elif isinstance(node, _ast.ImportFrom):
            mod = (node.module or "").split(".")[-1]
            for a in node.names:
                if a.name == "*":
                    stars.append(mod)
                    continue
                # `from bench import devices` binds a MODULE under a local name; the two
                # cases are told apart at resolution time by which table matches.
                aliases.setdefault(a.asname or a.name, a.name)
                names[a.asname or a.name] = (mod, a.name)
        elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if node.name not in names:
                shadowed.add(node.name)
        elif isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name):
                    shadowed.add(t.id)

    # Accessor functions: every return statement hands back a module imported above.
    accessors: dict[str, str] = {}
    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        returned = {r.value.id for r in _ast.walk(node)
                    if isinstance(r, _ast.Return) and isinstance(r.value, _ast.Name)}
        mods = {aliases[n] for n in returned if n in aliases}
        if len(mods) == 1 and len(returned) == len(mods):
            accessors[node.name] = mods.pop()
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Assign):
            continue
        mod = None
        value = node.value
        if isinstance(value, _ast.Call) and isinstance(value.func, _ast.Name):
            mod = accessors.get(value.func.id)
        elif isinstance(value, _ast.Name):
            mod = aliases.get(value.id)
        if mod is None:
            continue
        for t in node.targets:
            dotted = _dotted(t)
            if dotted:
                aliases[dotted] = mod
    return {"aliases": aliases, "names": names, "stars": stars,
            "self": self_module, "shadowed": shadowed}


def _call_target(node, binds: dict, by_module: dict, by_fn: dict):
    """Which instrument, if any, this call expression names. Fails closed.

    Returns ``(qualified_id, None)`` when the caller's own imports decide it, and
    ``(None, (why, fn_name))`` when a call *looks* like an instrument by name and the file
    does not say which module it came from.  ``(None, None)`` is the ordinary case: a call
    to something that is not an instrument at all.

    The three resolvable shapes, and nothing else:

    * ``mod.fn()`` where ``mod`` is a module this file imported;
    * ``fn()`` where ``fn`` was imported from a module by name;
    * ``fn()`` inside the module that defines it.

    `obj.summarise()` on an instance is NOT resolvable — ``win_gpu_counters.summarise`` and
    ``cuda_competition.summarise`` are both real, and picking one by name is how a census
    ends up crediting a module that was never called.
    """
    import ast as _ast

    def _plausible(name: str) -> bool:
        """Does this file import any module that defines an instrument of this name?

        The unresolved list is meant to be read, so it reports the sites where a reader
        could reasonably think an instrument was called: ones whose owning module this
        file actually imports.  ``Path(x).resolve()`` in a file that has never heard of
        ``bench_models`` is not an ambiguity about ``bench_models.resolve``; listing it
        would bury the real cases under a hundred method calls.  Nothing is credited
        either way — this only decides what gets printed.
        """
        owners = {q.split("::")[0].rsplit("/", 1)[-1][:-3] for q in by_fn.get(name, [])}
        seen = set(binds["aliases"].values()) | {m for m, _ in binds["names"].values()}
        seen |= set(binds["stars"]) | {binds["self"]}
        return bool(owners & seen)

    func = node.func
    if isinstance(func, _ast.Attribute):
        name = func.attr
        if name not in by_fn:
            return None, None
        value = func.value
        if isinstance(value, _ast.Name) or isinstance(value, _ast.Attribute):
            dotted = _dotted(value)
            mod = binds["aliases"].get(dotted) if dotted else None
            if mod is not None and (mod, name) in by_module:
                return by_module[(mod, name)], None
            if mod is not None:
                return None, None  # a real module, just not one with this instrument
        if not _plausible(name):
            return None, None
        return None, ("attribute call on a value this file does not bind to a module",
                      name)
    if isinstance(func, _ast.Name):
        name = func.id
        if name not in by_fn:
            return None, None
        bound = binds["names"].get(name)
        if bound is not None:
            key = (bound[0], bound[1])
            if key in by_module:
                return by_module[key], None
            return None, None  # imported from somewhere that is not a screened module
        if name in binds["shadowed"]:
            # This file defines the name itself, so the call is to its own function. Not
            # an instrument call, and not an ambiguity either.
            if (binds["self"], name) in by_module:
                return by_module[(binds["self"], name)], None
            return None, None
        if (binds["self"], name) in by_module:
            return by_module[(binds["self"], name)], None
        if binds["stars"]:
            owners = [q for m in binds["stars"] if (m, name) in by_module
                      for q in [by_module[(m, name)]]]
            if len(owners) == 1:
                return owners[0], None
            return None, ("star import makes the origin of this name undecidable", name)
        if not _plausible(name):
            return None, None
        return None, ("bare call to an instrument name this file never imported", name)
    return None, None


def harness_survey(tests_root=None, files=None, fn_re=None, prefix="tests",
                   unresolved_out=None) -> list[dict]:
    """Screen every harness instrument for callers and for two-polarity coverage.

    *tests_root* and *files* are parameters rather than constants so this screen can be
    pointed at a synthetic tree and watched to disagree — see
    ``tests/ops/test_harness_census.py``.  A screen that has only ever been run against the
    real repository, where it happens to print a plausible answer, is precisely the Guard D
    shape it exists to catch.

    CALL SITES ARE RESOLVED MODULE-AWARE (2026-08-07).  A call is credited to an instrument
    only when the caller file's own imports say which module the name came from — see
    :func:`_call_target`.  Before this, ``stats`` was keyed by bare function name, so
    ``bench/test_cuda_workloads.py``'s ``pytest.raises(...)`` around
    ``cuda_workloads.build_feeds`` was recorded as reject polarity for
    ``real_model.py::build_feeds``, an instrument that test has never called.  A polarity
    credit that arrives from another module is worse than a missing one: `unfalsified` is
    an honest "nobody has watched this", and a borrowed `screened` is a claim that somebody
    has.

    Unresolved and ambiguous sites are **failed closed** — no credit to anybody — and
    appended to *unresolved_out* when one is supplied, because a screen that silently drops
    evidence it could not attribute is reporting a smaller world than it looked at.
    """
    import ast as _ast

    tests_root = TESTS if tests_root is None else Path(tests_root)
    files = HARNESS_INSTRUMENT_FILES if files is None else files
    fn_re = HARNESS_FN if fn_re is None else fn_re
    instruments = _harness_instruments(tests_root, files, fn_re, prefix)
    fixtures = _fixture_instruments(tests_root, files, fn_re, prefix)
    by_module: dict[tuple[str, str], str] = {
        (rec["module"], rec["fn"]): qual for qual, rec in instruments.items()}
    by_fn: dict[str, list[str]] = {}
    for qual, rec in instruments.items():
        by_fn.setdefault(rec["fn"], []).append(qual)
    stats = {qual: {"calls": 0, "reject": 0, "accept": 0} for qual in instruments}
    unresolved: list[dict] = [] if unresolved_out is None else unresolved_out

    def _record_unresolved(path, node, name, why) -> None:
        unresolved.append({"file": str(path), "line": getattr(node, "lineno", 0),
                           "name": name, "why": why,
                           "candidates": sorted(by_fn.get(name, []))})

    owner_files = {(tests_root / rel).resolve() for rel in files}
    # Calls from inside the owner module count as callers (an instrument invoked at import
    # time, like the Q/DQ oracle probe, is wired) but can never supply a polarity: polarity
    # is a property of a test that was written to watch it disagree.
    for rel in files:
        path = (tests_root / rel).resolve()
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        binds = _module_bindings(tree, Path(rel).stem)
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            target, why = _call_target(node, binds, by_module, by_fn)
            if target is not None:
                stats[target]["calls"] += 1
            elif why:
                _record_unresolved(path, node, why[1], why[0])

    for path in sorted(tests_root.rglob("*.py")):
        if path.resolve() in owner_files:
            continue
        try:
            tree = _ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        binds = _module_bindings(tree, path.stem)
        for fn in [n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)]:
            gated = _is_gated(fn)
            # A fixture is depended on by naming it as a parameter.  This is the only
            # invocation it ever gets, so it is counted here and nowhere else.  It never
            # supplies a polarity: nothing in the parameter list says which answer the
            # fixture gave.  A fixture name defined by two modules is ambiguous — pytest
            # would resolve it by conftest scope, which is not visible here — so it is
            # failed closed rather than credited to whichever one sorts first.
            for arg in fn.args.args:
                owners = [q for q in by_fn.get(arg.arg, []) if q in fixtures]
                if len(owners) == 1:
                    stats[owners[0]]["calls"] += 1
                elif len(owners) > 1:
                    _record_unresolved(path, fn, arg.arg,
                                       "fixture parameter names an instrument defined in "
                                       "more than one screened module")
            # Map every node in this function to whether it sits inside `pytest.raises`.
            raising: set[int] = set()
            for node in _ast.walk(fn):
                if isinstance(node, _ast.With):
                    if any(
                        "raises" in _ast.dump(item.context_expr) for item in node.items
                    ):
                        for inner in _ast.walk(node):
                            raising.add(id(inner))
            # ...and whether it sits inside an enforcing value-polarity assertion, which is
            # how a TOTAL instrument's refusal is watched.  See VALUE_REJECT_FN above.
            value_reject = _polarity_wrapped(fn, VALUE_REJECT_FN)
            value_accept = _polarity_wrapped(fn, VALUE_ACCEPT_FN)
            # `value_accept` is computed and deliberately NOT used to award accept credit.
            # A bare call already scores accept under the original model, so requiring
            # `selects(...)` for it would silently re-score every screened row in the tree.
            # It is read by the frame report instead: `selects` earns nothing from this
            # screen, and does its work at RUN time, where it makes an accept credit mean
            # something a bare call never did.
            del value_accept
            for node in _ast.walk(fn):
                if not isinstance(node, _ast.Call):
                    continue
                target, why = _call_target(node, binds, by_module, by_fn)
                if target is None:
                    if why:
                        _record_unresolved(path, node, why[1], why[0])
                    continue
                stats[target]["calls"] += 1
                if gated:
                    continue
                if id(node) in raising or id(node) in value_reject:
                    stats[target]["reject"] += 1
                else:
                    stats[target]["accept"] += 1

    rows: list[dict] = []
    for qual, rec in sorted(instruments.items()):
        s = stats[qual]
        if s["calls"] == 0:
            state = "uninvoked"
        elif s["reject"] and s["accept"]:
            state = "screened"
        else:
            state = "unfalsified"
        rows.append({"id": qual, "fn": rec["fn"], "state": state, **s})
    return rows


def harness_report(
    rows: list[dict], title: str | None = None, files=None, note: str | None = None,
    unresolved: "list[dict] | None" = None
) -> tuple[list[str], list[str]]:
    """Print the harness screen; return (uninvoked, unfalsified) id lists."""
    title = (
        "HARNESS INSTRUMENT SCREEN (tests/ — a guard nothing falsifies is not a guard)"
        if title is None
        else title
    )
    files = HARNESS_INSTRUMENT_FILES if files is None else files
    print()
    print(title)
    print(f"  scanned {len(rows)} instrument fn(s) in {list(files)}")
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
    if unresolved is not None:
        print()
        print(f"  CALL SITES THIS SCREEN COULD NOT ATTRIBUTE ({len(unresolved)}), credited to")
        print("  nobody. Each names a function this census screens under a module it cannot")
        print("  decide from the caller's imports — a method on a value, or a bare name the")
        print("  file never imported. Fail-closed is the rule: a call that MIGHT be an")
        print("  instrument's earns it nothing, because the alternative is polarity credit")
        print("  borrowed from another module, which reads exactly like coverage.")
        for u in unresolved[:12]:
            where = f"{Path(u['file']).name}:{u['line']}"
            print(f"    {where:<44} {u['name']:<24} {u['why']}")
        if len(unresolved) > 12:
            print(f"    ... and {len(unresolved) - 12} more")
    if note:
        print()
        for line in note.strip("\n").splitlines():
            print(f"  {line}")
    return un, nf


def source_dirs(repo=None) -> list[str]:
    """Top-level directory names under *repo* that contain `.py` or `.rs` source."""
    repo = REPO if repo is None else Path(repo)
    out: list[str] = []
    for child in sorted(repo.iterdir()):
        if not child.is_dir() or child.name in FRAME_IGNORE_DIRS:
            continue
        for path in child.rglob("*"):
            if path.suffix not in (".py", ".rs"):
                continue
            if FRAME_IGNORE_DIRS & set(path.parts):
                continue
            out.append(child.name)
            break
    return out


def undeclared(present, declared) -> list[str]:
    """Names in *present* that nobody has declared either way. Pure, so it has a self-test.

    The whole frame declaration rests on this three-line function, which is exactly the
    Guard D shape if it is never watched to disagree — a screen for undeclared scope that
    silently returns `[]` reads identically to a fully declared tree. See `self_test`.
    """
    return sorted(n for n in present if n not in declared)


def frame_report(repo=None) -> tuple[list[str], list[str]]:
    """Print what this census scanned AND what it did not. Return the undeclared items.

    Niobe's tick screen is the model: "41 .rs files, 33 tick-bearing production lines,
    10979 lines held out as `#[cfg(test)]` — UNOBSERVABLE by frame, not zero findings."
    A screen that prints only its findings cannot be distinguished from one that has no
    frame at all, and for six weeks this one could not.
    """
    repo = REPO if repo is None else Path(repo)
    present = source_dirs(repo)
    stray_dirs = undeclared(present, FRAME_DIRS)

    bench_present = sorted(p.name for p in (repo / "bench").glob("*.py"))
    declared_bench = set(BENCH_INSTRUMENT_FILES) | set(BENCH_HELD_OUT)
    stray_files = undeclared(bench_present, declared_bench)

    rs_files = [p for p in (repo / "rust" / "src").rglob("*.rs")]
    tests_py = [p for p in (repo / "tests").rglob("*.py") if "__pycache__" not in p.parts]

    print("CENSUS FRAME (R12 applied to this screen: what it did not look at, said out loud)")
    print(f"  repository root: {repo}")
    print()
    print("  IN FRAME")
    print(
        f"    rust/src   {len(rs_files)} .rs file(s); {len(INSTRUMENT_FILES)} in the instrument "
        f"frame; {len(rs_files) - len(INSTRUMENT_FILES)} not screened — `pub fn` surface only, "
        "and only in files that emit."
    )
    print(
        f"    tests/     {len(tests_py)} .py file(s); {len(HARNESS_INSTRUMENT_FILES)} instrument "
        f"module(s) screened; callers and polarity read from ALL of them."
    )
    print(
        f"    bench/     {len(bench_present)} .py file(s); {len(BENCH_INSTRUMENT_FILES)} instrument "
        f"module(s) screened; {len(BENCH_HELD_OUT)} held out with a stated reason; callers and "
        "polarity read from ALL of them."
    )
    print()
    print("  OUT OF FRAME (declared, with the reason — this is the line whose absence was the bug)")
    for name in sorted(FRAME_DIRS):
        reason = FRAME_DIRS[name]
        if reason.startswith("IN FRAME"):
            continue
        seen = "" if name in present else "   [no source present]"
        print(f"    {name + '/':<14} {reason.split('— ', 1)[-1]}{seen}")
    print()
    print("  bench/ modules held out of the instrument screen:")
    for name in sorted(BENCH_HELD_OUT):
        print(f"    {name:<32} {BENCH_HELD_OUT[name]}")
    print()
    print("  NOTE: a directory or a bench module in neither list is FAIL(drift), not silence.")
    print("  Until 2026-08-02 this census scanned rust/src and tests/ops and printed PASS, and")
    print("  the reader could not tell that from a PASS over the tree. bench/ holds the SM-clock")
    print("  record that refused a device-clock series reading STEADY at RSD 0.0717% and 20.18x")
    print("  wrong; it had never been audited by anything.")
    if stray_dirs:
        print()
        print("  UNDECLARED DIRECTORIES (source present, no frame decision):", file=sys.stderr)
        for name in stray_dirs:
            print(f"    ? {name}/", file=sys.stderr)
    if stray_files:
        print()
        print("  UNDECLARED bench/ MODULES (neither screened nor held out):", file=sys.stderr)
        for name in stray_files:
            print(f"    ? bench/{name}", file=sys.stderr)
    return stray_dirs, stray_files


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

    # The frame screen's own falsifier. `undeclared` returning `[]` is what a fully
    # declared tree looks like AND what a broken screen looks like; the only way to tell
    # them apart is to hand it a tree with a known-undeclared directory and watch it say
    # so. Both polarities, because a screen that always fires is no better.
    frame_cases = [
        ((["rust", "bench"], {"rust": "", "bench": ""}), []),
        ((["rust", "bench", "newdir"], {"rust": "", "bench": ""}), ["newdir"]),
        (([], {"rust": ""}), []),  # a declaration for an absent dir is not a finding
        ((["b", "a"], {}), ["a", "b"]),  # sorted, so the output is stable to diff
    ]
    frame_bad = 0
    for (present, declared), want in frame_cases:
        got = undeclared(present, declared)
        if got != want:
            frame_bad += 1
            print(f"  SELF-TEST FAIL: undeclared({present}, {sorted(declared)}) -> {got} != {want}")
    print(f"  frame-declaration self-test: {len(frame_cases) - frame_bad}/{len(frame_cases)} cases pass")

    # The self-count arm's own falsifier, for the same reason the frame screen has one:
    # `counts == derived` and `no stale prose` are what a healthy baseline looks like AND
    # what an inert screen looks like. Both polarities, on synthetic baselines.
    #
    # The fixtures below DERIVE their own prose from the synthetic array beside them. A
    # literal would have been the defect this arm is for, written into the arm that
    # detects it, and this file is one of the files the source screen reads.
    _rows = ["a.py::f", "a.py::g", "b.py::h"]
    _in_a = sum(1 for r in _rows if r.startswith("a.py"))
    _agrees = {"bench_unfalsified": list(_rows),
               "note": f"{len(_rows)} rows here, {_in_a} rows in a.py"}
    _mutated = dict(_agrees, note=f"{len(_rows) + 1} rows here")
    count_cases = [
        # (baseline, hand-written `counts`, want counts-disagree, want stale prose)
        (_agrees, baseline_counts(_agrees), False, False),
        (_agrees, dict(baseline_counts(_agrees), bench_unfalsified=len(_rows) + 2),
         True, False),
        (_mutated, baseline_counts(_mutated), False, True),
    ]
    count_bad = 0
    for base_doc, published, want_mismatch, want_stale in count_cases:
        derived = baseline_counts(base_doc)
        got_mismatch = published != derived
        got_stale = bool(prose_row_claims(base_doc, derived))
        if (got_mismatch, got_stale) != (want_mismatch, want_stale):
            count_bad += 1
            print(f"  SELF-TEST FAIL: self-count arm on {base_doc!r}/{published!r} -> "
                  f"mismatch={got_mismatch} stale={got_stale}, "
                  f"want mismatch={want_mismatch} stale={want_stale}")
    print(f"  self-count self-test: {len(count_cases) - count_bad}/{len(count_cases)} cases pass")

    # The SOURCE claim screen's own falsifier. Its failure mode is the one that shipped:
    # a screen whose corpus is one file is green about every file it never opened, and
    # "no offenders" is indistinguishable from "nothing was read".
    #
    # Every specimen below is planted from `STALE_SPECIMENS` rather than retyped, in the
    # comment/docstring shapes that were actually missed, and the last two cases are the
    # must-pass polarity: a DERIVED sentence and a CURRENT one both survive the screen.
    _synth = {"bench_unfalsified": ["bench/synth_a.py::f", "bench/synth_a.py::g",
                                    "bench/synth_b.py::h"]}
    _synth_counts = baseline_counts(_synth)
    _mods = sorted(_synth_counts["bench_unfalsified_by_module"])
    _sum = sum(_synth_counts["bench_unfalsified_by_module"].values())
    _named = " and ".join(f"`{m}`" for m in _mods)
    source_cases = [
        (f"# {STALE_SPECIMENS['labelled_row']}\n", True),
        (f"# {STALE_SPECIMENS['labelled_array']}\n", True),
        (f"# {STALE_SPECIMENS['bare_row']}\n", True),
        (f"# {_named}: {STALE_SPECIMENS['module_set']}\n", True),
        (f"# {_named}: {STALE_SPECIMENS['module_sum']}\n", True),
        (f'def f():\n    """{STALE_SPECIMENS["labelled_row"]}."""\n', True),
        (f'x = "{STALE_SPECIMENS["labelled_row"]}"\n', True),
        # Must-pass polarity 1: the figure is derived, so the source carries no digit.
        ('x = f"{counts[\'bench_unfalsified\']} bench unfalsified rows"\n', False),
        # Must-pass polarity 2: the figure is current, spelled out, and correctly bound.
        (f"# {_synth_counts['bench_unfalsified']} bench unfalsified rows\n", False),
        (f"# {_named}: {_sum} rows across these {len(_mods)} modules\n", False),
        # ...and a screen that fired on prose with no claim in it would be useless.
        ("# R13 applies to every row of this census at once.\n", False),
    ]
    source_bad = 0
    for src, want_offender in source_cases:
        got = bool(source_claim_offenders(src, _synth_counts, "<self-test>"))
        if got != want_offender:
            source_bad += 1
            print(f"  SELF-TEST FAIL: source claim screen on {src!r} -> "
                  f"offender={got}, want {want_offender}")
    print(f"  source-claim self-test: {len(source_cases) - source_bad}/"
          f"{len(source_cases)} cases pass")
    return 1 if (bad or frame_bad or count_bad or source_bad) else 0


class CensusInstrumentError(RuntimeError):
    """The census did not reach its observation. R13: never a detection."""


def main(argv: list[str]) -> int:
    if self_test():
        # ERROR(instrument), not FAIL: a broken stripper means the census never reached its
        # observation. Raising rather than returning 1 keeps that distinction out of the
        # caller's hands — see `main_guarded`.
        raise CensusInstrumentError(
            "a self-test failed (comment/string stripper or frame declaration screen); "
            "census not run and nothing was screened"
        )
    stray_dirs, stray_files = frame_report()
    print()
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

    h_unresolved: "list[dict]" = []
    h_rows = harness_survey(unresolved_out=h_unresolved)
    h_uninvoked, h_unfalsified = harness_report(h_rows, unresolved=h_unresolved)

    b_unresolved: "list[dict]" = []
    b_rows = harness_survey(
        tests_root=BENCH, files=BENCH_INSTRUMENT_FILES, fn_re=BENCH_FN, prefix="bench",
        unresolved_out=b_unresolved,
    )
    b_uninvoked, b_unfalsified = harness_report(
        b_rows,
        title="BENCH INSTRUMENT SCREEN (bench/ — the certification apparatus, in frame since 2026-08-02)",
        files=BENCH_INSTRUMENT_FILES,
        unresolved=b_unresolved,
        note=(
            "READ THE UNFALSIFIED COUNT HERE AS A PROPERTY OF THIS SCREEN, NOT AS A VERDICT ON\n"
            "bench/'s TESTS. Most bench instruments are TOTAL functions: they return a token\n"
            "(`STEADY`, `MARGINAL_TAIL`, `NO_STEADY_TAIL`, `SOLE_TENANT`) instead of raising, so\n"
            "a test that watches `gpu_steady_tail` refuse a series was invisible to the original\n"
            "raise-based model. `unfalsified` here means THIS SCREEN has not seen a disagreement,\n"
            "which is exactly what R12 says it should say, and not that no test watches one.\n"
            "Crediting a polarity this screen did not observe would be the Guard D shape with the\n"
            "sign flipped.\n"
            "\n"
            "2026-08-06 (Tank): the value-polarity model handed to Niobe above now EXISTS, so this\n"
            "count is no longer a limit anyone has to accept. A call wrapped in\n"
            "`bench/_polarity.py::refuses(...)` scores reject polarity, and that helper raises when\n"
            "the thing inside it did not refuse — an assertion, not an annotation. First subject:\n"
            "`devices.py::identify_by_uuid`, moved unfalsified -> SCREENED with five planted\n"
            "mutants in `bench/test_devices_identity.py`. EVERY ROW BELOW IS NOW REACHABLE THE\n"
            "SAME WAY. A row that is still `unfalsified` is an instrument nobody has done this\n"
            "for yet, which is a smaller and more actionable statement than the one this note\n"
            "used to make."
        ),
    )

    if "--write-baseline" in argv:
        base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        # Rows held out of the baseline ON PURPOSE stay out of it mechanically. A comment
        # asking the next person not to absorb an open red is not a mechanism; this is.
        # Present specimen: Trinity's two `assert_*` guards from 0f9a4e9, which are
        # correctly flagged `unfalsified` and owe a two-polarity self-test. Baselining
        # them would turn an open item into a green tick, which is the one thing a census
        # must never do.
        held = set(base.get("not_baselined_on_purpose", {}))
        keep = lambda xs: [x for x in xs if x not in held]  # noqa: E731
        base["uninvoked"] = keep(found)
        base["ambiguous"] = sorted(f"{r['file']}::{r['fn']}" for r in ambiguous)
        base["harness_uninvoked"] = keep(h_uninvoked)
        base["harness_unfalsified"] = keep(h_unfalsified)
        base["bench_uninvoked"] = keep(b_uninvoked)
        base["bench_unfalsified"] = keep(b_unfalsified)
        base["counts"] = baseline_counts(base)
        stale = prose_row_claims(base, base["counts"])
        stale += corpus_claim_offenders(base["counts"])
        stale += specimen_offenders(base, base["counts"])
        BASELINE.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {BASELINE} ({len(found)} uninvoked, {len(h_unfalsified)} unfalsified)")
        if stale:
            # Written first, then reported: the arrays and `counts` are now right, and the
            # prose that disagrees with them is a finding the writer must not walk past.
            print(
                "\nFAIL: a census count claim is not witnessed by the collection it "
                "names:", file=sys.stderr)
            for x in stale:
                print(f"  ! {x}", file=sys.stderr)
            print("\nCENSUS VERDICT: FAIL(drift)")
            return 1
        if held:
            print(f"held out of the baseline on purpose ({len(held)}): {sorted(held)}")
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
        ("bench_uninvoked", b_uninvoked),
        ("bench_unfalsified", b_unfalsified),
    ):
        if key not in base:
            print(f"\nFAIL: baseline has no `{key}`; run --write-baseline.", file=sys.stderr)
            h_bad = True
            continue
        exp = sorted(base[key])
        added = [x for x in current if x not in exp]
        removed = [x for x in exp if x not in current]
        label = key.split("_", 1)[1]
        if added:
            h_bad = True
            print(f"\nFAIL: {len(added)} NEW {label} instrument(s):", file=sys.stderr)
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
                f"\nFAIL: {len(removed)} instrument(s) left `{label}` — "
                "good news, update the baseline:",
                file=sys.stderr,
            )
            for x in removed:
                print(f"  - {x}", file=sys.stderr)

    # The frame arm. An undeclared directory or bench module is drift in the scope itself:
    # the census would otherwise render a verdict over a tree it cannot account for, which
    # is the defect this whole section exists to remove.
    frame_bad = False
    if stray_dirs or stray_files:
        frame_bad = True
        print(
            f"\nFAIL: {len(stray_dirs) + len(stray_files)} undeclared item(s) in the census "
            "frame. Add each to FRAME_DIRS / BENCH_INSTRUMENT_FILES / BENCH_HELD_OUT with a "
            "reason. 'Not scanned' is an acceptable answer; not saying which is not.",
            file=sys.stderr,
        )

    # The self-count arm. A summary that disagrees with the collection it summarises is
    # drift of a kind the four array comparisons above cannot see: they compare the
    # baseline's arrays against the tree, and both copies of a count can be wrong together
    # while every array matches. `counts` is re-derived here from the arrays that were just
    # confirmed, so a hand-edited summary, or prose quoting a number nothing derives, is red.
    counts_bad = False
    want_counts = baseline_counts(base)
    have_counts = base.get("counts")
    if have_counts is None:
        counts_bad = True
        print(
            "\nFAIL: baseline has no `counts`; run --write-baseline. A census that publishes "
            "no derived summary invites a hand-typed one.",
            file=sys.stderr,
        )
    elif have_counts != want_counts:
        counts_bad = True
        print("\nFAIL: the baseline's `counts` disagree with the arrays they summarise:",
              file=sys.stderr)
        for key in sorted(set(want_counts) | set(have_counts)):
            if have_counts.get(key) != want_counts.get(key):
                print(f"  ! {key}: baseline says {have_counts.get(key)!r}, "
                      f"the array has {want_counts.get(key)!r}", file=sys.stderr)
        print(
            "  Counts are derived, never typed: re-run --write-baseline rather than "
            "editing the number.",
            file=sys.stderr,
        )
    stale_prose = prose_row_claims(base, want_counts)
    if stale_prose:
        counts_bad = True
        print("\nFAIL: the baseline's prose quotes a row count this census does not derive:",
              file=sys.stderr)
        for x in stale_prose:
            print(f"  ! {x}", file=sys.stderr)
        print(
            "  Cite `counts` instead of restating a number. A figure a reader cannot "
            "re-derive is the defect this arm was added for.",
            file=sys.stderr,
        )

    # The SOURCE corpus arm. The screen above reads the generated document; this one reads
    # the comments, docstrings and test prose beside it, where the same stale figures
    # survived the first repair verbatim. It also fails on an in-frame file that starts
    # making census count claims without being declared, so the corpus cannot go blind by
    # standing still while the tree moves.
    source_stale = corpus_claim_offenders(want_counts)
    if source_stale:
        counts_bad = True
        print("\nFAIL: a census count claim in source or test prose is not witnessed by "
              "the collection it names:", file=sys.stderr)
        for x in source_stale:
            print(f"  ! {x}", file=sys.stderr)
        print(
            "  Derive the figure (`f\"{counts['bench_unfalsified']} rows\"`), cite the "
            "field by name, or drop the number. A comment is where a reader goes to find "
            "out what a mechanism means, so a stale count in one is quoted as readily as "
            "a stale count in the artifact.",
            file=sys.stderr,
        )

    # The specimen arm. `STALE_SPECIMENS` is the one place a stale figure may still be
    # spelled, so it owes the opposite witness: every entry must STILL be convicted by the
    # live screen. A specimen that has quietly become true is a claim, not a quotation.
    specimen_stale = specimen_offenders(base, want_counts)
    if specimen_stale:
        counts_bad = True
        print("\nFAIL: a historical specimen is no longer stale:", file=sys.stderr)
        for x in specimen_stale:
            print(f"  ! {x}", file=sys.stderr)

    if new or gone or h_bad or frame_bad or counts_bad:
        print("\nCENSUS VERDICT: FAIL(drift)")
        return 1
    print(f"\nOK: uninvoked set matches the baseline ({len(found)} known).")
    print(
        f"OK: harness screen matches the baseline "
        f"({len(h_uninvoked)} uninvoked, {len(h_unfalsified)} unfalsified)."
    )
    print(
        f"OK: bench screen matches the baseline "
        f"({len(b_uninvoked)} uninvoked, {len(b_unfalsified)} unfalsified)."
    )
    print("OK: every source directory and every bench/ module has a frame decision on record.")
    print("OK: every summary count in the baseline was re-derived from the array it summarises.")
    print(
        f"OK: no unwitnessed census count claim in the baseline or in the "
        f"{len(SOURCE_CLAIM_CORPUS)} declared source file(s), and every historical "
        f"specimen is still stale."
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
