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

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
REPO = HERE.parents[2]
TESTS = REPO / "tests"
BENCH = REPO / "bench"
BASELINE = HERE.parent / "instrument_census.json"

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
# Consequence, stated up front so it is not read as a regression: this admits ~90 rows,
# most of them `unfalsified`. `unfalsified` is the honest state of an instrument nothing
# has watched disagree. It was always the state of these; the census simply could not say
# so, because it had never looked.
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
    # Arrived 2026-08-08 (Morpheus, issue #81). Declared on the way in rather than left for
    # the frame arm to catch, which is what this dict is for.
    "public_paths.py": (
        "screen — it decides whether a RECORD may be published (absolute local paths) and "
        "whether a DESTINATION may be written (git-tracked evidence). Neither is a verdict "
        "about a measurement, which is BENCH_INSTRUMENT_FILES' criterion: it never reads a "
        "number, and a record it passes is one it has said nothing about. Its own two "
        "polarities are screened in test_public_paths.py, which plants each recognised path "
        "form and requires the screen to fire."
    ),
    "test_public_paths.py": (
        "test module — a caller, screened as polarity, not as an instrument."
    ),
    "test_withdrawn_streaming_claim.py": (
        "test module — a caller. It reads docs/PERF.md and committed JSON and rules on "
        "whether the WITHDRAWN §26.4 figures stay withdrawn; the arithmetic it checks is "
        "the artifact's, not its own."
    ),
}


def _harness_instruments(tests_root=None, files=None, fn_re=None, prefix="tests") -> dict[str, str]:
    """Return {fn_name: "file::fn"} for every harness instrument."""
    import ast as _ast

    tests_root = TESTS if tests_root is None else Path(tests_root)
    files = HARNESS_INSTRUMENT_FILES if files is None else files
    fn_re = HARNESS_FN if fn_re is None else fn_re
    out: dict[str, str] = {}
    for rel in files:
        path = tests_root / rel
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if fn_re.search(node.name):
                    out[node.name] = f"{prefix}/{rel}::{node.name}"
    return out


def _fixture_instruments(tests_root=None, files=None, fn_re=None) -> set[str]:
    """Return the subset of harness instruments that are pytest fixtures.

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
                    out.add(node.name)
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
VALUE_REJECT_FN = frozenset({"refuses"})
VALUE_ACCEPT_FN = frozenset({"selects"})


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


def harness_survey(tests_root=None, files=None, fn_re=None, prefix="tests") -> list[dict]:
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
    fn_re = HARNESS_FN if fn_re is None else fn_re
    names = _harness_instruments(tests_root, files, fn_re, prefix)
    fixtures = _fixture_instruments(tests_root, files, fn_re)
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
            # A fixture is depended on by naming it as a parameter.  This is the only
            # invocation it ever gets, so it is counted here and nowhere else.  It never
            # supplies a polarity: nothing in the parameter list says which answer the
            # fixture gave.
            for arg in fn.args.args:
                if arg.arg in fixtures:
                    stats[arg.arg]["calls"] += 1
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
                if id(node) in raising or id(node) in value_reject:
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


def harness_report(
    rows: list[dict], title: str | None = None, files=None, note: str | None = None
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
    return 1 if (bad or frame_bad) else 0


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

    h_rows = harness_survey()
    h_uninvoked, h_unfalsified = harness_report(h_rows)

    b_rows = harness_survey(
        tests_root=BENCH, files=BENCH_INSTRUMENT_FILES, fn_re=BENCH_FN, prefix="bench"
    )
    b_uninvoked, b_unfalsified = harness_report(
        b_rows,
        title="BENCH INSTRUMENT SCREEN (bench/ — the certification apparatus, in frame since 2026-08-02)",
        files=BENCH_INSTRUMENT_FILES,
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
        BASELINE.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {BASELINE} ({len(found)} uninvoked, {len(h_unfalsified)} unfalsified)")
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

    if new or gone or h_bad or frame_bad:
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
