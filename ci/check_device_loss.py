#!/usr/bin/env python3
"""``check_device_loss`` — a lost device that exits 0 is not a run, and not a number.

WHAT HAPPENED
=============

Tank, measuring KV bytes at ctx 512 (``bench/results/ctx512_device_lost.txt``)::

    [vulkan-ep] ERROR: vkWaitForFences failed: The logical device has been lost.
    EP Error: ... Status Message: vkWaitForFences failed
    Falling back to ['CPUExecutionProvider'] and retrying.
    EXIT=0

The device was lost, the work moved to the CPU EP, and the process reported success.
Both ctx-512 points were truncated by it, and **differencing two truncated points produced
an apparent 6.7% KV saving that was an observation ending early.**  That is the shape to
keep in mind: a run that dies partway and exits 0 does not look like a failure — *it looks
like a smaller number*.

WHY A SECOND CHECK, WHEN ``check_fatal_log`` EXISTS
===================================================

Because ``check_fatal_log`` would not have caught this, and I can show it rather than
argue it.  Its markers come from ``tests/ops/_verdict.py``::

    FATAL_LOG_MARKERS = ("Falling back to CPUExecutionProvider", "Falling back to CPU")

ORT's actual line is ``Falling back to ['CPUExecutionProvider'] and retrying.`` — a list
repr, so neither marker is a substring of it.  Run
``_verdict.find_fatal_log_lines`` over Tank's real artifact and it returns **zero hits on a
log that announces the fallback twice**.  A marker list that has been cited as the second
witness for five incidents does not match the line it was written for.  The cross-check at
the bottom of this file reports that as its own condition so it routes to the vocabulary's
owner rather than being quietly patched here — one vocabulary, not two.

Three further reaches this check has and that one does not:

  * It reads **artifacts**, not only the lane's teed pytest log.  Tank's run was a bench
    probe; no captured-log check was ever pointed at it.
  * It keys on the **EP's own** device-lost text as well as ORT's announcement.  A device
    loss during a run ORT never fell back from prints the first and not the second.
  * It has a **structural** rule that needs no text at all (see below), which is the one
    that survives a log format change.

SIGNAL PREFERENCE — the coordinator asked for something stronger than text if it exists
=======================================================================================

In descending order of strength, and each is applied where its input is present:

  1. **Structural (no text).**  A counters artifact that declares what it expected —
     ``iters``, ``expected_compute_calls`` — and observed less.  Or ``uploads ==
     readbacks + 1``: an inference caught in flight, which is Tank's own screen in
     ``probe_device_memory_kv.py::point_validity``.  This cannot be defeated by a log
     format change or by a driver that words its error differently.
  2. **The EP's own device-lost text.**  ``VK_ERROR_DEVICE_LOST`` and "the logical device
     has been lost" are *specification* language, not vendor prose, so they are stable
     across drivers in a way ORT's Python-side formatting is not.
  3. **The EP's BROKEN COMMITMENT warning** — the EP saying it claimed a subgraph and did
     not keep the claim.
  4. **ORT's runtime fallback announcement**, matched form-tolerantly.

There is no EP counter for device loss.  If Tank adds one — a monotonically increasing
``device_lost`` — rule 1 would subsume rules 2-4 for the counters lane and this file would
prefer it; that is an ask recorded in the decision, not a change made to his file here.

EXIT CODES ARE NOT AN INPUT
===========================

This check never reads the exit status of the thing that produced its evidence.  The
defect it exists for *is* an exit status of 0, so accepting one would be accepting the
defect as a filter.

TERMINAL STATES (§10.0.1 R13)
=============================

    0  DEVICE-LOSS: PASS
    1  DEVICE-LOSS: FAIL(condition=device_lost_reported)
       DEVICE-LOSS: FAIL(condition=observation_ended_early)
       DEVICE-LOSS: FAIL(condition=broken_commitment_reported)
       DEVICE-LOSS: FAIL(condition=runtime_fallback_announced)
       DEVICE-LOSS: FAIL(condition=marker_list_misses_real_line)
    4  DEVICE-LOSS: ERROR(instrument=...)  — including "nothing was scanned", because a
       check with no input has observed nothing, and UNOBSERVABLE is not zero hits (R12).

Findings quote the matching text.  Never a count without its text (R13).

USAGE
    python ci/check_device_loss.py <path> [<path> ...] [--lane-marker=PATH]
    python ci/check_device_loss.py --lane            # the default artifact set
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

TAG = "DEVICE-LOSS"

#: Specification language, quoted from the Vulkan spec and from the loader's own message.
#: These are stable across vendors in a way a host-side format string is not.
DEVICE_LOST_MARKERS: tuple[str, ...] = (
    "VK_ERROR_DEVICE_LOST",
    "logical device has been lost",
    "device has been lost",
    "ERROR_DEVICE_LOST",
)

#: The EP's own admission that it claimed a subgraph and did not keep the claim.
BROKEN_COMMITMENT_MARKERS: tuple[str, ...] = ("BROKEN COMMITMENT",)

#: ORT's runtime fallback announcement, matched form-tolerantly. `Falling back to
#: ['CPUExecutionProvider']` is a list repr; `Falling back to CPUExecutionProvider` is
#: not. Anchoring on the two words plus the provider name, with anything between, covers
#: both and every quoting style either side of them.
FALLBACK_RE = re.compile(r"Falling back to\b[^\n]{0,40}?CPUExecutionProvider")

TEXT_SUFFIXES = {".txt", ".log", ".out", ".err", ".md"}
JSON_SUFFIXES = {".json"}

#: Historical incident records. `bench/results/ctx512_device_lost.txt` IS a device loss —
#: it is the artifact of one, deliberately kept — so a directory scan would find it
#: forever and this check would be permanently red on evidence of a defect it did not
#: catch live. Naming such files in a small, owned, reasoned file is the same discipline
#: as ci/tick_conversion_allowlist.json: the exclusion is visible in the diff that adds
#: it, it is counted and printed on every run, and an entry pointing at a file that no
#: longer exists is itself a finding.
INCIDENT_RECORDS = Path(__file__).resolve().parent / "device_loss_incident_records.json"

#: Conditions this check decides about its own EXCLUSION LIST rather than about a run.
#: They are decided on every invocation that has an exclusion list, whatever the paths.
#:
#: WHY THE FIRST TWO ARE NOT ENOUGH, MEASURED ON `main` 2026-08-07 (issue #24)
#: --------------------------------------------------------------------------
#: Until today an entry excluded its file *wholesale and forever*, and said nothing about
#: WHAT it excluded. Two failures follow from that and both were live on `main`:
#:
#:   1. Eight artifacts of real, already-diagnosed device losses (Tank's ctx-4096 gate
#:      captures, Switch's ctx-4096 KV-chain record, Trinity's criterion-3(a) devunset
#:      liveness arm) landed committed with NO record at all. The lane screen went red and
#:      stayed red on every run that reached it, which is the state the exclusion list
#:      exists to prevent: a check that is always red is a check nobody reads.
#:   2. The obvious repair — name the eight files and move on — would have bought green with
#:      an exclusion that also blinds the screen to the NEXT loss recorded in the same
#:      file. `kv_bytes_earned-armed.json` has been excluded whole since 2026-08-02; a new
#:      truncated point appended to it would have been invisible.
#:
#: So an exclusion now has to declare the findings it accounts for, and the check re-reads
#: every excluded file to hold it to that. An exclusion that covers nothing is deleted; an
#: exclusion that covers more than it declares is a NEW incident hiding behind an old one.
RECORD_CONDITIONS = (
    "incident_record_rot",
    "incident_record_covers_nothing",
    "incident_record_widened",
)


def report_pass(detail: str) -> int:
    print(f"{TAG}: PASS — {detail}", flush=True)
    return EXIT_PASS


def report_fail(condition: str, detail: str) -> int:
    print(f"{TAG}: FAIL(condition={condition})", flush=True)
    print(detail, flush=True)
    print(
        f"{TAG}: this is a finding about a run, not about this check. The evidence is "
        "quoted above with the file it came from. Note that the producing process may "
        "well have exited 0 — that is the defect, not a reason to discount the finding.",
        flush=True,
    )
    return EXIT_FAIL_CONDITION


def report_instrument_error(instrument: str, detail: str) -> int:
    print(f"{TAG}: ERROR(instrument={instrument})", flush=True)
    print(detail, flush=True)
    print(
        f"{TAG}: the check did not reach its observation, so this is NOT a detection "
        "(§10.0.1 R13). Do not route it as a device fault and do not read it as a clean "
        "run.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------


def gather(paths: list[Path]) -> tuple[list[Path], list[str]]:
    """Expand directories; return the files this check will read, and what it declined.

    What it declined is printed, always. A scan that does not say what it skipped invites
    a reader to take its silence for coverage.
    """
    files: list[Path] = []
    skipped: list[str] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if not child.is_file():
                    continue
                if child.suffix.lower() in TEXT_SUFFIXES | JSON_SUFFIXES:
                    files.append(child)
                else:
                    skipped.append(f"{child} (suffix {child.suffix or '<none>'})")
        elif path.is_file():
            files.append(path)
        else:
            skipped.append(f"{path} (does not exist)")
    return files, skipped


def load_incident_records(
    records_path: "Path | None" = None,
) -> tuple[dict[Path, dict], list[str], str]:
    """Return {resolved path: entry}, rot findings, and an instrument-error reason.

    Rot is a finding, not a silent no-op: an entry naming a file that is gone is an
    exclusion nobody can check any more.
    """
    path = records_path or INCIDENT_RECORDS
    if not path.exists():
        return {}, [], ""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {}, [], f"{path} is not readable JSON: {exc}"
    entries = doc.get("records", [])
    if not isinstance(entries, list):
        return {}, [], f"{path}: 'records' is not a list"
    mapping: dict[Path, dict] = {}
    rot: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("file"):
            return {}, [], f"{path}: a record has no 'file'"
        for field in ("reason", "owner", "date"):
            if not entry.get(field):
                return {}, [], (
                    f"{path}: record {entry['file']} has no '{field}'. "
                    "An exclusion without a reason, an owner and a date is an "
                    "exclusion nobody can review."
                )
        # `witness` is what makes the exclusion BOUNDED. Without it an entry silences a
        # file for all time and for all conditions, so the day a second, different loss is
        # recorded in the same file nothing says so. Refusing to run without it is
        # deliberate: a missing bound is an instrument outage, not a permissive default.
        witness = entry.get("witness")
        if not isinstance(witness, dict) or not witness:
            return {}, [], (
                f"{path}: record {entry['file']} has no 'witness'. An exclusion must "
                "name the finding(s) it accounts for, by condition and count, or it is "
                "an exclusion over everything that file will ever say."
            )
        for condition, count in witness.items():
            if condition not in TIER_TREE_WIDE or condition in RECORD_CONDITIONS:
                return {}, [], (
                    f"{path}: record {entry['file']} accounts for {condition!r}, which "
                    "is not a condition a directory scan decides on an artifact "
                    f"({', '.join(c for c in TIER_TREE_WIDE if c not in RECORD_CONDITIONS)})."
                )
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                return {}, [], (
                    f"{path}: record {entry['file']} accounts for {condition} with "
                    f"{count!r}. A witness count below 1 excludes nothing and cannot be "
                    "reviewed."
                )
        target = (REPO_ROOT / entry["file"]).resolve()
        if not target.exists():
            rot.append(
                f"{entry['file']}: named as a historical incident record but no longer "
                f"present, so this exclusion can no longer be reviewed "
                f"(owner {entry['owner']}, recorded {entry['date']})"
            )
            continue
        mapping[target] = entry
    return mapping, rot, ""


def scan_text(path: Path, text: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {
        "device_lost_reported": [],
        "broken_commitment_reported": [],
        "runtime_fallback_announced": [],
    }
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(m in line for m in DEVICE_LOST_MARKERS):
            hits["device_lost_reported"].append(line)
        if any(m in line for m in BROKEN_COMMITMENT_MARKERS):
            hits["broken_commitment_reported"].append(line)
        if FALLBACK_RE.search(line):
            hits["runtime_fallback_announced"].append(line)
    return hits


def json_embedded_text(doc: object) -> str:
    """Every string VALUE in a JSON artifact, joined, so a captured log inside one is read.

    MEASURED BLINDNESS, 2026-08-04, and the reason this exists
    ----------------------------------------------------------

    A probe that captures a worker's stderr and stores it under a key such as
    ``stderr_tail`` produces a JSON artifact whose device-loss witness is *inside a JSON
    string*.  ORT's C++ sink writes UTF-16LE, so that captured text arrives NUL-separated,
    and ``json.dumps`` escapes each NUL as the six literal characters ``\\u0000``.  The
    file's raw text therefore contains ``V\\u0000K\\u0000_\\u0000E...`` and **never** the
    substring ``VK_ERROR_DEVICE_LOST``.

    ``searchable_text`` cannot help: it normalises real NUL bytes, and inside a JSON source
    file there are none — the NULs only exist after the string is decoded.  So the check
    scanned the raw file, matched nothing, and reported PASS.

    Demonstrated rather than argued.  Take the committed incident record
    ``bench/results/phi35_kv_chain-ctx4096-BOTH-dev0.json``, delete the Python traceback
    from ``stderr_tail`` and keep only ORT's own wide-encoded line — the exact artifact a
    run produces when ORT falls back and the process exits 0, which is the original ctx-512
    incident this whole file was written for — and this check returned::

        DEVICE-LOSS: PASS — 1 artifact(s) read; nothing found for: device_lost_reported, ...

    The committed record is red today only because the *Python traceback* in the same field
    happens to be plain ASCII.  The check was reading the accident, not the witness.

    THE SHAPE THIS IS AN INSTANCE OF
    --------------------------------

    Switch read ``alloc_device failed`` in a stderr, stopped, and missed
    ``vkQueueSubmit failed: The logical device has been lost`` further down the same
    capture.  Three losses are now on record where the reports named one.  A reader that
    stops at the first form it can read misses the second, and an instrument that reads one
    encoding of a capture is that reader wearing a badge.

    Scanning decoded values *in addition to* the raw text, rather than instead of it, so
    nothing that was previously visible becomes invisible: a JSON file's own keys and its
    non-string structure stay in the raw scan, and the two scans are de-duplicated by the
    caller.
    """
    out: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(doc)
    return "\n".join(out)


#: Fields an artifact may use to declare what it expected to observe. A structural rule
#: needs the producer to state its expectation; where none is stated the rule reports
#: UNOBSERVABLE for that artifact rather than assuming the run was whole.
EXPECTED_FIELDS = ("iters", "iterations", "expected_compute_calls", "expected_dispatches")


def scan_counters(doc: object) -> tuple[list[str], bool, int]:
    """Structural truncation rules.

    Returns (findings, this_artifact_was_decidable, declared_rejections).

    A truncated point the producer has already put under a `rejected_*` key is not a
    silent truncation — it is the producer reporting one. Those are counted and printed,
    never counted as findings: the defect this check exists for is a short run that was
    KEPT. Counting them as findings would punish the one artifact that did the right
    thing and would make the honest producer look like the defective one.
    """
    findings: list[str] = []
    decidable = False
    declared = 0
    if not isinstance(doc, dict):
        return findings, decidable, declared

    def walk(node: object, where: str) -> None:
        nonlocal decidable, declared
        already_rejected = any(
            seg.startswith("rejected") or seg.startswith("discarded")
            for seg in where.replace("[", ".[").split(".")
        )
        if not isinstance(node, dict):
            if isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, f"{where}[{i}]")
            return
        expected = None
        expected_key = None
        for key in EXPECTED_FIELDS:
            value = node.get(key)
            if isinstance(value, int) and value > 0:
                expected, expected_key = value, key
                break
        observed = node.get("compute_calls")
        if expected is not None and isinstance(observed, int):
            decidable = True
            if observed < expected:
                if already_rejected:
                    declared += 1
                else:
                    findings.append(
                        f"{where}: {expected_key}={expected} but compute_calls={observed} — "
                        f"the run stopped {expected - observed} inference(s) short of what it "
                        "declared it would do, and any figure differenced from it is an "
                        "observation ending early rather than a smaller quantity"
                    )
        uploads = node.get("uploads")
        if not isinstance(uploads, int):
            uploads = node.get("session_staging_uploads")
        readbacks = node.get("readbacks")
        if not isinstance(readbacks, int):
            readbacks = node.get("session_staging_readbacks")
        if isinstance(uploads, int) and isinstance(readbacks, int):
            decidable = True
            if uploads == readbacks + 1:
                if already_rejected:
                    declared += 1
                else:
                    findings.append(
                        f"{where}: uploads={uploads}, readbacks={readbacks} — uploads == "
                        "readbacks + 1 is an inference caught in flight (Tank's point_validity "
                        "screen); the last upload has no matching readback"
                    )
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                walk(value, f"{where}.{key}")

    walk(doc, "$")
    return findings, decidable, declared


# ---------------------------------------------------------------------------
# The cross-check against the shared vocabulary
# ---------------------------------------------------------------------------


def searchable_text(raw: str) -> str:
    """Make a captured log searchable however ORT encoded it.

    Found 2026-08-02 by an arm of this file's own negative control, on that arm's first
    run. ORT's C++ log sink writes UTF-16LE into an otherwise UTF-8 file; read as UTF-8
    those regions arrive NUL-separated, and every regex in `scan_text` — device-lost text
    included — silently matches nothing. This check would then have read such a log as a
    clean run. `bench/results/trinity-suite-dev1.log` happens to carry its message in both
    encodings; had it carried only the wide one, this file would have seen an empty log.

    Deliberately delegated to `tests/ops/_verdict.py` rather than reimplemented, for the
    same reason the marker list is: one normalisation, one owner, no second dialect. If
    the helper is unavailable the raw text is returned rather than a private substitute —
    a wrong answer from a copy is worse than the original blindness, because it looks
    maintained.
    """
    sys.path.insert(0, str(REPO_ROOT / "tests" / "ops"))
    try:
        import _verdict  # type: ignore

        return _verdict.normalise_log_text(raw)
    except Exception:  # noqa: BLE001
        return raw


def marker_cross_check(samples: list[tuple[Path, str]]) -> list[str]:
    """Lines this file matches that the shared marker list does not.

    Not fixed here on purpose. `tests/ops/_verdict.py` is the one vocabulary and it has an
    owner; a second private list in ci/ is exactly the "two dialects" failure the vocabulary
    exists to prevent. So the divergence is reported as a finding addressed to its owner.

    TWO DEFECTS OF MY OWN, FOUND 2026-08-02 AND FIXED HERE
    ------------------------------------------------------

    Trinity repaired the shared markers so they match what ORT actually prints. This
    function went on reporting `marker_list_misses_real_line` against Tank's
    `ctx512_device_lost.txt` afterwards — a **false** finding, because
    `ci/check_fatal_log.py` now goes red on that exact file, quoting all three lines.
    Worse, my own negative control had an arm *requiring* that finding to appear, so
    repairing the divergence would have turned my control red and blamed the repair.
    That is R9 amendment 5 in my own file: a check that had stopped being able to tell
    repair from breakage, while still reading as confident.

    1. **It compared a physical line to a matched span by string equality.** That was
       adequate only while `find_fatal_log_lines` returned whole lines. A form-tolerant
       matcher necessarily returns the *span it matched*, so
       ``Falling back to ['CPUExecutionProvider'] and retrying`` (no trailing period, the
       regex ends at ``retrying``) never equals the stripped physical line that carries
       one. The comparison had silently become a test of punctuation rather than of
       coverage. The question worth asking is not "is my string in their list" but
       **"would the shared vocabulary see this line at all"**, so that is now the
       question asked.

    2. **My side scanned un-normalised text.** ORT's C++ sink writes UTF-16LE into an
       otherwise UTF-8 file; read as UTF-8 those regions arrive NUL-separated and
       `FALLBACK_RE` cannot match them. So on a log carrying *only* the wide form, `mine`
       would have been empty, no divergence would have been reported, and this function
       would have read as agreement. Two blind scanners agree perfectly. Both sides are
       now normalised through the one shared helper, which also means there is still
       exactly one implementation of the normalisation.

    The finding this function exists to raise is unchanged and still reachable: a line
    *I* can see that the shared vocabulary cannot. It is now reachable for that reason
    only.
    """
    sys.path.insert(0, str(REPO_ROOT / "tests" / "ops"))
    try:
        import _verdict  # type: ignore
    except Exception:  # noqa: BLE001
        return []
    normalise = getattr(_verdict, "normalise_log_text", None)
    findings: list[str] = []
    for path, text in samples:
        searchable = normalise(text) if normalise else text
        for raw_line in searchable.splitlines():
            line = raw_line.strip()
            if not FALLBACK_RE.search(line):
                continue
            # Coverage, not equality: hand the shared matcher this one line and ask
            # whether it finds anything in it. A line is only a divergence if the
            # vocabulary is blind to it.
            if not _verdict.find_fatal_log_lines(line):
                findings.append(f"{path}: {line}")
    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_LANE_PATHS = ("bench/results",)

#: Two tiers, and the reason is a real observation rather than a preference.
#:
#: Scanning the whole artifact tree for the weaker text signals produces findings on
#: files that are *supposed* to contain them: `broken-commitment-control.json` and the
#: criterion-4/5 ICD witnesses induce those failures deliberately and record the text as
#: their result, and `trinity-suite-dev*.log` contains an assertion message that quotes
#: the fallback line. Treating those as incidents would make the check red forever on
#: evidence that everything works, which is the fastest route to a check nobody reads.
#:
#: So the tree-wide tier carries only what no control on this project deliberately
#: produces: the device-lost text, which is Vulkan specification language, and the
#: structural rule, which is arithmetic on the producer's own declared expectation and
#: needs no text at all. Everything else requires a caller to say "this file is one run's
#: evidence" by naming it — positionally or with --run-log.
TIER_TREE_WIDE = ("device_lost_reported", "observation_ended_early", "incident_record_rot")
TIER_NAMED_RUN_ONLY = (
    "broken_commitment_reported",
    "runtime_fallback_announced",
    "marker_list_misses_real_line",
)

#: What a directory scan decides about an ARTIFACT, as opposed to about the record file.
#: These are the only conditions an exclusion may account for.
ARTIFACT_TREE_WIDE = tuple(c for c in TIER_TREE_WIDE if c not in RECORD_CONDITIONS)


class Scan(NamedTuple):
    """One artifact, read once.

    Extracted so the EXCLUDED files go through the identical reader as the scanned ones.
    A second, private reader for the exclusion audit would be a second dialect, and the
    audit would then be checking a file this screen never actually reads that way.
    """

    hits: dict[str, list[str]]
    decidable: bool
    declared_rejections: int
    embedded: bool
    searchable: str
    error: str


def scan_artifact(path: Path, named: bool) -> Scan:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return Scan({}, False, 0, False, "", f"{path}: {exc}")
    searchable = searchable_text(raw)
    hits: dict[str, list[str]] = {}
    decidable = False
    declared = 0
    embedded = False
    if path.suffix.lower() in JSON_SUFFIXES:
        try:
            doc = json.loads(raw)
        except Exception:  # noqa: BLE001
            doc = None
        if doc is not None:
            found, decidable, declared = scan_counters(doc)
            if found:
                hits["observation_ended_early"] = list(found)
            # A captured log stored inside a JSON string is not visible in the file's raw
            # text once its NULs are `\u0000` escapes. Scan the decoded values too. See
            # `json_embedded_text` — this blindness was measured, not supposed.
            emb = searchable_text(json_embedded_text(doc))
            if emb.strip():
                embedded = True
                searchable = searchable + "\n" + emb
    for condition, lines in scan_text(path, searchable).items():
        if not lines:
            continue
        if condition in TIER_NAMED_RUN_ONLY and not named:
            continue
        # De-duplicated: a line present in both the raw text and the decoded values of the
        # same file is one finding, not two.
        seen = hits.setdefault(condition, [])
        for line in lines:
            if line not in seen:
                seen.append(line)
    return Scan(hits, decidable, declared, embedded, searchable, "")


def audit_exclusions(
    excluded: list[Path], records: dict[Path, dict]
) -> tuple[dict[str, list[str]], list[str]]:
    """Hold every exclusion to the findings it declared it accounts for.

    Returns (findings by condition, frame notes).

    Two findings, and the asymmetry between them is the point:

    * **covers_nothing** — the file yields no finding at all. The entry silences nothing
      today and can only silence something tomorrow. That is an exclusion nobody will ever
      have a reason to look at again, which is how an exclusion list rots into a blindfold.

    * **widened** — the file yields a finding under a condition the record never named, or
      MORE findings of a named condition than it accounted for. That is a second incident
      arriving inside a file that was already forgiven, which is precisely the defect a
      whole-file exclusion cannot express.

    A *narrowed* witness (fewer findings than declared, none new) is printed in the frame
    and is deliberately NOT a finding. Every remaining finding is still inside a condition
    somebody reviewed, and every finding that vanished is one fewer thing excluded; a
    shrinking witness set cannot hide an incident. Making it red would only teach people to
    keep the numbers loose.
    """
    findings: dict[str, list[str]] = {}
    notes: list[str] = []
    for path in excluded:
        entry = records[path.resolve()]
        declared: dict[str, int] = entry.get("witness", {})
        scan = scan_artifact(path, named=False)
        if scan.error:
            findings.setdefault("incident_record_covers_nothing", []).append(
                f"{entry['file']}: excluded by name but could not be re-read "
                f"({scan.error}), so the exclusion cannot be held to what it accounts "
                f"for (owner {entry['owner']}, recorded {entry['date']})"
            )
            continue
        observed = {c: len(v) for c, v in scan.hits.items() if v and c in ARTIFACT_TREE_WIDE}
        if not observed:
            findings.setdefault("incident_record_covers_nothing", []).append(
                f"{entry['file']}: the exclusion is in force, but the file yields no "
                f"finding for {', '.join(ARTIFACT_TREE_WIDE)}. It excludes nothing today "
                f"and can only blind this check tomorrow — delete the entry "
                f"(owner {entry['owner']}, recorded {entry['date']})"
            )
            continue
        widened = [
            f"{condition}: {count} finding(s) present, {declared.get(condition, 0)} "
            "accounted for"
            for condition, count in sorted(observed.items())
            if count > declared.get(condition, 0)
        ]
        if widened:
            findings.setdefault("incident_record_widened", []).append(
                f"{entry['file']}: "
                + "; ".join(widened)
                + f" — the exclusion was reviewed for less than this file now says "
                f"(owner {entry['owner']}, recorded {entry['date']})"
            )
        narrowed = [
            f"{condition}: {observed.get(condition, 0)} of {count} accounted for"
            for condition, count in sorted(declared.items())
            if observed.get(condition, 0) < count
        ]
        if narrowed and not widened:
            notes.append(f"{entry['file']}: " + "; ".join(narrowed))
    return findings, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--lane", action="store_true", help="scan the default artifact set")
    ap.add_argument(
        "--run-log",
        action="append",
        default=[],
        type=Path,
        help=(
            "a file that is one run's own evidence. Named files get the full condition "
            "set; files reached by expanding a directory get only the two conditions no "
            "control on this project legitimately produces (see TIER_TREE_WIDE)."
        ),
    )
    ap.add_argument("--lane-marker", default="")
    ap.add_argument(
        "--incident-records",
        type=Path,
        default=None,
        help=(
            "read the exclusion list from PATH instead of ci/"
            "device_loss_incident_records.json. Exists so the negative control can plant "
            "a bad exclusion without editing the shipped one — a diagnostic must not "
            "mutate its subject. The path in use is printed whenever it is not the "
            "default, so an override can never be silent."
        ),
    )
    ap.add_argument(
        "--no-marker-cross-check",
        action="store_true",
        help="do not report lines the shared marker list misses (it is not my file)",
    )
    args = ap.parse_args(argv)

    paths = list(args.paths) + list(args.run_log)
    if args.lane:
        paths += [REPO_ROOT / p for p in DEFAULT_LANE_PATHS]
    if not paths:
        return report_instrument_error(
            "no_paths_given",
            "No evidence path was named. This check reports nothing rather than "
            "reporting a clean run: a check with no input has observed nothing.",
        )

    files, skipped = gather(paths)
    explicit = {p.resolve() for p in paths if p.is_file()}
    records_path = args.incident_records or INCIDENT_RECORDS
    records, record_rot, record_error = load_incident_records(records_path)
    if record_error:
        return report_instrument_error("incident_record_file_unreadable", record_error)
    excluded = [f for f in files if f.resolve() in records and f.resolve() not in explicit]
    files = [f for f in files if f not in excluded]
    if not files:
        if args.lane_marker and not Path(args.lane_marker).exists():
            print(f"{TAG}: ERROR(instrument=lane_did_not_reach_evidence)", flush=True)
            print(
                f"No evidence files under {[str(p) for p in paths]}, and the lane marker "
                f"{args.lane_marker} was never written — the lane failed before it "
                "produced anything to read. That failure is already red for its own "
                "reason and this check declines to add a second red on top of it.",
                flush=True,
            )
            print(
                "::warning title=Device-loss check had no subject::"
                "The lane produced no artifacts. Fix the earlier failure and re-read.",
                flush=True,
            )
            return EXIT_PASS
        return report_instrument_error(
            "nothing_scanned",
            f"No readable evidence under {[str(p) for p in paths]}.\n"
            + ("Declined: " + "; ".join(skipped) if skipped else ""),
        )

    findings: dict[str, list[str]] = {}
    samples: list[tuple[Path, str]] = []
    structural_decidable = 0
    declared_rejections = 0
    unreadable: list[str] = []
    embedded_scanned = 0
    for path in files:
        named = path.resolve() in explicit
        scan = scan_artifact(path, named)
        if scan.error:
            unreadable.append(scan.error)
            continue
        structural_decidable += 1 if scan.decidable else 0
        declared_rejections += scan.declared_rejections
        embedded_scanned += 1 if scan.embedded else 0
        if named:
            samples.append((path, scan.searchable))
        for condition, lines in scan.hits.items():
            seen = findings.setdefault(condition, [])
            for line in lines:
                entry = f"{path}: {line}"
                if entry not in seen:
                    seen.append(entry)
    findings = {k: v for k, v in findings.items() if v}

    if not args.no_marker_cross_check:
        missed = marker_cross_check(samples)
        if missed:
            findings["marker_list_misses_real_line"] = missed

    named_count = len([f for f in files if f.resolve() in explicit])
    print(f"\n{TAG}: frame — {len(files)} file(s) read.")
    print(
        f"  {named_count} named as one run's own evidence (full condition set); "
        f"{len(files) - named_count} reached by directory expansion (tree-wide tier only: "
        + ", ".join(TIER_TREE_WIDE)
        + ")."
    )
    if named_count == 0:
        print(
            "  UNOBSERVABLE in this run: "
            + ", ".join(TIER_NAMED_RUN_ONLY)
            + " — not zero findings. Controls on this project produce those texts "
            "deliberately, so they are only decidable on a file the caller declares is "
            "one run's evidence, and no file was so declared."
        )
    if declared_rejections:
        print(
            f"  {declared_rejections} truncated point(s) found under a rejected_* key and "
            "NOT counted as findings: the producer already reported those. This check is "
            "looking for a short run that was kept."
        )
    if excluded:
        print(
            f"  {len(excluded)} file(s) excluded by name as historical incident records "
            f"({records_path.name}), each with a reason, an owner, a date and the "
            "finding(s) it accounts for:"
        )
        for path in excluded:
            entry = records[path.resolve()]
            witness = ", ".join(
                f"{condition}×{count}" for condition, count in sorted(entry["witness"].items())
            )
            print(
                f"    {entry['file']} [accounts for {witness}] — {entry['reason']} "
                f"({entry['owner']}, {entry['date']})"
            )
        print(
            "  These are artifacts OF a device loss, kept deliberately. They are excluded "
            "so a past incident cannot make this check permanently red and therefore "
            "unread; they are named here so the exclusion is never silent."
        )
        audit_findings, audit_notes = audit_exclusions(excluded, records)
        for condition, lines in audit_findings.items():
            findings.setdefault(condition, []).extend(lines)
        print(
            f"  exclusion audit: all {len(excluded)} excluded file(s) re-read through the "
            "same reader. An exclusion that covers NO finding is red "
            "(incident_record_covers_nothing) because it silences nothing today and can "
            "only blind this check tomorrow; an exclusion that covers MORE than it "
            "accounts for is red (incident_record_widened) because that is a new incident "
            "inside a file that was already forgiven."
        )
        for note in audit_notes:
            print(
                f"    narrowed, not a finding — {note}. Fewer findings than accounted for "
                "cannot hide an incident; the count is stale, not permissive."
            )
    if records_path != INCIDENT_RECORDS:
        print(
            f"  exclusion list read from {records_path}, NOT the shipped "
            f"{INCIDENT_RECORDS}. An override is never silent."
        )
    if record_rot:
        findings.setdefault("incident_record_rot", []).extend(record_rot)
    print(
        f"  structural rule decidable on {structural_decidable} artifact(s): the rest "
        "declare no expected count, so truncation is UNOBSERVABLE there rather than "
        "absent (R12)."
    )
    print(
        f"  {embedded_scanned} JSON artifact(s) carried a captured log inside a string "
        "value and were scanned decoded as well as raw. A wide-encoded (UTF-16LE) capture "
        "stored in JSON has its NULs escaped as `\\u0000`, so it is INVISIBLE in the raw "
        "file text — measured 2026-08-04, see json_embedded_text."
    )
    if skipped:
        print(f"  declined {len(skipped)}: " + "; ".join(skipped[:8]))
    if unreadable:
        print(f"  unreadable {len(unreadable)}: " + "; ".join(unreadable[:8]))
    print(
        "  no exit status was read. The defect this check exists for exits 0, so an exit "
        "status cannot be one of its filters."
    )

    if findings:
        order = (
            "device_lost_reported",
            "observation_ended_early",
            "broken_commitment_reported",
            "runtime_fallback_announced",
            "marker_list_misses_real_line",
            "incident_record_rot",
            "incident_record_covers_nothing",
            "incident_record_widened",
        )
        primary = next(c for c in order if c in findings)
        blocks = []
        for condition in order:
            if condition not in findings:
                continue
            head = {
                "device_lost_reported": (
                    "The device was LOST during a run whose artifacts are in this tree. "
                    "Everything measured after this point is CPU output, and the process "
                    "very likely exited 0."
                ),
                "observation_ended_early": (
                    "A run declared what it would observe and observed less. A figure "
                    "differenced from it is not a smaller quantity; it is a shorter run."
                ),
                "broken_commitment_reported": (
                    "The EP claimed a subgraph and its Compute() then failed. ORT "
                    "re-executed those nodes on CPU while get_providers() still lists "
                    "this EP."
                ),
                "runtime_fallback_announced": (
                    "ORT abandoned this EP at run time and re-executed on CPU without "
                    "raising."
                ),
                "marker_list_misses_real_line": (
                    "tests/ops/_verdict.py::FATAL_LOG_MARKERS does NOT match these real "
                    "ORT lines, so ci/check_fatal_log.py reads this log as clean. The "
                    "list is the single shared vocabulary and belongs to Trinity; it is "
                    "reported here rather than patched here, because a second private "
                    "marker list in ci/ is the two-dialect failure the vocabulary exists "
                    "to prevent. One-line fix: a marker that tolerates the list repr, "
                    "e.g. matching 'Falling back to' and 'CPUExecutionProvider' on the "
                    "same line."
                ),
                "incident_record_rot": (
                    "An entry in ci/device_loss_incident_records.json names a file that "
                    "is no longer present. The exclusion is still in force and nobody "
                    "can review what it excludes. Delete the entry or restore the file."
                ),
                "incident_record_covers_nothing": (
                    "An exclusion is in force over a file that produces no finding. It "
                    "silences nothing today, so nobody will ever have a reason to look at "
                    "it again — and it will silence whatever that file says next. Delete "
                    "the entry."
                ),
                "incident_record_widened": (
                    "A file excluded as the record of ONE past incident now carries more "
                    "than that incident. The exclusion was reviewed against a smaller "
                    "witness, so this evidence has never been read by anyone. Read it, "
                    "then either raise it as the new incident it is or re-account for it "
                    "with the reason it is the same one."
                ),
            }[condition]
            body = "\n".join(f"  {ln}" for ln in findings[condition])
            blocks.append(f"[{condition}]\n{head}\n{body}")
        detail = "\n\n".join(blocks)
        if len(findings) > 1:
            detail += "\n\nConditions in this run: " + ", ".join(
                c for c in order if c in findings
            )
        return report_fail(primary, detail)

    observed = list(ARTIFACT_TREE_WIDE)
    if named_count:
        observed += list(TIER_NAMED_RUN_ONLY)
    if excluded:
        observed += list(RECORD_CONDITIONS)
    return report_pass(
        f"{len(files)} artifact(s) read; nothing found for: " + ", ".join(observed) + ".\n"
        "What this does NOT claim: that the EP executed anything (that is the verdict's "
        "job); that a truncated run which declared no expected count would have been "
        "caught; or anything at all about the conditions listed UNOBSERVABLE in the frame "
        "line above — those were not looked for in this run."
    )


def main_guarded(argv=None) -> int:
    try:
        return main(argv)
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)
    except Exception as exc:  # noqa: BLE001
        return report_instrument_error(
            "check_raised", f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    sys.exit(main_guarded())
