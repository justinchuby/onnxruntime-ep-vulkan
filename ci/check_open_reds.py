#!/usr/bin/env python3
"""Run the repository's own guards and rule on their colour against a declared register.

WHY THIS EXISTS
===============
Three checks were handed to me that reproduce in `main`'s own checkout:

    rust/tools/audit_instruments.py --check   -> FAIL(drift)
    ci/test_lane_checks.py                    -> 3 red
    tests/union_check.py --run                -> 5 red

Every one of them was already printing its failure, in full, with the drift quoted. None
of them was wired to anything that reads a colour. They were not silently passing; they
were **loudly failing into a void**, which is the same defect as a silent pass wearing a
louder coat: a check whose failure reaches no one has stopped being a check.

The obvious repair — turn them green — is the wrong one, and this file exists because it
is the wrong one. `audit_instruments --check` is red because nine `pub fn` accessors
arrived with the two-digest schema and nothing calls them. That red is TRUE. Baselining
it would, in `audit_instruments.py`'s own words, "turn an open item into a green tick,
which is the one thing a census must never do".

So the thing that needs building is not a fix for three reds. It is a way to tell a red
that somebody owns from a red that nobody has seen.

THE DEFECT THIS SCREEN REMOVES
==============================
For four sessions I described `ci/test_lane_checks.py` as "132 passed, 3 failed (the
known census reds)". Today, after merging `main`, it was **4** failed. The fourth was a
real regression in my own work: the merge reintroduced four `if: env.BUILD_SKIPPED != '1'`
guards whose writer I had deleted, and my own build-precondition screen caught it. I
found that by reading the node ids. Had I read the count — which is what "the known reds"
is, a count — I would have shipped it.

**An accepted red and a new red are indistinguishable when the only record of the
acceptance is a number in someone's head.** That is this screen's condition. It is the
`BUILD_SKIPPED` shape one level up: there, one missing file turned thirty steps green;
here, one remembered number turns an unbounded number of new failures invisible.

WHAT IT DOES
============
Every check in the register declares the colour it is EXPECTED to be, and both colours
are falsifiable:

  expect=green, observed green  -> PASS
  expect=green, observed red    -> FAIL(condition=unaccounted_red)   <- a new failure
  expect=red,   observed red    -> ACCOUNTED, and ANNOTATED to the merge UI
  expect=red,   observed green  -> FAIL(condition=stale_acceptance)  <- good news, close it
  expect=red,   observed red,
                signature gone  -> FAIL(condition=signature_changed) <- a DIFFERENT red

The last two rows are the load-bearing ones and they are why this is not an allowlist.

`stale_acceptance` is the arm that stops the register rotting. An allowlist that only
ever suppresses grows monotonically and ends up suppressing things that were fixed years
ago; nobody removes an entry because nothing ever asks them to. Here, the day Mouse wires
those nine accessors, this screen goes RED until the entry is deleted. The register can
only shrink by being asked to.

`signature_changed` is the arm that stops an acceptance covering more than it was granted
for. `audit_instruments --check` is accepted for the exact string
`9 NEW uninvoked instrument(s)`. If a tenth arrives, the signature no longer matches and
the lane fails — the acceptance does not stretch. This is the same reason
`check_suite_productivity.py` has no `--relax`: a waiver that widens to fit whatever
turns up is not a waiver, it is a deletion.

And every entry carries `review_by`. Past that date the entry is `lease_expired` and the
lane fails. Acceptance here is a **lease, not a grant**. This is deliberately a time bomb.
A red that nobody has looked at in three months is not accepted, it is forgotten, and the
whole point of this file is that those two states must not look the same. If that lands
on someone at an inconvenient moment, that is the mechanism working: the inconvenience is
the only thing that makes anyone re-read the entry.

WHAT IT DOES NOT CLAIM
======================
* Not that the accepted reds are harmless. It claims only that each has a named owner, a
  written reason and a stated closing condition — and it prints all three every run.
* Not that the register is complete. It rules on what it was told to run. A guard that is
  in no lane and in no register is invisible to this screen exactly as it was before;
  `ci/lane_inventory.py` is the screen for THAT question and the two must not be merged,
  because two censuses over one tree is the failure `audit_instruments.py` names.
* Not that a green check is a correct check. `--union-required`'s lesson applies: a check
  that quietly drops half its input is green.

WHY IT SHELLS OUT INSTEAD OF IMPORTING
======================================
The registered checks are what a human runs at a prompt. Importing them and calling an
entry point would screen a different thing from the one that is failing — the argv path,
the `if __name__` path and the exit-code path would all go unexercised, and the exit code
is the entire subject here. So it runs the command line, verbatim, and reads the code.

DEPENDENCIES
============
Standard library only, for the reason `ci/check_lane_inventory.py` states: the
`lane-checks` job installs pytest, onnx and numpy and nothing else, and a screen skipped
because an import failed is a screen that does not exist.

USAGE
=====
    python ci/check_open_reds.py                       # run the register
    python ci/check_open_reds.py --list                # show it without running anything
    python ci/check_open_reds.py --only audit_instruments
    python ci/check_open_reds.py --summary $GITHUB_STEP_SUMMARY
    python ci/check_open_reds.py --register other.json

There is no flag that suppresses a failure. Adding one would recreate the void.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REGISTER = HERE / "open_reds.json"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

# Every field is required. `reason` without `closes_when` is a shrug with a JSON key
# around it: it says why the red is there and not what would make it go away, so nobody
# can ever discharge it. `owner` without `review_by` is a name with no obligation
# attached. The register refuses partial entries rather than accepting them and hoping.
REQUIRED_FIELDS = ("id", "cmd", "expect", "owner", "opened", "review_by", "reason", "closes_when")

# Retiring an entry needs a reason and a name, for the same argument as opening one.
RETIRED_FIELDS = ("owner", "date", "reason")

# `signature` is required only for expect=red: it is the thing the acceptance is FOR.
STATE_PASS = "PASS"
STATE_ACCOUNTED = "ACCOUNTED"
STATE_UNACCOUNTED = "FAIL(condition=unaccounted_red)"
STATE_STALE = "FAIL(condition=stale_acceptance)"
STATE_SIGNATURE = "FAIL(condition=signature_changed)"
STATE_WIDENED = "FAIL(condition=extent_widened)"
STATE_NARROWED = "FAIL(condition=extent_narrowed)"
STATE_EXPIRED = "FAIL(condition=lease_expired)"
STATE_ERROR = "ERROR(instrument"


@dataclass
class Outcome:
    ident: str
    state: str
    expect: str
    observed: str
    detail: str
    entry: dict = field(default_factory=dict)

    @property
    def is_fail(self) -> bool:
        return self.state.startswith("FAIL(")

    @property
    def is_error(self) -> bool:
        return self.state.startswith("ERROR(")


def _annotating() -> bool:
    return bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("OPEN_REDS_FORCE_ANNOTATE"))


def _annotate(level: str, title: str, message: str) -> None:
    """Emit a GitHub check annotation.

    Annotations are the reason this screen is worth building rather than just printing.
    A merger reading a pull request sees annotations; they cannot be truncated away by a
    long log the way my own merge-gate failure was last week, when I lost the failing
    test's name to a cut-off tail. The same argument as `ci/check_flake_witness.py`.
    """
    if not _annotating():
        return
    flat = message.replace("\r", "").replace("\n", "%0A").replace("::", ":%3A")
    print(f"::{level} title={title}::{flat}", flush=True)


def _today() -> _dt.date:
    """The comparison date, overridable so the expiry arm is testable in both polarities.

    A lease that can only be observed to expire by waiting is a lease nobody ever tests.
    `OPEN_REDS_TODAY` exists for the negative control and for nothing else; it cannot
    suppress a failure, only move the date, and moving the date forward makes MORE things
    fail, not fewer.
    """
    override = os.environ.get("OPEN_REDS_TODAY")
    if override:
        return _dt.date.fromisoformat(override)
    return _dt.date.today()


def load_register(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"no register at {path}; the comparison input is missing, so nothing was "
            "ruled on either way — that is UNOBSERVABLE, not PASS"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    checks = doc.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{path}: `checks` must be a non-empty list")
    _check_subjects(path, doc, checks)
    seen: set[str] = set()
    for entry in checks:
        missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
        if missing:
            raise ValueError(f"{path}: entry {entry.get('id', '<no id>')!r} is missing {missing}")
        if entry["expect"] not in ("green", "red"):
            raise ValueError(f"{path}: entry {entry['id']!r} has expect={entry['expect']!r}")
        if entry["expect"] == "red" and not entry.get("signature"):
            raise ValueError(
                f"{path}: entry {entry['id']!r} expects red but declares no `signature`. "
                "An acceptance with no signature covers every future red of that check, "
                "which is a deletion of the check wearing an acceptance's name."
            )
        if entry["expect"] == "red":
            defect = _extent_defect(path, entry)
            if defect:
                entry["_extent_defect"] = defect
            closure = _closure_defect(path, entry)
            if closure:
                entry["_closure_defect"] = closure
        if entry["id"] in seen:
            raise ValueError(f"{path}: duplicate id {entry['id']!r}")
        seen.add(entry["id"])
        for key in ("opened", "review_by"):
            try:
                _dt.date.fromisoformat(entry[key])
            except ValueError as exc:
                raise ValueError(f"{path}: entry {entry['id']!r} has bad {key}: {exc}") from exc
    return checks


def _extent_defect(path: Path, entry: dict) -> str:
    """Read an expect=red entry's `extent` and say what is wrong with it, if anything.

    WHY THIS RETURNS A STRING INSTEAD OF RAISING, AND WHY THAT IS NOT A WEAKENING.

    `extent` became required in 69ac222 and the requirement was enforced by raising out of
    `load_register`, which happens BEFORE any check runs. Three of the four accepted reds in
    ci/open_reds_device.json predate the requirement, so from that commit onward the whole
    device register was UNLOADABLE: `--list`, `--only`, `--summary` and a plain run all exited
    2 having measured nothing. Ten entries, one of them a lease that could have expired
    unnoticed, and the tool's answer about every one of them was a usage error about a
    fourth. That is this repository's own `target_ran_nothing` shape appearing inside the
    screen written to prevent it -- a guard that refuses before it measures reports nothing
    about its subjects, and its silence reads exactly like their being fine.

    So a malformed `extent` is now a DEFECT OF THAT ENTRY, carried on the entry and turned
    into `ERROR(instrument=extent_undeclared)` when it is run. What does not change: the
    entry is not accepted, the run is not green, and the exit code is non-zero -- an entry
    with no declared extent gets NO acceptance, which is stricter than the raise was for
    that entry, because the raise let the file's author believe the problem was the file.
    What changes is that the other nine entries are ruled on. Two failures that are both
    loud are better than one failure that hides nine silences.

    A structural error in the file itself -- unreadable JSON, a missing `checks` list, a
    duplicate id, a bad date -- still raises, because those are properties of the register
    and not of one subject.
    """
    ident = entry.get("id", "<no id>")
    ext = entry.get("extent")
    if not isinstance(ext, dict):
        return (
            f"{path}: entry {ident!r} expects red but declares no `extent`. "
            "`signature` is a SUBSTRING test and a substring cannot notice a SECOND "
            "subject joining an accepted red (Switch, 2026-08-03). `extent` names the "
            "members the acceptance is granted over: "
            '{"pattern": "<regex with one capture group>", "members": [...]}. '
            "Until it does, this entry is UNCOLOURED: its red is neither accepted nor "
            "unaccounted, because nobody has said what the acceptance would be for."
        )
    if not ext.get("pattern"):
        return f"{path}: entry {ident!r} extent is missing 'pattern'"
    try:
        if re.compile(ext["pattern"]).groups != 1:
            return (
                f"{path}: entry {ident!r} extent.pattern must have EXACTLY one "
                "capture group — the member name. Zero groups matches whole lines and "
                "a set of whole lines is a signature again; more than one is ambiguous."
            )
    except re.error as exc:
        return f"{path}: entry {ident!r} extent.pattern: {exc}"
    if "members" not in ext or not isinstance(ext["members"], list):
        # Presence and type, not truthiness: an EMPTY members list is legal and is
        # the STRICTEST possible declaration — it says the acceptance is granted
        # over nothing, so any member the pattern finds is extent_widened. Rejecting
        # it as "missing" would have made the safest declaration unspellable.
        return f"{path}: entry {ident!r} extent.members must be present and a list"
    if len(set(ext["members"])) != len(ext["members"]):
        return f"{path}: entry {ident!r} extent.members has duplicates"
    return ""


#: Screens that own a machine-readable closure rule, keyed by the script an entry's `cmd`
#: invokes. The rule lives WITH the verdict it describes (see the module docstring of
#: ci/check_main_is_green.py); this table only says which entries are subject to one, so
#: that a register full of screens which own no such rule is unaffected.
_CLOSURE_RULE_OWNERS = ("ci/check_main_is_green.py",)


def _closure_defect(path: Path, entry: dict) -> str:
    """Say why this expect=red entry's `closes_when` disagrees with the screen it accepts.

    WHY A REGISTER FIELD IS PARSED BY THE SCREEN IT DESCRIBES.

    `closes_when` is not a comment. This tool quotes it verbatim in STATE_STALE the moment
    an accepted red turns green, and in `--list`, `--summary` and the merge annotation
    before that — it is the discharge instruction, handed to whoever has to act. So it can
    be WRONG, and a wrong one is worse than a missing one: it carries the register's
    authority into the wrong action.

    It was wrong. `main_is_green` was registered while ci/check_main_is_green.py ruled on
    the newest run, and said one green push to `main` closes it. PR #91 made the whole
    applicable window authoritative. From then on the entry's expected COLOUR stayed right
    — the screen is red and the register says red — while its instruction said a single
    green push would discharge an acceptance that the screen would go on refusing to
    discharge (issue #103). Nothing could notice, because no lane compared the sentence to
    the semantics; only a person reading both could, and the whole point of this register
    is not to need one.

    So the screen that owns the verdict also owns an identifier for its closure rule and a
    parser for prose claiming to describe it, and this function runs that parser here, in
    the production lane, before the acceptance is granted. A defect is carried on the
    entry and becomes ERROR(instrument=closure_condition_undeclared) at run time — the
    same shape as `_extent_defect`, and for the same reason (see its docstring): one
    entry's defect must not make the register unloadable and silence every other entry.

    Scope, deliberately: expect=red entries only, and only those whose `cmd` invokes a
    screen that owns a closure rule. An expect=green entry's `closes_when` is not an
    instruction anyone is about to follow, and a screen that has declared no rule has
    nothing to disagree with — inventing a defect there would be this tool ruling on a
    subject nobody gave it.

    Fails closed. If the owning screen cannot be imported, that is not "no defect": it is
    this tool being unable to ask, which is the same error as reporting green because the
    network was down.
    """
    ident = entry.get("id", "<no id>")
    cmd = entry.get("cmd")
    if not isinstance(cmd, list):
        return ""
    parts = [str(a).replace("\\", "/") for a in cmd]
    owner = next((o for o in _CLOSURE_RULE_OWNERS if any(p.endswith(o) for p in parts)), None)
    if owner is None:
        return ""
    script = REPO / owner
    try:
        spec = importlib.util.spec_from_file_location(f"_closure_owner_{ident}", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"no import spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        defects = module.closure_prose_defects(entry.get("closes_when"))
    except Exception as exc:  # noqa: BLE001 - unable-to-ask is a defect, not a pass
        return (
            f"{path}: entry {ident!r} accepts a red from {owner}, whose closure rule this "
            f"tool could not read ({type(exc).__name__}: {exc}). The instruction in "
            "`closes_when` is therefore unchecked, and an unchecked instruction is not a "
            "verified one — reporting no defect here would be asserting agreement that "
            "nothing established."
        )
    if not defects:
        return ""
    joined = " ".join(f"({i}) {d}" for i, d in enumerate(defects, 1))
    return (
        f"{path}: entry {ident!r} declares a `closes_when` that {owner} does not agree "
        f"describes its closure rule: {joined} Fix the sentence, or — if the screen's rule "
        "really did change — change the rule identifier in the screen and let every "
        "register quoting the old one go red, which is the point of its being an "
        "identifier."
    )


def _check_subjects(path: Path, doc: dict, checks: list[dict]) -> None:
    """The denominator arm. A subject may not leave this register silently.

    FOUND THE HARD WAY, BY THIS SCREEN'S FIRST REAL USER, ON ITS SECOND DAY.

    Mouse repaired three of the five accepted reds and — correctly, by the protocol this
    file states — DELETED their entries. But `audit_instruments --check` was not fully
    green: seven of eight uninvoked accessors had been wired and an eighth,
    `unprovable_decline_forms`, had not. Deleting the entry did not move that check to
    `expect: green`; it removed the check from the register altogether. The screen went
    from ruling on 8 subjects to ruling on 5, and printed PASS — with a red check in the
    tree that it had stopped looking at.

    That is this repository's `BUILD_SKIPPED` shape reproduced INSIDE the tool written to
    prevent it, and it is the same defect as `check_suite_productivity.py`'s
    `target_ran_nothing`: a sum cannot see one of its terms go silent. The frame line
    printed "8 declared checks" and then "5 declared checks" and nothing compared them.

    So `subjects` is APPEND-ONLY and every name in it must be accounted for, in exactly
    one of two ways: it is in `checks` (still ruled on, either colour), or it is in
    `retired` with an owner, a date and a reason. Closing a red now means flipping
    `expect` to `green` — which keeps the check running and turns any future regression
    into `unaccounted_red` — and deleting an entry is a deliberate, written act.

    `retired` is not a suppression list: a retired subject is not run and not ruled on.
    It exists so that the DIFFERENCE between "we decided to stop watching this" and "this
    fell out of the register" is written down, because those two were indistinguishable
    and one of them had already happened.
    """
    subjects = doc.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError(
            f"{path}: `subjects` must list every id this register has ever ruled on. "
            "Without it, an entry can leave the register and take its check with it, and "
            "the verdict line cannot tell a shrinking denominator from a clean tree."
        )
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"{path}: `subjects` has duplicates")
    retired = doc.get("retired", {})
    if not isinstance(retired, dict):
        raise ValueError(f"{path}: `retired` must be an object keyed by id")
    live = [c.get("id") for c in checks]
    for ident, rec in retired.items():
        if ident in live:
            raise ValueError(f"{path}: {ident!r} is both live and retired")
        missing = [f for f in RETIRED_FIELDS if not (isinstance(rec, dict) and rec.get(f))]
        if missing:
            raise ValueError(
                f"{path}: retired subject {ident!r} is missing {missing}. Retiring a check "
                "needs a name and a reason for the same argument as accepting a red does."
            )
    accounted = set(live) | set(retired)
    dropped = [s for s in subjects if s not in accounted]
    if dropped:
        raise ValueError(
            f"{path}: subject(s) {dropped} are in `subjects` but in neither `checks` nor "
            "`retired`. A check does not leave this register by being deleted. If it is "
            "fixed, set expect=green and keep running it — that is what turns the next "
            "regression into an unaccounted_red instead of a silence. If it should stop "
            "being watched, retire it with an owner, a date and a reason."
        )
    undeclared = [i for i in list(live) + list(retired) if i not in subjects]
    if undeclared:
        raise ValueError(
            f"{path}: {undeclared} appear in the register but not in `subjects`. Add them; "
            "`subjects` is the append-only record of everything this screen has ever been "
            "responsible for."
        )


def run_entry(entry: dict, repo: Path) -> Outcome:
    ident = entry["id"]
    defect = entry.get("_extent_defect")
    if defect:
        # Refused BEFORE the command runs, and said so: an entry whose acceptance has no
        # declared extent cannot be granted one by any output, so running it would spend
        # the time and still rule on nothing. `observed` is "?" and not "red", because
        # this check's colour was not looked at.
        return Outcome(
            ident,
            f"{STATE_ERROR}=extent_undeclared)",
            entry["expect"],
            "?",
            defect,
            entry,
        )
    cmd = list(entry["cmd"])
    closure = entry.get("_closure_defect")
    if closure:
        # Also refused BEFORE the command runs. The colour this entry expects may well be
        # the colour the screen produces — that is precisely how this defect survived: the
        # register was RIGHT about red and WRONG about what would end it. Running the
        # command would confirm the colour and rule on nothing about the instruction, so
        # `observed` is "?" here too.
        return Outcome(
            ident,
            f"{STATE_ERROR}=closure_condition_undeclared)",
            entry["expect"],
            "?",
            closure,
            entry,
        )
    if cmd and cmd[0] == "python":
        cmd[0] = sys.executable
    cwd = repo / entry.get("cwd", ".")
    env = dict(os.environ)
    # Do not let this screen's own annotation switch leak into the child. A registered
    # check that annotates would put ITS reds in the merge UI unlabelled, next to the
    # accounted ones, and the reader could not tell which was which.
    env.pop("OPEN_REDS_FORCE_ANNOTATE", None)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=int(entry.get("timeout", 1800)),
        )
    except FileNotFoundError as exc:
        return Outcome(ident, f"{STATE_ERROR}=command_absent)", entry["expect"], "?", str(exc), entry)
    except subprocess.TimeoutExpired:
        return Outcome(
            ident,
            f"{STATE_ERROR}=timeout)",
            entry["expect"],
            "?",
            "the check did not finish; its colour was not observed",
            entry,
        )

    out = (proc.stdout or "") + (proc.stderr or "")
    observed = "green" if proc.returncode == 0 else "red"
    expect = entry["expect"]

    if _today() > _dt.date.fromisoformat(entry["review_by"]):
        return Outcome(
            ident,
            STATE_EXPIRED,
            expect,
            observed,
            f"the lease on this entry ran out on {entry['review_by']}. Re-read it, then "
            f"either close it ({entry['closes_when']}) or extend it deliberately.",
            entry,
        )

    if expect == "green":
        if observed == "green":
            return Outcome(ident, STATE_PASS, expect, observed, "", entry)
        return Outcome(
            ident,
            STATE_UNACCOUNTED,
            expect,
            observed,
            _tail(out),
            entry,
        )

    # expect == "red"
    if observed == "green":
        return Outcome(
            ident,
            STATE_STALE,
            expect,
            observed,
            f"this check is GREEN and the register says it is red. Good news: delete the "
            f"entry. It was opened {entry['opened']} by {entry['owner']} and closes when: "
            f"{entry['closes_when']}",
            entry,
        )
    signature = entry["signature"]
    if signature not in out:
        return Outcome(
            ident,
            STATE_SIGNATURE,
            expect,
            observed,
            f"this check is red, but not for the reason it was accepted for. The "
            f"acceptance names {signature!r}; that string is not in the output. The "
            f"acceptance does not stretch to cover a red it was not granted for.\n"
            + _tail(out),
            entry,
        )
    verdict = _extent_verdict(entry, out)
    if verdict is not None:
        state, detail = verdict
        return Outcome(ident, state, expect, observed, detail + "\n" + _tail(out), entry)
    return Outcome(ident, STATE_ACCOUNTED, expect, observed, signature, entry)


def _extent_verdict(entry: dict, out: str) -> tuple[str, str] | None:
    """The set of things the acceptance is FOR, not just a string that appears in it.

    FOUND BY SWITCH, DECLARED BY ME, AND LEFT OPEN FOR A SESSION.

    `signature` is a SUBSTRING test, and a substring cannot notice a second subject joining
    an accepted red. `lane_checks_census_extent` accepts the string `CENSUS-EXTENT` over
    three named tests; a fourth test failing the same way still contains that string, so the
    acceptance absorbs it in silence — which is the identical defect the register was built
    to fix, one level down. `kv_caller_bind_reading` accepts `assert 'CUDA' in 'OK'` over one
    node; a second node in the same file failing differently would not move the substring.

    `extent` closes it by naming the MEMBERS. The set must match exactly: growing is
    `extent_widened` (a red the acceptance was not granted for), shrinking is
    `extent_narrowed` — good news, and the same arm as `stale_acceptance`, because an
    acceptance that has stopped covering something must be re-read rather than left to
    cover it in principle.
    """
    ext = entry.get("extent")
    if not ext:
        return None
    found = set(re.findall(ext["pattern"], out, flags=re.MULTILINE))
    members = set(ext["members"])
    if found == members:
        return None
    extra = sorted(found - members)
    gone = sorted(members - found)
    lead = (
        f"the acceptance names {len(members)} member(s) and this run has {len(found)}. "
        f"`signature` matched — a substring cannot see this, which is why `extent` exists."
    )
    if extra and not gone:
        return STATE_WIDENED, (
            f"{lead}\n  JOINED the accepted red, uncovered by it: {extra}\n"
            "  Either the newcomer belongs to the same owned red — add it to `members` and "
            "say so in `reason` — or it is a different red and needs its own entry."
        )
    if gone and not extra:
        return STATE_NARROWED, (
            f"{lead}\n  NO LONGER failing: {gone}\n"
            "  Good news. Narrow `members` to what is still red; if the list empties, the "
            "entry is `stale_acceptance` and `expect` flips to green."
        )
    return STATE_WIDENED, (
        f"{lead}\n  JOINED: {extra}\n  NO LONGER failing: {gone}\n"
        "  The membership moved in both directions, so this is a different red wearing the "
        "same signature. Re-read it before extending the acceptance."
    )


def _tail(text: str, limit: int = 2000) -> str:
    text = text.rstrip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def _frame(checks: list[dict], register: Path, doc: dict | None = None) -> str:
    reds = [c for c in checks if c["expect"] == "red"]
    greens = [c for c in checks if c["expect"] == "green"]
    doc = doc or {}
    subjects = doc.get("subjects", [])
    retired = doc.get("retired", {})
    lines = [
        "OPEN-REDS: frame (R12 applied to this screen: what it did not look at, said out loud)",
        f"  register: {register}",
        f"  {len(subjects)} subject(s) ever declared = {len(checks)} ruled on now "
        f"({len(greens)} expected green, {len(reds)} accepted red) + {len(retired)} retired.",
    ]
    for ident, rec in sorted(retired.items()):
        lines.append(f"    RETIRED {ident} — {rec['date']}, {rec['owner']}: {rec['reason']}")
    lines += [
        "  A subject cannot leave this register by being deleted; the arithmetic above is",
        "  checked, because a shrinking denominator and a clean tree printed the same line",
        "  here once already.",
        "  NOT IN FRAME — any guard that is in neither this register nor a workflow lane. This",
        "  screen rules on colour, not on coverage; ci/lane_inventory.py is the coverage census",
        "  and the two are deliberately separate tools over one tree.",
    ]
    return "\n".join(lines)


def screen(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--register", default=str(REGISTER))
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--only", action="append", default=[], help="run one id (repeatable)")
    ap.add_argument("--list", action="store_true", help="print the register; run nothing")
    ap.add_argument("--summary", help="append a markdown table to this file")
    ap.add_argument(
        "--emit-cmd",
        metavar="ID",
        help="print the declared command line for ID and exit, shell-quoted, running "
             "nothing. This exists because the workflow used to spell one register "
             "entry's command out a second time by hand: `lane_checks_suite` deselects "
             "the three census-extent nodes that `lane_checks_census_extent` accepts as "
             "red, and the CI step ran the same suite WITHOUT the deselects — so a red "
             "the register had accepted, with an owner and a closes_when, still failed "
             "main, every push, for days. Two spellings of one command is the same defect "
             "as two registers for one fact.",
    )
    args = ap.parse_args(argv)

    register = Path(args.register).resolve()
    repo = Path(args.repo).resolve()
    try:
        checks = load_register(register)
        doc = json.loads(register.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        print(f"OPEN-REDS: ERROR(instrument=register_absent): {exc}", file=sys.stderr)
        return EXIT_ERROR_INSTRUMENT
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"OPEN-REDS: usage: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.emit_cmd:
        match = [c for c in checks if c["id"] == args.emit_cmd]
        if not match:
            print(
                f"OPEN-REDS: usage: no such id: {args.emit_cmd!r}. A workflow step asking "
                "the register for a command it does not declare must fail loudly; falling "
                "back to a hand-written command is how the two spellings diverged.",
                file=sys.stderr,
            )
            return EXIT_USAGE
        print(shlex.join(match[0]["cmd"]))
        return EXIT_PASS

    print(_frame(checks, register, doc))
    print()

    if args.list:
        for c in checks:
            print(f"  {c['expect']:<5} {c['id']:<44} owner={c['owner']} review_by={c['review_by']}")
            print(f"        why: {c['reason']}")
            print(f"        closes when: {c['closes_when']}")
        print("\nOPEN-REDS: listed only; no check was run, so no colour was observed.")
        return EXIT_PASS

    if args.only:
        wanted = set(args.only)
        unknown = wanted - {c["id"] for c in checks}
        if unknown:
            print(f"OPEN-REDS: usage: no such id(s): {sorted(unknown)}", file=sys.stderr)
            return EXIT_USAGE
        checks = [c for c in checks if c["id"] in wanted]

    outcomes = [run_entry(c, repo) for c in checks]

    width = max(len(o.ident) for o in outcomes)
    print("  state       check")
    for o in outcomes:
        tick = {
            STATE_PASS: "PASS      ",
            STATE_ACCOUNTED: "ACCOUNTED ",
        }.get(o.state, o.state)
        print(f"  {tick:<11} {o.ident:<{width}}  expect={o.expect} observed={o.observed}")

    accounted = [o for o in outcomes if o.state == STATE_ACCOUNTED]
    fails = [o for o in outcomes if o.is_fail]
    errors = [o for o in outcomes if o.is_error]

    if accounted:
        print("\nACCOUNTED REDS — each of these is failing right now, on purpose, with an owner:")
        for o in accounted:
            e = o.entry
            print(f"\n  {o.ident}")
            print(f"    owner        {e['owner']}")
            print(f"    opened       {e['opened']}   review by {e['review_by']}")
            print(f"    why          {e['reason']}")
            print(f"    closes when  {e['closes_when']}")
            print(f"    signature    {e['signature']!r}  (present in this run's output)")
            _annotate(
                "warning",
                f"open red: {o.ident}",
                f"owner {e['owner']} - accepted red, opened {e['opened']}, review by "
                f"{e['review_by']}. {e['reason']} Closes when: {e['closes_when']}",
            )

    for o in errors:
        print(f"\n{o.state}  {o.ident}\n  {o.detail}", file=sys.stderr)
        _annotate("error", f"open-reds instrument error: {o.ident}", o.detail)

    for o in fails:
        print(f"\n{o.state}  {o.ident}\n  {o.detail}", file=sys.stderr)
        _annotate("error", f"{o.state} {o.ident}", o.detail)

    if args.summary:
        _write_summary(Path(args.summary), outcomes)

    limit_fails = screen_known_limits(doc, repo)

    print()
    if errors:
        print(
            "OPEN-REDS: a check whose colour was not observed is not a check that passed.",
            file=sys.stderr,
        )
        return EXIT_ERROR_INSTRUMENT
    if fails or limit_fails:
        if fails:
            print(
                f"OPEN-REDS: FAIL — {len(fails)} check(s) are not the colour the register declares."
            )
        if limit_fails:
            print(
                f"OPEN-REDS: FAIL — {len(limit_fails)} declared known limit(s) are no longer "
                "admitted by the screen that owns them."
            )
        return EXIT_FAIL_CONDITION
    print(
        f"OPEN-REDS: PASS — {len(outcomes)} check(s) are the colour the register declares "
        f"({len(accounted)} accepted red, each named above with an owner and a closing "
        "condition). PASS here does NOT mean the tree is clean; it means every red in it "
        "is one somebody is holding."
    )
    return EXIT_PASS


def screen_known_limits(doc: dict, repo: Path) -> list[str]:
    """A limit a tool is KNOWN to have, held by the tool's own author, run every time.

    WHY THIS IS NOT AN ACCEPTED RED. An accepted red is a failing check somebody OTHER than
    the person who accepted it is expected to close — the shipped-register test enforces
    exactly that, `owner != link`, and it is right to. A known limit is different in kind:
    it is a bounded gap in a screen, owned by the person who wrote the screen, which is
    precisely the case that rule forbids. Filing one as an accepted red would have meant
    either weakening that rule or writing `owner` I did not mean. So it gets its own
    category rather than a category it does not fit.

    WHY IT IS CHECKED AND NOT MERELY LISTED. Two limitations of mine lived in prose this
    quarter — a substring cannot notice a second file joining an accepted red (Switch's
    finding) and the census is blind in a shallow clone (mine, declared and unguarded) —
    and both stayed open until somebody tripped over them. A list nobody executes decays
    the same way. So each entry names a command that must make the SCREEN ITSELF admit the
    limit by token, and exit 1 doing it. If a limit is quietly closed the admission stops
    and this arm goes red: a stale declaration is as much a defect as an undeclared gap,
    which is the same asymmetry evidence/proof_rewitness.json is built on.
    """
    limits = doc.get("known_limits") or []
    print("\nKNOWN LIMITS — bounded gaps in these screens, held by the screen's own author:")
    if not limits:
        print("  (none declared)")
        return []
    bad: list[str] = []
    for lim in limits:
        for field in ("id", "cmd", "admits", "owner", "opened", "review_by", "limit",
                      "closes_when"):
            if not lim.get(field):
                bad.append(f"{lim.get('id', '?')}: declaration is missing {field!r}")
        if bad and bad[-1].startswith(lim.get("id", "?")):
            print(f"  MALFORMED  {lim.get('id', '?')}")
            continue
        out = subprocess.run(
            lim["cmd"], cwd=str(repo), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=lim.get("timeout", 300),
        )
        blob = (out.stdout or "") + (out.stderr or "")
        token = f"KNOWN-LIMIT {lim['admits']}"
        if out.returncode == 1 and token in blob:
            print(f"  ADMITTED   {lim['id']}")
            print(f"             owner {lim['owner']}, opened {lim['opened']}, "
                  f"review by {lim['review_by']}")
            print(f"             screen {lim['screen']} still admits {lim['admits']!r}")
            print(f"             closes when: {lim['closes_when'][:160]}")
        else:
            print(f"  FAIL(condition=limit_declaration_is_stale)  {lim['id']}")
            bad.append(
                f"{lim['id']}: {lim['screen']} no longer admits {lim['admits']!r} "
                f"(rc={out.returncode}). Either the limit was closed and this declaration "
                "must be retired with owner, date and reason, or the admission was removed "
                "while the gap remains — which is the register's own failure mode."
            )
    for b in bad:
        print(f"    - {b}", file=sys.stderr)
    return bad


def _write_summary(path: Path, outcomes: list[Outcome]) -> None:
    rows = ["", "### Open reds", "", "| state | check | owner | review by | closes when |", "|---|---|---|---|---|"]
    for o in outcomes:
        e = o.entry
        rows.append(
            f"| `{o.state}` | `{o.ident}` | {e.get('owner', '')} | {e.get('review_by', '')} | "
            f"{e.get('closes_when', '').replace('|', '/')} |"
        )
    rows.append("")
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
    except OSError as exc:  # pragma: no cover - a summary is a courtesy, not the verdict
        print(f"OPEN-REDS: could not write summary {path}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(screen())
