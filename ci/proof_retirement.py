#!/usr/bin/env python3
"""The one proof-retirement register, and the one rule for reading it.

WHY THIS FILE EXISTS
====================
A proof may legitimately leave `evidence/proof_ledger.jsonl`: a form is removed from the EP, an
op is withdrawn, a duplicate key is collapsed. That is a written act, in
`evidence/retired_proof_keys.json`, and it is the **exemption** that every loss invariant in this
repository has to honour — `ci/check_ledger_census.py` (did a key that was once committed leave?)
and `rust/tools/gen_proof_ledger.py` (is every key ever recorded MATCH still in the ledger?).

Until 2026-08-04 those two tools named two different *files* for that exemption. §8.9.23 retired
43 keys into the list the producer reads; the census read an object-shaped register that never
existed, saw zero retirements, and reported all 43 as VANISHED — exit 1, on every run for days.
Neither reader was wrong on its own terms. **Two registers for one fact was the defect**, and the
screen written to catch a silent deletion spent that week failing on a deliberate one.

Pointing both tools at one *path* fixed the number and left the shape of the defect intact: two
independent parsers, one requiring `owner`/`date`/`reason` and one requiring only `reason`, so a
retirement the producer honoured was a retirement the census refused. **One register means one
path and one rule.** A second implementation of "what counts as a withdrawal" is a second register
wearing the first one's filename, and its failure mode is the quiet one: the exemption is granted
by whichever tool asks first.

So the path, the superseded path, the required fields, and the parse all live here, exactly once,
and both readers import them. `ci/negative_control_ledger_census.py` asserts that they do — with a
register that is well-formed for one reader and malformed for the other, which is the arm that
would have caught the divergence this module removes.

THE SCHEMA
==========
    {"retired": [{"key": ..., "owner": ..., "date": ..., "reason": ...}, ...]}

A list, because that is the shape the producer already read and a second shape for one fact is how
the two readers came to disagree about the same 43 keys. `owner` and `date` are REQUIRED alongside
`reason` — the same three fields `ci/check_open_reds.py` requires to retire a subject, for the same
argument: *a withdrawal nobody has to sign is a blanket exemption*. A duplicate key is an error and
not a last-one-wins, because two retirements of one proof means two people withdrew it for two
reasons and only one of them survives being read.

Standard library only. No flag suppresses a failure, and there is deliberately no "lenient" mode:
a caller that wanted one would be the second rule this module exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The one register. Repo-relative so a caller can resolve it against a worktree OR a git revision.
RETIRED_REL = "evidence/retired_proof_keys.json"

#: The filename `check_ledger_census.py` used to read. It never existed; if it ever does, two
#: registers exist again and the 43-false-positive defect is back. Named here, once, so the guard
#: in the census cannot drift from the module that defines the register it guards.
SUPERSEDED_RETIRED_REL = "evidence/proof_retired.json"

#: What a withdrawal costs. Identical to `check_open_reds.RETIRED_FIELDS`, deliberately.
RETIRED_FIELDS = ("owner", "date", "reason")


class RetirementError(ValueError):
    """The register exists and cannot be read as a register.

    Never "nothing is retired": a reader that cannot see its subject must say so rather than
    answer about something else (§8.9.21). Silently degrading a malformed register to `{}` turns
    every deliberate withdrawal in it into a VANISHED proof, which is the exact false positive
    this module was extracted to make unrepeatable.
    """


def register_path(repo: Path) -> Path:
    """The register inside `repo`. Absent is a legal state; unreadable is not."""
    return Path(repo) / RETIRED_REL


def superseded_path(repo: Path) -> Path:
    """The register that must not exist. See `SUPERSEDED_RETIRED_REL`."""
    return Path(repo) / SUPERSEDED_RETIRED_REL


def parse(where: str, doc: object) -> dict[str, dict]:
    """`{"retired": [...]}` -> `{key: record}`. `where` names the file or `rev:path` for errors.

    Keyed by proof key on return so callers can ask `key in retired`, while the FILE stays a list.
    """
    rows = doc.get("retired", []) if isinstance(doc, dict) else None
    if rows is None:
        raise RetirementError(f"{where}: expected an object with a `retired` list")
    if isinstance(rows, dict):
        raise RetirementError(
            f"{where}: `retired` is an object keyed by proof key. That was the shape of the "
            "register `ci/check_ledger_census.py` read before 2026-08-04, and it is not the shape "
            "`rust/tools/gen_proof_ledger.py` reads. One register, one shape: a list of "
            "{key, owner, date, reason}."
        )
    if not isinstance(rows, list):
        raise RetirementError(
            f"{where}: `retired` must be a list of {{key, owner, date, reason}}"
        )
    retired: dict[str, dict] = {}
    for rec in rows:
        if not isinstance(rec, dict) or not rec.get("key"):
            raise RetirementError(f"{where}: every retirement needs a `key`; got {rec!r}")
        key = rec["key"]
        if key in retired:
            raise RetirementError(
                f"{where}: proof {key!r} is retired twice. Two withdrawal records for one "
                "proof means only one of the two reasons survives being read, and nothing "
                "says which."
            )
        missing = [f for f in RETIRED_FIELDS if not rec.get(f)]
        if missing:
            raise RetirementError(
                f"{where}: retired proof {key!r} is missing {missing}. Withdrawing a proof "
                "needs a name and a reason for the same argument that accepting a red does."
            )
        retired[key] = rec
    return retired


def load(path: Path) -> dict[str, dict]:
    """Read the register at `path`. A file that is not there retires nothing; `{}`.

    An absent register is the honest empty answer — at a revision before anything was withdrawn
    the file genuinely did not exist. An unreadable one raises: see `RetirementError`.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RetirementError(f"cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise RetirementError(f"{path} is not JSON: {exc}") from exc
    return parse(str(path), doc)


def load_text(where: str, text: str) -> dict[str, dict]:
    """Same rule, applied to a register body already in hand — e.g. `git show REV:path`."""
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise RetirementError(f"{where} is not JSON: {exc}") from exc
    return parse(where, doc)


def reasons(path: Path) -> dict[str, str]:
    """`{key: reason}` for callers that only need the exemption and its justification.

    A convenience over `load`, NOT a second rule: it validates through the same `parse`, so a
    register the census refuses is a register the producer refuses, in the same words.
    """
    return {key: rec["reason"] for key, rec in load(path).items()}
