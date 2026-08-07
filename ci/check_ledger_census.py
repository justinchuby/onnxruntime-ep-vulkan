#!/usr/bin/env python3
"""Census the proof ledger against its own history: has an entry ever gone MISSING?

WHY THIS EXISTS
===============
`gen_proof_ledger.py --check` asks, of every entry in the file, whether it agrees with the
build. It is a good question and it is answered well. It is also the wrong question for one
specific failure, because it can only be asked **about entries that are still there**.

On 2026-08-03 Tank found that `26fd93f` proved and committed three `Cast` forms —
`f32>i32`, `i32>f32`, `f32>bool` — and that they were absent from `main`. `git log -S` on
the ledger found the addition and **no removal**: the deletion happened inside a conflict
resolution in the merge `eb84364`, and history simplification then hid the original commit
from the file's own log. So:

* `--check` was green, because every entry that remained agreed with the build.
* the shrinking-write guard did not fire, because that guard covers writes by **the tool**,
  and a merge is not one.
* `check_open_reds.py` did not fire, because the ledger is not a check.

The only instrument that noticed was the op suite, which went red at that merge and stayed
red, **unaccounted** — and an unaccounted red is how a deletion survives. That is this
repository's own lesson from `check_open_reds.py`, one artifact over: a sum cannot see one
of its terms go silent, and a *proof* is a term.

THE ARITHMETIC
==============
The same shape `check_open_reds.py` applies to subjects, applied to proofs:

    N ever proven = M present now + K retired + D VANISHED

`D > 0` is the failure. Everything else is disclosure.

WHERE `N` COMES FROM, AND WHY IT IS NOT A FILE
==============================================
`check_open_reds.py` keeps its denominator in an append-only `subjects` list inside the
register it screens. That works there because a human edits that register deliberately.

It would **not** work here, and the reason is the whole point of this screen: the event
being detected is a merge silently rewriting the ledger. A denominator that lives in a
tracked file next to the ledger is rewritten by exactly the same merge, in the same
conflict resolution, by the same hand. It would have been deleted alongside the three Cast
entries and the census would have balanced.

So `N` is derived from **git history**: the union of every `key` that has ever appeared in
any revision of the ledger reachable from HEAD (see DEFAULT_SCOPE). History is append-only in
the sense that matters here — a merge commit adds to it and cannot subtract from it. The
denominator is therefore held somewhere the failure mode cannot reach.

`--simplify` is deliberately NOT used when listing revisions (`git rev-list --full-history HEAD`), because
history simplification is precisely what hid `26fd93f` from the file's own log.

RETIREMENT
==========
A proof may legitimately go away: a form is removed from the EP, an op is withdrawn, a
duplicate key is collapsed. That is a written act, in `evidence/retired_proof_keys.json`, with
an owner, a date and a reason — the same three fields `check_open_reds.py` requires to retire a
subject, for the same argument. Retiring is not suppression: a retired key is *named* in the
output every run.

ONE REGISTER, AND WHY IT IS THIS FILENAME
-----------------------------------------
Until 2026-08-04 this screen read `evidence/proof_retired.json` — an object keyed by proof key
— while `rust/tools/gen_proof_ledger.py`, the tool that PRODUCES the ledger, read
`evidence/retired_proof_keys.json`, a list of `{key, reason}`. Both readers were internally
correct. Only one of the two files was ever written. §8.9.23 retired 43 keys into the list, the
producer honoured them, and this screen — which cannot see a file that does not exist — read
zero retirements and reported all 43 as VANISHED, exit 1, on every run since.

Two registers for one fact is the defect, not the filename. The list wins for one reason that
is not taste: it is the register that EXISTS and that a live reader consumes. Renaming a file
a producer reads has a silent failure mode — a reader left pointing at the old name reads an
absent file as "nothing is retired", which is exactly the 43-false-positive shape, inverted and
harder to see. Pointing the consumer at the producer's register has no such mode: if it is
wrong, it is wrong loudly and immediately.

The schema is the list's, extended: `owner` and `date` are now REQUIRED alongside `reason`,
because "who withdrew this and when" is the part a bare reason cannot answer. `proof_retired.json`
is not merely unread now — its existence is an ERROR(instrument), so the ambiguity cannot come
back by someone re-creating the file this screen used to want.

ONE PATH IS NOT YET ONE REGISTER
--------------------------------
Pointing both tools at one filename left the *shape* of the defect standing: two independent
parsers, this one requiring `owner`/`date`/`reason` and the producer's requiring only `reason`.
A retirement missing an owner was therefore honoured by `gen_proof_ledger.py` and refused here —
one file, two rules, and the exemption granted by whichever tool asked first. So the path, the
required fields and the parse now live in `ci/proof_retirement.py`, which both readers import;
this file re-exports the names it used to define and adds no rule of its own. A malformed
register is `ERROR(instrument=unreadable_retirement_register)`, never "nothing is retired".

A RETIREMENT IS A POSITIVE STATE, AND IT MUST HAVE AN ARM
---------------------------------------------------------
`negative_control_ledger_census.py` had 13 arms and every one of them planted or replayed a
DELETION. The exemption branch — key gone, retirement recorded, screen stays green and names it
— had never been executed by a green run. A branch with no positive state, in the tool built to
have positive states. `retirement_acquits_a_deletion` and `retirement_without_owner_is_refused`
are that arm and its polarity.

WHAT THIS DOES NOT CLAIM
========================
* Not that the surviving entries are correct — `gen_proof_ledger.py --check` is that screen
  and the two must stay separate. This one never opens the built EP.
* Not that every key ever written was *deliberately* written; a key added by mistake and
  removed on purpose is a retirement, and must be recorded as one.
* Not that the ledger is complete. "Which forms ought to be proven" is the census question
  `probe_model_op_census.py` asks. This asks only whether something that WAS proven has
  quietly stopped being proven.

USAGE
=====
    python ci/check_ledger_census.py                    # screen HEAD's ledger against history
    python ci/check_ledger_census.py --at <rev>         # replay: screen an older revision
    python ci/check_ledger_census.py --json <path>      # machine-readable record
    python ci/check_ledger_census.py --list-retired

Standard library only, and `git`. No flag suppresses a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The register, its schema and its parse live in ONE module that the ledger's PRODUCER imports
# too (`rust/tools/gen_proof_ledger.py`). Re-implementing the parse here is how this screen and
# the producer came to disagree about the same 43 keys with one filename between them: one
# required an owner and a date, the other did not, so a withdrawal one honoured the other refused.
import proof_retirement  # noqa: E402
from proof_retirement import (  # noqa: E402
    RETIRED_FIELDS,
    RETIRED_REL,
    SUPERSEDED_RETIRED_REL,
    RetirementError,
)

LEDGER_REL = "evidence/proof_ledger.jsonl"

# WAS `--all`, AND `--all` WAS WRONG — found on 2026-08-04 by running this screen twice.
#
# The first run of the session was green. The second, minutes after a teammate pushed an
# in-progress branch, reported 28 VANISHED proofs "first proven in 18ddece" — a commit on
# `squad/mouse` that is not an ancestor of anything I have. The screen was convicting my
# branch for not containing somebody else's unmerged work, and the sentence it printed
# ("committed to this ledger and no longer in it") was false: they were never in this line
# of history at all.
#
# `HEAD` is the right denominator and loses nothing the screen exists for. The failure it
# was built to catch is a proof dropped inside a merge conflict resolution, and BOTH merge
# parents are reachable from HEAD, so `--full-history` still sees the side the deletion
# came from. What `--all` added was only refs that were never merged — which is not history,
# it is other people's drafts.
#
# This is the third time in one session that a framing choice, not a value test, was the
# defect: a symmetric value comparison convicted my own repair; an unresolvable boundary
# made every comparison vacuous; and now too WIDE a scope convicted a branch for a proof it
# never had. `--full-history` stays: that one is load-bearing and is separately asserted.
DEFAULT_SCOPE = "HEAD"
RETIRED = REPO / RETIRED_REL
FRAME_WITNESSES = ("source_digest", "toolchain")
REWITNESS = REPO / "evidence" / "proof_rewitness.json"
REWITNESS_FIELDS = ("revision", "field", "owner", "date", "reason")
# A v2 record names no `revision`: see `load_rewitness` for why naming one cannot work.
REWITNESS_V2_FIELDS = ("schema", "field", "owner", "date", "reason", "caused_by", "transitions")
# A v3 record names no commit at ALL: see `load_rewitness` for why no commit can work when the
# cause is in the same change as the move.
REWITNESS_V3_FIELDS = (
    "schema", "field", "owner", "date", "reason", "caused_by_content", "transitions"
)
SCHEMA_V1 = "rewitness/1"
SCHEMA_V2 = "rewitness/2"
SCHEMA_V3 = "rewitness/3"
KNOWN_SCHEMAS = (SCHEMA_V1, SCHEMA_V2, SCHEMA_V3)
CONTENT_SCHEMAS = (SCHEMA_V2, SCHEMA_V3)
WORKTREE = "WORKTREE"

# ── where a `source_digest` COMES FROM, as paths in a tree ──────────────────────────────
# These mirror `rust/build.rs` exactly (`glsl_dir`, `include_dir`, `variant_table`) and they are
# the only inputs `rust/build_support/shader_source_digest.rs` hashes that live in the repository.
# A v3 cause is checked against them, so if `build.rs` moves a directory this screen must move
# with it — which is why they are named here once and asserted by
# `tests/ops/test_rewitness_cause_paths.py` rather than spelled inline.
GLSL_DIR_REL = "rust/shaders/glsl"
GLSL_INCLUDE_DIR_REL = "rust/shaders/include"
SHADER_VARIANTS_REL = "rust/src/ops/shader_variants.txt"
#: `kind` values a v3 cause may declare. One, for now, and an unknown one is refused.
CAUSE_KIND_SAME_CHANGE = "same_change"
KNOWN_CAUSE_KINDS = (CAUSE_KIND_SAME_CHANGE,)
#: Roots a cause path may never name. Every one of them is either this screen's own evidence or a
#: generated reading of it, and a cause that points at generated evidence proves itself.
GENERATED_CAUSE_ROOTS = (
    "evidence/", "bench/", "docs/", ".squad/", "ci/", ".github/", "target/", "tests/",
)


def _git(args: list[str], repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, encoding="utf-8"
    )


def _git_bytes(args: list[str], repo: Path) -> subprocess.CompletedProcess:
    """`_git`, without the text decoding — for content that is hashed rather than read.

    `text=True` is universal-newlines: it turns CRLF into LF on the way in. That is harmless
    for a `rev-list` but fatal for a content digest, because it would silently make a file
    read differently from the bytes git stores, and the digest would stop being a fact about
    the tree. Every byte a v3 cause is judged on comes through here.
    """
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True)


def normalize_source_text(data: bytes) -> bytes:
    """Byte-for-byte the rule `rust/build_support/shader_source_digest.rs` applies.

    Strip a UTF-8 BOM; fold CRLF and lone CR to LF; change nothing else. The Rust function's
    own doc comment says why, and this copy exists for the same reason it does: a digest that
    moves because a checkout has `core.autocrlf=true` is a machine fingerprint, and a screen
    that convicts a Windows checkout of a change a Linux checkout does not have is worse than
    no screen. `tests/ops/test_rewitness_cause_paths.py` pins the two implementations against
    the same inputs so this copy cannot drift from the one that computes the ledger's values.
    """
    body = data[3:] if data.startswith(b"\xef\xbb\xbf") else data
    out = bytearray()
    i = 0
    while i < len(body):
        if body[i : i + 1] == b"\r":
            out += b"\n"
            if body[i + 1 : i + 2] == b"\n":
                i += 1
        else:
            out += body[i : i + 1]
        i += 1
    return bytes(out)


def content_id(data: bytes) -> str:
    """The canonical identity of one production source file: sha256 of its normalised text.

    NOT the git blob sha, deliberately. A blob sha is a fact about an object database, so it
    is unavailable for a working-tree file that has not been added, it is not comparable
    across a checkout that applied a filter, and it names a container rather than a content.
    sha256-of-normalised-bytes is computable identically from `git show <rev>:<path>`, from a
    file on disk, and from a tarball with no git at all.
    """
    return hashlib.sha256(normalize_source_text(data)).hexdigest()


_ABSENT = "ABSENT"


def _blob_at(repo: Path, rev: str | None, rel: str) -> bytes | None:
    """The bytes of `rel` at `rev`, or from the working tree for `WORKTREE`/`None`."""
    if rev is None or rev == WORKTREE:
        path = repo / rel
        if not path.is_file():
            return None
        return path.read_bytes()
    r = _git_bytes(["show", f"{rev}:{rel}"], repo)
    if r.returncode != 0:
        return None
    return r.stdout


def _content_id_at(repo: Path, rev: str | None, rel: str, cache: dict) -> str:
    key = (rev, rel)
    if key not in cache:
        data = _blob_at(repo, rev, rel)
        cache[key] = _ABSENT if data is None else content_id(data)
    return cache[key]


def _variant_rows(repo: Path, rev: str | None, cache: dict) -> dict[str, tuple[str, str]]:
    """stem -> (glsl source relative to shaders/glsl, the `-D` defines as written).

    The same parse `rust/build.rs::parse_shader_variants` does: tab-separated, `#` comments
    and blank lines skipped. A malformed row is skipped here rather than fatal, because this
    screen is not the build — it only needs to know which template a stem is generated from.
    """
    key = (rev, "<variants>")
    if key in cache:
        return cache[key]
    rows: dict[str, tuple[str, str]] = {}
    data = _blob_at(repo, rev, SHADER_VARIANTS_REL)
    if data is not None:
        for line in normalize_source_text(data).decode("utf-8", "replace").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue
            rows[parts[0].strip()] = (parts[1].strip(), parts[2].strip() if len(parts) > 2 else "")
    cache[key] = rows
    return rows


def _include_names(body: bytes) -> list[str]:
    """`#include "x"` / `#include <x>` targets, by the same parse `glslc` and the Rust do."""
    names: list[str] = []
    for line in normalize_source_text(body).decode("utf-8", "replace").splitlines():
        rest = line.lstrip()
        if not rest.startswith("#"):
            continue
        rest = rest[1:].lstrip()
        if not rest.startswith("include"):
            continue
        rest = rest[len("include") :].lstrip()
        if not rest:
            continue
        close = {'"': '"', "<": ">"}.get(rest[0])
        if close is None:
            continue
        end = rest[1:].find(close)
        if end < 0:
            continue
        name = rest[1 : 1 + end].strip()
        if name:
            names.append(name)
    return names


def source_closure(repo: Path, rev: str | None, stem: str, cache: dict) -> tuple[set[str], str]:
    """-> (repo-relative paths whose bytes `source_digest_for(stem)` hashes, error or "").

    Resolution mirrors `rust/build_support/shader_source_digest.rs`: the including file's own
    directory, then `-I rust/shaders/include`, then `rust/shaders/glsl`. An include that
    resolves nowhere contributes no path — it also contributes no *content*, so it cannot be
    the corroborating transition of a cause and must not be silently counted as one.

    This is the set that makes a same-change cause checkable at all: it is derived from the
    tree under comparison, not from the record, so a record cannot widen it.
    """
    rows = _variant_rows(repo, rev, cache)
    if stem in rows:
        src = f"{GLSL_DIR_REL}/{rows[stem][0]}"
    else:
        src = f"{GLSL_DIR_REL}/{stem}.comp"
    body = _blob_at(repo, rev, src)
    if body is None:
        return set(), (
            f"stem {stem!r} has no shader source at {src} in {rev or 'the working tree'}, so "
            "its source closure cannot be computed and no cause can be corroborated against it"
        )
    paths = {src}
    pending = [(src, body)]
    seen: set[str] = set()
    while pending:
        frm, data = pending.pop()
        parent = frm.rsplit("/", 1)[0]
        for name in _include_names(data):
            if name in seen:
                continue
            seen.add(name)
            for cand in (f"{parent}/{name}", f"{GLSL_INCLUDE_DIR_REL}/{name}",
                         f"{GLSL_DIR_REL}/{name}"):
                sub = _blob_at(repo, rev, cand)
                if sub is not None:
                    paths.add(cand)
                    pending.append((cand, sub))
                    break
    return paths, ""


def _keys_of(text: str) -> set[str]:
    """Every `key` in a ledger body. A line without one is a header, not an entry."""
    return set(_entries_of(text))


def _entries_of(text: str) -> dict[str, dict]:
    """key -> entry for a ledger body. A line without a key is a header, not an entry."""
    out: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = entry.get("key")
        if isinstance(key, str) and key:
            out[key] = entry
    return out


def history_is_complete(repo: Path) -> tuple[bool, str]:
    """Can this checkout see the history the census takes its denominator from?

    DECLARED AND NOT GUARDED IN SESSION 18; GUARDED HERE. The census answers "was this
    key ever proven" by walking every revision that touched the ledger. In a shallow or
    partial clone that walk terminates at the graft boundary, the denominator silently
    shrinks to whatever was fetched, and a key deleted *before* the boundary reads as
    never-proven rather than as VANISHED. The screen would then print PASS — which is the
    exact failure mode this screen was written to make impossible, arriving through the
    clone depth instead of through a deletion.

    `--depth=1` is the CI default for most runners, so this is not a hypothetical: it is
    the configuration the screen is most likely to be run under. A check that cannot see
    its subject has not checked anything, so this returns an instrument error rather than
    a colour.
    """
    shallow = _git(["rev-parse", "--is-shallow-repository"], repo)
    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
        return False, (
            "this is a SHALLOW clone. `git rev-list --full-history` stops at the graft "
            "boundary, so 'ever proven' would mean 'proven since the fetch depth' and a "
            "proof deleted before it would read as never-existing. Re-run with a full "
            "clone (`git fetch --unshallow`)."
        )
    filt = _git(["config", "--get", "remote.origin.promisor"], repo)
    if filt.returncode == 0 and filt.stdout.strip() == "true":
        return False, (
            "this is a PARTIAL (promisor) clone. Blob fetches are lazy, so `git show "
            "<rev>:ledger` can fail per revision and those revisions drop silently out of "
            "the denominator. Re-run with a full clone."
        )
    if (repo / ".git" / "shallow").exists():
        return False, (
            "a .git/shallow graft file is present. The revision walk is truncated and the "
            "denominator is not 'every revision that ever touched the ledger'."
        )
    return True, ""


def witness_transitions(
    repo: Path, upto: str | None = None
) -> tuple[list[tuple[str, str, str, str, str]], list[str], dict]:
    """-> (transitions, walk_order, origins). Each transition is (revision, key, field, old, new).

    `origins` maps a transition's content signature `(field, key, old, new)` to the list of
    COMPARISONS it happened in: `(before_revision, after_revision)` pairs, where `before` is
    the last revision whose ledger still carried `old` for that key and `after` is the
    revision that carried `new`. That pair — and not the after-revision alone — is what a
    schema `rewitness/3` cause is corroborated against, because "the source changed" is a
    statement about two trees, and only naming both makes it checkable.

    FOUND BY THE LINUX LANE, 2026-08-04, ONE DAY AFTER THE REPAIR IT UNDID.

    115 of 121 entries had `source_digest` re-witnessed under the current hashing rule at
    `aea0147`, and `eee65aa` — an author regenerating the ledger from a base that predated
    that merge — put the withdrawn values back. Nothing saw it. No key went missing, so the
    key census and the loss invariant both reported 0; and on Windows a stale source digest
    with matching SPIR-V is `SOURCE-COSMETIC`, which forgives, so `--check` printed PASS
    over all 115. Only Linux declined, and it took the whole op suite with it.

    THE FIRST VERSION OF THIS ARM WAS WRONG AND THE WAY IT WAS WRONG IS THE POINT. It asked
    "did this value go back to something a later revision replaced", which is SYMMETRIC:
    it convicted my own repair exactly as loudly as it convicted the regression, because
    from the values alone the two are the same event seen from opposite ends. A screen
    cannot rank two alternating values without a frame. So the question asked here is not
    "which value is right" — it is "did the writer say they were moving it", which has an
    answer in the repository and needs no build and no platform.
    """
    scope = [upto] if upto else [DEFAULT_SCOPE]
    revs = _git(
        ["rev-list", "--full-history", "--topo-order", *scope, "--", LEDGER_REL], repo
    ).stdout.split()
    walk = list(reversed(revs))
    last: dict[tuple[str, str], str] = {}
    last_rev: dict[tuple[str, str], str] = {}
    origins: dict[tuple[str, str, str, str], list[tuple[str, str]]] = {}
    out: list[tuple[str, str, str, str, str]] = []
    for rev in walk:
        blob = _git(["show", f"{rev}:{LEDGER_REL}"], repo)
        if blob.returncode != 0:
            continue
        for key, entry in _entries_of(blob.stdout).items():
            for field in FRAME_WITNESSES:
                val = entry.get(field)
                if not val:
                    continue
                prev = last.get((key, field))
                if prev is not None and prev != val:
                    out.append((rev, key, field, prev, val))
                    origins.setdefault((field, key, prev, val), []).append(
                        (last_rev[(key, field)], rev)
                    )
                last[(key, field)] = val
                last_rev[(key, field)] = rev
    if upto is None:
        for key, entry in present_entries(repo, None).items():
            for field in FRAME_WITNESSES:
                val = entry.get(field)
                if not val:
                    continue
                prev = last.get((key, field))
                if prev is not None and prev != val:
                    out.append((WORKTREE, key, field, prev, val))
                    origins.setdefault((field, key, prev, val), []).append(
                        (last_rev[(key, field)], WORKTREE)
                    )
    return out, walk + [WORKTREE], origins


def _record_schema(rec: dict) -> str:
    """Which schema a record is written in. Absent `schema` means v1, and that is the ONLY
    thing absence is allowed to mean: an unknown value is an error, never a v1 fallback."""
    raw = rec.get("schema")
    if raw is None:
        return SCHEMA_V1
    if raw not in KNOWN_SCHEMAS:
        raise ValueError(
            f"a rewitness record declares schema {raw!r}, which this checker "
            f"does not know (known: {list(KNOWN_SCHEMAS)}). Refusing to read it. A checker "
            "that silently treats an unrecognised schema as the oldest one it knows will "
            "screen a record it has not understood and print PASS — which is how a "
            "declaration format becomes unenforceable the moment it is extended."
        )
    return raw


_HEX16 = re.compile(r"\A[0-9a-f]{16}\Z")
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


def _validate_transitions(rec: dict, path: Path, seen: set, schema: str) -> None:
    """The enumerated `(field, key, old, new)` moves a v2 or v3 record declares.

    Shared verbatim by both content-addressed schemas, because the matching rule is the
    same one: a move IS its content. v3 changes only what CAUSED the move, never what the
    move is, so a second copy of this rule would be a second answer to a question that has
    one.
    """
    trans = rec.get("transitions")
    if not isinstance(trans, list) or not trans:
        raise ValueError(f"{path}: {schema} `transitions` must be a non-empty list.")
    field = rec["field"]
    for t in trans:
        if not isinstance(t, dict) or set(t) != {"key", "old", "new"}:
            raise ValueError(
                f"{path}: every {schema} transition needs exactly key/old/new, got "
                f"{sorted(t) if isinstance(t, dict) else type(t).__name__}. Extra keys are "
                "refused rather than ignored: an ignored key is a claim nobody checked."
            )
        if t["old"] == t["new"]:
            raise ValueError(
                f"{path}: transition for {t['key']!r} declares old == new ({t['old']!r}). "
                "That is not a move, and declaring it hides the real one."
            )
        for side in ("old", "new"):
            if not _HEX16.match(str(t[side])):
                raise ValueError(
                    f"{path}: transition for {t['key']!r} has {side}={t[side]!r}, which is "
                    "not a 16-hex-digit digest. A malformed digest can never match a real "
                    "transition, so it would declare nothing while looking like a record."
                )
        sig = (field, t["key"], t["old"], t["new"])
        if sig in seen:
            raise ValueError(
                f"{path}: transition {sig} is declared more than once (within or across "
                "records). Declarations are consumed one-for-one, so a duplicate lets a "
                "second, undeclared move borrow the first one's declaration."
            )
        seen.add(sig)
    if "keys" in rec and rec["keys"] != len(trans):
        raise ValueError(
            f"{path}: record says keys={rec['keys']} but enumerates {len(trans)} "
            "transition(s). The enumeration is the declaration; the count is a summary of "
            "it, and a summary that disagrees with its subject is the defect this schema "
            "was introduced to remove."
        )


def _validate_v2(rec: dict, path: Path, seen: set[tuple[str, str, str, str]]) -> None:
    """Structural validation of a v2 record. Every branch here is a fail, never a warning.

    The rules exist because each of them, left unchecked, lets a record *look* like a
    declaration while ruling on nothing:

    * a bare `keys: 55` with no enumeration declares a NUMBER, not a set of moves — it
      matches any 55 moves, including 55 moves nobody intended;
    * a transition whose `old`/`new` are equal declares a non-event;
    * a duplicate transition lets one real move be "declared" twice and silently covers a
      second, different move of the same key;
    * a `keys` count that disagrees with the enumeration means one of the two is a lie and
      the checker cannot tell which.
    """
    missing = [f for f in REWITNESS_V2_FIELDS if not rec.get(f)]
    if missing:
        raise ValueError(
            f"{path}: a {SCHEMA_V2} record is missing {missing}. Unlike v1 this schema "
            "cannot fall back on a revision, so every one of these is load-bearing."
        )
    if "caused_by_content" in rec:
        raise ValueError(
            f"{path}: a {SCHEMA_V2} record carries `caused_by_content`, which only "
            f"{SCHEMA_V3} defines. Two cause rules in one record means the checker picks "
            "one and silently ignores the other — write the record in the schema whose "
            "cause you mean."
        )
    _validate_transitions(rec, path, seen, SCHEMA_V2)


def _validate_v3(rec: dict, path: Path, seen: set[tuple[str, str, str, str]]) -> None:
    """Structural validation of a v3 record: the SAME-CHANGE cause, declared as content.

    Everything refused here is refused because accepting it would let the cause become
    unfalsifiable:

    * a `caused_by` alongside `caused_by_content` — two rules, one record, and whichever
      the checker reads first silently excuses the other;
    * an unknown `kind` — a cause form nobody has written a corroboration for is a cause
      nobody checks, and reading it as the one form we do know is how a schema stops
      meaning anything the moment it is extended;
    * a `path` that is absolute, has a drive letter, uses `\\`, or contains `..` — none of
      those name a position in a git tree, so none of them could ever be read from the
      trees under comparison, and a path that cannot be read cannot be corroborated;
    * an `old` == `new` content id — that declares the source did NOT change, which is the
      opposite of a cause;
    * a duplicated path — the same file cannot have two different content transitions in
      one comparison, so the second is either a lie or a copy;
    * a `field` other than `source_digest` — the corroboration below reads shader source
      closures, and `toolchain` does not move because the source moved. A same-change cause
      for it would be corroborated by nothing while looking corroborated.
    """
    missing = [f for f in REWITNESS_V3_FIELDS if not rec.get(f)]
    if missing:
        raise ValueError(
            f"{path}: a {SCHEMA_V3} record is missing {missing}. Every one is load-bearing: "
            "this schema names no commit at all, so the content fields are the whole claim."
        )
    if "caused_by" in rec:
        raise ValueError(
            f"{path}: a {SCHEMA_V3} record also carries `caused_by`. A record declares ONE "
            "cause rule. `caused_by` is for a cause that has already landed in some earlier "
            "commit; `caused_by_content` is for a cause that arrives in this same change. "
            "Carrying both leaves the checker to choose, and a checker that chooses is a "
            "checker that excuses the other half without saying so."
        )
    if rec["field"] != "source_digest":
        raise ValueError(
            f"{path}: a {SCHEMA_V3} record declares field={rec['field']!r}. The same-change "
            "cause is corroborated against the SHADER SOURCE CLOSURE that `source_digest` "
            "hashes; no other frame witness is a function of that content, so for any other "
            "field this form would assert a corroboration it never performs."
        )
    cause = rec["caused_by_content"]
    if not isinstance(cause, dict) or set(cause) != {"kind", "paths"}:
        raise ValueError(
            f"{path}: `caused_by_content` needs exactly kind/paths, got "
            f"{sorted(cause) if isinstance(cause, dict) else type(cause).__name__}. An extra "
            "key is a claim nobody checked, and a missing one is a rule nobody applied."
        )
    if cause["kind"] not in KNOWN_CAUSE_KINDS:
        raise ValueError(
            f"{path}: `caused_by_content.kind` is {cause['kind']!r}, which this checker does "
            f"not know (known: {list(KNOWN_CAUSE_KINDS)}). Refusing to read it — a cause form "
            "with no corroboration behind it authorises everything it names."
        )
    paths = cause["paths"]
    if not isinstance(paths, list) or not paths:
        raise ValueError(
            f"{path}: `caused_by_content.paths` must be a non-empty list. A same-change cause "
            "with no content is prose, and prose authorises nothing."
        )
    seen_paths: set[str] = set()
    for p in paths:
        if not isinstance(p, dict) or set(p) != {"path", "old", "new"}:
            raise ValueError(
                f"{path}: every `caused_by_content.paths` entry needs exactly path/old/new, "
                f"got {sorted(p) if isinstance(p, dict) else type(p).__name__}."
            )
        rel = p["path"]
        if (
            not isinstance(rel, str)
            or not rel
            or rel.startswith("/")
            or "\\" in rel
            or ".." in rel.split("/")
            or re.match(r"\A[A-Za-z]:", rel)
        ):
            raise ValueError(
                f"{path}: cause path {rel!r} is not a repository-relative POSIX path. A path "
                "that does not name a position in a git tree can never be read from the trees "
                "under comparison, so it would declare a transition nothing could check."
            )
        if rel in seen_paths:
            raise ValueError(
                f"{path}: cause path {rel!r} is declared twice in one record. One file has one "
                "content transition per comparison; a second row is either wrong or a copy."
            )
        seen_paths.add(rel)
        for side in ("old", "new"):
            if not _HEX64.match(str(p[side])):
                raise ValueError(
                    f"{path}: cause path {rel!r} has {side}={p[side]!r}, which is not a "
                    "64-hex-digit sha256 of the normalised file content. A malformed content "
                    "id can never match a real file, so it would declare nothing."
                )
        if p["old"] == p["new"]:
            raise ValueError(
                f"{path}: cause path {rel!r} declares old == new. That says the source did not "
                "change, which is the opposite of a cause for a witness that did."
            )
    _validate_transitions(rec, path, seen, SCHEMA_V3)


def load_rewitness(path: Path) -> dict:
    """The declarations that make a witness move legible.

    Same discipline as `open_reds.json` and `proof_retired.json`: an event that is allowed
    to happen silently is an event nobody can tell from its opposite. `screened_since`
    bounds the window so adopting this screen does not retroactively convict history that
    predates it — transitions older than that revision are counted and named, not failed.

    ── WHY THERE IS A SECOND SCHEMA ────────────────────────────────────────────────────
    v1 matched a declaration to a move by REVISION. That is unsatisfiable under a
    squash merge, and not as an edge case — as the normal path. A declaration must be
    written *before* the merge exists, so the only revisions an author can name are
    branch commits; a squash merge then replays the change under a brand-new sha and
    erases every one of them. The declaration is now stale and the move it describes is
    undeclared: one event, two failures, and the register is worse than useless because
    it reports a red for a move that WAS declared, correctly, in advance.

    This happened twice on this project. `601ddcf` was named by a #35 declaration and
    erased by that PR's squash (issue #43's fourth red). The repair — re-pointing at the
    squash sha after the fact — is not a fix: it needs a human to notice, and it cannot
    be done before the merge, so the register is guaranteed to be wrong in the window
    where it matters.

    v2 removes the revision from the matching rule entirely. A move IS its content:
    `(field, key, old, new)`. That tuple is identical on the branch commit, on a squash
    commit, on a rebase, on a cherry-pick and in the working tree, because it is a fact
    about the ledger's bytes and not about the commit that carried them. `caused_by`
    still names a revision, but a different one and for a different purpose: the LANDED
    commit whose source change made the re-witness necessary. That one is knowable in
    advance, is checked for ancestry, and is evidence rather than an index.

    ── WHY THERE IS A THIRD SCHEMA: THE SAME-CHANGE CAUSE ──────────────────────────────
    v2's one surviving revision reopened v1's hole for exactly one case, and it is the
    common one: when the SOURCE change that made the re-witness necessary is in the SAME
    change as the ledger edit. `caused_by` must be an ancestor, and the only sha an author
    can write for a change that has not landed yet is a branch commit — which a squash
    erases. Measured on PR #53: green on the branch head and on a merge-commit landing,
    `FAIL(condition=unlanded_rewitness_cause)` on a squash. Main turns red *because* the
    change landed. There is no value an author can write today that is both knowable
    before the merge and an ancestor after it, so the gap is in the schema, not in the
    author.

    v3 removes the revision from the CAUSE rule too, and replaces it with the same kind of
    fact the matching rule already runs on: content under comparison.

        caused_by_content: {kind: "same_change", paths: [{path, old, new}, ...]}

    `old`/`new` are sha256 of the file's NORMALISED text (`normalize_source_text`, the
    identical rule `rust/build_support/shader_source_digest.rs` applies before hashing) in
    the two trees the witness moved between — `before` being the last revision whose ledger
    carried `old`, `after` the revision that carried `new`. Both trees exist in every
    landing shape: on a branch commit, on a squash (whose parent is main), on a rebase, on
    a merge commit, and in the working tree. Nothing in the record names a commit, a
    branch, a PR, a timestamp or an author, so nothing in it can be erased by a merge
    strategy.

    WHY THIS IS NOT CIRCULAR. The corroborating content is production shader source —
    `rust/shaders/**` and the variant row in `rust/src/ops/shader_variants.txt` — which is
    exactly the input `source_digest` is a hash of, and it is READ FROM THE TREES, never
    from the record and never from `evidence/`. The record cannot widen the set it is
    checked against: the closure is derived from the stems the moved ledger entries name,
    resolved through the same include rules the build uses. A cause path under `evidence/`
    or any other generated root is refused outright, so a record can never prove itself
    with the very file it lives in.
    """
    if not path.is_file():
        return {"screened_since": "", "rewitness": []}
    doc = json.loads(path.read_text(encoding="utf-8"))
    seen: set[tuple[str, str, str, str]] = set()
    for rec in doc.get("rewitness", []):
        schema = _record_schema(rec)
        if schema == SCHEMA_V2:
            _validate_v2(rec, path, seen)
            continue
        if schema == SCHEMA_V3:
            _validate_v3(rec, path, seen)
            continue
        missing = [f for f in REWITNESS_FIELDS if not rec.get(f)]
        if missing:
            raise ValueError(
                f"{path}: a rewitness record is missing {missing}. Moving a frame witness "
                "needs a name and a reason for the same argument accepting a red does: "
                "otherwise a regeneration from a stale base is indistinguishable from a "
                "deliberate re-witness, and on 2026-08-04 it was."
            )
    return doc


def _refused_cause_path(rel: str) -> str:
    """Why a v3 cause may not name this path, or "" if it may.

    The refusal is not a taste about tidiness. `evidence/` holds the ledger this screen
    reads, the register the record lives in, and every generated reading of both; a cause
    that points there is the tautology this schema exists to avoid — "the new proof digest
    proves itself". `ci/`, `docs/`, `bench/` and `tests/` are refused for the weaker but
    sufficient reason that no byte in them reaches `source_digest_for`, so a transition in
    one of them corroborates nothing while looking like corroboration.
    """
    for root in GENERATED_CAUSE_ROOTS:
        if rel == root.rstrip("/") or rel.startswith(root):
            return (
                f"is under {root!r}, which is generated evidence or lane machinery, not "
                "production shader source. A cause read from the evidence the record is "
                "part of proves itself; a cause read from a path the build never hashes "
                "proves nothing"
            )
    return ""


def uncorroborated_causes(
    repo: Path,
    doc: dict,
    origins: dict,
    at_scope: str | None = None,
    live: set[int] | None = None,
) -> list[tuple[str, str]]:
    """v3 records whose declared same-change cause is not corroborated by the trees.

    For every comparison `(before, after)` in which one of the record's declared moves
    actually happened, ALL of these must hold, and each is a separate sentence in the
    output because each is a different way to launder an undeclared move:

    1. every declared cause path reads `old` in `before` and `new` in `after` — so a cause
       that is absent, stale, or already landed before the witness moved fails;
    2. every declared cause path is in the source closure of a stem that one of the moved
       ledger entries names, or is the variant manifest with THAT stem's row changed — so
       an unrelated or over-broad path fails even when it did change;
    3. every path in that closure whose content DID move is declared — so a record cannot
       under-declare the cause and quietly cover a second, unrelated edit;
    4. every moved key has at least one declared, changed path in its OWN stems' closure —
       so one key's real cause cannot vouch for a second key it has nothing to do with;
    5. the entry's `shaders` list is the same in both trees — if the subject itself moved,
       the source content is not what explains the witness and this form must not pretend
       it does.

    A record whose transitions matched nothing is not judged here at all; that is
    `stale_rewitness_declaration`'s job, and judging it twice would convict a replay
    bounded before the change for a cause that had not happened yet.
    """
    out: list[tuple[str, str]] = []
    cache: dict = {}
    for i, rec in enumerate(doc.get("rewitness", [])):
        if _record_schema(rec) != SCHEMA_V3:
            continue
        if live is not None and i not in live:
            continue
        ident = f"{rec['field']} [{rec['owner']} {rec['date']}]"
        declared = {p["path"]: (p["old"], p["new"]) for p in rec["caused_by_content"]["paths"]}
        refused = [(p, why) for p in declared if (why := _refused_cause_path(p))]
        if refused:
            out.extend((ident, f"cause path {p!r} {why}") for p, why in refused)
            continue

        comparisons: dict[tuple[str, str], list[dict]] = {}
        for t in rec["transitions"]:
            sig = (rec["field"], t["key"], t["old"], t["new"])
            for before, after in origins.get(sig, []):
                comparisons.setdefault((before, after), []).append(t)
        for (before, after), moved in sorted(comparisons.items()):
            out.extend(
                (ident, why)
                for why in _corroborate_same_change(
                    repo, rec, declared, before, after, moved, cache
                )
            )
    return out


def _corroborate_same_change(
    repo: Path,
    rec: dict,
    declared: dict[str, tuple[str, str]],
    before: str,
    after: str,
    moved: list[dict],
    cache: dict,
) -> list[str]:
    """The five checks of `uncorroborated_causes`, for ONE `(before, after)` comparison."""
    where = f"{before[:12]}..{'the working tree' if after == WORKTREE else after[:12]}"
    problems: list[str] = []

    # 1. The declared content transition must BE THERE, exactly, in these two trees.
    for rel, (old, new) in sorted(declared.items()):
        was = _content_id_at(repo, before, rel, cache)
        now = _content_id_at(repo, after, rel, cache)
        if was != old:
            problems.append(
                f"{where}: cause path {rel!r} declares old={old[:16]}… but the base tree "
                + (
                    "does not contain that file at all"
                    if was == _ABSENT
                    else f"has {was[:16]}…"
                )
                + ". The declared change did not start from this tree — either the cause "
                "already landed without the witness move, or the record describes a "
                "different edit"
            )
        if now != new:
            problems.append(
                f"{where}: cause path {rel!r} declares new={new[:16]}… but the tree the "
                "witness moved in "
                + (
                    "does not contain that file at all"
                    if now == _ABSENT
                    else f"has {now[:16]}…"
                )
                + ". A cause whose content is not present in the same comparison as the "
                "move is not a cause, it is prose"
            )

    # 2/5. Which stems the moved entries name, read from BOTH trees.
    led_before = _ledger_entries(repo, before, cache)
    led_after = _ledger_entries(repo, after, cache)
    stems: set[str] = set()
    stems_by_key: dict[str, set[str]] = {}
    for t in moved:
        ea = led_after.get(t["key"])
        eb = led_before.get(t["key"])
        sa = {s for s in (ea or {}).get("shaders") or [] if isinstance(s, str)}
        sb = {s for s in (eb or {}).get("shaders") or [] if isinstance(s, str)}
        if not sa:
            problems.append(
                f"{where}: the moved entry {t['key']!r} names no `shaders`, so there is no "
                "source closure to corroborate a cause against. A same-change cause is only "
                "checkable for an entry that says which shader it was proven from"
            )
            continue
        if eb is not None and sb != sa:
            problems.append(
                f"{where}: the moved entry {t['key']!r} changed its `shaders` from "
                f"{sorted(sb)} to {sorted(sa)}. The SUBJECT moved, not just its source text, "
                "so a source-content cause would corroborate a different claim than the one "
                "the ledger is making"
            )
        stems_by_key[t["key"]] = sa
        stems |= sa

    changed_by_stem: dict[str, set[str]] = {}
    for stem in sorted(stems):
        cb, err_b = source_closure(repo, before, stem, cache)
        ca, err_a = source_closure(repo, after, stem, cache)
        if err_b or err_a:
            problems.append(f"{where}: {err_b or err_a}")
            continue
        changed = {
            rel
            for rel in cb | ca
            if _content_id_at(repo, before, rel, cache) != _content_id_at(repo, after, rel, cache)
        }
        row_b = _variant_rows(repo, before, cache).get(stem)
        row_a = _variant_rows(repo, after, cache).get(stem)
        if row_b != row_a:
            changed.add(SHADER_VARIANTS_REL)
        changed_by_stem[stem] = changed

    all_changed = set().union(*changed_by_stem.values()) if changed_by_stem else set()
    in_closure = set()
    for stem in changed_by_stem:
        for rev in (before, after):
            paths, err = source_closure(repo, rev, stem, cache)
            if not err:
                in_closure |= paths
    in_closure.add(SHADER_VARIANTS_REL)

    if stems and not all_changed:
        problems.append(
            f"{where}: no production source in the closure of "
            f"{sorted(stems)} changed between these two trees, so nothing here caused the "
            "witness to move. This is what a re-witness REPLAYED after its cause already "
            "landed looks like, and what a hand-edited digest looks like"
        )

    # 2. Relevance: declared, and actually part of what this witness is a hash of.
    for rel in sorted(declared):
        if rel in all_changed:
            continue
        if rel not in in_closure:
            problems.append(
                f"{where}: cause path {rel!r} is not in the source closure of "
                f"{sorted(stems)} — the stems the moved entries name. A path the build never "
                "hashes into these keys' `source_digest` cannot be why it moved, however "
                "real the edit to it was"
            )
        else:
            problems.append(
                f"{where}: cause path {rel!r} is in the closure but its content is identical "
                "in both trees, so it declares a transition that did not happen here"
            )

    # 3. Under-declaration: everything that moved in the closure must be named.
    for rel in sorted(all_changed - set(declared)):
        problems.append(
            f"{where}: {rel!r} is in the source closure of {sorted(stems)} and its content "
            "moved in this comparison, and the record does not declare it. A cause must be "
            "the WHOLE cause, or a second, undeclared edit rides in under the first one's "
            "authorisation"
        )

    # 4. Per-key relevance: every moved key's own closure must carry a declared change.
    for key, key_stems in sorted(stems_by_key.items()):
        mine = set().union(*(changed_by_stem.get(s, set()) for s in key_stems)) if key_stems else set()
        if not (mine & set(declared)):
            problems.append(
                f"{where}: the move on {key!r} is authorised by no declared cause path in its "
                f"own stems' closure ({sorted(key_stems)}). One key's real source change must "
                "not vouch for a key it does not share a shader with"
            )
    return problems


def _ledger_entries(repo: Path, rev: str | None, cache: dict) -> dict[str, dict]:
    key = (rev, "<ledger>")
    if key not in cache:
        try:
            cache[key] = present_entries(repo, None if rev == WORKTREE else rev)
        except FileNotFoundError:
            cache[key] = {}
    return cache[key]


def unlanded_causes(
    repo: Path,
    doc: dict,
    at_scope: str | None = None,
    live: set[int] | None = None,
) -> list[tuple[str, str]]:
    """v2 records whose `caused_by` is not an ancestor of the scope — i.e. has not landed.

    This is the one place a v2 record IS judged on a revision, and the direction matters.
    `caused_by` does not index the declaration; it names the source change that made the
    re-witness necessary. That change must already be in the history the declaration is
    being read against, because a re-witness justified by a commit that is not in this
    history is justified by nothing a reader here can see.

    A BRANCH-ONLY revision therefore fails: it resolves (the object exists in the clone)
    but is not an ancestor, which is exactly the state a squash merge leaves behind and
    exactly the state that must not be quietly accepted.

    Only records that are DOING WORK in this scope are judged — `live` is the set of record
    indices whose transitions matched a move in this walk. A replay bounded at an older
    revision must not convict a declaration written for a change that had not happened yet;
    that is the same append-only discipline `screened_since` enforces for the walk.
    """
    out: list[tuple[str, str]] = []
    head = at_scope or DEFAULT_SCOPE
    for i, rec in enumerate(doc.get("rewitness", [])):
        if _record_schema(rec) != SCHEMA_V2:
            continue
        if live is not None and i not in live:
            continue
        cb = rec["caused_by"]
        ok = _git(["rev-parse", "--verify", "--quiet", cb + "^{commit}"], repo)
        if ok.returncode != 0:
            out.append((cb, "does not resolve to a commit in this repository"))
            continue
        anc = _git(["merge-base", "--is-ancestor", cb, head], repo)
        if anc.returncode != 0:
            out.append((cb, f"resolves but is NOT an ancestor of {head} — it has not landed"))
    return out


def screen_transitions(
    repo: Path,
    transitions: list[tuple[str, str, str, str, str]],
    walk: list[str],
    doc: dict,
    at_scope: str | None = None,
) -> tuple[list, list, int, list, set[int]]:
    """-> (undeclared, stale_declarations, out_of_frame_count, overdeclared, live_v2).

    The frame boundary is a POSITION IN THE WALK, not an ancestry test, and that is the
    second thing this arm got wrong before it got it right. `merge-base --is-ancestor` said
    the offending commit was out of frame — correctly, because it was authored on a side
    branch that forked BEFORE the boundary and only reached main through a later merge.
    That is not an edge case: it is exactly how the regression happened, so a boundary that
    excuses it excuses the only event the arm exists for.
    """
    since = (doc.get("screened_since") or "").strip()
    cut = -1
    if since:
        ok = _git(["rev-parse", "--verify", "--quiet", since + "^{commit}"], repo)
        if ok.returncode != 0:
            raise ValueError(
                f"screened_since={since!r} does not resolve to a commit in this repository. "
                "This is not a detail: an unresolvable boundary puts EVERY transition out of "
                "frame and the screen prints PASS having ruled on nothing. That is the same "
                "failure this whole file exists to prevent, arriving through a typo — and it "
                "happened on the first run, with a hand-written sha."
            )
        full = ok.stdout.strip()
        if full in walk:
            cut = walk.index(full)
        elif at_scope is None:
            raise ValueError(
                f"screened_since={since!r} resolves, but never touched {LEDGER_REL}, so it "
                "names no position in this walk and cannot bound it. Use a revision that "
                "wrote the ledger."
            )
        else:
            # A replay bounded at an EARLIER revision than the boundary. Everything in this
            # walk predates the rule, which is the honest answer, not an error: adopting a
            # screen must not retroactively convict history it could not have governed.
            cut = len(walk)
    pos = {rev: i for i, rev in enumerate(walk)}
    decls = doc.get("rewitness", [])
    in_frame, out_of_frame = [], 0
    for t in transitions:
        if cut >= 0 and pos.get(t[0], len(walk)) < cut:
            out_of_frame += 1
            continue
        in_frame.append(t)

    # v2/v3 index: (field, key, old, new) -> record position. Content, not commit — see
    # `load_rewitness` for why the commit cannot be the index.
    by_content: dict[tuple[str, str, str, str], int] = {}
    for i, d in enumerate(decls):
        if _record_schema(d) not in CONTENT_SCHEMAS:
            continue
        for t in d["transitions"]:
            by_content[(d["field"], t["key"], t["old"], t["new"])] = i

    matched: set[int] = set()
    matched_content: set[tuple[str, str, str, str]] = set()
    undeclared = []
    for rev, key, field, old, new in in_frame:
        sig = (field, key, old, new)
        hit = by_content.get(sig)
        if hit is not None:
            matched.add(hit)
            matched_content.add(sig)
            continue
        for i, d in enumerate(decls):
            if _record_schema(d) != SCHEMA_V1 or d["field"] != field:
                continue
            dr = d["revision"]
            if dr == rev or (dr != WORKTREE and rev != WORKTREE and rev.startswith(dr)):
                hit = i
                break
        if hit is None:
            undeclared.append((rev, key, field, old, new))
        else:
            matched.add(hit)
    stale = []
    overdeclared: list[tuple[str, str, str, str]] = []
    cache: dict = {}
    for i, d in enumerate(decls):
        schema = _record_schema(d)
        if schema in CONTENT_SCHEMAS:
            if schema == SCHEMA_V2:
                # A v2 record is in scope when the LANDED revision it blames is in this walk.
                # `caused_by` is the source change that made the re-witness necessary, so if
                # that change is not in frame the re-witness cannot be either.
                cb = d["caused_by"]
                in_scope = any(r == cb or r.startswith(cb) for r in walk)
            else:
                # A v3 record names no revision, so "is this record's cause in this history"
                # is asked the same way everything else about it is asked: on content. The
                # cause is in scope when the tree being screened already carries the `new`
                # content of at least one declared path. A replay bounded before the source
                # change therefore does not convict a declaration for an edit that had not
                # happened yet — the same discipline `screened_since` enforces for the walk.
                tip = at_scope or DEFAULT_SCOPE
                in_scope = any(
                    _content_id_at(repo, tip if at_scope else None, p["path"], cache) == p["new"]
                    for p in d["caused_by_content"]["paths"]
                )
            unmatched = [
                t for t in d["transitions"]
                if (d["field"], t["key"], t["old"], t["new"]) not in matched_content
            ]
            # Over-declaration is reported separately from staleness on purpose. A wholly
            # unmatched record is a record for an event that did not happen; a PARTIALLY
            # matched one is worse, because it looks live while quietly carrying rows that
            # match nothing — which is how a wrong `old` or a typo'd key would otherwise
            # ride along inside an otherwise-correct declaration and never be read.
            if in_scope and unmatched and i in matched:
                for t in unmatched:
                    overdeclared.append((d["field"], t["key"], t["old"], t["new"]))
            if i in matched:
                continue
            if in_scope:
                stale.append(d)
            continue
        if i in matched:
            continue
        dr = d["revision"]
        if dr == WORKTREE:
            in_scope = at_scope is None
        else:
            in_scope = any(r == dr or r.startswith(dr) for r in walk)
        if in_scope:
            stale.append(d)
    live_content = {i for i in matched if _record_schema(decls[i]) in CONTENT_SCHEMAS}
    return undeclared, stale, out_of_frame, overdeclared, live_content


def accidental(doc: dict) -> list[dict]:
    """Declared moves whose declaration says they were NOT deliberate.

    A record is not a suppression. `deliberate: false` means the move happened and was an
    accident; the entry stays so the check keeps ruling on that revision and so the next
    reader can see that this class of accident has happened, how often, and what it cost.
    Printing the count on every run is the difference between a record and a silence.
    """
    return [d for d in doc.get("rewitness", []) if d.get("deliberate") is False]


def ever_proven(repo: Path, upto: str | None = None) -> dict[str, str]:
    """key -> the EARLIEST revision that carried it.

    `git rev-list --full-history HEAD` and not `--simplify-merges`/`--follow`: the removal
    this screen exists for was invisible to a simplified log, so a simplified log cannot be
    the input.

    `--full-history` is load-bearing and was added after the first replay arm reported PASS on
    the very merge it was written to convict. Default history simplification drops commits
    whose change is "not interesting" for the path once a merge has picked a side — at
    `eb84364`, `git rev-list <rev> -- evidence/proof_ledger.jsonl` lists **13** revisions and
    `26fd93f`, the commit that proved the three `Cast` forms, is not one of them; with
    `--full-history` it lists **55** and `26fd93f` is there. That is not a detail: the
    simplification that hid the proving commit from the file's own log is the same
    simplification that would hide it from this census, and a screen for a deletion must not
    take its denominator from the view the deletion is invisible in.

    `upto` bounds the walk to the ancestors of one revision, and it is not an optimisation.
    Without it the replay arm is dishonest in both directions: a key proven *after* the
    revision under test would be reported VANISHED from it (it was never there to lose), and
    the count would answer a question about the future. The denominator must be "what this
    revision's own history had already proven".

    `rev-list` is reverse-chronological, so the walk is reversed to make "first", first.
    """
    scope = [upto] if upto else [DEFAULT_SCOPE]
    revs = _git(["rev-list", "--full-history", *scope, "--", LEDGER_REL], repo).stdout.split()
    seen: dict[str, str] = {}
    for rev in reversed(revs):
        blob = _git(["show", f"{rev}:{LEDGER_REL}"], repo)
        if blob.returncode != 0:
            continue
        for key in _keys_of(blob.stdout):
            seen.setdefault(key, rev)
    return seen


def present_at(repo: Path, rev: str | None) -> set[str]:
    return set(present_entries(repo, rev))


def present_entries(repo: Path, rev: str | None) -> dict[str, dict]:
    if rev is None:
        path = repo / LEDGER_REL
        if not path.is_file():
            raise FileNotFoundError(
                f"no ledger at {path}; the comparison input is missing, so nothing was ruled "
                "on either way — that is UNOBSERVABLE, not PASS"
            )
        return _entries_of(path.read_text(encoding="utf-8"))
    blob = _git(["show", f"{rev}:{LEDGER_REL}"], repo)
    if blob.returncode != 0:
        raise FileNotFoundError(f"{rev} has no {LEDGER_REL}: {blob.stderr.strip()}")
    return _entries_of(blob.stdout)


def load_retired_at(repo: Path, rev: str | None) -> dict[str, dict]:
    """Read the retirement register AS OF the revision under test.

    A retirement written today cannot govern a deletion that predates it. `--at` replays a
    historical revision, and reading the *worktree's* register while screening a *historical*
    ledger mixes two moments: the 43 keys retired on 2026-08-04 were, at `eb84364`, still in
    the ledger — so today's register applied to that revision reports them
    `retired_but_present` and returns 1 before the real `proof_vanished` finding is ever
    printed. The replay then convicts the right revision for the wrong reason, which is the
    same defect as acquitting it.

    This is the argument `screened_since` already encodes for frame witnesses: adopting a
    screen must not retroactively rule on history. If the register does not exist at `rev`,
    nothing was retired then — `{}`, not "the file is missing", because at that revision it
    genuinely was not.
    """
    if rev is None:
        return load_retired(proof_retirement.register_path(repo))
    blob = _git(["show", f"{rev}:{RETIRED_REL}"], repo)
    if blob.returncode != 0:
        return {}
    return proof_retirement.load_text(f"{rev}:{RETIRED_REL}", blob.stdout)


def load_retired(path: Path) -> dict[str, dict]:
    """Read the one retirement register: `{"retired": [{key, owner, date, reason}, ...]}`.

    Thin by design. The shape, the required fields and the refusals are `ci/proof_retirement.py`,
    which `rust/tools/gen_proof_ledger.py` also imports, so "what counts as a withdrawal" is
    answered in exactly one place. A second answer here would be a second register with the first
    one's filename.
    """
    return proof_retirement.load(path)


def guard_one_register(repo: Path) -> int:
    """Refuse to run while two retirement registers exist.

    Not a FAIL(condition): with two registers this screen cannot say which one is the record,
    so it has no observation to report — R13 makes that ERROR(instrument), never a detection
    and never a pass.
    """
    superseded = proof_retirement.superseded_path(repo)
    if not superseded.is_file():
        return 0
    print("LEDGER-CENSUS: ERROR(instrument=two_retirement_registers)")
    print(
        f"  {SUPERSEDED_RETIRED_REL} exists alongside {RETIRED_REL}. These are two registers "
        "for one fact, which is the arrangement that made this screen report 43 deliberate "
        "retirements as VANISHED on every run for days: it read one file, the ledger's "
        "producer read the other, and each was internally correct. Fold it into "
        f"{RETIRED.name} and delete it."
    )
    return 2


KNOWN_LIMITS = {
    "shallow_clone_is_unobservable_not_clean": (
        "This census walks every revision that touched the ledger. In a shallow clone most "
        "of them are absent, so the walk is short and the answer 'nothing vanished' is a "
        "statement about the fetch depth, not about the project. history_is_complete() now "
        "GUARDS this — ERROR(instrument=truncated_history), exit 2 — but a guard is not a "
        "fix: in a shallow CI checkout the screen still rules on nothing, and a lane that "
        "only tests for exit 1 would read that as tolerable. See ci/open_reds.json "
        "known_limits id=ledger_census_is_unobservable_in_a_shallow_clone."
    ),
}


def _assert_known_limit(name: str) -> int:
    if name not in KNOWN_LIMITS:
        print(
            f"ERROR(instrument=unknown_limit): {name!r} is not a declared limit of this "
            f"screen. Declared: {sorted(KNOWN_LIMITS)}. A register entry accepting a limit "
            "the screen does not admit to is an acceptance of nothing."
        )
        return 2
    print(f"KNOWN-LIMIT {name}")
    print(f"  {KNOWN_LIMITS[name]}")
    print(
        "\nFAIL(condition=known_limit_still_open): declared, owned, bounded, and red on "
        "purpose. It goes green when the limit is closed, not when somebody stops looking."
    )
    return 1


def screen(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument(
        "--at",
        default=None,
        help="screen the ledger as of REV instead of the working tree. This is the replay "
             "arm: it is how the screen is shown to convict a real, historical deletion "
             "rather than only a planted one.",
    )
    ap.add_argument("--json", default="")
    ap.add_argument("--list-retired", action="store_true")
    ap.add_argument(
        "--allow-shallow",
        action="store_true",
        help="run the census against a truncated history anyway. The result is not a "
             "census: 'ever proven' becomes 'proven since the fetch depth'. Exists so the "
             "guard is testable in both polarities, and it is reported in the frame line.",
    )
    ap.add_argument("--assert-known-limit", default="")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()

    if args.assert_known_limit:
        return _assert_known_limit(args.assert_known_limit)

    complete, why = history_is_complete(repo)
    if not complete and not args.allow_shallow:
        print(f"ERROR(instrument=truncated_history): {LEDGER_REL} census cannot run here.")
        print(f"  {why}")
        print(
            "  This is deliberately NOT a PASS and NOT a FAIL. The screen's whole claim is "
            "about what the history contains; with the history truncated it has no "
            "denominator, and an answer computed from a denominator that silently shrank is "
            "the failure this screen exists to prevent, arriving through the clone depth."
        )
        return 2

    rc = guard_one_register(repo)
    if rc:
        return rc
    try:
        retired = load_retired_at(repo, args.at)
    except RetirementError as exc:
        # R13: a screen that cannot read the register granting the exemption has not observed
        # anything, so this is never a colour. It is the same class as two registers — and it is
        # printed as a verdict rather than raised as a traceback, because a traceback is not a
        # token and this lane rules on tokens.
        print("LEDGER-CENSUS: ERROR(instrument=unreadable_retirement_register)")
        print(f"  {exc}")
        print(
            f"  Every key withdrawn in {RETIRED_REL} would otherwise read as VANISHED, which is "
            "the false positive this screen spent days producing when the register it read did "
            "not exist. Repair the register; do not delete it to make this green."
        )
        return 2
    if args.list_retired:
        if not retired:
            print("no proofs are retired.")
            return 0
        for key, rec in sorted(retired.items()):
            print(f"{key}\n    {rec['owner']} {rec['date']}: {rec['reason']}")
        return 0

    ever = ever_proven(repo, args.at)
    now = present_at(repo, args.at)
    rw_doc = load_rewitness(repo / "evidence" / "proof_rewitness.json")
    _trans, _walk, _origins = witness_transitions(repo, args.at)
    undeclared, stale_decl, out_of_frame, overdeclared, live_content = screen_transitions(
        repo, _trans, _walk, rw_doc, args.at
    )
    unlanded_cause = unlanded_causes(repo, rw_doc, args.at, live_content)
    uncorroborated = uncorroborated_causes(repo, rw_doc, _origins, args.at, live_content)
    where = args.at or "the working tree"

    vanished = sorted(k for k in ever if k not in now and k not in retired)
    retired_present = sorted(k for k in retired if k in now)
    unhistoried = sorted(k for k in now if k not in ever)

    n, m, k, d = len(ever), len(now), len([x for x in retired if x not in now]), len(vanished)
    print(f"LEDGER CENSUS of {LEDGER_REL} at {where}")
    print(
        f"  retirement register {RETIRED_REL} read at {where}"
        f" ({len(retired)} record(s))"
    )
    print(
        f"  {n} ever proven = {m - len(unhistoried)} present now + {k} retired + {d} VANISHED"
        + (f" (+{len(unhistoried)} present but not yet in history — uncommitted)" if unhistoried else "")
    )
    print(
        f"  frame witnesses {list(FRAME_WITNESSES)}: {len(undeclared)} UNDECLARED move(s), "
        f"{len(stale_decl)} declaration(s) matching nothing, {len(overdeclared)} "
        f"over-declared transition(s), {len(unlanded_cause)} unlanded cause(s), "
        f"{len(uncorroborated)} uncorroborated same-change cause(s), "
        f"{out_of_frame} out of frame "
        f"(before screened_since={rw_doc.get('screened_since') or '<unset>'})"
        + ("  [history truncated: --allow-shallow]" if not complete else "")
    )
    for d in accidental(rw_doc):
        if d.get("revision") is None:
            continue
        if not any(r == d["revision"] or r.startswith(d["revision"]) for r in _walk):
            continue
        print(
            f"  DECLARED ACCIDENT: {d['revision'][:12]} moved `{d['field']}` on "
            f"{d.get('keys', '?')} entr(ies) and nobody meant to "
            f"[{d['owner']} {d['date']}]. Recorded, not excused; the check still rules on it."
        )
    if retired:
        print(f"  retired ({len(retired)}):")
        for key in sorted(retired):
            rec = retired[key]
            print(f"    - {key}  [{rec['owner']} {rec['date']}: {rec['reason']}]")

    if retired_present:
        print("")
        print(
            f"FAIL(condition=retired_but_present): {len(retired_present)} key(s) are recorded "
            "as withdrawn and are in the ledger anyway. A retirement is a statement that the "
            "proof is gone; if it came back, the retirement is now a false record of the "
            "file's contents and must be deleted, not left to agree with nothing."
        )
        for key in retired_present:
            print(f"  - {key}")
        return 1

    if vanished:
        print("")
        print(
            f"FAIL(condition=proof_vanished): {len(vanished)} proof(s) were committed to this "
            "ledger and are no longer in it, with no retirement record."
        )
        for key in vanished:
            print(f"  - {key}\n      first proven in {ever[key][:12]}; "
                  f"`git log -S {key!r} -- {LEDGER_REL}` finds the addition")
        print(
            "\n  A proof does not leave this ledger by being absent. If the form is genuinely "
            f"withdrawn, record it in {RETIRED_REL} with an owner, a "
            "date and a reason. If it is not, this is a deletion — most likely inside a merge "
            "conflict resolution, which is the one write neither `--check` nor the "
            "shrinking-write guard can see."
        )
        rc = 1
    elif undeclared:
        print("")
        print(
            f"FAIL(condition=undeclared_witness_move): {len(undeclared)} §8.9.19 frame "
            "witness(es) moved with no record of anyone deciding to move them. No key went "
            "missing, so the census above and the loss invariant both read clean; the loss "
            "is INSIDE surviving entries."
        )
        seen_rev: dict[tuple[str, str], int] = {}
        for rev, key, field, old, new in undeclared:
            seen_rev[(rev, field)] = seen_rev.get((rev, field), 0) + 1
        for (rev, field), cnt in sorted(seen_rev.items(), key=lambda x: -x[1]):
            sample = next(t for t in undeclared if t[0] == rev and t[2] == field)
            print(
                f"  - {rev[:12]}: {field} moved on {cnt} entr(ies), "
                f"e.g. {sample[1]}\n      {sample[3]!r} -> {sample[4]!r}"
            )
        print(
            f"\n  A frame witness is the field that decides, on a SECOND platform, whether a "
            f"difference was the compiler or the kernel. Moving one is legitimate and routine "
            f"(--reprove, --backfill-frame --rewitness-source); moving one WITHOUT SAYING SO "
            f"is not, because a regeneration from a stale base then looks exactly like a "
            f"deliberate re-witness. On 2026-08-04 it looked exactly like one on 115 of 121 "
            f"entries, Windows forgave every one of them as SOURCE-COSMETIC, and the Linux "
            f"lane declined all 115. Declare the move in "
            f"{REWITNESS.relative_to(REPO).as_posix()} with an owner, a date and a reason — "
            f"or, if it was not deliberate, repair it."
        )
        rc = 1
    elif stale_decl:
        print("")
        print(
            f"FAIL(condition=stale_rewitness_declaration): {len(stale_decl)} declaration(s) "
            "in " + REWITNESS.relative_to(REPO).as_posix() + " match no witness move in the "
            "history. Good news, and the same arm as `stale_acceptance`: a declaration that "
            "has stopped describing anything is a record of a decision nobody can check, and "
            "deleting it is what stops this register rotting."
        )
        for d in stale_decl:
            ident = d.get("revision") or f"caused_by={d.get('caused_by')}"
            print(f"  - {ident} {d['field']} [{d['owner']} {d['date']}: {d['reason']}]")
        rc = 1
    elif overdeclared:
        print("")
        print(
            f"FAIL(condition=overdeclared_witness_move): {len(overdeclared)} transition(s) "
            f"declared in {REWITNESS.relative_to(REPO).as_posix()} match NO move in the "
            "history, inside records whose other transitions do match."
        )
        for field, key, old, new in overdeclared[:12]:
            print(f"  - {field} {key}\n      declared {old!r} -> {new!r}, which never happened")
        if len(overdeclared) > 12:
            print(f"  ... +{len(overdeclared) - 12} more")
        print(
            "\n  A partially-matching declaration is worse than a wholly stale one, because it "
            "looks live. A wrong `old`, a typo'd key or a row copied from another move rides "
            "along inside an otherwise-correct record and is never read — which is precisely "
            "the state a bare `keys: <n>` count made unobservable, and the reason this schema "
            "enumerates transitions instead of counting them. Fix the row or remove it; do not "
            "widen the matcher."
        )
        rc = 1
    elif unlanded_cause:
        print("")
        print(
            f"FAIL(condition=unlanded_rewitness_cause): {len(unlanded_cause)} declaration(s) "
            "blame a source change that is not in this history."
        )
        for cb, why in unlanded_cause:
            print(f"  - caused_by={cb}: {why}")
        print(
            "\n  `caused_by` is the landed commit whose source change made the re-witness "
            "necessary. A branch-only revision RESOLVES but is not an ancestor — the exact "
            "state a squash merge leaves behind — so accepting it would reintroduce, through "
            "the one revision this schema still reads, the failure the schema removed from "
            "the matching rule. If the cause is in THIS change rather than an earlier one, "
            f"there is no sha that can be right: write the record in {SCHEMA_V3} and declare "
            "the source content transition instead."
        )
        rc = 1
    elif uncorroborated:
        print("")
        print(
            f"FAIL(condition=uncorroborated_rewitness_cause): {len(uncorroborated)} "
            f"{SCHEMA_V3} declaration(s) claim a same-change cause the compared trees do not "
            "show."
        )
        for ident, why in uncorroborated[:12]:
            print(f"  - {ident}: {why}")
        if len(uncorroborated) > 12:
            print(f"  ... +{len(uncorroborated) - 12} more")
        print(
            f"\n  `caused_by_content` says: THIS change's own edit to production shader "
            "source is why the witness moved. That is checkable without any commit name — "
            "the declared path must read its `old` content in the tree the ledger left and "
            "its `new` content in the tree the ledger arrived at, it must be in the source "
            "closure of the stems the moved entries name, and everything in that closure "
            "that moved must be declared. A cause that is absent, stale, already landed, "
            "unrelated, over-broad or under-declared fails here rather than authorising a "
            "witness move nobody can trace to a source change."
        )
        rc = 1
    else:
        print("")
        print(
            f"PASS: every key this ledger has ever carried is either present or retired with a "
            f"reason, and no §8.9.19 frame witness has moved without a record of someone "
            f"deciding to move it. Read as: nothing has been proven and then quietly "
            f"unproven, and nothing has been re-witnessed and then quietly un-re-witnessed."
        )
        rc = 0

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "at": args.at or "worktree",
                    "ever_proven": n,
                    "present": m,
                    "retired": sorted(retired),
                    "vanished": vanished,
                    "witnesses_undeclared_moves": [
                        {"revision": r, "key": k2, "field": f, "from": o, "to": nv}
                        for r, k2, f, o, nv in undeclared
                    ],
                    "rewitness_declarations_matching_nothing": stale_decl,
                    "rewitness_transitions_overdeclared": [
                        {"field": f, "key": k2, "from": o, "to": nv}
                        for f, k2, o, nv in overdeclared
                    ],
                    "rewitness_causes_not_landed": [
                        {"caused_by": cb, "why": why} for cb, why in unlanded_cause
                    ],
                    "rewitness_causes_uncorroborated": [
                        {"record": ident, "why": why} for ident, why in uncorroborated
                    ],
                    "witness_moves_out_of_frame": out_of_frame,
                    "history_complete": complete,
                    "retired_but_present": retired_present,
                    "not_yet_in_history": unhistoried,
                    "verdict": "PASS" if rc == 0 else "FAIL(condition)",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return rc


if __name__ == "__main__":
    try:
        sys.exit(screen())
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR(instrument=register_unusable): {exc}")
        print(
            "  Exit 2, deliberately distinct from 1. Nothing was ruled on either way — that "
            "is UNOBSERVABLE, not PASS and not FAIL."
        )
        sys.exit(2)


