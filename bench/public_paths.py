#!/usr/bin/env python3
"""Where a probe is allowed to write, and what it is allowed to write about a path.

Two rules, both of which this tree already had as habits and neither of which it had as code.

RULE 1 — A PROBE DOES NOT DIRTY COMMITTED EVIDENCE
==================================================
`bench/results/*.json` are *records*: a committed statement that a particular run, on a
particular machine, observed particular numbers. Overwriting one is not "refreshing" it, it is
replacing evidence with different evidence under the same name — and if the new run had a
different model, a different driver or a different kernel, the file's git history now reads as
though one measurement drifted when in fact two different measurements were conflated.

The concrete failure this was written for: `probe_weight_reread.py` ended `main()` with an
unconditional `write_text` onto the tracked `bench/results/weight_reread_phi35.json`. Anyone who
ran the probe to *read* it — including to check that it still ran at all — silently modified
tracked evidence and then, more often than not, committed it along with whatever else was in
their working tree.

So: **writing is opt-in, the default destination is untracked, and writing over a tracked file
requires saying so.** `bench/.gitignore` already ignores `_scratch/`, which is where the default
goes.

RULE 2 — A RECORD NAMES A PATH IN PUBLIC FORM
=============================================
`weight_reread_phi35.json` carries

    "onnx_file": "C:\\\\Users\\\\<someone>\\\\<model-cache>\\\\..."

An absolute path under a home directory in a file that is pushed to a public repository is a
leak of the operator's account name, and it is *not* provenance: `onnx_sha256` is the provenance,
and it is exact where the path is merely suggestive. Two operators with the same model have
different paths and the same hash; one operator with two different models has the same-shaped
path and different hashes. The path was never the identifying field.

[`public_path`] converts a filesystem path into the form the record should carry — repo-relative
where the file is in the repo, `<home>`-relative where it is under the operator's home, and
name-only where it is neither. [`assert_public`] is the screen: it *fails closed* on anything
that still looks absolute or still contains the current account name.

WHY THIS IS A MODULE AND NOT A PATCH IN THE PROBE
=================================================
Because it is testable without a GPU, without a model, and without Foundry. `probe_weight_reread`
cannot run in CI — it needs a 2 GB Phi-3.5 checkout and a compiled SPIR-V module — so a guard
that lives inside its `main()` is a guard nothing exercises. Everything here is pure except
`is_tracked`, which shells out to `git`, and that one fails closed.

One thing deliberately NOT moved here: the probes' `out.write_text(json.dumps(...))` call
itself. `ci/phi35_identity_audit.py` finds record producers structurally, by looking for a write
sink whose argument subtree contains `json.dumps(...)`; hoisting the write into a helper would
make every probe that used the helper vanish from that audit's producer set. The screen would go
quiet rather than red, which is the failure mode this tree spends most of its CI budget refusing
to accept. The helper therefore hands back a *destination*, and the probe still does its own
writing in its own file where the audit can see it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

#: Repository root: `bench/public_paths.py` -> `bench/` -> repo.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Untracked scratch directory. `bench/.gitignore` ignores `_scratch/`, so a default write here
#: cannot dirty the index however carelessly it is committed.
SCRATCH = ROOT / "bench" / "_scratch"


class LeakError(Exception):
    """A value that would have gone into a published record and should not have.

    Its own type rather than `ValueError` so a caller cannot catch it by accident while catching
    something else — the whole point is that it is not recoverable in place.
    """


class TrackedWriteRefused(Exception):
    """A probe tried to overwrite committed evidence without being told it could."""


def _account_tokens() -> list[str]:
    """Strings that name the operator and must never reach a published record.

    The environment's idea of who is running (`USERNAME` on Windows, `USER`/`LOGNAME` on POSIX)
    plus the home directory's own name, which is the one that survives when the environment has
    been scrubbed by a CI runner but the paths have not.

    Names of two characters or fewer are dropped. A user called `jo` would otherwise make
    `bench/json/...` unpublishable, and a guard that refuses everything is not a stricter guard —
    it is a guard that gets switched off.
    """
    seen: list[str] = []
    for candidate in (
        os.environ.get("USERNAME"),
        os.environ.get("USER"),
        os.environ.get("LOGNAME"),
        pathlib.Path.home().name,
    ):
        if candidate and len(candidate) > 2 and candidate.lower() not in [s.lower() for s in seen]:
            seen.append(candidate)
    return seen


def _scrub_account(text: str) -> str:
    """Replace every case-spelling of the operator's name in `text` with ``<user>``.

    Necessary because making a path relative to `<home>` removes the account name from the
    *prefix* and not from the rest of it: `<home>/AppData/Local/Temp/pytest-of-hortensia/...` is
    home-relative, POSIX, and still says who ran it. Directory names carry account names far more
    often than one expects — temp directories, cache keys, scratch mounts.
    """
    out = text
    for token in _account_tokens():
        lowered = out.lower()
        needle = token.lower()
        if needle not in lowered:
            continue
        rebuilt, i = [], 0
        while True:
            j = out.lower().find(needle, i)
            if j < 0:
                rebuilt.append(out[i:])
                break
            rebuilt.append(out[i:j])
            rebuilt.append("<user>")
            i = j + len(needle)
        out = "".join(rebuilt)
    return out


def public_path(p: pathlib.Path | str, root: pathlib.Path | None = None) -> str:
    """The form of `p` that a published record may carry.

    Three cases, in order, and the order matters — the repo is very often *inside* the home
    directory, and a repo-relative path is strictly more useful than a home-relative one:

    * under the repository root  -> POSIX path relative to it, e.g. ``bench/results/x.json``;
    * under the operator's home  -> ``<home>/`` + POSIX remainder, e.g.
      ``<home>/<model-cache>/vendor/...`` — the *shape* of the location is public information and
      is what a reader actually wants; the account name is not;
    * anywhere else              -> ``<external>/`` + the file name alone. A path on another
      volume can encode a customer name, a ticket number or a mount point, and none of that is
      needed to read the record.

    The result is then scrubbed of the operator's account name wherever it still appears, because
    stripping a home prefix does not strip an account name that also occurs further down the
    path. POSIX separators throughout, so a record written on Windows and one written on Linux
    compare as text rather than as two spellings of the same thing.
    """
    path = pathlib.Path(p)
    root = (root or ROOT).resolve()
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover — a path the OS cannot even normalise
        resolved = path
    for base, prefix in ((root, ""), (pathlib.Path.home().resolve(), "<home>/")):
        try:
            return _scrub_account(prefix + resolved.relative_to(base).as_posix())
        except ValueError:
            continue
    return _scrub_account("<external>/" + resolved.name)


def assert_public(value: str, *, field: str = "value") -> str:
    """Return `value`, or raise [`LeakError`] naming what is wrong with it.

    Fail-closed and deliberately blunt. The checks are on the *rendered string*, not on the path
    it came from, because the string is what gets published — a screen that re-derives the answer
    from the input has assumed the conversion worked, which is the one thing it is here to doubt.

    Refuses a value that:

    * is a Windows drive-absolute path (``C:\\...``) or a UNC path (``\\\\server\\share``);
    * is a POSIX absolute path (``/home/...``);
    * contains a backslash at all — every public form this module produces is POSIX, so a
      backslash means something bypassed [`public_path`];
    * contains the current account name, wherever it appears. This is the check that catches the
      case the others miss: a path that was made relative to the *wrong* base can be relative,
      POSIX, and still spell out who ran it. [`public_path`] already scrubs this, so a value that
      trips this clause did not come from [`public_path`] — which is exactly what it is for.
    """
    if not isinstance(value, str):
        raise LeakError(f"{field} is {type(value).__name__}, not a string")
    if "\\" in value:
        raise LeakError(
            f"{field} contains a backslash and so did not come from public_path: {value!r}"
        )
    if len(value) >= 2 and value[1] == ":":
        raise LeakError(f"{field} is a drive-absolute path: {value!r}")
    if value.startswith("/"):
        raise LeakError(f"{field} is an absolute path: {value!r}")
    for token in _account_tokens():
        if token.lower() in value.lower():
            raise LeakError(f"{field} names the account that produced it: {value!r}")
    return value


def is_tracked(path: pathlib.Path, root: pathlib.Path | None = None) -> bool:
    """Is `path` a file git is tracking? Indeterminate answers count as yes.

    `git ls-files --error-unmatch` is the only question with an unambiguous answer here: exit 0
    is tracked, exit 1 is not. Anything else — git missing, not a repository, a timeout, a
    permissions error — is *not knowing*, and this function reports not-knowing as tracked.

    That asymmetry is the whole design. The cost of a false "tracked" is an operator having to
    pass `--allow-tracked` or pick another destination. The cost of a false "untracked" is
    committed evidence silently overwritten, which is the failure this module exists to prevent.
    A guard that opens when it cannot see is not a guard.
    """
    root = (root or ROOT).resolve()
    try:
        rel = pathlib.Path(path).resolve().relative_to(root)
    except (ValueError, OSError):
        return False  # outside the repository: git does not track it, by definition
    try:
        done = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel.as_posix()],
            cwd=root,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return True  # cannot tell -> treat as tracked
    if done.returncode == 0:
        return True
    if done.returncode == 1:
        return False
    return True


def resolve_out(
    out: pathlib.Path | str | None,
    default_name: str,
    *,
    allow_tracked: bool = False,
    root: pathlib.Path | None = None,
) -> pathlib.Path:
    """The destination a probe should write to, or raise [`TrackedWriteRefused`].

    `out=None` means the untracked scratch default, which is what an operator who just ran the
    probe to see its output gets. An explicit `--out` is honoured, and is checked: if it names a
    tracked file, this refuses unless the caller passed `--allow-tracked`, which is the operator
    saying in as many words that they intend to replace committed evidence.

    Creates the parent directory, because a probe that has just spent ten minutes walking SPIR-V
    should not fail on `mkdir`.
    """
    root = (root or ROOT).resolve()
    dest = pathlib.Path(out) if out is not None else (SCRATCH / default_name)
    if not dest.is_absolute():
        dest = (root / dest).resolve()
    if is_tracked(dest, root) and not allow_tracked:
        raise TrackedWriteRefused(
            f"{public_path(dest, root)} is tracked by git; writing it would replace committed "
            "evidence with a different run's numbers under the same name. Pass --allow-tracked "
            "if that is what you mean, or leave --out unset to write to "
            f"{public_path(SCRATCH, root)}/{default_name}."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest
