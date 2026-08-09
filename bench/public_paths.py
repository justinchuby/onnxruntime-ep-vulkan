#!/usr/bin/env python3
"""One seam for writing a JSON record that is safe to publish and safe to re-run.

WHY THIS EXISTS
===============
``bench/results/probe_weight_reread.py`` ended its ``main()`` with a single unconditional
``write_text(json.dumps(report, ...))`` onto ``bench/results/weight_reread_phi35.json``, and that
one line carried two defects.

**One: an implicit write to tracked evidence.** ``weight_reread_phi35.json`` is a *committed
witness*; ``docs/PERF.md`` quotes it. Running the probe to *read* a number therefore rewrote the
number and left the tree dirty, so ``git status`` came back changed from an operation the operator
believed was read-only, and the honest recovery was to remember which of your own runs to
``git checkout``. A probe that dirties the evidence it is quoted against is a probe you cannot run
while you are trying to reproduce a disagreement about that evidence — which is exactly when you
want to run it.

**Two: an absolute local path in a published record.** The same committed file carries an
``onnx_file`` value that is the resolved model path under a user's own home directory, so the
artifact names a person and a machine. ``python/verify_cleanroom.py`` already established the rule
for URLs — exactly one scrub seam, everything through it — and this is the same rule for
filesystem paths. It is not only a privacy point: a path under somebody else's home directory is
not reproducible by anybody else, so as *provenance* it is worse than useless. It looks like
provenance and is not.

**Scope note, stated rather than implied.** This module changes what *future* runs write. It does
not rewrite any committed record, and no committed witness value moves as part of landing it. The
absolute path currently in ``bench/results/weight_reread_phi35.json`` is still there; it will be
scrubbed the next time somebody deliberately regenerates that witness.

This module deliberately spells no example of a real cache path in its own text.
``ci/check_hardcoded_foundry_paths.py`` exists because literal cache paths in source go stale
silently, and a module about not publishing paths should not be the one thing that publishes one.
The planted examples live in ``bench/test_public_paths.py``, where they are fixture strings with an
invented user and an invented layout.

WHAT THIS MODULE DOES, AND WHAT IT REFUSES TO DO
================================================
* :func:`scrub_public` rewrites every absolute path it recognises. A path inside the repository
  becomes repo-relative — scrubbed *and* more useful, because a reader can open it. A path outside
  the repository becomes :data:`REDACTED`, because there is no honest shorter form of somebody
  else's home directory.
* :func:`assert_public` is the screen, run *after* the scrub. A scrub that silently missed a form
  is the failure this catches, and it is deliberately a hard error rather than a warning: a record
  that reaches disk with a leak has already been copied into whatever collected it.
* :func:`dump_public_json` writes, and **refuses to overwrite a git-tracked file** unless the
  caller passes ``allow_tracked=True`` — an argument that has to be typed by somebody who meant it,
  not a default that can be forgotten.
* :func:`untracked_default` gives probes a place to write that git already ignores, so "explicit
  ``--out`` or a safe default" does not degrade into "explicit ``--out`` or nothing".

WHAT IT IS NOT
==============
It is not a general secret scanner. It knows about filesystem paths and says so. URLs and
credentials are ``python/verify_cleanroom.py``'s seam and stay there; two half-scanners over one
record would be a second register of the same fact.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

__all__ = [
    "PublicPathError",
    "REDACTED",
    "assert_public",
    "dump_public_json",
    "is_git_tracked",
    "repo_root",
    "scrub_public",
    "untracked_default",
]

#: What an unrepresentable path becomes. A fixed token rather than a truncated path: half of a home
#: directory is still a username, and a reader who sees this token knows exactly what happened,
#: where a truncated path invites them to guess.
REDACTED = "<redacted-absolute-path>"

#: Absolute-path forms this module recognises, assembled from named parts.
#:
#: Deliberately *over*-broad rather than exact: the cost of a false positive is a redacted string
#: in a record, and the cost of a false negative is a published home directory.
#:
#: Assembled from named constants rather than written as one ``re.VERBOSE`` block. A verbose
#: pattern documents itself with ``#`` comments, and a trailing backslash inside one of those
#: comments silently swallows the following alternative — a pattern whose comments can eat its own
#: branches is a pattern that documents itself into a defect.
_DRIVE = r"[A-Za-z]:[\\/]"
_UNC = r"\\\\[^\\/\s\"]+[\\/]"

#: POSIX roots, as an explicit list rather than "any string starting with ``/``".
#:
#: The blanket rule is tempting and wrong **in this repository specifically**: an ONNX node name is
#: spelled exactly like a POSIX absolute path (``/model/layers.0/n0``) and these records are full
#: of them. A scrub that redacted every leading-slash string would shred the data the records exist
#: to carry.
#:
#: So the list is explicit and deliberately long. ``github`` is here because ``/github/home`` is the
#: container HOME on GitHub Actions; ``workspaces``, ``tmp``, ``var``, ``opt``, ``srv``, ``data``,
#: ``private`` and ``Volumes`` are roots a CI job or a developer box actually resolves a cache into.
#: The residual risk is stated rather than papered over: a POSIX root outside this list is not
#: recognised, and the fix is to add it here with a planted example in
#: ``bench/test_public_paths.py``.
_POSIX_ROOTS = (
    "home",
    "Users",
    "users",
    "root",
    "mnt",
    "media",
    "tmp",
    "var",
    "opt",
    "srv",
    "data",
    "private",
    "Volumes",
    "workspaces",
    "github",
)
_POSIX_ROOT = "/(?:" + "|".join(_POSIX_ROOTS) + ")/"
_TILDE = r"~[\\/]"

#: With a no-whitespace tail: used to find where an embedded path ends.
_ABSOLUTE = re.compile("(?:" + "|".join((_DRIVE, _UNC, _POSIX_ROOT, _TILDE)) + r")[^\s\"']*")

#: The same prefixes without the tail. Used to decide whether a *whole* string is a path, which is
#: how a path containing a space is handled correctly — an entirely ordinary Windows profile has
#: one. ``_ABSOLUTE``'s tail stops at the first space, so matching on it alone would replace the
#: head of such a path and leave the rest of the username in the record.
_ABSOLUTE_PREFIX = re.compile("(?:" + "|".join((_DRIVE, _UNC, _POSIX_ROOT, _TILDE)) + ")")

#: The same prefixes, but only where a path can actually *begin*: at the start of the string, or
#: after one of a few delimiters.
#:
#: This is the correction that keeps :data:`_POSIX_ROOTS` from eating live data. An ONNX node name
#: is a slash-joined path of *module attribute names*, and this model family really does contain
#: submodules whose names collide with the roots above — a bare ``search`` for ``/data/`` or
#: ``/var/`` would fire in the middle of a node name and redact a field that is not a path at all.
#: Anchoring means an interior segment can never start a match, while a genuine embedded path (in
#: prose, after a space or a quote or an ``=``) still does.
_ABSOLUTE_EMBEDDED = re.compile(
    "(?:^|(?<=[\\s\"'=(\\[,]))(?:" + "|".join((_DRIVE, _UNC, _POSIX_ROOT, _TILDE)) + ")"
)


class PublicPathError(RuntimeError):
    """A record that would have published an absolute local path, or a write that would have
    clobbered tracked evidence."""


def repo_root(start: Path | None = None) -> Path:
    """The repository root, found by walking up for ``.git``.

    Not ``git rev-parse``: this is called from inside a scrub that must work when git is absent (a
    wheel, a clean-room extraction) and must never itself shell out on that path.

    ``.git`` is tested with ``exists()`` rather than ``is_dir()`` on purpose. In a **worktree** it
    is a *file* holding a ``gitdir:`` pointer, and a check that insisted on a directory would walk
    straight past the root of every worktree in this project.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    # No `.git` — a source extraction rather than a checkout. `bench/` sits directly under the root.
    return Path(__file__).resolve().parent.parent


def _relativise(text: str, root: Path) -> str | None:
    """``text`` as a repo-relative POSIX path, or ``None`` if it is not inside the repo.

    Refuses anything not already absolute. ``~/x`` is not absolute, and resolving it would quietly
    interpret it against the *current working directory* — which, when the cwd happens to be inside
    the repository, relativises an unexpanded home reference back to itself and hands the tilde
    straight through the scrub.
    """
    try:
        candidate = Path(text)
    except (ValueError, OSError):
        return None
    if not candidate.is_absolute():
        return None
    try:
        resolved = candidate.resolve(strict=False)
    except (ValueError, OSError):
        return None
    try:
        rel = resolved.relative_to(root.resolve(strict=False))
    except ValueError:
        return None
    return rel.as_posix()


def _scrub_text(text: str, root: Path) -> str:
    """Every absolute path in one string, rewritten. Idempotent.

    Three cases, in order, and the first is the one that matters most in practice:

    1. **The whole string is a path.** This is what a record field like ``onnx_file`` actually
       holds. Handled first and as a unit, so a path containing a space is relativised or redacted
       whole.
    2. **A path is embedded in prose and is inside the repository.** Relativised in place; the
       surrounding text is left alone.
    3. **A path is embedded in prose and is not inside the repository.** Redacted from the match to
       the end of the line. Deliberately over-broad: a filesystem path may contain spaces, so there
       is no way to know where it ends, and the alternative — stopping at the first space — is what
       leaves half a username in the record.
    """
    if not text:
        return text
    lines = text.split("\n")
    if len(lines) > 1:
        return "\n".join(_scrub_text(line, root) for line in lines)

    stripped = text.strip()
    if stripped and _ABSOLUTE_PREFIX.match(stripped):
        rel = _relativise(stripped, root)
        replacement = rel if rel is not None else REDACTED
        return text.replace(stripped, replacement, 1)

    match = _ABSOLUTE_EMBEDDED.search(text)
    if match is None:
        return text
    head, tail = text[: match.start()], text[match.start():]
    rel = _relativise(tail.strip(), root)
    if rel is not None:
        return head + rel
    narrow = _ABSOLUTE.match(tail)
    if narrow is not None:
        rel = _relativise(narrow.group(0), root)
        if rel is not None:
            return head + rel + _scrub_text(tail[narrow.end():], root)
    return head + REDACTED


def scrub_public(obj, *, root: Path | None = None):
    """Rewrite every absolute path in a JSON-shaped object. **Keys as well as values.**

    Keys matter: a record that maps a resolved model path to a sub-record leaks exactly as much as
    one that stores the same string in a value, and a scrub that walked only values would be the
    kind of screen that passes while the thing it screens for is in the file.
    """
    root = root or repo_root()
    if isinstance(obj, str):
        return _scrub_text(obj, root)
    if isinstance(obj, dict):
        return {scrub_public(k, root=root): scrub_public(v, root=root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_public(v, root=root) for v in obj]
    return obj


def assert_public(obj, *, where: str = "record", root: Path | None = None) -> None:
    """Refuse a record that still contains an absolute path after scrubbing.

    Run *after* :func:`scrub_public`, never instead of it. The scrub is the fix and this is the
    screen; a screen nobody has seen fire is not a screen, which is why
    ``bench/test_public_paths.py`` plants each recognised form and requires this to raise.
    """
    root = root or repo_root()
    leaks: list[str] = []

    def walk(node, path: str) -> None:
        if isinstance(node, str):
            for match in _ABSOLUTE_EMBEDDED.finditer(node):
                leaks.append(f"{path}: {node[match.start(): match.start() + 60]!r}")
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(key, f"{path}.<key>")
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(obj, where)
    if leaks:
        raise PublicPathError(
            f"{where} would publish {len(leaks)} absolute local path(s), which name a machine and "
            "a person and are reproducible by nobody else:\n  " + "\n  ".join(sorted(leaks))
        )


def is_git_tracked(path: Path, *, root: Path | None = None) -> bool:
    """Is ``path`` a file git already tracks?

    Answered by asking git, because the alternative — matching ``.gitignore`` by hand — is a second
    implementation of a rule that already has one, and the two would disagree eventually.

    **Fails safe in every direction.** This guard exists to prevent a write, so the answer it gives
    when it cannot tell must be the one that refuses. Only ``git ls-files --error-unmatch`` exiting
    **1** means "definitely untracked"; that is the documented exit for a path git knows about and
    does not track. Exit 128 (not a repository, corrupt index, permission denied), any other exit,
    a timeout, or a missing binary all mean *the question was not answered*, and are reported as
    tracked. Returning ``done.returncode == 0`` would quietly turn every one of those into
    permission to overwrite evidence.

    ``path`` is resolved before it is handed to git, because ``git -C <root>`` resolves a relative
    path against ``root`` while :func:`dump_public_json`'s write resolves it against the process
    working directory. Those two disagreeing is not theoretical: running a probe from
    ``bench/results/`` with a bare ``--out`` filename makes the guard ask about a file at the repo
    root — untracked — while the write lands on the committed witness beside the script. Resolving
    also collapses symlinks and Windows directory junctions, so a destination reached through one
    is checked as the file it really is rather than as the name it was reached by.

    A destination outside ``root`` is answered directly as untracked rather than by asking git,
    which returns 128 for it and would otherwise trip the fail-safe on every ordinary scratch path.
    """
    root = root or repo_root()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", str(resolved)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if done.returncode == 0:
        return True
    if done.returncode == 1:
        return False
    return True


def untracked_default(name: str, *, root: Path | None = None) -> Path:
    """A default output path git already ignores: ``bench/_scratch/<name>``.

    ``bench/.gitignore`` has ignored ``_scratch/`` since ``bench/devices.py`` needed somewhere to
    put device-probe dumps, so this reuses a decision rather than making a second one. The
    directory is created here so a caller never has to decide whether creating it is its job.
    """
    root = root or repo_root()
    out = root / "bench" / "_scratch" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def dump_public_json(
    obj,
    path: Path,
    *,
    allow_tracked: bool = False,
    root: Path | None = None,
) -> Path:
    """Scrub, screen, then write — and refuse to clobber tracked evidence by accident.

    Returns the path written, so a caller can print it without restating it.

    ``allow_tracked`` exists because regenerating a committed witness is a legitimate act; it is an
    argument rather than a default so that it appears in the command line of the run that did it,
    which is the only place a reviewer will look.
    """
    root = root or repo_root()
    # Resolve ONCE and use the same object for the guard and the write, so the two cannot be
    # answering about different files.
    path = Path(path).resolve()
    if not allow_tracked and is_git_tracked(path, root=root):
        raise PublicPathError(
            f"refusing to write {path}: git tracks it, so this is committed evidence and an "
            "implicit write would dirty the tree of anyone who ran this probe to READ a number. "
            "Pass an explicit --out, or --allow-tracked if regenerating the witness is the intent "
            "(and then say so in the commit message)."
        )
    scrubbed = scrub_public(obj, root=root)
    assert_public(scrubbed, where=str(path), root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scrubbed, indent=2, sort_keys=True), encoding="utf-8")
    return path
