#!/usr/bin/env python3
"""One seam for writing a JSON record that is safe to publish, and safe to re-run.

WHY THIS EXISTS
===============
`bench/results/probe_weight_reread.py` ended its `main()` with

    out = ROOT / "bench" / "results" / "weight_reread_phi35.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

and that is two defects wearing one line.

**One: an implicit write to tracked evidence.** `weight_reread_phi35.json` is a *committed
witness* — `docs/PERF.md` §22 and §25 quote it, and `ci/check_ledger_census.py`'s whole argument
is that a witness nobody wrote down moving is indistinguishable from an accident. Running the
probe to *read* a number silently rewrote the number, so `git status` came back dirty from an
operation the operator believed was read-only, and the honest way to recover was to remember
which of your own runs to `git checkout`. A probe that dirties the evidence it is quoted against
is a probe you cannot run while you are trying to reproduce a disagreement about that evidence —
which is exactly when you want to run it.

**Two: an absolute local path in a published record.** The same file carries an ``onnx_file`` value
that is the resolved model path under the operator's own home directory, so the artifact names a
person and a machine. `python/verify_cleanroom.py` already established the rule for URLs — there
is exactly one scrub seam and everything goes through it — and this is the same rule for
filesystem paths. It is not only a privacy point: a path under a home directory is not
reproducible by anybody else, so as *provenance* it is worse than useless. It looks like
provenance and is not.

This module deliberately spells no example of such a path in its own text.
`ci/check_hardcoded_foundry_paths.py` exists because literal cache paths in source go stale
silently, and a module about not publishing paths should not be the one thing that publishes one.
The planted examples live in `bench/test_public_paths.py`, where they are test data with an
invented user and no real cache layout.

WHAT THIS MODULE DOES, AND WHAT IT REFUSES TO DO
================================================
* :func:`scrub_public` rewrites any absolute path it recognises. A path inside the repository
  becomes repo-relative — which is both scrubbed *and* more useful, because a reader can open it.
  A path outside the repository is replaced by :data:`REDACTED`, because there is no honest
  shorter form of somebody else's home directory.
* :func:`assert_public` is the screen, run *after* the scrub. A scrub that silently missed a form
  is the failure mode this catches, and it is deliberately a hard error rather than a warning: a
  record that reaches disk with a leak has already been copied into whatever collected it.
* :func:`dump_public_json` writes, and **refuses to overwrite a git-tracked file** unless the
  caller passes ``allow_tracked=True``. Not a default that can be forgotten — an argument that
  has to be typed, in the same call, by somebody who meant it.
* :func:`untracked_default` gives probes a place to write that is already git-ignored, so
  "explicit ``--out`` or a safe default" does not degrade into "explicit ``--out`` or nothing".

WHAT IT IS NOT
==============
It is not a general secret scanner. It knows about filesystem paths, and it says so. URLs and
credentials are `python/verify_cleanroom.py`'s seam and stay there; two half-scanners over one
record would be the two-registers defect `ci/check_ledger_census.py` names.
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

#: What an unrepresentable path becomes. A fixed token rather than a partial path: half of a home
#: directory is still a username, and a reader who sees `<redacted-absolute-path>` knows exactly
#: what happened, where a truncated path invites them to guess.
REDACTED = "<redacted-absolute-path>"

#: Absolute-path forms this module recognises. Deliberately *over*-broad rather than exact: the
#: cost of a false positive is a redacted string in a record, and the cost of a false negative is
#: a published home directory.
#:
#: * ``C:\...`` / ``C:/...`` — a Windows drive path, either separator.
#: * ``\\server\share`` — a UNC path.
#: * ``/home/x``, ``/Users/x``, ``/root/``, ``/mnt/x``, ``/media/x`` — POSIX user-ish roots.
#: * ``~/...`` — an unexpanded home reference, which is not a leak but is not reproducible either.
#:
#: Assembled from named parts rather than written as one ``re.VERBOSE`` block with inline
#: comments. The first draft *was* verbose-with-comments, and a trailing backslash inside one of
#: those comments silently swallowed the following alternative — the POSIX branch never matched,
#: and only the planted positive controls in `bench/test_public_paths.py` caught it. A pattern
#: whose comments can eat its own alternatives is a pattern that documents itself into a defect.
_DRIVE = r"[A-Za-z]:[\\/]"
_UNC = r"\\\\[^\\/\s\"]+[\\/]"
#: POSIX roots, as an explicit list rather than "any string starting with `/`".
#:
#: The blanket rule is tempting and it is wrong **in this repository specifically**: an ONNX node
#: name is spelled exactly like a POSIX absolute path (`/model/layers.0/n0`), and these records are
#: full of them. A scrub that redacted every leading-slash string would shred the very data the
#: records exist to carry, which is the failure mode `test_a_clean_record_is_left_exactly_alone`
#: and `test_onnx_node_names_are_not_mistaken_for_paths` guard against.
#:
#: So the list is explicit and deliberately long. `github` is here because `/github/home` is the
#: container HOME on GitHub Actions; `workspaces`, `tmp`, `var`, `opt`, `srv`, `data`, `private`
#: and `Volumes` are the roots a CI job or a developer box actually resolves a cache into. The
#: residual risk is stated rather than papered over: a POSIX root outside this list is not
#: recognised, and the fix for that is to add it here with a planted example in
#: `bench/test_public_paths.py`.
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
_ABSOLUTE = re.compile("(?:" + "|".join((_DRIVE, _UNC, _POSIX_ROOT, _TILDE)) + r")[^\s\"']*")

#: The same prefixes without the no-whitespace tail. Used to decide whether a *whole* string is a
#: path, which is how a path containing a space (`C:\Users\Jane Smith\...` — an entirely ordinary
#: Windows profile) is handled correctly. `_ABSOLUTE`'s tail stops at the first space, so matching
#: on it alone would replace `C:\Users\Jane ` and leave `Smith\.foundry\...` in the record.
_ABSOLUTE_PREFIX = re.compile("(?:" + "|".join((_DRIVE, _UNC, _POSIX_ROOT, _TILDE)) + ")")


def repo_root(start: Path | None = None) -> Path:
    """The repository root, found by walking up for ``.git``.

    Not ``git rev-parse``: this is called from inside a scrub that must work when git is absent
    (a wheel, a clean-room extraction) and must never itself shell out on the hot path.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    # No `.git` — a source extraction rather than a checkout. `bench/` is directly under the root.
    return Path(__file__).resolve().parent.parent


def _relativise(text: str, root: Path) -> str | None:
    """``text`` as a repo-relative POSIX path, or ``None`` if it is not inside the repo.

    Refuses anything that is not already absolute. `~/x` is not absolute, and resolving it would
    quietly interpret it against the *current working directory* — which, when the cwd happens to
    be inside the repository, relativises `~/.foundry/model.onnx` back to itself and hands the
    tilde straight through the scrub. That is not hypothetical; it is what the first draft did,
    and the planted `unexpanded home` control is what caught it.
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
       holds. Handled first and as a unit, so a path containing a space — ``C:\\Users\\Jane
       Smith\\...``, an entirely ordinary Windows profile — is relativised or redacted whole.
    2. **A path is embedded in prose, and it is inside the repository.** Relativised in place; the
       surrounding text is left alone.
    3. **A path is embedded in prose and is not inside the repository.** Redacted from the match to
       the end of the line. Deliberately over-broad: a filesystem path may contain spaces, so
       there is no way to know where it ends, and the alternative — stopping at the first space —
       is what leaves half a username in the record.
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

    match = _ABSOLUTE_PREFIX.search(text)
    if match is None:
        return text
    head, tail = text[: match.start()], text[match.start() :]
    rel = _relativise(tail.strip(), root)
    if rel is not None:
        return head + rel
    narrow = _ABSOLUTE.match(tail)
    if narrow is not None:
        rel = _relativise(narrow.group(0), root)
        if rel is not None:
            return head + rel + _scrub_text(tail[narrow.end() :], root)
    return head + REDACTED


def scrub_public(obj, *, root: Path | None = None):
    """Rewrite every absolute path in a JSON-shaped object. **Keys as well as values.**

    Keys matter: a record that maps ``{"C:\\\\Users\\\\x\\\\model.onnx": {...}}`` leaks exactly as
    much as one that stores the same string in a value, and a scrub that walked only values would
    be the kind of screen that passes while the thing it screens for is in the file.
    """
    root = root or repo_root()
    if isinstance(obj, str):
        return _scrub_text(obj, root)
    if isinstance(obj, dict):
        return {
            scrub_public(k, root=root): scrub_public(v, root=root) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [scrub_public(v, root=root) for v in obj]
    return obj


class PublicPathError(RuntimeError):
    """A record that would have published an absolute local path."""


def assert_public(obj, *, where: str = "record", root: Path | None = None) -> None:
    """Refuse a record that still contains an absolute path after scrubbing.

    Run *after* :func:`scrub_public`, never instead of it. The scrub is the fix and this is the
    screen, and a screen that has never been seen firing is not a screen — see
    ``test_public_paths.py``, which plants each recognised form and asserts this raises.
    """
    root = root or repo_root()
    leaks: list[str] = []

    def walk(node, path: str) -> None:
        if isinstance(node, str):
            for match in _ABSOLUTE_PREFIX.finditer(node):
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
            f"{where} would publish {len(leaks)} absolute local path(s), which name a machine "
            "and a person and are reproducible by nobody else:\n  " + "\n  ".join(sorted(leaks))
        )


def is_git_tracked(path: Path, *, root: Path | None = None) -> bool:
    """Is ``path`` a file git already tracks?

    Answered by asking git, because the alternative — matching ``.gitignore`` by hand — is a
    second implementation of a rule that already has one, and the two would disagree eventually.

    **Fails safe in every direction.** This guard exists to prevent a write, so the answer it gives
    when it cannot tell must be the one that refuses. Only `git ls-files --error-unmatch` exiting
    **1** means "definitely untracked"; that is the documented exit for a path git knows about and
    does not track. Exit 128 (not a repository, corrupt index, permission denied), any other exit,
    a timeout, or a missing binary all mean *the question was not answered*, and are reported as
    tracked. The first draft of this function returned `done.returncode == 0`, which quietly turned
    every one of those into permission to overwrite evidence.

    ``path`` is resolved before it is handed to git, because ``git -C <root>`` resolves a relative
    path against ``root`` while :func:`dump_public_json`'s write resolves it against the process
    working directory. Those two disagreeing is not theoretical: running the probe from
    ``bench/results/`` with ``--out weight_reread_phi35.json`` made the guard ask about
    ``<root>/weight_reread_phi35.json`` — untracked — while the write landed on the committed
    witness.

    A destination outside ``root`` is answered directly as untracked rather than by asking git,
    which returns 128 for it and would otherwise trip the fail-safe on every ordinary scratch path.
    """
    root = root or repo_root()
    resolved = Path(path).resolve()
    # A path outside the repository cannot be in that repository's index, and asking git about one
    # returns 128 ("is outside repository") — which the fail-safe below would read as "cannot tell"
    # and refuse. Answering it here keeps the fail-safe meaning "git could not answer a question it
    # should have been able to answer" rather than "the destination is somewhere else".
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
    """A default output path that git already ignores: ``bench/_scratch/<name>``.

    ``bench/.gitignore`` has ignored ``_scratch/`` since `bench/devices.py` needed somewhere to
    put vulkaninfo dumps, so this reuses a decision rather than making a second one. The directory
    is created here so a caller never has to decide whether creating it is its job.
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

    ``allow_tracked`` exists because regenerating a committed witness is a legitimate act; it is
    an argument rather than a default so that it appears in the command line of the run that did
    it, which is the only place a reviewer will look.
    """
    root = root or repo_root()
    # Resolve ONCE, and use the same object for the guard and the write. `git -C <root>` resolves
    # a relative path against `root` while `write_text` resolves it against the process working
    # directory; letting those two disagree is how `--out weight_reread_phi35.json` run from
    # `bench/results/` passed the tracked check and then overwrote the committed witness.
    path = Path(path).resolve()
    if not allow_tracked and is_git_tracked(path, root=root):
        raise PublicPathError(
            f"refusing to write {path}: git tracks it, so this is committed evidence and an "
            "implicit write would dirty the tree of anyone who ran this probe to READ a number. "
            "Pass an explicit --out, or --allow-tracked if regenerating the witness is the "
            "intent (and then say so in the commit message)."
        )
    scrubbed = scrub_public(obj, root=root)
    assert_public(scrubbed, where=str(path), root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scrubbed, indent=2, sort_keys=True), encoding="utf-8")
    return path
