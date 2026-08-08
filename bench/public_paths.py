"""Machine-independent paths for artifacts that get committed. Owner: Tank (issue #69).

WHY THIS EXISTS
---------------
Link rejected `62d85f3` because eight committed evidence JSONs carried
``C:\\Users\\<username>\\...``: the operator's account name, the name of the
worktree the run happened in, and the absolute location of a multi-gigabyte
model cache. None of it is a measurement. All of it is a fact about one laptop,
published in a public repository, in files whose whole purpose is to be read by
someone who does not have that laptop.

The defect was not that somebody forgot to redact. It was that ``str(path)``
was called at eleven serialisation sites and nothing between those calls and
``git add`` could tell the difference between a path that means something and a
path that means "this ran on my machine". So this module is not a redaction
pass. It is the **only** way this suite is allowed to put a path into a
committed artifact, and :func:`dump_public_json` refuses to write a payload
that still contains one.

WHAT IS KEPT AND WHAT IS DROPPED
--------------------------------
A path in an evidence record answers one of two questions, and they have
different answers here:

* *Which file did you measure?* — semantic, kept. ``mobilenetv2-12.onnx`` under
  the model cache is rewritten ``<model-cache>/mobilenetv2_12/mobilenetv2-12.onnx``:
  the root is named by role, the part below it is preserved verbatim, and a
  reader on another machine can locate the same file from their own cache root.
  The bytes are identified by ``sha256`` regardless, which is the identity that
  actually survives a move.
* *Where did this process happen to put a scratch file?* — non-semantic, and
  it is dropped to the same rooted form rather than deleted, because a record
  that silently omits a field reads the same as a record whose field was never
  populated. ``<repo>/bench/results/_cuda69/scratch/rep0/out0.npy`` says what
  the file was for; the drive letter and the account name said nothing.

Anything under no known root becomes ``<elsewhere>/<basename>``. That is a
deliberate loss: a path this suite cannot attribute to a role is a path it
cannot promise is machine-independent, so it keeps only the leaf and says so.

WHY THE SCREEN IS STRUCTURAL AND NOT A LIST OF NAMES
----------------------------------------------------
:data:`LEAK_PATTERNS` matches ``C:\\Users\\<anything>``, ``/home/<anything>``
and ``/Users/<anything>`` by shape, never a specific account by name. A screen
keyed to the current operator's account passes for everyone else and would have
to be edited by each new contributor before it protected them — which is the
same failure as no screen at all, arriving one hire later. The account names
this process can discover are added on top of the structural patterns, never
instead of them.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent

#: Token for a path this module could not attribute to any declared root.
ELSEWHERE = "<elsewhere>"

#: Substituted for an account name that appears outside a path (in an error
#: message, a traceback, a captured log line).
REDACTED_USER = "<user>"

#: A value this module has already rewritten. Matching it lets `public_path` be
#: applied twice without damage, which matters because sanitisation happens both
#: where a field is built and again at the serialisation boundary — belt and
#: braces, deliberately, so neither one alone is load-bearing.
_ROOTED = re.compile(r"^<[a-z-]+>(/|$)")

#: The one key whose contents are **runtime state, not evidence**: absolute paths that
#: one process hands to another inside a single run and that must never reach a file.
#:
#: WHY A RESERVED KEY AND NOT A NAMING CONVENTION.  Before this existed, a worker put an
#: absolute ``out0.npy`` into its record, the record went through `dump_public_json`, and
#: the path came back rooted — ``<repo>/.../out0.npy``. The parent then asked
#: ``Path(that).is_file()``, got ``False``, and reported "reference output file missing",
#: which its caller read as "nothing was disqualified". A wrong answer published as a
#: fast one. Both halves of that were path handling: the internal channel and the public
#: artifact were the same field, so sanitising one necessarily broke the other.
#:
#: They are separated here. Anything a later process must *open* travels under this key
#: and is dropped by :func:`public_payload` before serialisation; anything a later reader
#: must *understand* travels as a rooted path plus a relative handle, which
#: :func:`resolve_public_path` and :func:`contained_child` turn back into a real file with
#: a containment check. A record therefore cannot be both readable-by-the-parent and
#: leaky, and it cannot be silently neither.
RUNTIME_ONLY_KEY = "_runtime"


class PathLeak(ValueError):
    """A payload bound for a committed artifact still names a machine."""


def _model_cache_root() -> Path:
    """The bench model cache, resolved the same way ``bench_models.cache_root`` does.

    Duplicated rather than imported: this module is a dependency of
    ``bench_models``, and a cycle would make the sanitiser importable only when
    the thing it sanitises already imported cleanly.
    """
    override = os.environ.get("ONNXRUNTIME_EP_VULKAN_BENCH_MODEL_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "onnxruntime-ep-vulkan" / "bench-models"


def public_roots() -> "list[tuple[str, Path]]":
    """Declared roots, longest-path first so a nested root wins over its parent.

    Order is load-bearing: the model cache and the temp directory both live
    under ``$HOME`` on the machines this runs on, and rewriting them as
    ``<home>/...`` would keep the account name out but throw away the role,
    which is the part a reader needs.
    """
    roots = [
        ("<repo>", REPO),
        ("<model-cache>", _model_cache_root()),
        # The Windows AI Foundry cache. `bench_models._resolve_foundry` resolves
        # Phi-3.5 out of it by identity, so "which cache" is a fact about the
        # provenance of the weights, not about the operator, and collapsing it
        # into `<home>` would throw that away.
        ("<foundry-cache>", Path.home() / ".foundry" / "cache" / "models"),
        ("<tmp>", Path(tempfile.gettempdir())),
        ("<home>", Path.home()),
    ]
    resolved: "list[tuple[str, Path]]" = []
    for token, path in roots:
        try:
            resolved.append((token, path.resolve()))
        except OSError:  # pragma: no cover - a root that cannot be stat'd is simply not a root
            continue
    # Longest first, so `<model-cache>` beats `<home>` for a path under both.
    resolved.sort(key=lambda kv: len(str(kv[1])), reverse=True)
    return resolved


def _relative_to(child: Path, root: Path) -> "PurePosixPath | None":
    """Is *child* under *root*? Decided by path parts, never by string prefix.

    ``str.startswith`` says ``C:\\Users\\ann-backup`` is under
    ``C:\\Users\\ann``. Comparing resolved parts does not, and the false
    positive would be a path rewritten under the wrong root, which is worse
    than one left alone because it looks sanitised.
    """
    c = [p.lower() for p in child.parts] if os.name == "nt" else list(child.parts)
    r = [p.lower() for p in root.parts] if os.name == "nt" else list(root.parts)
    if len(c) < len(r) or c[: len(r)] != r:
        return None
    return PurePosixPath(*child.parts[len(r):]) if len(c) > len(r) else PurePosixPath()


def public_path(value, *, roots=None, scrub_remainder: bool = True) -> str:
    """Rewrite one path as ``<root>/relative/posix/path``.

    Forward slashes on every platform: a record whose separators depend on the
    operating system that produced it is a machine fingerprint one level down,
    and the same run on Linux would diff against itself.

    ``scrub_remainder=False`` is used by :func:`scrub_text`, which calls this to
    re-root paths it found inside prose and then finishes the remaining shapes
    itself. It exists to break the mutual recursion, not as a way to publish
    less.
    """
    if value is None or value == "":
        return ""
    text = str(value)
    # Idempotence. An already-rooted value has no drive and no leading slash, so
    # `Path.resolve()` would re-anchor it at the current directory and produce
    # `<repo>/<model-cache>/...`. Serialisers legitimately run this twice (a
    # dataclass field rooted at construction, then rooted again by `to_dict`).
    if _ROOTED.match(text):
        return text.replace("\\", "/")
    raw = Path(text)
    try:
        resolved = raw.resolve()
    except OSError:  # pragma: no cover - unresolvable path, judged as written
        resolved = raw
    tail = (lambda s: scrub_text(s)) if scrub_remainder else (lambda s: s)
    candidates = public_roots() if roots is None else list(roots)
    if roots is None:
        foreign = _foreign_checkout_root(resolved)
        if foreign is not None:
            # Inserted as a root and re-sorted rather than tried as a fallback:
            # it must beat `<home>` (an artifact from another checkout under the
            # same home) and lose to `<model-cache>` (which the rule excludes
            # anyway). Specificity is length, exactly as for declared roots.
            #
            # Rooted at `<foreign-repo>`, never `<repo>`: this path is inside a checkout
            # that is not the one being served. See :data:`FOREIGN_REPO`.
            candidates = sorted(candidates + [(FOREIGN_REPO, foreign)],
                                key=lambda kv: len(str(kv[1])), reverse=True)
    for token, root in candidates:
        rel = _relative_to(resolved, root)
        if rel is None:
            continue
        if str(rel) == ".":
            return token
        # The part BELOW a declared root can still name the operator — pytest's
        # `pytest-of-<user>` under the temp directory is the case that found
        # this. Naming the root is not sufficient; the remainder is scrubbed too.
        return f"{token}/{tail(rel.as_posix())}"
    return f"{ELSEWHERE}/{tail(resolved.name)}"


#: Names that are DEVICES on Windows in every directory, with or without a suffix.
#: ``base/NUL`` is not a file under *base*; it is the null device, and it opens.
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _malformed_component(part: str) -> bool:
    """Is *part* a name this module refuses to turn into a path component?

    Every case here is one where the *filesystem* would have accepted something other
    than what the handle literally says, which is the property that makes a handle
    unreadable as evidence:

    * ``:`` anywhere — a drive spec (``C:evil.npy``) or an NTFS alternate data stream
      (``out0.npy:hidden``). Checked in EVERY component, not only the first: joining
      ``sub`` with ``C:evil.npy`` on Windows *discards* ``sub`` and yields a
      drive-relative path, so a colon buried in the tail is the same escape as one at
      the head with the check looking the other way.
    * a trailing ``.`` or space — Windows strips both before it opens the file, so
      ``out0.npy.`` and ``out0.npy `` name ``out0.npy`` while reading as different
      handles. Repairing them silently is how two records can disagree about which
      file they mean and both look right.
    * NUL and other C0 control characters — ``Path.resolve`` raises ``ValueError`` on an
      embedded NUL rather than ``OSError``, which is why this is caught here and again
      around the resolution below.
    * a Windows device name, with or without a suffix.
    """
    if not part or part in (".", ".."):
        return True
    if ":" in part or any(ch <= "\x1f" for ch in part):
        return True
    if part[-1] in ". ":
        return True
    return part.split(".")[0].lower() in _WINDOWS_DEVICE_NAMES


def contained_child(base, rel) -> "Path | None":
    """``base/rel`` as a real path, or ``None`` if *rel* does not stay inside *base*.

    The one operation allowed to turn a handle that came out of another process into a
    file this one opens. ``rel`` is a *relative POSIX name* by contract — that is what
    :func:`cuda_competition.relative_handle` writes — so everything that is not one is
    REFUSED RATHER THAN REPAIRED, and the distinction is the point of the function.

    Refused: a non-string; an empty or absent ``base``; an absolute, rooted (``<repo>/…``),
    UNC, device or drive-relative path; a backslash anywhere (a POSIX filename may contain
    one, so rewriting it to a separator renames the file this handle points at); an empty,
    ``.`` or ``..`` component; and every shape :func:`_malformed_component` lists — colon
    in ANY component, trailing dot or space, embedded NUL, a Windows device name.

    Two properties this function is required to have and did not:

    * **Total.** It returns ``None``; it does not raise. ``Path`` raises ``ValueError``
      for an embedded NUL and ``TypeError`` for a non-path object, and a caller that
      wrote ``if contained_child(...) is None`` would have been bypassed by the
      traceback rather than told no.
    * **No repair.** Every check above is a case where the previous version handed back
      a *different* file from the one the handle named — ``sub/C:evil.npy`` collapsing
      to a drive-relative path, ``out0.npy.`` opening ``out0.npy``. A resolver that
      quietly fixes a malformed name is indistinguishable from one reading a well-formed
      one, and the record it validated is then evidence about a file nobody chose.

    The syntactic checks are not the guarantee — ``resolve()`` followed by
    :func:`_relative_to` is. A symlink or junction inside *base* pointing at
    ``C:\\Windows`` passes every string test and fails this one, which is why containment
    is decided on the resolved parts and not on the text.

    Refusing is the whole point. A caller that gets ``None`` has learned that it cannot
    read the file, which is a different fact from "the file said nothing" — and the
    second is what a silently-empty result would have told it.
    """
    if not isinstance(rel, str):
        # A handle arrives from JSON or from `relative_handle`, and both produce `str`.
        # Anything else was built by a caller that is not honouring the contract; coercing
        # it with `str()` would turn a `Path`, an `int` or a `None` into a plausible name.
        return None
    if base is None or base == "" or isinstance(base, (bool, int, float)):
        return None
    text = rel
    if not text or "\\" in text or "\x00" in text:
        return None
    if text.startswith("/") or _ROOTED.match(text):
        return None
    parts = text.split("/")
    if any(_malformed_component(p) for p in parts):
        return None
    try:
        root = Path(base).resolve()
        candidate = (root / Path(*parts)).resolve()
    except (OSError, ValueError, TypeError):
        # Unresolvable, malformed or not a path at all. Refused, not guessed, and not
        # raised: every caller of this function records the refusal beside the handle.
        return None
    inside = _relative_to(candidate, root)
    if inside is None or str(inside) in ("", "."):
        # `""`/`"."` is *base* itself, reachable only through a link. A handle names a
        # file inside the directory; the directory is not a file inside itself.
        return None
    return candidate


def resolve_public_path(value, *, roots=None) -> "Path | None":
    """Inverse of :func:`public_path`: ``<repo>/a/b`` → this machine's ``a/b``.

    This is what makes a rooted path *evidence a second machine can act on* rather than
    a decorative string. ``--reanalyse`` re-derives equivalence from a committed record
    whose paths were rewritten at write time; without an inverse it would be reading
    ``<repo>/…`` off disk, finding nothing, and reporting "missing" for a file that is
    present under the reader's own checkout.

    ``None`` — refusal, never a guess — for a value that is not rooted, for a value
    rooted at a token this module cannot map back to a directory (:data:`ELSEWHERE` and
    :data:`FOREIGN_REPO` name places that are, by construction, not here), and for a
    remainder that escapes its root. An absolute path is refused too: a *published*
    record is not allowed to contain one, so accepting it here would be the sanitiser
    and the resolver disagreeing about what a record may say.
    """
    text = str(value or "").replace("\\", "/")
    if not _ROOTED.match(text):
        return None
    token, _, rest = text.partition("/")
    if token in (ELSEWHERE, FOREIGN_REPO):
        return None
    for tok, root in (public_roots() if roots is None else list(roots)):
        if tok != token:
            continue
        if not rest:
            return root
        return contained_child(root, rest)
    return None


def _foreign_checkout_root(resolved: Path) -> "Path | None":
    """The checkout directory *resolved* lives in, when it is not this one.

    Artifacts get repaired from a different worktree than the one that produced
    them — that is the whole situation this module was written for. Without
    this, a path from ``…/repos/<project>-<n>/bench/x`` sanitised from a
    sibling checkout falls through to ``<home>`` and publishes
    ``<home>/.../repos/<repo>/bench/x``: no account name, but a spelling of a
    repo-relative path no reader can match against the tree.

    The rule is structural, not a name list: a directory component that is the
    project name plus a checkout suffix is a checkout of this project. Matched
    from the right, so a path inside a checkout inside a checkout is rooted at
    the innermost one. The bare project name does not qualify — see
    :data:`_CHECKOUT_DIR`.
    """
    parts = resolved.parts
    for i in range(len(parts) - 1, -1, -1):
        if _CHECKOUT_DIR.fullmatch(parts[i]):
            root = Path(*parts[: i + 1])
            # The checkout this process is running in is `<repo>`, and it is already a
            # declared root. Returning it here too would put two roots of identical length
            # in the candidate list and let a sort decide which token a path gets.
            try:
                if root.resolve() == REPO.resolve():
                    return None
            except OSError:  # pragma: no cover - unresolvable, treated as foreign
                pass
            return root
    return None


def _account_names() -> "list[str]":
    """Account names to scrub from free text, longest first."""
    names = set()
    for candidate in (Path.home().name, os.environ.get("USERNAME", ""),
                      os.environ.get("USER", ""), os.environ.get("LOGNAME", "")):
        if candidate and len(candidate) >= 3:
            names.add(candidate)
    try:
        who = getpass.getuser()
    except Exception:  # pragma: no cover - getuser raises on some CI images
        who = ""
    if who and len(who) >= 3:
        names.add(who)
    return sorted(names, key=len, reverse=True)


def _account_patterns() -> "list[tuple[str, re.Pattern]]":
    """The account names, compiled with word boundaries. Detector and scrubber share these.

    THE BOUNDARY IS NOT A REFINEMENT, IT IS THE DIFFERENCE BETWEEN A SCREEN AND A CORRUPTER.
    An account name is a word this process happens to have been started under, and the
    shortest ones this module accepts are three characters: ``dev``, ``ann``, ``bot``, ``ci2``.
    Without boundaries, an operator called ``dev`` turns ``device_index`` into
    ``<user>ice_index`` in every scrubbed traceback, and the repository-wide screen reports a
    leak in every file that says ``devices``. Both failures are silent on the machine that
    wrote the code and loud on somebody else's, which is the exact class of defect this
    module exists to remove.

    ``\\w`` on both sides, so ``dev`` matches in ``C:\\Users\\dev\\x`` and in ``(dev)`` and
    not in ``device`` or ``dev_tools``. The cost is stated plainly: an account name glued to
    a longer word is not caught here. That case is not silent, because the *structural*
    patterns match a home directory by shape whoever lives in it — the account screen has
    always been the layer on top, never the load-bearing one.
    """
    return [(name, re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE))
            for name in _account_names()]


#: Structural leak shapes. Each group 1 is the account name that follows the root.
#:
#: The separator class is ``[\\/]+`` rather than ``[\\/]{1,2}``: a Windows path that has
#: been through two rounds of JSON escaping reads ``C:\\\\\\\\Users\\\\\\\\<name>``, and the
#: bounded form walked past it. Committed evidence in this tree contains exactly that
#: spelling, and it was being caught only by the account-name screen — which is to say, only
#: on the machine whose account it was.
LEAK_PATTERNS: "list[tuple[str, re.Pattern]]" = [
    ("windows_home", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+([^\\/\"'\s,;)]+)")),
    ("posix_home", re.compile(r"/home/([^/\"'\s,;)]+)")),
    ("macos_home", re.compile(r"/Users/([^/\"'\s,;)]+)")),
    ("windows_drive_abs",
     re.compile(r"[A-Za-z]:[\\/]+(Program Files(?: \(x86\))?|ProgramData)")),
    # Anchored on a non-word boundary, not on a leading separator: a reproduction
    # command printed at line start (``.venv-cu12/Scripts/python.exe -m ...``) is a
    # local interpreter path with no separator in front of it, and the previous form
    # walked straight past it into a committed log. The lookbehind still refuses to
    # match a word that merely ends in "venv".
    ("virtualenv", re.compile(r"(?<![\w.-])\.?venv[\w.-]*[\\/]")),
]

#: Directory names that identify a private checkout rather than a project.
_WORKTREE = re.compile(r"onnxruntime-ep-vulkan-[0-9a-z][\w.-]*")

#: Token for a checkout of this project that is **not** the one this process is running
#: in. `<repo>` is reserved for the current checkout, and the distinction is load-bearing
#: rather than cosmetic: this suite repairs artifacts from a sibling worktree, so a record
#: can contain paths from both trees at once. Spelling them the same way told a reader that
#: two different files were one file — and on a machine where the current checkout is a
#: suffixed worktree, the *canonical* checkout serialised as `<home>/...`, so `<repo>` was
#: simultaneously ambiguous and not the project directory.
FOREIGN_REPO = "<foreign-repo>"


def _scrub_checkout_name(m: "re.Match") -> str:
    """A checkout directory name in prose: this one is ``<repo>``, any other is foreign."""
    return "<repo>" if m.group(0) == REPO.name else FOREIGN_REPO


#: A directory component that is a *foreign* checkout of this project: the
#: project name plus a worktree suffix. The bare project name is deliberately
#: excluded — ``~/.cache/onnxruntime-ep-vulkan/bench-models`` is a cache, not a
#: checkout, and rooting it at ``<foreign-repo>`` would attribute model weights to a
#: source tree. The current checkout, whatever it is called, is already a
#: declared root.
_CHECKOUT_DIR = _WORKTREE

#: A whole absolute path embedded in prose, anchored on a home-directory root.
#: Matched greedily to the next delimiter so the *entire* path can be handed to
#: :func:`public_path` and come back rooted (``<model-cache>/x``) rather than
#: merely beheaded (``<home>/.cache/onnxruntime-ep-vulkan/bench-models/x``).
#: Beheading is not wrong, but it publishes two different spellings of the same
#: directory depending on whether it arrived as a field or as a sentence, and a
#: reader cannot tell those are the same place.
_ABS_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)[^\"'\s,;)\]]*")


def _scrub_system_root(m: "re.Match") -> str:
    """``C:\\Program Files\\NVIDIA\\...`` -> ``<program-files>\\NVIDIA\\...``.

    Named by role like every other root here rather than deleted: a DLL that failed to
    load says something a reader needs, and which system directory it was under is part
    of it. What is dropped is the drive letter, which is the machine-dependent half.
    """
    return _SYSTEM_ROOT_TOKENS.get(m.group(1).lower(), "<system>")


#: One token per system root this module recognises, spelled out rather than derived, so
#: the published name is a decision and not a side effect of how Windows capitalises.
_SYSTEM_ROOT_TOKENS = {
    "program files": "<program-files>",
    "program files (x86)": "<program-files-x86>",
    "programdata": "<program-data>",
}


#: How :func:`scrub_text` rewrites each shape :func:`scan` detects.
#:
#: WHY THIS TABLE EXISTS AT ALL. `scan` and `scrub_text` are the two halves of one
#: contract: `dump_public_json` scrubs a payload and then refuses to write it if `scan`
#: still finds something. If the detector knows a shape the scrubber does not, that
#: contract is **unsatisfiable** — no input the caller can construct passes — and the
#: writer raises `PathLeak` on text it has already sanitised. That was the state of
#: ``windows_drive_abs``: added to `LEAK_PATTERNS`, never added to the scrubber, which
#: hand-indexed ``[:3]`` and ``[4]`` and stepped over it. The first Windows DLL-load error
#: to reach a committed record would have been unwritable, and the fix under deadline
#: would have been to delete the field.
#:
#: So the scrubber no longer indexes into `LEAK_PATTERNS`; it iterates it, and a pattern
#: with no entry here fails at import rather than at the write boundary of whoever adds
#: the next one.
LEAK_SCRUB: "dict[str, object]" = {
    "windows_home": "<home>",
    "posix_home": "<home>",
    "macos_home": "<home>",
    "windows_drive_abs": _scrub_system_root,
    "virtualenv": "/<venv>/",
}

_UNSCRUBBABLE = [kind for kind, _ in LEAK_PATTERNS if kind not in LEAK_SCRUB]
if _UNSCRUBBABLE:  # pragma: no cover - import-time structural guard
    raise RuntimeError(
        f"LEAK_PATTERNS entries with no LEAK_SCRUB rule: {_UNSCRUBBABLE}. A shape this "
        f"module detects and cannot rewrite makes dump_public_json unsatisfiable: the "
        f"payload is scrubbed, still scans dirty, and the write raises PathLeak.")


def scrub_text(text: str) -> str:
    """Remove machine identity from free text — details, tracebacks, captured logs.

    Whole absolute paths are re-rooted through :func:`public_path` so prose and
    structured fields agree on how to spell a directory; anything left that
    still names a home, a system directory, a private checkout, a virtualenv or the
    account is collapsed to a token. This is the lossy half of the module and it is for
    text nobody parses; structured fields go through :func:`public_path`
    directly.

    Postcondition, asserted in ``bench/test_public_paths.py`` for every entry of
    :data:`LEAK_PATTERNS`: ``scan(scrub_text(x)) == []``.
    """
    if not isinstance(text, str) or not text:
        return text
    out = _ABS_PATH.sub(lambda m: public_path(m.group(0), scrub_remainder=False), text)
    for kind, pattern in LEAK_PATTERNS:
        out = pattern.sub(LEAK_SCRUB[kind], out)
    out = _WORKTREE.sub(_scrub_checkout_name, out)
    for _, pattern in _account_patterns():
        out = pattern.sub(REDACTED_USER, out)
    return out


def scan(text: str, *, accounts: bool = True) -> "list[tuple[str, str]]":
    """Every machine-identifying substring in *text*, as ``(kind, sample)``.

    Total: it returns findings instead of raising, so a caller can report all of
    them at once. The polarity that matters is that it is watched to find
    something — see ``bench/test_public_paths.py``, which plants each shape.

    ``accounts=False`` drops the ``account_name`` kind and leaves every structural
    shape in place. It exists for :func:`scan_structural`; see the reasoning there.
    """
    if not isinstance(text, str):
        text = str(text)
    found: "list[tuple[str, str]]" = []
    for kind, pattern in LEAK_PATTERNS:
        for m in pattern.finditer(text):
            found.append((kind, m.group(0)[:120]))
    if accounts:
        for _, pattern in _account_patterns():
            for m in pattern.finditer(text):
                found.append(("account_name", text[max(0, m.start() - 20):m.end() + 20]))
    for m in _WORKTREE.finditer(text):
        found.append(("worktree_name", m.group(0)))
    return found


#: The kinds :func:`scan_structural` can report: every shape that is decided from the
#: *text*, and none that is decided from the process that happens to be reading it.
STRUCTURAL_KINDS: "tuple[str, ...]" = tuple(
    [kind for kind, _ in LEAK_PATTERNS] + ["worktree_name"])


def scan_structural(text: str) -> "list[tuple[str, str]]":
    """:func:`scan` minus the account-name screen: same answer on every machine.

    WHY A SECOND ENTRY POINT RATHER THAN A FLAG NOBODY SETS. The account screen asks
    "does this text contain the name of the account *this interpreter is running under*".
    That is the right question at a write boundary — the payload about to be committed was
    produced by this process, on this machine, and its account name is the thing being
    published.

    It is the wrong question for a survey of files somebody else committed. Run over the
    whole repository it makes the answer a function of the runner: the operator's own
    account name finds hundreds of files, a CI account called ``runner`` finds none of
    those and some others, and the checked-in ratchet those numbers are compared against
    is then either stale, or
    undeclared-red, or — with a short account name like ``dev`` — red on every file that
    contains the word ``devices``. A baseline that a second machine cannot reproduce is not
    a baseline; it is a record of one laptop, which is the defect this module was written
    to stop publishing.

    The username leaks the ratchet exists to count are still counted, by shape:
    ``C:\\Users\\justinchu\\...`` is ``windows_home`` on every machine, including the ones
    where nobody is called justinchu.
    """
    return scan(text, accounts=False)


def scan_payload(payload, *, where: str = "") -> "list[str]":
    """Walk a JSON-able object and report every leak with the key path that holds it."""
    problems: "list[str]" = []

    def walk(node, at: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str):
                    for kind, sample in scan(k):
                        problems.append(f"{at}<key {k!r}>: {kind}: {sample}")
                walk(v, f"{at}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{at}[{i}]")
        elif isinstance(node, str):
            for kind, sample in scan(node):
                problems.append(f"{at}: {kind}: {sample}")

    walk(payload, where or "$")
    return problems


def runtime_only_keys(payload, *, where: str = "") -> "list[str]":
    """Every place *payload* still carries the runtime-only sidecar, by key path.

    :data:`RUNTIME_ONLY_KEY` holds absolute paths for one process to hand to another
    within a run. `public_payload` drops it, so a sanitised payload never reaches here
    carrying one; this exists for the other polarity, ``sanitise=False``, where the
    caller asserts the payload is already public.

    The check is *structural* rather than a leak scan because the two questions have
    different answers: a runtime path of ``/srv/run/out0.npy`` names no machine and no
    account, so every pattern in :data:`LEAK_PATTERNS` passes it, and it is still a
    process-local handle with no meaning to a reader. Publishing it would put the
    parent's private channel into the evidence — the exact confusion this key was
    introduced to end.
    """
    found: "list[str]" = []

    def walk(node, at: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == RUNTIME_ONLY_KEY:
                    found.append(
                        f"{at}.{k}: runtime_only_key: {RUNTIME_ONLY_KEY!r} is an "
                        f"in-process channel and must not be serialised; route the "
                        f"payload through public_paths.public_payload() first")
                    continue
                walk(v, f"{at}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{at}[{i}]")

    walk(payload, where or "$")
    return found


def assert_public(payload, *, where: str = "") -> None:
    """Raise :class:`PathLeak` if *payload* would publish machine identity."""
    problems = scan_payload(payload, where=where)
    problems.extend(runtime_only_keys(payload, where=where))
    if problems:
        head = "\n  ".join(problems[:12])
        more = f"\n  ... and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise PathLeak(
            f"{len(problems)} machine-identifying value(s) in a payload bound for a "
            f"committed artifact. Route every path through public_paths.public_path() "
            f"and every free-text field through public_paths.scrub_text():\n  {head}{more}"
        )


def public_payload(payload):
    """Return a copy of *payload* with every path rooted and every string scrubbed.

    The in-memory record keeps real, openable paths — a record that lied about where its
    own scratch directory is would be a working harness turned into a broken one for the
    sake of a clean file. So the rewrite happens at the boundary: real in memory, rooted
    on disk.

    WHICH CHANNEL A LATER READER ACTUALLY USES.  There is no ``profile_path`` field on
    these records, and the sentence that said this module's rewrite protected one
    outlived the code by a revision. ``cuda_profile`` reads ``profile_rel`` — a relative
    handle — through ``cuda_competition.resolve_scratch_file``, which roots it at the
    arm's own scratch directory and puts it through :func:`contained_child`. The absolute
    directory it is rooted at travels under :data:`RUNTIME_ONLY_KEY` and is dropped here;
    the published record carries the rooted ``scratch_dir`` instead, which
    :func:`resolve_public_path` turns back into a directory on a second machine.
    ``bench/test_public_paths.py::test_no_prose_names_a_channel_the_reader_does_not_read``
    checks the claim in this paragraph against the field ``cuda_profile.py`` actually
    takes off a record, which is what stops it going stale again — a presence check would
    not have, because ``profile_path`` is still a parameter name in that module.
    """
    return _sanitise_node(payload)


def public_text(text: str) -> str:
    """Scrubbed copy of rendered Markdown/log text, for the same boundary reason."""
    return scrub_text(text)


def dump_public_json(payload, path, *, indent: int = 2, sort_keys: bool = False,
                     default=str, sanitise: bool = True) -> str:
    """Serialise *payload* to *path*, refusing to write machine identity.

    With ``sanitise=True`` (the default) the payload is rooted first and the
    screen then runs on the result, so the screen is testing the sanitiser
    rather than merely repeating it. With ``sanitise=False`` the payload must
    already be public — that polarity is what the tests use to prove the screen
    fires at all.

    The check is on the **serialised text**, not on the object: ``default=str``
    turns unknown objects into strings after any object-level walk, so a `Path`
    that reached the writer unconverted would be invisible to an object-level
    screen and perfectly visible in the file. Round-tripping through
    ``json.loads`` closes that gap.
    """
    if sanitise:
        payload = public_payload(json.loads(
            json.dumps(payload, default=default)))
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys, default=default)
    assert_public(json.loads(text), where=str(path))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return text


def write_public_text(text: str, path, *, sanitise: bool = True) -> str:
    """Write Markdown/log output, refusing machine identity for the same reason."""
    if sanitise:
        text = public_text(text)
    problems = scan(text)
    if problems:
        head = "\n  ".join(f"{k}: {s}" for k, s in problems[:12])
        raise PathLeak(
            f"{len(problems)} machine-identifying value(s) in text bound for "
            f"{path}:\n  {head}"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return text


def sanitise_file(path) -> "tuple[int, bool]":
    """Rewrite an existing artifact in place. Returns ``(leaks_before, changed)``.

    For repairing artifacts that were produced before the writers were fixed.
    JSON is walked and rewritten field by field so the structure survives;
    anything else is scrubbed as text.
    """
    target = Path(path)
    original = target.read_text(encoding="utf-8", errors="replace")
    before = len(scan(original))
    if target.suffix == ".json":
        payload = json.loads(original)
        cleaned = _sanitise_node(payload)
        text = json.dumps(cleaned, indent=2, sort_keys=target.name == "models.json")
    else:
        text = scrub_text(original)
    changed = text != original
    if changed:
        target.write_text(text, encoding="utf-8")
    return before, changed


#: JSON keys whose values are paths and must be rewritten as rooted paths rather
#: than scrubbed as prose.
PATH_KEYS = frozenset({
    "path", "file", "outputs_dir", "profile_path", "cache_root", "model_path",
    "interpreter", "executable", "out", "dump", "log_path", "onnx_path",
})

#: Extensions that carry committed evidence a reader is expected to read. The
#: scanner CLI and the legacy ratchet both enumerate from this, so "what counts
#: as an evidence file" has one answer.
EVIDENCE_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".log", ".txt", ".csv"})


def _sanitise_node(node, key: str = ""):
    if isinstance(node, dict):
        # The runtime-only sidecar is dropped, not rewritten: rooting it would keep a
        # dead handle in the artifact that reads like a live one.
        return {k: _sanitise_node(v, k) for k, v in node.items()
                if k != RUNTIME_ONLY_KEY}
    if isinstance(node, list):
        return [_sanitise_node(v, key) for v in node]
    if isinstance(node, str):
        if key in PATH_KEYS and scan(node):
            return public_path(node)
        return scrub_text(node)
    return node


def _expand(paths) -> list[Path]:
    """A directory argument means every evidence file under it, not a read error.

    The operator entry point is what people reach for after a run ("did this leak?"),
    and a run writes a directory. Making them glob it themselves is how a file gets
    missed.
    """
    out: list[Path] = []
    for p in paths:
        t = Path(p)
        if t.is_dir():
            out.extend(sorted(f for f in t.rglob("*")
                              if f.is_file() and f.suffix.lower() in EVIDENCE_SUFFIXES))
        else:
            out.append(t)
    return out


def main(argv=None) -> int:  # pragma: no cover - operator entry point
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fix", action="store_true", help="rewrite in place")
    a = ap.parse_args(argv)

    bad = 0
    targets = _expand(a.paths)
    if not targets:
        print("no evidence files matched")
        return 0
    for target in targets:
        if a.fix:
            before, changed = sanitise_file(target)
            after = len(scan(target.read_text(encoding="utf-8", errors="replace")))
            print(f"{target}: {before} -> {after} leak(s){' (rewritten)' if changed else ''}")
            bad += after
        else:
            found = scan(target.read_text(encoding="utf-8", errors="replace"))
            print(f"{target}: {len(found)} leak(s)")
            for kind, sample in found[:5]:
                print(f"    {kind}: {sample}")
            bad += len(found)
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
