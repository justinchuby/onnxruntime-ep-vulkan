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
            candidates = sorted(candidates + [("<repo>", foreign)],
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
            return Path(*parts[: i + 1])
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


#: Structural leak shapes. Each group 1 is the account name that follows the root.
LEAK_PATTERNS: "list[tuple[str, re.Pattern]]" = [
    ("windows_home", re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}([^\\/\"'\s,;)]+)")),
    ("posix_home", re.compile(r"/home/([^/\"'\s,;)]+)")),
    ("macos_home", re.compile(r"/Users/([^/\"'\s,;)]+)")),
    ("windows_drive_abs", re.compile(r"[A-Za-z]:[\\/]{1,2}(?:Program Files|ProgramData)")),
    # Anchored on a non-word boundary, not on a leading separator: a reproduction
    # command printed at line start (``.venv-cu12/Scripts/python.exe -m ...``) is a
    # local interpreter path with no separator in front of it, and the previous form
    # walked straight past it into a committed log. The lookbehind still refuses to
    # match a word that merely ends in "venv".
    ("virtualenv", re.compile(r"(?<![\w.-])\.?venv[\w.-]*[\\/]")),
]

#: Directory names that identify a private checkout rather than a project.
_WORKTREE = re.compile(r"onnxruntime-ep-vulkan-[0-9a-z][\w.-]*")

#: A directory component that is a *foreign* checkout of this project: the
#: project name plus a worktree suffix. The bare project name is deliberately
#: excluded — ``~/.cache/onnxruntime-ep-vulkan/bench-models`` is a cache, not a
#: checkout, and rooting it at ``<repo>`` would attribute model weights to the
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
    r"(?:[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}|/home/|/Users/)[^\"'\s,;)\]]*")


def scrub_text(text: str) -> str:
    """Remove machine identity from free text — details, tracebacks, captured logs.

    Whole absolute paths are re-rooted through :func:`public_path` so prose and
    structured fields agree on how to spell a directory; anything left that
    still names a home, a private checkout, a virtualenv or the account is
    collapsed to a token. This is the lossy half of the module and it is for
    text nobody parses; structured fields go through :func:`public_path`
    directly.
    """
    if not isinstance(text, str) or not text:
        return text
    out = _ABS_PATH.sub(lambda m: public_path(m.group(0), scrub_remainder=False), text)
    for _, pattern in LEAK_PATTERNS[:3]:
        out = pattern.sub("<home>", out)
    out = _WORKTREE.sub("<repo>", out)
    out = LEAK_PATTERNS[4][1].sub("/<venv>/", out)
    for name in _account_names():
        out = re.sub(re.escape(name), REDACTED_USER, out, flags=re.IGNORECASE)
    return out


def scan(text: str) -> "list[tuple[str, str]]":
    """Every machine-identifying substring in *text*, as ``(kind, sample)``.

    Total: it returns findings instead of raising, so a caller can report all of
    them at once. The polarity that matters is that it is watched to find
    something — see ``bench/test_public_paths.py``, which plants each shape.
    """
    if not isinstance(text, str):
        text = str(text)
    found: "list[tuple[str, str]]" = []
    for kind, pattern in LEAK_PATTERNS:
        for m in pattern.finditer(text):
            found.append((kind, m.group(0)[:120]))
    for name in _account_names():
        for m in re.finditer(re.escape(name), text, flags=re.IGNORECASE):
            found.append(("account_name", text[max(0, m.start() - 20):m.end() + 20]))
    for m in _WORKTREE.finditer(text):
        found.append(("worktree_name", m.group(0)))
    return found


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


def assert_public(payload, *, where: str = "") -> None:
    """Raise :class:`PathLeak` if *payload* would publish machine identity."""
    problems = scan_payload(payload, where=where)
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

    The in-memory record keeps real, openable paths — ``cuda_profile`` reads
    ``profile_path`` back out of the very record it is about to serialise, and a
    record that lied about where its own profile lives would be a working
    harness turned into a broken one for the sake of a clean file. So the
    rewrite happens at the boundary: real in memory, rooted on disk.
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
        return {k: _sanitise_node(v, k) for k, v in node.items()}
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
