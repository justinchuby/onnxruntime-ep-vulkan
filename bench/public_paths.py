"""Public-path leak screening for committed `bench/` artifacts (issue #78).

WHY THIS MODULE EXISTS
=======================
`bench/real_model.py::resolve_model` puts a real, absolute, machine-specific filesystem path
into the dict it returns, and `bench/results/probe_real_model_latency.py` writes that dict
straight into `bench/results/real_model_latency.json` and `real_model_diagnostics.json` —
files this repository commits. Both currently contain a literal
`C:\\Users\\<name>\\...` path (the Foundry cache and the repo-cache model directory on the
machine that ran the probe): a public record naming a private machine's home directory. The
same shape of leak generalises to `/var/...`/`/opt/...` service roots, UNC shares, Windows
extended-device paths (`\\\\?\\...`), and interpreter/venv/cache internals
(`site-packages`, `.venv`, `__pycache__`) — none of which belong in a file this project
publishes.

This module is the **sole writer** of committed JSON records that may carry a filesystem
path, so there is exactly one place that can leak one, and exactly one place that screens
for it. `bench/real_model.py` and `bench/results/probe_real_model_latency.py` route every
committed write through `write_public_json`, never through a bare `json.dumps` +
`Path.write_text`.

WHAT IT DELIBERATELY DOES NOT DO
=================================
It does not try to redact a leak and ship the redacted text anyway. A path that slipped
through construction is evidence the record's shape is wrong, not evidence the file needs a
find-and-replace: `write_public_json` refuses (raises `PublicPathLeak`) rather than
silently rewriting the record. It also does not touch valid ONNX node names such as
`/model/layers.0/self_attn/q_proj/MatMul` — those begin with a segment (`model`) that is not
on the sensitive-root list, and the pattern used here requires the sensitive segment to be
the FIRST segment of a path-shaped token (immediately after a quote, whitespace, bracket, or
string start), so a later occurrence of one of these words *inside* an already-public node
name (e.g. `/model/etc_block/...`) does not match either.

PUBLIC API: `root_public_path` (raises `TypeError` on a non-path-like value),
`write_public_json` (raises `PublicPathLeak`), `runtime_path` (raises
`RuntimePathUnavailable`) — all three render a verdict (accept the value / refuse it) and
are screened by `rust/tools/audit_instruments.py`. `_find_leaks` is a private scanning
primitive used only by `write_public_json`: it always returns a list (empty or not) and
never raises, which does not fit the raise-based reject/accept polarity contract the census
reads (`bench/_polarity.py`'s `refuses`/`selects` are for a `(value, why)`-returning total
instrument, not a "scan and report" one) — kept private and exercised directly by
`bench/test_public_paths.py` instead of forcing it into a shape it is not.
"""

from __future__ import annotations

import copy
import json
import re
import tempfile
from pathlib import Path
from typing import Any

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent

#: The private, real-path channel every public record may carry alongside its public form.
#: A key with this prefix is stripped by `write_public_json` before anything is screened or
#: written, so it is a channel `runtime_path` reads and the public artifact never contains.
PRIVATE_KEY_PREFIX = "_runtime_path"


class PublicPathLeak(RuntimeError):
    """A record about to be written to a committed, public JSON file names a private path.

    Raised, not redacted: a leak reaching this point means the record's shape let a real
    filesystem path in, and the fix belongs at the point the record was built, not in this
    module rewriting text after the fact.
    """


class RuntimePathUnavailable(RuntimeError):
    """A caller asked to open the real path of a record that never recorded one.

    Distinguishes "no runtime path was ever attached to this record" from "the path is
    private and none of your business" — both would look like a missing key to a bare
    ``rec[key]``, and only the first is this module's problem.
    """


#: Named public roots, longest-`Path.parts`-prefix matched — never `str.startswith`, which a
#: sibling directory (`C:\Users\ann-backup` next to `C:\Users\ann`) would defeat.
def _named_roots() -> "tuple[tuple[str, Path], ...]":
    import os

    cache_override = os.environ.get("ONNXRUNTIME_EP_VULKAN_MODEL_CACHE")
    cache_root = Path(cache_override) if cache_override else (
        Path.home() / ".cache" / "onnxruntime-ep-vulkan"
    )
    return (
        ("<model-cache>", cache_root),
        ("<venv>", _ROOT / ".venv"),
        ("<repo>", _ROOT),
        ("<tmp>", Path(tempfile.gettempdir())),
        ("<home>", Path.home()),
    )


def root_public_path(path: "str | Path") -> str:
    """Rewrite an absolute, private path to a rooted, public token string.

    Refuses (``TypeError``) anything that is not path-like: a public record must not
    silently stringify an unexpected object (``None``, an int, a live handle) into text that
    merely *looks* like a rooted path.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(
            f"root_public_path expects a path-like value (str or Path); got "
            f"{type(path).__name__} ({path!r}). Silently str()-ing an unexpected object "
            f"would produce a public field that looks rooted without being derived from a "
            f"real path."
        )
    p = Path(path)
    try:
        p = p.resolve()
    except OSError:
        pass

    best: "tuple[str, tuple[str, ...]] | None" = None
    for token, root in _named_roots():
        try:
            root_r = root.resolve()
        except OSError:
            continue
        root_parts = root_r.parts
        if len(root_parts) <= len(p.parts) and p.parts[: len(root_parts)] == root_parts:
            if best is None or len(root_parts) > len(best[1]):
                best = (token, root_parts)
    if best is None:
        return f"<elsewhere>/{p.name}"
    token, root_parts = best
    rel = p.parts[len(root_parts):]
    return token if not rel else f"{token}/{'/'.join(rel)}"


#: Sensitive root segments. Matched only as the FIRST segment of a path-shaped token (see
#: `_LEAK_RE` below) so a later occurrence inside an already-public node name — e.g.
#: `/model/etc_block/...` — is not flagged: the census this module exists to pass would be
#: false-positive noise on every graph that happens to name a layer `home`, `root` or `tmp`.
_SENSITIVE_SEGMENTS = (
    "var", "opt", "home", "root", "Users", "tmp", "private", "proc", "etc", "mnt", "media",
    "usr",
)
_SEG_ALT = "|".join(_SENSITIVE_SEGMENTS)

#: `/var/...`, `/opt/...`, ..., and their Windows-backslash form (`\Users\...`, escaped to
#: two-or-more literal backslashes by `json.dumps`). The leading-boundary lookbehind requires
#: the match NOT be preceded by a word character, `/` or `\` — so it fires only at the start
#: of a path-shaped token (right after a quote, whitespace, bracket, comma, or string start),
#: never after another path segment.
_LEAK_SEGMENT_RE = re.compile(
    rf'(?<![\w/\\])(?:/|\\{{2,}})(?:{_SEG_ALT})(?=/|\\{{2,}}|["\'\s,\]}}]|$)'
)
#: A Windows drive letter followed by a separator — `C:\` (escaped) or `C:/`.
_WINDOWS_DRIVE_RE = re.compile(r'\b[A-Za-z]:(?:\\{2,}|/)')
#: A UNC share (`\\server\share`), escaped to four-or-more backslashes by `json.dumps`, at a
#: token boundary and not immediately followed by `?` or `.` (which is the extended-device
#: form below, screened and reported separately).
_UNC_RE = re.compile(r'(?:^|(?<=["\'\s\[,]))\\{2,}(?![?.]\\)[A-Za-z0-9_.$-]')
#: Windows extended-device paths: `\\?\...` or `\\.\...`.
_EXTENDED_DEVICE_RE = re.compile(r'\\{2,}[?.]\\{2,}')
#: Interpreter/venv/build-cache signatures — these name a machine's toolchain layout, not a
#: model or a result, and have no reason to appear in a committed record at all.
_TOOLCHAIN_RE = re.compile(
    r'site-packages|\.venv|__pycache__|Scripts[\\/]{1,2}python|bin/python|AppData[\\/]{1,2}Local'
)

_LEAK_PATTERNS: "tuple[tuple[str, re.Pattern], ...]" = (
    ("posix_or_windows_root_segment", _LEAK_SEGMENT_RE),
    ("windows_drive_root", _WINDOWS_DRIVE_RE),
    ("unc_path", _UNC_RE),
    ("windows_extended_device_path", _EXTENDED_DEVICE_RE),
    ("interpreter_or_cache_signature", _TOOLCHAIN_RE),
)

#: The public root tokens `root_public_path` itself emits. A match whose path-shaped token
#: already BEGINS with one of these is this module's own output, not a leak: `<venv>/Lib/
#: site-packages/onnxruntime/capi` (a real, legitimate `onnxruntime_lib` field) contains the
#: literal substring `site-packages`, and flagging it would make `write_public_json` refuse
#: to write the very tokens this module exists to produce.
_PUBLIC_ROOT_TOKENS = ("<repo>", "<venv>", "<model-cache>", "<tmp>", "<home>", "<elsewhere>")


def _token_start(text: str, at: int) -> int:
    """The index where the path-shaped token containing position *at* begins.

    Scans backward to the nearest quote/whitespace/bracket/comma — the same boundary set
    `_LEAK_SEGMENT_RE`'s lookbehind treats as "the start of a path-shaped token".
    """
    i = at
    while i > 0 and text[i - 1] not in '"\'\t\n \\[{,':
        i -= 1
    return i


def _find_leaks(text: str) -> "list[dict]":
    """Every private-path-shaped match in *text*, named by which pattern caught it.

    Operates on already-serialized text (the `json.dumps(..., default=str)` output), not on
    the pre-serialization object: a value that reaches text only via `default=str` (e.g. a
    bare `Path`) is exactly as visible to this scan as a plain string field, which is the
    point — screening the object graph would have to know about every type that might
    stringify into a path, and screening the text does not need to.

    A match is skipped, not reported, when the path-shaped token it sits in already BEGINS
    with one of this module's own public root tokens (``_PUBLIC_ROOT_TOKENS``): that token
    is proof the value was already produced by `root_public_path`, and a screen that refused
    its own output would make every legitimate record — `<venv>/Lib/site-packages/...` is a
    real, correct `onnxruntime_lib` value — unwritable.
    """
    hits = []
    for name, pattern in _LEAK_PATTERNS:
        for m in pattern.finditer(text):
            start = _token_start(text, m.start())
            if any(text.startswith(tok, start) for tok in _PUBLIC_ROOT_TOKENS):
                continue
            hits.append({"pattern": name, "match": m.group(0), "at": m.start()})
    return hits


def _strip_private(obj: Any) -> Any:
    """Deep-copy *obj*, dropping every dict key that starts with an underscore.

    The private channel (`_runtime_path` and any other leading-underscore key) is for the
    process that just resolved a model to open the real file; it is never part of the record
    this function's caller is about to make public.
    """
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_private(v) for v in obj)
    return obj


def write_public_json(path: "str | Path", obj: Any, *, indent: "int | None" = 2) -> None:
    """Serialize *obj* to *path*, refusing (not redacting) if the text leaks a private path.

    The sole writer for committed `bench/` JSON records. Strips leading-underscore keys
    (the private channel), serializes with `default=str` exactly as every writer in this
    tree already did, screens the resulting TEXT for the patterns in `_LEAK_PATTERNS`, and
    raises `PublicPathLeak` — before anything touches disk — if any pattern hits. A record
    that needs a real path for anything (opening the file to run inference, printing where a
    model was found on THIS machine) uses `runtime_path` at the point of use instead of
    putting the real path in this file.
    """
    public_obj = _strip_private(copy.deepcopy(obj))
    text = json.dumps(public_obj, indent=indent, default=str)
    hits = _find_leaks(text)
    if hits:
        sample = hits[0]
        raise PublicPathLeak(
            f"refusing to write {path}: {len(hits)} private-path-shaped match(es) in the "
            f"serialized record, first is pattern={sample['pattern']!r} "
            f"text={sample['match']!r} at offset {sample['at']}. A public artifact must not "
            f"carry a real filesystem path — route the real path through the private "
            f"`_runtime_path` channel and `runtime_path()` instead."
        )
    Path(path).write_text(text, encoding="utf-8")


def runtime_path(rec: dict, key: str = PRIVATE_KEY_PREFIX) -> Path:
    """The real, absolute path a record's private channel carries, for opening the file.

    Call sites that need to actually open a model — build an ONNX Runtime session, hash a
    blob — read the path through here rather than through `rec["path"]` (which, after this
    module's conventions are followed, holds the rooted PUBLIC token, not a usable path).
    Making this the one accessor is what keeps every real-path call site greppable.
    """
    value = rec.get(key)
    if not value:
        raise RuntimePathUnavailable(
            f"{key!r} is missing or empty on this record; there is no real path to open. "
            f"This is not the same state as \"the path is private\" — a record that was "
            f"never resolved to a file has no runtime path to withhold."
        )
    return Path(value)
