"""A general screen for filesystem paths that must never reach a public artifact.

WHY THIS MODULE EXISTS (issue #78, "Non-goals")
===============================================
Issue #78 ends with a line that is easy to read as housekeeping and is not: *"Public records for
this work must never contain a local absolute path — only the pinned repo/revision/file/digest
identity."* The committed evidence in this repository shows why. `bench/results/rust-model-
runner/mobilenetv2-12.json` — a public artifact, in the tree, on the web — carries
``C:\\Users\\<name>\\.cache\\…`` and ``\\\\?\\C:\\Users\\<name>\\.copilot\\repos\\…`` in four
separate fields. That is a username, a directory layout and a machine's shape, published by an
instrument nobody asked to publish them.

WHAT MAKES THIS A SCREEN AND NOT AN ALLOWLIST
=============================================
The obvious implementation is a list of known-private roots — ``/home``, ``C:\\Users``,
``/tmp`` — and it fails on the first machine that is not the author's. A model cache under
``/srv/models``, ``/data/hf``, ``/nix/store``, ``/Volumes/ext``, ``/run/user/1000`` or
``/opt/ml`` leaks exactly as much and matches none of the list. So the rule here is inverted:

    **Any string with a filesystem root anchor is a finding unless it is an explicit public
    placeholder.**

A root anchor is a Windows drive (``C:\\``, ``c:/``), a UNC share (``\\\\host\\share``,
``//host/share``), a Windows device or extended-length prefix (``\\\\?\\``, ``\\\\.\\``), an
absolute POSIX path (``/anything/…``), or a home/temp/cache macro (``~/``, ``%USERPROFILE%``,
``$HOME``, ``%TEMP%``, ``$XDG_CACHE_HOME``). No root is named as private; every root is.

THE ONNX NODE-NAME PROBLEM, AND WHY THE ANSWER IS STRUCTURAL
============================================================
ONNX exporters name nodes with a leading slash: ``/encoder/layer.0/attention/self/query/MatMul``
is a legitimate value that a provenance artifact may legitimately publish, and it is
character-for-character the shape of an absolute POSIX path. **No string-level rule can tell
that from ``/srv/models/minilm``** — a screen claiming to is guessing, and a guessing screen
either loses node names or keeps private paths.

So the exemption is structural, not lexical: `screen_public_record` walks a JSON object and
grants the POSIX-absolute exemption **only** to values under a declared graph-name key
(`GRAPH_NAME_KEYS`), and only when the value also has ONNX-name *shape*. Everywhere else — every
key that is not a declared name field — an absolute POSIX path is a finding regardless of its
root. Drive letters, UNC, device paths and home/temp macros are findings **even inside a name
field**, because no ONNX exporter emits those and a leak hiding in a name field is still a leak.

TEXT THAT IS NOT PLAIN TEXT
===========================
A path that has been through ``json.dumps`` is ``"C:\\\\Users\\\\…"``; through a URL encoder it
is ``C%3A%5CUsers``; through a Windows API that speaks UTF-16 it is bytes with a NUL after every
character. Each of those is the same disclosure and none of them matches a naive scan, so
`screen_public_text` screens the raw text *and* de-escaped and percent-decoded variants of it,
and `decode_wide_text` turns UTF-16 bytes into something the screen can read.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

#: What a redacted local path is replaced by. A screen that deletes the field instead would make
#: "there was no path here" and "there was one and we removed it" the same artifact.
PUBLIC_PLACEHOLDER = "<redacted-local-path>"

#: Placeholders a public artifact may legitimately carry. Matched exactly, not by prefix: a
#: value that merely *starts* with the placeholder can still carry a path after it.
PUBLIC_PLACEHOLDERS = frozenset({PUBLIC_PLACEHOLDER, "<redacted>", "REDACTED", ""})

#: JSON keys whose values may be ONNX graph names, which are POSIX-path-shaped by construction.
#: This is the ONLY exemption in the module, it is keyed on structure rather than on the value's
#: text, and it never exempts a drive letter, a UNC path or a home/temp macro.
GRAPH_NAME_KEYS = frozenset(
    {"name", "names", "tensor", "tensor_name", "node", "node_name", "node_names",
     "input", "inputs", "output", "outputs", "where", "initializer", "op_type"}
)

#: An ONNX node name: `/`-separated identifier segments. Note what it does NOT permit — spaces,
#: backslashes, drive colons, `..`, or an empty segment.
ONNX_NAME = re.compile(r"\A(?:/[A-Za-z_][A-Za-z0-9_.\-]*)+\Z")

_WIN_DRIVE = re.compile(r"(?:\A|[^A-Za-z0-9])([A-Za-z]:[\\/])")
_DEVICE = re.compile(r"(?:\\\\|//)[?.](?:\\|/)")
_UNC = re.compile(r"(?<![A-Za-z0-9+.\-:])(?:\\\\|//)[A-Za-z0-9._-]+(?:\\|/)[A-Za-z0-9._$-]+")
#: An absolute POSIX path: a leading `/` and at least one segment character. Deliberately NOT
#: `two or more segments` — `/data` and `/srv` are private roots on their own, and a rule that
#: needed a second slash would publish them. Single-segment ONNX node names like `/Shape` are
#: caught by the same rule and are exempted by *position*, not by weakening the rule.
_POSIX_ABS = re.compile(
    r"(?:\A|[\s\"'=(\[,;:])(/[A-Za-z0-9_.~$-][^\s\"'<>|*?\\]*)"
)
_HOME_MACRO = re.compile(
    r"(?:\A|[\s\"'=(\[,;])(~[\\/])"
    r"|%(?:USERPROFILE|HOMEPATH|HOMEDRIVE|APPDATA|LOCALAPPDATA|TEMP|TMP|PROGRAMDATA)%"
    r"|\$(?:HOME|TMPDIR|XDG_CACHE_HOME|XDG_DATA_HOME|USER)\b",
    re.IGNORECASE,
)
#: A scheme-qualified URL is not a filesystem path. `https://huggingface.co/...` is the pinned
#: public identity this work exists to publish, so it must survive the screen intact. Note the
#: set is explicit and small: `file://` is a filesystem path wearing a scheme, and is NOT here.
_PUBLIC_URL = re.compile(r"\A(?:https|http)://[^\s\"'<>]*\Z")
_PUBLIC_URL_ANY = re.compile(r"(?:https|http)://[^\s\"'<>]*")
_FILE_URL = re.compile(r"\bfile:/{2,3}", re.IGNORECASE)


class PrivatePathLeak(RuntimeError):
    """A value that would have published a local filesystem path was refused.

    Raised by the production serializer rather than returned, because a caller that ignores a
    return value publishes the artifact anyway, and the whole point is that it must not.
    """

    def __init__(self, findings: "tuple[str, ...]"):
        super().__init__("; ".join(findings))
        self.findings = tuple(findings)


def decode_wide_text(data: object) -> "tuple[str | None, str]":
    """Turn UTF-16/UTF-8 bytes into text the screen can read, or refuse them.

    TOTAL. A Windows API that hands back ``C:\\Users\\…`` in UTF-16 produces bytes with a NUL
    after every ASCII character; a screen that only reads ``str`` sees no path in them at all.
    Decoding is therefore part of the screen, not a caller's chore.
    """
    if isinstance(data, str):
        return data, "already text"
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return None, f"not text or bytes ({type(data).__name__}); nothing to decode"
    raw = bytes(data)
    if not raw:
        return "", "empty"
    for bom, enc in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"),
                     (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            try:
                return raw.decode(enc), f"decoded {enc} by BOM"
            except UnicodeDecodeError as exc:
                return None, f"{enc} BOM present but the bytes do not decode: {exc}"
    if len(raw) >= 4 and raw[1::2].count(0) > len(raw) // 4:
        try:
            return raw.decode("utf-16-le"), "decoded utf-16-le by NUL interleave"
        except UnicodeDecodeError:
            pass
    if len(raw) >= 4 and raw[0::2].count(0) > len(raw) // 4:
        try:
            return raw.decode("utf-16-be"), "decoded utf-16-be by NUL interleave"
        except UnicodeDecodeError:
            pass
    try:
        return raw.decode("utf-8"), "decoded utf-8"
    except UnicodeDecodeError as exc:
        return None, f"bytes are not decodable text: {exc}"


def _variants(text: str) -> "list[str]":
    """The text as written, plus the shapes an escaper would have left it in."""
    out = [text]
    if "\\\\" in text:
        out.append(text.replace("\\\\", "\\"))
    if "\\/" in text:
        out.append(text.replace("\\/", "/"))
    if "%" in text:
        decoded = text
        for enc, dec in (("%5C", "\\"), ("%5c", "\\"), ("%2F", "/"), ("%2f", "/"),
                         ("%3A", ":"), ("%3a", ":"), ("%7E", "~"), ("%7e", "~")):
            decoded = decoded.replace(enc, dec)
        if decoded != text:
            out.append(decoded)
    if "\\u" in text.lower():
        try:
            out.append(text.encode("utf-8", "surrogatepass").decode("unicode_escape"))
        except Exception:
            pass
    return out


def _findings(text: str, *, allow_graph_names: bool) -> "list[str]":
    found: list[str] = []
    for variant in _variants(text):
        if _FILE_URL.search(variant):
            found.append("`file://` URL — a filesystem path wearing a scheme")
        # Public http(s) URLs are excised before the detectors run, rather than the whole
        # value being waved through when it *starts* with one. Otherwise
        # `"https://ok.example and C:\\Users\\me"` would publish the second half.
        scrubbed = _PUBLIC_URL_ANY.sub("<url>", variant)
        m = _DEVICE.search(scrubbed)
        if m:
            found.append(f"Windows device/extended-length prefix {m.group(0)!r}")
        m = _WIN_DRIVE.search(scrubbed)
        if m:
            found.append(f"Windows drive-absolute path {m.group(1)!r}")
        m = _HOME_MACRO.search(scrubbed)
        if m:
            found.append(f"home/temp/cache macro {m.group(0)!r}")
        m = _UNC.search(scrubbed)
        if m:
            found.append(f"UNC path {m.group(0)!r}")
        for m in _POSIX_ABS.finditer(scrubbed):
            candidate = m.group(1)
            if allow_graph_names and ONNX_NAME.match(candidate):
                continue
            found.append(f"absolute POSIX path {candidate!r}")
    # De-duplicate while keeping the order the screen found them in.
    seen: set[str] = set()
    return [f for f in found if not (f in seen or seen.add(f))]


def screen_public_text(text: object, *, allow_graph_names: bool = False
                       ) -> "tuple[object | None, str]":
    """Pass a value through, or refuse it because it carries a filesystem path.

    TOTAL: returns ``(text, why)`` when the value is publishable and ``(None, why)`` when it is
    not, for any input at all. The returned value is the *same object* that was passed in, so a
    caller cannot accidentally publish a normalised copy of something the screen has not seen.

    ONE AMBIGUITY, STATED
    ---------------------
    ``screen_public_text(None)`` returns ``(None, "not a string…")`` — an accept whose value is
    indistinguishable from a refusal. That is unavoidable for a total instrument whose accept
    value is the input. It is why `screen_public_record` dispatches on *type* before calling
    this, instead of testing the returned value for ``None``: a JSON ``null`` in a record must
    not read as "the screen refused this field", and a refusal must not read as "it was null".
    """
    if text is None or isinstance(text, (bool, int, float)):
        return text, "not a string; carries no path"
    decoded, why = decode_wide_text(text)
    if decoded is None:
        return None, f"value could not be read as text and so could not be screened: {why}"
    if decoded in PUBLIC_PLACEHOLDERS:
        return text, "explicit public placeholder"
    found = _findings(decoded, allow_graph_names=allow_graph_names)
    if found:
        return None, "; ".join(found)
    if _PUBLIC_URL.match(decoded.strip()):
        return text, "scheme-qualified public URL, not a filesystem path"
    return text, "no filesystem root anchor"


def screen_public_record(obj: object, *, path: str = "$") -> "tuple[object | None, str]":
    """Walk a JSON-shaped object and refuse it if any value would publish a local path.

    TOTAL. The walk is where the ONNX-name exemption is applied, and it is applied by **key**:
    a value under a `GRAPH_NAME_KEYS` key may be POSIX-path-shaped, and a value anywhere else
    may not. That is the structural boundary the module docstring argues for — the text of
    ``/srv/models/minilm`` and the text of ``/encoder/layer.0/MatMul`` are indistinguishable, so
    the screen distinguishes their *positions* instead of guessing at their meanings.

    The scalar arm dispatches on type rather than on `screen_public_text`'s returned value,
    because a JSON ``null`` and a refusal both come back as ``None``. Conflating them would let
    a null field read as a leak and — far worse — let a leak read as a null field.
    """
    findings: list[str] = []

    def walk(node: object, where: str, allow_names: bool) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(key, str):
                    findings.append(f"{where}: non-string key {key!r}")
                    continue
                kept, why = screen_public_text(key, allow_graph_names=False)
                if kept is None:
                    findings.append(f"{where} key {key!r}: {why}")
                walk(value, f"{where}.{key}", key in GRAPH_NAME_KEYS)
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                walk(value, f"{where}[{i}]", allow_names)
        elif node is None or isinstance(node, (bool, int, float)):
            return
        elif isinstance(node, (str, bytes, bytearray, memoryview)):
            kept, why = screen_public_text(node, allow_graph_names=allow_names)
            if kept is None:
                findings.append(f"{where}: {why}")
        else:
            findings.append(
                f"{where}: value of type {type(node).__name__} is not JSON-shaped; this screen "
                f"cannot read it and will not publish a thing it has not read"
            )

    walk(obj, path, False)
    if findings:
        return None, "; ".join(findings)
    return obj, "no value in this record carries a filesystem root anchor"


def public_model_record(record: object, *, extra: "dict | None" = None) -> dict:
    """Serialise a `pinned_bytes.ProvenanceRecord` for publication, or raise.

    This is the production serializer — the one `bench/real_model.py` calls and the one the
    tests drive. It never receives a local path to redact, because it never asks for one: what
    it publishes is the pinned public identity (repo, revision, file, immutable URL), the
    observed digest and size, the derived verdict and the named disagreements. The screen then
    runs over its own output and raises `PrivatePathLeak` rather than returning a leaky dict,
    because an artifact writer that ignores a return value publishes it anyway.
    """
    to_dict = getattr(record, "to_dict", None)
    if not callable(to_dict):
        raise PrivatePathLeak(
            (f"{type(record).__name__} is not a provenance record; refusing to publish an "
             f"object this screen cannot account for",)
        )
    payload = dict(to_dict())
    if extra:
        for key, value in extra.items():
            if key in payload:
                raise PrivatePathLeak(
                    (f"extra field {key!r} would overwrite the record's own {key!r}; a "
                     f"published field that two writers can set is a field neither owns",)
                )
            payload[key] = value
    kept, why = screen_public_record(payload)
    if kept is None:
        raise PrivatePathLeak((why,))
    return payload
