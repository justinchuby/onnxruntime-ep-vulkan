"""One authority that decides whether the bytes on disk are the bytes that were pinned.

WHY THIS MODULE EXISTS (issue #78)
==================================
`bench/real_model.py` could resolve a model and report a provenance block, but nothing in it
ever *refused*. `resolve_model` hashed whatever it found and put the digest on the face of the
result next to ``agrees_with_recorded_provenance``, a field that was allowed to be ``None`` and
allowed to be ``False`` while the run continued. A default cache holding a MiniLM blob with
sha256 ``759c3cd2…`` (90,387,606 bytes) — a different export from a different repository — would
have been benchmarked, classified and published as "MiniLM" with nobody's assertion violated.

That is the whole defect: **"a file is here" was standing in for "these are the pinned bytes".**

WHAT THIS MODULE IS
===================
The single place where that question is answered. There is exactly one verifier
(`check_pinned_bytes`), exactly one identity type (`PinnedIdentity`), exactly one record type
(`ProvenanceRecord`), and one reader for untyped metadata (`read_pinned_identity`). Callers do
not re-implement any part of the decision — `bench/real_model.py::resolve_model` calls this and
reports what it is told.

THE RULE THAT MAKES A RECORD UNABLE TO LIE
------------------------------------------
``ProvenanceRecord.provenance_ok`` is **not a stored field**. It is a pure function of the other
recorded fields, recomputed on every read (see the property). It is therefore impossible for a
record to travel with ``provenance_ok = True`` while the metadata beside it disagrees: change the
recorded digest, the recorded size, the recorded source state, the sidecar digest or the external
scan, and the verdict changes with it. The class of defect where a summary field and its own
evidence disagree cannot be expressed here. A second reader (`bench/path_screen.py`) publishes
the record and cannot re-derive the verdict differently, because it does not re-derive it at all.

TOTALITY OF THE METADATA GATE
-----------------------------
`read_pinned_identity` is **total over arbitrary input**. It is handed mappings out of JSON, out
of test fixtures and out of hand-written specs, and it must never raise ``AttributeError`` or
``KeyError`` on any of them — an uncaught attribute error exits 1, which is the same exit code a
genuine provenance refusal uses, and a reader cannot tell the two apart. Every field is checked
for presence, type and shape, and *every* rejection returns a stable reason string. In
particular:

* ``pinned_bytes`` must be a positive ``int`` that is **not a ``bool``**. ``True`` is an
  ``int`` in Python and ``isinstance(True, int)`` is ``True``; a spec that said
  ``pinned_bytes: true`` would otherwise have pinned the size to 1 byte.
* ``revision`` must be 40 lowercase hex — a branch name is a mutable ref and pins nothing.
* ``sha256`` must be 64 lowercase hex. A truncated or upper-case digest is refused rather than
  normalised, because normalising input is how a screen starts accepting shapes nobody chose.

WHAT IT DELIBERATELY DOES NOT DO
================================
* **It does not reach the network.** It verifies bytes that are already on disk against an
  identity that was recorded elsewhere. Offline is not a failure mode here; it is the normal one.
* **It does not import onnxruntime and does not run inference.** Provenance is decided before
  anything is executed, or the thing that got executed is what decided.
* **It does not repair.** A disagreeing digest is refused, never re-pinned, never "updated".
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Iterator

_BENCH = Path(__file__).resolve().parent
for _p in (str(_BENCH),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Schema of the provenance block this module produces. Readers key off it.
PROVENANCE_SCHEMA = "pinned_bytes/1"

_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_REPO = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_REPO_BAD = re.compile(r"\.\.|[._-]/|/[._-]|[._-]\Z")

#: The only source states a verified record may carry. A state this module has never heard of
#: is refused rather than passed through: "verified" must mean one of a closed set of ways the
#: bytes got here, or it means "somebody wrote a word in a field".
VERIFIED_SOURCE_STATES = frozenset({"pinned-cache"})

#: Source states that are legitimate but are NOT evidence of pinned bytes. They exist so a
#: refusal can say which one it hit instead of collapsing to "not verified".
UNVERIFIED_SOURCE_STATES = frozenset({"unresolved", "offline", "download-failed", "unpinned"})

#: Recursion and work bounds for the graph walk. A malformed or hostile graph must not be able
#: to turn a provenance check into an unbounded traversal; exceeding a bound is a refusal, which
#: is the honest answer ("I could not finish looking"), not a pass.
MAX_GRAPH_DEPTH = 64
MAX_TENSORS_SCANNED = 2_000_000
#: Largest byte offset/length this module will accept from an `external_data` entry. ONNX stores
#: these as strings; a value beyond the signed 64-bit range is a malformed record, not a big file.
MAX_EXTENT = 2 ** 63 - 1

#: Windows reserved device names. `CON`, `NUL`, `COM1`… are not files, and opening one by
#: relative name from a model directory does not read the bytes anybody pinned.
_WIN_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


class ProvenanceError(RuntimeError):
    """The provenance question could not be answered, or was answered `no`.

    Carries a stable ``reason`` token so a caller can branch on the *kind* of refusal without
    parsing prose, and the ``record`` (when one exists) so a refusal can be published as
    evidence rather than as an absence.
    """

    def __init__(self, reason: str, detail: str, record: "ProvenanceRecord | None" = None):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.record = record


# --------------------------------------------------------------------------------------------
# The pinned identity, and the total reader that admits one
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PinnedIdentity:
    """The immutable identity of one model file: where it came from and what it must hash to.

    Constructed only through `read_pinned_identity`, which is where the validation lives. The
    dataclass re-validates in ``__post_init__`` so an identity cannot be assembled by hand — or
    by a future caller in a hurry — that the reader would have refused.
    """

    repo: str
    file: str
    revision: str
    sha256: str
    pinned_bytes: int
    source: str
    declared_external_files: int

    def __post_init__(self) -> None:
        ident, why = _validate_identity_fields(
            {
                "repo": self.repo,
                "file": self.file,
                "revision": self.revision,
                "sha256": self.sha256,
                "pinned_bytes": self.pinned_bytes,
                "source": self.source,
                "declared_external_files": self.declared_external_files,
            }
        )
        if ident is None:
            raise ProvenanceError("invalid_pin", why)

    @property
    def url(self) -> str:
        """The immutable public URL these bytes came from. No credentials, no local path."""
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file}"

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "file": self.file,
            "revision": self.revision,
            "sha256": self.sha256,
            "pinned_bytes": self.pinned_bytes,
            "source": self.source,
            "declared_external_files": self.declared_external_files,
            "url": self.url,
        }


_REQUIRED_FIELDS = (
    "repo",
    "file",
    "revision",
    "sha256",
    "pinned_bytes",
    "source",
    "declared_external_files",
)


def _text_field(raw: dict, name: str) -> "tuple[str | None, str]":
    if name not in raw:
        return None, f"{name} is absent; a pin with a missing field pins nothing"
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, str):
        return None, (
            f"{name} must be a str, got {type(value).__name__} ({value!r}); a pin read out of "
            f"JSON can be any type and the wrong one must be refused, not coerced"
        )
    if not value or not value.strip():
        return None, f"{name} is empty or whitespace-only ({value!r})"
    if value != value.strip():
        return None, (
            f"{name} carries surrounding whitespace ({value!r}); it is refused rather than "
            f"stripped, because a pin that is normalised on the way in is a pin somebody else "
            f"chose"
        )
    return value, ""


def _int_field(raw: dict, name: str, *, minimum: int) -> "tuple[int | None, str]":
    if name not in raw:
        return None, f"{name} is absent; a pin with a missing field pins nothing"
    value = raw[name]
    if isinstance(value, bool):
        return None, (
            f"{name} is a bool ({value!r}). `isinstance(True, int)` is True in Python, so a "
            f"spec saying `{name}: true` would otherwise pin the value to 1"
        )
    if not isinstance(value, int):
        return None, (
            f"{name} must be an int, got {type(value).__name__} ({value!r}); a float or a "
            f"decimal string is refused rather than converted"
        )
    if value < minimum:
        return None, f"{name} is {value}, which is below the minimum {minimum}"
    return value, ""


def _validate_identity_fields(raw: dict) -> "tuple[PinnedIdentity | None, str]":
    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS))
    if unknown:
        return None, (
            f"unknown pin field(s) {unknown}; an unrecognised key is refused because a typo in "
            f"a field name is otherwise a silently unpinned property"
        )

    repo, why = _text_field(raw, "repo")
    if repo is None:
        return None, why
    if not _REPO.match(repo) or _REPO_BAD.search(repo):
        return None, (
            f"repo {repo!r} is not `owner/name`; a bare name does not say which of several "
            f"re-exports of a model this is, and that ambiguity is issue #78 itself"
        )

    file, why = _text_field(raw, "file")
    if file is None:
        return None, why
    ok, why = _relative_posix_ok(file)
    if not ok:
        return None, f"file {file!r}: {why}"

    revision, why = _text_field(raw, "revision")
    if revision is None:
        return None, why
    if not _HEX40.match(revision):
        return None, (
            f"revision {revision!r} is not a 40-char lowercase commit sha. A branch or tag is a "
            f"mutable ref: it names whatever it points at today, which is the unpinned-source "
            f"pattern issue #78 was filed about"
        )

    digest, why = _text_field(raw, "sha256")
    if digest is None:
        return None, why
    if not _HEX64.match(digest):
        return None, (
            f"sha256 {digest!r} is not 64 lowercase hex chars. A truncated or upper-case digest "
            f"is refused rather than normalised"
        )

    size, why = _int_field(raw, "pinned_bytes", minimum=1)
    if size is None:
        return None, why

    source, why = _text_field(raw, "source")
    if source is None:
        return None, why
    if source not in VERIFIED_SOURCE_STATES and source not in UNVERIFIED_SOURCE_STATES:
        return None, (
            f"source {source!r} is not one of the states this module knows "
            f"({sorted(VERIFIED_SOURCE_STATES | UNVERIFIED_SOURCE_STATES)}); an unrecognised "
            f"source state is refused rather than treated as a verified one"
        )

    external, why = _int_field(raw, "declared_external_files", minimum=0)
    if external is None:
        return None, why

    ident = PinnedIdentity.__new__(PinnedIdentity)
    object.__setattr__(ident, "repo", repo)
    object.__setattr__(ident, "file", file)
    object.__setattr__(ident, "revision", revision)
    object.__setattr__(ident, "sha256", digest)
    object.__setattr__(ident, "pinned_bytes", size)
    object.__setattr__(ident, "source", source)
    object.__setattr__(ident, "declared_external_files", external)
    return ident, "pin is total: repo, file, revision, sha256, size, source and external count"


def read_pinned_identity(raw: object) -> "tuple[PinnedIdentity | None, str]":
    """Admit an untyped mapping as a `PinnedIdentity`, or refuse it with a reason.

    TOTAL: never raises, for any input whatsoever. The refusal path is the interesting one —
    every metadata defect enumerated in issue #78 (absent field, empty string, whitespace, wrong
    type, mutable revision, short digest, ``pinned_bytes`` of 0/False/None) arrives here as a
    ``(None, why)`` and can therefore never reach `check_pinned_bytes`, which is the only thing
    that can set ``provenance_ok``.
    """
    if raw is None:
        return None, "pin is absent (None); an unpinned model is refused, not benchmarked"
    if isinstance(raw, PinnedIdentity):
        return raw, "already a validated PinnedIdentity"
    if not isinstance(raw, dict):
        return None, (
            f"pin must be a mapping, got {type(raw).__name__} ({raw!r}); a sequence or a string "
            f"cannot carry named fields and is refused rather than indexed"
        )
    if not all(isinstance(k, str) for k in raw):
        return None, "pin has non-string key(s); it did not come from a JSON object"
    try:
        return _validate_identity_fields(dict(raw))
    except Exception as exc:  # pragma: no cover - the point is that nothing escapes
        return None, f"pin could not be read: {type(exc).__name__}: {exc}"


def read_sidecar_sha256(path: "Path | str | None") -> "tuple[str | None, str]":
    """The sha256 an independent tool recorded for this model, or a refusal.

    TOTAL. Read from `rust/modelrunner`'s artifact rather than from a constant in this file: a
    digest this module both writes and checks proves nothing. The sidecar is a **required**
    second witness — `check_pinned_bytes` refuses when it is absent — because "the pin agrees
    with itself" is the tautology issue #78's `agrees_with_recorded_provenance = None` field
    already shipped.
    """
    import json

    if path is None:
        return None, "no sidecar path was given; a pin with one witness has no second opinion"
    try:
        p = Path(path)
    except Exception as exc:
        return None, f"sidecar path is unusable: {type(exc).__name__}: {exc}"
    if not p.is_file():
        return None, f"sidecar {p.name} does not exist; the second witness is absent"
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"sidecar {p.name} is not readable JSON: {type(exc).__name__}: {exc}"
    if not isinstance(blob, dict):
        return None, f"sidecar {p.name} is a {type(blob).__name__}, not a JSON object"
    value = blob.get("onnx_sha256")
    if value is None:
        return None, f"sidecar {p.name} carries no onnx_sha256 field"
    if isinstance(value, bool) or not isinstance(value, str):
        return None, (
            f"sidecar {p.name} onnx_sha256 is {type(value).__name__} ({value!r}), not a string"
        )
    if not _HEX64.match(value):
        return None, (
            f"sidecar {p.name} onnx_sha256 {value!r} is not 64 lowercase hex chars"
        )
    return value, f"sidecar {p.name} recorded onnx_sha256 {value[:12]}…"


# --------------------------------------------------------------------------------------------
# Complete traversal of every tensor-bearing container in an ONNX model
# --------------------------------------------------------------------------------------------


def _tensor_bearing_attribute_fields() -> tuple:
    """(single, repeated) attribute field names that can hold a tensor or a subgraph."""
    return (
        ("t", "sparse_tensor", "g"),
        ("tensors", "sparse_tensors", "graphs"),
    )


def iter_tensor_protos(model: object, *, max_depth: int = MAX_GRAPH_DEPTH,
                       max_tensors: int = MAX_TENSORS_SCANNED) -> Iterator[tuple]:
    """Yield ``(path, TensorProto)`` for **every** tensor anywhere in an ONNX model.

    A partial walk is the failure mode this replaces. Hashing only ``graph.initializer`` — which
    is what `bench/real_model.py::external_data_provenance` did — misses an external-data
    declaration hiding in an ``If`` branch, in a ``Loop`` body, in a sparse initializer's index
    tensor, in a local function's body, or in a training-info graph. Each of those is a place
    where a model can name a blob on disk that the check never looked at, and "we did not look
    there" reported as "there is nothing there" is precisely the substitution issue #78 names.

    Covered, exhaustively:

    * ``model.graph``: ``initializer``; ``sparse_initializer`` (**both** the ``values`` tensor
      and the ``indices`` tensor — the indices carry bytes too, and can be external);
    * every ``node.attribute`` in every graph: ``t``/``tensors``, ``sparse_tensor``/
      ``sparse_tensors``, and ``g``/``graphs``, recursively, so nested and repeated subgraphs
      are reached at any depth;
    * ``model.functions``: each ``FunctionProto``'s node attributes **and** its
      ``attribute_proto`` default values, which are ``AttributeProto``s that can themselves
      carry tensors and subgraphs;
    * ``model.training_info``: each ``TrainingInfoProto``'s ``initialization`` and
      ``algorithm`` graphs.

    Bounded: ``max_depth`` on nesting and ``max_tensors`` on total work. Exceeding either raises
    ``ProvenanceError`` — refusing to finish is a refusal, and a refusal is not a pass. A member
    of the wrong type (a graph where a tensor was declared) raises rather than being skipped.
    """
    if model is None:
        raise ProvenanceError("malformed_graph", "model is None; there is nothing to traverse")
    seen = [0]

    def _tensor(where: str, obj: object):
        if obj is None:
            return
        if not hasattr(obj, "data_location") and not hasattr(obj, "external_data"):
            raise ProvenanceError(
                "malformed_graph",
                f"{where} is {type(obj).__name__}, which is not a TensorProto; a container "
                f"holding the wrong type is a graph this module cannot account for",
            )
        seen[0] += 1
        if seen[0] > max_tensors:
            raise ProvenanceError(
                "traversal_bounded",
                f"more than {max_tensors} tensors while scanning {where}; the walk is bounded "
                f"so a malformed graph cannot turn a provenance check into an unbounded one",
            )
        yield_list.append((where, obj))

    yield_list: list = []

    def _attribute(where: str, attr: object, depth: int):
        _, repeated = _tensor_bearing_attribute_fields()
        name = getattr(attr, "name", "?")
        tensor = getattr(attr, "t", None)
        if _tensor_is_set(tensor):
            _tensor(f"{where}.{name}.t", tensor)
        sparse = getattr(attr, "sparse_tensor", None)
        if _sparse_is_set(sparse):
            _sparse(f"{where}.{name}.sparse_tensor", sparse)
        subgraph = getattr(attr, "g", None)
        if _graph_is_set(attr, "g"):
            _graph(f"{where}.{name}.g", subgraph, depth + 1)
        for field in repeated:
            values = getattr(attr, field, None)
            if not values:
                continue
            for i, value in enumerate(values):
                if field == "graphs":
                    _graph(f"{where}.{name}.graphs[{i}]", value, depth + 1)
                elif field == "sparse_tensors":
                    _sparse(f"{where}.{name}.sparse_tensors[{i}]", value)
                else:
                    _tensor(f"{where}.{name}.tensors[{i}]", value)

    def _sparse(where: str, sp: object):
        values = getattr(sp, "values", None)
        indices = getattr(sp, "indices", None)
        if values is None and indices is None:
            raise ProvenanceError(
                "malformed_graph",
                f"{where} is {type(sp).__name__}, which is not a SparseTensorProto",
            )
        _tensor(f"{where}.values", values)
        _tensor(f"{where}.indices", indices)

    def _graph(where: str, graph: object, depth: int):
        if depth > max_depth:
            raise ProvenanceError(
                "traversal_bounded",
                f"graph nesting deeper than {max_depth} at {where}; the walk is bounded so a "
                f"cyclic or hostile graph cannot exhaust the stack",
            )
        if graph is None:
            return
        if not hasattr(graph, "node"):
            raise ProvenanceError(
                "malformed_graph",
                f"{where} is {type(graph).__name__}, which is not a GraphProto",
            )
        for i, init in enumerate(getattr(graph, "initializer", ()) or ()):
            _tensor(f"{where}.initializer[{i}]", init)
        for i, sp in enumerate(getattr(graph, "sparse_initializer", ()) or ()):
            _sparse(f"{where}.sparse_initializer[{i}]", sp)
        for n, node in enumerate(getattr(graph, "node", ()) or ()):
            for attr in getattr(node, "attribute", ()) or ():
                _attribute(f"{where}.node[{n}]", attr, depth)

    def _function(where: str, fn: object, depth: int):
        for n, node in enumerate(getattr(fn, "node", ()) or ()):
            for attr in getattr(node, "attribute", ()) or ():
                _attribute(f"{where}.node[{n}]", attr, depth)
        for a, attr in enumerate(getattr(fn, "attribute_proto", ()) or ()):
            _attribute(f"{where}.attribute_proto[{a}]", attr, depth)

    graph = getattr(model, "graph", None)
    if graph is None:
        raise ProvenanceError("malformed_graph", "model carries no graph")
    _graph("graph", graph, 0)
    for f, fn in enumerate(getattr(model, "functions", ()) or ()):
        _function(f"functions[{f}]", fn, 0)
    for t, info in enumerate(getattr(model, "training_info", ()) or ()):
        init = getattr(info, "initialization", None)
        if init is not None and getattr(init, "node", None) is not None:
            _graph(f"training_info[{t}].initialization", init, 0)
        algo = getattr(info, "algorithm", None)
        if algo is not None and getattr(algo, "node", None) is not None:
            _graph(f"training_info[{t}].algorithm", algo, 0)

    for item in yield_list:
        yield item


def _tensor_is_set(value: object) -> bool:
    """A protobuf singular message field is always present; `name`/`dims`/data tell us."""
    if value is None:
        return False
    return bool(
        getattr(value, "name", "")
        or getattr(value, "dims", None)
        or getattr(value, "raw_data", b"")
        or getattr(value, "external_data", None)
        or getattr(value, "data_type", 0)
    )


def _sparse_is_set(value: object) -> bool:
    if value is None:
        return False
    return _tensor_is_set(getattr(value, "values", None)) or _tensor_is_set(
        getattr(value, "indices", None)
    )


def _graph_is_set(attr: object, field: str) -> bool:
    graph = getattr(attr, field, None)
    if graph is None:
        return False
    return bool(
        getattr(graph, "node", None)
        or getattr(graph, "initializer", None)
        or getattr(graph, "sparse_initializer", None)
        or getattr(graph, "name", "")
    )


# --------------------------------------------------------------------------------------------
# External data: confinement, stable identity, and deterministic extent hashing
# --------------------------------------------------------------------------------------------


def _relative_posix_ok(location: object) -> "tuple[bool, str]":
    """Is *location* a plain relative POSIX path with no way out of its root?"""
    if isinstance(location, bool) or not isinstance(location, str):
        return False, f"must be a str, got {type(location).__name__} ({location!r})"
    if not location or not location.strip():
        return False, "is empty or whitespace-only"
    if "\x00" in location:
        return False, "contains a NUL byte"
    if any(ord(c) < 0x20 for c in location):
        return False, "contains a control character"
    if "\\" in location:
        return False, (
            "contains a backslash. ONNX external-data locations are POSIX-relative; a backslash "
            "is a separator on one platform and a filename character on another, and a location "
            "that means two different files is not a location"
        )
    if re.match(r"\A[A-Za-z][A-Za-z0-9+.\-]*:", location):
        return False, (
            "is URI-like (has a scheme). External data is a file beside the model, never a "
            "thing this process will fetch"
        )
    if location.startswith("/"):
        return False, "is an absolute POSIX path; external data must live under the model root"
    if re.match(r"\A[A-Za-z]:", location):
        return False, "carries a drive letter; external data must live under the model root"
    if location.startswith("//"):
        return False, "is a UNC-style path; external data must live under the model root"
    parts = location.split("/")
    if any(p == "" for p in parts):
        return False, (
            "has an empty path segment (a doubled or trailing separator); it is refused rather "
            "than collapsed, because collapsing is a normalisation nobody chose"
        )
    if any(p == ".." for p in parts):
        return False, "contains `..`, which is a traversal out of the model root"
    if any(p == "." for p in parts):
        return False, "contains `.`, which is refused rather than collapsed"
    for p in parts:
        base = p.split(".")[0].lower()
        if base in _WIN_DEVICE_NAMES:
            return False, (
                f"segment {p!r} is a Windows reserved device name; opening it reads a device, "
                f"not the bytes anybody pinned"
            )
        if p.endswith(" ") or p.endswith("."):
            return False, (
                f"segment {p!r} ends with a space or a dot, which Windows silently strips — two "
                f"different declared names would then open one file"
            )
    return True, "relative POSIX path with no traversal, no scheme, no root anchor"


def _has_reparse_point(path: Path) -> bool:
    try:
        st = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attrs & reparse)


def _reparse_component(path: Path, root: Path) -> "str | None":
    """The first component of *path* below *root* that is a reparse point, or ``None``.

    Checking only the final component is not enough. ``weights/w.bin`` where ``weights`` is a
    junction resolves wherever ``weights`` points; ``w.bin`` itself is an ordinary file at the
    far end of it and answers "no" to every question asked about it alone.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    candidate = root
    for part in rel.parts:
        candidate = candidate / part
        if _has_reparse_point(candidate):
            return part
    return None


def confine_external_location(location: object, *, model_root: "Path | str"
                              ) -> "tuple[Path | None, str]":
    """Resolve one declared external-data location under *model_root*, or refuse it.

    TOTAL. This is the confinement gate, and it refuses far more than `..`:

    * anything that is not a plain relative POSIX path (`_relative_posix_ok`): a URI, an
      absolute POSIX path, a drive letter, a UNC or device path, a backslash, an empty or
      traversing segment, a Windows reserved device name, a trailing-space/dot segment;
    * a path whose **resolved** location escapes the model root even though its text did not —
      the symlink, junction and reparse-point case, which is the interesting one, because
      `a/b.data` is a perfectly innocent-looking location when `a` is a junction to `C:\\`.

    Every component from the root down is checked for a reparse point, not only the final one:
    an escape one directory up is an escape.
    """
    ok, why = _relative_posix_ok(location)
    if not ok:
        return None, f"external location {location!r} {why}"
    try:
        root = Path(model_root).resolve(strict=False)
    except Exception as exc:
        return None, f"model root {model_root!r} is unusable: {type(exc).__name__}: {exc}"
    if not root.is_dir():
        return None, f"model root {root.name!r} is not a directory"

    rel = PurePosixPath(str(location))
    part = _reparse_component(root / Path(*rel.parts), root) if rel.parts else None
    if part is not None:
        return None, (
            f"external location {location!r} passes through {part!r}, which is a symlink, "
            f"junction or other reparse point. A confinement check that only compares text "
            f"is satisfied by a name that points anywhere on the machine"
        )
    candidate = root
    for p in rel.parts:
        candidate = candidate / p
    try:
        resolved = candidate.resolve(strict=False)
    except Exception as exc:
        return None, f"external location {location!r} could not be resolved: {exc}"
    try:
        resolved.relative_to(root)
    except ValueError:
        return None, (
            f"external location {location!r} resolves to somewhere outside the model root; the "
            f"declared name stayed inside and the resolved path did not"
        )
    return resolved, f"external location {location!r} is confined under the model root"


@dataclasses.dataclass(frozen=True)
class ExternalRef:
    """One declared external byte extent: which file, which bytes of it, for which tensor."""

    tensor: str
    where: str
    location: str
    offset: int
    length: "int | None"


def external_references(model: object, *, model_root: "Path | str") -> "list[ExternalRef]":
    """Every external byte extent the model declares, validated, or raise.

    Refuses, rather than skipping, when a tensor's ``external_data`` is not exactly one
    non-empty ``location`` plus optional well-formed ``offset``/``length``:

    * **duplicate keys** — protobuf permits repeating ``key: "location"``, and a reader that
      takes the first while an ORT build takes the last is reading a different file from the one
      it verified;
    * a missing, empty or whitespace ``location``;
    * an ``offset``/``length`` that is not a base-10 integer, is negative, or exceeds the signed
      64-bit range (an out-of-range extent is a malformed record, not a large file);
    * a location that does not survive `confine_external_location`.
    """
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - onnx is present in this repo's venv
        raise ProvenanceError(
            "onnx_unavailable",
            f"the onnx package is not importable ({exc}); external weights cannot be scanned, "
            f"and UNSCANNED is not a synonym for absent",
        ) from exc

    external_enum = onnx.TensorProto.EXTERNAL
    refs: list[ExternalRef] = []
    for where, tensor in iter_tensor_protos(model):
        location_field = getattr(tensor, "data_location", None)
        if location_field != external_enum:
            continue
        name = getattr(tensor, "name", "") or "<unnamed>"
        pairs = list(getattr(tensor, "external_data", ()) or ())
        keys = [getattr(kv, "key", "") for kv in pairs]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ProvenanceError(
                "external_malformed",
                f"tensor {name!r} at {where} repeats external_data key(s) {dupes}. Which value "
                f"wins is a reader's choice, so two readers can verify and load different bytes",
            )
        kv = {getattr(p, "key", ""): getattr(p, "value", "") for p in pairs}
        location = kv.get("location")
        if location is None or not isinstance(location, str) or not location.strip():
            raise ProvenanceError(
                "external_malformed",
                f"tensor {name!r} at {where} declares EXTERNAL data with no usable `location` "
                f"({location!r}); a declared-but-unnamed blob can never be verified",
            )
        offset = _extent_int(kv.get("offset"), name=f"{name}.offset", default=0)
        length = (
            None if kv.get("length") in (None, "")
            else _extent_int(kv.get("length"), name=f"{name}.length", default=0)
        )
        resolved, why = confine_external_location(location, model_root=model_root)
        if resolved is None:
            raise ProvenanceError("external_unsafe", why)
        refs.append(ExternalRef(tensor=name, where=where, location=location,
                                offset=offset, length=length))
    return refs


def _extent_int(value: object, *, name: str, default: int) -> int:
    if value in (None, ""):
        return default
    if not isinstance(value, str):
        raise ProvenanceError(
            "external_malformed",
            f"{name} is {type(value).__name__} ({value!r}); ONNX stores external offsets and "
            f"lengths as decimal strings",
        )
    text = value.strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ProvenanceError(
            "external_malformed",
            f"{name} is {value!r}, which is not a non-negative base-10 integer. A negative or "
            f"hex or floating value is refused rather than coerced — a negative offset seeks "
            f"backwards from somewhere",
        )
    parsed = int(text)
    if parsed > MAX_EXTENT:
        raise ProvenanceError(
            "external_malformed",
            f"{name} is {parsed}, beyond the signed 64-bit range; that is a malformed record, "
            f"not a large file",
        )
    return parsed


def open_stable_file(path: "Path | str", *, model_root: "Path | str"):
    """Open a confined file and return ``(fd, stat_result)`` whose identity has been checked.

    The confinement check in `confine_external_location` runs against the *name*. Between that
    check and the read, the name can be replaced — the classic time-of-check/time-of-use race,
    and on Windows a directory junction can be swapped in with no elevated privileges at all.

    So the bytes are read from a **file descriptor**, and the descriptor's own ``fstat``
    identity (``st_dev``/``st_ino``, populated on Windows too) must match the ``lstat`` of the
    name that was checked. If they differ, the thing that was opened is not the thing that was
    validated and this raises. ``O_NOFOLLOW`` is added where the platform has it; where it does
    not, the identity comparison is the portable half that still holds.
    """
    p = Path(path)
    root = Path(model_root).resolve(strict=False)
    try:
        p.resolve(strict=False).relative_to(root)
    except ValueError:
        raise ProvenanceError(
            "external_unsafe",
            f"{p.name} is not under the model root; opening it was refused before any read",
        ) from None
    if _reparse_component(p, root) is not None:
        raise ProvenanceError(
            "external_unsafe",
            f"{p.name} is reached through {_reparse_component(p, root)!r}, a symlink, junction "
            f"or reparse point, at open time",
        )
    try:
        before = p.lstat()
    except OSError as exc:
        raise ProvenanceError(
            "external_missing",
            f"{p.name} could not be stat'd: {exc.strerror or exc}; declared external bytes that "
            f"are not there must never verify",
        ) from exc
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(p), flags)
    except OSError as exc:
        raise ProvenanceError(
            "external_missing",
            f"{p.name} could not be opened: {exc.strerror or exc}",
        ) from exc
    try:
        after = os.fstat(fd)
    except OSError as exc:  # pragma: no cover - fstat on a live fd
        os.close(fd)
        raise ProvenanceError("external_missing", f"{p.name} could not be fstat'd: {exc}") from exc
    if not stat.S_ISREG(after.st_mode):
        os.close(fd)
        raise ProvenanceError(
            "external_unsafe",
            f"{p.name} is not a regular file; a device, pipe or directory does not hold pinned "
            f"bytes",
        )
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) and after.st_ino:
        os.close(fd)
        raise ProvenanceError(
            "external_unsafe",
            f"{p.name} changed identity between the confinement check and the open "
            f"({before.st_dev}/{before.st_ino} -> {after.st_dev}/{after.st_ino}); the file that "
            f"was validated is not the file that was opened",
        )
    return fd, after


def hash_external_refs(refs: "list[ExternalRef]", *, model_root: "Path | str") -> dict:
    """Hash every declared external byte extent, deterministically, or raise.

    Deterministic means: extents are hashed in sorted ``(location, offset, length)`` order and
    folded into one digest over the *content* plus the extent geometry, so the same bytes under
    two orderings produce the same combined digest and the same bytes at a different offset do
    not. Per-file digests are kept beside it, because a combined value alone cannot say which
    blob moved.

    A declared extent that runs past the end of its file raises: a short read that silently
    hashes fewer bytes is exactly "missing external bytes verified anyway".
    """
    root = Path(model_root).resolve(strict=False)
    files: dict[str, dict] = {}
    combined = hashlib.sha256()
    for ref in sorted(refs, key=lambda r: (r.location, r.offset, r.length if r.length else -1)):
        resolved, why = confine_external_location(ref.location, model_root=root)
        if resolved is None:
            raise ProvenanceError("external_unsafe", why)
        fd, st = open_stable_file(resolved, model_root=root)
        try:
            end = st.st_size if ref.length is None else ref.offset + ref.length
            if ref.offset > st.st_size or end > st.st_size:
                raise ProvenanceError(
                    "external_short",
                    f"{ref.location} declares bytes [{ref.offset}, {end}) but the file holds "
                    f"{st.st_size}; declared external bytes that are not present must never "
                    f"verify",
                )
            digest = _hash_fd_extent(fd, ref.offset, end - ref.offset)
        finally:
            os.close(fd)
        combined.update(f"{ref.location}\0{ref.offset}\0{end - ref.offset}\0{digest}\0".encode())
        entry = files.setdefault(
            ref.location, {"location": ref.location, "bytes": st.st_size, "extents": []}
        )
        entry["extents"].append(
            {"offset": ref.offset, "length": end - ref.offset, "sha256": digest}
        )
    return {
        "scanned": True,
        "files": [files[k] for k in sorted(files)],
        "combined_sha256": combined.hexdigest() if files else None,
        "extents": sum(len(f["extents"]) for f in files.values()),
    }


def _hash_fd_extent(fd: int, offset: int, length: int) -> str:
    h = hashlib.sha256()
    os.lseek(fd, offset, os.SEEK_SET)
    remaining = length
    while remaining > 0:
        chunk = os.read(fd, min(1 << 20, remaining))
        if not chunk:
            raise ProvenanceError(
                "external_short",
                f"the file ended {remaining} byte(s) before the declared extent did",
            )
        h.update(chunk)
        remaining -= len(chunk)
    return h.hexdigest()


def _sha256_and_size(path: Path) -> "tuple[str, int]":
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# --------------------------------------------------------------------------------------------
# The record, whose verdict is derived and therefore cannot disagree with itself
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProvenanceRecord:
    """What was pinned, what was found, and — derived from both — whether they agree."""

    identity: PinnedIdentity
    observed_sha256: str
    observed_bytes: int
    source_state: str
    sidecar_sha256: "str | None"
    external: dict
    checked_at_schema: str = PROVENANCE_SCHEMA

    @property
    def provenance_ok(self) -> bool:
        """Recomputed from the recorded fields on every read. Never stored.

        This is the property that makes gate 2 structural rather than aspirational: there is no
        assignment anywhere in this repository that can set a record's verdict, so no record can
        be published claiming agreement while the digest, the size, the source state, the second
        witness or the external scan beside it says otherwise.
        """
        return (
            isinstance(self.identity, PinnedIdentity)
            and isinstance(self.observed_sha256, str)
            and _HEX64.match(self.observed_sha256) is not None
            and self.observed_sha256 == self.identity.sha256
            and isinstance(self.observed_bytes, int)
            and not isinstance(self.observed_bytes, bool)
            and self.observed_bytes == self.identity.pinned_bytes
            and self.source_state in VERIFIED_SOURCE_STATES
            and self.source_state == self.identity.source
            and isinstance(self.sidecar_sha256, str)
            and self.sidecar_sha256 == self.identity.sha256
            and isinstance(self.external, dict)
            and self.external.get("scanned") is True
            and len(self.external.get("files") or ()) == self.identity.declared_external_files
        )

    @property
    def disagreements(self) -> "tuple[str, ...]":
        """Every reason `provenance_ok` is False, named. Empty exactly when it is True."""
        out: list[str] = []
        if not isinstance(self.identity, PinnedIdentity):
            return ("no validated pin was admitted",)
        if self.observed_sha256 != self.identity.sha256:
            out.append(
                f"sha256 mismatch: pinned {self.identity.sha256}, found {self.observed_sha256}"
            )
        if self.observed_bytes != self.identity.pinned_bytes:
            out.append(
                f"size mismatch: pinned {self.identity.pinned_bytes} bytes, found "
                f"{self.observed_bytes}"
            )
        if self.source_state not in VERIFIED_SOURCE_STATES:
            out.append(
                f"source state {self.source_state!r} is not one that carries verified bytes "
                f"({sorted(VERIFIED_SOURCE_STATES)})"
            )
        elif self.source_state != self.identity.source:
            out.append(
                f"source state {self.source_state!r} disagrees with the pin's {self.identity.source!r}"
            )
        if not isinstance(self.sidecar_sha256, str):
            out.append(
                "no independently recorded sha256 (sidecar); a pin that agrees only with itself "
                "is not a second opinion"
            )
        elif self.sidecar_sha256 != self.identity.sha256:
            out.append(
                f"sidecar sha256 {self.sidecar_sha256} disagrees with the pin "
                f"{self.identity.sha256}"
            )
        if not isinstance(self.external, dict) or self.external.get("scanned") is not True:
            out.append("external-data scan did not complete; UNSCANNED is not a synonym for none")
        else:
            found = len(self.external.get("files") or ())
            if found != self.identity.declared_external_files:
                out.append(
                    f"external data disagrees with the pin: {found} file(s) found, "
                    f"{self.identity.declared_external_files} declared"
                )
        return tuple(out)

    def to_dict(self) -> dict:
        """The record as it lands in an artifact. Carries no local path — see `path_screen`."""
        return {
            "schema": self.checked_at_schema,
            "pin": self.identity.to_dict(),
            "observed_sha256": self.observed_sha256,
            "observed_bytes": self.observed_bytes,
            "source_state": self.source_state,
            "sidecar_sha256": self.sidecar_sha256,
            "external_data": self.external,
            "provenance_ok": self.provenance_ok,
            "disagreements": list(self.disagreements),
        }


def check_pinned_bytes(path: "Path | str", pin: object, *,
                       sidecar: "Path | str | None" = None,
                       source_state: str = "pinned-cache") -> ProvenanceRecord:
    """Decide whether the bytes at *path* are the bytes *pin* names. The only such decision.

    Raises `ProvenanceError` — carrying a stable ``reason`` and, where one exists, the refusal
    ``record`` — on every disagreement. Returns a record whose derived ``provenance_ok`` is
    ``True`` only after **all** of: the pin was admitted as total and typed; the file exists and
    is a regular file; its bytes hash to the pinned digest; its size equals the pinned size; an
    independently recorded sidecar digest exists and equals the pinned digest; the complete
    graph traversal ran; and the number of external-data files found equals the number declared,
    with every declared extent confined, opened by stable identity and hashed.

    Order matters and is deliberate: the pin is admitted *first*, so a malformed pin is refused
    before a single byte is read, and no expensive or fallible step can be mistaken for the
    reason a bad pin failed.
    """
    identity, why = read_pinned_identity(pin)
    if identity is None:
        raise ProvenanceError("invalid_pin", why)
    if identity.source not in VERIFIED_SOURCE_STATES:
        raise ProvenanceError(
            "unpinned_source",
            f"the pin's own source state is {identity.source!r}, which does not carry verified "
            f"bytes; it is refused here rather than reported as an unverified pass",
        )
    if not isinstance(source_state, str) or source_state not in VERIFIED_SOURCE_STATES:
        raise ProvenanceError(
            "unpinned_source",
            f"resolved source state {source_state!r} is not one that carries verified bytes "
            f"({sorted(VERIFIED_SOURCE_STATES)}); offline, a failed download and an unpinned "
            f"cache are all UNAVAILABLE, never success-shaped",
        )

    p = Path(path)
    if not p.exists():
        raise ProvenanceError("model_missing", f"{p.name} does not exist under the model root")
    if not p.is_file():
        raise ProvenanceError("model_missing", f"{p.name} is not a regular file")

    observed_sha, observed_bytes = _sha256_and_size(p)

    sidecar_sha, sidecar_why = read_sidecar_sha256(sidecar)

    # Cheapest and most specific first. A file whose digest already disagrees with the pin is
    # not "a graph that failed to parse" — reporting it that way would name the wrong finding,
    # and a reader chasing a parser bug is a reader not chasing a substituted model.
    if observed_sha != identity.sha256 or observed_bytes != identity.pinned_bytes:
        record = ProvenanceRecord(
            identity=identity,
            observed_sha256=observed_sha,
            observed_bytes=observed_bytes,
            source_state=source_state,
            sidecar_sha256=sidecar_sha,
            external={"scanned": False, "files": [], "combined_sha256": None, "extents": 0,
                      "reason": "not scanned: the file's own digest already disagrees"},
        )
        raise ProvenanceError(
            "provenance_mismatch", "; ".join(record.disagreements), record=record
        )

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover
        raise ProvenanceError(
            "onnx_unavailable",
            f"the onnx package is not importable ({exc}); the graph cannot be traversed and an "
            f"unscanned graph is not a scanned one",
        ) from exc
    try:
        model = onnx.load(str(p), load_external_data=False)
    except Exception as exc:
        raise ProvenanceError(
            "malformed_graph", f"{p.name} could not be parsed as ONNX: {type(exc).__name__}: {exc}"
        ) from exc

    refs = external_references(model, model_root=p.parent)
    external = hash_external_refs(refs, model_root=p.parent)

    record = ProvenanceRecord(
        identity=identity,
        observed_sha256=observed_sha,
        observed_bytes=observed_bytes,
        source_state=source_state,
        sidecar_sha256=sidecar_sha,
        external=external,
    )
    if not record.provenance_ok:
        detail = "; ".join(record.disagreements) or "the derived verdict is False"
        if sidecar_sha is None:
            detail = f"{detail} [{sidecar_why}]"
        raise ProvenanceError("provenance_mismatch", detail, record=record)
    return record
