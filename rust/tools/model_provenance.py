#!/usr/bin/env python
"""The model provenance contract: URL, byte size, and SHA-256 for every public model this repo
uses as a differential-testing input. Owner: Trinity.

WHY THIS EXISTS
----------------
`bench/results/model_provenance.json` has recorded the URL and pinned SHA-256 for MobileNetV2-12
and BERT-SQuAD-12 since 2026-08-03/04, but nothing in the repo ever *read* it programmatically --
it was prose evidence in `docs/OP_COVERAGE.md`, not a contract anything enforced. A model file on
disk and the provenance record describing it could silently diverge (wrong download, partial
download, a stale re-fetch from a moved upstream URL) and nothing would notice. This module makes
the JSON file an actual contract: `load_provenance` reads it, `verify_file` enforces it, and
`tests/ops/test_small_model_provenance.py` calls both before any differential test trusts a model
file it did not download itself in-process.

CONTRACT SHAPE
---------------
`bench/results/model_provenance.json` is `{"models": [{"name", "url", "sha256", "bytes",
"fetched", "why"}, ...]}`. `name` is the key tests look models up by (e.g. ``"mnist-12"``); it is
not required to match the on-disk filename, though by convention it does (`<name>.onnx`).

NO CLOCK. Identity and integrity only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO / "bench" / "results" / "model_provenance.json"

_CHUNK_SIZE = 1 << 20  # 1 MiB; MobileNetV2 is ~14MB, BERT-SQuAD is ~436MB -- stream, don't slurp.


class ProvenanceMismatch(RuntimeError):
    """A file on disk does not match its `model_provenance.json` entry, or the entry is missing."""


@dataclasses.dataclass(frozen=True)
class ModelProvenance:
    name: str
    url: str
    sha256: str
    bytes: int
    fetched: str
    why: str


def load_provenance(path: pathlib.Path = DEFAULT_PATH) -> "dict[str, ModelProvenance]":
    """Reads the contract, keyed by ``name``. Raises if the file is missing or malformed --
    the contract not existing is never treated the same as "no models are provenance-pinned"."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, ModelProvenance] = {}
    for row in doc["models"]:
        entry = ModelProvenance(
            name=row["name"],
            url=row["url"],
            sha256=row["sha256"],
            bytes=row["bytes"],
            fetched=row["fetched"],
            why=row["why"],
        )
        out[entry.name] = entry
    return out


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: pathlib.Path, entry: ModelProvenance) -> None:
    """Raises ``ProvenanceMismatch`` with actionable detail unless ``path`` matches ``entry``
    exactly in both byte size and SHA-256. Size is checked first and reported on its own --
    a truncated download most often fails on size alone, and a size-only report is enough
    to tell a reader "the download did not finish" without waiting for a full hash pass."""
    if not path.is_file():
        raise ProvenanceMismatch(f"{entry.name}: expected a file at {path}, found none")
    actual_bytes = path.stat().st_size
    if actual_bytes != entry.bytes:
        raise ProvenanceMismatch(
            f"{entry.name}: size mismatch at {path}: expected {entry.bytes} bytes, got "
            f"{actual_bytes} bytes. Re-download from {entry.url}."
        )
    actual_sha256 = sha256_of(path)
    if actual_sha256 != entry.sha256:
        raise ProvenanceMismatch(
            f"{entry.name}: SHA-256 mismatch at {path}: expected {entry.sha256}, got "
            f"{actual_sha256}. Size matched but the contents did not -- this file does not "
            f"match the pinned provenance contract "
            f"({DEFAULT_PATH.relative_to(REPO)}); re-download from {entry.url} rather than "
            f"trusting the local copy."
        )
