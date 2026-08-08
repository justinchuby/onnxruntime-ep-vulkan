"""Model provenance for the CUDA-competition suite (issue #69). Owner: Niobe.

WHY THIS EXISTS
---------------
Issue #69 asks whether the Vulkan EP beats ORT's CUDA EP.  Every arm of that
comparison must run *the same bytes*.  "The same model" is not a property of a
name, a URL, or a Hugging Face repo id — all three are mutable — it is a
property of a digest.  This module resolves each benchmark model to a concrete
path plus a ``sha256`` of every file that participates (the graph proto **and**
its external-data blobs), and refuses rather than substitutes.

Three refusals are load-bearing:

* **No silent download.**  A model is fetched only when ``--allow-download`` is
  passed.  Otherwise a missing model is ``MODEL_ABSENT``, never an implicit
  network call whose result nobody reviewed.
* **No hardcoded cache versions.**  The Foundry artifact is resolved through
  ``rust/tools/foundry_discovery.py`` (issue #11's contract), which fails on
  zero/ambiguous/wrong-provider matches instead of globbing for the newest.
* **No substitution.**  If a pinned digest is recorded and the bytes on disk
  disagree, that is ``DIGEST_MISMATCH`` — a finding — not a re-pin.

External data is deliberately not treated as an implementation detail.  The
Phi-3.5 graph proto is 26 MB; the weights it references are 2.2 GB in a sibling
``.onnx.data``.  A digest of the proto alone would call two different weight
sets "the same model", which is exactly the substitution this module exists to
make impossible.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_paths import dump_public_json, public_path, scrub_text  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_RUST_TOOLS = _REPO / "rust" / "tools"

#: Where downloaded public models are cached.  Outside the repo tree by default so a
#: benchmark never mutates the repository it is measuring (the ledger-probe ruling,
#: PR #51: a diagnostic that writes into tracked state is mutating what it reads).
CACHE_ENV = "ONNXRUNTIME_EP_VULKAN_BENCH_MODEL_CACHE"

MODEL_OK = "MODEL_OK"
MODEL_ABSENT = "MODEL_ABSENT"
MODEL_DIGEST_MISMATCH = "MODEL_DIGEST_MISMATCH"
MODEL_RESOLVE_ERROR = "MODEL_RESOLVE_ERROR"


def cache_root() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "onnxruntime-ep-vulkan" / "bench-models"


@dataclass(frozen=True)
class ModelSpec:
    """One benchmark model, named by identity rather than by path.

    ``source`` is either ``"foundry"`` (resolved through the discovery contract) or
    ``"url"`` (a public download).  ``license_id`` and ``license_url`` are recorded
    because a benchmark that redistributes numbers about a model still has to say
    under what terms the model was obtained.
    """

    key: str
    description: str
    source: str
    license_id: str
    license_url: str
    #: foundry-only
    foundry_variant: str = ""
    foundry_provider: str = ""
    foundry_onnx_filename: str = ""
    foundry_alias: str = ""
    #: url-only — (relative filename, url) pairs; the first entry is the graph proto
    url_files: "tuple[tuple[str, str], ...]" = ()
    #: opset/producer note carried into the report so a reader knows what graph shape to expect
    producer_note: str = ""


PHI35 = ModelSpec(
    key="phi35_int4",
    description="Phi-3.5-mini-instruct, ORT GenAI builder, cuda-int4-rtn-block-32 "
                "(MatMulNBits + GroupQueryAttention, fp16 activations, external data)",
    source="foundry",
    license_id="MIT",
    license_url="https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE",
    foundry_variant="Phi-3.5-mini-instruct-cuda-gpu",
    foundry_provider="CUDAExecutionProvider",
    foundry_onnx_filename="phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    foundry_alias="phi-3.5-mini",
    producer_note="com.microsoft contrib ops: MatMulNBits, GroupQueryAttention, "
                  "SkipSimplifiedLayerNormalization",
)

MOBILENETV2 = ModelSpec(
    key="mobilenetv2_12",
    description="MobileNetV2 opset 12, float32, ONNX Model Zoo validated vision set",
    source="url",
    license_id="Apache-2.0",
    license_url="https://github.com/onnx/models/blob/main/LICENSE",
    url_files=(
        ("mobilenetv2-12.onnx",
         "https://github.com/onnx/models/raw/main/validated/vision/classification/"
         "mobilenet/model/mobilenetv2-12.onnx"),
    ),
    producer_note="ai.onnx opset 12, Conv/BatchNormalization/Clip/GlobalAveragePool",
)

MINILM = ModelSpec(
    key="all_minilm_l6_v2",
    description="sentence-transformers/all-MiniLM-L6-v2 exported to ONNX by Xenova — "
                "a float32 BERT-family transformer encoder, the third-model arm",
    source="url",
    license_id="Apache-2.0",
    license_url="https://huggingface.co/Xenova/all-MiniLM-L6-v2",
    url_files=(
        ("all-MiniLM-L6-v2.onnx",
         "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"),
    ),
    producer_note="transformers.onnx / optimum export; MatMul + LayerNormalization + "
                  "Softmax attention written out as primitive ai.onnx ops",
)

SPECS: "dict[str, ModelSpec]" = {m.key: m for m in (PHI35, MOBILENETV2, MINILM)}


@dataclass
class ResolvedModel:
    key: str
    status: str
    path: "str | None" = None
    files: "list[dict]" = field(default_factory=list)
    bundle_sha256: "str | None" = None
    total_bytes: int = 0
    license_id: str = ""
    license_url: str = ""
    description: str = ""
    producer_note: str = ""
    source: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        """Serialise for publication.

        ``path`` stays a real, openable path in memory — ``run_arm`` hands it to
        ``ort.InferenceSession`` — but nothing that leaves this process needs the
        operator's account name to be true.  The rooted form (``<model-cache>/...``)
        keeps every fact a reader can act on: which root, which layout beneath it.
        ``files[].path`` is already rooted by :func:`_describe_files`; running it
        through :func:`public_path` again is a no-op, so this is idempotent.
        """
        d = asdict(self)
        if d.get("path"):
            d["path"] = public_path(d["path"])
        d["files"] = [{**e, "path": public_path(e["path"])} if e.get("path") else e
                      for e in d.get("files", [])]
        d["detail"] = scrub_text(d.get("detail", ""))
        return d


def sha256_file(path: Path, *, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def external_data_files(onnx_path: Path) -> "list[Path]":
    """Every external-data file the graph at ``onnx_path`` actually references.

    Read from the proto's ``external_data`` entries rather than by globbing the
    directory: a sibling ``.onnx.data`` that the graph does *not* reference is not
    part of this model, and a referenced file in another directory would be missed
    by a glob.  Falls back to the conventional ``<name>.data`` sibling only when the
    proto cannot be parsed without loading 2 GB of weights.
    """
    found: "list[Path]" = []
    try:
        import onnx

        model = onnx.load(str(onnx_path), load_external_data=False)
        seen: "set[str]" = set()
        for tensor in onnx.external_data_helper._get_all_tensors(model):
            if tensor.HasField("data_location") and \
                    tensor.data_location == onnx.TensorProto.EXTERNAL:
                for kv in tensor.external_data:
                    if kv.key == "location" and kv.value not in seen:
                        seen.add(kv.value)
                        found.append((onnx_path.parent / kv.value).resolve())
    except Exception:
        sibling = onnx_path.with_suffix(onnx_path.suffix + ".data")
        if sibling.is_file():
            found.append(sibling.resolve())
    return [p for p in found if p.is_file()]


def _bundle_digest(entries: "list[dict]") -> str:
    """One digest over the whole file set, order-independent by construction.

    Individual per-file digests are also kept; this is the single value a report can
    quote as "the model" without a reader having to compare seven hex strings.
    """
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda d: d["name"]):
        h.update(e["name"].encode("utf-8"))
        h.update(b"\0")
        h.update(e["sha256"].encode("ascii"))
        h.update(b"\0")
        h.update(str(e["bytes"]).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _describe_files(paths: "list[Path]") -> "list[dict]":
    return [{"name": p.name, "path": public_path(p), "bytes": p.stat().st_size,
             "sha256": sha256_file(p)} for p in paths]


def _resolve_foundry(spec: ModelSpec) -> ResolvedModel:
    if str(_RUST_TOOLS) not in sys.path:
        sys.path.insert(0, str(_RUST_TOOLS))
    try:
        import foundry_discovery as fd  # type: ignore
    except Exception as exc:
        return ResolvedModel(key=spec.key, status=MODEL_RESOLVE_ERROR,
                             detail=f"foundry_discovery not importable: {exc}")
    fspec = fd.FoundryModelSpec(
        variant_name=spec.foundry_variant,
        execution_provider=spec.foundry_provider,
        onnx_filename=spec.foundry_onnx_filename,
        download_alias=spec.foundry_alias,
    )
    try:
        onnx_path = Path(fd.resolve_model_path(fspec))
    except fd.FoundryDiscoveryError as exc:
        return ResolvedModel(key=spec.key, status=MODEL_ABSENT, detail=str(exc)[:1200])
    except Exception as exc:
        return ResolvedModel(key=spec.key, status=MODEL_RESOLVE_ERROR, detail=repr(exc)[:1200])

    files = [onnx_path.resolve()] + external_data_files(onnx_path)
    entries = _describe_files(files)
    return ResolvedModel(
        key=spec.key, status=MODEL_OK, path=str(onnx_path), files=entries,
        bundle_sha256=_bundle_digest(entries),
        total_bytes=sum(e["bytes"] for e in entries),
        license_id=spec.license_id, license_url=spec.license_url,
        description=spec.description, producer_note=spec.producer_note, source="foundry",
        detail="resolved by identity through rust/tools/foundry_discovery.py (issue #11 "
               "contract: exactly one cached, correctly-provisioned variant, or refuse)",
    )


def _download(url: str, dest: Path, *, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    # The product token deliberately avoids the "<repo>-<suffix>" shape: that is
    # what a sibling worktree directory looks like, and the leak scanner flags it.
    # It is a false positive here -- this is a constant, not a path -- but the
    # remedy is to stop colliding, never to narrow a leak pattern.
    req = urllib.request.Request(url, headers={"User-Agent": "ort-ep-vulkan-bench/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)


def _resolve_url(spec: ModelSpec, *, allow_download: bool) -> ResolvedModel:
    root = cache_root() / spec.key
    wanted = [(root / name, url) for name, url in spec.url_files]
    missing = [(p, u) for p, u in wanted if not p.is_file()]

    if missing and not allow_download:
        names = ", ".join(p.name for p, _ in missing)
        return ResolvedModel(
            key=spec.key, status=MODEL_ABSENT,
            detail=(f"{names} absent from {public_path(root)}. Pass --allow-download to fetch "
                    f"from the recorded public URL(s), or set {CACHE_ENV} to a directory that "
                    f"has them. This harness never downloads implicitly."),
            license_id=spec.license_id, license_url=spec.license_url,
            description=spec.description, source="url",
        )

    for path, url in missing:
        try:
            _download(url, path)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return ResolvedModel(
                key=spec.key, status=MODEL_ABSENT,
                detail=f"download of {url} failed: {exc!r}",
                license_id=spec.license_id, license_url=spec.license_url,
                description=spec.description, source="url",
            )

    graph = wanted[0][0]
    files = [p for p, _ in wanted] + external_data_files(graph)
    # dedupe while preserving order (a graph proto may also be listed in url_files)
    seen: "set[str]" = set()
    ordered: "list[Path]" = []
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            ordered.append(p.resolve())
    entries = _describe_files(ordered)
    urls = {name: url for name, url in spec.url_files}
    for e in entries:
        e["url"] = urls.get(e["name"], "(external data, sibling of the graph proto)")
    return ResolvedModel(
        key=spec.key, status=MODEL_OK, path=str(graph), files=entries,
        bundle_sha256=_bundle_digest(entries),
        total_bytes=sum(e["bytes"] for e in entries),
        license_id=spec.license_id, license_url=spec.license_url,
        description=spec.description, producer_note=spec.producer_note, source="url",
        detail=f"cached under {public_path(root)}",
    )


def resolve(key: str, *, allow_download: bool = False,
            expect_bundle_sha256: "str | None" = None) -> ResolvedModel:
    """Resolve one model by key.  Never guesses, never substitutes."""
    spec = SPECS.get(key)
    if spec is None:
        return ResolvedModel(key=key, status=MODEL_RESOLVE_ERROR,
                             detail=f"unknown model key {key!r}; known: {sorted(SPECS)}")
    if spec.source == "foundry":
        rec = _resolve_foundry(spec)
    else:
        rec = _resolve_url(spec, allow_download=allow_download)

    if rec.status == MODEL_OK and expect_bundle_sha256:
        if rec.bundle_sha256 != expect_bundle_sha256:
            rec.status = MODEL_DIGEST_MISMATCH
            rec.detail = (f"bundle digest {rec.bundle_sha256} does not match the pinned "
                          f"{expect_bundle_sha256}. The bytes on disk are not the bytes the "
                          f"pinned number was measured on; this is a finding, not a re-pin.")
    return rec


def resolve_all(keys: "list[str] | None" = None, *,
                allow_download: bool = False) -> "dict[str, ResolvedModel]":
    keys = list(SPECS) if keys is None else keys
    return {k: resolve(k, allow_download=allow_download) for k in keys}


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--allow-download", action="store_true",
                    help="permit fetching the public URL-sourced models")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    resolved = resolve_all(a.models, allow_download=a.allow_download)
    payload = {"schema": "bench_models/1",
               "cache_root": public_path(cache_root()),
               "models": {k: v.to_dict() for k, v in resolved.items()}}
    if a.out:
        dump_public_json(payload, Path(a.out), indent=2, sort_keys=True)

    for key, rec in resolved.items():
        print(f"{key:<20} {rec.status}")
        if rec.status == MODEL_OK:
            # Rooted on stdout too: console output is redirected into committed
            # `.log` artifacts, so "it is only a print" is not a place the account
            # name gets to survive.
            print(f"    path   : {public_path(rec.path)}")
            print(f"    bundle : {rec.bundle_sha256}")
            print(f"    bytes  : {rec.total_bytes:,} across {len(rec.files)} file(s)")
            print(f"    licence: {rec.license_id} ({rec.license_url})")
        else:
            print(f"    detail : {scrub_text(rec.detail)[:400]}")
    return 0 if all(r.status == MODEL_OK for r in resolved.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
