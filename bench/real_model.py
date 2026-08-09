"""Real-model latency: the models a user actually runs, on a clock, with a reference arm.

WHY THIS MODULE EXISTS (issue #56)
==================================
Every real-model instrument in this repository was, until this file, **deliberately clock-free**
(`bench/exec_census.py`, `bench/island_attribution.py`,
`bench/results/probe_real_matmulnbits_rows.py` — none of them contains a timer), and every
clocked instrument was **synthetic** (`bench/results/ab_row_tile.py` at K=N=4096 f32,
`bench/bench.py`'s op cases). That is a coherent design and it is exactly the hole #56 names:
*"I want to see some perf numbers on real models."*

This module is the shared, testable half of that instrument. It holds the parts that can be
checked without a GPU — model identity and provenance, feed construction, the arm definitions,
the statistics, the throughput arithmetic and the equivalence verdicts — so that
`bench/results/probe_real_model_latency.py` is a driver rather than a pile of logic that only
runs when a device is present. `bench/test_real_model.py` tests everything here.

WHAT IT REFUSES
===============
The refusals are inherited rather than invented; each names a wrong number this project has
actually produced:

* **No hardcoded Foundry path.** Resolution goes through `foundry_discovery.resolve_model_path`
  (issue #11/#19: the cache moved from `…-cuda-gpu/` to `…-cuda-gpu-2/v2/` with no code change on
  either side). A declared model that is absent is a **failure**, never a silent skip.
* **No unattributed device.** The device is pinned and named in the artifact
  (`bench.bench.select_device`), because two GPUs on one desk are not interchangeable.
* **No unverified arm.** Every arm that is compared is first checked against a CPU reference
  **in the same process, on the same session objects that are then timed** — the `argmax 0`
  defect (161 nodes dispatched, `compute_failures: 0`, all-zero logits) is why.
* **No single ms.** A median without a spread is a rumour; `latency_stats` always returns the
  distribution, and `tokens_per_second` is derived from the median rather than from a best run.
* **No unlabelled arm ordering.** Arms alternate per repeat. A fixed order produced a spurious
  0.905x at M=1 in `ab_row_tile.py`, on a shape where both arms bind *identical* SPIR-V.

WHAT IT WILL NOT LET YOU CONCLUDE
=================================
`M = 1` prefill tiled-vs-untiled is the **null control**, not a result: both arms resolve
`QB_ROWS = 1` and build the identical pipeline, so its spread is the harness's noise floor and
no ratio narrower than it is a measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import re
import statistics
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Schema of the artifact `probe_real_model_latency.py` writes. Bumped when a *meaning* changes,
#: never when a field is added — readers key off it to know what a row is a row of.
SCHEMA = "real_model_latency/1"

#: The row-tile kill switch. Also the A/B control: pinned to "1" the tiled path cannot be
#: selected at all, so the two arms differ by exactly one specialisation constant.
ROWS_ENV = "ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS"
EP_NAME = "VulkanExecutionProvider"
CPU_EP = "CPUExecutionProvider"

# --------------------------------------------------------------------------------------------
# Model identity and provenance
# --------------------------------------------------------------------------------------------


class ModelUnavailable(RuntimeError):
    """A declared model does not resolve on this machine.

    Raised rather than skipped: #56 asks for numbers on *real models*, and a lane that quietly
    drops a model when its cache entry moves produces a result that reads as complete while
    covering less than it claims.
    """


class ModelProvenanceRefused(RuntimeError):
    """A pinned model's source metadata is missing, malformed, or does not match the pin.

    Fail-closed, with a named reason (or several) rather than a bare boolean: #78 exists
    because a cached substitute (`759c3cd2...`, 90,387,606 bytes) was once accepted as "the"
    MiniLM artifact with no check of its own. ``reasons`` always has at least one entry
    naming which of the immutable identity fields the evidence disagreed with, was absent,
    was empty, or was malformed (e.g. carried a newline) — "unchecked" and "verified" are
    different states, and this exception is what stands between them.
    """

    def __init__(self, spec_key: str, reasons: "list[str]"):
        self.spec_key = spec_key
        self.reasons = list(reasons)
        super().__init__(
            f"{spec_key}: source metadata refused ({len(self.reasons)} reason(s)): "
            + "; ".join(self.reasons)
        )


#: Exact 40 lower-hex-digit git/HF commit revision. `re.fullmatch`, never `re.match`: `match`
#: accepts a trailing newline this pattern's `$` would otherwise seem to forbid (`$` matches
#: just before a trailing `\n` as well as at the true end of string), and a revision string
#: read from an untrusted sidecar or a copy-paste can carry exactly that.
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
#: A mutable ref name a resolver could silently follow forward. `source_revision` must never
#: fullmatch this — it must fullmatch `_REVISION_RE` instead, which "main"/"master"/"HEAD"
#: cannot.
_MUTABLE_REF_NAMES = frozenset({"main", "master", "head", "latest", "HEAD"})


def _non_empty_no_whitespace(value: str, field: str, reasons: "list[str]") -> None:
    if value is None:
        reasons.append(f"{field} is missing (None)")
    elif value == "":
        reasons.append(f"{field} is empty")
    elif value != value.strip() or "\n" in value or "\r" in value or "\t" in value:
        reasons.append(f"{field} carries leading/trailing whitespace or a newline/tab: {value!r}")


def _reasons_has(reasons: "list[str]", field: str) -> bool:
    """True if *field* already has a recorded reason — guards a second, redundant check.

    All-or-nothing validation still runs every clause (never returns early on the first
    miss), but a field that is already known missing/empty/malformed should not ALSO be
    reported as failing its format regex — that would read as two independent problems
    where there is one.
    """
    return any(r.startswith(field) for r in reasons)


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One real model this lane benchmarks, and how to find it without guessing.

    ``resolver`` is either ``"foundry"`` (goes through the Foundry cache manifest) or
    ``"repo-cache"`` (the pinned download cache `rust/modelrunner` already uses, whose sha256 is
    recorded in `bench/results/rust-model-runner/<key>.json`). No other resolution exists — a
    literal path in a benchmark is a guess about a version.

    ``capability`` is ``"full"`` (the default: this model is fed and timed) or
    ``"provenance_only"`` (issue #78: the model's *bytes* are resolved and verified, but no
    case/feed builder exists for it yet, so it is never fed to a session). A model in either
    state is still, always, resolved and verified — capability decides whether it is *also*
    benchmarked, never whether its identity is checked.

    A model is **pinned** when any of ``source_repo``/``source_revision``/``source_file``/
    ``pinned_sha256``/``pinned_bytes`` is set. All five identity fields become required, and
    are validated here, at construction, with ``re.fullmatch`` — never an early return on the
    first missing field, so a spec with four correct fields and one blank one names the blank
    one rather than silently short-circuiting.
    """

    key: str
    family: str
    resolver: str
    #: Foundry identity (resolver == "foundry")
    variant_name: str = ""
    execution_provider: str = ""
    onnx_filename: str = ""
    download_alias: str = ""
    #: repo-cache identity (resolver == "repo-cache")
    cache_filename: str = ""
    #: Where the recorded provenance for this model lives, relative to `bench/results/`.
    recorded_provenance: str = ""
    note: str = ""
    #: "full" (fed and timed) or "provenance_only" (resolved and verified, never fed).
    capability: str = "full"
    #: Immutable source identity (issue #78). Every field here, or none.
    source_repo: str = ""
    source_revision: str = ""
    source_file: str = ""
    pinned_sha256: str = ""
    pinned_bytes: "int | None" = None
    #: Expected external-data blob filenames, or ``()`` if the pin asserts NONE exist —
    #: distinct from "not checked", which is what an unpinned spec's empty tuple would
    #: otherwise look identical to.
    pinned_external: "tuple[str, ...]" = ()

    @property
    def pinned(self) -> bool:
        return bool(
            self.source_repo or self.source_revision or self.source_file
            or self.pinned_sha256 or self.pinned_bytes
        )

    def __post_init__(self) -> None:
        if not self.pinned:
            return
        reasons: "list[str]" = []
        _non_empty_no_whitespace(self.source_repo, "source_repo", reasons)
        if self.source_repo and not _reasons_has(reasons, "source_repo") \
                and not _REPO_RE.fullmatch(self.source_repo):
            reasons.append(f"source_repo does not fullmatch owner/name: {self.source_repo!r}")

        _non_empty_no_whitespace(self.source_revision, "source_revision", reasons)
        if self.source_revision and not _reasons_has(reasons, "source_revision"):
            if self.source_revision.lower() in _MUTABLE_REF_NAMES:
                reasons.append(
                    f"source_revision is a mutable ref name, not an immutable commit SHA: "
                    f"{self.source_revision!r}"
                )
            elif not _REVISION_RE.fullmatch(self.source_revision):
                reasons.append(
                    f"source_revision does not fullmatch 40 lowercase hex digits: "
                    f"{self.source_revision!r}"
                )

        _non_empty_no_whitespace(self.source_file, "source_file", reasons)
        if self.source_file and not _reasons_has(reasons, "source_file"):
            if self.source_file.startswith("/") or self.source_file.startswith("\\"):
                reasons.append(f"source_file must be repo-relative, not rooted: {self.source_file!r}")
            elif ".." in Path(self.source_file).parts:
                reasons.append(f"source_file must not contain '..': {self.source_file!r}")

        _non_empty_no_whitespace(self.pinned_sha256, "pinned_sha256", reasons)
        if self.pinned_sha256 and not _reasons_has(reasons, "pinned_sha256") \
                and not _SHA256_RE.fullmatch(self.pinned_sha256):
            reasons.append(
                f"pinned_sha256 does not fullmatch 64 lowercase hex digits: "
                f"{self.pinned_sha256!r}"
            )

        if self.pinned_bytes is None:
            reasons.append("pinned_bytes is missing (None)")
        elif isinstance(self.pinned_bytes, bool) or not isinstance(self.pinned_bytes, int):
            reasons.append(f"pinned_bytes must be an int, got {type(self.pinned_bytes).__name__}")
        elif self.pinned_bytes <= 0:
            reasons.append(f"pinned_bytes must be positive, got {self.pinned_bytes!r}")

        if reasons:
            raise ModelProvenanceRefused(self.key, reasons)


PHI35 = ModelSpec(
    key="phi-3.5-mini-instruct-cuda-int4-rtn-block-32",
    family="phi3",
    resolver="foundry",
    variant_name="Phi-3.5-mini-instruct-cuda-gpu",
    execution_provider="CUDAExecutionProvider",
    onnx_filename="phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    download_alias="phi-3.5-mini",
    recorded_provenance="rust-model-runner/phi-3.5-mini.json",
    note="Foundry Local Phi-3.5-mini-instruct, int4 RTN block-32, fp16 activations. "
         "32 layers, 32 KV heads, head dim 96, 64 KV inputs and 64 KV outputs.",
)

MOBILENETV2 = ModelSpec(
    key="mobilenetv2-12",
    family="cnn",
    resolver="repo-cache",
    cache_filename="mobilenetv2-12.onnx",
    recorded_provenance="rust-model-runner/mobilenetv2-12.json",
    note="ONNX model zoo MobileNetV2-12, f32. Broad-EP-overhead arm: dense elementwise/Conv "
         "graph with a single free batch dimension, so it prices boundary crossing on a model "
         "whose arithmetic has nothing to do with MatMulNBits.",
)

#: Issue #78: `all-MiniLM-L6-v2`, pinned to an immutable Hugging Face commit revision rather
#: than the mutable `main` branch a prior draft used (a Hugging Face `/resolve/main/` URL, or
#: a bare-repo reference with no revision, both silently track new commits — the exact defect
#: this pin exists to close). `capability="provenance_only"`: this repository has no case or
#: feed builder for a sentence embedding model yet (#74/#75/#76), so `MINILM` is resolved and
#: its provenance is verified on every run, and it is never fed to a session. That is a
#: capability gap, not a provenance gap — the two must not be conflated by leaving the model
#: out of `MODELS` entirely, which would make it invisible rather than explicitly deferred.
MINILM = ModelSpec(
    key="all-MiniLM-L6-v2",
    family="bert",
    resolver="repo-cache",
    cache_filename="all-MiniLM-L6-v2.onnx",
    recorded_provenance="rust-model-runner/all-MiniLM-L6-v2.json",
    capability="provenance_only",
    source_repo="sentence-transformers/all-MiniLM-L6-v2",
    source_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    source_file="onnx/model.onnx",
    pinned_sha256="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    pinned_bytes=90405214,
    pinned_external=(),
    note="sentence-transformers/all-MiniLM-L6-v2, onnx/model.onnx, pinned to an immutable "
         "commit revision (never the mutable `main` HF ref). Provenance-only: no case/feed "
         "builder exists for a sentence-embedding model yet (#74/#75/#76), so this model is "
         "always resolved and verified and never fed to a session.",
)

MODELS = {m.key: m for m in (PHI35, MOBILENETV2, MINILM)}

#: Where `rust/modelrunner` puts pinned downloads. Read from the environment first so a machine
#: that keeps its cache elsewhere is not silently missed.
REPO_CACHE_ENV = "ONNXRUNTIME_EP_VULKAN_MODEL_CACHE"


def repo_cache_dir() -> Path:
    override = os.environ.get(REPO_CACHE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "onnxruntime-ep-vulkan" / "models"


def sha256_file(path: "Path | str") -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_sha256(spec: ModelSpec, results_dir: "Path | None" = None) -> "str | None":
    """The sha256 a previous, independent tool recorded for this model, or ``None``.

    Deliberately read from `rust/modelrunner`'s artifacts rather than from a constant in this
    file: a hash this module both writes and checks proves nothing.
    """
    import json

    base = Path(results_dir) if results_dir else (_BENCH / "results")
    p = base / spec.recorded_provenance
    if not spec.recorded_provenance or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("onnx_sha256")
    except Exception:
        return None


def recorded_source_metadata(spec: ModelSpec, results_dir: "Path | None" = None) -> dict:
    """Independently recorded source-repo/revision/file/bytes for *spec*, or ``{}``.

    ``source_repo``/``source_revision``/``source_file`` cannot be derived from the resolved
    file's bytes alone — unlike the sha256 and the byte count, which `resolve_model` computes
    fresh from the real file every time, "which Hugging Face repo and commit produced these
    bytes" is external metadata this module can only read from a sidecar another process
    wrote. An absent sidecar is ``{}``, not a dict of empty strings — the two are different
    refusal reasons in `verify_source_metadata` (missing vs. empty).
    """
    import json

    base = Path(results_dir) if results_dir else (_BENCH / "results")
    p = base / spec.recorded_provenance
    if not spec.recorded_provenance or not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        "source_repo": raw.get("source_repo"),
        "source_revision": raw.get("source_revision"),
        "source_file": raw.get("source_file"),
        "onnx_bytes": raw.get("onnx_bytes"),
    }


def _malformed_string_reason(name: str, value) -> "str | None":
    """One named reason *value* fails as recorded metadata for *name*, or ``None`` if clean.

    Four states, never conflated: missing (``None``), empty (``""``), malformed (wrong type,
    or a string carrying whitespace/newline/tab a real Hugging Face field never legitimately
    has), and — checked by the caller, not here — disagreement with the pin.
    """
    if value is None:
        return f"{name}: recorded metadata is missing (None)"
    if not isinstance(value, str):
        return f"{name}: recorded metadata is not a string (got {type(value).__name__})"
    if value == "":
        return f"{name}: recorded metadata is empty"
    if value != value.strip() or "\n" in value or "\r" in value or "\t" in value:
        return f"{name}: recorded metadata carries whitespace/newline/tab: {value!r}"
    return None


def verify_source_metadata(
    spec: ModelSpec,
    *,
    resolved_sha256: str,
    resolved_bytes: int,
    metadata: dict,
    external: dict,
) -> dict:
    """Check freshly-resolved bytes AND independently-recorded metadata against *spec*'s pin.

    Three states, never conflated: an unpinned spec returns ``{"state": "unpinned",
    "provenance_ok": None}`` without checking anything (there is nothing pinned to check
    against); a pinned spec that agrees on every field returns ``{"state": "verified",
    "provenance_ok": True}``; a pinned spec with ANY missing, empty, malformed, or
    disagreeing field RAISES `ModelProvenanceRefused` naming every reason found — there is
    no ``provenance_ok=False`` return value, because a refusal a caller can `if not ok:`
    past is exactly the "unchecked read as verified" shape #78 exists to close.

    ``resolved_sha256``/``resolved_bytes`` are computed by the caller from the actual
    resolved file — non-tautological, because a substituted file's bytes genuinely produce a
    different hash. ``source_repo``/``source_revision``/``source_file`` cannot be derived
    from bytes alone, so they are checked against ``metadata`` (an independent sidecar
    record) instead; an absent sidecar means every one of those fields reads as "missing",
    which correctly refuses rather than silently passing a pin with nothing to verify it.
    """
    if not spec.pinned:
        return {"state": "unpinned", "provenance_ok": None}

    reasons: "list[str]" = []
    for name, expected, got in (
        ("source_repo", spec.source_repo, metadata.get("source_repo")),
        ("source_revision", spec.source_revision, metadata.get("source_revision")),
        ("source_file", spec.source_file, metadata.get("source_file")),
    ):
        bad = _malformed_string_reason(name, got)
        if bad:
            reasons.append(bad)
        elif got != expected:
            reasons.append(
                f"{name}: recorded metadata {got!r} does not match the pin {expected!r} "
                f"(source drift)"
            )

    rec_bytes = metadata.get("onnx_bytes")
    if rec_bytes is None:
        reasons.append("onnx_bytes: recorded metadata is missing (None)")
    elif isinstance(rec_bytes, bool) or not isinstance(rec_bytes, int):
        reasons.append(f"onnx_bytes: recorded metadata is not an int (got {type(rec_bytes).__name__})")
    elif rec_bytes != spec.pinned_bytes:
        reasons.append(
            f"onnx_bytes: recorded metadata {rec_bytes!r} does not match the pin "
            f"{spec.pinned_bytes!r} (source drift)"
        )

    # Non-tautological: computed from the real resolved file, so a cached substitute
    # (#78's `759c3cd2...`, 90,387,606 bytes) genuinely fails here rather than being waved
    # through by matching only the spec its own code carries.
    if resolved_sha256 != spec.pinned_sha256:
        reasons.append(
            f"resolved sha256 {resolved_sha256!r} does not match the pin "
            f"{spec.pinned_sha256!r}"
        )
    if resolved_bytes != spec.pinned_bytes:
        reasons.append(
            f"resolved size {resolved_bytes!r} bytes does not match the pin "
            f"{spec.pinned_bytes!r} bytes"
        )

    if not external.get("scanned"):
        reasons.append(
            f"external_data: graph could not be scanned ({external.get('reason')!r}); a pin "
            f"cannot certify external-data state over an unscanned graph"
        )
    else:
        got_locations = tuple(sorted(f["location"] for f in external.get("files", [])))
        expected_locations = tuple(sorted(spec.pinned_external))
        if got_locations != expected_locations:
            reasons.append(
                f"external_data: pin expects location(s) {expected_locations!r}, graph has "
                f"{got_locations!r}"
            )

    if reasons:
        raise ModelProvenanceRefused(spec.key, reasons)

    return {
        "state": "verified",
        "provenance_ok": True,
        "source_repo": spec.source_repo,
        "source_revision": spec.source_revision,
        "source_file": spec.source_file,
        "sha256": resolved_sha256,
        "bytes": resolved_bytes,
    }


def pinned_source_url(spec: ModelSpec) -> str:
    """The immutable HTTPS URL a pinned spec's bytes came from, for a remediation message.

    Refuses (``ValueError``) an unpinned spec: there is no immutable URL to build for a
    model with no pin, and building one from empty fields would produce a URL-shaped string
    that names nothing. Never `/resolve/main/` — `source_revision` was already validated at
    construction to fullmatch 40 lowercase hex digits, so it cannot be the literal ``"main"``
    or any other mutable ref by the time this runs.
    """
    if not spec.pinned:
        raise ValueError(
            f"{spec.key}: pinned_source_url requires a pinned spec; this spec has no "
            f"source_repo/source_revision/source_file to build a URL from"
        )
    return (
        f"https://huggingface.co/{spec.source_repo}/resolve/"
        f"{spec.source_revision}/{spec.source_file}"
    )


def resolve_model(spec: ModelSpec, *, results_dir: "Path | None" = None) -> dict:
    """Resolve one model to a path plus a full provenance block, or raise.

    The returned dict is what lands in the artifact. ``agrees_with_recorded_provenance`` is
    reported rather than enforced — a model that legitimately moved forward should be visible as
    a disagreement, not as a crash — but it is a *field on the face of the result*, so a reader
    cannot miss it.

    For a **pinned** spec (issue #78), verification is unconditional: `verify_source_metadata`
    runs on every call, before anything is returned, and raises `ModelProvenanceRefused`
    rather than returning a record with ``provenance_ok`` set to anything but ``True`` — there
    is no early-success path and no path that returns a pinned record un-verified.

    ``path`` in the returned dict is the rooted, PUBLIC form (`public_paths.root_public_path`);
    the real, absolute path lives under the private ``_runtime_path`` key, read only through
    `public_paths.runtime_path`. This is what lets `resolve_model`'s own result be written
    straight into a committed JSON record without leaking the machine that produced it.
    """
    import public_paths as pp

    if spec.resolver == "foundry":
        import foundry_discovery as fd

        fspec = fd.FoundryModelSpec(
            variant_name=spec.variant_name,
            execution_provider=spec.execution_provider,
            onnx_filename=spec.onnx_filename,
            download_alias=spec.download_alias,
        )
        try:
            path = Path(fd.resolve_model_path(fspec))
        except Exception as exc:
            raise ModelUnavailable(
                f"{spec.key}: Foundry resolution failed: {exc}. "
                f"Fetch it with `foundry model download {spec.download_alias}`. "
                f"This lane fails rather than skipping: #56 asks for numbers on real models."
            ) from exc
        provenance = "foundry-resolved"
    elif spec.resolver == "repo-cache":
        path = repo_cache_dir() / spec.cache_filename
        if not path.is_file():
            # The public, rooted form: this message reaches both the console AND (via
            # `main()`'s `unavailable_models`/`failures` list) a committed artifact, so it
            # must never carry the real absolute cache path (issue #78, blocker #3).
            public_path = pp.root_public_path(path)
            if spec.pinned:
                remediation = (
                    f"Fetch it from the immutable pinned source: "
                    f"{pinned_source_url(spec)} (sha256 {spec.pinned_sha256}, "
                    f"{spec.pinned_bytes} bytes), and place it at {public_path}, or set "
                    f"{REPO_CACHE_ENV} to the directory that holds it. "
                    f"`ort-model-runner` has no manifest entry for {spec.key} — do not "
                    f"recommend it for a pinned, provenance-only model."
                )
            else:
                remediation = (
                    f"Fetch it with `cargo run -p ort-model-runner -- --model {spec.key}` "
                    f"(which pins and hashes it), or set {REPO_CACHE_ENV} to the directory "
                    f"that holds it."
                )
            raise ModelUnavailable(f"{spec.key}: {public_path} is absent. {remediation}")
        provenance = "pinned-cache" if spec.pinned else "cache"
    else:  # pragma: no cover - guarded by MODELS
        raise ValueError(f"unknown resolver {spec.resolver!r}")

    if not path.is_file():
        raise ModelUnavailable(
            f"{spec.key}: resolver returned {pp.root_public_path(path)}, which does not exist"
        )

    digest = sha256_file(path)
    size = path.stat().st_size
    recorded = recorded_sha256(spec, results_dir)
    external = external_data_provenance(path)

    provenance_record: dict
    if spec.pinned:
        metadata = recorded_source_metadata(spec, results_dir)
        # Never returns a refusal — raises. No early-success path: this call happens on
        # EVERY resolution of a pinned spec, unconditionally, before the function returns.
        provenance_record = verify_source_metadata(
            spec, resolved_sha256=digest, resolved_bytes=size, metadata=metadata,
            external=external,
        )
    else:
        provenance_record = {"state": "unpinned", "provenance_ok": None}

    return {
        "key": spec.key,
        "family": spec.family,
        "capability": spec.capability,
        "path": pp.root_public_path(path),
        "_runtime_path": str(path),
        "sha256": digest,
        "bytes": size,
        "provenance": provenance,
        "resolver": spec.resolver,
        "recorded_sha256": recorded,
        "agrees_with_recorded_provenance": (recorded == digest) if recorded else None,
        "pinned": spec.pinned,
        "provenance_ok": provenance_record["provenance_ok"],
        "provenance_state": provenance_record["state"],
        "source_metadata": (
            {k: v for k, v in provenance_record.items() if k not in ("state", "provenance_ok")}
            if spec.pinned else None
        ),
        "external_data": external,
        "weights_bytes": sum(f["bytes"] for f in external["files"]),
        "note": spec.note,
    }


def _iter_graph_tensors(graph):
    """Every ``TensorProto``-shaped weight reachable from *graph*, deterministically ordered.

    Depth-first in protobuf declaration order: initializers, then sparse initializers (each
    contributing its ``values`` AND ``indices`` — both are themselves ``TensorProto`` and
    either can independently declare external data), then every node's attributes, which is
    how an ``If``/``Loop``/``Scan`` subgraph's own initializers are reached.
    """
    for t in graph.initializer:
        yield t
    for st in graph.sparse_initializer:
        yield st.values
        yield st.indices
    for node in graph.node:
        yield from _iter_node_tensors(node)


def _iter_node_tensors(node):
    for attr in node.attribute:
        yield from _iter_attribute_tensors(attr)


def _iter_attribute_tensors(attr):
    """Every tensor an ``AttributeProto`` can carry: the four tensor-bearing fields, then a
    recursive descent into any subgraph the attribute names (``g``/``graphs`` — how If/Loop/
    Scan bodies, and their own initializers, are reached)."""
    if attr.HasField("t"):
        yield attr.t
    if attr.HasField("sparse_tensor"):
        yield attr.sparse_tensor.values
        yield attr.sparse_tensor.indices
    for t in attr.tensors:
        yield t
    for st in attr.sparse_tensors:
        yield st.values
        yield st.indices
    if attr.HasField("g"):
        yield from _iter_graph_tensors(attr.g)
    for g in attr.graphs:
        yield from _iter_graph_tensors(g)


def _iter_function_tensors(func):
    """A local ``FunctionProto``'s own tensors: its body's nodes, AND its
    ``attribute_proto`` entries — the function-attribute DEFAULT VALUES, which are full
    ``AttributeProto`` messages and can themselves carry a default tensor. The plain
    ``attribute`` field (just names, no values) carries nothing to scan."""
    for node in func.node:
        yield from _iter_node_tensors(node)
    for attr in func.attribute_proto:
        yield from _iter_attribute_tensors(attr)


def _iter_model_tensors(model):
    """Every tensor reachable from a ``ModelProto``: the main graph, every local function,
    and every ``training_info`` entry's ``initialization``/``algorithm`` graphs — training
    graphs carry their own initializers and are not part of the main inference graph at
    all, so a scan of ``model.graph`` alone misses them entirely."""
    yield from _iter_graph_tensors(model.graph)
    for func in model.functions:
        yield from _iter_function_tensors(func)
    for ti in model.training_info:
        if ti.HasField("initialization"):
            yield from _iter_graph_tensors(ti.initialization)
        if ti.HasField("algorithm"):
            yield from _iter_graph_tensors(ti.algorithm)


def _external_location(tensor, onnx_mod) -> "tuple[bool, str] | None":
    """``None`` if *tensor* is not external; else ``(True, location)`` or ``(False, reason)``.

    A malformed EXTERNAL tensor — zero ``location`` entries, more than one (whether they
    agree or conflict: BOTH are refused, since a graph that names the same blob twice has
    already shown its external-data bookkeeping cannot be trusted), or a location that is
    empty/whitespace — is reported as a REFUSAL reason, never silently treated as "no
    location" (which would under-report) or "first location wins" (which would guess).
    """
    if not (tensor.HasField("data_location") and tensor.data_location == onnx_mod.TensorProto.EXTERNAL):
        return None
    locations = [kv.value for kv in tensor.external_data if kv.key == "location"]
    name = tensor.name or "<unnamed>"
    if len(locations) == 0:
        return (False, f"tensor {name!r}: EXTERNAL with no 'location' entry in external_data")
    if len(locations) > 1:
        return (
            False,
            f"tensor {name!r}: EXTERNAL with {len(locations)} 'location' entries "
            f"{locations!r} (duplicate or conflicting; refused regardless of whether they "
            f"agree)",
        )
    loc = locations[0]
    if not isinstance(loc, str) or loc.strip() == "":
        return (False, f"tensor {name!r}: EXTERNAL 'location' entry is empty/whitespace: {loc!r}")
    return (True, loc)


def external_data_provenance(path: Path) -> dict:
    """Hash the tensors the `.onnx` file does *not* contain.

    The Foundry Phi-3.5 graph is 26 MB and its weights are 2.29 GB in a sibling `.data` file.
    A provenance block that hashes only the `.onnx` therefore identifies the *topology* and
    says nothing about the numbers the benchmark actually multiplies — swap the `.data` and the
    recorded hash is unchanged. This is not hypothetical for a quantised model shipped by a
    downloader that can re-materialise weights independently of the graph.

    The file list comes from the graph's own `external_data` locations, not from a `.data`
    suffix guess, so a model that names its blob something else is still covered.

    Recursive (issue #78): every ``TensorProto``/``SparseTensorProto`` location the pinned
    ONNX schema (``onnx==1.22.0``) can hold — graph initializers and sparse initializers,
    node attribute tensors and sparse tensors, nested subgraphs (``If``/``Loop``/``Scan``),
    local function bodies and their attribute-default tensors, and ``training_info``
    initialization/algorithm graphs — is walked via `_iter_model_tensors`. A malformed
    EXTERNAL tensor anywhere in that walk (`_external_location` returning a refusal) makes
    the WHOLE scan refuse (``scanned: False``): an unscanned graph is reported as unscanned,
    never silently read back as "no external data found", which would be indistinguishable
    from a graph that is genuinely clean.
    """
    rec = {"scanned": False, "files": [], "reason": None}
    try:
        import onnx
    except ImportError:  # pragma: no cover - onnx is present in this repo's venv
        rec["reason"] = "onnx package not importable; external weights are UNHASHED"
        return rec
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:  # pragma: no cover - a corrupt graph is its own failure
        rec["reason"] = f"could not parse graph: {exc}; external weights are UNHASHED"
        return rec

    locations = set()
    malformed = []
    for tensor in _iter_model_tensors(model):
        result = _external_location(tensor, onnx)
        if result is None:
            continue
        ok, value = result
        if ok:
            locations.add(value)
        else:
            malformed.append(value)

    if malformed:
        rec["reason"] = (
            f"{len(malformed)} malformed EXTERNAL tensor(s), refusing rather than scanning "
            f"clean: " + "; ".join(malformed)
        )
        rec["malformed"] = malformed
        return rec

    rec["scanned"] = True
    for loc in sorted(locations):
        blob = path.parent / loc
        if not blob.is_file():
            rec["files"].append({"location": loc, "bytes": 0, "sha256": None,
                                 "missing": True})
            continue
        rec["files"].append({"location": loc, "bytes": blob.stat().st_size,
                             "sha256": sha256_file(blob)})
    if not locations:
        rec["reason"] = "graph carries no external initializers; the .onnx hash covers the weights"
    return rec



# --------------------------------------------------------------------------------------------
# Cases — what is fed, and what the numbers may be divided by
# --------------------------------------------------------------------------------------------

PHI35_LAYERS = 32
PHI35_KV_HEADS = 32
PHI35_HEAD_DIM = 96
PHI35_F16_BYTES = 2
#: 32 layers x 2 (key,value) x 32 heads x 96 dim x 2 B. Niobe's slope, exact to the byte
#: (`docs/PERF.md` §21): the host KV round trip costs this many bytes per past token, each way.
PHI35_BYTES_PER_PAST_TOKEN = (
    PHI35_LAYERS * 2 * PHI35_KV_HEADS * PHI35_HEAD_DIM * PHI35_F16_BYTES
)

#: Deterministic and fixed: the KV cache content must be byte-identical across arms or the arms
#: are not comparable, and it must be identical across runs or two runs are not comparable.
KV_SEED = 0x5EED0056


@dataclasses.dataclass(frozen=True)
class Case:
    """One (model, phase, size) point of the matrix.

    ``tokens`` is what a tokens/s figure may divide by, and it is **not** always ``m``: a decode
    step emits one token no matter how large its KV cache is, and a MobileNetV2 batch emits no
    tokens at all (``tokens = None``, and the throughput is reported as images/s).
    """

    model_key: str
    phase: str          # "prefill" | "decode" | "batch"
    m: int              # sequence positions fed this run (or batch size, for "batch")
    past: int = 0       # past_sequence_length; decode requires > 0 for at least one point
    tokens: "int | None" = None
    unit: str = "tokens"

    @property
    def label(self) -> str:
        if self.phase == "batch":
            return f"{self.model_key}/batch/N{self.m}"
        return f"{self.model_key}/{self.phase}/M{self.m}/past{self.past}"


def phi35_cases(prefill_m, decode_past) -> "list[Case]":
    """Prefill sweep (empty cache) plus decode points (non-empty cache).

    The decode arm exists because **every** wall-clock measurement in this repository before
    #56 ran at `past_sequence_length == 0`. Real decode runs in the hundreds to thousands, where
    the KV traffic — `PHI35_BYTES_PER_PAST_TOKEN` bytes per past token, each way — and not
    MatMulNBits may dominate. A prefill-only benchmark of a decoder model measures the case the
    user spends the least time in.
    """
    cases = [Case(PHI35.key, "prefill", int(m), 0, tokens=int(m)) for m in prefill_m]
    cases += [Case(PHI35.key, "decode", 1, int(p), tokens=1) for p in decode_past]
    return cases


def mobilenet_cases(batch) -> "list[Case]":
    return [Case(MOBILENETV2.key, "batch", int(b), 0, tokens=None, unit="images")
            for b in batch]


#: One fixed (phase, m, past) plan per model key, for `run_diagnostics`. A key absent from
#: this table (issue #78: `MINILM.key`) gets no diagnostic plan at all — never the `else`
#: branch's MobileNetV2-shaped batch plan, which is a different model's arithmetic and would
#: silently mis-describe what was actually run.
DIAGNOSTIC_PLANS = {
    PHI35.key: (("prefill", 1, 0), ("prefill", 8, 0), ("prefill", 128, 0), ("decode", 1, 1024)),
    MOBILENETV2.key: (("batch", 1, 0), ("batch", 16, 0)),
}


def cases_for(spec: ModelSpec, *, prefill_m=(), decode_past=(), batch=()) -> "list[Case] | None":
    """The case list for *spec*, or ``None`` if none exists yet — never a fallback.

    Raises ``ValueError`` for a spec key this repository does not know at all (`MODELS`
    fails closed on an unknown key rather than silently reusing some other model's cases).
    Returns ``None`` for a known key with no case/feed builder wired for it — issue #78's
    `MINILM`, ``capability="provenance_only"`` — which a caller must handle as an explicit
    skip (recorded in ``skipped_models``), not as "no cases means use MobileNetV2's".
    """
    if spec.key not in MODELS:
        raise ValueError(f"{spec.key!r} is not a known model key (see MODELS)")
    if spec.key == PHI35.key:
        return phi35_cases(prefill_m, decode_past)
    if spec.key == MOBILENETV2.key:
        return mobilenet_cases(batch)
    if spec.capability == "provenance_only":
        return None
    raise ValueError(  # pragma: no cover - guarded by MODELS/capability today
        f"{spec.key!r} has capability={spec.capability!r} but no case/feed builder is wired "
        f"for it; add one rather than falling back to another model's cases"
    )


def phi35_feeds(case: Case, np):
    """Build one feed dict. Pure given ``KV_SEED``; identical bytes for every arm."""
    if case.model_key != PHI35.key:
        raise ValueError(f"{case.model_key} is not {PHI35.key}")
    total = case.past + case.m
    feeds = {
        # Token ids are arbitrary but fixed: a benchmark that feeds random ids feeds a different
        # embedding-gather pattern each run, and the gather is not free.
        "input_ids": np.arange(1, case.m + 1, dtype=np.int64).reshape(1, case.m),
        "attention_mask": np.ones((1, total), dtype=np.int64),
    }
    rng = np.random.default_rng(KV_SEED)
    shape = (1, PHI35_KV_HEADS, case.past, PHI35_HEAD_DIM)
    for layer in range(PHI35_LAYERS):
        for kind in ("key", "value"):
            if case.past == 0:
                arr = np.empty(shape, dtype=np.float16)
            else:
                # Scaled to the magnitude a real cache carries; a cache of N(0,1) would push the
                # fp16 softmax into a regime the model never sees.
                arr = (rng.standard_normal(shape) * 0.02).astype(np.float16)
            feeds[f"past_key_values.{layer}.{kind}"] = arr
    return feeds


def mobilenet_feeds(case: Case, np):
    rng = np.random.default_rng(KV_SEED)
    return {"input": rng.standard_normal((case.m, 3, 224, 224), dtype=np.float32)}


def build_feeds(case: Case, np):
    if case.model_key == PHI35.key:
        return phi35_feeds(case, np)
    if case.model_key == MOBILENETV2.key:
        return mobilenet_feeds(case, np)
    raise ValueError(f"no feed builder for {case.model_key}")


def feeds_digest(feeds: dict) -> str:
    """A hash of the exact bytes fed, so two runs can be shown to have fed the same thing.

    Names *and* bytes: a feed dict with the right values under the wrong keys is a different
    input, and hashing values alone would call the two identical.
    """
    h = hashlib.sha256()
    for name in sorted(feeds):
        arr = feeds[name]
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(memoryview(arr).tobytes() if arr.size else b"")
    return h.hexdigest()


def kv_round_trip_bytes(case: Case) -> "dict | None":
    """Host bytes the KV cache costs this case, each way. ``None`` for non-KV models.

    This is a *model* of the traffic, derived from the shapes in the graph, not a measurement —
    and it is labelled as such in the artifact. It exists so the decode rows can be read against
    a bandwidth number instead of being a mystery.
    """
    if case.model_key != PHI35.key:
        return None
    upload = case.past * PHI35_BYTES_PER_PAST_TOKEN
    readback = (case.past + case.m) * PHI35_BYTES_PER_PAST_TOKEN
    return {
        "bytes_per_past_token": PHI35_BYTES_PER_PAST_TOKEN,
        "past_upload_bytes": upload,
        "present_readback_bytes": readback,
        "total_host_bytes": upload + readback,
        "derivation": "shape model (layers x 2 x kv_heads x head_dim x 2 B), not a measurement",
    }


# --------------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Arm:
    """One thing being compared. ``env`` is applied before the session is built, never after.

    Ordering is load-bearing and cost this project a run: the row tile is chosen at *translate*
    time, so a session built before the variable was set keeps the arm it was built with, and
    `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` set after `register_execution_provider_library`
    registers no allocator at all.
    """

    name: str
    providers: "tuple[str, ...]"
    env: "tuple[tuple[str, str], ...]" = ()
    role: str = "arm"

    def apply_env(self, environ) -> dict:
        """Set this arm's env, returning the previous values so a caller can restore them."""
        prev = {}
        for k, v in self.env:
            prev[k] = environ.get(k)
            environ[k] = v
        return prev


VULKAN_TILED = Arm("vulkan_tiled", (EP_NAME, CPU_EP), ((ROWS_ENV, "4"),), role="subject")
VULKAN_UNTILED = Arm("vulkan_untiled", (EP_NAME, CPU_EP), ((ROWS_ENV, "1"),), role="kill-switch")
CPU_ARM = Arm("cpu", (CPU_EP,), (), role="baseline")
ARMS = (VULKAN_TILED, VULKAN_UNTILED, CPU_ARM)


def arm_order(arms, repeat: int):
    """Alternate arm order per repeat.

    A fixed order does not cancel: GPU clock and page cache both drift monotonically inside a
    repeat, so whichever arm always runs last inherits that drift as a systematic bias. This is
    not a hypothetical — `ab_row_tile.py`'s first version reported a 0.905x "slowdown" at M=1,
    a shape where both arms bind identical SPIR-V, purely from the order.
    """
    seq = list(arms)
    if repeat % 2:
        seq.reverse()
    return seq


def is_null_control(case: Case) -> bool:
    """True where ``vulkan_tiled`` and ``vulkan_untiled`` build the *identical* pipeline.

    `ops::quant::gemv_tile` returns ``rows = 1`` whenever ``M == 1`` regardless of the cap, so
    at M=1 the two arms differ in no specialisation constant at all. Their measured ratio is
    therefore a direct read of this harness's noise floor and is not a result. Decode is M=1 by
    definition, so every decode row is also a null control for the row tile — which is exactly
    why decode needs the *CPU* arm to say anything.
    """
    return case.m == 1


# --------------------------------------------------------------------------------------------
# Statistics and throughput
# --------------------------------------------------------------------------------------------


def latency_stats(samples: "list[float]") -> dict:
    """Full distribution, never a single number.

    ``min`` is reported but is deliberately *not* the headline: on a shared, boost-clocked box
    the minimum is the run that got the machine to itself, which is the case a user least often
    experiences. The median is the headline and the spread travels with it.
    """
    xs = sorted(float(x) for x in samples)
    if not xs:
        return {"n": 0}
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs])
    return {
        "n": len(xs),
        "median_ms": med,
        "mean_ms": statistics.fmean(xs),
        "min_ms": xs[0],
        "max_ms": xs[-1],
        "p05_ms": _quantile(xs, 0.05),
        "p95_ms": _quantile(xs, 0.95),
        "mad_ms": mad,
        "stdev_ms": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "rsd": (statistics.stdev(xs) / med) if len(xs) > 1 and med else 0.0,
    }


def _quantile(ordered: "list[float]", q: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def throughput(case: Case, median_ms: float) -> "dict | None":
    """tokens/s (or images/s) from the median latency.

    Prefill divides by ``m`` because the run consumed ``m`` tokens; decode divides by 1 because
    a decode step emits one token however long its cache is. Dividing decode by ``past`` would
    manufacture a throughput that grows with the cache — the number getting better as the model
    gets slower.
    """
    if median_ms is None or median_ms <= 0:
        return None
    if case.tokens is not None:
        return {"unit": f"{case.unit}/s", "value": case.tokens * 1000.0 / median_ms,
                "divisor": case.tokens, "divisor_meaning": f"{case.unit} advanced by one run"}
    return {"unit": f"{case.unit}/s", "value": case.m * 1000.0 / median_ms,
            "divisor": case.m, "divisor_meaning": f"{case.unit} in one batch"}


def paired_ratios(a: "list[float]", b: "list[float]") -> dict:
    """Per-repeat ratios of two arms measured adjacent in time.

    Paired, because the two arms of a repeat share whatever the machine was doing; the *median
    of the per-repeat ratios* is the estimate that survives one disturbed repeat, and the
    [min, max] is the honest error bar. A ratio of medians would hide a repeat where one arm was
    displaced.
    """
    ratios = [x / y for x, y in zip(a, b) if y]
    if not ratios:
        return {"n": 0}
    return {
        "n": len(ratios),
        "median": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "ratios": [round(r, 6) for r in ratios],
    }


def exceeds_noise_floor(ratio: float, floor: dict) -> "bool | None":
    """Is a ratio bigger than the identical-pipeline control's own spread?

    Returns ``None`` when there is no control to read it against, which is a different state
    from "no" and must not collapse into it.
    """
    if not floor or floor.get("n", 0) == 0:
        return None
    lo, hi = floor.get("min"), floor.get("max")
    if lo is None or hi is None:
        return None
    return bool(ratio > hi or ratio < lo)


# --------------------------------------------------------------------------------------------
# Equivalence — every compared arm, every case
# --------------------------------------------------------------------------------------------

#: fp16 logits over a 32-layer decoder. `max_rel` over a GEMM output is a **cancellation meter,
#: not an accuracy meter** (§25.3), so the gate is argmax + top-k agreement plus an absolute
#: bound, and the relative figure is reported without being gated on.
PHI35_TOP_K = 10
#: The logit budget is a **fraction of the reference's own scale**, not a constant. A constant
#: was the first form and it is arbitrary: 0.5 is 3.8% of the logit scale at an empty cache and
#: 2.1% at past=1024, so a fixed number silently tightens as the cache grows. What fp16 supports
#: is relative: the error of a length-`K` fp16 accumulation grows like `sqrt(K) * 2^-11 * scale`,
#: and Phi-3.5 decode reduces over 3072-wide projections through 32 layers, so `sqrt(3072*32) *
#: 2^-11 ~= 0.15` — 15% of scale is the *theoretical* envelope. This budget is 3x tighter than
#: that envelope and still admits what the device actually produces.
#:
#: MEASURED 2026-08-06 on this box: the CPU EP is **bit-identical to itself** at
#: `intra_op_num_threads` 1, 2 and 4 (`_issue56_scratch/reorder_control.py`), so there is no
#: reference-side reordering noise to calibrate against and this bound is argued from fp16's
#: numerics, not fitted to the failure.
PHI35_LOGIT_SCALE_FRACTION = 0.05
#: What a decoder's logits are *for* is the next-token distribution. Two logit vectors that
#: disagree by 0.58 at scale 23 but induce the same distribution to within this bound are the
#: same model behaviour; two that agree in argmax while moving a probability by 0.3 are not.
PHI35_MAX_PROB_DELTA = 0.02
#: f32 CNN, fifty-plus Conv/BatchNorm/Clip layers. Same tolerance `rust/modelrunner` justifies
#: for this exact model in `bench/results/rust-model-runner/mobilenetv2-12.json`, borrowed
#: rather than re-invented so the two tools cannot disagree about what "AGREE" means.
MOBILENET_RTOL = 1e-2
MOBILENET_ATOL = 1e-4

MATCH = "MATCH"
DIVERGENT = "DIVERGENT"
UNMEASURED = "UNMEASURED"


def softmax(x, np):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def classify_logits(candidate, reference, np, *, top_k: int = PHI35_TOP_K,
                    scale_fraction: float = PHI35_LOGIT_SCALE_FRACTION,
                    max_prob_delta: float = PHI35_MAX_PROB_DELTA) -> dict:
    """Compare a decoder's logits against the CPU reference for the *same* run.

    Four clauses, all required: argmax agreement, top-k agreement, an absolute bound scaled to
    the reference's own logit magnitude, and a bound on the induced softmax distribution. The
    last one is the clause that means something: a sampler reads probabilities, so a difference
    that does not move a probability by more than `max_prob_delta` has not changed the model's
    behaviour, whatever it did to the raw logits. An all-zero reference is reported as its own
    state: the `argmax 0` defect produced two all-zero tensors that agreed perfectly.
    """
    cand = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    out = {
        "shape_candidate": list(cand.shape),
        "shape_reference": list(ref.shape),
        "top_k": top_k,
    }
    if cand.shape != ref.shape:
        out["verdict"] = DIVERGENT
        out["reason"] = "shape mismatch"
        return out
    # The last position is the one a sampler reads; earlier positions of a prefill are compared
    # too, but the argmax gate is on the position the model is actually asked about.
    last_c = cand.reshape(-1, cand.shape[-1])[-1]
    last_r = ref.reshape(-1, ref.shape[-1])[-1]
    out["reference_all_zero"] = bool(not np.any(ref))
    out["argmax_candidate"] = int(np.argmax(last_c))
    out["argmax_reference"] = int(np.argmax(last_r))
    k = min(top_k, last_r.shape[-1])
    topk_c = set(np.argsort(-last_c)[:k].tolist())
    topk_r = set(np.argsort(-last_r)[:k].tolist())
    out["topk_overlap"] = len(topk_c & topk_r)
    diff = np.abs(cand - ref)
    out["max_abs"] = float(diff.max()) if diff.size else 0.0
    out["reference_scale"] = float(np.abs(ref).max()) if ref.size else 0.0
    out["abs_budget"] = scale_fraction * out["reference_scale"]
    denom = np.maximum(np.abs(ref), 1e-6)
    out["max_rel"] = float((diff / denom).max()) if diff.size else 0.0
    out["max_rel_note"] = ("relative error over logits is a cancellation meter, not an accuracy "
                           "meter; reported, not gated on")
    p_c = softmax(last_c, np)
    p_r = softmax(last_r, np)
    out["max_prob_delta"] = float(np.abs(p_c - p_r).max())
    out["top1_prob_reference"] = float(p_r.max())
    out["top1_prob_candidate"] = float(p_c[out["argmax_reference"]])
    ok = (out["argmax_candidate"] == out["argmax_reference"]
          and out["topk_overlap"] == k
          and out["max_abs"] <= out["abs_budget"]
          and out["max_prob_delta"] <= max_prob_delta
          and not out["reference_all_zero"])
    out["verdict"] = MATCH if ok else DIVERGENT
    out["gate"] = (f"argmax equal AND top-{k} identical AND max_abs <= {scale_fraction} of the "
                   f"reference's own logit scale AND max softmax probability delta <= "
                   f"{max_prob_delta} AND reference not all-zero")
    return out


def classify_tensor(candidate, reference, np, *, rtol: float = MOBILENET_RTOL,
                    atol: float = MOBILENET_ATOL) -> dict:
    """Elementwise combined tolerance: ``|diff| <= atol + rtol * |ref|`` for every element.

    Elementwise and not aggregate. An aggregate ``max_rel <= rtol OR max_abs <= atol`` has a
    hole a planted error walks straight through: a large absolute error on a *small* element
    passes the relative clause while a large relative error on a *large* element passes the
    absolute one, so the two clauses cover for each other. `rust/modelrunner`'s recorded verdict
    for this model uses the aggregate form; this is strictly stricter, so a MATCH here implies a
    MATCH there and the two tools cannot disagree in the dangerous direction.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    out = {"shape_candidate": list(cand.shape), "shape_reference": list(ref.shape),
           "rtol": rtol, "atol": atol}
    if cand.shape != ref.shape:
        out["verdict"] = DIVERGENT
        out["reason"] = "shape mismatch"
        return out
    diff = np.abs(cand - ref)
    out["max_abs"] = float(diff.max()) if diff.size else 0.0
    out["max_rel"] = (float((diff / np.maximum(np.abs(ref), 1e-12)).max())
                      if diff.size else 0.0)
    out["reference_all_zero"] = bool(not np.any(ref))
    offenders = diff > (atol + rtol * np.abs(ref)) if diff.size else np.zeros(0, dtype=bool)
    out["elements_outside_tolerance"] = int(offenders.sum())
    # Per-row argmax: for a classifier the label is the answer, and a tolerance that passes while
    # the predicted class moves is not a useful tolerance.
    if cand.ndim == 2 and cand.shape[-1] > 1:
        out["argmax_agreement"] = int((np.argmax(cand, axis=-1) == np.argmax(ref, axis=-1)).sum())
        out["argmax_rows"] = int(cand.shape[0])
    ok = (out["elements_outside_tolerance"] == 0
          and not out["reference_all_zero"]
          and out.get("argmax_agreement", out.get("argmax_rows", 0)) == out.get("argmax_rows", 0))
    out["verdict"] = MATCH if ok else DIVERGENT
    out["gate"] = (f"every element |diff| <= {atol} + {rtol}*|ref| AND every row's argmax "
                   f"agrees AND the reference is not all-zero")
    return out


#: fp16 has ~2^-10 relative resolution, so an absolute difference of one ULP at the tensor's own
#: largest magnitude is not an error — it is the representation. `KV_ULP_BUDGET` ULPs of the
#: reference's own scale is the absolute floor; `KV_REL_TOL` is the relative slope.
FP16_EPS = 2.0 ** -10
KV_ULP_BUDGET = 16.0
KV_REL_TOL = 5e-2
#: An element may sit above the floor without being evidence of a defect — fp16 accumulation
#: over a 3072-wide reduction lands one element in ~400,000 a third of a floor above it. It may
#: **not** sit arbitrarily far above it. `KV_GROSS_MULTIPLE` floors is the point past which no
#: rounding story survives and the tensor is DIVERGENT on that element alone, however few there
#: are; `KV_MARGINAL_FRACTION` bounds how many elements may occupy the band in between. Without
#: the ceiling this would be a fudge — with it, the planted-error control still fails, because a
#: planted 7.0 at scale 4 is 50 floors out, not 1.3.
KV_GROSS_MULTIPLE = 8.0
KV_MARGINAL_FRACTION = 1e-4
#: Elements below this fraction of the tensor's RMS are excluded from the *reported* relative
#: figure. Borrowed verbatim from §25.3's discipline: relative error over an accumulation output
#: is a **cancellation meter, not an accuracy meter**, and near-zero elements make it grow with
#: the number of terms summed for a kernel that is exactly as accurate. The first version of this
#: gate had no such restriction and called every Vulkan arm DIVERGENT at `max_rel = 2.73` on a
#: tensor whose largest absolute error was one fp16 ULP.
KV_SIGNAL_FRACTION = 0.1


def classify_activation(candidate, reference, np, *, rel_tol: float = KV_REL_TOL,
                        ulp_budget: float = KV_ULP_BUDGET,
                        signal_fraction: float = KV_SIGNAL_FRACTION,
                        gross_multiple: float = KV_GROSS_MULTIPLE,
                        marginal_fraction: float = KV_MARGINAL_FRACTION) -> dict:
    """Compare an fp16 activation tensor (a `present.*` KV block) against the CPU reference.

    Three bands, elementwise::

        clean:    |diff| <= floor,             floor = ulp_budget*fp16_eps*max|ref| + rel_tol*|ref|
        marginal: floor < |diff| <= gross_multiple * floor
        gross:    |diff| >  gross_multiple * floor

    Any gross element is DIVERGENT on its own. Marginal elements are tolerated only up to
    `marginal_fraction` of the tensor, so a systematic small error across the tensor still fails
    while fp16's one-element-in-400k accumulation tail does not.

    The floor is a **sum, not an OR**: an aggregate ``max_rel <= rtol OR max_abs <= atol`` lets a
    large absolute error on a small element pass under the relative clause, which is exactly how
    a planted-error control fell through the first version of this function.

    ``max_rel_all`` and ``max_rel_signal`` are reported; only the band structure decides.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    out = {"shape_candidate": list(cand.shape), "shape_reference": list(ref.shape),
           "rel_tol": rel_tol, "ulp_budget": ulp_budget,
           "signal_fraction": signal_fraction, "gross_multiple": gross_multiple,
           "marginal_fraction": marginal_fraction}
    if cand.shape != ref.shape:
        out["verdict"] = DIVERGENT
        out["reason"] = "shape mismatch"
        return out
    if ref.size == 0:
        # An empty tensor agrees with an empty tensor, and that is a real state (a decode step
        # with an empty cache emits `present` of extent 1, never 0) — but say so on the face of
        # the record rather than letting a vacuous MATCH look like a checked one.
        out["verdict"] = MATCH
        out["empty"] = True
        return out
    diff = np.abs(cand - ref)
    absref = np.abs(ref)
    scale = float(absref.max())
    rms = float(np.sqrt(np.mean(ref ** 2)))
    out["max_abs"] = float(diff.max())
    out["reference_scale"] = scale
    out["reference_rms"] = rms
    out["abs_floor"] = ulp_budget * FP16_EPS * scale
    signal = absref >= signal_fraction * rms
    out["signal_elements"] = int(signal.sum())
    out["max_rel_signal"] = (float((diff[signal] / absref[signal]).max())
                             if signal.any() and absref[signal].min() > 0 else 0.0)
    out["max_rel_all"] = float((diff / np.maximum(absref, 1e-12)).max())
    out["max_rel_all_note"] = ("relative error over every element is a cancellation meter; "
                               "reported, not gated on")
    out["reference_all_zero"] = bool(not np.any(ref))
    floor = out["abs_floor"] + rel_tol * absref
    offenders = diff > floor
    gross = diff > gross_multiple * floor
    out["elements_outside_tolerance"] = int(offenders.sum())
    out["elements_gross"] = int(gross.sum())
    out["elements_total"] = int(ref.size)
    out["marginal_fraction_observed"] = float(offenders.sum()) / float(ref.size)
    out["worst_floor_multiple"] = (float((diff / np.maximum(floor, 1e-300)).max())
                                   if diff.size else 0.0)
    ok = (out["elements_gross"] == 0
          and out["marginal_fraction_observed"] <= marginal_fraction
          and not out["reference_all_zero"])
    out["verdict"] = MATCH if ok else DIVERGENT
    out["gate"] = (f"no element above {gross_multiple}x the floor "
                   f"({ulp_budget} fp16 ULP of the reference's own scale + {rel_tol}*|ref|), "
                   f"at most {marginal_fraction} of elements above the floor at all, "
                   f"AND the reference is not all-zero")
    return out


def classify_outputs(case: Case, candidate_outputs, reference_outputs, np) -> dict:
    """Verdict for one arm on one case. The *first* output is the task-relevant one.

    Phi-3.5's `logits` and MobileNetV2's `output` are both output 0. The KV outputs are compared
    too, under the tensor gate, because a decode step whose logits agree but whose `present`
    cache is wrong produces a *correct first token and a wrong sequence* — a failure that a
    single-step comparison would otherwise call MATCH.
    """
    if len(candidate_outputs) != len(reference_outputs):
        return {"verdict": DIVERGENT, "reason": "output count mismatch",
                "n_candidate": len(candidate_outputs), "n_reference": len(reference_outputs)}
    if case.model_key == PHI35.key:
        primary = classify_logits(candidate_outputs[0], reference_outputs[0], np)
    else:
        primary = classify_tensor(candidate_outputs[0], reference_outputs[0], np)
    worst_kv = None
    kv_divergent = 0
    for c, r in zip(candidate_outputs[1:], reference_outputs[1:]):
        v = classify_activation(c, r, np)
        if v["verdict"] == DIVERGENT:
            kv_divergent += 1
            if worst_kv is None or v.get("max_abs", 0) > worst_kv.get("max_abs", 0):
                worst_kv = v
    verdict = MATCH if primary["verdict"] == MATCH and kv_divergent == 0 else DIVERGENT
    return {
        "verdict": verdict,
        "primary": primary,
        "secondary_outputs": len(candidate_outputs) - 1,
        "secondary_divergent": kv_divergent,
        "worst_secondary": worst_kv,
        "secondary_note": ("KV/present outputs compared under `classify_activation` — a floor "
                           "scaled to the tensor's own fp16 resolution, a gross ceiling, and a "
                           "bound on how many elements may sit between them; logits agreement "
                           "alone would not catch a wrong cache"),
    }


def bitwise_identical(a_outputs, b_outputs, np) -> dict:
    """Are two arms' outputs *byte for byte* the same?

    The identical-pipeline null control claims that at `M = 1` the tiled and untiled arms compile
    and bind the same SPIR-V, so their outputs must be bit-identical — not close, identical. A
    latency ratio alone cannot tell a real 1.04x from a scheduling artefact, but a bitwise
    disagreement at `M = 1` would falsify the claim that the pipelines are the same, and that is
    worth stating on the face of the record rather than assuming.
    """
    out = {"outputs": len(a_outputs), "identical": True, "first_difference": None}
    if len(a_outputs) != len(b_outputs):
        out["identical"] = False
        out["first_difference"] = "output count"
        return out
    for i, (a, b) in enumerate(zip(a_outputs, b_outputs)):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            out["identical"] = False
            out["first_difference"] = f"output[{i}]"
            if a.shape == b.shape and a.dtype == b.dtype:
                out["max_abs"] = float(np.abs(a.astype(np.float64)
                                              - b.astype(np.float64)).max())
            break
    return out


# --------------------------------------------------------------------------------------------
# Diagnostics — "utilize the device" made falsifiable
# --------------------------------------------------------------------------------------------


def dispatch_diagnosis(counters: "dict | None", inference_count: int) -> dict:
    """What the EP actually did, per inference, from the EP's own counters.

    Every figure is per-inference where that is meaningful, because the raw counters are
    process-cumulative and a cumulative counter divided by the wrong denominator is how this
    project once read a 1.78x "improvement" that was really an iteration ratio.
    """
    c = counters or {}
    live = c.get("subgraphs_live")
    disp = c.get("dispatches_executed")
    out = {
        "islands": live,
        "compute_calls": c.get("compute_calls"),
        "dispatches_executed": disp,
        "compute_failures": c.get("compute_failures"),
        "inferences": inference_count,
        "dispatches_per_inference": (disp / inference_count) if disp and inference_count else None,
        "island_crossings_per_inference": live,
        "outputs_host_resident": c.get("outputs_host_resident"),
        "outputs_device_resident": c.get("outputs_device_resident"),
        "note": ("counters are process-cumulative; per-inference figures divide by the number of "
                 "inferences this process ran, never by a repeat count"),
    }
    return out


def fallback_diagnosis(provider_node_counts: "dict | None") -> dict:
    """How much of the graph ORT gave to somebody else.

    `claimed` is what the EP asked for; this is what ORT's own profile says *executed* where.
    They are different questions and the second is the one a latency number is about.
    """
    counts = dict(provider_node_counts or {})
    total = sum(counts.values())
    vk = counts.get(EP_NAME, 0)
    return {
        "provider_node_counts": counts,
        "total_node_executions": total,
        "vulkan_node_executions": vk,
        "cpu_fallback_node_executions": total - vk,
        "vulkan_share": (vk / total) if total else None,
        "witness": "onnxruntime profile (cat=Node, args.provider)",
    }


def bandwidth_proxy(case: Case, median_ms: float) -> "dict | None":
    """Achieved host-transfer bandwidth implied by the KV round trip, if any.

    A *proxy*, and named one: it assumes the modelled bytes actually crossed the link, which is
    true for the shipping (unbound, host-staged) lane and false for the device-resident lane.
    It exists so a decode row that is three times slower than the CPU has a number attached to
    the mechanism, instead of a story.
    """
    kv = kv_round_trip_bytes(case)
    if not kv or not median_ms:
        return None
    gb = kv["total_host_bytes"] / (1 << 30)
    return {
        "modelled_host_gib": gb,
        "implied_gib_per_s": gb / (median_ms / 1000.0),
        "assumes": "the shipping host-staged lane; false if outputs are bound device-resident",
    }
