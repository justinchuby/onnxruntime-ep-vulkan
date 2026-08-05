#!/usr/bin/env python
"""Where is a specific Foundry Local model, on disk, right now? Owner: Trinity.

WHY THIS EXISTS
---------------
`tests/ops/test_phi35.py` used to hardcode a Foundry cache subpath:

    C:\\Users\\...\\.foundry\\cache\\models\\Microsoft\\Phi-3.5-mini-instruct-cuda-gpu
                                              \\cuda-int4-rtn-block-32\\...\\.onnx

Foundry Local 0.10.2 downloads the same model to a *different* subpath --
`Phi-3.5-mini-instruct-cuda-gpu-2\\v2\\...` -- because Foundry's own on-disk layout is versioned
by the CLI's internal catalog revision, not by anything this repo controls. The hardcoded path
went stale under us with no code change on either side, and the only way anyone noticed was a
manually-created directory junction bridging the two (issue #11).

A hardcoded path is a *guess* about a version. Guessing again, only with a wildcard glob and
"pick the first/newest one found", replaces one arbitrary choice with another -- exactly the
failure mode this module refuses to reproduce. **Two facts must both be true before this module
returns a path: exactly one cached, correctly-provisioned variant matches the exact model
identity asked for, and the file the manifest says is there is actually there.** Zero matches,
more than one match, a match under the wrong execution provider, or a match the catalog has not
actually downloaded are all reported as distinct, actionable failures -- never as a silent pick.

TWO DISCOVERY STRATEGIES, IN ORDER
-----------------------------------
1. **The Foundry CLI's own cache manifest** (`foundry cache list --verbose --variants -o json`).
   This is authoritative: it is Foundry's own bookkeeping of what is cached, under which
   `executionProvider`, at which `cachePath`. Preferred whenever the CLI is on `PATH` and
   answers.
2. **A constrained, version-tolerant filesystem search**, used only when the CLI itself is
   unavailable (not installed, not on `PATH`, or the invocation failed/timed out). The search is
   restricted to `<cache_root>/Microsoft/<variant_name>[-*]/**/<onnx_filename>` -- an *exact*
   family-name prefix, never a fuzzy or wildcard family guess -- so a hit can only ever be a
   different *version* of the model asked for, never a different model. Foundry's own naming
   convention folds the execution provider into `variant_name` itself (e.g. the `-cuda-gpu`
   suffix), so this constraint also rules out a wrong-provider substitution by construction; see
   `select_variant` for the CLI-manifest path, where `executionProvider` is a separate field and
   is therefore checked explicitly.

NO CLOCK, NO ARBITRARY CHOICE. Identity and presence only.

USAGE
    import foundry_discovery as fd

    spec = fd.FoundryModelSpec(
        variant_name="Phi-3.5-mini-instruct-cuda-gpu",
        execution_provider="CUDAExecutionProvider",
        onnx_filename="phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
        download_alias="phi-3.5-mini",
    )
    onnx_path = fd.resolve_model_path(spec)  # raises FoundryDiscoveryError, never guesses
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess


class FoundryDiscoveryError(RuntimeError):
    """The exact model identity requested does not resolve to exactly one usable cache entry.

    Raised for all four negative cases this module refuses to paper over: missing, ambiguous
    (duplicate), stale (catalogued but not actually cached, or cached but missing from disk), and
    wrong execution provider. The message always names the exact identity that was asked for and
    the remedy command, so a reader never has to re-derive either.
    """


@dataclasses.dataclass(frozen=True)
class FoundryModelSpec:
    """The exact model identity this repo depends on -- never a family name alone.

    ``variant_name`` must match Foundry's own ``variantName`` (== the on-disk directory name
    minus any catalog-revision suffix), not a display name or alias. ``execution_provider`` must
    match Foundry's own ``executionProvider`` string exactly (e.g. ``CUDAExecutionProvider``).
    ``download_alias`` is what a human types to fetch it (``foundry model download <alias>``);
    it is only ever used in error messages, never in discovery itself.
    """

    variant_name: str
    execution_provider: str
    onnx_filename: str
    download_alias: str


def default_cache_root() -> pathlib.Path:
    """The Foundry Local model-cache root, overridable for tests and non-default installs."""
    override = os.environ.get("ONNXRUNTIME_EP_VULKAN_FOUNDRY_CACHE") or os.environ.get(
        "FOUNDRY_CACHE_DIR"
    )
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".foundry" / "cache" / "models"


def _foundry_cache_list_json(foundry_exe: str, timeout: float = 30.0) -> "list[dict] | None":
    """Runs the Foundry CLI's own cache manifest. Returns ``None`` -- not ``[]`` -- if the CLI
    itself could not be asked (not installed, not on `PATH`, timed out, or answered something
    this module cannot parse), so the caller can fall back to the filesystem strategy rather than
    misreading "the CLI has nothing to say" as "the cache is empty".
    """
    try:
        r = subprocess.run(
            [foundry_exe, "cache", "list", "--verbose", "--variants", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        doc = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    variants = doc.get("variants")
    if not isinstance(variants, list):
        return None
    return variants


def _variants_from_filesystem(cache_root: pathlib.Path, spec: FoundryModelSpec) -> "list[dict]":
    """The version-tolerant filesystem fallback. See the module docstring for the exact
    constraint (family-name prefix, never a fuzzy guess) and why it does not need to check
    ``executionProvider`` separately.
    """
    family_root = cache_root / "Microsoft"
    if not family_root.is_dir():
        return []
    variants: list[dict] = []
    for child in sorted(family_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name != spec.variant_name and not child.name.startswith(
            spec.variant_name + "-"
        ):
            continue
        for onnx_path in sorted(child.rglob(spec.onnx_filename)):
            variants.append(
                {
                    "variantName": spec.variant_name,
                    "executionProvider": spec.execution_provider,
                    "variantId": child.name,
                    "cached": True,
                    "cachePath": str(onnx_path.parent),
                }
            )
    return variants


def select_variant(variants: "list[dict]", spec: FoundryModelSpec) -> dict:
    """Pure selection over an already-fetched variant list -- no subprocess, no filesystem.

    Kept separate from ``resolve_model_path`` so the four negative cases (missing, ambiguous,
    stale, wrong-provider) are unit-testable against synthetic manifests, deterministically and
    without a real Foundry install. Raises ``FoundryDiscoveryError`` rather than returning
    ``None`` or picking arbitrarily -- there is no case in which this function silently guesses.
    """
    by_name = [v for v in variants if v.get("variantName") == spec.variant_name]
    if not by_name:
        raise FoundryDiscoveryError(
            f"no cached Foundry variant named {spec.variant_name!r} found. "
            f"Run `foundry model download {spec.download_alias}` to fetch it."
        )

    by_provider = [v for v in by_name if v.get("executionProvider") == spec.execution_provider]
    if not by_provider:
        found = sorted({v.get("executionProvider") for v in by_name})
        raise FoundryDiscoveryError(
            f"found {spec.variant_name!r} cached, but not under execution provider "
            f"{spec.execution_provider!r} -- the cache has provider(s) {found} instead. "
            f"This repo requires exactly {spec.execution_provider!r} and will not silently "
            f"substitute another provider's build. Run "
            f"`foundry model download {spec.download_alias}` to fetch the correct variant."
        )

    cached = [v for v in by_provider if v.get("cached") is True]
    if not cached:
        raise FoundryDiscoveryError(
            f"{spec.variant_name!r} ({spec.execution_provider}) is known to the Foundry catalog "
            f"but is not actually cached locally (stale catalog entry, or never downloaded). "
            f"Run `foundry model download {spec.download_alias}` to fetch it."
        )

    if len(cached) > 1:
        candidates = [(v.get("variantId"), v.get("cachePath")) for v in cached]
        raise FoundryDiscoveryError(
            f"ambiguous: {len(cached)} cached variants all match "
            f"{spec.variant_name!r} ({spec.execution_provider}): {candidates}. Refusing to "
            f"choose arbitrarily -- remove the stale entries with `foundry cache clear "
            f"<variant>` (keeping the one you want) before running this again."
        )

    return cached[0]


def resolve_model_path(
    spec: FoundryModelSpec,
    *,
    foundry_exe: str = "foundry",
    cache_root: "pathlib.Path | None" = None,
) -> pathlib.Path:
    """Resolves ``spec`` to exactly one ``.onnx`` file, or raises ``FoundryDiscoveryError``.

    Tries the Foundry CLI's cache manifest first; falls back to a constrained filesystem search
    only if the CLI is unavailable. Either way, the manifest's/filesystem's claim that the file
    is cached is verified against the actual file on disk before returning -- a cache entry that
    says ``cached: true`` for a file that is not actually there is reported as the same kind of
    staleness a duplicate or missing entry is, not silently trusted.
    """
    resolved_cache_root = cache_root if cache_root is not None else default_cache_root()

    variants = _foundry_cache_list_json(foundry_exe)
    if variants is not None:
        source = "Foundry CLI cache manifest (`foundry cache list --variants -o json`)"
    else:
        variants = _variants_from_filesystem(resolved_cache_root, spec)
        source = f"filesystem search under {resolved_cache_root / 'Microsoft'}"

    try:
        chosen = select_variant(variants, spec)
    except FoundryDiscoveryError as exc:
        raise FoundryDiscoveryError(f"[{source}] {exc}") from exc

    onnx_path = pathlib.Path(chosen["cachePath"]) / spec.onnx_filename
    if not onnx_path.is_file():
        raise FoundryDiscoveryError(
            f"[{source}] resolved {spec.variant_name!r} ({spec.execution_provider}) to "
            f"{chosen['cachePath']}, but {spec.onnx_filename} is not there -- the cache "
            f"manifest is stale relative to disk. Run "
            f"`foundry model download {spec.download_alias}` again, or `foundry cache clear "
            f"{spec.download_alias}` first if the entry is corrupted."
        )
    return onnx_path
