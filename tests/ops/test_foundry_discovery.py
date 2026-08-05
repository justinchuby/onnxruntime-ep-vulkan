"""Negative tests for `rust/tools/foundry_discovery.py`. Owner: Trinity.

WHY THIS EXISTS
----------------
Issue #11: `tests/ops/test_phi35.py` hardcoded a Foundry cache subpath that went stale when
Foundry's own on-disk layout changed between the Phi-3.5 block being written and the gpt-oss-20b
block being written (see the module docstring in `foundry_discovery.py`). The fix is a resolver
that reports four distinct failure modes loudly instead of ever guessing:

  1. missing       -- nothing cached under the requested variant name at all
  2. duplicate     -- more than one cached entry matches (ambiguous; refuse to pick)
  3. stale         -- catalogued but not actually cached, or `cached: true` yet the file the
                      manifest points at is not on disk
  4. wrong-provider-- the variant name is cached, but under a different execution provider

Every one of these must raise `FoundryDiscoveryError` with a message a human can act on
directly, not skip silently and not fall through to a "close enough" match.

These tests exercise `select_variant` (pure, over synthetic CLI-schema manifests -- no real
Foundry install needed) for cases 1-4, and `resolve_model_path`'s filesystem fallback (real
`tmp_path` directory trees, `foundry_exe` pointed at a name that can never resolve so the CLI
strategy always misses) for the subset of cases that are meaningful on disk. The happy path is
covered at both layers so a change to either layer alone cannot silently break resolution.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "rust" / "tools"))
import foundry_discovery as fd  # noqa: E402

_SPEC = fd.FoundryModelSpec(
    variant_name="Widget-mini-instruct-cuda-gpu",
    execution_provider="CUDAExecutionProvider",
    onnx_filename="widget-mini.onnx",
    download_alias="widget-mini",
)


def _variant(
    *,
    variant_name: str = _SPEC.variant_name,
    execution_provider: str = _SPEC.execution_provider,
    cached: bool = True,
    variant_id: str = "v1",
    cache_path: str = r"C:\fake\cache\Microsoft\Widget-mini-instruct-cuda-gpu\v1",
) -> dict:
    """A synthetic row matching the `foundry cache list --variants -o json` schema."""
    return {
        "variantName": variant_name,
        "executionProvider": execution_provider,
        "variantId": variant_id,
        "cached": cached,
        "cachePath": cache_path,
    }


# ---------------------------------------------------------------------------
# `select_variant` — pure, synthetic-manifest tests. No Foundry install required.
# ---------------------------------------------------------------------------


class TestSelectVariantHappyPath:
    def test_unique_cached_correct_provider_is_selected(self) -> None:
        variants = [_variant()]
        chosen = fd.select_variant(variants, _SPEC)
        assert chosen["cachePath"] == _variant()["cachePath"]

    def test_unrelated_variants_are_ignored(self) -> None:
        variants = [
            _variant(variant_name="SomeOtherModel-cuda-gpu", variant_id="other"),
            _variant(),
        ]
        chosen = fd.select_variant(variants, _SPEC)
        assert chosen["variantId"] == "v1"


class TestSelectVariantMissing:
    """Case 1: nothing cached under the requested variant name."""

    def test_empty_manifest_raises_with_download_remedy(self) -> None:
        with pytest.raises(fd.FoundryDiscoveryError, match="no cached Foundry variant"):
            fd.select_variant([], _SPEC)

    def test_manifest_with_only_unrelated_models_raises(self) -> None:
        variants = [_variant(variant_name="TotallyDifferentModel-cpu", variant_id="x")]
        with pytest.raises(fd.FoundryDiscoveryError, match=_SPEC.variant_name):
            fd.select_variant(variants, _SPEC)

    def test_missing_error_names_the_download_alias(self) -> None:
        with pytest.raises(fd.FoundryDiscoveryError, match=_SPEC.download_alias):
            fd.select_variant([], _SPEC)


class TestSelectVariantDuplicate:
    """Case 2: more than one cached entry matches — refuse to pick arbitrarily."""

    def test_two_cached_variants_raise_ambiguous(self) -> None:
        variants = [
            _variant(variant_id="v1", cache_path=r"C:\fake\v1"),
            _variant(variant_id="v2", cache_path=r"C:\fake\v2"),
        ]
        with pytest.raises(fd.FoundryDiscoveryError, match="ambiguous"):
            fd.select_variant(variants, _SPEC)

    def test_ambiguous_error_lists_all_candidates(self) -> None:
        variants = [
            _variant(variant_id="v1", cache_path=r"C:\fake\v1"),
            _variant(variant_id="v2", cache_path=r"C:\fake\v2"),
        ]
        with pytest.raises(fd.FoundryDiscoveryError) as exc_info:
            fd.select_variant(variants, _SPEC)
        msg = str(exc_info.value)
        assert "v1" in msg and "v2" in msg
        # The candidate list is embedded via repr(), so backslashes come out doubled.
        assert repr(r"C:\fake\v1") in msg and repr(r"C:\fake\v2") in msg

    def test_never_silently_picks_the_first_or_newest(self) -> None:
        # Regression guard for the exact anti-pattern this module refuses to reproduce: a
        # duplicate must never resolve to *any* single candidate, first or last in list order.
        variants = [
            _variant(variant_id="v1", cache_path=r"C:\fake\v1"),
            _variant(variant_id="v2", cache_path=r"C:\fake\v2"),
        ]
        with pytest.raises(fd.FoundryDiscoveryError):
            fd.select_variant(variants, _SPEC)
        with pytest.raises(fd.FoundryDiscoveryError):
            fd.select_variant(list(reversed(variants)), _SPEC)


class TestSelectVariantStale:
    """Case 3: catalogued but `cached: false` — never treated as present."""

    def test_uncached_entry_raises_stale(self) -> None:
        variants = [_variant(cached=False)]
        with pytest.raises(fd.FoundryDiscoveryError, match="not actually cached"):
            fd.select_variant(variants, _SPEC)

    def test_stale_error_names_download_remedy(self) -> None:
        variants = [_variant(cached=False)]
        with pytest.raises(fd.FoundryDiscoveryError, match=_SPEC.download_alias):
            fd.select_variant(variants, _SPEC)


class TestSelectVariantWrongProvider:
    """Case 4: right variant name, wrong execution provider — never silently substituted."""

    def test_cpu_variant_is_not_substituted_for_cuda_request(self) -> None:
        variants = [_variant(execution_provider="CPUExecutionProvider")]
        with pytest.raises(fd.FoundryDiscoveryError, match="not under execution provider"):
            fd.select_variant(variants, _SPEC)

    def test_wrong_provider_error_names_both_providers(self) -> None:
        variants = [_variant(execution_provider="CPUExecutionProvider")]
        with pytest.raises(fd.FoundryDiscoveryError) as exc_info:
            fd.select_variant(variants, _SPEC)
        msg = str(exc_info.value)
        assert _SPEC.execution_provider in msg
        assert "CPUExecutionProvider" in msg

    def test_correct_provider_among_others_is_still_selected(self) -> None:
        # Right provider present alongside an unrelated wrong-provider entry for the same name:
        # must select the right one, not raise merely because *a* mismatch exists somewhere.
        variants = [
            _variant(execution_provider="CPUExecutionProvider", variant_id="cpu-build"),
            _variant(execution_provider="CUDAExecutionProvider", variant_id="cuda-build"),
        ]
        chosen = fd.select_variant(variants, _SPEC)
        assert chosen["variantId"] == "cuda-build"


# ---------------------------------------------------------------------------
# `resolve_model_path` filesystem fallback — real `tmp_path` trees, CLI forced to miss by
# pointing `foundry_exe` at a name that can never be found on PATH.
# ---------------------------------------------------------------------------

_UNRESOLVABLE_FOUNDRY_EXE = "foundry-exe-that-does-not-exist-anywhere-on-this-machine.exe"


def _make_cache_file(cache_root: pathlib.Path, variant_dir: str, onnx_filename: str) -> None:
    d = cache_root / "Microsoft" / variant_dir / "v1"
    d.mkdir(parents=True, exist_ok=True)
    (d / onnx_filename).write_bytes(b"not a real onnx file, just a presence marker")


class TestFilesystemFallback:
    def test_missing_raises_when_family_dir_absent(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(fd.FoundryDiscoveryError, match="no cached Foundry variant"):
            fd.resolve_model_path(
                _SPEC, foundry_exe=_UNRESOLVABLE_FOUNDRY_EXE, cache_root=tmp_path
            )

    def test_missing_raises_when_only_other_models_present(self, tmp_path: pathlib.Path) -> None:
        _make_cache_file(tmp_path, "SomeOtherModel-cpu", "other.onnx")
        with pytest.raises(fd.FoundryDiscoveryError, match="no cached Foundry variant"):
            fd.resolve_model_path(
                _SPEC, foundry_exe=_UNRESOLVABLE_FOUNDRY_EXE, cache_root=tmp_path
            )

    def test_happy_path_resolves_the_real_file(self, tmp_path: pathlib.Path) -> None:
        _make_cache_file(tmp_path, _SPEC.variant_name, _SPEC.onnx_filename)
        resolved = fd.resolve_model_path(
            _SPEC, foundry_exe=_UNRESOLVABLE_FOUNDRY_EXE, cache_root=tmp_path
        )
        assert resolved.name == _SPEC.onnx_filename
        assert resolved.is_file()

    def test_versioned_variant_suffix_is_still_found(self, tmp_path: pathlib.Path) -> None:
        # This is the exact regression issue #11 reports: Foundry renamed
        # "Phi-3.5-mini-instruct-cuda-gpu" to "Phi-3.5-mini-instruct-cuda-gpu-2" between the
        # hardcoded path being written and today. A version-tolerant search must still find it.
        _make_cache_file(tmp_path, _SPEC.variant_name + "-2", _SPEC.onnx_filename)
        resolved = fd.resolve_model_path(
            _SPEC, foundry_exe=_UNRESOLVABLE_FOUNDRY_EXE, cache_root=tmp_path
        )
        assert resolved.is_file()

    def test_ambiguous_versions_both_present_raises(self, tmp_path: pathlib.Path) -> None:
        _make_cache_file(tmp_path, _SPEC.variant_name, _SPEC.onnx_filename)
        _make_cache_file(tmp_path, _SPEC.variant_name + "-2", _SPEC.onnx_filename)
        with pytest.raises(fd.FoundryDiscoveryError, match="ambiguous"):
            fd.resolve_model_path(
                _SPEC, foundry_exe=_UNRESOLVABLE_FOUNDRY_EXE, cache_root=tmp_path
            )

    def test_unrelated_family_prefix_is_not_matched(self, tmp_path: pathlib.Path) -> None:
        # A model whose name merely *starts with* the same characters but is not actually a
        # version of the requested variant (e.g. a differently-named sibling model) must not
        # be treated as a match — the fallback's prefix check requires a "-" separator, not a
        # bare string-prefix match.
        _make_cache_file(tmp_path, _SPEC.variant_name + "XL", _SPEC.onnx_filename)
        with pytest.raises(fd.FoundryDiscoveryError, match="no cached Foundry variant"):
            fd.resolve_model_path(
                _SPEC, foundry_exe=_UNRESOLVABLE_FOUNDRY_EXE, cache_root=tmp_path
            )

    def test_manifest_says_cached_but_file_absent_raises_stale(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Directory exists (as Foundry would leave one mid-download or after a partial cleanup)
        # but the actual .onnx file inside it does not — must not be silently treated as present.
        d = tmp_path / "Microsoft" / _SPEC.variant_name / "v1"
        d.mkdir(parents=True, exist_ok=True)
        with pytest.raises(fd.FoundryDiscoveryError, match="no cached Foundry variant"):
            fd.resolve_model_path(
                _SPEC, foundry_exe=_UNRESOLVABLE_FOUNDRY_EXE, cache_root=tmp_path
            )


# ---------------------------------------------------------------------------
# Why there is no dedicated filesystem-fallback "wrong provider" test.
# ---------------------------------------------------------------------------
#
# In the CLI-manifest strategy, `executionProvider` is an independent field on the same
# `variantName` and must be checked explicitly (see TestSelectVariantWrongProvider above) —
# that is real defense-in-depth, because the manifest could in principle report any provider
# string next to any variant name.
#
# In the filesystem fallback, Foundry itself folds the execution provider into the variant
# *directory name* (e.g. the "-cuda-gpu" suffix). A wrong-provider build of the same model
# lives under a differently-named directory (e.g. "Widget-mini-instruct-cpu"), which the exact
# `variant_name` prefix match in `_variants_from_filesystem` will not match at all — so a
# provider mismatch at the filesystem layer surfaces as "missing", not "wrong provider", by
# construction. Asserting a distinct "wrong provider" error class here would test a
# distinction Foundry's own directory-naming scheme does not preserve on disk.
def test_filesystem_fallback_wrong_provider_directory_reports_as_missing(
    tmp_path: pathlib.Path,
) -> None:
    _make_cache_file(tmp_path, "Widget-mini-instruct-cpu", _SPEC.onnx_filename)
    with pytest.raises(fd.FoundryDiscoveryError, match="no cached Foundry variant"):
        fd.resolve_model_path(_SPEC, foundry_exe=_UNRESOLVABLE_FOUNDRY_EXE, cache_root=tmp_path)


# ---------------------------------------------------------------------------
# `default_cache_root` — override behaviour.
# ---------------------------------------------------------------------------


class TestDefaultCacheRoot:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ONNXRUNTIME_EP_VULKAN_FOUNDRY_CACHE", r"C:\somewhere\else")
        monkeypatch.delenv("FOUNDRY_CACHE_DIR", raising=False)
        assert fd.default_cache_root() == pathlib.Path(r"C:\somewhere\else")

    def test_default_is_under_home_dot_foundry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ONNXRUNTIME_EP_VULKAN_FOUNDRY_CACHE", raising=False)
        monkeypatch.delenv("FOUNDRY_CACHE_DIR", raising=False)
        root = fd.default_cache_root()
        assert root == pathlib.Path.home() / ".foundry" / "cache" / "models"


# ---------------------------------------------------------------------------
# `_foundry_cache_list_json` — CLI-unavailable path must return None, never [] or raise.
# ---------------------------------------------------------------------------


def test_cli_unavailable_returns_none_not_empty_list() -> None:
    # `None` means "could not ask the CLI"; `[]` would incorrectly mean "asked, cache is empty".
    # Conflating the two would make `resolve_model_path` skip the filesystem fallback on a
    # machine that simply does not have Foundry's CLI on PATH.
    assert fd._foundry_cache_list_json(_UNRESOLVABLE_FOUNDRY_EXE, timeout=5.0) is None
