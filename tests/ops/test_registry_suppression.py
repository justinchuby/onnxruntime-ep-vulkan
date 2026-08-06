"""Falsification suite for `_registry_suppression.py` — issue #1's registry-based arm.

R9: *evidence scales only with falsifying instruments, not agreeing ones.* The claims
this file exists to falsify:

  - **"This module can be imported and called on a non-Windows platform without
    crashing."** `suppress_registry_entries` (and every convenience wrapper built on it)
    must turn platform unavailability into `RegistryMechanismUnavailable`, not let a bare
    `ModuleNotFoundError`/`ImportError` escape. This is a real regression this file is
    named for: an earlier revision imported `winreg` unconditionally at the top of
    `suppress_registry_entries` itself (not just inside the already-guarded
    `_open_key_all_access` helper it calls), so on the Linux CI lane — which reaches this
    arm's `with ctx:` line whenever the env-var arms already suppressed the ICD and the
    generator body still runs to no-op through it — the test crashed with
    `ModuleNotFoundError: No module named 'winreg'` instead of degrading gracefully
    (`test_criterion4_icd_polarity_witness`, Linux job of PR #45's own CI, 2026-08-06).
    Every public entry point is exercised here under a simulated non-Windows platform so
    this cannot regress silently again.
  - **"A missing registry key is indistinguishable from a real crash."** Both must raise
    `RegistryMechanismUnavailable`, but only the missing-key path may claim "nothing is
    registered there".
  - **"The manifest classifier only works when the manifest is readable."** It must also
    classify correctly from the path alone when the file cannot be read, and must not
    over-match an unrelated layer manifest.

Everything here runs without a GPU, without the EP library, and without real HKLM write
access — these are properties of the module's own control flow, observable on any
platform including the ones the mechanism does not apply to.
"""

from __future__ import annotations

import sys

import pytest

import _registry_suppression as rs


# ---------------------------------------------------------------------------
# The regression this file is named for: no bare ImportError/ModuleNotFoundError on a
# non-Windows platform, from any public entry point.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_context_manager",
    [
        lambda: rs.suppress_registry_entries(rs.KEY_PATH_DRIVERS),
        lambda: rs.suppress_registry_entries(rs.KEY_PATH_EXPLICIT_LAYERS, name_filter=lambda _n: True),
        rs.suppress_icd_registry,
        rs.suppress_validation_layer_registry,
    ],
    ids=["suppress_registry_entries(drivers)", "suppress_registry_entries(layers)",
         "suppress_icd_registry", "suppress_validation_layer_registry"],
)
def test_every_entry_point_degrades_on_a_simulated_non_windows_platform(
    monkeypatch, make_context_manager
):
    """Simulates the exact failure this test is named for (Linux CI, 2026-08-06): every
    public way to obtain the context manager, then entering it, must raise
    `RegistryMechanismUnavailable` — never a bare `ModuleNotFoundError`/`ImportError` —
    when `sys.platform` is not `"win32"`. Monkeypatching `sys.platform` (rather than
    running this only on real Linux CI) makes the assertion machine-independent: it is
    checked on every platform this suite runs on, including this Windows dev box.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    cm = make_context_manager()
    with pytest.raises(rs.RegistryMechanismUnavailable):
        with cm:
            pytest.fail("the context manager must never yield on a non-Windows platform")


def test_non_windows_platform_message_names_the_platform(monkeypatch):
    """The `RegistryMechanismUnavailable` raised for a non-Windows platform must name the
    offending platform string, distinguishing it from the missing-key/no-access messages
    below (a caller reading `suppression_attempts` needs to tell these apart)."""
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(rs.RegistryMechanismUnavailable, match="platform='linux'"):
        with rs.suppress_registry_entries(rs.KEY_PATH_DRIVERS):
            pass


# ---------------------------------------------------------------------------
# Missing key vs. no access: two distinct RegistryMechanismUnavailable causes, both on
# the real Windows registry API where this dev box's own credentials genuinely differ.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_missing_key_is_mechanism_unavailable_not_a_crash():
    with pytest.raises(rs.RegistryMechanismUnavailable, match="does not exist"):
        with rs.suppress_registry_entries(r"SOFTWARE\Khronos\Vulkan\ThisKeyDoesNotExist12345"):
            pass


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_write_protected_key_is_mechanism_unavailable_not_a_crash():
    """A real HKLM key that exists but this process cannot write — the actual shape of
    "no admin rights" on a non-elevated dev box, as opposed to the simulated-platform
    tests above."""
    with pytest.raises(rs.RegistryMechanismUnavailable, match="no write access"):
        with rs.suppress_registry_entries(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"):
            pass


# ---------------------------------------------------------------------------
# The manifest classifier: JSON-name path, path-fallback path, and the negative case
# that proves the fallback does not over-match.
# ---------------------------------------------------------------------------


def test_manifest_classifier_reads_the_declared_layer_name(tmp_path):
    manifest = tmp_path / "some_odd_filename.json"
    manifest.write_text(
        '{"layer": {"name": "VK_LAYER_KHRONOS_validation"}}', encoding="utf-8"
    )
    assert rs._manifest_declares_validation_layer(str(manifest)) is True


def test_manifest_classifier_rejects_a_different_declared_layer(tmp_path):
    manifest = tmp_path / "khronos_validation_lookalike.json"
    manifest.write_text('{"layer": {"name": "VK_LAYER_SOME_OTHER_THING"}}', encoding="utf-8")
    # The path substring ("khronos_validation") would say yes; the declared name must win.
    assert rs._manifest_declares_validation_layer(str(manifest)) is False


def test_manifest_classifier_falls_back_to_path_substring_when_unreadable():
    missing = r"C:\does\not\exist\VkLayer_khronos_validation.json"
    assert rs._manifest_declares_validation_layer(missing) is True


def test_manifest_classifier_falls_back_negative_for_an_unrelated_missing_path():
    missing = r"C:\does\not\exist\VkLayer_api_dump.json"
    assert rs._manifest_declares_validation_layer(missing) is False
