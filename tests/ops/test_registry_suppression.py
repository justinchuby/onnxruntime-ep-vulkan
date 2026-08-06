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


# ---------------------------------------------------------------------------
# Real winreg round trips (Morpheus's Blocker 1 on PR #45): the classifier/platform-guard
# tests above never once call `winreg.SetValueEx`/`winreg.QueryValueEx` for real, so they
# cannot falsify a bug in the actual enumerate -> disable -> read-back -> restore ->
# verify cycle itself. Everything below runs the real Win32 registry API against a
# disposable HKEY_CURRENT_USER scratch key — writable without elevation, unlike the real
# HKLM keys probed above — via `suppress_registry_entries(..., hive=winreg.HKEY_CURRENT_USER)`.
# Only the *hive* differs from production; the enumerate/disable/restore code path
# exercised is identical to what `suppress_icd_registry`/`suppress_validation_layer_registry`
# run against HKLM.
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import winreg

_SCRATCH_KEY_PATH = r"SOFTWARE\onnxruntime-ep-vulkan-registry-suppression-test-scratch"


@pytest.fixture
def hkcu_scratch_key():
    """A fresh HKCU scratch key for real winreg round-trip tests, writable without
    elevation. Fails loudly (rather than silently reusing/overwriting) if a previous
    crashed run already left one behind, matching this module's own "verify restoration,
    don't assume it" convention; always removes every value plus the key itself in
    teardown, regardless of what the test did to it.

    Yields a small helper object exposing `set_value`/`read_value`/`delete_value`, each
    implemented directly against `winreg` (not via `_registry_suppression`), so the test
    module under test is verified from a genuinely independent vantage point.
    """
    if sys.platform != "win32":
        pytest.skip("HKCU round-trip tests only apply on Windows")

    try:
        stale = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _SCRATCH_KEY_PATH)
        stale.Close()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            f"stale scratch key HKCU\\{_SCRATCH_KEY_PATH} already exists from a previous "
            "crashed test run — inspect and remove it manually with regedit before "
            "re-running (refusing to silently reuse or overwrite it)"
        )

    key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _SCRATCH_KEY_PATH, 0, winreg.KEY_ALL_ACCESS)

    class _Scratch:
        path = _SCRATCH_KEY_PATH
        hive = winreg.HKEY_CURRENT_USER

        def set_value(self, name: str, value: int, value_type: int = winreg.REG_DWORD) -> None:
            winreg.SetValueEx(key, name, 0, value_type, value)

        def read_value(self, name: str):
            return winreg.QueryValueEx(key, name)

        def delete_value(self, name: str) -> None:
            winreg.DeleteValue(key, name)

        def value_names(self) -> "list[str]":
            names = []
            index = 0
            while True:
                try:
                    names.append(winreg.EnumValue(key, index)[0])
                except OSError:
                    return names
                index += 1

    try:
        yield _Scratch()
    finally:
        # Unconditional cleanup: delete every value this test (or a helper) left behind,
        # then the key itself, regardless of whether the test body raised.
        index = 0
        names = []
        while True:
            try:
                names.append(winreg.EnumValue(key, index)[0])
            except OSError:
                break
            index += 1
        for name in names:
            try:
                winreg.DeleteValue(key, name)
            except OSError:
                pass
        key.Close()
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _SCRATCH_KEY_PATH)


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_real_round_trip_disables_and_restores_a_pre_existing_enabled_value(hkcu_scratch_key):
    """The core contract: a value that starts at 0 (enabled) reads back as 1 (disabled)
    while suppressed, and is restored to exactly 0 afterward — verified via the
    independent `hkcu_scratch_key` helper, not the module's own internal bookkeeping."""
    hkcu_scratch_key.set_value("C:\\fake\\driver_a.json", 0)
    with rs.suppress_registry_entries(
        hkcu_scratch_key.path, hive=hkcu_scratch_key.hive
    ) as disabled:
        assert "C:\\fake\\driver_a.json" in disabled
        assert hkcu_scratch_key.read_value("C:\\fake\\driver_a.json") == (1, winreg.REG_DWORD)
    assert hkcu_scratch_key.read_value("C:\\fake\\driver_a.json") == (0, winreg.REG_DWORD)


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_real_round_trip_restores_a_pre_existing_disabled_value_to_disabled(hkcu_scratch_key):
    """A value that was already disabled (1) before suppression must come back as 1, not
    0 — restoration must save and replay the *original* value, not assume "enabled"."""
    hkcu_scratch_key.set_value("C:\\fake\\driver_b.json", 1)
    with rs.suppress_registry_entries(hkcu_scratch_key.path, hive=hkcu_scratch_key.hive):
        assert hkcu_scratch_key.read_value("C:\\fake\\driver_b.json") == (1, winreg.REG_DWORD)
    assert hkcu_scratch_key.read_value("C:\\fake\\driver_b.json") == (1, winreg.REG_DWORD)


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_real_absent_value_is_mechanism_unavailable(hkcu_scratch_key):
    """An existing, writable key with zero matching DWORD values must raise
    `RegistryMechanismUnavailable` ("nothing registered there"), not silently no-op and
    not crash — this is the real-registry counterpart of the missing-*key* test above,
    for the missing-*value* case."""
    with pytest.raises(rs.RegistryMechanismUnavailable, match="no matching DWORD value"):
        with rs.suppress_registry_entries(hkcu_scratch_key.path, hive=hkcu_scratch_key.hive):
            pass


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_real_non_dword_value_is_skipped_not_disabled(hkcu_scratch_key):
    """A REG_SZ value under the same key (e.g. some unrelated tool's own bookkeeping)
    must be left completely alone — only REG_DWORD values are ever candidates."""
    hkcu_scratch_key.set_value("some_string_value", "leave me alone", winreg.REG_SZ)
    hkcu_scratch_key.set_value("C:\\fake\\driver_c.json", 0)
    with rs.suppress_registry_entries(hkcu_scratch_key.path, hive=hkcu_scratch_key.hive) as disabled:
        assert "some_string_value" not in disabled
        assert hkcu_scratch_key.read_value("some_string_value") == (
            "leave me alone",
            winreg.REG_SZ,
        )
    assert hkcu_scratch_key.read_value("some_string_value") == ("leave me alone", winreg.REG_SZ)


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_real_round_trip_handles_unicode_and_space_value_names(hkcu_scratch_key):
    """Manifest paths are real filesystem paths and can contain spaces (``Program
    Files``) or non-ASCII characters (a user profile directory, an installer's localized
    path); the value *name* — which is the manifest path — must round-trip byte-for-byte
    through disable/restore for both."""
    spaced = r"C:\Program Files\Some Vendor\driver with spaces.json"
    unicode_name = "C:\\Users\\jos\u00e9\\AppData\\vk_ICD_\u00e9\u00e9.json"
    hkcu_scratch_key.set_value(spaced, 0)
    hkcu_scratch_key.set_value(unicode_name, 0)
    with rs.suppress_registry_entries(hkcu_scratch_key.path, hive=hkcu_scratch_key.hive) as disabled:
        assert spaced in disabled
        assert unicode_name in disabled
        assert hkcu_scratch_key.read_value(spaced) == (1, winreg.REG_DWORD)
        assert hkcu_scratch_key.read_value(unicode_name) == (1, winreg.REG_DWORD)
    assert hkcu_scratch_key.read_value(spaced) == (0, winreg.REG_DWORD)
    assert hkcu_scratch_key.read_value(unicode_name) == (0, winreg.REG_DWORD)


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_real_round_trip_restores_even_when_the_body_raises(hkcu_scratch_key):
    """Restoration must happen in `finally`: if the caller's own body blows up mid-way
    (a real subprocess failure, say), the disabled value must still come back exactly as
    it was — this is what makes the mechanism safe to use inside a witness test that
    itself might fail."""
    hkcu_scratch_key.set_value("C:\\fake\\driver_d.json", 0)

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with rs.suppress_registry_entries(hkcu_scratch_key.path, hive=hkcu_scratch_key.hive):
            assert hkcu_scratch_key.read_value("C:\\fake\\driver_d.json") == (1, winreg.REG_DWORD)
            raise _Boom("simulated failure inside the suppressed region")
    assert hkcu_scratch_key.read_value("C:\\fake\\driver_d.json") == (0, winreg.REG_DWORD)


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_real_restore_verification_failure_raises_runtime_error(hkcu_scratch_key, monkeypatch):
    """The one seam that cannot be forced with a genuine OS-level failure on a
    non-elevated HKCU key (this process always has full rights to its own HKCU key, so
    there is no permission-style way to make a real restore write fail): the disable and
    restore *writes* below are both real `winreg.SetValueEx` calls that actually execute
    and succeed. Only the post-restore *verification read* is monkeypatched, and only for
    this one value name — `winreg.QueryValueEx` is wrapped so it lies (reports the
    disabled value 1) exactly when asked about this test's own value, while every other
    call (including this fixture's own teardown reads) passes through to the real
    function untouched. This is intentionally the sole non-end-to-end-real seam among
    these round-trip tests, and exists only to prove the module's own `RuntimeError`
    guard actually fires when a real restore write is contradicted by a real read."""
    target_name = "C:\\fake\\driver_lying_restore.json"
    hkcu_scratch_key.set_value(target_name, 0)

    real_query_value_ex = winreg.QueryValueEx

    def _lying_query_value_ex(key, name):
        if name == target_name:
            return (1, winreg.REG_DWORD)  # lie: claim still-disabled after a real restore write
        return real_query_value_ex(key, name)

    monkeypatch.setattr(winreg, "QueryValueEx", _lying_query_value_ex)

    with pytest.raises(RuntimeError, match="registry restoration FAILED"):
        with rs.suppress_registry_entries(hkcu_scratch_key.path, hive=hkcu_scratch_key.hive):
            pass

    # Tear down the monkeypatch before the fixture's own teardown reads run, and confirm
    # via the real API that the actual SetValueEx restore write underneath the lie did
    # genuinely put the value back to 0 — the module's RuntimeError was correctly a false
    # alarm manufactured by this test, not evidence of a real restore bug.
    monkeypatch.undo()
    assert hkcu_scratch_key.read_value(target_name) == (0, winreg.REG_DWORD)


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Windows registry API")
def test_real_round_trip_leaves_unrelated_filtered_out_values_untouched(hkcu_scratch_key):
    """`name_filter` must exclude a value from being disabled at all, not just from being
    reported — a real registered layer manifest sitting alongside the target one under
    the same key must be completely unaffected, proving `suppress_validation_layer_registry`'s
    filtering genuinely isolates its target from siblings under the same real key."""
    hkcu_scratch_key.set_value("C:\\fake\\target_layer.json", 0)
    hkcu_scratch_key.set_value("C:\\fake\\unrelated_layer.json", 0)
    with rs.suppress_registry_entries(
        hkcu_scratch_key.path,
        name_filter=lambda name: "target_layer" in name,
        hive=hkcu_scratch_key.hive,
    ) as disabled:
        assert disabled == ["C:\\fake\\target_layer.json"]
        assert hkcu_scratch_key.read_value("C:\\fake\\target_layer.json") == (1, winreg.REG_DWORD)
        # The unrelated value must never have been touched, even transiently.
        assert hkcu_scratch_key.read_value("C:\\fake\\unrelated_layer.json") == (0, winreg.REG_DWORD)
    assert hkcu_scratch_key.read_value("C:\\fake\\target_layer.json") == (0, winreg.REG_DWORD)
    assert hkcu_scratch_key.read_value("C:\\fake\\unrelated_layer.json") == (0, winreg.REG_DWORD)


# ---------------------------------------------------------------------------
# `is_high_integrity()`: the High-integrity detection Morpheus's Blocker 2 depends on.
# ---------------------------------------------------------------------------


def test_is_high_integrity_is_false_on_a_simulated_non_windows_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert rs.is_high_integrity() is False


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real Win32 token API")
def test_is_high_integrity_runs_the_real_win32_call_path_without_raising():
    """A smoke test against the real Win32 API on this dev box: it must return a plain
    `bool` without raising. This dev box is known (module docstring, prior turns'
    established constraints) to run at Medium integrity — a non-elevated, UAC-split
    admin token — so the real answer here must be `False`; if this ever starts
    returning `True` on an unelevated shell, that is itself evidence something about this
    box's security context changed and worth investigating, not evidence of a bug in the
    function."""
    result = rs.is_high_integrity()
    assert isinstance(result, bool)
    assert result is False
