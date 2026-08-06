"""Windows registry-based ICD/layer suppression — the mechanism that actually takes.

WHY THIS FILE EXISTS
====================
Criterion 4 (test_criterion_4_5_witness.py) and criterion 3d (test_validation.py) each
need to take a real Vulkan component away from a real ``epctl`` child process and prove
the gate's answer changes. Both files' original mechanism was environment variables:
``VK_ICD_FILENAMES`` / ``VK_DRIVER_FILES`` / ``VK_LOADER_DRIVERS_DISABLE`` for the ICD,
``VK_LAYER_PATH`` / ``VK_LOADER_LAYERS_DISABLE`` for the layer. On the GitHub-hosted
Windows runner every one of those arms comes back "ineffective" (run 31094738484 and
reproduced again in later runs) and both files' comments record the resulting confusion:
the loader's own documentation only lists ``VK_ICD_FILENAMES`` / ``VK_DRIVER_FILES`` /
``VK_LAYER_PATH`` as "ignored when running elevated" (LoaderInterfaceArchitecture.md
§Elevated Privilege Caveats) — it does not list the newer filter variables
``VK_LOADER_DRIVERS_DISABLE`` / ``VK_LOADER_LAYERS_DISABLE`` at all, so their failure
looked like a second, unexplained defect.

THE PROVEN ROOT CAUSE (issue #1, 2026-08-06 — read the loader's own source, not its docs)
===========================================================================================
``loader/loader_environment.c`` in KhronosGroup/Vulkan-Loader (upstream, unmodified by
this project) shows ``parse_generic_filter_environment_var`` — the function that reads
BOTH ``VK_LOADER_DRIVERS_DISABLE``/``VK_LOADER_DRIVERS_SELECT`` and
``VK_LOADER_LAYERS_DISABLE``/``VK_LOADER_LAYERS_ENABLE`` — calls ``loader_secure_getenv``,
the exact same gate used for ``VK_ICD_FILENAMES``/``VK_DRIVER_FILES``/``VK_LAYER_PATH``.
On Windows that gate is::

    bool is_high_integrity() {
        ... GetTokenInformation(..., TokenIntegrityLevel, ...) ...
        return integrity_level >= SECURITY_MANDATORY_HIGH_RID;
    }

    char *loader_secure_getenv(...) {
        if (is_high_integrity()) { ...; return NULL; }
        return loader_getenv(name, inst);
    }

So *every* VK_* environment variable this project's negative controls use — filter
variables included — is silently dropped whenever the calling process token's mandatory
integrity level is High or above. That is a documentation gap, not a loader bug: the
markdown docs enumerate a narrower list than the C source actually gates. It was verified,
not assumed — this project's real Windows CI run (PR #42 CI, job 92670932473, step
"Probe Vulkan loader") reports ``loader version = 1.3.301`` via
``vkEnumerateInstanceVersion``, which is *newer* than 1.3.234/1.3.262 (the minimum loader
versions the filter variables require per LoaderDriverInterface.md) and newer than the
1.3.296 LunarG SDK CI installs — the loader on the runner is not stale, so an "old
vulkan-1.dll" theory is disproved by the same evidence that proves the real cause: this
runner's job process token is High integrity, and no amount of loader version pinning
changes what ``is_high_integrity()`` returns.

THE FIX: DO NOT ASK THE LOADER TO HONOUR AN ENV VAR IT WILL NOT READ
=====================================================================
Both the mesa lavapipe ICD (this project's own CI step) and the LunarG SDK's
``VK_LAYER_KHRONOS_validation`` layer are registered the *other* way Windows discovers
Vulkan components: the registry keys documented in LoaderDriverInterface.md's "Driver
Discovery on Windows" section —

    HKEY_LOCAL_MACHINE\\SOFTWARE\\Khronos\\Vulkan\\Drivers          (ICDs)
    HKEY_LOCAL_MACHINE\\SOFTWARE\\Khronos\\Vulkan\\ExplicitLayers   (explicit layers)

— where each value name is a manifest path and each value is a DWORD: 0 enables it, 1
disables it. This scan is **not gated by ``is_high_integrity()``** — the loader's own doc
for this key says it is used "regardless of elevation", the same property that let the
CI job register it here in the first place (`ci.yml`'s own "Register ICD in Windows
registry" step) — so flipping the DWORD to 1 and back is a suppression mechanism that
does not depend on loader version, elevation, or which environment variable the loader
happens to trust this build.

This module is that mechanism: a context manager that finds the matching registry
value(s), flips them to disabled, yields control to the caller's subprocess, and restores
the original value afterwards — verified restored, not merely reset-and-hoped.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Iterator


class RegistryMechanismUnavailable(Exception):
    """This suppression mechanism cannot be tried on this machine.

    Distinct from ``_verdict.InstrumentError``: a caller trying several suppression arms
    in sequence should treat this as "skip to the next arm", not as the final verdict.
    Only if *every* arm raises this (or otherwise fails to suppress) is the negative
    control itself an instrument outage.
    """


#: Windows registry keys the Vulkan loader scans "regardless of elevation"
#: (LoaderDriverInterface.md, "Driver Discovery on Windows"). Both live under
#: HKEY_LOCAL_MACHINE, hence the CI job's own registration step needing HKLM write
#: access — the same access this module reuses to *remove* the registration.
KEY_PATH_DRIVERS = r"SOFTWARE\Khronos\Vulkan\Drivers"
KEY_PATH_EXPLICIT_LAYERS = r"SOFTWARE\Khronos\Vulkan\ExplicitLayers"

#: The layer manifest's own declared name, read from the JSON rather than guessed from
#: the file's path, so a differently-named manifest for the same layer is still matched
#: and an unrelated layer with a similar filename is not.
VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation"


def _manifest_declares_validation_layer(manifest_path: str) -> bool:
    """True if the JSON at ``manifest_path`` declares ``VK_LAYER_KHRONOS_validation``.

    Falls back to a path substring check if the manifest cannot be read (e.g. the path
    itself is stale) — a permissive fallback is fine here because this predicate only
    ever *widens* which entries get temporarily disabled, never which ones a positive
    reading depends on, and the disabled set is always restored in ``finally``.
    """
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        name = data.get("layer", {}).get("name", "")
        if name:
            return name == VALIDATION_LAYER_NAME
    except (OSError, ValueError, AttributeError):
        pass
    return "khronos_validation" in manifest_path.lower()


@contextlib.contextmanager
def _open_key_all_access(key_path: str):
    """Open ``HKEY_LOCAL_MACHINE\\key_path`` for read/write, or raise unavailable."""
    if sys.platform != "win32":
        raise RegistryMechanismUnavailable(
            f"registry-based suppression only applies on Windows (platform={sys.platform!r})"
        )
    import winreg  # noqa: PLC0415 — Windows-only import, guarded above

    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_ALL_ACCESS | winreg.KEY_WOW64_64KEY
        )
    except FileNotFoundError as exc:
        raise RegistryMechanismUnavailable(
            f"HKLM\\{key_path} does not exist — nothing is registered there to suppress"
        ) from exc
    except PermissionError as exc:
        raise RegistryMechanismUnavailable(
            f"no write access to HKLM\\{key_path} (need admin rights on this machine); "
            "the environment-variable arms are the fallback here"
        ) from exc
    try:
        yield key
    finally:
        key.Close()


@contextlib.contextmanager
def suppress_registry_entries(
    key_path: str, *, name_filter: "callable[[str], bool] | None" = None
) -> "Iterator[list[str]]":
    """Temporarily set matching HKLM registry driver/layer values to disabled (1).

    ``name_filter`` receives each value name (a manifest path) and decides whether that
    entry should be disabled; ``None`` disables every DWORD value under the key (used for
    the ICD key, where this project expects to find only its own registered lavapipe
    entry). Restoration is unconditional (``finally``) and verified by reading each value
    back; a value that fails to restore raises loudly rather than leaving the runner's
    Vulkan registration mutated for whatever runs after this test.

    Raises ``RegistryMechanismUnavailable`` (not ``_verdict.InstrumentError``) when the
    key is absent, unwritable, or matches nothing — callers trying several suppression
    arms in sequence should catch this and move to the next arm.
    """
    # Same guard as `_open_key_all_access`, repeated here: this function's own body
    # (not just that helper's) calls `winreg.EnumValue`/`winreg.SetValueEx` directly
    # below, so it needs its own import — and that import must not even be attempted
    # on a platform where the module does not exist (issue #1 Linux-lane regression,
    # 2026-08-06: an unguarded `import winreg` here raised a bare `ModuleNotFoundError`
    # instead of the intended `RegistryMechanismUnavailable`, which no caller catches).
    if sys.platform != "win32":
        raise RegistryMechanismUnavailable(
            f"registry-based suppression only applies on Windows (platform={sys.platform!r})"
        )
    import winreg  # noqa: PLC0415 — Windows-only import, guarded immediately above

    with _open_key_all_access(key_path) as key:
        originals: dict[str, tuple[int, int]] = {}
        index = 0
        while True:
            try:
                value_name, value_data, value_type = winreg.EnumValue(key, index)
            except OSError:
                break
            index += 1
            if value_type != winreg.REG_DWORD:
                continue
            if name_filter is not None and not name_filter(value_name):
                continue
            originals[value_name] = (value_data, value_type)

        if not originals:
            raise RegistryMechanismUnavailable(
                f"no matching DWORD value under HKLM\\{key_path} — nothing registered "
                "there for this mechanism to disable (checked "
                f"{index} value(s) total)"
            )

        for name in originals:
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, 1)

        try:
            yield list(originals)
        finally:
            failures: list[str] = []
            for name, (original_value, value_type) in originals.items():
                winreg.SetValueEx(key, name, 0, value_type, original_value)
                restored, _ = winreg.QueryValueEx(key, name)
                if restored != original_value:
                    failures.append(f"{name} (wanted {original_value}, read back {restored})")
            if failures:
                raise RuntimeError(
                    "registry restoration FAILED after suppression — the runner's Vulkan "
                    f"registration under HKLM\\{key_path} is left mutated: "
                    + "; ".join(failures)
                )


def suppress_icd_registry() -> "contextlib._GeneratorContextManager[list[str]]":
    """Disable every registered ICD under ``HKLM\\SOFTWARE\\Khronos\\Vulkan\\Drivers``.

    This project's own CI step is the only thing expected to have written to this key
    (the mesa lavapipe registration in ``ci.yml``), so no name filter is applied — every
    DWORD value found there is disabled, mirroring the intent of the ineffective
    ``VK_LOADER_DRIVERS_DISABLE=*`` arm, via a mechanism the loader does not gate on
    process integrity.
    """
    return suppress_registry_entries(KEY_PATH_DRIVERS)


def suppress_validation_layer_registry() -> "contextlib._GeneratorContextManager[list[str]]":
    """Disable ``VK_LAYER_KHRONOS_validation`` under ``HKLM\\...\\ExplicitLayers``.

    Filtered by the manifest's own declared layer name (falling back to a path substring
    if the manifest cannot be read) so any other explicit layer registered on the same
    machine — the mesa layer, RenderDoc, API dump, etc. — is left alone.
    """
    return suppress_registry_entries(
        KEY_PATH_EXPLICIT_LAYERS, name_filter=_manifest_declares_validation_layer
    )
