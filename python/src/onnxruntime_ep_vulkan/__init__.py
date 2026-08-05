"""Load the Vulkan execution provider into a stock ONNX Runtime.

    import onnxruntime as ort
    import onnxruntime_ep_vulkan

    onnxruntime_ep_vulkan.register_execution_provider_library()
    sess = ort.InferenceSession(model, providers=onnxruntime_ep_vulkan.providers())
    onnxruntime_ep_vulkan.assert_ep_selected(sess)

Why this exists, given that it wraps one ORT call
-------------------------------------------------
``ort.register_execution_provider_library(name, path)`` is the whole API, and a wrapper
around one call is usually worse than a documented one-liner. It is not, here, and the
reason is four measured properties of that call — every one of them recorded in
``bench/results/consumption_surface_dev0.json``, six cases, each in its own subprocess
because plugin registration is process-global state:

1. **A relative path does not resolve against the caller's CWD.** ORT anchors it at its own
   ``onnxruntime/capi`` directory. The absolute path is mandatory and nothing says so.
2. **The registration name is not checked against the library.** Registering the artifact
   under ``"NotOurNameAtAll"`` succeeds and advertises its GPU devices under that name. The
   string in ``providers=[...]`` and the string passed to ``register`` must agree, and
   nothing but the caller enforces the agreement.
3. **Registering the same name twice raises.** The documented call is not safe to run twice
   in one process — a re-run notebook, or two modules that both register.
4. **The failure mode is silence.** A session that asks for an unregistered EP name does not
   raise: it warns, falls back to CPU, and returns numerically correct results. A user who
   hits ``ModuleNotFoundError`` on the import line, deletes it, and keeps the providers list
   gets a working session that never touches the GPU.

(4) is the one that decides it. This project's whole claim discipline is built on "an
unclaimed op is always correct; a wrongly-claimed one is silently wrong" — and the very
first thing a user does had the silent failure in it. :func:`assert_ep_selected` is the
answer to (4); the rest of the module is the answer to (1)-(3).

What this module deliberately does not do
-----------------------------------------
It does not create sessions, choose devices, set EP options, or wrap ``InferenceSession``.
Everything past registration is ORT's API and stays ORT's API.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

__all__ = [
    "PROVIDER_NAME",
    "ARTIFACT_FILENAME",
    "LIB_ENV_VAR",
    "EpVulkanError",
    "EpArtifactNotFound",
    "EpNotSelected",
    "ProvenanceMismatch",
    "library_path",
    "register_execution_provider_library",
    "unregister_execution_provider_library",
    "is_registered",
    "providers",
    "assert_ep_selected",
    "provenance",
    "verify_provenance",
    "describe",
]

__version__ = "0.28.0"

#: The name this package registers the EP under, and the name it puts in ``providers()``.
#: Both sides come from this one constant precisely because ORT checks neither against the
#: other (measured: ``registration_name_is_arbitrary``).
PROVIDER_NAME = "VulkanExecutionProvider"

#: Escape hatch, and the variable the repository's own pytest suite already uses. An
#: explicit path always wins over the bundled artifact, so a developer can point a
#: pip-installed package at a freshly built cdylib without reinstalling.
LIB_ENV_VAR = "ONNXRUNTIME_VULKAN_EP_LIB"

_PROVENANCE_FILENAME = "_provenance.json"
_BUNDLE_DIR = "_lib"


def _artifact_filename() -> str:
    if sys.platform == "win32":
        return "onnxruntime_vulkan_ep.dll"
    if sys.platform == "darwin":
        return "libonnxruntime_vulkan_ep.dylib"
    return "libonnxruntime_vulkan_ep.so"


#: Platform-correct cdylib filename. Pinned by ``[lib] name`` in ``rust/Cargo.toml``.
ARTIFACT_FILENAME = _artifact_filename()

_HERE = Path(__file__).resolve().parent

# name -> absolute path this process registered under that name. Used to make
# registration idempotent without pretending that a *different* library under the same
# name is the same registration.
_REGISTERED: dict[str, str] = {}


# ---------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------


class EpVulkanError(RuntimeError):
    """Base class for every error this package raises."""


class EpArtifactNotFound(EpVulkanError):
    """The cdylib could not be located. The message lists every path that was tried."""


class EpNotSelected(EpVulkanError):
    """A session did not select the Vulkan EP, and would have failed silently."""


class ProvenanceMismatch(EpVulkanError):
    """The bundled artifact does not match the digest its own provenance record names."""


# ---------------------------------------------------------------------------------------
# Locating the artifact
# ---------------------------------------------------------------------------------------


def _candidate_paths(explicit: str | os.PathLike[str] | None) -> list[tuple[str, Path]]:
    """Every place the artifact is looked for, in priority order, with a label each.

    The labels exist so that :class:`EpArtifactNotFound` can say *where it looked* rather
    than only *that it failed*. A "file not found" with no search list is the packaging
    equivalent of a benchmark with no environment record.
    """
    out: list[tuple[str, Path]] = []
    if explicit is not None:
        out.append(("explicit path argument", Path(explicit)))
    env = os.environ.get(LIB_ENV_VAR)
    if env:
        out.append((f"${LIB_ENV_VAR}", Path(env)))
    out.append(("bundled in this wheel", _HERE / _BUNDLE_DIR / ARTIFACT_FILENAME))
    # Source-checkout fallback: an editable/`pip install -e` or a plain `sys.path` use from
    # inside the repository, where the artifact lives in cargo's target directory and was
    # never staged into the package.
    for parent in (_HERE, *_HERE.parents):
        rust = parent / "rust"
        if rust.is_dir():
            for profile in ("release", "debug"):
                out.append(
                    (
                        f"source checkout (cargo {profile})",
                        rust / "target" / profile / ARTIFACT_FILENAME,
                    )
                )
            break
    return out


def library_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the absolute path to the EP cdylib, or raise :class:`EpArtifactNotFound`.

    Always absolute: ORT resolves a relative library path against its *own* ``capi``
    directory, not the caller's working directory (measured), so a relative path that
    happens to exist for the caller is still the wrong path for ORT.
    """
    tried: list[str] = []
    for label, candidate in _candidate_paths(path):
        resolved = candidate.expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:  # pragma: no cover - exotic path failures
            pass
        if resolved.is_file():
            return resolved
        tried.append(f"  - {label}: {resolved}")
    raise EpArtifactNotFound(
        f"could not find {ARTIFACT_FILENAME}. Looked in:\n"
        + "\n".join(tried)
        + "\n\nBuild it with `cargo build --release` from the repository's `rust/` "
        "directory (the Vulkan SDK, or `glslc` on PATH, is a required build "
        "prerequisite -- there is no checked-in SPIR-V, deliberately), then either "
        f"install a wheel built by `python python/build_wheel.py` or set ${LIB_ENV_VAR} "
        "to the built artifact."
    )


# ---------------------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------------------


def register_execution_provider_library(
    name: str = PROVIDER_NAME,
    path: str | os.PathLike[str] | None = None,
) -> str:
    """Register the Vulkan EP with ONNX Runtime. Returns the absolute path registered.

    Idempotent **for the same library**: calling twice in one process is a no-op the second
    time, because ORT itself raises ``library is already registered under <name>`` and a
    package that registers on import must survive being imported twice. Calling twice with
    two *different* libraries under one name is not a no-op and raises, because that is a
    real conflict and silently keeping the first one would mean the caller is running a
    binary they did not ask for.
    """
    import onnxruntime as ort

    resolved = str(library_path(path))
    previous = _REGISTERED.get(name)
    if previous is not None:
        if os.path.normcase(previous) == os.path.normcase(resolved):
            return previous
        raise EpVulkanError(
            f"{name!r} is already registered in this process from {previous!r}; "
            f"refusing to silently ignore a request to register {resolved!r} instead. "
            f"Call unregister_execution_provider_library({name!r}) first."
        )

    try:
        ort.register_execution_provider_library(name, resolved)
    except Exception as exc:  # ORT raises a bare RuntimeError subclass
        if "already registered" not in str(exc):
            raise
        # Registered by something else in this process (the repo's pytest conftest does
        # exactly this). ORT gives us no way to read back which path that was, so we
        # record ours and say what we cannot know rather than assuming they match.
        warnings.warn(
            f"{name!r} was already registered in this process by something other than "
            f"this package. ORT exposes no way to read back which library that was, so "
            f"it is not verified to be {resolved!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    _REGISTERED[name] = resolved
    return resolved


def unregister_execution_provider_library(name: str = PROVIDER_NAME) -> bool:
    """Unregister the EP. Returns True if ORT unregistered it, False if it was not there."""
    import onnxruntime as ort

    _REGISTERED.pop(name, None)
    try:
        ort.unregister_execution_provider_library(name)
    except Exception as exc:
        if "was not registered" in str(exc):
            return False
        raise
    return True


def is_registered(name: str = PROVIDER_NAME) -> bool:
    """True if ORT currently advertises *name* as an available provider."""
    import onnxruntime as ort

    return name in ort.get_available_providers()


def providers(name: str = PROVIDER_NAME, cpu_fallback: bool = True) -> list[str]:
    """The provider list to hand to ``InferenceSession``.

    Exists so the name in the providers list comes from the same constant as the name used
    at registration. ORT checks neither against the other, and a mismatch is silent.
    """
    return [name, "CPUExecutionProvider"] if cpu_fallback else [name]


def assert_ep_selected(session: Any, name: str = PROVIDER_NAME) -> None:
    """Raise :class:`EpNotSelected` unless *session* actually selected the Vulkan EP.

    ORT does **not** raise when asked for a provider it does not have. It emits a
    ``UserWarning``, falls back to CPU, and returns correct numbers — measured, case
    ``unregistered_name_failure_mode``. Warnings are routinely filtered, swallowed by
    notebooks, or lost in log noise, so the only reliable reading is the session's own
    ``get_providers()``. Call this once after constructing a session that is supposed to
    run on the GPU.

    This asserts that the EP was *selected for the session*. It does not assert that any
    node was claimed, or that any dispatch executed: ORT can select an EP that claims
    nothing. ``claimed_nodes`` is not what executes either (Mouse, 2026-08-04); the honest
    execution-side metric is dispatch count, which is out of this package's scope.
    """
    got = list(session.get_providers())
    if name in got:
        return
    raise EpNotSelected(
        f"session did not select {name!r}; it is running on {got}. "
        f"ORT does not raise for this — it warns and falls back to CPU, so a session that "
        f"never touches the GPU still returns correct-looking results.\n"
        f"Registered with ORT right now: {is_registered(name)}. "
        f"Did you call register_execution_provider_library() before constructing the "
        f"session, and pass providers={providers(name)!r}?"
    )


# ---------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------


def provenance() -> dict | None:
    """The provenance record of the bundled artifact, or None if none is bundled.

    ``docs/DESIGN.md`` §7.8 forbids checked-in SPIR-V because a binary that drifts from its
    source silently changes what runs. A wheel is a checked-in binary from the consumer's
    point of view, and gets the same treatment one level out: it carries the commit it was
    built from, whether that tree was dirty, a digest over the shader sources at that
    commit, and its own sha256 — so "does this binary still correspond to that source?" is
    a question with an answer, instead of a question nobody can ask.
    """
    record = _HERE / _BUNDLE_DIR / _PROVENANCE_FILENAME
    if not record.is_file():
        return None
    return json.loads(record.read_text(encoding="utf-8"))


def verify_provenance() -> dict:
    """Check the bundled artifact against its own provenance record.

    Returns a report dict with a ``verdict``: ``MATCH``, ``MISMATCH``, or ``UNBUNDLED``
    (no artifact is bundled, so there is nothing to verify — never ``MATCH``, because a
    check that was not performed must not read like a check that passed).
    """
    rec = provenance()
    if rec is None:
        return {
            "verdict": "UNBUNDLED",
            "reason": "no artifact is bundled in this installation; nothing to verify",
        }
    artifact = _HERE / _BUNDLE_DIR / ARTIFACT_FILENAME
    if not artifact.is_file():
        return {
            "verdict": "MISMATCH",
            "reason": f"provenance record present but {ARTIFACT_FILENAME} is not",
        }
    got = hashlib.sha256(artifact.read_bytes()).hexdigest()
    want = rec.get("artifact_sha256")
    shaders = rec.get("shader_sources") or {}
    return {
        "verdict": "MATCH" if got == want else "MISMATCH",
        "artifact_sha256": got,
        "recorded_sha256": want,
        "commit": rec.get("commit"),
        "tree_dirty": rec.get("tree_dirty"),
        "spirv_modules_embedded": rec.get("spirv_modules_embedded"),
        # Reported as digest-plus-count, never the digest alone: a sha256 over zero files
        # is a perfectly valid-looking hex string, and that failure would read as a match.
        "shader_source_digest": shaders.get("digest"),
        "shader_source_file_count": shaders.get("file_count"),
        "note": (
            "A digest match proves the shipped bytes are the bytes that were recorded. It "
            "does not prove they were built from the named commit -- that is checked by "
            "rebuilding at that commit and comparing shader_source_digest."
        ),
    }


def describe() -> dict:
    """Everything this package can say about the installation, as a dict.

    Rendered by ``python -m onnxruntime_ep_vulkan``.
    """
    info: dict[str, Any] = {
        "package_version": __version__,
        "provider_name": PROVIDER_NAME,
        "artifact_filename": ARTIFACT_FILENAME,
        "platform": sys.platform,
    }
    try:
        info["library_path"] = str(library_path())
    except EpArtifactNotFound as exc:
        info["library_path"] = None
        info["library_path_error"] = str(exc)
    info["provenance"] = verify_provenance()
    try:
        import onnxruntime as ort

        info["onnxruntime_version"] = ort.__version__
        info["registered"] = is_registered()
        info["available_providers"] = list(ort.get_available_providers())
    except ImportError:
        info["onnxruntime_version"] = None
    return info
