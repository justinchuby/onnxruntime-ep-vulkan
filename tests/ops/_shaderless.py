"""Build the shader-less artifact M0 criterion 5 is about, in-lane.

DESIGN.md §10 M0 exit criterion 5:

  > A shader-less build (``ALLOW_MISSING_GLSLC=1``) advertises zero devices and claims
  > nothing, with a ``built without shaders`` reason — it never loads, claims, and then
  > fails at pipeline creation (§7.8 condition 3).

M0 table, criterion 5 (2026-07-31): *"mechanism landed, artifact not seen ... the
``ALLOW_MISSING_GLSLC=1`` negative and its paired positive must be distinguishable from a
build that failed for an unrelated reason — the reason string is the mechanism, and R13
says it must be emitted on the failing path, not only on the passing one."*

The artifact is a second binary.  It cannot be produced by an environment variable at run
time, so it is produced here, from the same source tree, into a scratch directory outside
the repository.  A control that has to be built by hand before the lane can see it is a
control that must be opted into, and this project has already ruled that such a control is
not in the lane.

STATED LIMIT — THE ``ALLOW_MISSING_GLSLC`` ROUTE IS UNREACHABLE ON ANY BOX WITH THE SDK
========================================================================================
``build.rs::find_glslc()`` looks in three places, in order:

  1. ``$VULKAN_SDK/bin/glslc``
  2. ``glslc`` on ``$PATH`` (probed by running it)
  3. ``installed_sdk_glslc()`` — an **unconditional scan of ``C:\\VulkanSDK``**
     (build.rs:175-198), which honours no environment variable at all.

``ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1`` is only consulted when all three fail
(build.rs:247).  On a Windows developer box or CI runner with the Vulkan SDK installed —
which is every machine that can run the rest of this suite — step 3 always succeeds, so
**the criterion-5 negative control as DESIGN.md words it cannot be executed there.**  That
is a finding about the build, not about the EP, and it is Tank's file: ``ALLOW_MISSING_GLSLC=1``
should be checked *before* the search, as a hard override, so the shader-less lane can be
entered deliberately rather than only by accident of a missing toolchain.

What this module does instead, and why it is the same artifact
--------------------------------------------------------------
It builds from a copy of ``rust/`` whose shader source set is empty.  ``build.rs`` has an
early return for that case (build.rs:239-243) which executes ``write_shader_modules(out_dir,
&[])`` and returns — the *same statement* the ``ALLOW_MISSING_GLSLC`` arm executes
(build.rs:255-256).  Both produce an artifact whose ``shaders::SHADER_MODULES`` is empty.

That equivalence is an argument, so it is **not** what is relied on.  §7.8 condition 3's
guard reads ``shaders::has_any()`` (engine.rs:679, ep.rs:508) and nothing else; its subject
is emptiness, not the route to emptiness.  The witness asserts the artifact's own emitted
reason string — ``built without shaders (ALLOW_MISSING_GLSLC build)`` — which the guard
prints and which the shader-full binary never prints.  The observation is the artifact's,
not this docstring's.

R13
---
Every failure to *produce* the artifact raises ``InstrumentError``.  A shader-less binary
that could not be built has told us nothing about criterion 5, and a lane that scores that
as a criterion-5 failure has fabricated a detection.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import _verdict

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
RUST = REPO / "rust"
#: A compile input of the crate that does not live under ``rust/`` — see ``_sync_sources``.
_LEDGER = REPO / "evidence" / "proof_ledger.jsonl"

#: Outside the repository on purpose: a second `target/` inside it would be picked up by
#: every glob in the tree and would double the size of a `git status`.
SCRATCH = REPO.parent / f"{REPO.name}-shaderless"

_DLL_NAME = {
    "win32": "onnxruntime_vulkan_ep.dll",
    "darwin": "libonnxruntime_vulkan_ep.dylib",
}.get(sys.platform, "libonnxruntime_vulkan_ep.so")

_EPCTL_NAME = "epctl.exe" if sys.platform == "win32" else "epctl"

#: The reason string §7.8 condition 3 requires on the failing path.  It is emitted by
#: `engine::probe_devices` and by `ep::get_capability_impl`, and it is the discriminator:
#: a shader-full binary never prints it, and a build that failed for an unrelated reason
#: never gets far enough to print it either.
SHADERLESS_REASON = "built without shaders"


def artifact_paths() -> "tuple[Path, Path]":
    """(library, epctl) inside the scratch build.  Neither is guaranteed to exist."""
    rel = SCRATCH / "rust" / "target" / "release"
    return rel / _DLL_NAME, rel / _EPCTL_NAME


def _newest_source_mtime() -> float:
    newest = 0.0
    for root in (RUST / "src", RUST / "shaders"):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                newest = max(newest, path.stat().st_mtime)
    for path in (RUST / "Cargo.toml", RUST / "build.rs", RUST / "Cargo.lock"):
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
    # The crate `include_str!`s the proof ledger from outside `rust/`. It is a compile input
    # like any other, so a ledger newer than the scratch tree must force a re-sync — otherwise
    # this witness builds the previous ledger and reports on a binary nobody shipped.
    if _LEDGER.is_file():
        newest = max(newest, _LEDGER.stat().st_mtime)
    return newest


def _sync_sources() -> Path:
    """Copy ``rust/`` (minus ``target/``) into the scratch tree and empty its shader set."""
    dst = SCRATCH / "rust"
    dst.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("target", "*.pyc", "__pycache__")
    for entry in RUST.iterdir():
        if entry.name == "target":
            continue
        target = dst / entry.name
        if entry.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(entry, target, ignore=ignore)
        else:
            shutil.copy2(entry, target)

    # §8.9 bakes the proof ledger with `include_str!("../../evidence/proof_ledger.jsonl")`,
    # which resolves *outside* the crate directory. Copying only `rust/` therefore produced
    #     error: couldn't read `src\../../evidence/proof_ledger.jsonl`
    #     error: could not compile `onnxruntime-ep-vulkan` (lib) due to 1 previous error
    # and the witness reported ERROR(instrument) — correctly, since a build that failed for an
    # unrelated reason is not a criterion-5 result. The scratch tree has to carry every compile
    # input, not just the ones under `rust/`.
    if _LEDGER.is_file():
        led_dst = SCRATCH / _LEDGER.relative_to(REPO)
        led_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_LEDGER, led_dst)

    glsl = dst / "shaders" / "glsl"
    if glsl.is_dir():
        shutil.rmtree(glsl)
    glsl.mkdir(parents=True, exist_ok=True)

    variants = dst / "src" / "ops" / "shader_variants.txt"
    if variants.parent.is_dir():
        variants.write_text("", encoding="utf-8")

    return dst


def build(*, force: bool = False) -> "tuple[Path, Path]":
    """Return (library, epctl) for a shader-less build, building it if it is stale.

    Raises ``_verdict.InstrumentError`` — never ``AssertionError`` — on every failure to
    produce the artifact.
    """
    lib, epctl = artifact_paths()
    if not force and lib.is_file() and epctl.is_file():
        built = min(lib.stat().st_mtime, epctl.stat().st_mtime)
        if built >= _newest_source_mtime():
            return lib, epctl

    try:
        manifest_dir = _sync_sources()
    except OSError as exc:
        raise _verdict.InstrumentError(
            "[criterion 5 instrument failure] ERROR(instrument): could not stage the "
            f"shader-less source tree at {SCRATCH}: {exc}.  Nothing was built, so nothing "
            "has been observed about criterion 5."
        ) from exc

    env = dict(os.environ)
    # `build.rs` resolves ORT headers relative to its own manifest dir, which is now the
    # scratch copy and has no `third_party/`.  Point it back at the repository's vendored
    # copy rather than duplicating 30 MB of headers.
    env["ORT_INCLUDE_DIR"] = str(REPO / "third_party" / "onnxruntime" / "include")
    # Set even though the empty-source-set early return fires first: when Tank makes this
    # variable a hard override, this build takes the route DESIGN.md names, unchanged.
    env["ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC"] = "1"
    # PIN the output directory to the one `artifact_paths()` reads.
    #
    # Found on the Linux lane, where `CARGO_TARGET_DIR` is exported (it must be: left unset it
    # lands on `/mnt/c/.../rust/target` and races the Windows build). `dict(os.environ)` carried
    # it into this build, so the SHADER-LESS artifact was written into the SHARED target
    # directory — over the real one. Two failures from one line, and the loud one was the wrong
    # one: this function raised `cargo exited 0 but <scratch>/rust/target/release/... does not
    # exist`, which reads as a build problem, while the silent damage was that every later step
    # in the same run then loaded an EP that advertises zero devices. The gate went UNATTRIBUTED,
    # the EP wrote no counters file, and `epctl --check-counters` fell back to reading a snapshot
    # left by an earlier session and reported its AGE as `counters ABI 4 vs 8`.
    #
    # A witness that must produce a deliberately broken binary has to own the directory it writes
    # it to. Inheriting one is how the broken binary escapes into the run it was measuring.
    env["CARGO_TARGET_DIR"] = str((SCRATCH / "rust" / "target").resolve())

    result = _verdict.run_subprocess_checked(
        [
            "cargo", "build", "--release",
            "--manifest-path", str(manifest_dir / "Cargo.toml"),
        ],
        what="criterion 5 shader-less build",
        quiet_seconds=240,
        env=env,
        cwd=str(manifest_dir),
    )
    if result.returncode != 0:
        raise _verdict.InstrumentError(
            "[criterion 5 instrument failure] ERROR(instrument): the shader-less build "
            f"did not produce an artifact (cargo exited {result.returncode}).  A build "
            "that failed for an unrelated reason must be distinguishable from a build "
            "that correctly claims nothing — that is criterion 5's own wording — so this "
            "is an outage and NOT a criterion-5 failure.  Quote this text:\n"
            f"{(result.stderr or '')[-3000:]}"
        )

    if not lib.is_file() or not epctl.is_file():
        raise _verdict.InstrumentError(
            "[criterion 5 instrument failure] ERROR(instrument): cargo exited 0 but "
            f"{lib if not lib.is_file() else epctl} does not exist."
        )
    return lib, epctl
