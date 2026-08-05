"""Build the `onnxruntime-ep-vulkan` wheel, with a provenance record it can be checked by.

Run from anywhere:

    python python/build_wheel.py                 # cargo build --release, stage, wheel
    python python/build_wheel.py --no-build      # use an existing cargo artifact
    python python/build_wheel.py --lib <path>    # stage a specific artifact

How a wheel reconciles with DESIGN.md §7.8
------------------------------------------
§7.8 forbids checked-in SPIR-V, and the reason is not "binaries are bad" -- it is that a
checked-in `.spv` which no longer matches its `.comp` **is undetectable at a glance and
changes what runs**. The hazard is silent drift, not the binary.

A wheel is a binary from the consumer's point of view, so the same hazard has to be
answered, and it is answered in three parts:

1. **Nothing binary enters the tree.** `src/onnxruntime_ep_vulkan/_lib/` is gitignored.
   The wheel is a build product of this script; the repository is unchanged by running it,
   and `git status` after a build is the check on that claim. §7.8's invariant is about the
   *source tree* and it still holds exactly.
2. **The binary names its own source.** The staged artifact is accompanied by
   `_provenance.json`: the commit, whether that tree was dirty, a digest over every shader
   source at that commit, the artifact's sha256, and the toolchain. Drift stops being
   silent, because "does this wheel correspond to that source?" becomes a question with a
   procedure: rebuild at `commit` and compare `shader_source_digest`.
3. **An escape-hatch build cannot become a wheel.** §7.8 condition 4 says no release
   artifact may be produced from an `ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` build.
   Rather than trust the environment variable not to have been set in some earlier shell,
   this script counts SPIR-V magic words (`0x07230203`) in the artifact itself and refuses
   at zero. That is a measurement of the bytes being shipped, not of the intent of whoever
   built them.

What (2) does not establish: a matching `artifact_sha256` proves the shipped bytes are the
recorded bytes and nothing more. It does not prove they were compiled from the named
commit. Only a rebuild does that, and this script does not pretend otherwise -- which is
why the digest and the commit are separate fields with separate meanings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY_DIR = Path(__file__).resolve().parent
REPO = PY_DIR.parent
RUST = REPO / "rust"
BUNDLE = PY_DIR / "src" / "onnxruntime_ep_vulkan" / "_lib"

#: SPIR-V module magic number, little-endian on every platform we build for.
_SPIRV_MAGIC = bytes([0x03, 0x02, 0x23, 0x07])

_ESCAPE_HATCH = "ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC"


def _artifact_filename() -> str:
    if sys.platform == "win32":
        return "onnxruntime_vulkan_ep.dll"
    if sys.platform == "darwin":
        return "libonnxruntime_vulkan_ep.dylib"
    return "libonnxruntime_vulkan_ep.so"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _tool_version(*cmd: str) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None


def shader_source_digest() -> dict:
    """A digest over every shader source in the tree, plus the file list it covers.

    Content-addressed and path-ordered, so it is reproducible on any checkout of the same
    commit regardless of clone path or filesystem ordering. The file count ships alongside
    the digest because a digest over *zero* files is a perfectly valid-looking hex string,
    and that failure would otherwise read as a match.
    """
    roots = [RUST / "shaders"]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(p for p in root.rglob("*") if p.is_file())
    files.sort(key=lambda p: p.relative_to(REPO).as_posix())
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(REPO).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return {
        "digest": h.hexdigest() if files else None,
        "file_count": len(files),
        "roots": [str(r.relative_to(REPO).as_posix()) for r in roots],
    }


def count_spirv_modules(path: Path) -> int:
    """How many SPIR-V module headers are embedded in the artifact."""
    return path.read_bytes().count(_SPIRV_MAGIC)


def cargo_build() -> None:
    if os.environ.get(_ESCAPE_HATCH):
        raise SystemExit(
            f"refusing to build a wheel with ${_ESCAPE_HATCH} set: an escape-hatch build "
            f"carries no shaders, advertises zero devices, and must never be shipped "
            f"(DESIGN.md §7.8, condition 4)."
        )
    print("$ cargo build --release  (in rust/)")
    proc = subprocess.run(["cargo", "build", "--release"], cwd=RUST)
    if proc.returncode != 0:
        raise SystemExit(
            "cargo build failed. The Vulkan SDK (or `glslc` on PATH) is a required build "
            "prerequisite; there is no checked-in SPIR-V (DESIGN.md §7.8)."
        )


def stage(lib: Path) -> dict:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    for stale in BUNDLE.iterdir():
        stale.unlink()

    modules = count_spirv_modules(lib)
    if modules == 0:
        raise SystemExit(
            f"refusing to package {lib}: it contains zero SPIR-V modules. This is a "
            f"shader-less artifact -- it can create no compute pipeline and advertises no "
            f"device. DESIGN.md §7.8 condition 4 forbids shipping it."
        )

    dest = BUNDLE / _artifact_filename()
    shutil.copy2(lib, dest)

    record = {
        "package_version": "0.28.0",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "tree_dirty": bool(_git("status", "--porcelain")),
        "artifact_filename": dest.name,
        "artifact_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "artifact_size_bytes": dest.stat().st_size,
        "spirv_modules_embedded": modules,
        "shader_sources": shader_source_digest(),
        "source_lib": str(lib),
        "platform": sys.platform,
        "machine": os.uname().machine if hasattr(os, "uname") else os.environ.get(
            "PROCESSOR_ARCHITECTURE", "unknown"
        ),
        "rustc": _tool_version("rustc", "--version"),
        "cargo": _tool_version("cargo", "--version"),
        "glslc": _tool_version("glslc", "--version"),
        "escape_hatch_set_at_package_time": bool(os.environ.get(_ESCAPE_HATCH)),
        "note": (
            "artifact_sha256 proves the shipped bytes are the recorded bytes. It does not "
            "prove they were compiled from `commit`; that is checked by rebuilding at that "
            "commit and comparing shader_sources.digest. If tree_dirty is true, `commit` "
            "does not fully identify the source at all."
        ),
    }
    (BUNDLE / "_provenance.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def build_wheel(outdir: Path) -> Path:
    try:
        import build  # noqa: F401,PLC0415

        cmd = [sys.executable, "-m", "build", "--wheel", "--no-isolation",
               "--outdir", str(outdir)]
    except ImportError:
        cmd = [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(outdir)]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=PY_DIR)
    if proc.returncode != 0:
        raise SystemExit("wheel build failed")
    wheels = sorted(outdir.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit(f"no wheel appeared in {outdir}")
    return wheels[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-build", action="store_true",
                    help="do not run cargo; stage an artifact that already exists")
    ap.add_argument("--lib", type=Path, default=None,
                    help="path to the cdylib to package (default: rust/target/release/...)")
    ap.add_argument("--outdir", type=Path, default=PY_DIR / "dist")
    args = ap.parse_args(argv)

    if not args.no_build and args.lib is None:
        cargo_build()

    lib = args.lib or (RUST / "target" / "release" / _artifact_filename())
    lib = lib.resolve()
    if not lib.is_file():
        raise SystemExit(
            f"artifact not found: {lib}\nBuild it with `cargo build --release` in rust/, "
            f"or pass --lib."
        )

    record = stage(lib)
    print(f"staged: {BUNDLE / record['artifact_filename']}")
    print(f"  commit         : {record['commit']} (dirty={record['tree_dirty']})")
    print(f"  sha256         : {record['artifact_sha256']}")
    print(f"  spirv modules  : {record['spirv_modules_embedded']}")
    print(f"  shader sources : {record['shader_sources']['digest']} "
          f"({record['shader_sources']['file_count']} files)")

    wheel = build_wheel(args.outdir)
    print(f"\nwheel: {wheel}")
    print(f"install with: pip install {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
