"""Tests for `python/build_wheel.py` — the piece that reconciles a wheel with DESIGN.md §7.8.

§7.8 forbids checked-in SPIR-V because a binary that drifts from its source silently
changes what runs. A wheel is a binary from the consumer's point of view, so the same
hazard is answered three ways, and each answer is tested here:

1. nothing binary enters the tree (the staged directory is gitignored — checked below);
2. the binary names its own source (shader-source digest — checked below);
3. an escape-hatch build cannot become a wheel (SPIR-V module count — checked below,
   in **both** states, because a detector never seen firing has no demonstrated positive
   state).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD_WHEEL = REPO / "python" / "build_wheel.py"

_spec = importlib.util.spec_from_file_location("_build_wheel", BUILD_WHEEL)
assert _spec and _spec.loader
bw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bw)

LIB = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
needs_lib = pytest.mark.skipif(
    not (LIB and Path(LIB).is_file()),
    reason="ONNXRUNTIME_VULKAN_EP_LIB is unset or does not point at a built artifact",
)


# ---------------------------------------------------------------------------------------
# The SPIR-V detector, in both states
# ---------------------------------------------------------------------------------------


def test_spirv_detector_negative_control(tmp_path):
    """A file with no SPIR-V in it counts zero — including a large one full of near misses."""
    p = tmp_path / "no_shaders.bin"
    # 0x07230203 with one byte wrong, repeatedly: the detector must not fire on these.
    p.write_bytes(bytes([0x03, 0x02, 0x23, 0x08]) * 4096 + b"\x00" * 65536)
    assert bw.count_spirv_modules(p) == 0


def test_spirv_detector_positive_control(tmp_path):
    """A file with N planted module headers counts exactly N."""
    p = tmp_path / "planted.bin"
    p.write_bytes(b"\xff" * 100 + bytes([0x03, 0x02, 0x23, 0x07]) * 7 + b"\xff" * 100)
    assert bw.count_spirv_modules(p) == 7


@needs_lib
def test_shipping_artifact_carries_shaders():
    """The real artifact is in the detector's positive state, not merely assumed to be."""
    assert bw.count_spirv_modules(Path(LIB)) > 0


def test_stage_refuses_a_shaderless_artifact(tmp_path, monkeypatch):
    """§7.8 condition 4, enforced against the bytes rather than against an env var.

    The rule is "no release artifact from an escape-hatch build". Checking
    ``$ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC`` at packaging time would only check the
    intent of whoever ran the packaging step, not the artifact they handed it — the
    variable may have been set in the shell that ran cargo hours earlier, or unset in the
    one that runs the wheel build.
    """
    fake = tmp_path / bw._artifact_filename()
    fake.write_bytes(b"no spirv here at all" * 100)
    monkeypatch.setattr(bw, "BUNDLE", tmp_path / "bundle")
    with pytest.raises(SystemExit) as exc:
        bw.stage(fake)
    assert "zero SPIR-V modules" in str(exc.value)
    assert "§7.8" in str(exc.value)


def test_cargo_build_refuses_under_the_escape_hatch(monkeypatch):
    """Belt as well as braces: the env var is refused too, before a build even starts."""
    monkeypatch.setenv("ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC", "1")
    with pytest.raises(SystemExit) as exc:
        bw.cargo_build()
    assert "escape-hatch" in str(exc.value)


# ---------------------------------------------------------------------------------------
# The shader-source digest
# ---------------------------------------------------------------------------------------


def test_shader_source_digest_covers_real_files():
    """The digest must be reported with the file count it covers.

    A sha256 over zero files is a perfectly valid-looking hex string. Publishing the digest
    alone would let "the shader directory moved and nothing was hashed" read as a match.
    """
    got = bw.shader_source_digest()
    assert got["file_count"] > 0, (
        "no shader sources found under rust/shaders — either the tree moved or the digest "
        "is covering nothing, and both must fail loudly"
    )
    assert got["digest"] is not None
    assert len(got["digest"]) == 64


def test_shader_source_digest_is_content_addressed(tmp_path, monkeypatch):
    """Same contents, different clone path — same digest. Otherwise it cannot be re-checked."""
    a = tmp_path / "a"
    (a / "rust" / "shaders" / "glsl").mkdir(parents=True)
    (a / "rust" / "shaders" / "glsl" / "x.comp").write_text("void main() {}\n")
    b = tmp_path / "b"
    (b / "rust" / "shaders" / "glsl").mkdir(parents=True)
    (b / "rust" / "shaders" / "glsl" / "x.comp").write_text("void main() {}\n")

    monkeypatch.setattr(bw, "REPO", a)
    monkeypatch.setattr(bw, "RUST", a / "rust")
    first = bw.shader_source_digest()
    monkeypatch.setattr(bw, "REPO", b)
    monkeypatch.setattr(bw, "RUST", b / "rust")
    second = bw.shader_source_digest()
    assert first["digest"] == second["digest"]
    assert first["file_count"] == second["file_count"] == 1


def test_shader_source_digest_changes_with_content(tmp_path, monkeypatch):
    root = tmp_path / "r"
    (root / "rust" / "shaders").mkdir(parents=True)
    f = root / "rust" / "shaders" / "x.comp"
    f.write_text("void main() {}\n")
    monkeypatch.setattr(bw, "REPO", root)
    monkeypatch.setattr(bw, "RUST", root / "rust")
    before = bw.shader_source_digest()["digest"]
    f.write_text("void main() { int y = 1; }\n")
    assert bw.shader_source_digest()["digest"] != before


# ---------------------------------------------------------------------------------------
# Nothing binary enters the tree
# ---------------------------------------------------------------------------------------


def test_staged_artifact_directory_is_ignored_by_git():
    """The invariant §7.8 is actually about: the *source tree* carries no binary.

    `git check-ignore` is the authority here, not a reading of .gitignore by eye.
    """
    rel = "python/src/onnxruntime_ep_vulkan/_lib/onnxruntime_vulkan_ep.dll"
    proc = subprocess.run(
        ["git", "check-ignore", "-q", rel], cwd=REPO, capture_output=True
    )
    assert proc.returncode == 0, f"{rel} is NOT gitignored — a wheel build would dirty the tree"


def test_no_binary_artifact_is_tracked_in_the_python_package():
    proc = subprocess.run(
        ["git", "ls-files", "python/"], cwd=REPO, capture_output=True, text=True, check=True
    )
    tracked = [f for f in proc.stdout.split() if f]
    offenders = [
        f for f in tracked
        if f.endswith((".dll", ".so", ".dylib", ".spv", ".pyd", ".lib", ".a"))
    ]
    assert offenders == [], f"binary artifacts tracked under python/: {offenders}"


# ---------------------------------------------------------------------------------------
# Wheel tagging
# ---------------------------------------------------------------------------------------


def test_platform_filename_matches_the_shim():
    """build_wheel and the shim must agree on the artifact name, or the wheel is unloadable."""
    sys.path.insert(0, str(REPO / "python" / "src"))
    import onnxruntime_ep_vulkan as vk

    assert bw._artifact_filename() == vk.ARTIFACT_FILENAME
