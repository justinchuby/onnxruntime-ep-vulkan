"""Pytest-driven ONNX conformance (cbourjau/onnx-tests) for the Vulkan EP.

This is the pytest-native front end to the property-based ONNX conformance suite: each
claimed op is fuzzed against the ONNX reference in its **own subprocess** (the Vulkan EP is
native code and can hard-crash on an unhandled op form — a crash must not abort the whole
run). pytest reports per-op pass/fail, so CI can simply run ``pytest tests/conformance``.

It **skips** (never errors) when prerequisites are absent, so it is safe to include in any
``pytest`` invocation:

  * ``VULKAN_EP_LIB`` — the built EP cdylib (e.g.,
    ``rust/target/release/libonnxruntime_vulkan_ep.so``).
  * ``ONNX_TESTS_DIR`` — a cbourjau/onnx-tests clone (defaults to sibling ``../../onnx-tests``).
  * ``pixi`` — the onnx-tests python env (``pixi run postinstall`` done, with onnxruntime 1.27
    installed to match ORT_API_VERSION 27).

Env knobs: ``MAX_EXAMPLES`` (20), ``SEED`` (0), ``OPS`` (space-separated subset),
``VULKAN_EP_NAME``, ``ORT_LIB_DIR``, ``PIXI``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _ep_lib() -> str:
    return (
        os.environ.get("VULKAN_EP_LIB")
        or os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
        or str(REPO_ROOT / "rust" / "target" / "release" / "libonnxruntime_vulkan_ep.so")
    )


def _onnx_tests_dir() -> str:
    return os.environ.get("ONNX_TESTS_DIR") or str(REPO_ROOT.parent / "onnx-tests")


def _pixi() -> str:
    return (
        os.environ.get("PIXI")
        or shutil.which("pixi")
        or str(Path.home() / ".pixi" / "bin" / "pixi")
    )


def _ort_lib_dir() -> str:
    return os.environ.get("ORT_LIB_DIR", "")


def _ops() -> list[str]:
    env = os.environ.get("OPS")
    if env:
        return env.split()
    ops_file = HERE / "claimed_ops.txt"
    if not ops_file.is_file():
        return []
    return [
        line.strip()
        for line in ops_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


_EP = _ep_lib()
_OTD = _onnx_tests_dir()
_PIXI = _pixi()

pytestmark = pytest.mark.skipif(
    not (Path(_EP).is_file() and Path(_OTD).is_dir() and Path(_PIXI).is_file()),
    reason=(
        "Conformance prerequisites missing — need VULKAN_EP_LIB (built EP cdylib), "
        "ONNX_TESTS_DIR (cbourjau/onnx-tests clone), and pixi. "
        "See tests/conformance/README.md."
    ),
)


@pytest.mark.parametrize("op", _ops())
def test_conformance(op: str) -> None:
    """Fuzz one op against the ONNX reference via onnx-tests, in its own process."""
    env = dict(os.environ)
    env["RUN_CANDIDATE"] = "vulkan_runtime_wrapper.run_vulkan"
    env["VULKAN_EP_LIB"] = _EP
    env["VULKAN_EP_NAME"] = os.environ.get("VULKAN_EP_NAME", "VulkanExecutionProvider")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    ort_lib_dir = _ort_lib_dir()
    if ort_lib_dir:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [ort_lib_dir, env.get("LD_LIBRARY_PATH", "")]
        ).strip(os.pathsep)

    cmd = [
        _PIXI,
        "run",
        "python",
        "-m",
        "pytest",
        "tests",
        "-k",
        f"test_{op}_",
        f"--hypothesis-max-examples={os.environ.get('MAX_EXAMPLES', '20')}",
        f"--hypothesis-seed={os.environ.get('SEED', '0')}",
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    proc = subprocess.run(
        cmd, cwd=_OTD, env=env, capture_output=True, text=True, check=False
    )
    rc = proc.returncode
    tail = (proc.stdout or "")[-3000:] + (proc.stderr or "")[-1000:]

    if rc == 5:  # pytest "no tests collected"
        pytest.skip(f"onnx-tests has no test_{op}_ cases")
    if rc >= 128 or rc in (134, 139):  # signal / SIGABRT / SIGSEGV
        pytest.fail(f"{op}: Vulkan EP CRASHED (rc={rc})\n{tail}", pytrace=False)
    assert rc == 0, f"{op}: conformance failed (rc={rc})\n{tail}"
