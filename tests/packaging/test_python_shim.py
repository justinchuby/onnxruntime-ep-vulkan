"""Tests for the `onnxruntime_ep_vulkan` consumer shim.

Split by what they need:

* Most tests need only the package on ``sys.path`` and never touch ORT registration.
* Tests marked ``needs_lib`` need the built cdylib (``$ONNXRUNTIME_VULKAN_EP_LIB``).
* Tests that *register* run in a subprocess, because plugin-EP registration is
  process-global and irreversible enough that doing it in the pytest interpreter would
  change the result of every later test in the session — including tests in other files.

The point of the shim is that ORT's one documented call has four sharp edges
(``bench/results/consumption_surface_dev0.json``). Each of those edges gets a test here,
and each test names which measured case it corresponds to.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG_SRC = REPO / "python" / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

import onnxruntime_ep_vulkan as vk  # noqa: E402

LIB = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
needs_lib = pytest.mark.skipif(
    not (LIB and Path(LIB).is_file()),
    reason="ONNXRUNTIME_VULKAN_EP_LIB is unset or does not point at a built artifact",
)


def _in_subprocess(body: str, env: dict[str, str] | None = None) -> dict:
    """Run *body* in a fresh interpreter; it must print one ``@@<json>`` line."""
    full = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(PKG_SRC)!r})\n"
        "def emit(**kw): print('@@' + json.dumps(kw))\n" + body
    )
    e = dict(os.environ)
    if env:
        e.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", full], capture_output=True, text=True, env=e, cwd=str(REPO)
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise AssertionError(
        f"subprocess emitted no reading (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------------------
# Names and constants
# ---------------------------------------------------------------------------------------


def test_provider_name_is_single_sourced():
    """The registration name and the providers-list name come from one constant.

    Corresponds to measured case ``registration_name_is_arbitrary``: ORT never checks the
    two against each other, so the only defence is that they cannot differ.
    """
    assert vk.providers()[0] == vk.PROVIDER_NAME
    assert vk.providers(cpu_fallback=False) == [vk.PROVIDER_NAME]
    assert vk.providers()[-1] == "CPUExecutionProvider"


def test_artifact_filename_matches_platform():
    expected = {
        "win32": "onnxruntime_vulkan_ep.dll",
        "darwin": "libonnxruntime_vulkan_ep.dylib",
    }.get(sys.platform, "libonnxruntime_vulkan_ep.so")
    assert vk.ARTIFACT_FILENAME == expected


# ---------------------------------------------------------------------------------------
# Locating the artifact
# ---------------------------------------------------------------------------------------


def test_library_path_is_absolute(tmp_path, monkeypatch):
    """A relative path in, an absolute path out.

    Corresponds to measured case ``relative_path_anchor``: ORT resolves a relative library
    path against its own ``capi`` directory, so handing ORT a relative path that happens to
    exist for the caller loads nothing, or worse, loads something else.
    """
    fake = tmp_path / vk.ARTIFACT_FILENAME
    fake.write_bytes(b"not a real library")
    monkeypatch.chdir(tmp_path)
    got = vk.library_path(vk.ARTIFACT_FILENAME)
    assert got.is_absolute()
    assert got == fake.resolve()


def test_library_path_prefers_explicit_over_env(tmp_path, monkeypatch):
    a = tmp_path / "a" / vk.ARTIFACT_FILENAME
    b = tmp_path / "b" / vk.ARTIFACT_FILENAME
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"x")
    monkeypatch.setenv(vk.LIB_ENV_VAR, str(b))
    assert vk.library_path(a) == a.resolve()
    assert vk.library_path() == b.resolve()


def test_missing_artifact_names_every_path_it_tried(tmp_path, monkeypatch):
    """A 'not found' with no search list is the packaging equivalent of an unlabelled number."""
    # Relocate the package's notion of where it lives, so neither a bundled artifact (this
    # tree has one after a wheel build) nor the source-checkout fallback can satisfy it.
    monkeypatch.setattr(vk, "_HERE", tmp_path / "pkg")
    monkeypatch.setenv(vk.LIB_ENV_VAR, str(tmp_path / "nowhere" / vk.ARTIFACT_FILENAME))
    with pytest.raises(vk.EpArtifactNotFound) as exc:
        vk.library_path(tmp_path / "also_nowhere" / vk.ARTIFACT_FILENAME)
    msg = str(exc.value)
    assert "explicit path argument" in msg
    assert vk.LIB_ENV_VAR in msg
    assert "bundled in this wheel" in msg
    assert "also_nowhere" in msg
    # It must also say what to do, not only what failed.
    assert "cargo build --release" in msg


def test_env_var_is_honoured_for_a_path_that_exists(tmp_path, monkeypatch):
    p = tmp_path / vk.ARTIFACT_FILENAME
    p.write_bytes(b"x")
    monkeypatch.setenv(vk.LIB_ENV_VAR, str(p))
    assert vk.library_path() == p.resolve()


# ---------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------


def test_unbundled_provenance_is_not_a_pass():
    """No artifact bundled means UNBUNDLED, never MATCH.

    A check that was not performed must not read like a check that passed. In a source
    checkout there is no ``_lib/`` directory, so this is the state the test suite sees.
    """
    report = vk.verify_provenance()
    assert report["verdict"] in {"UNBUNDLED", "MATCH", "MISMATCH"}
    if vk.provenance() is None:
        assert report["verdict"] == "UNBUNDLED"


def test_provenance_mismatch_is_detected(tmp_path, monkeypatch):
    """Positive control for the detector: plant a wrong digest and require MISMATCH.

    Without this the UNBUNDLED path above is the only state the detector is ever seen in,
    and a detector never seen firing has no demonstrated positive state.
    """
    bundle = tmp_path / "_lib"
    bundle.mkdir()
    (bundle / vk.ARTIFACT_FILENAME).write_bytes(b"the shipped bytes")
    (bundle / "_provenance.json").write_text(
        json.dumps(
            {
                "artifact_sha256": "0" * 64,
                "commit": "deadbeef",
                "shader_sources": {"digest": "abc", "file_count": 17},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(vk, "_HERE", tmp_path)
    report = vk.verify_provenance()
    assert report["verdict"] == "MISMATCH"
    assert report["recorded_sha256"] == "0" * 64
    assert report["artifact_sha256"] != "0" * 64
    assert report["shader_source_file_count"] == 17


def test_provenance_reports_file_count_beside_digest(tmp_path, monkeypatch):
    """A digest over zero files is a valid-looking hex string; the count is what disproves it."""
    import hashlib

    bundle = tmp_path / "_lib"
    bundle.mkdir()
    payload = b"bytes"
    (bundle / vk.ARTIFACT_FILENAME).write_bytes(payload)
    (bundle / "_provenance.json").write_text(
        json.dumps(
            {
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "shader_sources": {"digest": None, "file_count": 0},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(vk, "_HERE", tmp_path)
    report = vk.verify_provenance()
    assert report["verdict"] == "MATCH"  # the bytes are the recorded bytes ...
    assert report["shader_source_file_count"] == 0  # ... and this says they came from nothing


def test_provenance_without_an_artifact_is_a_mismatch(tmp_path, monkeypatch):
    """A record with no binary beside it is not 'nothing to check'; it is a broken install."""
    bundle = tmp_path / "_lib"
    bundle.mkdir()
    (bundle / "_provenance.json").write_text(
        json.dumps({"artifact_sha256": "0" * 64}), encoding="utf-8"
    )
    monkeypatch.setattr(vk, "_HERE", tmp_path)
    assert vk.verify_provenance()["verdict"] == "MISMATCH"


# ---------------------------------------------------------------------------------------
# assert_ep_selected — the answer to the silent-fallback measurement
# ---------------------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, providers):
        self._p = list(providers)

    def get_providers(self):
        return list(self._p)


def test_assert_ep_selected_passes_when_selected():
    vk.assert_ep_selected(_FakeSession([vk.PROVIDER_NAME, "CPUExecutionProvider"]))


def test_assert_ep_selected_raises_on_silent_cpu_fallback():
    """Corresponds to measured case ``unregistered_name_failure_mode``.

    ORT warns and falls back to CPU with correct numbers. This turns that into a raise.
    """
    with pytest.raises(vk.EpNotSelected) as exc:
        vk.assert_ep_selected(_FakeSession(["CPUExecutionProvider"]))
    msg = str(exc.value)
    assert "CPUExecutionProvider" in msg
    assert "falls back to CPU" in msg


# ---------------------------------------------------------------------------------------
# Registration semantics — subprocess, because registration is process-global
# ---------------------------------------------------------------------------------------


@needs_lib
def test_double_registration_is_a_noop_for_the_same_library():
    """Corresponds to measured case ``double_registration``: ORT itself raises.

    A package that registers is imported more than once in real life, so the second call
    must be a no-op rather than an exception.
    """
    got = _in_subprocess(
        """
import onnxruntime_ep_vulkan as vk
a = vk.register_execution_provider_library()
b = vk.register_execution_provider_library()
emit(first=a, second=b, same=(a == b))
"""
    )
    assert got["same"] is True


@needs_lib
def test_registering_a_different_library_under_one_name_raises(tmp_path):
    """Idempotence must not become 'silently keep the first one'."""
    other = tmp_path / vk.ARTIFACT_FILENAME
    other.write_bytes(b"a different library entirely")
    got = _in_subprocess(
        f"""
import onnxruntime_ep_vulkan as vk
vk.register_execution_provider_library()
try:
    vk.register_execution_provider_library(path={str(other)!r})
except vk.EpVulkanError as exc:
    emit(raised=True, msg=str(exc))
else:
    emit(raised=False)
"""
    )
    assert got["raised"] is True
    assert "already registered" in got["msg"]


@needs_lib
def test_register_then_run_selects_the_ep():
    """The positive state, end to end, from the source tree.

    The clean-room equivalent (from a wheel, in a fresh venv, with the repository
    unreachable) is ``python/verify_cleanroom.py`` and its record
    ``bench/results/cleanroom_install_dev0.json``.
    """
    got = _in_subprocess(
        """
import numpy as np, onnxruntime as ort
from onnx import TensorProto, helper
import onnxruntime_ep_vulkan as vk
vk.register_execution_provider_library()
g = helper.make_graph(
    [helper.make_node("Add", ["a", "b"], ["c"])], "g",
    [helper.make_tensor_value_info("a", TensorProto.FLOAT, [4]),
     helper.make_tensor_value_info("b", TensorProto.FLOAT, [4])],
    [helper.make_tensor_value_info("c", TensorProto.FLOAT, [4])])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 10
sess = ort.InferenceSession(m.SerializeToString(), providers=vk.providers())
out = sess.run(None, {"a": np.ones(4, np.float32),
                      "b": np.full(4, 2.0, np.float32)})[0]
vk.assert_ep_selected(sess)
emit(providers=list(sess.get_providers()), correct=bool(np.allclose(out, 3.0)))
"""
    )
    assert vk.PROVIDER_NAME in got["providers"]
    assert got["correct"] is True


@needs_lib
def test_unregister_of_an_unregistered_name_returns_false():
    got = _in_subprocess(
        """
import onnxruntime_ep_vulkan as vk
emit(before=vk.unregister_execution_provider_library("NeverRegisteredAnywhere"))
"""
    )
    assert got["before"] is False


@needs_lib
def test_describe_reports_a_library_and_does_not_register():
    got = _in_subprocess(
        """
import onnxruntime as ort, onnxruntime_ep_vulkan as vk
info = vk.describe()
emit(has_lib=info["library_path"] is not None,
     registered=info["registered"],
     provenance_verdict=info["provenance"]["verdict"])
"""
    )
    assert got["has_lib"] is True
    assert got["registered"] is False, "describe() must not have side effects"
    assert got["provenance_verdict"] in {"UNBUNDLED", "MATCH", "MISMATCH"}
