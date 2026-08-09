"""Self-tests for `bench/public_paths.py` — issue #78, blocker #3 (public path leak screening).

Named for the plausible-but-wrong reading each test prevents, in the style of
`bench/test_real_model.py`. `bench/results/rust-model-runner/mobilenetv2-12.json` (committed,
on `main`) already carries a literal `C:\\Users\\...` path — this file exists to make that
defect impossible to repeat through the one writer this module is.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import public_paths as pp  # noqa: E402


# ---------------------------------------------------------------------------
# root_public_path
# ---------------------------------------------------------------------------


def test_root_public_path_refuses_a_non_path_like_value():
    """A public record must not silently `str()` an unexpected object into path-shaped text."""
    for bad in (None, 42, 3.14, [], {}, object()):
        with pytest.raises(TypeError):
            pp.root_public_path(bad)


def test_root_public_path_roots_a_repo_relative_file():
    p = _BENCH / "real_model.py"
    token = pp.root_public_path(p)
    assert token == "<repo>/bench/real_model.py"
    assert "\\" not in token
    assert str(_BENCH.parent) not in token


def test_root_public_path_prefers_the_longest_matching_root():
    """`<venv>` is inside `<repo>`; a file under it must root to the more specific token, not
    the outer one — the longest-`Path.parts`-prefix rule this function documents."""
    venv = _BENCH.parent / ".venv"
    token = pp.root_public_path(venv / "Scripts" / "python.exe")
    assert token.startswith("<venv>/"), token


def test_root_public_path_honours_the_model_cache_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_VULKAN_MODEL_CACHE", str(tmp_path))
    token = pp.root_public_path(tmp_path / "models" / "all-MiniLM-L6-v2.onnx")
    assert token == "<model-cache>/models/all-MiniLM-L6-v2.onnx"


def test_root_public_path_falls_back_to_elsewhere_for_an_unrooted_path():
    """A path under none of the named roots is still never emitted raw — `<elsewhere>` plus
    only the basename, never the parent directories that would otherwise leak."""
    with tempfile.TemporaryDirectory() as td:
        # A location outside every named root this process knows (repo/venv/home/tmp/cache):
        # simulate by picking a path that does not resolve under any of them, using a name
        # unlikely to collide with the real tempdir root itself.
        odd = Path(td) / "not-a-named-root" / "weights.bin"
        token = pp.root_public_path(odd)
        # Under <tmp> if td happens to BE inside the system tempdir (likely on CI runners);
        # accept either correct outcome rather than asserting a specific one, since both are
        # public-safe.
        assert token.startswith("<tmp>/") or token == "<elsewhere>/weights.bin"


def test_root_public_path_survives_a_missing_file():
    """A path that does not exist must still root (a record about `ModelUnavailable` names a
    path that, by definition, is absent) — `.resolve()` failures are swallowed, not raised."""
    token = pp.root_public_path(_BENCH / "results" / "rust-model-runner" / "does-not-exist.onnx")
    assert token == "<repo>/bench/results/rust-model-runner/does-not-exist.onnx"


# ---------------------------------------------------------------------------
# _find_leaks — private scanning primitive, exercised directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaky", [
    '"/var/log/onnxruntime.log"',
    '"/opt/vendor/lib.so"',
    '"/home/justin/models/m.onnx"',
    '"/root/.cache/thing"',
    '"/private/tmp/x"',
    '"/proc/self/status"',
    '"/mnt/data/model.onnx"',
    '"/media/usb/model.onnx"',
    '"/usr/bin/python3"',
    r'"C:\\Users\\justin\\.cache\\model.onnx"',
    r'"C:/Users/justin/model.onnx"',
    r'"\\\\server\\share\\model.onnx"',
    r'"\\\\?\\C:\\Users\\justin\\model.onnx"',
    r'"\\\\.\\C:\\Users\\justin\\model.onnx"',
    r'".venv\\Lib\\site-packages\\onnxruntime"',
    r'"site-packages/onnxruntime/capi"',
    r'"__pycache__/real_model.cpython-312.pyc"',
    r'"Scripts\\python.exe"',
    r'"AppData\\Local\\Temp\\x"',
])
def test_find_leaks_catches_every_named_pattern(leaky):
    hits = pp._find_leaks(leaky)
    assert hits, f"expected a leak hit in {leaky!r}"


@pytest.mark.parametrize("clean", [
    '"/model/layers.0/self_attn/q_proj/MatMul"',
    '"/model/decoder/etc_block/MatMul"',
    '"/model/etc/block"',
    '"<repo>/bench/real_model.py"',
    '"<model-cache>/all-MiniLM-L6-v2.onnx"',
    '"<venv>/Scripts/python.exe"',
    '"<home>/notes.txt"',
    '"<tmp>/scratch.json"',
    '"<elsewhere>/weights.bin"',
    '"var_output/thing"',
    '"a_home_grown_value"',
])
def test_find_leaks_survives_public_and_node_shaped_strings(clean):
    hits = pp._find_leaks(clean)
    assert hits == [], f"false-positive leak hit(s) in {clean!r}: {hits}"


def test_find_leaks_reports_pattern_name_and_offset():
    hits = pp._find_leaks('prefix "/var/log/x" suffix')
    assert hits[0]["pattern"] == "posix_or_windows_root_segment"
    assert hits[0]["match"] == "/var"
    assert isinstance(hits[0]["at"], int)


def test_find_leaks_is_total_never_raises_on_arbitrary_text():
    """The accept polarity for a scanning primitive: garbage text is not an exception, it is
    zero or more hits — a screen that can itself crash is worse than no screen."""
    for text in ("", "no paths here at all", "\x00\x01 binary-ish \ufeff", "/", "\\"):
        assert isinstance(pp._find_leaks(text), list)


# ---------------------------------------------------------------------------
# write_public_json
# ---------------------------------------------------------------------------


def test_write_public_json_writes_a_clean_record(tmp_path):
    out = tmp_path / "clean.json"
    pp.write_public_json(out, {"key": "all-MiniLM-L6-v2", "path": "<model-cache>/m.onnx"})
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["key"] == "all-MiniLM-L6-v2"


def test_write_public_json_refuses_before_touching_disk(tmp_path):
    out = tmp_path / "leaky.json"
    with pytest.raises(pp.PublicPathLeak):
        pp.write_public_json(out, {"path": r"C:\Users\justin\models\m.onnx"})
    assert not out.exists(), "a refused write must never partially land on disk"


def test_write_public_json_strips_leading_underscore_keys(tmp_path):
    """The private channel never reaches the public file, even though it is a real path that
    would otherwise leak — `_runtime_path` is dropped, not screened-then-rejected."""
    out = tmp_path / "with_private.json"
    real_path = str(tmp_path / "models" / "m.onnx")
    pp.write_public_json(out, {"path": "<model-cache>/m.onnx", "_runtime_path": real_path})
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "_runtime_path" not in doc
    assert doc["path"] == "<model-cache>/m.onnx"


def test_write_public_json_screens_nested_structures(tmp_path):
    out = tmp_path / "nested.json"
    with pytest.raises(pp.PublicPathLeak):
        pp.write_public_json(out, {"models": [{"note": "see /var/log/audit.log"}]})
    assert not out.exists()


def test_write_public_json_leak_message_names_the_pattern_and_never_writes(tmp_path):
    out = tmp_path / "unc.json"
    with pytest.raises(pp.PublicPathLeak, match="unc_path|windows_drive_root"):
        pp.write_public_json(out, {"lib": r"\\server\share\onnxruntime.dll"})
    assert not out.exists()


# ---------------------------------------------------------------------------
# runtime_path
# ---------------------------------------------------------------------------


def test_runtime_path_returns_the_real_path(tmp_path):
    real = tmp_path / "m.onnx"
    rec = {"path": "<model-cache>/m.onnx", "_runtime_path": str(real)}
    assert pp.runtime_path(rec) == real


def test_runtime_path_refuses_a_record_with_no_private_channel():
    with pytest.raises(pp.RuntimePathUnavailable):
        pp.runtime_path({"path": "<model-cache>/m.onnx"})


def test_runtime_path_refuses_an_empty_private_value():
    with pytest.raises(pp.RuntimePathUnavailable):
        pp.runtime_path({"path": "<model-cache>/m.onnx", "_runtime_path": ""})


def test_runtime_path_honours_a_custom_key():
    rec = {"_alt_runtime_path": "/some/real/path"}
    assert pp.runtime_path(rec, key="_alt_runtime_path") == Path("/some/real/path")


# ---------------------------------------------------------------------------
# No network — this module never imports anything that could reach one
# ---------------------------------------------------------------------------


def test_public_paths_module_imports_no_networking_primitive():
    """`public_paths.py` screens text and writes local files; it has no business importing
    anything that could open a socket. Asserted on the module's own source text rather than
    by breaking `socket.connect` at runtime, because the property under test is "this module
    never even mentions the machinery", which a source scan proves directly."""
    src = (_BENCH / "public_paths.py").read_text(encoding="utf-8")
    for banned in ("socket", "urllib", "http.client", "requests"):
        assert banned not in src, f"public_paths.py must not import {banned!r}"
