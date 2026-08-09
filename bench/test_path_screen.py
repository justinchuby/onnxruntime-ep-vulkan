"""Behaviour tests for `bench/path_screen.py` — issue #78's published-artifact gate.

WHAT THIS FILE IS FOR
=====================
The leak is not hypothetical and it is not in a draft.  `bench/results/rust-model-runner/
mobilenetv2-12.json` is checked in, is published, and contains a home directory and an
extended-length path belonging to a named person.  Nothing in the repository stopped it,
because nothing in the repository looked.

The rejection of PR #100 named the shape of the fix precisely: a screen that recognises a
*known* set of cache roots is a screen that a path outside that set walks straight through.
So every test here is written against a root the author of the screen had no reason to think
of — `/srv`, `/nix`, `/Volumes`, `Q:\\`, `\\\\fileserver\\share` — and the screen has to catch
them because of what they *are*, not because they were enumerated.

The hard part, stated honestly
------------------------------
An ONNX node is named `/encoder/layer.0/attention/MatMul`.  A private model directory is named
`/srv/models/minilm`.  As text these are the same kind of object and no regex separates them;
any rule that rejects one rejects the other.  The screen therefore does not try to tell them
apart by text.  It uses *position*: a POSIX-absolute value is tolerated only under a key that
declares itself to hold graph names, and never anywhere else — and a drive letter, UNC path or
home macro is refused even there, because no ONNX node is called `C:\\Users\\justinchu`.

That boundary is a real limitation and these tests pin both of its sides: the exemption works
(node names survive), and the exemption is narrow (the same string under a different key does
not survive).

Every test drives the shipped production entry points — `screen_public_text`,
`screen_public_record`, `public_model_record` — never a copy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import pinned_bytes as pb  # noqa: E402
import path_screen as ps  # noqa: E402
from _polarity import refuses, selects  # noqa: E402


# --------------------------------------------------------------------------------------
# General private roots — the point of gate 5. None of these is a "known cache root".
# --------------------------------------------------------------------------------------

ARBITRARY_PRIVATE_ROOTS = [
    ("/srv/models/all-MiniLM-L6-v2/model.onnx", "a service root"),
    ("/data/scratch/justin/model.onnx", "a data mount"),
    ("/run/user/1000/models/model.onnx", "a runtime dir"),
    ("/Volumes/BuildSSD/models/model.onnx", "a macOS mounted volume"),
    ("/nix/store/abc123-onnx/model.onnx", "a nix store path"),
    ("/var/lib/ci/models/model.onnx", "a var path"),
    ("/opt/models/minilm/model.onnx", "an opt path"),
    ("/mnt/nas/models/model.onnx", "a network mount"),
    ("/media/usb0/models/model.onnx", "removable media"),
    ("/home/justin/.cache/huggingface/model.onnx", "a POSIX home"),
    ("/Users/justinchu/.cache/hub/model.onnx", "a macOS home"),
    ("/tmp/pytest-1234/model.onnx", "a temp dir"),
    ("/private/var/folders/zz/T/model.onnx", "a macOS temp"),
    ("/workspace/checkouts/repo/bench/model.onnx", "a CI workspace"),
    ("/github/home/.cache/model.onnx", "an actions home"),
]


@pytest.mark.parametrize("text,label", ARBITRARY_PRIVATE_ROOTS)
def test_an_arbitrary_private_posix_root_is_refused_even_though_no_list_names_it(text, label):
    """The screen must generalise. An allowlist of cache roots would pass every one of these."""
    refuses(ps.screen_public_text(text), because=label)


ARBITRARY_DRIVES = [
    ("C:\\Users\\justinchu\\.cache\\model.onnx", "the obvious one"),
    ("D:\\models\\minilm\\model.onnx", "a second drive"),
    ("Q:\\build\\artifacts\\model.onnx", "a drive nobody enumerates"),
    ("z:/lowercase/drive/model.onnx", "lowercase with forward slashes"),
    ("E:/CI/work/1/s/model.onnx", "an agent work dir"),
]


@pytest.mark.parametrize("text,label", ARBITRARY_DRIVES)
def test_any_drive_letter_is_refused_not_just_c(text, label):
    refuses(ps.screen_public_text(text), because=label)


UNC_AND_DEVICE = [
    ("\\\\?\\C:\\Users\\justinchu\\.copilot\\repos\\wt\\bench", "extended-length prefix"),
    ("\\\\.\\PIPE\\onnx", "device namespace"),
    ("\\\\fileserver\\share\\models\\model.onnx", "a UNC share"),
    ("//fileserver/share/models/model.onnx", "a UNC share, forward slashes"),
    ("\\\\?\\UNC\\server\\share\\model.onnx", "extended-length UNC"),
]


@pytest.mark.parametrize("text,label", UNC_AND_DEVICE)
def test_unc_and_device_and_extended_length_paths_are_refused(text, label):
    refuses(ps.screen_public_text(text), because=label)


HOME_MACROS = [
    ("~/.cache/huggingface/model.onnx", "tilde home"),
    ("%USERPROFILE%\\.cache\\model.onnx", "Windows user profile macro"),
    ("%LOCALAPPDATA%/Temp/model.onnx", "local appdata macro"),
    ("%TEMP%\\model.onnx", "temp macro"),
    ("$HOME/.cache/model.onnx", "POSIX home macro"),
    ("$XDG_CACHE_HOME/huggingface/model.onnx", "XDG cache macro"),
]


@pytest.mark.parametrize("text,label", HOME_MACROS)
def test_an_unexpanded_home_or_temp_macro_is_still_a_private_path(text, label):
    """`%USERPROFILE%` names a person as surely as the expansion does."""
    refuses(ps.screen_public_text(text), because=label)


# --------------------------------------------------------------------------------------
# Obfuscation: the same path, written the way a serialiser would have left it.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,label", [
    ("C:\\\\Users\\\\justinchu\\\\.cache\\\\model.onnx", "JSON-escaped backslashes"),
    ("{\"path\": \"C:\\\\Users\\\\justinchu\\\\model.onnx\"}", "a JSON document as a string"),
    ("path=C%3A%5CUsers%5Cjustinchu%5Cmodel.onnx", "percent-encoded"),
    ("\\/srv\\/models\\/minilm\\/model.onnx", "escaped forward slashes"),
    ("C:\\u005cUsers\\u005cjustinchu", "unicode-escaped separator"),
])
def test_an_escaped_or_encoded_path_is_still_a_path(text, label):
    refuses(ps.screen_public_text(text), because=label)


def test_a_path_hidden_in_utf16_bytes_is_decoded_and_refused():
    """A Windows API hands back UTF-16. A screen that only reads `str` sees nothing in it."""
    wide = "C:\\Users\\justinchu\\.cache\\model.onnx".encode("utf-16-le")
    refuses(ps.screen_public_text(wide), because="UTF-16-LE with no BOM")
    refuses(ps.screen_public_text(b"\xff\xfe" + wide), because="UTF-16-LE with BOM")
    refuses(ps.screen_public_text("/srv/models/x".encode("utf-16-be")), because="UTF-16-BE")


def test_wide_decoding_is_itself_a_two_polarity_instrument():
    value, why = ps.decode_wide_text("already text")
    assert value == "already text" and why
    value, why = ps.decode_wide_text(b"plain ascii")
    assert value == "plain ascii" and why
    refuses(ps.decode_wide_text(object()), because="not text or bytes")
    refuses(ps.decode_wide_text(b"\xff\xfe\x00"), because="BOM present, bytes do not decode")


def test_a_json_blob_carrying_a_path_is_refused_when_screened_as_a_record():
    payload = json.loads(json.dumps({"model": {"cache": "C:\\Users\\justinchu\\model.onnx"}}))
    refuses(ps.screen_public_record(payload), because="nested private path")


# --------------------------------------------------------------------------------------
# The exemption, both sides of it.
# --------------------------------------------------------------------------------------

LEGITIMATE_NODE_NAMES = [
    "/encoder/layer.0/attention/self/MatMul",
    "/embeddings/LayerNorm/Add_1",
    "/Constant_output_0",
    "/encoder/layer.11/output/dense/MatMul_output_0",
    "/Shape",
]


@pytest.mark.parametrize("name", LEGITIMATE_NODE_NAMES)
def test_a_real_onnx_node_name_survives_under_a_declared_name_key(name):
    """If the screen eats node names, every diagnostic that quotes one becomes unpublishable."""
    record = {"nodes": [{"name": name, "op_type": "MatMul"}]}
    kept, why = ps.screen_public_record(record)
    assert kept is record, why


@pytest.mark.parametrize("name", LEGITIMATE_NODE_NAMES)
def test_the_same_string_under_an_undeclared_key_is_not_exempt(name):
    """The exemption is positional. Under `cache_path` this is a directory, not a node."""
    refuses(ps.screen_public_record({"cache_path": name}), because="undeclared key")


def test_the_name_key_exemption_does_not_extend_to_drives_or_unc_or_macros():
    """No ONNX node is called `C:\\Users\\justinchu`."""
    for bad in ("C:\\Users\\justinchu\\model.onnx", "\\\\server\\share\\m", "~/x/y",
                "%TEMP%\\m.onnx"):
        refuses(ps.screen_public_record({"name": bad}), because=bad)


def test_a_posix_path_shaped_node_name_is_refused_by_the_text_screen_by_default():
    """`allow_graph_names` is opt-in. A caller that does not ask does not get the exemption."""
    refuses(ps.screen_public_text("/encoder/layer.0/MatMul"), because="default is strict")
    text = "/encoder/layer.0/MatMul"
    selects(ps.screen_public_text(text, allow_graph_names=True), text)


# --------------------------------------------------------------------------------------
# Things that must survive, or the screen is useless in production.
# --------------------------------------------------------------------------------------

PUBLISHABLE = [
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/model.onnx",
    "sentence-transformers/all-MiniLM-L6-v2",
    "onnx/model.onnx",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "pinned-cache",
    "bench/results/pinned-bytes/all-MiniLM-L6-v2.json",
    "pinned_bytes/1",
    ps.PUBLIC_PLACEHOLDER,
    "",
]


@pytest.mark.parametrize("text", PUBLISHABLE)
def test_the_public_identity_this_work_exists_to_publish_survives_the_screen(text):
    selects(ps.screen_public_text(text), text)


def test_non_string_scalars_pass_through_unchanged():
    for value in (None, True, False, 0, 90405214, 1.5):
        kept, why = ps.screen_public_text(value)
        assert kept is value, why


# --------------------------------------------------------------------------------------
# The production serializer.
# --------------------------------------------------------------------------------------

def _record(tmp_path):
    """Build a real, verified ProvenanceRecord through the shipped resolver."""
    import onnx
    from onnx import helper, TensorProto

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="/encoder/layer.0/Identity")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    blob = path.read_bytes()
    import hashlib
    digest = hashlib.sha256(blob).hexdigest()
    pin = {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "file": "onnx/model.onnx",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "sha256": digest,
        "pinned_bytes": len(blob),
        "source": "pinned-cache",
        "declared_external_files": 0,
    }
    side = tmp_path / "side.json"
    side.write_text(json.dumps({"onnx_sha256": digest}), encoding="utf-8")
    return pb.check_pinned_bytes(path, pin, sidecar=side)


def test_a_published_provenance_record_carries_the_public_identity_and_no_local_path(tmp_path):
    record = _record(tmp_path)
    payload = ps.public_model_record(record)
    blob = json.dumps(payload)
    assert "huggingface.co" in blob
    assert str(tmp_path) not in blob
    assert "AppData" not in blob and "Users" not in blob
    assert payload["provenance_ok"] is True
    assert payload["schema"] == pb.PROVENANCE_SCHEMA


def test_the_serializer_raises_rather_than_returning_a_leaky_dict(tmp_path):
    """A caller that ignores a return value publishes the artifact anyway."""
    record = _record(tmp_path)
    with pytest.raises(ps.PrivatePathLeak) as exc:
        ps.public_model_record(record, extra={"cache_path": str(tmp_path / "m.onnx")})
    assert "path" in str(exc.value).lower()


def test_the_serializer_refuses_an_object_it_cannot_account_for():
    with pytest.raises(ps.PrivatePathLeak):
        ps.public_model_record({"provenance_ok": True})
    with pytest.raises(ps.PrivatePathLeak):
        ps.public_model_record(None)


def test_extra_fields_may_not_silently_overwrite_the_records_own_verdict(tmp_path):
    record = _record(tmp_path)
    with pytest.raises(ps.PrivatePathLeak):
        ps.public_model_record(record, extra={"provenance_ok": True})


def test_a_refused_records_public_form_still_publishes_and_still_says_false(tmp_path):
    """Refusals get published too. They must be publishable without leaking the path."""
    record = _record(tmp_path)
    import dataclasses
    broken = dataclasses.replace(record, observed_sha256="0" * 64)
    payload = ps.public_model_record(broken)
    assert payload["provenance_ok"] is False
    assert payload["disagreements"]
    assert str(tmp_path) not in json.dumps(payload)


def test_the_screen_runs_over_the_shipped_sidecar_witness():
    """The artifact this branch adds must itself pass the screen it adds."""
    side = _BENCH / "results" / "pinned-bytes" / "all-MiniLM-L6-v2.json"
    assert side.is_file(), side
    payload = json.loads(side.read_text(encoding="utf-8"))
    kept, why = ps.screen_public_record(payload)
    assert kept is payload, why


def test_the_screen_catches_the_leak_that_is_already_checked_in():
    """`mobilenetv2-12.json` publishes a home directory today. This is the proof it does.

    The file is left exactly as it is — fixing it is issue #78's sibling, not #78 — but a
    screen that does not catch a leak already in the tree is not a screen.
    """
    leaky = _BENCH / "results" / "rust-model-runner" / "mobilenetv2-12.json"
    if not leaky.is_file():
        pytest.skip("the pre-existing artifact is not present in this checkout")
    payload = json.loads(leaky.read_text(encoding="utf-8"))
    why = refuses(ps.screen_public_record(payload), because="checked-in private paths")
    assert "drive-absolute" in why or "device" in why or "home" in why


# --------------------------------------------------------------------------------------
# Record-walk edge cases.
# --------------------------------------------------------------------------------------

def test_a_private_path_used_as_a_dict_key_is_refused():
    refuses(ps.screen_public_record({"C:\\Users\\justinchu": "ok"}), because="path as key")


def test_a_non_string_key_is_refused_rather_than_stringified():
    refuses(ps.screen_public_record({1: "ok"}), because="non-string key")


def test_the_walk_reaches_arbitrary_depth_and_through_lists():
    deep = {"a": [{"b": [{"c": ["/srv/models/x/y"]}]}]}
    why = refuses(ps.screen_public_record(deep), because="deeply nested")
    assert "$.a[0].b[0].c[0]" in why


def test_a_clean_record_is_returned_as_the_same_object_not_a_copy():
    """A caller must publish the object the screen saw, not a normalisation of it."""
    record = {"repo": "sentence-transformers/all-MiniLM-L6-v2", "bytes": 90405214}
    selects(ps.screen_public_record(record), record)


# --------------------------------------------------------------------------------------
# Falsifier closures - written because a mutant survived the first battery pass
# --------------------------------------------------------------------------------------

def test_a_path_whose_only_readable_form_is_the_de_escaped_one_is_refused():
    """Kills `screen-drop-deescape`.

    Every earlier escaped case still carried something a detector saw without any
    de-escaping - a literal ``C:\\`` for the drive rule, a literal ``\\\\?\\`` for the device
    rule - so deleting the backslash de-escape left the suite green. This is the case that
    genuinely needs it. A JSON-escaped UNC share doubles every separator, and the doubled
    form matches the UNC rule nowhere: ``\\\\`` is followed by another ``\\`` rather than by a
    host name. Only after ``\\\\`` -> ``\\`` is there a path to see.
    """
    escaped = "\\\\" * 2 + "fileserver" + "\\\\" + "share" + "\\\\" + "models" + "\\\\" + "w.data"
    assert ps._UNC.search(escaped) is None, "this case must be invisible before de-escaping"
    why = refuses(ps.screen_public_text(escaped), because="only visible once de-escaped")
    assert "UNC" in why, why


def test_a_file_url_is_named_as_a_filesystem_path_not_waved_through_as_a_url():
    """Kills `screen-file-url-allowed`.

    `file:///srv/models/x` is caught by the POSIX detector anyway, so deleting the
    `file://` arm left the suite green while making the reported finding wrong - and the
    finding is what a reader acts on. It is also the arm that stops `file://` being added
    to the public-URL allowlist by someone reading `_PUBLIC_URL` and generalising.
    """
    for text in ("file:///srv/models/x/model.onnx",
                 "file://localhost/data/models/model.onnx",
                 "FILE:///opt/models/model.onnx"):
        why = refuses(ps.screen_public_text(text), because=text)
        assert "file://" in why, why


def test_an_https_url_does_not_launder_a_path_that_follows_it():
    """A value that merely STARTS with a public URL is not a public value."""
    refuses(ps.screen_public_text(
        "https://huggingface.co/x/y resolved to C:\\Users\\justinchu\\m.onnx"))
    refuses(ps.screen_public_text("see https://example.invalid and /srv/models/m.onnx"))
