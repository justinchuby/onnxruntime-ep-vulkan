"""Self-tests for `bench/pinned_bytes.py` — no GPU, no network, no runtime.

Named for the **plausible but wrong** reading each one prevents, in the style of
`bench/test_plausible_but_wrong.py`. Every test drives the shipped production entry points
(`read_pinned_identity`, `check_pinned_bytes`, `iter_tensor_protos`, `external_references`,
`confine_external_location`, `hash_external_refs`) rather than a copy of their logic, and no
test re-implements the predicate it is checking: the assertions are about *behaviour observed
through the shipped surface*, because a test that restates the implementation goes green with
it when the implementation is wrong.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import pinned_bytes as pb  # noqa: E402
from _polarity import refuses, selects  # noqa: E402

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: synthetic models. Nothing here is downloaded and nothing is 90 MB.
# ---------------------------------------------------------------------------

GOOD_PIN = {
    "repo": "sentence-transformers/all-MiniLM-L6-v2",
    "file": "onnx/model.onnx",
    "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "sha256": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "pinned_bytes": 90405214,
    "source": "pinned-cache",
    "declared_external_files": 0,
}


def _tiny_model(nodes=None, initializers=None, **graph_kw):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    nodes = nodes if nodes is not None else [helper.make_node("Identity", ["x"], ["y"])]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=initializers or [],
                              **graph_kw)
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 17
    return model


def _external_tensor(name="w", location="w.bin", offset=None, length=None, extra=()):
    t = helper.make_tensor(name, TensorProto.FLOAT, [4], [0.0, 0.0, 0.0, 0.0])
    t.ClearField("float_data")
    t.data_location = TensorProto.EXTERNAL
    pairs = [("location", location)]
    if offset is not None:
        pairs.append(("offset", offset))
    if length is not None:
        pairs.append(("length", length))
    pairs.extend(extra)
    for key, value in pairs:
        kv = t.external_data.add()
        kv.key = key
        kv.value = str(value)
    return t


def _write(model, tmp_path, name="m.onnx"):
    p = tmp_path / name
    onnx.save(model, str(p))
    return p


def _pin_for(path, **over):
    pin = dict(GOOD_PIN)
    pin["sha256"], pin["pinned_bytes"] = pb._sha256_and_size(Path(path))
    pin.update(over)
    return pin


def _sidecar(tmp_path, digest, name="side.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"onnx_sha256": digest}), encoding="utf-8")
    return p


def _verified(tmp_path, model=None, **pin_over):
    """A model on disk plus a pin and sidecar that agree with it. The accept polarity."""
    path = _write(model if model is not None else _tiny_model(), tmp_path)
    pin = _pin_for(path, **pin_over)
    return path, pin, _sidecar(tmp_path, pin["sha256"])


# ---------------------------------------------------------------------------
# Gate 1 — the metadata gate is TOTAL and TYPED
# ---------------------------------------------------------------------------

def test_a_complete_pin_is_admitted_and_says_why():
    ident, why = pb.read_pinned_identity(GOOD_PIN)
    assert ident is not None and why
    selects(pb.read_pinned_identity(ident), ident,
            because="an already-validated identity is passed through, not re-parsed")


@pytest.mark.parametrize("field", sorted(GOOD_PIN))
def test_every_pin_field_is_required_not_defaulted(field):
    """A pin missing any field pins less than it claims to; there is no default that is safe."""
    raw = {k: v for k, v in GOOD_PIN.items() if k != field}
    why = refuses(pb.read_pinned_identity(raw), because=f"{field} absent")
    assert field in why


@pytest.mark.parametrize("field", ["repo", "file", "revision", "sha256", "source"])
@pytest.mark.parametrize("bad", ["", "   ", "\t\n", None, 0, 1.5, True, False, [], {}, b"x"])
def test_no_text_pin_field_accepts_empty_whitespace_or_the_wrong_type(field, bad):
    raw = dict(GOOD_PIN)
    raw[field] = bad
    why = refuses(pb.read_pinned_identity(raw), because=f"{field}={bad!r}")
    assert field in why


@pytest.mark.parametrize("field", ["repo", "file", "revision", "sha256", "source"])
def test_surrounding_whitespace_is_refused_rather_than_stripped(field):
    """Normalising input is how a screen starts accepting shapes nobody chose."""
    raw = dict(GOOD_PIN)
    raw[field] = f" {GOOD_PIN[field]} "
    refuses(pb.read_pinned_identity(raw), because=f"{field} padded")


@pytest.mark.parametrize("bad", [
    "main", "HEAD", "refs/heads/main", "v2.0", "1110a24",
    "1110A243FDF4706B3F48F1D95DB1A4F5529B4D41",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d4",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d411",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d4g",
])
def test_a_mutable_or_malformed_revision_is_not_a_pin(bad):
    """#78: `Xenova/…@main` reported MODEL_OK for whatever `main` pointed at that day."""
    why = refuses(pb.read_pinned_identity({**GOOD_PIN, "revision": bad}))
    assert "revision" in why


@pytest.mark.parametrize("bad", [
    "6fd5d72f", GOOD_PIN["sha256"].upper(), GOOD_PIN["sha256"][:-1],
    GOOD_PIN["sha256"] + "0", "z" * 64,
])
def test_a_partial_or_upper_case_digest_is_refused_not_normalised(bad):
    why = refuses(pb.read_pinned_identity({**GOOD_PIN, "sha256": bad}))
    assert "sha256" in why


@pytest.mark.parametrize("bad", [0, -1, None, "90405214", 90405214.0, [], {}])
def test_pinned_bytes_of_zero_none_or_the_wrong_type_is_not_a_size(bad):
    why = refuses(pb.read_pinned_identity({**GOOD_PIN, "pinned_bytes": bad}))
    assert "pinned_bytes" in why


@pytest.mark.parametrize("bad", [True, False])
def test_pinned_bytes_of_a_bool_is_caught_even_though_bools_are_ints(bad):
    """`isinstance(True, int)` is True; `pinned_bytes: true` would otherwise pin one byte."""
    why = refuses(pb.read_pinned_identity({**GOOD_PIN, "pinned_bytes": bad}))
    assert "bool" in why


@pytest.mark.parametrize("bad", ["Xenova/all-MiniLM-L6-v2..", "all-MiniLM-L6-v2",
                                 "a/b/c", "/a/b", "a b/c", "https://hf.co/a/b"])
def test_a_repo_that_is_not_owner_slash_name_cannot_say_which_reexport_it_is(bad):
    why = refuses(pb.read_pinned_identity({**GOOD_PIN, "repo": bad}))
    assert "repo" in why


@pytest.mark.parametrize("bad", ["/onnx/model.onnx", "../model.onnx", "onnx//model.onnx",
                                 "C:/model.onnx", "onnx\\model.onnx", "onnx/./model.onnx",
                                 "http://x/model.onnx", "onnx/model.onnx/"])
def test_a_pinned_file_that_is_not_a_plain_relative_posix_name_is_refused(bad):
    why = refuses(pb.read_pinned_identity({**GOOD_PIN, "file": bad}))
    assert "file" in why


@pytest.mark.parametrize("bad", ["verified", "ok", "cache", "PINNED-CACHE", "trusted"])
def test_an_unrecognised_source_state_is_not_a_verified_one(bad):
    why = refuses(pb.read_pinned_identity({**GOOD_PIN, "source": bad}))
    assert "source" in why


@pytest.mark.parametrize("state", sorted(pb.UNVERIFIED_SOURCE_STATES))
def test_a_known_but_unverified_source_state_is_admitted_as_a_pin_and_refused_as_evidence(
        state, tmp_path):
    """Offline / download-failed / unpinned are legitimate STATES and never verified bytes."""
    ident, _ = pb.read_pinned_identity({**GOOD_PIN, "source": state})
    assert ident is not None
    path, pin, side = _verified(tmp_path)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(path, {**pin, "source": state}, sidecar=side)
    assert exc.value.reason == "unpinned_source"


@pytest.mark.parametrize("raw", [None, "a string", 17, [], (), b"{}", {1: "x"}, object()])
def test_the_metadata_gate_is_total_over_junk_and_never_raises(raw):
    """An uncaught AttributeError exits 1, which is what a genuine refusal exits."""
    refuses(pb.read_pinned_identity(raw))


def test_an_unknown_pin_field_is_refused_because_a_typo_is_a_silent_unpin():
    why = refuses(pb.read_pinned_identity({**GOOD_PIN, "sha_256": "x"}))
    assert "unknown" in why


def test_an_identity_cannot_be_hand_assembled_past_the_validation():
    """The dataclass re-validates, so a future caller in a hurry cannot bypass the reader."""
    with pytest.raises(pb.ProvenanceError):
        pb.PinnedIdentity(repo="a/b", file="m.onnx", revision="main",
                          sha256=GOOD_PIN["sha256"], pinned_bytes=1,
                          source="pinned-cache", declared_external_files=0)


def test_the_pinned_url_is_immutable_and_carries_no_credentials():
    ident, _ = pb.read_pinned_identity(GOOD_PIN)
    assert ident.url == (
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/model.onnx"
    )
    assert "@" not in ident.url and "token" not in ident.url


@pytest.mark.parametrize("field,bad", [
    ("revision", "main"), ("sha256", "6fd5d72f"), ("pinned_bytes", 0),
    ("pinned_bytes", False), ("source", "verified"), ("repo", ""),
])
def test_an_invalid_pin_is_refused_by_the_shipped_resolver_before_a_byte_is_read(
        field, bad, tmp_path):
    """Combinatorial metadata defects are driven through `check_pinned_bytes` itself."""
    path, pin, side = _verified(tmp_path)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(path, {**pin, field: bad}, sidecar=side)
    assert exc.value.reason == "invalid_pin"


# ---------------------------------------------------------------------------
# Gate 1/2 — the second witness, and a verdict that cannot disagree with itself
# ---------------------------------------------------------------------------

def test_a_present_sidecar_is_read_and_says_which_digest_it_recorded(tmp_path):
    side = _sidecar(tmp_path, GOOD_PIN["sha256"])
    value, why = pb.read_sidecar_sha256(side)
    assert value == GOOD_PIN["sha256"]
    assert "onnx_sha256" in why


@pytest.mark.parametrize("blob,label", [
    (None, "absent file"), ("not json", "unparseable"), ("[]", "not an object"),
    ('{"onnx_sha256": null}', "null digest"), ('{"onnx_sha256": 1}', "int digest"),
    ('{"onnx_sha256": "6fd5"}', "short digest"), ('{}', "no field"),
    ('{"onnx_sha256": true}', "bool digest"),
])
def test_every_unusable_sidecar_is_a_refusal_with_a_reason(blob, label, tmp_path):
    p = tmp_path / "side.json"
    if blob is not None:
        p.write_text(blob, encoding="utf-8")
    refuses(pb.read_sidecar_sha256(p), because=label)


def test_no_sidecar_path_at_all_is_a_refusal_not_a_none_shaped_pass():
    refuses(pb.read_sidecar_sha256(None))


def test_an_absent_sidecar_cannot_produce_a_verified_record(tmp_path):
    """#78 shipped `agrees_with_recorded_provenance: None` and ran anyway."""
    path, pin, _ = _verified(tmp_path)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(path, pin, sidecar=None)
    assert exc.value.reason == "provenance_mismatch"
    assert exc.value.record is not None and not exc.value.record.provenance_ok


def test_a_sidecar_that_disagrees_with_the_pin_is_a_refusal(tmp_path):
    path, pin, _ = _verified(tmp_path)
    other = _sidecar(tmp_path, "b" * 64, name="other.json")
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(path, pin, sidecar=other)
    assert "sidecar" in exc.value.detail


def test_agreeing_bytes_pin_and_sidecar_verify(tmp_path):
    path, pin, side = _verified(tmp_path)
    rec = pb.check_pinned_bytes(path, pin, sidecar=side)
    assert rec.provenance_ok is True
    assert rec.disagreements == ()


def test_provenance_ok_is_derived_so_a_record_cannot_carry_a_verdict_that_disagrees(tmp_path):
    """Gate 2, structurally: there is no assignment anywhere that can set the verdict."""
    path, pin, side = _verified(tmp_path)
    good = pb.check_pinned_bytes(path, pin, sidecar=side)
    assert good.provenance_ok is True
    import dataclasses as dc
    for field, value in [("observed_sha256", "c" * 64), ("observed_bytes", 1),
                         ("sidecar_sha256", None), ("source_state", "offline"),
                         ("external", {"scanned": False, "files": []})]:
        mutated = dc.replace(good, **{field: value})
        assert mutated.provenance_ok is False, field
        assert mutated.disagreements, field
        assert mutated.to_dict()["provenance_ok"] is False, field


def test_provenance_ok_is_not_a_settable_field(tmp_path):
    path, pin, side = _verified(tmp_path)
    rec = pb.check_pinned_bytes(path, pin, sidecar=side)
    with pytest.raises((AttributeError, TypeError)):
        rec.provenance_ok = True  # type: ignore[misc]
    assert "provenance_ok" not in {f.name for f in __import__("dataclasses").fields(rec)}


@pytest.mark.parametrize("mutate", ["byte", "truncate", "extend"])
def test_bytes_that_differ_from_the_pin_are_refused_however_they_differ(mutate, tmp_path):
    path, pin, side = _verified(tmp_path)
    raw = bytearray(path.read_bytes())
    if mutate == "byte":
        raw[-1] ^= 0xFF
    elif mutate == "truncate":
        del raw[-4:]
    else:
        raw.extend(b"\x00\x00\x00\x00")
    path.write_bytes(bytes(raw))
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(path, pin, sidecar=side)
    assert exc.value.reason == "provenance_mismatch"


def test_a_same_named_different_export_is_refused_which_is_issue_78_itself(tmp_path):
    """The 759c3cd2… blob: right name, right shape, bytes nobody verified."""
    path, pin, side = _verified(tmp_path)
    _write(_tiny_model([helper.make_node("Neg", ["x"], ["y"])]), tmp_path, "m.onnx")
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(path, pin, sidecar=side)
    assert exc.value.reason == "provenance_mismatch"


def test_an_absent_model_is_unavailable_never_success_shaped(tmp_path):
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(tmp_path / "nope.onnx", GOOD_PIN,
                              sidecar=_sidecar(tmp_path, GOOD_PIN["sha256"]))
    assert exc.value.reason == "model_missing"


def test_a_directory_where_the_model_should_be_is_not_a_model(tmp_path):
    (tmp_path / "m.onnx").mkdir()
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(tmp_path / "m.onnx", GOOD_PIN,
                              sidecar=_sidecar(tmp_path, GOOD_PIN["sha256"]))
    assert exc.value.reason == "model_missing"


def test_a_file_that_is_not_onnx_is_malformed_not_a_provenance_pass(tmp_path):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"not an onnx graph at all")
    pin = _pin_for(p)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(p, pin, sidecar=_sidecar(tmp_path, pin["sha256"]))
    assert exc.value.reason == "malformed_graph"


@pytest.mark.parametrize("state", ["offline", "download-failed", "unresolved", "unpinned", ""])
def test_a_resolved_source_state_that_is_not_verified_can_never_verify(state, tmp_path):
    path, pin, side = _verified(tmp_path)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(path, pin, sidecar=side, source_state=state)
    assert exc.value.reason == "unpinned_source"


# ---------------------------------------------------------------------------
# Gate 3 — the traversal reaches every tensor-bearing container
# ---------------------------------------------------------------------------

def _subgraph(nodes, initializers=(), name="sub"):
    y = helper.make_tensor_value_info("sy", TensorProto.FLOAT, [1])
    return helper.make_graph(list(nodes), name, [], [y], initializer=list(initializers))


def _hidden_external_models():
    """One model per place an external declaration can hide. Each must be found."""
    ext = _external_tensor
    out = {}

    out["graph.initializer"] = _tiny_model(initializers=[ext()])

    sp = helper.make_sparse_tensor(ext(name="sv"), ext(name="si"), [4])
    m = _tiny_model()
    m.graph.sparse_initializer.append(sp)
    out["graph.sparse_initializer.values"] = m

    sp2 = helper.make_sparse_tensor(
        helper.make_tensor("sv2", TensorProto.FLOAT, [1], [0.0]), ext(name="si2"), [4])
    m = _tiny_model()
    m.graph.sparse_initializer.append(sp2)
    out["graph.sparse_initializer.indices"] = m

    node = helper.make_node("Constant", [], ["y"], value=ext(name="cv"))
    out["node.attribute.t"] = _tiny_model([node])

    inner = _subgraph([helper.make_node("Identity", ["x"], ["sy"])], [ext(name="iw")])
    branch = helper.make_node("If", ["x"], ["y"], then_branch=inner,
                              else_branch=_subgraph(
                                  [helper.make_node("Identity", ["x"], ["sy"])]))
    out["node.attribute.g (then_branch)"] = _tiny_model([branch])

    deep_inner = _subgraph([helper.make_node("Identity", ["x"], ["sy"])], [ext(name="dw")])
    mid = _subgraph([helper.make_node(
        "If", ["x"], ["sy"], then_branch=deep_inner,
        else_branch=_subgraph([helper.make_node("Identity", ["x"], ["sy"])]))], name="mid")
    outer = helper.make_node("If", ["x"], ["y"], then_branch=mid,
                             else_branch=_subgraph(
                                 [helper.make_node("Identity", ["x"], ["sy"])]))
    out["nested subgraph, depth 2"] = _tiny_model([outer])

    m = _tiny_model()
    fn = helper.make_function(
        "test.dom", "F", ["fx"], ["fy"],
        [helper.make_node("Constant", [], ["fy"], value=ext(name="fw"))],
        [helper.make_opsetid("", 17)])
    m.functions.append(fn)
    out["functions[].node.attribute.t"] = m

    m = _tiny_model()
    fn = helper.make_function(
        "test.dom", "G", ["fx"], ["fy"],
        [helper.make_node("Identity", ["fx"], ["fy"])],
        [helper.make_opsetid("", 17)])
    ap = fn.attribute_proto.add()
    ap.name = "default_w"
    ap.type = onnx.AttributeProto.TENSOR
    ap.t.CopyFrom(ext(name="aw"))
    m.functions.append(fn)
    out["functions[].attribute_proto default"] = m

    m = _tiny_model()
    ti = m.training_info.add()
    ti.initialization.CopyFrom(
        _subgraph([helper.make_node("Identity", ["x"], ["sy"])], [ext(name="tw")]))
    out["training_info.initialization"] = m

    m = _tiny_model()
    ti = m.training_info.add()
    ti.algorithm.CopyFrom(
        _subgraph([helper.make_node("Identity", ["x"], ["sy"])], [ext(name="aw2")]))
    out["training_info.algorithm"] = m
    return out


@pytest.mark.parametrize("where", sorted(_hidden_external_models()))
def test_an_external_declaration_is_found_wherever_it_hides(where, tmp_path):
    """Hashing only `graph.initializer` is the partial walk this replaces."""
    model = _hidden_external_models()[where]
    refs = pb.external_references(model, model_root=tmp_path)
    assert refs, f"the walk did not reach {where}"
    assert any(r.location == "w.bin" for r in refs)


def test_a_graph_with_no_external_data_reports_zero_rather_than_unscanned(tmp_path):
    assert pb.external_references(_tiny_model(), model_root=tmp_path) == []
    scan = pb.hash_external_refs([], model_root=tmp_path)
    assert scan["scanned"] is True and scan["files"] == [] and scan["combined_sha256"] is None


def test_the_walk_reaches_every_initializer_not_just_the_first(tmp_path):
    inits = [helper.make_tensor(f"w{i}", TensorProto.FLOAT, [1], [0.0]) for i in range(5)]
    found = list(pb.iter_tensor_protos(_tiny_model(initializers=inits)))
    assert len(found) == 5


def test_traversal_depth_is_bounded_and_exceeding_it_refuses_rather_than_recursing():
    model = _hidden_external_models()["nested subgraph, depth 2"]
    with pytest.raises(pb.ProvenanceError) as exc:
        list(pb.iter_tensor_protos(model, max_depth=1))
    assert exc.value.reason == "traversal_bounded"


def test_traversal_work_is_bounded_so_a_hostile_graph_cannot_run_forever():
    inits = [helper.make_tensor(f"w{i}", TensorProto.FLOAT, [1], [0.0]) for i in range(10)]
    with pytest.raises(pb.ProvenanceError) as exc:
        list(pb.iter_tensor_protos(_tiny_model(initializers=inits), max_tensors=3))
    assert exc.value.reason == "traversal_bounded"


@pytest.mark.parametrize("model", [None, object(), "a string", 42])
def test_a_thing_that_is_not_a_model_fails_closed_rather_than_scanning_nothing(model):
    with pytest.raises(pb.ProvenanceError):
        list(pb.iter_tensor_protos(model))


def test_a_container_holding_the_wrong_type_is_refused_not_skipped():
    class FakeGraph:
        node = []
        initializer = ["not a tensor"]
        sparse_initializer = []

    class FakeModel:
        graph = FakeGraph()
        functions = []
        training_info = []

    with pytest.raises(pb.ProvenanceError) as exc:
        list(pb.iter_tensor_protos(FakeModel()))
    assert exc.value.reason == "malformed_graph"


# ---------------------------------------------------------------------------
# Gate 4 — external confinement, stable identity, deterministic extents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("location", [
    "/etc/passwd", "/srv/models/w.bin", "C:/Windows/w.bin", "c:\\w.bin",
    "\\\\server\\share\\w.bin", "//server/share/w.bin", "\\\\?\\C:\\w.bin",
    "\\\\.\\PhysicalDrive0", "../w.bin", "a/../../w.bin", "a//w.bin", "a/w.bin/",
    "./w.bin", "file:///w.bin", "https://example.com/w.bin", "a\\w.bin",
    "", "   ", "w.bin\x00", "CON", "nul.bin", "com1", "w.bin ", "w.bin.",
    None, 17, True, [], {},
])
def test_no_unsafe_external_location_is_ever_confined(location, tmp_path):
    refuses(pb.confine_external_location(location, model_root=tmp_path),
            because=f"location={location!r}")


@pytest.mark.parametrize("location", ["w.bin", "sub/w.bin", "a/b/c/w.bin", "w-1.data"])
def test_a_plain_relative_location_is_confined_and_lands_under_the_root(location, tmp_path):
    resolved, why = pb.confine_external_location(location, model_root=tmp_path)
    assert resolved is not None and why
    assert resolved.resolve().is_relative_to(tmp_path.resolve())


def test_a_model_root_that_is_not_a_directory_is_refused(tmp_path):
    f = tmp_path / "f"
    f.write_text("x", encoding="utf-8")
    refuses(pb.confine_external_location("w.bin", model_root=f))


@pytest.mark.parametrize("location", ["../w.bin", "/w.bin", "\\\\?\\C:\\w.bin", "a\\b"])
def test_an_unsafe_location_fails_the_shipped_scan_rather_than_being_skipped(location,
                                                                            tmp_path):
    model = _tiny_model(initializers=[_external_tensor(location=location)])
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.external_references(model, model_root=tmp_path)
    assert exc.value.reason == "external_unsafe"


def test_a_duplicated_location_key_is_refused_because_two_readers_would_disagree(tmp_path):
    t = _external_tensor(location="a.bin", extra=[("location", "b.bin")])
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.external_references(_tiny_model(initializers=[t]), model_root=tmp_path)
    assert exc.value.reason == "external_malformed"
    assert "location" in exc.value.detail


@pytest.mark.parametrize("pairs", [[], [("offset", "0")], [("location", "")],
                                   [("location", "   ")]])
def test_an_external_tensor_with_no_usable_location_can_never_be_verified(pairs, tmp_path):
    t = helper.make_tensor("w", TensorProto.FLOAT, [4], [0.0] * 4)
    t.ClearField("float_data")
    t.data_location = TensorProto.EXTERNAL
    for key, value in pairs:
        kv = t.external_data.add()
        kv.key, kv.value = key, value
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.external_references(_tiny_model(initializers=[t]), model_root=tmp_path)
    assert exc.value.reason == "external_malformed"


@pytest.mark.parametrize("field", ["offset", "length"])
@pytest.mark.parametrize("bad", ["-1", "0x10", "1.5", "1e6", " ", "abc", "99999999999999999999",
                                 str(2 ** 63)])
def test_a_negative_hex_float_or_out_of_range_extent_is_malformed(field, bad, tmp_path):
    t = _external_tensor(location="w.bin", **{field: bad})
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.external_references(_tiny_model(initializers=[t]), model_root=tmp_path)
    assert exc.value.reason == "external_malformed"


def test_declared_external_bytes_that_are_missing_never_verify(tmp_path):
    model = _tiny_model(initializers=[_external_tensor(location="w.bin")])
    refs = pb.external_references(model, model_root=tmp_path)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.hash_external_refs(refs, model_root=tmp_path)
    assert exc.value.reason == "external_missing"


def test_an_extent_that_runs_past_the_end_of_its_file_is_short_not_hashed(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"\x01" * 8)
    model = _tiny_model(initializers=[_external_tensor(location="w.bin", offset=4, length=16)])
    refs = pb.external_references(model, model_root=tmp_path)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.hash_external_refs(refs, model_root=tmp_path)
    assert exc.value.reason == "external_short"


def test_every_referenced_extent_is_hashed_and_the_result_is_deterministic(tmp_path):
    (tmp_path / "w.bin").write_bytes(bytes(range(64)))
    a = _external_tensor(name="a", location="w.bin", offset=0, length=16)
    b = _external_tensor(name="b", location="w.bin", offset=16, length=16)
    refs = pb.external_references(_tiny_model(initializers=[a, b]), model_root=tmp_path)
    first = pb.hash_external_refs(refs, model_root=tmp_path)
    second = pb.hash_external_refs(list(reversed(refs)), model_root=tmp_path)
    assert first["combined_sha256"] == second["combined_sha256"]
    assert first["extents"] == 2


def test_the_same_bytes_at_a_different_offset_do_not_hash_the_same(tmp_path):
    (tmp_path / "w.bin").write_bytes(bytes(range(64)))
    refs_a = pb.external_references(
        _tiny_model(initializers=[_external_tensor(location="w.bin", offset=0, length=16)]),
        model_root=tmp_path)
    refs_b = pb.external_references(
        _tiny_model(initializers=[_external_tensor(location="w.bin", offset=16, length=16)]),
        model_root=tmp_path)
    a = pb.hash_external_refs(refs_a, model_root=tmp_path)
    b = pb.hash_external_refs(refs_b, model_root=tmp_path)
    assert a["combined_sha256"] != b["combined_sha256"]


def test_changing_one_external_byte_changes_the_scan(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"\x01" * 32)
    refs = pb.external_references(
        _tiny_model(initializers=[_external_tensor(location="w.bin", offset=0, length=32)]),
        model_root=tmp_path)
    before = pb.hash_external_refs(refs, model_root=tmp_path)["combined_sha256"]
    (tmp_path / "w.bin").write_bytes(b"\x01" * 31 + b"\x02")
    after = pb.hash_external_refs(refs, model_root=tmp_path)["combined_sha256"]
    assert before != after


def test_a_regular_confined_file_opens_with_a_stable_identity(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"\x01" * 8)
    fd, st = pb.open_stable_file(tmp_path / "w.bin", model_root=tmp_path)
    try:
        assert st.st_size == 8
    finally:
        os.close(fd)


def test_opening_something_outside_the_model_root_is_refused_before_any_read(tmp_path):
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"secret")
    root = tmp_path / "root"
    root.mkdir()
    try:
        with pytest.raises(pb.ProvenanceError) as exc:
            pb.open_stable_file(outside, model_root=root)
        assert exc.value.reason == "external_unsafe"
    finally:
        outside.unlink()


def test_opening_a_missing_file_is_missing_not_empty(tmp_path):
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.open_stable_file(tmp_path / "nope.bin", model_root=tmp_path)
    assert exc.value.reason == "external_missing"


def test_opening_a_directory_is_refused_because_a_directory_holds_no_pinned_bytes(tmp_path):
    (tmp_path / "d").mkdir()
    with pytest.raises(pb.ProvenanceError):
        pb.open_stable_file(tmp_path / "d", model_root=tmp_path)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform has no symlink")
def test_a_symlink_that_escapes_the_root_is_refused_even_though_its_name_looks_innocent(
        tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "w.bin").write_bytes(b"secret")
    root = tmp_path / "root"
    root.mkdir()
    # `_make_escape_link` prefers a junction on Windows, which needs neither Developer Mode
    # nor elevation. This arm used to skip on a stock desk, and a confinement test that has
    # never run on the platform whose confinement is hardest is not evidence about it.
    if not _make_escape_link(root / "sub", outside):
        pytest.skip("this platform/user can create neither symlinks nor junctions")
    why = refuses(pb.confine_external_location("sub/w.bin", model_root=root))
    assert "symlink" in why or "outside" in why or "reparse" in why


def test_a_model_whose_external_count_disagrees_with_its_pin_is_refused(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"\x01" * 16)
    model = _tiny_model(initializers=[_external_tensor(location="w.bin", offset=0, length=16)])
    path = _write(model, tmp_path)
    pin = _pin_for(path, declared_external_files=0)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(path, pin, sidecar=_sidecar(tmp_path, pin["sha256"]))
    assert "external data disagrees" in exc.value.detail


def test_a_model_that_declares_external_data_and_has_it_verifies_when_the_pin_says_so(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"\x01" * 16)
    model = _tiny_model(initializers=[_external_tensor(location="w.bin", offset=0, length=16)])
    path = _write(model, tmp_path)
    pin = _pin_for(path, declared_external_files=1)
    rec = pb.check_pinned_bytes(path, pin, sidecar=_sidecar(tmp_path, pin["sha256"]))
    assert rec.provenance_ok is True
    assert rec.external["extents"] == 1 and rec.external["combined_sha256"]


def test_a_model_that_loses_its_external_data_relative_to_the_pin_is_refused(tmp_path):
    """#78: 'a pinned model unexpectedly declares external data it didn't at pin time'."""
    path, pin, side = _verified(tmp_path, declared_external_files=1)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.check_pinned_bytes(path, pin, sidecar=side)
    assert "external data disagrees" in exc.value.detail


# ---------------------------------------------------------------------------
# Gate 6 — provenance is decided before anything is executed
# ---------------------------------------------------------------------------

def test_deciding_provenance_does_not_import_onnxruntime_or_run_inference(tmp_path):
    """A check that had already loaded the runtime is trusting bytes it already handed it."""
    path, pin, side = _verified(tmp_path)
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys, json\n"
        f"sys.path.insert(0, {str(_BENCH)!r})\n"
        "import pinned_bytes as pb\n"
        f"pin = json.loads(open({str(pin_path)!r}, encoding='utf-8').read())\n"
        f"rec = pb.check_pinned_bytes({str(path)!r}, pin, sidecar={str(side)!r})\n"
        "assert rec.provenance_ok is True\n"
        "print(json.dumps(sorted(m for m in sys.modules if 'onnxruntime' in m)))\n",
        encoding="utf-8",
    )
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


# ---------------------------------------------------------------------------
# Falsifier closures - each of these was written because a mutant survived
# ---------------------------------------------------------------------------
#
# The battery in the PR body deleted one load-bearing behaviour at a time and re-ran the
# shipped suite. Twelve mutants survived the first pass. A surviving mutant is not a
# near-miss: it is a behaviour this repository relies on and nothing watches. The tests
# below are the ones that convict them, and each names the mutant it kills.


def test_an_empty_field_is_refused_for_being_empty_not_for_failing_a_later_shape_check():
    """Kills `meta-empty-ok`.

    Every empty field happens to fail a downstream regex too, so deleting the emptiness
    check left the suite green while making every refusal reason wrong. A refusal that
    says "revision is not a 40-char sha" about a field that is simply missing sends the
    reader to check the wrong thing.
    """
    for field in ("repo", "file", "revision", "sha256", "source"):
        for blank in ("", "   ", "\t", "\n"):
            why = refuses(pb.read_pinned_identity({**GOOD_PIN, field: blank}),
                          because=f"{field}={blank!r}")
            assert "empty or whitespace-only" in why, (field, blank, why)


def test_a_source_state_the_record_did_not_verify_under_cannot_be_ok(tmp_path):
    """Kills `verdict-ignores-source-state`."""
    path, pin, side = _verified(tmp_path)
    record = pb.check_pinned_bytes(path, pin, sidecar=side)
    assert record.provenance_ok is True
    for state in sorted(pb.UNVERIFIED_SOURCE_STATES) + ["", "verified", "ok", None]:
        mutated = dataclasses.replace(record, source_state=state)
        assert mutated.provenance_ok is False, state
        assert any("source" in d for d in mutated.disagreements), (state,
                                                                   mutated.disagreements)


def test_an_unverified_source_state_is_refused_even_when_the_pin_agrees_with_it(tmp_path):
    """Kills `verdict-ignores-source-state`.

    Agreement is not verification. A pin may legitimately record `source: "offline"` - that is
    a real state this module knows - and a record whose observed state matches it agrees
    perfectly while carrying no evidence at all. The verdict needs both clauses: the state must
    be one that carries verified bytes AND it must be the state the pin named. Dropping the
    first leaves "offline == offline" reading as success.
    """
    path, pin, side = _verified(tmp_path)
    record = pb.check_pinned_bytes(path, pin, sidecar=side)
    for state in sorted(pb.UNVERIFIED_SOURCE_STATES):
        agreeing = dataclasses.replace(
            dataclasses.replace(record, source_state=state),
            identity=dataclasses.replace(record.identity, source=state),
        )
        assert agreeing.source_state == agreeing.identity.source, state
        assert agreeing.provenance_ok is False, state
        assert any("carries verified bytes" in d for d in agreeing.disagreements), state


def test_a_sidecar_value_that_is_not_a_string_cannot_stand_in_for_the_second_witness(tmp_path):
    """Kills `verdict-ignores-sidecar`.

    The second witness is compared by value, and a value comparison alone trusts whatever the
    sidecar reader handed back. An object that answers "equal" to everything - a permissive
    wrapper, a mock left in a fixture, a NumPy-ish scalar - satisfies `== identity.sha256`
    without ever being a digest. The type guard is what makes "a sidecar exists and names these
    exact bytes" mean a string of 64 hex characters.
    """

    class AlwaysEqual:
        def __eq__(self, other):
            return True

        def __hash__(self):
            return 0

    path, pin, side = _verified(tmp_path)
    record = pb.check_pinned_bytes(path, pin, sidecar=side)
    spoofed = dataclasses.replace(record, sidecar_sha256=AlwaysEqual())
    assert spoofed.sidecar_sha256 == record.identity.sha256  # the value check is satisfied
    assert spoofed.provenance_ok is False
    assert any("sidecar" in d for d in spoofed.disagreements), spoofed.disagreements


def test_an_unverified_source_state_is_named_as_unverified_not_as_a_pin_disagreement(
        tmp_path):
    """The two source findings say different things and must not be reported as one.

    "offline" is not "the pin says something else" - it is "these bytes were never fetched
    under a state that carries verification". A reader told the wrong one goes and edits the
    pin.
    """
    path, pin, side = _verified(tmp_path)
    record = pb.check_pinned_bytes(path, pin, sidecar=side)
    for state in sorted(pb.UNVERIFIED_SOURCE_STATES):
        why = "; ".join(dataclasses.replace(record, source_state=state).disagreements)
        assert "is not one that carries verified bytes" in why, (state, why)
        assert "disagrees with the pin" not in why, (state, why)
    # ...and the other finding still exists, for a state that IS verified-capable but is not
    # the one the pin named.
    other = dict(pin)
    other["source"] = "pinned-cache"
    rec = pb.check_pinned_bytes(path, other, sidecar=side)
    mismatched = dataclasses.replace(
        dataclasses.replace(rec, source_state="pinned-cache"),
        identity=dataclasses.replace(rec.identity, source="unpinned"),
    )
    assert mismatched.provenance_ok is False
    assert any("disagrees with the pin" in d for d in mismatched.disagreements)


def _repeated_attribute_models():
    """Models that hide an external declaration in a REPEATED attribute field.

    `AttributeProto` has singular `t`/`sparse_tensor`/`g` AND repeated `tensors`/
    `sparse_tensors`/`graphs`. A walk that reads only the singular fields is silent about
    every model built by a converter that emits the repeated ones. Kills
    `traverse-drop-repeated-attrs`.
    """
    ext = _external_tensor
    out = {}

    node = helper.make_node("Identity", ["x"], ["y"])
    attr = node.attribute.add()
    attr.name = "many_tensors"
    attr.type = onnx.AttributeProto.TENSORS
    attr.tensors.append(ext(name="rt"))
    out["node.attribute.tensors[]"] = _tiny_model([node])

    node = helper.make_node("Identity", ["x"], ["y"])
    attr = node.attribute.add()
    attr.name = "many_sparse"
    attr.type = onnx.AttributeProto.SPARSE_TENSORS
    attr.sparse_tensors.append(
        helper.make_sparse_tensor(ext(name="rsv"), ext(name="rsi"), [4]))
    out["node.attribute.sparse_tensors[]"] = _tiny_model([node])

    node = helper.make_node("Identity", ["x"], ["y"])
    attr = node.attribute.add()
    attr.name = "many_graphs"
    attr.type = onnx.AttributeProto.GRAPHS
    attr.graphs.append(
        _subgraph([helper.make_node("Identity", ["x"], ["sy"])], [ext(name="rg")]))
    out["node.attribute.graphs[]"] = _tiny_model([node])
    return out


@pytest.mark.parametrize("where", sorted(_repeated_attribute_models()))
def test_a_repeated_attribute_field_is_walked_too(where, tmp_path):
    model = _repeated_attribute_models()[where]
    refs = pb.external_references(model, model_root=tmp_path)
    assert refs, f"the walk did not reach {where}"
    assert any(r.location == "w.bin" for r in refs)


# --- reparse points: junctions need no privilege on Windows, symlinks do -----------

def _make_escape_link(link: Path, target: Path) -> bool:
    """Point `link` at `target`, by whatever mechanism this machine allows. False if none.

    Junctions (`mklink /J`) are used on Windows in preference to symlinks because they need
    neither Developer Mode nor elevation, so this arm actually runs on a stock desk rather
    than skipping - and a confinement test that always skips is a confinement test that has
    never run.
    """
    if os.name == "nt":
        proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                              capture_output=True, text=True)
        return proc.returncode == 0 and link.exists()
    try:
        os.symlink(str(target), str(link), target_is_directory=target.is_dir())
        return True
    except (OSError, NotImplementedError):
        return False


def test_a_junction_in_any_path_component_is_refused(tmp_path):
    """Kills `external-allow-reparse-escape` and `external-skip-root-containment`.

    `weights/w.bin` is a textually perfect relative location. If `weights` is a junction to
    somewhere else on the machine, the bytes hashed are not under the model root at all -
    and the pre-#78 check would have hashed them and called the model verified.
    """
    root = tmp_path / "model"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "w.bin").write_bytes(b"bytes from outside the model root")
    if not _make_escape_link(root / "weights", outside):
        pytest.skip("this machine allows neither junctions nor symlinks")
    why = refuses(pb.confine_external_location("weights/w.bin", model_root=root),
                  because="component is a junction")
    assert "reparse" in why or "outside" in why


def test_a_reparse_point_is_refused_even_when_it_points_back_inside_the_root(tmp_path):
    """Kills `external-allow-reparse-escape` on its own.

    The escaping-junction case above is caught twice over - by the reparse scan and by the
    resolved-path containment check - so deleting either one alone left the suite green.
    They are not the same rule and this is the case that separates them: a junction that
    points *inside* the root passes containment and must still be refused, because what a
    reparse point resolves to today is not what it resolves to after someone re-points it,
    and the check that read it is not the code that will open it.
    """
    root = tmp_path / "model"
    (root / "real").mkdir(parents=True)
    (root / "real" / "w.bin").write_bytes(b"inside the root all along")
    if not _make_escape_link(root / "weights", root / "real"):
        pytest.skip("this machine allows neither junctions nor symlinks")
    resolved, why = pb.confine_external_location("real/w.bin", model_root=root)
    assert resolved is not None, why  # the non-junction route to the same bytes is fine
    why = refuses(pb.confine_external_location("weights/w.bin", model_root=root),
                  because="junction pointing inside the root")
    assert "reparse" in why or "symlink" in why or "junction" in why


def test_root_containment_refuses_on_its_own_when_the_reparse_scan_does_not_fire(tmp_path,
                                                                                monkeypatch):
    """Kills `external-skip-root-containment` on its own.

    The reparse scan and the containment check are deliberately redundant, which means
    neither is falsifiable while the other is present. Here the reparse scan is disabled -
    the mutant is injected into the production module rather than described - and the
    containment check must still refuse. If it does not, confinement rests on a single
    detector that a new platform, a new link type or a `False` return can defeat.
    """
    root = tmp_path / "model"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "w.bin").write_bytes(b"outside")
    if not _make_escape_link(root / "weights", outside):
        pytest.skip("this machine allows neither junctions nor symlinks")
    monkeypatch.setattr(pb, "_has_reparse_point", lambda p: False)
    why = refuses(pb.confine_external_location("weights/w.bin", model_root=root),
                  because="reparse detection disabled; containment must still hold")
    assert "outside the model root" in why, why


def test_opening_a_file_through_a_junction_is_refused_at_open_time(tmp_path):
    """Kills `external-drop-open-time-reparse`.

    Uses a junction that points back *inside* the root, for the same reason as the
    confinement case above: an escaping junction is refused by the containment check first,
    so it cannot falsify the open-time reparse check.
    """
    root = tmp_path / "model"
    (root / "real").mkdir(parents=True)
    (root / "real" / "w.bin").write_bytes(b"inside")
    if not _make_escape_link(root / "weights", root / "real"):
        pytest.skip("this machine allows neither junctions nor symlinks")
    fd, _ = pb.open_stable_file(root / "real" / "w.bin", model_root=root)
    os.close(fd)  # the direct route opens
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.open_stable_file(root / "weights" / "w.bin", model_root=root)
    assert exc.value.reason == "external_unsafe"
    assert "reparse" in exc.value.detail or "junction" in exc.value.detail


def test_a_file_swapped_between_the_check_and_the_open_is_refused(tmp_path, monkeypatch):
    """Kills `external-drop-toctou-identity`.

    The swap is simulated by making `fstat` report a different inode from the `lstat` the
    confinement check saw, which is exactly what a winning race produces. Racing a real
    rename in a test would be flaky; asserting the identity comparison is what makes the
    race lose is not.
    """
    root = tmp_path
    p = root / "w.bin"
    p.write_bytes(b"the bytes that were checked")
    real_fstat = os.fstat

    def swapped(fd):
        st = real_fstat(fd)
        return os.stat_result(tuple(st)[:1] + (st.st_ino + 1,) + tuple(st)[2:])

    monkeypatch.setattr(os, "fstat", swapped)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.open_stable_file(p, model_root=root)
    assert exc.value.reason == "external_unsafe"
    assert "changed identity" in exc.value.detail


def test_a_non_regular_file_does_not_hold_pinned_bytes(tmp_path, monkeypatch):
    """Kills `external-drop-regular-file`.

    A pipe or device cannot be opened portably inside a temp directory, so the mode is
    injected. What is under test is the decision, not the platform's ability to make one.
    """
    p = tmp_path / "w.bin"
    p.write_bytes(b"bytes")
    real_fstat = os.fstat

    def as_fifo(fd):
        st = real_fstat(fd)
        fields = list(tuple(st))
        fields[0] = (fields[0] & ~stat.S_IFMT(fields[0])) | stat.S_IFIFO
        return os.stat_result(tuple(fields))

    monkeypatch.setattr(os, "fstat", as_fifo)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.open_stable_file(p, model_root=tmp_path)
    assert exc.value.reason == "external_unsafe"
    assert "regular file" in exc.value.detail


def test_an_extent_that_runs_past_the_end_of_the_file_is_refused(tmp_path):
    """Kills `external-allow-short-extent`.

    A declared extent longer than the file is a truncated download. Hashing whatever was
    there and reporting a digest turns "the weights are incomplete" into "here is a hash",
    which is the exact substitution shape of #78 one level down.
    """
    (tmp_path / "w.bin").write_bytes(b"0123456789")
    ref = pb.ExternalRef(tensor="w", where="graph.initializer[0]", location="w.bin",
                         offset=0, length=64)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.hash_external_refs([ref], model_root=tmp_path)
    assert exc.value.reason == "external_short"

    past_eof = pb.ExternalRef(tensor="w", where="graph.initializer[0]", location="w.bin",
                              offset=100, length=1)
    with pytest.raises(pb.ProvenanceError):
        pb.hash_external_refs([past_eof], model_root=tmp_path)

    # The arm that the read loop cannot catch on its own, and therefore the one that makes the
    # up-front bounds check load-bearing rather than belt-and-braces: an offset past EOF with no
    # declared length. `end` becomes the file size, the extent length computes negative, the
    # read loop never executes, and a digest of nothing comes back looking like success. This is
    # a truncated download reporting a hash.
    open_ended = pb.ExternalRef(tensor="w", where="graph.initializer[0]", location="w.bin",
                                offset=100, length=None)
    with pytest.raises(pb.ProvenanceError) as exc:
        pb.hash_external_refs([open_ended], model_root=tmp_path)
    assert exc.value.reason == "external_short"
    assert "10" in exc.value.detail  # says how big the file actually is


def test_an_extent_that_fits_is_hashed_deterministically(tmp_path):
    """The accept polarity of the arm above: a whole extent hashes, twice, the same."""
    (tmp_path / "w.bin").write_bytes(b"0123456789")
    ref = pb.ExternalRef(tensor="w", where="graph.initializer[0]", location="w.bin",
                         offset=2, length=4)
    first = pb.hash_external_refs([ref], model_root=tmp_path)
    second = pb.hash_external_refs([ref], model_root=tmp_path)
    assert first["combined_sha256"] == second["combined_sha256"]
    assert first["scanned"] is True and first["extents"] == 1
