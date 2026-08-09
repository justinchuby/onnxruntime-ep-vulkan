"""Self-tests for `bench/real_model.py` — no GPU, no EP, no model file.

Each test is named for the **plausible but wrong** reading it prevents, in the style of
`bench/test_plausible_but_wrong.py`. The point of this file is that the parts of the #56
instrument that can be wrong without a device — provenance, feeds, throughput arithmetic,
pairing, verdicts — are checked on every `pytest bench` run, not only on the one machine that
has an RTX A1000 and a Foundry cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import real_model as rm  # noqa: E402


# ---------------------------------------------------------------------------
# Model identity and provenance
# ---------------------------------------------------------------------------

def test_no_model_spec_carries_a_literal_path():
    """A literal cache path is a guess about a version (#11: the cache moved under us)."""
    for spec in rm.MODELS.values():
        blob = repr(spec)
        assert ".foundry" not in blob
        assert ":\\" not in blob and ":/" not in blob


def test_phi35_is_resolved_through_foundry_not_a_filename():
    assert rm.PHI35.resolver == "foundry"
    assert rm.PHI35.variant_name and rm.PHI35.execution_provider


def test_missing_model_raises_rather_than_skipping(tmp_path, monkeypatch):
    """#56 asks for numbers on real models; a lane that silently skips one reads as complete."""
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    with pytest.raises(rm.ModelUnavailable):
        rm.resolve_model(rm.MOBILENETV2)


def test_repo_cache_dir_honours_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    assert rm.repo_cache_dir() == tmp_path


def test_recorded_sha256_comes_from_another_tools_artifact(tmp_path):
    """A hash this module both writes and checks proves nothing, so it is read from elsewhere."""
    (tmp_path / "rust-model-runner").mkdir()
    (tmp_path / rm.MOBILENETV2.recorded_provenance).write_text('{"onnx_sha256": "abc"}')
    assert rm.recorded_sha256(rm.MOBILENETV2, tmp_path) == "abc"


def test_external_weights_are_hashed_not_just_the_graph(tmp_path):
    """The Foundry Phi-3.5 `.onnx` is 26 MB and its weights are 2.29 GB in a sibling `.data`
    file. Hashing the graph alone identifies the topology and says nothing about the numbers the
    benchmark multiplies: replace the blob and the recorded hash does not move."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    blob = tmp_path / "weights.bin"
    payload = np.arange(16, dtype=np.float32)
    blob.write_bytes(payload.tobytes())
    t = numpy_helper.from_array(payload.reshape(4, 4), name="W")
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    t.external_data.extend([
        onnx.StringStringEntryProto(key="location", value="weights.bin"),
        onnx.StringStringEntryProto(key="offset", value="0"),
        onnx.StringStringEntryProto(key="length", value=str(payload.nbytes)),
    ])
    graph = helper.make_graph([], "g", [], [], initializer=[t])
    model = helper.make_model(graph)
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert [f["location"] for f in rec["files"]] == ["weights.bin"]
    assert rec["files"][0]["sha256"] == rm.sha256_file(blob)
    assert rec["files"][0]["bytes"] == payload.nbytes


def test_a_graph_with_no_external_data_says_so_rather_than_looking_unscanned(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), name="W")
    model = helper.make_model(helper.make_graph([], "g", [], [], initializer=[t]))
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())
    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert rec["files"] == []
    assert "covers the weights" in rec["reason"]


def test_a_missing_weight_blob_is_reported_not_silently_skipped(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), name="W")
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    t.external_data.extend([onnx.StringStringEntryProto(key="location", value="gone.bin")])
    model = helper.make_model(helper.make_graph([], "g", [], [], initializer=[t]))
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())
    rec = rm.external_data_provenance(mpath)
    assert rec["files"][0]["missing"] is True
    assert rec["files"][0]["sha256"] is None


def test_recorded_sha256_absent_is_none_not_empty_string(tmp_path):
    """``None`` (no record) and ``""`` (a record of nothing) are different states."""
    assert rm.recorded_sha256(rm.MOBILENETV2, tmp_path) is None


def test_resolve_model_reports_provenance_disagreement_rather_than_crashing(tmp_path,
                                                                           monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / rm.MOBILENETV2.cache_filename).write_bytes(b"not really an onnx file")
    (tmp_path / "rust-model-runner").mkdir()
    (tmp_path / rm.MOBILENETV2.recorded_provenance).write_text(
        '{"onnx_sha256": "0000000000000000"}')
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(models))
    rec = rm.resolve_model(rm.MOBILENETV2, results_dir=tmp_path)
    assert rec["agrees_with_recorded_provenance"] is False
    assert rec["sha256"] != rec["recorded_sha256"]


# ---------------------------------------------------------------------------
# The MiniLM pin itself (issue #78)
# ---------------------------------------------------------------------------

def test_minilm_pin_matches_the_exact_immutable_identity():
    """The pin this PR exists to hold — drifting any one of these fields silently would be
    exactly the defect #78 was opened over."""
    assert rm.MINILM.source_repo == "sentence-transformers/all-MiniLM-L6-v2"
    assert rm.MINILM.source_revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert rm.MINILM.source_file == "onnx/model.onnx"
    assert rm.MINILM.pinned_sha256 == (
        "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
    )
    assert rm.MINILM.pinned_bytes == 90405214
    assert rm.MINILM.pinned_external == ()


def test_minilm_source_revision_is_not_a_mutable_ref():
    assert rm.MINILM.source_revision.lower() not in ("main", "master", "head", "latest")


def test_minilm_is_provenance_only_and_never_reachable_through_cases_for():
    """Capability gap, not a provenance gap (#74/#75/#76): no sentence-embedding case/feed
    builder exists yet, so this model must never be fed to a session."""
    assert rm.MINILM.capability == "provenance_only"
    assert rm.cases_for(rm.MINILM) is None


def test_minilm_pinned_source_url_is_immutable_never_resolve_main():
    url = rm.pinned_source_url(rm.MINILM)
    assert url == (
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/model.onnx"
    )
    assert "/resolve/main/" not in url


def test_pinned_source_url_refuses_an_unpinned_spec():
    unpinned = rm.ModelSpec(key="k", family="f", resolver="repo-cache", cache_filename="x")
    with pytest.raises(ValueError):
        rm.pinned_source_url(unpinned)


# ---------------------------------------------------------------------------
# ModelSpec construction — all-or-nothing, fullmatch, every reason reported
# ---------------------------------------------------------------------------

def _minilm_kwargs(**overrides) -> dict:
    """The five pin fields plus key/family/resolver, as `ModelSpec(**kwargs)` accepts, so a
    single field can be overridden per test without repeating all the others."""
    base = dict(
        key="test-pin", family="bert", resolver="repo-cache", cache_filename="x.onnx",
        capability="provenance_only",
        source_repo="sentence-transformers/all-MiniLM-L6-v2",
        source_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        source_file="onnx/model.onnx",
        pinned_sha256="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        pinned_bytes=90405214,
    )
    base.update(overrides)
    return base


def test_unpinned_spec_construction_never_runs_the_pin_validation():
    spec = rm.ModelSpec(key="k", family="f", resolver="repo-cache", cache_filename="x.onnx")
    assert spec.pinned is False  # construction did not raise


def test_fully_correct_pinned_spec_constructs_without_raising():
    spec = rm.ModelSpec(**_minilm_kwargs())
    assert spec.pinned is True


@pytest.mark.parametrize("field", ["source_repo", "source_revision", "source_file",
                                   "pinned_sha256"])
def test_pinned_spec_refuses_a_single_missing_string_field(field):
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(**{field: None}))
    assert any(r.startswith(field) and "missing" in r for r in ei.value.reasons)


@pytest.mark.parametrize("field", ["source_repo", "source_revision", "source_file",
                                   "pinned_sha256"])
def test_pinned_spec_refuses_a_single_empty_string_field(field):
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(**{field: ""}))
    assert any(r.startswith(field) and "empty" in r for r in ei.value.reasons)


def test_pinned_spec_refuses_missing_pinned_bytes():
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(pinned_bytes=None))
    assert any(r.startswith("pinned_bytes") for r in ei.value.reasons)


@pytest.mark.parametrize("bad_bytes", [0, -1, -90405214])
def test_pinned_spec_refuses_zero_or_negative_pinned_bytes(bad_bytes):
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(pinned_bytes=bad_bytes))
    assert any(r.startswith("pinned_bytes") for r in ei.value.reasons)


def test_pinned_spec_refuses_a_bool_for_pinned_bytes():
    """`bool` is an `int` subclass in Python; `True == 1` would otherwise silently pass as a
    one-byte file."""
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(pinned_bytes=True))
    assert any(r.startswith("pinned_bytes") for r in ei.value.reasons)


@pytest.mark.parametrize("ref", ["main", "master", "HEAD", "head", "latest"])
def test_pinned_spec_refuses_a_mutable_source_revision_ref(ref):
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(source_revision=ref))
    assert any("mutable ref" in r for r in ei.value.reasons)


def test_pinned_spec_refuses_a_short_revision():
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(source_revision="1110a243"))
    assert any(r.startswith("source_revision") for r in ei.value.reasons)


def test_pinned_spec_refuses_an_uppercase_revision():
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(
            source_revision="1110A243FDF4706B3F48F1D95DB1A4F5529B4D41"))
    assert any(r.startswith("source_revision") for r in ei.value.reasons)


def test_pinned_spec_refuses_a_non_hex_sha256():
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(pinned_sha256="z" * 64))
    assert any(r.startswith("pinned_sha256") for r in ei.value.reasons)


def test_pinned_spec_refuses_a_short_sha256():
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(pinned_sha256="6fd5d72f"))
    assert any(r.startswith("pinned_sha256") for r in ei.value.reasons)


def test_pinned_spec_refuses_a_source_repo_with_no_owner_slash_name_shape():
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(source_repo="not-a-slug"))
    assert any(r.startswith("source_repo") for r in ei.value.reasons)


@pytest.mark.parametrize("bad", ["/onnx/model.onnx", "\\onnx\\model.onnx"])
def test_pinned_spec_refuses_a_rooted_source_file(bad):
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(source_file=bad))
    assert any("repo-relative" in r for r in ei.value.reasons)


def test_pinned_spec_refuses_a_dotdot_source_file():
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(source_file="../onnx/model.onnx"))
    assert any(".." in r for r in ei.value.reasons)


@pytest.mark.parametrize("field,bad", [
    ("source_repo", " sentence-transformers/all-MiniLM-L6-v2"),
    ("source_repo", "sentence-transformers/all-MiniLM-L6-v2\n"),
    ("source_revision", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41\n"),
    ("source_revision", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41 "),
    ("source_file", "\tonnx/model.onnx"),
    ("pinned_sha256",
     "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452\r\n"),
])
def test_pinned_spec_refuses_whitespace_or_newline_contaminated_fields(field, bad):
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(**{field: bad}))
    assert any(r.startswith(field) and ("whitespace" in r or "newline" in r)
              for r in ei.value.reasons)


def test_pinned_spec_reports_every_bad_field_not_just_the_first():
    """All-or-nothing means every clause runs; two bad fields must both be named, not just
    whichever the constructor happens to check first."""
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(source_repo="", pinned_bytes=-5))
    assert len(ei.value.reasons) >= 2
    assert any(r.startswith("source_repo") for r in ei.value.reasons)
    assert any(r.startswith("pinned_bytes") for r in ei.value.reasons)


def test_pinned_spec_does_not_double_report_an_already_missing_field():
    """A field already reported missing/empty must not ALSO be reported as failing its
    format regex — that reads as two independent problems where there is exactly one."""
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.ModelSpec(**_minilm_kwargs(source_repo=""))
    hits = [r for r in ei.value.reasons if r.startswith("source_repo")]
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# verify_source_metadata — unchecked vs. verified vs. refused, never conflated
# ---------------------------------------------------------------------------

def _pinned_spec_for_payload(payload: bytes, **overrides) -> "rm.ModelSpec":
    """A `MINILM`-shaped spec whose pin is self-consistent with *payload*'s own hash/size.
    The REAL pin (`6fd5d72f...`) can only be produced by the real Hugging Face bytes, so any
    test that needs `resolve_model`/`verify_source_metadata` to observe an actual
    ``state == "verified"`` result needs its own fabricated bytes/pin pair, never the genuine
    one."""
    fields = dict(pinned_sha256=hashlib.sha256(payload).hexdigest(), pinned_bytes=len(payload))
    fields.update(overrides)
    return dataclasses.replace(rm.MINILM, **fields)


def _clean_metadata(spec: "rm.ModelSpec") -> dict:
    return {
        "source_repo": spec.source_repo,
        "source_revision": spec.source_revision,
        "source_file": spec.source_file,
        "onnx_bytes": spec.pinned_bytes,
    }


def _clean_external() -> dict:
    return {"scanned": True, "files": [],
            "reason": "graph carries no external initializers"}


def test_verify_source_metadata_unpinned_is_a_third_state_not_true_or_false():
    """``state == "unpinned"`` must be distinct from both ``"verified"`` and a refusal — an
    unpinned spec has nothing pinned to check, and reporting it as anything else would say
    more than is known."""
    unpinned = rm.ModelSpec(key="k", family="f", resolver="repo-cache", cache_filename="x")
    result = rm.verify_source_metadata(
        unpinned, resolved_sha256="deadbeef", resolved_bytes=1, metadata={}, external={},
    )
    assert result == {"state": "unpinned", "provenance_ok": None}


def test_verify_source_metadata_verified_when_everything_agrees():
    payload = b"clean-room verified-state bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    result = rm.verify_source_metadata(
        spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
        metadata=_clean_metadata(spec), external=_clean_external(),
    )
    assert result["state"] == "verified"
    assert result["provenance_ok"] is True


@pytest.mark.parametrize("key", ["source_repo", "source_revision", "source_file",
                                 "onnx_bytes"])
def test_verify_source_metadata_refuses_a_missing_metadata_field(key):
    payload = b"clean-room missing-field bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    metadata = _clean_metadata(spec)
    metadata[key] = None
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
            metadata=metadata, external=_clean_external(),
        )
    assert any(key in r and "missing" in r for r in ei.value.reasons)


@pytest.mark.parametrize("key", ["source_repo", "source_revision", "source_file"])
def test_verify_source_metadata_refuses_an_empty_metadata_string(key):
    payload = b"clean-room empty-field bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    metadata = _clean_metadata(spec)
    metadata[key] = ""
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
            metadata=metadata, external=_clean_external(),
        )
    assert any(key in r and "empty" in r for r in ei.value.reasons)


@pytest.mark.parametrize("key", ["source_repo", "source_revision", "source_file"])
def test_verify_source_metadata_refuses_a_whitespace_newline_metadata_string(key):
    payload = b"clean-room whitespace-field bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    metadata = _clean_metadata(spec)
    metadata[key] = metadata[key] + "\n"
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
            metadata=metadata, external=_clean_external(),
        )
    assert any(key in r and ("whitespace" in r or "newline" in r) for r in ei.value.reasons)


def test_verify_source_metadata_refuses_source_drift_wrong_repo():
    """A mutable-metadata substitute (e.g. a different, byte-identical repo) cannot describe
    matching bytes as THIS pin — the repo/revision/file are checked independently of the
    hash."""
    payload = b"clean-room drift-repo bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    metadata = _clean_metadata(spec)
    metadata["source_repo"] = "Xenova/all-MiniLM-L6-v2"
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
            metadata=metadata, external=_clean_external(),
        )
    assert any("source drift" in r for r in ei.value.reasons)


def test_verify_source_metadata_refuses_source_drift_wrong_revision():
    payload = b"clean-room drift-revision bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    metadata = _clean_metadata(spec)
    metadata["source_revision"] = "0" * 40
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
            metadata=metadata, external=_clean_external(),
        )
    assert any("source_revision" in r and "does not match the pin" in r
              for r in ei.value.reasons)


def test_verify_source_metadata_refuses_wrong_hash_correct_size():
    """The cached-substitute shape (#78's ``759c3cd2...``): bytes of the right SIZE but the
    wrong content must still refuse."""
    payload = b"clean-room wrong-hash bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256="0" * 64, resolved_bytes=spec.pinned_bytes,
            metadata=_clean_metadata(spec), external=_clean_external(),
        )
    assert any("resolved sha256" in r for r in ei.value.reasons)


def test_verify_source_metadata_refuses_correct_hash_wrong_size():
    payload = b"clean-room wrong-size bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes + 1,
            metadata=_clean_metadata(spec), external=_clean_external(),
        )
    assert any("resolved size" in r for r in ei.value.reasons)


def test_verify_source_metadata_refuses_onnx_bytes_drift_in_the_sidecar_itself():
    """The recorded sidecar's own ``onnx_bytes`` field disagreeing with the pin is source
    drift in the metadata, independent of whatever the freshly-resolved file's real size
    is."""
    payload = b"clean-room onnx-bytes-drift bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    metadata = _clean_metadata(spec)
    metadata["onnx_bytes"] = spec.pinned_bytes + 1
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
            metadata=metadata, external=_clean_external(),
        )
    assert any("onnx_bytes" in r and "source drift" in r for r in ei.value.reasons)


def test_verify_source_metadata_refuses_an_unscanned_external_graph():
    """A pin cannot certify external-data state over a graph `external_data_provenance`
    itself refused to scan — that would let a malformed/unscannable graph pass by simply not
    looking."""
    payload = b"clean-room unscanned bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
            metadata=_clean_metadata(spec),
            external={"scanned": False, "files": [], "reason": "malformed tensor"},
        )
    assert any("could not be scanned" in r for r in ei.value.reasons)


def test_verify_source_metadata_refuses_unexpected_external_data():
    """``pinned_external=()`` asserts NONE exist; a graph that turns out to carry external
    weights disagrees with that assertion and must refuse, not silently accept extra data
    the pin never claimed."""
    payload = b"clean-room unexpected-external bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
            metadata=_clean_metadata(spec),
            external={"scanned": True,
                     "files": [{"location": "extra.bin", "bytes": 1, "sha256": "a" * 64}],
                     "reason": None},
        )
    assert any("external_data" in r and "location(s)" in r for r in ei.value.reasons)


def test_verify_source_metadata_never_returns_provenance_ok_false():
    """There is no ``provenance_ok=False`` return for a pinned spec — a bad pin RAISES. A
    caller that could write ``if not rec["provenance_ok"]:`` past a returned falsy value is
    exactly the "unchecked read as verified" shape #78 exists to close."""
    payload = b"clean-room no-false-return bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    with pytest.raises(rm.ModelProvenanceRefused):
        rm.verify_source_metadata(
            spec, resolved_sha256="0" * 64, resolved_bytes=spec.pinned_bytes,
            metadata=_clean_metadata(spec), external=_clean_external(),
        )


def test_verify_source_metadata_reports_every_disagreement_not_just_the_first():
    payload = b"clean-room multi-disagreement bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    metadata = _clean_metadata(spec)
    metadata["source_repo"] = "Xenova/all-MiniLM-L6-v2"
    metadata["source_file"] = None
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.verify_source_metadata(
            spec, resolved_sha256="0" * 64, resolved_bytes=spec.pinned_bytes + 1,
            metadata=metadata, external=_clean_external(),
        )
    reasons = ei.value.reasons
    assert any(r.startswith("source_repo") for r in reasons)
    assert any(r.startswith("source_file") for r in reasons)
    assert any("resolved sha256" in r for r in reasons)
    assert any("resolved size" in r for r in reasons)
    assert len(reasons) >= 4


# ---------------------------------------------------------------------------
# Recursive tensor discovery — every TensorProto/SparseTensorProto location
# ---------------------------------------------------------------------------

def _ext_tensor(onnx, numpy_helper, name, arr, location):
    t = numpy_helper.from_array(arr, name=name)
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    t.external_data.extend([onnx.StringStringEntryProto(key="location", value=location)])
    return t


def test_sparse_initializer_values_tensor_is_discovered(tmp_path):
    """A `SparseTensorProto`'s `values` (and `indices`) are themselves `TensorProto` and
    either can independently declare external data — a scan of `graph.initializer` alone
    never looks inside `graph.sparse_initializer` at all."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    (tmp_path / "sparse.bin").write_bytes(arr.tobytes())
    values = _ext_tensor(onnx, numpy_helper, "sv", arr, "sparse.bin")
    indices = numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), name="si")
    sparse = onnx.SparseTensorProto(values=values, indices=indices, dims=[3])
    graph = helper.make_graph([], "g", [], [], sparse_initializer=[sparse])
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(helper.make_model(graph).SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert [f["location"] for f in rec["files"]] == ["sparse.bin"]


def test_nested_if_subgraph_initializer_is_discovered(tmp_path):
    """An `If` node's `then_branch`/`else_branch` are full subgraphs with their own
    initializers, reached only by recursing into node attributes that carry a `g`/
    `graphs`."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    arr = np.array([9.0], dtype=np.float32)
    (tmp_path / "if_then.bin").write_bytes(arr.tobytes())
    w = _ext_tensor(onnx, numpy_helper, "W_then", arr, "if_then.bin")
    then_graph = helper.make_graph([], "then", [], [], initializer=[w])
    else_graph = helper.make_graph([], "else", [], [])
    if_node = helper.make_node("If", ["cond"], ["out"], then_branch=then_graph,
                               else_branch=else_graph)
    outer = helper.make_graph(
        [if_node], "outer",
        [helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1])],
    )
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(helper.make_model(outer).SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert [f["location"] for f in rec["files"]] == ["if_then.bin"]


def test_function_body_tensor_is_discovered(tmp_path):
    """A local `FunctionProto`'s own `Constant` node can carry an external tensor;
    functions are not part of `model.graph` at all and are reached only via
    `model.functions`."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    arr = np.array([5.0], dtype=np.float32)
    (tmp_path / "func_body.bin").write_bytes(arr.tobytes())
    const_t = _ext_tensor(onnx, numpy_helper, "body_const", arr, "func_body.bin")
    const_node = helper.make_node("Constant", [], ["c"], value=const_t)
    func = helper.make_function(
        domain="custom", fname="MyFunc", inputs=[], outputs=["c"], nodes=[const_node],
        opset_imports=[helper.make_opsetid("", 18)],
    )
    outer = helper.make_graph([], "outer", [], [])
    model = helper.make_model(
        outer, functions=[func],
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom", 1)],
    )
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert [f["location"] for f in rec["files"]] == ["func_body.bin"]


def test_function_attribute_default_tensor_is_discovered(tmp_path):
    """`attribute_proto` entries are a function's attribute DEFAULT VALUES — full
    `AttributeProto` messages that can themselves carry a default tensor. The plain
    `attribute` field (just names) carries nothing to scan; only `attribute_proto` does."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    arr = np.array([7.0], dtype=np.float32)
    (tmp_path / "func_attr_default.bin").write_bytes(arr.tobytes())
    default_t = _ext_tensor(onnx, numpy_helper, "attr_default", arr, "func_attr_default.bin")
    attr_proto = helper.make_attribute("myattr", default_t)
    func = helper.make_function(
        domain="custom", fname="MyFunc", inputs=[], outputs=[], nodes=[],
        opset_imports=[helper.make_opsetid("", 18)], attribute_protos=[attr_proto],
    )
    outer = helper.make_graph([], "outer", [], [])
    model = helper.make_model(
        outer, functions=[func],
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom", 1)],
    )
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert [f["location"] for f in rec["files"]] == ["func_attr_default.bin"]


def test_training_info_initialization_and_algorithm_graphs_are_discovered(tmp_path):
    """`training_info` initialization/algorithm graphs carry their own initializers and are
    not part of the main inference graph at all — a scan of `model.graph` alone misses them
    entirely."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    arr_init = np.array([1.0], dtype=np.float32)
    (tmp_path / "train_init.bin").write_bytes(arr_init.tobytes())
    t_init = _ext_tensor(onnx, numpy_helper, "t_init", arr_init, "train_init.bin")
    init_graph = helper.make_graph([], "init", [], [], initializer=[t_init])

    arr_algo = np.array([2.0], dtype=np.float32)
    (tmp_path / "train_algo.bin").write_bytes(arr_algo.tobytes())
    t_algo = _ext_tensor(onnx, numpy_helper, "t_algo", arr_algo, "train_algo.bin")
    algo_graph = helper.make_graph([], "algo", [], [], initializer=[t_algo])

    ti = helper.make_training_info(
        algorithm=algo_graph, algorithm_bindings=[],
        initialization=init_graph, initialization_bindings=[],
    )
    outer = helper.make_graph([], "outer", [], [])
    model = helper.make_model(outer)
    model.training_info.extend([ti])
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(model.SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is True
    assert sorted(f["location"] for f in rec["files"]) == ["train_algo.bin", "train_init.bin"]


def test_malformed_external_tensor_with_no_location_refuses_the_whole_scan(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="bad")
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    graph = helper.make_graph([], "g", [], [], initializer=[t])
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(helper.make_model(graph).SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is False
    assert "no 'location' entry" in rec["reason"]


def test_malformed_external_tensor_with_duplicate_location_refuses_even_if_they_agree(
        tmp_path):
    """Two identical `location` entries are refused exactly like two conflicting ones — a
    graph whose external-data bookkeeping names the same field twice has already shown it
    cannot be trusted, agreement or not."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="bad")
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    t.external_data.extend([
        onnx.StringStringEntryProto(key="location", value="a.bin"),
        onnx.StringStringEntryProto(key="location", value="a.bin"),
    ])
    graph = helper.make_graph([], "g", [], [], initializer=[t])
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(helper.make_model(graph).SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is False
    assert "duplicate or conflicting" in rec["reason"]


def test_malformed_external_tensor_with_conflicting_locations_refuses(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="bad")
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    t.external_data.extend([
        onnx.StringStringEntryProto(key="location", value="a.bin"),
        onnx.StringStringEntryProto(key="location", value="b.bin"),
    ])
    graph = helper.make_graph([], "g", [], [], initializer=[t])
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(helper.make_model(graph).SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is False
    assert "duplicate or conflicting" in rec["reason"]


def test_malformed_external_tensor_with_empty_location_refuses(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="bad")
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    t.external_data.extend([onnx.StringStringEntryProto(key="location", value="   ")])
    graph = helper.make_graph([], "g", [], [], initializer=[t])
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(helper.make_model(graph).SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is False
    assert "empty/whitespace" in rec["reason"]


def test_a_malformed_tensor_deep_in_a_nested_graph_refuses_the_whole_scan(tmp_path):
    """The malformed tensor is inside an `If` branch, not at the top level of `model.graph`
    — proving the refusal propagates from wherever in the recursive walk it is found, rather
    than only being checked at the outermost graph."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t_bad = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="bad2")
    t_bad.ClearField("raw_data")
    t_bad.data_location = onnx.TensorProto.EXTERNAL  # no location entries at all
    then_graph = helper.make_graph([], "then", [], [], initializer=[t_bad])
    else_graph = helper.make_graph([], "else", [], [])
    if_node = helper.make_node("If", ["cond"], ["out"], then_branch=then_graph,
                               else_branch=else_graph)
    outer = helper.make_graph(
        [if_node], "outer",
        [helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1])],
    )
    mpath = tmp_path / "m.onnx"
    mpath.write_bytes(helper.make_model(outer).SerializeToString())

    rec = rm.external_data_provenance(mpath)
    assert rec["scanned"] is False
    assert "bad2" in rec["reason"]


# ---------------------------------------------------------------------------
# resolve_model — MiniLM-shaped end-to-end (issue #78)
# ---------------------------------------------------------------------------

def _minimal_onnx_bytes() -> bytes:
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    t = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), name="W")
    model = helper.make_model(helper.make_graph([], "g", [], [], initializer=[t]))
    return model.SerializeToString()


def _stage_minilm(tmp_path, spec, payload, monkeypatch, *, results_dir=None,
                  sidecar_overrides=None):
    """Puts *payload* where `resolve_model`'s repo-cache resolver looks for *spec*, and
    writes a matching sidecar under *results_dir* (default *tmp_path*) so
    `recorded_source_metadata`/`recorded_sha256` find it."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / spec.cache_filename).write_bytes(payload)
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(models_dir))
    base = Path(results_dir) if results_dir else tmp_path
    meta = {
        "source_repo": spec.source_repo,
        "source_revision": spec.source_revision,
        "source_file": spec.source_file,
        "onnx_bytes": spec.pinned_bytes,
        "onnx_sha256": spec.pinned_sha256,
    }
    if sidecar_overrides:
        meta.update(sidecar_overrides)
    side = base / spec.recorded_provenance
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(meta), encoding="utf-8")
    return base


def test_resolve_model_verifies_a_minilm_shaped_spec_end_to_end(tmp_path, monkeypatch):
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm(tmp_path, spec, payload, monkeypatch)
    rec = rm.resolve_model(spec, results_dir=tmp_path)
    assert rec["provenance_ok"] is True
    assert rec["provenance_state"] == "verified"
    assert rec["capability"] == "provenance_only"
    assert rec["sha256"] == spec.pinned_sha256
    assert rec["bytes"] == spec.pinned_bytes


def test_resolve_model_public_path_never_carries_the_real_cache_directory(tmp_path,
                                                                          monkeypatch):
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm(tmp_path, spec, payload, monkeypatch)
    rec = rm.resolve_model(spec, results_dir=tmp_path)
    assert str(tmp_path) not in rec["path"]
    assert rec["_runtime_path"] == str(tmp_path / "models" / spec.cache_filename)


def test_resolve_model_refuses_a_cached_substitute_with_wrong_hash(tmp_path, monkeypatch):
    """The exact #78 defect: a resolved file whose bytes hash differently from the pin must
    refuse, never silently pass because the spec's own pin fields happen to be present."""
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm(tmp_path, spec, payload, monkeypatch)
    substitute = _minimal_onnx_bytes() + b"\x00" * 8  # different bytes, different hash
    (tmp_path / "models" / spec.cache_filename).write_bytes(substitute)
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.resolve_model(spec, results_dir=tmp_path)
    assert any("resolved sha256" in r or "resolved size" in r for r in ei.value.reasons)


def test_resolve_model_refuses_when_the_sidecar_is_entirely_missing(tmp_path, monkeypatch):
    """No sidecar means every recorded metadata field reads as missing — refusal, not a pin
    with nothing to verify it silently passing."""
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / spec.cache_filename).write_bytes(payload)
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(models_dir))
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.resolve_model(spec, results_dir=tmp_path)
    assert any("missing" in r for r in ei.value.reasons)


def test_resolve_model_refuses_a_partially_populated_sidecar(tmp_path, monkeypatch):
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm(tmp_path, spec, payload, monkeypatch, sidecar_overrides={"source_file": None})
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.resolve_model(spec, results_dir=tmp_path)
    assert any(r.startswith("source_file") for r in ei.value.reasons)


def test_resolve_model_refuses_a_drifted_source_revision(tmp_path, monkeypatch):
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm(tmp_path, spec, payload, monkeypatch,
                  sidecar_overrides={"source_revision": "0" * 40})
    with pytest.raises(rm.ModelProvenanceRefused) as ei:
        rm.resolve_model(spec, results_dir=tmp_path)
    assert any("source_revision" in r and "source drift" in r for r in ei.value.reasons)


def test_resolve_model_reports_provenance_only_honestly_not_promoted_to_full(tmp_path,
                                                                             monkeypatch):
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm(tmp_path, spec, payload, monkeypatch)
    rec = rm.resolve_model(spec, results_dir=tmp_path)
    assert rec["capability"] == "provenance_only"


def test_resolve_model_raises_model_unavailable_not_verified_when_bytes_are_absent(tmp_path,
                                                                                   monkeypatch):
    """Missing bytes must produce an unavailable record, never a "pinned and verified" one —
    a pinned spec existing in code is not evidence bytes were resolved."""
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(empty))
    with pytest.raises(rm.ModelUnavailable):
        rm.resolve_model(spec, results_dir=tmp_path)


def test_resolve_model_unavailable_message_never_leaks_the_real_cache_path(tmp_path,
                                                                           monkeypatch):
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(empty))
    with pytest.raises(rm.ModelUnavailable) as ei:
        rm.resolve_model(spec, results_dir=tmp_path)
    assert str(tmp_path) not in str(ei.value)


# ---------------------------------------------------------------------------
# cases_for / DIAGNOSTIC_PLANS — no fallback for a model with no builder
# ---------------------------------------------------------------------------

def test_cases_for_raises_on_a_totally_unknown_key():
    unknown = rm.ModelSpec(key="not-a-real-model", family="f", resolver="repo-cache",
                           cache_filename="x")
    with pytest.raises(ValueError):
        rm.cases_for(unknown)


def test_cases_for_phi35_returns_phi35_shaped_cases():
    cases = rm.cases_for(rm.PHI35, prefill_m=[1, 8], decode_past=[128])
    assert cases and all(c.model_key == rm.PHI35.key for c in cases)


def test_cases_for_mobilenet_returns_mobilenet_shaped_cases():
    cases = rm.cases_for(rm.MOBILENETV2, batch=[1, 4])
    assert cases and all(c.model_key == rm.MOBILENETV2.key for c in cases)


def test_cases_for_minilm_returns_none_never_another_models_cases():
    """`None`, never MobileNetV2's batch plan and never an empty list that could be misread
    as "zero cases ran" rather than "no builder exists"."""
    assert rm.cases_for(rm.MINILM) is None


def test_diagnostic_plans_has_no_entry_for_a_provenance_only_model():
    assert rm.MINILM.key not in rm.DIAGNOSTIC_PLANS
    assert rm.PHI35.key in rm.DIAGNOSTIC_PLANS
    assert rm.MOBILENETV2.key in rm.DIAGNOSTIC_PLANS


# ---------------------------------------------------------------------------
# The driver's provenance gates (issue #78, blocker #4) — every entry point
# ---------------------------------------------------------------------------

def _stage_minilm_for_driver(tmp_path, spec, payload, monkeypatch):
    """Like `_stage_minilm`, but also relocates `real_model`'s own `_BENCH` global so that
    `rm.resolve_model(spec)` — called with NO `results_dir`, exactly how
    `diagnose_worker`/`run_diagnostics` call it — reads the fabricated sidecar here rather
    than the real, committed `bench/results/rust-model-runner/all-MiniLM-L6-v2.json`."""
    _stage_minilm(tmp_path, spec, payload, monkeypatch, results_dir=tmp_path / "results")
    monkeypatch.setattr(rm, "_BENCH", tmp_path)


def test_diagnose_worker_refuses_a_provenance_only_model_before_importing_onnxruntime(
        tmp_path, monkeypatch):
    """A provenance-only model is resolved and verified above the capability gate — never
    silently accepted as "pinned and verified" if its bytes are actually missing — and then
    refused for feeding, named, BEFORE `onnxruntime` is ever imported into the process."""
    mod = _probe_module()
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm_for_driver(tmp_path, spec, payload, monkeypatch)
    monkeypatch.setitem(mod.rm.MODELS, spec.key, spec)
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)

    out = tmp_path / "worker_out.json"
    rc = mod.diagnose_worker([
        "--worker-diagnose", "--model", spec.key, "--arm", "vulkan_untiled",
        "--phase", "batch", "--m", "1", "--out", str(out),
    ])
    assert rc == 2
    assert "onnxruntime" not in sys.modules
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["provenance_ok"] is True
    assert rec["provenance_state"] == "verified"
    assert rec["capability"] == "provenance_only"


def test_diagnose_worker_refusal_record_is_written_through_the_public_path_screen(
        tmp_path, monkeypatch):
    """The refusal record still goes through `write_public_json`, not a raw
    `Path.write_text` — a leak screen that only applies to the success path is not a leak
    screen."""
    mod = _probe_module()
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm_for_driver(tmp_path, spec, payload, monkeypatch)
    monkeypatch.setitem(mod.rm.MODELS, spec.key, spec)

    out = tmp_path / "worker_out.json"
    mod.diagnose_worker([
        "--worker-diagnose", "--model", spec.key, "--arm", "vulkan_untiled",
        "--phase", "batch", "--m", "1", "--out", str(out),
    ])
    text = out.read_text(encoding="utf-8")
    assert mod.pp._find_leaks(text) == []


def test_run_diagnostics_records_provenance_for_a_model_with_no_diagnostic_plan(
        tmp_path, monkeypatch):
    """Issue #78, blocker #4: EVERY model key is resolved and verified, unconditionally,
    even one with no entry in `DIAGNOSTIC_PLANS` — "no cases" and "unchecked" must never be
    the same thing in the report."""
    mod = _probe_module()
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm_for_driver(tmp_path, spec, payload, monkeypatch)
    monkeypatch.setitem(mod.rm.MODELS, spec.key, spec)
    monkeypatch.setattr(mod.rm, "DIAGNOSTIC_PLANS", {})  # no plan spawns no subprocess

    class _Args:
        scratch = str(tmp_path / "scratch")
        diag_iters = 1

    class _Device:
        index = 0
        name = "test-device"

    out = mod.run_diagnostics(_Args(), [spec.key], _Device())
    assert out["provenance"][spec.key]["resolved"] is True
    assert out["provenance"][spec.key]["provenance_ok"] is True
    assert out["provenance"][spec.key]["provenance_state"] == "verified"
    assert out["runs"] == []  # no plan spawns nothing, but never simply absent above


def test_run_diagnostics_records_refusal_not_a_false_verified_when_bytes_are_absent(
        tmp_path, monkeypatch):
    """Missing bytes must produce an unavailable record in `provenance`, never
    ``provenance_ok=True``."""
    mod = _probe_module()
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    monkeypatch.setitem(mod.rm.MODELS, spec.key, spec)
    monkeypatch.setattr(mod.rm, "DIAGNOSTIC_PLANS", {})
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(mod.rm.REPO_CACHE_ENV, str(empty))

    class _Args:
        scratch = str(tmp_path / "scratch")
        diag_iters = 1

    class _Device:
        index = 0
        name = "test-device"

    out = mod.run_diagnostics(_Args(), [spec.key], _Device())
    assert out["provenance"][spec.key]["resolved"] is False
    assert out["provenance"][spec.key]["provenance_ok"] is None


# ---------------------------------------------------------------------------
# No network — provenance is bytes-on-disk plus static metadata, never a fetch
# ---------------------------------------------------------------------------

def test_real_model_module_never_imports_a_network_stack():
    src = (_BENCH / "real_model.py").read_text(encoding="utf-8")
    for forbidden in ("import socket", "import urllib", "import http.client",
                     "import requests"):
        assert forbidden not in src


def test_resolve_model_never_touches_the_network_even_if_asked_to(tmp_path, monkeypatch):
    """Breaks `socket.socket` itself rather than only checking the source text, so a network
    call reached through some indirect path would still be caught, not only a literal
    `import socket` in this file."""
    import socket

    def _refuse(*a, **kw):
        raise AssertionError("resolve_model must never open a network socket")

    monkeypatch.setattr(socket, "socket", _refuse)
    payload = _minimal_onnx_bytes()
    spec = _pinned_spec_for_payload(payload)
    _stage_minilm(tmp_path, spec, payload, monkeypatch)
    rec = rm.resolve_model(spec, results_dir=tmp_path)
    assert rec["provenance_ok"] is True


def test_verify_source_metadata_never_touches_the_network(monkeypatch):
    import socket

    def _refuse(*a, **kw):
        raise AssertionError("verify_source_metadata must never open a network socket")

    monkeypatch.setattr(socket, "socket", _refuse)
    payload = b"clean-room no-network bytes" * 10
    spec = _pinned_spec_for_payload(payload)
    result = rm.verify_source_metadata(
        spec, resolved_sha256=spec.pinned_sha256, resolved_bytes=spec.pinned_bytes,
        metadata=_clean_metadata(spec), external=_clean_external(),
    )
    assert result["provenance_ok"] is True


# ---------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------

def test_decode_cases_have_a_non_empty_kv_cache():
    """The single largest hole #56 names: every earlier measurement ran at past == 0."""
    cases = rm.phi35_cases([1, 2], [0, 128, 512])
    decode = [c for c in cases if c.phase == "decode"]
    assert decode, "there must be decode cases at all"
    assert any(c.past > 0 for c in decode), "at least one decode case must carry a real cache"


def test_prefill_feeds_have_an_empty_cache_and_decode_feeds_do_not():
    pre = rm.phi35_feeds(rm.Case(rm.PHI35.key, "prefill", 4, 0, tokens=4), np)
    dec = rm.phi35_feeds(rm.Case(rm.PHI35.key, "decode", 1, 7, tokens=1), np)
    assert pre["past_key_values.0.key"].shape == (1, 32, 0, 96)
    assert dec["past_key_values.0.key"].shape == (1, 32, 7, 96)
    assert dec["past_key_values.31.value"].shape == (1, 32, 7, 96)


def test_attention_mask_covers_past_plus_new_tokens():
    """GQA derives `seqlens_k` from the mask; a mask of length `m` at past>0 is rejected by ORT
    with `seqlens_k[0] = N is out of range`, which is exactly how the modelrunner's generic
    input generator fails on this model."""
    dec = rm.phi35_feeds(rm.Case(rm.PHI35.key, "decode", 1, 31, tokens=1), np)
    assert dec["attention_mask"].shape == (1, 32)
    pre = rm.phi35_feeds(rm.Case(rm.PHI35.key, "prefill", 8, 0, tokens=8), np)
    assert pre["attention_mask"].shape == (1, 8)


def test_feeds_are_byte_identical_across_calls():
    """Two arms must be fed the same bytes or the comparison is between two different inputs."""
    case = rm.Case(rm.PHI35.key, "decode", 1, 5, tokens=1)
    first = rm.feeds_digest(rm.phi35_feeds(case, np))
    second = rm.feeds_digest(rm.phi35_feeds(case, np))
    assert first == second


def test_feeds_digest_notices_a_renamed_key():
    """Hashing values alone would call a feed dict with the right values under the wrong keys
    identical to the correct one."""
    a = {"x": np.ones(4, dtype=np.float32)}
    b = {"y": np.ones(4, dtype=np.float32)}
    assert rm.feeds_digest(a) != rm.feeds_digest(b)


def test_feeds_digest_notices_a_changed_dtype_at_equal_values():
    a = {"x": np.ones(4, dtype=np.float32)}
    b = {"x": np.ones(4, dtype=np.float16)}
    assert rm.feeds_digest(a) != rm.feeds_digest(b)


def test_kv_round_trip_is_the_measured_slope_not_a_guess():
    assert rm.PHI35_BYTES_PER_PAST_TOKEN == 393216
    kv = rm.kv_round_trip_bytes(rm.Case(rm.PHI35.key, "decode", 1, 1024, tokens=1))
    assert kv["past_upload_bytes"] == 1024 * 393216
    assert kv["present_readback_bytes"] == 1025 * 393216


def test_kv_round_trip_is_none_for_a_model_without_a_cache():
    assert rm.kv_round_trip_bytes(rm.Case(rm.MOBILENETV2.key, "batch", 8, 0)) is None


# ---------------------------------------------------------------------------
# Arms and ordering
# ---------------------------------------------------------------------------

def test_the_two_vulkan_arms_differ_by_exactly_one_variable():
    tiled = dict(rm.VULKAN_TILED.env)
    untiled = dict(rm.VULKAN_UNTILED.env)
    assert set(tiled) == set(untiled) == {rm.ROWS_ENV}
    assert tiled[rm.ROWS_ENV] != untiled[rm.ROWS_ENV]
    assert untiled[rm.ROWS_ENV] == "1", "the kill switch pins the tile back to one row"


def test_arm_order_alternates():
    """A fixed order produced a spurious 0.905x at M=1 on identical SPIR-V (§25.4)."""
    a = rm.arm_order(rm.ARMS, 0)
    b = rm.arm_order(rm.ARMS, 1)
    assert [x.name for x in a] == list(reversed([x.name for x in b]))


def test_arm_env_is_restorable():
    environ = {rm.ROWS_ENV: "9"}
    prev = rm.VULKAN_UNTILED.apply_env(environ)
    assert environ[rm.ROWS_ENV] == "1"
    assert prev[rm.ROWS_ENV] == "9"


def test_m1_is_the_null_control_and_m4_is_not():
    assert rm.is_null_control(rm.Case(rm.PHI35.key, "prefill", 1, 0, tokens=1))
    assert rm.is_null_control(rm.Case(rm.PHI35.key, "decode", 1, 512, tokens=1))
    assert not rm.is_null_control(rm.Case(rm.PHI35.key, "prefill", 4, 0, tokens=4))


# ---------------------------------------------------------------------------
# Statistics and throughput
# ---------------------------------------------------------------------------

def test_latency_stats_reports_a_distribution_not_a_number():
    st = rm.latency_stats([10, 11, 12, 13, 100])
    for key in ("median_ms", "min_ms", "max_ms", "p05_ms", "p95_ms", "mad_ms", "rsd"):
        assert key in st
    assert st["median_ms"] == 12
    assert st["max_ms"] == 100, "an outlier must remain visible, not be smoothed away"


def test_latency_stats_of_nothing_is_not_zero():
    """An empty sample is `n = 0`, not a median of 0 — a fabricated fast number."""
    st = rm.latency_stats([])
    assert st == {"n": 0}


def test_prefill_throughput_divides_by_tokens_consumed():
    case = rm.Case(rm.PHI35.key, "prefill", 8, 0, tokens=8)
    tp = rm.throughput(case, 100.0)
    assert tp["value"] == pytest.approx(80.0)
    assert tp["unit"] == "tokens/s"


def test_decode_throughput_does_not_divide_by_the_cache_length():
    """Dividing a decode step by `past` manufactures a throughput that *grows* as the model
    slows down — the number getting better as the thing gets worse."""
    short = rm.Case(rm.PHI35.key, "decode", 1, 128, tokens=1)
    long = rm.Case(rm.PHI35.key, "decode", 1, 2048, tokens=1)
    assert rm.throughput(short, 50.0)["value"] == pytest.approx(20.0)
    assert rm.throughput(long, 50.0)["value"] == pytest.approx(20.0)


def test_batch_throughput_is_images_not_tokens():
    case = rm.Case(rm.MOBILENETV2.key, "batch", 16, 0, tokens=None, unit="images")
    tp = rm.throughput(case, 160.0)
    assert tp["unit"] == "images/s"
    assert tp["value"] == pytest.approx(100.0)


def test_throughput_of_a_zero_latency_is_none_not_infinity():
    assert rm.throughput(rm.Case(rm.PHI35.key, "prefill", 1, 0, tokens=1), 0.0) is None


def test_paired_ratio_survives_one_displaced_repeat():
    """The median of per-repeat ratios, not the ratio of pooled medians: a repeat where one arm
    was displaced by a co-tenant must not move the estimate."""
    a = [10.0, 10.0, 10.0, 90.0]
    b = [5.0, 5.0, 5.0, 45.0]
    assert rm.paired_ratios(a, b)["median"] == pytest.approx(2.0)


def test_paired_ratio_of_nothing_is_n_zero():
    assert rm.paired_ratios([], [])["n"] == 0


def test_noise_floor_refuses_to_call_a_sub_floor_ratio_a_speedup():
    floor = {"n": 3, "median": 1.0, "min": 0.94, "max": 1.15}
    assert rm.exceeds_noise_floor(1.05, floor) is False
    assert rm.exceeds_noise_floor(1.62, floor) is True


def test_missing_noise_floor_is_none_not_false():
    """"No control to read this against" is a different state from "not significant"."""
    assert rm.exceeds_noise_floor(1.62, {}) is None
    assert rm.exceeds_noise_floor(1.62, {"n": 0}) is None


# ---------------------------------------------------------------------------
# Equivalence verdicts
# ---------------------------------------------------------------------------

def _logits(argmax_at: int, n: int = 64, scale: float = 1.0):
    x = np.linspace(0.0, 0.1, n).astype(np.float32) * scale
    x[argmax_at] = 5.0 * scale
    return x.reshape(1, 1, n)


def test_identical_logits_match():
    v = rm.classify_logits(_logits(3), _logits(3), np)
    assert v["verdict"] == rm.MATCH


def test_moved_argmax_is_divergent_even_within_tolerance():
    a, b = _logits(3), _logits(3)
    b[0, 0, 3] = 0.0
    b[0, 0, 4] = 5.0
    v = rm.classify_logits(a, b, np)
    assert v["verdict"] == rm.DIVERGENT


def test_all_zero_reference_is_divergent_not_a_perfect_match():
    """The `argmax 0` defect: 161 nodes dispatched, `compute_failures: 0`, and both tensors
    all-zero, which any elementwise comparison calls a perfect agreement."""
    z = np.zeros((1, 1, 64), dtype=np.float32)
    v = rm.classify_logits(z, z, np)
    assert v["verdict"] == rm.DIVERGENT
    assert v["reference_all_zero"] is True


def test_shape_mismatch_is_divergent_not_an_exception():
    v = rm.classify_logits(np.zeros((1, 1, 8)), np.zeros((1, 1, 9)), np)
    assert v["verdict"] == rm.DIVERGENT
    assert v["reason"] == "shape mismatch"


def test_logit_budget_is_a_fraction_of_scale_not_a_constant():
    """A constant absolute budget silently tightens as the logit scale grows with the cache:
    0.5 is 3.8% of scale at past=0 and 2.1% at past=1024, so the same kernel gets a stricter
    exam for a longer sequence. The budget must track the reference's own magnitude."""
    small = _logits(3, scale=1.0)
    large = _logits(3, scale=10.0)
    v_small = rm.classify_logits(small, small, np)
    v_large = rm.classify_logits(large, large, np)
    assert v_large["abs_budget"] == pytest.approx(10.0 * v_small["abs_budget"])


def test_a_logit_error_above_the_scale_fraction_is_divergent():
    ref = _logits(3, scale=1.0)
    cand = ref.copy()
    cand[0, 0, 10] += 1.01 * rm.PHI35_LOGIT_SCALE_FRACTION * float(np.abs(ref).max())
    v = rm.classify_logits(cand, ref, np)
    assert v["verdict"] == rm.DIVERGENT
    assert v["max_abs"] > v["abs_budget"]


def test_a_logit_error_below_the_scale_fraction_is_a_match():
    ref = _logits(3, scale=1.0)
    cand = ref.copy()
    cand[0, 0, 63] += 0.5 * rm.PHI35_LOGIT_SCALE_FRACTION * float(np.abs(ref).max())
    v = rm.classify_logits(cand, ref, np)
    assert v["verdict"] == rm.MATCH


def test_logits_are_gated_on_the_distribution_they_induce():
    """argmax and top-10 can both agree while the sampler's distribution moves — scale the whole
    vector and the ranking is untouched but the probabilities are not. A logit-space bound alone
    would call that MATCH, and what a decoder emits is a distribution, not a ranking."""
    ref = _logits(3, scale=1.0)
    cand = (ref * 1.5).astype(np.float32)
    v = rm.classify_logits(cand, ref, np, scale_fraction=10.0)  # absolute clause disabled
    assert v["argmax_candidate"] == v["argmax_reference"]
    assert v["topk_overlap"] == v["top_k"]
    assert v["max_prob_delta"] > rm.PHI35_MAX_PROB_DELTA
    assert v["verdict"] == rm.DIVERGENT


def test_one_element_a_third_above_the_floor_is_not_a_defect():
    """MEASURED: at past=128 the Vulkan arms put exactly one element of 396,288 at 1.33x the
    fp16 floor. Failing the whole run on that is an instrument that cannot distinguish an
    accumulation tail from a wrong cache; passing anything at all is a fudge. The band does
    both — see the gross-ceiling and marginal-fraction controls below."""
    rng = np.random.default_rng(5)
    ref = rng.standard_normal((1, 32, 129, 96)).astype(np.float16).astype(np.float64)
    cand = ref.copy()
    idx = np.unravel_index(np.argmax(np.abs(ref)), ref.shape)
    floor = rm.KV_ULP_BUDGET * rm.FP16_EPS * np.abs(ref).max() + rm.KV_REL_TOL * abs(ref[idx])
    cand[idx] = ref[idx] + 1.33 * floor
    v = rm.classify_activation(cand, ref, np)
    assert v["elements_outside_tolerance"] == 1
    assert v["elements_gross"] == 0
    assert v["verdict"] == rm.MATCH


def test_a_gross_element_fails_however_few_there_are():
    """The ceiling is what keeps the marginal band from being a fudge: one element far out is
    DIVERGENT even though one element in 400,000 is a vanishing fraction."""
    rng = np.random.default_rng(6)
    ref = rng.standard_normal((1, 32, 129, 96)).astype(np.float16).astype(np.float64)
    cand = ref.copy()
    idx = np.unravel_index(np.argmax(np.abs(ref)), ref.shape)
    floor = rm.KV_ULP_BUDGET * rm.FP16_EPS * np.abs(ref).max() + rm.KV_REL_TOL * abs(ref[idx])
    cand[idx] = ref[idx] + (rm.KV_GROSS_MULTIPLE + 1.0) * floor
    v = rm.classify_activation(cand, ref, np)
    assert v["elements_gross"] == 1
    assert v["marginal_fraction_observed"] < rm.KV_MARGINAL_FRACTION
    assert v["verdict"] == rm.DIVERGENT


def test_a_systematic_small_error_fails_on_its_population():
    """A bias that puts every element just above the floor is the failure mode a per-element
    ceiling alone would miss: no single element is gross, and the tensor is still wrong."""
    rng = np.random.default_rng(7)
    ref = rng.standard_normal((1, 32, 129, 96)).astype(np.float16).astype(np.float64)
    floor = rm.KV_ULP_BUDGET * rm.FP16_EPS * np.abs(ref).max() + rm.KV_REL_TOL * np.abs(ref)
    cand = ref + 1.5 * floor
    v = rm.classify_activation(cand, ref, np)
    assert v["elements_gross"] == 0
    assert v["marginal_fraction_observed"] > rm.KV_MARGINAL_FRACTION
    assert v["verdict"] == rm.DIVERGENT


def test_bitwise_control_calls_identical_outputs_identical():
    a = [np.arange(12, dtype=np.float16).reshape(3, 4)]
    assert rm.bitwise_identical(a, [x.copy() for x in a], np)["identical"] is True


def test_bitwise_control_catches_a_one_bit_difference():
    """At M=1 the tiled and untiled arms are claimed to bind the same SPIR-V. If that claim is
    false the outputs will differ somewhere, and a *tolerance* would hide it — this is the one
    comparison in the harness that must be exact."""
    a = [np.arange(12, dtype=np.float16).reshape(3, 4)]
    b = [a[0].copy()]
    b[0][1, 1] = np.nextafter(b[0][1, 1], np.float16(1000.0))
    v = rm.bitwise_identical(a, b, np)
    assert v["identical"] is False
    assert v["first_difference"] == "output[0]"


def test_bitwise_control_does_not_crash_on_mismatched_counts():
    v = rm.bitwise_identical([np.zeros(4)], [np.zeros(4), np.zeros(4)], np)
    assert v["identical"] is False
    assert v["first_difference"] == "output count"


def test_classify_outputs_catches_a_wrong_kv_cache_behind_correct_logits():
    """A decode step whose logits agree but whose `present` cache is wrong produces a correct
    first token and a wrong sequence. Comparing output 0 alone would call it MATCH."""
    case = rm.Case(rm.PHI35.key, "decode", 1, 4, tokens=1)
    logits = _logits(3)
    rng = np.random.default_rng(11)
    good_kv = rng.standard_normal((1, 32, 5, 96)).astype(np.float16)
    bad_kv = good_kv.copy()
    bad_kv[0, 0, 0, 0] = np.float16(7.0)
    ok = rm.classify_outputs(case, [logits, good_kv], [logits, good_kv], np)
    bad = rm.classify_outputs(case, [logits, bad_kv], [logits, good_kv], np)
    assert ok["verdict"] == rm.MATCH
    assert bad["verdict"] == rm.DIVERGENT
    assert bad["secondary_divergent"] == 1


def test_activation_gate_is_elementwise_not_an_aggregate_or():
    """An aggregate `max_abs <= floor OR max_rel_signal <= tol` has a hole a planted error walks
    through: put the error on an element *below* the signal threshold and the relative clause
    excludes it, so the OR passes on a tensor with a 7.0 error in it. That is not hypothetical —
    it is what the first version of this gate did to `test_classify_outputs_catches_a_wrong_kv
    _cache_behind_correct_logits`."""
    rng = np.random.default_rng(11)
    ref = rng.standard_normal((1, 32, 5, 96)).astype(np.float16).astype(np.float64)
    idx = np.unravel_index(np.argmin(np.abs(ref)), ref.shape)  # a below-threshold element
    cand = ref.copy()
    cand[idx] = 7.0
    v = rm.classify_activation(cand, ref, np)
    assert v["verdict"] == rm.DIVERGENT
    assert v["elements_outside_tolerance"] == 1
    assert v["max_rel_signal"] == 0.0, "the excluded element proves the aggregate OR would pass"


def test_one_fp16_ulp_of_scale_is_not_a_divergence():
    """The gate this replaced called every Vulkan arm DIVERGENT at `max_rel = 2.73` on a tensor
    whose largest absolute error was one fp16 ULP — a cancellation meter read as an accuracy
    meter. That failure is the reason `classify_activation` exists."""
    rng = np.random.default_rng(5)
    ref = (rng.standard_normal((1, 32, 4, 96)) * 8.0).astype(np.float16)
    cand = ref.astype(np.float64).copy()
    # One ULP at magnitude 16 is 0.015625 — exactly the residual the real run produced.
    cand[0, 0, 0, 0] += 0.015625
    # ...and a near-zero element perturbed by the same absolute amount, which is what drives the
    # unrestricted relative figure to the hundreds.
    idx = np.unravel_index(np.argmin(np.abs(ref)), ref.shape)
    cand[idx] += 0.015625
    v = rm.classify_activation(cand, ref, np)
    assert v["verdict"] == rm.MATCH
    assert v["max_rel_all"] > 1.0, "the unrestricted relative figure is still huge — and unused"
    assert v["max_rel_signal"] < 0.05


def test_activation_gate_scales_with_the_tensors_own_magnitude():
    """A constant absolute tolerance is absurdly tight at magnitude 30 and absurdly loose at
    magnitude 1e-4. The budget is ULPs of the reference's own scale, so both are caught."""
    small = np.full((4, 4), 1e-3, dtype=np.float64)
    cand_small = small.copy()
    cand_small[0, 0] = 1e-1
    assert rm.classify_activation(cand_small, small, np)["verdict"] == rm.DIVERGENT
    big = np.full((4, 4), 30.0, dtype=np.float64)
    cand_big = big.copy()
    cand_big[0, 0] = 30.0 + 0.015625
    assert rm.classify_activation(cand_big, big, np)["verdict"] == rm.MATCH


def test_activation_gate_refuses_an_all_zero_reference():
    z = np.zeros((2, 3), dtype=np.float64)
    v = rm.classify_activation(z, z, np)
    assert v["verdict"] == rm.DIVERGENT


def test_empty_activation_match_is_labelled_as_vacuous():
    """An empty `present` is a real state; a vacuous MATCH must not read as a checked one."""
    e = np.zeros((1, 32, 0, 96), dtype=np.float16)
    v = rm.classify_activation(e, e, np)
    assert v["verdict"] == rm.MATCH and v["empty"] is True


def test_classify_outputs_notices_a_missing_output():
    case = rm.Case(rm.PHI35.key, "decode", 1, 4, tokens=1)
    v = rm.classify_outputs(case, [_logits(3)], [_logits(3), np.ones((1, 2))], np)
    assert v["verdict"] == rm.DIVERGENT


def test_classifier_output_is_gated_on_the_predicted_label():
    """A tolerance that passes while the predicted class moves is not a useful tolerance."""
    ref = np.zeros((2, 1000), dtype=np.float32)
    ref[:, 7] = 1.0
    cand = ref.copy()
    cand[1, 7] = 0.0
    cand[1, 8] = 1.0
    assert rm.classify_tensor(ref, ref, np)["verdict"] == rm.MATCH
    assert rm.classify_tensor(cand, ref, np)["verdict"] == rm.DIVERGENT


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_dispatch_diagnosis_divides_by_inferences_not_by_repeats():
    """A cumulative counter divided by the wrong denominator is how this project once read a
    1.78x "improvement" that was really an iteration ratio."""
    d = rm.dispatch_diagnosis({"dispatches_executed": 1000, "subgraphs_live": 33}, 10)
    assert d["dispatches_per_inference"] == pytest.approx(100.0)


def test_dispatch_diagnosis_with_no_counters_is_none_not_zero():
    d = rm.dispatch_diagnosis(None, 10)
    assert d["dispatches_per_inference"] is None
    assert d["islands"] is None


def test_fallback_diagnosis_separates_claimed_from_executed():
    f = rm.fallback_diagnosis({rm.EP_NAME: 3, rm.CPU_EP: 97})
    assert f["cpu_fallback_node_executions"] == 97
    assert f["vulkan_share"] == pytest.approx(0.03)


def test_fallback_diagnosis_of_an_absent_profile_is_not_a_perfect_score():
    f = rm.fallback_diagnosis(None)
    assert f["vulkan_share"] is None
    assert f["total_node_executions"] == 0


def test_bandwidth_proxy_says_what_it_assumes():
    p = rm.bandwidth_proxy(rm.Case(rm.PHI35.key, "decode", 1, 1024, tokens=1), 500.0)
    assert p["implied_gib_per_s"] > 0
    assert "device-resident" in p["assumes"]


def test_bandwidth_proxy_is_none_without_a_kv_cache():
    assert rm.bandwidth_proxy(rm.Case(rm.MOBILENETV2.key, "batch", 1, 0), 10.0) is None


# ---------------------------------------------------------------------------
# The probe's own output paths — a regression control
# ---------------------------------------------------------------------------


def _probe_module():
    """Import `probe_real_model_latency` without running it. It imports ORT lazily, so this is
    safe on a machine with no EP."""
    import importlib.util

    path = _BENCH / "results" / "probe_real_model_latency.py"
    spec = importlib.util.spec_from_file_location("probe_real_model_latency", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_two_passes_do_not_default_to_the_same_file():
    """REGRESSION (2026-08-07): `--diagnose` inherited the timed pass's default `--out` and
    overwrote a completed matrix with a profiling record carrying a different schema. The two
    passes measure different things; landing on one path by accident destroyed evidence that
    took thirteen minutes to produce and would have been reported as a matrix.

    Asserted on the parser's own defaults rather than by running the probe, because the failure
    was in argument resolution and that is the thing that must stay fixed.
    """
    src = (_BENCH / "results" / "probe_real_model_latency.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--out", default=None' in src, (
        "--out must not carry a pass-independent default; that is what caused the overwrite"
    )
    assert '"real_model_diagnostics.json" if args.diagnose else "real_model_latency.json"' in src


def test_the_diagnostics_pass_names_its_own_schema():
    """A file whose name says `latency` and whose schema says `diagnostics` is how the overwrite
    stayed invisible for one command. The schema string is the second, independent witness."""
    src = (_BENCH / "results" / "probe_real_model_latency.py").read_text(encoding="utf-8")
    assert '"schema": "real_model_diagnostics/1"' in src
    assert '"schema": rm.SCHEMA' in src
    assert rm.SCHEMA != "real_model_diagnostics/1"
