"""Self-tests for `bench/resnet.py` — no GPU, no EP, no CUDA, no model file required.

Issue #122 asks for ResNet numbers against the CUDA EP. The half of that instrument that can be
wrong *without* a device — the pin, the feeds, the polarity of the ratio, the classifier gate,
the static census, and the admissibility rule — is checked here, on every `pytest bench` run,
and not only on the one machine that happens to have an RTX A1000 and a CUDA-12 ORT wheel.

Named for the **plausible but wrong** reading each one prevents, in the house style of
`bench/test_plausible_but_wrong.py` and `bench/test_real_model.py`.
"""

from __future__ import annotations

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
import resnet as rn  # noqa: E402

_PROBE = _BENCH / "results" / "probe_resnet_vulkan_cuda.py"


# ---------------------------------------------------------------------------
# The pin — issue #78's discipline, applied to a second model
# ---------------------------------------------------------------------------

def test_resnet_is_pinned_to_an_immutable_revision_not_a_branch():
    """`main` names whatever was pushed last; a 40-hex commit names bytes."""
    rev = rn.RESNET50_PIN["revision"]
    assert len(rev) == 40
    assert rev == rev.lower()
    assert all(c in "0123456789abcdef" for c in rev)
    for movable in ("main", "master", "refs/heads", "latest", "HEAD"):
        assert movable not in rev


def test_the_pin_names_a_digest_and_a_byte_count_not_just_a_file():
    """A filename was the whole check once (#78). Here the size and the digest are the check."""
    pin = rn.RESNET50_PIN
    assert len(pin["sha256"]) == 64
    assert pin["sha256"] == pin["sha256"].lower()
    assert isinstance(pin["pinned_bytes"], int) and not isinstance(pin["pinned_bytes"], bool)
    assert pin["pinned_bytes"] > 0
    assert pin["repo"] == "onnxmodelzoo/resnet50-v1-12"
    assert pin["file"] == "resnet50-v1-12.onnx"
    assert pin["declared_external_files"] == 0


def test_the_url_resolves_the_pinned_revision_not_a_moving_ref():
    """A recorded URL that points at `main` documents a download, not a pin."""
    assert rn.RESNET50_PIN["revision"] in rn.RESNET50_URL
    assert rn.RESNET50_PIN["repo"] in rn.RESNET50_URL
    assert "/main/" not in rn.RESNET50_URL


def test_the_spec_carries_no_literal_local_path():
    blob = repr(rn.RESNET50)
    assert ":\\" not in blob and ":/" not in blob
    assert "Users" not in blob and "home" not in blob


def test_the_spec_resolves_through_the_pinned_bytes_authority():
    """Not through `repo-cache`, which decides identity by filename."""
    assert rn.RESNET50.resolver == "pinned"
    assert rn.RESNET50.pin is rn.RESNET50_PIN
    assert rn.RESNET50.recorded_provenance == "pinned-bytes/resnet50-v1-12.json"


def test_the_shipped_sidecar_names_the_same_bytes_the_code_pins():
    """If the witness and the pin disagree, one of them is stale and neither is evidence."""
    side = json.loads(
        (_BENCH / "results" / rn.RESNET50.recorded_provenance).read_text(encoding="utf-8")
    )
    assert side["onnx_sha256"] == rn.RESNET50_PIN["sha256"]
    assert side["onnx_bytes"] == rn.RESNET50_PIN["pinned_bytes"]
    assert side["pin"]["revision"] == rn.RESNET50_PIN["revision"]
    assert side["pin"]["repo"] == rn.RESNET50_PIN["repo"]
    assert side["pin"]["file"] == rn.RESNET50_PIN["file"]
    assert rn.RESNET50_PIN["revision"] in side["pin"]["url"]
    assert side["external_data"]["declared_files"] == 0
    assert side["outcome"] == "PASS"


def test_the_sidecar_carries_no_local_path_for_the_reader_to_trip_over():
    """`bench/path_screen.py`'s rule: a published record names bytes, not this machine."""
    blob = (_BENCH / "results" / rn.RESNET50.recorded_provenance).read_text(encoding="utf-8")
    assert "C:\\\\Users" not in blob and "/home/" not in blob
    assert "<redacted-local-path>" in blob


def test_the_recorded_histogram_is_re_derived_from_the_bytes_when_they_are_here():
    """A literal op count is a MEASUREMENT only while it still matches the file it describes."""
    path = rm.repo_cache_dir() / rn.RESNET50.cache_filename
    if not path.is_file():
        pytest.skip(f"the pinned ResNet-50 is not in this machine's cache ({path.name})")
    onnx = pytest.importorskip("onnx")
    model = onnx.load(str(path), load_external_data=False)
    hist: dict = {}
    for node in model.graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    assert hist == rn.RESNET50_OP_HISTOGRAM
    assert sum(hist.values()) == rn.RESNET50_NODES
    assert model.graph.input[0].name == rn.RESNET50_INPUT
    assert model.graph.output[0].name == rn.RESNET50_OUTPUT


def test_the_real_pinned_resnet_verifies_when_it_is_present():
    """Not a skip-shaped pass: when the file is here, this is the whole gate end to end."""
    path = rm.repo_cache_dir() / rn.RESNET50.cache_filename
    if not path.is_file():
        pytest.skip(f"the pinned ResNet-50 is not in this machine's cache ({path.name})")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rn.RESNET50_PIN["sha256"]
    rec = rm.resolve_model(rn.RESNET50)
    assert rec["provenance_ok"] is True
    assert rec["provenance"] == "pinned-immutable"
    assert rec["sha256"] == rn.RESNET50_PIN["sha256"]
    assert rec["bytes"] == rn.RESNET50_PIN["pinned_bytes"]
    assert rec["agrees_with_recorded_provenance"] is True
    assert rec["external_data"]["files"] == []


def test_an_absent_pinned_resnet_is_unavailable_not_an_unverified_pass(tmp_path, monkeypatch):
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    with pytest.raises(rm.ModelUnavailable) as exc:
        rm.resolve_model(rn.RESNET50)
    assert "absent" in str(exc.value)


def test_a_same_named_file_that_is_not_the_pinned_bytes_is_refused(tmp_path, monkeypatch):
    """The #78 exploit, re-run against this lane's model: the name is not the identity."""
    monkeypatch.setenv(rm.REPO_CACHE_ENV, str(tmp_path))
    (tmp_path / rn.RESNET50.cache_filename).write_bytes(b"not a resnet")
    with pytest.raises(rm.ModelUnavailable) as exc:
        rm.resolve_model(rn.RESNET50)
    assert "REFUSED" in str(exc.value)


def test_resnet_is_absent_from_the_shared_timed_matrix():
    """This lane owns its own driver; it must not silently join another agent's running one.

    `probe_real_model_latency.py` builds cases with a MobileNet-shaped guess for any key that
    is not Phi, so a ResNet spec injected into `rm.MODELS` would be fed a 224x224x3 *MobileNet*
    feed dict under the wrong input name and would fail — or worse, on a machine without the
    ResNet cache, would take a running lane down with a `ModelUnavailable`.
    """
    assert rn.RESNET50.key not in rm.MODELS
    assert rn.RESNET50.key not in rm.PROVENANCE_ONLY
    probe = (_BENCH / "results" / "probe_real_model_latency.py").read_text(encoding="utf-8")
    assert "resnet" not in probe.lower()


# ---------------------------------------------------------------------------
# Cases and feeds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [[True], [4.0], ["4"], [0], [-1], 4, "4"])
def test_resnet_cases_refuses_a_batch_nobody_chose(batch):
    """`int(4.7)` succeeds and means the caller did not say what they meant."""
    with pytest.raises((TypeError, ValueError)):
        rn.resnet_cases(batch)


def test_resnet_cases_carry_the_batch_they_were_asked_for():
    cases = rn.resnet_cases([1, 4, 16])
    assert [c.m for c in cases] == [1, 4, 16]
    assert {c.model_key for c in cases} == {rn.RESNET50.key}
    assert {c.unit for c in cases} == {"images"}
    assert all(c.tokens is None for c in cases)


def test_feeds_are_keyed_by_the_models_own_input_name():
    """A right-shaped array under a guessed key benchmarks whichever model uses that key."""
    feeds = rn.resnet_feeds(rn.resnet_cases([2])[0], np)
    assert list(feeds) == [rn.RESNET50_INPUT]
    assert feeds[rn.RESNET50_INPUT].shape == (2, 3, 224, 224)
    assert feeds[rn.RESNET50_INPUT].dtype == np.float32


def test_feeds_are_byte_identical_across_calls():
    """Two arms fed different bytes are not two arms; they are two experiments."""
    case = rn.resnet_cases([3])[0]
    a = rn.resnet_feeds(case, np)
    b = rn.resnet_feeds(case, np)
    assert rm.feeds_digest(a) == rm.feeds_digest(b)
    assert np.array_equal(a[rn.RESNET50_INPUT], b[rn.RESNET50_INPUT])


def test_feeds_of_two_batches_are_not_the_same_digest():
    """Guards against a digest that summarises the name and forgets the shape."""
    d1 = rm.feeds_digest(rn.resnet_feeds(rn.resnet_cases([1])[0], np))
    d4 = rm.feeds_digest(rn.resnet_feeds(rn.resnet_cases([4])[0], np))
    assert d1 != d4


def test_feeds_refuse_a_case_belonging_to_another_model():
    other = rm.Case("mobilenetv2-12", "batch", 1, 0, tokens=None, unit="images")
    with pytest.raises(ValueError):
        rn.resnet_feeds(other, np)


def test_the_resnet_seed_is_distinct_from_the_kv_seed():
    """So a ResNet feed digest can never collide with a Phi feed digest in a report."""
    assert rn.RESNET_SEED != getattr(rm, "KV_SEED", object())


# ---------------------------------------------------------------------------
# Arms — including this repository's first CUDA arm
# ---------------------------------------------------------------------------

def test_there_is_a_cuda_arm_at_all():
    """The whole of issue #122's second question. Before this file there was none."""
    assert rn.CUDA_EP == "CUDAExecutionProvider"
    assert rn.CUDA_ARM.providers[0] == rn.CUDA_EP
    assert rn.CUDA_ARM in rn.ARMS


def test_the_vulkan_and_cuda_arms_keep_the_cpu_fallback_the_shipped_ep_has():
    """Removing it would measure a configuration no user runs, and would turn every
    unsupported op into a session failure rather than into the partition cost being priced."""
    assert rn.VULKAN_ARM.providers == (rn.EP_NAME, rn.CPU_EP)
    assert rn.CUDA_ARM.providers == (rn.CUDA_EP, rn.CPU_EP)
    assert rn.CPU_ARM.providers == (rn.CPU_EP,)


def test_the_reference_arm_is_the_cpu_ep_and_is_labelled_as_such():
    assert rn.CPU_ARM.role == "reference"
    assert rn.VULKAN_ARM.role == "candidate"
    assert rn.CUDA_ARM.role == "baseline"
    assert rn.CPU_ARM not in rn.COMPARED_ARMS
    assert set(rn.COMPARED_ARMS) == {rn.VULKAN_ARM, rn.CUDA_ARM}


def test_no_arm_sets_an_environment_variable_behind_the_readers_back():
    """This lane compares EPs, not tunings; an env delta would be a third variable."""
    for arm in rn.ARMS:
        assert arm.env == ()


def test_arm_order_alternates_so_the_last_arm_is_not_always_the_same():
    first = [a.name for a in rm.arm_order(rn.ARMS, 0)]
    second = [a.name for a in rm.arm_order(rn.ARMS, 1)]
    assert first == list(reversed(second))


# ---------------------------------------------------------------------------
# Ratio polarity
# ---------------------------------------------------------------------------

def test_a_ratio_cannot_be_emitted_without_saying_which_way_it_points():
    rec = rn.ratio_record([10.0, 10.0, 10.0], [20.0, 20.0, 20.0],
                          baseline="cuda", candidate="vulkan")
    assert rec["baseline"] == "cuda"
    assert rec["candidate"] == "vulkan"
    assert "vulkan" in rec["polarity"] and "cuda" in rec["polarity"]
    assert "LONGER" in rec["polarity"]
    assert rec["provenance_class"] == "MEASUREMENT"


def test_the_ratio_is_candidate_over_baseline_and_a_slower_candidate_is_above_one():
    """The reading `ab_row_tile.py` once had to be recovered from source to interpret."""
    rec = rn.ratio_record([10.0] * 4, [20.0] * 4, baseline="cuda", candidate="vulkan")
    assert rec["median"] == pytest.approx(2.0)
    faster = rn.ratio_record([20.0] * 4, [10.0] * 4, baseline="cuda", candidate="vulkan")
    assert faster["median"] == pytest.approx(0.5)


def test_the_ratio_is_paired_per_repeat_not_a_ratio_of_pooled_medians():
    """A ratio of medians hides the repeat where one arm was displaced."""
    rec = rn.ratio_record([10.0, 10.0, 10.0, 10.0, 10.0],
                          [11.0, 11.0, 99.0, 11.0, 11.0],
                          baseline="cuda", candidate="vulkan")
    assert rec["n"] == 5
    assert rec["median"] == pytest.approx(1.1)
    assert rec["max"] == pytest.approx(9.9)


def test_a_ratio_of_nothing_is_n_zero_not_a_number():
    rec = rn.ratio_record([], [], baseline="cuda", candidate="vulkan")
    assert rec["n"] == 0
    assert "median" not in rec
    assert rec["baseline"] == "cuda"


# ---------------------------------------------------------------------------
# The classifier gate — positive and negative controls
# ---------------------------------------------------------------------------

def _logits(rows: int = 2, seed: int = 7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, rn.RESNET50_CLASSES)) * 4.0


def test_identical_logits_match():
    """The positive control. A gate never seen passing has no demonstrated passing state."""
    ref = _logits()
    got = rn.classify_resnet_logits(ref.copy(), ref, np)
    assert got["verdict"] == rm.MATCH
    assert got["top1_rows_agreeing"] == got["rows"] == 2
    assert got["topk_rows_agreeing"] == 2
    assert got["provenance_class"] == "MEASUREMENT"


def test_a_moved_top1_is_divergent_even_inside_the_numeric_budget():
    """What a classifier's user can see is the label; a swapped argmax is a wrong answer."""
    ref = _logits()
    cand = ref.copy()
    top = int(ref[0].argmax())
    runner = int(np.argsort(-ref[0])[1])
    cand[0, top], cand[0, runner] = ref[0, runner], ref[0, top]
    got = rn.classify_resnet_logits(cand, ref, np)
    assert got["verdict"] == rm.DIVERGENT
    assert got["top1_rows_agreeing"] < got["rows"]


def test_top1_is_checked_on_every_row_not_only_the_first():
    """A batch gate that reads row 0 passes a run that is wrong on rows 1..N-1."""
    ref = _logits(rows=4)
    cand = ref.copy()
    top = int(ref[3].argmax())
    runner = int(np.argsort(-ref[3])[1])
    cand[3, top], cand[3, runner] = ref[3, runner], ref[3, top]
    got = rn.classify_resnet_logits(cand, ref, np)
    assert got["verdict"] == rm.DIVERGENT
    assert got["top1_rows_agreeing"] == 3
    assert got["rows"] == 4


def test_a_reordered_topk_set_is_divergent_even_with_the_same_top1():
    """Membership of the top-5 is the claim; a class that left the set is a different answer."""
    ref = _logits()
    cand = ref.copy()
    order = np.argsort(-ref[0])
    fifth, sixth = int(order[4]), int(order[5])
    cand[0, fifth], cand[0, sixth] = ref[0, sixth], ref[0, fifth]
    got = rn.classify_resnet_logits(cand, ref, np)
    assert got["verdict"] == rm.DIVERGENT
    assert got["top1_rows_agreeing"] == got["rows"]
    assert got["topk_rows_agreeing"] < got["rows"]


def test_an_all_zero_candidate_is_divergent_not_an_argmax_0_pass():
    """THE defect: 161 dispatches executed, `compute_failures: 0`, and no numbers produced."""
    ref = _logits()
    got = rn.classify_resnet_logits(np.zeros_like(ref), ref, np)
    assert got["verdict"] == rm.DIVERGENT
    assert "zero" in got["reason"]


def test_a_nan_candidate_is_divergent_rather_than_an_exception():
    ref = _logits()
    cand = ref.copy()
    cand[0, 0] = np.nan
    got = rn.classify_resnet_logits(cand, ref, np)
    assert got["verdict"] == rm.DIVERGENT
    assert got["nonfinite_candidate"] == 1


def test_a_shape_mismatch_is_divergent_rather_than_a_broadcast():
    ref = _logits(rows=2)
    got = rn.classify_resnet_logits(_logits(rows=1), ref, np)
    assert got["verdict"] == rm.DIVERGENT
    assert "shape" in got["reason"]


def test_a_non_classifier_shape_is_refused_rather_than_scored():
    got = rn.classify_resnet_logits(np.ones((2, 10)), np.ones((2, 10)), np)
    assert got["verdict"] == rm.DIVERGENT
    assert "1000" in got["reason"]


def test_the_absolute_budget_is_a_fraction_of_scale_not_a_constant():
    """A constant floor silently tightens or loosens as the output scale moves."""
    ref = _logits() * 100.0
    got = rn.classify_resnet_logits(ref.copy(), ref, np)
    assert got["abs_budget"] == pytest.approx(
        rn.RESNET_LOGIT_SCALE_FRACTION * float(np.abs(ref).max()))
    assert got["abs_budget"] > 0


def test_a_perturbation_below_the_scale_fraction_still_matches():
    ref = _logits()
    scale = float(np.abs(ref).max())
    cand = ref + 1e-5 * scale
    got = rn.classify_resnet_logits(cand, ref, np)
    assert got["verdict"] == rm.MATCH
    assert got["numeric_ok"] is True


def test_a_probability_shift_fails_even_when_the_ranking_survives():
    """Two logit vectors can rank identically and still describe different confidences."""
    ref = np.zeros((1, rn.RESNET50_CLASSES))
    ref[0, 0] = 10.0
    cand = ref.copy()
    cand[0, 0] = 9.0
    got = rn.classify_resnet_logits(cand, ref, np)
    assert got["top1_rows_agreeing"] == 1
    assert got["max_prob_delta"] > rn.RESNET_MAX_PROB_DELTA
    assert got["verdict"] == rm.DIVERGENT


def test_classify_case_refuses_an_output_count_that_is_not_this_models():
    case = rn.resnet_cases([1])[0]
    ref = [_logits(rows=1)]
    assert rn.classify_case(case, ref, ref, np)["verdict"] == rm.MATCH
    two = rn.classify_case(case, ref + ref, ref, np)
    assert two["verdict"] == rm.DIVERGENT
    assert "count" in two["reason"]


# ---------------------------------------------------------------------------
# The static census — a MODEL, and it must say so
# ---------------------------------------------------------------------------

_CAPS = {
    "crate_version": "0.0.0-test",
    "ops": [
        # `ready` is the status the real EP reports for ResNet-50's Conv, Gemm and
        # GlobalAveragePool. It carries a kernel. See the `has_kernel` test below.
        {"name": "Conv", "status": "ready", "has_kernel": True, "dtypes": ["F32"]},
        {"name": "Relu", "status": "live", "has_kernel": True, "dtypes": ["F32", "F16"]},
        {"name": "Add", "status": "live", "has_kernel": True, "dtypes": ["F32", "F16"]},
        {"name": "Gemm", "status": "ready", "has_kernel": True, "dtypes": ["F32"]},
        {"name": "GlobalAveragePool", "status": "ready", "has_kernel": True, "dtypes": ["F32"]},
        {"name": "Softmax", "status": "staged", "has_kernel": False,
         "staged_reason": "no kernel"},
    ],
}


def test_a_ready_row_carries_a_kernel_and_counts_as_supported():
    """The trap this lane walked into once, locked so it cannot be walked into twice.

    `epctl`'s `status` is three-valued and its `live` token is the **deprecated**
    `OpStatus::Live` alias; `has_kernel` is true for `Live` AND `Ready` (§8.9.25 ruling 6, and
    `rust/src/bin/epctl.rs`'s own comment says a reader checking "76 rows carry a kernel"
    against the field once named `live` got 46). Reading the token instead of the predicate
    reports ResNet-50's `Conv`, `Gemm` and `GlobalAveragePool` — 55 of 175 nodes — as
    unsupported, i.e. announces that this EP cannot run a convolution.
    """
    census = rn.support_census({"Conv": 53, "Gemm": 1, "GlobalAveragePool": 1}, _CAPS)
    assert census["op_types_unsupported"] == []
    assert census["nodes_with_a_registered_kernel"] == 55
    assert census["supported"]["Conv"]["status"] == "ready"


def test_the_census_is_labelled_a_model_not_a_measurement():
    """ORT rewrites the graph before partitioning; a static count is an analytic construction."""
    census = rn.support_census(rn.RESNET50_OP_HISTOGRAM, _CAPS)
    assert census["provenance_class"] == "MODEL"
    assert "upper bound" in census["upper_bound_note"]


def test_the_census_names_the_ops_with_no_registered_kernel():
    """Issue #122's third question, statically: what is missing, by name and by node count."""
    census = rn.support_census(rn.RESNET50_OP_HISTOGRAM, _CAPS)
    assert set(census["op_types_unsupported"]) == {"BatchNormalization", "Flatten", "MaxPool"}
    assert census["unsupported"]["MaxPool"]["nodes"] == 1
    assert census["unsupported"]["Flatten"]["nodes"] == 1
    assert census["unsupported"]["BatchNormalization"]["nodes"] == 53
    assert census["unsupported"]["MaxPool"]["registered"] is False


def test_a_registered_but_staged_op_does_not_count_as_supported():
    """`registered` is necessary and not sufficient; a staged row has no kernel behind it."""
    census = rn.support_census({"Softmax": 3}, _CAPS)
    assert census["op_types_supported"] == []
    assert census["unsupported"]["Softmax"]["registered"] is True
    assert census["unsupported"]["Softmax"]["status"] == "staged"
    assert census["nodes_without_a_registered_kernel"] == 3


def test_the_census_node_counts_add_up_to_the_graph():
    census = rn.support_census(rn.RESNET50_OP_HISTOGRAM, _CAPS)
    assert census["graph_nodes"] == rn.RESNET50_NODES
    assert (census["nodes_with_a_registered_kernel"]
            + census["nodes_without_a_registered_kernel"]) == rn.RESNET50_NODES


def test_an_absent_capabilities_dump_is_zero_coverage_not_full_coverage():
    """A failed `epctl` invocation must not read as 'the EP supports everything'."""
    census = rn.support_census(rn.RESNET50_OP_HISTOGRAM, {"error": "epctl not found"})
    assert census["nodes_with_a_registered_kernel"] == 0
    assert census["op_types_supported"] == []


def test_the_census_against_the_really_built_ep_when_it_is_here():
    """The test that would have caught the `status == "live"` reading, on the real binary.

    A fixture agrees with whatever the fixture's author believed. This one asks the built
    `epctl` and asserts the shape of the answer §27 publishes: convolution is claimable, and
    the three gaps are `MaxPool`, `Flatten` and `BatchNormalization`.
    """
    epctl = _BENCH.parent / "rust" / "target" / "release" / "epctl.exe"
    if not epctl.is_file():
        epctl = _BENCH.parent / "rust" / "target" / "release" / "epctl"
    if not epctl.is_file():
        pytest.skip("epctl is not built in this tree; `cargo build --release` first")
    import subprocess

    out = subprocess.run([str(epctl), "--dump-capabilities", "--json"],
                         capture_output=True, text=True, timeout=120)
    caps = json.loads(out.stdout)
    census = rn.support_census(rn.RESNET50_OP_HISTOGRAM, caps)
    assert "Conv" in census["op_types_supported"], (
        "the EP registers a Conv kernel; a census that calls it unsupported is reading the "
        "deprecated `status` token instead of `has_kernel`")
    assert set(census["op_types_supported"]) == {
        "Add", "Conv", "Gemm", "GlobalAveragePool", "Relu"}
    assert set(census["op_types_unsupported"]) == {
        "BatchNormalization", "Flatten", "MaxPool"}
    assert census["nodes_with_a_registered_kernel"] == 120
    assert census["nodes_without_a_registered_kernel"] == 55


# ---------------------------------------------------------------------------
# Admissibility — the gate that decides whether anything may be quoted
# ---------------------------------------------------------------------------

def _admissible_kwargs(**over):
    base = dict(
        provenance_ok=True,
        equivalence_gate="PASS",
        vulkan_dispatched=161,
        cuda_ran=True,
        quiescence={"quiet": True, "reason": "box was idle"},
        device_identified=True,
        repeats=5,
        iters=10,
    )
    base.update(over)
    return base


def test_the_happy_path_is_admissible():
    """The positive control for the gate itself."""
    got = rn.admissibility(**_admissible_kwargs())
    assert got["verdict"] == rn.ADMISSIBLE
    assert got["failed_checks"] == []
    assert rn.quotable(got) is True


@pytest.mark.parametrize("override,expected_failure", [
    ({"provenance_ok": False}, "model_provenance"),
    ({"equivalence_gate": "FAIL"}, "outputs_agree_with_cpu"),
    ({"vulkan_dispatched": 0}, "vulkan_production_dispatch_witness"),
    ({"vulkan_dispatched": None}, "vulkan_production_dispatch_witness"),
    ({"cuda_ran": False}, "cuda_arm_executed"),
    ({"device_identified": False}, "device_identified"),
    ({"quiescence": {"quiet": False, "reason": "another benchmark was running"}},
     "quiescence"),
    ({"quiescence": None}, "quiescence"),
    ({"repeats": 2}, "enough_repeats"),
    ({"iters": 4}, "enough_repeats"),
])
def test_any_single_failure_makes_the_whole_reading_indeterminate(override, expected_failure):
    """No check is decorative; each one alone withholds the number."""
    got = rn.admissibility(**_admissible_kwargs(**override))
    assert got["verdict"] == rn.INDETERMINATE
    assert expected_failure in got["failed_checks"]
    assert rn.quotable(got) is False


def test_every_check_is_named_and_carries_its_own_detail():
    """A verdict whose reasons are not enumerable cannot be argued with."""
    got = rn.admissibility(**_admissible_kwargs())
    names = [c["name"] for c in got["checks"]]
    assert len(names) == len(set(names)) == 7
    for c in got["checks"]:
        assert isinstance(c["detail"], str) and c["detail"]


def test_the_verdict_is_recomputed_from_the_evidence_and_never_stored():
    """`ProvenanceRecord.provenance_ok`'s discipline: a stored verdict can outlive its facts."""
    src = (_BENCH / "resnet.py").read_text(encoding="utf-8")
    assert "def admissibility(" in src
    assert "verdict=" not in src.split("def admissibility(")[1].split("\ndef ")[0]


def test_quotable_of_nothing_is_false_not_true():
    assert rn.quotable(None) is False
    assert rn.quotable({}) is False
    assert rn.quotable({"verdict": rn.INDETERMINATE}) is False


def test_indeterminate_is_a_result_not_an_exception():
    """The lane must be able to report 'the machine state does not support this number'."""
    got = rn.admissibility(**_admissible_kwargs(quiescence={"quiet": False, "reason": "busy"}))
    assert got["verdict"] == rn.INDETERMINATE
    assert "INDETERMINATE" in got["rule"]


# ---------------------------------------------------------------------------
# Structural guards on the driver
# ---------------------------------------------------------------------------

def test_the_generalisation_limit_travels_with_the_numbers():
    """One device, one model, one driver is a reading. It is not parity."""
    assert "parity" in rn.GENERALISATION_LIMIT
    src = _PROBE.read_text(encoding="utf-8")
    assert "rn.GENERALISATION_LIMIT" in src


def test_the_driver_names_its_own_schema():
    src = _PROBE.read_text(encoding="utf-8")
    assert '"schema": rn.SCHEMA' in src
    assert rn.SCHEMA == "resnet_vulkan_cuda/1"
    assert rn.SCHEMA != rm.SCHEMA


def test_the_timed_pass_does_not_profile_and_the_diagnostic_pass_is_not_timed():
    """A wall time measured through an instrument is not the wall time without it."""
    src = _PROBE.read_text(encoding="utf-8")
    timed = src.split("def timed_worker(")[1].split("\ndef ")[0]
    assert "profiling=True" not in timed
    assert "enable_profiling" not in timed
    diag = src.split("def diagnose_worker(")[1].split("\ndef ")[0]
    assert "profiling=True" in diag
    assert "perf_counter" not in diag


def test_the_driver_pins_the_optimisation_level_rather_than_defaulting_it():
    """The level decides whether the 53 BatchNormalization nodes still exist at partition."""
    src = _PROBE.read_text(encoding="utf-8")
    assert "ORT_ENABLE_ALL" in src


def test_the_driver_pins_both_gpu_devices_by_index():
    """A result that does not name which card it ran on is not a result."""
    src = _PROBE.read_text(encoding="utf-8")
    assert "ep.device_index" in src
    assert '"device_id"' in src


def test_the_driver_preloads_cuda_and_records_whether_it_worked():
    """A CUDA arm that silently fell back to CPU produces plausible numbers for the wrong thing."""
    src = _PROBE.read_text(encoding="utf-8")
    assert "def _preload_cuda(" in src
    assert "cuda_preload" in src
    for worker in ("timed_worker", "diagnose_worker"):
        body = src.split(f"def {worker}(")[1].split("\ndef ")[0]
        assert "_preload_cuda(ort)" in body


def test_the_driver_records_the_providers_the_session_actually_got():
    """`providers=[CUDA, CPU]` is a request. `sess.get_providers()` is what happened."""
    src = _PROBE.read_text(encoding="utf-8")
    assert "get_providers()" in src


def test_the_driver_can_refuse_to_start_on_top_of_another_agents_benchmark():
    """This box is shared and other lanes measure on it (docs/PERF.md §20)."""
    src = _PROBE.read_text(encoding="utf-8")
    assert "--require-lock" in src
    assert "def _lock_held(" in src
    assert "REFUSED(instrument=gpu_lock_not_exclusive)" in src


def test_the_lock_refusal_happens_before_the_runtime_is_even_imported():
    """A refusal that first opened a device would itself perturb the run it declined to join."""
    src = _PROBE.read_text(encoding="utf-8")
    main = src.split("def main(")[1]
    assert main.index("if a.require_lock") < main.index("import onnxruntime as ort")
    assert main.index("if a.require_lock") < main.index("device_mod.probe()")


def test_an_empty_lock_directory_is_not_read_as_permission():
    """`held` is a file that exists, never the absence of somebody else's file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_probe_resnet_lock", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    empty = Path(__file__).resolve().parent / "results" / "no-such-lock-dir"
    got = mod._lock_held(empty, "niobe-11")
    assert got["held"] is False
    assert got["exclusive"] is False


def test_a_lock_held_by_somebody_else_is_not_exclusive(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_probe_resnet_lock2", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "niobe-11.lock").write_text("mine")
    assert mod._lock_held(tmp_path, "niobe-11")["exclusive"] is True
    (tmp_path / "niobe-10.lock").write_text("theirs")
    got = mod._lock_held(tmp_path, "niobe-11")
    assert got["held"] is True
    assert got["exclusive"] is False
    assert got["other_holders"] == ["niobe-10.lock"]


def test_the_driver_verifies_before_it_times():
    """A case that does not agree with the CPU reference is not timed at all."""
    src = _PROBE.read_text(encoding="utf-8")
    main = src.split("def main(")[1]
    assert main.index("verify_case(") < main.index("run_timed(")
    assert 'if v["gate"] == "PASS":' in main


def test_the_driver_gates_its_exit_code_on_admissibility():
    """An INDETERMINATE run must not look like a successful one to a script."""
    src = _PROBE.read_text(encoding="utf-8")
    assert 'return 0 if admis["verdict"] == rn.ADMISSIBLE else 1' in src


def test_the_driver_discloses_foreign_gpu_load_around_the_timed_pass():
    src = _PROBE.read_text(encoding="utf-8")
    assert "def foreign_load(" in src
    assert "nvidia_before" in src and "nvidia_after" in src


def test_the_driver_attributes_transfers_separately_from_compute():
    """MemcpyToHost/FromHost is the price of partition fragmentation, not of a kernel."""
    src = _PROBE.read_text(encoding="utf-8")
    assert "Memcpy" in src
    assert "transfer_nodes" in src
    assert "cpu_fallback_op_types" in src


def test_every_published_quantity_declares_a_provenance_class():
    """Round 10: SPECIFICATION, MEASUREMENT or MODEL — a number with no class is unquotable."""
    src = _PROBE.read_text(encoding="utf-8")
    block = src.split('"PROVENANCE": {')[1].split("},")[0]
    for key in ("latency_medians", "ratios", "dispatches_executed", "static_support_census",
                "device_and_driver_identity", "model_pin"):
        assert key in block
    for cls in ("MEASUREMENT", "MODEL", "SPECIFICATION"):
        assert cls in block
