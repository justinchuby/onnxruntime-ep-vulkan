"""Two-polarity screen and mutation battery for ``bench/attribution_gate.py``.

Each of the seven gates is exercised in **both** polarities in section 2 — a clean record it
must accept, and the specific defect the Fact Checker found in PR #94's artifact, which it must
refuse. Section 3 plants defective reimplementations of the two gates whose failure is silent
(the path screen and the share table) and puts them through the identical protocol, because a
gate that has only ever been watched agreeing is a gate nobody has watched.

Section 4 is the one that could not be written against PR #94's design at all: it asserts that a
refused record produces an object with **no attribution in it**. Not a withheld table, not a
null-valued one — nothing to lift out.

Nothing here is gated on a device, a model, a driver or a network. All of it runs in the
always-on lane, which is what makes it visible to the census.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_gate as gate  # noqa: E402
import phases  # noqa: E402
from _polarity import PolarityError, refuses, selects  # noqa: E402


# ---------------------------------------------------------------------------
# 1. FIXTURES — a record that passes every gate, and the trace it is about
# ---------------------------------------------------------------------------

_ABI, _ABI_NOTE = gate._expected_abi()

#: A digest-shaped string. Never a real one — the fixtures must not look like evidence.
_H = "0" * 64


def _phase(name: str, ts: int, dur: int, nested_in: str = "none") -> dict:
    return {
        "ph": "X",
        "cat": "ep.phase",
        "name": f"vulkan.{name}",
        "ts": ts,
        "dur": dur,
        "args": {"device": "host", "caveat": "…", "nested_in": nested_in},
    }


def _good_events() -> "list[dict]":
    """One admissible call: a 1000 µs total holding 700 µs of disjoint siblings.

    ``record`` is 500 µs and the ``cmd_upload`` nested inside it is 400 µs, which is the exact
    shape of the denominator error: 400/500 = 80.0% **of record** and 400/1000 = 40.0
    percentage points **of the whole call**. Both must appear, labelled.
    """
    return [
        {
            "ph": "X", "cat": "ep", "name": phases.COMPUTE_CALL, "ts": 1_000, "dur": 1_000,
            "args": {"device": "host", "nodes": 3, "outcome": "ok",
                     "boundary": phases.COMPUTE_CALL_BOUNDARY},
        },
        {"ph": "X", "cat": "ep", "name": phases.SUBGRAPH, "ts": 1_050, "dur": 900,
         "args": {"device": "host", "nodes": 3}},
        _phase("record", 1_060, 500),
        _phase("cmd_upload", 1_070, 400, nested_in="record"),
        _phase("submit", 1_600, 50),
        _phase("fence_wait", 1_660, 150),
    ]


def _good_record() -> dict:
    return {
        "model": {
            "key": "phi35-decode",
            "sha256": _H,
            "external_data": {
                "scanned": True,
                "files": [{"location": "model.onnx.data",
                           "bytes": 2_290_000_000, "sha256": _H}],
            },
        },
        "device": {"name": "a Vulkan device", "driver_version": "0.0.0",
                   "api_version": "1.1.0"},
        "host": {"onnxruntime": "0.0.0"},
        "build": {"sha256": _H, "profile": "release"},
        "source_commit": {"commit": "0" * 40, "dirty": False},
        "gpu_lock": {"held": True, "mechanism": "exclusive-lock witness recorded by the harness"},
        "machine_quiescence": {"verdict": "QUIET"},
        "started_at": "2026-08-09T01:00:00+00:00",
        "taken_at": "2026-08-09T01:05:00+00:00",
        "finished_at": "2026-08-09T01:20:00+00:00",
        "quotable_points": [{"case": "decode-past1024", "repeat": 0}],
        "equivalence": [{"case": "decode-past1024", "repeat": 0, "verdict": "MATCH",
                         "outputs_compared": 65, "outputs_total": 65}],
        "counters": {"abi_version": _ABI, "compute_calls": 1,
                     "dispatches_executed": 353, "queue_submits_completed": 1},
        "record_paths": {"first_record": 1, "replay": 0, "rerecord": 0},
        "children": {0: {"cmd_upload": {"parent": "record", "ms": 0.4}}},
    }


def _without(record: dict, *path):
    """A copy of *record* with one nested key removed — the shape of a missing witness."""
    import copy

    out = copy.deepcopy(record)
    cur = out
    for k in path[:-1]:
        cur = cur[k]
    cur.pop(path[-1], None)
    return out


def _with(record: dict, path, value):
    import copy

    out = copy.deepcopy(record)
    cur = out
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    return out


# ---------------------------------------------------------------------------
# 2. THE REAL GATES, IN BOTH POLARITIES
# ---------------------------------------------------------------------------


def test_the_abi_expectation_comes_from_the_shared_reader_and_not_from_a_constant():
    """The probe defect, asserted directly: there is no hard-coded version in this module.

    PR #94's probe carried its own idea of the ABI and accepted v8 after the struct had moved.
    ``_expected_abi`` parses ``rust/src/counters.rs`` through ``rust/tools/counters_abi.py`` —
    the one reader ``tests/ops/test_counters_abi_singleton.py`` enforces — and refuses rather
    than defaulting when it cannot.
    """
    assert isinstance(_ABI, int) and _ABI >= 1, (_ABI, _ABI_NOTE)
    source = Path(gate.__file__).read_text(encoding="utf-8")
    # A digit-only literal beside the word `abi` would be a second opinion about the version.
    assert not re.search(r"abi\w*\s*[=:]\s*\d+", source, re.IGNORECASE), (
        "attribution_gate.py appears to state an ABI version of its own. The expected version "
        "must come from counters_abi.abi_version() and from nowhere else."
    )


def test_a_record_with_no_absolute_paths_is_certified_by_identity():
    """Accept polarity, compared with ``is``: an equal-but-different empty set must not pass."""
    selects(gate.public_path_screen(_good_record()), gate.NO_PRIVATE_PATHS)
    selects(gate.public_path_screen({"a": ["bench/results/x.json", "https://host/a/b", "1/2"]}),
            gate.NO_PRIVATE_PATHS)


@pytest.mark.parametrize(
    "planted",
    [
        r"C:\Users\someone\models\model.onnx",
        r"D:\other-user\cache\phi-3.5\model.onnx.data",
        r"\\fileserver\share\models\model.onnx",
        "/home/someone/.cache/models/model.onnx",
        "/mnt/scratch/run/model.onnx",
    ],
)
def test_an_absolute_path_from_any_root_is_refused(planted: str):
    """The correction. PR #94's screen knew one home directory; these are five roots.

    The second entry is the exact disclosure the Fact Checker found: a path under another
    user's account on a second drive, published verbatim because the screen was a denylist.
    """
    why = refuses(gate.public_path_screen(_with(_good_record(), ("model", "path"), planted)))
    assert "absolute filesystem path" in why
    assert "known roots" in why


def test_a_full_weight_digest_binds_the_record_to_the_numbers_the_model_multiplies():
    bound, why = gate.weight_digest_binding(_good_record()["model"])
    assert bound["binding"] == "graph+external-bytes"
    assert bound["weight_bytes"] == 2_290_000_000
    assert bound["weight_files"][0]["sha256"] == _H
    assert "same-sized substitution" in why


def test_a_self_contained_graph_is_accepted_and_says_so():
    """The other honest shape: no external initializers, so the .onnx digest covers the weights."""
    model = {"sha256": _H,
             "external_data": {"scanned": True, "files": [],
                               "reason": "graph carries no external initializers"}}
    bound, _ = gate.weight_digest_binding(model)
    assert bound["binding"] == "self-contained"


def test_an_unhashed_external_weight_file_is_refused():
    """PR #94's artifact: graph sha256 + weight *size*, and 2.29 GB of unidentified numbers."""
    model = _good_record()["model"]
    model["external_data"]["files"][0]["sha256"] = None
    why = refuses(gate.weight_digest_binding(model))
    assert "not bound to a digest" in why and "2.29 GB" in why


def test_weights_that_were_never_scanned_are_refused():
    why = refuses(gate.weight_digest_binding(
        {"sha256": _H, "external_data": {"scanned": False, "reason": "onnx not importable"}}))
    assert "UNHASHED" in why and "same-sized substitution" in why


def test_a_fully_framed_record_names_what_ran_it_where_and_from_which_source():
    frame, why = gate.environment_witnesses(_good_record())
    assert frame["source_commit"] == "0" * 40
    assert frame["window"]["seconds"] == 1200.0
    assert "GPU held" in why and "machine QUIET" in why


@pytest.mark.parametrize("path", [
    ("device", "name"),
    ("device", "driver_version"),
    ("device", "api_version"),
    ("host", "onnxruntime"),
    ("build", "sha256"),
    ("build", "profile"),
    ("source_commit", "commit"),
    ("gpu_lock", "mechanism"),
])
def test_every_required_environment_witness_is_actually_required(path):
    """Each one, individually. A required-field list nothing removes a field from is a comment."""
    why = refuses(gate.environment_witnesses(_without(_good_record(), *path)))
    assert ".".join(path) in why


def test_a_hard_coded_taken_at_falls_outside_the_run_window_and_is_refused():
    """The literal defect: ``taken_at`` was a constant in PR #94's artifact.

    It is refused not because it is a constant — nothing can see that — but because a timestamp
    no clock produced does not lie between the run's own start and end.
    """
    why = refuses(gate.environment_witnesses(
        _with(_good_record(), ("taken_at",), "2026-01-01T00:00:00+00:00")))
    assert "HARD-CODED" in why and "outside the run window" in why


def test_a_dirty_tree_a_debug_build_an_unheld_lock_and_a_loud_machine_are_each_refused():
    r = _good_record()
    assert "dirty" in refuses(gate.environment_witnesses(
        _with(r, ("source_commit", "dirty"), True)))
    assert "not 'release'" in refuses(gate.environment_witnesses(
        _with(r, ("build", "profile"), "debug")))
    assert "gpu_lock.held" in refuses(gate.environment_witnesses(
        _with(r, ("gpu_lock", "held"), False)))
    assert "machine_quiescence" in refuses(gate.environment_witnesses(
        _with(r, ("machine_quiescence", "verdict"), "CONTENDED")))


def test_an_unknown_cleanliness_is_not_a_clean_tree():
    """``dirty: None`` means git could not be consulted. It must not read as ``False``."""
    why = refuses(gate.environment_witnesses(_with(_good_record(), ("source_commit", "dirty"),
                                                   None)))
    assert "dirty is None" in why


def test_equivalence_over_every_quoted_point_and_every_output_is_accepted():
    cover, why = gate.equivalence_coverage(_good_record())
    assert cover["points"] == 1 and cover["verdict"] == gate.EQUIVALENCE_MATCH
    assert "every output" in why


def test_quoting_a_context_length_that_was_never_checked_is_refused():
    """PR #94 checked past=128 and quoted 512, 1024 and 2048."""
    r = _good_record()
    r["quotable_points"] = [{"case": "decode-past1024", "repeat": 0},
                            {"case": "decode-past2048", "repeat": 0}]
    why = refuses(gate.equivalence_coverage(r))
    assert "NO equivalence entry" in why and "decode-past2048" in why


def test_comparing_only_the_logits_and_skipping_the_kv_outputs_is_refused():
    """The failure that looks most like success: right first token, wrong sequence."""
    r = _with(_good_record(), ("equivalence",),
              [{"case": "decode-past1024", "repeat": 0, "verdict": "MATCH",
                "outputs_compared": 1, "outputs_total": 65}])
    why = refuses(gate.equivalence_coverage(r))
    assert "1 of 65 outputs compared" in why and "wrong cache" in why


def test_a_non_match_verdict_at_a_quoted_point_is_refused():
    r = _with(_good_record(), ("equivalence",),
              [{"case": "decode-past1024", "repeat": 0, "verdict": "UNATTRIBUTED",
                "outputs_compared": 65, "outputs_total": 65}])
    assert "UNATTRIBUTED" in refuses(gate.equivalence_coverage(r))


def test_a_record_that_does_not_declare_what_it_quotes_is_refused():
    assert "unfalsifiable" in refuses(gate.equivalence_coverage(_without(_good_record(),
                                                                        "quotable_points")))


def test_counters_at_the_current_abi_with_the_record_path_wired_are_accepted():
    witness, why = gate.counters_witness(_good_record())
    assert witness["abi_version"] == _ABI
    assert witness["record_paths"]["first_record"] == 1
    assert "record path wired" in why


def test_counters_from_a_stale_abi_are_refused_by_the_shared_readers_number():
    """Not "v8 is refused" — *any* version that is not this checkout's is refused."""
    why = refuses(gate.counters_witness(_with(_good_record(), ("counters", "abi_version"),
                                              _ABI - 1)))
    assert f"ABI v{_ABI - 1}" in why and f"v{_ABI}" in why


def test_missing_counters_and_an_unwired_record_path_are_each_refused():
    r = _good_record()
    assert "no counters block" in refuses(gate.counters_witness(_without(r, "counters")))
    assert "queue_submits_completed" in refuses(gate.counters_witness(
        _without(r, "counters", "queue_submits_completed")))
    assert "not wired" in refuses(gate.counters_witness(
        _with(r, ("record_paths",), {"first_record": 0, "replay": 0, "rerecord": 0})))
    assert "never entered" in refuses(gate.counters_witness(
        _with(r, ("counters", "compute_calls"), 0)))


def test_every_percentage_carries_the_number_it_was_divided_by():
    """The 16.4%-of-record correction, in the fixture's own arithmetic.

    ``cmd_upload`` is 0.4 ms inside a 0.5 ms ``record`` inside a 1.0 ms call: **80.0% of
    record** and **40.0 percentage points of the whole call**. One label for those two numbers
    is a two-fold error, and in PR #94's real figures it was a 5.8-fold one.
    """
    rows, _ = phases.compute_call_attribution(_good_events())
    table, why = gate.attribution_shares(rows[0], {"cmd_upload": {"parent": "record", "ms": 0.4}})
    child = table["children"]["cmd_upload"]
    assert child["percent_of_parent"] == pytest.approx(80.0)
    assert child["denominator_parent_ms"] == pytest.approx(0.5)
    assert child["percent_of_compute_call"] == pytest.approx(40.0)
    assert child["denominator_compute_call_ms"] == pytest.approx(1.0)
    assert table["siblings"]["record"]["denominator"] == "compute_call"
    assert table["residual"]["percent_of_compute_call"] == pytest.approx(30.0)
    assert table["percent_sum"] == pytest.approx(100.0)
    assert "denominator" in why


def test_a_share_table_over_a_zero_total_or_an_orphan_child_is_refused():
    rows, _ = phases.compute_call_attribution(_good_events())
    assert "total_ms" in refuses(gate.attribution_shares(_with(rows[0], ("total_ms",), 0.0)))
    assert "not a sibling phase" in refuses(
        gate.attribution_shares(rows[0], {"cmd_upload": {"parent": "prepack", "ms": 0.4}}))
    assert "cannot outlast its parent" in refuses(
        gate.attribution_shares(rows[0], {"cmd_upload": {"parent": "record", "ms": 0.9}}))


def test_a_fully_witnessed_record_is_published():
    published, why = gate.publish(_good_record(), _good_events())
    assert published["calls"] == 1
    assert set(published["witnesses"]) == {
        "public_paths", "weights", "environment", "equivalence", "counters", "attribution"}
    assert published["shares"][0]["children"]["cmd_upload"]["percent_of_parent"] == \
        pytest.approx(80.0)
    assert "every percentage carrying its denominator" in why


def test_a_record_whose_trace_was_never_supplied_is_refused():
    assert "never ran" in refuses(gate.publish(_good_record()))


def test_a_call_missing_a_required_phase_is_refused():
    """D4's "missing required phase", structurally: a dispatch that reached the GPU submitted."""
    events = [e for e in _good_events() if e.get("name") != "vulkan.submit"]
    why = refuses(gate.publish(_good_record(), events))
    assert "required phase(s) absent" in why and "submit" in why


def test_every_blocker_is_reported_and_not_only_the_first():
    """An artifact repaired one blocker per round is one review round per blocker."""
    broken = _with(_without(_good_record(), "counters"), ("machine_quiescence", "verdict"),
                   "CONTENDED")
    broken["model"]["external_data"]["scanned"] = False
    why = refuses(gate.publish(broken, _good_events()))
    assert "3 blocker(s)" in why
    for expected in ("weights:", "environment:", "counters:"):
        assert expected in why


# ---------------------------------------------------------------------------
# 3. MUTATION BATTERY — the protocols, and reimplementations that must fail them
# ---------------------------------------------------------------------------


def _mutant_path_screen_knows_one_home(record):
    """PR #94's screen, preserved: a denylist of the roots its author happened to have."""
    for _where, text in gate._walk_strings(record):
        if re.search(r"C:\\Users\\", text, re.IGNORECASE):
            return None, "refused: absolute filesystem path (known roots)"
    return gate.NO_PRIVATE_PATHS, "clean"


def _mutant_path_screen_accepts_everything(record):
    return gate.NO_PRIVATE_PATHS, "clean"


def _mutant_path_screen_returns_a_look_alike(record):
    for _where, text in gate._walk_strings(record):
        if gate._ABS_PATH.search(text):
            return None, "refused: absolute filesystem path from unknown roots"
    look_alike = frozenset(set(gate.NO_PRIVATE_PATHS))
    assert look_alike == gate.NO_PRIVATE_PATHS and look_alike is not gate.NO_PRIVATE_PATHS
    return look_alike, "clean"


def _mutant_path_screen_fires_on_relative_paths(record):
    for _where, text in gate._walk_strings(record):
        if "/" in text or "\\" in text:
            return None, "refused: absolute filesystem path (known roots)"
    return gate.NO_PRIVATE_PATHS, "clean"


_PATH_MUTANTS = {
    "knows_one_home": _mutant_path_screen_knows_one_home,
    "accepts_everything": _mutant_path_screen_accepts_everything,
    "returns_a_look_alike": _mutant_path_screen_returns_a_look_alike,
    "fires_on_relative_paths": _mutant_path_screen_fires_on_relative_paths,
}


def _path_protocol(fn) -> None:
    selects(fn(_good_record()), gate.NO_PRIVATE_PATHS)
    selects(fn({"a": ["bench/results/x.json", "https://host/a/b", "1/2", "docs/PERF.md"]}),
            gate.NO_PRIVATE_PATHS)
    for planted in (r"C:\Users\someone\m.onnx", r"D:\other-user\m.onnx",
                    "/home/someone/m.onnx", r"\\server\share\m.onnx"):
        why = refuses(fn(_with(_good_record(), ("model", "path"), planted)))
        if "absolute filesystem path" not in why:
            raise PolarityError(f"refusal did not name the defect: {why!r}")


def test_the_real_path_screen_passes_the_protocol():
    _path_protocol(gate.public_path_screen)


@pytest.mark.parametrize("mutant_name", sorted(_PATH_MUTANTS))
def test_a_defective_path_screen_is_caught_by_this_protocol(mutant_name: str):
    with pytest.raises(PolarityError):
        _path_protocol(_PATH_MUTANTS[mutant_name])


def _mutant_shares_one_denominator(row, children=None):
    """The PR #94 defect exactly: the child's share of the TOTAL, labelled as of its PARENT."""
    table, why = gate.attribution_shares(row, children)
    if table is None:
        return None, why
    for c in table["children"].values():
        c["percent_of_parent"] = c["percent_of_compute_call"]
    return table, why


def _mutant_shares_drop_the_denominator(row, children=None):
    table, why = gate.attribution_shares(row, children)
    if table is None:
        return None, why
    for c in table["children"].values():
        c.pop("denominator_parent_ms", None)
    for s in table["siblings"].values():
        s.pop("denominator_ms", None)
    return table, why


def _mutant_shares_assume_a_zero_residual(row, children=None):
    """The residual asserted rather than computed — issue #88's original defect, in the table."""
    table, why = gate.attribution_shares(row, children)
    if table is None:
        return None, why
    table["residual"]["ms"] = 0.0
    table["residual"]["percent_of_compute_call"] = 0.0
    return table, why


def _mutant_shares_rescale_siblings_to_fill_the_call(row, children=None):
    """Absorbs the unattributed residual into the named phases by rescaling them to 100%.

    This is how an attribution table becomes a *complete-looking* one: nothing is missing from
    it, because the thing that was missing was quietly redistributed across the rows that were
    there.
    """
    table, why = gate.attribution_shares(row, children)
    if table is None:
        return None, why
    scale = 100.0 / sum(s["percent_of_compute_call"] for s in table["siblings"].values())
    for s in table["siblings"].values():
        s["percent_of_compute_call"] *= scale
    table["residual"]["percent_of_compute_call"] = 0.0
    table["percent_sum"] = 100.0
    return table, why


def _mutant_shares_accept_an_orphan_child(row, children=None):
    table, why = gate.attribution_shares(row, None)
    if table is None:
        return None, why
    for name, spec in (children or {}).items():
        table["children"][name] = {
            "ms": spec["ms"], "parent": spec["parent"],
            "percent_of_parent": 100.0 * spec["ms"] / row["total_ms"],
            "denominator_parent_ms": row["total_ms"],
            "percent_of_compute_call": 100.0 * spec["ms"] / row["total_ms"],
            "denominator_compute_call_ms": row["total_ms"],
        }
    return table, why


_SHARE_MUTANTS = {
    "one_denominator": _mutant_shares_one_denominator,
    "drop_the_denominator": _mutant_shares_drop_the_denominator,
    "assume_a_zero_residual": _mutant_shares_assume_a_zero_residual,
    "rescale_siblings_to_fill_the_call": _mutant_shares_rescale_siblings_to_fill_the_call,
    "accept_an_orphan_child": _mutant_shares_accept_an_orphan_child,
}


def _share_protocol(fn) -> None:
    rows, _ = phases.compute_call_attribution(_good_events())
    row = rows[0]
    table, why = fn(row, {"cmd_upload": {"parent": "record", "ms": 0.4}})
    if table is None:
        raise PolarityError(f"refused a good row: {why!r}")
    child = table["children"]["cmd_upload"]
    for key in ("percent_of_parent", "denominator_parent_ms", "percent_of_compute_call",
                "denominator_compute_call_ms"):
        if key not in child:
            raise PolarityError(
                f"the child share publishes no {key!r}. A percentage without the number it was "
                f"divided by is the '16.4% of record' defect with the label removed entirely."
            )
    if abs(child["percent_of_parent"] - 80.0) > 1e-9:
        raise PolarityError(
            f"cmd_upload is {child['percent_of_parent']}% of `record`, expected 80.0 — "
            f"0.4 ms inside a 0.5 ms parent. Reporting its share of the 1.0 ms TOTAL under this "
            f"name is the '16.4% of record' error."
        )
    if abs(child["percent_of_compute_call"] - 40.0) > 1e-9:
        raise PolarityError(f"cmd_upload is {child['percent_of_compute_call']} points of the "
                            f"whole call, expected 40.0")
    if abs(child["denominator_parent_ms"] - 0.5) > 1e-9:
        raise PolarityError(
            f"the child's parent denominator is {child['denominator_parent_ms']}, expected 0.5")
    for name, s in table["siblings"].items():
        if "denominator_ms" not in s:
            raise PolarityError(f"sibling {name} publishes a percentage with no denominator")
        if abs(s["percent_of_compute_call"] - 100.0 * s["ms"] / row["total_ms"]) > 1e-9:
            raise PolarityError(
                f"sibling {name}'s share does not equal its own ms over the call's total — it "
                f"has been rescaled, which is how a residual gets absorbed into the named rows."
            )
    # A residual that was RECONCILED rather than computed is the one thing that must not pass.
    computed = 100.0 * row["residual_ms"] / row["total_ms"]
    if abs(table["residual"]["percent_of_compute_call"] - computed) > 1e-9:
        raise PolarityError(
            f"the residual share is {table['residual']['percent_of_compute_call']}, but the "
            f"decomposition computes {computed}. A residual chosen to make the column sum to "
            f"100% is not a measurement."
        )
    # …and a refusal it must still produce.
    refuses(fn(row, {"cmd_upload": {"parent": "prepack", "ms": 0.4}}),
            because="the named parent is not a sibling phase of this call")


def test_the_real_share_table_passes_the_protocol():
    _share_protocol(gate.attribution_shares)


@pytest.mark.parametrize("mutant_name", sorted(_SHARE_MUTANTS))
def test_a_defective_share_table_is_caught_by_this_protocol(mutant_name: str):
    with pytest.raises(PolarityError):
        _share_protocol(_SHARE_MUTANTS[mutant_name])


# ---------------------------------------------------------------------------
# 4. THE STRUCTURAL PROPERTY — a refusal has no numbers to lift out of it
# ---------------------------------------------------------------------------


def _numbers_in(text: str) -> "set[str]":
    return set(re.findall(r"\d+\.\d+", text))


@pytest.mark.parametrize("break_it", [
    lambda r: _without(r, "counters"),
    lambda r: _with(r, ("machine_quiescence", "verdict"), "CONTENDED"),
    lambda r: _with(r, ("counters", "abi_version"), _ABI - 1),
    lambda r: _with(r, ("equivalence",), [{"case": "decode-past1024", "repeat": 0,
                                           "verdict": "MATCH", "outputs_compared": 1,
                                           "outputs_total": 65}]),
    lambda r: _with(r, ("model", "external_data", "scanned"), False),
    lambda r: _with(r, ("model", "path"), r"D:\other-user\m.onnx"),
    lambda r: _with(r, ("taken_at",), "2026-01-01T00:00:00+00:00"),
])
def test_a_refused_record_publishes_no_share_at_all(break_it):
    """The Fact Checker's D-blocker, as a property rather than an inspection.

    "Refused raw records retained complete-looking shares." Here the refused value **is**
    ``None``: there is no table, no percentage and no per-phase figure to copy out, and the
    refusal sentence does not contain the shares the accepted path would have produced.
    """
    published, why = gate.publish(break_it(_good_record()), _good_events())
    assert published is None, "a refused record must not carry an attribution at all"

    accepted, _ = gate.publish(_good_record(), _good_events())
    share_numbers = _numbers_in(str(accepted["shares"]))
    assert share_numbers, "the accept-path fixture produces no shares; this test proves nothing"
    leaked = share_numbers & _numbers_in(why)
    assert not leaked, f"the refusal text leaked share values: {sorted(leaked)}"


def test_the_refusal_names_every_blocker_so_it_can_be_acted_on():
    """Fail closed is not fail silent. A refusal with no reason is not reviewable."""
    _, why = gate.publish({"model": {"path": r"D:\other-user\m.onnx"}}, [])
    assert why.startswith("REFUSED")
    for expected in ("public_paths", "weights", "environment", "equivalence", "counters",
                     "attribution"):
        assert f"{expected}:" in why, f"{expected} did not report a blocker: {why}"


def test_an_empty_record_still_reports_every_blocker_it_can_see():
    """A record with no strings in it has no paths to disclose, and the screen says so.

    Worth asserting rather than assuming: a screen that fired on the empty record would be
    firing on its own absence of input, which is the vacuous positive this project keeps
    finding.
    """
    _, why = gate.publish({}, [])
    assert "public_paths:" not in why
    for expected in ("weights", "environment", "equivalence", "counters", "attribution"):
        assert f"{expected}:" in why
