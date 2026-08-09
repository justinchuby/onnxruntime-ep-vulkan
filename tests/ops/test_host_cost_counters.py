"""The host-cost instruments must fire from the *production* Compute path, not from a test double.

WHY THIS EXISTS
===============
Issue #88. Decode wall time was ~41% device and ~59% unnamed. The EP had no whole-``Compute``
timer, no descriptor/submit counters, and ``Tracer::record_path`` — the mechanism whose entire
purpose is to witness whether a command buffer was replayed or re-recorded — had **zero
production callers**. It was listed under ``uninvoked`` in ``rust/tools/instrument_census.json``
and ``log_summary`` printed "record-path breakdown NOT WIRED" on every run, forever.

An instrument with no caller is worse than a missing instrument, because the report still has a
row for it. This lane exists so that the new instruments cannot quietly return to that state.

WHAT THIS LANE ASSERTS, AND WHY IT IS NOT A TAUTOLOGY
=====================================================
Every reading here comes from a **child process that ran a real ORT session on a real Vulkan
device**. Nothing reads the Rust source. Nothing constructs a ``Tracer`` by hand. If a call site
is deleted from ``vk/session.rs`` or ``vk/pipeline.rs``, the corresponding count goes to zero and
a case here goes red — that is the mutation this lane is built to catch.

WHY A CHILD PROCESS
===================
Two reasons, and both are correctness rather than tidiness.

1. The tracer is a ``OnceLock`` initialised from ``ONNXRUNTIME_EP_VULKAN_TRACE`` on **first
   touch**. Any earlier test in the same pytest process has already initialised it with tracing
   off, so setting the variable in-process produces a session that silently traces nothing. The
   first draft of this lane did exactly that and reported "no trace file was written" — the
   harness was broken, not the product.
2. The counters are process-cumulative. In a fresh child, the absolute reading *is* this
   session's own contribution, so no differencing is required and no earlier test can lend it a
   number.

``R10`` — a value must vary with its input — is exercised directly by
:func:`test_the_submit_counter_tracks_how_many_times_the_graph_actually_ran`, which runs the same
graph once and then four times and requires the readings to differ by the right amount. A counter
that were hardcoded, or bumped once at teardown, passes a presence check and fails that one.

WHAT THIS LANE DELIBERATELY DOES NOT ASSERT
===========================================
No timing claim. Phase durations are wall clock on a shared machine; this lane asserts that the
phases are *emitted and named*, never that any of them is fast or that one is larger than
another. Attribution admissibility is a structural property — were all the parts observed? — and
is checked as such.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import onnx
import onnx.helper as oh
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"

#: How long one child gets. Generous: a cold child pays shader compilation and device creation.
CHILD_TIMEOUT_S = 600

#: Every counter this lane reads. Named here so that a rename in ``counters.rs`` surfaces as a
#: named failure rather than as a silently-absent ``.get()`` returning ``None``.
HOST_COST_COUNTERS = (
    "descriptor_pools_created",
    "descriptor_sets_allocated",
    "descriptor_writes",
    "command_buffers_recorded",
    "queue_submits",
    "record_path_first_record",
    "record_path_replay",
    "record_path_rerecord",
)

#: The host phases the Compute path must emit on every claimed island. ``execute`` is absent on
#: purpose: it is the *whole*, folded into the phase table without a span of its own because
#: ``vulkan.subgraph`` already brackets the same interval, and two spans over one interval is how
#: a decomposition learns to double-count.
REQUIRED_COMPUTE_PHASES = (
    "vulkan.prepare",
    "vulkan.buffer_alloc",
    "vulkan.record",
    "vulkan.cmd_alloc",
    "vulkan.submit",
    "vulkan.fence_wait",
    "vulkan.writeback",
)

#: Graph shape used by every case here: a straight-line f32 ``Add`` chain. ``Add`` is claimed on
#: every driver this repo supports, so a red result is about the instruments and never about op
#: coverage. Three nodes so that "one descriptor set per dispatch" is a ratio and not an identity
#: between two ones.
CHAIN_NODES = 3
CHAIN_SHAPE = [4, 8]


# ---------------------------------------------------------------------------------------------
# The model, built identically in parent and child
# ---------------------------------------------------------------------------------------------


def _chain_model(n_nodes: int = CHAIN_NODES) -> bytes:
    nodes = []
    prev = "x"
    for i in range(n_nodes):
        out = f"t{i}" if i < n_nodes - 1 else "y"
        nodes.append(oh.make_node("Add", [prev, "b"], [out], name=f"add{i}"))
        prev = out
    graph = oh.make_graph(
        nodes,
        "host_cost_chain",
        [
            oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, CHAIN_SHAPE),
            oh.make_tensor_value_info("b", onnx.TensorProto.FLOAT, CHAIN_SHAPE),
        ],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, CHAIN_SHAPE)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


def _feeds() -> dict:
    n = int(np.prod(CHAIN_SHAPE))
    return {
        "x": np.arange(n, dtype=np.float32).reshape(CHAIN_SHAPE),
        "b": np.ones(CHAIN_SHAPE, dtype=np.float32),
    }


# ---------------------------------------------------------------------------------------------
# Child: one fresh process, one session, N runs
# ---------------------------------------------------------------------------------------------


def _child_main(argv: list[str]) -> int:
    """``python test_host_cost_counters.py --child <runs> <outputs.npy>``.

    Deliberately free of pytest: this must run even when the parent's fixtures are unavailable,
    so that a failure here is legible as a failure of the EP rather than of the harness.
    """
    runs = int(argv[0])
    out_path = pathlib.Path(argv[1])

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import onnxruntime as ort  # noqa: PLC0415 — imported after the env is set

    import _models as m  # noqa: PLC0415

    # The child has none of conftest's fixtures, so it registers the EP itself. Registration is
    # process-scoped in ORT, and a child that skipped this would silently run on CPU and report a
    # counters file full of zeros — a broken harness spelled exactly like a broken product.
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib or not pathlib.Path(lib).is_file():
        print(
            "ONNXRUNTIME_VULKAN_EP_LIB is not set or does not name a file; the child cannot "
            "register the EP and will not pretend to have measured it",
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        ort.register_execution_provider_library(m.EP_NAME, str(pathlib.Path(lib).resolve()))
    except Exception as exc:  # noqa: BLE001 — re-raised unless it is the benign re-registration
        if "already registered" not in str(exc):
            raise

    session = ort.InferenceSession(
        _chain_model(), m._make_session_options(), providers=m.EP_PROVIDERS
    )
    m.assert_ep_in_session(session)
    feeds = _feeds()
    outputs = []
    for _ in range(runs):
        outputs = session.run(None, feeds)
    del session  # counters and the trace are flushed at EP teardown

    if outputs:
        np.save(out_path, np.asarray(outputs[0]))
    return 0


# ---------------------------------------------------------------------------------------------
# Parent: spawn, then read what the child left behind
# ---------------------------------------------------------------------------------------------


class Reading:
    """What one child process produced: counters, trace events, output, and its own exit state."""

    def __init__(self, counters, events, output, returncode, transcript):
        self.counters = counters
        self.events = events
        self.output = output
        self.returncode = returncode
        self.transcript = transcript

    def require_ok(self, what: str) -> None:
        if self.returncode != 0:
            pytest.fail(
                f"the child process for {what} exited {self.returncode}; "
                f"transcript tail:\n{self.transcript[-3000:]}"
            )
        if self.counters is None:
            pytest.fail(
                f"the child process for {what} wrote no counters file. This is a harness failure, "
                f"not a zero reading.\nTranscript tail:\n{self.transcript[-3000:]}"
            )

    def counter(self, name: str) -> int:
        if name not in self.counters:
            pytest.fail(
                f"the counters document has no field {name!r} — a counter was renamed or removed "
                f"without updating this lane. Host-cost fields present: "
                f"{sorted(k for k in self.counters if k in HOST_COST_COUNTERS)}"
            )
        return self.counters[name]

    def require_trace(self, what: str) -> list:
        if self.events is None:
            pytest.fail(
                f"tracing was requested for {what} but the child wrote no trace file.\n"
                f"Transcript tail:\n{self.transcript[-3000:]}"
            )
        return self.events


def _run_child(tmp_path: pathlib.Path, tag: str, *, runs: int, trace: bool) -> Reading:
    counters_path = tmp_path / f"counters_{tag}.json"
    trace_path = tmp_path / f"trace_{tag}.json"
    out_path = tmp_path / f"out_{tag}.npy"

    env = dict(os.environ)
    env[COUNTERS_ENV] = str(counters_path)
    if trace:
        env[TRACE_ENV] = str(trace_path)
    else:
        env.pop(TRACE_ENV, None)

    proc = subprocess.run(
        [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--child",
            str(runs),
            str(out_path),
        ],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=CHILD_TIMEOUT_S,
        cwd=str(REPO),
        check=False,
    )
    transcript = (proc.stdout or "") + "\n" + (proc.stderr or "")

    counters = None
    if counters_path.is_file():
        try:
            counters = json.loads(counters_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            counters = None

    events = None
    if trace and trace_path.is_file():
        doc = json.loads(trace_path.read_text(encoding="utf-8"))
        events = doc["traceEvents"] if isinstance(doc, dict) else doc

    output = np.load(out_path) if out_path.is_file() else None
    return Reading(counters, events, output, proc.returncode, transcript)


@pytest.fixture(scope="module")
def three_runs(tmp_path_factory) -> Reading:
    """One child, three inferences, tracing on. Shared by the cases that only read it.

    ``require_vulkan`` is function-scoped, so it cannot be requested here. The child registers the
    EP itself and exits non-zero if it cannot, and :meth:`Reading.require_ok` turns that into a
    named failure — the availability check is therefore in the child, where the session actually
    is, rather than in a parent process that never opens a device.
    """
    r = _run_child(tmp_path_factory.mktemp("hostcost3"), "shared3", runs=3, trace=True)
    r.require_ok("the shared three-run child")
    return r


# ---------------------------------------------------------------------------------------------
# The counters fire from production
# ---------------------------------------------------------------------------------------------


def test_every_host_cost_counter_is_reached_by_a_real_run(three_runs):
    """Each new counter must move on a real inference — an unmoved counter has no caller."""
    silent = [
        k
        for k in HOST_COST_COUNTERS
        # ``record_path_replay`` is structurally unreachable in this build; see its own case.
        if k != "record_path_replay" and three_runs.counter(k) == 0
    ]
    readings = {k: three_runs.counter(k) for k in HOST_COST_COUNTERS}
    assert not silent, (
        "these host-cost counters did not move across three real inferences in a fresh process, "
        f"so nothing in the production Compute path increments them: {silent}. "
        f"All readings: {readings}"
    )


def test_the_submit_counter_tracks_how_many_times_the_graph_actually_ran(require_vulkan, tmp_path):
    """R10: the value must vary with its input.

    A counter bumped once at session teardown, or a constant, passes a presence check. It cannot
    pass this one: one inference and four inferences of the same graph must differ by exactly the
    three extra Computes.
    """
    one = _run_child(tmp_path, "scale1", runs=1, trace=False)
    one.require_ok("the one-run child")
    four = _run_child(tmp_path, "scale4", runs=4, trace=False)
    four.require_ok("the four-run child")

    assert one.counter("queue_submits") > 0, "a single inference recorded no queue submit"
    assert four.counter("queue_submits") == 4 * one.counter("queue_submits"), (
        "queue_submits did not scale with the number of inferences — "
        f"1 run gave {one.counter('queue_submits')}, 4 runs gave {four.counter('queue_submits')}, "
        f"expected {4 * one.counter('queue_submits')}"
    )
    assert four.counter("dispatches_executed") == 4 * one.counter("dispatches_executed"), (
        "dispatches did not scale with the number of inferences — "
        f"{one.counter('dispatches_executed')} vs {four.counter('dispatches_executed')}"
    )
    assert four.counter("descriptor_sets_allocated") == 4 * one.counter(
        "descriptor_sets_allocated"
    ), (
        "descriptor set allocation did not scale with the number of inferences — "
        f"{one.counter('descriptor_sets_allocated')} vs "
        f"{four.counter('descriptor_sets_allocated')}. If this ever stops scaling because sets are "
        "being reused, that is the optimisation landing, and this lane should then be updated "
        "deliberately rather than left to pass by accident."
    )


def test_one_command_buffer_is_recorded_per_submit(three_runs):
    """The recording counter and the submit counter are taken at two different call sites.

    ``record_command_buffer_recorded`` fires inside the ``cmd_alloc`` phase around
    ``CommandPool::begin``; ``record_queue_submit`` fires after the submit block. If either lost
    its call site the identity breaks, and it breaks in a direction that names which one.
    """
    assert three_runs.counter("command_buffers_recorded") == three_runs.counter("queue_submits"), (
        "command buffer recordings and queue submits disagree: "
        f"{three_runs.counter('command_buffers_recorded')} recorded vs "
        f"{three_runs.counter('queue_submits')} submitted. In this build every Compute "
        "unconditionally resets and re-begins its command buffer and submits it exactly once, so "
        "these must be equal."
    )


def test_the_record_path_witness_accounts_for_every_recording(three_runs):
    """``record_path`` is the mechanism that had no caller. This is the lane that proves it does.

    Every command-buffer recording must be classified as exactly one of first-record, replay, or
    re-record. A witness that classifies fewer recordings than happened is a witness with a hole,
    and a hole in this particular witness is what made "is the command buffer being rebuilt on
    every call?" unanswerable from a production run.
    """
    classified = (
        three_runs.counter("record_path_first_record")
        + three_runs.counter("record_path_replay")
        + three_runs.counter("record_path_rerecord")
    )
    assert classified == three_runs.counter("command_buffers_recorded"), (
        f"{three_runs.counter('command_buffers_recorded')} command buffers were recorded but "
        f"{classified} were classified by the record-path witness "
        f"(first={three_runs.counter('record_path_first_record')}, "
        f"replay={three_runs.counter('record_path_replay')}, "
        f"rerecord={three_runs.counter('record_path_rerecord')})"
    )


def test_the_first_recording_of_a_subgraph_is_distinguished_from_the_later_ones(three_runs):
    """Three runs of a one-island graph is one first-record and two re-records.

    This is the observation the whole issue turns on: if warm calls were replaying a cached
    command buffer, ``rerecord`` would be zero. It is not zero, and this lane is where that stops
    being a hypothesis and becomes a reading.
    """
    assert three_runs.counter("record_path_first_record") == 1, (
        "a single-island graph run three times in one fresh process should report exactly one "
        f"first-record, got {three_runs.counter('record_path_first_record')}"
    )
    assert three_runs.counter("record_path_rerecord") == 2, (
        "the two warm calls should each report a re-record, got "
        f"{three_runs.counter('record_path_rerecord')} — if this is 0 the command buffer is being "
        "reused and the premise of issue #88 has changed"
    )


def test_no_recording_claims_to_be_a_replay_in_a_build_that_cannot_replay(three_runs):
    """``CommandPool::begin`` resets and re-begins unconditionally, so replay is unreachable here.

    Asserting the zero is not decoration. If a future change adds a replay path and forgets to
    change the docs, this goes red and points at the sentence that is now wrong.
    """
    assert three_runs.counter("record_path_replay") == 0, (
        f"{three_runs.counter('record_path_replay')} replays were reported, but no code path in "
        "this build reuses a recorded command buffer — either replay landed (in which case update "
        "docs/DESIGN.md and docs/PERF.md) or the witness is misclassifying"
    )


def test_a_descriptor_set_is_allocated_for_every_dispatch(three_runs):
    """Descriptor churn is the leading host-cost hypothesis, so the ratio must be observable.

    ``descriptor_sets_allocated`` is bumped in ``DispatchDescriptorPool::allocate_and_write`` and
    ``dispatches_executed`` is bumped on the dispatch itself — two independent call sites. Their
    equality is the statement "one fresh set per dispatch, every time".
    """
    dispatches = three_runs.counter("dispatches_executed")
    assert dispatches > 0, "no dispatches were executed"
    assert three_runs.counter("descriptor_sets_allocated") == dispatches, (
        f"{three_runs.counter('descriptor_sets_allocated')} descriptor sets for {dispatches} "
        "dispatches — this build allocates exactly one set per dispatch; a divergence means one "
        "of the two counters lost its call site, or a set is being reused"
    )
    assert three_runs.counter("descriptor_pools_created") == three_runs.counter(
        "descriptor_sets_allocated"
    ), (
        f"{three_runs.counter('descriptor_pools_created')} pools for "
        f"{three_runs.counter('descriptor_sets_allocated')} sets — this build creates a fresh pool "
        "per dispatch; a divergence means pooling changed"
    )
    assert three_runs.counter("descriptor_writes") >= three_runs.counter(
        "descriptor_sets_allocated"
    ), (
        f"{three_runs.counter('descriptor_writes')} writes for "
        f"{three_runs.counter('descriptor_sets_allocated')} sets — every set binds at least one "
        "buffer, so writes can never be fewer than sets"
    )


def test_the_counters_document_is_versioned_with_the_fields_it_carries(three_runs):
    """A counters document whose ABI version did not move cannot be told from an older one.

    The eight host-cost fields were appended in ABI v9. A reader that finds the fields but an
    older version stamp is reading a struct laid out differently from the one it thinks it has —
    the exact defect ``test_counters_abi_singleton.py`` exists to prevent, seen from the wire side.
    """
    version = three_runs.counters.get("abi_version")
    assert version is not None, (
        f"the counters document carries no abi_version; keys: {sorted(three_runs.counters)[:40]}"
    )
    assert version >= 9, (
        f"the counters document reports ABI v{version} but carries the v9 host-cost fields "
        f"{[k for k in HOST_COST_COUNTERS if k in three_runs.counters]}"
    )


# ---------------------------------------------------------------------------------------------
# The phases fire from production
# ---------------------------------------------------------------------------------------------


def test_every_compute_phase_is_emitted_by_a_real_run(three_runs):
    """A phase that emits no span on a real inference is a row in a report with nothing behind it."""
    events = three_runs.require_trace("the compute-phase case")
    seen = {e.get("name") for e in events if e.get("ph") == "X"}
    missing = [p for p in REQUIRED_COMPUTE_PHASES if p not in seen]
    assert not missing, (
        f"these phases emitted no span during a real inference: {missing}. "
        f"Phases actually seen: {sorted(str(n) for n in seen if str(n).startswith('vulkan.'))}"
    )


def test_the_whole_is_not_emitted_as_a_span_beside_its_own_parts(three_runs):
    """``execute`` is the total. If it also emitted a span, every host sum would double.

    ``bench/phases.py`` sums sibling spans. A ``vulkan.execute`` span would be summed alongside
    the parts it contains and the decomposition would appear to close at 200%.
    """
    events = three_runs.require_trace("the no-total-span case")
    offenders = [e for e in events if e.get("ph") == "X" and e.get("name") == "vulkan.execute"]
    assert not offenders, (
        f"{len(offenders)} vulkan.execute spans were emitted; the total must be folded into the "
        "phase table without a span, because vulkan.subgraph already brackets the same interval"
    )


def test_every_phase_span_declares_what_it_is_nested_in(three_runs):
    """A span without its caveat cannot be summed safely, and will be summed anyway."""
    events = three_runs.require_trace("the caveat case")
    bare = [
        e.get("name")
        for e in events
        if e.get("ph") == "X"
        and str(e.get("name", "")).startswith("vulkan.")
        and e.get("name") != "vulkan.subgraph"
        and not (e.get("args") or {}).get("caveat")
    ]
    assert not bare, f"these phase spans carried no caveat: {sorted(set(map(str, bare)))}"


def test_the_named_parts_of_a_recording_are_declared_as_parts_not_as_peers(three_runs):
    """``cmd_alloc`` happens inside ``record``; its caveat must say so in the load-bearing form.

    ``bench/phases.py`` reads the ``host/sub-record:`` prefix to decide what may be summed. A
    sub-record phase whose caveat forgot the prefix would be summed beside its own parent; a
    sibling that wrongly claimed the prefix would be reported as a containment MISMATCH.
    """
    events = three_runs.require_trace("the sub-record caveat case")
    caveats = {
        e["name"]: (e.get("args") or {}).get("caveat", "")
        for e in events
        if e.get("ph") == "X" and str(e.get("name", "")).startswith("vulkan.")
    }
    assert "vulkan.cmd_alloc" in caveats, "vulkan.cmd_alloc emitted no span"
    assert caveats["vulkan.cmd_alloc"].startswith("host/sub-record:"), (
        "vulkan.cmd_alloc is recorded inside the record phase but its caveat does not begin with "
        f"'host/sub-record:' — bench/phases.py will sum it as a sibling. Caveat was: "
        f"{caveats['vulkan.cmd_alloc']!r}"
    )
    for sibling in ("vulkan.prepare", "vulkan.buffer_alloc", "vulkan.writeback"):
        assert sibling in caveats, f"{sibling} emitted no span"
        assert not caveats[sibling].startswith("host/sub-record:"), (
            f"{sibling} is not recorded inside the record phase but claims to be. "
            "bench/phases.py checks that claim against timestamp containment and will report a "
            f"MISMATCH. Caveat was: {caveats[sibling]!r}"
        )


def test_the_session_summary_reports_the_record_path_as_wired(three_runs):
    """The summary used to print "record-path breakdown NOT WIRED" forever. It must not again."""
    events = three_runs.require_trace("the summary case")
    summaries = [e for e in events if e.get("name") == "vulkan.session_summary"]
    assert summaries, "no vulkan.session_summary event was emitted"
    args = summaries[-1].get("args") or {}

    assert args.get("record_path_wired") is True, (
        "the session summary reports the record-path witness as unwired after a real inference "
        f"reached it; args={args}"
    )
    assert args.get("execute_calls", 0) > 0, (
        f"the summary reports {args.get('execute_calls')!r} Execute calls after three inferences"
    )
    assert args.get("execute_us", 0) > 0, (
        "the whole-Execute timer reported zero elapsed microseconds across three inferences"
    )


def test_the_summary_never_reports_a_share_it_did_not_measure(three_runs):
    """Attribution must be admissible only when every named part was actually observed.

    This is the structural half of R11: a decomposition that *appears* to close is the hardest
    kind of wrong. The verdict and the residual must agree — an admissible verdict alongside a
    negative or absent residual is the failure this catches.
    """
    events = three_runs.require_trace("the admissibility case")
    summaries = [e for e in events if e.get("name") == "vulkan.session_summary"]
    assert summaries, "no vulkan.session_summary event was emitted"
    args = summaries[-1].get("args") or {}

    admissible = args.get("attribution_admissible")
    assert admissible is not None, f"the summary carries no admissibility verdict; args={args}"

    if admissible:
        assert not args.get("attribution_refusal"), (
            "the attribution is marked admissible and also carries a refusal reason: "
            f"{args.get('attribution_refusal')!r}"
        )
        attributed = args.get("attributed_us")
        execute = args.get("execute_us")
        assert attributed is not None and execute is not None, (
            f"an admissible attribution must state both parts and whole; args={args}"
        )
        assert attributed <= execute, (
            f"the named parts ({attributed}us) exceed the whole ({execute}us) and the verdict is "
            "still admissible — parts larger than the whole must be refused, not clamped"
        )
        assert args.get("unattributed_us") == execute - attributed, (
            f"unattributed_us={args.get('unattributed_us')} is not execute_us - attributed_us "
            f"({execute} - {attributed})"
        )
    else:
        assert args.get("attribution_refusal"), (
            "the attribution is inadmissible but states no reason — an instrument's failure must "
            "not be spelled the same way as its finding"
        )


def test_the_rendered_summary_does_not_claim_completeness_it_cannot_support(three_runs):
    """The human-readable summary is what a reader actually sees, so it carries the same verdict.

    A machine-readable ``attribution_admissible: false`` beside a printed table that reads as a
    closed decomposition is precisely the failure mode R13 names: the instrument's failure spelled
    the same way as its finding.
    """
    text = three_runs.transcript
    assert "Vulkan EP session summary" in text, (
        f"the child produced no rendered session summary at all; transcript tail:\n{text[-3000:]}"
    )
    assert "record-path breakdown NOT WIRED" not in text, (
        "the rendered summary still says the record-path breakdown is not wired"
    )
    assert "ATTRIBUTION:" in text, (
        "the rendered summary states no attribution verdict — a table of host phases with no "
        "verdict reads as complete whether or not it is"
    )
    assert "UNATTRIBUTED" in text, (
        "the rendered summary names no unattributed remainder; a host decomposition that lists "
        "only the parts it found reads as if it found all of them"
    )


def test_a_phase_cannot_go_silent_while_the_report_still_reads_as_complete(three_runs):
    """The completeness verdict must be *derived from* which phases actually spoke.

    This is the case that makes the whole lane load-bearing rather than decorative. Deleting a
    phase guard from ``vk/session.rs`` must not merely remove a row: it must flip the verdict, in
    both renderings, and name the phase that vanished.

    Verified by mutation while writing this lane — removing ``t.phase(Phase::Prepare)`` produced::

        ATTRIBUTION: NOT ADMISSIBLE — INCOMPLETE: 3 Compute call(s) ran and these top-level
        phases recorded none: prepare.

    so the two directions below are both known to be reachable, not merely asserted.
    """
    events = three_runs.require_trace("the completeness-linkage case")
    text = three_runs.transcript

    seen = {e.get("name") for e in events if e.get("ph") == "X"}
    silent = [p for p in REQUIRED_COMPUTE_PHASES if p not in seen]

    summaries = [e for e in events if e.get("name") == "vulkan.session_summary"]
    assert summaries, "no vulkan.session_summary event was emitted"
    admissible = bool((summaries[-1].get("args") or {}).get("attribution_admissible"))

    if silent:
        assert not admissible, (
            f"these phases emitted no span — {silent} — and the attribution still calls itself "
            "admissible. A report that keeps its completeness claim while one of its named parts "
            "disappears is the exact defect this lane exists to prevent."
        )
        assert "NOT ADMISSIBLE" in text, (
            "the machine-readable verdict is inadmissible but the rendered summary does not say "
            f"so; rendered tail:\n{text[-2000:]}"
        )
        for phase in silent:
            assert phase.removeprefix("vulkan.") in text, (
                f"the rendered refusal does not name {phase}, so a reader cannot tell which part "
                "went missing"
            )
    else:
        assert admissible, (
            "every named phase emitted a span and the whole was measured, yet the attribution "
            "refuses itself. Refusal reason: "
            f"{(summaries[-1].get('args') or {}).get('attribution_refusal')!r}. A refusal that "
            "fires on a healthy run makes every real refusal unreadable."
        )
        assert "NOT ADMISSIBLE" not in text, (
            f"the machine verdict is admissible but the rendered summary refuses; tail:\n"
            f"{text[-2000:]}"
        )


def test_tracing_does_not_change_what_the_graph_computes(require_vulkan, tmp_path):
    """Instrumentation that perturbs the result is not instrumentation.

    The same graph and the same feeds, run in two fresh processes — one with tracing off and one
    with tracing on — must produce bit-identical outputs.
    """
    off = _run_child(tmp_path, "eq_off", runs=1, trace=False)
    off.require_ok("the tracing-off child")
    on = _run_child(tmp_path, "eq_on", runs=1, trace=True)
    on.require_ok("the tracing-on child")

    assert off.output is not None and on.output is not None, "a child produced no output array"
    assert off.output.shape == on.output.shape, (
        f"output shape changed with tracing: {off.output.shape} vs {on.output.shape}"
    )
    differing = int(np.count_nonzero(off.output.view(np.uint32) != on.output.view(np.uint32)))
    assert differing == 0, f"tracing changed {differing} output elements bit-for-bit"


def test_the_counters_are_recorded_whether_or_not_tracing_is_on(require_vulkan, tmp_path):
    """The counters must not be a side effect of the trace switch.

    If the counters only moved when ``ONNXRUNTIME_EP_VULKAN_TRACE`` was set, every counter reading
    in CI — where tracing is off — would be a zero that looked like a measurement.
    """
    off = _run_child(tmp_path, "cnt_off", runs=2, trace=False)
    off.require_ok("the tracing-off counters child")
    on = _run_child(tmp_path, "cnt_on", runs=2, trace=True)
    on.require_ok("the tracing-on counters child")

    for name in HOST_COST_COUNTERS:
        assert off.counter(name) == on.counter(name), (
            f"{name} read {off.counter(name)} with tracing off and {on.counter(name)} with tracing "
            "on, for identical work — the counter is coupled to the trace switch"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        raise SystemExit(_child_main(sys.argv[2:]))
    raise SystemExit(
        "this module is a pytest lane; run it under pytest, or with --child <runs> <out.npy>"
    )
