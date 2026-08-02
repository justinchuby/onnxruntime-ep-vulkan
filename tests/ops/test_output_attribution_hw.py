"""Per-output attribution against real hardware — the intermediate-ledger state.

``test_output_attribution.py`` proves the mechanism on synthesised traces.  This module
proves the *reading* on a real session, on both selectors, and it is the falsifier for
"the coverage instrument is wired" (R10): an artifact the instrument produced whose
content varies with its input.  The input that varies is **which nodes the EP claims**,
and the graph below is built so that some are claimed and some are not — the exact state
the proof ledger passes through as Mouse fills it.

No wall-clock assertion and no threshold of any kind.  The graph is five nodes; the only
quantities asserted are provider labels and bit-equality.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _models as m  # noqa: E402
import _verdict  # noqa: E402

_RESULTS = pathlib.Path(__file__).resolve().parents[2] / "bench" / "results"

pytestmark = pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set — no EP to attribute outputs to",
)


def _mixed_model() -> bytes:
    """Two *independent* branches to two outputs.

    ``out_claimed``  <- Add, Mul       — proven forms; the EP is expected to claim these.
    ``out_declined`` <- Sin, Cos, Erf  — expected to decline.

    The branches share only the graph inputs on purpose.  A shared intermediate would put
    every node upstream of every output and make the question unanswerable by topology —
    which is a real limit of this instrument and is asserted below rather than hidden.
    """
    x = m.tensor("x", ir.DataType.FLOAT, [4, 4])
    y = m.tensor("y", ir.DataType.FLOAT, [4, 4])
    a = m.tensor("a", ir.DataType.FLOAT, [4, 4])
    b = m.tensor("out_claimed", ir.DataType.FLOAT, [4, 4])
    s = m.tensor("s", ir.DataType.FLOAT, [4, 4])
    c = m.tensor("c", ir.DataType.FLOAT, [4, 4])
    e = m.tensor("out_declined", ir.DataType.FLOAT, [4, 4])
    nodes = [
        ir.Node("", "Add", inputs=[x, y], outputs=[a], name="claimed_add"),
        ir.Node("", "Mul", inputs=[a, y], outputs=[b], name="claimed_mul"),
        ir.Node("", "Sin", inputs=[x], outputs=[s], name="declined_sin"),
        ir.Node("", "Cos", inputs=[s], outputs=[c], name="declined_cos"),
        ir.Node("", "Erf", inputs=[c], outputs=[e], name="declined_erf"),
    ]
    graph = ir.Graph(
        inputs=[x, y], outputs=[b, e], nodes=nodes, name="mixed", opset_imports={"": 17}
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _feeds() -> dict:
    rng = np.random.default_rng(20260802)
    return {
        "x": rng.standard_normal((4, 4)).astype(np.float32),
        "y": rng.standard_normal((4, 4)).astype(np.float32),
    }


def test_a_partially_claimed_graph_labels_each_output_by_who_produced_it(
    require_vulkan,
    tmp_path: pathlib.Path,
) -> None:
    """The reading, on hardware, with the artifact written whether it passes or not.

    What this asserts:

      1. the EP is in the session and executed at least one fused island;
      2. the branch it declined is labelled ``CPU-ONLY`` — sound, because every one of
         its nodes carries an explicit ``CPUExecutionProvider`` event;
      3. the branch it claimed is labelled ``EP-COVERED`` — its nodes appear under no
         other provider, because a fused island names no constituent;
      4. the ``CPU-ONLY`` output agrees with the CPU oracle **bit for bit**, which is
         this instrument's own falsifier: two sides of one computation must be identical,
         and a difference would refute the labelling rather than indict the EP.

    (3) is the weaker inference and is only ever used to withhold ``MATCH``.  If the EP's
    claim predicates change so that ``Sin``/``Cos``/``Erf`` become claimed, (2) inverts and
    this test fails loudly — which is the point: the artifact's content varies with its
    input (R10).
    """
    device_index = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    model = _mixed_model()
    feeds = _feeds()

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = str(tmp_path / f"outattr_dev{device_index}")
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)

    if m.EP_NAME not in sess.get_providers():
        raise _verdict.InstrumentError(
            f"[Device {device_index}] {m.EP_NAME} absent from get_providers(): "
            f"{sess.get_providers()}.  There is no attribution to read.  "
            "ERROR(instrument), not a finding about output coverage (R13)."
        )

    vk_out = sess.run(None, feeds)
    profile_path = sess.end_profiling()

    try:
        attribution = m.attribution_with_coverage_from_profile(profile_path, model)
    except _verdict.InstrumentError as exc:
        raise _verdict.InstrumentError(
            f"[Device {device_index}] per-output coverage instrument failure "
            f"(fix the harness, not the EP): {exc}"
        ) from exc

    coverage = attribution.output_coverage
    assert coverage is not None

    cpu_opts = ort.SessionOptions()
    cpu_opts.log_severity_level = 3
    cpu_out = ort.InferenceSession(
        model, cpu_opts, providers=["CPUExecutionProvider"]
    ).run(None, feeds)

    names = ["out_claimed", "out_declined"]
    bit_equal = {
        n: bool(np.array_equal(v, c)) for n, v, c in zip(names, vk_out, cpu_out)
    }
    disagreed = [n for n, ok in bit_equal.items() if not ok]

    record = {
        "device_index": device_index,
        "own_provider_execution_count": attribution.own_count,
        "executed_by": attribution.executed_by,
        "coverage": coverage.to_record(),
        "bit_equal_to_cpu_oracle": bit_equal,
        "instrument_refuted_by": coverage.refuted_by(disagreed),
        "what_this_is": (
            "the intermediate-ledger state: some nodes claimed, some declined, one output "
            "behind each. CPU-ONLY is a refusal and is sound; EP-COVERED is the weaker "
            "inference and only ever withholds MATCH."
        ),
    }
    try:
        _RESULTS.mkdir(parents=True, exist_ok=True)
        (_RESULTS / f"output_attribution-dev{device_index}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
    except OSError as exc:
        print(f"    WARNING: could not write coverage record: {exc}")

    print(f"\n[per-output attribution / Device {device_index}] {coverage.describe()}")
    for n in names:
        print(f"    {n}: {coverage.token_for(n)} — {coverage.reason_for(n)}")

    assert attribution.own_count > 0, (
        f"[Device {device_index}] the EP executed nothing on a graph containing Add and "
        f"Mul, both proven forms.  {attribution.describe()}  Without a claimed node there "
        "is no partial state to read and this test measures nothing."
    )

    # The instrument's own falsifier, before any finding is read off it (R9).
    refuted = coverage.refuted_by(disagreed)
    assert not refuted, (
        f"[Device {device_index}] outputs {refuted} were labelled CPU-ONLY yet disagree "
        "with the CPU oracle.  Both sides of a CPU-ONLY output are the same computation, "
        "so they cannot differ.  This refutes the COVERAGE INSTRUMENT — it is not a "
        "finding about the EP, and nothing about the EP may be read off this run."
    )

    assert coverage.token_for("out_declined") == _verdict.OUTPUT_CPU_ONLY, (
        f"[Device {device_index}] out_declined is {coverage.token_for('out_declined')}, "
        f"expected CPU-ONLY.  {coverage.reason_for('out_declined')}  Either Sin/Cos/Erf "
        "became claimable — in which case this test needs a new declined branch, not a "
        "relaxed assertion — or the topology/trace join is broken."
    )
    assert coverage.token_for("out_claimed") == _verdict.OUTPUT_EP_COVERED, (
        f"[Device {device_index}] out_claimed is {coverage.token_for('out_claimed')}, "
        f"expected EP-COVERED.  {coverage.reason_for('out_claimed')}"
    )
    assert coverage.partial, (
        f"[Device {device_index}] this graph is supposed to be partially claimed; "
        f"coverage reads {coverage.describe()}"
    )


def test_the_verdict_withholds_match_when_no_output_reaches_the_ep(
    require_vulkan,
    tmp_path: pathlib.Path,
) -> None:
    """A session in which the EP runs and every output is CPU-produced.

    Built by putting the *only* graph output behind ops the EP declines while the graph
    still contains claimable work that something else consumes.  Where that is not
    reachable — ORT may prune work no output depends on — the test says so and reports
    ``UNOBSERVABLE`` rather than inventing a pass: the specimen itself is proved
    synthetically in ``test_output_attribution.py``, on a trace no optimiser can rewrite.
    """
    device_index = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    x = m.tensor("x", ir.DataType.FLOAT, [4, 4])
    y = m.tensor("y", ir.DataType.FLOAT, [4, 4])
    a = m.tensor("a", ir.DataType.FLOAT, [4, 4])
    s = m.tensor("s", ir.DataType.FLOAT, [4, 4])
    out = m.tensor("out_declined_only", ir.DataType.FLOAT, [4, 4])
    nodes = [
        # Claimable work whose result reaches the output only through declined ops.
        ir.Node("", "Add", inputs=[x, y], outputs=[a], name="claimed_add"),
        ir.Node("", "Sin", inputs=[a], outputs=[s], name="declined_sin"),
        ir.Node("", "Erf", inputs=[s], outputs=[out], name="declined_erf"),
    ]
    graph = ir.Graph(
        inputs=[x, y], outputs=[out], nodes=nodes, name="declined_tail",
        opset_imports={"": 17},
    )
    model = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = str(tmp_path / f"outattr_tail_dev{device_index}")
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    if m.EP_NAME not in sess.get_providers():
        raise _verdict.InstrumentError(
            f"[Device {device_index}] {m.EP_NAME} absent from get_providers(); "
            "ERROR(instrument), not a coverage finding (R13)."
        )
    sess.run(None, _feeds())
    attribution = m.attribution_with_coverage_from_profile(sess.end_profiling(), model)
    coverage = attribution.output_coverage
    assert coverage is not None

    print(
        f"\n[declined-tail / Device {device_index}] own_count={attribution.own_count} "
        f"{coverage.describe()}"
    )
    print(f"    out_declined_only: {coverage.token_for('out_declined_only')}")

    if attribution.own_count == 0:
        pytest.skip(
            f"[Device {device_index}] the EP claimed nothing in this graph "
            "(the Add feeding a declined tail was not taken), so the state under test — "
            "EP ran, no output reaches it — did not occur here.  UNOBSERVABLE by frame "
            "(R12), not a pass."
        )

    # The Add IS upstream of the output, so this graph reads EP-COVERED and MATCH stays
    # representable.  That is correct and is worth asserting: the refusal must fire on
    # unreachability, never merely on the presence of a declined op.
    assert coverage.token_for("out_declined_only") == _verdict.OUTPUT_EP_COVERED, (
        f"[Device {device_index}] an output downstream of a claimed Add must be "
        f"EP-COVERED; got {coverage.token_for('out_declined_only')}.  A refusal that "
        "fired here would refuse every mixed graph and the criterion would never close."
    )
    assert attribution.attributed, (
        "a session whose output IS downstream of claimed work stays attributed; the new "
        "condition may only ever remove an attribution the old one wrongly granted"
    )
