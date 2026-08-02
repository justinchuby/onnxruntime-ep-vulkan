"""ORT's own refusal: ``session.disable_cpu_ep_fallback``.

WHERE THIS CAME FROM
====================
The user asked whether ORT has a flag that prevents EP fallback.  It does.  We had never
used it.  Seven CPU-vs-CPU incidents, an all-night hunt, a criterion closed and reopened,
and a session spent building detectors — Guard D, the attribution requirement,
``_verdict.py``'s unrepresentable ``MATCH``, per-output coverage — for a state the runtime
would have refused to enter.  The lesson recorded here is not about fallback.

WHY IT IS WORTH MORE THAN ANOTHER GUARD OF OURS
===============================================
1. **It is an instrument we do not own.**  Every mechanism above is ours and can share a
   blind spot with the thing it watches.  This refusal is ORT's, in ORT's code, agreeing
   with our attribution check from outside it — and capable of falsifying us (R9).
2. **It fires before any comparison exists.**  Guard D detects a vacuous pass once the
   numbers are in hand.  This makes the vacuous comparison *unconstructible*.
3. **It would have caught all seven**, including ``bench/results/criterion10-dev0.json``
   at ``c144210``: ``oracle_outputs_compared=65``, ``degenerate=0``,
   ``within_tolerance=65``, ``max_abs_diff=0.0`` — a flawless-looking record that only the
   attribution field saved.  With the flag set that session would have raised before
   producing a single number.

WHAT THE PROBE FOUND, AND WHY IT IS NOT A BLANKET FIX
=====================================================
``bench/results/probe_disable_cpu_fallback.py``, both selectors, ORT 1.28.0:

* With ``CPUExecutionProvider`` in the providers list the flag is a **configuration
  conflict** (``INVALID_ARGUMENT``) and fails **every** graph, including one the EP claims
  in full.  Our standard ``EP_PROVIDERS`` lists it.  So a naive wiring produces a refusal
  on a healthy run and a reader would score it as a detection.  These two texts are
  distinguished by the harness and only one of them is a finding (R13).
* An unknown session-config key is accepted **silently**.  A typo leaves the precondition
  inert while every test resting on it still passes.  Hence :func:`assert_no_cpu_fallback_is_live`.
* With ``providers=[EP_NAME]``: claimed single-op -> created; declined single-op -> refused;
  partially claimed -> refused.  Exact at single-op scale, wrong at whole-model scale,
  because Phi-3.5 legitimately declines ten edge ops when the EP is working correctly.

THE TWO MECHANISMS' EXTENTS ARE STATED SEPARATELY ON PURPOSE
===========================================================
Two gates whose extents differ compose to the weaker extent and the stronger name.  This
precondition reaches only graphs the EP must claim entirely; Guard D reaches any graph and
is indispensable exactly where this is unavailable.  Neither may borrow the other's reach.
"""

from __future__ import annotations

import os

import numpy as np
import onnx_ir as ir
import pytest

import _models as m
import _verdict

pytestmark = pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set - no EP whose claim can be required",
)

def _unary(op: str, dtype: ir.DataType = ir.DataType.FLOAT, n: int = 4) -> bytes:
    x = m.tensor("x", dtype, [n])
    out = m.tensor("out", dtype, [n])
    node = ir.Node("", op, inputs=[x], outputs=[out], name=f"the_{op.lower()}_node")
    graph = ir.Graph(inputs=[x], outputs=[out], nodes=[node], name=f"{op}_graph",
                     opset_imports={"": 17})
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _claimed_add() -> bytes:
    x = m.tensor("x", ir.DataType.FLOAT, [4])
    out = m.tensor("out", ir.DataType.FLOAT, [4])
    node = ir.Node("", "Add", inputs=[x, x], outputs=[out], name="claimed_add")
    graph = ir.Graph(inputs=[x], outputs=[out], nodes=[node], name="add_graph",
                     opset_imports={"": 17})
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _partially_claimed() -> bytes:
    x = m.tensor("x", ir.DataType.FLOAT, [4])
    mid = m.tensor("mid", ir.DataType.FLOAT, [4])
    out = m.tensor("out", ir.DataType.FLOAT, [4])
    nodes = [
        ir.Node("", "Add", inputs=[x, x], outputs=[mid], name="claimed_add"),
        ir.Node("", "Erf", inputs=[mid], outputs=[out], name="declined_erf"),
    ]
    graph = ir.Graph(inputs=[x], outputs=[out], nodes=nodes, name="mixed_graph",
                     opset_imports={"": 17})
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


# ---------------------------------------------------------------------------
# The instrument's own falsifier (R10) — runs first, everything below rests on it
# ---------------------------------------------------------------------------


def test_ort_refusal_is_live_and_not_a_silently_accepted_typo(require_vulkan):
    """ORT accepts unknown config keys silently, so the flag must prove it fires."""
    text = m.assert_no_cpu_fallback_is_live()
    assert m._ORT_FALLBACK_TEXT in text, text


def test_a_misspelled_key_is_accepted_silently_which_is_why_the_canary_exists(require_vulkan):
    """The hazard itself, asserted — if ORT ever starts rejecting typos, delete the canary.

    This is not a test of ORT's manners.  It is the reason ``assert_no_cpu_fallback_is_live``
    exists, kept executable so the justification cannot rot into a comment nobody rechecks.
    """
    import onnxruntime as ort

    opts = m._make_session_options()
    opts.add_session_config_entry(m.ORT_DISABLE_CPU_FALLBACK_KEY + "k", "1")
    # No exception: the typo is swallowed.  A precondition built on a typo'd key would be
    # inert and silent.
    session = ort.InferenceSession(_unary("Erf"), opts, providers=[m.EP_NAME])
    assert session is not None


# ---------------------------------------------------------------------------
# The three measured behaviours
# ---------------------------------------------------------------------------


def test_a_fully_claimed_single_op_graph_survives_the_refusal(require_vulkan):
    """The load-bearing arm: the precondition must not fire on a healthy single-op test.

    If ORT planted CPU nodes of its own (Cast/Memcpy/Identity) this would refuse, and the
    precondition would be useless at exactly the scale it is proposed for.  It does not.
    """
    model = _claimed_add()
    if not m.is_vulkan_claimed(model, {"x": np.ones(4, np.float32)}):
        pytest.skip("EP does not claim fp32 Add on this device; arm has no subject")
    m.assert_ep_owns_whole_graph(model, context="fp32 Add, claimed")


def test_a_declined_single_op_graph_is_refused_with_orts_own_text(require_vulkan):
    """FAIL(condition), and the text quoted is ORT's, not ours (R13)."""
    with pytest.raises(m.CpuFallbackRefused) as excinfo:
        m.assert_ep_owns_whole_graph(_unary("Erf"), context="Erf, expected decline")
    assert m._ORT_FALLBACK_TEXT in excinfo.value.ort_text


def test_a_partially_claimed_graph_is_refused_because_partial_is_the_dangerous_state(require_vulkan):
    """The intermediate-ledger state — some nodes claimed, some declined — is refused.

    This is the state that would have looked like partial acceleration with a clean
    oracle: attribution says yes, and the outputs downstream of the declined nodes are
    still CPU-against-CPU.  ORT refuses to build the session at all.
    """
    with pytest.raises(m.CpuFallbackRefused):
        m.assert_ep_owns_whole_graph(_partially_claimed(), context="Add->Erf")


def test_listing_the_cpu_ep_turns_the_flag_into_an_instrument_error_not_a_finding(require_vulkan):
    """A healthy graph + our standard EP_PROVIDERS = INVALID_ARGUMENT on every graph.

    The trap: that refusal looks like a detection and is not.  It must reach the reader as
    ERROR(instrument), never as FAIL(condition), or a working EP reads as a broken one.
    """
    import onnxruntime as ort

    model = _claimed_add()
    opts = m._no_cpu_fallback_options()
    with pytest.raises(Exception) as excinfo:
        ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    assert m._ORT_CONFLICT_TEXT in str(excinfo.value)
    assert m._ORT_FALLBACK_TEXT not in str(excinfo.value), (
        "ORT's two refusals must stay distinguishable; if they merge, the harness can no "
        "longer tell a finding from a misconfiguration"
    )


def test_the_refusal_is_a_finding_not_an_instrument_outage(require_vulkan):
    """R13 terminal-state placement, asserted rather than assumed.

    ``CpuFallbackRefused`` is an ``AssertionError`` — FAIL(condition).  It must NOT be an
    ``InstrumentError``, because "the EP did not claim the op" is exactly the condition
    single-op tests exist to detect.  Conversely a misconfiguration must not be a finding.
    """
    assert issubclass(m.CpuFallbackRefused, AssertionError)
    assert not issubclass(m.CpuFallbackRefused, _verdict.InstrumentError)
    assert not issubclass(_verdict.InstrumentError, AssertionError)


# ---------------------------------------------------------------------------
# The precondition is a precondition
# ---------------------------------------------------------------------------


def test_the_precondition_returns_nothing_and_therefore_cannot_report_pass():
    """It cannot say PASS about a thing it did not test — it has nothing to say it with."""
    import inspect

    sig = inspect.signature(m.assert_ep_owns_whole_graph)
    assert sig.return_annotation in (None, "None"), (
        "a precondition that returns a verdict is a gate wearing a precondition's name"
    )


def test_the_two_mechanisms_extents_are_written_down_separately():
    """Two gates whose extents differ compose to the weaker extent and the stronger name.

    So the extents must be stated, not inferred.  If this docstring loses either row a
    future reader will take the pair for one gate reaching everything.
    """
    doc = m.assert_ep_owns_whole_graph.__doc__ or ""
    assert "Guard D" in doc
    assert "single-op" in doc
    assert "whole model" in doc or "whole models" in doc
