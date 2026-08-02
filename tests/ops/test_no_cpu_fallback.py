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

# ---------------------------------------------------------------------------
# Fixtures — declined by CAPABILITY, never by POLICY
# ---------------------------------------------------------------------------
# 2026-08-02, `a0bd22d`.  Two arms below used `Erf` as the declined fixture.  Mouse's
# proof ledger landed with 73 entries in the same window, the real model went 0/363 ->
# 323/363, and `Erf` became a *proven* form.  Both branches were correct; the union was
# not.  A test built to prove *the EP did not run* was broken by *the EP now running more*.
#
# The premise that broke was implicit: "this op is unproven, therefore the EP declines it,
# therefore ORT refuses."  The middle step was a fact about the ledger on the day the test
# was written, and nothing in the test said so.  Two changes follow from that:
#
#   1. **The declined fixtures are declined by capability, not by policy.**  fp64 has no
#      path in this EP at all — its own reason is `[dtype] Add output 0 has no element
#      type this EP recognises` — so no ledger entry can ever claim it.  Mouse's
#      `mul_f16_unproven` planted control was the other candidate and is a fine one, but
#      it is declined by *policy*, and a policy can be changed by someone who has no idea
#      this file depends on it.  Capability cannot be changed by accident.
#   2. **The premise is checked, not assumed.**  ``_require_structurally_declined`` makes a
#      future capability addition arrive as ``ERROR(instrument)`` naming the premise,
#      rather than as ``Failed: DID NOT RAISE`` — which reads as a finding about ORT's
#      refusal and is nothing of the sort.  An instrument that changed under the test is
#      never a detection (R13).
#
# Verified on the merged binary (D45B3A8C...), both selectors:
#   fp64 Add                  -> declined      Cast fp32->fp64 -> declined
#   fp32 Add                  -> claimed       Erf             -> CLAIMED (was declined)

_F32 = ir.DataType.FLOAT
_F64 = ir.DataType.DOUBLE


def _build(nodes, inputs, outputs, name: str) -> bytes:
    graph = ir.Graph(inputs, outputs, nodes=nodes, name=name, opset_imports={"": 17})
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _claimed_add() -> bytes:
    """fp32 ``Add`` — a proven form, claimed on both selectors."""
    x = m.tensor("x", _F32, [4])
    out = m.tensor("out", _F32, [4])
    return _build([ir.node("Add", [x, x], outputs=[out])], [x], [out], "claimed_add_graph")


def _declined_fp64_add() -> bytes:
    """fp64 ``Add`` — declined by **capability**.  No ledger entry can claim this."""
    x = m.tensor("x", _F64, [4])
    out = m.tensor("out", _F64, [4])
    return _build([ir.node("Add", [x, x], outputs=[out])], [x], [out], "declined_fp64_graph")


def _declined_cast_to_fp64() -> bytes:
    """``Cast`` fp32->fp64 — declined by **capability**, and usable as a neighbour node."""
    x = m.tensor("x", _F32, [4])
    out = m.tensor("out", _F64, [4])
    return _build(
        [ir.node("Cast", [x], attributes={"to": int(_F64)}, outputs=[out])],
        [x], [out], "declined_cast_graph",
    )


def _partially_claimed() -> bytes:
    """A claimed node next to a **structurally** unclaimable one.

    Partial is the dangerous state — attribution says yes while some outputs are still
    CPU-against-CPU — so this fixture must stay partial permanently.  ``Add`` fp32 is
    claimed; ``Cast`` to fp64 cannot be claimed by any future ledger.
    """
    x = m.tensor("x", _F32, [4])
    mid = m.tensor("mid", _F32, [4])
    out = m.tensor("out", _F64, [4])
    return _build(
        [
            ir.node("Add", [x, x], outputs=[mid]),
            ir.node("Cast", [mid], attributes={"to": int(_F64)}, outputs=[out]),
        ],
        [x], [out], "partially_claimed_graph",
    )


def _require_structurally_declined(model: bytes, feeds, what: str) -> None:
    """Premise guard: the EP must NOT claim *model*.  Otherwise ERROR(instrument).

    Checked against the EP's own claim report, which is an observation independent of the
    refusal the arms below assert — so the premise is not established by the conclusion.

    If this ever raises it is not a finding about ORT and must not be scored as one: it
    says the EP grew a capability and this fixture stopped being a specimen.
    """
    if m.is_vulkan_claimed(model, feeds):
        raise _verdict.InstrumentError(
            f"[fixture premise broken] The EP now CLAIMS {what}, which this file requires "
            "it to decline structurally.\n"
            "Nothing here is evidence about ORT's refusal: the specimen stopped being a "
            "specimen, so the arm did not run (R13 — an instrument that changed under the "
            "test is never a detection).\n"
            "This exact failure happened on 2026-08-02 with `Erf`, when the proof ledger "
            "landed and a decline-by-policy became a claim.  The fixtures were moved to "
            "decline-by-capability (fp64, which the EP has no path for) precisely so that "
            "a ledger change could not do it again.  If fp64 support has genuinely "
            "arrived, pick a new structurally-unclaimable form — do not relax the arm."
        )


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

    The graph must be one ORT **would** refuse if the key were spelled correctly, or the
    arm proves nothing — a session created from a fully-claimed graph is created with or
    without the flag.  It used ``Erf`` until the ledger claimed ``Erf``, at which point
    this arm silently became vacuous alongside the two that went red.  fp64 ``Add`` is
    refused by capability, so the created session is attributable to the typo alone.
    """
    import onnxruntime as ort

    model = _declined_fp64_add()
    _require_structurally_declined(model, {"x": np.ones(4, np.float64)}, "fp64 Add")
    # Control: spelled correctly, ORT refuses this exact graph.
    with pytest.raises(m.CpuFallbackRefused):
        m.assert_ep_owns_whole_graph(model)

    opts = m._make_session_options()
    opts.add_session_config_entry(m.ORT_DISABLE_CPU_FALLBACK_KEY + "k", "1")
    # No exception: the typo is swallowed.  A precondition built on a typo'd key would be
    # inert and silent.
    session = ort.InferenceSession(model, opts, providers=[m.EP_NAME])
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
    """FAIL(condition), and the text quoted is ORT's, not ours (R13).

    The fixture is fp64 ``Add`` — declined by capability.  It was ``Erf`` until the proof
    ledger claimed ``Erf``; see the fixture section above.
    """
    model = _declined_fp64_add()
    _require_structurally_declined(model, {"x": np.ones(4, np.float64)}, "fp64 Add")
    with pytest.raises(m.CpuFallbackRefused) as excinfo:
        m.assert_ep_owns_whole_graph(model, context="fp64 Add, declined by capability")
    assert m._ORT_FALLBACK_TEXT in excinfo.value.ort_text


def test_a_partially_claimed_graph_is_refused_because_partial_is_the_dangerous_state(require_vulkan):
    """The intermediate-ledger state — some nodes claimed, some declined — is refused.

    This is the state that would have looked like partial acceleration with a clean
    oracle: attribution says yes, and the outputs downstream of the declined nodes are
    still CPU-against-CPU.  ORT refuses to build the session at all.

    The fixture must stay *permanently* partial, so the declined neighbour is ``Cast`` to
    fp64 — unclaimable by capability — rather than a form that is merely unproven today.
    Both halves of the premise are checked below against the EP's own claim report, which
    is independent of the refusal being asserted.
    """
    feeds = {"x": np.ones(4, np.float32)}
    _require_structurally_declined(
        _declined_cast_to_fp64(), feeds, "Cast fp32->fp64 (the declined neighbour)"
    )
    if not m.is_vulkan_claimed(_claimed_add(), feeds):
        pytest.skip("EP does not claim fp32 Add on this device; the graph cannot be partial")
    with pytest.raises(m.CpuFallbackRefused):
        m.assert_ep_owns_whole_graph(_partially_claimed(), context="Add(f32) -> Cast(f64)")


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


def test_every_declined_fixture_is_declined_by_capability_not_by_policy():
    """The 2026-08-02 defect, kept executable rather than written in a comment.

    ``Erf`` was declined because the proof ledger had no entry for it.  That is a policy,
    and Mouse changed it — correctly, and with no reason to know this file existed.  Three
    arms here depended on it: two went red and one went silently vacuous, which is the
    worse of the two outcomes.

    Every declined fixture must therefore be unclaimable by *capability*: fp64, for which
    this EP has no path at any point in the pipeline.  A ledger entry cannot create one.
    This asserts the property structurally, on the serialized graph, so that a future
    edit swapping in a convenient fp32 op fails here and not three arms later.
    """
    import onnx

    for build, label in (
        (_declined_fp64_add, "_declined_fp64_add"),
        (_declined_cast_to_fp64, "_declined_cast_to_fp64"),
    ):
        graph = onnx.load_from_string(build()).graph
        dtypes = {
            vi.type.tensor_type.elem_type for vi in list(graph.input) + list(graph.output)
        }
        assert onnx.TensorProto.DOUBLE in dtypes, (
            f"{label} no longer carries an fp64 tensor.  Its decline would then rest on "
            "the proof ledger — a policy — and the ledger changing would break these arms "
            "again, or worse, make one of them pass vacuously."
        )

    # The partial fixture must contain BOTH kinds, or it is not partial.
    partial = onnx.load_from_string(_partially_claimed()).graph
    ops = [n.op_type for n in partial.node]
    assert "Add" in ops and "Cast" in ops, ops
    assert any(
        vi.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE for vi in partial.output
    ), "the partial fixture's declined half must be the fp64 one"
