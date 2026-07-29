"""Domain-wide opt-in regression tests (Morpheus C1 requirement).

Morpheus C1: no domain-wide opt-in may exist. The implementation must not have a code
path like `if node.domain == "com.microsoft" { claim(node) }`. The EP must decline any
op not individually registered, including fabricated ops in any domain, with an ordinary
"not-registered" decline -- not a crash, not a panic, not a silent claim.

Tank bans the domain as a *value* statically in `rust/tests/layering.rs`. This module is
the runtime half: it proves the architecture holds end-to-end.

Machine-readable claim log (Mouse OP_COVERAGE.md section 10.2):
  Set ONNXRUNTIME_EP_VULKAN_CLAIM_LOG=<path> before session creation. The EP appends one
  JSON Lines record per decision, flushed immediately:
    {"op":"com.microsoft::NotARealOp","node":"n0","opset":1,"claimed":false,
     "code":"not-registered","reason":"[not-registered] no Vulkan handler registered..."}
  code is null when claimed. code == "not-registered" is the C1-specific assertion.

Tests:
  test_notarealop_ordinary_decline  -- asserts code="not-registered" via CLAIM_LOG + no crash
  test_notarealop_vulkan_does_not_claim -- belt-and-suspenders: zero Vulkan nodes (profiling)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

import tests.ops._models as m


# ---------------------------------------------------------------------------
# Shared model builder
# ---------------------------------------------------------------------------


def _make_not_a_real_op_model() -> tuple[bytes, dict[str, np.ndarray]]:
    """Build a model with a single com.microsoft::NotARealOp node."""
    import onnx.helper as oh
    from onnx import TensorProto as tp

    x_data = np.ones((2, 3), dtype=np.float32)
    feeds = {"X": x_data}

    node = oh.make_node(
        "NotARealOp",
        inputs=["X"],
        outputs=["Y"],
        domain="com.microsoft",
    )
    import onnx
    graph = oh.make_graph(
        [node],
        "not_a_real_op_test",
        [oh.make_tensor_value_info("X", tp.FLOAT, [2, 3])],
        [oh.make_tensor_value_info("Y", tp.FLOAT, [2, 3])],
    )
    model = oh.make_model(
        graph,
        opset_imports=[
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 8
    return model.SerializeToString(), feeds


# ---------------------------------------------------------------------------
# C1 regression
# ---------------------------------------------------------------------------

# ORT exception types that represent ordinary "no kernel" errors (not EP crashes).
_ORT_NO_KERNEL_ERRORS = (
    ort.capi.onnxruntime_pybind11_state.Fail,
    ort.capi.onnxruntime_pybind11_state.InvalidGraph,
    ort.capi.onnxruntime_pybind11_state.InvalidProtobuf,
    ort.capi.onnxruntime_pybind11_state.NotImplemented,
    ort.capi.onnxruntime_pybind11_state.NotFound,
)


def test_notarealop_ordinary_decline():
    """com.microsoft::NotARealOp must decline with code="not-registered" and not crash.

    Morpheus C1 runtime half. Assertion sequence:
    1. CLAIM_LOG (Mouse sec 10.2): if the EP is built, assert code == "not-registered".
       This is the definitive assertion -- it distinguishes "declined correctly" from a
       crash before reaching the claim predicate, and from domain-wide acceptance.
    2. ORT error type: session creation must raise an ORT Fail/similar, NOT SystemError
       (which would indicate an EP crash/panic).
    3. Zero Vulkan nodes claimed (via test_notarealop_vulkan_does_not_claim, belt-and-suspenders).
    """
    model_bytes, feeds = _make_not_a_real_op_model()

    log_path = Path(__file__).parent / f"_claim_log_{os.getpid()}.jsonl"
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        pass

    old_log = os.environ.pop("ONNXRUNTIME_EP_VULKAN_CLAIM_LOG", None)
    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log_path.absolute())

    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:
            sess = ort.InferenceSession(
                model_bytes,
                opts,
                providers=[m.EP_NAME, "CPUExecutionProvider"],
            )
            try:
                sess.run(None, feeds)
            except _ORT_NO_KERNEL_ERRORS:
                pass  # expected: no CPU kernel either
        except _ORT_NO_KERNEL_ERRORS:
            pass  # expected: ORT raises Fail/similar because no EP has a kernel
        except SystemError as exc:
            raise AssertionError(
                "com.microsoft::NotARealOp caused a SystemError during session creation -- "
                "this indicates an EP crash rather than an ordinary decline. "
                f"Original: {exc}"
            ) from exc

    finally:
        if old_log is not None:
            os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = old_log
        else:
            os.environ.pop("ONNXRUNTIME_EP_VULKAN_CLAIM_LOG", None)

    # Assertion 1: CLAIM_LOG decline code.
    claims = m.read_claim_log(log_path)
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        pass

    if claims:
        # EP wrote the log -- assert the specific decline code.
        key = "com.microsoft::NotARealOp"
        assert key in claims, (
            f"CLAIM_LOG did not contain an entry for '{key}'. "
            f"Keys present: {list(claims.keys())}"
        )
        record = claims[key]
        assert not record["claimed"], (
            f"EP claimed com.microsoft::NotARealOp -- domain-wide opt-in suspected. "
            f"Record: {record}"
        )
        code = record.get("code")
        assert code == "not-registered", (
            f"Expected decline code 'not-registered' but got {code!r}. "
            f"Full record: {record}\n"
            "If this is another code, check Mouse's DeclineCode definition and update "
            "the assertion to match the canonical 'unregistered op' code."
        )
    # If claims is empty (EP not built), the structural check in
    # test_notarealop_vulkan_does_not_claim covers this path.


def test_notarealop_vulkan_does_not_claim():
    """com.microsoft::NotARealOp must not appear as a Vulkan-claimed node (structural check).

    Belt-and-suspenders for test_notarealop_ordinary_decline: uses ORT profiling JSON to
    assert zero VulkanExecutionProvider nodes, regardless of whether CLAIM_LOG is available.
    Passes trivially when the EP is not built (no Vulkan provider -> zero claims guaranteed).
    """
    model_bytes, feeds = _make_not_a_real_op_model()

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "_notarealop_probe"

    try:
        sess = ort.InferenceSession(
            model_bytes,
            opts,
            providers=[m.EP_NAME, "CPUExecutionProvider"],
        )
        try:
            sess.run(None, feeds)
        except Exception:
            pass
        profile_path = sess.end_profiling()
        try:
            with open(profile_path) as fh:
                events = json.load(fh)
            providers_seen = {
                e["args"]["provider"]
                for e in events
                if e.get("cat") == "Node"
                and isinstance(e.get("args"), dict)
                and "provider" in e["args"]
            }
            assert m.EP_NAME not in providers_seen, (
                f"{m.EP_NAME} claimed com.microsoft::NotARealOp -- domain-wide opt-in "
                f"exists! Providers seen: {sorted(providers_seen)}"
            )
        finally:
            try:
                os.remove(profile_path)
            except OSError:
                pass
    except _ORT_NO_KERNEL_ERRORS:
        pass  # no kernel -> no claim; acceptable
    except SystemError as exc:
        raise AssertionError(
            f"com.microsoft::NotARealOp caused a SystemError -- EP crash suspected. {exc}"
        ) from exc
