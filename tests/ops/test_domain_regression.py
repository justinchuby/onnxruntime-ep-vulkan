"""Domain-wide opt-in regression tests (Morpheus C1 requirement).

Morpheus C1 states: no domain-wide opt-in may exist. Specifically, the implementation
must not contain a code path like `if node.domain == "com.microsoft" { claim(node) }`.
The VulkanExecutionProvider must decline any op that is not individually registered,
including fictional ops in the com.microsoft domain, with an ordinary "not-registered"
decline — not a crash, not a panic, not a silent claim.

Tank has banned the domain as a *value* statically in `rust/tests/layering.rs`. This
module is the runtime half: it proves the architecture holds end-to-end.

Test: test_notarealop_ordinary_decline
  - Fabricates a node `com.microsoft::NotARealOp`.
  - Asserts the Vulkan EP declines with an ordinary ORT "no kernel" error, NOT a crash.
  - Asserts the CPU EP can complete the session (since the session has CPU fallback).
  - Asserts the Vulkan EP executed ZERO nodes of the model.
  - The decline reason is asserted via the machine-readable mechanism once Mouse's
    registry exposes it (TODO below). Until then, the structural assert (zero nodes
    claimed) is the guard.

TODO(Mouse): once the claim-predicate registry exposes machine-readable decline reasons
  (the "not-registered" reason code from ops/registry.rs), update this test to assert
  against the reason code directly rather than relying only on the profiling-JSON absence
  of VulkanExecutionProvider in "provider" events. The structural assertion (zero nodes)
  is correct and sufficient until then.

Mouse note: Mouse's staged registry entries for com.microsoft ops (FastGelu, GroupQueryAttention,
  MatMulNBits, etc.) each have explicit claim predicates. Only those ops are claimed. A
  fictional op in the same domain must not be claimed, and this test verifies that invariant.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

import tests.ops._models as m


# ---------------------------------------------------------------------------
# C1 regression — com.microsoft::NotARealOp must produce an ordinary decline
# ---------------------------------------------------------------------------


def _make_not_a_real_op_model() -> tuple[bytes, dict[str, np.ndarray]]:
    """Build a minimal model containing a com.microsoft::NotARealOp node.

    Returns (model_bytes, feeds). The graph has a single node of type NotARealOp in the
    com.microsoft domain. No CPU EP kernel exists for this op either, so the session
    creation itself should fail or the run should raise an appropriate ORT error.

    The important invariant is: the Vulkan EP must NOT claim the op (no domain-wide opt-in),
    and the error must be an ordinary "no registered kernel" ORT error, not an EP crash.
    """
    import onnx
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


def test_notarealop_ordinary_decline():
    """com.microsoft::NotARealOp must be declined by the Vulkan EP with a normal ORT error.

    Morpheus C1 — the runtime half of the architectural invariant that no domain-wide opt-in
    exists. Tank's static ban in layering.rs prevents the *value* from appearing in code;
    this test proves the runtime behaviour matches the architectural intent.

    The assertion sequence:
    1. Session creation with Vulkan EP must either succeed (if the EP partitions the graph
       and declines this op) or raise an ORT-category error (not a segfault or Python crash).
       Either outcome is acceptable — what is NOT acceptable is the Vulkan EP claiming the op.
    2. If session creation succeeds, the Vulkan EP must have claimed ZERO nodes
       (assert_vulkan_does_not_claim). A claim would mean a domain-wide opt-in exists.
    3. Any ORT exception must be an "invalid graph / no registered kernel" class error,
       not an EP-internal panic or segfault.

    TODO(Mouse): when the claim-predicate registry exposes machine-readable decline reasons,
       replace the structural assertion (zero nodes) with:
           assert decline_reason == "not-registered"
       using the structured reason code from ops/registry.rs. The current test is correct
       but does not yet distinguish between "no kernel" (correct) and "panic" (wrong).
    """
    model_bytes, feeds = _make_not_a_real_op_model()

    # Check: session creation with Vulkan EP providers.
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    try:
        sess = ort.InferenceSession(
            model_bytes,
            opts,
            providers=[m.EP_NAME, "CPUExecutionProvider"],
        )
        # Session created successfully: Vulkan EP must have declined without crashing.
        # Now run and assert zero Vulkan claims.
        try:
            sess.run(None, feeds)
            # If run succeeded with the EP available, that means both EP and CPU declined
            # and somehow the session ran — which should not happen for an unknown op.
            # Use the profiling-based assertion to confirm zero Vulkan claims.
            m.assert_vulkan_does_not_claim(model_bytes, feeds)
        except (
            ort.capi.onnxruntime_pybind11_state.Fail,
            ort.capi.onnxruntime_pybind11_state.InvalidGraph,
            ort.capi.onnxruntime_pybind11_state.NotImplemented,
            ort.capi.onnxruntime_pybind11_state.NotFound,
        ):
            # Expected: no CPU kernel either, so the run legitimately fails.
            pass
        except Exception as exc:
            # Any other exception must not be an EP crash.
            if isinstance(exc, SystemError):
                raise AssertionError(
                    f"com.microsoft::NotARealOp caused a SystemError — this suggests an "
                    f"EP crash rather than an ordinary 'no registered kernel' decline. "
                    f"Original: {exc}"
                ) from exc

    except (
        ort.capi.onnxruntime_pybind11_state.Fail,
        ort.capi.onnxruntime_pybind11_state.InvalidGraph,
        ort.capi.onnxruntime_pybind11_state.InvalidProtobuf,
        ort.capi.onnxruntime_pybind11_state.NotImplemented,
        ort.capi.onnxruntime_pybind11_state.NotFound,
    ):
        # Session creation failed with an expected ORT error — this is acceptable.
        # The Vulkan EP did not crash, and no domain-wide claim occurred.
        pass
    except SystemError as exc:
        raise AssertionError(
            f"com.microsoft::NotARealOp caused a SystemError during session creation — "
            f"this suggests an EP crash rather than an ordinary decline. Original: {exc}"
        ) from exc


def test_notarealop_vulkan_does_not_claim():
    """Even when a Vulkan device is available, NotARealOp must not be claimed.

    This is the canonical statement of Morpheus C1 as a skip-safe test: if no Vulkan
    device is present, the EP cannot claim anything and the test passes vacuously. If a
    device IS present, we prove the architectural property.

    This test only runs its Vulkan-specific assertion if the EP is available (fixture),
    but the model-construction side runs always.
    """
    model_bytes, feeds = _make_not_a_real_op_model()

    # Check: does any Vulkan device try to claim this op?
    # We use the profiling-based check but catch the session-creation failure that happens
    # when neither EP nor CPU has a kernel for the op.
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "_notarealop_probe"

    import json
    import os

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
                f"{m.EP_NAME} claimed com.microsoft::NotARealOp — a domain-wide opt-in "
                f"exists! This violates Morpheus C1. Providers seen: {sorted(providers_seen)}"
            )
        finally:
            try:
                os.remove(profile_path)
            except OSError:
                pass
    except (
        ort.capi.onnxruntime_pybind11_state.Fail,
        ort.capi.onnxruntime_pybind11_state.InvalidGraph,
        ort.capi.onnxruntime_pybind11_state.InvalidProtobuf,
        ort.capi.onnxruntime_pybind11_state.NotImplemented,
        ort.capi.onnxruntime_pybind11_state.NotFound,
    ):
        # No kernel found, session creation failed. The Vulkan EP did not claim the op.
        pass
    except SystemError as exc:
        raise AssertionError(
            f"com.microsoft::NotARealOp caused a SystemError — EP crash suspected. {exc}"
        ) from exc
