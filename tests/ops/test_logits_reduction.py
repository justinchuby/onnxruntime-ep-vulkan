"""Device-free falsifiers for `bench/results/probe_logits_reduction.py`.

Round 36. The probe answers whether the logits' 12-ULP residual is made at the lm_head or
inherited from the chain, and it answers it with three instruments the criterion had never
used before: a **float64 reference** (so "divergent" can finally say *which* of the two is
further from the true value), an **accumulation-order envelope** (so a residual can be
compared against how far two correct kernels may legitimately sit apart), and **isolated
single-node runs** (so a node is judged on identical inputs).

Every one of those three can be wrong in a way that looks like a result:

  * a dequantisation I got wrong disagrees with both EPs and reads "both are wrong";
  * an order-spread routine that always returns 0 reads "order cannot matter here" whether
    or not that is true;
  * an isolated run whose graph the EP declined is CPU-vs-CPU, and reports 0 ULP -- the
    most convincing possible way to be measuring nothing;
  * and reading `claimed_nodes` off a **process-cumulative** counter credits the one-node
    lm_head run with the whole model's 355 nodes.

The last of those is not hypothetical: arm B's counters file reads `claimed_nodes: 356`
where arm A read 355, because both sessions live in one process. Every one of these is
pinned below, each with a demonstrated positive state -- an arm that has only ever been
seen green is indistinguishable from one that cannot go red.

These tests need no GPU and no model file.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "ops"))

_SPEC = importlib.util.spec_from_file_location(
    "probe_logits_reduction", REPO / "bench" / "results" / "probe_logits_reduction.py"
)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


# ---------------------------------------------------------------------------------------
# The float64 reference: is the dequantisation right?
# ---------------------------------------------------------------------------------------
def test_dequantize_matches_ort_cpu_on_a_real_matmulnbits_graph():
    """The reference is checked against ORT's own kernel, not against my reading of the spec.

    Arms C and D rest entirely on `dequantize_nbits`. If the nibble order or the implied
    zero point were wrong, the reference would disagree with *both* EPs and the probe would
    report "both are wrong", which is the most confident way to be useless. The probe's own
    liveness gate catches that at runtime; this catches it without a model or a GPU.
    """
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    import _models as m

    k, n = 128, 8
    model_bytes, feeds = m.make_matmulnbits_model(
        k, n, bits=4, block_size=32, with_zero_points=False
    )
    model = onnx.load_from_string(model_bytes)
    node = next(x for x in model.graph.node if x.op_type == "MatMulNBits")
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    packed = inits[node.input[1]]
    scales = inits[node.input[2]]

    w64 = probe.dequantize_nbits(packed, scales, n=n, k=k, block_size=32, bits=4)
    act = list(feeds.values())[0]
    mine = (w64 @ act.reshape(-1).astype(np.float64)).astype(np.float32)

    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    ort_out = np.asarray(sess.run(None, feeds)[0]).reshape(-1).astype(np.float32)

    np.testing.assert_allclose(mine, ort_out, rtol=1e-4, atol=1e-4)


def test_a_wrong_nibble_order_is_detectable():
    """The positive state of the check above: a reference that IS wrong must not agree.

    Without this, `test_dequantize_matches_ort_cpu_...` could be passing because the
    comparison is loose rather than because the dequantisation is right.
    """
    rng = np.random.default_rng(7)
    n, k, bs = 4, 64, 32
    nb = k // bs
    packed = rng.integers(0, 256, size=(n, nb, bs * 4 // 8), dtype=np.uint8)
    scales = (rng.random(n * nb).astype(np.float32) + 0.5).reshape(-1)

    right = probe.dequantize_nbits(packed, scales, n=n, k=k, block_size=bs, bits=4)

    # The same routine with the nibbles swapped -- a plausible off-by-one in the layout.
    b = packed.reshape(n, nb, -1)
    lo = (b & 0x0F).astype(np.int16)
    hi = (b >> 4).astype(np.int16)
    nib = np.empty((n, nb, b.shape[2] * 2), dtype=np.int16)
    nib[:, :, 0::2] = hi  # swapped
    nib[:, :, 1::2] = lo
    s = scales.reshape(n, nb).astype(np.float64)
    wrong = ((nib.astype(np.float64) - 8.0) * s[:, :, None]).reshape(n, -1)[:, :k]

    assert not np.allclose(right, wrong), (
        "swapping the nibble order produced the same weights, so this comparison cannot "
        "tell a correct dequantisation from an incorrect one"
    )


# ---------------------------------------------------------------------------------------
# The accumulation-order envelope: can it detect order sensitivity at all?
# ---------------------------------------------------------------------------------------
def test_the_order_envelope_can_go_red():
    """An envelope routine that always returns 0 reads "order cannot matter" either way.

    On the real lm_head the five orders agree exactly, and that is the *finding* -- fp16
    storage rounding swamps fp32 accumulation order over K=3072. A reader is entitled to
    know the routine can distinguish that from being broken. Here is a case built so the
    orders genuinely disagree: a term that cancels only if it is summed last.
    """
    k = 96
    a = np.ones(k, dtype=np.float32)
    w = np.ones((1, k), dtype=np.float32)
    # A +/- pair large enough that the 94 unit terms fall below its ULP (2**30 has an fp32
    # spacing of 128). Summed left-to-right the pair cancels first and the units survive;
    # summed right-to-left the units are absorbed before the pair cancels.
    w[0, 0], w[0, 1] = 2.0**30, -(2.0**30)

    orders = probe.reduce_orders(a, w, block_size=32)
    values = {name: float(v[0]) for name, v in orders.items()}
    assert len(set(values.values())) > 1, (
        f"all five accumulation orders agreed on a case built to separate them: {values}. "
        "The envelope instrument cannot go red and its 0 on the real model means nothing."
    )


def test_the_order_envelope_is_green_when_order_truly_cannot_matter():
    """The other polarity: exact small integers, where every order is exactly equal."""
    k = 64
    a = np.ones(k, dtype=np.float32)
    w = np.ones((3, k), dtype=np.float32)
    orders = probe.reduce_orders(a, w, block_size=32)
    for name, v in orders.items():
        np.testing.assert_array_equal(v, np.full(3, float(k), dtype=np.float32), err_msg=name)


# ---------------------------------------------------------------------------------------
# Attribution: the process-cumulative counter trap
# ---------------------------------------------------------------------------------------
def _counters(tmp_path: Path, name: str, claimed: int, dispatches: int) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(
        json.dumps(
            {
                "claimed_nodes": claimed,
                "dispatches_executed": dispatches,
                "outputs_device_bound": 0,
                "outputs_host_resident": 66,
                "alloc_device_frame_session_devices": "1=Fake GPU",
            }
        ),
        encoding="utf-8",
    )
    return p


def test_a_second_arm_that_claimed_nothing_reads_unattributed(tmp_path, monkeypatch):
    """355 then 355 is a second arm that claimed NOTHING, not a second arm that claimed 355.

    Counters are process-cumulative. Arm B's file genuinely reads `claimed_nodes: 356`
    after arm A's 355, and arm B claimed exactly one node. Reading the absolute number
    would report an isolated single-node run as fully attributed *even when the EP declined
    the graph entirely* -- and its 0-ULP residual would then be CPU-vs-CPU.
    """
    monkeypatch.setattr(probe, "_COUNTER_HISTORY", [])
    first = probe.route_and_device(_counters(tmp_path, "a", 355, 355))
    assert first["attribution"] == "ATTRIBUTED"
    assert first["claimed_nodes_this_arm"] == 355

    second = probe.route_and_device(_counters(tmp_path, "b", 355, 355))
    assert second["claimed_nodes_this_arm"] == 0
    assert second["attribution"].startswith("UNATTRIBUTED"), second["attribution"]


def test_a_second_arm_that_claimed_one_node_reads_attributed(tmp_path, monkeypatch):
    """The positive state: 355 then 356 is one claimed node, which is what arm B really is."""
    monkeypatch.setattr(probe, "_COUNTER_HISTORY", [])
    probe.route_and_device(_counters(tmp_path, "a", 355, 355))
    second = probe.route_and_device(_counters(tmp_path, "b", 356, 356))
    assert second["claimed_nodes_this_arm"] == 1
    assert second["attribution"] == "ATTRIBUTED"
    assert second["claimed_nodes_process_cumulative"] == 356


def test_the_route_is_read_off_the_counters_not_the_env_var(tmp_path, monkeypatch):
    """Round 35: Step 1c unbinds on refusal, so a declined bind must not read as taken."""
    monkeypatch.setenv("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", "1")
    monkeypatch.setattr(probe, "_COUNTER_HISTORY", [])
    fact = probe.route_and_device(_counters(tmp_path, "a", 1, 1))
    assert fact["kv_writeback_route"] == "host_staging", (
        "the env var requested a device-authoritative bind and the counters say the run "
        "did not take it; the record must follow the counters"
    )


def test_a_missing_counters_file_is_unobservable_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "_COUNTER_HISTORY", [])
    fact = probe.route_and_device(tmp_path / "does_not_exist.json")
    assert fact["kv_writeback_route"] == "UNOBSERVABLE"
    assert fact["device_name"] == "UNOBSERVABLE"
    assert fact["attribution"].startswith("UNATTRIBUTED")


# ---------------------------------------------------------------------------------------
# The ULP statistic itself
# ---------------------------------------------------------------------------------------
def test_ulp_stats_reports_median_and_max_separately():
    """One cancellation element must not be able to speak for the tensor (R11)."""
    cpu = np.ones(1000, dtype=np.float16)
    vk = cpu.copy()
    vk[0] = np.float16(1.0 + 2.0**-6)  # a single large outlier
    st = probe.ulp_stats(vk, cpu)
    assert st["median_ulp_diff"] == 0.0
    assert st["max_ulp_diff"] > 0.0, "the outlier must still be visible in the max"


# ---------------------------------------------------------------------------------------
# The published records say what this round actually measured
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("device", [0, 1])
def test_recorded_result_is_internally_consistent(device):
    """The artifact must not claim a conclusion its own arms do not support."""
    path = REPO / "bench" / "results" / f"logits_reduction-dev{device}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not present in this checkout")
    d = json.loads(path.read_text(encoding="utf-8"))

    b = d["arm_b_isolated_lm_head"]
    assert b["status"] == "MEASURED"
    assert b["attribution"] == "ATTRIBUTED", (
        "an isolated arm whose graph the EP declined measured CPU against CPU"
    )
    assert b["claimed_nodes_this_arm"] >= 1

    # The conclusion is a function of arm B's median and arm E's falsifier, not prose.
    assert d["conclusion"]["verdict"] == "H_depth"
    assert b["median_ulp_diff"] <= 2.0
    assert d["arm_e_falsifier_other_inputs"]["falsifier_fired"] is False

    # The float64 reference must have been live, or its verdict is not a verdict.
    for arm in ("arm_c_float64_reference", "arm_f_isolated_final_rmsnorm"):
        assert d[arm]["reference_liveness"]["live"] is True, arm

    # `atol` was not moved and no row was closed.
    assert any("atol is not moved" in s for s in d["not_done_here"])


def test_both_devices_reached_the_same_conclusion():
    """Round 31: a result seen on one vendor is a result about that vendor."""
    recs = {}
    for device in (0, 1):
        path = REPO / "bench" / "results" / f"logits_reduction-dev{device}.json"
        if not path.exists():
            pytest.skip("both device records required")
        recs[device] = json.loads(path.read_text(encoding="utf-8"))
    names = {d["arm_a_full_model_tap"]["device_name"] for d in recs.values()}
    assert len(names) == 2, f"both records name the same device: {names}"
    assert recs[0]["conclusion"]["verdict"] == recs[1]["conclusion"]["verdict"]
    hops = [
        tuple(h["median_ulp_diff"] for h in d["arm_a_full_model_tap"]["chain_hop_by_hop_median_ulp"])
        for d in recs.values()
    ]
    assert hops[0] == hops[1], f"the chain differs between vendors: {hops}"
