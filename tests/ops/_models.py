"""ONNX-IR model builders, session helpers, and assertion utilities for the Vulkan EP tests.

Models are constructed with ``onnx_ir`` (``ir.Value`` / ``ir.Node`` / ``ir.Graph`` /
``ir.Model``), never ``onnx.helper``. Comparisons run the model through the Vulkan EP (with
ORT CPU fallback available) and check the result against ORT's CPU EP, tolerance-gated.

TOLERANCE POLICY
================
Tolerances are per op-family, documented below. Widening any tolerance requires:
  1. Trinity's code-review sign-off.
  2. A comment in the test that names the driver exhibiting wider error and explains why
     the wider tolerance is acceptable.
Never widen a tolerance to turn a red test green without following this protocol.

fp32 elementwise (Add, Sub, Mul, Div, Neg, Abs, Sqrt, Reciprocal, Sign, Pow, Min, Max):
    rtol=1e-5, atol=1e-5
    Justification: IEEE 754 single-precision arithmetic with one rounding per operation.
    GPU and CPU should produce identical or near-identical results for these simple ops.
    1e-5 ≈ 1 ULP at typical activation magnitudes (~1.0). Tighter is not warranted because
    GPU and CPU may use slightly different FMA forms.

fp32 non-linear / transcendental (Exp, Log, Erf, Sin, Cos, Tan, etc.):
    rtol=1e-5, atol=1e-5
    Justification: Hardware transcendental units (GPU) may differ from libm (CPU) in the last
    1-2 ULPs at edge inputs. 1e-5 tolerates this. Do NOT widen without a per-driver data point.

fp32 activations (Relu, Clip, Sigmoid, Tanh, LeakyRelu, Elu, HardSigmoid, Softplus, Gelu):
    rtol=1e-5, atol=1e-5
    Justification: Composed of elementwise + non-linear operations; same tolerance applies.

fp32 comparison/logic (Equal, Greater, Less, And, Or, Not, Where):
    rtol=0, atol=0 (exact)
    Justification: Boolean/comparison results must be bit-exact — GPU and CPU must agree on
    every element. Any divergence is a correctness bug, not a tolerance issue.

fp16 any (M1+, gated on DeviceCapabilities.fp16_arithmetic):
    rtol=1e-3, atol=1e-3
    Justification: fp16 has 10-bit mantissa (~3.01 decimal digits). 1e-3 ≈ 0.5 ULP at
    typical activation magnitudes. Tighter than this is unreliable across GPU vendors
    because fp16 transcendentals vary by 1-2 ULPs. This is a measured, not pessimistic,
    bound: validated against NVIDIA RTX, AMD RDNA2, and lavapipe fp16 paths.

Reductions, GEMM, MatMul (M2+, OQ-10 in DESIGN.md §11):
    TBD — tolerance is accumulation-order-dependent and MUST be derived from test data
    per vendor (NVIDIA/AMD/lavapipe). A placeholder will be set when M2 ops land, with an
    explicit derivation comment. Do not guess; do not copy from fp32 elementwise.

Quantized ops — three-regime policy (Mouse's spec, OP_COVERAGE.md §10.1):

  Regime 1 — Unpack / dequantize (DequantizeLinear, int4→fp):
    rtol=0, atol=0 (bit-exact against NumPy reference)
    Justification: dequantize is purely deterministic arithmetic — (x - zp) * scale — with
    no accumulation. Any bit difference is a correctness bug, not a precision issue.
    Reference: NumPy (NOT ORT CPU EP, because a shared misreading of the schema would let
    both sides of the comparison encode the same wrong answer).

  Regime 2 — MatMulNBits output vs ORT CPU EP:
    fp32 activations: rtol=1e-3, atol=1e-4
    fp16 activations: rtol=2e-2, atol=1e-3
    Justification: the GPU unpack and accumulate in a different order from ORT's CPU int8
    VNNI path. The differences are accumulation-order-only, not correctness failures.
    Empirically measured (2026-07-28, Justin's dev machine, K=1024, N=512):
      accuracy_level=4 (int8 accumulator) vs accuracy_level=1 (fp32):
        max_abs=3.6e-3, mean_abs=7.5e-4, rtol_max=4.6e-3
    Oracle is pinned at accuracy_level=MATMULNBITS_ORACLE_ACCURACY_LEVEL (1, fp32 accumulator).
    The 2e-2 rtol for fp16 accommodates fp16 mantissa truncation on top of accumulation order.
    Source: Mouse's OP_COVERAGE.md §10.1; derived from measurement, not guessed.

    IMPORTANT: fp16 activations require ORT 1.28+ for a usable oracle. ORT 1.27 produces
    NaN/Inf for fp16 MatMulNBits on x86 (empirically confirmed 2026-07-28; suspected null-
    allocator PrePack bug, fixed in 1.28). Tests gated on ORT >= 1.28 for fp16 paths.

  Regime 3 — End-to-end LLM: top-1 token agreement over 64 greedy steps + KL bound.
    Tolerance: per-model, derived at M3+ when actual LLM tests land.
    Justification: final logit per-element comparison on a 150k-vocab model is meaningless —
    a broken kernel can maintain top-1 accuracy for many tokens before divergence is visible
    (Morpheus C6, DESIGN.md §9.1). Per-layer intermediate capture (compare_layers() below)
    is the correct fault-localisation mechanism.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EP_NAME = "VulkanExecutionProvider"
EP_PROVIDERS = [EP_NAME, "CPUExecutionProvider"]

# Tolerance constants — use these names in tests, not magic numbers.
FP32_ELEMENTWISE = {"rtol": 1e-5, "atol": 1e-5}
FP32_TRANSCENDENTAL = {"rtol": 1e-5, "atol": 1e-5}
FP32_ACTIVATION = {"rtol": 1e-5, "atol": 1e-5}
FP32_EXACT = {"rtol": 0, "atol": 0}
FP16_ANY = {"rtol": 1e-3, "atol": 1e-3}

# ---------------------------------------------------------------------------
# Machine-readable claim log helpers (Mouse's OP_COVERAGE.md §10.2)
# ---------------------------------------------------------------------------


def read_claim_log(log_path: "Path | str") -> dict[str, dict]:
    """Parse a CLAIM_LOG JSON Lines file into a dict keyed by qualified op name.

    Set ONNXRUNTIME_EP_VULKAN_CLAIM_LOG=<path> before session creation and the EP writes
    one JSON Lines record per claim decision, flushed immediately. Each record has:
        op      — domain-qualified op type  (e.g. "com.microsoft::MatMulNBits")
        node    — graph node name (or "" if unnamed)
        opset   — resolved since_version (or 0 if unresolved)
        claimed — bool
        code    — DeclineCode tag ("not-registered", "attribute", etc.) or null if claimed
        reason  — human-readable explanation string

    Returns {op_name: last_record} for each op seen in the file.
    Returns {} if the file doesn't exist (EP not built, or log not written).

    Usage:
        log_path = Path(__file__).parent / f"_claim_{os.getpid()}.jsonl"
        os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log_path)
        try:
            InferenceSession(model, ...)
            claims = read_claim_log(log_path)
            assert claims["Add"]["claimed"]
            assert claims["com.microsoft::NotARealOp"]["code"] == "not-registered"
        finally:
            del os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"]
            log_path.unlink(missing_ok=True)
    """
    from pathlib import Path as _Path

    p = _Path(log_path)
    if not p.exists():
        return {}
    result: dict[str, dict] = {}
    try:
        for line in p.read_text("utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    result[record["op"]] = record
                except (json.JSONDecodeError, KeyError):
                    pass
    except OSError:
        pass
    return result


# ---------------------------------------------------------------------------
# Quantization tolerance constants (Mouse's three-regime policy, OP_COVERAGE §10.1)
# ---------------------------------------------------------------------------

# Regime 1: Unpack / dequantize — must be bit-exact vs NumPy.
# Reference: NumPy (not ORT CPU EP — see module docstring for why).
DEQUANT_EXACT = {"rtol": 0, "atol": 0}

# Regime 2: MatMulNBits output vs ORT CPU EP oracle.
# Empirically derived 2026-07-28 (see module docstring for measurement details).
MATMULNBITS_FP32 = {"rtol": 1e-3, "atol": 1e-4}   # fp32 activations
MATMULNBITS_FP16 = {"rtol": 2e-2, "atol": 1e-3}   # fp16 activations

# Oracle pinning: always use accuracy_level=1 (fp32 accumulator) for the CPU EP oracle.
# Levels 0-3 are identical on x86 (all use fp32 path) but level 4 (int8 VNNI) diverges
# by ~4.6e-3 rtol — larger than MATMULNBITS_FP32.rtol. Pinning makes the reference
# deterministic and reproducible across runner hardware generations.
MATMULNBITS_ORACLE_ACCURACY_LEVEL: int = 1


# ---------------------------------------------------------------------------
# IR construction helpers
# ---------------------------------------------------------------------------


def tensor(name: str, dtype: ir.DataType, shape: list[int]) -> ir.Value:
    """A named, typed, shaped IR value — used for graph inputs and outputs."""
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def make_model(
    op_type: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
    opset: int = 21,
) -> bytes:
    """Build a single-node ONNX model. Pass fresh ir.Value instances per call."""
    node = ir.node(
        op_type,
        inputs,
        attributes=attributes or {},
        domain=domain,
        outputs=outputs,
    )
    opset_imports: dict[str, int] = {"": opset}
    if domain:
        opset_imports[domain] = 1
    # Absent optional inputs are represented as unnamed values (empty string) on the node
    # — they must not appear as graph inputs (would collide or be ambiguous).
    graph_inputs = [v for v in inputs if v.name]
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        name=f"vulkan_{op_type}",
        opset_imports=opset_imports,
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _make_session_options(*, profiling: bool = False, prefix: str = "vulkan_test") -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if profiling:
        opts.enable_profiling = True
        opts.profile_file_prefix = prefix
    validate = os.environ.get("ONNXRUNTIME_EP_VULKAN_VALIDATE", "").strip()
    if validate == "1":
        opts.add_session_config_entry("ep.enable_validation", "1")
    return opts


def _session(model: bytes, providers: list[str], *, profiling: bool = False) -> ort.InferenceSession:
    opts = _make_session_options(profiling=profiling)
    return ort.InferenceSession(model, opts, providers=providers)


def run_vulkan(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run a model through the Vulkan EP (CPU fallback available) and return its outputs."""
    return _session(model, EP_PROVIDERS).run(None, feeds)


def run_cpu(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run a model through ORT's CPU EP and return its outputs."""
    opts = _make_session_options()
    return ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)


# ---------------------------------------------------------------------------
# Claim assertion — the vacuous-pass guard
# ---------------------------------------------------------------------------


def assert_vulkan_claims(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert the Vulkan EP actually executed at least one node of *model*.

    Mechanism: ORT built-in profiling JSON (structured, not text parsing).
    - SessionOptions.enable_profiling = True
    - sess.end_profiling() → path to the trace JSON
    - Parse events where cat=="Node" and check args.provider for VulkanExecutionProvider.

    A test that passes *without* calling this function is vacuously correct: because CPU
    fallback is always correct, the output comparison will pass whether or not the EP ran
    anything. This function is the guard against that failure mode (DESIGN.md §9.1).
    """
    opts = _make_session_options(profiling=True, prefix="_vulkan_claim_probe")
    try:
        sess = ort.InferenceSession(model, opts, providers=EP_PROVIDERS)
        sess.run(None, feeds)
        profile_path = sess.end_profiling()
    except Exception as exc:
        raise AssertionError(
            f"Vulkan EP session failed — the EP may not be built or registered: {exc}"
        ) from exc

    try:
        with open(profile_path) as fh:
            events = json.load(fh)
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    providers_seen = {
        e["args"]["provider"]
        for e in events
        if e.get("cat") == "Node"
        and isinstance(e.get("args"), dict)
        and "provider" in e["args"]
    }
    assert EP_NAME in providers_seen, (
        f"{EP_NAME} did not execute any node of this model — the CPU-match check would be "
        f"a vacuous pass. Providers seen: {sorted(providers_seen) or 'none'}.\n"
        "Possible causes:\n"
        "  • No Vulkan device passed the capability gate (no ICD, or device below baseline).\n"
        "  • The claim predicate for this op rejected the node form — check "
        "    ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1 for decline reasons.\n"
        "  • The crate was not built yet (run cargo build --release first)."
    )


# ---------------------------------------------------------------------------
# Output comparison against the CPU oracle
# ---------------------------------------------------------------------------


def assert_matches_cpu(
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    rtol: float = FP32_ELEMENTWISE["rtol"],
    atol: float = FP32_ELEMENTWISE["atol"],
) -> None:
    """Assert the Vulkan EP output equals ORT's CPU EP output, tolerance-gated.

    The CPU EP is the sole correctness oracle (DESIGN.md §9.1). Never use numpy as the
    reference for ops that ORT's CPU EP supports.
    """
    expected = run_cpu(model, feeds)
    actual = run_vulkan(model, feeds)
    assert len(actual) == len(expected), (
        f"Output count mismatch: Vulkan EP returned {len(actual)}, CPU returned {len(expected)}"
    )
    for idx, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(
            got, want, rtol=rtol, atol=atol,
            err_msg=f"output[{idx}]: Vulkan EP vs ORT CPU EP mismatch"
        )


def cpu_can_run(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """Return True if ORT's CPU EP can execute this model (used to skip fp16 comparisons)."""
    try:
        opts = _make_session_options()
        ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
        return True
    except Exception:
        return False


def check(
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    rtol: float = FP32_ELEMENTWISE["rtol"],
    atol: float = FP32_ELEMENTWISE["atol"],
) -> None:
    """Assert the Vulkan EP claims the node; then (if CPU has a kernel) compare outputs.

    This is the standard test entry point: it first calls assert_vulkan_claims (the
    vacuous-pass guard), then assert_matches_cpu if the CPU EP can run the model. Use this
    in almost every test. Use assert_vulkan_claims alone only when the CPU EP lacks a kernel
    for the dtype and you only need to prove device placement.
    """
    assert_vulkan_claims(model, feeds)
    if cpu_can_run(model, feeds):
        assert_matches_cpu(model, feeds, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Inverse claim assertion — the conservative-claiming guard
# ---------------------------------------------------------------------------


def assert_vulkan_does_not_claim(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert that VulkanExecutionProvider executed ZERO nodes of *model*.

    This is the guard against over-claiming: if the EP claims an op form whose contract is
    not fully implemented it produces silent wrong results on some inputs or drivers. Use
    this for permanent CPU fallback ops (NonZero, fp64 arithmetic, data-dependent-shape ops)
    and for unsupported attribute combinations (e.g., Cast to fp64, dynamic shape metadata).

    The 'no Vulkan device' case is vacuously correct: with zero devices, the EP can never
    claim any node, so this assertion always passes — which is the right behaviour (no
    device → CPU fallback is guaranteed correct without further checks).
    """
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "_vulkan_noclaim_probe"
    try:
        sess = ort.InferenceSession(model, opts, providers=EP_PROVIDERS)
        sess.run(None, feeds)
        profile_path = sess.end_profiling()
    except Exception as exc:
        raise AssertionError(
            f"Session failed during not-claimed assertion — EP may be broken: {exc}"
        ) from exc

    try:
        with open(profile_path) as fh:
            events = json.load(fh)
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    providers_seen = {
        e["args"]["provider"]
        for e in events
        if e.get("cat") == "Node"
        and isinstance(e.get("args"), dict)
        and "provider" in e["args"]
    }
    assert EP_NAME not in providers_seen, (
        f"{EP_NAME} claimed a node it should NOT claim. "
        f"Providers seen: {sorted(providers_seen)}.\n"
        "This is a conservative-claiming violation: claiming an op form whose contract "
        "is not fully implemented produces silent wrong results on some inputs or drivers.\n"
        "Check the claim predicate for this op in rust/src/ops/ and ensure it rejects "
        "this input form."
    )


# ---------------------------------------------------------------------------
# Barrier-backend parity helper — used by test_barrier_parity.py
# ---------------------------------------------------------------------------


def run_with_backend(
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    force_legacy: bool = False,
) -> tuple[list[np.ndarray], str]:
    """Run *model* on the Vulkan EP with the specified barrier backend.

    Parameters
    ----------
    model :
        Serialised ONNX model bytes.
    feeds :
        Input name → ndarray mapping.
    force_legacy :
        If ``True``, adds ``ep.force_legacy_barriers=1`` to the session options, forcing
        ``Barriers::Legacy`` (``vkCmdPipelineBarrier``) even on a device that supports
        ``synchronization2``.  If ``False``, the device's natural selection from
        ``Barriers::select`` is used (sync2 if available, legacy otherwise).

    Returns
    -------
    (outputs, active_backend) where *active_backend* is one of:
        ``"sync2"``   — ``Sync2Backend`` was selected (``VK_KHR_synchronization2`` path).
        ``"legacy"``  — ``LegacyBackend`` was selected (``vkCmdPipelineBarrier`` path).
        ``"unknown"`` — EP has not yet implemented ``ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE``
                        (see TODO below); probe file was not written.

    Implementation contract — what Switch must implement
    ----------------------------------------------------
    When the environment variable ``ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE=<absolute-path>``
    is set at session creation time, the EP must:

    1. Write exactly ``"sync2"`` or ``"legacy"`` (no trailing newline, UTF-8) to ``<path>``
       **before ``CreateSession`` returns**, during ``Barriers::select`` inside ``Device::new``.
    2. Overwrite any previous content (create-or-truncate semantics).
    3. If the file cannot be written (permissions, etc.), log a warning and continue —
       the probe is test-only and must not affect correctness.

    This is the simplest IPC mechanism that requires zero ORT API changes.  The path is
    chosen by the test harness (unique per process, project-relative) to avoid /tmp and to
    ensure cleanup regardless of test outcome.

    The mechanism ties to DESIGN.md §7.5 item 5: "Trinity runs the differential suite twice
    per lane — once default, once forced."  Without the probe, a ``force_legacy=True`` session
    that silently ignores the option would make the parity test a false green.
    """
    # Project-relative probe file, unique per process (safe for pytest-xdist).
    # Lives in tests/ops/ alongside the test files; always cleaned up in finally.
    probe_path = Path(__file__).parent / f"_barrier_probe_{os.getpid()}.txt"
    if probe_path.exists():
        try:
            probe_path.unlink()
        except OSError:
            pass

    old_probe = os.environ.pop("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE", None)
    os.environ["ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE"] = str(probe_path.absolute())
    try:
        opts = _make_session_options()
        if force_legacy:
            opts.add_session_config_entry("ep.force_legacy_barriers", "1")
        sess = ort.InferenceSession(model, opts, providers=EP_PROVIDERS)
        outputs = sess.run(None, feeds)
    finally:
        # Always restore the env var, whether the session succeeded or not.
        if old_probe is not None:
            os.environ["ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE"] = old_probe
        else:
            os.environ.pop("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE", None)

    active_backend = "unknown"
    if probe_path.exists():
        try:
            active_backend = probe_path.read_text("utf-8").strip()
        finally:
            try:
                probe_path.unlink()
            except OSError:
                pass

    return outputs, active_backend


# ---------------------------------------------------------------------------
# MatMulNBits model builder
# ---------------------------------------------------------------------------


def make_matmulnbits_model(
    K: int,
    N: int,
    *,
    bits: int = 4,
    block_size: int = 32,
    accuracy_level: int = MATMULNBITS_ORACLE_ACCURACY_LEVEL,
    activation_dtype: ir.DataType = ir.DataType.FLOAT,
) -> tuple[bytes, dict[str, np.ndarray]]:
    """Build a minimal MatMulNBits ONNX graph and return (model_bytes, feeds).

    The packed weights and zero-points are random but deterministically seeded so that
    test results are reproducible across runs and machines.

    Parameters
    ----------
    K, N :
        Input/output feature dimensions (activations shape is [1, K]).
    bits :
        Quantization bit-width (4 or 8). Default 4.
    block_size :
        Quantization block size. Must be a multiple of 32.
    accuracy_level :
        CPU EP oracle accumulator precision (0=default, 1=fp32, 2=fp16, 3=bf16, 4=int8).
        Pin this to MATMULNBITS_ORACLE_ACCURACY_LEVEL in oracle comparisons.
    activation_dtype :
        ORT data type for the activations input (FLOAT or FLOAT16).

    Returns
    -------
    (model_bytes, feeds) — model_bytes is a serialized ONNX proto; feeds contains only the
    activation array (packed weights are baked in as initializers).
    """
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    from onnx import TensorProto as tp

    rng = np.random.default_rng(42)

    # --- Packed weight tensor ---
    # Shape: [N, ceil(K / block_size), block_size * bits / 8]
    blocks_per_col = -(-K // block_size)  # ceil division
    packed_bytes = block_size * bits // 8
    packed_shape = [N, blocks_per_col, packed_bytes]
    packed_data = rng.integers(0, 256, size=packed_shape, dtype=np.uint8)

    # --- Scale tensor ---
    # Shape: [N * blocks_per_col] as float32
    scale_shape = [N * blocks_per_col]
    scale_data = rng.uniform(0.001, 0.1, size=scale_shape).astype(np.float32)

    # --- Zero-point tensor (optional, pack two 4-bit zp per byte) ---
    zp_bytes_per_col = -(-blocks_per_col // 2)  # ceil(blocks_per_col / 2) bytes for 4-bit
    zp_shape = [N, zp_bytes_per_col]
    zp_data = rng.integers(0, 256, size=zp_shape, dtype=np.uint8)

    # --- Activation (the only dynamic input) ---
    np_dtype = np.float32 if activation_dtype == ir.DataType.FLOAT else np.float16
    act = rng.standard_normal((1, K)).astype(np_dtype)
    feeds = {"X": act}

    # Map ir.DataType → ONNX TensorProto type for the graph input declaration.
    _dtype_to_tp = {ir.DataType.FLOAT: tp.FLOAT, ir.DataType.FLOAT16: tp.FLOAT16}
    act_tp_dtype = _dtype_to_tp[activation_dtype]

    # Build initializers.
    b_tensor = onh.from_array(packed_data, name="B")
    scale_tensor = onh.from_array(scale_data, name="scale")
    zp_tensor = onh.from_array(zp_data, name="zero_points")

    # Build node.
    node = oh.make_node(
        "MatMulNBits",
        inputs=["X", "B", "scale", "zero_points"],
        outputs=["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=bits,
        block_size=block_size,
        accuracy_level=accuracy_level,
    )

    # Build graph.
    x_info = oh.make_tensor_value_info("X", act_tp_dtype, [1, K])
    y_info = oh.make_tensor_value_info("Y", tp.FLOAT, [1, N])

    graph = oh.make_graph(
        [node],
        "matmulnbits_test",
        [x_info],
        [y_info],
        initializer=[b_tensor, scale_tensor, zp_tensor],
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
# Per-layer intermediate capture — fault localisation for multi-layer models
# ---------------------------------------------------------------------------


def with_captured_outputs(model_bytes: bytes, extra_output_names: list[str]) -> bytes:
    """Return a copy of *model_bytes* with *extra_output_names* added as graph outputs.

    Use this to expose intermediate activations as outputs so that compare_layers() can
    check each layer independently instead of comparing only the final logits.

    Rationale (Morpheus C6, DESIGN.md §9.1): a broken kernel in a deep model can maintain
    correct top-1 token accuracy for many steps before divergence becomes visible. Per-layer
    capture makes a fault localisable in minutes rather than days.

    Only adds outputs that are not already graph outputs; silently skips unknown names
    (so callers can pass a superset list and let the graph answer what it has).

    Parameters
    ----------
    model_bytes :
        Serialised ONNX model bytes (not mutated; a modified copy is returned).
    extra_output_names :
        Names of internal nodes whose outputs should be exposed. Each name must match a
        value_info entry or a node output in the model.

    Returns
    -------
    Modified model bytes with the requested outputs appended to the graph's output list.
    """
    import onnx

    model = onnx.load_from_string(model_bytes)
    graph = model.graph

    existing_outputs = {vi.name for vi in graph.output}
    all_values: dict[str, onnx.TypeProto] = {}

    # Collect type information from value_info, initializers, and node outputs.
    for vi in graph.value_info:
        all_values[vi.name] = vi.type
    for init in graph.initializer:
        all_values.setdefault(init.name, onnx.TypeProto())
    for node in graph.node:
        for out in node.output:
            all_values.setdefault(out, onnx.TypeProto())

    for name in extra_output_names:
        if name in existing_outputs:
            continue
        tp = onnx.TypeProto()
        if name in all_values:
            tp.CopyFrom(all_values[name])
        vi = onnx.ValueInfoProto()
        vi.name = name
        vi.type.CopyFrom(tp)
        graph.output.append(vi)

    return model.SerializeToString()


def compare_layers(
    model_bytes: bytes,
    feeds: dict[str, np.ndarray],
    layer_names: list[str],
) -> list[dict]:
    """Run *model_bytes* on both the Vulkan EP and CPU EP, returning per-layer diffs.

    This is the primary fault-localisation tool: pass in the names of intermediate output
    nodes (obtained from model inspection or with_captured_outputs) and get back a sorted
    list showing which layers diverge most.

    Parameters
    ----------
    model_bytes :
        Serialised ONNX model bytes.  The model should already have the desired intermediate
        outputs exposed (e.g., via with_captured_outputs).
    feeds :
        Input feeds.
    layer_names :
        Intermediate output names to compare.  Names that are not present in the model
        output list are silently ignored.

    Returns
    -------
    A list of dicts (one per layer), sorted by max_abs_diff descending:
        {
            "name": str,          # layer name
            "max_abs_diff": float,
            "mean_abs_diff": float,
            "rtol_max": float,    # max(abs(a-b) / (atol + abs(b))) for elementwise comparison
        }
    The list is sorted by max_abs_diff descending so the worst-diverging layer is first.
    """
    expanded = with_captured_outputs(model_bytes, layer_names)

    vulkan_outputs = run_vulkan(expanded, feeds)
    cpu_outputs = run_cpu(expanded, feeds)

    # The graph outputs are the original outputs + layer_names (in that order).
    import onnx
    model = onnx.load_from_string(expanded)
    output_names = [vi.name for vi in model.graph.output]

    results = []
    for name, vulkan_val, cpu_val in zip(output_names, vulkan_outputs, cpu_outputs):
        if name not in layer_names:
            continue
        v = np.asarray(vulkan_val, dtype=float)
        c = np.asarray(cpu_val, dtype=float)
        abs_diff = np.abs(v - c)
        denom = 1e-8 + np.abs(c)
        results.append({
            "name": name,
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
            "rtol_max": float((abs_diff / denom).max()),
        })

    results.sort(key=lambda d: d["max_abs_diff"], reverse=True)
    return results
