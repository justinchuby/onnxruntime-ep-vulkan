# ONNX conformance (cbourjau/onnx-tests) vs. the Vulkan Execution Provider — RESULTS

Property-based (Hypothesis) fuzz-conformance of `VulkanExecutionProvider` against the ONNX
standard, using [`cbourjau/onnx-tests`](https://github.com/cbourjau/onnx-tests) as the
source-of-truth.

> **No conformance runs recorded yet** (conformance suite requires onnx-tests sibling clone;
> first run is a manual `workflow_dispatch` step — see `tests/conformance/README.md`).
> However, the **differential test suite** (`tests/ops/`) has been run against both local
> hardware devices; those results are recorded below.

---

## Differential test suite results — `tests/ops/` — M0 milestone

### Run date: 2026-07-29 (local, Windows, VULKAN_SDK 1.4.350.0)

| Device | Device name | Vulkan | Memory | PASSED | FAILED | SKIPPED |
|--------|-------------|--------|--------|--------|--------|---------|
| 0 | Intel Iris Xe Graphics | 1.4.309 | UMA 32 KiB shared | **46** | 120 | 74 |
| 1 | NVIDIA GeForce RTX 4060 Laptop GPU | 1.4.325 | Discrete 48 KiB | **46** | 120 | 74 |

**Validation layers:** `VK_LAYER_KHRONOS_validation` enabled. **Zero validation errors** on
both devices across all 240 test runs.

#### What the 120 failures mean

Every failure is a `claim=True` test for an op not yet marked Ready — Sub, Mul, Div, Relu,
Sigmoid, Tanh, Cast, etc. The claim assertion fires correctly (EP ran something other than
VulkanExecutionProvider). This is an **implementation checklist**, not a harness defect.

#### What the 74 skips mean

All 74 are `test_barrier_parity` cases for non-Ready ops. The claim guard in
`test_barrier_parity` confirms via profiling JSON that the EP did not claim the op, then
skips with an explanatory message. This is the intended behavior — vacuous passes are
excluded.

#### M0 exit criteria status

| Criterion | Status | Evidence |
|-----------|--------|---------|
| `pytest tests/ops` green for claimed ops | ✅ | `test_elementwise[Add-fp32]` PASSED both devices |
| Claim assertion proves Add ran on VulkanExecutionProvider | ✅ | profiling JSON `args.provider == "VulkanExecutionProvider"` confirmed |
| Zero validation-layer errors | ✅ | no VK_LAYER_KHRONOS_validation output on either device |
| No Vulkan ICD → zero devices → CPU fallback | ✅ | `test_build_no_icd_session` and `test_no_vulkan_icd_falls_back_to_cpu` PASSED |
| Decline diagnostics print decline reasons | ✅ | `test_claim_debug_prints_decline_reasons` PASSED |
| Barrier parity: Add runs both backends, parity confirmed | ✅ | `test_barrier_parity[Add-fp32]` and `[Add-i32]` PASSED both devices |

#### Key passing tests (46 total, identical on both devices)

```
test_a_ep_smoke.py::test_ep_dll_loads_via_ctypes
test_a_ep_smoke.py::test_ep_ort_registers
test_barrier_parity.py::test_barrier_parity[Add-fp32]          ← M0 Criterion 8
test_barrier_parity.py::test_barrier_parity[Add-i32]
test_claim_diagnostics.py::test_add_is_claimed
test_claim_diagnostics.py::test_unsupported_op_falls_back_to_cpu
test_claim_diagnostics.py::test_build_session_for_claim_debug
test_claim_diagnostics.py::test_claim_debug_prints_decline_reasons
test_claim_diagnostics.py::test_build_no_icd_session
test_claim_diagnostics.py::test_no_vulkan_icd_falls_back_to_cpu
test_domain_regression.py::test_notarealop_ordinary_decline
test_domain_regression.py::test_notarealop_vulkan_does_not_claim
test_elementwise.py::test_binary_elementwise[Add-fp32]         ← M0 Criterion 2
test_elementwise.py::test_binary_broadcast_scalar[Add-broadcast]
test_elementwise.py::test_binary_scalar_scalar[Add-scalar]
test_elementwise.py::test_add_rank4_large
test_elementwise.py::test_add_one_element
test_fallback.py::test_permanent_cpu_fallback_ops[NonZero]
test_fallback.py::test_permanent_cpu_fallback_ops[Unique]
test_fallback.py::test_fp64_not_claimed
test_fallback.py::test_mixed_session_claimed_and_fallback
test_fallback.py::test_cpu_only_session_still_works
test_matmulnbits.py::test_matmulnbits_accuracy_level_pinning
test_matmulnbits.py::test_layer_capture_mechanism
test_op_table.py::test_op_table[Add-fp32]
test_op_table.py::test_op_table[Add-i32]
test_op_table.py::test_op_table[Cast-fp32-to-fp64-declined]
test_op_table.py::test_op_table[Add-fp64-declined]
test_op_table.py::test_op_table[NonZero-declined]
test_shape_inference_delta.py::test_shape_inference_increases_resolved_count
test_shape_inference_delta.py::test_inferred_model_cpu_correctness
test_shape_inference_delta.py::test_inferred_shape_ep_claims[Add-fp32-dyn]    ← shape-inference delta
test_shape_inference_delta.py::test_inferred_shape_ep_claims[Add-fp32-dyn-3d]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Sub-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Mul-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Div-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Max-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Relu-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Neg-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Abs-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Exp-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Log-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Sqrt-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Sigmoid-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Tanh-fp32-dyn]
test_shape_inference_delta.py::test_uninferred_shape_ep_declines[Relu-fp32-dyn-3d]
```

---

## Conformance environment (template — fill in after first conformance run)

| | |
|---|---|
| EP cdylib | `rust/target/release/libonnxruntime_vulkan_ep.so` (ORT_API_VERSION 27) |
| EP name | `VulkanExecutionProvider` (+ `CPUExecutionProvider` fallback) |
| onnxruntime (python) | 1.28.0 |
| Vulkan device | _to be filled_ (lavapipe / SwiftShader / GPU name) |
| onnx-tests | sibling clone, `pixi run postinstall`, python 3.12 |
| Run parameters | `--hypothesis-seed=0`, `--hypothesis-max-examples=20` |
| Date | _to be filled_ |

## Op status — conformance (first run pending)

| Op | Status | Notes |
|---|---|---|
| `Add` | 🔲 pending first run | M0 op; differential tests pass on both local GPUs |

**Legend:** ✅ PASS — ⚠️ CPU-fallback (op claimed but some inputs fell back) — ❌ FAIL
(numerical mismatch) — 💥 CRASH (native EP crash, contained to subprocess) — 🔲 pending

## Claim summary — conformance (first run pending)

| | Count |
|---|---|
| Total claimed ops under test | 1 (M0: Add only) |
| PASS | pending |
| CPU-fallback only | pending |
| FAIL | pending |
| CRASH | pending |

## How to reproduce

```bash
# Prerequisites: see tests/conformance/README.md

cd tests/conformance
VULKAN_EP_LIB=../../rust/target/release/libonnxruntime_vulkan_ep.so \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
OPS="Add" \
MAX_EXAMPLES=20 SEED=0 \
./run_conformance.sh
```
