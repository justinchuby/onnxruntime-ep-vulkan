# ONNX conformance (cbourjau/onnx-tests) vs. the Vulkan Execution Provider — RESULTS

Property-based (Hypothesis) fuzz-conformance of `VulkanExecutionProvider` against the ONNX
standard, using [`cbourjau/onnx-tests`](https://github.com/cbourjau/onnx-tests) as the
source-of-truth.

> **No runs recorded yet.** The Vulkan EP crate (`rust/`) is under active development (M0).
> Once M0 exits (Add is claimed, the test suite is green in CI), the first conformance run
> will be recorded here. See `tests/conformance/README.md` for how to run.
>
> This file is updated manually after each opt-in conformance run (`workflow_dispatch`
> in `.github/workflows/conformance.yml`). Machine-readable results live in `results.csv`
> and per-op logs in `logs/`.

## Environment (template — fill in after first run)

| | |
|---|---|
| EP cdylib | `rust/target/release/libonnxruntime_vulkan_ep.so` (ORT_API_VERSION 27) |
| EP name | `VulkanExecutionProvider` (+ `CPUExecutionProvider` fallback) |
| onnxruntime (python) | 1.27.0 |
| Vulkan device | _to be filled_ (lavapipe / SwiftShader / GPU name) |
| onnx-tests | sibling clone, `pixi run postinstall`, python 3.12 |
| Run parameters | `--hypothesis-seed=0`, `--hypothesis-max-examples=20` |
| Date | _to be filled_ |

## Op status

<!-- Updated by run_conformance.sh → RESULTS.md rendering pass -->

| Op | Status | Notes |
|---|---|---|
| `Add` | 🔲 pending first run | M0 op; targeted for first conformance run |

**Legend:** ✅ PASS — ⚠️ CPU-fallback (op claimed but some inputs fell back) — ❌ FAIL
(numerical mismatch) — 💥 CRASH (native EP crash, contained to subprocess) — 🔲 pending

## Claim summary

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
