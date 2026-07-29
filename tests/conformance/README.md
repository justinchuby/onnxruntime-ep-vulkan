# ONNX conformance testing for the Vulkan Execution Provider

Wire the property-based ONNX conformance suite
[`cbourjau/onnx-tests`](https://github.com/cbourjau/onnx-tests) against this repo's
`VulkanExecutionProvider`, to fuzz-validate op coverage against the ONNX standard. **This
directory adds only a thin, non-invasive hook** — it does not fork or vendor onnx-tests.

Latest findings: **[RESULTS.md](RESULTS.md)**.

## Files

| File | Purpose |
|---|---|
| `vulkan_runtime_wrapper.py` | Candidate-runtime hook. `run_vulkan(model)` registers the Vulkan EP and runs each onnx-tests model on `["VulkanExecutionProvider","CPUExecutionProvider"]`. No onnx-tests source is modified. |
| `run_conformance.sh` | Orchestrator (Linux). Runs the claimed-op subset, **one op per pytest subprocess** (so a native EP crash cannot abort the whole run), and writes `results.csv` + per-op `logs/`. |
| `test_conformance.py` | pytest front end. Each claimed op is a parametrized test that runs in its own subprocess. |
| `claimed_ops.txt` | The ops under test. One name per line. Kept in sync with the registry. |
| `results.csv`, `logs/` | Last run's machine-readable results and per-op pytest output. |
| `RESULTS.md` | Human-readable conformance report: PASS / CPU-fallback / FAIL / CRASH with reproducible details. |

## How the injection works

onnx-tests selects its "candidate" runtime from the **`RUN_CANDIDATE`** env var — a dotted
import path to a `Callable[[onnx.ModelProto], dict[str, np.ndarray]]`. We set:

```
RUN_CANDIDATE=vulkan_runtime_wrapper.run_vulkan
PYTHONPATH=<this directory>
VULKAN_EP_LIB=<abs path to rust/target/release/libonnxruntime_vulkan_ep.so>
```

Each generated model is executed on the Vulkan EP (our wrapper) **and** on the ONNX reference
evaluator (the suite's built-in source-of-truth); the suite compares the two with its own
tolerances.

## Prerequisites

1. **Build the EP cdylib:**
   ```bash
   cd <repo root>/rust
   ORT_INCLUDE_DIR=<ort-include-dir> cargo build --release
   # → rust/target/release/libonnxruntime_vulkan_ep.so   (Linux)
   ```

2. **Clone onnx-tests as a sibling and install it with pixi:**
   ```bash
   git clone https://github.com/cbourjau/onnx-tests ../../onnx-tests
   curl -fsSL https://pixi.sh/install.sh | bash
   (cd ../../onnx-tests && ~/.pixi/bin/pixi run postinstall)
   ```

3. **Pin onnxruntime to 1.27 in the pixi env** (EP requires ORT_API_VERSION 27):
   ```bash
   (cd ../../onnx-tests && ~/.pixi/bin/pixi run python -m pip install "onnxruntime==1.27.0")
   ```

4. **Set VK_ICD_FILENAMES** (Linux, for GPU-less CI):
   ```bash
   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
   ```

## Run

```bash
cd tests/conformance

# Full claimed set, bounded fuzzing (20 examples, seed 0)
./run_conformance.sh

# Larger sample size
MAX_EXAMPLES=50 SEED=0 ./run_conformance.sh

# Restrict to specific ops
OPS="Add" ./run_conformance.sh

# With provider attribution (which nodes ran on VulkanEP vs CPU)
PROFILE=1 ./run_conformance.sh
```

Outputs: `results.csv`, per-op `logs/<Op>.log`, and (with `PROFILE=1`) `attr_<Op>.json`.

### Environment variables

| Var | Default | Meaning |
|---|---|---|
| `ONNX_TESTS_DIR` | sibling `../../onnx-tests` | onnx-tests clone location |
| `VULKAN_EP_LIB` | `<repo>/rust/target/release/libonnxruntime_vulkan_ep.so` | EP cdylib to register |
| `ORT_LIB_DIR` | auto-discovered | Directory with `libonnxruntime.so`, added to `LD_LIBRARY_PATH` |
| `PIXI` | `~/.pixi/bin/pixi` | pixi binary |
| `MAX_EXAMPLES` | `20` | Hypothesis `max_examples` per test |
| `SEED` | `0` | Hypothesis seed (reproducible) |
| `PROFILE` | `0` | `1` → ORT profiling + per-op attribution JSON |
| `OPS` | content of `claimed_ops.txt` | Space-separated op override |
| `VK_ICD_FILENAMES` | system default | Force a specific Vulkan ICD (lavapipe, SwiftShader) |

## Notes

- **Not wired into required CI.** This depends on pixi + network + a native build and is too
  heavy/flaky for the main CI gate. It runs as an **opt-in** `workflow_dispatch` workflow at
  `.github/workflows/conformance.yml`.
- **Per-op isolation is intentional.** The Vulkan EP is native code and can segfault on an
  unhandled op form; a per-op subprocess keeps one crash from taking down the whole suite.
- **CPU fallback.** Unclaimed op forms fall back to ORT CPU; a PASS can be a CPU pass — use
  `PROFILE=1` to see which ops actually ran on the Vulkan EP.
