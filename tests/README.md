# onnxruntime-ep-vulkan — tests

The EP is a Rust `cdylib` loaded by a stock ONNX Runtime at runtime. All suites are **Python
(pytest)**; there is no C++/CTest build.

Build the EP first (see the repo `README.md`):

```bash
# Linux / macOS
cd rust
ORT_INCLUDE_DIR=<ort-include-dir> cargo build --release
# => rust/target/release/libonnxruntime_vulkan_ep.so   (Linux)
# => rust/target/release/libonnxruntime_vulkan_ep.dylib (macOS)

# Windows (PowerShell)
cd rust
$env:ORT_INCLUDE_DIR = "<ort-include-dir>"
cargo build --release
# => rust\target\release\onnxruntime_vulkan_ep.dll
```

Then set `ONNXRUNTIME_VULKAN_EP_LIB` to the absolute path of the built library.

## `tests/ops` — op correctness (pytest)

Per-op differential tests: each ONNX node is run through the Vulkan EP and compared,
tolerance-gated, against ORT's CPU EP. Models are built with the ONNX IR (`onnx_ir`). The
**claim assertion** in every test proves the node actually ran on `VulkanExecutionProvider`,
preventing the vacuous CPU-fallback pass described in `docs/DESIGN.md §9.1`.

```bash
# Linux (lavapipe, GPU-less)
export ONNXRUNTIME_VULKAN_EP_LIB="$PWD/rust/target/release/libonnxruntime_vulkan_ep.so"
export LD_LIBRARY_PATH=<ort-prebuilt/lib>
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
python -m pytest tests/ops -q

# Windows (SwiftShader — TODO; build/clippy only for now)
# See docs/PLATFORMS.md §7.4 and the TODO in .github/workflows/ci.yml.
```

Running `pytest` without `ONNXRUNTIME_VULKAN_EP_LIB` set **skips** the suite with a clear
message rather than failing, so it is safe to include in any pytest invocation.

### Claim assertion (the vacuous-pass guard)

Every test in `tests/ops/` calls `_models.assert_vulkan_claims(model, feeds)` before comparing
against the CPU reference. The mechanism is ORT's built-in profiling JSON:

1. Enable `SessionOptions.enable_profiling = True`.
2. Run inference.
3. Call `sess.end_profiling()` to obtain the trace JSON path.
4. Parse the trace for `{"cat": "Node", "args": {"provider": "VulkanExecutionProvider"}}`.
5. Fail with a clear message if `VulkanExecutionProvider` is absent from the node providers.

This is a **structured** mechanism (ORT profiling JSON), not human-readable log text. It proves
device placement without relying on any custom EP-side Python API.

## `tests/conformance` — ONNX-standard fuzz conformance (opt-in)

Bounded property-based conformance of the Vulkan EP against the ONNX standard via
`cbourjau/onnx-tests`. Each op is fuzzed in its own subprocess so a native EP crash cannot abort
the run. See [`tests/conformance/README.md`](conformance/README.md).

## Tolerance policy

See `tests/ops/_models.py` — `TOLERANCE_POLICY` module docstring. Short version:

| Op family | rtol | atol | Justification |
|---|---|---|---|
| fp32 elementwise, activations | 1e-5 | 1e-5 | Single rounding; 1e-5 ≈ 1 ULP at typical magnitudes |
| fp32 non-linear (Exp, Log, Erf, transcendentals) | 1e-5 | 1e-5 | HW transcendentals differ from libm in last few ULPs |
| fp16 (M1+) | 1e-3 | 1e-3 | fp16 has 10-bit mantissa; 1e-3 ≈ 0.5 ULP at typical magnitudes |
| Reductions, GEMM, MatMul (M2+) | TBD | TBD | Accumulation-order-dependent; derived from test data per vendor at M2. See OQ-10. |

**Widening a tolerance** requires Trinity's code-review sign-off **and** an in-test comment
explaining which driver exhibits the wider error and why it is acceptable.

## Portability policy (standing directive 2026-07-29)

**Intel is the spec-conformance oracle.** Intel Iris Xe (Vulkan 1.4.309, UMA) is the
strictest Vulkan implementation available locally. A test that passes on NVIDIA and fails on
Intel means the EP or the test relied on undefined behavior — Intel is correct.

**Never vendor-special-case.** Adding a vendor-conditional skip or a wider tolerance for
one GPU hides a real bug. The permitted exception is a filed driver bug marked
`pytest.mark.xfail(reason="vendor bug: <URL>", strict=True)`.

**UMA awareness.** Iris Xe exposes memory that is both `DEVICE_LOCAL` and `HOST_VISIBLE`,
exactly as Adreno and Mali do. Running `--vulkan-devices 0,1` exercises both memory models
(UMA vs discrete) and catches staging-path assumptions.

**Local results are development loops.** A coverage number or timing measured only on
Justin's desk is not a project result. CI proves portability. Always state the source when
reporting a number.

To run against both local devices:
```bash
ONNXRUNTIME_VULKAN_EP_LIB=rust/target/release/libonnxruntime_vulkan_ep.so \
  pytest tests/ops/ --vulkan-devices 0,1 -v
# Device 0 = Intel Iris Xe (UMA, strictest)
# Device 1 = NVIDIA RTX 4060 (discrete)
# TODO(Switch): requires ep.device_index session option in rust/src/ep.rs
```

---

```bash
pip install -r tests/requirements.txt
# or
pip install onnxruntime>=1.27 numpy onnx onnx_ir pytest
```
