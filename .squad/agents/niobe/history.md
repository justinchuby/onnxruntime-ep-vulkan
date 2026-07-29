# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Performance — benchmarks, profiling, regression tracking
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

📌 Team update (2026-07-28T17:59:54-07:00): Vulkan API baseline is a capability-set: ≥1.1 core + compute queue + `synchronization2` + `subgroup_size_control` + subgroup BASIC+ARITHMETIC + `maxComputeWorkGroupInvocations ≥ 256` + `maxComputeSharedMemorySize ≥ 16 KiB`. Benchmark device coverage will vary by tier (desktop 2022+ / software rasterizer / Android 2022+ mid-high / pre-2022 Android that reports these extensions). — decided by Morpheus, Switch, Link, Fact Checker

📌 Team update (2026-07-28T17:59:54-07:00): Record-once / replay-many — `Compute` hashes input shapes and replays a cached `VkCommandBuffer`; re-records only on a cache miss. Benchmarks must distinguish first-inference latency (recording path) from steady-state latency (replay path). — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): Barrier model is per-edge (`vkCmdPipelineBarrier2` or `vkCmdPipelineBarrier` fallback), not global. This enables driver-side parallel dispatch scheduling. Benchmarks should profile this on real hardware to confirm the driver exploits parallelism. — decided by Switch

📌 Team update (2026-07-28T17:59:54-07:00): Benchmark reporting requirements — report island count and largest fused region alongside wall time. A change that increases claim rate but shrinks fused region size is a regression. These metrics must appear in every benchmark run. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): VkBuffer handle registry (opaque 64-bit tag → `(VkBuffer, offset)` map) — OQ-3 tracks whether the hash lookup is ever a bottleneck. Niobe's profiling will answer this; it is not expected to be hot. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): M0 benchmark — single `Add` node, software rasterizer. Numbers are not meaningful at M0; the goal is establishing the measurement infrastructure. — decided by Morpheus

---

## Cross-agent context appended (2026-07-28T22:28:08-07:00) — implementation-round decisions constraining Niobe

📌 **Baseline is now frozen (Morpheus OQ-1):** Device gate = Vulkan ≥1.1 + compute queue + `maxComputeWorkGroupInvocations ≥ 256` + `maxComputeSharedMemorySize ≥ 16384` + subgroup BASIC + one DEVICE_LOCAL + one HOST_VISIBLE memory type. No required extensions. `synchronization2` and `subgroup_size_control` are probed, not required. Capability shortfalls degrade op coverage, not device availability. Benchmark device tiers will vary accordingly.

📌 **`largest_island_flops` is the metric of record (Mouse OP_COVERAGE.md + Morpheus ratification).** Do NOT use `claimed_node_fraction` as a target. Required benchmark reporting fields per run: `island_count`, `largest_island_nodes`, `largest_island_flops`, `boundary_bytes_per_inference`, `boundary_time_fraction`, `declined_nodes` histogram keyed by `deny!` reason. A change that increases claim rate but shrinks fused region size is a regression.

📌 **OQ-12 pass bar (Morpheus §11.1): ≥ 1.5× over the device's own ORT CPU EP on a GEMM-anchored subgraph, zero numerical failures.** This is what §7.3's Android claim rests on. M3 Android tuning budget unlocks only if Android A+B class devices pass all three OQ-12 stages. Niobe owns stage 3 (execution timing measurement). Morpheus pre-committed reversal conditions: if ≥ 1.5× not met on the median Android tier, §7.3 is retracted.

📌 **MVS constants are provisional and must be re-derived by Niobe from M2 measurements.** Current values: `SAFETY = 3.0`, `node_count ≥ 4`, `64 KiB` output floor. These are placeholders set at 3× because the cost model is crude; they are not settled. `TransferModel::fit` (least squares over `(bytes, ns)` calibration samples) is the calibration hook in Mouse's partition code — Niobe replaces placeholder constants with real measurements. Do not treat these constants as fixed.

📌 **Trinity's oracle pinning rule (trinity-test-foundation):** For `MatMulNBits`, `accuracy_level` must be pinned at 1 (fp32 accumulator). Level 4 (int8 VNNI) diverges ~3.6e-3 max_abs at K=1024, N=512 — would present as a GPU kernel bug. Pin via `MATMULNBITS_ORACLE_ACCURACY_LEVEL=1`. Gate fp16 oracle on ORT ≥ 1.28 (1.27 produces NaN/Inf via null-allocator PrePack bug). When writing benchmark oracles: any knob ORT selects by sniffing host CPU must be pinned or expected values drift across runners.

📌 **ORT version policy (Tank + Fact Checker):** Compile against ORT 1.28 (`ORT_API_VERSION = 28`), minimum host 1.24, refuse 1.27. Three-number version negotiation: ship=28, min=24, negotiated version stamped into every `ort_version_supported` vtable field we hand ORT. Benchmark suite must pin `onnxruntime>=1.28`.

📌 **Engine: dual-backend barrier abstraction (Switch).** `synchronization2` probed at device init; legacy `vkCmdPipelineBarrier` backend required for devices without it. `ep.force_legacy_barriers=1` session option forces legacy backend in CI parity testing. If benchmarks show different performance between backends, report which backend was active.

📌 **OQ-3: reserved-VA opaque-handle registry (Tank proposal, Morpheus ruling).** No BDA. The handle table lookup may appear in profiling; it is not expected to be hot. If it shows up as a hotspot in Niobe's measurements, that is the signal to optimize the table.

📌 **Hard Vulkan SDK build dependency (Morpheus OQ-4).** Shader-less artifact advertises zero devices. Set `$env:ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC='1'` on machines without the SDK (produces inert build, not suitable for benchmarking).

📌 **llama.cpp accelerant (Rai 🟢 Green, Switch D-S4-10).** Tiling strategy, subgroup reduction shape, dequant-in-register patterns transfer as algorithmic reference. No code copying. Budget algorithm study time for GQA and MatMulNBits tiling specifically before implementing Niobe's GEMM benchmarks.

📌 **`concentration()` metric (Mouse registry).** `largest_island_flops ÷ total_claimed_flops`. Report this alongside `node_coverage`. Two graphs can have identical node coverage and wildly different concentration; concentration is the honest performance predictor.
