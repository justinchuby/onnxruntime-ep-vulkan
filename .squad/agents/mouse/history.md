# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Op Coverage — ONNX op implementations, registry, graph partitioning
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

📌 Team update (2026-07-28T17:59:54-07:00): Vulkan API baseline is a capability-set, not a version floor. Devices must have ≥1.1 core + compute queue + `synchronization2` + `subgroup_size_control` + subgroup BASIC+ARITHMETIC + workgroup and shared-memory minimums. Everything else (`shaderFloat16`, `shaderInt8`, etc.) is optional and gates shader variants. Mouse's op handlers choose variants via `DispatchContext`; they must never hard-require an optional capability. — decided by Morpheus, Switch, Link, Fact Checker

📌 Team update (2026-07-28T17:59:54-07:00): Hard layering rule — op handlers in `rust/src/ops/` must never reference `sys::`, `Ort`, `ash`, `vk::`, or `unsafe`. CI lint enforces this and fails the build on a violation. Op handlers see only `NodeDesc`, `NodeView`, `TensorRef`, and a `DispatchContext` trait. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): M0 op is a single `Add` node. This is the first deliverable for Mouse. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): Op growth strategy — grow by family (shared shader skeleton, descriptor layout, test file). Prioritize ops that merge existing graph islands or extend an island's edge. Benchmarks must report island count and largest fused region alongside wall time. Maximizing claim rate is actively harmful; maximize fused compute volume instead. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): v1 non-goals include quantized ops, contrib ops, attention fusion, graph-level op fusion, dynamic-shape fast paths, fp64, and data-dependent output shapes. Mouse must not invest in these families for v1. — decided by Morpheus

---

## 2026-07-28T18:51:35-07:00 — Op coverage plan (`docs/OP_COVERAGE.md`)

📌 The MLX op-coverage speed does NOT transfer at face value. In `onnxruntime-mlx` an op handler is
`ctx.binary(mlx_add, a, b)` — MLX supplied the kernel, numpy broadcasting, dtype promotion, unified
memory, and lazy fusion. Here every op is a hand-written GLSL compute shader against explicit device
memory. Our leverage must be manufactured: make hand-written *kernels* grow far more slowly than
*ops*. Target ratio ≥ 8 ops per kernel family in tiers 1–2.

📌 Real Qwen ONNX graphs come from the ORT GenAI model builder
(`onnxruntime-genai/src/python/py/models/builders/qwen.py`), and it EMITS `com.microsoft` ops
directly: `GroupQueryAttention`, `RotaryEmbedding`, `SimplifiedLayerNormalization`,
`SkipSimplifiedLayerNormalization`, `MatMulNBits`. Declining the `com.microsoft` domain means the EP
cannot run a Qwen graph at all. This reverses `DESIGN.md` §1.2.

📌 Qwen3.5 is a hybrid: full-attention layers use GQA + `Sigmoid`/`Mul` output gating; linear-attention
layers use `com.microsoft::CausalConvWithState` + `com.microsoft::LinearAttention` (rules
linear/gated/delta/gated_delta) with conv-state and `[B,H_kv,d_k,d_v]` recurrent-state cache I/O.
So "linear attention support" = two kernels, not a Mamba project. Mamba/Mamba2 do NOT export to ONNX
cleanly (custom selective-scan kernels; state-spaces/mamba#200) — do not target them.

📌 The ORT **WebGPU EP** (`onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc` +
`contrib_ops/webgpu/webgpu_contrib_kernels.cc`) is the closest existing analog to this project and its
op registry is a strong prior on what a compute-shader EP needs. Notably it registers `QMoE` but has
float `MoE` commented out — do `QMoE` first.

📌 `MatMulNBits` is the entry ticket for int4 LLMs, not an optimization: an int4 graph where we claim
everything except the quantized matmuls shatters into ~200 islands. It is a `B`-load variant of the
shared tiled GEMM, not a new algorithm. Never materialize dequantized weights in VRAM. Decode (M=1)
wants a memory-bound GEMV path; prefill (M>1) wants a shared-memory tile path.

📌 Broadcasting must be solved ONCE, in a shared `indexing.glsl` header, with broadcast expressed as a
**zero stride computed host-side** (no modulo, no branch, one code path). All ONNX semantics —
negative axes, keepdims, rank padding — normalize host-side in a shared `ShapePlan`. A shader must
never contain ONNX semantics. Corollary for testing: test the header exhaustively, then per-op tests
only check the expression.

📌 Registry decision: adopt MLX's registry *shape* (one table for both claim and translate; `deny!`/
`require!` with the reason colocated) but diverge on ergonomics — a `macro_rules!` `ops!` table with a
machine-readable `caps: DtypeSet` column that generates the dtype claim check, the `build.rs` shader
variant list, `docs/OP_SUPPORT.md`, and `--dump-capabilities`. CI fails if the checked-in matrix
differs. A hand-maintained support matrix is exactly the drift my charter exists to prevent.

📌 The LLM path is gated on **M2's device allocator**, not on op coverage. KV cache / conv state /
recurrent state cross the subgraph boundary every token; under M0/M1 host I/O that is a per-token
round-trip of the whole cache. Also flagged: LLM kernel dimensions should live in push constants from
day one so recorded command buffers are sequence-length-agnostic (record-once/replay-many is keyed on
shape).

📌 Partitioning needs a number, not a principle. Minimum Viable Subgraph:
`est_gpu_time > transfer_cost × 3.0`, floor `node_count ≥ 4 AND output_bytes ≥ 64 KiB`, waived when the
subgraph contains a GEMM/attention/QGEMM node; plus an anti-orphan pass dropping non-GEMM-anchored
1–3 node islands. `transfer_cost` is **calibrated at device init**, never hardcoded — on UMA parts
(Adreno/Mali/MoltenVK/integrated) the slope is ~0 and the rule must relax accordingly.

📌 Known bug classes inherited from the MLX project's scars, decided centrally up front: empty/zero-size
tensors are handled **on-device** (declining is a partition hazard); i64 indices are narrowed to i32
when shape bounds prove it safe and declined otherwise; **null interior optional inputs** (ORT returns
a null `OrtValueInfo` for `Clip` min/max, `Resize` roi) must be null-guarded in the clustering pass
before dataflow edges are built. MLX's conformance fuzzing found 16 crash classes here — and a bad
Vulkan dispatch can hang the GPU, so fuzz each op in its own subprocess.

📌 Timeline honesty, recorded so it can be checked later: tiers 0–2 (121 ops) is weeks-scale and
achievable — *conditional on building the template infrastructure before op #1*. "Qwen3.5 end-to-end"
is months-scale, gated on three XL kernels (GQA, MatMulNBits, LinearAttention) with zero template
leverage. Op count will look great long before any LLM runs; `largest_island_flops` is the metric that
keeps us honest.

📌 Highest-leverage tooling investment: `tools/graph_census.py` (node histogram by
`(domain, op_type, opset, dtypes, ranks)` + claimability diff against the registry dump). Half a day of
Python; converts every UNVERIFIED assumption into a verified one and auto-generates the coverage
backlog. Required before tier 3.
