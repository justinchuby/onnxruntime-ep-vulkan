# onnxruntime-ep-vulkan — Op Coverage Plan

**Owner:** Mouse (Op Coverage Engineer)
**Status:** Proposed — 2026-07-28T18:51:35-07:00
**Supersedes:** `docs/DESIGN.md` §8.2 (the v0 op set) and the op-related rows of §1.2 (v1
non-goals). Morpheus ratifies; see §12 for the exact supersession list.
**Reads against:** `docs/DESIGN.md` (architecture of record), `docs/ENGINE.md` (engine contract),
`.squad/decisions.md`.

---

## 0. TL;DR

- **174 ops inventoried** across 16 families, derived from *what real exported graphs contain*, not
  from the ONNX spec index. 2 of the 174 are inventoried-and-permanently-declined
  (`NonZero`, `NonMaxSuppression` — data-dependent output shape).
- **The tier-1 set is 87 ops and is written as roughly 5 shader templates.** That ratio — ops per
  hand-written kernel — is the entire thesis of this document. Op count is not the unit of work;
  *kernel families* are.
- **"Qwen3.5 runs end-to-end on Vulkan"** requires ~63 distinct ops, of which **9 are
  `com.microsoft` contrib ops** (`GroupQueryAttention`, `RotaryEmbedding`,
  `SkipSimplifiedLayerNormalization`, `SimplifiedLayerNormalization`, `MatMulNBits`,
  `LinearAttention`, `CausalConvWithState`, `QMoE`, `GatherBlockQuantized`). Contrib ops are
  **not optional** — the ONNX Runtime GenAI model builder emits them directly, so a decoder EP that
  declines `com.microsoft` cannot run a Qwen graph at all. This reverses `DESIGN.md` §1.2.
- **Leverage comes from four places, and only four:** (1) one generic elementwise kernel family
  covering ~66 ops, (2) one shared broadcast/indexing GLSL header used by every op, (3) one tiled
  GEMM specialized by dtype rather than N bespoke matmuls, (4) a table-driven registry so an op
  costs one table row + one claim predicate + zero boundary edits. Everything else is a bespoke
  kernel and must be budgeted as such.
- **Honest timeline read:** the elementwise/shape/reduction breadth (tiers 1–2, ~120 ops) is
  genuinely days-to-weeks. `GroupQueryAttention` + `MatMulNBits` + `LinearAttention` — the three ops
  that actually decide whether a Qwen3.5 runs — are **not**; each is an XL hand-written Vulkan
  kernel with a nontrivial numerical-correctness surface. See §11.

---

## 1. Why this document exists, and what it changes

`DESIGN.md` §8 was written before Justin's 2026-07-28 directive:

> "mlx 达到这样的 op coverage 只用了几天，不是两年，我们要 target 高 op coverage。当然 focus on
> llm，moe，multi modal，linear attention，qwen3.5，conv 这些类型的模型优先。"
> — `.squad/decisions/inbox/copilot-directive-2026-07-28-b.md`

§8's v0 set is a *risk-reduction* set: `Add`, then elementwise, then shape. It is correct as a
sequencing story and I keep it as tiers 0–1 unchanged. What it does not do is state a destination.
This document states the destination, orders it by model family, and — critically — says where the
speed comes from, because **the MLX comparison is not transferable at face value.**

### 1.1 The honest MLX comparison

`onnxruntime-mlx` reached 184/202 `ai.onnx` ops plus the full contrib set in days
(`onnxruntime-mlx/docs/OP_ARCHITECTURE.md` §2.1). It did that because MLX **already had the op
library**. Look at what an op handler there actually is:

```rust
// onnxruntime-mlx/rust/src/ops/elementwise.rs
fn add_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_add, a, b)?;   // <-- someone else wrote the kernel
    ctx.bind(&n.outputs[0], r);
    Ok(())
}
```

Every line of semantics is there; zero lines of *compute* are. MLX supplied: the kernel, numpy
broadcasting, dtype promotion, unified memory (no explicit transfers), and lazy graph fusion via
`mlx_compile`. That project's marginal cost per op was "read the ONNX spec, write a claim
predicate." Ours is "read the ONNX spec, write a claim predicate, **and author a GLSL compute
shader against explicit device memory with hand-placed barriers, then verify it on five vendors.**"

So the naive extrapolation — "they did 184 ops in days, so can we" — is wrong. Our leverage has to
be *manufactured*, and it is manufactured by making the number of hand-written kernels grow far
more slowly than the number of ops. That is §5, and it is the only interesting part of this plan.

**Target ratio: ≥ 8 ops per hand-written kernel family in tiers 1–2, ≥ 2 in tiers 3–5.** Tier 3+
ops (attention, quantized GEMM, SSM) are irreducibly bespoke; there is no template that writes
`GroupQueryAttention` for you.

---

## 2. Method — deriving the op list from real graphs

The op list below was derived by inspecting, in priority order:

1. **The ONNX Runtime GenAI model builder source** — `microsoft/onnxruntime-genai`,
   `src/python/py/models/builder.py` and `src/python/py/models/builders/qwen.py`. This is the
   authoritative answer to "what is in a Qwen ONNX graph", because it is the program that *emits*
   the graph. It dispatches Qwen / Qwen2.5 / Qwen3 / Qwen2.5-VL / Qwen3-VL / **Qwen3.5** to
   `builders/qwen.py`.
   <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builders/qwen.py>
2. **The GenAI quantization config** — `src/python/py/models/builders/quant_config.py`, which
   defines the dense-vs-MoE weight split, default `block_size` 32 (128 for TRT-RTX), and
   `accuracy_level`.
   <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builders/quant_config.py>
3. **ORT contrib op schemas** — `microsoft/onnxruntime`, `docs/ContribOperators.md` (generated from
   the schema registry, so it is authoritative for attributes/inputs).
   <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md>
4. **The ORT WebGPU EP registries** — `onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc`
   and `onnxruntime/contrib_ops/webgpu/webgpu_contrib_kernels.cc`. This is *the* closest analog to
   what we are building: a compute-shader EP for the same host, written by people who had to make
   the same decisions. Its op list is a strong prior on "what a compute EP actually needs."
   <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc>
   <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/webgpu/webgpu_contrib_kernels.cc>
5. **`onnxruntime-mlx`'s own registry** — a local, already-shipped ONNX→GPU op inventory for the
   same model families (`rust/src/ops/*.rs`, `docs/OP_ARCHITECTURE.md` §2.1). Independent
   corroboration: it has an `ssm.rs` registering `CausalConvWithState` and `LinearAttention`, which
   confirms these contrib ops are real and Qwen3.5-relevant, not speculative.
6. **llama.cpp's Vulkan backend** — `ggml/src/ggml-vulkan/` — for what a hand-written Vulkan LLM
   kernel set costs in practice.
   <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp>

### 2.1 Verification status — read this before trusting a row

| Claim | Status |
|---|---|
| GenAI builder emits `GroupQueryAttention`, `RotaryEmbedding`, `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, `MatMulNBits` for Qwen | **VERIFIED** from `builders/qwen.py` + `quant_config.py` |
| Qwen3.5 hybrid: full-attention layers use GQA + `Sigmoid`/`Mul` output gating; linear-attention layers use `com.microsoft::CausalConvWithState` + `com.microsoft::LinearAttention` with conv-state and `[B, H_kv, d_k, d_v]` recurrent-state cache I/O | **VERIFIED** from `builders/qwen.py` |
| Qwen2.5-VL MRoPE subgraph: `Shape → Gather → Unsqueeze/Expand/Cast → MatMul → Transpose → Concat → Cos/Sin → Mul`, then `Slice/Split/Gather/Squeeze/Concat/Reshape`, then `RotaryEmbedding`, then `GroupQueryAttention` | **VERIFIED** from `builders/qwen.py` |
| `com.microsoft::MoE` and `com.microsoft::QMoE` exist; GenAI defaults MoE experts to int4, block 32 | **VERIFIED** from `ContribOperators.md` + `quant_config.py` |
| WebGPU EP registers `QMoE`, `LinearAttention`, `CausalConvWithState`, `MatMulNBits`, `MatMulNBitsQkv`, `MatMulNBitsMlp`, `GatherBlockQuantized`; `MoE` (float) is **commented out** in that registry | **VERIFIED** from `webgpu_contrib_kernels.cc` |
| `LinearAttention` supports `linear` / `gated` / `delta` / `gated_delta` recurrence rules, packed 3-D QKV, optional decay/beta, present state | **VERIFIED** from `ContribOperators.md` |
| `MatMulNBits` attrs: required `K`, `N`, `bits`, `block_size`; optional `accuracy_level`. Inputs: `A`, packed `B`, `scales`, optional `zero_points`, optional `g_idx`, optional `bias` | **VERIFIED** from `ContribOperators.md` |
| Optimum (`optimum-cli export onnx`) Qwen graphs lower attention to *standard* ONNX primitives rather than GQA | **UNVERIFIED** — plausible and consistent with how torch.onnx export works, but I did not inspect a pinned artifact. Treat as a *hypothesis* that motivates keeping the primitive attention path (tier 2) alive alongside fused GQA. |
| `ArgMax` in the Qwen decoder graph | **UNVERIFIED** — likely a sampling/postprocess concern, not decoder. Do not budget as LLM-critical. |
| `MultiHeadAttention` / `Attention` in the GenAI Qwen path | **UNVERIFIED as emitted** — the verified Qwen paths use GQA. Budget MHA for *non-GenAI* exports (BERT-family, vision encoders), not for Qwen. |
| Unfused MoE export primitives (`TopK`/`Gather`/`ScatterND`/`ReduceSum`) for Mixtral/Qwen-MoE specifically | **UNVERIFIED** — architecturally near-certain, artifact not inspected. Do **not** put `Einsum`/`Loop`/`Scan` in the required MoE set on this basis. |
| Mamba/Mamba2/RWKV/RetNet ONNX exports | **UNVERIFIED, and likely not exportable cleanly** — selective-scan is a custom CUDA kernel with no ONNX lowering (`state-spaces/mamba` issue #200). **Consequence: our "linear attention" target is Qwen3.5's `LinearAttention` contrib op, not Mamba.** This is a significant scoping win and is treated as such below. |
| ResNet-50 / MobileNetV3 op sets | **PARTIALLY VERIFIED** — standard and uncontroversial; exact per-artifact sets (`Resize`/`Split`/`Concat` presence) depend on the chosen export. |

**Rule:** no op enters a tier's *exit criteria* on an UNVERIFIED row. UNVERIFIED ops may be
implemented opportunistically but cannot be load-bearing for a milestone until Trinity has an actual
`.onnx` artifact in `tests/models/` whose node histogram proves the need.

### 2.2 Action item — the graph census harness

Before tier 3 starts, we need `tools/graph_census.py`: takes a `.onnx`, emits a node histogram by
`(domain, op_type, since_version, input dtypes, ranks)`, plus a *claimability* diff against our
registry dump. This is a half-day of Python and it converts every "UNVERIFIED" row above into a
verified one. **It is the highest-leverage tooling investment in this plan** and I want it before
any tier-3 kernel is written. Coordinate with Trinity: the same artifacts become the conformance
corpus.

Target corpus:

| Slot | Artifact | Produced by |
|---|---|---|
| LLM fp16 | Qwen3-0.6B / Qwen3-1.7B | GenAI builder, `-p fp16` |
| LLM int4 | Qwen3-1.7B | GenAI builder, `-p int4` |
| LLM primitive | Qwen2.5-0.5B | `optimum-cli export onnx` (tests the non-fused path) |
| MoE | Qwen3-MoE (smallest available) | GenAI builder, int4 → `QMoE` |
| Multimodal | Qwen2.5-VL / Qwen3-VL vision tower + projector | GenAI builder / optimum |
| Linear attention | Qwen3.5 hybrid | GenAI builder |
| Conv | ResNet-50 (`onnx/models`), MobileNetV3 | model zoo |

---

## 3. Model-family graph profiles

What each family actually adds on top of the previous one. This is the ordering the directive asks
for, and it is *nearly* a superset chain — which is lucky and is what makes the tiering work.

### 3.1 Qwen3-family decoder LLM (fp16, GenAI builder)

**Compute-dominant:** `MatMul` (Q/K/V/O, gate/up/down, LM head) — 90%+ of FLOPs.
**Attention:** `com.microsoft::GroupQueryAttention` (with `past_key`/`past_value` →
`present_key`/`present_value`), `com.microsoft::RotaryEmbedding`.
**Norm:** `com.microsoft::SimplifiedLayerNormalization` (RMSNorm, incl. Q/K norm in Qwen3),
`com.microsoft::SkipSimplifiedLayerNormalization` (RMSNorm + residual, fused).
**MLP:** SwiGLU as `MatMul` ×3 + `Sigmoid` + `Mul` (+ `Add`).
**Qwen3.5 extra:** attention **output gating** = `Sigmoid` then `Mul` on the attention output.
**Glue (bandwidth/latency, not FLOPs):** `Reshape`, `Transpose`, `Shape`, `Gather`, `Concat`,
`Slice`, `Split`, `Squeeze`, `Unsqueeze`, `Expand`, `Range`, `Cast`, `Where`, `Constant`, `Add`.
**Embedding:** `Gather` (and `GatherBlockQuantized` when the embedding table is quantized).

The important structural fact: **without KV cache** the graph is a clean feed-forward chain. **With
KV cache** the graph acquires `past_*`/`present_*` graph inputs/outputs that are *large device
tensors crossing the subgraph boundary every token*. Under the M0/M1 host-I/O memory model
(`DESIGN.md` §6.3) that is a full KV-cache round-trip per token — catastrophically slow. **KV-cache
LLM inference is therefore gated on M2's device allocator, not on op coverage.** This is the single
most important dependency in this plan and it is not mine to fix.

### 3.2 Qwen3-family, int4 (the variant people actually run)

Replaces the dense `MatMul` projections with `com.microsoft::MatMulNBits` (packed `B`, per-block
`scales`, optional `zero_points`). Everything else is unchanged: activations, KV cache, attention,
norms stay float. Possibly adds `DequantizeLinear` if the export is QDQ-style rather than
weight-only, and `GatherBlockQuantized` for a quantized embedding table.

**Consequence: `MatMulNBits` is not an optimization, it is the entry ticket.** An int4 Qwen graph
where we claim everything *except* `MatMulNBits` is a graph shredded into ~200 islands around the
matmuls — strictly worse than not claiming at all (§7).

### 3.3 MoE

GenAI emits the expert block as a single fused `com.microsoft::MoE` (float) or
`com.microsoft::QMoE` (quantized, int4 default, block 32) node rather than per-expert `MatMul`s.
The router `Softmax`/`TopK` may be inside or outside depending on the schema form.

Two strategic notes:
- The ORT **WebGPU EP registers `QMoE` but has `MoE` commented out.** That is a meaningful signal
  about which one is worth a compute-shader implementation, and it aligns with the fact that
  real MoE deployments are quantized. **We do `QMoE` first, `MoE` second, and treat unfused
  routing (`TopK`/`Gather`/`ScatterND`/`ReduceSum`) as a fallback path for non-GenAI exports.**
- `onnxruntime-mlx` **declined `MoE`** — "data-dependent router top-k gather/scatter — cannot lower
  to a static MLX graph." We do not have that constraint: we emit command buffers, not a static
  lazy graph, and an indirect-dispatch or dense-masked formulation is available. **This is one of
  the few places where the Vulkan backend is structurally *better* positioned than the MLX one.**

### 3.4 Multimodal (ViT/CLIP/SigLIP encoder + projector)

Adds, on top of the LLM set: `Conv` (patch embedding — usually a large-stride, non-overlapping
conv that is really a GEMM), `LayerNormalization` (true LayerNorm, not RMSNorm), `Erf`/`Gelu`,
`MatMul`-based full attention (or `MultiHeadAttention` if fused), `Softmax`, `Resize`/`Pad` for
preprocessing, and the MRoPE construction subgraph for the VL text side (verified above).
`Einsum`, `InstanceNormalization`, `Split` are **UNVERIFIED for these specific models.**

The patch-embed `Conv` deserves a note: for a ViT it is `kernel = stride = patch_size` with no
overlap, so it lowers to **reshape + GEMM**, not to a real convolution kernel. That means a
multimodal encoder is reachable *before* we have a general `Conv` — worth exploiting in tiering.

### 3.5 Linear attention / SSM

**The scoping win of this entire document:** the directive names "linear attention" and "Qwen3.5"
together, and Qwen3.5's linear-attention layers are emitted as **two contrib ops**, not as a
`Scan`/`Loop` subgraph:

- `com.microsoft::CausalConvWithState` — short causal depthwise conv with a carried conv state.
- `com.microsoft::LinearAttention` — recurrence rules `linear` | `gated` | `delta` | `gated_delta`,
  packed 3-D Q/K/V, optional prior recurrent state, optional decay and beta, present state out.

So "support linear attention" = "write two kernels", not "make `Scan` fast". Meanwhile generic
Mamba/Mamba2 **does not export to ONNX cleanly at all** (custom selective-scan kernels;
`state-spaces/mamba` #200). **Recommendation: we do not target Mamba. We target Qwen3.5's
`LinearAttention`/`CausalConvWithState`, and we say so explicitly rather than leaving "SSM support"
as an open-ended commitment.** If a `Scan`-based export shows up later, `Scan`/`Loop` are in the
inventory at tier 6 as a recursive-subgraph translation (`onnxruntime-mlx/rust/src/ops/controlflow.rs`
proves the pattern works).

### 3.6 Conv models

ResNet-50: `Conv`, `Relu`, `MaxPool`, `GlobalAveragePool`, `Add`, `Flatten`, `Gemm`, plus
`BatchNormalization`/`Identity` if BN was not folded.
MobileNetV3: adds depthwise/grouped `Conv`, `Clip`, `HardSigmoid`, `HardSwish`, `Mul`, `Sigmoid`
(SE blocks), `Reshape`.

Conv is last per the directive, and that is the right call for a reason beyond priority: **`Conv` is
the op most likely to want image-backed tensors**, which `ENGINE.md` §3.6 deliberately deferred
("buffer-only for v0; image storage deferred until a specific op family demonstrates a measurable
benefit"). Conv is that family. Doing conv last means the buffer-vs-image question is answered with
data, after the LLM path has proven the engine.

---

## 4. The op inventory

174 ops, 16 families. Columns:

- **Dom** — `ai` = `ai.onnx`, `ms` = `com.microsoft`.
- **Families** — L=LLM, Q=quantized LLM, M=MoE, V=multimodal/vision, S=linear-attn/SSM, C=conv.
- **Diff** — S (≤ half a day incl. tests), M (1–2 days), L (3–5 days), XL (> 1 week, multi-vendor
  perf + numerics risk).
- **Tmpl** — the shader template it reuses (§5). `—` = bespoke kernel. `meta` = no dispatch at all.

### 4.1 Binary / variadic elementwise — 23 ops · template `EW-B`

| Op | Dom | Families | dtypes / ranks | Diff | Tmpl |
|---|---|---|---|---|---|
| `Add` `Sub` `Mul` `Div` | ai | L Q M V S C | f32,f16,i32,i64; rank ≤ 6; full numpy broadcast | S | EW-B |
| `Pow` `Mod` `Min` `Max` | ai | L V C | f32,f16,i32 | S | EW-B |
| `Sum` `Mean` | ai | V C | f32,f16; variadic fold | S | EW-B |
| `PRelu` | ai | C V | f32,f16 | S | EW-B |
| `And` `Or` `Xor` `BitShift` | ai | glue | bool, i32/i64 | S | EW-B |
| `BitwiseAnd` `BitwiseOr` `BitwiseXor` | ai | glue | i32,i64 | S | EW-B |
| `Equal` `Greater` `Less` `GreaterOrEqual` `LessOrEqual` | ai | L V | in f32/i32/i64 → out bool | S | EW-B |

### 4.2 Unary elementwise — 27 ops · template `EW-U`

`Neg` `Abs` `Sqrt` `Exp` `Log` `Reciprocal` `Floor` `Ceil` `Round` `Sign` `Erf` `Sin` `Cos` `Tan`
`Asin` `Acos` `Atan` `Sinh` `Cosh` `Tanh` `Asinh` `Acosh` `Atanh` `Not` `BitwiseNot` `IsNaN` `IsInf`

All `ai.onnx`. All **S**, all `EW-U`. f32/f16 (integer forms where the spec allows), any rank.
`Sin`/`Cos` are LLM-critical (dynamic RoPE cache construction in the VL builder — verified).
`Erf` is vision-critical (GELU). The rest are long-tail breadth that costs one table row each.

### 4.3 Activations — 16 ops · template `EW-U` with push-constant params

`Relu` `LeakyRelu` `Elu` `Selu` `Celu` `HardSigmoid` `HardSwish` `Softplus` `Softsign` `Sigmoid`
`Gelu` `Clip` `ThresholdedRelu` `Shrink` `Mish` `Swish`/`SiLU`

All `ai.onnx`, all **S**, all `EW-U`. `Sigmoid` is LLM-critical (SwiGLU gate *and* Qwen3.5 attention
output gating). `HardSigmoid`/`HardSwish`/`Clip` are MobileNet-critical. `Gelu` is
vision-critical. `Clip` claims only constant-or-absent `min`/`max`.

### 4.4 Select / cast — 3 ops

| Op | Dom | Families | Notes | Diff | Tmpl |
|---|---|---|---|---|---|
| `Where` | ai | L V | 3-way broadcast; verified in Qwen glue | S | EW-T |
| `Cast` `CastLike` | ai | L V C | f32↔f16↔i32↔i64↔bool; **no f64 ever** | S | EW-U |

### 4.5 Shape metadata — 9 ops · `meta` (zero dispatches)

`Reshape` `Squeeze` `Unsqueeze` `Flatten` `Identity` `Shape` `Size` `Constant` `ConstantOfShape`

All `ai.onnx`, all **S**. These are **view/alias operations in the plan, not kernels** — the
compiler rewrites the tensor descriptor and emits nothing. This is a large fraction of the node
count in a Qwen graph and it costs us zero GPU work, which is exactly why claiming them is
high-value: each one is an island-welder (§7). `Shape`/`Size` produce host-known values folded at
compile time when the shape is static; when it is not, they are declined.

### 4.6 Data movement / indexing — 20 ops · template `IDX`

| Op | Dom | Families | dtypes / constraints | Diff | Tmpl |
|---|---|---|---|---|---|
| `Transpose` | ai | L Q V | f32,f16,i32; rank ≤ 6 | S | IDX |
| `Concat` `Split` | ai | L Q M V | any axis, static sizes | S | IDX |
| `Slice` | ai | L V | constant `starts`/`ends`/`axes`/`steps` | M | IDX |
| `Gather` | ai | L Q M | **embedding lookup — LLM-critical**; i32/i64 indices | S | IDX |
| `GatherElements` `GatherND` | ai | M | i32/i64 indices | M | IDX |
| `ScatterElements` `ScatterND` | ai | M | **MoE-critical** (unfused routing); needs atomics or a no-duplicate-index precondition | L | IDX |
| `Tile` `Expand` | ai | L V | — | S | IDX |
| `Pad` | ai | V C | constant pads + `constant`/`reflect`/`edge` modes | M | IDX |
| `Range` | ai | L | static bounds only | S | IDX |
| `Compress` | ai | — | data-dependent output shape → **claim only when the condition is a constant initializer** | M | IDX |
| `DepthToSpace` `SpaceToDepth` | ai | V C | — | S | IDX |
| `Trilu` | ai | L V | causal masks | S | IDX |
| `EyeLike` `OneHot` | ai | — | — | S | IDX |
| `ReverseSequence` | ai | — | — | M | IDX |

### 4.7 Reduction — 15 ops · template `RED`

| Op | Dom | Families | Notes | Diff | Tmpl |
|---|---|---|---|---|---|
| `ReduceSum` `ReduceMean` `ReduceMax` `ReduceMin` `ReduceProd` | ai | all | keepdims, multi-axis, negative axes, **noop_with_empty_axes** | M (first) / S (rest) | RED |
| `ReduceL1` `ReduceL2` `ReduceSumSquare` `ReduceLogSum` `ReduceLogSumExp` | ai | V | pre/post elementwise composed with `RED` | S | RED+EW-U |
| `ArgMax` `ArgMin` | ai | V | index output; i64 | M | RED (index variant) |
| `CumSum` | ai | S? | prefix scan — **different kernel shape**, work-efficient Blelloch scan | L | SCAN |
| `TopK` | ai | M | **MoE router**; k small (2–8) → bitonic partial sort in shared memory | L | SORT |
| `NonZero` | ai | — | **DECLINED permanently** — data-dependent output shape, host-bound | — | — |

### 4.8 MatMul family — 4 ops

| Op | Dom | Families | Notes | Diff | Tmpl |
|---|---|---|---|---|---|
| `MatMul` | ai | L V C M | **the op**; batched, rank ≤ 6, broadcast batch dims; f32 then f16 | XL | GEMM |
| `Gemm` | ai | V C | alpha/beta/transA/transB → same GEMM with a prologue | M | GEMM |
| `MatMulInteger` | ai | Q | i8×i8→i32 | L | GEMM (int variant) |
| `Einsum` | ai | V? | **claim only equations that reduce to transpose+GEMM+reduce**; decline the rest | L | GEMM+IDX |

### 4.9 Normalization / softmax — 11 ops

| Op | Dom | Families | Notes | Diff | Tmpl |
|---|---|---|---|---|---|
| `Softmax` `LogSoftmax` | ai | L V | last-axis fast path first; general axis via IDX+RED | M | NORM |
| `LayerNormalization` | ai | V | mean+var two-pass or Welford single-pass | M | NORM |
| `RMSNormalization` | ai | L | opset-23 standard form | S | NORM |
| `SimplifiedLayerNormalization` | **ms** | L | RMSNorm — **VERIFIED emitted for Qwen Q/K norm** | S | NORM |
| `SkipSimplifiedLayerNormalization` | **ms** | L | RMSNorm + residual add, fused; **VERIFIED emitted** | M | NORM |
| `SkipLayerNormalization` | **ms** | V | LayerNorm + residual | M | NORM |
| `GroupNormalization` `InstanceNormalization` | ai | V C | — | M | NORM |
| `BatchNormalization` | ai | C | inference mode only (scale/bias fold) | S | EW-B |
| `LpNormalization` | ai | V | — | S | NORM |

### 4.10 Attention & fused activations — 11 ops

| Op | Dom | Families | Notes | Diff | Tmpl |
|---|---|---|---|---|---|
| `GroupQueryAttention` | **ms** | L | **the single most important op in this document.** Q/K/V + past K/V + seqlens; attrs: `num_heads`, `kv_num_heads`, `scale`, `local_window_size`, `do_rotary`, `softcap`, Q/K-norm controls. In-place KV-cache update. | XL | ATTN |
| `MultiHeadAttention` | **ms** | V | non-GQA fused attention; BERT/ViT exports | L | ATTN |
| `Attention` (contrib) | **ms** | V | packed-QKV form | L | ATTN |
| `Attention` (opset 23/24) | ai | V | standard-domain fused attention | L | ATTN |
| `RotaryEmbedding` | **ms** | L | `num_heads`, `rotary_embedding_dim`, `interleaved`; **VERIFIED emitted** | M | — |
| `RotaryEmbedding` (opset 23) | ai | L | standard-domain equivalent | S | — |
| `FastGelu` `QuickGelu` `BiasGelu` `BiasSplitGelu` | **ms** | V | fused bias+activation | S | EW-B/EW-U |
| `BiasAdd` | **ms** | V | — | S | EW-B |

### 4.11 Quantization — 12 ops

| Op | Dom | Families | Notes | Diff | Tmpl |
|---|---|---|---|---|---|
| `MatMulNBits` | **ms** | Q M | **entry ticket for int4 LLMs.** `K`,`N`,`bits`,`block_size`,`accuracy_level`; inputs `A`, packed `B`, `scales`, opt `zero_points`, opt `g_idx`, opt `bias` | XL | QGEMM |
| `MatMulNBitsQkv` `MatMulNBitsMlp` | **ms** | Q | fused multi-projection forms (present in the WebGPU registry) — **claim only after plain `MatMulNBits` is solid** | L | QGEMM |
| `MatMulBnb4` | **ms** | Q | bitsandbytes nf4/fp4 | L | QGEMM |
| `GatherBlockQuantized` | **ms** | Q | quantized embedding lookup | M | IDX+QGEMM |
| `DequantizeLinear` `QuantizeLinear` | ai | Q V | per-tensor / per-axis / blocked | M | EW-B |
| `DynamicQuantizeLinear` | ai | Q | needs a min/max reduction first | M | RED+EW-B |
| `DynamicQuantizeMatMul` | **ms** | Q | — | L | QGEMM |
| `QLinearMatMul` | ai | Q | — | L | QGEMM |
| `QLinearConv` `ConvInteger` | ai | Q C | — | XL | CONV |

### 4.12 MoE — 2 fused ops (+ the unfused routing path, counted in §4.6/§4.7)

| Op | Dom | Notes | Diff | Tmpl |
|---|---|---|---|---|
| `QMoE` | **ms** | int4 experts, block 32 default. **Do this one first** — it is what GenAI emits and what the WebGPU EP chose to implement. | XL | QGEMM+IDX |
| `MoE` | **ms** | float experts. WebGPU has it *commented out*; MLX declined it. We can do it — command-buffer recording admits a masked-dense or indirect-dispatch formulation that a static lazy graph cannot express. | XL | GEMM+IDX |

### 4.13 Linear attention / SSM — 3 ops

| Op | Dom | Notes | Diff | Tmpl |
|---|---|---|---|---|
| `LinearAttention` | **ms** | Qwen3.5. Rules `linear`/`gated`/`delta`/`gated_delta`; packed 3-D QKV; optional prior recurrent state, decay, beta; present state out. **Claim one rule at a time** — `gated_delta` is a different kernel from `linear`. | XL | — |
| `CausalConvWithState` | **ms** | Qwen3.5 short causal depthwise conv + carried conv state | L | CONV (1-D) |
| `TensorScatter` | ai (24) | in-place state/cache update | M | IDX |

### 4.14 Control flow — 3 ops

`If` `Loop` `Scan` — all `ai.onnx`, all **XL**, no template. Implemented as recursive subgraph
translation (the `onnxruntime-mlx/rust/src/ops/controlflow.rs` pattern) but far harder here:
`Loop`/`Scan` mean *re-recording or replaying a nested command buffer with a data-dependent trip
count*, which fights the record-once/replay-many decision (`decisions.md`). **Tier 6. `If` with a
compile-time-constant condition is the only cheap member.**

### 4.15 Conv / pool / vision — 12 ops

| Op | Dom | Families | Notes | Diff | Tmpl |
|---|---|---|---|---|---|
| `Conv` | ai | C V | **claim in stages:** (a) 1×1/patch-embed → GEMM, (b) depthwise, (c) general 2-D implicit-GEMM, (d) 3-D | XL | CONV |
| `ConvTranspose` | ai | C | — | XL | CONV |
| `MaxPool` `AveragePool` `LpPool` | ai | C | 2-D, no dilation initially | M | POOL |
| `GlobalAveragePool` `GlobalMaxPool` | ai | C | = full reduce | S | RED |
| `Resize` | ai | V C | nearest + linear only; **decline** cubic/roi/antialias | L | IDX |
| `GridSample` `Col2Im` `RoiAlign` `MaxUnpool` | ai | V | long tail, tier 6 | L | IDX |
| `NonMaxSuppression` | ai | — | **DECLINED permanently** — data-dependent, host-bound greedy loop | — | — |

### 4.16 Recurrent — 3 ops

`LSTM` `GRU` `RNN` — `ai.onnx`, **XL**, tier 6, static-unroll only when `seq_len` is constant.
Low priority: no target model family needs them.

### 4.17 Permanently declined (not counted in the 174)

Sequence types (`SequenceAt`/`Construct`/`Empty`/`Erase`/`Insert`/`Length`/`Map`,
`SplitToSequence`, `ConcatFromSequence`), string ops (`RegexFullMatch`, `StringConcat`,
`StringNormalizer`, `StringSplit`, `TfIdfVectorizer`), `ImageDecoder`, `Optional`, `DeformConv`,
`Unique`, `Det`, all training/loss ops, **all of float64**. Same list `onnxruntime-mlx` arrived at
(`OP_ARCHITECTURE.md` §2.1) — these need non-tensor value types, non-numeric data, or a codec. A
GPU compute EP has nothing to offer them.

---

## 5. Leverage strategy — how breadth gets cheap

This is the heart of the plan. Restating the target: **≥ 8 ops per hand-written kernel family in
tiers 1–2.** The 174-op inventory is served by roughly **13 kernel templates** plus ~14 bespoke
kernels.

### 5.1 The shared indexing header — write broadcasting once

Every non-trivial correctness bug in an elementwise op library is a broadcasting or
negative-axis bug. Solve it **once**, in a GLSL header that every op `#include`s, and validate it
with its own dedicated test matrix rather than re-testing it per op.

```glsl
// shaders/glsl/include/indexing.glsl
#define MAX_RANK 6

struct TensorMeta {
    uint  rank;
    uint  shape [MAX_RANK];
    uint  stride[MAX_RANK];   // 0 in a dim => broadcast that dim (no modulo needed)
};

layout(push_constant) uniform Params {
    TensorMeta a, b, y;
    uint n_elem;              // total output elements
} pc;

// Linear output index -> source offset, honouring stride-0 broadcasting.
uint src_offset(uint lin, in TensorMeta t) {
    uint off = 0u;
    for (int d = int(pc.y.rank) - 1; d >= 0; --d) {
        uint c = lin % pc.y.shape[d];
        lin   /= pc.y.shape[d];
        off   += c * t.stride[d];   // stride==0 => contributes nothing => broadcast
    }
    return off;
}
```

Two design points that matter:

1. **Broadcasting is expressed as a zero stride, computed host-side at compile time.** The shader
   never branches on "is this dim broadcast". No `%` against a source shape, no special cases for
   scalar/suffix/full broadcast — one code path.
2. **Host-side normalization is where the ONNX semantics live.** Negative axes, `keepdims`, axis
   permutation, unit-dim insertion, and rank padding to `MAX_RANK` are all resolved in Rust,
   in one shared `ShapePlan` helper, *before* any push constant is filled. Shaders see only
   normalized, non-negative, rank-padded metadata. **A shader must never contain ONNX semantics.**

A **fully-contiguous, identical-shape fast path** is selected by a specialization constant that
compiles the loop away entirely — the common LLM case (`Add` of two identically-shaped residuals)
must not pay a 6-iteration index decode per element.

### 5.2 Template `EW-U` / `EW-B` / `EW-T` — 66 ops, one kernel family

`§4.1 (23) + §4.2 (27) + §4.3 (16)` = 66 ops, plus `Where`/`Cast` = 69 nodes of ONNX surface, from
**one `.comp` per arity**:

```glsl
// shaders/glsl/elementwise_binary.comp
#version 450
#include "indexing.glsl"
#include "dtype.glsl"        // defines SCALAR (float/float16_t/int/uint) per variant

layout(local_size_x_id = 0) in;          // specialization constant: workgroup size

layout(set=0, binding=0) readonly  buffer A { SCALAR a[]; };
layout(set=0, binding=1) readonly  buffer B { SCALAR b[]; };
layout(set=0, binding=2) writeonly buffer Y { OUT_SCALAR y[]; };

OUT_SCALAR apply(SCALAR x, SCALAR v) { return OP_EXPR; }   // <-- the ONLY per-op text

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n_elem) return;
    y[i] = apply(a[src_offset(i, pc.a)], b[src_offset(i, pc.b)]);
}
```

The per-op delta is **one preprocessor define**. `build.rs` owns a table:

```rust
// build.rs
const EW_BINARY: &[EwOp] = &[
    EwOp { name: "add",  expr: "x + v",              dtypes: F_ALL | I_ALL },
    EwOp { name: "sub",  expr: "x - v",              dtypes: F_ALL | I_ALL },
    EwOp { name: "mul",  expr: "x * v",              dtypes: F_ALL | I_ALL },
    EwOp { name: "div",  expr: "x / v",              dtypes: F_ALL | I_ALL },
    EwOp { name: "pow",  expr: "pow(x, v)",          dtypes: F_ALL },
    EwOp { name: "max",  expr: "max(x, v)",          dtypes: F_ALL | I_ALL },
    EwOp { name: "less", expr: "(x < v) ? 1u : 0u",  dtypes: F_ALL | I_ALL, out: BOOL },
    // ... 23 rows
];
```

**Adding `Atanh` is a one-line diff in two tables.** This is the mechanism by which 66 ops cost
about as much as 3.

> **Caveat, stated plainly:** the per-op cost is one line; the per-op *test* cost is not, and the
> per-op *spec-reading* cost is not. `Mod` has an `fmod` attribute. `Pow` has integer-exponent
> semantics. `Clip` has optional inputs that ORT reports as **null interior `OrtValueInfo`s** — a
> real bug class the MLX project hit and fixed. `Round` is banker's rounding. Budget one careful
> hour per op for semantics even when the kernel is free.

### 5.3 Template `RED` — 15 ops, one kernel family

Two-stage reduction (per-workgroup partial via subgroup ops + shared memory, then a second pass),
parameterized by:

- `INIT` / `COMBINE` / `FINALIZE` defines → `Sum`, `Max`, `Min`, `Prod`, `Mean` (= Sum + scale),
  `L1` (= |x| prologue + Sum), `L2` (= x² + Sum + sqrt), `LogSumExp` (= max-shift + exp + Sum + log).
- An `INDEX_OUT` variant → `ArgMax`/`ArgMin` (carry `(value, index)` pairs through the combine).
- A layout mode: **last-axis contiguous** (one workgroup per row — the LLM/softmax case) vs
  **general strided** (uses `indexing.glsl`). Only the first is needed for tiers 1–3; the general
  form can land later without changing any op handler.

`subgroup_size_control` + subgroup ARITHMETIC are in our baseline (`decisions.md`), which is
precisely why Switch insisted on them: the reduction and GEMM kernels can use
`subgroupAdd`/`subgroupMax` with a *known* subgroup size instead of the defensive
shared-memory-only tree. That decision pays for itself here.

`Softmax`, `LogSoftmax`, `LayerNormalization`, `RMSNormalization`, `SimplifiedLayerNormalization`
are all "reduce along the last axis, then elementwise-normalize" — the `NORM` template is `RED`
with a fused epilogue, not a separate kernel family.

### 5.4 Template `GEMM` — one tiled matmul, specialized, not N matmuls

One `matmul_tiled.comp`. Everything else is a *specialization* or a *prologue/epilogue*, never a new
kernel:

| Axis | Mechanism |
|---|---|
| dtype (f32 / f16 / f16-in-f32-accum) | build-time variant (`-D`) |
| tile size `TM`×`TN`×`TK`, workgroup shape | **specialization constants** — driver re-specializes the SPIR-V; enables per-vendor tuning without new source |
| `M`,`N`,`K`, batch strides, lda/ldb/ldc | **push constants** — per-dispatch |
| transA/transB (`Gemm`) | stride swap host-side; **no shader change** |
| alpha/beta/C (`Gemm`) | epilogue define |
| batched / broadcast batch dims (`MatMul` rank>2) | batch index → `gl_WorkGroupID.z`, strides from `indexing.glsl` |
| bias add, GELU, SiLU-gate | epilogue define (fusion, §5.6) |

This mirrors llama.cpp exactly: spec constants for the tile sizes that survive into the pipeline
binary, push constants for the per-dispatch dimensions (`ENGINE.md` §4.4, verified against
`ggml-vulkan.cpp`).

`MatMulNBits` **is** a variant of this kernel with a different `B`-load path (unpack `bits`-wide
values, apply per-block `scale`/`zero_point`) — not a new algorithm. That is the single largest
piece of leverage available on the quantization path (§8).

### 5.5 The dtype variant matrix — generated, never hand-written

`build.rs` already owns GLSL→SPIR-V (`decisions.md`: Switch). Extend it to a **cartesian product**:

```
for op in TEMPLATE_TABLE:
    for dtype in op.dtypes:
        for target in op.spirv_targets:     # vulkan1.1 default; 1.2/1.3 as extra variants
            glslc -DSCALAR=... -DOP_EXPR=... --target-env=<target> -o OUT_DIR/spv/{op}_{dtype}_{target}.spv
```

and emit `OUT_DIR/shader_modules.rs` with a `phf`-style lookup
`(op_kind, dtype, target) → &'static [u8]`. Op handlers ask the `DispatchContext` for
`(op_kind, dtype)`; the engine resolves the best available variant against the device's capability
flags and falls back to an f32 variant with an upcast when `shaderFloat16` is absent
(`ENGINE.md` §4.3). **An op handler never names a SPIR-V file, and never checks a device
capability** — that would violate the layering rule in `decisions.md`.

Cost control: the product is large. Gate it — f16 variants are only generated for ops on the
`f16_relevant` list (everything on the LLM path), not for `Atanh`.

### 5.6 The compose-before-bespoke rule

**Rule: an op is implemented by composing existing templates unless it is on the fusion whitelist.
Adding a bespoke kernel requires a benchmark showing the composed form is the bottleneck.**

Composition means the op handler emits *multiple dispatches* into the same command buffer. That is
cheap for us in a way it is not for a per-op-submission design: `decisions.md` mandates one command
buffer per subgraph, so N dispatches cost N barriers, not N submissions.

| Op | Composed form | Fuse? |
|---|---|---|
| `Gelu` | `0.5·x·(1+erf(x/√2))` — 1 dispatch via `EW-U` | already 1 kernel |
| `Softmax` | max-reduce → sub/exp → sum-reduce → div (4 dispatches) | **Fuse.** Last-axis Softmax is on every attention path; 4 round-trips through VRAM for one row is the classic bandwidth mistake. One workgroup per row, whole row in shared memory. |
| `LayerNorm`/`RMSNorm` | reduce → normalize → scale/shift | **Fuse.** Same argument, plus it is emitted once per layer per token. |
| `SkipSimplifiedLayerNormalization` | `Add` then RMSNorm | **Fuse** — it arrives pre-fused from the exporter; splitting it would be actively regressive. |
| `SwiGLU` (`MatMul`,`MatMul`,`Sigmoid`,`Mul`,`MatMul`) | 5 dispatches | **Compose in tier 3, fuse the `Sigmoid`+`Mul` into the up-proj GEMM epilogue in tier 5** — only after a benchmark. |
| `ReduceL2` | `Mul`(x,x) → `ReduceSum` → `Sqrt` | compose; 3 dispatches, nobody's bottleneck |
| `HardSwish` | `x · clamp(x/6+0.5,0,1)` | compose into one `EW-U` expr |
| `DynamicQuantizeLinear` | min/max reduce → scale compute → quantize | compose |
| `GroupQueryAttention` | ~15 primitive dispatches | **Bespoke, no question.** A composed attention materializes the `[B,H,S,S]` score matrix in VRAM; at S=4096 that is gigabytes. Flash-style tiling is the *point* of the op. |

The whitelist is short on purpose: **Softmax, LayerNorm/RMSNorm(+skip), attention, quantized
GEMM.** Everything else composes until proven otherwise.

### 5.7 Table-driven registry and generated capability reporting

**Decision: table-driven registration via a declarative macro. Not hand-written `register()` calls.**

`onnxruntime-mlx`'s registry is the right *shape* — `(domain, op_type, [min_opset, max_opset]) →
{handler, claim}`, with claim and translate reading the **same table** so "claimed" can never
outrun "translatable" (`rust/src/registry.rs`). I am adopting that invariant verbatim; it is
non-negotiable and `DESIGN.md` §8.1 already requires it.

What I am **not** adopting is its registration ergonomics. There, each op is a 7-line struct
literal:

```rust
registry.register(OpRegistration {
    domain: "", op_type: "Add", min_opset: K_ANY_OPSET, max_opset: K_ANY_OPSET,
    handler: add_op, claim: add_claim,
});
```

At 174 ops that is ~1200 lines of boilerplate whose only job is to be a table. Replace it with one:

```rust
// rust/src/ops/elementwise.rs
ops! {
    // op_type    domain  opsets     handler          claim            caps
    Add          , ai   , 7..       , ew_binary::<Add>, ew_claim_binary, [F32, F16, I32, I64];
    Sub          , ai   , 7..       , ew_binary::<Sub>, ew_claim_binary, [F32, F16, I32, I64];
    Mul          , ai   , 7..       , ew_binary::<Mul>, ew_claim_binary, [F32, F16, I32, I64];
    Pow          , ai   , 12..      , ew_binary::<Pow>, ew_claim_pow   , [F32, F16];
    GroupQueryAttention, ms, ..     , gqa::handle     , gqa::claim     , [F16];
}
```

**Why the macro, defended:**

1. **The `caps` column is the point.** It is not decoration — it is machine-readable, and it
   generates: (a) the runtime dtype check inside the shared claim helper, (b) the `build.rs` shader
   variant list, (c) `docs/OP_SUPPORT.md`, and (d) the `--dump-capabilities` output Trinity's tests
   and `PLATFORMS.md` rows consume. **The support matrix cannot drift from reality because it is not
   written by a human.** My charter's first line is "an op you claim to support but silently get
   wrong is worse than an op you don't support"; a hand-maintained matrix *is* that failure mode.
2. **A 174-row table is reviewable; 1200 lines of struct literals are not.** Diffs in a table make
   coverage changes visible at review time.
3. **It is a `macro_rules!` that emits `const` slices** — zero runtime cost, no proc-macro
   dependency, no build-time hit, and it expands to exactly the struct literals it replaces, so
   debugging is unchanged.

**Alternative rejected — a proc macro / `#[op(...)]` attribute:** more magic, a `syn` dependency,
worse compile times, and the attribute form scatters the table back across files, losing benefit #2.

**Alternative rejected — hand-written, MLX-style:** it is what the reference does and it demonstrably
worked. But that project's registry rows carry *no* capability metadata (dtype support is a claim
predicate's internal business), so its support matrix is a hand-written prose table in
`OP_ARCHITECTURE.md` that has to be manually re-verified. We need the matrix to be generated because
we have five vendors, two dtypes, and optional-capability shader variants — a combinatorial space no
human table survives.

I also adopt MLX's `deny!`/`require!` macros verbatim — the "reason travels *with* the decision" idea
is genuinely good, and it means every declined node can explain itself at runtime under a
`ORT_EP_VULKAN_CLAIM_DEBUG` env var. That is worth more than it sounds: the number one debugging
question for an EP is "why didn't it take my node."

---

## 6. Tiered rollout

The unit of progress is **"model family X runs end-to-end on the EP"**, never op count. Each tier's
exit criterion names a model, an artifact, and a measurement.

| Tier | Milestone | Ops added (cum.) | Exit criterion |
|---|---|---|---|
| **T0** | M0 | 1 (1) | `Add`, f32, equal shapes, static. Stock ORT loads the plugin, runs one `Add` on a Vulkan device, matches ORT CPU, on Windows + Linux + lavapipe in CI. **Unchanged from `DESIGN.md` §8.2 — I am not touching this.** |
| **T1** | M1 | 87 (88) | §4.1–4.5 + the cheap half of §4.6. **Exit: a pure-elementwise ONNX graph of ≥ 20 nodes compiles to *one* island, runs on Vulkan, and matches CPU.** Plus: `tools/graph_census.py` exists and has produced node histograms for all 7 corpus artifacts. |
| **T2** | M2 | 33 (121) | §4.7 reductions, §4.8 `MatMul`/`Gemm`, §4.9 norms/softmax, remaining §4.6. **Exit: a small MLP and a BERT-base encoder (primitive attention, no contrib ops) run end-to-end on Vulkan and beat ORT CPU on a discrete GPU.** This is the first tier where a speedup is meaningful rather than dispatch-bound. Requires M2's device allocator. |
| **T3** | M2/M3 | 8 (129) | `GroupQueryAttention`, `RotaryEmbedding`, `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, f16 variants for the whole LLM path, `Gather` embedding, `Trilu`. **Exit: Qwen3-0.6B fp16, GenAI-built, with KV cache, generates correct tokens end-to-end on Vulkan with ≤ 2 islands.** |
| **T4** | M3 | 8 (137) | `MatMulNBits`, `DequantizeLinear`, `QuantizeLinear`, `GatherBlockQuantized`, `MatMulNBitsQkv`/`Mlp`, weight prepacking (§8). **Exit: Qwen3-1.7B int4 generates correct tokens end-to-end, ≤ 2 islands, and beats ORT CPU on tokens/sec on at least two vendors.** |
| **T5a** | M3 | 3 (140) | `LinearAttention` (`gated_delta` first — the Qwen3.5 rule), `CausalConvWithState`, `TensorScatter`. **Exit: Qwen3.5 hybrid runs end-to-end on Vulkan.** *This is the named target of the directive.* |
| **T5b** | M3 | 2 (142) | `QMoE`, then `MoE`. **Exit: a Qwen3-MoE int4 graph runs end-to-end with the expert block on Vulkan, not CPU.** |
| **T5c** | M3 | 12 (154) | `Conv` (patch-embed/1×1 form), `LayerNormalization`, `MultiHeadAttention`, `Erf`/`Gelu`, `Resize`, `Pad`, `Einsum` (restricted). **Exit: a Qwen-VL vision tower + projector runs end-to-end and feeds the T3/T4 decoder in one session.** |
| **T6** | post-M3 | 20 (174) | General `Conv`/`ConvTranspose`/pooling/`BatchNormalization`, `CumSum`, `TopK`, scatter family, control flow, recurrent, vision long tail. **Exit: ResNet-50 and MobileNetV3 run end-to-end and beat ORT CPU.** |

### 6.1 What "Qwen3.5 runs end-to-end on Vulkan" concretely requires

Not an op count — a checklist. All of the following must be true simultaneously:

**Ops (≈63 distinct, 9 of them `com.microsoft`):**

- `MatMulNBits` (int4 dense projections) **or** `MatMul` f16 (unquantized path)
- `GroupQueryAttention` with KV-cache in/out, and Qwen3.5's `Sigmoid`+`Mul` output gating
- `RotaryEmbedding`
- `SimplifiedLayerNormalization` (incl. Q/K norm) and `SkipSimplifiedLayerNormalization`
- `LinearAttention` (`gated_delta`) and `CausalConvWithState` with conv-state and recurrent-state
  cache I/O — for the linear-attention layers of the hybrid
- `QMoE` if the target variant is MoE
- `Gather` (embedding) or `GatherBlockQuantized`
- `Softmax`, `Sigmoid`, `Mul`, `Add`
- The full glue set: `Reshape`, `Transpose`, `Shape`, `Slice`, `Split`, `Concat`, `Squeeze`,
  `Unsqueeze`, `Expand`, `Range`, `Cast`, `Where`, `Constant`, `Cos`, `Sin`

**Engine/infrastructure preconditions — none of which are op work:**

1. **M2's device allocator must exist.** KV cache + conv state + recurrent state cross the subgraph
   boundary *every token*. Under M0/M1 host I/O this is a per-token round-trip of the entire cache.
   Without M2, "Qwen3.5 runs" is technically true and practically useless. **This is the critical
   path, and it is Morpheus/Switch's, not mine.**
2. **f16 end-to-end.** `shaderFloat16` + `storageBuffer16BitAccess` are *optional* capabilities per
   `decisions.md`. On a device lacking them we must either upcast (2× memory, likely OOM for a real
   model) or decline. **Open question for Link: what fraction of target devices have f16 storage +
   arithmetic?** If it is low, the LLM story is desktop-first regardless of what the op plan says.
3. **Dynamic shapes.** Decode has `seq_len=1`; prefill has `seq_len=N`. `DESIGN.md` §1.2 lists
   dynamic-shape fast paths as a v1 non-goal and record-once/replay-many is keyed on shape. Two
   shape buckets (prefill, decode) is the minimum; a growing KV length means either shape-bucketed
   re-records or shape-independent recording with dimensions in push constants. **I propose:
   dimensions live in push constants for every LLM-path kernel from day one, so the recorded
   command buffer is length-agnostic.** This must be agreed with Switch — it constrains kernel
   design and it is much cheaper to decide now than to retrofit.
4. **≤ 2 islands.** Measured, not assumed (§7).
5. **A correctness oracle that scales.** ORT CPU on a 1.7B int4 model is slow but feasible for a
   handful of tokens; per-layer intermediate comparison will be needed rather than final-logits
   comparison. Trinity's problem, flagged early.

---

## 7. Claiming and fallback policy

### 7.1 The hard rule (unchanged, restated)

**Only claim a node when the exact `(domain, op_type, opset, input count/presence, dtypes, ranks,
attribute values, broadcast form, static-shape availability)` tuple is genuinely handled.**
Everything else declines with a reason. Claim and translate read the same table. A wrong answer is
worse than a CPU answer, because CPU fallback is always correct.

The `caps` column of §5.7 plus a shared claim helper makes the common case automatic; ops with real
attribute surface (`Conv`, `Resize`, `GroupQueryAttention`, `LinearAttention`) get an explicit
hand-written predicate using `require!`/`deny!`.

**Specific traps I expect to bite us, from the reference project's scars:**

- **Null interior optional inputs.** ORT returns a null `OrtValueInfo` for an omitted *interior*
  optional input (`Clip` min/max, `Resize` roi, `GroupQueryAttention` optional caches). The MLX
  project shipped a crash here before fixing it. Null-guard input names in the clustering pass
  *before* building dataflow edges. Non-negotiable, and it belongs in the boundary layer, not in 60
  op modules.
- **Zero-size / empty tensors.** MLX's conformance fuzzing found 16 crash classes here. Decide once,
  centrally: **empty tensors are handled on-device** (dispatch with `n_elem == 0` is a no-op that
  still produces a correctly-shaped output), not declined. Declining is a partition hazard.
- **int64 indices.** ONNX says i64; many Vulkan devices have poor or absent 64-bit integer support
  in shaders. **Narrow i64 index tensors to i32 host-side when the shape bounds prove it is safe,
  and decline when it is not.** This is a claim predicate, not a shader concern.
- **Negative axes.** Normalized in the shared host-side `ShapePlan`. Never in a shader.

### 7.2 Death by fallback — the real failure mode

A claim rate of 95% can be *slower* than 0%. If the 5% we decline are distributed through the graph,
a 400-node Qwen graph becomes 40 islands with 39 device round-trips. `DESIGN.md` §8.3 states this;
here is the concrete rule.

**Proposed rule — Minimum Viable Subgraph (MVS). A candidate subgraph is claimed only if:**

```
est_gpu_time(subgraph)  >  transfer_cost(boundary_tensors)  ×  SAFETY   (SAFETY = 3.0)
```

with an unconditional floor:

```
node_count >= 4  AND  total_output_bytes >= 64 KiB
```

unless the subgraph **contains a GEMM/attention/QGEMM node**, which alone justifies a round trip at
realistic LLM sizes.

Where:

- `transfer_cost(bytes) = bytes / measured_pcie_bw + fixed_submit_overhead` — both **measured at
  device-init time**, not guessed. A one-time ~2 ms calibration: time a staged upload+download of a
  few sizes, fit a line. On UMA parts (Adreno, Mali, Apple/MoltenVK, integrated Intel/AMD) the slope
  is near zero and the rule correctly becomes permissive; on a discrete PCIe part it correctly
  becomes strict. **A single hardcoded constant would be wrong on half our platform matrix**
  (`PLATFORMS.md` is Link's).
- `est_gpu_time` = sum of per-op cost estimates from a coarse per-family model (bytes moved /
  measured device bandwidth for memory-bound ops; FLOPs / measured peak for GEMM). Deliberately
  crude — an order of magnitude is enough for a 3× threshold.

**Second rule — the island-merge preference.** When choosing what to implement next, prefer the op
that *merges* two islands over the op that extends one, over the op that creates one. The census
tool (§2.2) makes this measurable: for each unclaimed op type, report **"claiming this op would
reduce island count by N and increase largest-fused-region node count by M."** That number, not
"is this op easy", drives my backlog order.

**Third rule — the anti-orphan pass.** After partitioning, drop any island of 1–3 nodes that is not
GEMM-anchored. Giving 3 nodes back to CPU to avoid 2 round-trips is almost always correct.

### 7.3 Metric contract with Niobe

I need these reported by the benchmark harness, per model, **as first-class outputs alongside wall
time** (`DESIGN.md` §8.3 already requires the first two):

| Metric | Why |
|---|---|
| `island_count` | The fragmentation number. |
| `largest_island_nodes` / `largest_island_flops` | The real coverage metric. |
| `claimed_node_fraction` | Diagnostic only — **explicitly not a target** (`decisions.md`). |
| `boundary_bytes_per_inference` | Total bytes crossing subgraph boundaries. This is what MVS is minimizing. |
| `boundary_time_fraction` | Transfer+sync time ÷ total. **If this exceeds ~20%, coverage work is being wasted and the answer is a partitioning fix, not another op.** |
| `declined_nodes` histogram with reasons | Straight from the `deny!` reasons. This *is* my backlog, auto-generated. |

I will open this as a request to Niobe. The `declined_nodes` histogram is the one I care most about
— it turns "what should Mouse do next" from a judgement call into a sorted list.

---

## 8. Quantization

Mandatory, not optional (§3.2). The plan.

### 8.1 `MatMulNBits` — the shape of the work

Attributes (verified, `ContribOperators.md`): `K`, `N`, `bits`, `block_size` required;
`accuracy_level` optional. Inputs: `A` (float activations), `B` (packed low-bit weights), `scales`,
optional `zero_points` (packed), optional `g_idx`, optional `bias`.

**Claim policy — narrow first, widen with evidence:**

| Variant | T4 claim? |
|---|---|
| `bits = 4`, `block_size ∈ {32, 128}`, symmetric (no `zero_points`) | **Yes** — GenAI's default (`quant_config.py`: dense block 32; 128 for TRT-RTX) |
| `bits = 4` with `zero_points` | **Yes** (second) |
| `bits = 8` | Yes (same kernel, different unpack) |
| `bits ∈ {2, 3, 5, 6, 7}` | **No** — decline; nobody ships them |
| `g_idx` present (act-order / desc_act) | **No** at T4 — it makes the `B` access pattern data-dependent and destroys the coalesced load. Revisit only if a target model needs it. |
| `accuracy_level` | Honour as a *hint* for accumulator dtype; never as a correctness requirement |

**The kernel is a `GEMM` variant, not a new algorithm** (§5.4). The delta is the `B`-tile load:
instead of reading `TK×TN` floats, read `TK×TN/2` bytes, unpack nibbles, and multiply by the
per-block scale. Two sub-variants matter and both should exist:

- **GEMV path (decode, `M = 1`)** — the dominant case for token generation. Memory-bound; the goal
  is to read the packed weights exactly once at full bandwidth. Dequantize into registers, never
  into VRAM.
- **GEMM path (prefill, `M > 1`)** — compute-bound; dequantize a `B` tile into shared memory once
  and reuse it across the `M` tile.

**Never materialize a dequantized weight tensor in device memory.** It defeats the entire purpose —
int4 exists so a 1.7B model fits in 1 GB. This must be stated in the kernel spec I hand to Switch.

### 8.2 Prepacking and the memory model

`DESIGN.md` §5.5/§6.3: weights are uploaded once at `Compile` time in all phases. Quantization
interacts with this in three ways:

1. **Layout transform at compile time, not dispatch time.** The ONNX packing of `B` (row-major
   `[N, K/2]` nibbles) is not the layout a tiled GEMM wants. Repack **once**, on the host, during
   `Compile`, into a tile-friendly interleaved layout, and upload that. Cost: one pass over the
   weights at session creation. Benefit: every dispatch reads coalesced. This is exactly the kind of
   thing prepacking is for and it is free relative to the alternative.
2. **The repack layout is a function of the chosen tile size**, which is a specialization constant,
   which may be tuned per vendor. **Therefore the prepack must run after device selection and the
   packed buffer must be keyed by `(device, tile_config)`.** Flag for Switch: this couples the
   compile-time weight upload to the pipeline-tuning decision. If tile config is chosen per-device
   at init, this is fine; if it is chosen lazily at first dispatch, it is not.
3. **`scales` and `zero_points` are separate small buffers** — keep them separate (extra descriptor
   bindings), do not interleave them into `B`. Interleaving saves a binding and costs the ability to
   read `B` as a dense `uvec4` stream.
4. **Memory accounting.** GenAI int4 with `block_size=32` costs `4 + 16/32 = 4.5` bits/weight
   (fp16 scales) or `4 + 16/32 + 4/32` with zero-points. A 1.7B model ≈ 1.0 GB. **On a 4 GB mobile
   GPU with a KV cache this is tight** — Link should have a row for "minimum device memory for the
   LLM path."

### 8.3 The other quant ops

- `DequantizeLinear` / `QuantizeLinear` — `EW-B` variants with a per-tensor / per-axis / blocked
  scale-index mode. Claim per-tensor and per-axis at T4; blocked (opset 21) at T4 as well since it
  shares the `MatMulNBits` block-index math.
- `DynamicQuantizeLinear` — compose (`RED` min/max → scale → quantize). No bespoke kernel.
- `GatherBlockQuantized` — `Gather` whose gathered rows are dequantized on the fly. Reuses both the
  `IDX` gather index math and the `QGEMM` unpack helper. Needed for quantized embedding tables.
- `QLinearMatMul` / `MatMulInteger` / `ConvInteger` / `QLinearConv` — **tier 6**. These are the
  *activation*-quantized (QDQ CNN) path, a different world from weight-only LLM quantization. No
  target model family needs them.

---

## 9. Op module layout

### 9.1 File tree

Mirrors `onnxruntime-mlx/rust/src/ops/` (per `decisions.md`: mirror the reference; divergences
enumerated), with three deliberate differences noted below.

```
rust/src/ops/
  mod.rs            # module list + the single `register_all()` entry point
  common/
    mod.rs
    shape_plan.rs   # host-side: rank pad, negative-axis normalize, stride/broadcast computation
    claim.rs        # shared claim helpers: dtype sets, rank bounds, broadcast form, static-shape
    templates.rs    # ew_unary<Op> / ew_binary<Op> / reduce<Op> generic handlers
  elementwise.rs    # §4.1 §4.2 §4.3 §4.4  (66+3 ops, ~3 kernels)
  shape.rs          # §4.5 §4.6           (29 ops)
  reduction.rs      # §4.7                (15 ops)
  matmul.rs         # §4.8                (4 ops)
  norm.rs           # §4.9                (11 ops)
  attention.rs      # §4.10               (11 ops)
  quant.rs          # §4.11               (12 ops)
  moe.rs            # §4.12               (2 ops)          <-- divergence: MLX has no moe.rs
  ssm.rs            # §4.13               (3 ops)
  controlflow.rs    # §4.14               (3 ops)
  conv.rs           # §4.15 conv/pool     (8 ops)
  vision.rs         # §4.15 resize/vision (4 ops)
  recurrent.rs      # §4.16               (3 ops)
```

**Divergences from the MLX layout, stated per `decisions.md`:**

| MLX | Here | Why |
|---|---|---|
| `math.rs` separate from `elementwise.rs` | merged into `elementwise.rs` | There, the split is historical (a C++ file boundary). Here every op in both files is one `EW-*` template row; splitting them splits the template table, which is the thing that must stay in one place. |
| `MoE` inside `quant.rs`/declined | own `moe.rs` | We intend to implement both `MoE` and `QMoE`; MLX declined `MoE` outright. Distinct enough (routing + expert dispatch) to warrant a file. |
| `misc.rs`, `stragglers.rs` | **do not exist** | These are entropy sinks. Every op belongs to a named family or it does not get claimed. |
| `random.rs`, `signal.rs` | **do not exist (yet)** | No target model family needs RNG or FFT. Add if a real graph demands it. |

### 9.2 Registry entry format

See §5.7 for the `ops!` macro. The expanded entry:

```rust
pub struct OpRegistration {
    pub domain:    Domain,          // Ai | Ms  (typed, not a &str — prevents "" vs "ai.onnx" bugs)
    pub op_type:   &'static str,
    pub min_opset: i32,             // ANY_OPSET sentinel
    pub max_opset: i32,
    pub handler:   OpHandler,       // fn(&mut DispatchContext, &NodeDesc) -> Result<(), EpError>
    pub claim:     ClaimPredicate,  // fn(&NodeView) -> Result<(), Cow<'static, str>>
    pub caps:      DtypeSet,        // generates: claim check, shader variants, support matrix
}
```

`OpHandler` takes `&mut DispatchContext` — the typed engine trait from `ENGINE.md`. Per the CI-lint
layering rule, nothing in this directory may name `sys::`, `Ort`, `ash`, `vk::`, or `unsafe`.

### 9.3 Capability reporting

Generated, not written. `cargo xtask op-matrix` walks the registry table and emits
`docs/OP_SUPPORT.md`:

| Op | Domain | Opsets | f32 | f16 | i32 | i64 | bool | Notes |
|---|---|---|---|---|---|---|---|---|
| `Add` | ai.onnx | 7+ | ✅ | ✅ | ✅ | ✅ | — | full broadcast, rank ≤ 6 |

CI fails if the checked-in file differs from the generated one. Same table backs
`--dump-capabilities` for Trinity's assertions and Link's `PLATFORMS.md` rows.

---

## 10. Testing implications

Not my document (Trinity's), but three requirements fall out of this plan and I am asserting them:

1. **Template-level testing, not just op-level.** `indexing.glsl` gets its own exhaustive
   broadcast/rank/stride matrix test. If that header is right, 66 elementwise ops inherit
   correctness, and their per-op tests only need to check the *expression*, not the indexing. This
   is the testing analog of the shader-template leverage and it is what makes "87 ops in tier 1"
   testable at all.
2. **Every op test asserts the node ran on `VulkanExecutionProvider`** (`decisions.md`). Non-negotiable.
3. **A conformance fuzz pass per family, run in a per-op subprocess.** MLX's
   `tests/conformance/RESULTS.md` documents that a native EP *can hard-crash the host process* on an
   unhandled op form, and that isolating each op in its own pytest subprocess is what made the crash
   classes discoverable (16 → 0 after hardening). We will have the same problem, worse — a bad
   Vulkan dispatch can hang or reset the GPU. **Design the harness for crash containment from day
   one, not after the first hang.**

---

## 11. Risks — where this is slower than the MLX project, honestly

Ordered by how much I expect each to hurt.

| # | Risk | Assessment |
|---|---|---|
| 1 | **Three XL ops decide everything.** `GroupQueryAttention`, `MatMulNBits`, `LinearAttention` gate T3/T4/T5a. There is no template leverage for any of them. Each is a multi-week, multi-vendor, numerically-delicate kernel that the MLX project got *for free from MLX*. | **The single biggest schedule risk.** The 174-op inventory is genuinely cheap; these three are genuinely expensive, and they are the ones the directive actually cares about. |
| 2 | **Perf per op, not just correctness.** MLX handlers inherited Apple-tuned kernels. Ours are as fast as we write them, on 5 vendor architectures with different subgroup sizes, shared-memory sizes, and cache behaviour. "Correct on lavapipe" is not "useful on Adreno." | High. Budget a tuning pass per kernel family per vendor tier, not per op. |
| 3 | **The device-allocator dependency (M2).** LLM coverage is worthless without it (§6.1). It is the highest-uncertainty part of the plugin-EP ABI per `DESIGN.md` §6.3, and it is not on my critical path to fix. | High, and **not mine**. Flagging loudly. |
| 4 | **Dynamic shapes vs. record-once/replay-many.** Decode/prefill/growing-KV means the recorded command buffer must be shape-agnostic or bucketed. Deciding this late means rewriting every LLM kernel's parameter passing. | Medium-high. **Decide now** (§6.1 item 3). |
| 5 | **f16 is an optional capability.** If a meaningful fraction of target devices lack `shaderFloat16`/16-bit storage, the LLM story is desktop-only regardless of op coverage. | Medium. Depends on Link's OQ-1 follow-up. |
| 6 | **Contrib-op schema churn.** `com.microsoft` ops are versioned by ORT release, not by an opset, and `LinearAttention`/`CausalConvWithState`/`MatMulNBitsQkv` are *new*. A schema change silently changes what our claim predicate should accept. | Medium. Mitigation: pin the ORT version per release (already a decision), and make the census tool part of CI so a schema drift shows up as a claim-rate regression. |
| 7 | **int64 and zero-size edge cases.** The MLX project found 16 crash classes by fuzzing. We will find our own set, and ours can hang a GPU rather than raise a Python exception. | Medium. Mitigated by §10.3. |
| 8 | **`MoE`/`QMoE` are genuinely hard.** Data-dependent routing on a *pre-recorded* command buffer means either masked-dense execution (wastes ~7/8 of the FLOPs at top-2-of-8) or indirect dispatch with device-computed workgroup counts. Neither is a template. | Medium. But note this is an area where we are better positioned than MLX (§3.3). |
| 9 | **`Conv` breadth.** Full ONNX `Conv` (groups, dilation, asymmetric pad, autopad, 1-D/3-D) is a large surface. | Low priority by directive, so acceptable — stage it (§4.15). |

### 11.1 What would have to be true for days-to-weeks to hold

My honest read, stated as falsifiable preconditions rather than a promise:

1. **Tiers 0–2 (121 ops) in 2–3 weeks: realistic**, *if* the template infrastructure
   (`indexing.glsl`, `build.rs` variant generation, the `ops!` macro, the shared claim helpers) is
   built first and is not shortcut. If we hand-write op #1 through #20 before building the template,
   we lose the entire thesis and the schedule with it. **Order matters more than effort here.**
2. **Tier 3 (Qwen3 fp16 end-to-end): 3–5 weeks after T2**, dominated by `GroupQueryAttention` and
   by M2's allocator landing. Not parallelizable away — GQA is one person's deep work.
3. **Tier 4 (int4): 2–4 weeks after T3**, dominated by `MatMulNBits` performance (correctness is
   maybe a week; making the GEMV path hit memory bandwidth is the rest).
4. **Tier 5a (Qwen3.5): 2–4 weeks after T4.** `LinearAttention` `gated_delta` is a genuinely novel
   kernel with, as far as I can tell, no open reference Vulkan implementation to crib from.

**So: "high op coverage" — 121 ops, the whole elementwise/shape/reduction/GEMM surface — is a
weeks-scale goal and I am confident in it. "Qwen3.5 end-to-end on Vulkan" is a months-scale goal.**
Those are different claims and I want them separated in everyone's head, because the op *count*
will look great long before any LLM runs, and that gap is exactly where a coverage project deceives
itself. The metric that keeps us honest is §7.3's `largest_island_flops`, not the op matrix.

The one thing that would genuinely compress the schedule: **treating llama.cpp's Vulkan shaders as a
reference for the three XL kernels** (they are MIT-licensed and solve exactly the quantized-GEMV,
flash-attention, and RMSNorm problems on exactly our platform matrix). Not copying — reading, for
the tiling and subgroup strategies that took that project years to tune. That is a legitimate and
large accelerant and I recommend we plan for it explicitly, with license review, rather than
rediscovering their conclusions.

---

## 12. Supersession notice — for Morpheus

I propose, Morpheus ratifies. These conflict with the architecture of record:

| Doc | Current | Proposed | Rationale |
|---|---|---|---|
| `DESIGN.md` §1.2 non-goals | "quantized ops" out of scope for v1 | **In scope, tier 4** | `MatMulNBits` is the entry ticket for real LLM graphs (§3.2). Without it an int4 Qwen graph shatters into ~200 islands. |
| `DESIGN.md` §1.2 non-goals | "all `com.microsoft` contrib ops" out of scope for v1 | **9 contrib ops in scope, tiers 3–5** | The GenAI builder *emits* them (VERIFIED). Declining `com.microsoft` = cannot run a Qwen graph at all. |
| `DESIGN.md` §1.2 non-goals | "attention fusion" out of scope | **`GroupQueryAttention` in scope, tier 3** | It is not a fusion we perform; it arrives as a single node from the exporter. Decomposing it would materialize `[B,H,S,S]` scores in VRAM. |
| `DESIGN.md` §1.2 non-goals | "dynamic-shape fast paths" out of scope M0–M2 | **Shape-agnostic push-constant kernel parameters from tier 3** | LLM decode/prefill/growing-KV makes this structural, not an optimization (§6.1). Needs Switch. |
| `DESIGN.md` §8.2 | v0 op set ends at M2 with ~25 ops | **174-op inventory, 6 tiers, model-family exit criteria** | The directive. §8.1's *principles* are unchanged and I endorse all 7. |
| `DESIGN.md` §8.3 | fragmentation rule stated qualitatively | **MVS rule with a measured transfer-cost calibration (§7.2)** | "Does this merge two islands" needs a number to be enforceable. |
| `decisions.md` "Ruthless v1 non-goals" | as above | as above | Same three rows. |
| `ENGINE.md` §3.6 | buffer-only, image storage deferred until a family shows benefit | **`Conv` (tier 5c/6) is that family — re-evaluate then, not before** | Agreement, with a named trigger. |

Nothing here contradicts the two hard layering rules, the claim-conservatism rule, the ORT-CPU
oracle, the capability-set Vulkan baseline, or the record-once/replay-many decision. Those are load
-bearing and this plan is built on them.

---

## 13. Open questions

| ID | Question | Owner |
|---|---|---|
| OQ-M1 | Shape-agnostic recording: are LLM-path kernel dimensions in push constants from day one? | Mouse + Switch |
| OQ-M2 | What fraction of target devices support `shaderFloat16` + `storageBuffer16BitAccess`? Decides whether the LLM path is desktop-only. | Link |
| OQ-M3 | Does the prepacked `MatMulNBits` layout need to be keyed on a per-device tile config? (§8.2) | Mouse + Switch |
| OQ-M4 | Benchmark metric contract — `boundary_bytes_per_inference`, `boundary_time_fraction`, `declined_nodes` histogram. | Niobe |
| OQ-M5 | Conformance harness crash containment (per-op subprocess) and GPU-hang recovery. | Trinity |
| OQ-M6 | License review for reading llama.cpp Vulkan shaders as a reference for the 3 XL kernels. | Coordinator |
| OQ-M7 | Does ORT 1.28's `CreateExternalResourceImporterForDeviceImpl` remove the KV-cache round-trip before M2 lands? | Tank / Morpheus |

---

## 14. References

- ORT GenAI model builder — <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py>
- ORT GenAI Qwen builders (Qwen2.5-VL / Qwen3-VL / Qwen3.5) — <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builders/qwen.py>
- ORT GenAI quantization config — <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builders/quant_config.py>
- ORT contrib operator schemas — <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md>
- ORT WebGPU EP standard-op registry — <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc>
- ORT WebGPU EP contrib-op registry — <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/webgpu/webgpu_contrib_kernels.cc>
- llama.cpp Vulkan backend — <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp>
- Mamba ONNX export limitations — <https://github.com/state-spaces/mamba/issues/200>
- ONNX model zoo ResNet-50 — <https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx>
- Local reference: `onnxruntime-mlx/docs/OP_ARCHITECTURE.md`, `rust/src/registry.rs`, `rust/src/ops/*.rs`, `tests/conformance/RESULTS.md`
- This repo: `docs/DESIGN.md`, `docs/ENGINE.md`, `.squad/decisions.md`
