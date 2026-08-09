# onnxruntime-ep-vulkan — Op Coverage Plan

**Owner:** Mouse (Op Coverage Engineer)
**Status:** Ratified in part — 2026-07-28T21:01:56-07:00 (contrib domain and the XL kernels are
funded by direct user ruling; the remaining supersession rows await Morpheus)
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
- **User ruling 2026-07-28: the contrib domain is admitted and the XL kernels are committed
  deliverables** (`contrib op 要做`, `matmulnbits那些 都要做`). They move from the risk register into
  §6.0 with per-kernel exit criteria and owners. This does not make them cheap — an int4-quantized
  LLM is now a *functional requirement*, so weight prepacking (§8.2.1) is on the critical path and
  "Qwen3.5 end-to-end" remains a months-scale item. **Report the schedule as two numbers: tier-1
  breadth (weeks) and end-to-end LLM (months). Averaging them is how a coverage project deceives
  itself.**

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

**Accuracy is not uniformly the driver's problem.** Vulkan's precision table gives several of these
built-ins an allowance wider than the `1e-5` these ops are tested at: `asin`/`acos`/`atan` inherit
**4096 ULP** from `atan2`, and `sin`/`cos` are allowed an *absolute* error of `2⁻¹¹` (`4.9e-4`) in
`[-π, π]`, with `tan` inheriting from both. `Asin`/`Acos` therefore no longer call the built-in —
see **DESIGN.md §8.9.28**, which replaces them with a shared minimax core carrying a bound derived
from the specification rather than fitted to a device. `Sin`/`Cos`/`Tan`/`Atan` still call theirs
and are green only because the two drivers we can reach choose to beat their contract; that is
tracked in `BUILTIN_SCREEN` in `tests/ops/test_inverse_trig.py`, which fails if a new built-in call
appears without a recorded decision.

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

**GQA arity, verified 2026-07-28** against `docs/ContribOperators.md` @ `v1.28.0` and
`onnxruntime/core/graph/contrib_ops/bert_defs.cc` on main (the two are identical for this op):
**7–16 inputs, 1–4 outputs.**

```
in : 0 query*  1 key  2 value  3 past_key  4 past_value  5 seqlens_k*  6 total_sequence_length*
     7 cos_cache  8 sin_cache  9 position_ids  10 attention_bias  11 head_sink
     12 k_scale  13 v_scale  14 q_norm_weight  15 k_norm_weight              (* = required)
out: 0 output*  1 present_key  2 present_value  3 output_qk
```

Optional inputs are positional, so a node that uses input 15 has 16 slots with empty names in the
unused ones — the minimum of 7 comes from `seqlens_k` and `total_sequence_length` sitting at
indices 5 and 6, not from three required inputs. For scale: v1.21 was 7–9 inputs and exactly 3
outputs. This schema moves, which is why §9.4's fingerprint machinery exists.

**One finding here changes the T3 plan and I want it visible.** Inputs 14/15
(`q_norm_weight`/`k_norm_weight`) fuse a per-head RMS norm on Q and K into the attention kernel, and
the ORT GenAI model builder sets `q_norm`/`k_norm` for **every Qwen3-family decoder**
(`builders/qwen.py`: `Qwen3Model.make_attention_init` sets both), emitting a 16-input GQA node
whenever its fused-QK-norm path is enabled. ORT's own schema documentation states that an EP not
implementing this "must reject the node when this input is set", so our claim predicate declines it
— that is conformance, not caution. The builder takes a non-fused path for EPs without support,
emitting separate `SimplifiedLayerNormalization` nodes instead, which is the form we want; **whether
we get that form is a builder decision, not ours.** Confirm empirically before T3 is scheduled:
if the fused form is what lands, "GQA works" and "Qwen3 works" are separated by a Q/K-norm variant
of the hardest kernel in the project. Marked as a T3 precondition rather than an assumption.

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

### 4.18 Coverage is relative to a *producer* — and to a *revision* of that producer (2026-07-29)

The single most consequential thing the review of Justin's ONNX crates turned up, and it corrects a
premise this whole document was built on. Morpheus elevated the rule into `DESIGN.md` §8.5: *a
coverage number without a named producer is not well-formed.* The correction below extends it: a
named producer without a named revision is not well-formed either.

§4.1–4.17 derive the op inventory from what the **ORT GenAI model builder** emits. The
`mobius` builder — **authoritatively `onnxruntime/mobius`**, not the `justinchuby/onnx-genai-models`
mirror this section was first written against — builds the same models (Qwen3, Qwen3.5, Qwen2.5-VL,
Qwen3-VL, DeepSeek V3, Mamba) and emits a **substantially different op set**.

**Producer pinned:** `onnxruntime/mobius` @ `87fd878` (`main`, 2026-07-29), MIT-licensed,
`OPSET_VERSION = 24` in `src/mobius/_constants.py`.

| Role in a Qwen3 decoder layer | ORT GenAI builder emits | `onnxruntime/mobius` @ `87fd878` emits |
|---|---|---|
| Attention | `com.microsoft::GroupQueryAttention` (up to 16 inputs) | **`ai.onnx::Attention` @ opset 24** (6–7 inputs, `_outputs=3`) |
| RMS norm | `com.microsoft::SimplifiedLayerNormalization` | **`ai.onnx::RMSNormalization`** (schema v23), incl. **rank-4** for per-head Q/K norm |
| Residual + norm | `com.microsoft::SkipSimplifiedLayerNormalization` (fused) | plain `Add` + `RMSNormalization` at build time; **fused to `SkipSimplifiedLayerNormalization` post-optimization** for EPs that advertise it |
| Rotary | `com.microsoft::RotaryEmbedding` | **`ai.onnx::RotaryEmbedding`** (schema v23) |
| Q/K norm | inputs 14/15 of the GQA node, *or* separate nodes | always separate `RMSNormalization` nodes, on every EP path |
| SwiGLU activation | `Sigmoid` + `Mul` | **`ai.onnx::Swish` @ opset 24** |
| KV cache (static) | — | **`ai.onnx::TensorScatter` @ opset 24** + `Attention` input 6 `nonpad_kv_seqlen` |
| int4 weights | `com.microsoft::MatMulNBits` | `com.microsoft::MatMulNBits` — **the one op both agree on** |
| int4 embedding | — | `com.microsoft::GatherBlockQuantized` |
| quantized MoE | decomposed | **`com.microsoft::QMoE`** (`activation_type="swiglu"`, `swiglu_fusion=2`, 15 inputs) |
| float MoE | decomposed | decomposed — `MatMul`/`TopK`/`Softmax`/`Equal`/`ReduceSum`; **no fused float `MoE` node** |
| QDQ fallback path | — | `DequantizeLinear` (with `block_size`) + `MatMul`, via an inlined function body, for EPs without `MatMulNBits` |

Before this review the registry held only the left-hand column. A Qwen3 model built by Justin's own
toolchain would therefore have had **every norm, every rotary and every attention node declined
`[not-registered]`** — not because we lack the kernel, but because we never wrote the row. The
kernels are the same kernels. That is a coverage loss of roughly 5 nodes per decoder layer, ×28
layers, for zero technical reason.

Both spellings are now registered (`ops/norm.rs`, `ops/attention.rs`), sharing kernels and, where
the attribute vocabularies genuinely match, sharing claim predicates. `RMSNormalization` shares
`simplified_layer_norm` verbatim. `ai.onnx::Attention` gets its own predicate despite the shared
kernel, because the attribute names differ (`q_num_heads` vs `num_heads`), the illegal-combination
set differs (`is_causal`, `qk_matmul_output_mode`, `softmax_precision` have no contrib equivalent)
and the optional inputs sit at different indices — one predicate spanning both would be a predicate
that is wrong about one of them, in the permissive direction.

**This also reorders T3.** `ai.onnx::Attention` is the cheaper first attention target: no
`seqlens_k` indirection, no in-place KV-cache aliasing requirement, no `do_rotary` fold, and rotary
arrives as its own node. It unblocks a model family we can *build ourselves and iterate on locally*,
which matters far more now that we have two working GPUs on this machine. GQA remains committed and
remains the harder kernel. Morpheus ratified this in `DESIGN.md` §10.0.2.

#### 4.18.1 What the re-derivation against the authoritative repo changed

Checked rather than assumed, per the coordinator's instruction. Three of the earlier conclusions
survived; four things are new or corrected.

**Survived.** The three standard-domain rows (`Attention`, `RMSNormalization`, `RotaryEmbedding`)
are still what mobius emits, still with no `_domain=` kwarg, and `MatMulNBits` is still the
quantized linear op. The core §4.18 finding is not weakened.

**Corrected — 1: `GroupQueryAttention` is reachable from mobius after all.** The earlier reading
said mobius "never emits GQA at all". That is wrong. `onnxruntime/mobius` has both a
`RotaryAttentionToGQA` rewrite rule and a direct `Attention._forward_gqa()` fast path, and both are
gated on the *target EP advertising GQA support*. Which spelling arrives therefore depends on how
**we** are described to the builder, not on the model. Same for
`SkipSimplifiedLayerNormalization`, which mobius fuses post-optimization for every EP except QNN,
OpenVINO, TRT-RTX and `onnx-standard`. This is a stronger version of the §4.18 thesis, not a
weaker one: the op set is a function of the producer, the producer *revision*, **and the EP
description we hand it**. It also means our contrib rows are not dead code on the mobius path.

**Corrected — 2: opset 24, not 23**, with real schema consequences. See §4.19.

**New — 3: four ops mobius emits that we had no row for.** `ai.onnx::Swish` (opset 24, the SwiGLU
activation — `x·σ(αx)`, and mobius always uses α=1, i.e. SiLU), `ai.onnx::TensorScatter` (opset 24,
static KV-cache write), and confirmation of `com.microsoft::GatherBlockQuantized` and
`com.microsoft::QMoE` with a concrete attribute vocabulary rather than the low-confidence
fingerprint §9.4 flagged. `Swish` and `TensorScatter` now have rows; `Swish` also has a shader,
because it is one `#elif` branch in the existing `ew_unary` template — the leverage thesis working
exactly as §5.2 claims.

**New — 4: a third domain exists.** For native GGUF formats (MXFP4, IQ4_NL, IQ3_S) mobius emits
`pkg.nxrt::BlockQuantizedMatMul`. We do not register it and should not: it is an nxrt-runtime
extension, not something ORT will hand a plugin EP. Recorded so nobody rediscovers it as a mystery
`[not-registered]` bucket.

**Method consequence — pin the producer revision in the census.** Trinity pins `accuracy_level` on
the `MatMulNBits` oracle because an unpinned accumulator type drifts reference values across runner
hardware. The same argument applies one level up: an unpinned producer revision drifts the *op set*
under a coverage claim. §2.2's census harness must record the producer's commit SHA and default
opset alongside every census, and a coverage number quoted without them should be treated as
unsourced. Raised with Morpheus for `DESIGN.md` §8.5, which is his to edit.

**And it settles R1 (the fused-QK-norm GQA question) for this producer.** Definitive: mobius
applies Q/K norm as *separate* `RMSNormalization` nodes before the attention op, at **rank 4**
(`Reshape` to `[0,0,-1,head_dim]`, norm, `Reshape` back), the choice is *not* conditioned on
execution provider, and norm weights are never passed into the attention or GQA node. The 16-input
form is an ORT-GenAI construct. For the ORT GenAI path the hazard stands unchanged, so §4.10's
precondition is narrowed, not removed. One derived requirement: **the RMS-norm kernel must handle
rank 4**, not just the rank-3 hidden-state case.

Secondary consequence: mobius's `PackQKVForGQA` rewrite requires Q/K/V projections to share an
input with no intervening ops, and Qwen3's Q/K norm sits between them — so **QKV packing never
happens for Qwen3**. Three separate `MatMul` nodes, not one packed one. That is a partitioning
input, and it makes the GEMM the T2 target it already was.

> **Corrected 2026-07-29 (§4.21.2, contradiction 3).** The sentence above is true of Qwen3 and was
> wrongly generalised: I treated packed QKV as a form we would not meet. The Foundry Local census
> shows **both** cached ORT-GenAI models pack QKV on *every* layer — `GroupQueryAttention` inputs 1
> and 2 are empty strings and input 0 carries the fused `qkv_proj` output. "Never happens for
> Qwen3" is a statement about one model family, not about the producer. The predicate has been
> narrowed to require inputs 1 and 2 to be present; it previously never read them and would have
> claimed a packed node.

**Third recurrence, and the rule that supersedes this one (2026-07-29).** §4.18 was the wrong
producer; §4.19–§4.20 were the right producer at the wrong revision; §4.21 is the right producer at
a known revision whose *actual output* was never read. Each time the correction was one level more
concrete than the last, which is the tell that the underlying rule was still too abstract. The
stronger form, which §4.21.2 records and which §8.5 should carry:

> **A claim about what a producer emits is not evidence until it has been read off a graph that
> producer actually produced.** Builder source is a statement of intent; the model file is the fact.

---

### 4.19 Opset 24 — what moved, and why open-ended windows were a correctness bug (2026-07-29)

mobius defaults to opset 24. The standard-domain rows were windowed `23 ..= OPSET_ANY` — open-ended
upward — so opset-24 nodes were *nominally* in range. That is not the same as being correct at 24,
and for one op it was actively wrong.

Verified against onnx v1.22.0 (`defs.cc`, `old.cc`, `operator_sets.h`, `Changelog.md`). Opset 24
shipped in **onnx v1.19.0, 2025-08-26**, and is final. Its dominant theme is the new `float8e8m0`
scale type, which caused 19 pure type-constraint bumps. Two ops are genuinely new and one changed
its arity.

| Op | Changed at 24? | Consequence for us | Action |
|---|---|---|---|
| **`Attention`** | **Yes — new optional input 6 `nonpad_kv_seqlen`** (`int64`, `(batch,)`) | **Correctness bug.** With `is_causal=1` it moves the causal offset to `nonpad_kv_seqlen[b] − q_seq_len` *per batch element*. A kernel that ignores it attends to padding and returns plausible wrong logits. mobius supplies it on its static-cache path. | Window closed at `23 ..= 24`; predicate declines input 6 explicitly (`[missing-input]`… in the *present* direction) |
| **`TensorScatter`** | **New at 24** | The other half of the static-cache pattern; the spec explicitly permits aliasing `present_cache` onto `past_cache`, which is Switch's `bind_aliased_output` seam | Row added, `24 ..= 24`, `"circular"` mode and `axis = 0` declined |
| **`Swish`** | **New at 24** | mobius's SwiGLU activation | Row added, `24 ..= 24`, shader added to `ew_unary`, `alpha = 1` only |
| `RMSNormalization` | **No** — still schema v23, unrevised through opset 27 | none | Window closed at `23 ..= 23` |
| `RotaryEmbedding` | **No** — still schema v23, unrevised through opset 27 | none | Window closed at `23 ..= 23` |
| `QuantizeLinear` / `DequantizeLinear` | Yes — `float8e8m0` admitted as a scale type at 24; revised again at 25 | Low direct risk (we decline the dtype anyway) but the revision *rate* is the hazard | Window closed at `21 ..= 25` |
| `Cast`, `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `Pad`, `Shape`, `Size`, `Identity`, `Constant`, `ConstantOfShape`, `Flatten`, `If`, `Loop`, `Scan`, `SplitToSequence`, `TopK` | Type constraint only (`float8e8m0`, or `bfloat16` for the last two) | None. Our predicates check the *actual* edge dtype against the row's `caps`; a `float8e8m0` tensor is in no capability set we declare, so these decline on dtype without any window change | No change |

#### 4.19.1 The rule this establishes

An `ai.onnx` row's opset window is a **schema-version** window, not a model-opset window:
`Node_GetSinceVersion` reports the version of the op schema ORT resolved, so an `Add` in an
opset-24 model still reports 14. That is why closing `RMSNormalization` at 23 does not exclude
opset-24 graphs — mobius builds at 24 and its RMS-norm nodes still resolve to schema v23.

Given that, the policy is:

> **A standard-domain row may be open-ended upward only if the op has never gained an input or an
> attribute across a revision. Rows for ops in active revision are closed at the highest schema
> version somebody has actually read, recorded in `registry::ONNX_SPEC_READ`.**

An unread future revision then declines as `[opset]` — visible in the histogram, one row edit to
fix — instead of being claimed by a predicate written against an older schema. This is the
`ai.onnx` analogue of what `ContribSchema` + `SCHEMA_VERIFIED_ON` already do for `com.microsoft`
(§9.4). Both now answer "when did anyone last check?" with a date rather than with silence.

The elementwise rows stay open-ended, and that is a judgement, not an oversight: those ops have not
gained an input or attribute in a decade, closing ~70 windows at 27 would decline valid opset-28
graphs the day onnx 1.23 ships, and the only opset-24 change to any of them was a dtype the caps
set already refuses. The closed set is the set with evidence behind it.

**Errata, resolved 2026-07-29.** The `Attention`-24 *reference implementation* was itself wrong for
`nonpad_kv_seqlen != q_sequence_length` (top-left instead of bottom-right causal alignment, NaN for
fully-masked rows). Justin ruled that it is fixed **in place, with no opset bump** — bumping would
fragment compatibility and oblige ONNX to maintain a definition known to be wrong. So
`ai.onnx::Attention`-24 is defined by the corrected semantics and there is nothing for us to gate
on: no dual path, no legacy variant, no onnx-version branch in a claim predicate. Declining input 6
remains correct because we do not implement it, not because its meaning was uncertain.

What the episode leaves behind is a **detection blind spot**, recorded in §9.4.1: a correction
applied without a version change is invisible to both opset windows and contrib fingerprints,
because every number our detectors compare stays identical. The only signal is a differential test
against a *pinned* reference, which makes the oracle's `onnx` version a correctness input rather
than test hygiene — routed to Trinity beside the existing ORT pin.

---

### 4.20 The full opset range — what "the latest opset" actually means (2026-07-29)

Justin: *"onnx最新opset好像是26 我们都要支持"* — support the whole range up to current, not just
the window mobius happens to emit. The number needed resolving first: the coordinator measured
`onnx.defs.onnx_opset_version() -> 27` on the installed onnx 1.22.0, against Justin's 26.

#### 4.20.1 Both numbers are right; they answer different questions

`onnx/defs/schema.h` @ v1.22.0 carries **two** maps:

```c++
map_[ONNX_DOMAIN]                  = std::make_pair(1, 27);   // registered
last_release_version_map_[ONNX_DOMAIN] = 26;                  // last released
```

with the field comment saying the max version *"may be ahead of the last-release-version"* in
non-release builds. So:

| Question | Answer | Where it comes from |
|---|---|---|
| Highest opset ONNX declares **released** | **26** (onnx v1.21.0, 2026-03-27) | `last_release_version_map_` — Justin's number |
| Highest opset **registered**, i.e. that a model can be stamped with and the checker will accept | **27** (onnx v1.22.0, 2026-06-15) | `map_`, which is what `onnx_opset_version()` returns |

Opset 27 is not a draft in any operational sense: `OpSet_Onnx_ver27` registers three ops
(`CausalConvWithState`, `LinearAttention`, `Range`), `helper.make_model` stamps 27 by default, and
`checker` validates at 27. That `last_release_version_map_` still reads 26 in a *release* build
looks like a v1.22.0 oversight rather than a statement that 27 is provisional — but it is not our
call to make, so both numbers are recorded as constants
(`registry::ONNX_OPSET_LAST_RELEASED`, `registry::ONNX_OPSET_REGISTERED`) with a test asserting they
are distinct. **Our windows are written against 27**, the registered maximum, because that is the
number that decides what we can be handed.

#### 4.20.2 The extension was already done — because windows are schema-version windows

The directive assumes closed windows decline newer graphs. For every op we claim, they do not, and
§4.19.1 is why: a row's window is a **schema-version** window. The finding, verified against
`onnx/defs/operator_sets.h` @ v1.22.0 for opsets 25, 26 and 27:

| Op | Our window | Newest schema version that **exists** | Verdict |
|---|---|---|---|
| `ai.onnx::Attention` | `23 ..= 24` | **24** — no `Attention` at 25/26/27 | Already complete. Not extended: there is nothing above it |
| `ai.onnx::RMSNormalization` | `23 ..= 23` | **23** | Already complete |
| `ai.onnx::RotaryEmbedding` | `23 ..= 23` | **23** | Already complete |
| `ai.onnx::TensorScatter` | `24 ..= 24` | **24** | Already complete |
| `ai.onnx::Swish` | `24 ..= 24` | **24** | Already complete |
| `ai.onnx::QuantizeLinear` / `DequantizeLinear` | `21 ..= 25` | **25** — opset 26 is `BitCast`/`CumProd`, opset 27 is the three SSM/`Range` ops | Already complete |
| ~70 elementwise rows | open-ended | various, all ancient | Left open per §4.19.1 |

So an opset-27 model is claimable today for every op above: ORT resolves each node to its schema
version, and every one of those versions is inside a window we already have. **The closed bounds
are complete coverage of the operators, not a restriction on model opset.** The one place the
distinction would have bitten — an op revised above our bound — does not exist in the current spec.

That is a genuinely cheap result, and it is cheap *because* of the schema-version framing. Had the
rows been keyed on model opset, satisfying this directive would have meant re-reading and re-testing
every predicate at four more opsets.

#### 4.20.3 What the range check *did* turn up

Three things, and the first is the substantive coverage win of the exercise.

**1. `LinearAttention` and `CausalConvWithState` are now `ai.onnx` ops at opset 27.** We had them
only as `com.microsoft` rows carrying `MAIN_BASELINE` — main-branch-only contrib ops with low
fingerprint confidence. onnx v1.22.0 standardised both: `LinearAttention-27` (3D packed
`[B, T, H×D]`, GQA-aware, `update_rule ∈ {linear, gated, delta, gated_delta}`, optional
`past_state`/`decay`/`beta`, state type constrained separately from activation type) and
`CausalConvWithState-27` (stateful causal depthwise 1-D conv, weight `(channels, 1, k)`, optional
fused silu/swish, `past_state` in / `present_state` out).

This is §8.5 recurring on the hybrid path: **the same computation has two spellings from two
producers**, and registering one means Qwen3.5's linear-attention layers run on the EP for whoever
exported through ORT and fall back for whoever exported through the standard domain — with no
coverage number saying which. Both spellings now have rows. The `ai.onnx` rows are windowed
`27 ..= 27` and carry no `ContribSchema` (they are versioned by opset, which is the whole point of
standardisation), and their predicates assert only what was read line for line: dtype and
`update_rule` / `activation`. Head-count and chunking attributes were **not** verified in the
standard schema, so the predicates do not mention them; the rows are `Staged(XL_KERNEL)` and the
status check fires before the predicate, so nothing is claimable on an unverified assumption.

**2. `QuantizeLinear-25` has a `precision` attribute, and we were ignoring it.** Two readings of the
schema history disagree about whether `precision` arrived at 23 or 25; the disagreement does not
need settling, because the conservative action is the same either way. `precision` selects the
accumulation precision of the `x / y_scale` division — a *numeric* attribute, i.e. exactly the
`accuracy_level` failure shape Trinity measured on the oracle, where the wrong choice produces a
plausible answer rather than a visible error. `quant_linear` now declines any non-default
`precision`. No producer we census emits it, so this costs nothing and closes a silent-wrongness
path.

**3. Opsets 25/26/27 otherwise touch us only through type constraints.** Opset 25 is an IR-13
`uint2`/`int2`/`float8e8m0` expansion across 18 ops (`Cast`, `Reshape`, `Transpose`, `Pad`,
`Identity`, `Squeeze`, `Unsqueeze`, `Shape`, `Size`, `Flatten`, `Constant`, `ConstantOfShape`, `If`,
`Loop`, `Scan`, Q/DQ); opset 26 adds `BitCast` and `CumProd`; opset 27 bumps `Range` to
float16/bfloat16 with a new `stash_type`. None of these needs a window change — our predicates check
the *actual* edge dtype against the row's `caps`, and `uint2`/`int2`/`float8e8m0` are in no
capability set we declare, so they decline on dtype. `BitCast`, `CumProd` and `Range` have no rows;
they decline as `[not-registered]`, which is the correct answer for an op nobody has implemented.

#### 4.20.4 What this does not buy

Extending or confirming an upward bound protects against exactly one thing: a *declared interface*
moving. It says nothing about the class of change catalogued in §9.4.1, and confirming the bounds
here has, if anything, increased our exposure to it — we now hold windows that cover every published
version of six ops, with no version-shaped signal that would tell us if any of their meanings
changed underneath. See §9.4.2.

---

### 4.21 The Foundry Local census — two real production graphs, and what they overturn (2026-07-29)

Justin: *"搞定之后可以用机器上的 foundry local 模型测试一下"*. Two ORT-GenAI-built models are
cached on this machine, and they are the **first real production graphs this project has seen**.
Everything before §4.21 was derived from schema documents and from builder source. This section is
derived from the models themselves; where the two disagree, the models win.

Reproduced with `census.py`, `probe.py` and `islands.py` (scripts recorded in the decisions inbox).
All three read the proto with `load_external_data=False` — weights are irrelevant to a census and
13 GB of them is not worth loading.

#### 4.21.1 The census

| | **Phi-3.5-mini-instruct** `cuda-int4-rtn-block-32` | **gpt-oss-20b** `v1` |
|---|---|---|
| producer | `onnxruntime-genai` `'0.0.0'` | `onnxruntime-genai` `''` |
| ir_version | 7 | 10 |
| **opset imports** | **`ai.onnx` = 14**, `com.microsoft` = 1 | **`ai.onnx` = 21**, `com.microsoft` = 1 |
| main-graph nodes | 366 (+4 in subgraphs = 370) | 374 |
| subgraphs | **3** — one `If` with two 2-node branches | **1** — no control flow |
| activation dtype | fp16 throughout | fp16, **with fp32 norms** and 100 `Cast` nodes |

**Phi-3.5 histogram** (370 nodes incl. subgraphs): `com.microsoft::MatMulNBits` 161,
`com.microsoft::SkipSimplifiedLayerNormalization` 64, `ai.onnx::Mul` 64,
`com.microsoft::GroupQueryAttention` 32, `ai.onnx::Sigmoid` 32, `Constant` 7, `Cast` 2, `Gather` 2,
and one each of `Greater`, `If`, `ReduceSum`, `Sub`, `Shape`, `SimplifiedLayerNormalization`.

**gpt-oss-20b histogram** (374 nodes): `ai.onnx::Cast` **100**, `com.microsoft::MatMulNBits` 73,
`ai.onnx::Add` 72, `com.microsoft::SkipSimplifiedLayerNormalization` 48,
`com.microsoft::GroupQueryAttention` 24, `ai.onnx::Reshape` 24, `com.microsoft::QMoE` 24,
`Constant` 3, `Gather` 2, and one each of `Shape`, `ReduceSum`, `Sub`,
`SimplifiedLayerNormalization`.

Node-level detail that only a probe shows:

- **GQA, Phi-3.5:** 9 declared inputs, 7 occupied. Inputs 1 and 2 (`key`, `value`) are **empty
  strings** and input 0 is `qkv_proj/MatMul/output_0` — packed QKV. Inputs 5/6 are INT32
  `seqlens_k` / `total_sequence_length`, shared by all 32 layers. Inputs 7/8 are
  `cos_cache`/`sin_cache`. Attributes: `num_heads = kv_num_heads = 32` (so this is MHA, not
  grouped), `scale = 0.102062`, `softcap = 0`, **`do_rotary = 1`**, `rotary_interleaved = 0`.
- **GQA, gpt-oss:** 12 declared inputs, 8 occupied — additionally **input 11 = `attn.sinks`**
  (attention sinks). `num_heads = 64`, `kv_num_heads = 8` (real 8:1 grouping),
  **`local_window_size = 128` on 12 layers and `-1` on the other 12** — alternating sliding-window
  attention. `do_rotary = 1` again.
- **`MatMulNBits`, Phi-3.5:** 3 inputs — `B` is UINT8, `scales` fp16, **no zero-points** (symmetric
  RTN). `bits = 4`, `block_size = 32`, `accuracy_level = 0`, `K ∈ {3072, 8192}`,
  `N ∈ {3072, 8192, 9216, 32064}`.
- **`MatMulNBits`, gpt-oss:** 4 inputs — with zero-points. **`bits` is 4 on 60 nodes and 8 on 13**,
  in the same model. `block_size = 32`, `K ∈ {2880, 4096}`, `N ∈ {5120, 2880, 32, 201088}`.
- **`SkipSimplifiedLayerNormalization`:** 3 inputs, **4 declared outputs** of which only slots 0 and
  3 are occupied in all 112 nodes across both models. Slot 3 is the residual sum and it feeds the
  next block, so a kernel producing only output 0 breaks the residual stream.
- **`QMoE`, gpt-oss:** 11 declared inputs, 8 occupied (input, router probs, `gate_up_proj`
  qweight/scales/bias, `down_proj` qweight/scales/bias; the last three slots empty). Attributes:
  `k = 4`, `activation_type = "swiglu"`, `activation_alpha = 1.702`, `activation_beta = 1.0`,
  `expert_weight_bits = 4`, `normalize_routing_weights = 1`, `swiglu_limit = 7.0`,
  `swiglu_fusion = 0`, `use_sparse_mixer = 0`.
- **The `If`, Phi-3.5:** produces `cos_cache`/`sin_cache` in a one-time prologue. It is *not* in the
  decoder loop. gpt-oss has no `If` at all — it ships the caches as initializers.

#### 4.21.2 Five recorded conclusions the real graphs contradict

Stated plainly, in the §4.18 spirit. Four of the five are errors in the permissive or the
mis-scoped direction, which is the direction that matters.

**1. The opset *floor*, which I never considered at all.** §4.20 established the ceiling with some
care and treated the floor as settled. It is not: Phi-3.5 imports `ai.onnx` at **14** and gpt-oss at
**21** — same producer, different versions, eight opsets apart. `OPSET_STD_LLM = 23` is not merely
an upper-bounded window, it is a **floor that excludes both real models outright** for every
standard-domain row that uses it. `RMSNormalization`, `ai.onnx::Attention` and `ai.onnx::Swish`
cannot match anything here — not because the ops are missing but because the graphs are older than
the schema versions we window. Our elementwise rows start at 7 and are unaffected, which is the only
reason the census shows any claimable standard-domain nodes at all.

This is not a bug to fix by lowering bounds — `RMSNormalization` genuinely does not exist before
opset 23. It is a **scope correction**: the standard-domain LLM rows target mobius, and *no
ORT-GenAI graph will ever use them*, because GenAI puts all of that work in `com.microsoft`. Two
producers, two disjoint op sets, and the tier plan had silently assumed one.

**2. `do_rotary = 1` is universal, not optional.** `group_query_attention` declines any node with
`do_rotary != 0`, on the reasoning that fused rotary is a later variant. Every GQA node in both
models sets it, and **neither graph contains a separate rotary node at all** — Phi-3.5 has no
`RotaryEmbedding` in any domain. So the "claim GQA first, add fused rotary later" plan claims
exactly zero nodes. Fused rotary is not a variant of the GQA kernel; on this producer it *is* the
GQA kernel. Predicate unchanged (the kernel does not do it yet), scope corrected.

**3. Packed QKV was a permissive hole, and I have closed it.** §4.10 recorded that
`PackQKVForGQA` "never matches Qwen3" and treated the packed form as hypothetical. It is the form
both models use on every layer. Worse, `group_query_attention` never read inputs 1 or 2 at all, so
it would have **claimed** a packed node and handed the kernel a fused `[B, S, (Nq+2Nkv)·H]` tensor
where it expected a query. That is the exact failure §7 exists to prevent and it was sitting in the
predicate. Inputs 1 and 2 are now required to be present, which narrows the claim.

**4. Contrib ops appear in the *default domain*.** `SimplifiedLayerNormalization` is emitted with
`node.domain == ""` — not `"com.microsoft"` — by both models. ONNX opset 14 and 21 publish no such
operator. This is a third category the registry did not model: **no ONNX schema, but the standard
domain**, so `Node_GetSinceVersion` returns a number that means nothing and only a fingerprint can
detect drift. Our row was `com.microsoft::` only, so every real graph declined it as
`[not-registered]` — "we have never heard of this op", when the truth was "we registered it under a
name the producer does not use". Now handled with a second row and an explicit hazard register,
`registry::ORT_FUSED_IN_DEFAULT_DOMAIN`, whose test caps it at four entries: if it grows, the
registry needs a third `Domain` variant rather than a list.

**5. `QMoE` top-k and the T5b scope.** `supports_top_k` admits 1 and 2, written from schema reading.
The only real MoE graph we have routes **top-4**. The fingerprint was also missing
`activation_alpha`/`activation_beta`, so a real node declined as `[contrib-schema]` — the right
answer for the wrong reason, which is precisely the misattribution §9.2.1 exists to prevent. The
attributes are now known; the top-k bound is unchanged and now has a test explaining what evidence
would justify raising it.

**The recurrence is the finding.** This is the third time in two days that a conclusion drawn from a
document has been overturned by an artefact: §4.18 (wrong producer), §4.19/§4.20 (right producer,
wrong revision), and now §4.21 (right producer, but never actually looked at its output). The rule
in §8.5 — *a coverage number without a named producer at a version is not well-formed* — is
necessary and was not sufficient. The stronger form:

> **A claim about what a producer emits is not evidence until it has been read off a graph that
> producer actually produced.** Builder source is a statement of intent; the model file is the fact.

#### 4.21.3 What "Phi-3.5 runs end-to-end on Vulkan" actually requires

The partition simulation (`islands.py`, an *optimistic* approximation of `ops/partition.rs` — it
ignores dtype and rank predicates, so it upper-bounds coverage and lower-bounds island count):

| Claim set | Phi-3.5: claimed / islands / largest | gpt-oss: claimed / islands / largest |
|---|---|---|
| T0 nothing | 0 % / 0 / 0 | 0 % / 0 / 0 |
| T1 elementwise+shape | 28 % / **35** / 5 | 28 % / **52** / 49 |
| T1 + `Cast` | 29 % / 35 / 6 | 54 % / **125** / 49 |
| T2 + norms | 47 % / 35 / 66 | 67 % / 28 / 172 |
| T3 + `GroupQueryAttention` | 55 % / 34 / 66 | 74 % / 3 / 172 |
| **T4 + `MatMulNBits`** | **99 % / 1 / 364** | 93 % / 1 / 349 |
| T5b + `QMoE` | 99 % / 1 / 364 | **100 % / 1 / 373** |

Two things fall out, and both are sharper than anything the tier plan asserted.

**Partial coverage of these graphs is worth nothing.** Phi-3.5 sits at 34–35 islands from T1 all the
way through T3. 161 `MatMulNBits` nodes are interleaved through every layer, so until they are
claimed the graph is confetti no matter what else we add. Then `MatMulNBits` alone takes it from
55 % / 34 islands to **99 % / one island of 364 nodes**. There is no gradual approach to this model:
it is one op away from complete and, without that op, three tiers of work buy nothing measurable.

**Claiming more ops can make partitioning strictly worse.** On gpt-oss, adding `Cast` to the T1 set
raises coverage from 28 % to 54 % and raises the island count from 52 to **125** — a 2.4× increase
in the number of subgraph boundaries, each one a device transfer. This is death-by-fallback observed
rather than argued, and it is the concrete justification for §7's minimum-viable-subgraph rule.
It also identifies the metric Niobe needs: **island count and largest-island size, tracked together,
per producer at version** — coverage percentage alone would have called the `Cast` step a 26-point
improvement.

**The requirement list.** Five op types cover 353 of Phi-3.5's 366 main-graph nodes:

| Op | Count | Status against our predicates today |
|---|---|---|
| `com.microsoft::MatMulNBits` | 161 | **Claimable as specified** — bits 4, block 32, K%32 = 0, no `g_idx`, 3-input symmetric form. Only the kernel is missing |
| `com.microsoft::SkipSimplifiedLayerNormalization` | 64 | Fingerprint fits (3 in, 4 out, `epsilon`). Needs the row-reduction template and **must emit output 3** |
| `ai.onnx::Mul` / `Sigmoid` | 64 / 32 | Already in the elementwise table, opset 7 floor, fp16 |
| `com.microsoft::GroupQueryAttention` | 32 | **Declined on three counts**: `do_rotary = 1`, packed QKV, and the kernel. All three must land together |
| `ai.onnx::SimplifiedLayerNormalization` | 1 | Row added this turn; shares the RMSNorm kernel |

Plus a 13-node cold prologue (`Cast`, `Gather`, `Greater`, `If`, `ReduceSum`, `Sub`, `Shape`,
`Constant`) that can stay on CPU without shredding anything, because it runs once and feeds
`cos_cache`/`sin_cache` rather than sitting between decoder layers.

So the honest end-to-end requirement is **three kernels**: block-quantized GEMM/GEMV, fused
skip-RMSNorm with a residual output, and MHA-with-fused-rotary-and-packed-QKV over an fp16 paged KV
cache. Not 87 ops. Not a tier ladder. Three kernels and the elementwise table we already have.

#### 4.21.4 Is Phi-3.5 a better first target than Qwen3.5?

Yes, and it is not close.

- **It is on this disk and runnable.** No Qwen3 graph exists on this machine, and every statement we
  have made about Qwen3's exported form is inference from a builder we have not run.
- **Its attention is MHA.** `num_heads = kv_num_heads = 32`, so the grouped-query indexing is the
  identity and the first attention kernel can skip the KV-head broadcast entirely.
- **`softcap = 0`, `local_window_size` absent, no attention sinks, no QK norm.** Every numeric
  option we deliberately decline is off. gpt-oss, by contrast, has sliding windows on half its
  layers and attention sinks on all of them.
- **Its quantization is the easy corner.** Symmetric RTN, no zero-points, uniform `bits = 4`,
  `block_size = 32`, and every `K` a multiple of 32. gpt-oss mixes 4-bit and 8-bit weights in one
  model and uses zero-points.
- **Its control flow is cold.** The single `If` is a prologue, not a loop body.
- **It is small enough to hold.** 366 nodes, five op types, 2.2 GB of weights that fit on both GPUs
  in this machine.

The cost is that Phi-3.5 exercises **none** of the standard-domain LLM rows (§4.20) — it is a pure
`com.microsoft` graph at `ai.onnx` 14. So it does not validate the mobius path, and §8.5 means we
must report the two separately rather than letting one stand in for the other. That is a reporting
obligation, not an argument against it.

**Proposed amendment to the T3 exit criterion**, for Morpheus since §10.0.2 is his: keep
`ai.onnx::Attention` as the *implementation* entry point — it is the simpler schema and it serves
mobius — but make the *demonstration* Phi-3.5, because it is the only LLM we can actually run
end-to-end and measure. And add a T4 exit criterion that is now precisely measurable:
**`MatMulNBits` claimed ⇒ Phi-3.5 partitions into one island of ≥ 360 nodes.**

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

#### 5.1.1 The parameter tail — sixteen bytes that retired a whole blocker (2026-07-29)

Fourteen rows — `LeakyRelu`, `Elu`, `Selu`, `Celu`, `ThresholdedRelu`, `Shrink`, `HardSigmoid`,
`Swish` and their kin — were staged behind `NEEDS_PARAMS` for the whole of this project's life.
Their shaders were written and compiling from the start; each was one float away from claimable.
The blocker was §7.1's rule, correctly applied: their GLSL had the ONNX **default** attribute
value baked in as a literal, and claiming on that would have answered a graph setting
`alpha = 0.2` with `alpha = 0.01` — a wrong answer, not an error, on a graph we said we could run.

The fix was **one mechanism, not fourteen**. The push-constant block gained a fixed four-float
tail after the stride arrays:

```text
offset  size  field
0       4     rank
4       4     elem_count
8       24    out_shape[6]
32      24    strides[0][6]
56      24    strides[1][6]      (arity >= 2)
80      24    strides[2][6]      (arity >= 3)
<after last stride array>
        16    params[4]          (f32; all zero when the op has no attributes)
```

Worst case moves from 104 to 120 bytes, still inside the 128-byte `maxPushConstantsSize` floor
Vulkan 1.1 guarantees. Three decisions in it are load-bearing:

1. **The tail is unconditional.** Ops with no attributes push four zeros they never read. The
   alternative — a short block and a long one — means the shader's declared block and the bytes
   actually pushed can disagree, and `vk/pipeline.rs` declares a **fixed 128-byte** push-constant
   range for every pipeline, so the layout will not catch the disagreement: the shader would read
   bytes nobody wrote. Sixteen bytes of zeros is much cheaper than that bug class. (The fixed
   range is also why this change needed nothing from the engine layer — growing the block by 16
   bytes changes no pipeline layout.)
2. **Slot assignment lives in exactly one table**, `ops::common::params::SLOTS`, read by *both*
   the claim predicate and the translate handler through a small `FloatAttrs` trait implemented
   for `NodeView` (borrowed from ORT, seen at claim time) and `NodeDesc` (owned, seen at translate
   time). If those two ever read different tables we would claim on one set of values and dispatch
   with another. Slot order is the contract with the GLSL, and getting it wrong is invisible to
   both compilers — hence the test that `HardSigmoid` fills `alpha` then `beta` regardless of the
   attribute map's own ordering.
3. **A default read on the host is not the same as a default baked into a shader.** The tail still
   resolves an omitted attribute to the ONNX default — but it does so once, per node, and pushes
   it explicitly, so the *same* compiled shader is correct for every value. That is what turns the
   guess into a handled value and the row from `Staged` into `Live`.

Which attributes this does **not** cover, and why each is a different kind of thing rather than a
missing feature:

| op | attribute | why the tail does not apply |
|---|---|---|
| `Gelu` | `approximate` ∈ {`none`,`tanh`} | a **string selecting an expression**, not a coefficient. Needs a second shader variant. Claimed at `none` (the form the shader implements), declines `tanh` with `[attribute]`. |
| `Mod` | `fmod` | integer selector between two different arithmetics. |
| `BitShift` | `direction` | string selector. |
| `IsInf` | `detect_negative`/`detect_positive` | integer selectors, and the op has a bool output — a different store path (§7.1.2). |
| `Clip` | `min`/`max` | **optional inputs, not attributes**, from opset 11. See §5.1.2. |

The general rule this leaves behind: *a float parameter rides the tail; a selector needs a
variant.* The distinction is whether the attribute changes a coefficient or changes the
expression. Only the first is a value.

#### 5.1.2 `Clip` — why the bounds are not parameters (2026-07-29)

`Clip` looks like the parameterised activations and is not one. Since opset 11 its bounds are
**optional inputs**, so a bound may be a graph initializer or a value computed at runtime, and we
can read neither at Compile time — `TensorRef` carries `is_initializer` but not the initializer's
*contents*. The push-constant route is therefore unavailable in principle, not merely unbuilt.

But it does not need it. Three-input `Clip` is an ordinary ternary elementwise op: the bounds are
rank-0 tensors that broadcast against the value with a stride of zero, which the shared indexing
helper already does for free. So `Clip` is claimed in its three-input form, through the same
`ew_select` template `Where` uses, differing only in which input the common dtype is taken from
(`Where`'s first input is `bool`; `Clip`'s three are all the value dtype — hence a separate
`ew_clip` translate rather than a reuse that would silently take the dtype from input 1).

The one- and two-input forms **decline `[arity]`**, and this is a real loss: a min-only `Clip`
(`Relu6`-style clamping) is common in conv graphs. The fix is a shader variant substituting ±∞ for
the omitted bound, because an omitted bound is a different *dispatch shape* — a buffer that does
not exist — not a different value. Widening the predicate to cover it would mean binding a
descriptor to nothing. Tracked for T5; `tests/ops/test_elementwise.py::test_clip_no_bounds` fails
loudly against this decline today, which is the correct state: the harness is reporting a form we
decline rather than passing vacuously.

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

#### Status 2026-07-28 — the templates exist, and what "exist" means

The three templates landed as `rust/shaders/glsl/templates/ew_{unary,binary,select}.comp` with the
shared header `rust/shaders/include/indexing.glsl` and the op-selector header `op_codes.glsl`.
**All 168 variants in `src/ops/shader_variants.txt` compile to SPIR-V**, validated locally with a
standalone Khronos `glslangValidator` (fetched outside the repo — this machine has no Vulkan SDK)
using the same flags `build.rs` passes. The design above survived contact with the compiler with
three changes worth recording:

1. **Templates live one directory down**, in `shaders/glsl/templates/`, because `build.rs` compiles
   every `*.comp` directly in `shaders/glsl/` with no `-D` defines — that is the path for
   hand-written XL kernels, which need none. A template compiled with no `EW_OP` and no `DTYPE_*`
   fails on purpose, so the two populations must not share a directory. (Cleaner alternative for
   Tank if he wants it: have the direct scan skip any file named by the variant table.)
2. **`bool` and `uint8` are byte-packed into `uint` words**, since `storageBuffer8BitAccess` is not
   in the baseline capability set. Loads shift and mask. Stores use `atomicAnd` + `atomicOr` on the
   shared word, which is race-free *because the bit lanes are disjoint* — each invocation only ever
   clears and sets the eight bits of its own element, and disjoint-lane bit operations commute in
   any interleaving. This imposes one requirement on the allocator: **a buffer holding a byte-typed
   tensor must be allocated rounded up to a multiple of 4 bytes**, because the last word is written
   whole. Stated here because it is invisible from the Rust side.
3. **f16 is stored as `float16_t` and computed in `float`.** That is an accuracy gain rather than a
   loss, and it removes any dependency on f16 overloads of the transcendental builtins.

Two semantics the caveat above predicted, both now handled: ONNX `Round` compiles to `roundEven`,
not `round`; and GLSL's `pow` is undefined for a negative base while ONNX's is defined for integral
exponents, so `Pow` carries an explicit sign path.

**What has *not* happened: none of these have executed.** The engine has no pipeline or dispatch
path yet, so no variant has run on any device, on any vendor, including lavapipe. Every elementwise
row is therefore staged behind the new `UNEXERCISED` reason — a deliberately different blocker from
`NO_SHADER`, because "the compiler accepted it" and "it computes the right answer" are different
claims and the gap between them is exactly where a coverage project lies to itself. Claiming these
rows today would also simply fail: with no dispatch path a claimed node cannot execute, so
declining is not merely conservative, it is the only correct answer available.

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

**Status 2026-07-28: the table half has landed; the `build.rs` half is a request to Switch.**
`rust/src/ops/common/variants.rs` now owns the naming rule and generates every stem at compile
time with `concat!`, because `engine::KernelRequest::shader` is a `&'static str` and a stem may
never be formatted at runtime. The gating above is automatic and needs no `f16_relevant` list: a
row's `caps` column *is* the variant set, so `Atanh` declaring `FLOAT` and `BitwiseAnd` declaring
`INT` produce exactly the variants each op can actually use, and nothing else. 69 tier-1 rows
currently imply **168 SPIR-V modules**.

`variants::manifest()` walks the registry and emits one line per module, checked in at
`rust/src/ops/shader_variants.txt`:

```
ew_binary_add_f16   ew_binary.comp   EW_OP=OP_ADD,SCALAR_T=float16_t,DTYPE_F16
ew_binary_add_f32   ew_binary.comp   EW_OP=OP_ADD,SCALAR_T=float,DTYPE_F32
ew_unary_sqrt_f32   ew_unary.comp    EW_OP=OP_SQRT,SCALAR_T=float,DTYPE_F32
```

Tab-separated and deliberately boring, so `build.rs` can parse it without depending on the crate.
A unit test regenerates it (`MOUSE_BLESS_VARIANTS=1`) and fails when it drifts, so the build's view
of what shaders exist and the registry's view of what it will dispatch cannot silently diverge.
**The ask to Switch:** have `build.rs` read this file and, for each line, compile
`shaders/glsl/<source>` with the listed `-D` defines to `<stem>.spv`. That is the only change the
shader pipeline needs to go from one-module-per-`.comp` to full variant expansion.

### 5.6 The compose-before-bespoke rule

**Rule: an op is implemented by composing existing templates unless it is on the fusion allowlist.
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

The allowlist is short on purpose: **Softmax, LayerNorm/RMSNorm(+skip), attention, quantized
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

Tiers T3–T5b were previously written as "if we get there". As of the 2026-07-28 ruling they are
funded work; see §6.0 for the per-kernel exit criteria and the honest two-number timeline.

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

### 6.0 The XL kernels are committed work, not stretch goals (user ruling, 2026-07-28)

Justin ruled directly: *"matmulnbits那些 都要做"* and *"contrib op 要做"*
(`.squad/decisions/inbox/copilot-directive-xl-kernels.md`,
`.squad/decisions/inbox/copilot-directive-contrib-ops.md`). The earlier revision of this document
listed the following as the kernels with **no template leverage** and parked them in the risk
register (§11). They are now scheduled deliverables with named exit criteria:

| Kernel | Tier | Owner | Exit criterion (binary, measurable) |
|---|---|---|---|
| `MatMulNBits` | T4 | Mouse (kernel), Switch (prepack seam) | GEMV (`M=1`) and GEMM (`M>1`) variants, `bits=4/8`, `block_size ∈ {16,32,64,128}`, symmetric and zero-point forms, match ORT CPU EP on the *same quantized graph* within §10.1 tolerance. No dequantized weight tensor is ever allocated in device memory (asserted by an allocator high-water test). |
| `DequantizeLinear` / `QuantizeLinear` (block-wise) | T4 | Mouse | Per-tensor, per-axis and blocked (opset 21) scale/zero-point modes; blocked path shares the `MatMulNBits` block-index math bit-for-bit (same helper, proven by a unit test that runs both against one input). |
| `GatherBlockQuantized` | T4 | Mouse | Quantized embedding lookup matches CPU EP; reuses `IDX` gather math + the unpack helper — no third copy of either. |
| `RotaryEmbedding` | T3 | Mouse | Interleaved and non-interleaved, partial-rotary (`rotary_embedding_dim < head_size`), `is_packed_batching`; matches CPU EP on a Qwen3 GenAI graph's actual attribute set. |
| `GroupQueryAttention` | T3 | Mouse | **DELIVERED 2026-07-30.** Decode path (`seq_len=1`), packed QKV, `do_rotary=1`, neox RoPE, online-softmax GQA kernel (`gqa_f16.comp`), in-place KV-cache via aliased output. Phi-3.5: Intel MATCH 618 ms, NVIDIA MATCH 230 ms, 353 claimed, 1 island. Prefill (`seq_len>1`) needs inter-invocation sync and is not yet claimed. Declines `local_window_size ≠ -1`, `softcap ≠ 0`, `smooth_softmax > 0`. |
| `LinearAttention` (`gated_delta`) + `CausalConvWithState` | T5a | Mouse | Qwen3.5 hybrid layers run on Vulkan with conv-state and recurrent-state cache I/O. **Schema is main-branch-only and unverified — see §9.4; the fingerprint must be re-verified against the shipping release before the kernel is trusted.** |
| `QMoE` (then `MoE`) | T5b | Mouse | Qwen3-MoE int4 expert block runs on Vulkan, not CPU. Masked-dense first (correct, wasteful); indirect dispatch only after it is proven correct and only if Switch adds the seam (§9.5). |

**The honest timeline, restated in the same breath as the commitment.** These seven kernels are
roughly 60–70% of the total *kernel-writing* effort in this document while being 8 of 174 rows. The
template machinery makes 87 tier-1 ops a table; it does nothing for these. They are each a
multi-day-to-multi-week kernel with a numerics debugging tail, and they gate the only metric that
matters. So:

- **Tier-1 breadth (87 ops) is a days-to-two-weeks item.** That claim stands.
- **"Qwen3.5 runs end-to-end on Vulkan" is a months-scale item**, and would be even with a perfect
  op plan, because it is gated on M2's device allocator, f16 storage+arithmetic, a shape-bucketing
  decision, and six bespoke kernels — four of which have no reference semantics we can diff against
  other than ORT's own CPU implementation.
- The gap between those two sentences is the thing to watch, and `largest_island_flops` (§7.3) is
  the metric that makes it visible instead of letting op count paper over it.

Nothing here asks for the commitment to be reconsidered. It asks that the schedule be reported as
two numbers rather than one.

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

#### 7.1.1 A row's `caps` is not evidence that its variants ran (2026-07-29)

The first row went live this day and it exposed a gap in the rule as stated. `caps` is one column
serving two consumers: the shared claim helpers check input dtypes against it, and the shader
variant table generates exactly those variants. That is the leverage §5.7 claims, and it is right —
but it means a row that declares `NUMERIC` and flips to `Live` claims **four** dtypes on the
strength of however many actually executed.

`Add` executed as `add_f32`, on two devices, through the ORT wire. `add_f16`, `add_i32` and
`add_i64` compile and are shape-checked and have never run. Claiming them because they share a row
is the same argument as claiming `Sub` because it shares a template — an argument this document
exists to refuse.

So the live-set discipline has two parts, and `ops/elementwise.rs` implements both:

1. **`EXERCISED`, an `(op, dtype)` evidence list** naming the test and the devices. A test asserts
   the live set and the evidence list are the same set, so going live is a two-place edit where the
   second place demands a sentence claiming evidence.
2. **A predicate narrowed to the exercised dtypes**, not to `caps`. `ew_binary_f32` is
   `ew_binary` plus an f32 check, declining `[dtype]` for the rest. `caps` stays `NUMERIC` because
   we still want those variants *compiled* — the point is to stop claiming them, not to stop
   building them.

**Be precise about what claiming `Add` now bets on.** Not the shader; that has executed under
validation layers on an Intel Iris Xe and an NVIDIA RTX 4060 with zero errors. The bet is that the
**wire** is correct — `Compile` → `OrtNodeComputeInfo` → `dispatch_ort` — and the wire has so far
only carried a mock host's tensors. Flipping the row is what makes that bet *settleable*: an
unexercised path that nothing can execute never gets proven. Expect the first differential run to
fail; if it does, the failure is the deliverable and `EXERCISED` shrinks back.

`Sub`, `Mul`, `Div` and `Pow` share the exact shader family and stay staged. M0 asks for one node
to travel the wire and the differential assertion is per-op. Four rows flipped on one executed
shader would mean three of them rode on "it's the same template".

> **Superseded in part, same day, §7.1.2.** The wire was then proven end-to-end through real ORT on
> two vendors, which is the fact the paragraph above was waiting for. The paragraph's *reasoning*
> stands and is the reason §7.1.2 is a separate, weaker list rather than a widening of `EXERCISED`.

#### 7.1.2 Template evidence — a second, weaker list, and its exact boundary (2026-07-29)

Once `Add` was claimed and executing on both local devices through real ORT, the sentence "its
shader compiles but has never executed" stopped being the binding constraint on the rest of the
family. What had been unproven was the **wire**; the wire is now proven. What remains unproven per
op is one line of GLSL.

That is a genuinely different and much smaller bet, so it gets a genuinely different and much
smaller list: `TEMPLATE_LIVE`, alongside `EXERCISED` rather than merged into it. Each entry names
the op and the exercised op standing in for it.

**A row may join `TEMPLATE_LIVE` only if all three hold:**

1. its representative is in `EXERCISED` **and** currently `Live`;
2. it reaches the device through that representative's *exact* `translate` handler, template,
   descriptor layout and push-constant block; and
3. its claim predicate is narrowed to the representative's dtype.

All three are asserted by unit tests, including the one that matters most in a year: if `Add` is
ever demoted — which flipping it exists to make possible — every row standing on it fails the build
rather than quietly continuing to claim on withdrawn evidence.

**What this does not buy.** Anything that is a different *code path* rather than a different
expression:

| stays staged | why |
|---|---|
| `Equal`, `Greater`, `Less`, … | bool output from float input — a different store path in the template |
| `And`, `Or`, `Bitwise*`, `Not` | not f32 at all, so the narrowing that justifies the live rows says nothing about them |
| `Sum`, `Mean`, `Max`, `Min` | `ew_variadic` issues several dispatches; the wire has carried one |
| `Where` | third template (`ew_select`), never dispatched |
| `PRelu` | binary with a broadcast form the arithmetic ops do not exercise |
| any live op at f16/i32/i64 | the variant compiles, `caps` still generates it, and it has not run |
| `Mod`, `BitShift`, `IsInf` | selector attributes, not float parameters — `NEEDS_PARAMS`, unchanged (§5.1.1) |

**Why this is worth doing rather than waiting for 34 individual dispatch tests.** While a row is
`Staged`, its differential test does not compare anything — it fails with *"the EP executed no node;
the CPU-match check would be a vacuous pass"*. That failure is loud and correct and proves nothing
about the shader. Flipping the row is what converts it into an actual comparison against the CPU EP.
The evidence is therefore produced *by* the flip, on the next run, and if a body is wrong the suite
says so then. Waiting would mean holding 34 shaders unverifiable in order to avoid claiming them.

**And the reason to take the family together rather than one row at a time** is §7.2, not the op
count. These ops cluster: `Mul` is 64 nodes of Phi-3.5, `Add` 72 of gpt-oss, and they sit next to
each other. Claiming a scattered op raises coverage and shreds the graph — the gpt-oss `Cast`
result in §4.21 is 28 % → 54 % coverage against 52 → 125 islands. Coverage is not the metric;
islands are.

**Outcome, same day.** The flip was made and the suite run on both devices:

| run | before the flip | after |
|---|---|---|
| `test_op_table.py`, device 0 (Iris Xe) | 1 passed / 76 failed | **39 passed** / 38 failed |
| `test_op_table.py`, device 1 (RTX 4060) | 1 passed / 76 failed | **39 passed** / 38 failed |
| `test_barrier_parity.py`, both devices | 0 passed / 74 skipped | **36 passed** / 38 skipped |

Not one numerical mismatch against the CPU EP on either vendor, and the remaining 38 failures are
all the vacuous-pass guard firing on rows that are still `Staged`. The coordinator's expectation
was that the first real run would fail; it did not, and that is worth recording as plainly as a
failure would have been.

#### 7.1.3 The parameterised activations went into `EXERCISED`, not `TEMPLATE_LIVE` (2026-07-29)

The fourteen ops §5.1.1 unblocked look like the exact case `TEMPLATE_LIVE` was built for: same
template, same translate, same descriptor layout, same push-constant block, f32-narrowed
predicate, and a representative (`Relu`, `HardSwish`) already in `EXERCISED`. Condition (b) is
satisfied to the letter.

They were still put in `EXERCISED`, on their own dispatch evidence, because the letter is not the
point. `TEMPLATE_LIVE`'s argument is that *the only difference is one line of arithmetic inside a
body the pipeline generates from one source*. The parameter tail is not that: it is a **new code
path**. A wrong offset for `params[0]` — the arity-dependence of the tail's position makes that a
live possibility — would be invisible to every op already live, because all of them push zeros
there and read none of them. `Relu` passing says nothing whatsoever about whether `LeakyRelu`
reads the float the host wrote.

So the rule stands as written, with the boundary sharpened: **`TEMPLATE_LIVE` covers a different
expression in an exercised path, never a different path.** When in doubt, ask what a plausible bug
in the new code would do to the representative — if the answer is "nothing", the representative is
not evidence.

They were flipped, run, and promoted in the same turn:

| run | before | after |
|---|---|---|
| `test_elementwise.py`, device 0 (Iris Xe) | 25 passed / 11 failed | **33 passed** / 3 failed |
| `test_elementwise.py`, device 1 (RTX 4060) | 25 passed / 11 failed | **33 passed** / 3 failed |
| `test_barrier_parity.py`, both devices | 36 passed / 38 skipped | **46 passed** / 28 skipped |

The three remaining `test_elementwise` failures are `Min`, `Max` (variadic — several dispatches,
still staged) and `test_clip_no_bounds` (§5.1.2, deliberately declined). `test_op_table.py` is
unchanged at 28 failures, all of them the vacuous-pass guard on staged families. Crucially, the
suite covers **non-default** attribute values — `LeakyRelu(alpha=0.1)`, `Elu(alpha=1.5)`,
`HardSigmoid(alpha=0.15, beta=0.4)` — so what passed is the mechanism, not the defaults that were
already baked into the shader.


`TEMPLATE_LIVE` is consequently **empty again**: all 34 entries were promoted into `EXERCISED`,
each naming `test_op_table[<Op>-fp32]` and the two devices. The list stays defined because that
two-step — flip on template evidence, promote on dispatch evidence — is the mechanism the next
family will use, not an accident of this one. An entry still sitting in `TEMPLATE_LIVE` after a
differential run is itself a finding: it means the run is not covering that row.

**Barrier parity was the other beneficiary.** It is M0 criterion 8 and had never executed a case,
because it only runs on ops the EP claims. 36 cases now run on both the legacy and the sync2
barrier backend, so the legacy path we carry for ~31 % of Android is finally exercised. That was
not the goal of the flip; it is a consequence of §7.2's point that claims and coverage are
different things from *tested* claims.

#### 7.1.4 Optional-input population is a coverage axis — not a simpler expression of the same form (2026-07-30)

This section exists because of a silent defect that was found only after Phi-3.5 ran end-to-end
with all-zero logits, `compute_failures: 0`, and no validation error on either vendor.

**The defect.** `MatMulNBits` has an optional `zero_points` input (slot 3). Switch's translate
handler allocated **4 binding tokens** for the 3-input form (without `zero_points`). The shader
writes its output to **binding 4** — the fifth slot — which was undeclared in the 4-entry pipeline
layout. The GPU silently ignored the write. The unit tests in `test_matmulnbits.py` all exercised
the form *with* `zero_points`; Phi-3.5 has none. A test of the 4-input form is not evidence for
the 3-input form, because the two forms have different pipeline layout entry counts.

**The structural point.** §7.1.2 condition (b) says:

> a row may join `TEMPLATE_LIVE` only if it reaches the device through that representative's *exact*
> `translate` handler, template, descriptor layout and push-constant block.

An op with an optional input absent takes a *different code path in the translate handler* — it
produces a different binding count, a different pipeline layout, and often a different push-constant
block. Morpheus's proof-ledger key (`§8.9`) captures this exactly: the final dimension of the key
is `populated_optional_input_set`. A proof keyed on `{0,1,2,3}` cannot be returned for a query
keyed on `{0,1,2}`. The silence set was always there; nobody was reading it.

**The rule this adds.** §7.1.2 condition (b) now carries an explicit corollary:

> **An op with an optional input absent is a different form, not a simpler expression of the same
> form.** It must be separately exercised. A proof of `(op, dtype, shape_class, {full inputs})` is
> not a proof of `(op, dtype, shape_class, {inputs minus optional slot k})`.

This is true even when the absent input triggers a code path that re-uses the slot (as `MatMulNBits`
does, binding `scales` twice as an inert `zero_points` placeholder). That placeholder is invisible
to the test that never exercises the no-`zero_points` form.

**What would have caught this before the logit failure.** A probe that runs the session more than
once (Tank's multi-run discriminator is what made the KV-cache signal observable). Any probe that
checks `binding_count(pipeline_layout) == binding_count(translate output)`. The proof ledger
(§8.9), by construction: `populated_optional_input_set` in the key means the 3-input and 4-input
forms are different keys and can never satisfy each other's proof obligation.

**Island measurement note, 2026-07-30.** After `SkipSimplifiedLayerNormalization` was promoted from
`Staged` to `Ready`, the bench run showed `claimed 321 of 363, islands 321` — up from `257` on the
prior run. The island count increased, not decreased. This is the falsifier the coordinator's
prediction specified: "if `subgraphs_live` drops by less than 128, some SkipNorm nodes are not
between two claimed nodes." The result was that *none* of the newly claimed SkipNorm nodes merged
neighbouring islands — each became its own island. The coordinator's hypothesis about SkipNorm
sitting between two MatMulNBits islands was wrong on the Phi-3.5 graph as ORT partitions it.
The correct reading: **claiming a new op type adds islands before it removes them**, unless the
newly claimed op is the sole unclaimed gap between two existing islands. Op priority order should
be chosen by what *removes* gaps, not by what adds the most nodes. `declined_nodes` histogram
(§7.3) is the right instrument for this — each declined op's island-removal potential is computable
before the op is implemented.

#### 7.1.5 Island-count == claimed-count is the partition-wiring falsifier (2026-07-30)

A second silent defect was found on the same day as the binding-arity bug. After `GetCapability`
was wired to offer maximal convex connected subgraphs rather than one capability per node,
`bench/phi35.py` reported `claimed 321 of 363, islands 321` — the island count equalled the
claimed-node count exactly. `partition.rs` was correct; `GetCapability` was producing one
fused node per claimed op. ORT fused one node per capability entry, so 321 capabilities became
321 subgraphs, and `compute_calls 1 != expected 1023` (the dispatch-accounting falsifier) went
red immediately on the first run — then the partition wiring was fixed.

**The falsifier.** `island_count == claimed_node_count` means the partition pass did not run or
produced no merges. This equality is not noise: ORT always fuses exactly the nodes in one
capability entry into one subgraph, so a 1:1 ratio is a precise symptom of the same defect that
produced 321 islands before the partition wiring was fixed. The bench now asserts:

```
dispatch_accounting: ok — compute_calls {N_INFERENCES} × {islands} == compute_calls_actual
```

Any future regression in partitioning will make this red by construction, without requiring a
human to notice that wall time went up.

**The general lesson.** A mechanism that exists in a file (`partition.rs`) and is not in a call
graph (`GetCapability`) is indistinguishable from a mechanism that does not exist. Per R9: name the
instrument that goes red if "partitioning is working" is false. Before this fix there was none —
`island_count` was *reported*, and nobody compared it against `claimed_node_count`. That comparison
is one line and it would have caught this on day one.

**Multi-node island dispatch: a new axis in the intermediate-buffer space.** A subgraph with
multiple nodes requires intermediate GPU buffers — one per inter-node edge — with stable token
assignment across all kernels in the island. The prior token scheme (positional, reset per kernel)
aliased intermediate output tokens onto external output tokens, so the write of an intermediate
value silently clobbered or missed the external output slot. The fix uses a name-based
token map built from the island's `plan.inputs`/`plan.outputs` and inner node outputs at Compile
time, making the token ranges non-overlapping by construction:

```
0..n_plan_inputs                                    → external ORT inputs
n_plan_inputs..n_plan_inputs+n_plan_outputs         → external ORT outputs
n_plan_inputs+n_plan_outputs..first_temp_token      → intermediate buffers
first_temp_token..                                  → alloc_temp scratch
```

This is a new coverage axis: **a multi-node island has different token routing than a single-node
island**. A test that runs only single-node subgraphs cannot catch an aliasing bug that only appears
when two kernels share an intermediate. The `dispatch_accounting` check (`compute_calls == islands ×
inferences`) is the falsifier: if any multi-node island fails to dispatch, `compute_calls` drops
below the expected value.

#### 7.1.6 GroupQueryAttention is Live — island attribution methodology and results (2026-07-30)

**Attribution method.** After SkipSimplifiedLayerNormalization was promoted to `Live` (§7.1.4), a
union-find over the full 366-node Phi-3.5 graph showed 2 connected components, while ORT reported
33 islands. The gap between 2 and 33 came from topological "runs" of consecutive claimed nodes being
broken by gaps of declined nodes. The question is which declined op types create gaps vs which sit
at the graph's edges and create none.

The attribution script (`tests/ops/test_island_attribution.py`) works as follows:

1. Enable `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` during a real EP session; capture one JSONL record per
   node (`op`, `claimed`, `code`).
2. Run a union-find over *claimed* nodes only — two claimed nodes are merged if they share a
   topological edge (i.e., one feeds the other and no unclaimed node lies between them in the
   dependency chain). The number of resulting components is the minimum island count achievable
   given the current claim set — it is what ORT would see if its partitioner were omniscient.
3. Test each declined node type: for each node `d` of type `T`, temporarily add it to the claimed
   set and recount components. The *reduction* in component count is the number of ORT islands that
   claiming `T` would merge.
4. The sum of reductions for a type is its **island-attribution score** — the measure of its
   gap-creating contribution.

**Decline histogram — Phi-3.5-mini int4, device 0 and device 1 (identical):**

| op | declined | code | attribution (islands removed if claimed) |
|---|---|---|---|
| `GroupQueryAttention` | 32 | staged | **32 of 33 cuts → 1 island** |
| `Gather` | 2 | not-registered | 0 |
| `Cast` | 2 | staged | 0 |
| `SkipSimplifiedLayerNormalization` | 1 | staged | 0 |
| `Shape` | 1 | not-registered | 0 |
| `ReduceSum` | 1 | not-registered | 0 |
| `Sub` | 1 | dtype | 0 |
| `Greater` | 1 | staged | 0 |
| `If` | 1 | not-registered | 0 |

GQA is the *only* cut-creator in Phi-3.5. Every other declined node is at a graph edge or sits in an
already-disconnected component; claiming it would add nodes to an existing island but not merge any
two islands. A decline can appear 32 times and create 32 cuts (GQA), or appear 2 times and create
zero cuts (Gather). **Frequency is not attribution.**

**Prediction (stated before implementation):**
- Islands: 33 → ~1 (GQA is the sole cut creator; attributing all 32 cuts to it predicts one island)
- Timing (NVIDIA RTX 4060, device 0): 1156 ms → 400–900 ms
- Timing (Intel Iris Xe, device 1): 807 ms → 300–600 ms
- Falsifier: `island_count > 3` after GQA → something else cuts

**Post-GQA results (`bench/phi35.py`, 2026-07-30) — corrected device labels (see below):**

| device | claimed | islands | vulkan median | cpu median | verdict |
|---|---|---|---|---|---|
| 0 — NVIDIA RTX 4060 | 353 / 363 | **1** | 618 ms (16% RSD — noisy, taken under load) | 345 ms | **MATCH** |
| 1 — Intel Iris Xe | 353 / 363 | **1** | **230 ms** (2.7% RSD, taken under load) | 254 ms | **MATCH** |

**Device-label correction (2026-07-30):** `ONNXRUNTIME_EP_VULKAN_DEVICE` indexes the
capability-gated sorted list (discrete first), not the raw Vulkan enumeration index.
`Device 0: Intel / Device 1: NVIDIA` in probe output is enumeration order;
selector 0 → NVIDIA RTX 4060, selector 1 → Intel Iris Xe. Every prior label in this document
that said "device 0 = Intel" was backwards. The **measurements are correct; only the names
were wrong** (coordinator correction, 2026-07-30T22:00).

Prediction: islands = 1 ✓. Timing comparisons are **not reliable** — measurements were taken under
load (six agents building simultaneously; coordinator confirmed 9.5× inflation under contention, so
these numbers are absolute floor estimates only). Quiet-machine numbers are owed.

**Upload dominates (Tank's finding, 2026-07-30):** The EP re-uploads the entire ~2 GB weight set
on every inference call. `Phase::Record` wraps the staging memcpy, so the 68% attributed to
"recording" in earlier traces was actually weight upload — CPU work, not GPU command recording.
Actual GPU command recording is ~1–3% of wall. `alloc_device_authoritative_spans = 0` is the
counter that must move (Switch/Tank: persistent weight residency). This makes
`retain_viable`'s `transfer_ns` weighting *more* correct than initially understood — the boundary
cost is upload-dominated, not PCIe-latency-dominated.

The GQA kernel is serial (1 thread per head, online softmax, decode path `seq_len=1`). The island
consolidation is still the right action — it eliminated 32 boundary round-trips and removed
32 inter-island upload cycles from the 33-island path.

**Remaining declines (10 nodes, 0 additional islands):** Gather × 2, Shape × 1, ReduceSum × 1,
If × 1 (not-registered / control-flow); Cast × 2, SkipSimplifiedLayerNormalization × 1 (staged,
non-GQA variant); Sub × 1 (dtype — fp32 residual); Greater × 1 (staged). None of these create
cuts on this graph.

**⚠️ Runtime status as of 2026-07-31 (main `86a815f`):** The numbers above describe the EP's
*offer* at `GetCapability` time — 353 nodes claimed, 1 island. **At Compute time the EP fails and
ORT falls back entirely to CPU. Zero nodes execute on Vulkan.** Root cause: Phi-3.5's KV-cache
inputs have shape `[1, 32, 0, 96]` at first-token prefill (empty KV sequence). `vk/alloc.rs:214`
returns `None` for `size=0` because `vkCreateBuffer(size=0)` is forbidden by the Vulkan spec;
this propagates to `session.rs:1045` as `bail!("alloc_device failed for input buffer")`, which
ORT silently swallows and retries on CPU. **All timings cited in this section (Vulkan median,
CPU median, 618 ms, 230 ms, 254 ms) are CPU measurements of a CPU fallback run.** The island
count and claim data are correct; the timings are not comparisons of the EP against CPU.

*Fix ownership:* Switch owns `vk/alloc.rs` and `vk/session.rs`. The fix is a zero-size sentinel
path in `alloc.rs::alloc()` (return a 1-byte placeholder buffer; skip descriptor writes for
zero-size inputs in `session.rs`). `retain_viable`/`should_claim_island` is **not** the fix —
the gate runs at capability time against static/symbolic shapes; the zero dimension is dynamic and
is not visible at that stage. See `.squad/decisions/inbox/copilot-zero-size-alloc-partition-analysis.md`.

*Post-fix action for Mouse:* re-run `bench/phi35.py` with ORT profiling to confirm non-zero
Vulkan node counts, then report the first valid EP-vs-CPU timing.

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

**Status 2026-07-28: the metric definitions are now code, not prose.** `rust/src/ops/partition.rs`
implements every row of the table above as `Island`, `CoverageReport` and `TransferModel`, so Niobe
can adopt the definitions rather than re-derive them:

* `CoverageReport::{island_count, largest_island_nodes, largest_island_flops, node_coverage,
  concentration, boundary_bytes_per_inference, boundary_time_fraction}`.
* `TransferModel { fixed_ns, bytes_per_ns }` with `UMA` and `DISCRETE` starting points and
  **`TransferModel::fit(&[(bytes, ns)])`** — a least-squares calibration hook. That is the
  measurement handoff: Niobe times a staircase of transfer sizes on the target device, hands the
  samples to `fit`, and the MVS rule stops being a guess. The provisional constants are explicitly
  labelled as placeholders in the source.
* `Policy { min_nodes: 4, margin: 3.0, flops_per_ns }` and `evaluate(island, model, policy) ->
  Verdict`, plus `retain_viable` for the whole set. A rejection renders through `decline_for` as a
  `[partition]` reason, so partitioning rejections land in the *same* histogram as dtype and rank
  rejections instead of disappearing into a log line.
* `concentration()` — claimed FLOPs in the largest island ÷ total claimed FLOPs — is the single
  number that separates "80% of nodes across 40 islands" from "80% of nodes in one island". The
  unit tests assert those two cases have identical `node_coverage` and wildly different honest
  metrics, which is the whole argument in executable form.

The margin is 3×, not 1×. A cost model this crude, calibrated on a different device, under a driver
that schedules differently, is easily wrong by 2×; requiring a margin means the rule fails towards
"run it on the CPU", which is always correct.

#### 7.3.1 `retain_viable` is now wired — R10 resolved (2026-07-30)

**Status 2026-07-30:** R10 (Morpheus) identified that `retain_viable` existed in `partition.rs` but
was not in the call graph for single-cluster models. Wiring `partition.rs` into `GetCapability`
(the prior session's 321→33 island fix) wired the connected-component grouping. It did not wire
the economics gate — `should_claim_island` had never declined an island in production.

**What was wired:** The partition economics gate (`partition::evaluate`) now runs for every cluster
in multi-cluster graphs. A cluster that fails the gate produces a `[partition]` decline code in
the CLAIM_LOG, observable by tests and tooling.

**Artifact proving it fires:** `tests/ops/test_partition_gate.py` builds a model with two
independent Sigmoid branches (two disjoint claimed clusters, each 1 node, no anchors). The gate
fires: TooSmall (1 < min_nodes=4, anchors=0) → both clusters declined → `[partition]` codes in
CLAIM_LOG. This is the artifact R10 requires — content that varies with the gate's input.

**C ABI counter (ABI version 2):** `viable_islands_retained` added to `VulkanEpCounters` (the
struct `OrtEpVulkanGetExecutionCounters` fills). Emitted per multi-cluster `GetCapability` call;
present even at 0, so the wiring census can distinguish "gate ran, all rejected" from "UNWIRED
(key absent)". The census (`test_wiring_census.py`) now reads this counter and marks `retain_viable`
WIRED. The test `test_retain_viable_wired` has its `xfail(strict=True)` removed — it passes.

> **Amendment 2026-07-31 — read §7.10 before quoting this section.** "WIRED" here means the gate
> is in the production call graph, and that is all it means. Phi-3.5 partitions into exactly one
> cluster, takes the single-cluster bypass, and therefore **never runs the economics gate**;
> `viable_islands_retained == 0` on our only real model means *bypassed*, not *all-rejected*. The
> gate's only exercise is a synthetic two-branch test, which is one step from unwired (R10).

**PartitionStats populated:** `ep.rs` now emits real values for `island_count`,
`largest_island_nodes`, `largest_island_flops`, `concentration`, and `boundary_bytes_per_inference`
from the surviving island set after the economics gate runs. The `boundary_time_fraction` slot
remains 0.0 until Niobe wires VkQueryPool timestamps for calibration.

**Guards against both failure modes (§7.0.2):**
- *Over-declination* (gate declines everything): the anchor exemption —
  `if island.anchors > 0 { return Verdict::Claim }` — ensures any island containing a node that
  carries a resident weight at a schema-designated site is always claimed. On Phi-3.5 the island is
  dense with `MatMulNBits` nodes holding resident packed weights, so the gate never declines.
  Falsifier: bench/phi35.py → 0 claimed nodes would indicate a broken anchor exemption.

  > **Amendment 2026-08-08T17:01:06-07:00 — Niobe, issue #73.** This bullet used to read "any island
  > containing MatMulNBits or GQA is always claimed. On Phi-3.5 (353 claimed, 1 island, 225
  > anchors)". Both halves were readings of the **pre-#73 name-only** `is_anchor` and neither is
  > re-asserted. `GroupQueryAttention` designates **no** weight site — its operands are activations,
  > KV cache, position-indexed RoPE tables, cache scales and elementwise learned vectors — so GQA
  > nodes contribute zero anchors under the shipped predicate, and the anchor total for Phi-3.5 is
  > **not restated here** because no post-repair census has been run on that model. The claimed-node
  > count and island count are separate observations and are not affected by the anchor repair. See
  > `DESIGN.md` §5.4.2 for the designated-site table and its provenance.
- *Under-declination* (gate declines nothing): `test_partition_gate.py`. A non-anchor two-cluster
  model must produce `[partition]` codes. If it does not, the gate is inert. Falsifier: the test
  asserting `claims["Sigmoid"]["code"] == "partition"` goes red.

**Single-cluster exemption retained:** when all claimed nodes form one connected component (the
common case for unit tests of individual ops), the gate is bypassed. The gate applies to
multi-cluster graphs, where it has a real scheduling decision to make.

**TransferModel calibration — post-residency update (Switch 2026-07-30):** Persistent weight
residency is landed. Per-inference boundary bytes: **~0.756 MiB** on Phi-3.5 (Switch's byte sweep:
inference 0 uploads 1997.596 MiB, subsequent inferences add only 0.756 MiB each — activation-only).
The provisional `TransferModel` constants (`DISCRETE: 12 bytes/ns`, `UMA: 40 bytes/ns`) now model
the correct regime.

**Gate behavior post-residency — stated before verifying:** the gate will decline far less. With
`cost_ns(0.378 MiB) ≈ 93,000 ns` per direction on `DISCRETE`, the total transfer cost for Phi-3.5's
one island is ~186 µs, and the 3× margin threshold is ~558 µs. Any island whose GPU compute exceeds
558 µs (GQA: ~4 ms, MatMulNBits: ~1 ms) passes easily. Only sub-millisecond non-anchor islands
should be declined now.

**Correction on my earlier estimate:** I estimated "~16–128 bytes per boundary" post-residency. The
measurement says 0.756 MiB — three orders of magnitude larger — because the KV cache (32 layers,
each 2 × [B, Hkv, seq, D]) and model activations cross the boundary even without weights.
**An estimate without a measurement is not a calibration** (R6 amendment 4: a result surprising
enough to be a discovery is first a reason to check the instrument).

**What still needs Niobe:** `TransferModel::fit(samples)` awaits a timed staircase calibration.
The current constants model nominal PCIe bandwidth (12 GB/s) from published specs. For the
activation-only regime, the fixed overhead (60 µs / 20 µs) dominates over the variable term;
Niobe's measurements will tell us whether the fixed cost is accurate or needs adjustment.



Every previous version of this document reasoned about *which ops a model contains*. This section
reasons about *why the EP declines the nodes it declines*, which turns out to be a different
question with a different answer. It is §8.5's lesson landing for the fourth time, and the fourth
landing is itself the finding: we keep planning against a property of the model that is easy to
read off, rather than the property that actually gates us.

#### 7.4.1 Method

Two passes, in separate processes because loading the EP and running onnx's C++ shape inference in
one process faults on a 2.2 GB external-data model:

1. Real session creation over the cached Foundry model with `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` set —
   one JSONL record per node carrying `op`, `node`, `opset`, `claimed`, `code`, `reason`.
2. `onnx.shape_inference.infer_shapes_path` over the same file, giving every edge's shape with
   symbolic dimensions preserved as `dim_param` names.

Joined on node name. Scripts were scratch; the numbers below are reproducible from the claim log
plus stock `onnx`, and nothing here is simulated.

#### 7.4.2 The decline histogram is first-match, and cannot be read as a partition of causes

A node records **one** reason — the first predicate that rejected it — but may have several
disqualifying properties. So `staged: 100` does not mean "100 nodes that only need a kernel". It
means "100 nodes that were rejected by the staging check *before* anything else was examined".
Reading the histogram as a partition is how "kernels are what stand between us and a model" became
a working assumption. Cross-tabulating the code against an independently computed shape class
dissolves it:

**Phi-3.5-mini-instruct, int4 RTN block-32 — 363 records over 366 nodes**

| code | STATIC | EXTENT-ONLY | STRUCTURAL | total |
|---|---:|---:|---:|---:|
| `dynamic-shape` | 0 | 258 | 0 | 258 |
| `staged` | 2 | 66 | 32 | 100 |
| `not-registered` | 2 | 0 | 3 | 5 |
| **total** | **4** | **324** | **35** | **363** |

**gpt-oss-20b — 371 records over 374 nodes**

| code | STATIC | EXTENT-ONLY | STRUCTURAL | total |
|---|---:|---:|---:|---:|
| `dynamic-shape` | 0 | 146 | 0 | 146 |
| `staged` | 1 | 148 | 48 | 197 |
| `not-registered` | 1 | 24 | 3 | 28 |
| **total** | **2** | **318** | **51** | **371** |

**Four nodes in Phi-3.5 and two in gpt-oss have fully static shapes.** Of the 100 `staged` Phi-3.5
nodes, 2 are static; of gpt-oss's 197, 1 is. So landing `SkipSimplifiedLayerNormalization`,
`GroupQueryAttention`, `Cast` and `QMoE` tomorrow, while `REQUIRE_STATIC_SHAPES` stands, moves the
claimed count from 0 to **2** on Phi-3.5 and **1** on gpt-oss.

The ratio "dynamic shapes 258, kernels 100" is therefore not a comparison between two comparable
quantities, and the honest statement is stronger than the ratio suggests: **the shape gate sits
upstream of the kernel gate for 99% of the nodes in both graphs.** Dynamic-shape support is not
higher-priority than the three kernels; it is the precondition that decides whether those kernels
are worth anything at all on a real model. Note also that the two models' histograms look
*opposite* — Phi-3.5 is shape-dominated (258 vs 100), gpt-oss is kernel-dominated (146 vs 197) —
purely because gpt-oss has 100 `Cast` nodes that get tested for staging first. Two models, opposite
apparent conclusions, identical underlying cause. That is what a first-match histogram does to you.

#### 7.4.3 Which dimension is symbolic — the answer is narrower than "dynamic shapes"

Every `dynamic-shape` decline in both models, without exception:

| model | op | n | symbolic dims |
|---|---|---:|---|
| Phi-3.5 | `com.microsoft::MatMulNBits` | 161 | `A` axis 0 `batch_size`, axis 1 `sequence_length`; `A` axis 2 = 3072 or 8192 literal; `B`/`scales` fully static |
| Phi-3.5 | `Mul` | 64 | both inputs `[batch_size, sequence_length, 8192]` — identical dim_params |
| Phi-3.5 | `Sigmoid` | 32 | `[batch_size, sequence_length, 8192]` |
| Phi-3.5 | `Sub` | 1 | `[batch_size, 1]` against a scalar |
| gpt-oss | `com.microsoft::MatMulNBits` | 73 | `A` `[batch_size, sequence_length, 2880]` |
| gpt-oss | `Add` | 72 | `[batch_size, sequence_length, 5120]` + `[5120]` |
| gpt-oss | `Sub` | 1 | `[batch_size, 1]` against a scalar |

The whole graph uses five dim_params — `batch_size`, `sequence_length`, `total_sequence_length`,
`past_sequence_length`, `max_sequence_length` — and **the last axis of every declined tensor is a
literal**. The symbolic dimensions are exclusively leading dimensions. They determine *how many
rows there are*, never the contraction length, never the channel count, never the block count,
never the rank. That is a much smaller problem than "dynamic shapes".

I classify each node as:

* **STATIC** — no symbolic dim anywhere.
* **EXTENT-ONLY** — every symbolic dim is a leading axis, the last axis of every input is a
  literal, and all symbolic-carrying inputs agree dim_param-for-dim_param on their leading pattern.
  Under those conditions "rows" is one runtime number and broadcasting is decidable *symbolically*,
  because two dims carrying the same `dim_param` are equal by construction.
* **STRUCTURAL** — anything else: symbolic in a contraction/channel axis, or shapes whose broadcast
  relationship cannot be settled without the values.

324 of Phi-3.5's 359 non-static nodes and 318 of gpt-oss's 369 are EXTENT-ONLY. The STRUCTURAL
remainder is `GroupQueryAttention` (32 / 24), `QMoE` (24), and the `Shape`/`ReduceSum`/`Gather`
prologue (3 each) — all of which are staged or unregistered anyway, so structural dynamic shapes
are not on anyone's critical path yet.

#### 7.4.4 The three options, costed — and (a) already works

**(a) Shapes known only at `Compile` time.** `Compile` runs once at session creation, where the
fused node's dims are still symbolic — *unless* the caller pins them. ORT exposes exactly that:
`SessionOptions::AddFreeDimensionOverrideByName`. Measured on the cached model, device 1:

| pinned | claimed | islands | remaining declines |
|---|---:|---:|---|
| nothing | 0 | 0 | `dynamic-shape` 258, `staged` 100, `not-registered` 5 |
| `batch_size=1` | 0 | 0 | `dynamic-shape` 257, `staged` 100, `dtype` 1, `not-registered` 5 |
| `sequence_length=1` | 0 | 0 | unchanged from nothing — it is the *conjunction* that matters |
| **`batch_size=1, sequence_length=1`** | **161** | **161** | `staged` 100, `dtype` 97, `not-registered` 5 |

And with both pinned the model **runs**:

```
Phi-3.5, batch=1 seq=1, decode:   161 MatMulNBits nodes on VulkanExecutionProvider,
                                  298 node instances on CPU, 65 outputs
  device 0 (Intel Iris Xe)  : argmax 30751, top-5 match, max|Δ| 0.0078 vs CPU EP
  device 1 (NVIDIA RTX 4060): argmax 30751, top-5 match, max|Δ| 0.0078 vs CPU EP
Phi-3.5, batch=1 seq=16, prefill: same 161 claimed, argmax 30751, max|Δ| 0.0488, both devices
```

Logits are fp16 with magnitude ~13, so 0.0078 is one ULP at that scale; the prefill figure is
larger because the GEMV path accumulates 16 rows. **This is the first real production model to run
on this EP with nodes executing on the GPU, and it took no predicate change of any kind.** It also
independently confirms §8.1.2 on real hardware: `MatMulNBits` claimed *alone* produces 161 islands
of one node each — the partition is shredded exactly as the simulation said, which is why the
number to chase is the pair with `SkipSimplifiedLayerNormalization`, not this one.

The cost of (a) is real and must not be glossed: a session pinned at `sequence_length=1` is valid
for exactly that token count. A decoder needs one pinned session per shape bucket — one for decode
and K for prefill — each carrying its own compiled plans, though the 2.2 GB of weights can be
shared through ORT's prepacked-weight container. That is the standard shape-bucketing approach and
it is a legitimate T3 demonstration vehicle, but it is not a general EP.

**(b) Shapes known only at `Compute` time.** This is what ORT actually offers and what a general EP
needs: the claim predicate accepts EXTENT-ONLY symbolic dims, and the extents arrive per call.
Unlocks all 324 Phi-3.5 EXTENT-ONLY nodes and all 318 on gpt-oss with no bucketing and no second
session. What has to change is precisely three things, and none of them is a shader:

1. **Claim.** `claim::REQUIRE_STATIC_SHAPES` is a single global `bool` in `ops/common/claim.rs`,
   deliberately placed there so this flips in one location rather than in sixty predicates. It
   becomes a per-shape-class test: EXTENT-ONLY accepted, STRUCTURAL still declined.
2. **Plan format.** `vk::session::CompiledKernel` stores `push_constants: Vec<u8>` and
   `workgroups: [u32; 3]` **baked at Compile**. Those are the only two fields that carry an extent.
3. **Compute.** `VulkanSession::dispatch_ort` currently takes `input_byte_sizes` and
   `output_byte_sizes` from Compile and never calls `GetTensorTypeAndShape`. It must read the real
   shapes and size its allocations from them.

**(c) Fully dynamic, kernel handles it via push constants.** For every one of these ops, **(c) is
already true and (b) and (c) are the same change.** Checked, not assumed:

* `templates::dispatch_elementwise` emits `spec_constants: [EW_LOCAL_SIZE, plan.all_identical]`.
  `EW_LOCAL_SIZE` is a compile-time constant; `all_identical` is shape-*structure*, decidable from
  the symbolic shapes because same-`dim_param` implies equality. Extents appear only in the push
  constants and the 1-D grid.
* `q_gemv` uses `[local_size_x, QB_BITS, QB_BLOCK, QB_HAS_ZP]`. `local_size_x` is derived from
  `K / block_size` and `K` is a literal on every real node. Extents appear only in the `m_total`
  push constant and in `workgroups.y`.

**No pipeline in the Live set is keyed on a runtime extent**, so nothing recompiles per shape and
the pipeline cache is untouched. `ShapePlan::broadcast` already takes `&[&[i64]]` and does not care
where the numbers came from.

The cheapest implementation is therefore not a new symbolic plan IR: it is **running the existing
translate handler a second time at Compute against a `DispatchContext` that holds real shapes**.
`CompileRecorder` is one implementor of that trait; a `ComputeRecorder` is another. Cost is
host-side integer arithmetic over ~360 nodes per `Run`, which is noise beside the weight upload the
same call performs. Memoising the recording on the runtime shape vector reduces it to a hash lookup
after the second call, because a decoder has exactly two shape regimes.

This is an engine change and it is Switch's and Tank's to make. What is mine is item 1, and I will
not flip it until 2 and 3 exist — a claim predicate that accepts symbolic dims while the plan still
bakes extents produces a *wrong answer*, not an error, which is the same class of bug as `Compute`
returning `null` on failure.

#### 7.4.5 The second gate, which nobody had seen: fp16 elementwise

With shapes pinned, 97 Phi-3.5 nodes stop declining on shape and immediately decline on **dtype**:

> ``[dtype] `Mul` is live for f32 only; this node is f16. The f16 variant of the elementwise shader
> compiles but has never executed on a device, and the CPU EP is correct for it``

`Mul` ×64, `Sigmoid` ×32 are f16; `Sub` ×1 is i64. The elementwise family was flipped to `Live` for
f32 only, correctly, because f32 is what `Add` proved. **Every elementwise node in a real fp16
decoder is f16.** So the elementwise coverage this project celebrated is, on a production graph,
worth zero nodes until the f16 variants are exercised — and those variants already compile and are
already in the manifest. This is the cheapest remaining work in the whole plan and it is now the
only thing between the current state and `MatMulNBits` + elementwise clustering on the real model.
It is also a fourth instance of the same lesson: the family was measured against synthetic f32
graphs, and the model is f16.

#### 7.4.6 gpt-oss-20b does not run on any EP on this machine

Recorded because the T5b plan assumes otherwise. Session creation fails in ORT's own CPU kernel
before the Vulkan EP is reached:

> `QMoECPU<MLFloat16>::QMoECPU activation_type_ != ActivationType::SwiGLU || swiglu_fusion_ == 1
> was false. CPU QMoE only supports interleaved SwiGLU format. Please set swiglu_fusion=1.`

`GetCapability` still runs, so the 371-record census above is valid, but there is **no CPU EP
reference to differentiate against for this model**. Any T5b numerical claim needs a different
oracle or a patched graph, and "run it and compare to CPU" — the method every other verification in
this project rests on — is simply unavailable here. Flagging to Trinity rather than solving it.

#### 7.4.7 Corrections to my own recorded conclusions

* §8.1.3 said the island numbers in §8.1.2 were "the ceiling the partitioner reaches once dynamic
  shapes land in the engine, not what the EP does today". That was right about the mechanism and
  wrong about the availability: free-dimension overrides reach a large part of that ceiling
  **today**, and the 161-island measurement above is the simulation confirmed on hardware. I
  described a blocker without checking whether the caller could step around it.
* I have repeatedly written that the blocker for Phi-3.5 is dynamic shapes, singular. There are
  **two** gates in series, and the second (f16 elementwise) is in my own area and cheaper than the
  first.
* The claim-log sink in `ops/claim_log.rs` reopens only when the *path* changes. A harness that
  deletes and reuses one path across sessions writes to the unlinked file and silently records
  nothing. It cost me one confused measurement; noting it as a sharp edge for anyone writing a
  multi-session census.

---

### 7.5 The full-set decline audit, and runtime extents (2026-07-29)

Morpheus's §8.8/§10.0.3/R8 ruling moves dynamic-shape support ahead of the three staged kernels,
and corrects the histogram in §7.4.2 by reading the producer rather than the output. This section
records what I changed, and what the corrected measurement says.

#### 7.5.1 R8 landed in the producer, because a first-match histogram looks exactly like a complete one

`claim_decision` evaluated key → opset → contrib schema → status → predicate and returned on the
**first** failure, so every census this project will ever run inherited the same defect: an early
code is a **ceiling** (nodes that failed there were never shown to the later checks) and a late
code is a **floor** (nodes that reached it had already passed everything before it). Two decline
counts are not comparable without knowing the check order — and nothing in the output said so.

`registry::claim_audit(view, with_counterfactual)` now runs **all** checks in the same canonical
order and collects every failure into `ClaimAudit`:

| field | meaning |
| --- | --- |
| `primary` | first failure — unchanged semantics, so every existing `code` assertion still means what it meant |
| `failures` | the complete set. This is the field planning must use |
| `unevaluated` | checks that genuinely could not run (an unregistered op has no row, so opset/status/predicate are *unknown*, not *passing*) |
| `shape_class` | computed from the node's edges, **independent of its registry row** |
| `predicate_ok` | the row's own predicate, evaluated even for staged rows |
| `predicate_ok_with_runtime_extents` | the same predicate under the counterfactual |

The JSONL record gains `codes`, `reasons`, `unevaluated`, `shape_class`, `predicate_ok`,
`predicate_ok_runtime_extents`; `code` and `reason` are untouched. Extending rather than replacing
is deliberate — Trinity's harness and my own earlier measurements keep working.

`shape_class` must not be routed through the op's predicate: for a staged row the predicate may be
a stub, so its answer is not evidence. Reading the node's edges directly is the only way a staged
node's shape viability is knowable at all, which is precisely the thing R8 says we were missing.

#### 7.5.2 The predicate now distinguishes three cases it used to collapse into one

Per §8.8:

* **rank known, extents symbolic → claimable.** This is the LLM case.
* **rank unknown → decline** (`unknown-rank`, a new code — previously reported as `dynamic-shape`,
  which is wrong: one is a hard decline, the other a floor). Since issue #8 a rank ORT did not
  report may still be *proven* by this EP before the predicate runs — see §7.5.9.
* **data-dependent output shape → permanently declined** (`data-dependent-shape`, new). `NonZero`,
  `Unique`, `Compress`, `StringSplit`, `TopK`, `RoiAlign`, `NonMaxSuppression`. This is a property
  of ONNX, not of our progress, so it does not move. `Reshape`/`Slice`/`Expand` are deliberately
  *not* on the list: whether their shape is data-dependent is a per-node fact, not an op fact.

`REQUIRE_STATIC_SHAPES` is replaced by **`ENGINE_ACCEPTS_RUNTIME_EXTENTS`** (inverted sense). The
rename carries the point: the constant is a statement about `vk::session`, not about claim logic —
claim is *already* correct for symbolic extents, and is being held back by dispatch. To be plain
about it in my own record: rejecting symbolic extents **was right for a static-shape EP and is
wrong for an LLM EP**. That is a design correction, not a defect.

`check_broadcast` had a real latent bug found while making this change: it returned `Ok(())` early
whenever static shapes were not required, so the moment symbolic extents became acceptable,
broadcast compatibility would have gone **unchecked**. It is now symbolic-aware — every pair of
*literal* extents is still checked per right-aligned axis; symbolic and `1` are compatible with
anything.

#### 7.5.3 The counterfactual is a measurement, not a switch

`AssumeRuntimeExtents` is an RAII guard over an `AtomicBool` that makes `runtime_extents_ok()`
report true for one predicate evaluation. It exists so the question *"how many nodes would this
unlock?"* is answered by **running the real predicates**, not by re-implementing them in a Python
probe — which is how §7.4's first attempt got the shape story only half right. The second
evaluation is only paid when the claim log is enabled.

It is not a way to turn the feature on. Flipping `ENGINE_ACCEPTS_RUNTIME_EXTENTS` before the engine
changes lands produces a **wrong answer, not an error**: symbolic dims arrive as `-1`, so push
constants and grid dimensions would be computed from `-1`. Same failure class as `Compute`
returning `null`.

#### 7.5.4 The corrected numbers — Phi-3.5, both devices, identical

363 records. First-match said `dynamic-shape` 258 / `staged` 100 / `not-registered` 5.

| check | full-set | first-match | hidden |
| --- | ---: | ---: | ---: |
| `dynamic-shape` | **356** | 258 | +98 |
| `staged` | 100 | 100 | 0 |
| `not-registered` | 5 | 5 | 0 |

`shape_class`: 360 `extents-symbolic`, 3 `static`, **0** `rank-unknown`, **0** `data-dependent`.

**98 of the 100 staged nodes also fail the shape check.** So the answer the coordinator asked for:

> **Landing all three staged kernels and nothing else unlocks 0 nodes of Phi-3.5.**

Not "at most 100, plausibly fewer" — zero. Every one of `SkipSimplifiedLayerNormalization`×64 and
`GroupQueryAttention`×32 is `extents-symbolic`. The two staged nodes with static shapes are a
`Cast` and a `Greater`, and two nodes is not a milestone.

The asymmetry is therefore not 2.5× and not "larger than 2.5×" — it is **total**. There is no
ordering of the kernel work that produces a claimed node on this graph before the extent work
lands. Kernels-first does not merely manufacture rework; on this model it produces nothing at all.

#### 7.5.5 How many of the 258 become claimable under rank-known/extents-symbolic: 161

Measured by the counterfactual, with no predicate widened:

| | nodes |
| --- | ---: |
| predicate accepts today (status ignored) | 2 |
| predicate accepts with runtime extents | 229 |
| **unlocked by runtime extents alone** | **227** |

The 227: `MatMulNBits`×161, `SkipSimplifiedLayerNormalization`×64, `SimplifiedLayerNormalization`×1,
`Cast`×1. Of those, 161 are **claimable immediately** (their rows are `Live`); the other 66 are
staged, so runtime extents is a **precondition** for their kernels, not a substitute.

Of the 258 first-match `dynamic-shape` nodes, exactly **161 become claimable**. The other 97 do
not, and the reason is the second gate from §7.4.5: `Mul`×64 and `Sigmoid`×32 are **f16**, `Sub`×1
is **i64**. Confirmed independently — with free-dimension overrides pinning the symbolic dims, the
residual histogram is `dtype: 97`. That is R8 one level down: `dynamic-shape` was itself masking
`dtype` for 97 nodes, and the full-set log does not yet decompose failures *inside* a predicate,
which returns one reason. Noted as a known limit of the audit rather than papered over.

#### 7.5.6 gpt-oss-20b — the reversal condition would have been triggered by an artifact

Morpheus's stated condition for revisiting the ruling is gpt-oss showing `dynamic-shape` **below**
`staged`. First-match says `dynamic-shape` 146 < `staged` 197 — the condition appears met. Full-set
says `dynamic-shape` **342** > `staged` 197, with 196 of the 197 staged nodes also shape-blocked and
369 of 371 nodes `extents-symbolic`.

So the reversal condition is **not** met on gpt-oss, and reading the first-match histogram would
have reversed a correct ruling. This is the strongest available demonstration that R8 is not a
tidiness rule. (Session init still fails in ORT's own CPU `QMoE`, so there remains no oracle for
this model — but `GetCapability` runs, so the census is complete and valid. §7.4.6.)

Also surfaced by full-set only: one gpt-oss `Cast` fails on `attribute`, a code that appeared
**nowhere** in the first-match histogram.

#### 7.5.7 What each kernel must take as a runtime parameter

Verified by reading every `Live` dispatch: **no pipeline in the current set is keyed on a runtime
extent**, so options (b) and (c) from §7.4.4 are the same option, and there is **no shader work**.

* `dispatch_elementwise` — spec constants are `[EW_LOCAL_SIZE, plan.all_identical]`. `all_identical`
  is *structure*, not extent. Extents enter only through push constants (shape, strides) and the
  1-D grid. **Consequence:** when any extent is symbolic, `all_identical` cannot be decided at
  Compile, so the handler must select the **general broadcast path**. A performance choice, not a
  correctness one.
* `q_gemv` — spec constants are `[local_size_x, QB_BITS, QB_BLOCK, QB_HAS_ZP]`, and `local_size_x`
  derives from `K / block_size` where `K` is a literal on all 161 real nodes. Needs `m_total` as a
  push constant and `workgroups.y` from the runtime row count.

What must change is in the engine, not in `ops/`:

1. `vk::session::CompiledKernel::{push_constants, workgroups}` stop being baked at Compile.
2. `VulkanSession::dispatch_ort` reads real shapes at Compute — it currently takes byte sizes from
   Compile and never calls `GetTensorTypeAndShape`.
3. Translate handlers re-run against those real shapes.

Recommended cheapest form, for Switch: **not** a symbolic plan IR, but a `ComputeRecorder`
implementing `DispatchContext` with real shapes, re-running the existing handler unchanged
(`ShapePlan::broadcast` already takes `&[&[i64]]`). Memoise on the runtime shape vector — a decoder
has exactly two regimes, prefill and decode.

One constraint of ours makes this harder than it looks: `EdgeType.shape` carries `-1` for symbolic
and **discards the `dim_param` name**, so the EP cannot prove two symbolic dims are equal even when
the graph says they are. ORT's C API does expose `GetSymbolicDimensions`; `NodeView` does not use
it. Until it does, symbolic-vs-symbolic equality must be treated as *unknown*, never as *equal*.

#### 7.5.8 Writing a decode-path kernel against static extents is forbidden

Morpheus's constraint, recorded here because it binds my area: such a kernel is **not a partial
version of the one we need, it is a different one**. Free-dimension overrides (§7.4.4) remain
legitimate as a *harness* device for producing a CPU-EP comparison, and `onnx-shape-inference`
remains an oracle — but neither may be presented as progress on inference, because a decoder whose
claim depends on static extents claims nothing on the second token. Resolving dims statically
improves our test numbers without making inference work, which is the §9.1.2 hazard in its purest
form.

#### 7.5.9 `unknown-rank` is now a *measured* decline, not an accepted one (issue #8, 2026-08-06)

§7.5.2 treats "ORT did not report a rank" as a terminal fact about the node. On transformer
graphs it is not: it is a fact about ORT's *propagator*, and one this EP can often discharge
itself. BERT-SQuAD-12 declined **1,773 edges** as `unknown-rank`, and every one of them traced
back to a single structural cause — the reshape targets are computed at runtime through

```
Shape → Cast(FLOAT) → Slice → Squeeze → Cast(INT32) → Unsqueeze → Concat → Cast(INT64) → Reshape
```

and ORT's partial-data propagation follows only *integral* tensors. The `Cast` to `FLOAT`
destroys ORT's knowledge of the shape tensor's **values**, so it can no longer fold the `Concat`
and no longer knows the `Reshape` output's rank. 58 of 71 `Reshape` outputs and all 98 `MatMul`
A-inputs went unranked for that one reason, and the EP executed **4 dispatches** on 797 nodes.

`rust/src/shape_infer.rs` closes it by proving only what ONNX guarantees — crucially, that a
cast preserves a tensor's *shape* even when it destroys its *values*, so the shape tensor's
**length** survives the float round trip, and the length is the whole of what fixes the rank.
The three rules and the full contract are in `docs/DESIGN.md` §8.11. For this document the
coverage-facing points are:

* The pass runs once per `GetCapability`, before any predicate. Predicates are unchanged: they
  still decline `unknown-rank`, they are just asked fewer times.
* A row claimed on a rank this EP proved is marked `rank_inferred: true` in the claim log, so
  coverage attributable to inference is separable from coverage ORT handed us.
* `ONNXRUNTIME_EP_VULKAN_RANK_INFERENCE=0` restores the old decisions exactly, which is how the
  A/B below was taken.
* `Reshape` is deliberately excluded: its output rank is a property of a runtime *value*, so a
  proven rank is not a shape `Compute()` can bind. It gates on ORT's raw reading.
* **Rank 0 is not a fact.** ORT reports dimension-count 0 both for a real scalar and for a value
  whose shape was never established. Treating the second as the first was a live broken
  commitment (a `Mul` planned for 4 bytes and handed 1,024); `tensor_desc` now demotes any
  uncorroborated rank-0 reading to the dynamic path.

Measured on `NVIDIA RTX A1000`, release build, ORT 1.28.0, `bertsquad-12.onnx`
(sha256 `5f0d96a9…9659e55`):

| | OFF | ON |
|---|---|---|
| dispatches executed | 4 | **367** |
| claimed nodes | 481 | 489 |
| islands retained | 4 | 52 |
| profile-attributed CPU nodes | 781 | 418 |

The `claimed_nodes` column is the reason this section exists in a coverage document with a
warning attached: **+8 claimed is not the result**. The result is +363 dispatches actually
executed and −363 nodes attributed to the CPU EP in the profile. A claim that ends in CPU
re-execution is not coverage, and the two columns move independently — quoting the first would
have overstated a 4→367 change as an 8-node one, and understated it as a fraction. Outputs
agree with the CPU oracle on all three BERT outputs (max-abs `6.68e-06`, max-rel `2.19e-06`);
MNIST-12 (2→2) and MobileNetV2-12 (97→97) are unchanged. No timing claim is made.

---

### 7.6 The fp16 elementwise path — the second gate, and two bugs it was hiding (2026-07-30)

§7.5.5 named a second gate behind `dynamic-shape`: with the symbolic dims pinned, 97 nodes of
Phi-3.5 declined on **`dtype`** — `Mul`×64 and `Sigmoid`×32 at f16, `Sub`×1 at i64. Those 96 f16
nodes are now claimed. The `Sub` is i64 and remains declined; that is correct, not pending.

This section records the work because *how* the gate was shut matters more than that it opened.

#### 7.6.1 The claim narrowing was hardcoded where the evidence already lived

The elementwise rows advertise `NUMERIC`/`FLOAT` capability sets, and the build pipeline was already
emitting f16 SPIR-V for every one of them. The claim was nevertheless f32-only, because the
predicate called a helper literally named `only_f32`. The evidence list `EXERCISED` — the record of
which `(op, dtype)` pairs have actually run against the CPU oracle on a device — sat beside it and
was consulted by nothing in the claim path.

Two sources of truth for the same question, and the weaker one won. Replaced by
`only_proved_dtypes`, which **reads `EXERCISED` directly** (falling through `TEMPLATE_LIVE` to the
representative op). The change is semantics-preserving on introduction, because every live row was
listed at f32 and nothing else. Its value is forward: widening a claim is now the *single* act of
adding a differential result to the evidence list, and the two can no longer disagree, because
there is only one of them.

**Rule.** A predicate that narrows a claim must derive the narrowing from the evidence, not restate
it. A restated narrowing is a copy that will drift, and drift in this direction is invisible: it
declines nodes we can serve, which no test fails on.

#### 7.6.2 Bug 1 — every f16 shader required a device feature the engine never enables

`indexing.glsl` defined `SCALAR_T = float16_t` under `GL_EXT_shader_16bit_storage`. `spirv-dis`
on `ew_binary_mul_f16.spv` shows the consequence: `OpCapability StorageBuffer16BitAccess`. The
engine's `VkDeviceCreateInfo` feature chain carries **only `synchronization2`**. Every f16 module
in the binary was therefore unloadable on every device we support.

Nothing had ever failed, because nothing had ever asked a device to load one — the claim was
f32-only, so the f16 variants were compiled, embedded, shipped and never bound. The census reported
the resulting nodes as declining on `[dtype]`: true, and completely uninformative about the fact
that the alternative would not have worked either.

**Fix:** f16 becomes a *packed* storage path — `uint` buffers with `unpackHalf2x16`/`packHalf2x16`,
which are core GLSL 4.2 and require no device feature at all. Arithmetic was already carried in
`float`, so the only cost is a cast per element. This is exactly the trade `q_gemv.comp` documented
and is why its f16 path worked while the elementwise family's did not.

**Generalised as a test, not a fix.** `no_shader_requires_a_device_feature_the_engine_does_not_enable`
decodes `OpCapability` out of every embedded SPIR-V module and asserts an allowlist. The class of
bug — a shader whose requirements exceed what the device was created with — is silent by
construction whenever the corresponding claim is closed, so it must be caught by inspecting the
artifact rather than by running it.

> This is §8.5's lesson in a new place: a capability we *generate* is not a capability we *have*.
> Generation is cheap and proves nothing; the binding is where the claim is tested.

#### 7.6.3 Bug 2 — a partial final word, which Intel catches and NVIDIA hides

The fp16 differential ran green on device 1 (NVIDIA, 12/12) and **6/12 on device 0** (Intel). Every
failure was the **last element of an odd-length tensor**, and only on the unary rows, whose shape
was `(3, 5)` = 15 elements.

Packed two to a word, 15 f16 elements occupy 30 bytes. The store for element 14 addresses bytes
28..31 — outside the bound buffer range. The RTX 4060 absorbs the overrun and returns the right
answer. The Iris Xe applies `robustBufferAccess`, discards the write, and leaves a zero.

`indexing.glsl` already *asked* the allocator to round sub-word buffers up to four bytes. That
request is unenforceable for ORT-owned tensors: ORT sizes them exactly and the EP binds what it is
given. **A requirement the EP cannot enforce has to be met by declining, not by asking.**

`claim::check_subword_tail` therefore declines any f16 edge whose element count this EP cannot
*prove* even. `provably_even_elements` is sound under symbolic extents — a product is even as soon
as any one factor is even, so a single literal even extent settles it whatever the symbolic dims
turn out to be, and it returns `false` when unprovable rather than guessing. This is why the
restriction costs nothing on Phi-3.5: its f16 tensors have symbolic leading dims and literal even
last axes (3072, 8192).

**Named lift condition** (engine-side, Switch): bind sub-word tensors with
`VkDescriptorBufferInfo.range` rounded up to a multiple of four. Then `check_subword_tail` is
deleted, not relaxed.

**The same latent defect exists on the byte-packed `bool`/`uint8` path.** It has not bitten only
because every row using it is `Staged`. It must not be allowed to become the first thing a future
reader discovers when they flip one.

> Two devices, one right answer, one wrong one, and the wrong one was the *quiet* one. Device 0 is
> the stricter conformance oracle by policy; this is the run that paid for the policy. Had we
> tested only on the faster card, the bug would have shipped and surfaced as a wrong logit on
> somebody else's laptop.

#### 7.6.4 Result on the real model

Phi-3.5, both devices, identical:

| | unpinned | pinned (free-dim overrides) |
| --- | ---: | ---: |
| records | 363 | 358 |
| **claimed** | 0 | **257** |
| declined | 363 | 101 |
| full-set `dynamic-shape` | 356 | 0 |
| full-set `staged` | 100 | 98 |
| full-set `dtype` | 0 | 1 |

Claimed, pinned: `MatMulNBits`×161, `Mul`×64, `Sigmoid`×32. The residual `dtype: 1` is the i64
`Sub`. **`dtype` has disappeared from the unpinned full-set histogram entirely** — after this work,
dynamic shape is the *sole* remaining blocker on 257 nodes, and the claim log now says so directly:
`predicate_ok_runtime_extents` is true, with `codes == ["dynamic-shape"]`, on exactly those 257.

The pinned session also **runs**, on both devices, against the CPU EP as oracle: 65 outputs, max
absolute logit deviation 0.035 on fp16 logits, and the **same argmax token** (30751). That is the
first execution of a real production model's arithmetic on this EP.

The pinned number is a *measurement device*, not a milestone — §7.5.8 stands, and a decoder that
needs free-dimension overrides serves no second token. What it establishes is that the 257 are
blocked by one thing and one thing only, and that when that thing lifts the arithmetic is right.

---

### 7.7 The variant census — which *dtype* each remaining op needs (2026-07-30)

§7.4's rule is that coverage planning is driven by the decline histogram of a real graph rather
than its op histogram. §7.6 showed that rule has a second level: an op census says *which op*, a
decline census says *which op first*, and neither says **which variant of it is worth anything**.
On an fp16 model that last question decides whether a kernel claims 64 nodes or 0.

Measured directly from Phi-3.5's graph (ONNX shape inference in its own process; the EP DLL must
not be loaded alongside it, §7.4):

| n | op | signature |
| ---: | --- | --- |
| 161 | `com.microsoft::MatMulNBits` | `in(f16, u8, f16) -> out(f16)` |
| 64 | `Mul` | `in(f16, f16) -> out(f16)` |
| 63 | `com.microsoft::SkipSimplifiedLayerNormalization` | `in(f16, f16, f16) -> out(f16, f16)` |
| 32 | `Sigmoid` | `in(f16) -> out(f16)` |
| 32 | `com.microsoft::GroupQueryAttention` | `in(f16, f16, f16, i32, i32, f16, f16) -> out(f16, f16, f16)` |
| 1 | `com.microsoft::SkipSimplifiedLayerNormalization` | `in(f16, f16, f16) -> out(f16)` |
| 1 | `SimplifiedLayerNormalization` | `in(f16, f16) -> out(f16)` |
| 1 | `Sub`, `ReduceSum`, `Shape`, `Greater`, `Gather` (one each) | i64 throughout |
| 1 | `Gather` | `in(f16, i64) -> out(f16)` |
| 2 | `Cast` | `in(i64) -> out(i32)` |

Three consequences, none of which the op histogram shows:

1. **A staged kernel written at f32 claims zero nodes of this model.** Every one of the 97 staged
   nodes that matter — `SkipSimplifiedLayerNormalization`×64, `GroupQueryAttention`×32,
   `SimplifiedLayerNormalization`×1 — is **f16 end to end**. This is §7.6 about to happen again,
   one kernel later: the elementwise family was worth 0 nodes on this model for exactly as long as
   it was f32-only. **Raised for whoever writes those kernels; the f16 variant is not a follow-up.**
2. **`SkipSimplifiedLayerNormalization` has a varying output count** — 63 nodes bind two outputs,
   one binds a single output. A predicate requiring exactly two claims 63 of 64, and a kernel that
   writes two where one is bound is a bug, not a decline. The optional second output has to be in
   the predicate and in the dispatch.
3. **`GroupQueryAttention` mixes dtypes within one node** — f16 tensors with **i32** sequence-length
   inputs. It is not an "f16 kernel"; it is a kernel with a per-input dtype contract, and the
   variant axis for it is not a single dtype.

The i64 tail (`Sub`, `ReduceSum`, `Shape`, `Greater`, `Gather`, and `Cast`'s i64→i32) is 7 nodes and
should be treated as a group, not one op at a time — see §7.7.1, which says why it is currently
worth zero regardless.

#### 7.7.1 The i64 variants cannot be loaded either — the same bug, found by looking for it

`ew_binary_sub_i64.spv` declares `OpCapability Int64`. That requires
`VkPhysicalDeviceFeatures::shaderInt64` to be **enabled** at device creation; `vk::device` builds
`VkDeviceCreateInfo` with a feature chain carrying only `synchronization2` and passes no
`pEnabledFeatures` at all. So every `_i64` variant in the binary is uncreatable on every device we
run on, exactly as every f16 variant was.

**And the guard added in §7.6.2 would not have caught it, because I wrote the hole into it myself.**
Its allowlist admitted `Int64` with the comment *"core in Vulkan 1.0 via `shaderInt64`"* — which is
true about the feature *existing* and irrelevant to whether it is *enabled*. A guard whose allowlist
is written from the same misunderstanding as the bug it guards against inherits the bug.

The fix separates two things that were being conflated:

* `GENERATED_CAPABILITIES` — what a *built* variant may declare. Deliberately wide: an unloadable
  variant costs kilobytes, and the i64 modules must exist before the feature that makes them
  loadable is worth adding.
* `ENGINE_ENABLED_CAPABILITIES` — what the engine actually enables, and therefore the only thing a
  **live claim** may rest on. Currently `Shader`, and nothing else.

`no_live_claim_rests_on_an_unloadable_variant` walks every proved `(op, dtype)` pair, resolves its
module stem, decodes its capabilities, and fails if any is outside the enabled set. **Verified by
negative control**: adding `("Sub", "i64")` to `EXERCISED` fails with

> `` `Sub` is claimed at i64 via `ew_binary_sub_i64`, which declares SPIR-V capability 11 — the
> engine enables no such feature, so that module cannot be created on any device ``

A guard that has never fired is a guard nobody has tested, so it was fired on purpose and reverted.

**Requirement for Switch, if the i64 tail is ever worth 7 nodes:** enabling `shaderInt64` is three
edits together, not one — enable it in the feature chain, probe it in `vk::caps`, and decline the
i64 variants on devices that lack it. It is *not* universally available, so it gates variants; it
must never gate device admission (§7.2).

> **Rule.** Generation and admission are different claims. The build pipeline producing a variant
> says only that GLSL compiled; whether a device can create the module is a separate fact, and the
> only place the two are reconciled is at the claim.

---

### 7.8 The last ten nodes — two claimed, eight declined on purpose (2026-07-31)

At `77d5d2a` the Phi-3.5-mini-int4 graph partitioned as **353 claimed / 1 island / 10 declined**,
and the execution census confirmed it from the other side: 30 CPU node-executions over 3 runs =
exactly the 10 declined nodes × 3.

Island attribution had already closed the interesting question: **0 cut-creating declines remained;
GQA was the sole cut creator.** So none of these 10 merge islands. What they *were* was the entire
CPU-side cost of an inference, and each one forces a boundary crossing into and out of a 353-node
island. That reframes the work: this is not a coverage-percentage exercise, it is a
boundary-crossing exercise, and the right answer for some of the ten is to leave them where they
are.

#### 7.8.1 The ten are three structurally different things, not ten gaps

Reading each declined node's producers, consumers, dtypes and shapes out of the graph — rather
than reading the decline histogram — split them cleanly:

| group | nodes | what it is |
|---|---|---|
| **Data path** | 2 | `embed_tokens/Gather` → `layers.0/input_layernorm` (`SimplifiedLayerNormalization`) |
| **Control plane** | 7 | attn-mask + rotemb-cache scalar arithmetic, **every tensor INT64** |
| **Control flow** | 1 | `rotemb_caches_subgraph/If` |

Only the first group is on the tensor data path. The `Gather` emits `FLOAT16[batch,seq,3072]`
consumed by *both* the declined input LayerNorm and the already-claimed layer-0
`SkipSimplifiedLayerNormalization`; the LayerNorm's output feeds the claimed `qkv_proj/MatMul_Q4`.
Note also that **only layer 0 has an unfused input norm** — every other layer's is already fused
into `SkipSimplifiedLayerNormalization`. That is why claiming one norm buys one node, not 32.

#### 7.8.2 Prediction, written before building

Recorded in `history.md` before any code, per the standing requirement that predictions be
falsifiable in advance:

| # | prediction | falsifier | outcome |
|---|---|---|---|
| P1 | islands stay at 1 | islands ≠ 1 on either device | **CONFIRMED** |
| P2 | 353 → 355 claimed, declines 10 → 8 | any other pair | **CONFIRMED exactly** |
| P3 | per-inference upload drops 12,280 B at s=1 | drop < 6,144 B | **MISSED — 2× too large** |
| P4 | first-inference upload grows +187.9 MiB (embedding table becomes a resident weight) | growth < 150 MiB | **CONFIRMED**, measured +188.25 MiB |
| P5 | 0 cut-instances stays 0 | any cut instance appears | **CONFIRMED** |

P3 is the one worth reading. I predicted two hidden-state tensors' worth of saving; the measured
saving is **6,136 B**, almost exactly one. The upload counter instruments *uploads only*, and after
both claims land there is one crossing removed, not two. Four of five predictions confirming is
not evidence that the model of the graph was good — it is evidence that four easy predictions were
easy. The one that failed is the one that taught something.

#### 7.8.3 What was claimed

**`SimplifiedLayerNormalization` (RMSNorm).** `simplified_layer_norm_{f32,f16}.comp`, three
bindings, three-pass tree reduction. Needed **two rows** (`Ms` and `Ai` domains) because the ORT
GenAI builder emits it with `node.domain == ""`; Phi-3.5 keys against the `Ai` row. The claim
predicate requires `gamma` and declines any node declaring more than one output (`Arity`) — the
skip-norm spelling has a slot-3 output this shader does not produce, and silently dropping it
would be exactly the "correct claim that is a wrong claim" of §7.0.2. Retired the
`NEEDS_REDUCTION` blocker; moved `RMSNormalization` to `Staged(UNEXERCISED)`.
Result: **353 → 354, islands 1.**

**`Gather`.** `gather_{f32,f16}.comp`, new module `rust/src/ops/indexing.rs`. One three-extent
flattening (`outer / gathered / inner`, plus `n_idx`) covers every `axis` and every indices rank
with a single shader. int64 indices are read via their **low word**, so this does not depend on
`shaderInt64`. `gather_f16` gives one thread sole ownership of one output `uint` word — a
stronger race-freedom argument than the disjoint-lane `atomicAnd`/`atomicOr` the norm shaders
need, and worth preferring wherever the access pattern allows it.
Result: **354 → 355, islands 1.**

Caps are deliberately `FLOAT` only. Widening them to `ANY` would let this same row claim the
attn-mask `Gather`, whose output is `seqlens_k` — integer index data that this shader would
silently corrupt by round-tripping through `float`. The correct observable is that the attn-mask
`Gather` moved from `[not-registered]` to **`[dtype]`**: still declined, now declined *for the
right reason*. `gather_claims_float_data_only` is the guard.

#### 7.8.4 What was declined, permanently, and why

A decline with a reason is a result. These eight are not backlog.

**The INT64 control plane (6 nodes)** — `attn_mask_subgraph/{Shape, Gather, Gather/Cast, ReduceSum,
Sub, Sub/Cast}`. Every tensor in this cluster is an INT64 scalar or near-scalar: `INT64[]`,
`INT64[2]`, `INT64[batch,1]`. It produces `seqlens_k INT32[batch,1]` and `total_seq_len INT32[]`,
consumed by all 32 GQA nodes. Claiming it needs **three independent mechanisms** — `shaderInt64`
(a non-universal device feature that would gate variants, §7.7.1), a Cast dtype-pair matrix, and a
reduction template — to move a few hundred bytes of scalar integer arithmetic that the host
computes for free. The cost is three mechanisms and a device-feature dependency; the benefit is
measured in *bytes*, and the bytes are three digits. **Decline stands.**

**`Shape` (1 node)** — its output derives from a tensor's shape, not its data. The host already
knows the shape; a device round-trip to compute 16 bytes the host is holding is not an
optimisation under any transfer model. **Decline stands.**

**`If` (1 node)** — `rotemb_caches_subgraph/If`. Its `then_branch`/`else_branch` are GRAPH-typed
attributes and the EP has no subgraph-execution machinery. Worse, the predicate is `BOOL[]` and
would have to be read back host-side to select a branch, forcing a **fence stall in the middle of
a 355-node island** — the opposite of what claiming is for. Its outputs `cos_cache`/`sin_cache` are
session-invariant, so there is no per-inference work to win. This op does not belong on the GPU;
it belongs on the CPU, and control flow generally does. **Decline stands, and is not a coverage
gap.**

#### 7.8.5 Measured result, both devices

| state | claimed | islands | CPU node-exec / 3 runs | declines |
|---|---|---|---|---|
| baseline `77d5d2a` | 353 | 1 | 30 (10 × 3) | 10 |
| + `SimplifiedLayerNormalization` | 354 | 1 | 27 (9 × 3) | 9 |
| + `Gather` | **355** | **1** | **24 (8 × 3)** | **8** |

Selector 0 (NVIDIA RTX 4060) and selector 1 (Intel Iris Xe) agree on every figure.
`cross_run_identical = True`; `argmax = 30751`, matching CPU, on every run of both devices.

Per R13, no wall-clock figure appears in this section. Coverage counts, island counts and byte
counts are quotable; nanoseconds are not, because `phase_containment` is RED on both devices and
no run carries a `MATCH`-attributed verdict.

---

### 7.9 Post-residency transfer recalibration — and what is still not calibrated (2026-07-31)

I previously wrote that `transfer_ns` was "conservative in the right direction but 10–100× too
small vs actual cost", and deferred calibration until residency landed. It has landed. This is the
recalibration, and it contains a correction to a number that was about to be credited to the wrong
cause.

#### 7.9.1 The control run, and the 2× I did not earn

The brief carried "per-inference upload went 1997.6 MiB → 0.756 MiB". After claiming the two ops I
measured **0.38 MiB** and was one step from reporting a halving. Under R9 — evidence scales only
with falsifying instruments — I built the **pre-change commit `77d5d2a`** and ran the *same*
instrument on the *same* machine:

| build | per-inference upload |
|---|---|
| control, `77d5d2a`, pre-claim | **405,512 B** (0.3867 MiB) |
| after both claims | **399,376 B** (0.3809 MiB) |
| **attributable to the two claims** | **6,136 B** |

The halving was **already present in the control**. It belongs to residency, not to op coverage.
Had I skipped the control I would have published a 2× that was someone else's and mine by
accident. This is the concrete case for R13's second clause: the confirming measurement was the
dangerous one, and the number that needed scrutiny was the one I liked.

#### 7.9.2 The attribution is exact, not approximate

From a single 1-run counters snapshot:

```text
session_staging_upload_bytes   2,292,025,360
weight_cache_release_bytes     2,291,625,984
difference                           399,376   == the per-run delta, to the byte
```

That identity is what makes this an attribution rather than a number. 99.98% of the 2.19 GiB is
staged once and stays resident; the remainder is per-inference graph I/O plus the boundary tensors
of the 8 nodes still on CPU. It also closes an open question: `cos_cache`/`sin_cache` (~24 MiB of
`If` outputs) do **not** cross per inference — they cannot, since the entire per-inference upload
is 0.38 MiB.

Both directions, exactly linear over a 1/2/3-run sweep, **byte-identical on both devices**:

| direction | bytes / inference | share |
|---|---|---|
| upload (H→D) | 399,376 | 46.6% |
| readback (D→H) | 457,344 | 53.4% |
| **total boundary** | **856,720 (0.817 MiB)** | |

Two corrections to the previous constants' reasoning: the boundary is **0.817 MiB, not 0.756**,
and it is **asymmetric with readback larger** — the old doc comment modelled two symmetric
0.378 MiB halves. Both figures are now pinned as `TransferModel::MEASURED_PHI35_UPLOAD_BYTES` /
`MEASURED_PHI35_READBACK_BYTES` / `CONTROL_PHI35_UPLOAD_BYTES_PRE_CLAIM` with three unit tests, so
the prose cannot drift from the measurement.

#### 7.9.3 What is calibrated is bytes. Nanoseconds are not.

This must not be overstated. `TransferModel::fit` has **still never been handed a real sample**,
and under R13 it cannot be: no wall-clock figure is quotable from a run whose verdict is not
attributed `MATCH`, and `phase_containment` is RED on both devices. The counters *do* carry
`session_staging_upload_us`; it is deliberately unused. `fixed_ns` and `bytes_per_ns` remain
guesses. **What landed is a byte measurement, not a nanosecond measurement**, and the honest
statement of this recalibration is that it corrects the *input* to the model, not the model.

Evaluating `cost_ns` on the measured bytes:

| model | up | down | total | 3× gate threshold |
|---|---|---|---|---|
| `DISCRETE` | ~93,281 ns | ~98,112 ns | **~191 µs** | ~574 µs |
| `UMA` | ~29,984 ns | ~31,434 ns | **~61 µs** | ~184 µs |

#### 7.9.4 The consequence, stated plainly: the gate will decline far less

Pre-residency the EP re-uploaded ~2 GiB every inference, making `transfer_ns` ≈ 167 ms per
direction — larger than any kernel's compute time. The economics gate was pathologically strict:
essentially nothing but an anchor could pass, and **the anchor exemption was carrying the entire
partition**. Post-residency the modelled cost is ~191 µs, a ~1,750× drop, and the 3× threshold
falls from ~1 s to ~574 µs.

So: **with transfer nearly free, the net-benefit gate will decline far less, and islands that were
uneconomic may now be worth claiming.** That is coverage going up for a measured reason. It is also
the opposite of the direction I was told to expect a day ago, and the reason it reversed is that
the measurement changed, which is the only acceptable reason.

The matching risk, recorded now rather than discovered later: post-residency, `fixed_ns` is ~63% of
the modelled `DISCRETE` cost (`post_residency_the_gate_is_dominated_by_fixed_cost` asserts it). The
gate's remaining teeth are almost entirely in the **one parameter with no measurement behind it**,
and for any island with a small boundary the gate has degenerated to `2 × fixed_ns` — insensitive
to the byte count it is nominally reasoning about. An under-declining gate is a *silent* failure:
it shows up as slow inference, never as a wrong answer.

---

### 7.10 `retain_viable` is wired, and Phi-3.5 does not exercise it (2026-07-31)

§7.3.1 records `retain_viable` as R10-resolved: `viable_islands_retained` is in the counters ABI
(version 2), the wiring census reports it **WIRED**, and both guard directions have named
falsifiers — the anchor exemption against over-declination (falsifier: `bench/phi35.py → 0
claimed`), and `tests/ops/test_partition_gate.py` against under-declination (falsifier: the
`[partition]` code assertion goes red).

That is all true, and it is not the whole story. **This is a coverage gap in the gate's own
exercise, distinct from op coverage, and it should be stated where the coverage numbers are.**

#### 7.10.1 The bypass is structural, not incidental

In `ep.rs::GetCapability`:

```rust
let only_one_cluster = clusters.len() == 1;
...
if !only_one_cluster { n_viable_retained += 1; }
```

Phi-3.5 partitions into **exactly one cluster**. Therefore `only_one_cluster == true`, the
economics gate never has a decision to make, and `viable_islands_retained` is **structurally pinned
at 0** on our only real model — confirmed in every counters snapshot taken this session on both
devices.

The counter is doing its job: it is present-and-0, which the census can distinguish from
UNWIRED (key absent). But present-and-0 here does not mean "the gate ran and rejected everything".
It means **the gate did not run.** Those are different states and only the code tells them apart.

#### 7.10.2 Why this matters more now than it did yesterday

Per **R10**, a mechanism whose only exercise is a synthetic test is one step away from unwired.
`retain_viable`'s only exercise is `test_partition_gate.py`'s two-branch Sigmoid model — a graph
built specifically to produce two disjoint 1-node clusters so the gate has something to decline.
Nothing in the real workload touches it.

§7.9.4 raises the stakes: the gate's behaviour just changed by ~1,750× and its remaining teeth sit
in an uncalibrated constant. A mechanism that (a) has no real-model exercise, (b) just had its
operating point moved by three orders of magnitude, and (c) fails silently, is the highest-risk
combination in the partition path. **R11 applies directly: this is a decomposition that appears to
close.** The counter says WIRED, the test is green, the census is satisfied — and the gate has
never made a decision about our model.

#### 7.10.3 What would close it

Not a bigger synthetic test — a real one. The gate becomes genuinely exercised the first time a
production model partitions into ≥ 2 clusters. Today the only known cut creator was GQA, and it is
claimed, so **the very work that drove island count to 1 is what removed the gate's only real-model
exercise.** Concretely, closing this needs either a second real model that produces multiple
clusters, or an accepted counterfactual run with an anchor op force-declined to induce a multi-
cluster partition and assert the gate's verdicts against it. Until one of those exists, the correct
status is:

> `retain_viable`: **WIRED, exercised only synthetically.** Never evaluated on a production graph.
> `viable_islands_retained == 0` on Phi-3.5 means *bypassed*, not *all-rejected*.

**SUPERSEDED 2026-08-01 by §7.11.** The status line above was accurate when written and is no
longer the state of the code. The bypass is gone; the gate evaluates Phi-3.5's island in the
shipping configuration; `viable_islands_retained` on Phi-3.5 now reads `1`, and it reads `1`
*because the gate retained it*, not because a branch skipped the gate.

### 7.11 RAI-011 closed: one gate, always evaluated, and the override is a state of its own (2026-08-01)

Rai flagged §7.10 as the R10 shape one level up — *a mechanism true of one entry point, silent on
another* — and was right. It was worse than "unexercised". `GetCapability` read:

```rust
let verdict = if only_one_cluster { Verdict::Claim } else { partition::evaluate(..) };
```

The **call site decided whether the gate got to run.** On our only real model the gate was not
merely unexercised, it was *unreachable*. And per **R12** that made the artifact wrong, not just
thin: a count whose event cannot occur in its frame is not `0`.

#### 7.11.1 What changed, and why it is not a second check

Tank wired the observable half at the `GetCapability` site he owns — `net_benefit_gate` as a JSON
**string** (`UNWIRED` / `BYPASSED` / `EVALUATED` / `MIXED`), and `viable_islands_retained` emitted
as `"UNOBSERVABLE"` rather than `0` when nothing reached the gate. He also named the hazard in my
half precisely: **a second `partition.rs` path would reproduce RAI-011 inside its own fix.**

So there is no second path. There is one:

```rust
pub fn gate_islands(islands: &[Island], model: &TransferModel, policy: &Policy) -> Vec<GateOutcome>
```

* It is the **only** entry point. `retain_viable` is now a projection of it, so "the survivors" and
  "what the gate decided" cannot drift apart.
* It **always** calls `evaluate`, once per island, with no branch in front of it.
* The single-island exemption is a property of the *set*, which is why it lives here and not at a
  call site — a call site that decides whether to consult the gate **is** a second gate.
* The exemption is applied **after** evaluation, as `GateOutcome::SoleIslandOverride(RejectReason)`
  — it *carries the verdict it overrode*. An override that discards that verdict is a bypass
  wearing a different name.

Behaviour is unchanged: a sole island that the gate rejects is still claimed. What changed is that
the rejection now exists, is computed from that island's own bytes and FLOPs, and is visible.

#### 7.11.2 Three states, three fields, and one of them is a string

| fact about a run | field | value on Phi-3.5 today |
|---|---|---|
| nothing reached the gate | `viable_islands_retained` | `"UNOBSERVABLE"` (string — arithmetic on it fails loudly) |
| the gate ran and retained the island | `viable_islands_retained` | `1` |
| the gate ran, rejected, and was overridden | `net_benefit_sole_island_overrides` | `0` in the shipping config |
| a second un-evaluated path exists | `net_benefit_gate_bypasses` | `0`, and must stay `0` forever |

`viable_islands_retained` and `net_benefit_sole_island_overrides` are **different fields**, so
"the gate retained it" and "the gate rejected it and we kept it anyway" can never again share a
digit. In `partition.rs` they are different variants of a sum type, so no arithmetic can conflate
them either.

#### 7.11.3 The falsifier is an artifact, and it varies with its input

`rust/tools/probe_net_benefit_gate.py` runs Phi-3.5 at one commit under several partition
configurations and requires the observables to move. Both devices, byte-identical results
(`bench/results/net_benefit_gate_probe-dev{0,1}.json`):

| config | `net_benefit_gate` | `evaluations` | `bypasses` | `viable_islands_retained` | `sole_island_overrides` |
|---|---|---|---|---|---|
| **default (shipping)** | EVALUATED | 1 | 0 | **1** | 0 |
| anchor exemption off | EVALUATED | 1 | 0 | 0 | **1** |
| anchor off, `fixed_ns` 1e3 … 1e8 (6 runs) | EVALUATED | 1 | 0 | 0 | **1** |
| anchor off, bytes free, `fixed_ns` 1e6 | EVALUATED | 1 | 0 | **1** | 0 |
| anchor off, bytes free, `fixed_ns` 1e7 | EVALUATED | 1 | 0 | 0 | **1** |

`claimed_nodes` is **355 in every row**: the gate's verdict moves, the graph does not. The
counterfactual rows exist because Phi-3.5's island is anchor-bearing, so with the exemption on the
economics branch never answers and the artifact would be a constant — and a constant proves
nothing (R10). Every non-default row logs a WARN naming itself a counterfactual.

The last two rows are the load-bearing pair. **Prediction, written before they ran:** compute is
23,020,437,504 ÷ 1000 = 23,020,437.5 ns and transfer ≈ 2·`fixed_ns` once the byte term is removed,
so at margin 3 the flip is at 23,020,437.5 ÷ 6 = **3,836,739.6 ns**; 1e6 must claim and 1e7 must
override. *Falsifier: either landing on the other side, or neither moving.* **Both confirmed, on
both devices.** That is the observable changing with *this specific input*, which is what R10 asks
for and what a `net_benefit_gate: EVALUATED` line on its own does not supply.

The knobs are `ONNXRUNTIME_EP_VULKAN_PARTITION_{MARGIN,MIN_NODES,FLOPS_PER_NS,ANCHOR_EXEMPTION,
FIXED_NS,BYTES_PER_NS}`. They exist to make the gate falsifiable, not to be tuned.

### 7.12 `fixed_ns` sensitivity — the uncalibrated parameter cannot change any decision we make today

§7.9.4 recorded the risk that `fixed_ns` is ~63% of the modelled DISCRETE cost and has never seen
a measurement. R13 still forbids calibrating it — the device-clock gate that would have supplied
the sample is itself blind to bias (`gpu_steady_tail()` is a variance test over a suffix; a run
held at the 210 MHz idle clock is *perfectly steady* and earns the most confident verdict at 10.99×
wrong). So this is **a sensitivity statement, not a calibration.** No timing figure is quoted below;
every number is a count, a byte volume or a modelled ratio.

**Part 1 — fed the estimator's own bytes, the verdict is constant at every `fixed_ns`, including
zero.** `GetCapability` estimated Phi-3.5's single island at 23,020,437,504 FLOPs and
**89,199,100,032 boundary bytes** (`PartitionStats`, dev0). The byte term alone is ~968× the margin
requirement, so the economics check rejects at `fixed_ns = 0` and at `fixed_ns = 1e8` alike.
Confirmed by six measured runs across five orders of magnitude — all identical. *There is no value
of `fixed_ns` that changes this decision.*

**Part 2 — fed the measured bytes, the verdict is constant across the whole plausible range.**
Substituting the instrumented boundary (upload 399,376 B + readback 457,344 B = 856,720 B,
asymmetric with readback larger), the flip point solves to **~3.80 ms per transfer — 63× above the
current guess of 60 µs.** Anything from 0 to ~1 ms claims. *There is no plausible value of
`fixed_ns` that changes this decision either.*

**Conclusion: `fixed_ns` is not the parameter with the teeth, and calibrating it is not on the
critical path.** Both tests are pinned in `partition.rs`
(`fixed_ns_cannot_change_the_verdict_on_the_estimated_phi35_island`,
`with_measured_bytes_the_flip_point_is_far_outside_the_plausible_range`) so the claim goes red if
the constants move.

#### 7.12.1 What the sweep found instead, and it is worse

The two halves of Part 1 and Part 2 disagree about the same island's boundary by a factor of
**104,116**: 89,199,100,032 B estimated against 856,720 B measured. That is not measurement noise.
`GetCapability`'s estimator counts *every* claimed node's outputs as boundary bytes (documented as
deliberate — "over-counting is safe, it makes islands harder to claim") and substitutes `128` for
every unknown dimension. On a 355-node fused island with symbolic `sequence_length`, those two
choices compound into a figure five orders of magnitude off the instrumented one.

**R11 applies to my own model here.** The decomposition looked closed — `TransferModel` calibrated
in bytes, a gate comparing compute against transfer, a counter proving it ran — and the two sides
of the comparison came from different sources, one of which is not a measurement. The gate's
apparent strictness on Phi-3.5 is an artifact of its own byte estimator, and the *only* reason the
model is claimed at all is the anchor exemption. Remove the exemption and the EP declines the whole
graph, at every `fixed_ns` I swept.

I am **not** fixing the estimator in this change. It fails safe (towards the CPU), it is
load-bearing for the anchor exemption's design intent, and changing it would move the partitioner's
behaviour in the same commit that makes the partitioner observable — which is exactly how you lose
the ability to attribute a regression. Recorded as the next item, ahead of any nanosecond
calibration.

---

### 7.13 The eight declines, re-attributed against the current claim set (2026-08-01)

The histogram in §7.8.4 was written before `SimplifiedLayerNormalization` and `Gather` were
claimed. The graph shape changed, so the conclusion that closed the island lever — *no decline
creates a cut* — had to be re-derived rather than re-quoted.

Re-run at this commit, `bench/island_attribution.py`, **both devices, byte-identical**:

| | dev0 (RTX 4060) | dev1 (Iris Xe) |
|---|---|---|
| claimed | 355 | 355 |
| islands | 1 | 1 |
| declines | 8 | 8 |
| cut-instances | **0** | **0** |

Cut count alone is no longer sufficient. With one island, a decline can sit *between* two claimed
nodes and create zero cuts, and that is a different category from a decline at the edge — it is
what Justin asked me to check for. `bench/island_attribution.py` does not answer it, so
`rust/tools/probe_decline_position.py` does. It reads the CLAIM_LOG without running the model, so
it cannot perturb what it measures.

| position | ops | code |
|---|---|---|
| `DETACHED` (no claimed neighbour on either side) | `Gather`, `Greater`, `ReduceSum`, `Shape`, `Sub` | dtype / staged / not-registered |
| `EDGE_ENTRY` (feeds claimed nodes, fed by none) | `Cast` ×2, `If` | staged / not-registered |
| `INTERIOR` (claimed on **both** sides) | **none** | — |

**All eight are still 0-cut-creating, and none has changed category.** Five are fully detached —
the INT64 control plane computes `seqlens_k` and `total_seq_len` from graph inputs and never
touches a claimed tensor. The other three each feed exactly 32 claimed GQA nodes and are fed by
nothing claimed: they are prologue. Claiming any of them would shorten the prologue; it cannot
merge islands, because there is only one.

The probe also carries a falsifier that is not about backlog at all: **`INTERIOR` combined with
`island_count == 1` is a contradiction** — a claimed→declined→claimed path inside a single island
is a cycle ORT could not have fused. If that row ever appears, the island count is wrong, not the
op coverage. It is empty on both devices.

**Consequence for the island lever: it is closed, and closed for the second time on evidence
rather than on the first result being repeated.** 355/363 is not the number to move. The number to
move is boundary bytes — and §7.12.1 says the estimator's version of that number is off by 10⁵,
which is now the top of the partition backlog.

### 7.14 Which arm claimed Phi-3.5, and what the override overrode (2026-08-01)

A reconciliation of `bench/results/wiring_census-dev0.json` raised a sharper version of RAI-011:
on Phi-3.5 the fused island is retained — but *by what*? "The gate approved it" and "the gate
rejected it and the sole-island override kept it" are different facts that both produce one island,
claimed, executing, MATCH. If it were the override, the net-benefit gate would never have said *no*
on real input, and a gate that cannot decline is not a gate.

**The trap named in the question is the important part.** `net_benefit_gate: EVALUATED` and
`net_benefit_gate_bypasses: 0` both *look* like the gate is working, and both read exactly the same
in a world where the gate runs and always approves. `bypasses` is a tripwire for a different
failure — something skipping the gate — so it cannot falsify this one. Neither field is evidence
here.

**The field that is evidence: `net_benefit_sole_island_overrides`.**

| run | claimed | retained | overrides | override reason |
|---|---|---|---|---|
| Phi-3.5, shipping configuration | 355 | **1** | **0** | `UNOBSERVABLE` |
| one-node elementwise chain (the census lane's shape), shipping configuration | 1 | 0 | **1** | `TOO_SMALL` |
| a graph with no claimable node at all | 0 | `UNWIRED` | 0 | `UNOBSERVABLE` |

**Answer: on Phi-3.5 the predicate passed on its own.** `overrides = 0`, so the override did not
fire and cannot be what retained the island. The headline island is claimed by the gate.

**Which arm of the predicate, though, is a second question, and the answer is less comfortable.**
`evaluate` has two rejecting arms and returns `Claim` from three places; the counters record the
verdict, not the arm. Changing one input at a time answers it without reading the code
(`bench/results/net_benefit_gate_probe-dev0.json`, rows `default` and `no_anchor`, everything else
held fixed):

- anchor exemption **on** (shipping): retained 1, overrides 0.
- anchor exemption **off**: retained 0, overrides 1, reason `TRANSFER_DOMINATED`.

Nothing else differs between those two runs, so **the anchor exemption is the deciding term on
Phi-3.5**. The economics arithmetic is reached and, when allowed to decide, it *declines the graph
we ship* — at every `fixed_ns` in §7.12's range, for the reason §7.12.1 gives: the estimator's
boundary bytes are 10⁵ too large. The gate's economics arm is therefore not merely untested on
real input; it is wrong on real input, and the anchor exemption is what stands between that wrong
answer and the partition.

**Has the predicate ever returned `Reject` on a real graph at shipping defaults? Yes.** The census
lane is that case: a one-node elementwise chain, no `PARTITION_*` environment override anywhere in
the census tooling (only `probe_net_benefit_gate.py` sets those), `overrides = 1`. So the reject
branch does fire in production configuration today. The gate is not a decoration. What had never
happened is an *economics* rejection at shipping defaults — the census lane rejects on **size**,
and that is now read from an artifact rather than from the source: `rust/tools/probe_override_reason.py`
rebuilds the census lane's shape, predicts `TOO_SMALL` before running, and observes `TOO_SMALL` on
both devices.

**The observability gap that made this hard to answer, now closed.** `overrides: 1` said the
override fired and said nothing about what it overrode, even though `GateOutcome::SoleIslandOverride`
carries the `RejectReason` in memory: the reason died at the counter boundary. `counters.rs` now
emits `net_benefit_override_reason`, a token in Tank's idiom rather than a number —
`UNOBSERVABLE` when no override occurred (R12: the event whose reason is asked for did not happen
in this frame), else `TOO_SMALL` / `TRANSFER_DOMINATED` / `MIXED`, plus `UNRECORDED` for the
drift case where an override was counted and no reason arrived. It is a bitmask, not a
last-writer-wins slot, so two overrides with different reasons cannot collapse onto whichever ran
last. **It varies with its input in the artifact** — `UNOBSERVABLE` in the shipping Phi-3.5 row,
`TRANSFER_DOMINATED` in the eight anchor-off rows, `TOO_SMALL` on the one-node graph and
`UNOBSERVABLE` again where nothing is claimable — which is the R10 requirement for calling it
wired. Three distinct tokens from three distinct inputs, on both devices.

**No gate behaviour changed.** The override remains correct policy for a sole island: there is no
alternative partition, so declining it hands the whole graph back for no gain. This is
documentation plus an observable plus tests, exactly as asked.

**Backlog, needing sequencing rather than doing:** the *claim* side is still unattributed in the
artifact. `Verdict::Claim` does not say which of the three arms produced it, so "the anchor
exemption decided this" remains an inference from a two-run counterfactual rather than a field.
Making it a field means `Verdict::Claim` carries a reason, which is a `partition.rs` change, and
`partition.rs` was being edited concurrently. Deriving the arm at the `ep.rs` call site instead
would be a second copy of the predicate — the precise thing RAI-011 was about — so it is not an
option.

### 7.15 The proof ledger is wired — criterion 11, and the last `UNWIRED` (2026-08-01)

`ledger_lookup` was the twelfth mechanism and the only one still reporting `UNWIRED`. It now
reports a value it computed on the run:

```
ledger_lookup: ALL-PROVEN proven_key_lookups=6 ledger_hits=6 ledger_entries=9
               unproven_declines=0 unproven_forms_enabled=[]
               (hits is typed: 'UNWIRED'/'UNOBSERVABLE'/int)
```

The type is load-bearing, and it is the same fix R12 asked for when *bypassed* and *all-rejected*
shared one `0`: **`UNOBSERVABLE`** (no ledger in this frame — the counters surface carries no
ledger fields at all), **`UNWIRED`** (fields published, nothing consulted them), and an **int**
(something looked a key up). An increment cannot forge a type.

**The artifact.** `evidence/proof_ledger.jsonl`, 9 entries, header digest `e4436e93c19c8744`,
generated by `rust/tools/gen_proof_ledger.py` and never hand-edited. It is baked into the crate
with `include_str!`, so a build cannot claim a form whose proof is not in the binary that claims
it.

**The key** — six components, one `::`, exactly five `/`:

```
{domain}::{op_type}/{opset_bucket}/{in_dtypes}>{out_dtypes}/{variant}/{shape_class}/{optional_inputs}

ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2
com.microsoft::MatMulNBits/1+/f16,u8,f16>f16/q_gemv_matmul_nbits_f16/static/scales
com.microsoft::MatMulNBits/1+/f16,u8,f16,u8>f16/q_gemv_matmul_nbits_f16/static/scales+zero_points
```

Those last two are the §8.9 vindication pair, and they are in the ledger on purpose: the
2026-07-30 all-zero-logits defect was `MatMulNBits` **with** vs **without** `zero_points`. The two
forms differ in `populated_optional_input_set`, so they are two keys, so a proof of one can never
be returned for the other. This is asserted, not asserted-about: `distinct_forms_have_distinct_keys`
in `registry.rs` and the two-armed case pair in `tests/ops/test_proof_ledger.py`. **Switch's
`arms_must_differ` lesson applies here directly** — a ledger probe whose two arms produce the same
key is a perfectly stable, perfectly wrong answer.

**Two real defects my own controls caught, both worth recording:**

1. **The comma collision.** Proof keys contain `,` (`f32,f32>f32`) and the `CLAIM_UNPROVEN` escape
   hatch split its list on `,`. A well-formed key arrived as three invalid fragments, the list was
   correctly discarded, the run claimed nothing — and the differential comparison still returned
   `MATCH`, because it was comparing CPU against CPU. Only the **attribution** requirement
   (`claimed_nodes > 0 && dispatches_executed > 0`, §10.0's third amendment) caught it. The
   separator is now `;`, which cannot occur inside a key.
2. **The prefix-accepting validator.** The regression test for (1) found that
   `ai.onnx::Add/7+/f32` — the first comma-fragment of a real key — *passed* `ProofKey::validate`,
   which only required a `/`. A truncated key matches nothing and reads exactly like a key that
   matches something. `validate()` now requires the full structure.

**One `ERROR(instrument)`, and it is not a detection (R13).** `sqrt_f32` first returned `DIVERGENT`
with `worst_rel: 0.0` — self-contradictory on its face. Inputs were `standard_normal`, so `Sqrt` of
a negative produced NaN on *both* sides; `np.allclose` calls NaN≠NaN a divergence while
`max(0.0, nan)` returns `0.0`. Fixed in both halves: an explicit `ERROR` verdict when the
*reference* output is non-finite (EP-only non-finite remains a genuine `DIVERGENT`), and an
`INPUT_DOMAIN` table so `Sqrt` samples positive and `Div` samples non-zero.

**The price, paid and not softened.** Phi-3.5's claimed count goes **355 → 0**. Morpheus accepted
this explicitly when he ruled §8.9 and it is not negotiable downward. The fall is **temporary**:
the 355 nodes reduce to **8 distinct proof obligations**, and they are mechanically discoverable —
the claim log now carries `proof_key` on every audit line, so
`bench/results/_phi35_keys.txt` was extracted from a single gated run rather than enumerated by
hand. Populating them from existing differential runs is a harness job, not a design one.

**The open backlog item, named so it does not get lost.** The estimator's boundary-bytes number and
the measured one disagreed by 104,116×. That is now two independent defects, one closed and one
open:

- *Closed:* internal island edges were counted as boundary. A whole-graph per-value consumer map in
  `ep.rs` fixed it — 89,199,100,032 B → 13,936,509,056 B, and `net_benefit_sole_island_overrides`
  went **1 → 0**, so Phi-3.5's island is now claimed on the gate's own economics rather than on the
  no-alternative override.
- *Open:* `slot_bytes` substitutes **128 for every unknown dimension**, and every Phi-3.5 boundary
  tensor is `runtime-extent`. Residual ratio ~16,268×. This is a **fabricated** input, not merely an
  over-broad one, and it has a different fix (resolve the extents, or decline to answer). It is now
  self-disclosing: `Island::symbolic_boundary_slots` travels with the number and the sole-island
  WARN reports the fabricated-slot count, so the fabrication cannot be read as a measurement.


### 7.16 The row is open, the ledger is attributed, and the boundary fix is a bound — not an agreement (2026-08-02)

**Criterion 11 was reverted from MET to *not met — scaffolding only*, and not by me.** My write-up
claimed the row; Morpheus's did not, and the coordinator took his. His reason is the one worth
carrying:

> the cheapest satisfaction is a ledger generated from the claim table — derive the ledger from the
> same enumeration that produces the claims and the criterion is true **by construction**,
> `ledger_hits == proven_key_lookups` forever, and the check can never fail. That is an identity
> whose two sides come from the same source, and `6/6` looks identical under both readings.

The shipped ledger is not that shape. It is generated by `gen_proof_ledger.py` from executed
differential runs and baked in with `include_str!`. **But nothing in the artifact distinguished the
two shapes**, which means the good shape was being taken on trust — R11 on my own mechanism, and the
same failure mode as Switch's identical-census-for-both-arms probe: *a perfectly stable, perfectly
wrong answer.* Two of the four discharge conditions were mine.

#### (a) Provenance — the field the claim table cannot produce

Every ledger entry now records `claimed_nodes`, `dispatches_executed` and `worst_rel`. **A dispatch
count only exists after a session executed**; an enumeration over the claim table can produce a key
and a verdict, but it cannot produce a dispatch. That asymmetry is the whole defence.

Enforced on both sides and in the same direction:

| where | behaviour on an unattributed entry |
| --- | --- |
| `gen_proof_ledger.py` `entry_line()` | **raises `SystemExit`** rather than writing it |
| `gen_proof_ledger.py --check` | FAIL |
| `registry.rs::parse_ledger` | **faults** the ledger — the entry grants nothing |

Three details that are the actual content of the check:

- **Absent is treated exactly like zero.** Both mean *this entry does not record a run that ran*. A
  parser that accepted an absent field would have let the claim table write the ledger by omission.
- **A quoted count is not a count** (`"dispatches_executed": "1"`). A writer that stringified its
  counters did not read a counter. `json_u64_field` rejects it.
- **`dispatches_executed: 0` is the 2026-07-30 specimen**: a `MATCH` from a CPU-vs-CPU run, which is
  how the comma-shredded `CLAIM_UNPROVEN` list went undetected until attribution caught it.

Control: `an_entry_without_attribution_proves_nothing_however_well_formed` — **four ledgers differing
only in the attribution fields, four different outcomes** (R10: the falsifier varies with its input,
not with a flag its author set). Mutation-tested red at *“a run that dispatched nothing proves
nothing, whatever it compared.”* Plus `every_shipped_ledger_entry_carries_its_proof_run`, which is
the assertion that goes red if anyone regenerates the ledger with a tool that stopped recording
provenance.

Regenerating with provenance moved the digest **`e4436e93c19c8744` → `331003e0ff88df3f`**; all 9
entries re-attributed `MATCH` at `claimed_nodes=1 dispatches_executed=1`.

#### (b)(iii) A baked digest that disagrees with the disk **refuses**, it does not warn

This is a **second and distinct threat** from the header-vs-body digest that already existed:

| check | catches |
| --- | --- |
| header digest vs body (existing) | a hand-edit **before** the build |
| baked vs `ONNXRUNTIME_EP_VULKAN_LEDGER_FILE` (new) | the file changing **after** it |

The second is the case where *the artifact a reviewer reads is not the artifact the binary claimed
from* — and a WARN would leave the run claiming from evidence nobody can read. So the disagreement
is pushed into `Ledger::faults`, and non-empty faults makes every lookup return `Faulted`: **every
form declines.** A named file that cannot be read is also a refusal — *a ledger that was asked for
and is absent is not an empty ledger.*

**R9 amendment 5, asked honestly:** which way does this check move when its subject is wrong? It
moves **against** the reader's confidence. A mismatch can only remove claims; it can never add one.
That is why it is safe for the variable to be optional — setting it can only make the build stricter
— and it is the reason this check *can* be repaired by tightening, unlike the ones that cannot.

Control: `a_disk_ledger_that_disagrees_with_the_baked_one_refuses_to_claim`, three arms — identical
file → no fault (**this arm is what makes the other a detection rather than a check that fails on
everything**); one line appended → fault naming **both** digests; named-and-absent → fault.
Mutation-tested red.

#### (d) A miss is three findings, not one `false`

`LedgerLookup::{Hit, KeyAbsent, Faulted, NeverAttempted}`, one token each, and
`record_ledger_lookup` now takes the outcome instead of a `bool`. The three misses call for three
different repairs — *regenerate this form*, *fix the ledger file*, *nothing at all* — and a `bool`
spells all three `false`. This is exactly the collapse R12 made this project undo when *bypassed*
and *all-rejected* were sharing one `0`.

Two decisions inside it:

- **`LEDGER-FAULTED` outranks `KEY-ABSENT`** (R13). A run whose ledger failed has **no reading about
  any form** — the key might well be proven and this build cannot tell. Reporting `KEY-ABSENT`, a
  statement *about the form*, would spell an instrument outage exactly like a detection.
- **`NEVER-ATTEMPTED` is derived, never counted.** Recording one would be a lookup, which is
  precisely what it asserts did not happen. It is derived from `proven_key_lookups == 0`.

The counters artifact carries `"ledger_miss"`. Control:
`the_ledger_miss_token_names_which_of_three_things_happened` — four states driven, four tokens
asserted distinct.

#### The boundary fix is a **bound**, not a second opinion

I described the fix as making the economics arm *concur* with the exemption. Morpheus declined that
framing outright:

> agreement between two things fed the same fabricated input is not a second opinion.

He is right and the correction is sharper than the claim it replaced. The verdict flipped because
its input moved 6.4× **while remaining 16,268× wrong** — it flipped for a reason unrelated to the
proposition. What survives is not agreement but an inequality:

1. `transfer_ns` is **monotone non-decreasing in bytes** (asserted across six sizes including
   `u64::MAX/4`).
2. The gate **claims** at the inflated 13,936,509,056 B — not overrides, claims.
3. The measured boundary, 856,720 B, is **smaller**.
4. Therefore the truthful island claims **a fortiori**.

**The claim survives a 16,268× adversarial inflation of the term opposing it.** A number I do not
trust, used in the one direction where not trusting it is safe. This is §10.0.4's third form, after
*prefer the count* and *prefer the ratio*: **prefer the bound you can sign.** Asserted mechanically
by `the_claim_survives_an_adversarial_inflation_of_the_term_opposing_it`, not left as prose.

**The licence is narrow, and this is the part that will be forgotten first.** The sign is *not*
general. `slot_bytes` substitutes 128 for every unknown dimension, which **over-counts on our decode
window (extent 1) and under-counts on a long prefill (extent 4096)**. Under-counting makes the
island look cheaper to move than it is and the gate manufactures claims: **the bound does not
weaken, it evaporates.** Standing falsifier:
`the_substituted_extent_under_counts_on_a_long_prefill_and_the_bound_evaporates`. **If you touch
`slot_bytes` (`ep.rs`) or `symbolic_boundary_slots` (`partition.rs`), that asymmetry is the thing to
preserve.**

#### `Island::MEASURED_PHI35_DEV0` → `ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED`

A constant named `MEASURED` was holding an estimate now known wrong by 6.4×, sitting beside
`MEASURED_PHI35_DEV0_REAL_BYTES`, which holds the actually-measured bytes. Morpheus:

> **names outlive doc comments.**

Keeping both constants was correct — only the name was wrong. Renamed with the two tests that
reference it, and `the_override_carries_the_verdict_it_overrode` gained a note saying in one place
why its `TransferDominated` assertion is consistent with `overrides 1 → 0` shipping (it forces
`anchor_exemption: false` and feeds the *pre-fix* constant; the shipping path uses neither). It took
the coordinator three steps to establish that, and **a reader who stopped at the test name would
have concluded the opposite of what ships.**

### 7.17 What fraction of the model's work runs on the CPU EP — and why the answer is a curve (2026-08-02)

**The question was asked because a node count could not answer it.** The standing reading was
`executed_by = {'CPUExecutionProvider': 120, 'VulkanExecutionProvider': 99}`, and a large count of
small things is not a large thing. `rust/tools/roofline_split.py` replaces it with the two
quantities that are actually spent — **FLOPs** and **bytes moved** — split by execution provider,
and reports both as a **function of context length** rather than as a scalar.

Artifacts: `bench/results/roofline_split-dev{0,1}.json`,
`bench/results/roofline_split-dev0-cf-GroupQueryAttention.json`, prediction in
`bench/results/roofline_split-prediction.md`. **Not one figure in this section is a duration.**

#### (a) The answer

One decode step (`sequence_length = 1`, `past_sequence_length = ctx`), Phi-3.5-mini int4:

| ctx | CPU share of FLOPs | CPU share of bytes | model bytes/step | EP's own estimator says |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00% | **0.07%** | 2.302 GB | 16.58% |
| 128 | 0.67% | 4.26% | 2.403 GB | 16.58% |
| 512 | 2.63% | 14.95% | 2.705 GB | 16.58% |
| 2048 | 9.76% | **41.21%** | 3.913 GB | 16.58% |
| 8192 | 30.20% | **73.77%** | 8.769 GB | 16.58% |

**Neither 3% nor 30%. It is a curve spanning three orders of magnitude over the operating range,
and which end you stand at decides whether the GQA fix is a rounding error or the largest
performance item in the project.** At ctx=0 the declined work is 0.07% of bytes; at ctx=8192 it is
73.77%. The one regime in which the CPU EP looks free is exactly the regime our quotable figures
were taken in.

Identical on both devices (dev0 RTX 4060, dev1 Iris Xe) to every digit. The split is decided by op
type and shape, not by device, so this is one fact and not two.

#### (b) What the fix is worth, as a counterfactual on the same instrument

`--counterfactual GroupQueryAttention` moves the 32 declined nodes to the Vulkan side and re-runs
the same arithmetic against the same graph. Claimed goes 323 → **355** — the 355-node island we
already quote.

| ctx | CPU bytes, today | CPU bytes, GQA claimed |
| ---: | ---: | ---: |
| 0 | 0.07% | 0.03% |
| 2048 | 41.21% | 0.02% |
| 8192 | 73.77% | 0.29% |

**So the GQA fix is not a correctness fix with a rounding-error performance effect.** It is worth
**41.2 points of byte traffic at ctx=2048 and 73.5 points at ctx=8192**, and approximately nothing
at ctx=0. It also collapses **33 fused islands into 1** — see (d).

The residual 0.29% at ctx=8192 is not noise: it is the `If` at
`/model/rotemb_caches_subgraph/If` materialising the **131072 × 48** rotary cache (25.2 MB) instead
of the 4096-row one. That is a step function at `total_sequence_length > 4096`, read out of the
graph's own `Greater` predicate, and it is the only remaining context-dependent CPU cost once GQA
lands.

#### (c) The EP's own estimator cannot answer this question, and the artifact shows why

The last column of (a) is the split computed with `ep.rs`'s model reproduced exactly. **It is
16.58% at every context and does not move at all.** 16.58% is `32 / 193` — the declined anchors
over the total anchors. The estimator scores every anchor with the constant `2 * 3072 * 3072` and
everything else with `out_bytes / 2` under a substituted dim, so **its FLOP number is the anchor
count wearing a FLOP's clothes**, and it is blind to the one axis on which this model's cost
actually varies.

That is not a call to tighten it. Per R9 amendment 5, a verdict that moves with a constant nobody
measured is a fabricated input, not an over-broad one. The estimator is adequate for the job it has
(a coarse gate at `GetCapability`, where no shape inference is available) and inadequate for this
one, and the honest response is to use a different instrument, which is what this is.

#### (d) Reconciliation — three node counts that do not match, reported rather than smoothed

```
graph 366 = offered 363 + never-offered 3 {Constant: 3}
offered 363 = claimed 323 + declined 40
declined 40 = GroupQueryAttention 32 + the eight permanent declines
profile   = {VulkanExecutionProvider: 33, CPUExecutionProvider: 40}
```

- The **3 `Constant` nodes are never offered to `GetCapability`** — ORT folds them first. Per R12
  they are absent from the frame, not declined in it; counting them as declines would be a zero
  where `UNOBSERVABLE` belongs.
- **CPU profile events (40) match the declined set 1:1.** The eight are exactly the eight of §7.13,
  all inside `attn_mask_reformat` and `rotemb_caches_subgraph`.
- **The 33 Vulkan profile events are fused partitions, not nodes.** 323 claimed nodes in 33 islands:
  the 32 GQA declines cut the per-layer chain 32 times. **The island count is a consequence of the
  GQA decline, not an independent problem** — which is why the fragmentation lever measured 0.0376%
  on its own and why it should not be worked separately.
- **The standing `{CPU: 120, Vulkan: 99}` reading matches none of these numbers.** It sums to 219,
  not 366, not 363, not 73. I am not able to reproduce it from this build and I am not going to
  reconcile it by guessing; it should be re-taken or withdrawn.

#### (e) The denominator does not come from us (R11)

The whole model is enumerated straight from the ONNX file — all 366 nodes, their shapes, their
initializers' real stored byte lengths from `external_data[].length`. The **only** thing taken from
the EP is which nodes it claimed, read from the `CLAIM_LOG`. Two sources: the graph says how much
work exists, the EP says which part of it it took. Had the split been re-derived from
`registry.rs`'s predicates, both sides would come from one source and the identity could not fail.

Two independent checks that the totals are not self-consistent nonsense:
`MatMulNBits` moves **2.097 GB** of weights per step, which is the int4 file's own size; and its
**7.445 GFLOP** at one row implies 3.72 G weights, against a 3.8 B-parameter model.

#### (f) Fabricated-extent disclosure — and an instrument error I made getting here

**Zero.** Every extent resolves from the graph once a context length is stated:
`make_dim_param_fixed` on the four dim params plus ORT's symbolic shape inference leaves nothing
open. The EP's `128`-for-unknown-dim substitution is a property of the estimator at
`GetCapability`, where no shape inference is available — **it is not inherited by this
measurement.** So the answer in (a) is not `UNOBSERVABLE`; it is measured.

The one extent symbolic inference genuinely could not fold is `cos_cache` / `sin_cache`, left as
`[None, 48]` because they are produced by an `If`. That extent is **conditional, not unknown**: the
predicate is `total_sequence_length > 4096` and the two branches are Constants of `[4096, 48]` and
`[131072, 48]`. Reading it out of the graph is the difference between a resolved extent and a
fabricated one, and it is exactly the move `slot_bytes` does not make.

**R13, on myself.** The first run of this tool printed
`VERDICT: UNOBSERVABLE(fabricated extents carry 73.69% of the bytes)`. That was **ERROR(instrument),
not a detection**: I was flagging fabrication per *node* rather than per *tensor*, so one unresolved
48-wide operand condemned the whole 50 MB GQA node. The suspicious part was that the fabricated
fraction equalled the CPU fraction to two decimals at every context — an identity that had no reason
to hold. Fixed at the tensor level; the figure is now 0.00% at every context. **The number I would
have shipped was 73.69 percentage points wrong and it carried the more alarming verdict.**

#### (g) Predictions, scored

Written to `bench/results/roofline_split-prediction.md` before the tool ran.

| # | Prediction | Outcome |
| --- | --- | --- |
| 1 | monotone climb with ctx, near-flat FLOPs at ctx=0 | **held** |
| 2 | CPU bytes under 5% at ctx=0 | **held** (0.07%) |
| 3 | CPU bytes over 40% at ctx=8192 | **held, but I under-predicted** — 73.77% |
| 4 | the node count (32.8%) predicts neither end | **held** — byte share crosses it between ctx 512 and 2048 |
| 5 | the EP estimator is flat in ctx | **held** — 16.58% at every ctx, and it is `32/193` exactly |
| 6 | fabricated extents contribute nothing | **held, after I fixed my own instrument** — see (f) |

#### (h) Disagreement with the 60.5% KV figure, left standing

Switch's KV-traffic figure at ctx=8192 is **60.5%** of bytes; the GQA share here is **73.5%**. The
two are not the same quantity — mine charges GQA for Q, the output and the rotary indexing as well
as the cache — but that does not obviously account for 13 points. **Two independent estimates that
disagree are worth more than one that has been reconciled**, so both are recorded and neither is
adjusted. Anyone quoting either must state ctx alongside it, the way a timing states its device
state.

#### (i) The rule this establishes

**Any figure quoted against the roofline states its context length, or it is not a figure.** The
0.07% and the 73.77% are the same system, the same build and the same claim set. A reader given
either one alone would size Switch's work wrong by three orders of magnitude.


### 7.18 Which kernel produced the reading — `GEMV_PACKED` becomes observable, and a second ABI insertion falls out (2026-08-02)

Link's independent-whole work found twelve instrumented Rust surfaces no census mechanism
observes, and named this one first because it is different in kind: **`ONNXRUNTIME_EP_VULKAN_GEMV_PACKED`
selects a different kernel.** Every kernel measurement this project holds — the amplification
result, the packed-loads work, the `q_gemv` figures — is a reading of *a* kernel, and the artifact
could not say which. **A reading whose subject is unidentified is not a reading of that subject.**

Artifacts: `bench/results/gemv_kernel_identity-dev{0,1}.json`,
`bench/results/counters_abi_drift.json`. Tools: `rust/tools/probe_gemv_kernel_identity.py`,
`rust/tools/counters_abi.py`. **No timing figure.**

#### (a) Why the obvious closure is wrong

The switch does not change the shader stem. It moves **specialization constant 5** of
`q_gemv.comp`, and `vk/pipeline.rs` keys the pipeline cache on `(shader_stem, spec_constants)` — so
the two settings are two different pipelines wearing one stem. `shaders_dispatched`, the field this
project already had, reports `["q_gemv_matmul_nbits_f32"]` in **both** worlds.

And a host-side record of `GEMV_PACKED=1` would not close it either. That says what was *asked
for*, which is the distinction Trinity drew when `DEVICE=0` ran on `1=NVIDIA`: **the selector is a
request, not an identity.**

#### (b) The emission

`counters.rs` gains two **JSON-only** fields — no `abi_version` bump, no C-struct change, for the
reason (d) makes concrete:

- **`pipeline_variants`** — `["{stem}:{c0},{c1},…"]`, the distinct `(shader, spec_constants)` pairs
  the run actually built. Recorded from `vk/session.rs` at the `PipelineKey` construction site,
  from `eff_shader` and `eff_spec_constants` — the **effective** pair handed to `get_or_create`,
  after any substitution.
- **`gemv_packed_spec_constant`** — a **string**, five states:
  `"1"` / `"0"` / `"MIXED"` / `"UNRECORDED"` / **`"UNOBSERVABLE"`**. The last is the important one:
  no GEMV pipeline was built, so the constant was never resolved, and R12 forbids spelling that
  `0`. The census graph is a six-node elementwise chain with no `MatMulNBits`, so `UNOBSERVABLE` is
  the *common* case, and a `0` there would have quietly told the census that the unpacked kernel
  ran. A string rather than an int for the same reason `net_benefit_override_reason` is one: a
  reader who does arithmetic on it fails loudly.

#### (c) The falsifier — five arms, all predicted before the first run

| arm | env | graph | predicted | observed | `pipeline_variants` |
| --- | --- | --- | --- | --- | --- |
| A1 | unset | 4-bit / block 32 | `1` | **`1`** | `q_gemv_matmul_nbits_f32:32,4,32,0,8,1` |
| A2 | `=0` | 4-bit / block 32 | `0` | **`0`** | `q_gemv_matmul_nbits_f32:32,4,32,0,8,0` |
| A3 | `=1` | 4-bit / block 32 | `1` | **`1`** | `q_gemv_matmul_nbits_f32:32,4,32,0,8,1` |
| B | **unset** | 4-bit / **block 16** | `0` | **`0`** | `q_gemv_matmul_nbits_f32:32,4,16,0,8,0` |
| C | `=1` | elementwise `Add` | `UNOBSERVABLE` | **`UNOBSERVABLE`** | `ew_binary_add_f32:256,1` |

All five held on both devices. **Arm B is the one that makes this R10 evidence rather than an env
echo:** the environment is untouched in both A1 and B, and the recorded token still moves, because
the packed path is off *by shape* at an 8-byte blob. A field that read the env var would print the
same token for both.

Arm C is the R12 arm, and arm A3 is the control that would catch a field that merely echoed the
switch's presence.

**`shaders_dispatched` is byte-identical across A1 and A2.** That is the finding stated as a
disagreement between two instruments on the same run: the old field cannot tell the two kernels
apart, the new one can.

#### (d) What fell out — `a52024f` has happened again, and it is not mine

Adding a counter meant re-running the ABI guard, and it is red:

```
the DLL's VulkanEpCounters is 136 bytes, this mirror is 112.
A mirror that is the wrong size does not read smaller numbers, it reads different fields.
```

`898a2ba` inserted `outputs_device_resident`, `outputs_host_resident` and `outputs_device_bound`
**between `device_losses` and `dispatches_executed`** — mid-struct, same place, three fields where
`a52024f` inserted one. All three hand-maintained ctypes mirrors kept the old layout.

`rust/tools/counters_abi.py --compare` names what each stale field actually reads, which a size
mismatch does not:

```
tests/ops/test_wiring_census.py [mirror 0]: 15 fields, 112 bytes (struct is 18 fields, 136 bytes)
    dispatches_executed        reads  outputs_device_resident
    viable_islands_retained    reads  outputs_host_resident
    proven_key_lookups         reads  outputs_device_bound
    ledger_hits                reads  dispatches_executed
    unproven_declines          reads  viable_islands_retained
    ledger_entries             reads  proven_key_lookups
    unproven_forms_claimed     reads  ledger_hits
tests/ops/test_phi35.py [mirror 0]: dispatches_executed reads outputs_device_resident
tests/ops/test_phi35.py [mirror 1]: dispatches_executed reads outputs_device_resident
```

**One defect, three red tests.** `test_the_counters_mirror_matches_the_running_dll`,
`test_ledger_lookup_wired` and `test_wiring_census` are all the same shift: on the same DLL in the
same process, `ledger_entries` reads **0** through the stale mirror and **97** through a correct
one, which is why the ledger lane reports "published but nothing consulted".

And again the misread is the *plausible* one — `dispatches_executed` lands on
`outputs_device_resident`, which is 0 on any run whose outputs are host-resident. **Stable,
plausible, invisible.** That is the third time this exact signature has appeared.

**This change is not the cause and could not have been.** The two new fields are JSON-only; the
diff adds no `pub …: u64` line; the three fields arrived in `main` via `898a2ba`. The guard that
caught it is the `struct_size` **equality** check filed at `a52024f` — which is the one thing that
worked here, and it worked because equality was chosen over `<=`: an append is safe to read short,
an insertion is not, and the size alone cannot tell them apart.

#### (e) The real fix, built but not yet installed

The filed-not-done item was *three hand-maintained mirrors of one ABI*. `rust/tools/counters_abi.py`
removes the thing that drifts: it **parses the field list out of `counters.rs`** and builds the
`ctypes.Structure` from it, so there is no second list to forget. Verified against the running DLL:
18 fields, 136 bytes, exact size match, offsets printed.

It is not yet wired into the three call sites, which live in Trinity's `tests/`. The replacement is
one line each (`_fields_ = …` → `counters_abi.make_mirror()`); routed to her rather than done here
because four agents are live in this tree. Recorded in
`.squad/decisions/inbox/mouse-gemv-kernel-identity.md`.

**One instrument error of my own, disclosed.** The first `--compare` run exited
`STATUS_ACCESS_VIOLATION`. `OrtEpVulkanGetExecutionCounters(out, out_bytes)` takes a length and I
passed only the pointer, so `fill` clamped against whatever was in the register. That is
ERROR(instrument) in my probe, not a defect in the export — the EP clamps correctly, which is
exactly why a short mirror gets a truncated *prefix* and the positional misattribution above is the
right reading.

#### (f) Does this generalise to the other eleven?

**To the three counters, no; to kernel identity, much further than one switch.** `pipeline_variants`
records *every* pipeline the run built, so any future kernel reading can name its own subject
without a new field — which is the durable part. But the remaining env switches do not enter as
specialization constants, so each needs its own EP-side observable and none of them gets one for
free here. They stay with Link.

### 7.19 The mirrors are gone and the layout is compiler-checked (2026-08-02)

Artifacts: `rust/tools/counters_abi.py`, `tests/ops/test_counters_abi_singleton.py`,
`bench/results/phi35_claim_reading_summary.json`. **No timing figure.**

#### (a) The generator did not stop the second occurrence, and the reason is structural

§7.18 filed `counters_abi.py` as "the real fix, built but not yet installed", and routed the three
call sites elsewhere because four agents were live in the tree. That reasoning was sound about
concurrency and wrong about safety: **a generator that co-exists with the thing it replaces is a
fourth mirror.** Between `a52024f` and `898a2ba` the tool existed and the defect recurred anyway,
because the tests still read through the hand mirrors. Nothing in the tree made writing a fourth one
harder than writing the first three.

Verified count, rather than the reported one: **exactly three** offset-keyed mirrors —
`tests/ops/test_phi35.py` (two) and `tests/ops/test_wiring_census.py` (one). The JSON emission and
`snapshot()` are name-keyed and compiler-exhaustive, so they are not mirrors and were left alone.
All three are deleted; every call site reads `counters_abi.read_counters()` or
`counters_abi.counters_from_dll()`.

`tests/ops/test_counters_abi_singleton.py` fails if any file outside `counters.rs` and
`counters_abi.py` declares two or more counter field names inside a `_fields_` block. Two, not one:
a file that *reads* `dispatches_executed` is a consumer, and consumers are the point of having a
generator; it is the ordered **list** that carries layout. The lane carries both controls — a
planted mirror it must name, and a consumer it must not — because a detector that has never been
seen red is indistinguishable from one that cannot go red.

#### (b) The version discipline is computed, not remembered

A version bump is itself hand-maintained, so it fails the same way the mirrors did — `898a2ba` is
the proof, leaving `COUNTERS_ABI_VERSION` at 4 so that one number named two layouts.

`counters.rs` now declares the field list **once**, inside `counters_abi_struct!`, which emits both
`VulkanEpCounters` and `COUNTERS_LAYOUT: &[CounterField]` with offsets from `std::mem::offset_of!`.
`COUNTERS_LAYOUT_HASH` is const-evaluated from that (FNV-1a/64 over `name:offset:size;`), and a
`const _: () = assert!(…)` fails the **build** unless `(COUNTERS_ABI_VERSION, COUNTERS_LAYOUT_HASH)`
appears in `COUNTERS_LAYOUT_REGISTRY`. A compile-time assertion rather than a test, because a test
can be filtered out by the person inserting the field.

The DLL also exports `OrtEpVulkanGetCountersLayout`, a per-field offset manifest — the item my
2026-08-02 note asked for. A size check says *that* two layouts differ; only the manifest says
*how*, and `dispatches_executed reads outputs_device_resident` is the sentence a reader of a red
lane needs. `counters_abi.py` recomputes the same hash from the parsed source, so Python's `repr(C)`
model and rustc's `offset_of!` cross-check each other; if they ever disagree the struct has padding
and every ctypes reading in the tree is wrong.

**Acceptance, run and not reasoned about.** The exact `898a2ba` edit — three `u64` fields between
`device_losses` and `dispatches_executed`, version untouched — applied to `counters.rs`:

```
error[E0080]: evaluation panicked: VulkanEpCounters changed layout and COUNTERS_LAYOUT_REGISTRY
does not know about it. The compiler computed COUNTERS_LAYOUT_HASH from the field offsets; it does
not match the row for COUNTERS_ABI_VERSION. Run `python rust/tools/counters_abi.py` …
   --> src\counters.rs:709:5
```

and the tool, run as instructed, prints the repair:

```
FAIL(layout undeclared): VulkanEpCounters has layout hash 0x9ffcf374a1ca0e2b and
COUNTERS_LAYOUT_REGISTRY has no such row under COUNTERS_ABI_VERSION=5.
  Repair: bump COUNTERS_ABI_VERSION to 6 and append
      (6, 0x9ffcf374a1ca0e2b),
```

Reverted afterwards. Note what the build error also shows: `E0063 missing fields … in initializer`.
The struct is exhaustively initialised in five places, so *appends* were already compiler-checked;
what was never checked was the **meaning** of a version number, and that is what the registry adds.

**And then it fired again, unprompted, on work that was not about layout.** Renaming
`device_mismatch_declines` → `proven_elsewhere_declines` for §8.9.18's vocabulary changed no offset
and no size, and the build failed anyway: the hash covers `name:offset:size`, so v7
(`0x16eacc53e6e18d97`) differs from v6 (`0xf3fac68aa2c3a3ef`) at an identical 152 bytes and 20
fields. That is the correct behaviour and worth stating plainly — **two builds that disagree about
what a field is *called* disagree about what its number means**, and a ctypes reader keyed on names
would silently read `0` for the renamed field exactly as `ledger_entries` did under `898a2ba`. A
mechanism built for insertions caught a rename on its first unscripted outing.

### 7.20 `LedgerEntry.device` is load-bearing; `PROVEN-ELSEWHERE` discloses and does not promote (2026-08-02)

Artifacts: `rust/src/registry.rs` (`running_device_names`, `is_selector_ordinal`, `device_state`,
`ProofState`), `rust/src/disclosure.rs` (`FormEvidence`), `bench/results/phi35_claim_reading.json`.
**No timing figure.**

#### (a) What the ruling settled (§8.9.18 Part 1)

§7.19 originally reported `PROVEN-ELSEWHERE` as implemented, with a promotion licence. Fact
Checker's audit of the R12 rule declines returned ❌ on *"model-level ULP evidence cannot promote
unexercised per-form keys"*, and **Morpheus upheld the refutation and withdrew his own paragraph in
place** (§8.9.18). The cost argument was *the expensive proof establishes the form; the cheap
invariant establishes the port*; the cheap invariant is the **model-level** ULP series, whose
records are indexed by model output, and **there is no map from an output ordinal to a proof key**.
His own 12-ULP logits head-step was unattributable for exactly that reason.

The arithmetic is the part worth keeping: `wiring_census-dev1.json` reads `proven_key_lookups = 6`
against `ledger_entries = 95`. **One clean ULP curve would have promoted 89 keys that nothing ever
touched.** A promotion rule whose evidence is 6/95 exercised is not cheap, it is absent.

So the status **keeps disclosure and loses promotion**. The fatal-horn leg — *a run on a device with
no matching entry must say so, by name, in its own artifact* — is untouched and is what the code
implements: `ProofState::ProvenElsewhere` names the device the entry was obtained on, counts itself,
and **declines**.

#### (b) The precondition: the device field now decides something

`"device": "device0"` sat on 103 of 103 entries and **no predicate read it**. It does now.

| `ProofState` | condition | claimable | counted as |
|---|---|---|---|
| `PROVEN` | entry `MATCH`, witnesses present, `device` **names hardware this run opened** | yes | — |
| `DEVICE-UNATTRIBUTED` | entry sound, but its `device` is a **selector ordinal** (`device0`), or this run has opened no device | yes | `device_unattributed_claims`, `device_unattributed_forms`, `claimed_forms_device_unattributed` |
| `PROVEN-ELSEWHERE` | entry sound, obtained on a **named** device this run did not open | **no** | `proven_elsewhere_declines`, `proven_elsewhere_forms` |
| `UNPROVEN` | no entry, or a demoted one | no | `unproven_declines` |

Two design points, both of which are the difference between a predicate and a field:

* **The key is the device name read off the run, never the selector.** `running_device_names()`
  parses `allocator::tally::session_devices()` — what the EP *opened*. `is_selector_ordinal()`
  recognises `device` + digits and refuses to treat it as an identity. This is Morpheus's separate
  finding, and this session reproduced it live: with `ONNXRUNTIME_EP_VULKAN_DEVICE=0` the probe run
  reported `alloc_device_frame_session_devices: "1=NVIDIA GeForce RTX 4060 Laptop GPU"`. **The
  selector said 0 and the hardware was ordinal 1.** An entry stamped `device0` names neither.
* **`DEVICE-UNATTRIBUTED` still claims.** Declining it would take the EP from 355 claimed nodes to
  zero over a *frame* question — a bookkeeping defect used to withdraw working kernels. It is
  instead counted on every claim and named per form with `entry-device=` and `running-device=` in
  the session disclosure and the counters file. That is the answer to Fact Checker's question about
  `PROVEN-ELSEWHERE` — *what stops it becoming the default nobody looks at?* — applied to the state
  that actually exists today: it appears in every artifact of every run even while it changes no
  outcome.

Measured, device 0, this build, ledger `b0b06b464134fc33` / 103 entries:
`claimed_nodes` 355, `ledger_hits` 355, `unproven_declines` 3 — **identical to the pre-change
reading**. `device_unattributed_claims` 355, `claimed_forms_device_unattributed` 8,
`proven_elsewhere_declines` 0, `running_device_names` `NVIDIA GeForce RTX 4060 Laptop GPU`.
`claimed_form_evidence` moved `ALL-PROVEN` → `DEVICE-UNATTRIBUTED-PRESENT`. **That is a fix, not a
new defect**: no node changed hands, and `ALL-PROVEN` was an over-claim — it asserted a device frame
nothing had checked. (`ledger_entries` 97 → 103 is the `main` merge, not this change.)

`gen_proof_ledger.py` now writes the physical name into `device` when the proof run reports one, and
keeps the ordinal in a separate `device_selector`. The 103 baked entries predate that and are
therefore all `DEVICE-UNATTRIBUTED` until re-proven; they were **not** regenerated here.

#### (c) The four obligations, discharged against the ruling as issued

R12's obligations, and where each stands now that promotion is off the table:

1. **Claimable on purpose, never as a fallback.** *Discharged.* A missing key reaches
   `ProofState::Unproven` before any device comparison is made, so the status is unreachable by
   absence — it requires a **sound entry naming a device this run did not open**. Absence and
   elsewhere are different states by construction, not by convention.
2. **Counted.** *Discharged.* `proven_elsewhere_declines` (per claim) and `proven_elsewhere_forms`
   (per distinct form) are in the C ABI, the counters JSON and `epctl`.
3. **Disclosed with the device it was proven on.** *Discharged, conditionally on (b).* The
   disclosure prints `entry-device=` and `running-device=` by name. It could not have been
   discharged before the device field was load-bearing: until entries carry names rather than
   ordinals, "proven elsewhere" would name `device0`, which is not a device. Morpheus calls the
   ordering forced rather than tidy, and he is right — while `device0` is a selector, "proven on
   another device" is **undefined, not under-implemented**.
4. **A predicate must read it.** *Discharged in the disclosure direction, and that is now the only
   direction there is.* Under the withdrawn version this was the unsatisfiable condition: promotion
   needed a check that could come out negative, and no per-form evidence on the second device
   existed to fail. With promotion withdrawn, the predicate that must read the status is the
   **declining** one — `device_state()` returns it, the claim gate declines on it, and the §8.9.7
   disclosure prints it. Its positive state is exercised by
   `registry::tests::a_proof_is_a_property_of_a_form_on_a_device`, which plants an entry naming a
   device the run did not open and asserts the decline. **The status can no longer become the field
   nobody consults**, because the only thing it does is refuse.

The honest residue: `proven_elsewhere_declines` reads **0** on every real run today, because all 103
baked entries are ordinals and therefore `DEVICE-UNATTRIBUTED`, not elsewhere. That is a property of
the ledger, not of the predicate, and it is why the planted-entry test — not the probe — is the
control.

#### (d) Per-key replay: recorded, not commissioned — and I judge it right

The replacement Morpheus **recorded and explicitly did not commission** is per-key replay of the
entry's own stored case: `artifact`, `tolerance`, `shaders`, `shader_digest`. I judge it right, and
the reason is structural rather than deferential: it is **per-key by construction, so it cannot
promote anything it did not exercise** — which is precisely the failure mode that killed the ULP
route. Read off the ledger, the premise that per-device proof is too expensive to repeat also looks
false: every entry points at a per-form case model under `evidence/cases/*.onnx`, each a tiny graph,
and `gen_proof_ledger.py --reprove` re-measures a form against exactly that artifact. The
second-device cost is not "re-run Phi-3.5's proof programme"; it is **replay 103 recorded case
models**, producing real `PROVEN` entries with their own witnesses.

If that holds, the sound shape is not a promotion rule but a new *key*: an entry per
`(key, device)` rather than per `key`. `PROVEN-ELSEWHERE` then stays exactly what §8.9.18 leaves it
as — a disclosure state that declines and tells you which device to go replay on.

**I have not run that replay on a second device, so this remains a proposal and not a result.** What
is verified is that the artifacts, their digests and the `--reprove` path exist; what is unverified
is `--reprove` end-to-end on a second adapter. Stating it the other way round would be the mistake
this whole section exists to prevent.

#### (e) Fault scope follows localisation, not severity (§8.9.18 Part 3)

`parse_ledger`'s comment promised *"a stale entry demotes ITSELF and nothing else"* while it pushed
the message onto `Ledger::faults`, which `Ledger::get` consults for **every** key. One shader edit
therefore disarmed the whole artifact. Morpheus ruled the comment right and gave the principle:
**fault scope is set by the scope of what you cannot locate, not by the severity of what you
found.** The line is now drawn where the localisation is:

| damage | locatable? | scope | list |
|---|---|---|---|
| `STALE-SHADER` — recomputed digest ≠ recorded | yes, this key | demote the entry | `entry_faults` |
| `NO-SUBJECT-WITNESS` — empty shader set or missing digest | yes, this key | demote the entry | `entry_faults` |
| absent or zero `claimed_nodes`/`dispatches_executed` | yes, this key | demote the entry | `entry_faults` |
| non-`MATCH` verdict | yes, this key | demote the entry | `entry_faults` |
| header digest ≠ recomputed body digest | no — any line may be affected | fault the artifact | `faults` |
| `declared_count` ≠ parsed entries | no — an entry may have been dropped | fault the artifact | `faults` |
| duplicate key | no — neither is authoritative | fault the artifact | `faults` |
| a line that does not parse | no — you cannot read what it meant to say | fault the artifact | `faults` |

The decisive case is the one that has not happened yet: **`TOOLCHAIN-CHANGED` is ledger-wide by
nature.** A `glslc` bump changes every module's bytes at once, so under the old scope every routine
compiler upgrade was a 103-entry total fault for a change that touched no kernel — a fail-safe with
a scheduled date for being switched off.

The header-count check now compares against `entries + entry_faults`. Without that the demotion
re-creates the global fault through the back door, and it is the kind of second path that is easy to
leave open because the first one looks closed.

**Both obligations §8.9.18 attaches, discharged.**

1. *Demotions must be printed.* `disclosure::disclose_ledger_demotions()` is called on **every**
   session-creation disclosure, before the claim set is known and on the zero-claim path too,
   because the count is a property of the artifact and not of what this model touches. Zero demoted
   is an INFO stating `103/103 entries live` — the negative polarity still speaks, so "no demotions"
   is distinguishable from "no disclosure". Non-zero is a **WARN** quoting every reason verbatim and
   naming `--reprove`. `ledger_entry_faults` also appears in the counters JSON.
2. *The demotion count must not be zero-by-construction.* The baked ledger demotes nothing, so a
   test that only called this on it would assert `0` forever.
   `disclosure::tests::the_demotion_count_is_printed_and_is_not_zero_by_construction` plants two
   ledgers differing only in one `shader_digest`, asserts `(2, 0)` and `(1, 1)`, and asserts the two
   readings differ. `unlocatable_damage_still_faults_the_whole_artifact` holds the other side, so
   the correction cannot quietly become "nothing faults the ledger any more". The digest-drift
   census lane is unchanged and still reads `ledger_faults: 1`, `ledger_gate: FAULTED`,
   `ledger_hits: 0` — it exercises the *artifact* path, which the ruling leaves exactly where it
   was.

### 7.21 One key, two digests: a frame mismatch is no longer a key absence (§8.9.19, 2026-08-03)

**Supersedes the claimability column of §7.20(b) and the "declines" half of §7.20(c).**
`PROVEN-ELSEWHERE` **claims and discloses** as of §8.9.19; it still does not *promote*, which is the
only thing §8.9.18 withdrew. The rows below replace the table in §7.20(b).

#### (a) The defect was one `continue`

`parse_ledger` compared the entry's `shader_digest` against this build's and, on a mismatch,
`continue`d — so the entry never entered `Ledger::entries` and `Ledger::get` returned the **same
`None`** it returns for a form nobody ever proved. A frame mismatch and a key absence were the same
observation with two different repairs, and only one of them actionable.

That is the whole Linux symptom. Ubuntu ships shaderc 2023.8, the Windows SDK ships v2026.2, the
SPIR-V differs for byte-identical GLSL, and every one of the 103 entries silently ceased to exist.
Link proved the cause by perturbing one GLSL template *on Windows* and getting a superset of the
same failing test names.

#### (b) The schema, which is the generating rule for everything else

* **KEY** = the form. `ProofKey::from_node`, nothing else, ever.
* **SUBJECT** = what code was proven.
* **FRAME** = device, driver, `ort_build`, toolchain, tolerance.

Look up by key; compare frame **after**; a subject mismatch means the proof is about something
else. The device and the toolchain belong to the **entry and the predicate, not to the key** —
Morpheus correcting his own §8.9.17 wording.

#### (c) Two digests, because no single hash can be sensitive to the kernel and blind to the compiler

`shader_digest` hashes SPIR-V, which is a function of the compiler as much as of the kernel.
`source_digest` hashes the tree: the variant row, the resolved `#include` closure, the `glslc` argv
minus the version, and the source body. Their **disagreement** is the instrument.

| `shader_digest` | `source_digest` | `SubjectVerdict` | claimable | counted as |
|---|---|---|---|---|
| same | same | `IDENTICAL` | yes | — |
| **differs** | **same** | `TOOLCHAIN-DELTA` → `PROVEN-ELSEWHERE{toolchain}` | **yes, disclosed** | `proven_elsewhere_claims`, `proven_elsewhere_forms` |
| differs | differs | `SUBJECT-CHANGED` → `UNPROVEN{SUBJECT-CHANGED}` | no | `subject_changed_declines`, `subject_changed_forms` |
| same | differs | `SOURCE-COSMETIC` | yes, **named** | `source_cosmetic_forms` |
| differs | *absent* | `INDETERMINATE` → `UNPROVEN{SUBJECT-CHANGED}` | no | `subject_changed_declines` |

The fifth row is **not in the ruling** and is deliberate. Every entry written before §8.9.19 records
no `source_digest`, so on a second toolchain the pair cannot be evaluated. Guessing `TOOLCHAIN-DELTA`
there would grant every legacy entry a claim on a possibly-rewritten kernel; the fail-safe reading is
the other one, and the decline names `--backfill-frame` as the repair.

The pair is **jointly blind to a compiler bug** — identical source, different SPIR-V, and the
difference is wrong rather than merely different. That is exactly why row 2 is *disclosed* rather
than silently promoted.

`source_digest` is **platform-independent by construction**, which is the load-bearing property: both
platforms compute it from the same tree. The `-I` path is hashed as the literal placeholder
`-I<include>` and the `-o`/source paths are omitted, because the raw argv holds absolute paths that
differ between two checkouts of one tree — hashing them would turn the digest into a machine
fingerprint, the exact failure the ruling forbids.

#### (d) Frame and subject are different axes and a single token can carry only one

`ProofState` is single-valued and the frame verdict outranks a cosmetic subject move. Every entry in
today's ledger is `DEVICE-UNATTRIBUTED`, so a `source_cosmetic` count taken off the state alone could
only ever read **zero** — a counter whose only observable value is zero is not an instrument. The
subject verdict is therefore read off the entry and counted and printed **beside** the state, not
instead of it. This was found by running the acceptance rather than reasoning about it (see (f)).

#### (e) Entry survival, and the second route back

Only a genuinely **absent** subject — a shader set this build has no modules for — still deletes an
entry into `entry_faults`. A digest mismatch produces a live entry carrying a `SubjectVerdict`, so
`Ledger::get` returning `None` again means one thing only.

Moving that population out of `entry_faults` opened a second route to the state the ruling closed:
`parse_ledger` cross-checks the header's `entry_count` against what it parsed, and a stale count
would have faulted the **whole file**, re-creating the global decline through a different door.
`registry::tests::a_surviving_subject_mismatch_does_not_trip_the_declared_count` holds it.
`Ledger::demotion_count()` (entry faults **plus** subject-changed entries) is what the §8.9.18
disclosure now reads, for the same reason: sourcing it from `entry_faults.len()` would have made the
obligation read zero the day entry survival landed.

#### (f) The acceptance, run rather than reasoned about

All on this machine, which has exactly one Vulkan SDK:

* **Comment-only edit to `ew_binary.comp`** → SPIR-V byte-identical, `source_digest` moved, the
  affected entries read `SOURCE-COSMETIC` on the subject axis, all still claim, `cargo test --lib`
  green. Row 4 in its positive state, which is the row that proves the pair is being consulted at
  all. The *state* stayed `DEVICE-UNATTRIBUTED` throughout — which is how (d) was found.
* **Comment-only edit to `shaders/include/indexing.glsl`** (a *transitive* include) →
  `ew_binary_add_f32` source digest `c96284dee813bf70` → `c7edca19d8bd7644`, SPIR-V
  `c9f5eddc2471b772` unchanged. The include closure demonstrated in its positive state.
* **Code edit to `ew_binary.comp`** → both digests moved, 28 entries read
  `UNPROVEN{SUBJECT-CHANGED}`, **`entries=103, faults=0`**. Before this change those 28 entries
  would have been *deleted*. Row 3, and entry survival, in one run.
* **Row 2 is modelled, and says so.** One SDK is installed, so the second compiler cannot be
  produced locally. `the_digest_pair_separates_a_second_compiler_from_a_second_kernel` plants the
  artifact Link's lane presents — a SPIR-V digest this build did not produce beside a source digest
  read *out of* this build — and asserts the claim is granted with δ=`toolchain`. It also asserts
  the four rows do not agree with each other, without which every arm could be reading a constant.

#### (g) The ledger carries a frame now

`gen_proof_ledger.py` records `source_digest` and `toolchain` on every new entry and **refuses** to
write one without them. `--backfill-frame` stamps them onto pre-§8.9.19 entries, and the refusal
condition is the design: an entry is backfilled **only** when its recorded `shader_digest` equals
what this build hashes its shader set to right now. That equality is the evidence — the SPIR-V is
byte-identical, so the source that produced it is this source, and the stamp records a fact rather
than assuming one. Entries whose SPIR-V has moved are skipped and named; the repair for those is
`--reprove`, which measures.

All 103 baked entries backfilled with **zero skips**, which is itself a finding: the shipped ledger
is subject-consistent with the shipped binary. `--check` now fails on a frameless entry from the
Python side and `every_baked_entry_records_a_frame_that_can_be_compared` from the Rust side.

The digests are read **out of the artifact**, never re-derived in Python: `OrtEpVulkanGetShaderSubject`
answers `(toolchain, spirv_digest, source_digest)` for an arbitrary stem list. A Python
re-implementation of the hashing rule would have been a fourth mirror of it, and the mirror that
agrees the day it is written is the one that silently disagrees three weeks later.

#### (h) What is still open

**Runtime-chosen specialisation values sit outside both digests.** Morpheus named this and
explicitly did not fix it. The *variant row* — the `-D` defines the build bakes in — is hashed; a
spec-constant value chosen at dispatch time is not, so two runs that select different specialisation
constants for the same stem have identical digests and different pipelines. Switch's spec-constant
selectors are enlarging it. See §7.22.

### 7.22 Residual: runtime specialisation is outside both digests (unowned)

Recorded, not closed, and not silently absorbed.

**What the digests cover.** `source_digest` covers the source closure and the compile-time variant
row; `shader_digest` covers the SPIR-V the build emitted. Both are fixed at build time. What runs is
a *pipeline*, and a pipeline is `(SPIR-V, specialisation values, layout)`.

**What I saw.** The instrument for it already exists and I built it last round for a different
reason: the `pipeline_variants` and `gemv_packed_spec_constant` counters record the effective
`(shader_stem, spec_constants)` pair handed to `get_or_create`. That is precisely the
runtime-resolved value the digests miss — the observation exists, the digest does not consume it.

**What closing it would cost.** A third digest is the wrong shape, because the value is not known
until dispatch and a proof written before the dispatch cannot contain it. The cheap and honest
version is a **dispatch-time frame witness**: hash the sorted `(stem, spec_constants)` set the run
actually bound, expose it beside `shaders_dispatched_digest`, and record it in the entry. Then a
proof taken under one specialisation and replayed under another reads as a frame delta with a name
rather than as agreement. Estimated cost: the counter already accumulates the pairs, so it is a
digest over an existing collection, one counters field, one entry field, one `SubjectVerdict`-style
comparison, and one refusal in the generator — comparable to the `source_digest` work, materially
smaller because the collection instrument is already there.

**Why I did not do it here.** It is not on the blocking pair and it changes what an entry *means*,
which is a schema question. It also interacts directly with Switch's selector work, so whoever owns
it should own both. Nobody owns it today.

**Closed by §7.23 (§8.9.20).** Built as costed, and the estimate held: a digest over the existing
collection, one counters field, one entry field, one comparison, one generator refusal.

### 7.23 §8.9.20 — the dispatch-time frame witness

**The hole, demonstrated before it was closed.** `rust/tools/probe_specialisation_witness.py` runs
one `MatMulNBits` graph three times — with `ONNXRUNTIME_EP_VULKAN_GEMV_PACKED` unset, forced off,
and forced on — and reads all three digests out of the counters artifact:

| case | `shader_digest` | `source_digest` | `spec_digest` |
|---|---|---|---|
| unset | `4be613c24634ec9e` | `270e8086408f69a4` | `776968369d964eb4` |
| forced off | `4be613c24634ec9e` | `270e8086408f69a4` | `776cce369d9931dd` |
| forced on | `4be613c24634ec9e` | `270e8086408f69a4` | `776968369d964eb4` |

Rows 1 and 2 are the hole: **identical SPIR-V, identical source closure, different kernel.** Both
build-time digests call these the same frame and the ledger says `PROVEN` about whichever one was
not measured. Rows 1 and 3 are the control: arming a switch that was already on moves nothing. A
digest that always moves is a clock, not an instrument, and this section refuses to ship one.

**Why not a third build-time digest.** The value does not exist until `vkCreateComputePipelines`.
A claim is decided *before* any pipeline is created, so a witness consulted on the claim path would
report `SPEC-UNOBSERVED` on every run and `specialisation_delta_forms` would be a list whose only
possible content is empty. That is this project's own recorded defect class — a single-valued state
wearing the shape of a predicate — so the audit hangs off the **dispatch** path
(`vk/session.rs` → `registry::audit_dispatch_specialisation`), gated on `record_pipeline_variant`
returning `true` so it costs one ledger scan per distinct pipeline rather than one per dispatch.

**Five states, because four would lie.** `SpecWitness` is `UNOBSERVED` (nothing bound yet),
`PARTIAL` (some of the entry's stems bound), `UNRECORDED` (the entry records no specialisation),
`IDENTICAL`, `DELTA`. `PARTIAL` exists because a digest over *part* of an entry's stem set compared
against a recorded full-set digest would invent a delta out of a run that has merely not finished
binding. Only `DELTA` contributes a δ to `entry_state`.

**`SPEC-UNRECORDED` claims — and this is deliberately not §7.21's row 5.** A missing `source_digest`
is repairable from the tree (`--backfill-frame`), so declining on it buys a fix. A missing
`spec_digest` is a fact about a **run that has already ended**: no build can recover it and only
`--reprove` can. Declining would take the EP to zero claims for a repair nobody can perform.

**What an entry now means, and the disclosure that says so.** Every one of the 103 shipped entries
is `SPEC-UNRECORDED`. Their meaning has narrowed to *"this kernel's bytes, under a pipeline nobody
recorded"* — and a narrowing that is not disclosed is a quiet demotion of 103 proofs, so it is
disclosed twice: an INFO line on every session (`disclose_specialisation_frame_of`) and a
`NOTE(§8.9.20)` line on `gen_proof_ledger.py --check`. `entry_state` is consequently
**time-dependent** — its answer can change once a pipeline is bound. That is the finding, not a
defect; no node loses a claim mid-run, because a specialisation delta yields `ProvenElsewhere`,
which is still claimable.

**The generator refuses rather than backfills.** `entry_line()` will not write a *new* entry whose
`spec_digest` is absent, `NONE-DISPATCHED` or `PARTIAL-…`; `--backfill-frame` is deliberately not
extended, because there is nothing in the tree to read it from.

### 7.24 Three things the instrument was saying wrongly

**A warning emitted before anyone is listening was never emitted.** `registry::ledger()` is a
`OnceLock` initialised by the first key lookup, which happens **before ORT attaches its logger**.
Its `log::warn!` for a whole-file fault therefore went to a sink that did not exist, and being a
`OnceLock`, it was never repeated — which is how the log read `0 × "proof ledger fault"` while the
counters read `ledger_faults` on every entry. Per-entry demotions had already been moved to
session-disclosure time by §8.9.18; whole-file faults had not. `disclose_ledger_faults_of` now
re-emits them through the ORT sink every session.

**The decline text was false in every clause on a faulted ledger.** It said *"no proof ledger entry
for X … nothing has proven it correct on this form"*, when in truth nothing was **known**, and the
entry proving that form may well have been sitting in the file. `LedgerLookup` has carried the
Hit/KeyAbsent/Faulted distinction since R13 and nothing read it at the decline site. A blanket in
the *state* is a safety property; a blanket in the *text* is a false statement, and the two need
not travel together. `unproven_decline_detail` now names an instrument failure as one.

**A count with no keys cannot be acted on.** `unproven_declines` moved from 3 to 5 between two
builds of this repository, and nothing in the artifact could say which two forms had joined it;
establishing that they were not mine cost a second worktree and a second release build.
`subject_changed_forms` had carried its keys since §8.9.19 for exactly this reason; the older and
far more common decline had not. `unproven_decline_forms` now does — and answered the question it
was built for on its first run: the two new declines are
`ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32` in both its static and runtime-extent forms, i.e.
Tank's new Cast kernel arriving ahead of its proof. That is the ledger gate behaving: a kernel that
exists and is unproven declines, and says so by name.

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
| `accuracy_level` | **0–3 yes, 4 no.** Corrected 2026-07-29 — see below |

**`accuracy_level` is a hint at every value but one, and that was worth reading the kernel to
establish.** The row above previously said "honour as a hint, never a correctness requirement",
which is wrong at level 4 in the permissive direction. ORT's CPU kernel
(`contrib_ops/cpu/quantization/matmul_nbits.cc`, `GetComputeType<T1>`) branches on the attribute
exactly once:

```cpp
if (attr == Level4 && MlasIsQNBitGemmAvailable(nbits, block_size, SQNBIT_CompInt8))
    return SQNBIT_CompInt8;
return SQNBIT_CompFp32;          // <float>: every non-ARM64 host
```

So 0, 1, 2 and 3 all resolve to the same path and are indistinguishable in the output — declining
them would decline every real graph for no numerical reason. **Level 4 quantizes the activation `A`
to int8 at the weight's block size.** That is a different computation, not a wider accumulator, and
a kernel that multiplies against `A` at storage precision returns a plausible wrong answer for it.
The predicate now declines it with `[attribute]`.

Note also that ORT `ORT_ENFORCE`s `bits ∈ {2,4,8}` and `block_size ∈ {16,32,64,128,256}` at kernel
construction. Our claimed sets are strict subsets of both, which is the right direction: a node we
claim is always one the CPU oracle can also build.

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

### 8.1.1 The dequantisation semantics, derived from the oracle rather than the spec prose

**2026-07-29 — landed and executing on both devices.** Before writing a line of GLSL I fed
`A = I` through an ORT CPU EP `MatMulNBits` so that each output row *is* a dequantised weight
column, and read the layout off the result. Doing it this way rather than from memory of the
schema is the same discipline as §7: a shared misreading of the prose would have produced a kernel
that agrees with my own reference implementation and disagrees with ORT.

| Question | Answer, as measured |
| --- | --- |
| Orientation of `B` | `B` row `n` **is** output column `n`. `Y[m][n] = Σ_k A[m][k] · dequant(B[n][k])` — transposed relative to `MatMul`. |
| 4-bit nibble order | **Low nibble first.** Element `2i` is `byte[i] & 0xF`; element `2i+1` is `byte[i] >> 4`. |
| Implied zero point when `zero_points` is absent | `1 << (bits-1)` — **8** at 4 bits. Measured: nibble `i` dequantised to `i - 8`. |
| `zero_points` packing | Same packing as `B`: two blocks per byte at 4 bits, low nibble first; one byte per block at 8 bits; each column's run padded to a whole byte. |
| `scales` indexing | `n * blocks_per_col + blk`. |
| Sign | The zero point is **subtracted**: `(q - zp) * scale`. Verified at `zp = 0` and `zp = 3`. |

**Three things the kernel deliberately does not assume.**

1. **No subgroup operations.** Both development GPUs report a subgroup size of 32, which is the
   strongest possible invitation to bake 32 in and pass every local test. `subgroupSize` is not
   guaranteed to be anything, so the cross-thread reduction is a shared-memory tree sized by
   `gl_WorkGroupSize.x`, and shared storage is a fixed 256 floats = 1 KiB — inside the 16 KiB
   floor of §7.2, not the Iris Xe's 32 KiB.
2. **No 16-bit storage or arithmetic capability.** fp16 activations, scales and outputs go through
   `unpackHalf2x16`/`packHalf2x16` over `uint` buffers, which is core GLSL and gates nothing.
   Accumulation is fp32 regardless of storage, which is also what ORT's `SQNBIT_CompFp32` path
   does. **This is not an optimisation — it is the requirement:** all 161 of Phi-3.5's
   `MatMulNBits` nodes carry fp16 `A`, `scales` and `Y` (§4.21 re-census), so an f32-only kernel
   would decline the model the kernel exists to run.
3. **No tile-size query.** The workgroup size is the smallest power of two covering
   `K / block_size`, clamped to `[32, 256]`, where 256 is the *guaranteed*
   `maxComputeWorkGroupInvocations` floor rather than the larger figure either local device
   reports. Grid is `(N, M_total, 1)`: one workgroup per output element. That makes the kernel
   **correct for every `M`** — prefill is a performance problem, not a correctness one — which is
   why the row claims all ranks rather than only decode.

**Verified 2026-07-29 on both devices, identical results:** 29 passed / 1 failed in
`tests/ops/test_matmulnbits.py` (the one failure is `DequantizeLinear`, still `Staged`, failing
loudly by design). The passing set covers bits ∈ {4, 8} × block_size ∈ {16, 32, 64, 128} ×
{3-input symmetric, 4-input asymmetric} in fp32, `M ∈ {1, 2, 7, 32}`, and fp16 at
`M ∈ {1, 3}` both symmetric and asymmetric, each against the ORT CPU EP within the §10.1 Regime-2
tolerances.

### 8.1.2 The island result contradicts the premise this kernel was scheduled on

The brief that scheduled this kernel — and my own earlier note — said Phi-3.5 sits at 34–35 islands
without `MatMulNBits` and **collapses to one island of 364 with it**. Measured on the real graph,
that is wrong, and wrong in the direction §7.2 keeps warning about:

| claimed set | coverage | islands | largest |
| --- | ---: | ---: | ---: |
| elementwise only (today) | 27.3 % | 35 | 3 |
| **+ `MatMulNBits`** | **71.3 %** | **100** | **6** |
| + `SkipSimplifiedLayerNormalization` | 88.8 % | 5 | 320 |
| + `GroupQueryAttention` | 97.5 % | 2 | 356 |
| + `Reshape`/`Gather`/`Shape`/`ReduceSum` | 99.7 % | 1 | 365 |

gpt-oss-20b behaves the same way: 46.3 % / 148 islands → 65.8 % / **100** islands with
`MatMulNBits` → 78.6 % / 4 with `SkipSimplifiedLayerNormalization` → 85.0 % / 1 with
`GroupQueryAttention` → 100 % / 1 with `QMoE`.

**`MatMulNBits` alone makes the partition strictly worse on both models** — coverage nearly
triples while the island count also triples. The reason is structural and obvious once measured:
in a GenAI-built decoder block, every `MatMulNBits` is separated from the next by a
`SkipSimplifiedLayerNormalization` or a `GroupQueryAttention`, so claiming the projections without
the things between them shreds each block instead of fusing it. The collapse is caused by the
**pair** `(MatMulNBits, SkipSimplifiedLayerNormalization)`, and it is a collapse worth having: 5
islands with a largest of 320 on a 366-node graph.

This is the third landing of the same lesson (§4.18, §7.3, the gpt-oss `Cast` result) and the
recurrence is the finding. It also vindicates the metric of record being the triple
`(coverage, island_count, largest_island_flops)` rather than coverage: on coverage alone this
kernel looks like a 27 % → 71 % win, and shipped alone it would have made Phi-3.5 *slower*.

**Operational consequence:** `MatMulNBits` should not be enabled on a real model without
`SkipSimplifiedLayerNormalization`, which Switch is landing concurrently. The two are one unit of
value, not two.

### 8.1.3 What still blocks Phi-3.5, and it is not this kernel

**Superseded in part by §7.4 and §7.5 — read those first.** What follows was right about the
mechanism and wrong about the consequence. `REQUIRE_STATIC_SHAPES` no longer exists; the engine
precondition it encoded is now `ENGINE_ACCEPTS_RUNTIME_EXTENTS` (§7.5.2), and the counterfactual
in §7.5.5 measures these 161 nodes as the largest single block that runtime extents unlocks.

Measured, not inferred: a `MatMulNBits` node with static shapes is claimed and matches the CPU EP
at rank 2 and rank 3 (leading dimensions fold into the row count). The same node with symbolic
`batch`/`seq` dimensions is **declined**, because `claim::REQUIRE_STATIC_SHAPES` is a global
property of the wire — `Compile` bakes push constants and buffer byte sizes from compile-time
extents. Every `MatMulNBits` node in the real Phi-3.5 graph has symbolic leading dimensions.

What I concluded from that — that the island numbers in §8.1.2 are a ceiling reachable only once
dynamic shapes land in the engine — does not hold. §7.4.4 measures the same model with
`batch_size` and `sequence_length` pinned through ORT's free-dimension overrides: 161 nodes
claimed, 161 islands, correct logits on both devices. The ceiling is partly reachable **today**,
from the caller, with no EP change. The blocker was real; the assumption that only the EP could
move it was not.

The second gate, which this section missed entirely, is dtype: once shapes are pinned the 97
elementwise nodes decline on **f16**, not on shape. See §7.4.5.

### 8.1.4 P6 asserted structurally, and the multi-run verification

**P6** — *no dequantised weight is ever materialised in device memory* — has been the stated
constraint on this kernel since §8 was written, and until now it was aspirational: `allocator.rs`
carries `AllocStats::high_water_bytes` with two comments naming "Mouse's P6 assertion", and no such
assertion existed anywhere in the tree.

It is now asserted, and **not** as a high-water threshold. The argument for the structural form:

- `DispatchContext::alloc_temp` is the **only** way an op handler can obtain device memory. A
  handler that never calls it cannot have materialised anything, for **every** shape at once.
- A high-water threshold only proves the property for the shapes actually run, and needs a bound
  loose enough not to be flaky — which is loose enough to hide a small scratch buffer.
- **Zero is not a threshold.** `the_gemv_materialises_no_dequantised_weight` asserts exactly zero
  `alloc_temp` calls, exactly one dispatch, and that the only bound output is *activation*-sized
  (`n` elements), across `(K,N)` of (3072, 8192), (8192, 3072) and (3072, 3072) — the shapes all
  161 real nodes take. `gemv_allocation_is_independent_of_the_reduction_extent` quadruples `K` at
  fixed `N` and asserts the allocation record is byte-identical, which is the property that would
  break first if a dequantised `[K, N]` buffer ever appeared.
- The structural form also needs nothing from anyone: `high_water_bytes` is not in the counters
  JSON, and putting it there means editing Tank's `counters.rs`.

The numbers that make it matter: at K=8192, N=3072 the packed weight is 12 MiB and its f32
expansion is 96 MiB — times 161 nodes.

**Negative control.** A `ctx.alloc_temp(TensorDesc::new(dtype, vec![k, n]))` was temporarily
inserted into `matmul_nbits_gemv`; the test failed with *"the GEMV asked for 1 scratch buffer(s)
totalling 50331648 bytes"*, and was reverted. A guard that has never been seen to fail is a guard
whose failure mode is unknown.

**Multi-run verification, against Tank's interior-pointer finding.** Tank measured that from the
**second** run of a session, ORT's memory-pattern planner hands back interior pointers
(`handle + 48 KiB`, 52 of them across five runs, both devices) — a *wrong answer*, not a crash.
Every model-level check on record, mine included, had run exactly one inference per session, so
none of them could have seen it. `MatMulNBits` carries by far the largest bound buffers in the
graph, so it would meet the hazard first.

Five inferences in **one** session, with **different feeds per run** plus a late repeat of run 0's
feeds, every run compared against the CPU EP:

| device | worst `max abs diff` | argmax agreement | late repeat |
|---|---|---|---|
| 0 — Intel Iris Xe | 0.08984 | 5/5 runs | bit-identical to run 0 |
| 1 — RTX 4060 | 0.09473 | 5/5 runs | bit-identical to run 0 |

Clean on both. The insulation is structural rather than lucky: op code never sees a raw pointer —
the typed `DispatchContext` is the whole of the interface, and `transfer::host_backing_for`
resolves the offset below it. That was the design intent; this is the first time it has been
*measured* rather than assumed, and the distinction is the point.

### 8.2 Prepacking and the memory model

**Status 2026-07-28 — prepacking is on the critical path, not a tuning pass.** The XL-kernel ruling
makes an int4-quantized LLM a *functional* requirement, and an int4 LLM whose weights are repacked
per dispatch is not a working LLM. Prepacking is therefore a T4 blocker with the same standing as
the kernel itself, and it needs an engine seam that does not exist yet.

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

#### 8.2.1 The prepack seam — precise requirement for Switch

This is the engine change T4 depends on. Stated as a contract, not a design:

| # | Requirement | Why it is load-bearing |
|---|---|---|
| P1 | A host-side hook that runs **during `Compile`, after device selection and after tile/specialization-constant choice**, and can transform an initializer's bytes before upload. | The repack layout is a function of the tile config, which is a function of the device. Repacking before device selection produces the wrong layout; repacking at first dispatch produces a stall and a mutable-weight lifetime problem. |
| P2 | The packed result is cached keyed by **`(initializer identity, device, tile_config, kernel variant)`** and uploaded once. | Two nodes sharing a weight (rare) or one weight used by both the prefill and decode variant (common) must not repack twice, and a variant change must not silently reuse a stale layout. |
| P3 | `scales` and `zero_points` upload as **separate device buffers with their own descriptor bindings**, not interleaved into `B`. | Interleaving saves one binding and costs the ability to stream `B` as a dense `uvec4`, which is the entire GEMV bandwidth argument. |
| P4 | The original ONNX-layout weight must be **droppable** after repack — the engine must not keep both resident. | Otherwise int4 costs 2× and the memory argument for int4 evaporates. |
| P5 | A way for a kernel to declare "my weight input is prepacked, do not bind the raw initializer" so `DispatchContext` binds the packed buffer instead. | The op code must not learn about the difference; the binding table does. |
| P6 | The prepack function itself is **op-owned code** (`ops::quant::prepack`), called by the engine. Pure `&[u8] -> Vec<u8>` plus a shape/config struct — no Vulkan handles, so it stays inside the `src/ops/**` layering rules. | Keeps the nibble-unpack/interleave logic next to the kernel that consumes it, which is the only way the two stay in sync. |

**ORT version note.** ORT's own `PrePack` hook is the natural place for P1. ORT **1.27's plugin-EP
`PrePack` passes a null allocator**, which makes it unusable from a plugin EP — this is one of the
reasons the team pinned **ORT 1.28**, and `sys.rs` already encodes the version window
(`ORT_PINNED` 1.28.0 / `ORT_FLOOR` 1.24.0). If we cannot use `PrePack` for any reason on a given
release, the fallback is to do the transform inside our own `Compile` when we walk initializers —
strictly worse only in that ORT cannot then share the packed buffer across sessions. **The claim
predicate must not depend on which of the two paths is used.**

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

### 8.9 The proof ledger — claimability is derived, not hand-written (2026-07-30)

**Background.** Every `OpStatus::Live` row was previously hand-written by the op author who also
authored the test. That is a conflict of interest: the person who decides the test counts is the
person whose row benefits from it. The MatMulNBits defect (§7.1.4) is the sharpest example — the
4-input form was exercised, the author marked the row `Live`, and the 3-input form (Phi-3.5's
actual form) silently computed nothing for weeks. `compute_failures: 0` throughout.

**The ruling (Morpheus).** `OpStatus::Live` is no longer hand-written. Claimability is derived per
form from a harness-generated, never-hand-edited ledger. The ledger is the only source of truth for
which forms have been proven; `registry.rs` is the consumer; Trinity's differential harness
generates it from actual dispatch runs on both local devices.

**The proof key.** A form is a tuple:

```
(domain, op_type, opset_bucket, input_dtypes, output_dtypes, kernel_variant_key,
 shape_class, populated_optional_input_set)
```

Where:
- `opset_bucket` groups opset versions that share the same schema (e.g. `7..=12`, `13..=18`);
- `kernel_variant_key` is the shader stem without the dtype suffix (e.g. `"ew_binary"`, `"matmul_nbits_gemv"`);
- `shape_class` is `"static"` or `"dynamic"` (ORT-side symbolic axes present at Compile time);
- `populated_optional_input_set` is the frozenset of slot indices that are non-null (`{0,1,2}` vs `{0,1,2,3}`).

Two forms that differ in any dimension are different proof obligations. A proof of the 4-input
`MatMulNBits` form cannot satisfy the proof obligation for the 3-input form.

**The escape hatch — `CLAIM_UNPROVEN`.** When a form is needed in production before the harness has
run it — e.g. an op just shipped but Trinity's next run hasn't landed yet — the author may add a
`CLAIM_UNPROVEN` entry in `rust/src/ops/claim_unproven.rs`. The constraint:

- `CLAIM_UNPROVEN` takes a list of **explicit proof keys** and nothing else.
- A parser that can express "everything" (wildcards, ranges, regexes) must not exist.
- This is enforced by planted rejection tests: `test_claim_unproven_wildcard_is_rejected`,
  `test_claim_unproven_star_is_rejected`. These tests must go red if the parser is ever widened.
- Each entry expires when the harness adds the corresponding key to the ledger; a stale
  `CLAIM_UNPROVEN` is a warning (`CLAIM_UNPROVEN_STALE`), not a silence.

**What `registry.rs` reads.** At startup, it merges the ledger file (path from
`ONNXRUNTIME_EP_VULKAN_PROOF_LEDGER`, defaulting to `proof_ledger.json` next to the binary) with
`CLAIM_UNPROVEN`. A form is claimable iff its key appears in either. If the ledger file is absent
and `CLAIM_UNPROVEN` is empty, no form is claimable — but `OpStatus::Staged` rows remain as before
(they decline with `[staged]` regardless). This means the ledger is additive: a build without
the ledger file is safe, not broken. The transition from hand-written `Live` to ledger-derived `Live`
is gated on Trinity's first harness run producing a non-empty ledger.

**Falsifier for the ledger itself.** `test_no_live_row_without_ledger_key`: walks every `Live` row,
computes its proof key, and asserts it appears in the merged (ledger ∪ CLAIM_UNPROVEN) set. Must go
red if a row is marked `Live` and its key is in neither source.

### 8.9.8 Populating the ledger — the 154 reds and what they cost to clear (2026-08-02)

Landing the gate took `pytest tests/ops` to **154 failed / 276 passed**. Not one of those was a
regression: Guard D refuses to report a CPU-vs-CPU comparison as a pass, and a form the gate
declines runs on the CPU EP in both arms. The remedy is proofs, never a softer guard.

**Enumeration is mechanical, and the instrument that enumerates has a defect worth naming.** Every
claim-log line carries `proof_key`, so the set of forms a red suite needs is a parse away. But the
claim log is **truncated by whichever process opens it**, and several tests spawn a child that
loads the DLL. A whole-suite run therefore reports only the records written after the last such
child — 786 records, where the same tests enumerated **per file** produce 3,140. The first triage
built on the whole-suite log concluded the residual was one form; the per-file triage found five.
Enumerate per file. A census that under-counts silently is worse than one that refuses.

**The residual, triaged by decline code rather than by count (R13).** Of the 45 forms still
declining after the first population round:

| population | ledger authority | what it is |
|---|---|---|
| `unproven` alone | yes | a form nothing has proven — 5 keys |
| `unproven` + `dtype`/`arity` | no | the handler declines it on another axis too |
| `staged` / `not-registered` / `attribute` / `opset` | no | coverage gaps; §8.9 has no say |

Four of the five pure-`unproven` keys were `SkipSimplifiedLayerNormalization` forms differing from
the already-proven pair **only in `shape_class` (static vs runtime-extent) and dtype (f32 vs f16)**
— which is the key doing its job. Proving them took four case models and cleared **seven** tests.

**The fifth is `GroupQueryAttention`, and it is a finding, not an obstacle.** Its proof run returns

    DIVERGENT {'reason': 'output o0 outside tolerance', 'worst_rel': 16.72642029784887}

reproducibly, to the digit, on two runs. So GQA stays out of the ledger, Phi-3.5's five tests stay
red, and they stay red **for the correct reason**: the fused attention kernel does not reproduce the
CPU oracle on the decode form. This corroborates the pre-existing strict-`xfail` `_GQA_COMPUTE_BUG`
in `tests/ops/test_gqa.py` from a second, independent instrument.

A non-`MATCH` verdict **cannot** be recorded in `proof_ledger.jsonl`: `parse_ledger` pushes it to
`Ledger::faults`, and a ledger with faults refuses *every* claim, not just the bad one. So attempts
are appended to **`evidence/proof_attempts.jsonl`**, which grants nothing and is not baked into the
binary. It exists so that *"we measured GQA and it disagreed by 16.7×"* cannot decay into *"GQA has
no entry"* — the same distinction `Ledger::demoted` draws at session disclosure.

**Feed plans.** Some inputs are not free variables. GQA's `total_sequence_length` must equal past +
current and `seqlens_k` must agree with the cache extent; a random `int32` there is not a harsher
test, it is an invalid model, and the CPU arm raises rather than producing an oracle — an
`ERROR(instrument)`, not a verdict. `ledger_case_models.feed_plan()` pins those values and the
symbolic extents beside the case that needs them. The dims stay symbolic *in the model*, which is
what keeps the form in the `runtime-extent` shape class; only the run binds them.

**A compile input that lives outside `rust/`.** The ledger is baked with
`include_str!("../../evidence/proof_ledger.jsonl")`, and the criterion-5 shader-less witness builds
from a copy of `rust/` alone. It therefore failed with

    error: couldn't read `src\../../evidence/proof_ledger.jsonl`: The system cannot find the path specified.
    error: could not compile `onnxruntime-ep-vulkan` (lib) due to 1 previous error

and reported `ERROR(instrument)` — correctly, because a build that failed for an unrelated reason is
not a criterion-5 result. The scratch tree now carries the ledger, and the ledger's mtime now
participates in the witness's staleness check, so a re-generated ledger cannot be witnessed against
the previous binary.

**Result:** `154 failed / 276 passed` → **`37 failed / 393 passed`**; ledger **9 → 73 entries**;
census `ledger_lookup: ALL-PROVEN … ledger_entries=73`, byte-identical on device 0 and device 1.
Of the 37, **26** are `test_op_table` coverage gaps, **5** are Phi-3.5 behind the GQA divergence,
**3** are `Min`/`Max`/`Clip`-no-bounds (`[staged]`, `[arity]`), **1** is criterion 10 behind the same
GQA divergence, and **0** are instrument errors.

### 8.9.9 The real Phi-3.5: 0 → 323/363 claimed, with the ledger unchanged (2026-08-02)

The request that produced this section carried a diagnostic taken from the real Phi-3.5 artifact,
listing five `[unproven]` keys behind `claims 0/363`, and a premise: *everything the ledger proves is
`static`; everything Phi-3.5 needs is `runtime-extent`*.

**The premise was already stale when it arrived.** It was taken against the pre-`e97b186` ledger. At
the merged state, **four of the five keys are in the ledger at `runtime-extent`** — `MatMulNBits`
(`scales`), `Mul` f16, `SkipSimplifiedLayerNormalization` f16 (`>f16,-,-,f16`), and `Sigmoid` f16.
Only `GroupQueryAttention` is absent, and it is absent for a reason that is not an oversight (below).
This is worth recording because the correct response to a stale premise is not to act on it: minting
more `runtime-extent` keys would have been work whose effect was already achieved.

**What was written down before the run.** Per R10, `bench/results/phi35_runtime_extent_prediction.json`
was written before the real model was loaded, with five predictions and a falsifier for each. All five
scored CONFIRMED, P2 and P5 exactly:

| | Prediction | Outcome |
|---|---|---|
| P1 | not `0/363` any more | 7 proven forms; **323/363** |
| P2 | exactly 32 GQA nodes decline, `[unproven]` **alone** | exactly that, on that key |
| P3 | claimed count inside `[300, 340]` | 323 |
| P4 | criterion 10 does **not** go green; the *failure text* changes | it did, to a broken commitment |
| P5 | the ledger does **not** grow | held at 73 entries, digest `e3ea94196b4fd84f` |

**The guard the coordinator named was "a ledger that grows without the claimed count moving." What
happened is its inverse**: the claimed count moved `0 → 323` across `33/33` retained islands, with
**zero** new ledger entries. `ledger_hits=323`, `unproven_forms_claimed=0`,
`claimed_form_evidence=ALL-PROVEN`, `ledger_gate=MIXED` — `MIXED` because 40 nodes are still declined,
which is the gate reporting a partial claim rather than rounding it to either end.

**`GroupQueryAttention` stays out, and this is a finding rather than an obstacle.** Its proof run
returns `DIVERGENT` with `worst_rel=16.72642029784887`, reproducible to the digit across runs. It is
not written to the ledger — a non-`MATCH` verdict in `evidence/proof_ledger.jsonl` becomes a
`Ledger::fault`, and a faulted ledger refuses **every** claim, not just the bad one. It is recorded in
`evidence/proof_attempts.jsonl` instead, which grants nothing and is not baked in. The 32 GQA nodes
are the only pure-`[unproven]` declines left in the flagship model. Note what this combination means:
the handler *claims* the form and then disagrees with the CPU oracle by 16.7×, so claiming-then-
diverging is itself the defect; it corroborates the pre-existing strict-`xfail` `_GQA_COMPUTE_BUG`.

**The next defect is now visible, and it is not in `ops/`.** With 323 nodes claimed, ORT still fell
back to the CPU EP for the whole graph, and Tank's broken-commitment WARN fired correctly, naming
fused subgraph #15. The root text, obtained after a five-line R13 diagnostic change in
`rust/src/vk/session.rs` that logs the handler's own error instead of a bare "translate failed":

    Unsupported("`SimplifiedLayerNormalization` input 0 has no element type at compile time")

raised from `common_dtype(node, 0, 2)?` in `rust/src/ops/common/templates.rs::simplified_norm` during
the **dynamic** re-run of translate. Island #15 is `embed_tokens/Gather → layers.0/input_layernorm →
qkv_proj/MatMul_Q4`, so SLN's input 0 is an island-internal intermediate; the `patched_node` handed to
translate carries a shape but no element type. **The handler is right to refuse — the caller's
construction is the defect**, and `rust/src/vk/` is Switch's.

**A consequence that must not be read past.** Because the islands still execute zero times,
`criterion10-dev0.json` reports `own_provider_execution_count: 0` and
`executed_by: {CPUExecutionProvider: 1377}`. Its `oracle_outputs_degenerate: 0` was therefore measured
on a CPU-versus-CPU run. **Switch's all-65 oracle arm has still not had its first real reading**, and
the reopened degenerate-KV question is still open. `series verdict is UNATTRIBUTED` is the correct
finding, and it is now blocked on the SLN defect rather than on the ledger.

**Union defect repaired in the same change.** Trinity's criterion-11(c) controls referenced
`evidence/cases/mul_f16_unproven.onnx`, which `e97b186` deleted when the planted control moved to
`sub_f16_dyn_unproven` (the `Mul`/f16/static form had become *proven*, which disarms a control
silently — the worst failure mode a control has). Restoring the file would have restored a control
that passes for the wrong reason. Control 1's axis is *dtype*, so it now uses a different op:
`Abs` f32 (proven) against `Abs` f16 (unproven), the f16 arm **built in `tmp_path`** so the generator
cannot pick it up and quietly prove it. Control 2 gained its own static `Mul` arm, because a
shape-class control whose two arms are different ops is not a shape-class control. And
`test_ledger_key_discriminates_optional_inputs` selected the `MatMulNBits` pair by counting *all*
`MatMulNBits` entries (`== 2`); the ledger legitimately grew to five, so it now selects the pair by
form (f16, `static`) instead. A control keyed to a total is a control that goes red when the artifact
it guards gets better.

### 8.9.10 ORT's own refusal, and the branch that actually holds Phi-3.5 (2026-08-02)

**`session.disable_cpu_ep_fallback`, wired into the proof harness.** Our attribution was already
counter-based (`claimed_nodes`, `dispatches_executed`, per-key `admitted ⊆ offered`), but every one
of those counters is written by the thing being audited. This ORT session option is a refusal from
*outside* our code: if any node lands on the default CPU EP, ORT declines to build the session at
all. For a single-form evidence case — where the EP either takes the one node or the case proves
nothing — that converts a silently vacuous CPU-versus-CPU comparison into a raise at session
creation.

Two implementation facts, both learned by running it:

- **It conflicts with naming the CPU EP explicitly.** With
  `providers=["VulkanExecutionProvider", "CPUExecutionProvider"]` ORT raises
  `InvalidArgument: Conflicting session configuration: explicitly added the CPU EP to the session,
  but also disabled fallback to the CPU EP via session configuration options` — which reads nothing
  about our EP and is `ERROR(instrument)`. The strict arm offers this EP alone.
- **It is deliberately not set on the discovery pass.** Discovery runs before the hatch is open, the
  node is declined by design, and a refusal there is the expected state rather than a finding.

The refusal is raised as `CpuFallbackRefusal`, distinct from `InstrumentError`, and `prove()` turns
it into `UNATTRIBUTED` with the ORT text quoted. An outage and a reading must not spell the same
(R13).

**Mutation-tested in both polarities before it was trusted** —
`rust/tools/probe_cpu_fallback_guard.py`, witness `bench/results/cpu_fallback_guard_probe.json`.
Subject is the planted control `sub_f16_dyn_unproven`, chosen because the generator refuses to write
its key under any circumstance, so neither arm can contaminate the ledger:

| Arm | Hatch | Verdict | ORT refused |
|---|---|---|---|
| CLAIMED | the real key | `MATCH` | no |
| DECLINED | a key that does not exist | `UNATTRIBUTED` | **yes** |

A guard that fired in both arms would be worse than none — it would make every proof run look
vacuous. One that fired in neither is not wired. The probe asserts the arms differ, not that either
is non-zero.

**What it did not change: `GroupQueryAttention` is still `DIVERGENT`.** Re-run under the strict
guard, ORT did *not* refuse — the EP genuinely took the node and executed — and the verdict is
`worst_rel = 16.72642029784887`, identical to the digit across three runs now. **The divergence was
never vacuous.** It stays out of the ledger; it is a correctness finding on the flagship model, and
the handler claiming the form and then disagreeing by 16.7× is itself the defect.

**Current-state reading of the real model** (`rust/tools/probe_phi35_claim_reading.py`, witness
`bench/results/phi35_claim_reading_summary.json`), recorded because a `0/363` diagnostic has now been
routed twice from an older build — the frame of a result is the binary that produced it:

    claimed_nodes 323   ledger_hits 323   unproven_declines 34   unproven_forms_claimed 0
    islands_offered 33  viable_islands_retained 33
    ledger_gate MIXED   claimed_form_evidence ALL-PROVEN   ledger_entries 73

**The critical path is no longer the ledger.** With 323 nodes claimed the model still runs entirely
on the CPU EP, and the branch that causes it is now identified exactly rather than approximately. In
`rust/src/vk/session.rs`, the loop that patches concrete `TensorDesc`s onto a node before the dynamic
re-run of translate handles two token ranges — external ORT inputs, and intermediates from prior
kernels — and leaves the middle range, *island outputs that are also consumed internally*, as `None`,
under a comment reading "theoretically possible but unusual … the translate handler will degrade
gracefully". A temporary instrumented build, since reverted, measured which branch the real model
takes:

    node=/model/layers.0/input_layernorm/LayerNorm op=SimplifiedLayerNormalization
    slot=0 token=5 n_plan_inputs=5 n_plan_outputs=2 branch=island-output-consumed-internally

It is not unusual; it is island #15's normal shape, because `embed_tokens/Gather`'s result is both an
island output and an internal edge. The handler does not degrade gracefully — it refuses with
`Unsupported("`SimplifiedLayerNormalization` input 0 has no element type at compile time")`, which is
correct of the handler — so the island is dropped and ORT falls back wholesale.

**This was deliberately not repaired in `ops/`.** `common_dtype` could be made to infer the missing
element type from a sibling input, and it would have made the number move today. It would also be a
check that moves with the reader's confidence rather than with its subject (R9 amendment 5): the
information was lost by the caller, and a handler that fabricates it stops being able to detect that
the caller lost it. The branch is one `else` in `vk/session.rs`, which is Switch's.

### 8.9.11 The ledger had no re-proof path: an entry that outlived its subject (2026-08-02)

**The hole, found by Switch while fixing GQA.** `gen_proof_ledger.py --append` printed
`UNMEASURED … no unlockable keys` and then `PASS`, writing nothing. The GQA form was already
claimed, so the generator skipped it as an optimisation — and the entry proving it predated
*two* shader rewrites made that same day: the index-space fix that took GQA from
`DIVERGENT 16.726` to `MATCH 0.00072939`, and the prefill fix that took `present_key` from
`worst_rel 774.8` to `0`. The entry was minted against a shader that no longer existed.

This is the shape §8.9 was built to refuse, arriving from a direction none of us guarded. Not
*a ledger derived from the claim table* — Morpheus's refusal — but **a ledger whose entries
silently outlive their subject**. `ledger_hits == proven_key_lookups` stays true forever while
the thing proven drifts out from under the proof. The key is the same; the kernel is not.

**And note what caught it: nothing did.** Switch established GQA's correctness by his own
independent comparison against the CPU EP. The ledger agreed — but it would have agreed
identically had he broken the kernel, because it never re-measured.

> **A proof that cannot be invalidated by changing its subject is not a proof of that subject.**

#### The four repairs

1. **The entry carries a digest of its subject.** Each `MATCH` records `shaders` — the SPIR-V
   modules the proof run actually dispatched, by stem — and `shader_digest`, an FNV-1a/64 over
   the compiled bytes of those modules. `parse_ledger` recomputes the digest against the SPIR-V
   baked into *this* binary and demotes the entry on disagreement.
2. **`--reprove`.** A form that is already claimed can be re-measured deliberately instead of
   skipped. `discover_keys(model, reprove=True)` re-offers claimed keys and the census gained
   `claimed_reoffered`, so "re-offered and re-measured" is distinguishable from "never offered".
3. **`PASS` is no longer printed over a run that measured nothing.** `UNMEASURED … no unlockable
   keys` followed by `PASS` is a report whose two halves describe different things — the same
   defect that cost Morpheus a criterion, where `outputs_compared: 65` sat among oracle facts
   while counting cross-run comparisons. The generator now tracks `measured_any`, prints an
   explicit `NOT MEASURED:` line per model that offered no key, and appends a `NOTE:` to the
   `PASS` when the run as a whole measured nothing.
4. **Two new demotion tokens**, joining Tank's `DIVERGENT`/`UNMEASURED` at the §8.9.7 disclosure:
   `STALE-SHADER` (digest disagrees) and `NO-SUBJECT-WITNESS` (entry carries no `shaders`
   field at all). Both name `--reprove` as the remedy in their fault text (R13).

#### The digest frame — what it covers, and what it deliberately does not

The coordinator asked for this decision to be stated rather than assumed, in the shape Link uses
for his frames. Point 1 makes an entry stale on any change to a shader it dispatched, and the
pressure to relax that will arrive the first time someone is in a hurry. So:

**COVERS.** The compiled SPIR-V bytes of every module the proof run dispatched, keyed by stem
and hashed after sorting and dedup, so dispatch order does not move the digest. A formula
change, an index-space change, a workgroup-size change, a binding change, a deletion or a
rename all move it. A module named by an entry but absent from this build hashes a distinct
`\x01MODULE-ABSENT` marker rather than being skipped — a shader that has been deleted must not
silently produce the same digest as one that is unchanged.

**DOES NOT COVER, deliberately.**

- **Shaders the run did not dispatch.** A per-entry digest over the *whole* shader set would
  make every unrelated edit demand 73 re-proof runs, and a gate whose cost is that
  indiscriminate is a gate that gets relaxed. The narrower digest is the one that survives.
- **Host-side code** — translation, allocation, descriptor construction, push-constant values.
  **This is a named residual, and its falsifier is exact: a host-only numeric change leaves the
  entry green.** `dispatches_executed` catches part of that class; it is not claimed to catch
  all of it. Closing it properly needs a CI re-proof lane, which is Link's frame, not this one.
- **Comment-only GLSL edits**, which do not survive to SPIR-V under our `glslc` flags. This
  falls out of hashing the compiled bytes rather than the source, and it is the one relaxation
  taken — taken because it is verifiable from the compiler's behaviour rather than asserted.

**Staleness demotes per entry, never globally.** A blanket refusal would let one shader edit
disable every claim in the ledger — the blunt shape that gets switched off in a hurry.

#### Pre-existing entries are not grandfathered

All 73 shipped entries lacked the subject witness and therefore faulted under the new parser.
Admitting them "for compatibility" would have exempted exactly the entries that have had the
longest to drift — GQA's among them. The whole ledger was re-proved instead.

The bootstrap works because a fully-faulted ledger makes every form decline `[unproven]`, so
`discover_keys` finds them all even without `--reprove`, and `prove()` opens the
`CLAIM_UNPROVEN` hatch, which is independent of ledger state.

**Result of the re-proof: 74 entries, digest `d07643b0c4cd2e8f`.** One more than before, and the
new one is GQA — which under Switch's rewritten shader now proves `MATCH` where it proved
`DIVERGENT 16.726` this morning. That is the mechanism working in the direction that costs
something to state: the re-proof was run to invalidate entries, and it admitted one.

#### The thin margin, recorded rather than smoothed

Switch flagged GQA's `MATCH` margin as 1.37× — `worst_rel 0.00072939` against `rtol 0.001` —
and said so out loud rather than letting it pass as comfort. The re-proof reproduces it exactly.
Against the rest of the ledger it is not merely thin, it is an outlier:

| entry | `worst_rel` | margin vs its `rtol` |
|---|---|---|
| `GroupQueryAttention/…/runtime-extent/past_key+…` | 0.000729395 | **1.37×** |
| `Erf/9+/f32>f32/ew_unary_erf_f32/static/n1` | 0.0000045517 | ~220× |
| `MatMulNBits/…/f32,u8,f32>f32/…/static/scales` | 0.0000026391 | ~379× |

The next-tightest entry sits **160× further from its bound**. Switch believes the GQA figure is
one fp16 ULP at small denominators — rounding rather than formula. That reading is plausible and
is not asserted here as established. What is established is that the combination that made it
worth watching is gone: the entry can now be re-measured on demand, and a shader change makes it
stale rather than silently authoritative.

#### What the census reports

`shaders_dispatched` and `shaders_dispatched_digest` are emitted in the counters artifact beside
`unproven_forms_enabled`. When a session dispatched nothing the digest reads `NONE-DISPATCHED`
rather than the hash of an empty list — R12: "no shaders ran" and "shaders ran and hashed to X"
must not share a spelling, which is the defect this project already fixed once when *bypassed*
and *all-rejected* shared one `0`.

### 8.9.12 First real reading: 355/363 claimed, and criterion 10 goes DIVERGENT (2026-08-02)

With the re-proved ledger baked in, Phi-3.5 claims **355 of 363 nodes**, `ledger_hits 355`,
`unproven_declines 2`, `unproven_forms_claimed 0`, `claimed_form_evidence ALL-PROVEN`. That is
Morpheus's original honest cost figure — *"Phi-3.5's claimed count goes 355 → 0"* — arrived at
from the other side, with every one of the 355 backed by a proof run rather than by a table.

The guard that mattered while getting here held: **the ledger grew by one entry and the claimed
count moved by 32.** A ledger that grows while the claimed count does not is the failure mode of
minting `/static/` keys the model cannot use; that is not what happened.

**This gave Switch's all-65 oracle arm its first reading on real data, and it is DISAGREE.**
Three consecutive runs, bit-identical to one another across all 65 outputs, one fused island per
run, 65 outputs reaching the EP, **0 CPU-only, 0 unobservable, 0 degenerate**. Not vacuous, and
reproducible.

`oracle_outputs_degenerate = 0` is the field to read first, and it says **the reopened
all-zero-KV defect is not reproducing here.** The KV outputs carry information on both sides.

The disagreement is `[0, 63, 64]` — `logits`, `present.31.key`, `present.31.value`. Reading the
ratio against the criterion actually applied (`|a-b| / (atol + rtol·|b|)`, which is ≤1 exactly
when the output passes), the pattern is not a broken layer:

| output | ratio to bound |
|---|---|
| `present.0.key` | 0.0438 |
| `present.1.key` | 0.127 |
| `present.15.key` | 0.233 |
| `present.30.key` / `.value` | 0.812 / 0.854 |
| `present.31.key` / `.value` | **1.66 / 1.17** |

**The pass/fail line falls in the middle of a smooth curve, not at a discontinuity.** The error
grows monotonically with layer depth and crosses the bound at the last layer. Layer 30 passing
at 0.85 is the same phenomenon as layer 31 failing at 1.66 — so the 62 passes are not 62 clean
results. This argues *against* relaxing the tolerance: relaxing it moves the crossing point and
leaves the curve rising.

`logits` is the outlier at 46.9× the bound, driven by near-zero logits where a `max_abs_diff` of
0.0625 is relatively enormous. It does not move the model's decision: `argmax 30751` on both
sides, top-10 10/10, over a CPU range of `[-13.086, 13.031]`.

**No tolerance was changed and no gate was relaxed.** The reading is recorded and handed on; the
verdict on whether f16 accumulation through 32 residual layers is acceptable, or whether the f16
kernels should accumulate in f32, is a correctness ruling and not this section's to make. The
per-output table is reproducible with `rust/tools/probe_phi35_oracle_detail.py`, which computes
no verdict and grants nothing.

---

### 8.9.13 The staged-op sweep: 21 promoted, 3 refused, and two holes in the harness (2026-08-02)

`epctl --dump-capabilities` reported **91 rows, 50 live, 41 staged**. The brief called it "42 ops
whose shaders compile and have never executed". The tool's own grouping is sharper, and it is the
first thing this section records because it changes what the deliverable could reach:

| staging reason | rows | dischargeable by a proof run? |
| --- | ---: | --- |
| `UNEXERCISED` — shader compiles, never executed | 22 | **yes** |
| `XL_KERNEL` — shader still being written | 13 | no: missing code |
| `NEEDS_PARAMS` — attribute selects an expression, not a value | 3 | no: missing code |
| `NEEDS_CAST_MATRIX` — variant space keyed on a dtype *pair* | 2 | no: missing code |
| `NO_SHADER` | 1 | no: missing code |

**Only 22 of the 41 were evidence problems. The other 19 are missing code**, and the 13
`XL_KERNEL` rows are almost exactly the `com.microsoft` contrib set — `MoE`, `QMoE`,
`MultiHeadAttention`, `RotaryEmbedding`, `CausalConvWithState`, `LinearAttention`,
`GatherBlockQuantized`, plus `Attention`, `QuantizeLinear`, `DequantizeLinear`. **The contrib-op
commitment cannot be advanced by proof runs at all.** That is the honest bound on this sweep and
it was established before any op was promoted, not after.

The dump did not carry the staging reason in its JSON form, so the table above would have been a
code reading rather than an artifact (R10). `staged_reason` was added to `dump_json()` first.

#### Promotion is not a promotion

`OpStatus::Live` is deprecated in favour of `Ready`, which means *"the kernel exists; claimability
is derived from the proof ledger"*. So flipping a row `Staged(UNEXERCISED) → Ready` **grants
nothing**: the form still declines `[unproven]` until a proof run mints its key. This is what made
a 22-row sweep safe by construction rather than a batch of bets — the ledger, not the flip, is the
gate, and an op whose proof run declines is reverted so its informative decline text is not lost.

#### The degeneracy guard, which had to land before anything was proven

Twelve of the 22 return **bool**. `compare()` had no guard on the CPU reference, and a *constant*
reference makes the comparison vacuous: two constant tensors agree to any tolerance. `Equal` on
two independent normals is all-False. `IsNaN` on a finite tensor is all-False. Both would have
reported `MATCH worst_rel 0.0` having tested nothing — **the cheapest way to "prove" twelve ops
was to prove none of them**, and it is the same failure mode this project has hit seven times.

The guard is `ERROR(instrument)`, not `DIVERGENT`: the kernel has not been shown wrong, the case
model has been shown inadequate. ERROR neither proves nor demotes. It is mutation-tested in both
polarities **in the lane** (`test_the_degeneracy_guard_fires_and_stays_silent`), because a control
that must be opted into is not a control:

```
A mutated (equal_f32 without its `discrete` domain) -> ERROR(case_model_degenerate_reference)
B as shipped                                        -> MATCH
```

Input domains were added alongside it: `discrete` (integers in [-3,3), so `Equal` collides),
`withnan` (NaN/Inf injected into the *input*, not the reference), and `bits` (full-width int32 —
the default `[0,2)` exercises one bit of thirty-two, which is not a test of `BitwiseAnd`).

#### The 3-input case that caught a claim/translate violation

`Sum`/`Mean`/`Max`/`Min` were deliberately given a **3**-input evidence case. All four raised:

> `EP_FAIL … 'Sum' with 3 inputs needs the chained-dispatch lowering, which is not written yet`

The claim predicate allowed 1..=8 inputs; the lowering handled ≤2. A 2-input case would have
proved the binary path, minted an entry and left the fold untested — and a 3-input node in a real
graph would have been **claimed and then failed at session creation**, which is an `EP_FAIL`, not
a decline. `MAX_VARIADIC_INPUTS_LOWERED` now exists and `templates::ew_variadic` reads the same
constant the predicate does, so the two cannot drift. `MAX_VARIADIC_INPUTS = 8` is retained as the
design target, separately named.

The four were then proved at **`n2` only**. Arity rides the key's last component, so an `n2` proof
can never be returned for an `n3` node: the fold stays unclaimable and declines honestly. §8.9's
key design vindicated a second time, after the `zero_points` case.

**These were the two predictions written down before the runs that turned out wrong** — the
predictions file called `Sum`/`Mean`/`Max`/`Min` MATCH and `Swish` MATCH. Being wrong in a
recorded way is what the file is for.

#### Result

21 of 22 ops now carry at least one proven form; the ledger went **74 → 95** entries. The census
reads **91 rows, 71 live, 20 staged**, with `UNEXERCISED` down from 22 to 1.

**The falsifier is the op-suite red count, not Phi-3.5's claimed count.** None of these 22 ops
appears in Phi-3.5, so quoting 355/363 here would be dishonest. `pytest tests/ops` went
**43 → 18 reds**, and of those 18:

* **7 were `XPASS(stale expect)`** — the GQA `xfail(strict)` whose own removal condition ("remove
  when alloc handles absent optional inputs and these tests produce MATCH") is now met, because
  Switch's two GQA fixes plus the ledger entry that lets the form be claimed put the whole file at
  `FAIL(condition): 0`. Marker removed, not relaxed. **Now 7 green.**
* **1 is pre-existing and not ours** — `test_census_baseline_has_no_drift`, drift
  `- bench/phases.py::load`. Attributed rather than assumed: it fails identically with this
  branch's changes stashed.
* **8 remain, and each is a documented refusal** (below).

The same 8, byte-for-byte, are the only op-table reds on **device 1 (Intel)**, so the 21
promotions hold on the spec-conformance oracle as well as on the device they were proved on.

#### The refusals — what will not go Live, and why

| form | why it cannot be proven |
| --- | --- |
| `Swish` f32 | `EXERCISED` vetoes it before the ledger is consulted |
| `Add` i32, `Mul` i32 | same: `ew_binary_add_i32.spv` / `ew_binary_mul_i32.spv` exist and compile |
| `IsInf` f32 | `NEEDS_PARAMS` — needs a shader variant, not a proof |
| `Cast` ×3 | `NEEDS_CAST_MATRIX` — needs a template, not a proof |
| `Flatten`, `Reshape` | **no row in the op table at all** |

**`EXERCISED` is a named criterion-11 residual.** It is a *second, hand-written evidence list*,
consulted by `only_proved_dtypes()` in `elementwise.rs`, which vetoes a claim per-dtype
**independently of the ledger** and is applied inconsistently (only to ops whose predicate calls
it). It is exactly the shape §8.9 exists to remove: a flag its author set, standing beside an
artifact-derived ledger. Because it runs *inside the predicate*, it vetoes before `claim_audit`
computes a proof key, so the generator offers the key, the claim fails first, and the case reports
no key at all — **no proof run can reach these three forms.**

`("Swish", "f32")` was deliberately **not** added by hand. Doing so would make the list assert
something no run has shown, which is the thing the ledger exists to stop. The proper repair is to
derive `EXERCISED` from the ledger and reduce the predicate's dtype veto to a *shader-existence*
question (a caps/translate fact) rather than an *evidence* question — deferred as its own change,
not folded into a promotion sweep.

Swish is not a coverage hole in practice: ORT decomposes opset-24 `Swish` into `Sigmoid` + `Mul`,
both claimed and both proven.

#### Two holes in the ledger's own machinery, found while using it

**1. `--reprove` did not re-measure anything against a healthy ledger.** The generator offers an
already-proven key through `CLAIM_UNPROVEN`, but `claim_audit` only records
`unproven_forms_enabled` when the ledger *misses* — so on a consistent build the admission set came
back empty and every re-proof reported `UNATTRIBUTED`. The 74-entry re-proof of §8.9.11 succeeded
only because the on-disk ledger had drifted from the baked copy and every lookup was `Faulted`:
**an accident of state, not a path.** A re-proof that silently measures nothing is §8.9.11's own
defect one level up. Fixed with a distinct witness, `reproof_forms_admitted`, deliberately *not*
folded into `unproven_forms_enabled` — that list is the §8.9.4 disclosure of forms claimed
*without* evidence, and naming a proven form there would be false and would also fail
`epctl --check-counters`. The two arms now differ observably:

```
already proven : admitted_via_hatch=[]     admitted_via_reproof=[Equal/…]  -> MATCH, entry replaced
not yet proven : admitted_via_hatch=[…]    admitted_via_reproof=[]         -> MATCH, entry written
```

**2. The default write was destructive, and reported `PASS` over the file it had just emptied.**
Existing entries were loaded only under `--append`; without it the generator rewrote the ledger
with nothing but what that invocation proved. A single-model run reduced 95 entries to 1 — and
then printed `PASS`, because `--check` was asked whether the file it had just written was
internally consistent, which an empty file is. **Two halves of a report describing different
things**, which is the defect Scribe had in her health report and the one that cost Morpheus a
criterion. Entries are now always carried forward; discarding them is `--rebuild`, which has to be
asked for. (This is written from experience: it destroyed 95 entries during this sweep, and they
were restored by re-running the 21 proof runs, not by reconstructing them from the attempts log.)

#### A test that asserted a stand-in

`families_that_are_not_a_one_line_body_change_are_still_staged` asserted `Staged` for the fifteen
families that are not a one-line body change away from `add_f32`. Once each had its own proof run,
the assertion was backwards: it would have **passed** for a row flipped to `Ready` with nothing
measuring it, and **failed** after a genuine proof. `Staged` was only ever a stand-in for "nothing
has measured this". The test now asserts the invariant that survives — that none of the fifteen
rides `add_f32`'s evidence, because each names itself in a ledger key.

---

### 8.9.14 Three questions about the writer: a destructive success, a witness nobody could produce, and a staleness the digest cannot see (2026-08-02)

Three findings arrived together, all of them about the *writer* rather than about any kernel, and
all three from Switch's round against GQA. They are recorded together because they are one shape
seen from three angles: **an instrument that reports on something other than what it did.**

**1. `--reprove` without `--append` rewrote the ledger from 74 entries to 1 and printed `PASS`.**

This is the second time the same defect has been fixed in this file, and the second time it
arrived through the flag added to fix the first. §8.9.11's fix made entries always carry forward,
which makes a drop *unlikely*. It does not make a drop a **detection**, and the difference is the
whole lesson: carry-forward is a property of one code path, and the next flag gets its own path.

`write_ledger()` now compares against what is on disk **before** it writes. A write that would
leave fewer entries than it found is `FAIL(condition)`, names the keys that would have gone, and
**writes nothing**. `--rebuild` is the deliberate instruction and has to be asked for by name.

The second half matters as much as the first: on a refusal, `main()` returns immediately and does
**not** run `--check`. Running it would ask about the file on disk — which is now the *old*,
perfectly valid one — and it would answer `PASS`. That is exactly how the original `PASS` came to
sit under an emptied ledger, and how `outputs_compared: 65` came to sit among oracle facts while
counting something else. A report whose two halves describe different things is the defect; the
repair is not to reword it but to stop the second half from being produced.

**2. Can a proof run produce `NO-SUBJECT-WITNESS`, or did Switch only catch it by hand?**

His first isolation arm read `calls=0 dispatches=0` — it ran entirely on the CPU EP and would
have reported *"GQA is innocent"* from a run GQA never entered.

The answer is that the harness refuses that state in three independent places: `disable_cpu_ep_
fallback` makes **ORT** raise at session creation (§8.9.10), `prove()` returns `UNATTRIBUTED` on
`not claimed or not dispatches`, and `entry_line()` refuses to write an entry whose attribution
or subject witness is empty. Switch reading the counters by hand was a fourth.

But that answer was, until now, **a reading of the source, and R10 says a code reading is not a
falsifier.** So both shapes are now planted in the lane — a record with no attribution and a
record with attribution but no nameable shader — and the refusal is asserted, alongside a healthy
record that must still be written, because a test that refuses everything passes for free.

The distinction worth keeping: his arm was a **probe**, and probes have no writer to refuse them.
The guards protect the ledger, not every script that touches the EP. `NO-SUBJECT-WITNESS` is the
right name for the state a probe can be in, and naming it does not make a probe check for it.

**3. Could any of the 95 entries have been proven against stale inputs?**

Switch found the input cache was keyed on `(cpu_ptr, byte_size)` with a 32 KiB floor and no
content check, so every inference after the first in a session read the first one's inputs. All
95 entries were minted on a binary carrying that defect.

**The answer is none, and it is clean.** The defect requires a *second* `Compute()` in the same
session. Every proof arm is a fresh subprocess that builds one session and calls `sess.run`
exactly once — `sess.run` appears once in the whole harness, with no loop — and the EP's own
counters from a proof run agree: `compute_calls: 1`, `weight_cache_release_buffers: 0`,
`weight_cache_bytes_resident: 0`. There is no second inference for a stale cache to serve.

That is a true statement about today's harness and a fragile one about tomorrow's. A future case
needing two inferences — a KV-cache form is the obvious one — would reintroduce the exposure and
its entry would look identical, because **the shader digest cannot see it: the shaders do not
change.** This is the residual §8.9.11 named explicitly and does not cover — *host-side numeric
changes leave the entry green* — and this is the first concrete instance of it.

So the immunity is now a field rather than a habit. `compute_calls` is recorded in each new entry,
and `entry_line()` refuses a run that computed more than once until somebody has decided what a
multi-inference proof means. The prose answer above is correct today; the field is what keeps it
correct without anyone re-deriving it.

**What none of this closes.** The digest still covers SPIR-V only. Host-side changes — descriptor
layout, dispatch dimensions, the cache key that started this — leave every entry green. `compute_
calls` narrows that gap by one known instance; it does not close the class.

**One audit taken on the way past.** Switch's other transferable finding — *a classifier tested
only against strings we wrote ourselves is tested against the one input it cannot get wrong* — was
turned on this harness's only foreign-string classifier, the ORT CPU-fallback refusal match. It
holds, for a reason worth stating rather than a lucky one: **its miss produces an
`ERROR(instrument)`, not a verdict.** If ORT rewords the message the match fails, the run raises,
and the lane goes red — it cannot become a decline or a proof. A classifier whose miss produces a
verdict is the one to distrust; a classifier whose miss produces a raise costs a red lane.

---

### 8.9.15 A counter inserted mid-struct, and three mirrors that went on reading (2026-08-02)

Found while re-running the census after the merge. `test_wiring_census.py` reported:

```
partitioner: UNWIRED (dispatches_executed delta = 0 — EP ran nothing)
```

on a run that also reported `claimed=1, islands=1`, `compile_calls=1`, `compute_calls=1`, and,
from the counters *file*, `dispatches_executed=6`. Two artifacts from the same run disagreeing is
not a partitioner fault; it is a reader fault.

**The cause.** `a52024f` added `device_losses` to `VulkanEpCounters` and **inserted** it between
`compute_failures` and `dispatches_executed` rather than appending it — which the struct's own
doc comment forbids in the line directly above it: *"Fields are never removed or reordered."*
`COUNTERS_ABI_VERSION` was not bumped, and the three hand-written `ctypes` mirrors
(`test_wiring_census.py`, and two separate copies in `test_phi35.py`) kept the old layout.

Everything below the insertion shifted by eight bytes. The mapping was exact and silent:

| the mirror called it | it was actually reading |
| --- | --- |
| `dispatches_executed` | `device_losses` (always 0 on a healthy run) |
| `viable_islands_retained` | `dispatches_executed` |
| `unproven_declines` | `ledger_hits` |
| `ledger_entries` | `unproven_declines` |
| `unproven_forms_claimed` | `ledger_entries` (95) |

**Why nothing went red.** `device_losses` is `0` on every healthy run, so the census's criterion-8
number was a *stable, plausible, entirely wrong* `0` — Switch's `STEADY 21.4×` shape and his
`arms_must_differ` lesson, arriving through a struct offset. And the failure it produced was a
**false `UNWIRED`**: the census's most serious verdict, issued about a mechanism that was working.
Worse, `unproven_forms_claimed` read `95`, i.e. the ledger's *entry count* was being reported as
the number of forms claimed through the escape hatch — a number that is supposed to be `0` and
whose being non-zero is what `epctl --check-counters` fails on.

**The repair.** `COUNTERS_ABI_VERSION` is bumped to 4; all three mirrors carry `device_losses` in
its true position; and the census reader now checks `abi_version` and `struct_size` against the
running DLL and **raises** on a mismatch. The raise is deliberately outside the `try` that returns
`{}` for a missing library: `{}` becomes a delta of `0`, and a delta of `0` reads as `UNWIRED`.
Swallowing a layout mismatch into the same empty dict as a missing DLL is what let this hide.

**The decision worth stating, because it will be argued with.** The new lane test asserts
`struct_size` **equality**, not `>=`. The struct's documented contract is that an old reader can
read a new struct, and equality throws that away: every future counter added turns this lane red
until a human updates the mirror. That is the intended cost. From the reader's side an *append*
and an *insertion* are indistinguishable — both produce a larger struct and a bumped version — and
only one of them is safe. A check that cannot tell them apart has to assume the dangerous one.
The narrower alternative, a per-field offset manifest published by the DLL, is defensible and
strictly better; it is more machinery than this defect justifies today, and equality can be
relaxed the day that manifest exists.

**What this does not cover.** The two `test_phi35.py` mirrors are repaired but not guarded — they
have no equivalent version check, and they are copies of a struct that will move again. Three
hand-maintained mirrors of one C ABI is the actual defect; the honest fix is one shared reader,
and it is not mine to land unilaterally. Filed rather than done.

**Attribution, per R13.** This is not a detection of a kernel problem and it was not caught by any
gate. It surfaced because the census reported a mechanism `UNWIRED` that other artifacts from the
same run showed working, and the contradiction was followed instead of being explained away. Every
number any ctypes reader took between `a52024f` and this commit is suspect — including the
Phi-3.5 lane's `dispatches_executed`, which was reading `device_losses` throughout.

---

### 8.9.16 The second evidence list, and what was left when it went (2026-08-02)

The op suite came into this round at **11 red**. Eight were `test_op_table`, and the coordinator's
instruction was to separate them rather than batch them, because they are not one thing. They are
four things, and only one of them was ours to fix by proving something.

**The one that was a defect: `EXERCISED`.**

`Add-i32` and `Mul-i32` declined `[dtype]`, with the text *"that variant of the elementwise shader
compiles but has never executed on a device."* True, and unfixable — by design, though not by
anyone's intent. `elementwise::EXERCISED` was a hand-written `&[(&str, &str)]` of `(op, dtype)`
pairs that had run on a device, and `only_proved_dtypes` consulted it **inside the claim
predicate**. The claim predicate runs *before* a proof key is computed. So the sequence was:

1. `gen_proof_ledger.py` offers `add_i32.onnx` to the EP;
2. the predicate vetoes at `[dtype]`, before any key exists;
3. the run reports `no key at all` — not `[unproven]`, which the generator can unlock, but a
   decline it cannot see past;
4. `session.disable_cpu_ep_fallback=1`, which we adopted precisely so that a single-form case
   cannot silently prove nothing, then makes ORT refuse the session outright.

**The form was unproven because it was unproven.** The only exit was to type the pair into
`EXERCISED` by hand, which is the exact act — a claim widened by an assertion nobody measured —
that §8.9 exists to prevent. Three forms sat in that loop: `Add`/i32, `Mul`/i32, and `Swish`/f32,
which was reverted to `Staged` in §8.9.13 for this reason and documented there as a finding.

This is criterion 11's own shape arriving from inside criterion 11's own module: a second,
older answer to the question the ledger was built to answer, still wired in front of it.

**The split.** The list was answering two questions with one table:

| question | who answers it now | why |
| --- | --- | --- |
| *Does a kernel exist that this engine can create?* | the claim predicate | claiming a node whose module cannot be instantiated is an `EP_FAIL` at translate time, not a decline, and no ledger entry could make it safe |
| *Has anything measured this form?* | the proof ledger | a harness-generated entry naming artifact, device, shader digest and observed `worst_rel` — provenance a typed pair cannot have |

`only_proved_dtypes` is now `only_loadable_variants`, and the residual the list *did* carry
honestly is derived from the artifact instead of asserted. `variants::variant_is_loadable(stem)`
looks the stem up in `engine::shaders::SHADER_MODULES` and checks that every SPIR-V capability the
module declares is one `ENGINE_ENABLED_CAPABILITIES` contains. Today that refuses every `_i64`
variant: they declare `Int64` (11), which needs `VkPhysicalDeviceFeatures::shaderInt64`, and
`vk::device` passes no `pEnabledFeatures` at all. A device-lost on a user's machine becomes a
decline, computed rather than remembered.

`EXERCISED` and `TEMPLATE_LIVE` are **deleted**, not kept as documentation. A list nobody consults
is the next stale thing, and the deleted-here comment carries the reasoning that the lists carried.

**What the tests do now.** `no_live_claim_rests_on_an_unloadable_variant` used to scope itself by
`proved_at` — it looked only at pairs somebody had written down, which meant it could not see the
forms most at risk, the ones nobody had thought about. It now walks every dtype each live row's
`caps` accept and asserts the *predicate* refuses the unloadable ones. It also asserts
`refused > 0` (R12): a zero would let this test pass for the wrong reason the day the `_i64`
variants stop being generated, which is the same shape as `bypassed` and `all-rejected` sharing
one `0`. `every_template_live_row_stands_on_a_representative...` is deleted along with the list it
guarded; every row that stood on a representative's evidence now stands on a ledger entry of its
own, keyed on its own dtypes and its own shader digest, so there is no borrowed claim left to
invalidate.

**Stated before the run, per R10.** The prediction written down before `--append` was: `add_i32`
and `mul_i32` each offer exactly one unlockable key and clear it; `swish_f32` offers **no** key,
because the `Swish` *row* is still `Staged` and that is a separate gate from the list. The run:

```
[discover] add_i32.onnx:   1 unlockable key(s)
    would decline without a proof: ai.onnx::Add/7+/i32,i32>i32/ew_binary_add_i32/static/n2
[prove]    add_i32.onnx:   MATCH worst_rel 0.0  claimed_nodes 1  dispatches_executed 1
                           shaders_dispatched ['ew_binary_add_i32']  compute_calls 1
[discover] mul_i32.onnx:   1 unlockable key(s)
[prove]    mul_i32.onnx:   MATCH worst_rel 0.0  claimed_nodes 1  dispatches_executed 1
                           shaders_dispatched ['ew_binary_mul_i32']  compute_calls 1
[discover] swish_f32.onnx: 0 unlockable key(s)
[prove]    swish_f32.onnx: UNMEASURED  no unlockable keys on this model
```

`worst_rel 0.0` on integer arithmetic is exact agreement, not a degenerate comparison: the
generator's `case_model_degenerate_reference` instrument passes, `dispatches_executed=1` and the
shader is named, so the arm the EP ran is identified rather than assumed. Ledger **95 → 97**,
digest `6180b5a3f7d498fb` → `eb7c4e1f90cd7ec2`. `test_op_table` **8 red → 6 red**, and both new
greens hold on device 1 as well as device 0.

**The guard, applied to myself.** *A ledger that grows while the claimed count does not move.* It
did not move: Phi-3.5 reads `claimed_nodes 355`, `ledger_hits 355`, unchanged. That is the correct
outcome here and worth saying why rather than leaving it to be noticed — the two new forms are
i32 at static extent and Phi-3.5 is an f16 model with no i32 elementwise node, so no key of theirs
can be looked up on it. The falsifier that *did* move is the op suite, which is the surface these
two forms exist on.

**`Swish` is released but still staged, and the distinction is the point.** The list no longer
holds it; the row does. `Swish` is ai.onnx opset-24-only and ORT decomposes it into
`Sigmoid` + `Mul` on every graph we have, both claimed and both proven, so no model in this
repository can produce a `Swish` node for a proof run to measure. It stays `Staged(UNEXERCISED)`
because nothing can exercise it — an ordinary staging decision — rather than because a list forbids
it. The test that recorded the trap now records the release and asserts `ew_unary_swish_f32` is
loadable, so if flipping it ever becomes possible the failure will be about the graph, not the
kernel.

**The other three shapes, reported rather than proved.** Each decline below is quoted from the
running build, not from the source.

- **`clip_no_bounds` — not a claim-predicate defect.** The coordinator's suspicion was that
  `[arity] `Clip` has 1 inputs; this handler takes exactly 3` is a handler requiring inputs ONNX
  makes optional, and that it would keep declining however many forms we prove. The second half is
  right; the first half is not, and `claim::ew_clip` already says why: Clip's bounds are optional
  from opset 11, the three-input form rides the ternary template with zero-stride broadcast, and
  *"the fix is a shader variant that substitutes ±infinity for the omitted bound, not a widening of
  this predicate: an omitted bound is a different dispatch shape, not a different value, and
  claiming it here would bind a buffer that does not exist."* Widening the predicate would claim a
  node and then bind a descriptor for a tensor with no producer. It is a **coverage gap with a
  written repair**, and the repair is a shader variant, which is engineering rather than a proof
  run. Filed, not fixed here.

- **`Cast` ×3 — `NEEDS_CAST_MATRIX`, and the staging reason is exact.** *"its shader variant space
  is keyed on a source/destination dtype pair rather than a single dtype, so it needs its own
  template and manifest column."* Every other row in the variant table is keyed on one dtype; Cast
  is the only op whose stem is a *pair*, so it needs a manifest column that does not exist and a
  template that does not exist. This is the largest of the four pieces of real work here and it is
  not a proof-run problem: there is nothing to measure until the variants exist.

- **`IsInf` — `NEEDS_PARAMS`, and it is the selector case, not the coefficient case.** *"its
  attribute selects a different expression rather than supplying a value, so it needs its own
  shader variant rather than a push-constant parameter."* `detect_negative`/`detect_positive` are
  four combinations of two booleans, i.e. four bodies, not one body with a uniform. The push-
  constant path that serves `Elu`'s `alpha` cannot serve it.

- **`Flatten`, `Reshape` — `[not-registered] no Vulkan handler is registered`, and this may be
  correct.** These are shape-only ops: the output is the input's bytes under a different view. A
  lone shape op claimed in a one-node island buys **nothing** — the transfer to and from device
  costs more than the zero arithmetic performed, and the net-benefit gate should and would decline
  it. Their only value is *not breaking an island* that would otherwise be contiguous. So the
  question they ask is not "can we compute this" but "does a real graph have one of these between
  two islands we do claim", and on Phi-3.5 the answer today is no — its 8 remaining unclaimed
  nodes are the `island-output-consumed-internally` branch in `vk/session.rs`, which is a different
  finding and not ours. Registering them to turn a test green, absent a graph that needs it, would
  be widening the claim table for the suite's benefit rather than a model's. **Deliberately not
  done**, and the test staying red is the honest report of that.

**Count, stated plainly.** Two ops proven and promoted. Six red tests remaining across four
distinct pieces of work, none of which a proof run can reach: three need a shader variant that does
not exist (`Clip`, `IsInf`, and the Cast template), and two are a design question about shape-only
ops that a real graph has not yet asked. After today's round I would rather report four written
refusals than six promotions nobody can falsify.

**Still open, and it should not disappear into a history file:** *three hand-maintained ctypes
mirrors of one C ABI is the real defect; one shared reader crosses domains.* §8.9.15 caught the
mirrors drifting once. Making `struct_size` equality a hard assertion means the next drift is loud,
but loud in three places is still three places.

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

**Status 2026-07-28: landed.** The machinery below is implemented in `rust/src/registry.rs` and
`rust/src/ops/`; the shipped shape differs from the sketch in two ways, both improvements, and
this section now documents the real thing.

```rust
pub struct OpSpec {
    pub domain:    Domain,          // Ai | Ms  (typed, not a &str — prevents "" vs "ai.onnx" bugs)
    pub op_type:   &'static str,
    pub min_opset: i32,             // OPSET_ANY sentinel
    pub max_opset: i32,
    pub caps:      DTypeSet,        // generates: claim check, shader variants, support matrix
    pub kernel:    Kernel,          // template + template-op + the &'static str variant stems
    pub claim:     ClaimPredicate,  // fn(&NodeView, &OpSpec) -> Result<(), DeclineReason>
    pub translate: TranslateHandler,// fn(&OpSpec, &NodeDesc, &mut dyn DispatchContext) -> EpResult<()>
    pub status:    OpStatus,        // Live | Staged(&'static str)
}
```

Change 1 — **both function pointers take their own row.** That is what lets *one* predicate serve
sixty ops: it reads `spec.caps`, `spec.op_type` and `spec.kernel` instead of hard-coding them.
Without it, "adding an op is a row" is false, because every row needs a bespoke closure.

Change 2 — **`OpStatus::Staged`.** The claim/translate invariant is absolute, but the table is
worth landing before the shaders are. A staged row is fully described and fully claim-tested, and
`claim_decision` declines it with `[staged] …`. Flipping an op live is a one-word diff. This is
what makes it possible to build the machinery before op #1 without ever claiming a node we cannot
translate.

Adding `Add` is exactly this line:

```rust
"Add", Ai, 7 ..= OPSET_ANY, NUMERIC, kernel!(EwBinary, "add"),
       claim::ew_binary, templates::ew_binary, Staged(NO_SHADER);
```

### 9.2.1 Decline reasons are machine-readable by construction

`ep.rs` owns the `DeclineReason = Cow<'static, str>` seam, so widening it to a struct would be a
cross-owner ABI change for no benefit. Instead every reason is built by `registry::decline` and
rendered as `"[tag] sentence"`, where the tag is a `DeclineCode` from a closed set
(`not-registered`, `staged`, `opset`, `arity`, `missing-input`, `dtype`, `rank`, `shape`,
`dynamic-shape`, `attribute`, `partition`, `internal`). `DeclineCode::of_reason` parses it back.
One construction, three consumers: `CLAIM_DEBUG` prints the sentence, Trinity's harness asserts the
code, Niobe histograms it. A reason that did not come from `decline` returns `None` and buckets as
"other", which is why the histogram needs that bucket.

#### 9.2.2 The claim record must follow the environment, not latch it (2026-07-29)

`ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` names a file the registry appends one JSON line to per claim
decision. Its path was read through a `OnceLock` — read once per process — on the reasoning that an
environment variable is set before a process starts and does not change.

That reasoning is wrong for the only caller the record exists to serve. A pytest process loads the
EP once and then runs hundreds of tests, and `_models.is_vulkan_claimed` sets the variable **per
call**, around a single probe session. So the first claim decision in the process — made by
whichever fixture happened to create a session first — latched `None`, and every probe afterwards
found no file. The reader treats a missing file as *not claimed*, conservatively and reasonably.

**The failure therefore did not look like a broken diagnostic. It looked like a claim result.**
`test_barrier_parity` skipped with *"Add is not yet Ready — VulkanExecutionProvider did not claim
this node form"* in the same run in which `test_claim_diagnostics::test_add_is_claimed` passed on
`Add`. Both sentences were plausible; only one was true. Reproduced deterministically:

| env var set | `Add` at `[4,4]` | `Add` at `[3,4]` |
|---|---|---|
| before the first session | claimed | claimed |
| after one session had run | "not claimed" | "not claimed" |

The node form was never the difference. Both forms claim. The difference was the **mechanism**:
`assert_vulkan_claims` reads ORT's profiling JSON, which is ground truth about execution;
`is_vulkan_claimed` reads our record, which was silently disabled.

Three things follow, and the third is the general one:

1. **Fixed in `ops/claim_log.rs`:** the path is re-read per decision and the open handle is stored
   *with* the path it belongs to, so pointing the variable at a different file mid-process reopens.
   Cost is one `getenv` per node during `GetCapability`, which is nothing beside the schema lookups
   the same call already does. Pinned by a unit test that writes to two paths and back.
2. **`test_domain_regression.py` (C1) uses the same mechanism** and was therefore degraded the same
   way — it has an "if the EP wrote a log" guard, so it was passing without asserting. Flagged to
   Trinity; the fix restores it. A guard that silently weakens an assertion when its input is
   missing is the same shape of hazard as the skip.
3. **A diagnostic whose failure mode is indistinguishable from a negative result is not a
   diagnostic.** The record's own docs argued for JSON Lines flushed per record precisely so that a
   reader needs no lifecycle knowledge — and then the enablement check acquired exactly the
   lifecycle dependency that argument was avoiding. The absent-file case should have been
   distinguishable from the not-claimed case at the file level; it now is, because the file is
   always created when the variable names one.

**What actually caught it was two of our own tests disagreeing.** Neither was individually
suspicious: a skip saying "not Ready yet" is expected for most of the table, and a passing claim
test is expected for `Add`. It is worth noting that this is the second defect this session found by
cross-checking one artefact against another rather than by reading either on its own — §4.18's rule
in a new setting.

### 9.3 Capability reporting

Generated, not written. `cargo xtask op-matrix` walks the registry table and emits
`docs/OP_SUPPORT.md`:

| Op | Domain | Opsets | f32 | f16 | i32 | i64 | bool | Notes |
|---|---|---|---|---|---|---|---|---|
| `Add` | ai.onnx | 7+ | ✅ | ✅ | ✅ | ✅ | — | full broadcast, rank ≤ 6 |

CI fails if the checked-in file differs from the generated one. Same table backs
`--dump-capabilities` for Trinity's assertions and Link's `PLATFORMS.md` rows.

### 9.4 Contrib schema versioning — the hazard that comes with the domain

Admitting `com.microsoft` (user ruling, 2026-07-28) buys the Qwen3.5 path and imports one liability:
**contrib ops have no opset guarantee.** For an `ai.onnx` op, `min_opset ..= max_opset` is a complete
compatibility statement, because ONNX freezes a schema once published and issues a new version for
changes. Every `com.microsoft` op is `since_version = 1` forever; it is versioned by *ORT release*,
and inputs and attributes are added **in place**. `LinearAttention` and `CausalConvWithState` do not
exist in the pinned 1.28 release at all — they are main-branch ops.

The failure mode that matters is not "the op disappears" (that is a clean miss). It is: the schema
gains an attribute whose default changes the math, ORT materializes it on every node, our predicate
does not know the name, and we claim the node and compute a **confidently wrong answer**.

**Mechanism, implemented in `rust/src/registry.rs` (landed 2026-07-28):**

```rust
pub struct ContribSchema {
    pub baseline: sys::SchemaBaseline, // which ORT release, and when a human checked
    pub notes: &'static str,           // confidence, and what to re-verify
    pub min_inputs: usize, pub max_inputs: usize,
    pub min_outputs: usize, pub max_outputs: usize,
    pub required_attrs: &'static [&'static str],
    pub known_attrs: &'static [&'static str],
}
```

- Every `Domain::Ms` row **must** carry one — enforced by a unit test, not by review.
- `ContribSchema::check` runs **before** the staged check and before the claim predicate, so a
  schema that has moved declines even for an op we would otherwise have claimed.
- The load-bearing detector is **attribute-name enumeration** via `Node_GetNumAttributes` /
  `Node_GetAttributes` / `OpAttr_GetName`. ORT materializes defaulted optional attributes, so the
  observed name set is the *effective* schema. A name outside `known_attrs` ⇒ decline
  `[contrib-schema]`.
- `DeclineCode::ContribSchema` is a distinct bucket from `DeclineCode::Attribute` on purpose.
  `[attribute]` means *a value we chose not to support*; `[contrib-schema]` means **the schema moved
  under us**. The first is a backlog item; the second is an alarm. Conflating them would hide the
  alarm inside the backlog.
- **Failure direction is deliberate.** A too-narrow fingerprint costs a decline and a CPU fallback,
  which is always correct. A too-wide one produces a wrong answer. Fingerprints are therefore
  written narrow, their confidence is recorded in `notes`, and the `[contrib-schema]` histogram
  bucket is the evidence used to widen them.

**Layering constraints this satisfies** (`DESIGN.md` §1.4): **C1** — no domain-wide contrib opt-in;
the bare string `"com.microsoft"` appears only at its definition site, rows name the domain as a
table token and keys are always qualified. **C2** — every contrib row states the ORT release its
predicate was verified against and surfaces it, via `OpSpec::schema_baseline()` and the
`epctl --dump-capabilities` "schema baseline" column.

**Reconciliation, resolved 2026-07-28.** An earlier revision noted that `sys::CONTRIB_SCHEMA_BASELINES`
(Tank's) duplicated the pinned-release record that the contrib rows also carry, and that a registry
test papered over the disagreement by comparing only verification *dates*. Tank resolved it in the
direction I would have chosen but could not reach from my own files: he **deleted the `sys` table**,
making `ContribSchema.baseline` on the row the single source of the C2 baseline, and replaced the
cross-check with a test asserting exactly that. Reviewed and approved — one record, owned by whoever
writes the predicate it describes.

### 9.4.1 What C2 does **not** detect — behavioural drift without a version change

C2 is version-based by construction. `ContribSchema` compares an observed node against a shape
pinned to a named ORT release; `min_opset ..= max_opset` compares a node against a named ONNX
schema version. Both answer the question *"has the declared interface moved?"* Neither answers
*"has the meaning of an unchanged interface moved?"*

That second thing happens, and we hit it on the first op we looked at closely. `ai.onnx::Attention`
at opset 24 had a defective reference implementation — top-left instead of bottom-right causal
alignment when `nonpad_kv_seqlen != q_sequence_length`, and NaN for fully-masked rows. Justin ruled
on 2026-07-29 that it is **fixed in place, with no opset bump**:

> 不bump opset了 不然兼容性很麻烦还要维护一个错的def 你按照正确的实现就行了

That ruling is right and it is the one I would want as a consumer — bumping would fragment
compatibility across every producer and oblige ONNX to maintain a definition known to be wrong.
Its consequence for us is that **`ai.onnx::Attention`-24 is defined by the corrected semantics,
full stop.** No dual path, no legacy variant, nothing to gate on an onnx version inside a claim
predicate. Our narrowing to `23 ..= 24` with input 6 declined stands unchanged and on its own
merits: we decline `nonpad_kv_seqlen` because we do not implement it, not because its meaning was
ever in doubt.

But the *class* of event is worth naming, because our machinery is blind to it:

> **Blind spot:** a correction applied to an operator's semantics without a version change is
> invisible to opset-based and release-baseline checking alike. Every number our detectors compare
> stays identical across the change. `ContribSchema` will not fire. The opset window will not fire.
> The only signal is a differential test against a *pinned* reference — and if the reference is
> unpinned, the drift presents as a regression in our kernel rather than as a change in the spec.

Three consequences, recorded rather than solved:

1. **This is not a contrib-domain problem.** §9.4 exists because `com.microsoft` has no opset. This
   defect class attacks `ai.onnx` too, and `ai.onnx` is the domain we treat as the safe one. The
   opset window is a strong guarantee about *interface* and no guarantee at all about *behaviour*.
2. **The oracle version is a correctness input, not test hygiene.** The same opset-24 graph yields
   different expected outputs under onnx 1.22 and onnx 1.23, and the model carries no signal to
   distinguish them. Routed to Trinity: pin `onnx` in the harness and in CI beside the ORT pin,
   with the reason written next to the number. This is the `accuracy_level` argument one layer out.
   Noted in passing that `requirements.txt` pins `onnxruntime>=1.28` — whether `onnx` is pinned at
   all or arrives transitively is hers to check, and a transitive dependency is not a pin.
3. **We cannot detect it ourselves; we can only be told.** There is no fingerprint to write. What
   is actionable is the human process: Fact Checker is establishing whether this class recurs in
   other ops we claim, and any such finding belongs in the per-row `notes` field, which is the one
   place a fingerprint can carry a fact its own structure cannot express.

**How this surfaced, because the method is the transferable part.** Nothing detected it. It came
out of following §7.1's rule literally — *when the schema moves under us, narrow the predicate and
decline, never guess* — and therefore going to read what actually changed between opset 23 and 24
rather than trusting that an open-ended window was harmless. The interface change (input 6) is what
the rule was aimed at; the errata was sitting next to it in the same source file. With the original
`23 ..= OPSET_ANY` window we would have claimed the node and returned plausible wrong logits, and
we would have found out from a benchmark that looked slightly off. The rule paid for itself on its
first real application, and it paid off *twice* — once for the thing it was designed to catch, and
once for a thing it was not.

### 9.4.2 What confirming the opset range does **not** protect against

§4.20 confirmed that our closed windows cover every published schema version of every op we claim.
That is worth having, and it is worth being precise about what it buys, because "we support the
full opset range" reads as a stronger statement than it is.

An opset window is a guarantee about a **declared interface**: the inputs, the attributes, the type
constraints. Confirming its upper bound proves that no *interface* has moved above our bound
unread. It proves nothing about meaning. And ONNX has corrected the meaning of `Attention` **five
times without a single version bump**:

| PR | First released | What changed | Opsets | Bump? |
|---|---|---|---|---|
| onnx#7297 | v1.19.1 (2025-10-10) | Causal mask wrongly blocked past KV positions | 23, 24 | No |
| onnx#7274 | v1.20.0 (2025-12-01) | GQA key/value repetition changed from tiling to `repeat_interleave` | 24 | No |
| onnx#7867 | v1.22.0 (2026-06-15) | Softcap applied *after* mask/bias instead of before — masked positions got nonzero softmax mass | 23, 24 | No |
| onnx#7913 | v1.22.0 (2026-06-15) | `qk_matmul_output_mode` values 1 and 2 swapped meaning | 24 | No |
| onnx#8068 | **unreleased** | `is_causal` bottom-right alignment on external KV cache; NaN guards | 23, 24 | No |

`RotaryEmbedding`-23 has a sixth instance (onnx#7313, fixed in v1.19.1: reference implementation
only, spec unchanged). Every one of these is invisible to opset windows and to `ContribSchema`
alike, by construction — §9.4.1.

Three things follow, and one of them is encouraging:

1. **Confirming the range slightly increases exposure.** We now hold windows covering every
   published version of six ops. Every additional version inside a window is additional surface for
   an in-place correction we cannot see.
2. **Conservative claiming is the only structural defence, and it worked.** onnx#7913 swapped the
   meaning of `qk_matmul_output_mode` 1 and 2 with no version change — and it cannot affect us,
   because `std_attention` claims only `qk_matmul_output_mode == 0`. Declining every non-default
   value of an attribute is not just a scope decision; it is *immunity to that attribute's semantics
   changing*. This is a real argument for narrow predicates that I had not previously made:
   **the attributes you decline cannot drift under you.** Where §7's narrowness was justified on
   correctness-of-implementation grounds, it turns out also to shrink the C2 blind spot.
3. **The residual is the attributes we do claim.** For `Attention` that is the causal mask
   (onnx#7297, onnx#8068) and GQA repetition (onnx#7274) — both squarely inside what we intend to
   implement, both silently corrected. There is no fingerprint that catches this. The only
   instrument is Trinity's differential suite against a *pinned* onnx, which is why the pin is a
   correctness input and not hygiene.

### 9.5 Engine dependencies this plan creates

Recorded here so they are routed rather than discovered late:

1. **Prepack seam** — §8.2.1, P1–P6. Blocks T4.
2. **Indirect dispatch** — `QMoE` with device-computed workgroup counts needs a `KernelRequest`
   variant whose group count comes from a device buffer (`vkCmdDispatchIndirect`). Not needed for
   the masked-dense first implementation; needed before MoE is *fast*. Blocks T5b performance, not
   T5b correctness.
3. **In-place KV-cache aliasing** — `GroupQueryAttention` must write `present` into the same
   allocation it read `past` from, or every token pays a full cache copy. Needs the engine to allow
   an output binding to alias an input binding for a declared-safe op. Blocks T3 usefulness.
4. **`build.rs` consumes `src/ops/shader_variants.txt`** — §5.5. Blocks nothing today (the table and
   manifest are checked in and tested); blocks the first real shader.

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

### 10.1 Quantized paths need a different test story than fp32 — requirement for Trinity

int4/int8 block-quantized math is not "fp32 with a looser epsilon". It has two regimes and they need
two policies. Stating them as a requirement because the wrong policy here either passes a broken
kernel or fails a correct one, and both are expensive to discover late.

| Layer under test | Policy | Rationale |
|---|---|---|
| **Unpack + dequantize** (`b_packed`, `scales`, `zero_points` → dequantized block) | **Bit-exact.** `assert_array_equal` against a NumPy reference, no tolerance. | Nibble extraction, block indexing and zero-point subtraction are *integer* operations followed by one exact multiply. Any difference is a bug — an off-by-one in block indexing, a nibble-order swap, or a signed/unsigned mistake. A tolerance here hides exactly the bugs that are hardest to find downstream. This is also where a wrong answer looks *almost* right. |
| **`MatMulNBits` output** vs ORT CPU EP on the same graph | **Relative tolerance, accumulation-order aware:** `rtol = 2e-2`, `atol = 1e-3` for fp16 activations; `rtol = 1e-3` for fp32 activations. | Our GEMM's reduction order differs from CPU's by construction (tiles + subgroup reductions), and fp16 accumulation is not associative. The tolerance must cover reassociation, *not* cover a wrong dequant — which is why the layer above is bit-exact and this one is not the only check. |
| **End-to-end logits** (decode step) | **Top-1 token agreement over ≥ 64 greedy steps**, plus KL divergence of the softmax distribution below a fixed bound. Not per-element tolerance. | Per-element logit tolerance on a 150k vocab is meaningless — it either passes trivially or fails on one outlier. Token agreement is the property users care about and it is the only end-to-end assertion that survives a legitimate reassociation. |
| **Per-layer intermediates** | Same rtol as the op-level row, captured layer-by-layer. | §6.1 item 5: on a 1.7B model, final-logit comparison cannot localize a fault. The harness needs intermediate capture or T4 debugging becomes bisection by hand. |

**The oracle — CONFIRMED, with two constraints (Trinity, 2026-07-28).** The obvious reference was
**ORT CPU EP running the identical quantized graph**, so that dequantization semantics come from ORT
rather than from our reading of them, and I asked for that to be verified empirically rather than
assumed. Trinity ran the check and the answer is **yes, the CPU EP is usable as a `MatMulNBits`
oracle**. Two findings attach, and both are exactly the class of thing that would have silently
poisoned reference values if we had assumed instead:

1. **`accuracy_level` must be pinned, and is pinned at 1.** Level 4 (int8 VNNI) diverges from levels
   0–3 by ~3.6e-3 at K=1024, N=512. That is above the `rtol = 1e-3` this table asks for on fp32
   activations, so an unpinned oracle would have produced reference values that changed with the
   *runner's CPU*, and the resulting failures would have looked like GPU bugs. The oracle's
   accumulator type is part of the test configuration, not an implementation detail.
2. **The fp16 oracle is version-conditional.** fp16 activations produce NaN/Inf on ORT **1.27** —
   the same null-allocator `PrePack` defect that forced the version pin in the first place. The fp16
   oracle test is therefore gated on ORT ≥ 1.28 and runs in CI. So the fp16 row of this table is
   real but conditional, and any future support window widening below 1.28 removes it.

The bit-exact dequant row is compared against **NumPy, not the CPU EP**, and deliberately so: the
whole point of that row is to catch a misreading of the block layout, and comparing ORT against ORT
would let a shared misreading pass unnoticed. The CPU EP is the right oracle for the *arithmetic*;
it is the wrong oracle for the *schema*. Both are implemented.

Secondary asks: the bit-exact dequant test should run the *same* helper the blocked
`DequantizeLinear` path uses (§8.3), so the two cannot drift; and quantized tests need their own
tolerance config knob rather than inheriting the fp32 harness default.

### 10.2 The decline reason is an output, not a log line

Trinity's C1 regression test could originally only assert *structurally* — that the EP claimed zero
nodes — because `DeclineCode` was invisible outside the process. A zero-node assertion cannot tell
apart "declined because nothing is registered for it" (what C1 is actually about), "declined for
some unrelated reason" and "crashed before reaching the claim predicate". The distinction between
`[contrib-schema]` and `[attribute]` that §9.4 rests on has the same problem: it is only worth
drawing if something outside the process can read it.

So the record is now a real output. Set `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` to a file path and every
call to `registry::claim_decision` appends one self-contained JSON object per line:

```jsonc
{"op":"com.microsoft::NotARealOp","node":"n0","opset":1,"claimed":false,
 "code":"not-registered","reason":"[not-registered] no Vulkan handler is registered for ..."}
```

`op` is the domain-qualified type, `node` the graph node name (`""` if unnamed), `opset` the
resolved `since_version` (`0` if unresolved), `claimed` a bool, and `code` the `DeclineCode` tag —
`null` when the node was claimed, or when the reason did not originate in the registry. That makes
both of the assertions Trinity needs one lookup each:

```python
assert claims["com.microsoft::NotARealOp"].code == "not-registered"   # declined, and *why*
assert claims["Add"].claimed                                          # and the positive case
```

Three design points worth defending:

- **JSON Lines appended and flushed per decision, not a report written at teardown.** There is no
  point in the plugin-EP lifecycle where we are reliably told "the session is over, write your
  diagnostics". A test that reads a report the EP has not flushed is a flaky test; a file that is
  complete after every single decision cannot be read too early.
- **Recorded inside `claim_decision`, not in `ep.rs`.** The boundary layer aggregates declines to
  one reason per op *type*, which is right for a human reading claim-debug output and wrong for a
  test asserting about a specific node. Recording below the aggregation also means `epctl` and a
  future measurement harness get the same record with no further work.
- **Two declines never appear**, by construction: nodes inside a control-flow body and nodes
  excluded by `ep.max_claim_ops` are short-circuited *before* the registry is asked. Neither is a
  statement about op support. The record answers "what did the registry decide", not "what did the
  EP do", and a test that sets `max_claim_ops` must not expect lines for the excluded nodes.

Nothing in this path can fail a run: every I/O error is dropped. A diagnostic that can break
inference is worse than no diagnostic, and this runs inside a C ABI callback.

---

## 11. Risks — where this is slower than the MLX project, honestly

Ordered by how much I expect each to hurt.

| # | Risk | Assessment |
|---|---|---|
| 1 | **Three XL ops decide everything.** `GroupQueryAttention`, `MatMulNBits`, `LinearAttention` gate T3/T4/T5a. There is no template leverage for any of them. Each is a multi-week, multi-vendor, numerically-delicate kernel that the MLX project got *for free from MLX*. | **The single biggest schedule risk — and as of the 2026-07-28 ruling it is a scheduled risk, not an optional one** (§6.0). The 174-op inventory is genuinely cheap; these are genuinely expensive, and they are the ones the directive actually cares about. Mitigation is the OQ-M6 accelerant (§13.1), not descoping. |
| 2 | **Perf per op, not just correctness.** MLX handlers inherited Apple-tuned kernels. Ours are as fast as we write them, on 5 vendor architectures with different subgroup sizes, shared-memory sizes, and cache behaviour. "Correct on lavapipe" is not "useful on Adreno." | High. Budget a tuning pass per kernel family per vendor tier, not per op. |
| 3 | **The device-allocator dependency (M2).** LLM coverage is worthless without it (§6.1). It is the highest-uncertainty part of the plugin-EP ABI per `DESIGN.md` §6.3, and it is not on my critical path to fix. | High, and **not mine**. Flagging loudly. |
| 4 | **Dynamic shapes vs. record-once/replay-many.** Decode/prefill/growing-KV means the recorded command buffer must be shape-agnostic or bucketed. Deciding this late means rewriting every LLM kernel's parameter passing. | Medium-high. **Decide now** (§6.1 item 3). |
| 5 | **f16 is an optional capability.** If a meaningful fraction of target devices lack `shaderFloat16`/16-bit storage, the LLM story is desktop-only regardless of op coverage. | Medium. Depends on Link's OQ-1 follow-up. |
| 6 | **Contrib-op schema churn.** `com.microsoft` ops are versioned by ORT release, not by an opset, and `LinearAttention`/`CausalConvWithState`/`MatMulNBitsQkv` are *new*. A schema change silently changes what our claim predicate should accept. | Medium, and now **owned rather than noted** — the contrib domain is admitted, so this hazard is ours. Mitigation is mechanical (§9.4): a `ContribSchema` fingerprint per contrib row, checked before the predicate, with a dedicated `[contrib-schema]` decline bucket so drift is an alarm rather than a wrong answer. Plus the pinned ORT version and the census tool in CI. |
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
rediscovering their conclusions. **This was authorized on 2026-07-28 — see §13.1 and §13.2.**

**Post-ruling restatement (2026-07-28).** The XL kernels are now committed (§6.0), which does not
change any of the four estimates above — it removes the option of not paying them. The estimates
assume the OQ-M6 accelerant is used and that the four engine seams in §9.5 land on schedule; without
the prepack seam, item 3 does not start, and without M2's allocator, item 2 does not finish. If both
slip, the correct report upward is that op coverage is on schedule and "Qwen3.5 end-to-end" is not,
and those must not be averaged into one percentage.

---

## 12. Supersession notice — for Morpheus

I propose, Morpheus ratifies. These conflict with the architecture of record:

| Doc | Current | Proposed | Rationale |
|---|---|---|---|
| `DESIGN.md` §1.2 non-goals | "quantized ops" out of scope for v1 | **RATIFIED BY USER RULING 2026-07-28 — in scope, tier 4** ("matmulnbits那些 都要做") | `MatMulNBits` is the entry ticket for real LLM graphs (§3.2). Without it an int4 Qwen graph shatters into ~200 islands. An int4 LLM is now a *functional requirement*, not an optimization. |
| `DESIGN.md` §1.2 non-goals | "all `com.microsoft` contrib ops" out of scope for v1 | **RATIFIED BY USER RULING 2026-07-28 — 11 contrib ops in scope, tiers 3–5** ("contrib op 要做") | The GenAI builder *emits* them (VERIFIED). Declining `com.microsoft` = cannot run a Qwen graph at all. Morpheus no longer decides *whether*; the remaining question is only which constraints attach (§9.4, and `DESIGN.md` §1.4 C1/C2, which are already satisfied). |
| `DESIGN.md` §1.2 non-goals | "attention fusion" out of scope | **RATIFIED BY USER RULING 2026-07-28 — `GroupQueryAttention` in scope, tier 3** | It is not a fusion we perform; it arrives as a single node from the exporter. Decomposing it would materialize `[B,H,S,S]` scores in VRAM. |
| `DESIGN.md` §1.2 non-goals | "dynamic-shape fast paths" out of scope M0–M2 | **Shape-agnostic push-constant kernel parameters from tier 3** | LLM decode/prefill/growing-KV makes this structural, not an optimization (§6.1). Needs Switch. |
| `DESIGN.md` §8.2 | v0 op set ends at M2 with ~25 ops | **174-op inventory, 6 tiers, model-family exit criteria** | The directive. §8.1's *principles* are unchanged and I endorse all 7. |
| `DESIGN.md` §8.3 | fragmentation rule stated qualitatively | **MVS rule with a measured transfer-cost calibration (§7.2)** | "Does this merge two islands" needs a number to be enforceable. |
| `decisions.md` "Ruthless v1 non-goals" | as above | as above | Same three rows. |
| `ENGINE.md` §3.6 | buffer-only, image storage deferred until a family shows benefit | **`Conv` (tier 5c/6) is that family — re-evaluate then, not before** | Agreement, with a named trigger. |
| `DESIGN.md` §8.2, T3 entry point | T3 begins with `com.microsoft::GroupQueryAttention` | **T3 should begin with `ai.onnx::Attention` @ opset 23** (§4.18) | Justin's own `onnx-genai-models` emits the standard-domain op and never emits GQA. It is the cheaper kernel (no `seqlens_k` indirection, no KV-cache aliasing, no `do_rotary` fold) and it unblocks a model family we can build and iterate on locally. GQA stays committed; it is no longer obviously first. |

Nothing here contradicts the two hard layering rules, the claim-conservatism rule, the ORT-CPU
oracle, the capability-set Vulkan baseline, or the record-once/replay-many decision. Those are load
-bearing and this plan is built on them.

**OQ-11 (Morpheus's ratification of this plan) is closed for the two rows above by user ruling on
2026-07-28.** The remaining rows — shape-agnostic push-constant parameters, the 174-op inventory
superseding §8.2, the MVS rule superseding §8.3, and the `ENGINE.md` §3.6 image-storage trigger —
are still Morpheus's to ratify and are not blocked on him to *start*, because none of the machinery
landed so far depends on their outcome.

---

## 13. Open questions

| ID | Question | Owner |
|---|---|---|
| OQ-M1 | Shape-agnostic recording: are LLM-path kernel dimensions in push constants from day one? | Mouse + Switch |
| OQ-M2 | What fraction of target devices support `shaderFloat16` + `storageBuffer16BitAccess`? Decides whether the LLM path is desktop-only. | Link |
| OQ-M3 | Does the prepacked `MatMulNBits` layout need to be keyed on a per-device tile config? (§8.2) | Mouse + Switch |
| OQ-M4 | Benchmark metric contract — `boundary_bytes_per_inference`, `boundary_time_fraction`, `declined_nodes` histogram. | Niobe |
| OQ-M5 | Conformance harness crash containment (per-op subprocess) and GPU-hang recovery. | Trinity |
| ~~OQ-M6~~ | ~~License review for reading llama.cpp Vulkan shaders as a reference for the 3 XL kernels.~~ **RESOLVED 🟢 GREEN 2026-07-28 (Rai).** See below. | ~~Coordinator~~ |
| OQ-M7 | Does ORT 1.28's `CreateExternalResourceImporterForDeviceImpl` remove the KV-cache round-trip before M2 lands? | Tank / Morpheus |
| ~~R4~~ | ~~The `Attention`-24 reference implementation is wrong for `nonpad_kv_seqlen != q_sequence_length`; which semantics do we implement?~~ **RESOLVED 2026-07-29 (Justin).** Fixed in place, **no opset bump**: `Attention`-24 *is* the corrected semantics. No dual path, nothing to gate on an onnx version. The oracle must pin `onnx` (Trinity); the detection blind spot is recorded in §9.4.1. | ~~Mouse + Trinity~~ |

### 13.1 OQ-M6 resolution — reading llama.cpp's Vulkan shaders is authorized

Rai's ruling (`docs/THIRD_PARTY.md`, `.squad/rai/audit-trail.md`) closes OQ-M6 **🟢 GREEN**. Reading
llama.cpp's MIT-licensed Vulkan compute shaders as a *reference* for the three XL kernels —
`GroupQueryAttention`, `MatMulNBits`, `LinearAttention` — is fully permitted, with **no attribution
obligation for reading and learning**. The operative test, verbatim:

> *"could you write this code without looking at the original?"*

If the answer is yes, it is our code and nothing attaches. Obligations — MIT file header, a
`THIRD_PARTY_NOTICES.md` entry, a commit note, and shipping the notices file — trigger **only** if
we substantially adapt shader source rather than independently write our own after understanding
the algorithm. Rai addressed our specific build shape: **SPIR-V compiled from adapted GLSL is a
derived work**, and our build embeds SPIR-V into the cdylib, so an adapted shader would carry its
obligations all the way into the shipped binary. ExecuTorch (BSD-3), ORT (MIT) and gpuinfo.org
(CC-BY 4.0) are compatible on the same terms.

### 13.2 Per-kernel licensing record — which side of the line each kernel is on

Rai requires that the choice is recorded *per kernel*, before the kernel is written, so that the
answer to "could you write this without looking at the original?" is a decision rather than a
retrospective justification. The same record appears in each module's header doc comment; this table
is the index. It is updated when a kernel is written, not when it is planned — a row saying
"independent implementation" that later adapts source **must** be changed and must pick up the MIT
header, the `THIRD_PARTY_NOTICES.md` entry and the commit note.

| Kernel | Module | Choice | Obligation |
|---|---|---|---|
| `GroupQueryAttention` | `ops/attention.rs` | **Independent implementation.** Read llama.cpp's flash-attention Vulkan shaders for tiling and subgroup-reduction strategy; write our own against our `DispatchContext` and our KV-cache layout. | None. |
| `RotaryEmbedding` | `ops/attention.rs` | **Independent implementation.** The algorithm is a page of arithmetic; no reference needed. | None. |
| `MatMulNBits` | `ops/quant.rs` | **Independent implementation.** Read llama.cpp's `mul_mat_vec_q*` for the memory-access strategy — see the note below on what does and does not transfer. | None. |
| `GatherBlockQuantized`, `DequantizeLinear`/`QuantizeLinear` (blocked) | `ops/quant.rs` | **Independent implementation.** Shares our own unpack helper. | None. |
| `LinearAttention` (`gated_delta`) | `ops/ssm.rs` | **Independent implementation.** No open reference Vulkan implementation exists to adapt even if we wanted one. | None. |
| `CausalConvWithState` | `ops/ssm.rs` | **Independent implementation.** | None. |
| `QMoE` / `MoE` | `ops/moe.rs` | **Independent implementation.** Masked-dense first; the routing math is ours. | None. |
| `SimplifiedLayerNormalization` / `Skip*` | `ops/norm.rs` | **Independent implementation.** RMSNorm is four lines; our reduction template supplies the rest. | None. |
| `ai.onnx::Attention`, `ai.onnx::RotaryEmbedding`, `ai.onnx::RMSNormalization` | `ops/attention.rs`, `ops/norm.rs` | **Independent implementation.** Same kernels as the contrib spellings above; the standard-domain rows are table entries over them, not new code. | None. |

If any kernel later needs substantial adaptation of third-party shader source, the procedure is in
`docs/THIRD_PARTY.md` (Rai's) and must be followed **before** the adapted code lands, because our
build embeds SPIR-V into the cdylib and the compiled artifact is a derived work of the GLSL.

#### 13.2.1 Correction, 2026-07-28: "no obligation" is not "nothing to learn"

An earlier revision of the row above argued that llama.cpp's block formats are incompatible with
the ONNX `MatMulNBits` nibble layout, "so adaptation would not even work". Switch reviewed that for
`ENGINE.md` (D-S4-10) and is right that it was **too strong**, and I am withdrawing it. The two
claims it ran together are separate:

- *Licensing.* Unchanged, and Switch and I agree: we write our own code, nothing attaches. The data
  layout being incompatible is one more reason the answer is "independent implementation", but it
  was never the reason.
- *Value of reference reading.* This is Switch's domain, not mine, and he says the leverage is
  real. The data **layout** does not transfer, but the **algorithmic strategy** does: GEMV-vs-GEMM
  tile-size specialisation constants, subgroup reduction shape (per-lane partial dot →
  `subgroupAdd`), and dequantise-in-register patterns are all portable conclusions that took that
  project years of multi-vendor tuning to reach. Under Rai's green light we may read them.

So the accelerant stands at full strength and the estimates in §11.1 continue to assume it is used.
Morpheus is ruling on what that means for the T3/T4/T5a numbers, since he priced it in.

The general lesson, recorded because it will recur: *"we cannot copy this"* and *"there is nothing
here to learn"* are different sentences. A licensing record should only ever assert the first.

**Operational rule for this document's §6 tiers:** before touching T3's `GroupQueryAttention`, T2's
`MatMulNBits`, or T4's `LinearAttention`, read `docs/THIRD_PARTY.md` for the mechanics. Algorithm
study is unrestricted; source adaptation is a licensing event that must be declared. This removes
the largest single unknown from the XL-kernel estimates in §11 — the *algorithms* for flash-attention
tiling and int4 block dequant-in-register are no longer things we have to rediscover, which is worth
more to the schedule than any single template.

---

## 13.9. The vision lane: `Conv`, and what one model class costs (2026-08-03, Mouse)

### 13.9.1 The measurement that made this section necessary

Two LLMs were this project's entire evidence base. Neither contains a convolution, so **no
instrument the project had could see the gap** — §4.21's census reports 293/374 and 355/366 and
both are true and both are about transformers.

MobileNetV2-12 (`bench/results/model_provenance.json` records the URL and sha256), censused with
`rust/tools/probe_model_op_census.py` against a real EP session:

```
=== mobilenetv2: 105 nodes, opsets {'ai.onnx': 12}
    claimed 0 / declined 104 / no-decision 1
    Conv               52  {'not-registered': 52}
    Clip               35  {'unproven': 35}
    Add                10  {'unproven': 10}
    Shape/Gather/Unsqueeze/Concat/GlobalAveragePool/Reshape/Gemm  1 each
```

**Zero.** Not thin coverage — the registry contained none of `Conv`, `Gemm`, `MatMul`, `Softmax`,
`Transpose`, `Concat`, `Slice`, `Split`, `Reshape`, `Reduce*`, `LayerNormalization` or any pooling
op. No non-LLM model could run at all. That is the concrete content of the "llama.cpp ships 164
Vulkan shaders, we ship 12" ratio, measured on a graph instead of counted on a shelf.

And **45 of the 104 declines were `[unproven]` on ops shipped live** — `Clip` and `Add` at the
`runtime-extent` shape class, a form no LLM evidence case had ever produced at f32. That is
§8.10's pattern one level out: pointing the census at a new model *class* finds forms already
backed by a compiled variant and never proven. Their cost is a proof run, not a shader.

### 13.9.2 Why `Conv` and not `Reshape`

`Reshape` had 24 declining nodes on gpt-oss-20b and opens the shape family as a template — the
better-looking pick on a count. Reading the graph neighbourhood instead of the count:
**all 24 are `Add -> Reshape -> QMoE`**, and `QMoE` is staged. Claiming `Reshape` moves the island
boundary by one node and unblocks nothing.

This also re-tested the `Reshape` decline recorded at `26fd93f`, whose stated falsifier was "the
census shows no model contains one". **That falsifier has since fired** — 24 `Reshape` nodes now
exist. The conclusion survived anyway, for a reason the original ruling did not name: the
consumer is still declined. Both halves are recorded because a ruling that outlives its own
falsifier deserves the new reason written down, not a quiet reprieve.

### 13.9.3 What `conv_f32.comp` claims, and the four declines by name

A direct 2-D convolution, f32, one shader. **Grouped is the general case**, so depthwise
(`group == C`) — 17 of MobileNetV2's 52 convolutions — costs no second kernel. Only the *begin*
pads appear in the index arithmetic; the end pads change `OH`/`OW`, which the translate handler
computes. An out-of-range tap is a **skipped accumulation, not a clamped read**, which is what
ONNX's implicit zero padding means and what distinguishes a correct border from a plausible one.

Declined, each for a stated reason:

| decline | code | why |
|---|---|---|
| f16 | `[dtype]` | packed-`uint` half I/O addresses two elements per 32-bit word; a convolution reads single scattered elements. Needs `conv_f16.comp`, not a wider `caps`. |
| rank != 4 | `[rank]` | 1-D and 3-D convolution are different index arithmetic, not a longer loop. |
| `auto_pad != NOTSET` | `[attribute]` | `SAME_*` derives the pads from an *output* extent, which is not a fact about the node when the pipeline is built. ORT's optimizers rewrite most producers to explicit pads. |
| symbolic `C`/`H`/`W` | `[dynamic-shape]` | the padding arithmetic is not linear in `H`/`W`, so the output extent cannot be recovered from a ratio at Compute time the way a flat element count can. **Batch may be symbolic** — and on every real vision graph it is. A lift condition, not a permanent decline. |

### 13.9.4 The four keys are the whole key space — and the space the key omits

```
ai.onnx::Conv/1+/f32,f32,f32>f32/metadata/static/n3
ai.onnx::Conv/1+/f32,f32>f32/metadata/static/n2
ai.onnx::Conv/1+/f32,f32,f32>f32/metadata/runtime-extent/n3     <- MobileNetV2's own form
ai.onnx::Conv/1+/f32,f32>f32/metadata/runtime-extent/n2
```

That is `{bias, no bias} x {static, runtime-extent}` closed completely: arity and `shape_class`
are key components and **nothing else about a `Conv` is**.

`group`, `strides`, `dilations` and `pads` therefore appear in no key. Four entries say nothing
about whether a stride-2 asymmetric-pad depthwise convolution is right. That is not a defect in
the key — a key is about which module runs and how bindings are laid out — but it is a gap in
what a proof covers, and the honest response is to name the uncovered axis rather than let four
entries read as coverage of `Conv`. It is closed by `tests/ops/test_conv.py` (twelve attribute
cases plus two exact structural assertions a tolerance check cannot make: a one-hot depthwise
convolution must be a **bit-exact** channel copy, and all-ones input with `pads=1` must give
`4/6/9` taps at corner/edge/centre where a clamping kernel would give `9` everywhere).

### 13.9.5 The first derived tolerance

`tests/ops/_models.py` has reserved the accumulating ops since M0: *"tolerance is
accumulation-order-dependent and MUST be derived from test data per vendor ... Do not guess; do
not copy from fp32 elementwise."* `Conv` is the first accumulating op to land, so it is the first
one that clause binds.

`tests/ops/probe_conv_tolerance.py` measures it. RTX 4060 Laptop, twelve cases: worst
`max_rel = 1.858e-4`, `max_abs = 5.722e-6` (`bench/results/conv_tolerance_derivation.json`). ORT's
CPU EP lowers `Conv` to im2col + Eigen GEMM; the residual is accumulation order, not disagreement.
`FP32_CONV` is pinned at `rtol 1e-3 / atol 1e-5` — **exactly `gen_proof_ledger.py`'s defaults**, so
the conformance gate is precisely as strict as the proof gate and cannot be quoted as looser.
Re-derive before quoting it on AMD or lavapipe; the clause says per vendor and this is one vendor.

> The probe's **first** run printed `0.000e+00` for all twelve cases. ORT had answered `Unknown
> Provider Type: VulkanExecutionProvider`, fallen back to CPU, and compared the CPU against
> itself. It now registers the EP and refuses to print a number for any case the EP did not
> claim. A perfect residual from an instrument that observed nothing is `ERROR(instrument)`, and
> it very nearly became a pinned constant.

### 13.9.6 The delta

| | before | after |
|---|---|---|
| registry rows / kernel-carrying | 91 / 73 | 92 / 74 |
| ledger entries | 115 | 121 |
| **MobileNetV2-12** | **0 / 105** | **97 / 105** |
| gpt-oss-20b | 293 / 374 | 293 / 374 |
| Phi-3.5-mini | 355 / 366 | 355 / 366 |

**What it unblocks: one new model class.** MobileNetV2's remaining 7 declines are a single
`Shape -> Gather -> Unsqueeze -> Concat -> Reshape` classifier tail plus `GlobalAveragePool` and
`Gemm` — one contiguous non-backbone island, not seven scattered holes. `GlobalAveragePool` and
`Gemm` are the next two ops by this criterion, and they are also the two that a `Reduce`/GEMM
template would serve beyond this one graph.

The two LLMs are unchanged in both directions. That is the point: coverage, not a trade.

---

## 14. In-house ONNX crates — evaluation (2026-07-29)

Justin directed us to *参考* — reference — his own Rust/Python ONNX projects rather than build graph
handling from scratch. I read the source of all four rather than the READMEs. Verdicts first,
because three of the four are "defer", and the fourth turned out to be worth more than any of the
dependencies would have been.

| Crate | What it actually is today | Verdict |
|---|---|---|
| `onnx-ir-rust` (`onnx-ir-core`) | Rust, Apache-2.0, ~2,400 lines, **20% complete** by its own `IMPLEMENTATION_STATUS.md` | **Defer** |
| `onnx-shape-inference` | **Python**, Apache-2.0, v0.3.0 alpha, SymPy-backed, strong contrib-op coverage | **Defer as a dependency; adopt as an oracle and as reference** |
| `onnx-genai` (`onnx-runtime-ir`) | Rust, MIT, a real IR with full use-def, arenas, topological order | **Defer, with a named trigger** |
| `onnx-genai-models` (`mobius`) | Python, Apache-2.0, programmatic model builder | **Not a dependency — but the highest-value find, see §4.18** |

### 14.1 `onnx-ir-rust` — defer

Read from `crates/onnx-ir-core/src/`. It has `Graph`, `Node`, `Value`, `Attr`, `Shape`,
`SymbolicDim`, `Tensor` and an intrusive `DoublyLinkedList<Node>`. What it does **not** have is
everything we would adopt it for:

- **No use-def tracking.** `value.rs` carries the producer/consumer fields *commented out*, with
  the note that they "will be added when Node is properly defined with reference counting".
  Consumer queries are the single thing a partitioner needs most.
- **No topological iteration, no subgraph extraction, no reachability or dominance.** Listed as
  Phase 5–6 work.
- **No protobuf ingestion.** `prost` is declared as a dependency but no deserialization exists.
- **Mutation primitives are known-buggy** — `pop_front`/`pop_back`/`clear` have `#[ignore]`d tests.
- **Not published to crates.io** (404), so it would be a git dependency on a moving target.

And even at 100% completion it would not fit: we never see a protobuf. ORT hands us `OrtGraph` /
`OrtNode` handles across a C ABI, and there is no path to wrap those — we would have to *copy* the
whole graph into a second representation inside a cdylib living in someone else's process. Paying
`prost` (~500 KB) plus a full second graph, to replace roughly 150 lines of union-find over the
handles we already have, is not a trade. **Defer, with no expected revisit.**

### 14.2 `onnx-shape-inference` — defer as a dependency, adopt as an oracle

It is **Python**, not Rust, so the dependency question does not even arise for a cdylib. But the
substance is genuinely strong and directly touches two things I own:

- It is **really symbolic**: SymPy-backed dimension arithmetic with a safe recursive-descent parser
  (no `eval`), constraint propagation that anchors anonymous `_d0` symbols to author-declared names
  like `batch`/`seq`, and three merge policies (`refine`/`strict`/`merge`).
- Its `_ops/_microsoft.py` implements shape functions for `GroupQueryAttention`,
  `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, `RotaryEmbedding`,
  `MultiHeadAttention`, `BiasGelu`, `FastGelu`, `QuickGelu`, `BiasSplitGelu`, `GroupNorm` and more.

Two adoptions that cost us nothing and are worth real coverage:

1. **As a preprocessing step in Trinity's harness.** Our claim predicates decline symbolic dims
   (`symbolic_dims_are_rejected_not_guessed`). Running `infer_symbolic_shapes` over a test model
   before handing it to ORT would resolve many dims to concrete integers, converting
   `[dynamic-shape]` declines into claims **with no change to our Rust at all**. This is the
   cheapest coverage available anywhere in this document, and it is a test-harness change, not a
   runtime one. Routed to Trinity.
2. **As independent ground truth for the C2 contrib fingerprints.** Its `_microsoft.py` encodes
   contrib input ordering and output arity, derived independently of ORT's own docs. Where it and
   `ContribOperators.md` agree, my fingerprint confidence rises; where they disagree, that is a
   flag worth chasing rather than a coin toss. It does **not** solve schema *drift* — it is a
   snapshot like any other — but it doubles the sources behind each fingerprint.

Not a dependency. A tool and a second opinion.

### 14.3 `onnx-genai` (`onnx-runtime-ir`) — defer, with a named trigger

This one is not a toy. `crates/onnx-runtime-ir/src/graph.rs` is ~60 KB and the IR has exactly what
`onnx-ir-rust` lacks: `producer`/`consumers` on every `Value`, arena-allocated stable `NodeId`/
`ValueId`, sorted consumer snapshots for deterministic rewrites, per-node opset overrides,
`Option<ValueId>` inputs that model ONNX optional inputs correctly, symbol constraints, and
control-flow subgraph bodies. It is MIT, so licence-compatible with us with no `THIRD_PARTY.md`
consequence. The workspace also contains `onnx-runtime-ep-api`, `onnx-runtime-optimizer` and
`onnx-runtime-shape-inference`.

It is still **defer**, for one structural reason and one prudential one:

- *Structural:* the same C-ABI mismatch. It builds and loads its own graphs; it cannot adopt an
  `OrtGraph`. Using it would mean transcribing ORT's handles into its arenas — real work, a second
  copy of the graph resident in the host process, and a new failure mode (transcription bugs) in
  the layer whose correctness is hardest to test.
- *Prudential:* `0.1.0-dev.5`, unpublished, active development. A cdylib loaded into someone
  else's process is the worst possible place to take a fast-moving unpublished git dependency,
  because the lifetime is not ours and neither is the process.

**The named trigger, so this is a decision rather than a shrug:** if we ever need a graph
representation that outlives a single `GetCapability` call — a cross-session compiled-graph cache,
a rewrite pass, or a partitioner that needs repeated reachability queries over a mutable graph —
then we need an IR, and at that point `onnx-runtime-ir` is the first thing to reach for rather than
something to write. Today `ops/partition.rs` consumes an already-built `Island` and the clustering
that produces it is one union-find pass, so we do not.

#### 14.3.1 Re-evaluation after Justin retired the prudential objection (2026-07-29)

> "onnx-runtime-ir 是我们自己的 crate 所以可以放心用"

That retires the **prudential** half above, completely and correctly: version churn, unpublished
status and third-party lifetime risk are not arguments against a crate we own and can change. I am
not treating it as retiring the structural half, because the structural argument is about *where we
sit in the process*, not about who wrote the library — and Justin removed a constraint, he did not
issue an instruction to adopt.

Re-evaluating the structural argument on its merits, with trust removed as a factor:

**What the copy would cost, precisely.** ORT hands `GetCapability` an `OrtGraph` and `OrtNode`
handles across a C ABI. We never see a protobuf. Adopting the IR means walking every node and edge
through Tank's `sys.rs` accessors and materialising an arena — for a Qwen3-0.6B graph, on the order
of a few thousand nodes and values, per session, inside the host's address space. The cost is not
the memory; it is that the transcription becomes a correctness surface. A `NodeView` today is a
borrowed window onto ORT's own truth and cannot disagree with it. An arena can, and a transcription
bug in the layer that decides what we claim is the most expensive class of bug this project has,
because it fails as a wrong answer rather than as a decline.

**Does the recorded trigger fire?** I said: adopt it the day we need a graph representation that
outlives a single `GetCapability` call. Taking the three candidates the coordinator named, honestly:

1. **Compiled-subgraph caching — no.** What outlives the call is the *compiled artefact* (SPIR-V
   pipelines, prepacked weights) keyed by a structural hash, not the graph. We need a stable key,
   which is ~50 lines of hashing over the node list we already walk. Keeping an entire IR alive to
   own a hash key is the tail wagging the dog.
2. **Prepack keyed on graph structure — no, and it is keyed on less than that.** §8.2.1's prepack
   requests are keyed on the *initializer* plus `MatMulNBits`'s `K`/`N`/`bits`/`block_size`. That
   is node-local. It does not need edges at all, let alone an IR.
3. **Repeated reachability queries during fusion analysis — not yet, and this is the closest
   one.** `ops/partition.rs` today evaluates an already-built `Island`, and the clustering that
   produces one is a single union-find pass over claimed nodes — no repeated queries, no mutation.
   That stays true for the fusions T1–T2 need, which are all local patterns (`Add`+`RMSNorm`,
   `MatMul`+activation) matchable in one pass. It stops being true if we ever do **producer-side
   graph rewriting** — replacing a subgraph with a different node set rather than merely selecting
   one — and ORT's plugin-EP surface does not offer us that anyway.

**So: still defer, and now for exactly one reason instead of two.** The remaining reason is real
and is not about the crate. What has changed materially is the *disposition*: with the prudential
objection gone, the moment the trigger fires there is nothing left to weigh — no evaluation round,
no dependency-risk conversation. The trigger is now a switch rather than a question.

**Two sharpened triggers,** so the next person does not have to re-derive this:

- **T-IR-1:** the first time we need to answer "is node A reachable from node B" more than once per
  `GetCapability` call, or need a mutable graph across calls. That is when union-find stops being
  enough and an arena with real producer/consumer edges starts paying for itself.
- **T-IR-2:** the first time we need to *construct* ONNX — a decomposition emitted as nodes rather
  than as a dispatch sequence, or a graph written out for a test fixture. Building a graph is what
  this IR is unambiguously better at than anything we would write, and unlike consuming a graph it
  has no `OrtGraph` on the other side to compete with.

Neither fires today. Nothing to route to Tank; `Cargo.toml` is unchanged.

One thing worth flagging to Niobe rather than filing here: the same workspace contains
`onnx-runtime-optimizer` and `onnx-runtime-shape-inference`. `onnx-runtime-shape-inference` in
particular is the Rust home of the capability §14.2 wanted and could only get as a Python oracle —
if it can infer over an ORT-supplied graph without the full transcription, it fires a *different*
trigger (claiming nodes we currently decline for symbolic shapes) with a much smaller structural
cost, because shape inference wants a value graph and not the whole node arena. Not evaluated this
turn; recorded so it is not lost.

### 14.4 What we actually adopted

No new dependency lines. Nothing for Tank to add to `Cargo.toml`. What changed is the **op table**,
which is where the value turned out to be: five standard-domain rows
(`ai.onnx::Attention`, `ai.onnx::RotaryEmbedding`, `ai.onnx::RMSNormalization`, plus the shared
predicates) that make a model built by Justin's own toolchain claimable at all. See §4.18. That is
a larger coverage gain than any of the three libraries would have produced, and it came from
reading them.

I want to be explicit that "defer" here is not politeness-avoidance in the other direction: the
directive said *参考*, and the reference paid off. It simply paid off as information rather than as
linkage.

---

## 15. References

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
- In-house crates evaluated in §14 — <https://github.com/justinchuby/onnx-ir-rust>,
  <https://github.com/justinchuby/onnx-shape-inference>,
  <https://github.com/justinchuby/onnx-genai>,
  <https://github.com/justinchuby/onnx-genai-models> (mirror; superseded as the producer of record)
- **Producer of record** — `onnxruntime/mobius` @ `87fd878` (`main`, 2026-07-29), MIT,
  `OPSET_VERSION = 24`. Attention construction in `src/mobius/components/_attention.py`; norm in
  `src/mobius/components/_rms_norm.py`; rotary in `src/mobius/components/_rotary_embedding.py`;
  quantized linear in `src/mobius/components/_quantized_linear.py`; MoE in
  `src/mobius/components/_moe.py`; GQA fusion in `src/mobius/rewrite_rules/_group_query_attention.py`
- ONNX opset 23 additions (`Attention`, `RMSNormalization`, `RotaryEmbedding`) —
  <https://onnx.ai/onnx/operators/>
- **ONNX opset 24** (onnx v1.19.0, 2025-08-26; §4.19) — `onnx/defs/nn/defs.cc` (Attention-24),
  `onnx/defs/nn/old.cc` (Attention-23), `onnx/defs/tensor/defs.cc` (TensorScatter-24),
  `onnx/defs/math/defs.cc` (Swish-24), `onnx/defs/quantization/old.cc` (Q/DQ 21/23/24),
  `onnx/defs/operator_sets.h` (opset membership),
  `onnx/version_converter/adapters/Attention_24_23.h` (proof that 24 cannot be downgraded to 23
  when `nonpad_kv_seqlen` is present)
- **ONNX opset range 25–27** (§4.20) — `onnx/defs/schema.h` (`map_[ONNX_DOMAIN] = {1, 27}` vs
  `last_release_version_map_[ONNX_DOMAIN] = 26`, onnx v1.22.0), `onnx/defs/operator_sets.h`
  (opset-25 IR-13 type expansion across 18 ops; opset-26 `BitCast`/`CumProd`; opset-27
  `CausalConvWithState`/`LinearAttention`/`Range`), `onnx/defs/quantization/defs.cc`
  (QuantizeLinear-25 `output_dtype`/`precision`, DequantizeLinear-25 `output_dtype`),
  `onnx/docs/Changelog.md` @ v1.22.0
- **In-place ONNX errata, no opset bump** (§9.4.2) — onnx#7297 (v1.19.1, Attention causal mask),
  onnx#7274 (v1.20.0, Attention GQA `repeat_interleave`), onnx#7867 (v1.22.0, Attention softcap
  ordering), onnx#7913 (v1.22.0, `qk_matmul_output_mode` 1↔2 swap), onnx#8068 (unreleased,
  `is_causal` alignment + NaN guards), onnx#7313 (v1.19.1, RotaryEmbedding-23 reference impl)
- **Real production graphs censused (§4.21)** — Foundry Local model cache,
  `C:\Users\justinchu\.foundry\cache\models\Microsoft\`:
  - `Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32\model.onnx` — producer
    `onnxruntime-genai` `'0.0.0'`, ir_version 7, opset imports `ai.onnx` **14** + `com.microsoft` 1,
    366 main-graph nodes, fp16, int4 symmetric RTN block-32, 2.2 GB external data
  - `gpt-oss-20b-cuda-gpu\v1\model.onnx` — producer `onnxruntime-genai` `''` (empty version string),
    ir_version 10, opset imports `ai.onnx` **21** + `com.microsoft` 1, 374 nodes, fp16 activations
    with fp32 norms, mixed 4/8-bit `MatMulNBits`, 24 `QMoE` (top-4, swiglu)

  Per §8.5 these are two distinct producers-at-version and coverage must be reported separately for
  each; neither substitutes for the mobius path, which they share no standard-domain LLM ops with.
