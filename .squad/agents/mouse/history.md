# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Op Coverage — ONNX op implementations, registry, graph partitioning
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-28T22:28:08-07:00 — full session details in decisions.md -->

### [SUMMARY] Turns 1–5: op plan, template infrastructure, contrib rows, kernels, diagnostics (2026-07-28)

**Foundational decisions (turn 1 — OP_COVERAGE.md authored):**
- MLX op-coverage speed does NOT transfer: every op here is a hand-written GLSL shader. Leverage = kernels grow much slower than ops. Target ≥ 8 ops per kernel family in tiers 1–2.
- Qwen ONNX graphs emit `com.microsoft` ops (GQA, RotaryEmbedding, MatMulNBits, SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization). Declining the domain = EP cannot run Qwen at all.
- ORT WebGPU EP (`webgpu_contrib_kernels.cc`) is the closest analog; registers `QMoE` but float `MoE` commented out — do `QMoE` first.
- `MatMulNBits` is entry ticket for int4 LLMs (not optional). GEMV path (M=1) memory-bound; GEMM path (M>1) shared-memory tile. Never materialize dequantized weights in VRAM.
- Broadcasting solved ONCE in `indexing.glsl`: zero stride = stretched axis. All ONNX broadcasting semantics normalize host-side in `ShapePlan`. No ONNX semantics in shaders.
- Registry: `op_table!` macro with `caps: DtypeSet` column generating dtype claim check, build.rs shader variant list, `docs/OP_SUPPORT.md`, `--dump-capabilities`. CI fails on drift.
- LLM path gated on M2 device allocator. KV cache / conv state cross subgraph boundary per token under M0/M1 = per-token cache round-trip.
- MVS rule: `est_gpu_time > transfer_cost × 3.0`, floor `node_count ≥ 4 AND output_bytes ≥ 64 KiB`, waived for GEMM/attention/QGEMM anchored nodes. Anti-orphan pass for non-anchored 1–3 node islands.
- `largest_island_flops` is the metric of record. `claimed_node_fraction` is diagnostic only.
- Template infrastructure before op #1 is a milestone gate (M1 entry criterion). Ratified by Morpheus with this amendment.

**Template infrastructure landed (turn 2):**
- `OpStatus::Staged(reason)` — fully described row declines before predicate runs. Going live = one-word diff. Distinct staging reasons: `NO_SHADER`, `NEEDS_PARAMS`, `NEEDS_CAST_MATRIX`, `UNEXERCISED`.
- Claim predicate takes `(NodeView, OpSpec)` — the predicate reads `spec.caps`/`spec.op_type`/`spec.kernel` instead of closing over them. One predicate serves 60+ ops.
- Machine-readable declines: `decline(code, detail)` renders `"[tag] sentence"`, `DeclineCode::of_reason()` parses back. Three consumers (human log, Trinity assertions, Niobe histogram), zero cross-owner edits.
- `ShapePlan::broadcast` right-aligns into `[u32; MAX_RANK=6]`, zero stride on stretched axes. Push layout ≤ 128 bytes worst case (maxPushConstantsSize floor, asserted by test). `MAX_RANK=6` is not 8.
- Scalar inputs: leave `ShapePlan.rank` at 0 for all-scalar inputs; do NOT clamp to 1.
- `REQUIRE_STATIC_SHAPES = true` in one place. Dynamic shapes decline everything with `[dynamic-shape]` bucket. One constant to flip when OQ-15 (shape-agnostic dispatch) lands.
- `Recorder` mock `DispatchContext` = highest-value test asset: tests pure `NodeDesc → KernelRequest` with no Vulkan/ORT. Separates failure modes.
- MVS shipped as two ordered gates: size gate then economics gate. Margin 3× not 1× — cost model crude, must fail towards CPU. `TransferModel::fit` is Niobe's calibration hook.
- `concentration()` (largest-island FLOPs ÷ total claimed FLOPs) separates "80% in one island" from "80% across 40 islands". Unit tests assert these two have equal `node_coverage` and different honest metrics.
- 69 elementwise rows, 5 claim predicates, 5 translate handlers (demonstrated leverage).

**Contrib rows (turn 3 — 11 named ops admitted):**
- `com.microsoft` domain admitted by user ruling. Never a `domain == "com.microsoft"` predicate. Registry key is the allowlist.
- Contrib schemas version with ORT releases (not opset). `ContribSchema` + per-op recorded ORT version in the table (not in comments). Census claim rates in CI + version-bump-as-review-gate.
- `[attribute]` = deliberate limitation. `[contrib-schema]` = schema moved under us (alarm). Never lump these.
- 11 staged rows: GroupQueryAttention, RotaryEmbedding, MatMulNBits, LinearAttention, CausalConvWithState, SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, QMoE, MultiHeadAttention, MoE (float oracle for QMoE), SkipLayerNormalization.
- `SchemaBaseline` inside `ContribSchema` (not a parallel table) — impossible to record a schema without recording where it came from.
- GQA fingerprint: 7 required inputs (not 3); optional inputs are positional; `seqlens_k`/`total_sequence_length` at indices 5 and 6.
- GenAI builder sets `q_norm`/`k_norm` for Qwen3 — emits 16-input GQA node. Verify exported graph before scheduling a kernel against an assumed signature.
- In a 5-agent concurrent repo: re-read every file before editing; check git status before diagnosing a test failure.
- 174+26=205 tests after turn 3 (lib+layering+dump).

**First real kernels (turn 4):**
- `build.rs` scans `shaders/glsl/` non-recursively for `*.comp`. Parameterised templates must live in `shaders/glsl/templates/`.
- Validate shaders without SDK: `glslangValidator.exe` from Khronos release. Same flags as build.rs.
- COMPILED ≠ EXECUTED: `UNEXERCISED` staging reason distinct from `NO_SHADER`. Always report separately.
- ONNX scalar semantics: `Round` = roundEven; `Mod` default = sign of divisor (`a - floor(a/b)*b`); GLSL `pow` undefined for negative base but ONNX is not; `Mean` in binary template = divide by N once at end; `Erf` = load-bearing for exact Gelu.
- Byte-typed tensors: `bool`/`uint8` buffer must be allocated rounded up to 4 bytes (packed-byte stores write whole `uint` word). Tell Tank/Switch.
- Correction (D-S4-10, Switch): llama.cpp block format mismatch = no code copying, but tiling/subgroup reduction/dequant-in-register patterns DO transfer. Never bundle licensing conclusions with technical ones.
- `cargo build; cargo clippy --all-targets -- -D warnings; cargo test` → 245 passed. Set `$env:ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC='1'` on machines without Vulkan SDK. Regenerate variant manifest: `MOUSE_BLESS_VARIANTS=1 cargo test --lib variants`.

**Diagnostics (turn 5):**
- Diagnostic codes are worthless if they stop at the FFI boundary. Design the reader in the same breath as the codes.
- Per-event JSON (append-and-flush, one self-contained line per event). No lifecycle hook needed.
- Hook `claim_decision` not `ep.rs` aggregator — zero cross-owner edits; all future callers get the record.
- `GroupQueryAttention` fingerprint corrected: min_inputs=7, 1.28 and main forms identical.
- `cargo test` → 265 passed.

**ORT ABI facts verified against vendored 1.28 header:**
- `Node_GetInputs`/`Node_GetOutputs`: size-then-fill protocol.
- `GetValueInfoTypeInfo`: returns borrowed `const OrtTypeInfo**` — must NOT be released (double-free).
- `CastTypeInfoToTensorInfo`: `_Outptr_result_maybenull_` (non-tensor → null, no error).
- `Node_GetAttributeByName`: also `_Outptr_result_maybenull_`.
- `ReadOpAttr`: call once to size, again to fill.
- ORT returns null `OrtValueInfo` for omitted interior optional inputs (`Clip(x, , max)`).

---

## Cross-agent context appended (2026-07-28T22:28:08-07:00)

📌 **C2 item 7: fingerprint audit CI job (Morpheus §1.4):** A CI job running `graph_census.py` must execute before any tier-3 contrib work. Rows with `SchemaBaseline` pointing to a non-release (ORT `main` only) may not be set to `Live` — this is a build failure enforced by Tank. Your `ContribSchema` nested `SchemaBaseline` field wins over Tank's side table (deleted). Verify the CI job exists in `graph_census.py` and is wired in `.github/workflows/` before tier-3.

📌 **C1 domain regression test (Trinity):** `tests/ops/test_domain_regression.py` asserts `com.microsoft::NotARealOp` produces an ordinary decline (not a crash). TODO: upgrade Trinity's test to machine-readable reason code when your diagnostic JSON format is stable. `[contrib-schema]` and `[attribute]` must remain separate decline code buckets — never merge them.

📌 **Switch's `bind_aliased_output` seam (Switch engine-seams, D-S3):** KV-cache in-place update requires `bind_aliased_output(output_slot, input_slot)` on `DispatchContext`. GQA and LinearAttention handlers will need this for M2+. Default method returns resolved input — your handlers do not need to use it until KV-cache is required.

📌 **Switch's `compile_hook_for` stub in `registry.rs` (Switch engine-seams, Seam 1):** `registry.rs` now has `pub fn compile_hook_for(desc: &NodeDesc) -> Option<CompileHook>`. Mouse fills in per-op prepack hooks for GQA/MatMulNBits. `CompileHook` = `fn(&mut CompileContext, &NodeDesc)`. The `TileConfig`, `PackKey`, `PackInput`, `PackOutput`, `PrepackRequest`, `PrepackResult` vocabulary is in `engine.rs`.

📌 **`concentration()` metric is the honest performance predictor (Mouse partition rule).** `largest_island_flops ÷ total_claimed_flops` separates "80% coverage across 40 islands" from "80% in one island". Niobe reports this; always include it alongside `node_coverage` in any coverage summary you publish.

📌 **Byte-typed tensors: allocator rounding (Mouse turn-4):** `bool`/`uint8` buffers must be allocated rounded up to 4 bytes. This is invisible from Rust; it belongs in the allocator contract. Coordinate with Tank when the M2 allocator lands. Document in `OP_COVERAGE.md §8` alongside the `PackedWeights` memory class note.

📌 **GQA fingerprint correction (Mouse turn-5):** `min_inputs = 7` (not 3). Optional inputs are positional; `seqlens_k`/`total_sequence_length` at indices 5 and 6. GenAI builder emits 16-input GQA for Qwen3 (sets `q_norm`/`k_norm`). Verify the exported graph before finalizing the GQA claim predicate.

---

## Turn 6 — 2026-07-29T07:14:15-07:00 — the in-house crate review

📌 **Op coverage is relative to a *producer*, not to a model architecture.** The biggest finding of
the turn, and it invalidated a premise of my own document. I derived the entire op inventory from
what the ORT GenAI model builder emits. Justin's `onnx-genai-models` builds the *same models* and
emits `ai.onnx::Attention`, `RMSNormalization` and `RotaryEmbedding` @ opset 23 instead of the
`com.microsoft` spellings — so our table would have declined every norm, rotary and attention node
in a Qwen3 built by our own toolchain. Same kernels, missing rows. **Always ask "which exporter
produced this graph", never just "which model is this".**

📌 **Reading the source changed three verdicts that the READMEs would have gotten wrong.**
`onnx-ir-rust` looks like a Rust ONNX IR and is 20% of one — its producer/consumer fields are
literally commented out, and it cannot ingest a protobuf at all. `onnx-shape-inference` sounds like
a Rust crate and is pure Python. `onnx-genai` sounds like a model thing and contains the most
complete Rust IR of the three. Judge dependencies from `src/`, never from the front page.

📌 **The decisive objection to a graph IR here is architectural, not quality.** We are a plugin EP:
ORT hands us `OrtGraph`/`OrtNode` across a C ABI and we never see a protobuf. Any external IR would
require *copying the whole graph* into a second representation inside someone else's process. That
objection would survive the library becoming perfect, which is why it is worth stating separately
from maturity concerns — and why the deferral came with a named trigger (a representation that must
outlive one `GetCapability` call) rather than a vague "maybe later".

📌 **"Defer the dependency, adopt the information" is a real outcome.** Justin said *参考*, and the
review produced a bigger coverage gain (five standard-domain rows) than any of the three libraries
would have. `onnx-shape-inference` also became two free things: a preprocessing step for Trinity
that turns `[dynamic-shape]` declines into claims with zero Rust changes, and a second independent
source for the contrib fingerprints.

📌 **Share kernels freely; share claim predicates only when the vocabularies genuinely match.**
`RMSNormalization` reuses `simplified_layer_norm` verbatim. `ai.onnx::Attention` needed its own
predicate over the same kernel, because attribute names, the illegal-combination set and the
optional-input indices all differ. A predicate stretched to cover two schemas is wrong about one of
them, in the permissive direction.

📌 **`macro_rules!` gotcha:** `$min:literal ..= $max:expr` cannot accept a named constant, and you
cannot upgrade it to `$min:expr` either, because `..=` may not follow an `expr` fragment. `$min:tt`
takes both a literal and a bare ident.

📌 Verify commands, all green: `$env:ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC='1'`; `cargo ci`
→ **299 passed**, 7 ignored, 0 failed. (Clippy has one error in `src/trace.rs`, Niobe's file, not
mine.)
