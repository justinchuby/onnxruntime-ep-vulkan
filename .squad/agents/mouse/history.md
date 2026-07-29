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

---

## Cross-agent context appended (2026-07-29T09:00:39-07:00) — first-hardware round

📌 **Intel Iris Xe = spec-conformance oracle (2026-07-29, Morpheus D25 + Link):** Intel's implementation is strictest in the local device set (Vulkan 1.4.309, UMA, 32 KiB shared). When testing op claims and fingerprint predicates, verify passes on Intel first. A claim that only passes on NVIDIA but not Intel indicates either an undefined-behaviour exploit or a predicate that is too broad.

📌 **`onnx-shape-inference` (Python) adopted as Trinity harness preprocessing step (2026-07-29, Trinity + Morpheus D24):** Runs `infer_symbolic_shapes` over test models before ORT, converting `[dynamic-shape]` declines into claimable nodes. Mouse's fingerprint predicates will be exercised more thoroughly. `onnx-shape-inference` is also the C2 fingerprint cross-check oracle.

📌 **T3 sequencing finalised (2026-07-29, Morpheus D23):** `ai.onnx::Attention` first (standard-domain); GQA second (same exit criteria required). The `bind_aliased_output` seam must be confirmed with Switch before starting GQA implementation. No KV-cache or fp16 design decision as if only one consumer existed — design for both.

📌 **`norm.rs` rustfmt flagged red by `cargo ci` (2026-07-29, Niobe):** Must be fixed by Mouse before the next `cargo ci` pass.

📌 **Census corpus must be indexed by producer (2026-07-29, Morpheus D21):** A target model is "covered" only for a named producer. The M1 census item changes to per-producer. "Producer emits no GQA" must be an explicit row in the census output.
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

---

## Turn 8 — 2026-07-29 — mobius as producer of record, opset 24, `onnx-runtime-ir` on the merits

📌 **The mirror is not the repo.** I derived §4.18 from `justinchuby/onnx-genai-models`. The
authoritative producer is `onnxruntime/mobius`. Re-deriving found the earlier reading was wrong on
one substantive point: mobius **does** emit `com.microsoft::GroupQueryAttention`, via a
`RotaryAttentionToGQA` rewrite and a direct `_forward_gqa()` path, both gated on the target EP
advertising GQA support. So the op set is a function of producer × revision × **how we describe
ourselves to the producer**. The last factor is partly under our control and I had not considered
it at all.

📌 **Coverage claims need a producer *and* a revision.** Same argument Trinity used to pin
`accuracy_level`: unpinned, the reference drifts. Raised for Morpheus's §8.5.

📌 **The real find: an open-ended opset window was a correctness bug, not untidiness.**
`ai.onnx::Attention` gained optional input 6 `nonpad_kv_seqlen` at opset 24, and mobius defaults to
opset 24. My row said `23 ..= OPSET_ANY`, so the predicate would have claimed the static-cache form
and silently used the wrong per-batch causal offset. This is precisely the failure my own §7.1 rule
exists to prevent, and I had written the rule and then violated it in the window column. Lesson:
**the opset window is part of the claim predicate, not metadata about it.**

📌 Opset windows are **schema-version** windows, not model-opset windows. `Node_GetSinceVersion`
returns the resolved op schema version — an `Add` in an opset-24 model reports 14. That is why
closing `RMSNormalization` at `23 ..= 23` does not exclude opset-24 graphs, and it is the fact that
makes closed windows cheap. I nearly got this backwards.

📌 The policy I settled on, deliberately *not* blanket: close a window when the op has ever gained
an input or attribute across a revision (`Attention`, Q/DQ); leave it open when it has not
(elementwise). Closing all ~70 elementwise windows at 27 would decline valid opset-28 graphs the
day onnx 1.23 ships — that is death-by-fallback bought with no evidence. The closed set is the set
with evidence behind it.

📌 **`onnx-runtime-ir`: separating the two objections paid off.** Justin retired the prudential
half by owning the crate; the structural half was untouched by that, and because I had recorded
them separately I could say so without appearing to resist. Answer is still defer, but for one
reason instead of two, and the disposition changed: the trigger is now a switch rather than a
question. Generalisable: **when deferring, name the objections separately and name the trigger** —
it makes the later re-evaluation cheap and non-adversarial.

📌 Honest self-check on the trigger: I had to argue *against* my own preference here. Compiled-graph
caching and prepack keying both sounded like they fired it; neither does. Caching needs a stable
hash key, not a graph; prepack is node-local. Wanting a nice library is not a requirement.

📌 `Swish` was free coverage — one `#elif` in `ew_unary.comp`, one op code, one row, two variants.
That is the §5.2 leverage thesis behaving exactly as advertised, on an op that is in every LLM MLP.

⚠️ **Build environment:** Switch had uncommitted, non-compiling work in `rust/src/vk/caps.rs`
(four `E0503` borrow errors), so `cargo build` failed in the main tree through no fault of mine. I
verified in a throwaway `git worktree add .mouse-verify HEAD --detach`, copied my files in, built
and tested there, then removed the worktree. **Reusable technique**: it validates your own changes
against a green baseline without touching another owner's in-flight files. Do not `git stash`.

📌 Verify recipe with real hardware:
`$env:VULKAN_SDK="C:\VulkanSDK\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\Bin;$env:PATH"` — all 170
variants compile with `glslc`, no `ALLOW_MISSING_GLSLC` needed. Regenerate the variant table with
`MOUSE_BLESS_VARIANTS=1 cargo test --lib variants`. **306 passed / 0 failed / 7 ignored**, clippy
clean, in the verify worktree (HEAD + my files).

📌 A device dispatch test now passes: `vk::dispatch_integration::add_f32_dispatches_end_to_end`.
Switch's path is real. My kernels are still `UNEXERCISED` — none of mine has executed — but the
seam to exercise them exists now, and that is the T1 unblock.

### Turn 8 addendum — the `Attention`-24 ruling and what it taught about C2

📌 **Justin: no opset bump, implement the corrected semantics.** R4 closed. The lesson is not about
attention — it is that my two drift detectors are *both* version-based and therefore share one
blind spot: a semantic correction applied without a version change moves behaviour while every
number stays put. `ContribSchema` cannot fire. The opset window cannot fire. I had been treating
`ai.onnx` as the safe domain and `com.microsoft` as the risky one; that framing is wrong. **An
opset window is a strong guarantee about interface and no guarantee about behaviour.**

📌 There is no fix to write. That was the uncomfortable part — the honest output is a documented
limitation plus a routed dependency (Trinity pins `onnx`), not a mechanism. Recording a gap so the
machinery stops *looking* more complete than it is, is a legitimate deliverable.

📌 Generalisable: **write the limitation where the reader finds the mechanism.** I put §9.4.1 in the
doc, a "what this does not detect" section in the `ContribSchema` doc comment, and a line at the
decline site. Anyone reaching for the fingerprint struct now learns its boundary in the same breath
as its purpose.

📌 On how it surfaced: nothing detected it. Following §7.1 literally — narrow and decline, never
guess — meant going to *read* the 23→24 diff instead of assuming an open-ended window was harmless.
The errata was sitting next to the interface change in the same source file. Rules that force you
to go and look pay off on things they were not aimed at.

📌 Still true and worth repeating: **zero of my kernels have executed.** `add_f32` executing on both
local GPUs with validation layers clean is Switch's path working, not mine. The seam exists now;
the honest statement about my rows is unchanged.

---

## Turn 9 — the opset range (2026-07-29)

**Directive:** support the full ONNX opset range, not just what mobius emits.

**The number.** "Latest opset" has two correct answers and they disagree by one. onnx v1.22.0
registers 27 (`map_[ONNX_DOMAIN] = {1,27}`, what `onnx_opset_version()` returns, what
`make_model` stamps) but still reports `last_release_version_map_[ONNX_DOMAIN] = 26`. Justin
said 26, the coordinator measured 27, and neither was wrong. Lesson: when two people disagree about
a version number, the likely cause is that the library exposes two of them for different purposes.
Go read the header before picking a side. Both are now constants with a test.

**The result nobody expected: there was nothing to extend.** Every closed window I set last turn —
Attention 23..=24, RMSNorm 23..=23, RotaryEmbedding 23..=23, TensorScatter 24..=24, Swish 24..=24,
Q/DQ 21..=25 — is already at the newest schema version that *exists*. Opset 25 is a type-constraint
expansion, 26 is BitCast/CumProd, 27 is the SSM ops plus Range. So an opset-27 model is claimable
today for all of them.

That is entirely because windows are keyed on **schema version**, not model opset. Had I keyed them
on model opset, satisfying this directive would have meant re-reading and re-testing six predicates
at four more opsets. **A framing choice made for correctness reasons paid off as speed.** Worth
remembering the shape of that: the cheap answer to a scope directive usually comes from a
representation decision made earlier, not from working faster.

**The actual coverage win was a side effect.** Reading the opset-27 contents to answer a question
about *bounds* turned up that `LinearAttention` and `CausalConvWithState` are now `ai.onnx`
ops. I had them only as low-confidence `com.microsoft` main-branch rows. That is §4.18 recurring
on the Qwen3.5 hybrid path — same computation, two producers, two spellings — and it would have
gone unnoticed indefinitely because nothing about our contrib rows looks wrong. **Range checks find
things that are not range problems.** Read the whole diff, not the part that answers your question.

**A new argument for narrow predicates.** onnx#7913 swapped the meaning of `qk_matmul_output_mode`
1 and 2 with no opset bump. It cannot touch us, because we claim only mode 0. Generalised:
*the attributes you decline cannot drift under you.* I had been defending narrow claiming purely on
"we have not implemented that"; it is also a structural defence against the C2 blind spot. The
residual is exactly the attributes we do claim — for Attention that is the causal mask and GQA
repetition, both of which ONNX has silently corrected.

**Ambiguity is not always worth resolving.** Two sources disagreed about whether Q/DQ's
`precision` arrived at opset 23 or 25. I did not settle it: declining every non-default value is
correct under both readings and costs nothing. Check whether the conservative action is
reading-independent before spending time on the reading.

**Process.** Two `vk::barrier` tests fail under the parallel runner and pass with
`--test-threads=1` — a shared probe file in Switch's code. Confirmed it was pre-existing by
running the two tests in isolation rather than assuming. Flagged, not touched.

---

## Turn 10 — 2026-07-29 — The Foundry Local census: two real graphs, five wrong conclusions

**The lesson landed a third time and the recurrence is the finding.** Turn 6: wrong producer.
Turns 8-9: right producer, wrong revision. This turn: right producer at a pinned revision, whose
**actual output I had never read**. Each correction was one level more concrete than the last,
which is the tell that the underlying rule was still too abstract. The form I now hold:

> A claim about what a producer emits is not evidence until it has been read off a graph that
> producer actually produced. Builder source is intent; the model file is the fact.

I had written §8.5 ("a coverage number without a named producer at a version is not well-formed")
and still spent two turns reasoning about op sets from `builder.py` and schema headers. Writing a
rule is not the same as being governed by it. The concrete tell I missed: I could quote a builder's
rewrite-rule preconditions but could not quote a single node's input arity.

**I had been reasoning about the ceiling only.** Four turns of narrowing and extending upper
bounds — 23, 24, 26, 27 — and both real models import `ai.onnx` at **14** and **21**. The floor
excluded them outright from every standard-domain LLM row. Ranges have two ends and I had treated
one as interesting and one as settled. Ask which end the evidence actually constrains.

**Two of the five errors were permissive, which is the direction that costs.** `group_query_attention`
never read inputs 1 and 2, so it would have *claimed* a packed-QKV node and handed the kernel a
fused tensor where it expected a query. I had recorded "PackQKVForGQA never matches Qwen3" and
generalised a statement about one model family into a statement about the producer. Both cached
models pack on every layer. **A negative finding about one model is not a negative finding about
the world** — the scope of a "never" is exactly the scope of the evidence for it.

**Simulation beat argument, and contradicted it.** I had argued death-by-fallback abstractly in §7.
Running an island simulation on real graphs produced something I would not have predicted: adding
`Cast` to the claim set on gpt-oss raises coverage 28%→54% **and islands 52→125**. Claiming more
ops made partitioning strictly worse. And Phi-3.5 sits at 34-35 islands from T1 through T3 and then
collapses to *one island of 364* the moment `MatMulNBits` lands. Two conclusions I could not have
reached by reading: partial coverage of these graphs is worth nothing, and coverage percentage is
an actively misleading metric. **Cheap simulation against a real artefact is worth more than a
careful argument about a hypothetical one.** ~120 lines of Python.

**"Right answer, wrong reason" is a defect.** Real `QMoE` nodes were declining as
`[contrib-schema]` because my fingerprint was missing two attributes — but they should decline on
top-k (4, and we admit 1|2). The decline was correct and the diagnosis was wrong, which is exactly
what the machine-readable decline codes exist to prevent. **Check that declines fire for the reason
you think.** A census makes that checkable; nothing else does.

**A third registry category appeared that I had not modelled.** `SimplifiedLayerNormalization`
carries `domain == ""` — no ONNX schema, but the standard domain, so `Node_GetSinceVersion` is
meaningless and only a fingerprint detects drift. My taxonomy was binary (standard-with-opset vs
contrib-with-fingerprint) and reality had a third case. I handled it with an explicit allow-list
capped at four entries **by test**, so if it grows the test forces a real `Domain` variant rather
than letting the list quietly become the normal case. **When you patch a taxonomy with a list, put
a bound on the list.**

**Discipline held under temptation.** The census made it obvious how to make the coverage numbers
look good: lower the floor, admit `do_rotary`, widen top-k. I widened nothing and *narrowed* GQA.
The point of measuring what is out there is to know it, not to move predicates until the count
improves.

**Still true and worth repeating: zero of my kernels have executed.** Phi-3.5 is now a runnable
target sitting on this disk, which is a better first target than Qwen3.5 on every axis (MHA not
GQA, softcap 0, no SWA, no sinks, no QK norm, symmetric RTN, uniform bits/block, cold control
flow, 366 nodes). That changes what is worth building first, not what has been built.
