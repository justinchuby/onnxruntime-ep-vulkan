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

---

## 2026-07-28T19:16:08-07:00 — Turn 2: template infrastructure landed before op #1

📌 The coordinator funded the §5 thesis verbatim: *build the machinery, not the ops*. What landed is a
table-driven registry (`OpSpec` + `op_table!`), first-class claim predicates with machine-readable
declines, a compile-time shader-variant table with a checked-in build manifest, and the
minimum-viable-subgraph rule as arithmetic. 69 elementwise rows are served by 5 claim predicates and 5
translate handlers — that ratio *is* the leverage, and it is now demonstrated rather than argued.

📌 **The trick that makes one predicate serve 60 ops: pass the row into the predicate.**
`ClaimPredicate = fn(&NodeView, &OpSpec)`. The predicate reads `spec.caps` / `spec.op_type` /
`spec.kernel` instead of closing over them. Without this, "adding an op is a row" is false and you are
back to one function per op. Same for `TranslateHandler`. Remember this shape for the reduction, GEMM
and shape-op templates.

📌 **`OpStatus::Staged(reason)` is how you land a table before its shaders without breaking the
claim/translate invariant.** A staged row is fully described and fully claim-tested, but
`claim_decision` declines it with `[staged]` before the predicate runs, and `is_registered`/`spec_for`
return nothing so it can never reach `Compile`. M0's "claims zero nodes" stays literally true; Trinity
gets real "must NOT be claimed" assertions immediately; going live is a one-word diff. Use distinct
staging reasons (`NO_SHADER`, `NEEDS_PARAMS`, `NEEDS_CAST_MATRIX`) — the table then reads as a sorted
backlog and shows when 12 rows share one blocker.

📌 **Machine-readable declines without touching another owner's ABI.** `ep.rs` (Tank's) constructs
`Cow::Borrowed/Owned` reasons itself and calls `.clone().into_owned()`, so `DeclineReason` had to stay
`Cow<'static, str>`. Solution: structure by *construction* — `decline(code, detail)` renders
`"[tag] sentence"`, `DeclineCode::of_reason()` parses it back, `deny!`/`require!` stamp the code. One
construction site, three consumers (human log, Trinity assertions, Niobe histogram), zero cross-owner
edits. Generalisable lesson: when you need structure in someone else's type, put it in the value.

📌 **ORT C API facts verified against the vendored 1.28 header** (write these down, they are easy to get
wrong): `Node_GetInputs`/`Node_GetOutputs` are `_Out_writes_(n) const OrtValueInfo**` — you size then
fill. **`GetValueInfoTypeInfo` returns a `const OrtTypeInfo**` that is borrowed and must NOT be
released** — unlike the owning `GetTypeInfo`. Releasing it is a double-free.
`CastTypeInfoToTensorInfo` is `_Outptr_result_maybenull_` (non-tensor → null, no error).
`Node_GetAttributeByName` is also `_Outptr_result_maybenull_`. `ReadOpAttr` is
`(attr, type, void* data, size_t len, size_t* out)` — call it once to size, again to fill.

📌 ORT reports a **null `OrtValueInfo`** for an omitted *interior* optional input (`Clip(x, , max)`)
rather than shortening the input list. Hence `EdgeType`'s fields are both `Option` and `has_input(i)`
exists. Making "unknown" impossible to ignore is what stops a predicate from claiming an untranslatable
node. This is the MLX scar, confirmed in the Vulkan ABI.

📌 **`engine::KernelRequest::shader` is `&'static str`, so a variant stem can never be formatted at
runtime.** All six dtype stems are baked into `Kernel` at compile time via `concat!` inside the
`kernel!`/`stems!` macros. Stem order must match `ALL_DTYPES`; there is a test asserting it. Any future
template family must follow the same pattern.

📌 **A row's `caps` bitset *is* its variant set** — which killed §5.5's proposed `f16_relevant` gating
list before it was written. `Atanh` declaring `FLOAT` and `BitwiseAnd` declaring `INT` generate exactly
the variants each can use. 69 rows → 168 SPIR-V modules. The manifest
(`src/ops/shader_variants.txt`, TSV: stem / source / `-D` defines) is checked in, regenerable with
`MOUSE_BLESS_VARIANTS=1 cargo test`, and a test fails on drift — so the build's view of what shaders
exist and the registry's view of what it dispatches cannot silently diverge. `build.rs` consuming it is
a request to Switch, not something I can land.

📌 **Zero stride *is* the broadcasting implementation.** `ShapePlan::broadcast` right-aligns into
`[u32; MAX_RANK=6]`, computes contiguous strides over each input's own padded shape, and sets stride 0
on stretched axes. The GLSL contains no broadcasting logic whatsoever. `all_identical` flags the linear
fast path as a spec constant. Push layout `rank, elem_count, out_shape[6], strides[N][6]` = 104 bytes
worst case for a ternary — inside the 128-byte `maxPushConstantsSize` floor, asserted by a test. This
is why `MAX_RANK` is 6 and not 8.

📌 Subtlety I got wrong once: **leave `ShapePlan.rank` at 0 for all-scalar inputs**, don't clamp to 1,
or `out_dims()` returns `[1]` for what is genuinely a rank-0 output.

📌 **`REQUIRE_STATIC_SHAPES = true` in one place.** Honest consequence: symbolic `batch`/`seq` decoder
graphs currently decline *everything* with one dominant `[dynamic-shape]` bucket. That is the correct
signal and it promotes OQ-M1 (shape-agnostic recording) above almost every individual op. One constant
to flip when it lands.

📌 **Trinity's `tests/layering.rs` scans `src/ops/**` for forbidden *whole tokens*** after stripping
comments and strings: bare `sys`, `ort`, `ash`, `OrtNode`, `vk::`, `unsafe`. Word boundaries are
respected (`support`, `sort` are fine) but it reads doc comments too in some paths — my `ops/mod.rs`
docs deliberately never name them literally. All ABI work belongs in `registry.rs`, which is the
sanctioned exception.

📌 Named the table macro `op_table!`, not `ops!` — `ops!` is visually ambiguous with the `crate::ops`
module path in `crate::ops!{}` position. Invoked as `crate::op_table! { ... }`.

📌 Avoided `std::ptr::fn_addr_eq` in tests (MSRV/clippy risk); asserting on `status` /
`Staged(NEEDS_PARAMS)` tests the same intent without comparing function pointers.

📌 The `Recorder` mock `DispatchContext` is the highest-value test asset here — handlers are pure
`NodeDesc → KernelRequest`, so the entire template layer is tested with no Vulkan device, no shaders
and no ORT session. It also separates the failure modes: if `shape_plan` and the handler are asserted
correct, a conformance failure is in the GLSL.

📌 Partition rule shipped as two ordered gates: size (`min_nodes: 4` unless an anchor op is present)
then economics (`compute_ns > margin × transfer_ns`, `margin = 3.0`). **The margin is 3× and not 1×
because a cost model this crude is easily wrong by 2×, and it must fail towards "run on CPU", which is
always correct.** `TransferModel::fit` (least squares over `(bytes, ns)` samples) is Niobe's calibration
hook, so her measurements replace the placeholder constants instead of arguing with them.
`concentration()` (largest-island FLOPs ÷ total claimed FLOPs) is the number that separates "80% of
nodes across 40 islands" from "80% in one island" — the unit tests assert those two cases have
identical `node_coverage` and wildly different honest metrics.

📌 Rai ruled OQ-M6 🟢 GREEN: reading llama.cpp's MIT Vulkan shaders is permitted with **no attribution
obligation for reading and learning**. The operative test is *"could you write this code without
looking at the original?"* Obligations attach only on substantial source adaptation — and note SPIR-V
compiled from adapted GLSL is a derived work, which matters because we embed SPIR-V in the cdylib.
Practical discipline for GQA/MatMulNBits/LinearAttention: read, understand, close the tab, write.

📌 Verify commands, all green on Windows: `cargo build`; `cargo clippy --all-targets -- -D warnings`;
`cargo test --lib` → **141 passed** (was 45 at Tank's baseline). `cargo test --test layering` is 14/15,
the failure being Switch/Trinity's in-flight barrier lint on the untracked `src/vk/command.rs` — not
mine. Always re-check the baseline before blaming your own change for a red test in a repo with four
agents editing concurrently.

---

## 2026-07-28T21:01:56-07:00 — turn 3: contrib domain admitted, XL kernels funded

📌 The two rulings I was most careful about both went the ambitious way: `com.microsoft` is admitted
and the XL kernels are committed. Being honest about cost did **not** get the work descoped — it got
it scheduled with the cost acknowledged. The lesson is that flagging expense clearly is not the same
as arguing against something, and the coordinator explicitly valued the split claim (breadth in weeks,
end-to-end LLM in months) enough to report it upward that way. Keep splitting the claim.

📌 Contrib ops have **no opset guarantee**. `since_version = 1` forever, versioned by ORT *release*,
inputs and attributes added **in place**. So the opset window — the whole compatibility contract for
an `ai.onnx` row — is information-free for exactly the ops the LLM path needs. That is why
`ContribSchema` exists. Never treat a contrib row's opset column as meaning anything.

📌 The drift detector that actually works is **attribute-name enumeration**, not shape checking. ORT
materializes defaulted optional attributes, so the observed name set is the *effective* schema. An
unknown attribute name means the op may not be the op we think it is. `Node_GetNumAttributes` /
`Node_GetAttributes` / `OpAttr_GetName` are all in the vendored 1.28 header — check the header before
assuming a diagnostic is impossible.

📌 Decline codes must not be lumped. `[attribute]` = a value we chose not to support (backlog, bulk,
boring). `[contrib-schema]` = the schema moved under us (alarm, should be zero). Sharing a bucket
hides the alarm inside the backlog at exactly the moment it matters.

📌 Asymmetric failure directions decide how narrow to write a predicate. Too narrow ⇒ decline ⇒ CPU
fallback ⇒ *always correct*. Too wide ⇒ wrong answer at full speed. So write narrow, record the
confidence in the row, and widen with evidence from the decline histogram.

📌 In a repo with five agents editing concurrently, **re-read the file before every edit**. `epctl.rs`
grew a `sys::schema_baseline_for` lookup between my reading it and my editing it, and `sys.rs` grew a
`CONTRIB_SCHEMA_BASELINES` side table covering my eleven rows before I registered them. When someone
else's mechanism already solves your problem, do not delete it and do not silently duplicate it —
add a test asserting the two agree, and record the part where they don't.

📌 Two records of one fact is a drift hazard; a test is cheaper than a merge. The registry now asserts
its fingerprints and `sys`'s side table agree on the verification *date*, and deliberately does not
compare release strings because `LinearAttention`/`CausalConvWithState`/`QMoE` are main-branch-only.
The divergence is written down in the decision file rather than hidden by loosening the test.

📌 Predicate before kernel. The claim predicate holds the long uncertainty tail (schema reading,
attribute semantics, decline taxonomy) and needs no GPU and no shader to test. Landing eleven staged
rows with real predicates means the kernel author starts from a settled contract and Trinity can
assert "must NOT be claimed" today.

📌 Layering constraints C1/C2 (`DESIGN.md` §1.4) landed *while* I was writing the contrib rows and my
first cut tripped C2. Check `tests/layering.rs` for new constraints before assuming a red test is
your own regression — and note C2's check is a grep for the literal `schema_baseline` in non-test
`registry.rs`, so the accessor name is load-bearing.

📌 Verify commands, all green on Windows: `cargo build`; `cargo clippy --all-targets -- -D warnings`;
`cargo test` → **174 lib + 25 layering + 6 dump_capabilities, 0 failed** (was 141 lib last turn, 45 at
Tank's baseline).

📌 Clippy gotcha, again worth remembering: `for x in iter.filter(..) { return Err(..) }` trips
`clippy::never_loop`. Use `if let Some(x) = iter.find(..)`.
