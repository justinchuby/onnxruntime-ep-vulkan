# Mouse (Op-Coverage) — history.md

## Learnings

### [SUMMARY] Turns 1–16: op plan, infrastructure, contrib rows, census, Live ops, extents, runtime (2026-07-28–2026-07-30)

**Turns 1–5 (archived):** 174-op inventory (ai.onnx standard domain). Table-driven registry (`registry.rs`). Eleven staged contrib rows for `com.microsoft` ops. Machine-readable claim log (JSON Lines, flushed per decision). GQA fingerprint self-audit found two permissive bugs (corrected).

**Turn 6 — in-house crate review (2026-07-29T07:14:15-07:00):**
`onnx-genai-models` / `mobius` builder emits `ai.onnx::Attention`/`RMSNormalization`/`RotaryEmbedding` when our EP advertises GQA support, not the `com.microsoft` variants. Standard-domain rows are required for the mobius path, not optional.

**Turn 8 — mobius as producer of record (2026-07-29):**
Authoritative producer is `onnxruntime/mobius@87fd878`, not `justinchuby/onnx-genai-models`. Default opset 24. `ai.onnx::Attention` gained optional input 6 `nonpad_kv_seqlen` at opset 24 — predicate written against opset 23 would have claimed and returned wrong logits. `onnx-runtime-ir` trust objection withdrawn (it is Justin's own crate); structural objection stands independently.

**Turn 9 — opset range (2026-07-29):**
`ONNX_OPSET_LAST_RELEASED=26`, `ONNX_OPSET_REGISTERED=27`. Two constants, test asserts. `LinearAttention-27` and `CausalConvWithState-27` registered (Qwen3.5-hybrid ops standardised in onnx 1.22.0).

**Turn 10 — Foundry Local census (2026-07-29):**
Phi-3.5-mini-instruct (`ai.onnx`=14) and gpt-oss-20b (`ai.onnx`=21) read from disk. Five findings: `OPSET_STD_LLM=23` excludes both; `do_rotary=1` universal; packed QKV predicate requires both inputs; `SimplifiedLayerNormalization` has `domain=""` not `ai.onnx` in both graphs; `QMoE` top-4 (not top-1|2). §8.5 third strengthening: "builder source is intent; the model file is the fact." Metric upgraded to triple: `(claimed_coverage, island_count, largest_island_flops)`.

**Turn 11 — first Live row, and oracle boundary (2026-07-29):**
`Add` Live for f32 only. `EXERCISED` evidence list introduced. `Sub/Mul/Div/Pow` stay Staged — template similarity is not evidence (D-M11-02). "An oracle knows ORT's correctness, not our dispatch correctness."

**Turn 12 — test contradiction and elementwise flip (2026-07-29):**
`OnceLock` bug in claim log path fixed — re-reads env var per decision. Profiling-JSON retained for `is_vulkan_claimed` (post-load env var changes unreliable for DLL on Windows; CLAIM_LOG still correct for subprocess use). Three-layer skip contradiction found and closed.

**Parameter tail (2026-07-29):**
Four-float push-constant tail unconditionally at block end. 7 activations unlocked (Selu, Elu, HardSigmoid, Shrink, ThresholdedRelu, LeakyRelu, CeluAlpha). `Clip` excluded (two params, NaN semantic mismatch).

**MatMulNBits Live (2026-07-29):**
`com.microsoft::MatMulNBits` Live for all `M`, fp32 and fp16. GEMV layout from oracle (`A=I`). All 161 Phi-3.5 nodes are fp16 (`bits=4`, `block_size=32`, 3-input symmetric, `K∈{3072,8192}`, `N∈{3072,8192,9216,32064}`). fp16 through `unpackHalf2x16/packHalf2x16` — no 16-bit storage capability needed.

**Decline census (2026-07-29 evening):**
First-match histogram is a ceiling, not a measurement. Full-set Phi-3.5: `dynamic-shape=356` (not 258); 98 of 100 staged nodes are also shape-blocked; landing all staged kernels unlocks 0 nodes. gpt-oss: `dynamic-shape=342 > staged=197` — first-match would have reversed a correct ruling.

**Runtime extents (2026-07-29):**
`ENGINE_ACCEPTS_RUNTIME_EXTENTS` flag; `shape_class` computed independently; `predicate_ok_runtime_extents` field in JSONL. 227 Phi-3.5 nodes predicate-clean under runtime extents; 161 (`MatMulNBits`) claimable immediately once Switch flips the flag (now done).

**Current state:**
- 45 Live rows. `cargo ci` — green.
- 161 Phi-3.5 nodes claimable; GQA + SkipSimplifiedLayerNorm are next for remaining coverage.
- `onnxruntime/mobius@87fd878` is the pinned authoritative producer.
- Standing rules: closed windows = schema-version windows; `do_rotary=1` must precede GQA claim; metric is the triple, reported per producer at version.
---

## 📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)

### Worktree layout and inbox portability constraint
The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree. `.squad/decisions/inbox/` is **gitignored** — records written in a worktree do NOT travel with the branch. The inbox in `main` is authoritative.

### Vulkan SDK path
`C:\VulkanSDK\1.4.350.0` — installed but **not on the default PATH**. `glslc` discovery must search this path; `VULKAN_SDK` env var is the canonical pointer.

### Local hardware — both GPUs pass the §7.2 gate
- Intel Iris Xe: Vulkan 1.4.309, UMA, `subgroup_size=32`, 32 KiB shared. Spec-conformance oracle. Do not special-case Intel.
- RTX 4060 Laptop: Vulkan 1.4.325, discrete, `subgroup_size=32`, 48 KiB shared.
- Lavapipe (CI): `subgroup_size=8`, 32 KiB shared, `is_uma=true`. CI exercises the mobile-warp path. LVP2 retracted.

### ORT's planner hands back interior pointers from run 2 onward
Memory-pattern planner does not engage on run 1. From run 2 onward hands back interior pointers. 52 observed, identical on both devices, all within span, `pointers_in_guard_band=0`. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON. `dispatches_executed > 0` is the only reliable indicator.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards pNext chain. Rebind, never discard. Root cause of LVP2, `subgroup_size=0`, ReBAR UMA misclassification.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE (Niobe — critical)
`(claimed_op_coverage, island_count, largest_island_flops)` per producer at version. Portability floor = §7.2. `SUBGROUP_SIZE_IS_GUARANTEED=False`.
