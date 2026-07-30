# Switch (Vulkan-Compute) — history.md

## Learnings

### [SUMMARY] Sessions 1–16+: ash engine, first execution, probe correctness, runtime extents (2026-07-28–2026-07-30)

**Sessions 1–11 (archived):** ENGINE.md produced; ash+gpu-allocator selected; buffer-only tensor storage; build-time GLSL→SPIR-V; one command buffer per subgraph; per-edge barriers. Dual-backend barrier abstraction (`legacy` + `sync2`). Capability probe implemented. Loader diagnostics added; `apiVersion` capped to loader-reported to avoid `ERROR_INCOMPATIBLE_DRIVER` on 1.0 loaders. R5 (subgroup BASIC) removed as gate criterion — now a probed capability governing shader variants, per §7.0.

**Session 12 (2026-07-29T08:13:58-07:00) — push_next chain bug root-caused:**
`let _ = props2.push_next(..)` discards the pNext chain in ash 0.38. Every chained capability (subgroup size, stages, float16, SSC) read zero before fix. Three-state probe added (`subgroup_probe_valid: bool`). Intel Iris Xe validated as spec-conformance oracle. `is_uma` predicate corrected: "every heap is DEVICE_LOCAL" (not "largest DEVICE_LOCAL heap is HOST_VISIBLE") — ReBAR on RTX 4060 was returning wrong `true`. `timestamp_period_ns` and `timestamp_valid_bits` added to `Capabilities`.

**Session 13 (2026-07-29) — teardown order:**
Vulkan struct teardown order enforced by field declaration order: `instance` must be LAST. Rust drops fields top-to-bottom; declaring `instance` first caused STATUS_ACCESS_VIOLATION in `cdylib_load`.

**Session 14 (2026-07-29T13:42:45-07:00) — §7.9 and R5 re-evaluation:**
§7.9 five rules for capability probe discipline. R5 rationale was false (lavapipe reading was probe bug); policy stands on §7.0 principle that shortfalls degrade op coverage, not device availability. `cargo ci` — green, 300 tests.

**Session 15 (2026-07-29T20:26:56-07:00) — SkipSimplifiedLayerNorm:**
`SkipSimplifiedLayerNormalization` kernel: single-pass, two outputs, 1 KiB shared memory, LOCAL_SIZE_X=256.

**Session 16 (2026-07-29T21:14:03-07:00) — descriptor-set lifetime:**
Fixed VUID-03047: descriptor sets released before command buffer submission. Descriptor pool now per-dispatch.

**Sessions 17–18 (2026-07-30) — runtime extents:**
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. `CompiledKernel` reads real shapes at Compute via `GetTensorTypeAndShape`. OQ-15 resolved for M0/M1: re-record per shape. Device 0 (Intel): 161 claimed, zero validation errors. Device 1 (RTX 4060): 161 claimed. Variable seqlen correct on both devices. 97 additional nodes unlocked without new kernels.

**Current state:**
- `cargo ci` — green, 300 tests.
- 45 op rows Live on two GPUs.
- 161 Phi-3.5 nodes claimed under runtime extents.
- M0 open: validation positive control (criterion 3); CI lanes.
- Outstanding seam for Niobe: tile_config, workgroup size, shared-memory bytes, memory path must be reported by the engine.
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
Memory-pattern planner does not engage on run 1 — records during run 1, from run 2 onward hands back interior pointers. 52 interior pointers observed, identical on both devices, all within span. Every probe that ran each session once was pointed at the wrong moment. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON, written on first dispatch and at teardown. `dispatches_executed > 0` is the only reliable indicator that a shader ran.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards the pNext chain in ash 0.38. Every chained capability reads zero. Rule: rebind, never discard. This was the root cause of LVP2, `subgroup_size=0`, and ReBAR UMA misclassification.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. 45 op rows Live. 161 Phi-3.5 `MatMulNBits` nodes claimable. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE (Niobe — critical)
`(claimed_op_coverage, island_count, largest_island_flops)` reported together, per producer at version. `largest_island_flops` alone is not quotable. Portability floor = §7.2 (16 KiB / 256 invocations). `SUBGROUP_SIZE_IS_GUARANTEED=False`.
