# Squad Decisions

<!-- Round 1: 2026-07-28T17:59:54-07:00 (design session) -->
<!-- Round 2: 2026-07-28T22:28:08-07:00 (implementation round) -->

## Active Decisions

### 2026-07-28T17:59:54-07:00: Commit and push work to origin/main regularly

**By:** Justin Chu (via Copilot)
**What:** The team must commit and push work to `origin/main` regularly — do not accumulate large uncommitted working trees. Commit at meaningful checkpoints: a design doc landed, a module scaffolded, a test suite added.
**Why:** User directive — keeps work durable and reviewable.

---

### 2026-07-28T17:59:54-07:00: Mirror the `onnxruntime-mlx` architecture rather than design fresh

**By:** Morpheus
**What:** The crate layout, module boundaries, claim→fuse→compile→run pipeline, registry design, `NodeView`/`NodeDesc` split, panic-guarded FFI, RAII teardown, version scheme (`0.<ORT_API_VERSION>.<patch>`), and the `rust/` `docs/` `tests/` `bench/` `python/` repo shape are taken from `onnxruntime-mlx` one-to-one. Divergences are enumerated in `docs/DESIGN.md` §12.
**Why:** Justin's directive, and independently correct: that architecture is proven against a real ORT plugin-EP ABI, has survived a C++→Rust rewrite, and its structure is what made op coverage grow linearly (one handler + one claim + one registration line, zero boundary-layer edits). Re-deriving it would cost weeks and produce something worse.
**Alternatives rejected:** (1) Design a Vulkan-native architecture from scratch — no upside; loses a proven ABI integration and the claim/translate-can-never-disagree invariant. (2) Copy ExecuTorch's AOT structure — wrong host model; ORT partitions through `GetCapability`, not a Python export pass.

---

### 2026-07-28T17:59:54-07:00: EP identity: `VulkanExecutionProvider`, `libonnxruntime_vulkan_ep`, `0.27.0`

**By:** Morpheus
**What:** Registered EP/device name `VulkanExecutionProvider`. Library base name pinned to `onnxruntime_vulkan_ep` via `[lib] name` regardless of crate name. Vendor string `onnxruntime-ep-vulkan`. Version `0.<ORT_API_VERSION>.<patch>` = `0.27.0`, single-sourced from `CARGO_PKG_VERSION`. Vendor ID read from `VkPhysicalDeviceProperties.vendorID`, not hardcoded.
**Why:** The EP name is in every user's provider list and can never change without breaking them. The version scheme encodes plugin-ABI compatibility. A per-device vendor ID is required by the cross-platform mandate.
**Alternatives rejected:** `VulkanEP` / `VkExecutionProvider` (inconsistent with ORT naming); semantic versioning independent of ORT (hides ABI compatibility fact); fixed vendor ID (meaningless cross-platform).

---

### 2026-07-28T17:59:54-07:00: Two enforced layering rules, checked by CI

**By:** Morpheus
**What:** (1) The ORT C ABI never appears in `rust/src/ops/`. (2) Raw Vulkan handles never appear in `rust/src/ops/`. Op handlers see only `NodeDesc`, `NodeView`, `TensorRef`, and a `DispatchContext` trait. Enforced by module privacy *and* a CI lint that greps `rust/src/ops/` for `sys::`, `Ort`, `ash`, `vk::`, and `unsafe`, failing the build on a hit.
**Why:** If ORT/Vulkan concerns bleed into 60 op modules, the first driver quirk becomes a 60-file change. This boundary is the only thing that keeps op coverage a linear cost. A rule not mechanically checked is a suggestion.
**Alternatives rejected:** Convention plus code review (fails at scale or under time pressure); per-op `vkCmdDispatch` (no such thing as simple once barriers and non-coherent memory are involved).

---

### 2026-07-28T17:59:54-07:00: Phased memory model: host I/O in M0/M1, device allocator in M2

**By:** Morpheus
**What:** M0/M1 advertise no device allocator, return null from `GetDefaultMemoryDevice`, keep subgraph I/O in host memory, and stage uploads/downloads inside `Compute`. M2 implements `OrtAllocator` + `OrtDataTransferImpl` + a real `OrtMemoryDevice`. Weights are uploaded once at `Compile` time in all phases.
**Why:** The allocator/data-transfer surface is the highest-uncertainty part of the plugin-EP ABI and has nothing to do with proving Vulkan dispatch correctness. Deferring it gets a correct, CPU-verified, cross-platform op running in the shortest path without designing into a corner — the M2 contract is written in `DESIGN.md` §6.2.
**Alternatives rejected:** Device allocator in M0 (couples riskiest ABI work to "does anything work" milestone); permanent host I/O (guarantees device-round-trip at every partition boundary, slower than CPU for fragmented graphs).

---

### 2026-07-28T17:59:54-07:00: `VkBuffer` handle identity via opaque-handle registry, not buffer device addresses

**By:** Morpheus
**What:** ORT's allocator API is pointer-based; a `VkBuffer` is not a pointer. `Alloc` returns a unique tagged 64-bit value from a reserved range, resolved to `(VkBuffer, offset)` through a process-wide map.
**Why:** `VK_KHR_buffer_device_address` is optional on every baseline and MoltenVK support is partial. A hash lookup on a handle is not the bottleneck in a graph that just submitted a command buffer.
**Alternatives rejected:** Require `bufferDeviceAddress` (costs macOS and some Android for a nanosecond); one `VkBuffer` per allocation (defeats suballocation, blows past `maxMemoryAllocationCount`). Tracked as OQ-3; revisit only if profiling says so.

---

### 2026-07-28T17:59:54-07:00: Record-once / replay-many, keyed on shape

**By:** Morpheus
**What:** `Compute` hashes the concrete input shapes, looks up a cached recorded `VkCommandBuffer`, and replays it. A miss re-records and caches. Prepacked weights and compile-time-created pipelines mean a replay is submission-only.
**Why:** Direct analog of MLX EP's `mlx_compile` fast path and ExecuTorch's model. llama.cpp re-records every eval — fine for a few large matmuls, wrong for a graph of many small dispatches. Recording is where descriptor updates and barrier emission cost live; per-inference recording would dominate runtime.
**Alternatives rejected:** Re-record every `Compute` (simpler, pays graph-construction cost per inference forever — kept as M0 implementation only because M0 has one node); fully shapeless recording from the start (premature, M3+).

---

### 2026-07-28T17:59:54-07:00: Vulkan API baseline: capability-set, not version floor (consolidated)

**⚠️ SUPERSEDED by:** "Vulkan baseline frozen: minimal device gate, no required extensions" (2026-07-28T22:28:08-07:00). The intermediate capability-set requiring `synchronization2` and `subgroup_size_control` extensions was replaced by a more minimal gate with no required extensions. Preserved for reasoning trail.

**By:** Morpheus, Switch, Link, Fact Checker, Justin Chu via Copilot
**What (at time of writing):** Vulkan ≥1.1 + compute queue + `synchronization2` (core or ext) + `subgroup_size_control` (core or ext) + subgroup BASIC+ARITHMETIC + workgroup/shared-memory minimums. Everything else capability-probed.
**Why superseded:** Link measured a 31.43% Android gap for sync2 and 14.12% for subgroup_size_control. Under Justin's compatibility-first directive, any hard device requirement must be justified by "no op we will ever ship can work without it." Those extensions are not universally required; capability shortfalls degrade op coverage, not device availability.

---

### 2026-07-28T17:59:54-07:00: Ruthless v1 non-goals

**⚠️ SUPERSEDED by:** "`com.microsoft` contrib ops admitted: eleven named ops" (2026-07-28T22:28:08-07:00). The exclusion of all `com.microsoft` contrib ops was reversed by user ruling. Preserved for reasoning trail.

**By:** Morpheus
**What (at time of writing):** Out of scope for v1: training, opset completeness, dynamic-shape fast paths (M0–M2), data-dependent output shapes, fp64, quantized ops, attention fusion, graph-level op fusion, mobile-first tuning, image/texture-backed tensors, multi-GPU/multi-queue, cooperative matrix, and all `com.microsoft` contrib ops.
**Why superseded:** Mouse verified from ORT GenAI model builder source that a Qwen3.5 graph contains 9 contrib ops emitted directly by the builder. An EP that declines the domain cannot run Qwen at all. Justin ruled the contrib domain in scope. The XL kernels (MatMulNBits, GQA, etc.) are now committed deliverables. Non-goals excluding training, fp64, multi-GPU, etc. remain in force.

---

### 2026-07-28T17:59:54-07:00: `VkBuffer` handle identity via opaque-handle registry, not buffer device addresses

**⚠️ SUPERSEDED by:** "OQ-3 resolved: reserved virtual-address opaque-handle registry, no BDA" (2026-07-28T22:28:08-07:00). The mechanism is now more precisely defined (reserved-VA, not a synthetic token). BDA is explicitly not carried.

**By:** Morpheus
**What (at time of writing):** ORT's allocator API is pointer-based. `Alloc` returns a unique tagged 64-bit value from a reserved range, resolved to `(VkBuffer, offset)` through a process-wide map.
**Why superseded:** Tank's analysis showed a synthetic token breaks under ORT's internal pointer arithmetic (base+offset, align_up). Reserved virtual address space answers this by construction — spans are real OS memory reservations.

---

### 2026-07-28T17:59:54-07:00: Two enforced layering rules, checked by CI

**By:** Morpheus
**What:** (1) The ORT C ABI never appears in `rust/src/ops/`. (2) Raw Vulkan handles never appear in `rust/src/ops/`. Op handlers see only `NodeDesc`, `NodeView`, `TensorRef`, and a `DispatchContext` trait. Enforced by module privacy *and* a CI lint that greps `rust/src/ops/` for `sys::`, `Ort`, `ash`, `vk::`, and `unsafe`, failing the build on a hit. Additionally: all Vulkan barrier types (`vkCmdPipelineBarrier`, `VkBufferMemoryBarrier`, `VK_PIPELINE_STAGE*`, `VK_ACCESS*`) are restricted to `rust/src/vk/barrier.rs` by a separate layering rule.
**Why:** If ORT/Vulkan concerns bleed into 60+ op modules, the first driver quirk becomes a 60-file change. A rule not mechanically checked is a suggestion.
**Alternatives rejected:** Convention plus code review (fails at scale); per-op `vkCmdDispatch` (no such thing as simple once barriers are involved).

---

### 2026-07-28T17:59:54-07:00: Phased memory model: host I/O in M0/M1, device allocator in M2

**By:** Morpheus
**What:** M0/M1 advertise no device allocator, return null from `GetDefaultMemoryDevice`, keep subgraph I/O in host memory, and stage uploads/downloads inside `Compute`. M2 implements `OrtAllocator` + `OrtDataTransferImpl` + a real `OrtMemoryDevice`. Weights are uploaded once at `Compile` time in all phases.
**Why:** The allocator/data-transfer surface is the highest-uncertainty part of the plugin-EP ABI. Deferring it gets a correct, CPU-verified, cross-platform op running in the shortest path without designing into a corner — the M2 contract is written in `DESIGN.md` §6.2.
**Alternatives rejected:** Device allocator in M0 (couples riskiest ABI work to "does anything work" milestone); permanent host I/O (guarantees device-round-trip at every partition boundary, slower than CPU for fragmented graphs).

---

### 2026-07-28T17:59:54-07:00: Record-once / replay-many, keyed on shape

**By:** Morpheus
**What:** `Compute` hashes the concrete input shapes, looks up a cached recorded `VkCommandBuffer`, and replays it. A miss re-records and caches. Prepacked weights and compile-time-created pipelines mean a replay is submission-only.
**Why:** Direct analog of MLX EP's `mlx_compile` fast path. llama.cpp re-records every eval — fine for a few large matmuls, wrong for a graph of many small dispatches.
**Alternatives rejected:** Re-record every `Compute` (pays graph-construction cost per inference forever — kept as M0 implementation only because M0 has one node); fully shapeless recording from the start (premature, M3+).

---

### 2026-07-28T17:59:54-07:00: Op growth by family, prioritized by island-merging

**⚠️ SUPERSEDED by:** "Op coverage: model-family-driven, 174-op, 6-tier plan (OP_COVERAGE.md)" (2026-07-28T22:28:08-07:00). Op growth by island-merging and the anti-fragmentation metric remain valid principles, now embedded in the authoritative plan. The original entry's sequencing-by-family-type approach is superseded by sequencing-by-model-family (LLM → MoE → multimodal → SSM → conv).

---

### 2026-07-28T17:59:54-07:00: Milestone M0 defined as an end-to-end vertical slice, not a layer

**By:** Morpheus
**What:** M0 = a stock ORT loads the plugin, enumerates a Vulkan device, runs a graph containing a single `Add` node on that device, and matches the ORT CPU EP within tolerance, on Windows and Linux, on a software rasterizer, in CI. Every team member ships something into it.
**Why:** The MLX Rust rewrite began exactly this way — a single-`Add` spike proved the two unknown boundaries and immediately caught a real per-session leak. A vertical slice proves the boundaries; a horizontal layer proves nothing until the last layer lands.
**Alternatives rejected:** M0 = "Vulkan engine works standalone" (defers all ABI risk); M0 = "EP loads and claims nodes, no compute" (proves the easy half).

---

### 2026-07-28T17:59:54-07:00: Divergences from the reference are enumerated, not implied

**By:** Morpheus
**What:** `DESIGN.md` §12 lists all deliberate differences from `onnxruntime-mlx` with reasons. Anything not on that list is intended to match the reference. A PR that diverges without adding a row is a review rejection.
**Why:** "We'll refactor later" is a decision, not an excuse. Both need to be written down at the moment they are made, or the reference stops being a reference.
**Alternatives rejected:** Track divergences in commit messages (not discoverable at review time).

---

### 2026-07-28T17:59:54-07:00: Rust Vulkan crate: `ash` + `gpu-allocator`

**By:** Switch
**What:** Use `ash` (raw Vulkan bindings) as the Vulkan dependency, supplemented by `gpu-allocator` for suballocation.
**Why:** `ash` is a thin binding over the Vulkan C API with zero abstraction overhead. `vulkano` adds a redundant ownership abstraction conflicting with `engine.rs`. `wgpu` hides push constants, specialization constants, and pipeline cache. `gpu-allocator` is the pure-Rust VMA equivalent, used in production by Bevy and wgpu-hal.
**Alternatives rejected:** `vulkano` (conflicts with engine abstraction layer); `wgpu` (hides required Vulkan primitives).

---

### 2026-07-28T17:59:54-07:00: Buffer-only tensor storage for v0

**By:** Switch
**What:** All tensors are backed by `VkBuffer` (storage buffer). Image storage is deferred until a specific op family (Conv, tiers 5c/6) demonstrates a measurable benefit.
**Why:** Target workloads are decoder-dominated with linear memory access. Image storage requires layout transitions without performance benefit for these access patterns. Barrier reasoning is simpler with buffers only.

---

### 2026-07-28T17:59:54-07:00: One command buffer per subgraph; no per-op submission

**By:** Switch
**What:** The entire fused subgraph is recorded into one `VkCommandBuffer` and submitted once via `vkQueueSubmit`. Per-op submissions are prohibited.
**Why:** `vkQueueSubmit` overhead is measured in microseconds per call. Per-op submission on a 100-op subgraph would add milliseconds of CPU-side overhead. Single submission mirrors the MLX EP's single `mlx_eval`.

---

### 2026-07-28T17:59:54-07:00: Barrier placement: per data edge, not per dispatch

**By:** Switch
**What:** A `vkCmdPipelineBarrier2` (or `vkCmdPipelineBarrier` fallback) is inserted after each dispatch, once per consumer edge of each output buffer. No global `ALL_COMMANDS → ALL_COMMANDS` barrier.
**Why:** Per-edge barriers let the driver schedule independent dispatch pairs in parallel. A global barrier serializes the entire GPU pipeline unnecessarily.

---

### 2026-07-28T17:59:54-07:00: Shader source: GLSL compiled to SPIR-V at build time; embedded in cdylib

**By:** Switch
**What:** Shaders are written in GLSL, compiled by `glslc` during `cargo build` via `build.rs`, and embedded as byte slices in the cdylib. Tank's `build.rs` must locate and invoke `glslc`, iterate `shaders/glsl/`, write SPIR-V to `OUT_DIR/spv/`, generate `OUT_DIR/shader_modules.rs`. See also: "OQ-4 resolved: hard Vulkan SDK dependency" (2026-07-28T22:28:08-07:00) — no checked-in SPIR-V fallback.
**Why:** Both reference implementations (llama.cpp, ExecuTorch) use this pattern. Guarantees a self-contained plugin with deterministic SPIR-V output.

---

### 2026-07-28T17:59:54-07:00: ORT CPU EP as the sole correctness oracle, with mandatory claim assertions

**By:** Morpheus
**Updated:** Oracle validated for quantized path (Trinity empirical result, 2026-07-28T22:28:08-07:00). See "Oracle validated for quantized path; oracle knobs pinned."
**What:** Every op test compares against ORT's own CPU EP running the same ONNX model. Every op test **must** assert the node actually ran on `VulkanExecutionProvider`. Tolerances are derived and documented per family; widening one requires Trinity's sign-off and an in-test note. Validation-layer-clean is part of "done" for any engine change.
**Why:** CPU fallback is always correct; a plain output comparison passes whether or not the EP ran anything — the vacuous-pass trap.
**Alternatives rejected:** numpy reference (re-derives ONNX semantics, bugs go in both); ONNX reference evaluator as primary oracle (good for conformance fuzzing, slow).

---

### 2026-07-28T17:59:54-07:00: ORT Plugin EP C ABI is experimental — accept and isolate the risk

**By:** Fact Checker
**Updated (2026-07-28T22:28:08-07:00):** ABI floor is 1.24 (not 1.22); runtime target is 1.28; three-number version negotiation scheme adopted — see "ORT version: 1.28 ship, 1.24 floor, negotiated version scheme."
**What:** The ORT plugin EP system is functional but the ABI stability guarantee is weak — API redesigned after 1.22, major additions at 1.23 and 1.24, Qualcomm's first production plugin EP shipped May 2026. Strategy: pin to a specific ORT version for versioned releases; invest early in an FFI abstraction layer so breakages are contained.
**Why:** ORT 1.22/1.23 headers confirm experimental status. Raw FFI in Rust requires regenerating/updating unsafe bindings at each API change.

---

### 2026-07-28T17:59:54-07:00: No existing Vulkan EP or ORT Rust plugin-EP bindings — we write raw FFI

**By:** Fact Checker
**What:** There is no existing Vulkan EP for ORT (feature request open, no release). The `ort` Rust crate covers built-in providers only. We write raw FFI bindings from scratch via `bindgen` over vendored headers.
**Why:** Verified from ORT issue tracker and published crate registry. Opportunity (no prior art) and risk (no ecosystem).

---

<!-- ═══════════════════════════════════════════════════════════════════════════════ -->
<!-- ROUND 2 DECISIONS — 2026-07-28T22:28:08-07:00 (implementation round)          -->
<!-- ═══════════════════════════════════════════════════════════════════════════════ -->

### 2026-07-28T22:28:08-07:00: Compatibility outranks API version — coverage wins when they conflict

**By:** Justin Chu (via Copilot)
**What:** If Vulkan 1.3 compatibility turns out to be poor, the team takes Vulkan 1.2. Broad device compatibility is the top-priority property of every baseline decision — ahead of engine-code simplicity. When coverage and elegance conflict, **coverage wins**. Link's OQ-1 findings decide the final shape.
**Why:** User directive. Resolves the trade-off the round-1 capability-set design left open.

---

### 2026-07-28T22:28:08-07:00: Target HIGH op coverage fast — model-family priority order

**By:** Justin Chu (via Copilot)
**What:** The op coverage ambition is HIGH; the timeline is days-to-weeks for tiers 0–2, months for Qwen3.5 end-to-end. Prioritization by model family: (1) LLMs / decoder transformers — Qwen3.5 explicitly named as target; (2) MoE; (3) Multimodal (vision encoder + LLM); (4) Linear attention / SSM; (5) Conv-based. The v0 op set must be driven by "what a real Qwen3.5/MoE/multimodal ONNX graph contains", not an abstract op list.
**Why:** User directive. Broad correct coverage with clean CPU fallback beats a small perfectly-tuned set.

---

### 2026-07-28T22:28:08-07:00: `com.microsoft` contrib ops admitted: eleven named ops with five constraints

**By:** Justin Chu (user ruling), Morpheus (constraints), Mouse (evidence)
**Supersedes:** "Ruthless v1 non-goals" round 1 entry (exclusion of all contrib ops).
**What:** The `com.microsoft` contrib domain is in scope by user ruling. Admitted v1 set (eleven): `GroupQueryAttention`, `MultiHeadAttention`, `RotaryEmbedding`, `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, `MatMulNBits`, `LinearAttention`, `CausalConvWithState`, `QMoE`, `GatherBlockQuantized`, `MoE`. A twelfth requires a new decision record. **The domain is NOT admitted — named ops only; no `if domain == "com.microsoft"` predicate exists.** Five constraints: (C1) no domain-wide opt-in in code, registry keyed by `(domain, op_type)`, CI regression test; (C2) ORT version pinned per release, `graph_census.py` in CI before tier 3, schema drift alarm; (C3) contrib declines use the same machine-readable decline path as `ai.onnx` declines; (C6) CPU fallback stays the safety net, per-layer numerical verification, never final-logits; (C7) fingerprints audited against the ORT schema by a CI job that re-derives arity/attribute names from the pinned release and fails on disagreement.
**Why:** Mouse verified from GenAI builder source that Qwen3.5 graphs contain these ops directly. An EP that declines the domain cannot run a Qwen graph at all. Justin named Qwen3.5 as an explicit target. The constraints replace the missing opset version-check guarantee that `ai.onnx` provides.
**Alternatives rejected:** Admit the whole domain (unbounded, unversioned surface); hold the non-goal and target primitive-attention graphs only (abandons the named target); admit without constraints (same hazard as domain-wide opt-in).

---

### 2026-07-28T22:28:08-07:00: XL kernels are committed deliverables, not stretch goals

**By:** Justin Chu (via Copilot)
**What:** `MatMulNBits`, `GroupQueryAttention`, `LinearAttention`, `QMoE`, `GatherBlockQuantized`, `RotaryEmbedding`, `CausalConvWithState`, and the block-quantized dequant path are all in scope and must be implemented. They cannot be deferred without deferring the actual goal: an int4-quantized LLM is the real workload. Weight prepacking at subgraph-compile time is on the critical path.
**Why:** User ruling following the contrib-domain admission. Mouse's plan classified these as the XL kernels gating "Qwen3.5 end-to-end." Rai's 🟢 Green ruling on OQ-M6 means llama.cpp's shaders can be read as reference for structural patterns (not copied).

---

### 2026-07-28T22:28:08-07:00: Vulkan baseline frozen: minimal device gate, no required extensions

**By:** Morpheus
**Supersedes:** "Vulkan API baseline: capability-set, not version floor (consolidated)" round 1 entry.
**What:** `docs/DESIGN.md` §7.2 is frozen with six hard requirements and **no required extensions**: (1) Vulkan ≥ 1.1 core; (2) a compute queue; (3) `maxComputeWorkGroupInvocations >= 256`; (4) `maxComputeSharedMemorySize >= 16384`; (5) subgroup `BASIC` in `COMPUTE` stage; (6) at least one `DEVICE_LOCAL` and one `HOST_VISIBLE` memory type. Everything else — `synchronization2`, `subgroup_size_control`, `shaderFloat16`, `shaderInt8`, timeline semaphores, `bufferDeviceAddress`, cooperative matrix — is probed into `vk::caps::Capabilities` and used to select an engine strategy or gate a claim predicate. **None may become a device gate without a new decision record.** Governing principle: a hard device requirement must be justified by "no op we will ever ship can work without it."
**Why:** The previous capability-set required extensions that only some ops need. Under the compatibility-first directive, that is indefensible once measured. Link's OQ-1 data: 31.43% Android gap for sync2, 14.12% for subgroup_size_control. Making the gate minimal also makes the failure mode strictly better: a weak device gets "runs the ops it can, CPU EP runs the rest" instead of "this device does not exist."
**Alternatives rejected:** Keep two-extension requirement (31.43% Android and 12.22% Windows measured exclusions, now indefensible); Vulkan 1.3 hard floor (~36pp Android); Vulkan 1.2 floor (barely exists on Android, bimodal 1.1/1.3 distribution).

---

### 2026-07-28T22:28:08-07:00: `synchronization2` is probed; Switch carries legacy `vkCmdPipelineBarrier` backend

**By:** Morpheus (ruling), Link (measurement), Switch (implementation)
**What:** `synchronization2` is a probed capability, not a device gate. The engine implements **both** `vkCmdPipelineBarrier2` and legacy `vkCmdPipelineBarrier` behind a single internal seam (`rust/src/vk/barrier.rs`), selected once at device init via `Barriers::select(&caps, &device, force_legacy)`. Session option `ep.force_legacy_barriers` forces legacy on sync2-capable hardware (required by CI for parity testing). Trinity runs the full differential suite twice per lane and asserts identical results.
**Why:** Link measured 31.43% Android gap and 12.22% Windows gap. The missing Android population is structurally missing (Adreno 5xx on frozen OEM blobs, Mali Bifrost on MediaTek). It does not shrink on any schedule we control. Cost is bounded and one-time; declined devices can never be won back.
**Alternatives rejected:** Bundle `VK_LAYER_KHRONOS_synchronization2` (retail Android cannot load layers from a plugin; wgpu/Dawn/Godot cited as using this layer was false — all three use legacy barriers exclusively); scope Android to 2021+ (reads the directive backwards).

---

### 2026-07-28T22:28:08-07:00: `VK_LAYER_KHRONOS_synchronization2` shim rejected as a shippable mechanism

**By:** Morpheus
**What:** The Khronos sync2 layer is not bundled or shipped. If an Android integrator independently deploys it in their own APK, our EP will light it up automatically (we probe the extension). That is documented as an optional integrator note in `PLATFORMS.md`, not a mechanism we depend on.
**Why:** The retail Android Vulkan loader does not read `VK_LAYER_PATH`, does not use JSON manifests, and enumerates layers only from the host application's `nativeLibraryDir`. A plugin cannot enable layers on retail Android — the platform that was 100% of the motivation for the proposal. Additionally, the cited precedent (wgpu, Dawn, Godot) was found to be false: all three use legacy `vkCmdPipelineBarrier` exclusively. And `setenv`-ing before `vkCreateInstance` in a `dlopen`ed plugin is a race against the host's own `getenv` in any multi-threaded application.

---

### 2026-07-28T22:28:08-07:00: Barrier abstraction: one seam (`barrier.rs`), two backends, selected once at init

**By:** Switch (implementation), Morpheus (design requirement)
**What:** `rust/src/vk/barrier.rs` is the **only** file in the crate permitted to name `vkCmdPipelineBarrier`, `vkCmdPipelineBarrier2`, `VkBufferMemoryBarrier(2)`, `VkDependencyInfo`, or any `VK_PIPELINE_STAGE*`/`VK_ACCESS*` flag families — enforced by the layering lint. `Access` and `Stage` are our own closed enums with no `None` variant (the abstraction is total by construction). The sync2↔legacy mapping is one table in one place. Batching semantics are identical in both backends.
**Why:** Scattered `if caps.sync2 { … } else { … }` at every call site would make the legacy path (carried for ~31% of Android) chronically untested in CI where sync2 is available ~99% of the time. Selecting once at init converts a recurring cost into a one-time cost.
**Alternatives rejected:** `if` at each call site (bug farm); emit legacy barriers always (would require rewriting ENGINE.md which is already written around sync2's finer-grained access masks, though retained as the fallback plan if dual-path proves expensive).

---

### 2026-07-28T22:28:08-07:00: `subgroup_size_control` is a query, never a feature gate

**By:** Morpheus
**What:** Not a device gate at all. Where consulted: (1) always read Vulkan 1.1 core `VkPhysicalDeviceSubgroupProperties::subgroupSize`; (2) if `VK_EXT_subgroup_size_control` / 1.3 core is present, chain `VkPhysicalDeviceSubgroupSizeControlProperties` and record `minSubgroupSize`/`maxSubgroupSize` as *better information about the range*, nothing more; (3) only if the feature flag is additionally `VK_TRUE` may a pipeline use `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`. (4) A shader whose correctness depends on a specific subgroup width may only be selected when the width is known exactly (`minSubgroupSize == maxSubgroupSize`, or the required-size path was used); otherwise the portable shared-memory variant is selected.
**Why:** Requiring `subgroupSizeControl == VK_TRUE` would exclude all of macOS/iOS (MoltenVK reports `VK_FALSE`; Metal cannot control SIMD-group width per pipeline) and very likely lavapipe and SwiftShader — the CI machines. A requirement that excludes the machines you run CI on is a requirement you have not tested.

---

### 2026-07-28T22:28:08-07:00: Op coverage: model-family-driven, 174-op inventory, 6-tier plan

**By:** Morpheus (ratification), Mouse (authorship)
**Supersedes:** "Op growth by family, prioritized by island-merging" round 1 entry.
**What:** `docs/OP_COVERAGE.md` is the authoritative coverage plan. 174 ops in 16 families, 6 tiers (T0–T6). Exit criteria are "model family X runs end-to-end on the EP", never an op count. **Milestone split: tiers 0–2 (121 ops) are weeks-scale; Qwen3.5 end-to-end is months-scale.** These are never collapsed into one number. `largest_island_flops` on the corpus artifacts (including Qwen artifacts where it will be near zero for a long time) is the metric of record — it cannot be gamed by breadth. `claimed_node_fraction` is diagnostic only, explicitly not a target. Four non-negotiable constraints attach: conservative claiming, clean CPU fallback, minimum viable subgraph size, every claimed op ships with its differential test and platform row on the same PR.
**Why:** Justin's directive ("high op coverage, days-to-weeks, Qwen3.5 target"). The inventory is derived from emitted graphs (GenAI model builder source, contrib schemas, ORT WebGPU EP registries), not the ONNX spec index. UNVERIFIED rows may not be load-bearing for tier exit criteria.
**Alternatives rejected:** Keep conservative §8 as authoritative (wrong sequencing axis — op-family vs. model-family); ratify sight-unseen without constraints.

---

### 2026-07-28T22:28:08-07:00: Template infrastructure must exist before op #1 lands

**By:** Morpheus (A2 amendment), Mouse (thesis)
**What:** M1 does not begin until `indexing.glsl`, the `build.rs` variant generation, the `ops!` macro, and the shared claim helpers are merged. M1 exit criteria include an ops-per-hand-written-kernel ratio (≥ 8) as a **reported number**. A ratio that collapses is the earliest possible signal the thesis is failing.
**Why:** Mouse established that 87 tier-1 ops are served by ~5 kernel templates. If ops #1–#20 are hand-written first, the template architecture is lost and the schedule collapses. MLX reached its coverage in days because MLX already owned the kernels. Our leverage has to be manufactured first. Order matters more than effort.

---

### 2026-07-28T22:28:08-07:00: Registry is table-driven via `OpSpec` / `ops!` macro with machine-readable `caps` column

**By:** Mouse (proposal), Morpheus (ratification)
**What:** `OpSpec { domain, op_type, min_opset, max_opset, caps, kernel, claim, translate, status }` replaces the `(&str, ClaimPredicate)` tuple. Both function pointers take `&OpSpec`. A `caps: DtypeSet` column generates: the runtime dtype claim check, the `build.rs` shader variant list, `docs/OP_SUPPORT.md`, and `--dump-capabilities`. `cargo xtask op-matrix` regenerates; CI fails if the checked-in matrix differs. `domain` is a typed enum. `OpStatus::{Live, Staged(reason)}` — staged rows decline with a machine-readable `[staged]` reason, allowing the table to land and be tested before shaders exist.
**Why:** At 174 ops the struct-literal form is ~1200 lines of unreviewable boilerplate. Without the `caps` column, the op support matrix must be hand-maintained prose across five vendors and two dtypes — the drift this project exists to prevent.

---

### 2026-07-28T22:28:08-07:00: Minimum Viable Subgraph (MVS) rule with measured transfer cost

**By:** Mouse (design), Morpheus (A3 amendment)
**What:** Claim a candidate subgraph only if `est_gpu_time > transfer_cost(boundary_tensors) × SAFETY(3.0)`, plus unconditional floor `node_count ≥ 4 AND total_output_bytes ≥ 64 KiB` (waived for GEMM/attention/quantized-GEMM). Anti-orphan pass: drop any 1–3 node non-GEMM-anchored island after partitioning. `transfer_cost` is **calibrated at device init** (~2 ms: time staged upload+download, fit a line) — not hardcoded. On UMA parts the slope is near zero; on discrete PCIe it correctly tightens. `SAFETY = 3.0`, `node_count >= 4`, and the 64 KiB floor are **provisional and must be re-derived from Niobe's M2 measurements**. Declined subgraphs are returned (not discarded) for `CLAIM_DEBUG` visibility — a silently-declined region is indistinguishable at the console from a missing op.
**Why:** A coverage-count mindset shreds graphs into transfer-dominated fragments that run slower than CPU. This is the rule that makes coverage a performance metric rather than a vanity metric.

---

### 2026-07-28T22:28:08-07:00: Quantized weights prepacked at `Compile` time; never dequantized into VRAM

**By:** Mouse (design), Switch (engine seam)
**What:** `MatMulNBits` `B` weight is repacked once on the host during `Compile` into a tile-friendly interleaved layout and uploaded; `scales`/`zero_points` are separate bindings. Two sub-variants: **GEMV** (decode, M=1, memory-bound — dequantize into registers) and **GEMM** (prefill, M>1 — dequantize a B tile into shared memory once, reuse across the M tile). A dequantized weight tensor must **never** appear in device memory. Prepacked buffer is keyed on `(initializer, TileConfig, variant)` — two ops sharing the same weight produce one upload; two tile configs produce separate uploads.
**Why:** Materializing a dequantized weight defeats the entire purpose of int4. The GEMV path is how llama.cpp achieves full bandwidth on decode. Weight high-water invariant: only packed nibbles + scales on device.

---

### 2026-07-28T22:28:08-07:00: OQ-3 resolved: reserved virtual-address opaque-handle registry; no BDA

**By:** Morpheus (ruling), Tank (proposal)
**Supersedes:** "VkBuffer handle identity via opaque-handle registry" round 1 entry.
**What:** `Alloc(size)` sub-allocates Vulkan memory via `gpu-allocator`, carves a matching span from a large region of **reserved-but-uncommitted virtual address space** (`VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` on Windows; `mmap(PROT_NONE)` on POSIX), records `span_base → (VkBuffer, offset, size, generation)`, and returns `span_base` as the `void*`. Resolution is `idx = (ptr - region_base) >> 21; entry = table[idx]; offset = ptr - entry.span_base`. `Free` quarantines the span with a generation bump. **`VK_KHR_buffer_device_address` is not carried at all.**
**Why:** BDA is not an optimization of the registry — it is a different shader architecture (`GL_EXT_buffer_reference` / `PhysicalStorageBuffer` addressing) requiring a second shader family, second variant axis, and second conformance run set. It does not even remove the side table: building a descriptor set still needs a `VkBuffer`. Reserved VA answers ORT's internal pointer arithmetic (`base + offset`, `align_up`) by construction, not convention. A stray dereference is an MMU fault at a recognizable address, not silent corruption.
**Alternatives rejected:** Synthetic dense token (breaks under ORT's pointer arithmetic); BDA primary with registry fallback (Tank's decisive argument above); host-visible staging memory as the allocator return value (not zero-copy, silently preempts a memory-type decision).

---

### 2026-07-28T22:28:08-07:00: ORT bindings: `bindgen` over vendored version-pinned headers

**By:** Tank
**What:** `third_party/onnxruntime/include/` holds `onnxruntime_c_api.h`, `onnxruntime_ep_c_api.h`, and `onnxruntime_error_code.h` copied verbatim from `microsoft/onnxruntime` tag `v1.28.0`. `build.rs` runs `bindgen` over them into `$OUT_DIR/ort.rs`, which `src/sys.rs` includes. `$ORT_INCLUDE_DIR` / `$ORT_HOME/include` override for local ORT checkout testing. All raw ORT types stay behind `sys.rs`; the layering lint enforces this.
**Why:** `OrtEp`, `OrtEpFactory`, and `OrtApi` are `#[repr(C)]` vtables; a mistake in field order does not fail to compile or load — it calls the wrong function pointer with the wrong arguments (silent UB). `bindgen` derives layout from the same bytes ORT was compiled from. Vendoring buys byte-reproducible builds, no network access, and reviewable ORT version bumps.
**Alternatives rejected:** Hand-written `#[repr(C)]` structs (silent UB failure mode); bindgen against system ORT (non-reproducible); existing `ort`/`ort-sys` crates (they bind the inference API, not the plugin-EP author API).

---

### 2026-07-28T22:28:08-07:00: ORT version: compile 1.28, minimum runtime 1.24, three-number negotiation

**By:** Tank
**What:** Three distinct numbers: `ORT_API_VERSION_EXPECTED = 28` (vendored headers), `ORT_API_VERSION_MIN = 24` (oldest host we run against), `NegotiatedApi::version = 24–28` (stamped back into all four EP vtable `ort_version_supported` fields so a downlevel host stops reading our vtables exactly where its own header stops describing them). `check_api_version()` walks down from `GetApi(28)` to `GetApi(24)`; below 24 it refuses to load. Optional entry points gated by `NegotiatedApi::supports(since::*)`. ORT 1.27 explicitly excluded (null-allocator `PrePack` bug, deleter lifetime bug).
**Why:** `OrtEp` and `OrtEpFactory` are append-only, so version *v*'s layout is a prefix of 28's — but safe only if we never touch fields added after *v*. `ort_version_supported` stamping plus `supports()` gating are the mechanism that makes negotiation safe rather than a silent-UB trap.
**Alternatives rejected:** Hard-pin 28 only (unnecessarily narrow, 1.24 floor is already set); accept any version (below 24 the layout is unvalidated).

---

### 2026-07-28T22:28:08-07:00: OQ-4 resolved: hard Vulkan SDK build dependency; shader-less artifact presents as EP absent

**By:** Morpheus (ruling), Switch (implementation)
**What:** There is no checked-in SPIR-V fallback. `glslc` must be present on shader-writing machines (escape hatch: `ALLOW_MISSING_GLSLC=1` for SDK-less machines). When `SHADER_MODULES.is_empty()`: (1) `probe_devices()` returns `vec![]` and logs `WARN "built without shaders"`; (2) `get_capability_impl()` returns `ptr::null_mut()`. Both guards fire independently. A shader-less artifact advertises zero devices and claims nothing — identical posture to "EP is absent."
**Why:** A checked-in SPIR-V fallback (168 modules, ~1–3 MiB) has an invisible staleness hazard: a clone that misses the latest blob builds silently but runs old shaders. Dawn, llama.cpp, and ExecuTorch all require the SDK on shader-writing machines. The safety property of the escape hatch is the same as Trinity's no-ICD test one level up: no devices, no claims, graph runs on CPU.
**Alternatives rejected:** Checked-in SPIR-V (staleness hazard, binary bloat); OQ-4 provisional "checked-in fallback" plan (overturned by Switch and Morpheus on this session).

---

### 2026-07-28T22:28:08-07:00: OQ-13: zero-copy IO binding via `CreateExternalResourceImporterForDevice` (post-M2)

**By:** Morpheus, Tank
**What:** `OrtEpFactory::CreateExternalResourceImporterForDevice` (vtable member since ORT 1.24) is the complete answer for zero-copy IO binding from external callers. Scope: post-M2, since it presupposes the device-memory tensor path. Design contract to document explicitly: the caller's `VkDeviceMemory` must have been allocated with `VkExportMemoryAllocateInfo` up front — it cannot be retrofitted. This is an integration contract for downstream applications, not a transparent optimization.
**Why:** Real, supported upstream since 1.24 with an in-tree reference (`onnxruntime/test/providers/nv_tensorrt_rtx/nv_vulkan_test.cc`). Orthogonal to OQ-3 — it is caller-driven for caller's memory; OQ-3 is EP-driven for our memory.

---

### 2026-07-28T22:28:08-07:00: ORT 1.28 external resource importer API name corrected

**By:** Fact Checker
**What:** The correct symbol is `CreateExternalResourceImporterForDevice` (not `…Impl`). The `Impl` suffix appears only as a local static in ORT example/test code. It landed in **ORT 1.24**, not 1.28. The earlier entry in Morpheus's OQ-11 record naming it as landing in 1.28 and proposing it as an OQ-3 candidate are both withdrawn.
**Why:** ORT's `onnxruntime/core/session/interop_api.cc` comment: "added in ORT 1.24"; `ep_factory_provider_bridge.h` guard: `if (ep_factory_.ort_version_supported < 24)`. Tank had already set `ORT_API_VERSION_MIN = 24`, so no ABI floor change was ever required.

---

### 2026-07-28T22:28:08-07:00: Claim assertion mechanism: ORT profiling JSON

**By:** Trinity
**What:** Use ORT's built-in session profiling JSON to assert device placement. The helper `assert_vulkan_claims` enables profiling, runs the session, reads the profile file, checks `"VulkanExecutionProvider" in providers` from `Node`-category events, and removes the file in a `finally` block. This is the authoritative mechanism; ORT 1.27+ supports it and the reference project (`onnxruntime-mlx`) uses it.
**Why:** Structured (JSON not text), official ORT API, cross-platform, no changes to Tank's crate required. Alternatives rejected: parse `CLAIM_DEBUG` log text (not a stable API); EP-side Python counter (requires new ABI); infer from output values (cannot distinguish GPU same result from CPU fallback — the vacuous-pass trap).

---

### 2026-07-28T22:28:08-07:00: Tolerance policy per op-family; widening requires Trinity sign-off

**By:** Trinity
**What:** fp32 elementwise: rtol=atol=1e-5; fp32 transcendental/activation: same; fp32 comparison/logic: exact (0/0); fp16 (M1+): rtol=atol=1e-3; reductions/GEMM/MatMul (M2+): TBD from measurement per vendor (OQ-10). Named constants in `tests/ops/_models.py`; widening a tolerance requires Trinity's sign-off AND an in-test comment explaining which driver exhibits wider error.
**Why:** Tolerances as magic numbers scattered across tests are invisible in code review. Named constants make widening a visible diff. The protocol exists because a tolerance wide enough to hide a real error class is worse than no test.

---

### 2026-07-28T22:28:08-07:00: CI lanes: lavapipe Linux primary; lavapipe Windows via mesa-dist-win; force-legacy parity lane

**By:** Trinity
**What:** Three jobs: (1) `format` (ubuntu-latest, no Vulkan); (2) `build-test-linux` (ubuntu-22.04, lavapipe from `mesa-vulkan-drivers`, Vulkan 1.3, validation layers on, `ep.force_legacy_barriers` second pass); (3) `build-test-windows` (windows-latest, lavapipe via `mesa-dist-win 26.1.3`, LunarG SDK for glslc/validation, Vulkan tests now active). SwiftShader on Windows is explicitly rejected (Google publishes no prebuilt DLLs; build from source ~20 min). The force-legacy parity lane runs the full differential suite twice per lane (sync2-default and `force_legacy_barriers=true`) and asserts identical results — without this, the legacy path carried for ~31% of Android would never be tested.
**Why:** lavapipe (Mesa) is in Ubuntu 22.04 apt repos; mesa-dist-win provides it on Windows. Both pass Vulkan 1.3 conformance. The no-ICD fallback test runs with `VK_ICD_FILENAMES` overridden to `/nonexistent/path`.

---

### 2026-07-28T22:28:08-07:00: License ruling — llama.cpp reference: 🟢 Green

**By:** Rai
**What:** Reading llama.cpp's MIT-licensed Vulkan shaders (`ggml/src/ggml-vulkan/vulkan-shaders/`) as reference for GQA, MatMulNBits, and LinearAttention is fully permitted. Conditions activate **only when writing code that substantially adapts** (not merely learns from) the original: (1) file header in the adapted shader; (2) entry in `docs/THIRD_PARTY_NOTICES.md`; (3) commit message noting the source; (4) `THIRD_PARTY_NOTICES.md` distributed with any binary containing adapted SPIR-V.
**Why:** The idea/expression dichotomy: algorithms, tiling strategies, and quantization approaches are not protected expression. No attribution obligation for reading and studying.

---

### 2026-07-28T22:28:08-07:00: llama.cpp structural ideas transfer; direct code copying does not apply

**By:** Morpheus (ruling), Switch (analysis)
**Supersedes:** Mouse's narrower claim that "adaptation is useless" due to format mismatch.
**What:** Block-format mismatch is real (llama.cpp uses K-quants / superblocks; ONNX `MatMulNBits` uses row-major int4 with separate scales). Direct instruction-level code copying would not work. **Structural ideas DO transfer:** (1) tiling strategy — outer loop over output tiles, inner loop over K-blocks, tile sizes as specialization constants, GEMV vs. GEMM path selection based on M=1; (2) subgroup reduction shape — per-subgroup partial dot-product accumulation followed by `subgroupAdd`, scalar float accumulation as portable fallback; (3) dequant-in-register — load packed nibbles, unpack, multiply by scales, accumulate; no intermediate dequantized buffer. Morpheus's timeline estimates assume this accelerant is used for tiers 3 and 4. Pre-committed test: if Switch's independent review concludes the tiling/subgroup-reduction shape does not transfer, Morpheus widens those tier estimates and says so plainly.
**Why:** Rai's 🟢 ruling permits algorithm study with no obligation. Refusing to study it would abandon the single largest available schedule accelerant.

---

### 2026-07-28T22:28:08-07:00: Oracle validated for quantized path; oracle knobs pinned explicitly

**By:** Morpheus (ratification), Trinity (empirical result)
**What:** The ORT CPU EP works as a differential oracle for a GenAI-built int4 Qwen3-0.6B graph — confirmed by Trinity's execution. New rule generalized from Trinity's `accuracy_level` finding: **any oracle knob the runtime selects from the host machine must be pinned explicitly.** `MatMulNBits` at `accuracy_level = 4` (int8/VNNI) diverges from levels 0–3 by ~3.6e-3 at K=1024, N=512; ORT picks a level from the host CPU. Oracle pinned at level 1. Bit-layout interpretation (`MatMulNBits` unpack) is checked against an independently-written specification (numpy), not against the CPU EP — the shared-misreading hazard applies here too.
**Why:** An oracle that changes with the machine is not an oracle. Unpinned, reference values would drift silently across CI runner hardware and present as bugs in our kernel. Trinity's fp16 NaN/Inf on ORT 1.27 independently confirms the 1.27 PrePack bug.

---

### 2026-07-28T22:28:08-07:00: Execution status disclosure — no shader has executed on any device yet ⚠️ SUPERSEDED by 2026-07-30 D26 below

**By:** Morpheus
**What:** `DESIGN.md` §9.1.2 and `README.md` status block state plainly: no shader in this repository has executed on any device; the only device execution will be on lavapipe; every green count to date measures host-side logic; the first real execution evidence is M0's exit criteria. A test count is a claim about what was executed and must not imply more execution than occurred.
**Why:** The project now reports large test counts (227 at `cbb1a0d`; 206 collected / 8 passing with no EP built). A reader will reasonably interpret those as evidence about GPU numerics. They are not.

---

### 2026-07-28T22:28:08-07:00: C1 enforced from both ends — static ban and runtime assertion

**By:** Morpheus (ratification), Tank (static ban), Trinity (runtime assertion)
**What:** C1 (no domain-wide opt-in in code) is enforced by: (1) Tank's static ban — `layering.rs` prevents `"com.microsoft"` as a *value* in `src/ops/**`; (2) Trinity's runtime assertion — `com.microsoft::NotARealOp` takes the ordinary decline path and `ep.rs` checks `code == "not-registered"` (not merely that the EP didn't run). A constraint checked only statically can be satisfied by code that never runs; a constraint checked only at runtime can be reintroduced in a path no test reaches.
**Why:** The vacuous-pass trap applies to negative-property tests too. "Assert the reason, not the absence" — zero claimed nodes is equally consistent with "declined correctly", "declined for the wrong reason", and "crashed before claiming."

---

### 2026-07-28T22:28:08-07:00: `GetCapability` retain_viable placement: stage 3 of 4

**By:** Morpheus
**What:** `GetCapability` runs: (1) per-node `registry::claim_decision`; (2) maximal convex connected clustering → `Island`s; (3) **`retain_viable`** (MVS rule applied to islands); (4) `AddNodesToFuse` on survivors. Stage-3 rejections carry `DeclineCode::Partition` with modelled numbers into `CLAIM_DEBUG` output. `retain_viable` is pure; `ep.rs` builds inputs and translates verdicts. `largest_island_flops` and the dropped set come from the same `(kept, dropped)` pair — the rule and its metric live in one module deliberately.
**Why:** A per-node approximation of a whole-island rule would re-introduce the graph-shredding the rule exists to prevent. The decline vocabulary requirement ensures silently-declined regions are distinguishable from missing ops at the console.

---

### 2026-07-28T22:28:08-07:00: Milestone reconciliation: M0/M1 unaffected by contrib; M2 gets census; M3+ carries XL cost

**By:** Morpheus
**What:** M0 and M1 are unaffected by the contrib ruling — no contrib op appears in either. M2 gains one obligation: `tools/graph_census.py` in CI must exist before the first contrib op; it is built in M1/M2 and has an M2 exit criterion. M3+ carries the entire cost: three XL kernels with no template leverage (GQA, MatMulNBits, LinearAttention) that are not parallelizable away. Tier 0–2 is weeks-scale; Qwen3.5 end-to-end is months-scale. These do not collapse. Every milestone from M1 onward reports `largest_island_flops` on corpus artifacts.
**Why:** The ambition was raised (contrib admitted, XL kernels committed) and an accelerant authorized (llama.cpp study). Neither shortens the months-scale number. Before the ruling, "Qwen3.5 end-to-end" had no completion path at all; now it has a long one. That is a different kind of progress from "it got faster."

---

### 2026-07-28T22:28:08-07:00: OQ-12 decisive experiment specified in advance, before hardware exists

**By:** Morpheus (ruling), Link (specification)
**What:** Validates whether the 31.43% Android sync2-missing population is actually usable (the gpuinfo.org data proves only absence of an extension, not usability). Four device slots: (A) Adreno 5xx on stock pre-2021 OEM blob; (B) Mali Bifrost on MediaTek; (C) Adreno 6xx on Android 12+; (D) Mali Valhall on Android 12+. Three stages: (1) §7.2 gate check with full capability dump; (2) full M1/M2 differential suite on-device, run twice on C/D with `ep.force_legacy_barriers=1`; (3) GEMM-anchored subgraph vs. that device's own ORT CPU EP, threshold ≥ 1.5×. Pre-committed outcomes: A+B fail 1–2 → Android half of §7.3 void, legacy backend stays for Windows only; A+B pass 1–2 fail 3 → supported but "runs, not recommended", no tuning budget; A+B pass all three → §7.3 vindicated, Android gets M3 tuning budget; C/D fail only under forced-legacy → `LegacyBackend` bug, most valuable possible result.
**Why:** An experiment designed after the data arrives gets designed to confirm the decision already made. Writing reversal conditions down in advance is the only protection against that, and §7.3 is Morpheus's decision.

---

### 2026-07-28T22:28:08-07:00: contrib schema census alarm (C2); schema fingerprints audited by CI

**By:** Morpheus
**What:** (C2) ORT version pinned per release in `Cargo.toml`, docs, and CI matrix. Each contrib row in the registry table records the ORT version its predicate was written against. `tools/graph_census.py` runs in CI against pinned `.onnx` artifacts — an ORT version bump that changes any contrib claim rate is a review gate. C2 item 6: a contrib row whose baseline is not a released ORT version (`main`-only schema) may not be flipped from `Staged` to `Live` — enforced as a build-failing test. C2 item 7: a CI job re-derives arity and attribute names for all contrib rows from the pinned ORT release and fails on disagreement with the recorded fingerprint; runs on ORT version bump **and** on a schedule. Mouse's self-audit of `GroupQueryAttention` found two permissive errors (`min_inputs 3` vs. true minimum 7). Both errors were permissive — a too-permissive baseline fails silently; only a too-strict one is self-announcing.
**Why:** `ai.onnx` ops carry an opset number we can range-check. Contrib ops do not — their schemas version with ORT releases. Drift surfaces as a CI delta in the week it happens rather than as wrong logits in a bug report months later.

---

### 2026-07-28T22:28:08-07:00: OQ-16 raised: LinearAttention/CausalConvWithState schema stabilization is a schedule risk

**By:** Morpheus
**What:** `LinearAttention` and `CausalConvWithState` exist only on ORT main (post-1.28.0) and gate T5a — the named Qwen3.5 target. Tracked with an owner: Fact Checker watches upstream, Mouse re-verifies, Morpheus rules on T5a scope. If it slips, the correct report is "gated on an upstream schema", not "Qwen3.5 slipped for Vulkan reasons."
**Why:** This is a different kind of risk from "the Vulkan kernel is hard": upstream-dependent, may mean writing T5a kernels twice, and its mitigations are different from engineering effort. It needs to be visible at the architecture level, not buried in a `notes:` field.

---

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction


<!-- ═══════════════════════════════════════════════════════════════════════════════ -->
<!-- ROUND 3 DECISIONS — 2026-07-29T09:00:39-07:00 (first-hardware round)          -->
<!-- ═══════════════════════════════════════════════════════════════════════════════ -->

### 2026-07-29T09:00:39-07:00: Use the local GPU — diagnose on hardware, use CI to prove portability

**By:** Justin Chu (via Copilot)
**What:** This development machine has real GPUs and must be used for iteration. Two Vulkan devices present: `Intel(R) Iris(R) Xe Graphics` (Vulkan 1.4.309, UMA 32 KiB shared memory) and `NVIDIA GeForce RTX 4060 Laptop GPU` (Vulkan 1.4.325, discrete 48 KiB shared memory). Both pass the §7.2 capability gate on every criterion. Vulkan SDK 1.4.350.0 is installed at `C:\VulkanSDK\1.4.350.0` (not on default PATH). With SDK on PATH, all 168 shader variants compile; `ALLOW_MISSING_GLSLC` is no longer needed locally. Local build recipe: `$env:VULKAN_SDK="C:\VulkanSDK\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\Bin;$env:PATH"; cargo build --release`. Intel Iris Xe is the stricter implementation and should be treated as the spec-conformance oracle.
**Why:** User directive. Three consecutive CI failures (missing package, bad YAML, PowerShell array bug) all masqueraded as "Vulkan is broken" — a single local command would have distinguished the hypotheses hours earlier. Principle: **diagnose on local hardware; use CI to prove portability, not to answer questions.**
**Corollary:** `rustfmt --edition 2021` silently no-ops on this edition-2024 crate. Always use `cargo fmt --all` (which reads the `edition` from `Cargo.toml`).

---

### 2026-07-29T09:00:39-07:00: Prefer the project's own ONNX crates — reference first, adopt only what earns its place

**By:** Justin Chu (via Copilot), Mouse (evaluation)
**What:** Justin's own crates (`onnx-runtime-tracer`, `onnx-ir-rust`, `onnx-shape-inference`, `onnx-genai`, `onnx-genai-models`) are to be referenced and, where they earn their place, adopted. Adoption is not automatic — we ship a cdylib loaded into someone else's process, so every dependency is binary weight and a lifetime we do not control. Outcomes of evaluation: `onnx-runtime-tracer` adopted at `0.1.0-dev.5, default-features = false` (Niobe). `onnx-shape-inference` adopted as a **Python oracle** (Trinity harness preprocessing step, converts `[dynamic-shape]` declines into claims with no Rust changes). `onnx-ir-rust` and `onnx-runtime-ir` deferred with named triggers. `onnx-genai-models` (`mobius` builder) produced the decisive finding of this round: see "Op coverage is relative to a producer."

---

### 2026-07-29T09:00:39-07:00: R5 (subgroup BASIC in compute) removed from the device gate — now a probed capability ⚠️ RATIONALE CORRECTED by 2026-07-30 D36 (lavapipe reading was instrument bug, policy stands on §7.0)

**By:** Switch
**Supersedes:** R5 criterion in "Vulkan baseline frozen: minimal device gate, no required extensions" (2026-07-28T22:28:08-07:00). Only the gate status changes; the capability is still queried.
**What:** `passes_gate` now checks R1–R4, R6 only. `Capabilities::subgroup_basic_in_compute` is probed (from `VkPhysicalDeviceSubgroupProperties`) and used by ops that require subgroup intrinsics in their claim predicates. `assess_gate` (new function) evaluates all criteria without early exit and drives `epctl --probe-loader` verbose output.
**Why:** Mesa llvmpipe on Ubuntu 22.04 reports `supportedStages = 0` — lavapipe cannot pass R5, but lavapipe is the only device available on both CI lanes. Direct application of §7.0's governing principle: capability shortfalls degrade op coverage, not device availability. Requiring a compute-stage subgroup feature that software renderers and many integrated GPUs do not satisfy was a gate that should have been a probe from the start.
**Alternatives rejected:** Keep R5 and exclude lavapipe from CI (no Vulkan CI whatsoever, blocking all shader execution until physical hardware is obtained).

---

### 2026-07-29T09:00:39-07:00: `VK_ICD_FILENAMES` rejected as Windows CI mechanism — use registry registration

**By:** Link (root cause), Trinity (fix)
**Supersedes:** Mesa lavapipe ICD mechanism in "CI lanes: lavapipe Linux primary; lavapipe Windows via mesa-dist-win; force-legacy parity lane" (2026-07-28T22:28:08-07:00). The env-var approach was the wrong mechanism.
**What:** After extracting Mesa on the Windows CI lane, register the ICD in the Windows Vulkan driver registry key rather than using `VK_ICD_FILENAMES`: `New-ItemProperty -Path "HKLM:\SOFTWARE\Khronos\Vulkan\Drivers" -Name <icd_path> -Value 0 -PropertyType DWord -Force`. Add `VK_LOADER_DEBUG=warn` to test steps.
**Why:** The LunarG Vulkan loader 1.3+ silently ignores `VK_ICD_FILENAMES`, `VK_DRIVER_FILES`, and `VK_ADD_DRIVER_FILES` when the calling process has elevated privileges. GitHub Actions Windows runners run as `runneradmin` (Administrators group, UAC disabled). Verified from primary source: KhronosGroup/Vulkan-Loader LoaderDriverInterface.md v1.3.274. The env-var is ignored; the registry is not.
**Alternatives rejected:** `VK_DRIVER_FILES` (same elevated-privilege restriction); switching to mmozeiko/build-mesa (unrelated root cause).

---

### 2026-07-29T09:00:39-07:00: `glslc` must come from the LunarG apt repository on Ubuntu 22.04

**By:** Trinity
**What:** `glslc` is not in Ubuntu 22.04's own repos. The LunarG Vulkan SDK apt repository for Jammy provides the `shaderc` package (`/usr/bin/glslc`). `VULKAN_SDK_VERSION: "1.3.296.0"` pinned at workflow-level env (covers both Linux repo URL and Windows installer). A "Verify GLSL compiler (glslc)" precondition step runs `glslc --version` before the build step and fails with `::error::` if not found.
**Why:** `glslang-tools` (Ubuntu default) provides `glslangValidator`, not `glslc`. CI was red for ≥4 consecutive runs because a step named "Install glslc" installed the wrong tool. A claim about an external system is not usable until something has executed and confirmed it.
**Alternatives rejected:** LunarG Linux SDK tarball (~600 MB for one binary); Google shaderc GitHub release (inconsistently published); build from source (~10 min).

---

### 2026-07-29T09:00:39-07:00: `null` file_path to `Logger_LogMessage` causes access violation — always pass a real string

**By:** Tank
**What:** `Logger_LogMessage` is annotated `_In_z_` on `file_path` — ORT dereferences it unconditionally via `ToUTF8String`/`CodeLocation`. Passing `std::ptr::null()` causes a Windows access violation at the first log record emitted after `CreateEp`. Fixed: always pass a real NUL-terminated `ORTCHAR_T` string (`file!()` macro when available, the literal `<onnxruntime-ep-vulkan>` otherwise). Two `cfg` branches for the `wchar_t`/`char` width difference between Windows and Unix.
**Why:** The bug manifested as a crash at `conftest.py:60` (`register_execution_provider_library`) — appearing to be a device-probe crash because the *warning about* the probe was what triggered the first log record. The crash was ours, not ORT's. Rule: for FFI, testing your own code is the easy half. Every `// SAFETY:` comment was about invariants we owe when touching ORT's memory; this bug was in the category of invariants we owe ORT about the arguments we pass.

---

### 2026-07-29T09:00:39-07:00: `attach_default_ort_logger` / `restore_default_ort_logger` — OrtLogger lifetime contract

**By:** Tank
**What:** `CreateEp` must not attach the session's `OrtLogger` to the global bridge permanently. `ReleaseEp` must call `restore_default_ort_logger()` before dropping the EP. If there is no default, detach entirely. A static holding a dangling `OrtLogger*` from a destroyed session is UB at the next log record.
**Why:** The lifetime bug (D-T15) was found by audit, not by test, because CI's suite never reached a second session. `tests/host_registration.rs` now exercises the unwind: emits a record after `ReleaseEp` and before `ReleaseEpFactory`.

---

### 2026-07-29T09:00:39-07:00: `cargo ci` — one command that mirrors CI's Rust lanes exactly

**By:** Tank
**What:** `cargo xtask ci` (aliased as `cargo ci`) runs in order: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -D warnings`, `cargo build`, `cargo test`. Runs all checks even after one fails. `cargo ci --release` passes `--release` to the build and test steps. `cargo ci --fix` rewrites via `rustfmt` rather than checking. On success, prints a caveats block explicitly stating what it does NOT check (no shaders compiled without SDK, no Python lane, no Vulkan dispatch, no `cfg(unix)` from Windows).
**Why:** CI was red for four consecutive runs without any agent noticing because each ran `cargo build; cargo clippy; cargo test` and saw green — `cargo fmt --check` was not in the loop. **The verification loop must live in one place and one place only**: `CHECKS` in `xtask/src/main.rs`. If CI gains a Rust check, it is added there in the same commit.
**Design constraints:** Zero dependencies (xtask). Works with no SDK (sets `ALLOW_MISSING_GLSLC=1` automatically). `--workspace` scope for clippy. Separate package from `epctl` so it runs when the crate fails to build.

---

### 2026-07-29T09:00:39-07:00: `tests/cdylib_load.rs` — mock host that dlopens the shipped cdylib

**By:** Tank
**What:** `tests/host_registration.rs` tests the rlib path (fast, catches ABI violations). `tests/cdylib_load.rs` dlopens the compiled `onnxruntime_vulkan_ep.dll`/`.so`/`.dylib`, resolves `CreateEpFactories`/`ReleaseEpFactory` by name, and runs the identical scenario with the same `_In_z_` contract checks. `libloading` added as a `[dev-dependencies]` entry (test-only). The mock ORT callbacks check SAL annotations: `_In_z_` arguments must be non-null and NUL-terminated; every `OrtStatus` released exactly once.
**Why:** The access-violation crash lived between "the crate compiles" and "ORT's `GetProcAddress` call succeeds" — an interval previously untested. Re-planting the null file_path makes the test fail with a diagnostic rather than process death, before CI runs.

---

### 2026-07-29T09:00:39-07:00: `apiVersion` capped to loader's reported version — latent loader-1.0 bug fixed

**By:** Switch
**What:** `Instance::create` calls `vkEnumerateInstanceVersion` before building `VkApplicationInfo`. If the loader version is Vulkan 1.0 (function not present) or < 1.1, returns `None` with a clear diagnostic. The requested `apiVersion` is capped to the loader version.
**Why:** Vulkan spec requires `apiVersion` ≤ the instance version reported by `vkEnumerateInstanceVersion`. Requesting 1.1 against a 1.0 loader produces `ERROR_INCOMPATIBLE_DRIVER`. This was a latent bug for any user with a 1.0 loader + 1.1 ICD. `vkEnumerateInstanceVersion` is a loader-level function that returns correctly even when no ICD is installed.

---

### 2026-07-29T09:00:39-07:00: Full loader diagnostic on `vkCreateInstance` failure; `epctl --probe-loader` as CI pre-check

**By:** Switch
**What:** On any `vkCreateInstance` error, `Instance::create` emits at WARN: `VK_ICD_FILENAMES`/`VK_DRIVER_FILES`/`VK_INSTANCE_LAYERS` values, loader version, available layer names/count, instance extension count. `ERROR_INCOMPATIBLE_DRIVER` gets an additional hint. `epctl --probe-loader` runs a standalone probe and exits 1 when no device passes the §7.2 gate. Used as a CI pre-check step before pytest.
**Why:** The pre-change log was a single unactionable line. The post-change log shows which ICD paths the loader was given, whether the ICD's library could be found, and which layers/extensions were visible — the difference between "maybe reinstall the driver" and a precise diagnosis.

---

### 2026-07-29T09:00:39-07:00: Linux mock_ort `wchar_t` bug fixed — `OrtChar` is platform-width

**By:** Tank (fix), Link (root cause)
**What:** `tests/mock_ort/mod.rs` used `ort::wchar_t` unconditionally. On Linux, `OrtChar = char` and bindgen emits `c_char`; `wchar_t` only exists on Windows. Fixed with `#[cfg(target_os = "windows")]` / `#[cfg(not(target_os = "windows"))]` branches.
**Why:** Linux compile error was blocking the CI lane entirely, masking any Vulkan outcome. The error was in test infrastructure, not in the EP code.

---

### 2026-07-29T09:00:39-07:00: Op coverage is relative to a producer, not to a model architecture

**By:** Morpheus (D21), Mouse (D-M6-04 — the finding)
**What:** A coverage number is meaningless without naming the producer it was measured against. "We support Qwen3" is not a well-formed claim; "we support Qwen3 as emitted by producer P at version V" is. Four obligations follow: (1) the census corpus is indexed by producer and reports per producer; (2) a target model is "covered" only for a named producer; (3) standard-domain and contrib forms of the same computation get separate claim predicates even when sharing a kernel; (4) the standard domain is preferred where a producer offers one. `largest_island_flops` reported per producer from T3 onward; a green column cannot mask a near-zero one.
**Why:** The inventory was derived from the ORT GenAI builder and then reasoned about as "what a Qwen3 graph looks like." The `onnx-genai-models` (`mobius`) builder builds the same target models but emits `ai.onnx::Attention` @ opset 23, `ai.onnx::RMSNormalization`, `ai.onnx::RotaryEmbedding` with no fused skip-norm — not the contrib equivalents. A Qwen3 built by Justin's own toolchain would have had ~5 nodes per decoder layer × 28 layers declined `[not-registered]` for want of a table row, not a kernel.
**Alternatives rejected:** treating as a one-time registry fix (it is a class of error, not an incident); averaging coverage across producers (hides the gap).

---

### 2026-07-29T09:00:39-07:00: Standard-domain rows registered: `ai.onnx::Attention`, `RMSNormalization`, `RotaryEmbedding`

**By:** Mouse (D-M6-04)
**What:** `ai.onnx::Attention` (opset 23+), `ai.onnx::RMSNormalization` (opset 23+), `ai.onnx::RotaryEmbedding` (opset 23+) registered in the op table with `OPSET_STD_LLM = 23`. `RMSNormalization` reuses the `simplified_layer_norm` kernel (function-pointer identity asserted by test). `ai.onnx::Attention` gets its own predicate despite the shared kernel — attribute names differ (`q_num_heads` vs `num_heads`), illegal-combination set differs, optional inputs at different indices. Standard-domain rows carry no `ContribSchema` fingerprint (enforced by test — `ai.onnx` versions by opset which the row window already expresses). `op_table!` macro's opset lower bound now accepts a `tt` fragment (was `literal`) to allow named constants like `OPSET_STD_LLM`.
**Why:** Without these rows, a Qwen3 built by `onnx-genai-models` declines every norm, every rotary, and every attention for no technical reason. A shared kernel, separate predicates: one predicate spanning both standard and contrib forms would be wrong about one of them in the permissive direction.

---

### 2026-07-29T09:00:39-07:00: T3 begins with `ai.onnx::Attention`, not `GroupQueryAttention` (sequencing)

**By:** Morpheus (D23)
**What:** `ai.onnx::Attention` is the first T3 kernel: no `seqlens_k` indirection, no in-place KV-cache aliasing, no `do_rotary` fold, rotary as its own node. GQA stays committed, stays T3 scope, and T3 does not exit without it — this is a sequencing decision only. Binding constraints: T3's exit criterion is **per producer** (decoder layer as one island for both `mobius` and ORT GenAI); no KV-cache or fp16 design decision may be made as though `ai.onnx::Attention` were the only consumer; reporting T3 progress in a way that implies the GenAI path is served is a §1.5 error.
**Why:** (1) Decouples T3 from `bind_aliased_output` (required by GQA, not yet finished — a critical path through two unfinished owners' work simultaneously is chosen badly). (2) Unblocks a model family buildable locally on two gate-passing GPUs. (3) T3 is also the first real exercise of the entire dispatch path; the faster loop should be where that happens. The standard form is lower-risk (opset-versioned) and the faster loop point the same direction — had the standard form been riskier, Morpheus would have ruled the other way.

---

### 2026-07-29T09:00:39-07:00: `onnx-shape-inference` adopted as a Python oracle; `onnx-ir-rust` / `onnx-runtime-ir` deferred

**By:** Morpheus (D24), Mouse (D-M6-02, D-M6-03)
**What:** `onnx-shape-inference` (Python, Apache-2.0, v0.3.0): adopted as (1) a Trinity harness preprocessing step — runs `infer_symbolic_shapes` over test models before ORT, converting `[dynamic-shape]` declines into claims with zero Rust changes (cheapest coverage in the plan); (2) independent ground truth for C2 contrib fingerprints. `onnx-ir-rust` deferred: no use-def tracking, no topological iteration, no deserialization; wrong structural fit (we never see a protobuf). `onnx-runtime-ir` deferred with named trigger: adopt when we need a graph representation outliving a single `GetCapability` call.
**Why:** We are a guest in ORT's address space, handed `OrtGraph`/`OrtNode` across a C ABI, never seeing a protobuf. Any external IR means copying the whole graph into a second representation inside someone else's process. Today's partitioner is one union-find pass and does not need an IR.

---

### 2026-07-29T09:00:39-07:00: R1 narrowed: ORT GenAI GQA arity risk stands; `mobius` path definitively resolved

**By:** Morpheus (D22)
**What:** For the `onnx-genai-models` (`mobius`) path: Q/K norm is always separate `RMSNormalization` nodes; the 16-input GQA form never occurs because `mobius` never emits GQA. R1 (fused Q/K-norm GQA) is now a question about the ORT GenAI producer only. M1 census item changes from per-model to per-producer; a producer emitting no GQA must report that explicitly. `mobius` landing on the good side does not retire the risk for the ORT GenAI path.
**Why:** An empty cell and "this producer does not emit this op" are different findings and must not look the same (§C3's rule).

---

### 2026-07-29T09:00:39-07:00: §9.1.2 refreshed: SDK installed, two GPUs pass gate, no shader dispatched ⚠️ SUPERSEDED: 45 ops now Live and executing on two GPUs; see D26 below

**By:** Morpheus (D25)
**What:** `DESIGN.md` §9.1.2 records: SDK installed at `C:\VulkanSDK\1.4.350.0`, two local GPUs pass the §7.2 gate, all 168 variants compile. **No shader has still been dispatched on any device.** Local GPUs are a development loop, not coverage — nothing they run is recorded, gated, or reproducible by anyone else.
**Why:** A disclosure section that goes stale in the favourable direction is worse than none: compiling a shader and dispatching one are different facts, only the first has occurred.

---

### 2026-07-29T09:00:39-07:00: `onnx-runtime-tracer` adopted at `0.1.0-dev.5`; seven Vulkan span phases defined

**By:** Niobe (D-N1 through D-N7)
**What:** Dependency: `onnx-runtime-tracer = "0.1.0-dev.5", default-features = false`. The pin is not incidental: that release's clock uses an absolute UNIX-microsecond domain whose origin is machine-level, so a span emitted from inside a plugin cdylib overlays the host's timeline with no offset negotiation. Bumping requires re-verifying the epoch property. Seven span phases: `compile`, `prepack`, `upload`, `record`, `submit`, `fence_wait`, `readback`. `Phase::Submit` is the only phase where `observes_gpu_work()` is `false` (unit test asserts this). `fence_wait` is labelled "UPPER BOUND." GPU timestamps are opt-in under `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` (separate from the main trace env var) to avoid perturbing command buffers on mobile tile-based GPUs.
**Why:** MLX has unified memory and lazy eval; our vocabulary would be actively misleading there. Separating Submit from fence_wait makes CPU/GPU overlap visible, which is what we tune.

---

### 2026-07-29T09:00:39-07:00: `trace.rs` names no Vulkan type; GPU timestamps arrive as raw ticks with metadata

**By:** Niobe (D-N5, D-N6)
**What:** `rust/src/vk/` hands `GpuTimestampReport { calibration, queue_family, intervals }` with unconverted ticks to `trace.rs`. `trace.rs` owns masking, `timestampPeriod` conversion, single-wrap recovery, axis placement. GPU spans go on synthetic device lane `tid = 0x7600_0000 + queue_family` (never the submitting thread's lane). `anchor_uncertainty_us` carried through and printed. Full timestamp-query requirement spec in `docs/PERF.md` §3, routed to Switch.
**Why:** Keeps `trace.rs` on the right side of the layering lint (no `ash`). Puts arithmetic with interesting failure modes (`timestampPeriod != 1.0`, `timestampValidBits < 64`, counter wrap) in the module with unit tests, not the one that needs a GPU to test anything.

---

### 2026-07-29T09:00:39-07:00: Benchmark harness design (Niobe D-N7 through D-N11)

**By:** Niobe
**What:** (D-N7) A case the EP did not claim yields no Vulkan number — `speedup=null`, row marked, `--fail-on-unclaimed` makes it fatal. Claim status from `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` JSON-Lines, not stderr. (D-N8) Noise gate: median + MAD/IQR/p05/p95; robust RSD > 10% marks noisy; delta flagged only when exceeding both the threshold and twice the spread. (D-N9) `bench/environment.py` stamps OS, CPU, ORT version, EP artifact path, Vulkan devices from `epctl --probe-loader`, every `ONNXRUNTIME_EP_VULKAN_*` var; `compare.py` refuses cross-env comparison without warning. (D-N10) OQ-12 anchor: `matmulnbits_q4_b32_K4096_N4096` carries `oq12_anchor=True`; ≥1.5× bar measured there only. (D-N11) `bench/transfer_calibration.py` sweeps a doubling byte staircase, fits `fixed_ns + bytes / bytes_per_ns`, prints paste-ready Rust literal. MVS constants (`SAFETY=3.0`, `node_count≥4`, 64 KiB floor) replaced per device via review.
**Why:** A benchmark without a baseline and variance is a rumour. A harness that cries wolf on jitter gets ignored, and an ignored regression detector is worse than none. CPU fallback is always correct and hides in wall-clock tables.

---

### 2026-07-29T09:00:39-07:00: CI is the only place shaders execute — red CI blocks all merges

**By:** Trinity
**What:** README badge + `.github/CI_POLICY.md` documenting lane structure, failure investigation commands, and branch protection TODO (requires GitHub admin from Justin). Rule: a red CI badge blocks all merges. Every agent checks the badge before reporting work complete.
**Why:** CI was red for ≥4 consecutive runs without detection. Four consecutive CI failures — wrong package, bad YAML, PowerShell array bug, null file_path — all stacked silently because local `cargo ci` passed and CI was not being watched. CI is the ONLY mechanism in this project that verifies any shader executes. A loss of CI is a loss of all empirical evidence.

---

## Round 4 — 2026-07-30T02:49:12-07:00 (first-execution round)

<!-- Merging 25 inbox files from this session. Supersessions resolved: LVP2 retracted, R5 rationale corrected, "no shader dispatched" superseded, kernels-first sequencing superseded by §8.8. -->

### 2026-07-30T02:49:12-07:00: `push_next` chain bug — `let _ = props2.push_next(..)` silently discards pNext chain [D-S12-01]

**By:** Switch
**What:** In ash 0.38, `push_next` takes `self` by value and returns `Self`; discarding the return discards the chain. Consequence: every chained capability field (subgroup size, subgroup stages, shader_float16, SSC) read zero on all devices. Fix: `let mut props2 = { let p = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_props); ... };` — rebind, never discard. Rule: every `push_next` call must rebind, not discard. `#[must_use]` was the hint; treating it as cosmetic was the mistake.
**Why:** This was the root cause behind LVP2 (lavapipe `supportedStages=0`), the `subgroup_size=0` readings, and the UMA misclassification — three "device facts" that were instrument failures. The bug invalidated data that was used in two architectural decisions.

---

### 2026-07-30T02:49:12-07:00: LVP2 (lavapipe `supportedStages=0`) RETRACTED — was instrument failure, not device fact [D-S14-01 / Link]

**By:** Switch + Link
**What:** The quirks watchlist entry LVP2 is retracted. Mesa 23.2.1 lavapipe reports `subgroup_size=8`, `subgroup_stages_raw: FRAGMENT|COMPUTE|TASK_EXT|MESH_EXT`, `subgroup_basic_in_compute=true`. The original `supportedStages=0` was the zeroed default of the discarded `push_next` chain. ⚠️ **SUPERSEDES** the original LVP2 observation recorded earlier in PLATFORMS.md.
**Why:** "We changed a frozen architectural decision on the strength of a number our own bug produced." (D36) Consequence: lavapipe subgroup size is **8** (not 32); CI now exercises the subgroup arithmetic path with a smaller warp than either local GPU. `PLATFORMS.md` LVP2 updated to reflect this.

---

### 2026-07-30T02:49:12-07:00: R5 rationale correction — policy on §7.0; lavapipe premise was false [D36]

**By:** Morpheus
**What:** §7.2's R5 removal rationale is corrected. The decision **stands** but the correct reason is: §7.0 requires capability shortfalls to degrade op coverage, not device availability. Subgroup arithmetic selects between shader variants — a variant choice, not a device-admission criterion. The "lavapipe lacks subgroups" reading was our probe bug. ⚠️ **CORRECTS** the rationale in the 2026-07-29 R5 decision; no code change.
**Why:** A right answer reached through false evidence is an unaudited answer. Leaving the false premise in four documents lets the next decision inherit it silently.

---

### 2026-07-30T02:49:12-07:00: §7.9: capability probing distinguishes "not supported" from "not asked correctly" [D28]

**By:** Morpheus
**What:** Five rules for `vk/caps.rs` and every caller. (1) A probe reports three states including *not determined*, never coerced to "not supported". (2) Every chained query is validated after the call; an all-zero `pNext` chain on a ≥1.1 device is **probe failure** until proven otherwise. (3) `--dump-capabilities` prints raw values, not only derived booleans. (4) Heap predicates stated positively and universally; UMA = "every heap is DEVICE_LOCAL", not "a heap is HOST_VISIBLE". (5) Capability-derived behaviour is tested on one integrated and one discrete device before it is trusted.
**Why:** Two bugs, both silent, from the first two-vendor run: the `push_next` chain discard (D-S12-01) zeroed every chained capability; `detect_uma` returned true on the discrete 4060 because ReBAR maps VRAM `HOST_VISIBLE`. Both are natural mistakes requiring structural prevention.

---

### 2026-07-30T02:49:12-07:00: `is_uma` predicate corrected — "every heap is DEVICE_LOCAL" not "largest heap is HOST_VISIBLE" [D-S12b-01]

**By:** Switch
**What:** Old predicate: "largest DEVICE_LOCAL heap is also HOST_VISIBLE" — wrongly reports discrete ReBAR GPU as UMA. New predicate: every memory heap is DEVICE_LOCAL. Confirmed: Intel Iris Xe `uma=true` (single DEVICE_LOCAL heap); RTX 4060 `uma=false` (system-RAM heap without DEVICE_LOCAL exists). Five unit tests, including one explicitly documenting the previously broken ReBAR case.
**Why:** Staging copy bypass on a discrete GPU (RTX 4060) would have been fast and wrong.

---

### 2026-07-30T02:49:12-07:00: §9.1.2 rewritten — 45 ops execute on two GPUs; three qualifiers travel with any citation [D26]

**By:** Morpheus
**What:** §9.1.2 rewritten. 45 op rows (`Live`), barrier parity 46/28 on both devices under both backends. Three qualifiers: (1) executed through a Rust integration test, not through ONNX Runtime; (2) one kernel on one OS; (3) every contrib op, quantized path, and ORT-mediated route remain unexecuted. "A result obtained only on this desk is not a result this project has" is now more load-bearing, not less. ⚠️ **SUPERSEDES** the 2026-07-29 §9.1.2 "no shader dispatched" entry.
**Why:** The risk inverted: from overclaiming execution we had not performed, to letting "we dispatch on two GPUs" stand in for "the EP works".

---

### 2026-07-30T02:49:12-07:00: M0 is NOT declared; six met, one partial, one not met [D27 / D41 / D42 / D43 / D46]

**By:** Morpheus
**What:** M0 criterion status — Met: build/clippy (1), no-ICD zero devices (4), shader-less claims nothing (5), CLAIM_DEBUG reasons (6), layering lint (7), doc consistency (9). Partial/Met: criterion 8 — legacy backend executed first time ever (46/28 on two GPUs, both backends, bit-exact). Not met: ORT-mediated `Add` claim assertion (2); validation-layer positive control (3 — "no errors surfaced" is what a run with the layer NOT loaded produces). Three items remain: validation positive control (Switch + Trinity); PLATFORMS.md LVP2 retracted (Link — done, see above); CI lanes green on lavapipe (Link + Trinity).
**Why:** A standard that yields the first time it costs something was never a standard. M0 was placed in CI precisely because local success is the easiest and least transferable result.

---

### 2026-07-30T02:49:12-07:00: §10.0.1 R6 — a decision can be right, reasonably reached, and rest on evidence we manufactured [D37]

**By:** Morpheus
**What:** Three rules: (1) a decision record names its load-bearing reason; a number is never load-bearing alone — when supported by both a principle and a measurement, record which would have to fail for the decision to change; label provisional if the answer is the measurement. (2) A number produced by our own tooling is evidence about our tooling until corroborated by a second instrument. (3) A correction that leaves the conclusion standing must still be published.
**Why:** R5 removal was reached by false evidence. Asymmetry noted: R5's direction (removing a capability) was safe; the same mistake in a permissive direction (adding a claim) would have shipped wrong answers.

---

### 2026-07-30T02:49:12-07:00: §10.0.1 R7 — instruments fabricate negatives; "absence of signal must not read as success" [D44]

**By:** Morpheus
**What:** Three-layer skip contradiction: `OnceLock`-cached claim-log path (dead instrument reads as "not claimed"), profiling-JSON workaround crashed on Intel, per-op `live` flag introduced a vacuous pass on `Add-i32` against an f32-only predicate. Two rules: (1) absence of an instrument must not read as a negative result — a probe that cannot find its data raises, it does not return False; (2) *derive, do not declare* — any fact the code knows is computed from the code, per form, never restated in a second language.
**Why:** R7 is R6's twin and arguably worse: a manufactured negative asks for no explanation. Layer 2 and 3 were competent engineering applied to a false premise, each making the system worse.

---

### 2026-07-30T02:49:12-07:00: Execution counters are always-on at the ORT boundary; `dispatches_executed` is the gate [D-T30]

**By:** Tank
**What:** `rust/src/counters.rs`: six process-wide relaxed atomics (`compile_calls`, `subgraphs_live`, `subgraphs_stub`, `compute_calls`, `compute_failures`, `dispatches_executed`), `VulkanEpCounters` C ABI struct, two exported C symbols, JSON file gated on `ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE`, and teardown summary logged at WARN when count is zero. `dispatches_executed` incremented **after `dispatch_ort` returns null status** (fence waited, device finished). `epctl --check-counters <file>` exits 0/1/3 (0=pass, 1=below threshold, 3=lane did not report). File written twice: on first dispatch and at teardown.
**Why:** M0 criterion 8 requires a *reported* non-zero dispatch count. Four consecutive red CI runs went unnoticed; two fabricated speedups retracted — all because "GPU work happened" was inferred from something else being green. A counter ORT's teardown emits cannot be inferred away. Debug-only would be absent in CI; log lines are greppable prose that breaks across owners.

---

### 2026-07-30T02:49:12-07:00: `Compute` must never return null on failure; null means success to ORT [D-T22]

**By:** Tank
**What:** `SubgraphComputeInfo` carries the `OrtApi` pointer; `Compute` returns a real `OrtStatus` on every internal error. ORT reads null from `Compute` as success — a failed compute reporting success leaves output tensors holding whatever was in them.
**Why:** This is the same shape as the two fabricated speedups: a precondition dressed as an effect. A crash is better than a silent wrong answer. Pinned by `a_failing_compute_returns_a_real_status_not_a_silent_success`, verified adversarially.

---

### 2026-07-30T02:49:12-07:00: Real `OrtAllocator` — reserved-VA handle scheme built and verified [D-T35 / D-T38 / D-T45 / D-T48]

**By:** Tank
**What:** `src/allocator.rs`: 64 GiB virtual-address reservation per device (`MEM_RESERVE`/`PROT_NONE`), page-aligned span carving, `BTreeMap<usize, Span>` for interior-pointer resolution. `attach_buffer(addr, BufferView)` is Switch's seam. Data transfer (`src/transfer.rs`) implemented: `CanCopy`, `CopyTensors`, `Release`. Device memory behind `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` (default off, blocked on `OrtDataTransferImpl`). **ORT's planner does arithmetic on our handles (D-T48):** 52 interior pointers observed, identical on both devices, all within span, `pointers_in_guard_band=0`. The planner does not engage on run 1 — every earlier "0 interior" probe was pointed at the wrong moment. `EpDevice_AddAllocatorInfo` leaks memory info by design (D-T36) — ORT dereferences it after `GetSupportedDevices`, so releasing causes an AV.
**Why:** ORT pointer arithmetic on allocator return values would silently corrupt data with integer handles. The reserved-address-space design makes `ptr + n` stay in-span by construction, and makes the address unreadable so stray dereferences fault.

---

### 2026-07-30T02:49:12-07:00: `glslc` discovery includes installed-but-unexported Windows SDK [D-T29]

**By:** Tank
**What:** `build.rs::find_glslc` now searches (1) `$VULKAN_SDK/bin`, (2) `$PATH`, (3) highest-versioned `C:\VulkanSDK\<version>\Bin\glslc.exe`. Emits `cargo:warning` naming the compiler when it falls back to (3). Prevents phantom test failures when the SDK is installed but not on `PATH` or `VULKAN_SDK`.
**Why:** A parity tool red for a reason CI is not is worse than no tool. False reds teach you to ignore the tool; that habit is how four consecutive red CI runs went unnoticed.

---

### 2026-07-30T02:49:12-07:00: Worktree layout and inbox-portability constraint

**By:** Coordinator (structural fact)
**What:** The team now works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree. **`.squad/decisions/inbox/` is gitignored — records written in a worktree do not travel with the branch.** The inbox in `main` is authoritative. If the inbox count looks short against a manifest, sweep worktrees rather than assuming a record was never written.
**Why:** Records have already been missed due to worktree isolation. Tank's `cargo ci` now warns about it.

---

### 2026-07-30T02:49:12-07:00: Hardware matrix — CI verified, local GPUs confirmed, Intel = spec oracle [Link]

**By:** Link
**What:** Linux CI lane VERIFIED (`llvmpipe`, Vulkan 1.3.255, driverID MESA_LLVMPIPE); Windows CI lane VERIFIED (lavapipe enumerated after ICD registry registration). Local: Intel Iris Xe Vulkan 1.4.309 PASS, RTX 4060 Laptop Vulkan 1.4.325 PASS. Intel is the spec-conformance oracle — Intel failures predict MoltenVK failures and real portability problems. **Do not special-case Intel.** UMA = first-class platform column: Intel Iris Xe and mobile (Adreno, Mali) are UMA; test results on Iris Xe are a closer mobile proxy than results on the RTX 4060.
**Why:** First execution-derived hardware matrix entries replace inferred/expected entries.

---

### 2026-07-30T02:49:12-07:00: Vulkan struct teardown order enforced by field declaration order [D-S13-01]

**By:** Switch
**What:** `VulkanSession` fields declared in reverse creation order: pipeline_cache → cmd_pool → alloc → device → **instance last**. Rust drops fields top-to-bottom; declaring `instance` first caused `vkDestroyInstance` before `vkDestroyDevice` → STATUS_ACCESS_VIOLATION in `cdylib_load`.
**Why:** Correct Vulkan teardown must be structurally enforced, not dependent on a `Drop` impl that can drift from the creation order.

---

### 2026-07-30T02:49:12-07:00: 45 op rows Live; Add and elementwise f32 execute on two GPUs; `Add` via ORT confirmed [D-M11-01 / D-M12-01 / D-M12-02]

**By:** Mouse
**What:** 1 → 45 Live rows. `Add` is Live for f32 only; `caps` stays `NUMERIC` for variant generation but `ew_binary_exercised` predicate declines non-f32 (`EXERCISED` evidence list: `add_f32_dispatches_end_to_end`, Intel Iris Xe 1.4.309, RTX 4060 Laptop 1.4.325). `Sub/Mul/Div/Pow` stay Staged — same template is not evidence (D-M11-02). `OnceLock`-cached claim log path fixed (D-M12-02) — `path()` now re-reads env var per decision so mid-process changes redirect. Profiling-JSON retained as the `is_vulkan_claimed` mechanism (D-M12-01 Trinity §22 finding: post-load env var changes unreliable for the DLL on Windows).
**Why:** Seven activations unlocked by attribute push-constant tail (D below). The three-layer skip contradiction identified and closed.

---

### 2026-07-30T02:49:12-07:00: Op attributes ride a push-constant parameter tail; Clip does not [Mouse attribute-params]

**By:** Mouse
**What:** Four-float parameter tail unconditionally at the end of every push-constant block (worst case 104 → 120 bytes, within 128-byte limit). Parameterless ops push four zeros they never read. One slot table read by both predicate and handler. Unlocks Selu, Elu, HardSigmoid, Shrink, ThresholdedRelu, LeakyRelu, CeluAlpha (7 activations). `Clip` excluded: two parameters, and the ORT kernel and GLSL semantics differ for NaN inputs.
**Why:** A pipeline layout dependent on op would make a wrong push range UB that validation may not catch. Fixed range is cheaper than the bug class.

---

### 2026-07-30T02:49:12-07:00: `MatMulNBits` GEMV is Live (fp32 and fp16); layout read from the oracle [Mouse matmulnbits]

**By:** Mouse
**What:** `com.microsoft::MatMulNBits` Live for all `M`, fp32 and fp16. One workgroup per output element, grid `(N, M_total, 1)`. Layout derived by feeding `A=I` through ORT CPU EP. fp16 through `unpackHalf2x16/packHalf2x16` over `uint` — no 16-bit storage capability gated. Verified on both devices: `bits∈{4,8}`, `block_size∈{16,32,64,128}`, 3-input symmetric and 4-input asymmetric, fp32 `M∈{1,2,7,32}`, fp16 `M∈{1,3}`. All 161 Phi-3.5 `MatMulNBits` nodes are fp16 (`K∈{3072,8192}`, `N∈{3072,8192,9216,32064}`, bits=4, block_size=32, 3-input symmetric).
**Why:** Claiming `MatMulNBits` alone actually worsens partition (D4 below) due to interleaving with GQA. Correct but incomplete coverage can be worse than less coverage.

---

### 2026-07-30T02:49:12-07:00: Foundry Local census — two real production graphs; §8.5 third strengthening [D-M10-01 through D-M10-06 / D31 / D32]

**By:** Mouse + Morpheus
**What:** Phi-3.5-mini-instruct (`ai.onnx`=14, `com.microsoft`=1) and gpt-oss-20b (`ai.onnx`=21, `com.microsoft`=1) read from disk. Five findings: (1) `OPSET_STD_LLM=23` excludes both models — standard-domain rows serve mobius only; (2) `do_rotary=1` is universal, packed QKV on all GQA nodes — "claim GQA first, add rotary later" claims 0 nodes; (3) packed QKV predicate narrowed to require both inputs present; (4) `SimplifiedLayerNormalization` has `domain=""` in both graphs with no ONNX schema; (5) `QMoE` top-4 found (not top-1|2 as in schema). §8.5 rule: "a claim about what a producer emits is not evidence until read off a graph that producer actually produced. Builder source is intent; the model file is the fact." Metric of record upgraded to triple: `(claimed_op_coverage, island_count, largest_island_flops)` reported together, per producer at version.
**Why:** Death-by-fallback observed: claiming `Cast` on gpt-oss raised coverage 28%→54% while raising islands 52→125. Coverage percentage alone scores it a 26-point win while partitioning worsens.

---

### 2026-07-30T02:49:12-07:00: mobius at `onnxruntime/mobius@87fd878`, default opset 24; producer revision pinned [D-M8-01 / D-M8-02 / D-M8-03]

**By:** Mouse (corrected by Justin directive)
**What:** Authoritative mobius is `onnxruntime/mobius` (not `justinchuby/onnx-genai-models`). Default opset 24. `ai.onnx::Attention` gained optional input 6 `nonpad_kv_seqlen` at opset 24 — a predicate written against opset 23 would have claimed that node and returned wrong logits. New ops found: `ai.onnx::Swish` (opset 24), `ai.onnx::TensorScatter` (opset 24). Producer revision must be pinned in census. Emitted op set is a function of producer, revision, **and how we describe ourselves** — contrib rows are not dead code on the mobius path if we advertise GQA support.
**Why:** "The op inventory was correct in method but drawn from the wrong revision of the producer — the same class of error one level up." (Justin directive)

---

### 2026-07-30T02:49:12-07:00: ONNX Attention-24 `nonpad_kv_seqlen` — no opset bump; implement corrected semantics [copilot-directive / Fact Checker]

**By:** Justin Chu (directive) + Fact Checker
**What:** ONNX reference implementation of `ai.onnx::Attention`-24 was wrong for `nonpad_kv_seqlen` and was fixed in onnx 1.23 (commit 2816da65, June 20) **without an opset bump**. Justin ruled: implement the corrected semantics; no dual-path. Oracle must pin `onnx>=1.23` (fix not yet in any stable release as of 2026-07-29; onnx 1.22.0 is latest). ORT CPU EP (`>=1.28.0`) is already correct — pin matters only when using standalone onnx Python reference evaluator. **Opset-based version checks cannot see this class of defect by construction** — this is a known limitation of C2.
**Why:** The same opset-24 graph yields different expected outputs under onnx 1.22 vs 1.23 with no signal in the model. An unpinned onnx drifts the oracle exactly as an unpinned `accuracy_level` drifts the ORT EP confidence knob.

---

### 2026-07-30T02:49:12-07:00: Opset range through current version; both ONNX opset answers recorded [D-M9-01 / D-M9-02 / D-M9-03]

**By:** Mouse (Justin directive: support opset up to current)
**What:** `ONNX_OPSET_LAST_RELEASED=26`, `ONNX_OPSET_REGISTERED=27` (distinct constants, test asserts). All closed windows already cover every published schema version of their op; no window widened. `LinearAttention-27` and `CausalConvWithState-27` registered — onnx v1.22.0 standardised the Qwen3.5-hybrid ops (formerly `com.microsoft` main-branch).
**Why:** Closed windows are schema-version windows; an opset-27 model is claimable for all of these — ORT resolves each node to its schema version. The distinction matters for elementwise rows (open-ended upward) vs LLM rows (versioned and closed).

---

### 2026-07-30T02:49:12-07:00: §8.8 RULING — dynamic-shape support is a claim-path capability; moves ahead of kernels [D47 / D48]

**By:** Morpheus
**What:** Symbolic extents are no longer a decline per se. Predicate now distinguishes: (a) rank known / extents symbolic → **claimable** if kernel takes extents as runtime parameters; (b) rank unknown → decline; (c) data-dependent → permanently declined. `REQUIRE_STATIC_SHAPES` renamed to `ENGINE_ACCEPTS_RUNTIME_EXTENTS` (inverted, gated on Switch's work). OQ-15 (indirect dispatch) promoted from tier-3 evaluation to **blocking** for M1. M1 gains the second-token criterion: same session, two different concrete values of a symbolic dim, no session re-compile. ⚠️ **SUPERSEDES** "kernels-first" sequencing; dynamic shapes are upstream of kernels, not in competition.
**Why:** Phi-3.5 declined 363 nodes: `dynamic-shape` 258, `staged` 100. Full-set audit (D below) shows the asymmetry is larger: 98 of the 100 staged nodes are also shape-blocked. Landing all staged kernels while static shapes required yields 0 claimed nodes.

---

### 2026-07-30T02:49:12-07:00: Decline census reports full set of failing checks; first-match is a ceiling [D-M decline-census / D3 mouse-runtime-extents]

**By:** Mouse
**What:** `registry::claim_audit` runs all checks, returns full set. JSONL record gains `codes`, `reasons`, `unevaluated`, `shape_class`, `predicate_ok`, `predicate_ok_runtime_extents`. Full-set Phi-3.5: `dynamic-shape` is **356 of 363**, not 258. 98 of 100 staged nodes are also shape-blocked. Landing all three staged kernels alone unlocks **0** nodes. 227 nodes predicate-clean under runtime extents; 161 (`MatMulNBits`) claimable immediately. gpt-oss full-set: `dynamic-shape` 342 > `staged` 197 — reading first-match would have reversed a correct ruling.
**Why:** "A decline census is only usable for planning if it reports every failing check." R8: §10.0.1 — a decline code names the first failing check, not the only one.

---

### 2026-07-30T02:49:12-07:00: `ENGINE_ACCEPTS_RUNTIME_EXTENTS` flipped; 161 nodes claimed on Phi-3.5 [D-S18-01]

**By:** Switch
**What:** `ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. `CompiledKernel` stops baking extent-dependent fields at Compile for dynamic nodes; `dispatch_ort` reads real shapes at Compute via `GetTensorTypeAndShape`/`GetDimensions`. OQ-15 resolved for M1: re-record per shape (M2+ bucketing is an optimisation). Device 0 (Intel Iris Xe): 161 claimed, zero validation errors; Device 1 (RTX 4060): 161 claimed. Variable seqlen (seq=1 and seq=5 in same session) correct on both devices.
**Why:** No new kernels needed — only the dispatch path changed. 97 nodes on Phi-3.5 (Mul×64, Sigmoid×32, Sub×1) were declined solely because the engine baked extents at Compile.

---

### 2026-07-30T02:49:12-07:00: §10.0.1 R8 — a decline code names the first failing check, not the only one [D47 D morpheus-dynamic-shape]

**By:** Morpheus
**What:** A first-match histogram answers "why did this node fail?" for one node; it cannot answer "what should we build first?" for a graph, and it fails in the direction of looking authoritative. The 100 `staged` nodes had never reached the shape check — their shape viability was unknown. The correct asymmetry is larger than 2.5×; it is total.
**Why:** Mouse's full-set audit proved the first-match histogram on gpt-oss would have reversed a correct architectural ruling. Risk R8 added to §10.0.1.

---

### 2026-07-30T02:49:12-07:00: Trinity — real Phi-3.5 run; barrier parity on two devices; Attention-24 oracle pin; profiling-JSON for `is_vulkan_claimed` [trinity-test-foundation §21 / §22]

**By:** Trinity
**What:** Phi-3.5-mini-instruct (2.2 GB) through the EP: loads, runs, 65 outputs, bit-identical across sessions, variable sequence length correct. Barrier parity 46 passed / 28 skipped on both devices (both barrier backends) with non-zero dispatch counts — first execution of the legacy backend. `_assert_oracle_versions()` in conftest.py: refuses if ORT<1.28 or onnx<1.22.0. Attention-24 tests marked `xfail(strict=True)` until `onnx>=1.23`. `is_vulkan_claimed` reverted to profiling-JSON (post-load env var changes unreliable for loaded DLL on Windows). Multi-device parametrization wired (blocked on Switch `ep.device_index`).
**Why:** First production-model run through the EP. Legacy barrier backend had never executed; its compatibility claim now rests on bit-exact agreement rather than just code review.

---

### 2026-07-30T02:49:12-07:00: Niobe — producer provenance structural; two fabricated speedups caught; portability floor is §7.2 [D-N19 through D-N28]

**By:** Niobe
**What:** `bench/producers.py`: `Producer{name,kind,version,digest,opsets,family}`, fingerprint `name@version#digest`. A case cannot carry a model-family label unless the producer built for that family. `compare.py` refuses cross-producer comparison (exit 2) — a timing without provenance is a mislabelled number. Portability floor = §7.2 (16 KiB shared, 256 invocations) not the smaller local GPU. `SUBGROUP_SIZE_IS_GUARANTEED=False` — both local GPUs happen to report 32, not a guarantee. `bench/portability.py`: verdicts `portable/needs-fallback/unknown`; `fits_device` from reported limits. UMA and discrete transfer models may not be blended. Two fabricated speedups caught: 1.70× (ORT 1.27 prints failure, doesn't raise) and 1.45× (EP loads, declines everything, all columns are CPU EP). Neither claimed.
**Why:** A benchmark artefact is relative to its producer. "A timing has no shape to disagree about and so nothing fails loudly."

---

### 2026-07-30T02:49:12-07:00: Cross-platform generality is a standing constraint, checked continuously [copilot-directive-cross-platform]

**By:** Justin Chu (directive)
**What:** Cross-platform generality must be kept in mind at all times — not audited at the end. Specific structural forms: derive workgroup sizes from `maxComputeSharedMemorySize`, never from local constants; Intel Iris Xe = UMA = mobile proxy for memory model; Intel = spec oracle; no required extensions per §7.2; every `cfg` is a portability hazard (test `tests/portability.rs`); local results are development loop, not coverage.
**Why:** This EP is the *cross-platform* backend. A Vulkan EP that only works on desktop NVIDIA has no reason to exist.

---

### 2026-07-30T02:49:12-07:00: onnx-runtime-ir trust objection withdrawn; structural objection stands independently [copilot-directive-mobius-opset24 / D30]

**By:** Justin Chu (directive) + Morpheus
**What:** `onnx-runtime-ir` (in `justinchuby/onnx-genai`) is the team's own crate; prudential trust objection withdrawn. Structural objection unchanged: ORT hands us `OrtGraph/OrtNode` across a C ABI; we never see a protobuf; adopting an external IR means copying the whole graph inside someone else's address space. Must be answered on merits. Named trigger still active: adopt when a graph representation must outlive a single `GetCapability` call.
**Why:** Ownership retires the prudential half only. The structural argument must be engaged on its own merits.

---

### 2026-07-30T02:49:12-07:00: Phi-3.5 is a better first-execution target than Qwen3 [D-M10-06 / D-M10-05]

**By:** Mouse (routed to Morpheus)
**What:** Phi-3.5-mini-instruct: on disk, runnable, MHA (no KV-head broadcast), softcap=0, no SWA, symmetric RTN quantization (bits=4, block=32, K%32=0), cold-path control flow. Five op types cover 353/366 main-graph nodes; T4 exit criterion: `MatMulNBits` claimed → Phi-3.5 partitions into one island of ≥360 nodes. `ai.onnx::Attention` stays T3 implementation entry point (simpler schema, serves mobius); T3 demonstration is Phi-3.5.
**Why:** The only LLM on disk, measureable, exercises none of the standard-domain LLM rows (§8.5 requires reporting separately). gpt-oss mixes 4-bit and 8-bit quantization and has SWA on 12/24 layers.


---

### 2026-07-30T02:49:12-07:00: fp16 elementwise Live; `OpCapability` allowlist; sub-word tail constraint [D-M1 / D-M2 / D-M3 — mouse-f16-elementwise.md]

**By:** Mouse (not in original manifest — found in inbox)
**What:** (D-M1) `only_proved_dtypes` replaces `only_f32` — predicate reads `EXERCISED` directly, so widening a claim is one edit, not two that can diverge silently. (D-M2) f16 elementwise via `uint` buffers + `unpackHalf2x16/packHalf2x16` — not via `OpCapability StorageBuffer16BitAccess` (which crashes under the engine's current feature set). `no_shader_requires_a_device_feature_the_engine_does_not_enable` assertion decodes `OpCapability` from every embedded module against an allowlist. This class of bug is silent when the matching claim is closed — must be caught in artifact, not at runtime. (D-M3) `claim::check_subword_tail` declines tensors where last-axis byte count is odd — ORT sizes tensors exactly and the EP binds what it is given; rounding up is Switch's named lift condition (bind `VkDescriptorBufferInfo.range` rounded up to multiple of 4). **Cross-owner flag:** `rust/shaders/include/indexing.glsl` edited under Switch's ownership (pre-ruling) — Switch should review the f16 packed-half hunk.
**Result:** 257 nodes have `codes=["dynamic-shape"]` and `predicate_ok_runtime_extents=true` on Phi-3.5 — one blocker, named. `MatMulNBits×161 + Mul×64 + Sigmoid×32 = 257` claimed when shapes pinned. Same argmax token (30751) as CPU EP; first real-model arithmetic on this EP. **The 97 dtype and 258 dynamic-shape counts were not disjoint** — the correct unified statement is 257 extent-blocked with predicate OK.
**Why:** A closed claim hides bugs in the variant it excludes — `StorageBuffer16BitAccess` modules had been unloadable since written, invisible because f16 was closed. The RTX 4060/Iris Xe divergence (6/12 fp16 failures on device 0) found the sub-word overrun before it could mislead.

---

### 2026-07-30T05:48:29-07:00: P6 (no dequantised weight in device memory) — proved by `alloc_temp` call count, not byte threshold; multi-run insulation verified structurally [mouse-p6-and-multirun.md]

**By:** Mouse (Op Coverage Engineer)
**What:** (1) P6 is proved by asserting zero `alloc_temp` calls per GEMV dispatch, not by a high-water byte threshold. Zero calls proves the property for all shapes at once; any non-zero threshold proves it only for tested shapes. Two tests in `ops::quant`: zero allocations / one dispatch / activation-sized output across three real (K, N) pairs; and allocation record byte-identical when K is quadrupled at fixed N — the shape that breaks first if a dequantised [K, N] buffer appears. Negative-controlled: deliberate `alloc_temp` in the GEMV made it fail, then reverted. (2) Model-level checks must run more than once per session. ORT's memory-pattern planner returns interior pointers from run 2 onward — five runs with differing feeds verified clean (worst max|d| 0.08984 device 0, 0.09473 device 1; 5/5 argmax; late repeats bit-identical). Insulation is structural: op code sees only `DispatchContext`; `transfer::host_backing_for` resolves the offset below the raw pointer.
**Why:** A single-inference model check covers run 1 only. Constraints named in design docs and code comments are not enforced until something fails when violated. A threshold loose enough not to be flaky is loose enough to hide a scratch buffer. Zero is not a threshold.

---

### 2026-07-30T05:48:29-07:00: Variant census — staged-f32 kernels claim 0 Phi-3.5 nodes; i64 variants have same device-feature bug as f16; 257 nodes total with shape pin [mouse-variant-census.md]

**By:** Mouse (Op Coverage Engineer)
**What:** (1) All staged nodes on Phi-3.5 that matter are f16 end-to-end. `skip_simplified_layer_norm_f32.comp` claims 0 Phi-3.5 nodes — same mechanism as the elementwise f32 family. The f16 variant is the kernel. SkipSimplifiedLayerNormalization has a varying output count (63 nodes bind two outputs, 1 binds one); GroupQueryAttention mixes dtypes within one node (f16 tensors + i32 sequence-length inputs). (2) `_i64` variants declare `OpCapability Int64` requiring `VkPhysicalDeviceFeatures::shaderInt64`, which `vk::device` passes no `pEnabledFeatures` to enable. Every `_i64` module is uncreatable on every current device — same class as the f16 bug. Fixed by splitting into `GENERATED_CAPABILITIES` (what a built variant may declare) vs `ENGINE_ENABLED_CAPABILITIES` (what a live claim may rest on). `no_live_claim_rests_on_an_unloadable_variant` assertion walks every proved (op, dtype) pair. Fired deliberately as negative control and reverted. (3) 257 nodes claimed with shapes pinned: MatMulNBits×161 + Mul×64 + Sigmoid×32. Check to run when Switch's runtime-extent branch merges: unpinned census should claim 257; if 161, f16 nodes are lost on the dynamic path. i64 tail (7 nodes: Sub, ReduceSum, Shape, Greater, Gather, Cast) recommended left declined. Note: Switch's earlier record says 97 nodes unlocked by runtime-extent work; the Sub is i64 and stays declined — correct number is 96, plus 161 MatMulNBits = 257 total.
**Why:** A guard whose allowlist is written from the same misunderstanding as the bug it guards against inherits the bug. The dangerous review comment is one that sounds like a justification but is a restatement.

---

### 2026-07-30T01:32:15-07:00: Counter scoping corrected — `FIRST_DISPATCH_DUMPED` removed; CLAIM_LOG measures GetCapability offers not ORT acceptance; island regex fixed [switch-counter-scoping.md]

**By:** Switch
**What:** Three apparent contradictions (Claimed=161, Islands=0, counters={1,1,1}) each had an independent explanation: (1) `counters::record_dispatches()` used a `FIRST_DISPATCH_DUMPED` one-shot atomic; `conftest.py::_probe_vulkan_device()` creates an Add session before any test, so the counters file was written with the probe's state. Fix: removed one-shot; `record_dispatches()` calls `dump_if_requested()` on every dispatch. `test_phi35_session_loads_and_declines_cleanly` now resets counters before the Phi-3.5 session and reads in-process via ctypes. (2) `CLAIM_LOG` records `GetCapability` *offers*, not ORT *acceptance*. Correct Phi-3.5 counters with scoped measurement: `{compile_calls:1, subgraphs_live:161, compute_calls:161, dispatches_executed:161}` on both devices. (3) `_count_islands` regex required two short numeric suffixes; actual ORT plugin-EP event names have a large hash followed by indices. Fix: count distinct `args["op_name"]` values among VulkanExecutionProvider events → 161 islands. All three verified on both devices.
**Why:** Three counters that read as contradictory were three different instrument-scoping bugs. The distinction between GetCapability offers and ORT acceptance is permanent context every agent must carry.

---

### 2026-07-30T01:32:15-07:00: Validation positive control for M0 C3 — VUID-00332 caught in dispatch path; VUID-03047 deferred to submit-time by SDK 1.4.350.0 [switch-positive-control.md]

**By:** Switch
**What:** `rust/src/vk/dispatch_integration.rs` `#[test] #[ignore]` function `descriptor_set_updated_while_bound_fires_vuid_03047` creates a Vulkan instance with `VK_LAYER_KHRONOS_validation` + `VK_EXT_debug_utils`, installs a `VkDebugUtilsMessengerEXT` callback (counter incremented per message), deliberately violates VUID-00332 (buffer created with `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT` written as `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` descriptor), and asserts counter > 0. Confirmed: 31 validation errors captured on Intel Iris Xe (SDK 1.4.350.0). Finding: VUID-03047 (update a bound descriptor set) fires lazily at submit time, not at the `vkUpdateDescriptorSets` call site, in SDK 1.4.350.0. The control uses VUID-00332 (checked immediately). The session-16 `DispatchDescriptorPool` fix addressed the same descriptor-writing domain; this control demonstrates the capture path is live for that code path.
**Why:** "No validation errors surfaced" is only a meaningful claim if the validation layer demonstrably catches real errors. A positive control that has never been observed failing is a guard nobody has tested.

---

### 2026-07-30T02:33:16-07:00: Guard-band assertion in `epctl`; no EP debug messenger; `epctl --probe-validation`; planted violation; skips must be loud [tank-validation-control.md — D-T60 through D-T66]

**By:** Tank (Integration/Validation Engineer)
**What:** (D-T60) `epctl --check-counters` fails `OutOfBounds` on non-zero `pointers_in_guard_band`, ordered before `--require-dispatches`. Key absent = "no ledger", not zero, not pass. (D-T61) `vk/instance.rs` requests `VK_LAYER_KHRONOS_validation` but attaches no `VkDebugUtilsMessengerEXT` — Switch's file, deferred. Two independent reasons criterion 3 proved nothing before this session. (D-T62) `epctl --probe-validation [--plant-violation]` with three states: `VALIDATION ARMED` (exit 0), `VALIDATION LAYER ABSENT` (exit 3), `NO VULKAN LOADER` (exit 3). Prints `EPCTL-VALIDATION-CAUGHT:` per captured message. `tests/validation_control.rs` asserts both directions. (D-T63) Plant: `vkCreateDebugUtilsMessengerEXT` with zero `messageSeverity`/`messageType` masks (VUID parameter check; no device needed; caught on any ICD including lavapipe). (D-T64) EP's own instance (`vk/instance.rs`) is a separate scope; epctl probe does not cover it. Closure: env-gated fence leak in dispatch path (`ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION`; VUID-vkDestroyDevice-device-05137). Routed to Switch. (D-T65) `ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION=1` turns skip→failure in both epctl and harness. (D-T66) `cargo ci` epilogue corrected — "no `vkCreateInstance`" was false after `validation_control.rs` landed.
**Why:** A check that verifies the value but not the provider is a gate that does not gate. A skip configured to silently pass is a check a lane can lose forever. The honest-caveat epilogue must be maintained as carefully as an assertion.

---

### 2026-07-30T09:14:00-07:00: False-premise test audit — `test_phi35_vulkan_matches_cpu_logits` xfail correctness gate; two mandatory guards; determinism test renamed [trinity-false-premise-audit-2026-07-30.md]

**By:** Trinity (Test Architect)
**What:** `test_phi35_cpu_output_matches_between_sessions` compared two VulkanEP sessions and passed vacuously when both produced all-zero logits (bit-identical zeros). Triggered by Switch's runtime-extents merge: 161 MatMulNBits claimed and dispatched with `compute_failures: 0`, but producing all-zero logits; entire suite was green. (1) New gate `test_phi35_vulkan_matches_cpu_logits`: one `EP_PROVIDERS` session vs one `["CPUExecutionProvider"]` session; Guard A — `EP_NAME in sess.get_providers()` before `sess.run()`; Guard B — `max|logit_vk| > 0.1` before comparison; asserts argmax agreement + top-10 overlap ≥ 5/10. Marked `xfail(strict=True)` — currently XFAIL because `max|logit_vk| = 0.0`. (2) `test_phi35_cpu_output_matches_between_sessions` renamed `test_phi35_vulkan_session_determinism` with corrected premise. (3) `test_phi35_variable_seqlen_fallback` renamed and docstring corrected. (4) `assert_ep_in_providers` helper added to `_models.py`. (5) Module and per-test docstrings updated throughout. Standing rule: any test comparing two sessions both created with `EP_PROVIDERS` is a determinism gate only. A correctness gate requires one `EP_PROVIDERS` + one `["CPUExecutionProvider"]` session plus Guard A.
**Why:** A green suite was shown not to imply a correct model. The coordinator's own VulkanEP-vs-CPU comparison reported bit-identical on both devices because it never called `register_execution_provider_library` — ORT printed `Unknown Provider Type ... Falling back to CPUExecutionProvider` without raising, comparing CPU to CPU. Same defect class, third instance: absence of an instrument reads as a positive result.

---

### 2026-07-30T05:48:29-07:00: `compute_failures == 0` is a detector statement, not a correctness statement; `model_output_equivalence` verdict required [morpheus-compute-failures-ruling.md]

**By:** Morpheus (Lead / EP Architect)
**What:** `compute_failures` counts times `Compute` returned non-null — the times our code detected a fault. Its sole licensed reading: "no dispatch reported an error it was able to detect." It does not license "kernels are correct", "graph produced the right answer", "run is usable". A kernel that writes zeros into every output, submits, waits on the fence, and returns null is a complete success by all six counters simultaneously. General rule: an execution-status counter's zero is a statement about the detector, not about the computation. Its silence set is everything downstream of "the dispatch returned." Mechanism: (1) `model_output_equivalence` ∈ {`MATCH`, `DIVERGENT`, `UNMEASURED`} emitted next to the counters from the same run, against a CPU-only execution of the same session. `UNMEASURED` is the default. (2) No counters summary may be quoted without its verdict. `epctl --check-counters` reports it alongside; a file with no verdict field reports `UNMEASURED` explicitly. Owners: Switch (emission), Trinity (comparison), Niobe (`PERF.md`). (3) Counter is not renamed; `VulkanEpCounters` is a published C ABI; verdict is an addition — new optional field, absent means `UNMEASURED`.
**Why:** The absence of a report is the success report — identical shape one layer down to ORT's null-pointer success convention. Every "no error" signal on the compute path is treated as `UNMEASURED` until a positive control or differential comparison converts it. Prose constraint on a reading is a declaration; R7 applies to documents as much as to the harness.

---

### 2026-07-30T05:48:29-07:00: R9 — for every claim, name the instrument that would go red if the claim were false [morpheus-r9-name-the-red-instrument.md]

**By:** Morpheus (Lead / EP Architect)
**What:** New rule of record R9: "A set of individually sound instruments can be jointly silent on the property that matters, and their agreement raises confidence without raising evidence. Therefore: for every claim, name the instrument that would go red if the claim were false. If no such instrument exists, the claim is not evidenced — however much telemetry surrounds it." Four operational rules: (1) Every claim carries a named falsifier; if the sentence "this would have gone red if the claim were false: ___" cannot be completed, claim is downgraded to `UNMEASURED`. (2) A criterion whose falsifier does not exist yet is not met, however much evidence surrounds it. (3) Positive controls are the standard mechanism; a check never observed failing is a check of unknown polarity. (4) The composite is not a free instrument — combining sound readings produces a new claim requiring its own falsifier. Additionally: when an instrument is added, its silence set is recorded with it.
**Why:** 161 MatMulNBits nodes offered and accepted by ORT; `compute_failures: 0`; entire test suite green; and `vk logits: [0.0000, 0.0000] argmax 0 top-10 overlap 0/10` (vs CPU `argmax 30751`). Output #64 differs by 25.27. Every instrument was individually correct, several freshly repaired. The composite reading — "161 nodes execute on the GPU" — was true, was used as a correctness claim, and not one instrument in the set measured correctness. R7 covers instruments that fabricate negatives; R9 covers corroborating instruments that answer the wrong question. More diligence, more devices, more telemetry makes this failure *worse* — the wrong conclusion becomes more persuasive.

---

### 2026-07-30T05:48:29-07:00: M0 criteria amended — criterion 10 added; criteria 2, 4, 5 reopened; criterion 8 relabelled [morpheus-m0-criteria-reopened.md]

**By:** Morpheus (Lead / EP Architect)
**What:** Criterion 10 (NEW, NOT MET): a real model at a named producer/version with non-zero claimed-node count produces output equivalent to a CPU-only run — `model_output_equivalence = MATCH`, reported next to counters. `dispatches_executed > 0` and claimed count > 0 so CPU-fallback cannot satisfy it. Currently measured `DIVERGENT`. Criterion 2 REOPENED (Met → Partially met): bottoms out in a single `Add`; the cheapest thing satisfying its words computes one op correctly and writes zeros elsewhere — that EP exists. Closes when criterion 10 closes. Criteria 4 and 5 REOPENED (Met → Partially met): negative-space criteria with no positive control; same fix — the same binary in the same lane must advertise a non-zero device count with an ICD present and claim a non-zero node count with shaders built. Owners: Trinity with Switch. Criterion 8 relabelled: parity criterion only, not a correctness criterion — two backends agreeing bit-exactly on a wrong value satisfies it completely. Criterion 7 untouched — the pattern all others are held to. Tally: four met, four partial, two not met (previously seven met, one partial, one not met).
**Why:** M0 as written could be fully met by an EP that computes zeros on every real model. No criterion required a model-level comparison. That is a defect in the criteria, not in the engineering. Reopening a met criterion is cheaper than declaring a milestone that means nothing. M0 is further from met than it was yesterday, and none of that movement is a regression in the code.

---

### 2026-07-30T05:48:29-07:00: Metric of record gated on `model_output_equivalence`; M0 sequencing — criterion 10 before CI tail [morpheus-metric-correctness-gate.md]

**By:** Morpheus (Lead / EP Architect)
**What:** The metric of record (triple: `claimed_op_coverage`, `island_count`, `largest_island_flops`) is now gated on `model_output_equivalence`: `MATCH` → triple may be reported as a result; `DIVERGENT` → triple may not be reported as progress (run reports which outputs, max-abs-diff, argmax/top-k agreement); `UNMEASURED` → triple may not be reported as progress (may be reported as a claim-path diagnostic, labelled `UNMEASURED`). Four rulings: (1) `UNMEASURED` is the default, not a soft `MATCH`. (2) Verdict is per artifact at producer-at-version; never generalises. (3) Gate not a term — correctness is not commensurable with coverage so must not appear in a row that invites arithmetic. (4) Comparison is against a CPU-only run of the same session on the same artifact, not a stored golden vector. Owners: Trinity emits, Niobe carries in `PERF.md`, Mouse carries in census. Sequencing ruling: criterion 10 goes `MATCH` before the M0 CI tail (lavapipe/Windows/Linux) is worth closing; CI lanes must carry criterion 10's gate when they run, or they measure the same silence in a new location. Link is not blocked — parallel Linux lane work is correct and prerequisite for running criterion 10 on CI.
**Why:** Coverage went 0 → 161 nodes and the model went from correct via CPU fallback to wrong via GPU. A coverage number that rises while the answer becomes wrong is worse than no number — it recruits effort in the wrong direction with the full authority of a metric of record. `UNMEASURED` as the default is R7 arriving for the fifth time on the execution path.
