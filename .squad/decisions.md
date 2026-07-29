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

### 2026-07-28T22:28:08-07:00: Execution status disclosure — no shader has executed on any device yet

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

**By:** Morpheus, Switch, Link, Fact Checker, Justin Chu via Copilot
**What:** A device is advertised if and only if it satisfies: Vulkan ≥ 1.1 core; a compute queue; `synchronization2` (core in 1.3 **or** `VK_KHR_synchronization2`); `subgroup_size_control` (core in 1.3 **or** `VK_EXT_subgroup_size_control`); subgroup BASIC+ARITHMETIC in the COMPUTE stage; `maxComputeWorkGroupInvocations ≥ 256`; `maxComputeSharedMemorySize ≥ 16 KiB`. `VkApplicationInfo::apiVersion = min(vkEnumerateInstanceVersion(), VK_API_VERSION_1_3)`. Everything else — `shaderFloat16`, 16-bit storage, `shaderInt8`, timeline semaphores, `bufferDeviceAddress`, cooperative matrix, integer dot product — is capability-probed and gates a shader variant, never a hard requirement. Default SPIR-V target `vulkan1.1`; higher targets as separate variants.
**Why:** Switch identified the only two features that materially simplify engine design: `synchronization2` (single barrier code path) and `subgroup_size_control` (eliminates defensive workgroup-size-unknown handling in GEMM shaders). Both exist as standalone extensions on 1.1/1.2 drivers, so we get both without paying a version floor's coverage cost. Link's platform analysis puts Vulkan 1.3 at ~26% and Vulkan 1.1 at ~89% of Android Vulkan-capable devices; Android is in scope per the project charter. Fact Checker verified: the premise behind the original 1.3 proposal does not hold — llama.cpp's base shader target is `vulkan1.2` (only the NVIDIA coopmat-2 path uses `vulkan1.3`), and ExecuTorch hardcodes `VK_API_VERSION_1_1`. On every platform tested first (desktop 2022+, lavapipe, SwiftShader, MoltenVK 1.2.5+) the capability-set requirement is satisfied by devices reporting 1.3, so in practice this *is* Justin's 1.3 baseline; the difference only appears on older Android, where the EP cleanly declines instead of failing to load.
**Alternatives rejected:**
- *Hard Vulkan 1.3 floor (Justin's initial proposal):* Buys only the two features already available as extensions; costs ~63 percentage points of Android installed-base coverage and pre-1.3.0 MoltenVK, for zero engine simplification. "llama.cpp does it" is not a valid justification — llama.cpp does not require 1.3 (Fact Checker claims 1–2, both contradicted).
- *Hard Vulkan 1.2 floor (Link's recommendation):* Sane for desktop; on Android the 1.2 tier barely exists (bimodal 1.1/1.3 distribution), so it pays nearly the full Android cost of 1.3 while delivering less on desktop. Timeline semaphores — the main 1.2 addition — are unused in v0 and available as `VK_KHR_timeline_semaphore` on 1.1 when needed.
- *Bare Vulkan 1.1 floor (ExecuTorch's position):* Forces the dual barrier path Switch warned about and leaves subgroup size to driver choice, which can silently produce wrong GEMM results.
**Open:** OQ-1 — how many real 1.1/1.2 devices lack the two required extensions (Link, in progress). `DESIGN.md` §7 is final once OQ-1 lands.

---

### 2026-07-28T17:59:54-07:00: Ruthless v1 non-goals

**By:** Morpheus
**What:** Out of scope for v1: training, opset completeness, dynamic-shape fast paths (M0–M2), data-dependent output shapes, fp64, quantized ops, attention fusion, graph-level op fusion, mobile-first tuning, image/texture-backed tensors, multi-GPU/multi-queue, cooperative matrix, and all `com.microsoft` contrib ops. Full table in `DESIGN.md` §1.2.
**Why:** The MLX EP reached 184/202 ops because MLX supplied op semantics. We supply everything. A broad v1 would be broad, shallow, and wrong — wrong is worse than absent because CPU fallback is always correct. A narrow correct v1 with clean fallback is strictly more useful than a wide one with a silent numerical bug.
**Alternatives rejected:** Start with quantized matmul/attention (the MLX lesson) — those are the hardest possible ops to write from scratch; it would be a year before anything was verifiable on an unproven ABI.

---

### 2026-07-28T17:59:54-07:00: ORT CPU EP as the sole correctness oracle, with mandatory claim assertions

**By:** Morpheus
**What:** Every op test compares against ORT's own CPU EP running the same ONNX model. Every op test **must** assert the node actually ran on `VulkanExecutionProvider`. Tolerances are derived and documented per family; widening one requires Trinity's sign-off and an in-test note. Validation-layer-clean is part of "done" for any engine change. Software rasterizers are a smoke test, not a correctness claim.
**Why:** CPU fallback is always correct; a plain output comparison passes whether or not the EP ran anything — the vacuous-pass trap. Using ORT CPU as oracle means we cannot encode our own misreading of an ONNX spec into both the implementation and the expectation.
**Alternatives rejected:** numpy reference (re-derives ONNX semantics, bugs go in both); ONNX reference evaluator as primary oracle (good for conformance fuzzing, slow, not what user output is compared to).

---

### 2026-07-28T17:59:54-07:00: Op growth by family, prioritized by island-merging

**By:** Morpheus
**What:** Coverage grows in families that share a shader skeleton, descriptor layout, and test file. A new op is worth claiming when it connects two existing claimed regions or extends one at the graph's edge. Benchmarks report island count and largest fused region alongside wall time.
**Why:** One unclaimed node in the middle of a graph splits it into two islands with a device round-trip between them. Claim rate is a bad metric; fused-region compute volume is the good one. The MLX project learned this expensively.
**Alternatives rejected:** Maximize claim rate (actively harmful — can make graphs slower); claim ops in ONNX-spec order (no amortization across shaders or tests).

---

### 2026-07-28T17:59:54-07:00: Milestone M0 defined as an end-to-end vertical slice, not a layer

**By:** Morpheus
**What:** M0 = a stock ORT loads the plugin, enumerates a Vulkan device, runs a graph containing a single `Add` node on that device, and matches the ORT CPU EP within tolerance, on Windows and Linux, on a software rasterizer, in CI. Every team member ships something into it.
**Why:** The MLX Rust rewrite began exactly this way — a single-`Add` spike proved the two unknown boundaries and immediately caught a real per-session leak. A vertical slice proves the boundaries; a horizontal layer proves nothing until the last layer lands.
**Alternatives rejected:** M0 = "Vulkan engine works standalone" (defers all ABI risk); M0 = "EP loads and claims nodes, no compute" (proves the easy half).

---

### 2026-07-28T17:59:54-07:00: Divergences from the reference are enumerated, not implied

**By:** Morpheus
**What:** `DESIGN.md` §12 lists all 11 deliberate differences from `onnxruntime-mlx` with reasons. Anything not on that list is intended to match the reference. A PR that diverges without adding a row is a review rejection.
**Why:** "We'll refactor later" is a decision, not an excuse. Both need to be written down at the moment they are made, or the reference stops being a reference and the two projects drift into unrelated codebases that can no longer share lessons.
**Alternatives rejected:** Track divergences in commit messages (not discoverable at review time).

---

### 2026-07-28T17:59:54-07:00: Rust Vulkan crate: `ash` + `gpu-allocator`

**By:** Switch
**What:** Use `ash` (raw Vulkan bindings) as the Vulkan dependency, supplemented by `gpu-allocator` for suballocation.
**Why:** `ash` is a thin binding over the Vulkan C API with zero abstraction overhead. `vulkano` adds a redundant ownership abstraction that conflicts with `engine.rs`'s own abstraction layer. `wgpu` abstracts over the WebGPU model and hides push constants, per-vendor specialization constants, and pipeline cache — all required for this EP. `gpu-allocator` is the pure-Rust VMA equivalent, used in production by Bevy and wgpu-hal.
**Alternatives rejected:** `vulkano` (conflicts with engine abstraction layer); `wgpu` (hides required Vulkan primitives).

---

### 2026-07-28T17:59:54-07:00: Buffer-only tensor storage for v0

**By:** Switch
**What:** All tensors are backed by `VkBuffer` (storage buffer). Image storage (`VkImage` / `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE`) is deferred until a specific op family (e.g., convolution) demonstrates a measurable benefit.
**Why:** Target workloads are decoder-dominated with linear memory access. Image storage requires layout transitions and format probing without performance benefit for these access patterns. Barrier reasoning is simpler with buffers only.

---

### 2026-07-28T17:59:54-07:00: One command buffer per subgraph; no per-op submission

**By:** Switch
**What:** The entire fused subgraph is recorded into one `VkCommandBuffer` and submitted once via `vkQueueSubmit`. Per-op submissions are prohibited.
**Why:** `vkQueueSubmit` overhead is measured in microseconds per call on both NVIDIA and Qualcomm. Per-op submission on a 100-op subgraph would add milliseconds of CPU-side overhead. Single submission mirrors the MLX EP's single `mlx_eval` at the subgraph boundary.

---

### 2026-07-28T17:59:54-07:00: Barrier placement: per data edge, not per dispatch

**By:** Switch
**What:** A `vkCmdPipelineBarrier2` (or `vkCmdPipelineBarrier` fallback) is inserted after each dispatch, once per consumer edge of each output buffer. No global `ALL_COMMANDS → ALL_COMMANDS` barrier.
**Why:** Per-edge barriers let the driver schedule independent dispatch pairs in parallel. A global barrier serializes the entire GPU pipeline unnecessarily.

---

### 2026-07-28T17:59:54-07:00: Shader source: GLSL compiled to SPIR-V at build time; embedded in cdylib

**By:** Switch
**What:** Shaders are written in GLSL, compiled by `glslc` during `cargo build` via `build.rs`, and embedded as byte slices in the cdylib. No runtime shader compiler is present in the deployed artifact. Tank's `build.rs` must locate and invoke `glslc`, iterate `shaders/glsl/`, write SPIR-V to `OUT_DIR/spv/`, and generate `OUT_DIR/shader_modules.rs`.
**Why:** Both reference implementations (llama.cpp, ExecuTorch) use this pattern. Guarantees a self-contained plugin with deterministic SPIR-V output.

---

### 2026-07-28T17:59:54-07:00: ORT Plugin EP C API is experimental — accept and isolate the risk

**By:** Fact Checker
**What:** The ORT plugin EP system (`OrtEpFactory`, `CreateEpFactories`) is functional but the ABI stability guarantee is weak — API redesigned after 1.22, major additions at 1.23 and 1.24, Qualcomm's first production plugin EP shipped May 2026. Strategy: pin to a specific ORT version for versioned releases; invest early in an abstraction layer that isolates the FFI from the rest of the codebase so breakages are contained.
**Why:** ORT 1.22/1.23 headers confirm `@since Version 1.22`/`1.23` with experimental status. We are building on evolving infrastructure; raw FFI in Rust requires regenerating/updating unsafe bindings at each API change. Containment is cheaper than distributed updates.

---

### 2026-07-28T17:59:54-07:00: No existing Vulkan EP or ORT Rust plugin-EP bindings — we write raw FFI

**By:** Fact Checker
**What:** There is no existing Vulkan EP for ORT (feature request open, no release). The `ort` Rust crate covers built-in providers only. We write raw FFI bindings or our own bindings wrapper from scratch.
**Why:** Verified from ORT issue tracker and published crate registry. This is an opportunity (no prior art to be compatible with) and a risk (no ecosystem to draw from). The FFI abstraction strategy above applies directly.

---

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
