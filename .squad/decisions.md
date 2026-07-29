# Squad Decisions

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
