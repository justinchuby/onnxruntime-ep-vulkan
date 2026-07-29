# Vulkan Runtime & Shader Architecture

**Status:** Design — v0 baseline
**Date:** 2026-07-28T17:59:54-07:00
**Author:** Switch (Vulkan Compute Engineer)
**Scope:** `rust/src/engine.rs` and the Vulkan abstraction layer; does NOT cover ONNX graph
partitioning (Mouse), ORT C ABI plumbing (Tank), platform matrix (Link), or DESIGN.md (Morpheus).

---

## 0. Document Purpose

This document specifies the Vulkan runtime and shader architecture for the
`onnxruntime-ep-vulkan` plugin Execution Provider. It covers nine topics: layering contract,
device and context model, memory strategy, shader pipeline, pipeline and descriptor management,
command submission and synchronization, Rust crate choices, Vulkan version feature analysis, and
the minimum v0 surface for a first end-to-end elementwise op.

The reference implementations studied are:

- **llama.cpp `ggml/src/ggml-vulkan/`** — device/queue/allocator structure, build-time
  `vulkan-shaders-gen` SPIR-V embedding, push-constant and specialization-constant discipline,
  vendor-aware feature detection, cooperative-matrix variant paths.
  Sources: [ggml-vulkan.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp),
  [DeepWiki analysis](https://deepwiki.com/ggml-org/llama.cpp/5.3-vulkan-backend-(cross-platform)).

- **ExecuTorch `backends/vulkan/`** — `Context`/`Adapter` object model, `vTensor` with
  buffer-or-image storage modes, VMA-based memory allocator, descriptor pool per-context,
  `ComputeGraph` for dispatch scheduling, GLSL-template + yaml variant codegen (compiled offline
  with `glslc`), weight prepacking at lowering time.
  Sources: [ET-VK overview](https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html),
  [DeepWiki analysis](https://deepwiki.com/pytorch/executorch/5.2-vulkan-gpu-backend),
  [Context.h](https://github.com/embecosm/rv-europe-2026-executorch/blob/riscv-europe-2026-workshop-update-1/backends/vulkan/runtime/api/Context.h).

The MLX EP (`onnxruntime-mlx`) defines the structural template we mirror: `engine.rs` owns the
backend execution context, `registry.rs` routes ops, `ops/*` modules implement per-family
handlers, `ep.rs` does claim/fuse/compile, and `build.rs` handles code generation and
foreign-library binding.

---

## 1. Layering — The Engine Boundary

**Rule:** Vulkan objects (handles, command buffers, pipelines, memory allocations) live behind a
safe Rust wrapper inside `rust/src/engine.rs` and the sub-modules it owns. Raw Vulkan handles
**never** cross into op handler code. Op handlers in `rust/src/ops/` see only the
`DispatchContext` trait and typed buffer handles — they cannot hold a `vk::CommandBuffer` or call
`vkCmdDispatch` directly.

This is an **enforced boundary**, not a convention: the types in `engine.rs` that wrap raw
handles are not `pub` outside the engine module. Op code is structurally incapable of touching
raw Vulkan regardless of what it imports.

```
┌───────────────────────────────────────────────────────┐
│  rust/src/ep.rs + ops/*  (ONNX semantics, no Vulkan)  │
│  uses: DispatchContext trait, typed buffer handles     │
├───────────────────────────────────────────────────────┤
│  rust/src/engine.rs  (Vulkan objects, sync, dispatch)  │
│  owns: VkDevice, VkQueue, VkCommandBuffer, pipelines  │
│        allocator, staging, pipeline cache              │
│  NOT pub outside this module                           │
├───────────────────────────────────────────────────────┤
│  ash (raw Vulkan bindings, unsafe)                    │
│  gpu-allocator (safe suballocator over ash buffers)   │
└───────────────────────────────────────────────────────┘
```

The validation layers (`VK_LAYER_KHRONOS_validation`) are always enabled in debug builds.
A clean validation run is part of "done" for any engine change.

---

## 2. Device & Context

### 2.1 Instance Creation

The engine creates exactly one `VkInstance` per plugin load. Extensions requested at instance
creation:

| Extension | Purpose |
|---|---|
| `VK_KHR_get_physical_device_properties2` | Required for feature chain queries (Vulkan 1.1 core, kept explicit for compat) |
| `VK_EXT_debug_utils` | Debug labels and validation messenger; enabled in debug builds only |

No surface or presentation extensions — this is compute-only.

### 2.2 Physical Device Selection

The selection algorithm runs at EP factory time when ORT calls the factory to advertise devices,
matching the `OrtEpDevice` model (one `OrtEpDevice` per usable physical device):

1. Enumerate all physical devices.
2. Score by device type: discrete GPU (highest), integrated GPU, virtual GPU, CPU (lavapipe /
   SwiftShader, usable for GPU-less CI), other (rejected).
3. Check that the device supports the features gated at the chosen baseline (§8).
4. Expose each usable device as a separate `OrtEpDevice` so multi-GPU hosts can register more
   than one.
5. Prefer the highest-scoring device as the default when the user does not pin a device.

**Vendor detection** (following llama.cpp's approach — verified in
[ggml-vulkan.cpp lines 163–168](https://github.com/ggml-org/llama.cpp/blob/0cea3622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L163-L168)):
the `VkPhysicalDeviceProperties.vendorID` is stored on the device handle. This drives
per-vendor shader variant selection and workaround paths later; it does not affect device
selection scoring.

**UMA detection:** if `VkPhysicalDeviceMemoryProperties` shows the largest heap is both
`DEVICE_LOCAL` and `HOST_VISIBLE` (i.e., a unified-memory GPU such as Apple/MoltenVK or
integrated Intel), the device is flagged `uma = true` and the staging copy path is suppressed
(§3.3).

### 2.3 Logical Device and Queues

One logical `VkDevice` per physical device handle. The engine requests two queue families at
device creation:

| Queue | Role |
|---|---|
| **Compute queue** | All shader dispatch; `VK_QUEUE_COMPUTE_BIT`. In practice most GPUs expose a universal graphics+compute family; a dedicated async-compute family is preferred when available. |
| **Transfer queue** | Staging uploads and downloads; `VK_QUEUE_TRANSFER_BIT`. Falls back to the compute queue when no separate DMA engine is present. |

The compute queue owns its command pool (`RESET_COMMAND_BUFFER`). The transfer queue owns its
own command pool. Both pools are created once per device context and are not shared across
threads (a per-thread command pool strategy can be added later when parallel recording is
needed).

The compute queue uses one pre-allocated primary command buffer that is reset and re-recorded
for each subgraph execution. The transfer queue uses short-lived secondary command buffers for
staging operations (allocated from the pool, freed on completion).

---

## 3. Memory

### 3.1 Allocator Strategy: `gpu-allocator`

We adopt the [`gpu-allocator`](https://github.com/Traverse-Research/gpu-allocator) crate as the
suballocator. Decision rationale:

- **Pure Rust, safe API** — wraps `ash` handles; the allocation side-effects are confined to
  the engine module.
- **Active maintenance** — used by Bevy, wgpu's Vulkan backend, and other production Rust GPU
  projects; API surface is stable.
- **Feature parity with VMA** — supports TLsf (fast) and dedicated allocation strategies,
  heap budget queries, residency tracking. This is functionally equivalent to ExecuTorch's use
  of VMA.
- **Cross-platform** — works on all platforms in our matrix (Windows, Linux, Android,
  macOS/MoltenVK).
- **Avoids a C++ dependency** — linking VMA directly would pull C++ into the cdylib's build
  graph. `gpu-allocator` avoids this.

We do **not** roll a bespoke allocator for v0. The performance argument for a custom allocator
does not arise until we have profiling data showing `gpu-allocator` is a bottleneck.

### 3.2 Memory Heaps and Buffer Kinds

| Buffer kind | Heap flags | Used for |
|---|---|---|
| `DeviceLocal` | `DEVICE_LOCAL` | Tensor data live on GPU; model weights after upload |
| `HostVisible` / `HostCoherent` | `HOST_VISIBLE \| HOST_COHERENT` | Staging buffers for upload/download |
| `HostCached` | `HOST_VISIBLE \| HOST_CACHED` | Download readback where CPU reads are frequent |

On a UMA device all three collapse to a single heap and no staging is needed (§3.3).

### 3.3 Staging / Upload / Download Paths

**Discrete GPU path (UMA = false):**

```
ORT CPU tensor  →  vkMapMemory(staging)  →  vkCmdCopyBuffer  →  device-local buffer
device-local buffer  →  vkCmdCopyBuffer  →  vkMapMemory(readback)  →  ORT CPU tensor
```

Staging buffers for upload are owned by `StagingPool`, a fixed-size ring of host-visible
buffers. When the pool is exhausted, a temporary dedicated allocation is made. After the
compute queue signals completion (fence), staging allocations are returned to the pool.

**UMA path:** the ORT tensor pointer is wrapped in a `HOST_VISIBLE | DEVICE_LOCAL` buffer
using `VK_EXT_external_memory_host` (when available) or copied once into a persistent
host-visible device-local buffer. No transfer-queue submission is needed.

### 3.4 Buffer Suballocation and Alignment

All tensor allocations go through `gpu-allocator`'s `LinearAllocator` for transient
intermediates (freed at subgraph end) and its TLsf allocator for persistent tensors (weights,
KV cache). The minimum alignment is `VkPhysicalDeviceLimits.minStorageBufferOffsetAlignment`
(guaranteed ≥ 4 bytes, typically 16 or 64 bytes on real hardware).

Suballocation within a `VkBuffer` is used for small intermediates to reduce `vkAllocateMemory`
calls (the spec permits at least 4096 allocations; we target staying under 256 live allocations
at any given time).

### 3.5 Where ORT-Allocated Tensors Live

ORT inputs arrive as CPU-side pointers (the plugin EP does not register a custom device
allocator in v0). On first use, the engine uploads each unique input to a device-local buffer
keyed by the ORT tensor pointer. Outputs are written to device-local buffers during dispatch,
then downloaded to ORT-provided output pointers after the fence fires. Weight initializers that
survive across `Run` calls are uploaded once at `Compile` time and reused.

### 3.6 Buffer-Only vs. Buffer + Image Storage — v0 Recommendation

**Recommendation: buffer-only for v0.**

ExecuTorch supports both `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` and
`VK_DESCRIPTOR_TYPE_STORAGE_IMAGE` tensor backing. The image path offers sampler-based
interpolation and automatic tiling for spatially-local access patterns (convolution). llama.cpp
uses buffers exclusively.

For onnxruntime-ep-vulkan v0, the target workloads are decoder-dominated (elementwise,
normalization, matmul, attention) — all row-major linear access patterns. Image storage adds:

- `VkImage` creation + layout transition barriers (undefined → general → shader-read-only-optimal)
- Format negotiation across vendors
- Per-format capability probing (`vkGetPhysicalDeviceFormatProperties`)

None of these translate into a performance benefit for linear-access compute. Buffer storage
requires only `VkBuffer` + `vkCmdCopyBuffer` and does not need layout transitions. The
barrier reasoning is simpler: every buffer access barrier is
`COMPUTE_SHADER_READ|WRITE → COMPUTE_SHADER_READ|WRITE` with a buffer memory barrier.

**Decision:** buffer-only until profiling on a specific operator (e.g., convolution) shows
a measurable gap versus an image-backed path on a target device.

---

## 4. Shader Strategy

### 4.1 Source Language: GLSL

We write shaders in **GLSL (Vulkan dialect)** compiled to SPIR-V at build time.

Rationale:

| Option | Assessment |
|---|---|
| **GLSL** | Both reference implementations use it. Mature tooling (`glslc` from Vulkan SDK, `glslangValidator`). Subgroup ops, push constants, specialization constants all map cleanly. No runtime compiler dependency. |
| Slang | Cross-compiles to SPIR-V, HLSL, MSL. Useful for the future compute-shader-as-source-of-truth vision. Tooling is less proven and would add a non-standard build dependency. Defer until slang-to-SPIR-V output is stable enough for CI on all platforms. |
| WGSL → SPIR-V | WGSL has no first-class push constant or subgroup extension support as of 2024. Not suitable for a high-performance compute EP. |
| Hand-written SPIR-V | Correct but unmaintainable. Rejected. |

### 4.2 Build-Time SPIR-V Compilation and Embedding

Shaders live under `shaders/glsl/` in the repo. The build pipeline (coordinated with Tank, who
owns `build.rs`) is:

```
shaders/glsl/<op>_<dtype>.comp
    │
    ▼ glslc (Vulkan SDK) — run by build.rs
shaders/spv/<op>_<dtype>.spv
    │
    ▼ include_bytes! (emitted by build.rs into OUT_DIR/shader_modules.rs)
embedded in cdylib — zero runtime compiler dependency
```

**Requirements for `build.rs` (Tank's file — Switch describes, Tank implements):**

1. Locate `glslc` via `$VULKAN_SDK/bin/glslc`, or fall back to `glslc` on `$PATH`. Fail the
   build with a clear error if absent.
2. Iterate over `shaders/glsl/*.comp`, compile each to `OUT_DIR/spv/<stem>.spv` with
   `-O` (optimization) and `-fshader-stage=compute`.
3. Generate `OUT_DIR/shader_modules.rs` containing one `pub const <STEM>_SPV: &[u8] =
   include_bytes!("spv/<stem>.spv");` per shader.
4. Emit `cargo:rerun-if-changed=shaders/glsl/` so Cargo re-runs only when shaders change.
5. **No runtime `glslc` invocation.** The compiled cdylib is self-contained.

llama.cpp's `vulkan-shaders-gen` (verified: [vulkan-shaders-gen.cpp lines 33–75](https://github.com/ggml-org/llama.cpp/blob/0cea3622/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp#L33-L75)) takes a similar approach but generates C++ header literals; we embed SPIR-V bytes directly via Rust's `include_bytes!`, which is simpler and removes the C++ toolchain dependency.

### 4.3 Shader Variant Generation for dtype / Layout Combinations

Each op requires variants for supported dtypes (f32, f16). We generate variants by passing
`-D` preprocessor defines to `glslc`:

```glsl
// elementwise_add.comp
#extension GL_EXT_shader_16bit_storage : require   // guarded by variant
#ifdef DTYPE_F16
layout(set=0, binding=0) buffer In0  { float16_t data[]; };
#else
layout(set=0, binding=0) buffer In0  { float    data[]; };
#endif
```

The build script instantiates one `glslc` invocation per `(op, dtype)` pair, naming outputs
`elementwise_add_f32.spv`, `elementwise_add_f16.spv`, etc. ExecuTorch's `gen_vulkan_spv.py`
uses a yaml-driven variant table for this; we use a Rust struct in `build.rs` — same concept.

At runtime, the engine selects the variant matching the tensor dtype reported by ORT. If the
device does not support `shaderFloat16`, the f16 pipeline is never created and ops on f16
tensors fall back to f32 upcasting (tracked as a capability flag on the device handle).

### 4.4 Specialization Constants vs. Push Constants

Following both reference implementations:

**Specialization constants** — for values that are fixed at pipeline creation time and allow the
driver to specialize the SPIR-V binary:

- Local workgroup size (`local_size_x`, `local_size_y`, `local_size_z`) — tuned per device
  after querying `maxComputeWorkGroupInvocations` and `subgroupSize`.
- Feature toggles that affect the code path (e.g., whether to use subgroup operations).

These are set when calling `vkCreateComputePipelines` and are part of the pipeline cache key.
Once a specialized pipeline is in the cache, it is reused for every subsequent dispatch with
the same constants.

**Push constants** — for values that change every dispatch:

- Tensor element count (total number of elements for elementwise ops)
- Per-dimension strides (for ops that need them)
- Scalar parameters (scale, bias, axis index)

Push constant block layout is defined in GLSL as `layout(push_constant) uniform Params { ... };`.
The total size must not exceed `maxPushConstantsSize` (guaranteed ≥ 128 bytes by the spec; we
cap our push constant structs at 128 bytes).

llama.cpp (verified: [ggml-vulkan.cpp pipeline field `layout`](https://github.com/ggml-org/llama.cpp/blob/0cea3622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L167-L185)) uses exactly this split: specialization constants for the GEMM tile sizes that survive into the pipeline binary, push constants for per-dispatch matrix dimensions.

---

## 5. Pipeline & Descriptors

### 5.1 Pipeline Cache

One `VkPipelineCache` per logical device, populated at startup from a file written at previous
session teardown (path determined by an env-var override or a platform cache dir, mirroring
llama.cpp's approach). On first run the cache is empty and pipelines JIT-compile from SPIR-V.
On subsequent runs the cache provides the compiled ISA, cutting startup time.

The cache is keyed implicitly by the Vulkan spec: the driver validates `pipelineCacheUUID` and
`deviceID` before using cached data, so stale entries from a different GPU or driver version
are automatically discarded.

Pipelines are created **lazily** — the first time an op is dispatched for a given
`(shader_variant, specialization_constants)` tuple. Creation is guarded by a `Mutex<HashMap>`
so a multi-threaded ORT session does not double-create pipelines. After creation, the pipeline
lives for the lifetime of the device context.

### 5.2 Descriptor Set Layout Strategy

Each op family uses a fixed descriptor set layout declared in the GLSL source. For a
simple elementwise binary op:

```
Set 0, Binding 0: storage buffer (input A)
Set 0, Binding 1: storage buffer (input B)
Set 0, Binding 2: storage buffer (output)
```

Descriptor set layouts are created once per op family at device initialization and cached in a
`HashMap<OpFamily, VkDescriptorSetLayout>`. They are not per-pipeline because GLSL binding
slots are defined by us and kept consistent across dtype variants of the same op.

### 5.3 Descriptor Pool Management

A single `VkDescriptorPool` per device context, sized to hold the maximum number of concurrent
descriptor sets needed by one subgraph execution (empirically bounded at startup to
`max_ops_per_subgraph × max_sets_per_op`). The pool is created with
`VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` so individual sets can be returned after a
subgraph completes.

Allocation strategy: before recording a subgraph, allocate all descriptor sets at once
(`vkAllocateDescriptorSets` with the full count). After the subgraph fence fires, free all sets
in one call. This keeps the per-dispatch allocation cost to a single bulk allocation, matching
ExecuTorch's pool-per-context design (verified in the Context.h reference).

If the pool is exhausted (a transient subgraph is unexpectedly large), the engine creates a
temporary overflow pool for that execution and logs a warning.

---

## 6. Command Submission & Synchronization

### 6.1 Recording Model: One Command Buffer Per Subgraph

**Do not submit one command buffer per op.** The overhead of `vkQueueSubmit` on both NVIDIA and
Qualcomm drivers is measured in microseconds per call; submitting per-op turns a 100-op
subgraph into hundreds of round-trips.

The engine records the entire subgraph into a single primary command buffer and submits it once:

```
vkBeginCommandBuffer(cb)
  ┌─ for each op in subgraph (topological order) ─────────────────────────────┐
  │   vkCmdBindPipeline(cb, COMPUTE, pipeline)                                │
  │   vkCmdBindDescriptorSets(cb, ...)                                        │
  │   vkCmdPushConstants(cb, ...)                                             │
  │   vkCmdDispatch(cb, ...)                                                  │
  │   [barrier if successor reads this op's output — see §6.3]                │
  └───────────────────────────────────────────────────────────────────────────┘
vkEndCommandBuffer(cb)
vkQueueSubmit(compute_queue, cb, fence)
vkWaitForFences(fence)          // CPU blocks until GPU done
```

This maps the ORT "fused subgraph → one `Compute` call" contract directly onto one GPU
submission, identical in structure to the MLX EP's single `mlx_eval` at the subgraph boundary.

### 6.2 Barrier Strategy

**Every storage buffer write must be covered by a barrier before a subsequent read from the same
buffer.** There is no implicit ordering of compute dispatches in Vulkan.

Barrier placement rule: after recording `vkCmdDispatch` for op N, walk N's output tensors. For
each output tensor T that is an input to a later op M in the same subgraph, emit:

```
vkCmdPipelineBarrier2(cb,
  srcStageMask  = COMPUTE_SHADER,
  srcAccessMask = SHADER_WRITE,
  dstStageMask  = COMPUTE_SHADER,
  dstAccessMask = SHADER_READ,
  buffer = T.buffer, offset = 0, size = VK_WHOLE_SIZE
)
```

**Why this is correct:** the Vulkan spec (§7.1, "Implicit synchronization guarantees") provides
no ordering between two compute dispatches in the same command buffer unless a pipeline barrier
or event covers the memory dependency. The barrier above inserts an execution barrier
(`COMPUTE_SHADER → COMPUTE_SHADER`) and a memory barrier (`SHADER_WRITE → SHADER_READ`),
making the write from op N visible to op M.

**Why we don't coarsen to one global barrier:** a single
`vkCmdPipelineBarrier(ALL_COMMANDS → ALL_COMMANDS)` at the start of each dispatch would be
correct but would serialize the entire GPU pipeline. Per-edge barriers let the driver overlap
independent op pairs on hardware that supports async compute tiling.

Tensors that feed multiple consumers get one barrier per consumer edge. If two consumers are
independent (no data dependency between them), they can be dispatched back-to-back without a
barrier between them — only the barrier from the producer to each consumer is needed.

### 6.3 Synchronization Primitives

| Primitive | v0 usage |
|---|---|
| `VkFence` | One per subgraph execution. CPU waits on it after `vkQueueSubmit`. Reset and reused for the next execution. |
| `vkCmdPipelineBarrier2` | Inter-op memory barriers within the command buffer (via `synchronization2` when available; see §8). Falls back to `vkCmdPipelineBarrier` when `synchronization2` is absent. |
| `VkSemaphore` (binary) | Used between transfer-queue staging upload and compute-queue dispatch: the transfer queue signals a binary semaphore on staging completion; the compute queue waits on it before executing the subgraph. |
| Timeline semaphores | Not used in v0. Useful for multi-stream pipelining (overlapping prefill and decode dispatches). Deferred to post-v0 when ORT IoBinding patterns are understood. |

**Staging → compute synchronization:** the upload path runs on the transfer queue. Before
recording the compute command buffer, the engine records a transfer command buffer that copies
ORT inputs to device-local staging buffers. The transfer queue submission includes a signal
semaphore; the compute queue submission includes a wait on that semaphore with
`PIPELINE_STAGE_COMPUTE_SHADER_BIT`. This ensures all host data is resident before any shader
reads it.

---

## 7. Rust Crate Dependency Choices

### 7.1 Decision: `ash`

We use [`ash`](https://github.com/ash-rs/ash) (raw Vulkan bindings) as the sole Vulkan Rust
dependency, supplemented by `gpu-allocator` for suballocation.

**Against vulkano:**

- vulkano sits on top of `ash` and adds its own abstraction layer (object ownership model,
  lifetime tracking). That layer is redundant — we are building our own abstraction in `engine.rs`.
  Layering vulkano's abstraction beneath ours would mean fighting two ownership models.
- vulkano's API surface evolves in breaking ways. A plugin EP lives inside ORT's process and
  must remain stable; fewer transitive dependencies reduce the risk of version conflicts.
- Bindless and subgroup extension support in vulkano lags the raw API in some areas (unverified
  as of the date of this document; assessed from published issue tracker history).

**Against wgpu:**

- wgpu targets the WebGPU abstraction layer, not Vulkan directly. It hides per-vendor
  specialization constants, subgroup operations, and push constant flexibility behind a
  cross-API model that does not expose these concepts at the native level.
- wgpu does not expose `VkPipelineCache`, per-vendor workgroup tuning, or cooperative-matrix
  paths — all of which are in scope for post-v0 matmul performance.
- The wgpu dependency graph is substantial (`naga`, `wgpu-hal`, `wgpu-core`, `wgpu-types`) and
  would conflict with any application that also uses wgpu.

**For ash:**

- `ash` is a thin binding-generation over the Vulkan headers. It adds essentially zero binary
  size beyond the Vulkan symbols themselves.
- Used by vulkano, `gpu-allocator`, and `wgpu-hal` as their own internal backend — it is the
  de facto standard for low-level Rust Vulkan work.
- Full access to every Vulkan extension, including experimental and vendor extensions, without
  waiting for a mid-level wrapper to expose them.
- The `ash` crate is `unsafe` by design; our safety contract lives in `engine.rs`, not in a
  crate boundary.

**Companion crates:**

| Crate | Version pin | Role |
|---|---|---|
| `ash` | `^0.38` | Raw Vulkan bindings |
| `gpu-allocator` | `^0.27` | Suballocator (device-local, host-visible, TLsf + Linear) |
| `ash-window` | `^0.13` | Surface extensions (only needed for debug overlay, guarded behind a feature flag) |

No `winit`, no `vulkano`, no `wgpu`. The cdylib link graph must remain clean for a process that
already embeds ORT.

---

## 8. Vulkan Version Baseline — Switch's Input

Justin proposes Vulkan 1.3. This section analyses which 1.2/1.3 features actually change the
engine design and which are cosmetic. **Morpheus makes the final baseline call** informed by
Link's platform coverage analysis.

### Features that structurally change the engine

| Feature | Where it enters | Impact |
|---|---|---|
| **`synchronization2`** (VK_KHR_synchronization2, core in 1.3) | `vkCmdPipelineBarrier2`, `vkQueueSubmit2` | Replaces the coarse `VkPipelineStageFlagBits` enum with a 64-bit `VkPipelineStageFlags2`. This simplifies barrier code — e.g., `PIPELINE_STAGE_2_ALL_TRANSFER_BIT` correctly covers both copy and blit without combining six flags. If we baseline 1.3, we can write `vkCmdPipelineBarrier2` unconditionally and drop the fallback path. If we baseline 1.1, we maintain two barrier code paths. **This is the feature with the largest engine-code-simplification benefit.** |
| **`VK_EXT_subgroup_size_control`** (core in 1.3) | Pipeline create-info `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` | Allows us to require the driver to use a specific subgroup size (e.g., 32 on RDNA, 64 on Ampere) in pipeline creation rather than receiving whatever the driver chooses. This directly affects workgroup sizing correctness for GEMM shaders — an incorrectly-sized subgroup can silently produce wrong results in cooperative ops. If baseline 1.1/1.2, we must probe `VK_EXT_subgroup_size_control` as an optional extension and code defensively around absent support. Baselining 1.3 makes the guarantee unconditional. |
| **Timeline semaphores** (VK_KHR_timeline_semaphore, core in **1.2**) | `vkSignalSemaphore`, wait with value | Enables fine-grained, monotonically-increasing wait values per semaphore instead of one binary signal/wait pair. Changes how we'd build an async multi-queue pipeline (prefill + decode overlap). Not used in v0 (single-queue fence model), but if we baseline 1.2+ the timeline API is unconditionally available for post-v0 pipelining. Strictly a 1.2 contribution, not 1.3. |
| **`shaderFloat16` / `16bit_storage`** (VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage) | Shader variant selection, buffer descriptor types | Allows `float16_t` in storage buffers and shader arithmetic. Changes which dtype variants we compile and ship. **Not guaranteed by any baseline** — must be probed at runtime regardless of whether we baseline 1.1, 1.2, or 1.3. The baseline bump does not give this for free. |
| **`bufferDeviceAddress`** (core in **1.2**) | Push constant payload (64-bit pointer) | Enables passing raw `VkDeviceAddress` values in push constants, allowing bindless buffer indexing without descriptor sets. Enables richer indirection (e.g., chained buffer access for MoE routing). **MoltenVK support is partial and restricted** (verified via platform coverage research). Do not rely on this feature in v0; probe at runtime and use only when fully supported. |

### Features that are cosmetic for the engine

| Feature | Why cosmetic |
|---|---|
| `VK_KHR_cooperative_matrix` | Changes matmul shader strategy dramatically — but it is not guaranteed by 1.3 baseline and must be capability-probed regardless. A 1.3 baseline does not give cooperative matrix; it remains an optional extension even on 1.3 devices. |
| `dynamicRendering` | Render-pass-related; irrelevant for compute-only. |
| `inlineUniformBlock` | Useful for small constant data embedded in descriptor sets; reduces one `vkUpdateDescriptorSets` call. Minor ergonomics improvement, not a structural change. |
| `privateData` | Driver-private per-object data slots; developer tooling convenience only. |
| `extendedDynamicState` | Graphics pipeline state; irrelevant for compute. |

### Switch's Recommendation

The **only two features that materially simplify the engine design** if guaranteed by baseline
are `synchronization2` and `VK_EXT_subgroup_size_control`. Both are core in 1.3.

Platform cost: Android requires API 33+ (Android 13) for guaranteed Vulkan 1.3. Android 12
(API 31) guarantees only Vulkan 1.1. MoltenVK supports Vulkan 1.3 since v1.2.5 with growing
(but not complete) coverage. (Link is analyzing the full coverage matrix independently.)

**If the team can accept Android 13+ as the minimum Android target:** baseline 1.3, eliminate
the dual barrier code path, and get unconditional subgroup size control. The engine is simpler
and the correctness argument for barriers is cleaner.

**If Android 12 or broad MoltenVK compatibility is required:** baseline 1.1 or 1.2. Probe
`synchronization2` and `subgroup_size_control` as optional extensions, implement the fallback
barrier path, and gate subgroup-size-aware GEMM shaders behind the capability flag.

Either way, `shaderFloat16`, `bufferDeviceAddress`, and `VK_KHR_cooperative_matrix` must be
capability-probed at runtime — the baseline version does not change that requirement.

---

## 9. v0 Scope — Minimum Surface for One Elementwise Op End-to-End

The minimum engine surface to run a single elementwise op (e.g., ONNX `Add` on two f32
tensors) from ORT `Compute` to output:

### 9.1 Engine components required

| Component | v0 implementation |
|---|---|
| `VulkanContext` struct | Holds `VkInstance`, `VkPhysicalDevice`, `VkDevice`, one compute `VkQueue`, `VkCommandPool`, `VkPipelineCache` (file-backed), descriptor pool, `gpu-allocator` handle |
| `DeviceBuffer` | Wraps a `gpu-allocator` allocation + `VkBuffer` handle; typed by memory kind (device-local / host-visible) |
| `StagingPool` | Ring of N host-visible `VkBuffer`s (N=4 for v0) for upload/download |
| `PipelineRegistry` | `HashMap<(ShaderVariant, SpecConstants), VkPipeline>` with lazy creation |
| `DescriptorSetLayout` map | One `VkDescriptorSetLayout` per op family (one for v0: the binary elementwise layout) |
| `SubgraphExecutor` | Records one `VkCommandBuffer` for the full subgraph; emits barriers; calls `vkQueueSubmit` + `vkWaitForFences` |

### 9.2 Shader required

`shaders/glsl/elementwise_add_f32.comp` — reads two storage buffers, writes one, dispatches
in 1D workgroups of 256 threads. Parameterized by element count via push constant.

### 9.3 Data flow for one `Add` op

```
ORT inputs (CPU ptr)
  → upload via staging (vkCmdCopyBuffer, transfer queue)
  → binary semaphore signal
compute queue waits on semaphore
  → vkCmdBindPipeline(elementwise_add_f32)
  → vkCmdBindDescriptorSets(A_buf, B_buf, out_buf)
  → vkCmdPushConstants({n_elements})
  → vkCmdDispatch(ceil(n/256), 1, 1)
  → (no barrier needed — this is the only op and there is no subsequent read in-graph)
vkQueueSubmit → vkWaitForFences
  → download output (vkCmdCopyBuffer, transfer queue or compute queue)
ORT output (CPU ptr written)
```

### 9.4 What is explicitly out of scope for v0

- Multi-op subgraph with inter-op barriers
- f16 dtype variant
- Cooperative matrix (matmul)
- Timeline semaphores
- Async compute (second queue for overlap)
- Weight prepacking
- Pipeline cache persistence (file I/O can be added without touching the core path)
- Per-vendor workgroup tuning

---

## 10. Cross-Cutting Concerns Not Covered Here

| Topic | Owner |
|---|---|
| ORT plugin C ABI, `CreateEpFactories`, `OrtEpFactory` vtable | Tank |
| ONNX op semantics, claim predicates, registry | Mouse |
| Graph partitioning and fused-subgraph construction | Mouse |
| Platform coverage matrix, Android API level, MoltenVK version policy | Link |
| Vulkan baseline version final decision | Morpheus |
| CI matrix, test harness, conformance runner | Trinity |
| Benchmark methodology | Niobe |

---

## 11. References

| Source | Link |
|---|---|
| llama.cpp ggml-vulkan.cpp | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp |
| llama.cpp vulkan-shaders-gen | https://github.com/ggml-org/llama.cpp/blob/0cea3622/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp |
| llama.cpp Vulkan DeepWiki | https://deepwiki.com/ggml-org/llama.cpp/5.3-vulkan-backend-(cross-platform) |
| ExecuTorch Vulkan overview | https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html |
| ExecuTorch Vulkan DeepWiki | https://deepwiki.com/pytorch/executorch/5.2-vulkan-gpu-backend |
| ExecuTorch Context.h | https://github.com/embecosm/rv-europe-2026-executorch/blob/riscv-europe-2026-workshop-update-1/backends/vulkan/runtime/api/Context.h |
| ash crate | https://github.com/ash-rs/ash |
| gpu-allocator crate | https://github.com/Traverse-Research/gpu-allocator |
| Vulkan 1.3 release notes | https://www.khronos.org/blog/vulkan-1-3-released |
| MoltenVK feature coverage | https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/features.md |
| Vulkan Memory Allocator (VMA) | https://gpuopen.com/vulkan-memory-allocator/ |
| onnxruntime-mlx reference EP | C:\Users\justinchu\dev\onnxruntime-mlx |
