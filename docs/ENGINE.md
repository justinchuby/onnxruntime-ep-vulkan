# Vulkan Runtime & Shader Architecture

**Status:** In progress — first real dispatch verified on both local devices (Intel + NVIDIA)
**Date:** 2026-07-28T17:59:54-07:00 (updated 2026-07-29T08:13:58-07:00)
**Author:** Switch (Vulkan Compute Engineer)
**Scope:** `rust/src/engine.rs` and the Vulkan abstraction layer; does NOT cover ONNX graph
partitioning (Mouse), ORT C ABI plumbing (Tank), platform matrix (Link), or DESIGN.md (Morpheus).

## Implementation Status (2026-07-29T08:13:58-07:00)

| Module | Status | Notes |
|---|---|---|
| `vk/instance.rs` | ✅ Real | Instance creation, physical device enumeration, §7.2 gate (R1–R4, R6), `assess_gate` verbose output, `ONNXRUNTIME_EP_VULKAN_DEVICE` selector, 19 unit tests |
| `vk/caps.rs` | ✅ Real | Capability probe (push_next chain fixed D-S12-01), MoltenVK trap, `subgroup_basic_in_compute`, `test_caps()` helper, `DeviceFeatureChain` |
| `vk/barrier.rs` | ✅ Real | Dual-backend (`Core`/`Khr`), mapping tables, probe write, 20 unit tests |
| `vk/device.rs` | ✅ Real | Logical device creation, queue retrieval, `Barriers::select`, `DeviceFeatureChain::apply` |
| `vk/alloc.rs` | ✅ Real | `gpu-allocator` backed, 4 memory classes, staging helpers, 6 unit tests |
| `vk/cmd.rs` | ✅ Real | Command pool, `CommandRecorder`, `submit_and_wait` |
| `vk/pipeline.rs` | ✅ Real | Pipeline cache, `DispatchDescriptorPool`, spec constants, push constants, 8 unit tests |
| `vk/dispatch_integration.rs` | ✅ **Verified** | **1024-element f32 `Add` dispatched on BOTH Intel Iris Xe (1.4.309) and NVIDIA RTX 4060 (1.4.325). Zero validation layer errors on both devices.** |
| `engine.rs` seams | ✅ Stubbed | Vocabulary types, `Plan`, `CompileContext`, `DispatchContext`; real dispatch pending |
| Shader-less guard | ✅ Real | §7.8 condition 3: `probe_devices()` + `GetCapability` return zero/claim-nothing when `SHADER_MODULES` empty. Tests assert correctly in both build modes. |
| Loader diagnostics | ✅ Real | `loader_state_lines()` always emitted on `vkCreateInstance` failure; INFO-gated pre-creation on `ONNXRUNTIME_EP_VULKAN_VERBOSE=1`. `apiVersion` capped to loader version. |
| `epctl --probe-loader` | ✅ Real | Per-device per-criterion gate assessment via `assess_gate`; shows measured values, PASS/FAIL per criterion, and `ONNXRUNTIME_EP_VULKAN_DEVICE` selection result. |
| Session lifecycle | 🔲 Pending | `VulkanEp` in `ep.rs` must hold `Instance` + `Device` across Compile/Compute |
| Real `DispatchContext` | 🔲 Pending | Concrete implementor over `VkCommandBuffer` using `cmd.rs` + `pipeline.rs` |
| `alloc` integration | 🔲 Pending | Tank's `BufferView` handle table ↔ `GpuBuffer` side-table |
| Prepack hook (real) | 🔲 Pending | Seam 1 vocab is real; actual staging upload behind it pending |
| KV-cache aliasing | 🔲 Pending | Seam 2 default impl correct for stubs; real aliasing pending |

Build status: `cargo ci` green (rustfmt + clippy + build + test). Test count: **258** lib (+ 6 dump-capabilities + 26 layering + 7 portability).

**Morpheus — §9.1.2 update required:** The sentence "no shader has ever executed on any device" is no longer accurate as of session 11. `add_f32_dispatches_end_to_end` completed with 1024 f32 elements verified on NVIDIA GeForce RTX 4060 Laptop GPU (Vulkan 1.4.325, Discrete), with `VK_LAYER_KHRONOS_validation` reporting zero errors. As of session 12, the same dispatch verified on Intel Iris Xe (Vulkan 1.4.309, Integrated) as well — zero validation errors on both devices.

---

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

**Explicit device selection — `ONNXRUNTIME_EP_VULKAN_DEVICE`:** set this environment variable to
either an integer index (0-based, within the ordered list of capable devices after gate scoring)
or a case-insensitive name substring to pin to a specific device. Example:
`ONNXRUNTIME_EP_VULKAN_DEVICE=Intel` or `ONNXRUNTIME_EP_VULKAN_DEVICE=0`. The selected device
and reason are reported by `epctl --probe-loader`. This allows deterministic per-device testing
without recompiling.

**Intel as conformance oracle:** Intel's Vulkan driver is stricter than NVIDIA's about undefined
behaviour, missing synchronisation, and extension interactions. When both devices are present,
if a dispatch is correct on NVIDIA and raises a validation error on Intel, Intel's finding is
the actionable one. Encode results per device rather than aggregating — a pass on NVIDIA alone
does not constitute a pass. This is not special-casing Intel: it is the correct reading of the
spec conformance asymmetry between the two driver families.

**Vendor detection** (following llama.cpp's approach — verified in
[ggml-vulkan.cpp lines 163–168](https://github.com/ggml-org/llama.cpp/blob/0cea3622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L163-L168)):
the `VkPhysicalDeviceProperties.vendorID` is stored on the device handle. This drives
per-vendor shader variant selection and workaround paths later; it does not affect device
selection scoring.

**UMA detection:** `is_uma = true` when the largest `DEVICE_LOCAL` heap also has a
`HOST_VISIBLE` memory type. This covers two physically distinct configurations:
- **True integrated UMA** (e.g. Intel Iris Xe): device-local memory is the same DRAM as system
  RAM; no VRAM heap exists independently.
- **Discrete GPU with ReBAR** (e.g. NVIDIA RTX 4060 Laptop with Resizable BAR enabled): the full
  VRAM heap is mapped `HOST_VISIBLE` via PCIe BAR, allowing direct CPU writes to VRAM.

Both cases allow bypassing the staging copy path (§3.3) because the GPU can read from memory
that the CPU can also write. The distinction (integrated vs ReBAR discrete) is tracked separately
via `VkPhysicalDeviceType` and reported in `epctl --probe-loader`. **The correct predicate for
UMA is: every memory heap is `DEVICE_LOCAL`.** A discrete GPU with ReBAR has a system-RAM heap
without `DEVICE_LOCAL` even though its VRAM heap is also `HOST_VISIBLE`; ReBAR does not make
a discrete GPU a UMA device. Tile sizes and workgroup shapes derived from reported limits (not
hardcoded constants) must fit within the device's `maxComputeSharedMemorySize` — Intel Iris Xe
reports 32 KiB, NVIDIA RTX 4060 Laptop reports 48 KiB; any constant tuned on the NVIDIA device
may silently be too large for Intel.

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

### 3.2 Memory Classes and `gpu-allocator` Location Hints

Four memory classes are in use (matching `alloc.rs::MemClass`):

| Class | `MemoryLocation` hint | Flags requested | Used for |
|---|---|---|---|
| `DeviceLocal` | `GpuOnly` | `DEVICE_LOCAL` | Live tensor data, KV-cache |
| `Upload` | `CpuToGpu` | `HOST_VISIBLE \| HOST_COHERENT` | CPU→GPU staging (write-combining preferred) |
| `Download` | `GpuToCpu` | `HOST_VISIBLE \| HOST_COHERENT` | GPU→CPU readback (cached preferred) |
| `PackedWeights` | `GpuOnly` | `DEVICE_LOCAL` | Compile-time prepacked weights; never freed mid-run |

`gpu-allocator` maps these hints to the best available `VkMemoryType`. On discrete GPUs the
`GpuOnly` and `CpuToGpu` types are on different physical heaps (VRAM vs PCIe-mapped BAR). On
UMA devices (Intel Iris Xe, Adreno, Mali) and on discrete GPUs with ReBAR all four types may
reside on the same physical heap — `gpu-allocator` still picks the right type within that heap
(write-combining vs cached when both are exposed).

**The `Upload`/`Download` distinction matters on every memory architecture.** `CpuToGpu` signals
that the CPU writes and the GPU reads; `GpuToCpu` signals the reverse. On discrete hardware
these may select different BAR regions. Using the wrong hint is not a crash but degrades
throughput on non-UMA devices, so they are kept separate.

### 3.3 Staging / Upload / Download Paths

**v0 always stages, even on UMA devices.** Correctness is proven first; the staging bypass is
a future M1+ optimisation.

```
CPU data  →  vkMapMemory(Upload staging)  →  vkCmdCopyBuffer  →  DeviceLocal buffer
DeviceLocal buffer  →  vkCmdCopyBuffer  →  vkMapMemory(Download staging)  →  CPU data
```

On UMA devices (Intel Iris Xe, Adreno, Mali) the `vkCmdCopyBuffer` is an in-heap copy — the
GPU reads from a different `VkBuffer` that maps to the same physical DRAM. It is correct and
produces no data hazard. The Intel Iris Xe dispatch test confirms this path works without
validation errors.

**Future UMA bypass (not in v0):** when `caps.is_uma == true`, the `DeviceLocal` buffer could
be allocated as `HOST_VISIBLE | DEVICE_LOCAL` and the CPU could write to it directly via
`vkMapMemory`, skipping the `vkCmdCopyBuffer`. This requires that the shader's `TRANSFER_DST`
usage is changed to just `STORAGE_BUFFER` and that the barrier pipeline changes from
`TRANSFER_WRITE → SHADER_READ` to `HOST_WRITE → SHADER_READ`. The barrier abstraction in
`barrier.rs` already has `Access::HostWrite` for this purpose. The feature gate for this path
is `caps.is_uma`, not the device type — a discrete GPU with ReBAR has `is_uma=true` and
should also benefit.

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

### 3.5.1 Weight Prepacking (Block-Quantized Kernels)

Mouse's quantized kernels (`MatMulNBits`, `BlockDequant`, `GroupQueryAttention`, `QMoE`,
`LinearAttention`) are on the critical path and require weight prepacking at subgraph-compile
time (`Plan::compile`). This changes the memory model in the following ways:

**New buffer class: `PackedWeights`**

Prepacked weight buffers are device-local, written once (at `Compile` time), and read-only for
the lifetime of the `Plan`. They are distinct from activation buffers in three ways:
- *Lifetime:* allocated in `compile()`, freed in `Drop` of the `Plan` — not at subgraph end.
- *Access pattern:* read-only from shader; never a barrier destination after initial upload.
- *Layout:* block-quantized weights require interleaved or separated scale/zero-point layout
  matching the GLSL kernel's expected memory layout; the packing kernel must run once at
  `Compile` time, not at every `Compute` call.

The allocator design needs a `PackedWeights` memory class separate from `TempBuffer`. Both use
`gpu-allocator`'s TLSF strategy, but the lifetime difference matters for eviction heuristics
and for the `#[must_use]` invariants we will enforce on deallocation.

**Staging for bulk uploads**

Model weights can be multi-GB. The existing `StagingPool` (a fixed ring of buffers sized for
activation upload at inference time) is not suitable for one-shot GiB-sized transfers. The
`Compile`-time path needs a separate "bulk staging" mechanism: either a temporary dedicated
host-visible allocation (`gpu-allocator`'s `LinearAllocator` is ideal here — allocate, upload,
copy, free), or a configurable max-upload-size option in the staging pool.

This does not change the current allocator stub's interface — it adds a new entry point.

**Impact on barrier design:** none. A `vkCmdCopyBuffer` from staging to the packed-weights
buffer at compile time is `TransferWrite`. The first shader read at `Compute` time is
`ShaderRead`. The existing `Barriers::buffer_deps` with
`src: Access::TransferWrite, dst: Access::ShaderRead` is the correct barrier for the
staging-to-shader transition. The `PackedWeights` buffers then have no further src-side
barriers (they are never written again).

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

Shaders live under `shaders/glsl/` in the repo. The build pipeline (owned by Tank, whose
`build.rs` now consumes both sources and the variant table) is:

```
shaders/glsl/<source>.comp             src/ops/shader_variants.txt
       │                                        │
       │           ┌────────────────────────────┘
       ▼           ▼
  build.rs  (parses variant table, runs glslc with -D defines per row)
       │
       ▼  glslc → OUT_DIR/spv/<stem>.spv
       │
       ▼  include_bytes! (emitted into OUT_DIR/shader_modules.rs)
embedded in cdylib — zero runtime compiler dependency
```

**Two compilation paths in `build.rs` (Seam 3, implemented):**

1. **Direct sources:** every `shaders/glsl/*.comp` file is compiled once with no `-D` flags.
   Used for hand-written XL kernels (`matmul_nbits.comp`, etc.) where the variant logic is
   internal to the shader.

2. **Variant rows from `src/ops/shader_variants.txt`:** tab-separated `<stem>\t<source>\t<defines>`.
   Each row produces one SPIR-V module named `<stem>.spv`. `build.rs` compiles the shared
   template (`<source>`) with the per-row `-D` flags, generating all dtype/layout variants from
   one source file. `cargo:rerun-if-changed` tracks the variant table so Cargo re-runs only when
   it changes.

**No runtime `glslc` invocation.** The compiled cdylib is self-contained; the build machine
needs the Vulkan SDK (or `ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` for lint-only lanes).

**Current build-machine requirement vs. Morpheus's OQ-4 provisional decision (open):**

Morpheus's provisional OQ-4 decision was "build-time glslc with a checked-in SPIR-V fallback
so that a plain `cargo build` works". The current implementation does NOT have checked-in
SPIR-V; a machine without glslc on PATH sees:
```
glslc not found but 168 shader(s) exist. Set ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1.
```

Switch's position for Morpheus's ruling:

- **Checked-in SPIR-V is a real cost:** 168 SPIR-V modules × ≈8–20 KB each ≈ 1–3 MiB of
  binary in the repo. Every shader edit forces a re-checked-in blob. Binary diffs are
  unreadable and staleness is invisible (a `git clone` that misses the latest blob will build
  silently but run old shaders).
- **The escape hatch plus clear documentation is the better engineering answer:** Require the
  Vulkan SDK on any machine that touches shader source, just as any shader-writing project does.
  Document it clearly in `README.md` and `rust/README.md`. The `ALLOW_MISSING_GLSLC=1` flag
  satisfies the "doc/lint-only lane" case without a checked-in binary burden.
- **Precedent:** Dawn, llama.cpp, ExecuTorch all require the Vulkan SDK on build machines that
  compile shaders. None ship checked-in SPIR-V as a universal fallback.

**OQ-4 RESOLVED (2026-07-29T01:51:01-07:00 — `DESIGN.md §7.8`)**: Morpheus ruled in favour of
the hard SDK dependency, changing the provisional decision rather than the code. The escape hatch
stays. The ruling comes with five binding conditions, of which condition 3 is Switch's:

> **A shader-less artifact must advertise zero devices and claim nothing** (`DESIGN.md §7.8` cond. 3).

Implementation (`engine.rs`, `ep.rs`):
- `shaders::has_any()` — returns `SHADER_MODULES.is_empty()`. Used by both guards below.
- `probe_devices()` — if `!shaders::has_any()`, logs a `WARN` with reason `"built without shaders"` and returns `vec![]` immediately. ORT sees no devices → no session is created with this EP.
- `get_capability_impl()` — belt-and-suspenders: if `!shaders::has_any()`, logs `[built-without-shaders]` at DEBUG and returns `null_mut()` (zero claims). A shader-less artefact can never claim a node, dispatch to a pipeline that doesn't exist, and fail silently.

Both guards produce the same outcome: zero devices, zero claims, clean CPU fallback. This is the same property as Trinity's no-ICD assertion, one level up in the build.

This decision is tracked at `.squad/decisions/inbox/switch-engine-seams.md` (D-S-I-04, now resolved).

llama.cpp's `vulkan-shaders-gen` (verified: [vulkan-shaders-gen.cpp lines 33–75](https://github.com/ggml-org/llama.cpp/blob/0cea3622/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp#L33-L75)) takes a similar approach but generates C++ header literals; we embed SPIR-V bytes directly via Rust's `include_bytes!`, which is simpler and removes the C++ toolchain dependency.

### 4.3 Shader Variant Generation for dtype / Layout Combinations

Mouse's `src/ops/shader_variants.txt` (69 rows, 168 modules, already committed) is the single
source of truth for variant generation. Format per row:

```
<stem>TAB<glsl_source>TAB<comma-separated -D defines>
```

Example:
```
ew_binary_add_f32    ew_binary.comp    EW_OP=OP_ADD,SCALAR_T=float,DTYPE_F32
ew_binary_add_f16    ew_binary.comp    EW_OP=OP_ADD,SCALAR_T=float16_t,DTYPE_F16
```

`build.rs` compiles each row with `-D<define>` for each comma-separated entry. The template
shader guards code paths:

```glsl
// ew_binary.comp
#ifdef DTYPE_F16
layout(set=0, binding=0) buffer In0 { float16_t data[]; };
#else
layout(set=0, binding=0) buffer In0 { float data[]; };
#endif
```

At runtime, the engine selects the variant matching the tensor dtype reported by ORT. If the
device does not support `shaderFloat16`, the f16 pipeline is never created and ops on f16
tensors fall back to f32 upcasting (tracked as a capability flag on the device handle).

XL kernels (`matmul_nbits.comp`, etc.) are direct sources compiled without a variant-table row:
their variant branching is internal to the GLSL file, and they carry their own tile-config
specialization-constant paths.

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
each output tensor T that is an input to a later op M in the same subgraph, call
`Barriers::buffer_deps` with one `BufferDep` per (T, M) consumer edge:

```rust
// After dispatching op N, emit one barrier per consumer edge.
let deps: Vec<BufferDep> = op_n.outputs
    .iter()
    .flat_map(|tensor| tensor.consumer_ops(subgraph))
    .map(|consumer| BufferDep {
        buffer:  tensor.buffer,
        offset:  0,
        size:    vk::WHOLE_SIZE,
        src:     Access::ShaderWrite,
        dst:     Access::ShaderRead,
    })
    .collect();

// SAFETY: cb is in recording state; all buffers are live for the command buffer lifetime.
if !deps.is_empty() {
    unsafe { barriers.buffer_deps(cb, &deps) };
}
```

`barriers` is the `Barriers` instance stored on the device, selected once at init from
`Capabilities` (see §8 and `rust/src/vk/barrier.rs`). Call sites never branch on sync2 — they
call `buffer_deps` and the backend handles the dispatch.

**Why this is correct:** the Vulkan spec (§7.1) provides no ordering between two compute
dispatches in the same command buffer unless a pipeline barrier or event covers the memory
dependency. `buffer_deps` emits exactly one barrier command: either `vkCmdPipelineBarrier2`
(sync2 path) or `vkCmdPipelineBarrier` (legacy path), both carrying N `VkBufferMemoryBarrier(2)`
structs for the N consumer edges. The access mask pair (`SHADER_WRITE` → `SHADER_READ`) makes
the write from op N visible to op M.

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
| `Barriers` (see `DESIGN.md` §7.5 and `rust/src/vk/barrier.rs`) | Inter-op memory barriers within the command buffer. Selected once at device init: `Barriers::Sync2` uses `vkCmdPipelineBarrier2` when `synchronization2` is available; `Barriers::Legacy` uses `vkCmdPipelineBarrier` otherwise. No call site may branch on `capabilities.synchronization2` — they call `barriers.buffer_deps()` unconditionally. |
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

> **Note (2026-07-28T19:16:08-07:00):** Morpheus has frozen the capability set in `DESIGN.md` §7
> after Link measured real platform data. The analysis below is updated to reflect the frozen
> decision. The original recommendation (1.3 baseline requiring `synchronization2`) is superseded.

### 8.0 The Frozen Capability Principle

**Capability shortfalls degrade op coverage, not device availability** (`DESIGN.md` §7.0).

The device gate is **five** hard requirements (R1–R4, R6) — no optional extensions:

1. Vulkan ≥ 1.1
2. At least one compute queue
3. `maxComputeWorkGroupInvocations ≥ 256`
4. `maxComputeSharedMemorySize ≥ 16384`
5. At least one DEVICE_LOCAL memory heap and one HOST_VISIBLE memory type

**R5 (subgroup BASIC in COMPUTE stage) was removed from the gate in session 10.**

Lavapipe/llvmpipe — the only device available on CI — reported `supportedStages = 0` for
subgroup operations on the Mesa version present in Ubuntu 22.04. The old R5 check rejected
it, violating Morpheus's §7.0 governing principle verbatim: *"capability shortfalls degrade
op coverage, not device availability."* Subgroup support is now recorded in
`Capabilities::subgroup_basic_in_compute`; ops that use subgroup intrinsics check that field
in their claim predicates instead of requiring it at device admission.

`passes_gate` is a thin wrapper over `assess_gate`, which evaluates **every** criterion and
records its measured value — no early exit. This lets `epctl --probe-loader` show the exact
criterion and value that rejected a device, and lets `enumerate_capable_devices` log the full
assessment at DEBUG when a device fails the gate.

No extension is required. Capability shortfalls (no sync2, no fp16, no subgroup size control,
no subgroup BASIC in compute) mean fewer ops are available, not that the device is rejected.

### 8.1 Why `synchronization2` Is Not Required

Link measured `VK_KHR_synchronization2` availability: **68.57% of Android devices** have it
(vulkan.gpuinfo.org, 2026-07-28). The gap — 31.43% — is concentrated in Adreno 5xx/6xx with
frozen pre-2021 OEM blobs and Mali Bifrost on MediaTek. These are real devices carrying real
inference workloads.

The Khronos `VK_LAYER_KHRONOS_synchronization2` emulation layer was evaluated and **rejected**:
the AOSP Vulkan loader only enumerates validation layers from the host application's
`nativeLibraryDir`. A plugin inside someone else's APK process cannot inject layers. This path
is closed.

The established precedent among production Vulkan compute runtimes: wgpu, Dawn, and Godot all
use `vkCmdPipelineBarrier` (legacy) exclusively on Android. We follow the same approach.

**Engine resolution: dual-backend behind a single seam.**

`rust/src/vk/barrier.rs` implements two backends (`Sync2Backend`, `LegacyBackend`) behind the
`Barriers` enum. `Barriers::select` is called **once** in `Device::new`:

```rust
let barriers = unsafe {
    Barriers::select(&caps, &instance, &device, opts.force_legacy_barriers)
};
// Stored on the device. No other code reads caps.synchronization2.
```

No call site anywhere else may branch on `caps.synchronization2`. The selection is transparent
to all dispatch code; both backends produce identical submission shapes for parity testing.

`ep.force_legacy_barriers` (default false) forces the legacy path on sync2-capable hardware so
Trinity's CI harness can exercise both paths and assert identical results.

### 8.2 `subgroup_size_control` Is a Query, Not a Requirement

`VK_EXT_subgroup_size_control` (core in 1.3) is probed but never required.

**Critical MoltenVK quirk**: MoltenVK 1.3.0 reports the extension present (via 1.3 core) but
`subgroupSizeControl = VK_FALSE`. Metal cannot set SIMD-group width per pipeline. "Extension
present" must never be read as "width controllable".

The `Capabilities` struct (`rust/src/vk/caps.rs`) distinguishes:
- `subgroup_size_range: Option<SubgroupSizeRange>` — whether the range is queryable (has values)
- `can_require_subgroup_size: bool` — whether `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`
  is actually honoured (the `subgroupSizeControl` feature flag is `VK_TRUE`)

**Shader variant selection rule:** a shader whose correctness depends on a specific subgroup
width may only be selected when the width is **known exactly** — either:
- `subgroup_size_range.min == subgroup_size_range.max`, OR
- `can_require_subgroup_size == true` and the required-size pipeline creation flag was used

Otherwise the portable shared-memory variant runs. This rule prevents silent wrong-result bugs
on hardware where the subgroup size is implementation-defined.

### 8.3 Feature-by-Feature Analysis (updated)

| Feature | Changed verdict |
|---|---|
| **`synchronization2`** | **Probed only.** 31.43% Android gap. Dual-backend abstraction in `barrier.rs` resolves the code-simplicity cost at zero availability cost. |
| **`VK_EXT_subgroup_size_control`** | **Probed only.** MoltenVK quirk: present ≠ controllable. See §8.2. |
| **Timeline semaphores** (1.2 core) | Not used in v0. Probed for post-v0 pipelining. |
| **`shaderFloat16` / `16bit_storage`** | Probed only. Not guaranteed by any baseline. F16 op variants only loaded when both flags are true. |
| **`bufferDeviceAddress`** (1.2 core) | Not used in v0. MoltenVK support is partial. Probe at runtime; bindless deferred to post-v0. |
| **`VK_KHR_cooperative_matrix`** | Not guaranteed by 1.3; must probe regardless. Deferred to GEMM milestone. |

### 8.4 Switch's Updated Recommendation

The correct baseline is **Vulkan 1.1 with the six hard requirements above**. The
`synchronization2` and `subgroup_size_control` simplifications that originally motivated a 1.3
baseline are now obtained by other means:

- Synchronization: dual-backend `Barriers` abstraction costs one `match` at init time and zero
  code at every call site.
- Subgroup size: the `can_require_subgroup_size` flag + the portable fallback variant provide
  the same safety property without the platform exclusion cost.

**Morpheus makes the final call** after reviewing Link's full platform analysis.

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
in 1D workgroups. Workgroup size is `EW_LOCAL_SIZE` (256), selected as the minimum guaranteed
by `maxComputeWorkGroupInvocations` and matching every vendor's preferred subgroup size. It
reaches the shader as spec constant ID 0 so a per-device tuner can override it later without
recompiling GLSL.

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

### 9.5 Engine Seams for XL Kernels (post-v0, implemented)

These four seams are vocabulary additions to `rust/src/engine.rs` and `rust/src/registry.rs`.
They do not require a running Vulkan device and are therefore implemented now, before the
shaders exist, so Mouse can build XL-kernel claim predicates and translate handlers against
a stable contract.

**Seam 1 — Prepack hook (`OP_COVERAGE.md` §8.2.1)**

Weight prepacking for block-quantized kernels (`MatMulNBits`, `GroupQueryAttention`,
`GatherBlockQuantized`). Contract: `CompileContext::request_prepack(PrepackRequest)` emits a
pack request at Compile time; the engine calls `req.pack_fn(PackInput)` once per unique
`PackKey`, uploads the result, and stores `PrepackResult` in `Plan::prepacked`.
At Compute time, `DispatchContext::resolve_prepacked(&PackKey)` returns the handles.

Key invariants (P1–P6):
- Runs **after** device selection and tile-config choice, because the packed layout depends on
  `TileConfig` (`tile_n`, `tile_m`, `block_size`).
- Cached on `PackKey(initializer, config, variant)` — one upload per unique key regardless of
  how many ops reference the weight.
- Scales and zero-points are **separate `Vec<u8>` outputs** and separate `BufferView` bindings.
- The ONNX-layout weight is droppable after packing (engine clears the staging buffer).
- `pack_fn: fn(PackInput) -> PackOutput` is pure (no Vulkan); lives in `ops::quant::prepack`.
- The claim predicate does not depend on the prepack path (claiming and prepacking are decoupled).

`compile_hook_for(&NodeDesc) -> Option<CompileHook>` in `registry.rs` is the stub dispatch
point; Mouse fills in the per-op hooks.

**Seam 2 — KV-cache aliasing (`GroupQueryAttention`)**

`DispatchContext::bind_aliased_output(&TensorRef, &OutRef)` declares that `present` should
write into the `past` allocation. Default impl returns `resolve(input)`, which is correct
for test stubs and trivially correct when KV cache is not needed.

**Coordination note for Tank:** the handle-based allocator must not trigger its generation-stamped
quarantine-on-free between the aliased input read and the aliased output write. The engine's
alias table marks the handle as both input and output within the same Compute pass. Tank should
confirm or propose an alternative before the real GQA kernel lands.

**Seam 3 — `build.rs` consuming `src/ops/shader_variants.txt`**

Implemented in `build.rs`. Mouse's 69-row, 168-module variant table is now the authoritative
source for all template-based shader compilation. See §4.2–§4.3 above.

**Seam 4 — Indirect dispatch for `QMoE`**

`DispatchContext::dispatch_indirect(IndirectKernelRequest)` issues `vkCmdDispatchIndirect`
from `k.dispatch_buffer` at `k.dispatch_offset`. A prior dispatch writes the `[u32; 3]`
workgroup counts; the engine inserts a `ShaderWrite → ShaderRead` barrier automatically on
`dispatch_buffer`. Default impl returns `Err(EpError::Internal(...))` — concrete engine
implementation required before QMoE's fast path is usable.

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
