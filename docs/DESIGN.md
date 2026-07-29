# onnxruntime-ep-vulkan — Architecture Design

**Status:** v0 architecture of record — accepted for M0/M1 implementation. **§7 (Vulkan baseline) is frozen.**
**Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-07-28T19:16:08-07:00 (§7 frozen, OQ-1 resolved)
**Author:** Morpheus (Lead / EP Architect)
**Repo:** `onnxruntime-ep-vulkan`
**Reference architecture:** `onnxruntime-mlx` (Justin Chu's MLX plugin EP for Apple Silicon)
**Sibling documents:** [`ENGINE.md`](./ENGINE.md) (Switch — Vulkan runtime & shaders), [`PLATFORMS.md`](./PLATFORMS.md) (Link — platform & hardware matrix), [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse — op coverage plan)

---

## 0. TL;DR

`onnxruntime-ep-vulkan` is an **out-of-tree ONNX Runtime plugin Execution Provider** that runs
fused ONNX subgraphs on any Vulkan compute device. It is loaded by a **stock, unmodified** ORT
build through the plugin-EP C ABI. No ORT fork, no ORT rebuild, no link against
`libonnxruntime`.

| Field | Value |
|---|---|
| Repository / vendor string | `onnxruntime-ep-vulkan` |
| Cargo crate | `rust/` — `onnxruntime-ep-vulkan` |
| Library artifact | `libonnxruntime_vulkan_ep.so` / `onnxruntime_vulkan_ep.dll` / `libonnxruntime_vulkan_ep.dylib` |
| Registered EP / device name | **`VulkanExecutionProvider`** |
| Crate type | `cdylib` |
| ORT ABI | plugin-EP C ABI, `ORT_API_VERSION 27` (ORT 1.27.x) |
| Version scheme | `0.<ORT_API_VERSION>.<patch>` → `0.27.0` |
| Backend | Vulkan compute, GLSL → SPIR-V, `ash` bindings |
| **Device requirement** | **Vulkan 1.1 core + a compute queue + four limits (§7.2).** No required extensions. Everything else is probed and degrades op coverage, never device availability. |

The architecture is **deliberately the same shape as `onnxruntime-mlx`**: a registry-driven,
claim → fuse → compile → run plugin EP with conservative node claiming and clean CPU fallback.
Every module in `onnxruntime-mlx` has a counterpart here. Where we diverge, §12 records the
divergence and the reason.

**The single biggest divergence from the MLX reference:** MLX runs on Apple unified memory, so
the MLX EP advertises *no device allocator* and copies out with one `memcpy` at the subgraph
boundary. Vulkan has **explicit, non-coherent, non-unified device memory**. That reshapes the
tensor/memory contract (§6), the factory surface (§2.5), the compile step (weight prepacking,
§5), and the milestone plan (§10). It is the reason this document exists rather than a
find-and-replace of the MLX design.

---

## 1. Goals and non-goals

### 1.1 Goals for v1

1. **Cross-platform GPU inference from one codebase.** Windows, Linux, Android, and macOS
   (MoltenVK) on NVIDIA / AMD / Intel / Adreno / Mali, plus software rasterizers (lavapipe,
   SwiftShader) so CI is possible without a GPU runner. No vendor-specific code path is
   permitted to be load-bearing for correctness.
2. **Zero ORT fork.** Ship a single shared library that a stock ORT loads via
   `RegisterExecutionProviderLibrary`.
3. **Correctness before performance.** The ORT CPU EP is the oracle. A claimed op must match it
   within a stated tolerance on every supported platform before we quote a single speedup number.
4. **Conservative claiming with clean CPU fallback.** Claim only node forms whose exact
   dtype / attribute / shape / layout contract the Vulkan translator implements. Everything else
   runs on ORT CPU. Falling back is a feature, not a gap.
5. **Compile-once, replay-many execution.** Weight upload and prepacking happen at `Compile`
   time; per-inference work is command-buffer submission, not graph construction.
6. **A layering that survives contact with contributors.** The ORT C ABI never reaches op code;
   raw Vulkan handles never reach op code. Enforced by module privacy, not by review vigilance.

### 1.2 Non-goals for v1 — explicit and ruthless

These are **out of scope**. Each is a decision, not an oversight. Re-opening any of them requires
a decision record.

| Non-goal | Why |
|---|---|
| **Training / gradients** | ORT training EPs are a different ABI surface and a different correctness problem. Inference only. |
| **ONNX opset completeness** | `onnxruntime-mlx` reached 184/202 ops *after* two years and a mature backend that supplies op semantics for free. Vulkan supplies nothing — every op is a shader we write. Broad coverage in v1 would mean broad, shallow, wrong. |
| **Dynamic shapes in the fast path (M0–M2)** | Static/shape-keyed subgraphs only. A shape change re-records the command buffer. Truly shapeless execution (the analog of MLX's shapeless decode compile) is M3+. |
| **Data-dependent output shapes** | `NonZero`, `Unique`, value-dependent `Reshape` targets. These need a mid-graph host readback that a recorded command buffer cannot express. Permanent CPU fallback. |
| **fp64** | Most consumer GPUs have no usable double precision and Vulkan makes `shaderFloat64` optional. Permanent CPU fallback. |
| **Quantized ops (int4/int8 matmul, MatMulNBits, GatherBlockQuantized)** | The highest-value target eventually, and the reason llama.cpp's Vulkan backend exists — but each is a hand-written packed-format shader. Not in v1. |
| **Attention fusion (GQA / MHA / SDPA / flash attention)** | Same reason, one level harder. M3+ at the earliest. |
| **Graph-level op fusion** | llama.cpp fuses `MUL_MAT+ADD`, `RMS_NORM+MUL`, etc. Real wins, but fusion patterns are a perf optimization layered on a correct per-node dispatcher. Not v1. |
| **Mobile-first tuning** | Android must *work* (M3) and must not be architecturally excluded (§7). Tile sizes, memory budgets, and Adreno/Mali-specific tuning are not v1. |
| **Images / texture-backed tensors** | ExecuTorch defaults to 3D image textures for mobile bandwidth. Buffers only in v1 (see `ENGINE.md` §3.6). Revisit when Niobe shows a bandwidth-bound case. |
| **Multi-GPU / multi-queue overlap** | One `VkDevice`, one compute queue, one submission per subgraph execution. |
| **Cooperative matrix / tensor-core paths** | Optional extension on every baseline. Capability-probed later, never required. |
| **Custom / contrib domain ops (`com.microsoft`)** | v1 is `ai.onnx` only. |
| **Shipping wheels on PyPI in v1** | The Python package exists for testing from M0. Publishing is a release decision, not an architecture one. |

### 1.3 Why "conservative claiming" is a hard requirement, not a preference

Because the fallback is not free but it *is* always correct. The failure mode we are designing
against is not "we didn't claim enough ops" — it is "we claimed a node form our shader gets
subtly wrong on one driver, and a user gets silently wrong logits." Every claim predicate is a
promise. The rule from the MLX reference stands verbatim: **when in doubt, do not claim.**

---

## 2. How it plugs into ONNX Runtime

### 2.1 The plugin-EP model

ORT exposes a public C ABI for registering an out-of-tree EP as a shared library. The host
resolves two symbols by name:

```c
OrtStatus* CreateEpFactories(const char* registered_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories);

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

`rust/src/lib.rs` exports both. ORT is reached **only** through the `OrtApi` function-pointer
table handed to `CreateEpFactories`; we never link `libonnxruntime`. Ownership crosses the C
boundary with `Box::into_raw` / `Box::from_raw`. Every `extern "C"` entry point that runs real
logic is wrapped in a panic guard that converts a Rust panic into an `ORT_EP_FAIL` status —
unwinding into ORT's C++ is undefined behaviour and a plugin must never take down its host.

Usage from the application side:

```python
import onnxruntime as ort
import onnxruntime_ep_vulkan

onnxruntime_ep_vulkan.register_execution_provider_library()
sess = ort.InferenceSession(model, providers=["VulkanExecutionProvider", "CPUExecutionProvider"])
```

### 2.2 Object lifecycle

```
dlopen / LoadLibrary
   └─ CreateEpFactories ────────────► VulkanEpFactory      (process-lived, one per registration)
        ├─ GetName / GetVendor / GetVendorId / GetVersion
        ├─ GetSupportedDevices(OrtHardwareDevice[]) ──────► OrtEpDevice[]   (device enumeration)
        ├─ CreateAllocator(OrtMemoryInfo) ────────────────► OrtAllocator    (device memory)
        ├─ CreateDataTransfer() ──────────────────────────► OrtDataTransferImpl
        └─ CreateEp(devices, session_options, logger) ────► VulkanEp        (one per session)
                ├─ GetName
                ├─ GetDefaultMemoryDevice ────────────────► OrtMemoryDevice
                ├─ GetCapability(OrtGraph, OrtEpGraphSupportInfo)      ← node claiming
                ├─ Compile(OrtGraph[], OrtNode[], OrtNodeComputeInfo[]) ← plan build + prepack
                │     └─ OrtNodeComputeInfo { CreateState, Compute, ReleaseState }  ← inference
                └─ ReleaseNodeComputeInfos
   └─ ReleaseEpFactory
```

The `VulkanEpFactory` struct embeds `OrtEpFactory` as its **first field** under `#[repr(C)]`, so
the pointer ORT holds is pointer-identical to our Rust struct at offset 0. Same for `VulkanEp` and
`OrtEp`, and for the per-subgraph compute-info object and `OrtNodeComputeInfo`. This is the exact
pattern proven in `onnxruntime-mlx/rust/src/factory.rs`.

### 2.3 Device enumeration — where Vulkan differs from MLX

The MLX EP's `GetSupportedDevices` picks *the first* `OrtHardwareDeviceType_GPU` ORT presents and
advertises exactly one `OrtEpDevice`. Apple Silicon has one GPU; that is sufficient there.

Vulkan does not have that luxury. The factory must:

1. Create a `VkInstance` (once per plugin load) and enumerate physical devices.
2. For each physical device, evaluate the **capability gate** (§7): does it meet the required
   feature set? If not, it is not advertised — an unusable device must never be offered to ORT.
3. Correlate each usable `VkPhysicalDevice` with the `OrtHardwareDevice` entries ORT presents.
   The correlation key is vendor ID + device ID, both of which appear in
   `VkPhysicalDeviceProperties` and in ORT's hardware-device metadata. Where correlation fails
   (software rasterizers, virtualized GPUs, MoltenVK), we fall back to type matching (GPU, then
   CPU) and record which strategy was used in the EP device metadata.
4. Create one `OrtEpDevice` per usable device via `EpApi::CreateEpDevice`, attaching EP metadata
   (Vulkan API version, device name, driver version, vendor) and EP options.

Consequences that follow, and which the MLX design never had to answer:

- **Device selection is a user-visible session option.** Multi-GPU machines are normal on Windows
  and Linux. `ep.device_index` selects among advertised devices; the default is the
  highest-scoring device (discrete > integrated > virtual > CPU), matching `ENGINE.md` §2.2.
- **`VkInstance` lifetime is factory-scoped, `VkDevice` lifetime is EP-scoped.** Two sessions on
  the same physical device share the instance but get independent logical devices, queues,
  allocators, and pipeline caches. Sharing a `VkDevice` across sessions is a post-v1
  optimization with real thread-safety cost; we do not take it now.
- **Enumeration must never abort the host.** A machine with no Vulkan loader, no ICD, or a broken
  driver must produce zero advertised devices and a warning — not a crash and not an error status
  that fails session creation. This is a tested requirement (Trinity, M0).

### 2.4 Session options

Prefixed `ep.` and read in `CreateEp` from the `OrtSessionOptions`. The v1 set is small on
purpose:

| Option | Type | Default | Meaning |
|---|---|---|---|
| `ep.device_index` | int | auto | Which advertised Vulkan device to bind. |
| `ep.enable_validation` | bool | `false` (release), `true` (debug) | Enable `VK_LAYER_KHRONOS_validation`. |
| `ep.pipeline_cache_path` | string | platform cache dir | On-disk `VkPipelineCache` blob location. |
| `ep.max_claim_ops` | string list | unset | Restrict claiming to a named op set. Debugging and bisecting only. |
| `ep.disable_device_memory` | bool | `false` | Force the M0 host-memory I/O path (see §6.3). Escape hatch for driver bugs. |
| `ep.force_legacy_barriers` | bool | `false` | Force the legacy `vkCmdPipelineBarrier` backend on a device that supports `synchronization2` (§7.5). Exists so CI exercises both barrier backends on the same hardware; also an escape hatch for a broken sync2 driver. |

Environment variables mirror the MLX EP's convention for observability, and are *not* a
configuration surface: `ONNXRUNTIME_EP_VULKAN_VERBOSE`, `ONNXRUNTIME_EP_VULKAN_TRACE=<path>`,
`ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG`, `RUST_LOG=onnxruntime_ep_vulkan=<level>`.

### 2.5 Node claiming (`GetCapability`) and subgraph compilation

**Claim.** For each node in the `OrtGraph`, `ep.rs` builds a `NodeView` (a read-only FFI wrapper
over `OrtNode` exposing op type, domain, since-version, input/output slot info, and attributes)
and asks the registry a single question: `claimable(&NodeView)`. There is **no per-op logic in
`ep.rs`**. This is the invariant that makes "claimed" and "translatable" impossible to
desynchronize, and it is inherited directly from the MLX reference.

**Fuse.** Claimed nodes are grouped into **maximal convex connected clusters** — the union-find +
reachability-bitset algorithm from the MLX EP. Convexity is not optional: a non-convex fusion
creates a cycle in the partitioned graph and ORT rejects it. Each cluster is handed to
`EpGraphSupportInfo_AddNodesToFuse`.

**Compile.** ORT calls `Compile` with one fused `OrtGraph` per cluster. For each we:
1. Extract a `NodeDesc` per node — op type, domain, since-version, generically-copied attributes
   (ints / floats / int arrays / float arrays / strings / tensors), and input/output tensor refs.
2. Build a `Plan`: the topologically ordered `NodeDesc` list plus the I/O binding table.
3. **Prepack**: read every constant initializer, convert it to the layout the shader wants, and
   upload it into a device-local buffer owned by the plan. This happens **once**. (§5.4)
4. Create or fetch the `VkPipeline` for every node in the plan, so the first inference does not
   pay shader compilation.
5. Hand ORT an `OrtNodeComputeInfo` that owns the plan.

**Run.** `Compute` binds ORT's input tensors, records (or replays) a command buffer, submits it,
waits on a fence, and makes the outputs visible to ORT.

### 2.6 CPU fallback

Unclaimed nodes are assigned to ORT's CPU EP by the ORT partitioner. ORT inserts the required
memcpy nodes at partition boundaries, using the `OrtDataTransferImpl` our factory supplies
(§6.2). Nothing about fallback is our code path — which is precisely why it is trustworthy.

The cost model, which the op-coverage strategy in §8 is built around: **one unclaimed node in the
middle of a graph splits it into two islands with a device round-trip between them.** Claim rate
is a bad metric; fused-region compute volume is the good one. The MLX EP learned this the
expensive way and we inherit the lesson rather than repeating it.

---

## 3. Repository and crate layout

Mapped one-to-one against `onnxruntime-mlx`. New paths carry a ✨.

```text
onnxruntime-ep-vulkan/
├── README.md                          # what it is, how to build, how to run
├── LICENSE
├── docs/
│   ├── DESIGN.md                      # ← this file: architecture of record
│   ├── ENGINE.md                      # ✨ Switch: Vulkan runtime, memory, shaders, sync
│   ├── PLATFORMS.md                   # ✨ Link: platform/driver matrix, toolchains, CI lanes
│   ├── OP_ARCHITECTURE.md             # Mouse: op registry + authoritative coverage table
│   └── BENCHMARKS.md                  # ✨ Niobe: methodology + published baselines
├── rust/
│   ├── Cargo.toml                     # cdylib crate, lib name onnxruntime_vulkan_ep
│   ├── build.rs                       # bindgen(ORT C ABI) + GLSL→SPIR-V compile+embed
│   ├── README.md                      # crate-level notes for contributors
│   ├── shaders/                       # ✨ GLSL compute sources (Switch)
│   │   ├── include/                   #    shared GLSL headers (indexing, broadcast, dtype)
│   │   ├── elementwise_binary.comp
│   │   ├── elementwise_unary.comp
│   │   └── ...
│   └── src/
│       ├── lib.rs                     # CreateEpFactories / ReleaseEpFactory, panic guards
│       ├── factory.rs                 # OrtEpFactory vtable: devices, allocator, data transfer
│       ├── ep.rs                      # OrtEp vtable: GetCapability, clustering, Compile, Compute
│       ├── engine.rs                  # Plan, NodeDesc, DispatchContext — the op-facing API
│       ├── recorded.rs                # ✨ command-buffer recording cache (shape-keyed replay)
│       ├── registry.rs                # op registry + NodeView / GraphView + claim helpers
│       ├── allocator.rs               # ✨ OrtAllocator over device memory
│       ├── transfer.rs                # ✨ OrtDataTransferImpl: host↔device staging copies
│       ├── vk/                        # ✨ the Vulkan layer — Switch owns, nothing else enters
│       │   ├── mod.rs                 #    re-exports the safe surface only
│       │   ├── instance.rs            #    VkInstance, layers, debug messenger
│       │   ├── device.rs              #    physical-device scoring, VkDevice, queues
│       │   ├── caps.rs                #    Capabilities struct: the single capability oracle
│       │   ├── memory.rs              #    gpu-allocator integration, DeviceBuffer, StagingPool
│       │   ├── pipeline.rs            #    VkPipeline creation, VkPipelineCache, spec constants
│       │   ├── descriptor.rs          #    descriptor set layouts and pools
│       │   ├── command.rs             #    command pool/buffer recording, barriers, submission
│       │   └── shaders.rs             #    embedded SPIR-V module table + variant selection
│       ├── ops/                       # per-family ONNX handlers + claim predicates (Mouse)
│       │   ├── mod.rs
│       │   ├── elementwise.rs
│       │   ├── math.rs
│       │   ├── reduction.rs
│       │   ├── shape.rs
│       │   ├── matmul.rs
│       │   └── norm.rs
│       ├── sys.rs                     # raw bindgen output for the ORT plugin-EP C ABI
│       ├── logging.rs                 # in-crate `log` subscriber, env-gated, silent by default
│       └── trace.rs                   # env-gated Chrome/Perfetto tracer + GPU timestamp queries
├── tests/
│   ├── README.md
│   ├── ops/                           # pytest op-correctness: Vulkan EP vs ORT CPU EP
│   │   ├── conftest.py                #    registers the plugin from ONNXRUNTIME_VULKAN_EP_LIB
│   │   ├── _models.py                 #    ONNX IR model builders, shared with bench/
│   │   └── test_*.py
│   ├── backend/                       # ONNX backend node tests through the EP
│   └── conformance/                   # opt-in broader conformance (onnx-tests harness)
│       ├── README.md
│       ├── RESULTS.md
│       ├── claimed_ops.txt
│       └── run_conformance.sh
├── bench/
│   ├── README.md
│   ├── bench.py                       # per-op-family timings, Vulkan vs CPU
│   ├── cases.py
│   └── compare.py                     # base-vs-PR regression table for CI comment
├── python/
│   ├── README.md
│   ├── pyproject.toml
│   ├── hatch_build.py                 # builds the cargo cdylib into the wheel
│   └── src/onnxruntime_ep_vulkan/
│       ├── __init__.py                # register_execution_provider_library(), EP_NAME, paths
│       └── py.typed
└── .github/workflows/
    ├── ci.yml                         # fmt, clippy, build matrix, op tests on lavapipe/SwiftShader
    ├── conformance.yml                # opt-in workflow_dispatch
    ├── bench.yml                      # informational perf comment on PRs
    └── publish.yml                    # wheel build + release
```

### 3.1 Naming decisions

- **EP name `VulkanExecutionProvider`.** Matches ORT's naming convention for every other EP and
  is what a user will guess. Frozen — changing it later breaks every user's provider list.
- **Library base name `onnxruntime_vulkan_ep`,** pinned via `[lib] name` so it is stable
  regardless of the crate name, exactly as the MLX EP does. Python, tests, CI, and any downstream
  runtime load it by that exact filename.
- **Version `0.<ORT_API_VERSION>.<patch>`.** A plugin EP is bound to one plugin-EP C-ABI version,
  so the version must state which ORT it works with. `0.27.0` pairs with ORT 1.27.x. When ORT
  ships API version 28, we move to `0.28.0`. The EP reports this to ORT from
  `env!("CARGO_PKG_VERSION")` so it can never drift from the manifest.
- **Vendor ID.** Unlike the MLX EP (which reports Apple's `0x106B`), there is no single hardware
  vendor here. The factory reports the **Vulkan `vendorID` of the bound physical device** from
  `VkPhysicalDeviceProperties`, so a user querying EP devices sees NVIDIA/AMD/Intel/Qualcomm/ARM
  correctly. Open question OQ-6 covers the no-device case.

---

## 4. Module responsibilities and boundaries

### 4.1 Layer map

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L0  ORT C ABI boundary        lib.rs · factory.rs · ep.rs · allocator.rs │
│                               transfer.rs · sys.rs                       │
│     Owns: OrtEpFactory/OrtEp/OrtNodeComputeInfo/OrtAllocator vtables,     │
│           Box::into_raw ownership, panic guards, OrtStatus construction.  │
│     Owner: Tank                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ L1  Plan & dispatch           engine.rs · recorded.rs · registry.rs      │
│     Owns: NodeDesc, Plan, DispatchContext, the op registry, NodeView,     │
│           claim dispatch, command-buffer recording cache.                 │
│     Owner: Morpheus (contract) · Tank (plumbing) · Mouse (registry)      │
├──────────────────────────────────────────────────────────────────────────┤
│ L2  ONNX op semantics         ops/*.rs                                   │
│     Owns: per-op claim predicates and translate handlers. Reads           │
│           attributes, validates dtypes/shapes, requests dispatches.       │
│     Owner: Mouse                                                         │
├──────────────────────────────────────────────────────────────────────────┤
│ L3  Vulkan engine             vk/*.rs · shaders/*.comp                   │
│     Owns: VkInstance/VkDevice/VkQueue, allocator, staging, descriptors,   │
│           pipelines, barriers, submission, SPIR-V modules.                │
│     Owner: Switch                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ L4  Raw bindings              ash · gpu-allocator · bindgen(ORT)         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 The two hard rules

**Rule 1 — The ORT C ABI never leaks into op code.**
No type from `sys::ort` may appear in a signature, field, or local in `rust/src/ops/`. Op handlers
see `NodeDesc`, `NodeView`, `TensorRef`, and `DispatchContext`. They never see `OrtNode`,
`OrtValue`, `OrtKernelContext`, `OrtStatus`, or an `OrtApi` function pointer. The exception that
proves the rule: `NodeView` and `NodeDesc` are *where* the ABI is translated into safe Rust, and
they live in `registry.rs` / `engine.rs`, not in `ops/`.

*Enforcement:* `sys` is `pub(crate)` but a CI lint (`ci.yml`) greps `rust/src/ops/` for `sys::`,
`Ort`, and `unsafe` and fails the build on a hit. A rule that is not mechanically checked is a
suggestion.

**Rule 2 — Op code never touches raw Vulkan handles.**
No `ash::vk::*` type may appear in `rust/src/ops/`. Op handlers cannot hold a `vk::CommandBuffer`,
call `vkCmdDispatch`, allocate memory, or create a pipeline. They express intent through
`DispatchContext`:

```rust
// The entire vocabulary an op handler has. Illustrative signature — the real one lands with M0.
pub trait DispatchContext {
    fn resolve(&mut self, r: &TensorRef) -> Result<BufferView, EpError>;
    fn bind_output(&mut self, o: &OutRef, desc: TensorDesc) -> Result<BufferView, EpError>;
    fn alloc_temp(&mut self, desc: TensorDesc) -> Result<BufferView, EpError>;
    fn dispatch(&mut self, k: KernelRequest) -> Result<(), EpError>;
    fn read_const_i64(&self, r: &TensorRef) -> Option<Vec<i64>>;
}
```

`BufferView` is an opaque handle. `KernelRequest` names a shader variant, its specialization
constants, its push-constant payload, its bindings, and a workgroup count. The engine decides
descriptor sets, barriers, pipeline selection, and submission. `ENGINE.md` §1 states the same
boundary from the engine side and enforces it with module privacy: the wrapper types in `vk/` are
not `pub` outside the engine.

*Enforcement:* the same CI lint greps `rust/src/ops/` for `ash`, `vk::`, and `unsafe`. It also
enforces the barrier seam of §7.5: `cmd_pipeline_barrier`, `cmd_pipeline_barrier2`,
`BufferMemoryBarrier`, `DependencyInfo`, `PipelineStageFlags*` and `AccessFlags*` may appear
**only** in `rust/src/vk/barrier.rs`, and `Capabilities::synchronization2` may be read **only** in
`rust/src/vk/barrier.rs` and `rust/src/vk/caps.rs`.

**Why this matters enough to reject a working PR over.** The MLX EP got a mature backend that
handled memory, scheduling, and dtypes. We do not. Every op we add is a shader, a descriptor
layout, a barrier, and a workgroup calculation. If those details are allowed to bleed into 60 op
modules, the first driver quirk Link finds becomes a 60-file change instead of a 1-file change.
The boundary is the only thing that keeps op coverage a linear cost.

### 4.3 What each module may depend on

| Module | May use | May **not** use |
|---|---|---|
| `lib.rs`, `factory.rs`, `ep.rs`, `allocator.rs`, `transfer.rs` | `sys::ort`, `engine`, `registry`, `vk` (safe surface) | `ash` directly, `ops::*` internals |
| `engine.rs`, `recorded.rs` | `vk` safe surface, `registry` | `sys::ort` FFI calls outside `NodeDesc` construction |
| `registry.rs` | `sys::ort` (for `NodeView` only), `engine` types | `ash`, `vk` |
| `ops/*.rs` | `engine::{DispatchContext, NodeDesc, TensorRef}`, `registry` helpers | `sys`, `ash`, `vk`, `unsafe` |
| `vk/*.rs` | `ash`, `gpu-allocator` | `sys::ort`, `ops` |

---

## 5. Execution flow, end to end

### 5.1 Library load

`RegisterExecutionProviderLibrary` → `dlopen`/`LoadLibrary` → `CreateEpFactories`.
Initialise logging, negotiate `ORT_API_VERSION 27` (fail with a clear status if the host is
older), construct one `VulkanEpFactory`. **No Vulkan work happens yet** — a plugin must be cheap
to load even on a machine that will never use it.

### 5.2 Device enumeration

ORT calls `GetSupportedDevices`. *Now* we create the `VkInstance`, enumerate physical devices,
apply the capability gate (§7), score and sort, correlate with ORT's `OrtHardwareDevice` list,
and create one `OrtEpDevice` per usable device. Zero usable devices → advertise none, log a
warning, return success. The instance is kept alive on the factory.

### 5.3 Session creation

ORT calls `CreateEp`. We read session options, select the physical device, create the `VkDevice`,
compute queue, `gpu-allocator` arena, command pool, descriptor pools, staging pool, and load the
on-disk `VkPipelineCache`. The `VulkanEp` owns all of it and drops it in `ReleaseEp`. RAII, not
manual teardown — this is where the MLX rewrite found a real per-session leak that three lines of
`impl Drop` fixed, and we take the same posture from day one.

`GetDefaultMemoryDevice` returns our device's `OrtMemoryDevice` (M2+) or null (M0/M1, §6.3).

### 5.4 `GetCapability` — claiming

Per node: build `NodeView` → `registry::claim_decision(&view)`. Rejections carry a *reason string*
and are aggregated per op type; with `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` or tracing on, the EP
prints exactly which ops were declined, how many, and why. This is the single most valuable
diagnostic the MLX EP has, and it is cheap. It ships in M0.

Claimed nodes → convex clustering → `EpGraphSupportInfo_AddNodesToFuse` per cluster.

### 5.5 `Compile` — plan build and prepacking

For each fused subgraph, in order:

1. **Extract.** `NodeDesc` per node, attributes copied generically into typed maps, inputs and
   outputs resolved to `TensorRef`/`OutRef` with dtype and static shape where known.
2. **Validate.** Every node must resolve to a registry entry with a claim predicate that still
   accepts it. A mismatch here is an internal invariant violation, not a user error — fail the
   compile loudly.
3. **Plan shapes.** Compute static shapes for every intermediate. Determine the workgroup counts
   and the temporary-buffer set. Assign temporaries to a shared arena with a greedy
   liveness-based packing (ExecuTorch's `SharedObject` idea) so a 40-node subgraph does not
   allocate 40 buffers.
4. **Prepack weights.** For each constant initializer: read host bytes through the ORT graph API,
   convert to the shader's expected layout/dtype, and upload into a **device-local buffer owned by
   the plan**, via a staging buffer. Done once, at compile time. This is the ExecuTorch
   `prepack()` model and the direct analog of the MLX plan's repacked-weight cache, and it is the
   reason inference does not re-upload weights.
5. **Warm pipelines.** Create every `VkPipeline` the plan needs, populating the `VkPipelineCache`.
   First-inference latency is a real user-visible cost on Vulkan; paying it at compile time is the
   right trade.
6. **Wrap.** Return an `OrtNodeComputeInfo` owning the plan.

### 5.6 `Compute` — inference

```
ORT calls Compute(state, OrtKernelContext)
 ├─ 1. Resolve inputs.
 │      M0/M1 (host I/O):   ORT hands host pointers → upload into device buffers via staging.
 │      M2+  (device I/O):  ORT hands device "pointers" our allocator produced → resolve to
 │                          (VkBuffer, offset) with no copy at all.
 ├─ 2. Shape key.  Hash the concrete input shapes. Look up recorded.rs.
 │      hit  → reuse the recorded VkCommandBuffer.
 │      miss → record: for each NodeDesc in topo order, registry::translate() runs the handler,
 │             the handler calls DispatchContext::dispatch(), the engine binds descriptors,
 │             emits the memory barrier the dependency edge requires, and records vkCmdDispatch.
 │             Cache the recording under the shape key.
 ├─ 3. Bind outputs. KernelContext_GetOutput for each subgraph output.
 ├─ 4. Submit once to the compute queue. One submission per subgraph execution.
 ├─ 5. Wait on the fence.
 └─ 6. Outputs.
        M0/M1: download device → staging → ORT's host output tensor.
        M2+:   nothing — the output already lives in the device buffer ORT allocated.
```

**Where CPU fallback happens.** Three distinct places, and it is worth keeping them straight:

1. **Claim time (the main one).** A node the registry declines is never in a plan. ORT assigns it
   to the CPU EP and inserts partition-boundary copies through our `OrtDataTransferImpl`. This is
   the designed path and it is always correct.
2. **Compile time.** If a plan cannot be built (a shape that cannot be resolved statically, an
   initializer that cannot be read, a pipeline that fails to create), `Compile` returns an error
   status and ORT falls the whole subgraph back to CPU. Loud, logged, and rare by construction —
   §5.5 step 2 makes it an invariant violation.
3. **Runtime.** A device-lost, out-of-memory, or panic condition returns `ORT_EP_FAIL` from
   `Compute`. ORT surfaces the failure. We do **not** attempt a silent per-node CPU rescue inside
   `Compute`: a partially-executed command buffer with half-written outputs is not a state we can
   reason about, and silently producing CPU results after a GPU fault hides real bugs.

---

## 6. Tensor and memory model

This is the section where Vulkan and MLX genuinely part ways, so it states the **contract**;
`ENGINE.md` §3 owns the implementation.

### 6.1 The problem

MLX: unified memory. An ORT CPU tensor and an MLX array can point at the same bytes. The MLX EP
therefore advertises no device allocator, returns null from `GetDefaultMemoryDevice`, and copies
out once per subgraph with a `memcpy`.

Vulkan: explicit device memory. Device-local memory is generally not host-visible; host-visible
memory is generally not device-local; and on discrete GPUs the two are across a PCIe bus. There is
no pointer we can hand ORT that a shader can also read. Everything below follows from that.

### 6.2 The contract

| Concern | Contract |
|---|---|
| **Tensor identity** | A device tensor is `(VkBuffer, offset, size, dtype, shape)`. The `VkBuffer` may be a suballocation of a larger arena. Op code sees this only as an opaque `BufferView`. |
| **Layout** | Row-major dense, ONNX semantics, no implicit padding, no implicit transposition. Any op needing a different internal layout materializes it explicitly as a temporary. Prepacked *constants* may use a shader-specific layout; **activations may not**. |
| **Alignment** | Every allocation satisfies `minStorageBufferOffsetAlignment` for the bound device. Enforced by the allocator, never by op code. |
| **Who allocates weights** | The plan, at `Compile` time, device-local, uploaded once, freed when the plan drops. |
| **Who allocates activations at partition boundaries** | ORT, through our `OrtAllocator` (M2+) or ORT's CPU allocator (M0/M1). |
| **Who allocates intermediates** | The engine, from a plan-owned arena sized at compile time, reused across inferences. |
| **Who transfers** | Only `transfer.rs` (`OrtDataTransferImpl`) and the engine's staging path. Op code never initiates a transfer. |
| **When transfers happen** | At partition boundaries (ORT-inserted) and, in M0/M1, at subgraph entry/exit. Never per node. |
| **Coherence** | Non-coherent host-visible memory is flushed/invalidated by the engine around every host access. This is not optional and it is not the op author's problem. |
| **Synchronization** | The engine emits the barriers the plan's dataflow edges imply. A read-after-write between two dispatches always gets a barrier. Correctness first; barrier-batching optimization is Niobe's ticket, not a design assumption. |

### 6.3 Avoiding a copy on every inference — the phased plan

This is the crux, so it is explicit.

**M0/M1 — host I/O (the MLX shape).** `GetDefaultMemoryDevice` returns null; the factory's
`CreateAllocator`/`CreateDataTransfer` return null (valid — ORT tolerates it, as the MLX EP
proves). Subgraph I/O lives in CPU memory; `Compute` uploads inputs and downloads outputs each
call. Weights are still uploaded once at compile time, so the per-inference traffic is
activations only.

*Why start here:* it removes the entire allocator/data-transfer/ORT-memory-placement surface from
M0, which is the highest-uncertainty part of the ABI. It gets a correct, cross-platform,
CPU-oracle-verified elementwise op running on Windows and Linux in the shortest path. It is
honestly slow for anything small, and we will say so rather than benchmark it.

**M2 — device I/O (the real model).** The factory implements `CreateAllocator` (returning an
`OrtAllocator` backed by the device arena) and `CreateDataTransfer` (host↔device staging copies).
`GetDefaultMemoryDevice` returns the device's `OrtMemoryDevice`. ORT then:
- places tensors that only ever cross Vulkan partitions in device memory, so **two adjacent Vulkan
  subgraphs separated by a CPU node still avoid one of the two round-trips**;
- uses our data transfer for the boundary copies it does need;
- lets a user with `IoBinding` keep inputs and outputs resident on the device across inferences,
  which is what makes a per-inference copy disappear entirely.

The one hard problem M2 must solve: **ORT's allocator API is pointer-based, and a `VkBuffer` is
not a pointer.** The v1 answer is an opaque-handle registry — `Alloc` returns a unique tagged
64-bit value from a reserved range, and the EP resolves it to `(VkBuffer, offset)` through a
process-wide map. We deliberately do **not** build on `VK_KHR_buffer_device_address` (which would
give a real GPU virtual address) because it is optional, and MoltenVK support is partial
(`ENGINE.md` §8). Revisit if profiling shows the map lookup matters; it will not.

**M3+ — persistence.** Keep prepacked weights and, where a graph allows, activation buffers
resident across `Compute` calls; shapeless recording so a growing dimension does not retrace.

---

## 7. Vulkan API baseline — decision

> **Status: FROZEN as of 2026-07-28T19:16:08-07:00.** OQ-1 is **resolved** with measured data
> (Link, [`PLATFORMS.md`](./PLATFORMS.md) §8, vulkan.gpuinfo.org pulled 2026-07-28) and the answer
> **reversed the provisional §7.2 requirement set**. This section is now the binding contract for
> Switch's [`ENGINE.md`](./ENGINE.md) and Link's CI matrix. Changing it requires a new decision
> record, not an edit.
>
> **Governing directive (Justin, 2026-07-28):** 「如果 1.3 兼容性不好 那 1.2 更好。可以保证兼容性
> 是最好。」 — *broad device compatibility is the top-priority property of this decision.* Where
> device coverage and engine-code simplicity conflict, **coverage wins**. Every ruling below is
> made under that constraint, and where it costs Switch complexity, it costs Switch complexity.
>
> Justin separately ratified the *framing* — a capability set rather than a version number
> (「拿能力集很聪明，听你的」). What changed is where the bar sits, not how the bar is expressed.

### 7.0 The frozen principle

**The device gate is minimal. Capability shortfalls degrade op coverage, not device availability.**

This one sentence replaces the previous "require the two features Switch wants" posture. A device
that lacks an optional capability must still load, still be advertised, and still run every op
that does not need that capability. It declines the ops it cannot do correctly, and ORT's
partitioner sends those to CPU (§2.6). We never refuse a device for a reason that only affects
*some* ops.

Consequences, stated so they are not re-litigated per op:

- A hard device requirement must be justified by *"no op we will ever ship can work without it."*
- A per-op requirement is expressed as a claim predicate in `ops/` (§8), never as a device gate.
- Anything we make optional, we must be able to run **both ways in CI** (§7.5 item 5, §9.1).

### 7.1 The evidence

| Source | Finding |
|---|---|
| Justin's proposal | Vulkan 1.3, citing llama.cpp. |
| llama.cpp `ggml-vulkan.cpp` | Hard runtime floor is **Vulkan 1.2** — `if (api_version < VK_API_VERSION_1_2) throw`. `VkApplicationInfo::apiVersion` is set to *whatever the instance reports*, not to a hardcoded 1.3. |
| llama.cpp `vulkan-shaders-gen.cpp` (Fact Checker, claim 1, SHA `3e6b395`) | **Base shaders are compiled with `--target-env=vulkan1.2`.** Only the cooperative-matrix-2 (`_cm2`) variants — an NVIDIA Ampere+/Ada optimization path — target `vulkan1.3`. The `--target-env=vulkan1.3` in CMakeLists is an extension-availability probe, not the default. **Verdict: the "llama.cpp requires 1.3" claim is contradicted.** |
| ExecuTorch `vk_api/Runtime.cpp` (Fact Checker, claim 2, SHA `8001512`) | Hardcodes `VK_API_VERSION_1_1` in `VkApplicationInfo`; `Device.cpp` branches feature queries at `>= VK_API_VERSION_1_1`; VMA is initialized with `VK_API_VERSION_1_0`. **Verdict: contradicted** — ExecuTorch targets 1.1. |
| MoltenVK (Fact Checker, claim 3) | **Verified:** MoltenVK 1.3.0 (2025) advertises Vulkan 1.3 on macOS/iOS. Older MoltenVK does not. |
| Android (Fact Checker, claim 4 — *unverified*, plausible) | Vulkan 1.3 ≈ **26%** of active Android devices; Vulkan 1.1 ≈ **62%**. The Android CDD does not mandate 1.3 at any API level as of Android 15. Link (`PLATFORMS.md` §4) reports ~89% for 1.1 measured against *devices that expose Vulkan at all* — a different denominator, same conclusion. |
| lavapipe / SwiftShader (Fact Checker, claim 5) | **Verified:** both support Vulkan 1.3 and both pass 1.3 conformance. Adequate for GPU-less CI. |
| Link (`PLATFORMS.md` §4) | Recommends 1.2 core + mandatory device features. Explicitly does **not** recommend a hard 1.3 baseline if Android coverage is a goal. |
| Switch (`ENGINE.md` §8) | Exactly **two** features materially simplify the engine: `synchronization2` and `subgroup_size_control`. Both are core in 1.3 — **and both are available as standalone extensions on 1.1/1.2 drivers.** `shaderFloat16`, `bufferDeviceAddress`, and cooperative matrix must be capability-probed at runtime *regardless of baseline*. |
| **Link, OQ-1 (`PLATFORMS.md` §8), vulkan.gpuinfo.org, pulled 2026-07-28** | **`VK_KHR_synchronization2`: Android 68.57%, Windows 87.78%, Linux 99.05%, macOS 97.5%, iOS 100%.** A **31.43-point Android gap** and a **12.22-point Windows gap.** The Android shortfall is concentrated in Adreno 5xx (Snapdragon 625–660, frozen pre-2021 OEM blobs), Adreno 6xx on unupdated Android 10/11, and Mali Bifrost (G52/G57/G72/G76) especially on MediaTek — populations with no update cadence, so this does not decay with time on any schedule we control. Link's verdict: the hard requirement is **not safe**. |
| **Link, OQ-1** | **`VK_EXT_subgroup_size_control`: Android 85.88%, Windows 93.33%, Linux 98.81%, macOS/iOS 100%.** A **14.12-point Android gap.** |
| **Link, OQ-1 — the MoltenVK artifact** | The macOS/iOS 100% figure is **extension-string presence only**. MoltenVK reports Vulkan 1.3, which promotes `subgroup_size_control` to core, so the string is always there — but the `subgroupSizeControl` **feature flag is `VK_FALSE`**, because Metal cannot control SIMD-group width per pipeline. **Requiring the feature flag to be `VK_TRUE` would silently exclude all of macOS and iOS** — and probably lavapipe and SwiftShader too, which have a single fixed CPU SIMD width. |
| **Link, OQ-1 — limits that *are* safe** | `maxComputeWorkGroupInvocations ≥ 256`: ~1% of 8,206 Android reports show 128. `maxComputeSharedMemorySize ≥ 16 KiB`: the Vulkan spec minimum, universal. Subgroup `BASIC`: spec-guaranteed in the compute stage on 1.1+. Subgroup `ARITHMETIC`: >95%, but *query, never assume*. |
| **Morpheus, layer-shim feasibility research (2026-07-28, primary sources below)** | The Khronos `VK_LAYER_KHRONOS_synchronization2` shim **cannot be shipped by us on Android.** The AOSP Vulkan loader does not read `VK_LAYER_PATH`, does not use JSON manifests, and searches only the **host application's** `nativeLibraryDir` (derived from the installed APK via `GraphicsEnv::getAppNamespace()`) plus `/data/local/debug/vulkan` (debuggable/userdebug only). Khronos' own `docs/synchronization2_layer.md` states the `.so` "needs to be packaged **inside the APK**". A plugin `.so` `dlopen`ed into someone else's process has no mechanism to add a layer search path. Sources: `developer.android.com/ndk/guides/graphics/validation-layer`; `KhronosGroup/Vulkan-Loader` `docs/LoaderLayerInterface.md` ("The Android loader does not use manifest files"; "There is No Support For Implicit Layers on Android"); `KhronosGroup/Vulkan-ExtensionLayer` `docs/synchronization2_layer.md`. |
| **Morpheus, prior-art check on barrier strategy** | **wgpu, Dawn, and Godot all use legacy `vkCmdPipelineBarrier` exclusively and none of them ships the sync2 layer.** `gfx-rs/wgpu` `wgpu-hal/src/vulkan/command.rs` calls `cmd_pipeline_barrier` in `transition_buffers`/`transition_textures` with no sync2 variant and no sync2 entry in its `Workarounds` bitflags; `google/dawn` `src/dawn/native/vulkan/CommandBufferVk.cpp` calls `fn.CmdPipelineBarrier` and mentions sync2 only in a spec comment; `godotengine/godot` `drivers/vulkan/rendering_device_driver_vulkan.cpp` calls `vkCmdPipelineBarrier`. The cited precedent for Option B does not survive contact with the source. |

The premise that motivated 1.3 — "llama.cpp requires it" — is contradicted by llama.cpp's own
source at both the runtime check and the shader target, and independently verified as contradicted
by Fact Checker. That does not make 1.3 wrong; it makes the *reason* wrong, and I would rather we
decide this on the two features Switch identified than on a misattribution.


### 7.2 Decision — the frozen capability set

**We require a capability set, not a version number.** The set is deliberately small.

A physical device is advertised to ORT **if and only if** it satisfies all of:

| # | Hard requirement | Why it is a *device* gate and not a per-op gate |
|---|---|---|
| R1 | Vulkan **≥ 1.1** core, instance and device | `VkPhysicalDeviceFeatures2` / `VkPhysicalDeviceProperties2` chains and `VkPhysicalDeviceSubgroupProperties` are core at 1.1. Below 1.1 we cannot even *ask* what a device can do, so no op can be claimed safely. This is also the Android floor. |
| R2 | A queue family with `VK_QUEUE_COMPUTE_BIT` | Without it there is nothing to dispatch to. |
| R3 | `maxComputeWorkGroupInvocations ≥ 256` | Every shader skeleton we will write assumes a 256-invocation workgroup. ~1% of Android reports fall below. |
| R4 | `maxComputeSharedMemorySize ≥ 16384` | The Vulkan spec minimum; universal. Listed so the assumption is written down, not because it filters anything. |
| R5 | Subgroup `BASIC` in the `COMPUTE` stage | Spec-guaranteed on 1.1+; listed for the same reason as R4. |
| R6 | At least one `DEVICE_LOCAL` memory type and at least one `HOST_VISIBLE` memory type | The staging path (§6) has no meaning otherwise. |

**That is the entire gate.** It is satisfied by essentially every device that exposes Vulkan 1.1
at all, on every platform, including MoltenVK, lavapipe and SwiftShader.

**Everything else is capability-probed** into a single `vk::caps::Capabilities` struct, read once at
device init, and used in exactly two ways: (a) to select an implementation strategy inside the
engine, or (b) to gate an op's claim predicate. Nothing on this list may ever become a device gate
without a new decision record:

| Capability | Probed how | What it changes |
|---|---|---|
| `synchronization2` | 1.3 core **or** `VK_KHR_synchronization2` device extension | Selects the barrier backend (§7.3). **Not required.** |
| `subgroup_size_control` **properties** | 1.3 core **or** `VK_EXT_subgroup_size_control` — *properties queryable only* (§7.4) | Narrows the known subgroup-size range; enables the subgroup-cooperative shader variants. |
| Subgroup `ARITHMETIC` / `BALLOT` / `SHUFFLE` | `VkPhysicalDeviceSubgroupProperties::supportedOperations` | Gates the subgroup-reduction shader variants. Absent → shared-memory tree-reduction variant. |
| `shaderFloat16`, `storageBuffer16BitAccess` | `VkPhysicalDeviceVulkan12Features` / `VK_KHR_shader_float16_int8` + `VK_KHR_16bit_storage` | Gates fp16 op variants; absent → those ops are not claimed for fp16. |
| `shaderInt8`, integer dot product | extension probe | Gates future quantized ops. |
| Timeline semaphores | 1.2 core or `VK_KHR_timeline_semaphore` | Post-v0 multi-stream pipelining. Unused in v0. |
| `bufferDeviceAddress` | 1.2 core or `VK_KHR_buffer_device_address` | OQ-3 alternative only. |
| Cooperative matrix | `VK_KHR_cooperative_matrix` / `VK_NV_cooperative_matrix2` | Post-v0 GEMM variants, llama.cpp's `_cm2` split. |

`VkApplicationInfo::apiVersion` is set to `min(vkEnumerateInstanceVersion(), VK_API_VERSION_1_3)`
— llama.cpp's pattern. We ask for the highest the loader will give us and then *use* only what the
device actually reports.

### 7.3 `synchronization2` — dropped from the hard requirement; Switch carries a legacy path

**Ruling: Option A.** `synchronization2` is **not required**. Switch implements a legacy
`vkCmdPipelineBarrier` backend alongside the `vkCmdPipelineBarrier2` backend, selected once at
device init (§7.5 defines the seam).

This reverses the provisional §7.2 of 2026-07-28T17:59:54-07:00, which required it.

**Why.** Under the compatibility-first directive, a 31.43-point Android exclusion and a
12.22-point Windows exclusion cannot be traded for one internal code path. The Windows number
matters as much as the Android one and is easy to overlook: nearly one desktop Windows device in
eight in Link's sample would be silently declined. The missing Android population is
*structurally* missing — Adreno 5xx blobs frozen before the 2021 extension, Mali Bifrost on
MediaTek with no update cadence — so it does not shrink on any timeline we control.

The cost is bounded and one-time: two implementations of a five-function internal API, written
once, tested in CI on every run (§7.5). The cost of the alternative is unbounded and permanent:
every device we decline is a device we can never win back with engineering.

#### Ruling on the layer-shim proposal (Link's Option B) — **rejected as a shippable mechanism**

The coordinator asked me to examine rather than adopt this. I did, and the concern is correct and
decisive.

| Platform | Can *our plugin* enable `VK_LAYER_KHRONOS_synchronization2`? | Basis |
|---|---|---|
| **Retail Android (non-rooted, non-debuggable)** | **No.** | The AOSP loader does not read `VK_LAYER_PATH`, does not use JSON manifests, and enumerates layers only from the **host application's** `nativeLibraryDir` (set by the framework at process launch from the installed APK via `GraphicsEnv::getAppNamespace()`) and from `/data/local/debug/vulkan`, which requires a debuggable app or a userdebug build. Khronos' own layer documentation says the `.so` must be "packaged inside the APK". We do not own the APK. |
| Windows | Conditionally yes | `SetEnvironmentVariable("VK_ADD_LAYER_PATH", …)` before *our* `vkCreateInstance` works, because the desktop loader re-scans layer paths at `vkCreateInstance`, not at load time. **Fails silently** if the host process runs at High Integrity Level (`loader_secure_getenv` returns NULL). Mutating the environment of a host process we do not own is also a `setenv`/`getenv` data race in any multi-threaded host. |
| Linux / macOS | Conditionally yes | Same mechanism; fails under setuid/setgid (`secure_getenv`). Manifest must carry an absolute `.so` path. Same race. |

Two independent reasons to reject it even where it technically works:

1. **It does not solve the platform it was proposed for.** Android is 100% of the reason we were
   considering it, and Android is the one platform where it cannot work from a plugin.
2. **The cited precedent does not exist.** wgpu, Dawn, and Godot were offered as evidence that
   shipping this layer is normal practice. Reading their source, all three use legacy
   `vkCmdPipelineBarrier` exclusively and none of them ships the sync2 layer. The precedent
   actually supports Option A.

Add to that: silently mutating a host process's environment variables from inside a `dlopen`ed
plugin is behaviour I would reject in code review on its own merits, independent of Vulkan.

**What survives.** Nothing that we ship. If an *Android integrator* independently packages
`libVkLayer_khronos_synchronization2.so` in their own APK and enables it, our sync2 backend will
light up automatically — because we probe the extension, and the layer's documented behaviour is
to advertise the extension and disable itself when the driver already provides it. That is a
**documented, optional, integrator-side deployment note** in `PLATFORMS.md`, not a mechanism we
depend on and not a substitute for the legacy path. Labelled as materially weaker, exactly as the
coordinator required.

**Option C (scope Android to a 2021+ population) is rejected outright** — it is the directive read
backwards.

### 7.4 `subgroup_size_control` — required as a *query*, never as a *feature*

**Ruling.** `subgroup_size_control` is **not** a device gate at all, and where we do consult it we
require only that the **properties struct is queryable**. We **never** require
`VkPhysicalDeviceSubgroupSizeControlFeatures::subgroupSizeControl == VK_TRUE`, and we never call
`vkCmdSetRequiredSubgroupSize`-style per-pipeline sizing as a correctness dependency.

Precisely what the engine does:

1. **Always** read `VkPhysicalDeviceSubgroupProperties::subgroupSize` and `supportedOperations`
   (Vulkan 1.1 core, universally available). This is the baseline knowledge.
2. **If** `VK_EXT_subgroup_size_control` is present *or* the device reports 1.3 core, chain
   `VkPhysicalDeviceSubgroupSizeControlProperties` into `vkGetPhysicalDeviceProperties2` and record
   `minSubgroupSize` / `maxSubgroupSize` / `requiredSubgroupSizeStages`. Treat this as *better
   information about the range*, nothing more.
3. **Only if** the `subgroupSizeControl` feature flag is additionally `VK_TRUE` may a pipeline be
   created with `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`. This is an *optimization
   path*, gated at pipeline-creation time.
4. **A shader whose correctness depends on a specific subgroup width may only be selected when the
   width is known exactly** — i.e. `minSubgroupSize == maxSubgroupSize`, or the required-size
   pipeline path from (3) is available and was used. Otherwise the engine selects the portable
   variant, which uses shared memory and workgroup barriers and makes no subgroup-width assumption.

Rule 4 is the substantive part and it is a correctness rule, not a performance rule. Assuming a
subgroup width silently produces wrong numbers in cooperative GEMM and reduction shaders; that was
the original reason for wanting this extension, and this formulation preserves the guarantee
without excluding anyone.

**Why this matters beyond macOS.** Requiring the feature flag would have excluded all of
macOS/iOS (MoltenVK reports `VK_FALSE`; Metal has no per-pipeline SIMD-group width control) and
very likely lavapipe and SwiftShader — meaning our own CI lanes. A requirement that excludes the
machines you test on is a requirement you have not tested. Requiring the extension *string* would
still have cost 14.12 points of Android for information we can approximate from 1.1 core.

**Link's third open item — the 12.22% Windows `synchronization2` gap — is moot** under this
ruling. We accept nothing, because we require nothing; those devices run the legacy backend.

### 7.5 The barrier abstraction contract — binding on `ENGINE.md`

Switch wrote `ENGINE.md` §6.2 around a single `vkCmdPipelineBarrier2` path (§6.3 already noted a
fallback, but only as a sentence). This section is the contract that replaces it.

**Rule: one internal barrier API, two backends, selected exactly once at device init. Not
`if caps.sync2 { … } else { … }` at call sites.** A dual path scattered across the recorder is how
this decision turns into a bug farm; a single seam is how it stays a one-time cost.

The seam lives in **`rust/src/vk/barrier.rs`** and is the *only* file in the crate permitted to
name `vkCmdPipelineBarrier`, `vkCmdPipelineBarrier2`, `VkBufferMemoryBarrier`,
`VkBufferMemoryBarrier2`, `VkDependencyInfo`, or the `VK_PIPELINE_STAGE*` / `VK_ACCESS*` flag
families. The layering lint (§4.2) is extended to enforce this: those tokens outside
`vk/barrier.rs` fail CI.

Shape of the seam (illustrative — Switch owns the final signatures):

```rust
// rust/src/vk/barrier.rs — the ONLY module that names Vulkan barrier types.

/// Our own closed set. Deliberately contains no `None`/`NONE` variant: `VK_PIPELINE_STAGE_2_NONE`
/// has no legacy equivalent, so the abstraction must not be able to express it.
pub(crate) enum Access { ShaderRead, ShaderWrite, TransferRead, TransferWrite, HostRead, HostWrite }

pub(crate) struct BufferDep {
    pub buffer: vk::Buffer, pub offset: u64, pub size: u64,
    pub src: Access, pub dst: Access,
}

pub(crate) enum Barriers { Sync2(Sync2Backend), Legacy(LegacyBackend) }

impl Barriers {
    /// Chosen ONCE, in `Device::new`, from `Capabilities`. Never re-evaluated.
    pub(crate) fn select(caps: &Capabilities, dev: &ash::Device) -> Self;

    pub(crate) fn buffer_deps(&self, cb: vk::CommandBuffer, deps: &[BufferDep]);
    pub(crate) fn execution_only(&self, cb: vk::CommandBuffer, src: Access, dst: Access);
}
```

Binding requirements on the implementation:

1. **`Barriers::select` is called once, in `Device::new`, and the result is stored on the device
   handle.** `recorded.rs` and every op path call `dev.barriers().buffer_deps(...)`. No call site
   anywhere else may branch on `caps.synchronization2`.
2. **`Access` and `Stage` are our own closed enums, not Vulkan flag re-exports.** This is what makes
   the legacy backend total rather than best-effort: every value we can express has an exact 32-bit
   legacy equivalent by construction. `VK_PIPELINE_STAGE_2_NONE`, `SHADER_STORAGE_*`-only bits, and
   any other sync2-only concept are simply not representable.
3. **The mapping is one table, in one place.** `ShaderRead → (COMPUTE_SHADER, SHADER_READ)`,
   `ShaderWrite → (COMPUTE_SHADER, SHADER_WRITE)`, `TransferRead → (TRANSFER, TRANSFER_READ)`,
   `TransferWrite → (TRANSFER, TRANSFER_WRITE)`, `HostRead → (HOST, HOST_READ)`,
   `HostWrite → (HOST, HOST_WRITE)`. The sync2 backend widens the same table to the `_2_` flag
   names. If the two tables ever disagree, that is a bug in one file.
4. **Batching semantics are identical in both backends.** `buffer_deps` takes a slice and emits
   **one** barrier command covering all of them — `VkDependencyInfo` with N
   `VkBufferMemoryBarrier2` for sync2, one `vkCmdPipelineBarrier` with N `VkBufferMemoryBarrier`
   and OR-ed stage masks for legacy. The barrier-placement algorithm in `ENGINE.md` §6.2 does not
   change at all; only the emission does.
5. **Both backends are exercised on every CI run.** A new session option
   **`ep.force_legacy_barriers`** (bool, default `false`) forces `Barriers::Legacy` on a device
   that supports sync2. Trinity runs the differential suite twice per lane — once default, once
   forced — so the legacy path is never the untested path. Without this, the ~99%-Linux-coverage
   sync2 path is the only one our CI ever sees, and the 31% of Android we just bought would be
   running code no test has executed.
6. **`ENGINE.md` §6.2's worked example must be rewritten in terms of `buffer_deps`**, and §6.3's
   `vkCmdPipelineBarrier2` row must point at this section. §6.2's *reasoning* — per-edge barriers
   rather than one global barrier, one barrier per consumer edge — is correct and unchanged.

### 7.6 Why this and not the alternatives

**Why not a hard 1.3 baseline (Justin's original proposal).** It buys the two features Switch
wanted, which we have now stopped requiring anyway. It costs roughly **36 points of Android
installed-base coverage** (Fact Checker: 26% at 1.3 vs 62% at 1.1) and any MoltenVK older than
1.3.0, for zero engine simplification. The premise that motivated it — "llama.cpp requires 1.3" —
is contradicted by llama.cpp's own source at both the runtime check (floor 1.2) and the shader
target (base shaders `--target-env=vulkan1.2`), independently verified by Fact Checker (claims
1–2). It is also flatly incompatible with the compatibility-first directive.

**Why not a hard 1.2 baseline.** On Android the 1.2 tier barely exists — devices jumped 1.1 → 1.3
— so a 1.2 floor pays nearly the full Android cost of a 1.3 floor while getting less than 1.3
gives on desktop. The only 1.2 core feature we care about is timeline semaphores, which v0 does not
use and which are available as `VK_KHR_timeline_semaphore` on 1.1 when we do.

**Why not keep the two-extension requirement (the provisional 2026-07-28T17:59:54 position).**
Because Link measured it and it costs 31.43 points of Android and 12.22 points of Windows. It was
a defensible position under "we don't know the number"; it is indefensible now that we do.

**Why not require `synchronization2` on desktop only, and legacy on Android.** A per-platform
requirement is the worst of both: we still write both backends, and we additionally get a matrix
where a Windows-only contributor cannot reproduce an Android-only code path. If we are writing the
legacy backend at all, it must be the one that runs everywhere it is needed and gets tested
everywhere.

**What we lose by being this permissive.** Two things, both accepted: (1) `Barriers` has two
implementations forever, and the CI matrix doubles for barrier-sensitive tests (§7.5 item 5);
(2) a device can now be advertised, claim an op, and produce a slow result where a stricter gate
would have declined the device and let CPU handle it. Mitigation for (2) is Niobe's job, not the
gate's: if a device class is measurably worse than CPU, that is a *scoring* and *claim* decision
(§8), recorded per device class in `PLATFORMS.md` — not a reason to refuse to load.

### 7.7 Shader targets

SPIR-V is compiled with `--target-env=vulkan1.1` by default (SPIR-V 1.3), which every device
meeting §7.2 can consume. This is one notch below llama.cpp's `vulkan1.2` default and is the
conservative choice for Android breadth; if a base shader ever needs a 1.2-only SPIR-V capability
we raise the default and record it. Variants needing higher SPIR-V (fp16 arithmetic, integer dot
product, cooperative matrix) or a known subgroup width (§7.4 rule 4) are compiled as **separate
variants** and selected at runtime from `Capabilities` — the same split llama.cpp uses for its
`_cm2` shaders. Never a single fat module with runtime-dead capabilities; some drivers validate
the whole module.

Note that shader targets are **independent of the barrier decision**: `synchronization2` is a
host-side API, not a SPIR-V capability. Nothing in `shaders/` changes because of §7.3.


---

## 8. Op coverage strategy and the v0 op set

### 8.1 Strategy

> **Authoritative coverage plan: [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse).** Justin has raised
> the ambition sharply — *"mlx 达到这样的 op coverage 只用了几天，不是两年，我们要 target 高 op
> coverage。当然 focus on llm，moe，multi modal，linear attention，qwen3.5，conv 这些类型的模型
> 优先。"* Coverage is therefore driven by **model families** (LLM/Qwen3.5 → MoE → multimodal →
> linear attention/SSM → conv), not by the incremental family list in §8.2. Mouse proposes;
> **Morpheus ratifies.** When `OP_COVERAGE.md` is ratified it **supersedes §8.2 and §8.3** as the
> sequencing plan, and §8.2 below is retained only as the M0/M1 floor — the minimum that must exist
> for the pipeline to be provable.
>
> **The constraints §8.1 imposes on `OP_COVERAGE.md` are not negotiable by the coverage plan.** An
> aggressive schedule changes *what order* ops land in and *how many* land per week. It does not
> change the claiming discipline, the fallback guarantee, or the fragmentation rule. Specifically,
> the coverage plan must honour:
>
> 1. **Conservative claiming (§1.3, §8.1 items 2–3).** Every claim predicate validates domain, op
>    type, opset range, arity, dtypes, attributes, static-shape availability, and broadcast form.
>    A partially-implemented op is an unclaimed op. Speed of coverage is never a reason to claim
>    something we cannot translate correctly for every input we accepted.
> 2. **Clean CPU fallback (§2.6, §5).** Declining must be free and silent. No op may be claimed
>    whose failure mode is an error at `Compile` or `Compute` time rather than a decline at
>    `GetCapability` time.
> 3. **A minimum viable subgraph size.** A claimed region must be large enough that its compute
>    outweighs the device round-trip at its boundary (§8.3). High op *count* that shreds a graph
>    into transfer-dominated fragments is a regression wearing a coverage badge. The metric of
>    record is Niobe's island count and largest fused region (§9.2), not the number of ops in the
>    registry.
> 4. **Every claimed op ships with its differential test and its platform row on the same PR**
>    (items 4–5 below). This is what makes a fast schedule survivable rather than a debt pile.
>
> A model-family-driven plan is the right instinct and I expect to ratify it. These four constraints
> are what I will check it against.

Mouse owns `docs/OP_COVERAGE.md`, `docs/OP_ARCHITECTURE.md` and the registry. The architectural
constraints on that work:

1. **One op = one handler + one claim predicate + one registration line, in one
   `ops/<family>.rs`.** Zero edits to `ep.rs`, `engine.rs`, or the registry core. If adding an op
   requires touching the boundary layer, the boundary layer is wrong — that is a bug report
   against L1, not a reason to edit it.
2. **Claim and translate share one table.** Claimed can never outrun translatable.
3. **Claim predicates validate everything:** domain, op type, opset range, input/output count and
   presence, dtypes, required attributes, static-shape availability, and broadcast form. When in
   doubt, do not claim.
4. **Every claimed op ships with a differential test against ORT CPU on the same PR.** No
   exceptions, no "tests in a follow-up."
5. **Every claimed op ships with a `PLATFORMS.md` row or an explicit "untested on X" note.** An op
   verified only on lavapipe is not verified.
6. **Coverage is grown in families, not alphabetically.** A family shares a shader skeleton, a
   descriptor layout, and a test file, so families amortize; scattered ops do not.
7. **fp32 first.** fp16 is a variant per family, gated on `shaderFloat16` +
   `storageBuffer16BitAccess`, added family by family once fp32 is green. No fp64, ever.

### 8.2 The M0/M1 op floor

> Superseded as a *plan* by [`OP_COVERAGE.md`](./OP_COVERAGE.md) once ratified (§8.1). Retained as
> the minimum that must exist for the pipeline to be provable; nothing here is a ceiling.

**M0 — one op, end to end.** `Add`, fp32, identical shapes, 2 inputs, 1 output, static shape.
That is the whole M0 claim set. It exists to prove the ABI, the device, the memory path, the
dispatch, and the test harness — not to be useful.

**M1 — the elementwise and shape families.**

| Family | Ops | Constraints |
|---|---|---|
| Binary elementwise | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max` | fp32; equal shapes or suffix/scalar broadcast |
| Unary elementwise | `Neg`, `Abs`, `Sqrt`, `Exp`, `Log`, `Reciprocal`, `Floor`, `Ceil`, `Round`, `Sign`, `Erf` | fp32 |
| Activations | `Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Elu`, `HardSigmoid`, `Softplus`, `Clip`, `Gelu` | fp32; `Clip` with constant or absent min/max |
| Comparison / logic | `Equal`, `Greater`, `Less`, `GreaterOrEqual`, `LessOrEqual`, `And`, `Or`, `Not`, `Where` | fp32/bool; same broadcast rule |
| Cast | `Cast` | fp32 ↔ int32 ↔ bool only |
| Shape (metadata-only) | `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten`, `Identity` | constant target shape; **no data movement** |
| Shape (copying) | `Transpose`, `Concat`, `Slice`, `Gather` | fp32/int32; constant axes/indices where the op allows |

Explicitly **not** claimed in M1, and each with a stated reason in the coverage table: anything
fp16/fp64, anything with a data-dependent shape, `Resize`, `Pad` with non-constant pads, and every
`com.microsoft` op.

**M2 — the compute families that make the EP worth using.**
`MatMul`, `Gemm`, `ReduceSum`/`ReduceMean`/`ReduceMax`/`ReduceMin`/`ReduceProd`, `Softmax`
(last-axis), `LogSoftmax`, `LayerNormalization`, `RMSNormalization`, `ArgMax`/`ArgMin`.
This is the first milestone where a real model (a small MLP, then a small CNN once `Conv` lands)
has a fused region big enough for a speedup to be meaningful rather than dispatch-bound.

### 8.3 The fragmentation rule

A new op is worth claiming when it **connects** existing claimed regions or **extends** one at the
edge. An op claimed in isolation, in the middle of a graph of unclaimed ops, makes the graph
*slower* — two extra device round-trips for one dispatch. Mouse prioritizes by "does this merge
two islands", not by "is this op easy." Niobe's benchmark must report island count and largest
fused region, not just wall time, so this is measurable rather than folklore.

---

## 9. Testing and benchmarking strategy

### 9.1 Differential testing against the ORT CPU EP — Trinity

The oracle is **ORT's own CPU EP**, running the same ONNX model. Not numpy, not a reference we
wrote. This is the single most important testing decision in the project: it means a test failure
is unambiguous, and it means we cannot accidentally encode our own misreading of an ONNX spec into
both the implementation and the expectation.

| Layer | Location | Purpose | Gate |
|---|---|---|---|
| Op correctness | `tests/ops/` (pytest) | Per-op, per-dtype, per-shape differential vs ORT CPU. Models built with the ONNX IR API in `_models.py`. | **Required on every PR.** |
| Claim assertion | `tests/ops/test_claim_diagnostics.py` | Asserts the node *actually ran on `VulkanExecutionProvider`*, via the claim diagnostics. Prevents vacuous CPU-fallback passes. | **Required.** |
| ONNX backend node tests | `tests/backend/` | The ONNX project's own node tests through the EP. | Required. |
| Conformance fuzzing | `tests/conformance/` | Bounded property-based fuzzing of claimed ops against the ONNX standard, one op per subprocess so a native crash cannot abort the run. | Opt-in `workflow_dispatch`. |
| Validation layers | all suites, debug builds | `VK_LAYER_KHRONOS_validation` clean is part of "done" for any engine change. | Required in the CI debug lane. |
| Leak / teardown | stress scripts across many sessions | RAII teardown leaves no `VkDeviceMemory`, no pipelines, no descriptor pools. | Required. |
| **Barrier-backend parity** | every lane, run twice | The full suite with the default backend and again with `ep.force_legacy_barriers=1`, asserting **identical** numerical results (§7.5 item 5). Without this, the legacy `vkCmdPipelineBarrier` path we carry for 31% of Android and 12% of Windows would never be executed by any test we own. | **Required on every PR.** |

**The vacuous-pass trap, stated plainly.** Because CPU fallback is always correct, a test that
merely compares outputs will pass whether or not the EP ran anything. Every op test **must** assert
the claim. This is non-negotiable and it is the first thing I will look for in a review.

**Tolerances.** Stated per family in `tests/ops/`, defaulting to `rtol=1e-5, atol=1e-5` for fp32
elementwise. Reductions and GEMM get looser tolerances tied to accumulation order — but a
tolerance is *derived and documented*, never widened to make a red test green. Widening a
tolerance requires Trinity's sign-off and a note in the test.

**Cross-platform.** The same suite runs on every CI lane: lavapipe (Linux), SwiftShader (Windows),
and any GPU runner we acquire. A pass on lavapipe alone is a smoke test, not a correctness claim —
software rasterizers do not reproduce driver-specific subgroup, denorm, or precision behaviour.
Link owns which lanes exist; Trinity owns what runs on them.

### 9.2 Benchmarking — Niobe

- **Baselines are versus the ORT CPU EP on the same machine, same model, same ORT build.** Any
  other comparison is marketing.
- `bench/` reuses `tests/ops/_models.py` builders so the benchmark cannot drift from what is
  tested.
- Reported per case: median wall time on Vulkan, median on CPU, ratio, **and** the claim
  diagnostics — island count, largest fused region, node count claimed. A speedup number without
  those three is not accepted.
- GPU-side timing uses `VkQueryPool` timestamp queries once the engine exposes them, so we can
  separate submit overhead from actual GPU time. Sub-millisecond cases are dispatch-bound and will
  be slower than CPU; that is expected, must be labelled, and must not be hidden.
- `bench.yml` posts an informational base-vs-PR table. **It does not gate**, because shared-runner
  timings are noise. It flags a regression as a prompt to re-measure locally.
- **No performance claim leaves this repo before the corresponding op is green in `tests/ops/` on
  at least one real GPU.**

---

## 10. Milestones

Each milestone's exit criteria are verifiable by a command, not by an opinion.

### M0 — "It loads, it runs, it matches"

> **A stock ORT loads the plugin, enumerates a Vulkan device, runs a graph containing a single
> `Add` node on that device, and the output matches the ORT CPU EP within tolerance — on both
> Windows and Linux, on a software rasterizer, in CI.**

| Work | Owner |
|---|---|
| `Cargo.toml`, `build.rs` (ORT bindgen + GLSL→SPIR-V embedding), crate scaffolding | Tank |
| `lib.rs`, `factory.rs`, `ep.rs` — factory/EP/compute-info vtables, panic guards, RAII teardown | Tank |
| `vk/` — instance, physical-device scoring + capability gate (§7.2), device, queue, allocator, staging, descriptor pool, pipeline cache, command recording, single-fence submit | Switch |
| **`vk/barrier.rs` — the barrier seam of §7.5: `Access`/`BufferDep` enums, `Barriers::select`, and *both* the `Sync2Backend` and `LegacyBackend` implementations** | Switch |
| `vk/caps.rs` — `Capabilities` probe incl. `synchronization2`, `subgroup_size_control` **properties-only** query (§7.4), subgroup ops, fp16 | Switch |
| `shaders/elementwise_binary.comp` + the SPIR-V embedding pipeline | Switch |
| `registry.rs` + `NodeView`; `ops/elementwise.rs` with `Add` claim + handler; claim diagnostics | Mouse |
| `engine.rs` — `NodeDesc`, `Plan`, `DispatchContext`; per-run command recording (no cache yet) | Tank + Morpheus (contract) |
| `tests/ops/conftest.py`, `_models.py`, `test_elementwise.py`, claim assertion helper | Trinity |
| CI: fmt, clippy, build on windows-latest + ubuntu-latest, lavapipe + SwiftShader lanes, **the `ep.force_legacy_barriers=1` duplicate lane (§7.5 item 5)**, Vulkan SDK provisioning, layering lint | Link |
| `python/` package with `register_execution_provider_library()` | Tank |
| Baseline harness stub; no numbers published | Niobe |

**Exit criteria.**
1. `cargo build --release` and `cargo clippy -- -D warnings` clean on Windows and Linux.
2. `pytest tests/ops -q` green on both, with the claim assertion proving `Add` ran on
   `VulkanExecutionProvider`.
3. Validation layers report zero errors and zero warnings in the debug lane.
4. A machine with no Vulkan ICD loads the plugin, advertises zero devices, logs a warning, and the
   session still runs on CPU.
5. `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` prints per-op decline reasons.
6. The layering lint is in CI and fails a deliberately-planted violation, including a planted
   `cmd_pipeline_barrier` outside `vk/barrier.rs` (§4.2, §7.5).
7. **The full test suite passes twice per lane — once with the default barrier backend and once
   with `ep.force_legacy_barriers=1` — with identical numerical results** (§7.5 item 5).
8. Both sibling docs and this one are consistent; §12 lists every divergence.

### M1 — "A useful elementwise EP"

The §8.2 M1 op set, claimed, tested, and documented; shape-keyed command-buffer recording cache
(`recorded.rs`); convex clustering with multi-node fused subgraphs; the `OP_ARCHITECTURE.md`
coverage table as the authoritative contract.

| Work | Owner |
|---|---|
| Op families + claim predicates + coverage table | Mouse |
| Shader variants, spec constants, broadcast indexing, descriptor layout per family | Switch |
| Convex clustering, `recorded.rs`, arena-based temporaries | Tank |
| Per-family differential tests, tolerance policy, `tests/backend/` node tests | Trinity |
| `bench/` harness + first published baselines with island counts | Niobe |
| macOS/MoltenVK lane; first real-GPU lane if a runner is available; driver quirk log | Link |

**Exit criteria.** Every M1 op green vs CPU on ≥2 platforms; a 10-node elementwise chain fuses into
**one** subgraph and **one** submission; a shape change re-records once and then replays;
`OP_ARCHITECTURE.md` matches the registry exactly (checked by a test that diffs them).

### M2 — "Real memory, real compute"

Device allocator + data transfer (§6.3), `GetDefaultMemoryDevice` returning a real device, and the
compute families from §8.2 M2. This is the first milestone that can honestly claim a speedup.

| Work | Owner |
|---|---|
| `allocator.rs`, `transfer.rs`, handle registry, `OrtMemoryDevice` wiring | Tank |
| Device-local arena, staging ring, coherence handling, barrier batching | Switch |
| `MatMul`/`Gemm`/reductions/`Softmax`/norms — semantics, claim, tiling requirements | Mouse + Switch |
| Timestamp queries and the trace exporter | Switch + Niobe |
| Numerical tolerance policy for accumulation-order-sensitive ops | Trinity |
| GPU CI lane; per-vendor result matrix | Link |
| First published speedup table, with island counts and CPU baseline | Niobe |

**Exit criteria.** A small MLP and a small CNN-shaped graph run majority-on-Vulkan; with
`IoBinding`, per-inference host↔device traffic is **zero** for a fully-claimed graph; a measured
speedup vs the ORT CPU EP on at least one real GPU, published with methodology.

### M3+ — "Breadth and platforms"

Android (NDK cross-compile, Adreno + Mali validation — **including deliberate validation on the
Adreno 5xx / Mali Bifrost population that §7.3 bought us, which is the whole point of carrying the
legacy barrier backend**), fp16 variants across families, `Conv` and pooling, persistent activation
buffers, shapeless recording for dynamic dimensions, graph-level fusion patterns, quantized ops,
attention. Sequencing is decided at the M2 retrospective, informed by what Niobe's numbers,
Link's matrix, and the ratified `OP_COVERAGE.md` actually say — not scheduled now.

---

## 11. Open questions

| # | Question | Decided by | Blocks |
|---|---|---|---|
| **OQ-1** | ~~How many real devices report Vulkan 1.1/1.2 **without** `VK_KHR_synchronization2` or `VK_EXT_subgroup_size_control`?~~ **RESOLVED 2026-07-28T19:16:08-07:00.** Link measured it (`PLATFORMS.md` §8, vulkan.gpuinfo.org 2026-07-28): `VK_KHR_synchronization2` is missing on **31.43% of Android** and **12.22% of Windows**; `VK_EXT_subgroup_size_control` on **14.12% of Android**, and its *feature flag* is `VK_FALSE` on all of macOS/iOS. **Ruling (§7.2–§7.5): both are dropped from the hard requirement.** `synchronization2` becomes a probed capability selecting one of two barrier backends behind a single seam (`vk/barrier.rs`); `subgroup_size_control` is consulted as a *properties query* only and never as a required feature. Link's layer-shim option is **rejected** — the AOSP loader cannot discover a layer we ship from a plugin `.so`, and the cited wgpu/Dawn/Godot precedent turned out to be legacy-barrier-only in all three. | Link investigated → **Morpheus decided** | — (§7 is frozen) |
| **OQ-2** | ~~Do llama.cpp and ExecuTorch's stated version floors survive verification?~~ **RESOLVED 2026-07-28T17:59:54-07:00.** Fact Checker claims 1–2: both "requires 1.3" claims **contradicted**. llama.cpp base shaders target `vulkan1.2` (only `_cm2` variants target 1.3); ExecuTorch hardcodes `VK_API_VERSION_1_1`. Claim 4 (Android share) remains *unverified but plausible*. | **Fact Checker** (done) | — |
| **OQ-3** | The ORT allocator's pointer problem (§6.3): ORT allocators return `void*`, a Vulkan allocation is a `(VkBuffer, offset)` pair. Three candidates now: (a) an **opaque-handle registry** — a monotonic token cast to `*mut c_void`, resolved through a side table (provisional); (b) `VK_KHR_buffer_device_address`; (c) **ORT 1.28's `CreateExternalResourceImporterForDeviceImpl`** — *live alternative, added 2026-07-28T19:16:08-07:00*. If it does what its name suggests — importing an externally-allocated device resource such as a `VkBuffer` into ORT without a host round-trip — it is a **better answer than (a)**, because it makes the mapping ORT's contract rather than our side table, and it removes the "a pointer we hand out must never be dereferenced" hazard entirely. **Not resolved:** Fact Checker is verifying the exact symbol and semantics; Tank is building `sys.rs` against the 1.28 headers in parallel. Note this also reopens the ORT ABI version pinned in §0 (currently `ORT_API_VERSION 27`) — adopting (c) means a 1.28 floor, which is its main cost. | Fact Checker verifies → Tank proposes → **Morpheus decides** | M2 |
| **OQ-4** | Shader compilation: build-time `glslc` from the Vulkan SDK (SDK becomes a build dependency) vs checked-in pre-generated SPIR-V (reviewable diffs, but binary artifacts in git) vs both with SDK preferred. Provisionally: build-time with checked-in fallback. | **Switch** proposes → Link validates on all CI lanes → Morpheus decides | M0 |
| **OQ-5** | `gpu-allocator` vs a hand-rolled suballocator. `ENGINE.md` §3.1 picks `gpu-allocator`; I concur provisionally. Confirm it cross-compiles cleanly for Android and works under MoltenVK. | **Switch** owns → Link validates | M0/M3 |
| **OQ-6** | What vendor ID does the factory report when it advertises zero devices, or before a device is bound? ORT calls `GetVendorId` on the factory, not per device. | **Tank** proposes → Morpheus decides | M0 |
| **OQ-7** | Do we need a real GPU CI runner for M2's exit criteria, and if so, self-hosted or a cloud GPU lane? Software rasterizers cannot validate a speedup claim. | **Link** proposes → Justin decides (cost) | M2 |
| **OQ-8** | Is `com.microsoft` contrib-op support ever in scope? It is where the MLX EP's value concentrated (MatMulNBits, GQA), but it is a much larger surface. | **Morpheus**, at the M2 retrospective | M3+ scope |
| **OQ-9** | Threading model: one `VkDevice` per session (chosen) vs a process-shared device with a mutex. Sharing saves memory and pipeline-cache warmth for multi-session hosts. | **Tank + Switch** propose → Morpheus decides | post-M2 |
| **OQ-10** | Tolerance policy for accumulation-order-sensitive ops (GEMM, reductions) across vendors, where fp32 associativity differs. Needs a stated, derived rule before M2's ops land, not after. | **Trinity** proposes → Morpheus ratifies | M2 |
| **OQ-11** | Ratification of `OP_COVERAGE.md` (§8.1) — does the model-family-driven, high-ambition coverage plan honour conservative claiming, clean CPU fallback, and the minimum-viable-subgraph rule? Raised 2026-07-28T19:16:08-07:00. | **Mouse** proposes → **Morpheus ratifies** | §8 supersession; M1 sequencing |
| **OQ-12** | Does carrying the legacy barrier backend (§7.3) actually buy usable devices, or does the Adreno 5xx / Mali Bifrost population fail us for some *other* reason (driver bugs, `maxComputeWorkGroupInvocations`, absent subgroup ARITHMETIC, unusably slow)? If the answer is "these devices are unusable anyway", the legacy backend is still correct for the 12.22% Windows gap, but the Android argument weakens. Must be answered with a real device, not a database. Raised 2026-07-28T19:16:08-07:00. | **Link** measures → Niobe benchmarks → Morpheus reviews | M3 Android scope |

---

## 12. Divergences from the `onnxruntime-mlx` reference

Every deliberate difference, with its reason. Anything not listed here is intended to match the
reference.

| # | Divergence | Reason |
|---|---|---|
| D1 | **Real device allocator + data transfer** (`allocator.rs`, `transfer.rs`); MLX returns null stubs. | No unified memory. Without them, every partition boundary is a full host round-trip. Phased: null in M0/M1, real in M2 (§6.3). |
| D2 | **`GetSupportedDevices` advertises N devices, with a capability gate**; MLX advertises one. | Multi-GPU is normal on Windows/Linux, and an unusable Vulkan device must never be offered. |
| D3 | **A whole Vulkan engine layer (`vk/`) that has no MLX counterpart.** MLX supplies scheduling, memory, and op semantics; Vulkan supplies none. | Unavoidable. It is also why Rule 2 (§4.2) is enforced rather than encouraged. |
| D4 | **`recorded.rs` (command-buffer recording cache) replaces `compiled.rs` (`mlx_compile` tracing).** | Same intent — pay graph construction once, replay many. Different mechanism: Vulkan's unit of replay is a recorded `VkCommandBuffer`, following ExecuTorch's model rather than llama.cpp's re-record-every-eval. |
| D5 | **Weight prepacking is a first-class `Compile` step**, not a first-`Run` cache fill. | Vulkan uploads are explicit and expensive; ORT gives us initializer bytes at compile time; doing it lazily would put a staging copy on the first inference for no benefit. |
| D6 | **`shaders/` directory and a SPIR-V build step.** MLX explicitly *deleted* its `.metal` kernels. | We are the backend. This is the one place where the MLX project's history is an anti-pattern for us — its lesson was "don't hand-write kernels when a good backend exists", and for Vulkan no such backend exists. |
| D7 | **fp32-only v0; fp16 as a per-family gated variant.** MLX is dtype-generic for free. | MLX carries dtype through its ops with no per-dtype code. Every Vulkan dtype is a separate SPIR-V variant plus a device feature probe. |
| D8 | **No contrib-domain (`com.microsoft`) ops in v1.** MLX's highest-value ops are contrib ops. | Those are the *hardest* ops to write from scratch (int4 packing, GQA/KV-cache). Doing them first would mean a year before anything is verifiable. OQ-8. |
| D9 | **Vendor ID is read from the bound device**, not hardcoded to one vendor. | Cross-platform mandate. |
| D10 | **Validation layers are part of the definition of done.** No MLX equivalent. | Vulkan's error surface is enormous and mostly silent without layers; MLX's C API validates for us. |
| D11 | **`ash` (safe-ish Rust Vulkan bindings) rather than bindgen over `vulkan.h`.** MLX bindgens `mlx-c` directly. | `ash` is the ecosystem standard, handles the loader/extension-function-pointer problem correctly, and removes a large class of hand-written FFI bugs. The ORT side still uses bindgen, matching the reference. |
| D12 | **Two barrier backends behind one internal seam (`vk/barrier.rs`), selected once at device init**, plus a session option to force the legacy one. MLX has no counterpart — MLX's C API owns all synchronization. | §7.3. Requiring `synchronization2` would exclude 31.43% of Android and 12.22% of Windows; the compatibility-first directive makes that unacceptable. The cost is one file with two implementations and a doubled CI lane, which is bounded; the cost of the alternative is permanent device exclusion. wgpu, Dawn and Godot all ship legacy-only barriers, so this is the mainstream position, not an exotic one. |

---

## 13. References

- **Reference architecture:** `onnxruntime-mlx` — `docs/DESIGN.md`, `docs/OP_ARCHITECTURE.md`,
  `docs/COMPILED_CAPTURE.md`, `rust/src/{lib,factory,ep,engine,registry,compiled,sys}.rs`,
  `rust/{Cargo.toml,build.rs}`, `tests/`, `bench/`, `python/`.
- **Sibling docs:** [`ENGINE.md`](./ENGINE.md) (Switch), [`PLATFORMS.md`](./PLATFORMS.md) (Link),
  [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse), `OP_ARCHITECTURE.md` (Mouse, forthcoming),
  `BENCHMARKS.md` (Niobe, forthcoming).
- **Vulkan layer deployment (§7.3 ruling):** `KhronosGroup/Vulkan-Loader`
  `docs/LoaderLayerInterface.md` and `docs/LoaderApplicationInterface.md` (Android layer discovery;
  "The Android loader does not use manifest files"; elevated-privilege `secure_getenv` caveats),
  `loader/loader_environment.c`; `KhronosGroup/Vulkan-ExtensionLayer`
  `docs/synchronization2_layer.md` ("needs to be packaged inside the APK");
  `developer.android.com/ndk/guides/graphics/validation-layer`.
- **Barrier prior art (§7.3):** `gfx-rs/wgpu` `wgpu-hal/src/vulkan/command.rs`
  (`cmd_pipeline_barrier`, legacy only); `google/dawn`
  `src/dawn/native/vulkan/CommandBufferVk.cpp` (`fn.CmdPipelineBarrier`, legacy only);
  `godotengine/godot` `drivers/vulkan/rendering_device_driver_vulkan.cpp` (`vkCmdPipelineBarrier`,
  legacy only). None ships `VK_LAYER_KHRONOS_synchronization2`.
- **ORT plugin-EP C ABI:** `onnxruntime_ep_c_api.h`, `RegisterExecutionProviderLibrary`,
  `SessionOptionsAppendExecutionProvider_V2`, `CreateEpFactories`, `ReleaseEpFactory`.
- **Prior art — llama.cpp Vulkan backend:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
  `vulkan-shaders/vulkan-shaders-gen.cpp`. Per-node eager dispatch with graph-level fusion passes;
  re-records command buffers every eval; buffers only, no VMA; hard runtime floor Vulkan 1.2; base
  shaders compiled at `--target-env=vulkan1.2` with `vulkan1.3` reserved for cooperative-matrix-2
  variants; build-time SPIR-V with runtime binary patching.
- **Prior art — ExecuTorch Vulkan backend:** `backends/vulkan/`. Ahead-of-time partitioning
  (`vulkan_partitioner.py`), serialized graph, `prepack()` at init, command buffer recorded once
  and replayed, buffers **and** image textures, VMA, on-disk `VkPipelineCache`, hard floor
  Vulkan 1.1.
- **Decision records:** `.squad/decisions/inbox/morpheus-architecture-v0.md`,
  `.squad/decisions/inbox/morpheus-oq1-resolution.md`,
  `.squad/decisions/inbox/link-oq1-extension-availability.md`.
