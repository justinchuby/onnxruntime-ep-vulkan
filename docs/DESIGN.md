# onnxruntime-ep-vulkan — Architecture Design

**Status:** v0 architecture of record — accepted for M0/M1 implementation
**Date:** 2026-07-28T17:59:54-07:00
**Author:** Morpheus (Lead / EP Architect)
**Repo:** `onnxruntime-ep-vulkan`
**Reference architecture:** `onnxruntime-mlx` (Justin Chu's MLX plugin EP for Apple Silicon)
**Sibling documents:** [`ENGINE.md`](./ENGINE.md) (Switch — Vulkan runtime & shaders), [`PLATFORMS.md`](./PLATFORMS.md) (Link — platform & hardware matrix)

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

*Enforcement:* the same CI lint greps `rust/src/ops/` for `ash`, `vk::`, and `unsafe`.

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

> **Status: reconciled, with one item outstanding.** Link's [`PLATFORMS.md`](./PLATFORMS.md) §4,
> Switch's [`ENGINE.md`](./ENGINE.md) §8, and Fact Checker's audit trail
> (`.squad/fact-checker/audit-trail.md`, claims 1–5) all exist and are incorporated below. The one
> item still open is **OQ-1** — how many real devices report Vulkan 1.1/1.2 *without*
> `VK_KHR_synchronization2` or `VK_EXT_subgroup_size_control`. §7.2 is the working decision and
> becomes final when Link answers OQ-1.

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

The premise that motivated 1.3 — "llama.cpp requires it" — is contradicted by llama.cpp's own
source at both the runtime check and the shader target, and independently verified as contradicted
by Fact Checker. That does not make 1.3 wrong; it makes the *reason* wrong, and I would rather we
decide this on the two features Switch identified than on a misattribution.


### 7.2 Decision

**We require a capability set, not a version number.**

A physical device is advertised to ORT if and only if it satisfies all of:

| Requirement | Rationale |
|---|---|
| Vulkan **≥ 1.1** core (instance and device) | `VkPhysicalDeviceFeatures2` / property chains in core; subgroup properties in core; the Android floor. |
| A queue family with `VK_QUEUE_COMPUTE_BIT` | Obvious. |
| **`synchronization2`** — core in 1.3 *or* `VK_KHR_synchronization2` | Switch's #1 simplification. One barrier code path, not two. |
| **`subgroup_size_control`** — core in 1.3 *or* `VK_EXT_subgroup_size_control` | Switch's #2. A wrong subgroup size silently produces wrong results in cooperative GEMM shaders; guessing is not acceptable. |
| Subgroup `BASIC` + `ARITHMETIC` ops in the `COMPUTE` stage | Required by reductions, softmax, and GEMM. |
| `maxComputeWorkGroupInvocations ≥ 256`, `maxComputeSharedMemorySize ≥ 16 KiB` | Below this, our tiling assumptions do not hold. |

`VkApplicationInfo::apiVersion` is set to `min(vkEnumerateInstanceVersion(), VK_API_VERSION_1_3)`
— llama.cpp's pattern. Every feature above the required set — `shaderFloat16`,
`storageBuffer16BitAccess`, `shaderInt8`, timeline semaphores, `bufferDeviceAddress`, cooperative
matrix, integer dot product — is **capability-probed at runtime** through a single
`vk::caps::Capabilities` struct and gates a shader variant, never a hard requirement.

### 7.3 Why this and not the alternatives

**Why not a hard 1.3 baseline (Justin's proposal).** It buys us exactly the two features above,
which we get anyway as extensions. It costs roughly **36 points of Android installed-base
coverage** (Fact Checker: 26% at 1.3 vs 62% at 1.1; Link reports the same gap against a
Vulkan-capable denominator) and any MoltenVK older than 1.3.0 — for zero engine simplification,
because our required set already guarantees a single barrier path. Adopting it would also mean the
project's own stated cross-platform mandate ("Windows, Linux, Android, macOS") comes with an
unwritten asterisk. If we ever want that asterisk, it should be a scope decision with a decision
record, not a side effect of an API constant.

**Why not a hard 1.2 baseline (Link's recommendation).** Link is right that 1.2 is a sane desktop
balance, but on Android the 1.2 tier barely exists — devices jumped 1.1 → 1.3. So a 1.2 floor
pays nearly the full Android cost of a 1.3 floor while getting less than 1.3 gives on desktop.
The only thing 1.2 core adds that we care about is timeline semaphores, which v0 does not use
(single-queue fence model, `ENGINE.md` §6.3) and which are available as
`VK_KHR_timeline_semaphore` on 1.1 when we do.

**Why not a bare 1.1 floor (ExecuTorch's position).** It would force the dual-barrier path Switch
warned about and leave subgroup size a driver's choice. The extension requirement buys us both
guarantees at a small, measurable cost in device coverage — and unlike a version bump, we can
*measure* that cost: Link can enumerate how many real devices expose 1.1 without
`VK_KHR_synchronization2`. That is OQ-1.

**In practice, on the platforms we test first, this is Justin's 1.3 baseline.** Every desktop
driver from 2022 onward, lavapipe, SwiftShader (both verified 1.3-conformant), and MoltenVK 1.3.0+
report 1.3 and satisfy the set trivially. The difference only shows up on older Android, and there
it shows up as "works" or "cleanly declines" rather than "fails to load."

### 7.4 Shader targets

SPIR-V is compiled with `--target-env=vulkan1.1` by default (SPIR-V 1.3), which every device
meeting the requirement above can consume. This is one notch below llama.cpp's `vulkan1.2` default
and is the conservative choice for Android breadth; if a base shader ever needs a 1.2-only SPIR-V
capability we raise the default and record it. Variants needing higher SPIR-V (fp16 arithmetic,
integer dot product, cooperative matrix) are compiled as **separate variants at a higher target**
and selected at runtime by `Capabilities` — the same split llama.cpp uses for its `_cm2` shaders.
Never a single fat module with runtime-dead capabilities; some drivers validate the whole module.


---

## 8. Op coverage strategy and the v0 op set

### 8.1 Strategy

Mouse owns `docs/OP_ARCHITECTURE.md` and the registry. The architectural constraints on that work:

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

### 8.2 The v0 op set (M0–M1)

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
| `vk/` — instance, physical-device scoring + capability gate, device, queue, allocator, staging, descriptor pool, pipeline cache, command recording, single-fence submit | Switch |
| `shaders/elementwise_binary.comp` + the SPIR-V embedding pipeline | Switch |
| `registry.rs` + `NodeView`; `ops/elementwise.rs` with `Add` claim + handler; claim diagnostics | Mouse |
| `engine.rs` — `NodeDesc`, `Plan`, `DispatchContext`; per-run command recording (no cache yet) | Tank + Morpheus (contract) |
| `tests/ops/conftest.py`, `_models.py`, `test_elementwise.py`, claim assertion helper | Trinity |
| CI: fmt, clippy, build on windows-latest + ubuntu-latest, lavapipe + SwiftShader lanes, Vulkan SDK provisioning, layering lint | Link |
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
6. The layering lint is in CI and fails a deliberately-planted violation.
7. Both sibling docs and this one are consistent; §12 lists every divergence.

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

Android (NDK cross-compile, Adreno + Mali validation, the 1.1-without-extensions coverage question
from OQ-1), fp16 variants across families, `Conv` and pooling, persistent activation buffers,
shapeless recording for dynamic dimensions, graph-level fusion patterns, quantized ops, attention.
Sequencing is decided at the M2 retrospective, informed by what Niobe's numbers and Link's matrix
actually say — not scheduled now.

---

## 11. Open questions

| # | Question | Decided by | Blocks |
|---|---|---|---|
| **OQ-1** | How many real devices report Vulkan 1.1/1.2 **without** `VK_KHR_synchronization2` or `VK_EXT_subgroup_size_control`? If that set is large on Android, §7.2's requirement needs the dual barrier path after all. | Link investigates → **Morpheus decides** | Final §7 sign-off; M3 |
| **OQ-2** | ~~Do llama.cpp and ExecuTorch's stated version floors survive verification?~~ **RESOLVED 2026-07-28T17:59:54-07:00.** Fact Checker claims 1–2: both "requires 1.3" claims **contradicted**. llama.cpp base shaders target `vulkan1.2` (only `_cm2` variants target 1.3); ExecuTorch hardcodes `VK_API_VERSION_1_1`. Claim 4 (Android share) remains *unverified but plausible*. | **Fact Checker** (done) | — |
| **OQ-3** | Opaque-handle registry vs `VK_KHR_buffer_device_address` for the ORT allocator's pointer problem (§6.3). Provisionally the registry; BDA if profiling ever justifies it *and* MoltenVK coverage improves. | Tank proposes → **Morpheus decides** | M2 |
| **OQ-4** | Shader compilation: build-time `glslc` from the Vulkan SDK (SDK becomes a build dependency) vs checked-in pre-generated SPIR-V (reviewable diffs, but binary artifacts in git) vs both with SDK preferred. Provisionally: build-time with checked-in fallback. | **Switch** proposes → Link validates on all CI lanes → Morpheus decides | M0 |
| **OQ-5** | `gpu-allocator` vs a hand-rolled suballocator. `ENGINE.md` §3.1 picks `gpu-allocator`; I concur provisionally. Confirm it cross-compiles cleanly for Android and works under MoltenVK. | **Switch** owns → Link validates | M0/M3 |
| **OQ-6** | What vendor ID does the factory report when it advertises zero devices, or before a device is bound? ORT calls `GetVendorId` on the factory, not per device. | **Tank** proposes → Morpheus decides | M0 |
| **OQ-7** | Do we need a real GPU CI runner for M2's exit criteria, and if so, self-hosted or a cloud GPU lane? Software rasterizers cannot validate a speedup claim. | **Link** proposes → Justin decides (cost) | M2 |
| **OQ-8** | Is `com.microsoft` contrib-op support ever in scope? It is where the MLX EP's value concentrated (MatMulNBits, GQA), but it is a much larger surface. | **Morpheus**, at the M2 retrospective | M3+ scope |
| **OQ-9** | Threading model: one `VkDevice` per session (chosen) vs a process-shared device with a mutex. Sharing saves memory and pipeline-cache warmth for multi-session hosts. | **Tank + Switch** propose → Morpheus decides | post-M2 |
| **OQ-10** | Tolerance policy for accumulation-order-sensitive ops (GEMM, reductions) across vendors, where fp32 associativity differs. Needs a stated, derived rule before M2's ops land, not after. | **Trinity** proposes → Morpheus ratifies | M2 |

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

---

## 13. References

- **Reference architecture:** `onnxruntime-mlx` — `docs/DESIGN.md`, `docs/OP_ARCHITECTURE.md`,
  `docs/COMPILED_CAPTURE.md`, `rust/src/{lib,factory,ep,engine,registry,compiled,sys}.rs`,
  `rust/{Cargo.toml,build.rs}`, `tests/`, `bench/`, `python/`.
- **Sibling docs:** [`ENGINE.md`](./ENGINE.md) (Switch), [`PLATFORMS.md`](./PLATFORMS.md) (Link),
  `OP_ARCHITECTURE.md` (Mouse, forthcoming), `BENCHMARKS.md` (Niobe, forthcoming).
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
- **Decision records:** `.squad/decisions/inbox/morpheus-architecture-v0.md`.
