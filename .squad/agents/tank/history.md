# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Runtime & FFI — ORT plugin EP C ABI, sys/ep/factory, build & packaging
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

📌 Team update (2026-07-28T17:59:54-07:00): Vulkan API baseline is a capability-set, not a version floor — device is advertised if it has Vulkan ≥1.1 core + a compute queue + `synchronization2` (ext or 1.3 core) + `subgroup_size_control` (ext or 1.3 core) + subgroup BASIC+ARITHMETIC + `maxComputeWorkGroupInvocations ≥ 256` + `maxComputeSharedMemorySize ≥ 16 KiB`. Tank's device-enumeration code must enforce exactly these capability checks; everything else is optional probe. `VkApplicationInfo::apiVersion = min(vkEnumerateInstanceVersion(), VK_API_VERSION_1_3)`. — decided by Morpheus, Switch, Link, Fact Checker

📌 Team update (2026-07-28T17:59:54-07:00): Vulkan crate stack is `ash` + `gpu-allocator`. Tank's `Cargo.toml` must include these; `vulkano` and `wgpu` are rejected. `gpu-allocator` is the pure-Rust VMA equivalent. — decided by Switch

📌 Team update (2026-07-28T17:59:54-07:00): `build.rs` (Tank's responsibility) must locate and invoke `glslc`, iterate `shaders/glsl/`, write SPIR-V to `OUT_DIR/spv/`, and generate `OUT_DIR/shader_modules.rs`. No runtime shader compiler in the deployed artifact. — decided by Switch

📌 Team update (2026-07-28T17:59:54-07:00): ORT Plugin EP C API (`OrtEpFactory`, `CreateEpFactories`) is experimental since ORT 1.22, no ABI stability guarantee. Strategy: pin to a specific ORT version; invest early in an FFI abstraction layer so breakages are contained. — decided by Fact Checker

📌 Team update (2026-07-28T17:59:54-07:00): M0 definition — stock ORT loads the plugin, enumerates a Vulkan device, runs a graph with a single `Add` node, matches ORT CPU EP within tolerance, on Windows and Linux, on a software rasterizer, in CI. Tank wires the ORT FFI boundary for this milestone. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): No existing Vulkan EP for ORT and no Rust crate for ORT plugin-EP bindings. We write raw FFI from scratch. — decided by Fact Checker

---

### M0 foundation — the `rust/` crate landed (2026-07-28T19:16:08-07:00)

**`onnxruntime_ep_c_api.h` has no include guard.** Including it explicitly in a bindgen wrapper
header alongside `onnxruntime_c_api.h` (which already includes it) produces ~20 clang
"redefinition of `OrtDataTransferImpl`" errors. `rust/wrapper_ort.h` must include **only**
`onnxruntime_c_api.h`. Cost an hour; will recur on every ORT re-vendor if forgotten.

**Bindings are bindgen over vendored headers, not hand-written.** The plugin-EP types are `repr(C)`
vtables — `OrtApi` has several hundred function-pointer fields, and `OrtEp`/`OrtEpFactory` each
gained fields in 1.23, 1.24 and 1.28. Wrong field *order* compiles, loads, and then calls the wrong
function pointer: silent UB. Vendoring the headers (`third_party/onnxruntime/`, tag `v1.28.0`,
commit `da9b5e364c465de65c49d91e696cd6485270757f`, MIT) keeps builds reproducible and offline while
bindgen guarantees layout fidelity. **Consequence for CI: every runner needs LLVM/libclang**
(`LIBCLANG_PATH=C:\Program Files\LLVM\bin` on Windows).

**ORT pinned at 1.28.0 / `ORT_API_VERSION` 28; crate version `0.28.0`.** Two compile-time
assertions in `sys.rs` (header version, and crate minor version) plus a runtime `GetApi(28)` that
**refuses to load** on null. Never fall back to a lower API version — a lower `n` returns a
differently-shaped vtable and calling through it is UB.

**ORT 1.28 vtable field order, verified from the headers** (write it down; it is the thing bindgen
protects us from getting wrong):
`OrtEp` = ort_version_supported, GetName, GetCapability, Compile, ReleaseNodeComputeInfos,
GetPreferredDataLayout, ShouldConvertDataLayoutForOp, SetDynamicOptions, OnRunStart, OnRunEnd,
CreateAllocator, CreateSyncStreamForDevice, GetCompiledModelCompatibilityInfo, GetKernelRegistry,
IsConcurrentRunSupported, Sync, CreateProfiler, IsGraphCaptureEnabled, IsGraphCaptured, ReplayGraph,
GetGraphCaptureNodeAssignmentPolicy, GetAvailableResource, OnSessionInitializationEnd,
GetDefaultMemoryDevice, ReleaseCapturedGraph.
`OrtEpFactory` = ort_version_supported, GetName, GetVendor, GetSupportedDevices, CreateEp, ReleaseEp,
GetVendorId, GetVersion, ValidateCompiledModelCompatibilityInfo, CreateAllocator, ReleaseAllocator,
CreateDataTransfer, IsStreamAware, CreateSyncStreamForDevice,
GetHardwareDeviceIncompatibilityDetails, CreateExternalResourceImporterForDevice,
GetNumCustomOpDomains, GetCustomOpDomains, InitGraphicsInterop, DeinitGraphicsInterop,
SelectBestModelCandidate.
`OrtNodeComputeInfo` = ort_version_supported, CreateState, Compute, ReleaseState.

**`OrtEp::Compile` takes `*mut *const OrtGraph` / `*mut *const OrtNode`**, not `*const *const`.
The mutability is on the outer pointer. Cost a compile error; bindgen caught it, which is the whole
argument for bindgen in one line.

**`Logger_LogMessage` takes `file_path: *const wchar_t`** — `u16` on Windows, `u32` on Unix. We pass
null (ORT reads that as "no source location") rather than write cfg-gated wide-string conversion.

**`GetSessionConfigEntry` is a two-call protocol.** First call with a null buffer and `size = 0` to
query the length — it returns a status you must release *even on the success path*. Then call again
with a buffer of that size. Getting this wrong leaks an `OrtStatus` per option per session.

**`clippy::undocumented_unsafe_blocks` is positional.** The `// SAFETY:` comment must be on the line
immediately preceding `unsafe {`, not merely above the statement. Two clean-looking blocks failed
`-D warnings` for this. The generated bindings need the lint allowed at the `mod ort` level.

**Layering rules are enforced by `rust/tests/layering.rs`, not by an attribute.** No `deny` can
express "this directory may not name this crate" — `ash` is a legitimate dependency. A test cannot
be forgotten, since `cargo test` already runs it. The comment/string stripper is load-bearing:
`src/ops/mod.rs` documents the rules by naming every forbidden token. Verified live — a planted
`src/ops/planted_violation.rs` produced 7 findings across both rules before removal.

**`GetSupportedDevices` with no Vulkan returns success + zero devices, never an error.** An error
fails session creation, turning "no GPU here" into "your model does not load". Same reflex in
`GetCapability`: decline every node inside a control-flow subgraph body (non-null
`Graph_GetParentNode`), or ORT raises `INVALID_GRAPH` "no opset import for domain" — our bug
presenting as the user's model bug.

**`CreateExternalResourceImporterForDevice` does not answer OQ-3.** It is caller-driven (the app
exports a `VkDeviceMemory`, we import it); OQ-3 is EP-driven (what `void*` do *we* hand ORT for
memory *we* allocated?). Different directions, no overlap. The opaque-handle registry stands. The
importer slot is left `None`, which ORT reads as "cannot import external memory".

**Status:** `cargo build`, `cargo clippy --all-targets -- -D warnings` and `cargo test` all clean on
Windows. 37 tests pass. `onnxruntime_vulkan_ep.dll` exports exactly `CreateEpFactories` and
`ReleaseEpFactory`.

### Correction + version policy (2026-07-28T19:48:05-07:00)

**Supersedes the "never fall back to a lower API version" line above.** That was right about the
danger and wrong about the remedy. `OrtApi`, `OrtEp` and `OrtEpFactory` are **append-only**, so
version *v*'s layout is a prefix of 28's — running on an older host is safe *if and only if* we
never touch a field added after *v*. That is an obligation, not a property, and it needs two
mechanisms to be real:

1. Write the **negotiated** version (not the compiled-against one) into every
   `ort_version_supported` field we hand ORT — `OrtEpFactory`, `OrtEp`, `OrtNodeComputeInfo`,
   `OrtNodeFusionOptions`. That is how ORT knows where to stop reading our vtables.
2. Gate every optional entry point on `NegotiatedApi::supports(since::*)`, one named constant per
   feature. A gate that only ever returns true is theatre, so there is a test asserting it returns
   **false** at version 23.

With both in place, negotiating down is a supported configuration rather than the silent-UB trap.
Policy now: **compile and ship against 1.28, minimum supported host 1.24, refuse below that.**
Excluding 1.27 is deliberate — it has a null-allocator bug in plugin-EP `PrePack` plus a deleter
lifetime bug. 1.24 as the floor because that is where the surface settled, and ORT's own
`ep_factory_provider_bridge.h` uses `ort_version_supported < 24` as its compatibility line.

**Two of Justin's premises were wrong; the Fact Checker caught both.**
`CreateExternalResourceImporterForDeviceImpl` does **not** exist as public API — it is only a local
static in ORT example/test code. The real symbols are
`OrtEpFactory::CreateExternalResourceImporterForDevice` (EP side, ours, returns an
`OrtExternalResourceImporterImpl*`) and `OrtInteropApi::CreateExternalResourceImporterForDevice`
(caller side, `core/session/interop_api.h`). And it shipped in **1.24**, not 1.28. Lesson for the
team: version claims about this API are worth verifying against the `\since` annotations and the
bridge guards before designing around them.

**What the importer actually does** (write it down; it gets misremembered): the *caller* allocates
`VkDeviceMemory` with `VkExportMemoryAllocateInfo` — exportability must be chosen at allocation
time, memory not allocated as exportable can never be imported — exports it via
`vkGetMemoryWin32HandleKHR` / `vkGetMemoryFdKHR`, and passes the OS handle to `ImportMemory`. The
EP re-imports into its own `VkDeviceMemory` and wraps it as a zero-copy tensor. Timeline-semaphore
import for GPU↔GPU sync uses the same interface. ORT already defines the Vulkan enum values:
`ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_WIN32` / `..._OPAQUE_FD` and two
`ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_*`. **Working in-tree reference:
`onnxruntime/test/providers/nv_tensorrt_rtx/nv_vulkan_test.cc`** — the NV TensorRT RTX EP doing
exactly our case, Vulkan on both sides. Match its contract; do not invent one.

**"Bound but not implemented" needs teeth.** Claiming a seam is correctly shaped is worthless if
nothing checks it. `sys::importer_seam` names `OrtExternalResourceImporterImpl`, the two handle
base structs, `OrtInteropApi` and the four Vulkan enum values explicitly, so an upstream rename is
a build failure at the moment of the ORT bump rather than a discovery months later when someone
tries to implement it. Generalise this: for any deferred seam, name the types in code.

**OQ-3 is still open and the importer does not close it.** Caller-driven, caller's memory (the
importer) vs EP-driven, our memory (OQ-3: what does our `Alloc()` return to a pointer-based API
when a Vulkan allocation is `VkBuffer` + offset). Recorded as "evaluated and rejected as
orthogonal" in the decision inbox so nobody re-litigates it. Opaque-handle registry vs
`VK_KHR_buffer_device_address` is mine to propose, Morpheus's to decide.

**Status after the corrections:** `cargo build`, `cargo clippy --all-targets -- -D warnings` and
`cargo test` all clean. 45 tests (36 unit + 9 layering).

---

## 2026-07-28T21:01:56-07:00 — OQ-3 proposal, OQ-13 design, and the C1/C2 code obligations

**Ban the value, not the comparison.** C1 (no domain-wide contrib opt-in) could have been linted
by enumerating predicate shapes — `== Domain::Ms`, `!=`, `matches!`, `if let`, `starts_with`. That
list is never finished. Banning the contrib domain as a *value* in non-test code, with one
exemption for the arm that defines its spelling, forbids every shape at once and leaves no third
spelling to forget. Generalise: when linting "you must not decide on X", ban naming X, not the
comparisons against it.

**A grep-based tripwire on macro-generated data can never fire.** My first C2 tripwire looked for
`domain: Domain::Ms` in `registry.rs`. Rows are declared through `op_table!`, so that string
appears only inside the macro body — the test would have passed forever while proving nothing.
Rewrote it to read the *linked registry* via `all_specs()`, which an integration test can do
because it links the crate. Lesson: if the data is generated, assert against the data, not the
source text.

**ORT does pointer arithmetic on allocator return values, and that decides OQ-3's encoding.** The
memory-pattern planner allocates one block and constructs tensors at `base + offset`. Any
synthetic token scheme breaks under `ptr + n`. Reserving real virtual address space
(`VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` / `mmap(PROT_NONE, MAP_NORESERVE)`) and handing out
real, unique, never-dereferenceable addresses makes arithmetic, alignment and uniqueness correct
*by construction* — and turns a stray dereference from silent corruption into an immediate fault
at an address we recognise. Making an invariant MMU-enforced beats documenting it.

**BDA is not an optimization, it is a second shader architecture.** I nearly wrote "registry
primary, BDA optional" to agree with Morpheus. Costing it out properly changed the answer: a
`VkDeviceAddress` is unusable by a descriptor-bound shader, so adopting BDA means a whole second
shader family — and it does not even remove the side table, because building a descriptor set
still needs a `VkBuffer`. Combined with MoltenVK's Apple-Silicon-only support and §7.2 making it
probed-not-required, the honest recommendation is *no BDA at all*. Lesson: when the lead narrows
the ground for you, cost the remaining option out anyway; sometimes the answer is further than
they went, not nearer.

**Read the upstream reference before designing the contract.** `nv_vulkan_test.cc` gave me five
things I would have got wrong: both `VkExternalMemoryBufferCreateInfo` *and*
`VkExportMemoryAllocateInfo` are required; DMA-BUF has no ORT enum and is explicitly unsupported;
`OrtExternalTensorDescriptor` carries `offset_bytes`, so several tensors can live in one import;
teardown order is ORT handles → importer → deinit → `vkDeviceWaitIdle` → Vulkan objects; and the
test calls `DisableMemPattern()`, which is a warning not a licence — an EP cannot force a caller's
session options.

**OS handle ownership is asymmetric across platforms.** `vkImportMemoryFdKHR` takes ownership of
the fd on Linux; the Win32 `HANDLE` is not transferred and the caller must close it. The reference
test simply leaks it at process exit. Documented both rather than copying the leak.

**Three agents in one crate means the tree is red through no fault of yours.** Mouse landed
`registry.rs` references to `ops::{attention,quant,moe,ssm}` before the modules existed. Nothing
to fix on my side — verify, wait, re-verify — but worth knowing that "cargo build fails" is no
longer evidence about my own changes. Check `git status` and the error's file before reacting.

**Shipped this turn:** `sys::{OrtRelease, ORT_PINNED, ORT_FLOOR, SchemaBaseline}` (C2 machinery);
C1 contrib-domain rules plus five planted-violation and three false-positive tests in
`tests/layering.rs`; `src/bin/epctl.rs` with `--dump-capabilities [--json]` and
`tests/dump_capabilities.rs`; README sections for the zero-copy integration contract, `epctl`, and
C1. Owed by Mouse: one line, `pub schema_baseline: Option<SchemaBaseline>` on `OpSpec`.

**C2 arrived from both ends at once, and the other end was better.** Mouse and I independently
implemented the contrib schema baseline within the same hour — me as a keyed side table in
`sys.rs`, him as a field inside `ContribSchema` in `registry.rs`. His placement wins for a reason
worth keeping: nesting the baseline inside the schema fingerprint makes it *impossible to record a
shape without recording where the shape came from*, whereas a parallel table can be half-filled.
It also carried information mine could not — two ops exist only on ORT main, so `1.28.0` would
have been a lie for them. Deleted my table. Generalise: when two implementations collide, do not
split the difference and do not defer on seniority; ask which one makes the wrong state
unrepresentable.

**Two places recording the same fact should be deleted, not cross-checked.** Mouse's instinct was
a test asserting the two agreed. That test is real work forever and buys nothing a single source
would not. Deleting one side is the cheaper and stronger fix.

**Own the type, let the domain own the data.** `sys.rs` owns `OrtRelease`/`SchemaBaseline`/
`ORT_PINNED` because ORT version tracking is mine; `registry.rs` owns which release each row was
read from because that belongs beside the schema it describes. This split also kept the layering
clean — a `sys` → `registry` lookup helper would have inverted the dependency direction of the
lowest layer in the crate, which was the tell that my design was in the wrong place.
