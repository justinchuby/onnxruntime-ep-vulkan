# `onnxruntime-ep-vulkan` — Rust crate

The Vulkan compute **plugin execution provider** for ONNX Runtime, built as a standalone shared
library that ONNX Runtime loads at runtime. Nothing here links against ONNX Runtime, and nothing
here links against the Vulkan SDK.

| | |
|---|---|
| Crate | `onnxruntime-ep-vulkan` |
| Artifact base name | `onnxruntime_vulkan_ep` |
| ORT ABI targeted | **1.28.0** (`ORT_API_VERSION` 28) — see [version policy](#version-policy) |
| Minimum supported ORT | **1.24** (`ORT_API_VERSION` 24) |
| EP registration name | `VulkanExecutionProvider` |
| Vendor | `onnxruntime-ep-vulkan` |
| Crate version scheme | `0.<ORT_API_VERSION>.<patch>` |
| Edition / MSRV | 2024 / 1.85 |

**Status: M0.** The crate builds, loads, negotiates the ABI, enumerates devices and reports
capability. It claims **zero nodes** — by design. Every model runs on CPU, correctly. The engine
(`src/engine.rs`) and the op registry (`src/ops/`) are documented seams awaiting their owners.

---

## Building

### Prerequisites

| Tool | Required? | Why |
|---|---|---|
| Rust ≥ 1.85 | **yes** | edition 2024 |
| **libclang** (LLVM) | **yes** | `bindgen` parses the vendored ORT headers at build time |
| ONNX Runtime install | no | the headers are vendored; nothing links against ORT |
| Vulkan SDK / `glslc` | not yet | only needed once `shaders/glsl/` is non-empty |
| Vulkan loader | at runtime only | absent loader ⇒ zero devices advertised, not an error |

Install libclang:

```powershell
# Windows
winget install LLVM.LLVM
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'   # only if bindgen cannot find it
```

```bash
# Debian / Ubuntu
sudo apt-get install -y libclang-dev
# macOS
brew install llvm      # or rely on the Xcode CLT libclang
```

### Build

```powershell
cd rust
cargo build                  # debug
cargo build --release        # release
```

Artifacts land in `target/{debug,release}/`:

| OS | File |
|---|---|
| Windows | `onnxruntime_vulkan_ep.dll` (+ `.dll.lib`, `.pdb`) |
| Linux | `libonnxruntime_vulkan_ep.so` |
| macOS | `libonnxruntime_vulkan_ep.dylib` |

Two C symbols are exported, and only two: `CreateEpFactories` and `ReleaseEpFactory`.

### Verify

```powershell
cargo build
cargo clippy --all-targets -- -D warnings
cargo test
cargo test --test layering        # the layering lint on its own
```

All four must be clean before a change lands.

---

## Where the ORT headers come from

The plugin-EP C ABI headers are **vendored** in
[`../third_party/onnxruntime/include/`](../third_party/onnxruntime/include/), taken verbatim from
`microsoft/onnxruntime` tag `v1.28.0` (commit `da9b5e364c465de65c49d91e696cd6485270757f`). MIT
licence, reproduced alongside them. Full provenance and the re-vendoring procedure are in
[`../third_party/onnxruntime/PROVENANCE.md`](../third_party/onnxruntime/PROVENANCE.md).

`build.rs` resolves the include directory in this order, so you can point at a local ORT checkout
when you need to test an unreleased ABI:

1. `$ORT_INCLUDE_DIR`
2. `$ORT_HOME/include`
3. `third_party/onnxruntime/include` *(default)*

### bindgen, not hand-written bindings

The ORT plugin-EP ABI is **experimental and still moving** — `OrtEp` and `OrtEpFactory` both gained
fields in 1.23, 1.24 and 1.28. That is exactly the argument people usually make *for* hand-writing
bindings ("the API is unstable, keep it under our control"), and it is why we do the opposite:

* These are `#[repr(C)]` **vtables**. Getting a field *order* wrong does not fail to compile and
  does not fail to load. It calls the wrong function pointer with the wrong arguments — silent
  undefined behaviour, discovered later, somewhere else. `OrtApi` alone has several hundred
  function-pointer fields. Transcribing that by hand, repeatedly, is a bet we lose eventually.
* bindgen derives the layout from the same bytes ORT was compiled from, so field order cannot
  drift from the header.
* Vendoring the headers recovers everything the hand-written camp actually wants: builds are
  byte-reproducible, no network access and no ORT install are needed, and an ORT bump becomes a
  reviewable diff of two header files.

The cost is a libclang dependency on build machines. That is a CI provisioning line, paid once.

### Version policy

Three numbers, and conflating them is how plugin EPs corrupt themselves:

| | value | meaning |
|---|---|---|
| `ORT_API_VERSION_EXPECTED` | 28 | what we compile and ship against |
| `ORT_API_VERSION_MIN` | 24 | oldest host we will run against |
| negotiated version | 24–28 | what the host in front of us actually serves |

`sys::check_api_version()` asks `GetApi(28)` and walks down to `GetApi(24)`, taking the first
version the host serves. Below 24 the plugin **refuses to load** with a message naming both the
requirement and the host's version.

Running below 28 is safe because `OrtApi`, `OrtEp` and `OrtEpFactory` are append-only — version *v*
is a prefix of 28. But that safety is an *obligation* (never touch a field added after *v*), not a
property, so two mechanisms discharge it:

* The **negotiated** version, not 28, is written into `OrtEpFactory::ort_version_supported`,
  `OrtEp::ort_version_supported` and friends — so ORT stops reading our vtables exactly where its
  own header stops describing them. This is the same signal ORT uses on its side of the boundary.
* Every optional entry point is gated by `NegotiatedApi::supports(since::*)`, one named constant
  per feature (today: `since::EXTERNAL_RESOURCE_IMPORTER = 24`). A test asserts the gate returns
  **false** at version 23 — a gate that only ever says yes would pass every other test.

Three compile-time assertions back this up: the vendored header's `ORT_API_VERSION` is 28, the
crate minor version is 28, and the floor does not exceed the ceiling.

**Why ship against 1.28 specifically.** 1.27 has a critical plugin-EP defect — a null allocator in
`PrePack` plus a deleter lifetime bug — that would hit us directly. 1.27 is excluded on purpose,
not merely superseded.

**Why 1.24 as the floor.** `OrtEpFactory` has existed since 1.22, but 1.24 is where the surface we
depend on settled, and ORT's own bridge uses `ort_version_supported < 24` as its compatibility
line. We never have to reason about the 1.22/1.23 layouts.

The plugin EP API is **still experimental in 1.28** — it did not graduate. That is why every raw
ORT type stays behind `src/sys.rs`: a vtable change should be a one-file fix.

### Zero-copy IO binding (bound, not implemented)

`OrtEpFactory::CreateExternalResourceImporterForDevice` (ORT 1.24+) is the OS-handle-based external
memory import path: the caller allocates `VkDeviceMemory` **with `VkExportMemoryAllocateInfo`**,
exports it via `vkGetMemoryWin32HandleKHR` or `vkGetMemoryFdKHR`, and the EP re-imports it as a
zero-copy tensor. Timeline-semaphore import for GPU↔GPU sync rides the same interface.

M0 **binds the seam without implementing it**: `sys::importer_seam` names
`OrtExternalResourceImporterImpl`, the handle base structs, and the four Vulkan-specific enum
values ORT defines (`..._VK_MEMORY_WIN32`, `..._VK_MEMORY_OPAQUE_FD`, and the two timeline
semaphore types). The factory slot is left `None` — which ORT reads as "cannot import external
memory", and which is true today — behind an already-written `supports()` gate. Adding the
implementation is one new `importer.rs` plus one line; nothing else moves. The contract to match is
ORT's own `nv_vulkan_test.cc`.

This does **not** answer OQ-3 (what our `Alloc()` returns to ORT's pointer-based allocator API when
a Vulkan allocation is a `VkBuffer` + offset). Different direction, different memory, different
owner. That one is still open.

---

## Loading the plugin

### Python

```python
import onnxruntime as ort

ort.register_execution_provider_library(
    "VulkanExecutionProvider",
    r"C:\path\to\onnxruntime_vulkan_ep.dll",   # or lib....so / .dylib
)

sess = ort.InferenceSession(
    "model.onnx",
    providers=["VulkanExecutionProvider", "CPUExecutionProvider"],
)
```

In M0 every node falls through to CPU. That is the expected result, and
`ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` will tell you exactly why, node type by node type.

### C / C++

```c
OrtEnv* env = /* ... */;
g_ort->RegisterExecutionProviderLibrary(env, "VulkanExecutionProvider", ORT_TSTR("onnxruntime_vulkan_ep.dll"));
```

---

## Environment variables

| Variable | Effect |
|---|---|
| `ONNXRUNTIME_EP_VULKAN_VERBOSE=1` | verbose EP logging through ORT's logger |
| `ONNXRUNTIME_EP_VULKAN_TRACE=1` | per-node trace during capability and compile |
| `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` | log every node the EP declined **and why**, aggregated by op type |
| `RUST_LOG=debug` | Rust-side `log` filtering, independent of ORT's level |
| `LIBCLANG_PATH` | build only — where `bindgen` finds libclang |
| `ORT_INCLUDE_DIR` / `ORT_HOME` | build only — override the vendored headers |
| `VULKAN_SDK` | build only — where `build.rs` looks for `glslc` |
| `ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` | build only — allow a build with shaders present but no compiler |

Session-option equivalents (`ep.vulkan.*`) are read in `src/ep.rs` and take precedence over the
environment.

---

## Layering lint

`DESIGN.md` §4.2 states two rules that protect the op layer:

1. The ORT C ABI never appears in `src/ops/`.
2. Raw Vulkan — and `unsafe` — never appears in `src/ops/`.

Both are enforced mechanically by [`tests/layering.rs`](tests/layering.rs):

```powershell
cargo test --test layering
```

It scans `src/ops/**/*.rs` for a forbidden vocabulary (`crate::sys`, `Ort*`, `ash`, `vk::`, `Vk*`,
`unsafe`, …) after stripping comments and string literals, so documentation that *names* a
forbidden token is not a false positive — `src/ops/mod.rs` names all of them on purpose. A
mirror-image check keeps `ash`/`vk::` out of the ORT boundary modules.

The lint is itself tested: several cases run the scanner over deliberately planted violations and
assert it catches each one, so a refactor that neuters the detector fails too. It was also verified
against a real planted file under `src/ops/`, which produced seven findings before being removed.

**Why a test rather than a `deny` attribute or an xtask.** A lint attribute cannot express "this
identifier must not appear in this directory"; `ash` is a legitimate dependency of the crate, and
there is no built-in way to forbid it per-module. An xtask is a second binary to build and invoke
on every CI lane. A test is already run by `cargo test`, so it cannot be forgotten — CI cannot be
green without it, and a contributor sees the failure locally before pushing.

*CI wiring is Trinity's.* This crate owns the check and the local command; `.github/workflows/`
needs `cargo test --all-targets` (which includes it) plus LLVM/libclang on every runner.

---

## Module map

| Module | Owner | What it is |
|---|---|---|
| `src/lib.rs` | Tank | crate root, `guard_ffi_status` panic guard, the two exported C symbols |
| `src/sys.rs` | Tank | generated ORT bindings, version gates, status helpers |
| `src/factory.rs` | Tank | `OrtEpFactory` vtable, device enumeration and correlation |
| `src/ep.rs` | Tank | `OrtEp` vtable, session options, capability reporting |
| `src/logging.rs` | Tank | Rust `log` → ORT logger bridge |
| `src/registry.rs` | Mouse | the op table; `NodeView`, the ABI→safe-Rust translation point |
| `src/engine.rs` | **Switch** | **stub** — the Vulkan engine seam and shared vocabulary |
| `src/ops/mod.rs` | **Mouse** | **stub** — op handlers |
| `build.rs` | Tank | bindgen + GLSL→SPIR-V pipeline |

### FFI discipline

Non-negotiable in this crate, and visible in every file:

* Every exported `extern "C"` function body runs inside `guard_ffi_status`, which catches unwinds
  and converts them to an `OrtStatus`. **No panic crosses the FFI boundary** — unwinding into
  ORT's C++ is UB.
* Every `unsafe` block carries a `// SAFETY:` comment stating the invariant it relies on.
  `clippy::undocumented_unsafe_blocks` is `warn`, and CI runs with `-D warnings`, so an
  undocumented block fails the build.
* No `unwrap()` / `expect()` anywhere near the ABI boundary. Null pointers from ORT are checked,
  not assumed.
* `panic = "unwind"` in **both** dev and release profiles — `panic = "abort"` would stop the guards
  from working.

---

## What is stubbed, and for whom

**Switch** (`src/engine.rs`) — replace `probe_devices()` with real `vkEnumeratePhysicalDevices`
(its contract: never fails, returns sorted best-first, returns empty when there is no loader), add
the `vk/` tree, implement `DispatchContext`, and add SPIR-V shaders under `shaders/glsl/` for
`build.rs` to compile.

**Mouse** (`src/registry.rs`, `src/ops/`) — add rows to `REGISTRY` and the matching handlers. The
invariant to preserve: a node is claimed *only if* it can actually be translated. The layering lint
is on and will reject an op handler that reaches for `sys` or `ash`.

**Tank**, next — real `Compile`, the allocator and data-transfer vtable slots (M2), a proposal for
OQ-3 (opaque-handle registry vs `VK_KHR_buffer_device_address`) for Morpheus to decide, and the
external resource importer when zero-copy IO binding is wanted. `src/sys.rs` carries the bound
seam and a marked TODO showing exactly where that slots in.
