# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Runtime & FFI — ORT plugin EP C ABI, sys/ep/factory, build & packaging
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-29T09:00:39-07:00 — full session details in decisions.md -->

### [SUMMARY] Sessions 1–7: crate foundation, version policy, logging crash fix, cargo ci (2026-07-28–2026-07-29)

**M0 crate foundation (sessions 1–2):**
- `onnxruntime_ep_c_api.h` has no include guard. `rust/wrapper_ort.h` must include ONLY `onnxruntime_c_api.h`.
- Bindings via bindgen over vendored headers (`third_party/onnxruntime/`, tag v1.28.0). Wrong vtable field order = silent UB. CI: every runner needs LLVM/libclang.
- `clippy::undocumented_unsafe_blocks`: `// SAFETY:` must be on the line immediately preceding `unsafe {`. Generated bindings: allow at `mod ort` level.
- `GetSupportedDevices` with no Vulkan: return success + zero devices. `GetCapability`: decline all nodes inside a control-flow subgraph body (non-null `Graph_GetParentNode`) or ORT raises `INVALID_GRAPH`.
- `GetSessionConfigEntry`: two-call protocol; release status even on success. Getting this wrong leaks `OrtStatus` per option per session.
- `Logger_LogMessage` `file_path` is `_In_z_` — ORT dereferences unconditionally; u16 on Windows, u32 on Unix. Was passing `null()` — caused the first real ORT load to crash.
- cdylib exports exactly `CreateEpFactories` and `ReleaseEpFactory`. 37 tests at M0 baseline.

**OrtEp/OrtEpFactory vtable field order (ORT 1.28 — authoritative):**
- `OrtEp`: ort_version_supported, GetName, GetCapability, Compile, ReleaseNodeComputeInfos, GetPreferredDataLayout, ShouldConvertDataLayoutForOp, SetDynamicOptions, OnRunStart, OnRunEnd, CreateAllocator, CreateSyncStreamForDevice, GetCompiledModelCompatibilityInfo, GetKernelRegistry, IsConcurrentRunSupported, Sync, CreateProfiler, IsGraphCaptureEnabled, IsGraphCaptured, ReplayGraph, GetGraphCaptureNodeAssignmentPolicy, GetAvailableResource, OnSessionInitializationEnd, GetDefaultMemoryDevice, ReleaseCapturedGraph.
- `OrtEpFactory`: ort_version_supported, GetName, GetVendor, GetSupportedDevices, CreateEp, ReleaseEp, GetVendorId, GetVersion, ValidateCompiledModelCompatibilityInfo, CreateAllocator, ReleaseAllocator, CreateDataTransfer, IsStreamAware, CreateSyncStreamForDevice, GetHardwareDeviceIncompatibilityDetails, **CreateExternalResourceImporterForDevice**, GetNumCustomOpDomains, GetCustomOpDomains, InitGraphicsInterop, DeinitGraphicsInterop, SelectBestModelCandidate.

**Version negotiation (session 2):** Compile/ship against ORT 1.28 (`ORT_API_VERSION=28`); min host 1.24; exclude 1.27 (null-allocator PrePack + deleter lifetime bugs). Write negotiated version (not compiled-against) into every `ort_version_supported` field.

**OQ-3 — reserved VA registry (session 3, adopted by Morpheus):** ORT pointer arithmetic on allocator return values — synthetic tokens break. `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` on Windows; `mmap(PROT_NONE, MAP_NORESERVE)` on POSIX. Unique spans; stray dereference = MMU fault. No BDA at all.

**C1/C2 linting (session 3):** C1 bans the contrib domain VALUE (not the comparison). C2 drift alarm: assert against `all_specs()` linked data, not source text. `SchemaBaseline` inside `ContribSchema` (not a parallel table). C2 item 7: fingerprint audit CI job; non-release baseline rows may not go `Live`.

**`cargo ci` xtask (session 4, D-T13):** `rust/xtask/` package + `rust/.cargo/config.toml` alias. Runs fmt → clippy → build → test. `--release` flag. `ALLOW_MISSING_GLSLC=1` set automatically when no SDK. Zero deps. Lesson: CI verification must be an artefact, not a habit.

**Null `file_path` crash fix (session 5, D-T14):** `Logger_LogMessage` `file_path: *const wchar_t` is `_In_z_`. Always pass real NUL-terminated string with two `cfg` branches. Bug manifested as a crash at the first `log::warn!` after `CreateEp`. Lesson: SAL annotations (`_In_z_` vs `_In_opt_z_`) are contract text — read the implementation on ambiguity.

**OrtLogger lifetime bug (session 5, D-T15):** `CreateEp` overwrote the process-default logger permanently. `ReleaseEp` now calls `restore_default_ort_logger()`.

**`tests/host_registration.rs` mock ORT (session 5, D-T16):** Zeroed vtable with Rust callbacks asserting `_In_z_` non-null/NUL-terminated, `_Outptr_` written, `OrtStatus` released exactly once. Verified adversarially (plant the bug, test fails). Blind spot documented: cannot catch packaging faults (missing exports, wrong crate-type).

**`tests/cdylib_load.rs` (session 6, D-T18):** dlopens the shipped cdylib, resolves exports by name as ORT does. `libloading` added as `[dev-dependencies]`. 272 tests, `cargo ci --release` green.

**`ort::wchar_t` Linux fix (session 7):** `ORTCHAR_T` behind a single `cfg`-selected alias. `tests/portability.rs` added. Lesson: writing a caveat in a caveats section discharges the feeling of owing something about it — the countermeasure must be structural: the commit either closes the gap or explains in the caveat itself why it was rejected.

**External resource importer (OQ-13, post-M2):** `OrtEpFactory::CreateExternalResourceImporterForDevice` landed ORT 1.24 (not 1.28). Does NOT answer OQ-3. `sys::importer_seam` names all types so upstream rename = build failure. Teardown order: ORT handles → importer → deinit → `vkDeviceWaitIdle` → Vulkan.

---

## Cross-agent context appended (2026-07-29T09:00:39-07:00) — first-hardware round

📌 **Standard-domain LLM rows registered (2026-07-29, Mouse D-M6-04):** `ai.onnx::Attention`, `ai.onnx::RMSNormalization`, `ai.onnx::RotaryEmbedding` all registered at `OPSET_STD_LLM = 23`. Without these, a Qwen3 built by Justin's own `onnx-genai-models` (mobius builder) would have declined ~5 nodes/layer × 28 layers. Tank's `GetCapability` path must correctly handle these standard-domain rows — they are not contrib-domain and must not be filtered by any `com.microsoft` domain check.

📌 **Niobe's `onnx-runtime-tracer` is now a dependency (2026-07-29, Niobe D-N1):** Pin: `0.1.0-dev.5, default-features = false`. The absolute UNIX-microsecond clock in this crate is critical for correct overlay of plugin cdylib spans onto the host timeline. If Tank's `Cargo.toml` patches or replaces this pin, coordinate with Niobe — wrong clock semantics silently corrupt the trace.

📌 **Vulkan SDK at `C:\VulkanSDK\1.4.350.0` (2026-07-29):** Not on default PATH. `cargo ci` sets `ALLOW_MISSING_GLSLC=1` automatically when no SDK is found; for full builds including shader compilation, prefix the SDK `bin/` directory explicitly.

📌 **`rustfmt --edition 2021` silently no-ops on this edition-2024 crate (2026-07-29, D-T12):** Always use `cargo fmt --all` or the `cargo ci` xtask.