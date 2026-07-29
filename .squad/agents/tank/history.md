# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Runtime & FFI — ORT plugin EP C ABI, sys/ep/factory, build & packaging
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-28T22:28:08-07:00 — full session details in decisions.md -->

### [SUMMARY] Sessions 1–3: crate foundation, version policy, OQ-3 proposal, C1/C2 linting (2026-07-28)

**M0 crate foundation:**
- `onnxruntime_ep_c_api.h` has no include guard. `rust/wrapper_ort.h` must include ONLY `onnxruntime_c_api.h` (not both). Costs an hour on every ORT re-vendor.
- Bindings are bindgen over vendored headers (`third_party/onnxruntime/`, tag v1.28.0, commit `da9b5e364c465de65c49d91e696cd6485270757f`). Wrong vtable field order compiles and calls wrong fn pointer: silent UB. Bindgen guarantees layout fidelity. CI: every runner needs LLVM/libclang (`LIBCLANG_PATH=C:\Program Files\LLVM\bin` on Windows).
- `clippy::undocumented_unsafe_blocks` is positional: `// SAFETY:` must be on the line immediately preceding `unsafe {`. Generated bindings need lint allowed at `mod ort` level.
- `GetSupportedDevices` with no Vulkan: returns success + zero devices (not an error). `GetCapability`: decline every node inside a control-flow subgraph body (`non-null Graph_GetParentNode`), or ORT raises `INVALID_GRAPH`.
- `GetSessionConfigEntry` is a two-call protocol: first call with null buffer to query length (release status even on success path), then fill. Getting this wrong leaks an `OrtStatus` per option per session.
- `Logger_LogMessage` takes `file_path: *const wchar_t` — u16 on Windows, u32 on Unix. Pass null.
- `layering.rs` tests enforce the rules; cannot be forgotten since `cargo test` runs them.
- 37 tests passing at M0 baseline. cdylib exports exactly `CreateEpFactories` and `ReleaseEpFactory`.

**OrtEp vtable field order (ORT 1.28 — authoritative):**
- `OrtEp`: ort_version_supported, GetName, GetCapability, Compile, ReleaseNodeComputeInfos, GetPreferredDataLayout, ShouldConvertDataLayoutForOp, SetDynamicOptions, OnRunStart, OnRunEnd, CreateAllocator, CreateSyncStreamForDevice, GetCompiledModelCompatibilityInfo, GetKernelRegistry, IsConcurrentRunSupported, Sync, CreateProfiler, IsGraphCaptureEnabled, IsGraphCaptured, ReplayGraph, GetGraphCaptureNodeAssignmentPolicy, GetAvailableResource, OnSessionInitializationEnd, GetDefaultMemoryDevice, ReleaseCapturedGraph.
- `OrtEpFactory`: ort_version_supported, GetName, GetVendor, GetSupportedDevices, CreateEp, ReleaseEp, GetVendorId, GetVersion, ValidateCompiledModelCompatibilityInfo, CreateAllocator, ReleaseAllocator, CreateDataTransfer, IsStreamAware, CreateSyncStreamForDevice, GetHardwareDeviceIncompatibilityDetails, **CreateExternalResourceImporterForDevice**, GetNumCustomOpDomains, GetCustomOpDomains, InitGraphicsInterop, DeinitGraphicsInterop, SelectBestModelCandidate.

**Version negotiation policy:**
- Compile and ship against ORT 1.28 (`ORT_API_VERSION` = 28). Minimum supported host: 1.24 (`ORT_API_VERSION_MIN = 24`). Exclude 1.27 explicitly (null-allocator PrePack bug + deleter lifetime bug).
- `OrtApi`/`OrtEp`/`OrtEpFactory` are append-only: version *v* layout is a prefix of 28's. Write the **negotiated version** (not compiled-against) into every `ort_version_supported` field.
- `NegotiatedApi::supports(since::*)` gates every optional entry point. Test asserts it returns false at version 23.
- ORT floor `ep_factory_provider_bridge.h` uses `ort_version_supported < 24` as the compatibility line.

**External resource importer (OQ-13 — post-M2):**
- Real symbol: `OrtEpFactory::CreateExternalResourceImporterForDevice` (landed ORT 1.24, not 1.28).
- `CreateExternalResourceImporterForDeviceImpl` does NOT exist as public API (test-code local only).
- Does NOT answer OQ-3 (orthogonal: caller-driven vs EP-driven memory).
- `sys::importer_seam` names all types explicitly so upstream rename = build failure at ORT bump.
- Teardown order: ORT handles → importer → deinit → `vkDeviceWaitIdle` → Vulkan objects.
- DMA-BUF has no ORT enum; DMA-BUF import is unsupported.
- Reference: `onnxruntime/test/providers/nv_tensorrt_rtx/nv_vulkan_test.cc`.

**OQ-3 proposal (adopted by Morpheus):**
- ORT pointer arithmetic on allocator return values (`base + offset`). Any synthetic token breaks.
- **Reserved VA registry**: `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` on Windows; `mmap(PROT_NONE, MAP_NORESERVE)` on POSIX. Real unique spans. Stray dereference = MMU fault.
- **No BDA**: `VkDeviceAddress` unusable by descriptor-bound shaders without `GL_EXT_buffer_reference`. Does not remove side table. MoltenVK Apple-Silicon-only. No BDA at all.
- For Android narrow address space: probe-and-halve, not a per-platform constant table.

**C1/C2 linting:**
- C1 (no domain-wide contrib opt-in): ban the value (contrib domain as a value in non-test code), not the comparison. One exemption: the arm that defines its spelling.
- C2 drift alarm: assert against `all_specs()` (linked registry data), not against source text. Macro-generated data cannot be grepped.
- `SchemaBaseline` inside `ContribSchema` (not a parallel table). Two records of one fact = drift hazard; delete one.
- C2 item 7: fingerprint audit CI job; rows with non-release baseline may not go `Live` (build failure enforced).
- Ban `CONTRIB_SCHEMA_BASELINES` side table (Tank's duplicate deleted when Mouse's nested field wins).

**Key ORT API facts:**
- `OrtEp::Compile` takes `*mut *const OrtGraph` / `*mut *const OrtNode` (mutability on outer pointer).
- `GetSessionConfigEntry`: two-call protocol; release status even on success.
- Three-agent concurrent repo: `cargo build` can be red through no fault of yours. Check `git status` and the error's file before reacting.
- 45 tests after session 2 (corrections). Test commands: `cargo build; cargo clippy --all-targets -- -D warnings; cargo test` (all clean Windows).

---

## Cross-agent context appended (2026-07-28T22:28:08-07:00)

📌 **Switch's `bind_aliased_output` seam (Switch engine-seams):** `DispatchContext::bind_aliased_output` default method returns the resolved input buffer. Tank's allocator must support aliased buffer handles for KV-cache in-place updates (M2+). When implementing the allocator in M2, check `dispatch_indirect` and `resolve_prepacked` seams in `engine.rs` for the full contract.

📌 **ORT 1.28 vtable `CreateExternalResourceImporterForDevice` (Fact Checker):** Real symbol name is `OrtEpFactory::CreateExternalResourceImporterForDevice` (not `…Impl`). Landed ORT 1.24. `sys::importer_seam` already names all types; no ABI change needed. Zero-copy IO binding is OQ-13 (post-M2, Tank owns).

📌 **ORT 1.27 excluded (Tank + Trinity):** ORT 1.27 null-allocator PrePack bug in plugin EP path. Both Tank build experience and Trinity fp16 NaN/Inf independently confirm. Pin `ORT_VERSION=1.28.0` in CI (`ci.yml` workflow-level env). Tank's `ORT_API_VERSION_MIN = 24` floor is correct; 1.27 is excluded mid-range.

📌 **C2 item 7 fingerprint audit CI job (Morpheus §1.4):** A fingerprint audit CI job (`graph_census.py`) must run before any contrib tier-3 work. Rows with non-release `SchemaBaseline` may not go `Live` (build failure enforced). Tank's `OrtRelease`/`SchemaBaseline` types in `sys.rs` are the shared vocabulary; Mouse's `ContribSchema` owns the per-row data. Do not add a duplicate side table.

📌 **Hard Vulkan SDK dependency (Morpheus OQ-4):** `ALLOW_MISSING_GLSLC=1` escape hatch produces an inert artifact (zero devices, zero claims). No release artifact from escape-hatch builds. `shaders::has_any()` in `engine.rs` is the truth point; `probe_devices()` and `get_capability_impl()` both early-exit when false (Switch belt-and-suspenders guards).

---

## Session 4 — 2026-07-28T22:28:08-07:00 — crate-wide fmt pass + `cargo ci`

**The lesson of the turn: a verification loop must be an artefact, not a habit.** CI was red for
four consecutive runs and nobody noticed, including me. Every agent ran build/clippy/test, saw
green, reported green. `cargo fmt --check` was never in that loop — not because anyone was
careless, but because "green" meant *"the commands I happen to remember passed"*, and nothing in
the repo knew what CI actually runs. Memory drifts from CI the first time CI changes. Fix:
`cargo ci` (xtask + alias in `rust/.cargo/config.toml`) IS the list; a new CI check gets added to
`CHECKS` in `xtask/src/main.rs` in the same commit.

**Corollary I want to remember: an all-green local command is itself a claim, and it can lie the
same way.** `cargo ci` prints its caveats on *success*, not just on failure — no shader has ever
executed, none are even compiled without glslc, no device is touched, no Python lane, one OS.
Otherwise it would recreate one level up the exact false confidence it was built to remove.

**Technique — proving a formatting pass is formatting-only, mechanically.** Strip `[\s,]` from
both the `HEAD` and working-tree copies of each changed file and compare: identical means no
token changed. Survivors are usually rustfmt's `reorder_imports` sorting names *inside*
`use a::{...}` braces — confirm with a sorted character multiset. Only what is left needs
eyeballing via `git diff -w --ignore-blank-lines`. 26 files reduced to 5 needing human reading.

**Gotcha: rustfmt can break `clippy::undocumented_unsafe_blocks`.** It collapsed a closure body
onto one line in `ep.rs`, which moved a `// SAFETY:` comment off the line preceding its `unsafe`
block, and clippy (denied in CI) fired. Put SAFETY comments *inside* the block's enclosing scope,
adjacent to the block, not before a wrapper expression that rustfmt may reflow. And: never run
fmt and clippy separately — this is the second reason `cargo ci` sequences them.

**`[workspace]` + `default-members = ["."]`** lets you add an xtask without changing what a bare
`cargo build`/`cargo test` means, and without touching CI's `--manifest-path rust/Cargo.toml`
invocations. Verified: still 268 tests (236 lib + 26 layering + 6 dump-capabilities).

**Zero dependencies in the xtask, on purpose.** The tool that tells you the tree is healthy must
not be able to fail for a reason of its own. Same instinct as clippy `--workspace` there: it
lints itself.

**State at end of session:** tree green, `cargo ci` passes all four checks on a machine with no
Vulkan SDK. Open coordination: `build.rs` glslc-vs-glslangValidator (Switch may want the change;
the file is mine — settle ownership before either of us edits). CI's grep-based layering lint
still has a `TODO(Tank)` and can now become `cargo test --test layering`, but `ci.yml` is
Trinity's.
