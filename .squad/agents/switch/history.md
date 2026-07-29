# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Vulkan Compute — device/memory/sync, SPIR-V shaders, pipelines
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-28T22:28:08-07:00 — full session details in decisions.md -->

### [SUMMARY] Sessions 1–6+: ENGINE.md, barrier abstraction, seams, device/memory/pipeline (2026-07-28)

**ENGINE.md authored (session 1):**
- Reference study: llama.cpp (build-time GLSL→SPIR-V, specialization constants, per-vendor tuning, lazy pipeline creation). ExecuTorch (VK_API_VERSION_1_1, buffer-only, one-time record, weight prepacking at compile phase, yaml variant tables).
- Chosen stack: `ash` + `gpu-allocator` (not vulkano, not wgpu).
- Buffer-only tensor storage for v0. One command buffer per subgraph (no per-op submissions).
- Per data-edge barriers (`vkCmdPipelineBarrier2`), not global. GLSL→SPIR-V at build time.
- `synchronization2` and `subgroup_size_control` structurally simplify engine; baseline decision delegated to Morpheus.

**Barrier abstraction (session 2) — `rust/src/vk/barrier.rs`:**
- Dual-backend `Barriers` enum: `Sync2Backend`, `LegacyBackend`. `Access`/`Stage` closed enums (no None). Single mapping table.
- Backend selected once at `Device::new`. `ep.force_legacy_barriers` session option forces legacy.
- `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE` env var: EP writes "sync2" or "legacy" to a file during `Barriers::select`. Used by Trinity's parity test.
- Layering lint: `barrier.rs` is the ONLY file allowed to name barrier types; `BARRIER_RULES` + `SYNC2_FIELD_RULES` in `tests/layering.rs`.
- ash 0.38 notes: `push_next` is safe but `#[must_use]`; extension paths are `ash::khr::*`; `vk::DependencyInfo` uses `vk::MemoryBarrier2` for execution-only sync2 barriers.
- Rust 2024: `#![deny(unsafe_op_in_unsafe_fn)]` — unsafe fn calls inside unsafe fn still need explicit `unsafe {}` + SAFETY comment. `const { assert!(...) }` preferred form.

**Backend probe + force_legacy wiring (session 3) — `rust/src/vk/device.rs`:**
- `Device` struct owns `ash_device`, `physical_device`, `caps`, `barriers`. `Device::new` is sole call site of `Barriers::select`.
- `should_use_sync2(caps, force_legacy) -> bool` extracted for testability (ash::Device cannot be zeroed — non-nullable fn pointers → UB).
- `caps::test_caps(sync2: bool)` defined outside `mod tests` so `device.rs` tests import without touching `synchronization2` token (layering compliance).
- Total: 185 tests after session.

**Engine seams for XL kernels (session 4) — `rust/src/engine.rs`:**
- Seam 1 (prepack): `TileConfig`, `PackKey`, `PackInput`, `PackOutput`, `PrepackRequest`, `PrepackResult`, `CompileContext` trait.
- Seam 2 (KV-cache aliasing): `bind_aliased_output` default method on `DispatchContext` (returns resolved input by default).
- Seam 3 (build.rs variant table): `VariantRow`, `parse_shader_variants`, two-path compile in `build.rs`. `cargo:rerun-if-changed` for `shader_variants.txt`.
- Seam 4 (indirect dispatch): `IndirectKernelRequest`, `dispatch_indirect` default method.
- llama.cpp assessment: block format mismatch = no code copying. Tiling strategy, subgroup reduction shape, dequant-in-register patterns **do transfer**. (D-S4-10 correction of Mouse's "useless" claim.)
- Rust trait default methods returning `Err(...)` = correct pattern for stubs that concrete engine impls override. All new methods have defaults; no existing implementors broken.
- Total: 195 tests after session.

**Real device enumeration (session 5) — `rust/src/vk/instance.rs`:**
- `Instance` struct: `_entry` declared first (dropped last); `ash::Entry::new()` returns None gracefully when no loader.
- `Instance::enumerate_capable_devices()` applies R1–R6 gate (pure `passes_gate` function, 15 unit tests).
- `Capabilities::required_device_extensions(api_version)` lives in `caps.rs` (keeps `synchronization2` token out of `instance.rs`).
- `Device::create(instance, capable, force_legacy)` — logical device creation, compute queue retrieval.
- `probe_devices()` sorts discrete-first.
- glslc fallback: Switch recommended hard SDK dep + escape hatch (168 SPIR-V blobs ≈ 1–3 MiB binary weight + staleness hazard). Morpheus ruled for hard SDK dep (OQ-4 resolved).
- Total: 245 tests after session.

**Memory / command / pipeline (session 6) — `alloc.rs`, `cmd.rs`, `pipeline.rs`:**
- `MemClass`: `DeviceLocal`, `Upload`, `Download`, `PackedWeights` (maps to `GpuOnly` — enforces "no dequantized weight in VRAM" at type level).
- `CommandPool` + `CommandRecorder<'pool>`: lifetime prevents use-after-pool-drop at compile time. `Drop` logs warning if `finish()` not called.
- `submit_and_wait()`: fence-based blocking submit. V0: one submission per subgraph.
- `PipelineCache`: lazy build+cache `(shader_stem, spec_constants) → (VkPipeline, VkPipelineLayout, VkDescriptorSetLayout)`. Shader module destroyed after pipeline creation.
- `DispatchDescriptorPool`: per-dispatch pool-and-reset. V0 simple model; M2+ replaces with persistent.
- `vk::SpecializationInfo` borrows both map_entries and data — return the storage from a helper, construct in caller scope.
- Total: 265 tests after session.

**Shader-less guard (session 7+):**
- `shaders::has_any()` = `SHADER_MODULES.is_empty()`. `probe_devices()` returns `vec![]` + logs warn. `get_capability_impl()` early-returns null + logs `[built-without-shaders]`.
- Belt-and-suspenders: `probe_devices` (factory init) + `get_capability_impl` (per-session). Future refactor can skip either; both together prevent claiming.
- OQ-4 condition 3 implemented: shader-less artifact advertises zero devices, claims nothing.
- Total: 268 tests.

**Key ash 0.38 / Rust 2024 facts (permanent reference):**
- All `ash::Instance` methods are `unsafe`. `ash::khr::synchronization2::Device::new` is safe.
- `push_next` is `#[must_use]`; use `let _ = props2.push_next(...)`.
- Extension paths: `ash::khr::*` (not `ash::extensions::khr::*`).
- `ash::Device::clone()` is cheap (Arc internally).
- `gpu_allocator::vulkan::Allocator::new()` is safe.
- `c"main"` is the modern c-string literal (Rust 1.77+).
- `bytes.len().div_ceil(4)` — clippy `manual_div_ceil` enforces this.
- `ash::Device` / `ash::Instance` cannot be zeroed (non-nullable fn pointers).

**Current test count: 268 (233 lib + 6 dump-capabilities + 26 layering + 3 shader-guard). All passing.**