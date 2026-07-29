# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Vulkan Compute — device/memory/sync, SPIR-V shaders, pipelines
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-07-28T17:59:54-07:00 — ENGINE.md authored; reference implementation study complete

**Produced:** `docs/ENGINE.md` — full Vulkan runtime and shader architecture design (9 sections,
~550 lines). Decisions inbox: `.squad/decisions/inbox/switch-engine-design.md`.

**Reference study findings:**

- **llama.cpp ggml-vulkan:** `vk_device_struct` holds compute + transfer queues, vendor ID,
  UMA flag, pinned memory list, subgroup size. Build-time `vulkan-shaders-gen` compiles GLSL
  `.comp` files into SPIR-V and embeds them as C++ headers. Specialization constants tune
  workgroup/tile sizes per vendor; push constants carry per-dispatch matrix dims. Cooperative
  matrix variants (`flash_attn_cm1.comp`, `flash_attn_cm2.comp`) are behind
  `VK_KHR_cooperative_matrix` / `VK_NV_cooperative_matrix2` capability probes. Lazy pipeline
  creation with mutex-deduplication.

- **ExecuTorch ET-VK:** Vulkan 1.1 baseline. `Context` owns device, queues, command pools,
  descriptor pool, VMA allocator. `Adapter` wraps the physical device + feature detection.
  `vTensor` supports buffer or image backing; buffer-only recommended for linear ops. GLSL
  shaders compiled offline by `glslc`; yaml variant tables drive dtype/layout specialization.
  Weight prepacking at lowering time (ExecuTorch's equivalent of our Compile phase). Descriptor
  pool per context, freed and reallocated per inference graph execution.

**Key decisions made:**
- `ash` + `gpu-allocator` as Vulkan crate stack (not vulkano, not wgpu).
- Buffer-only tensor storage for v0.
- One command buffer per subgraph (no per-op submissions).
- Per data-edge barriers (`vkCmdPipelineBarrier2`), not global barriers.
- GLSL → SPIR-V at build time, embedded in cdylib; no runtime shader compiler.
- `synchronization2` and `VK_EXT_subgroup_size_control` are the only 1.3 features that
  structurally simplify engine code. Morpheus decides the final baseline.

**Unverified / follow-up:**
- MoltenVK 1.3 feature coverage completeness — Link is analyzing.
- `bufferDeviceAddress` on MoltenVK: assessed as partial; do not rely on in v0.
- `VK_KHR_cooperative_matrix` on Android: rare as of 2024–2026; capability-gated path only.
- vulkano advanced-feature gap (bindless, subgroup extensions) — assessed from issue tracker
  history, not verified against current vulkano HEAD.

---

### 2026-07-28T19:16:08-07:00 — Barrier abstraction implemented; ENGINE.md §6.2/6.3/§8 updated

**Context:** Morpheus froze `DESIGN.md §7` after Link found `VK_KHR_synchronization2` present
on only 68.57% of Android devices. The Khronos emulation layer was rejected (AOSP loader only
searches the APK owner's `nativeLibraryDir`; we are a plugin, not an APK owner). Device gate
becomes 6 hard requirements, no required extensions.

**Produced:**
- `rust/src/vk/barrier.rs` — dual-backend `Barriers` enum (`Sync2Backend`, `LegacyBackend`),
  `Access`/`Stage` closed enums (no `None`), `BufferDep`, single mapping table, 10 unit tests.
- `rust/src/vk/caps.rs` — `Capabilities` struct, `SubgroupSizeRange`, `probe()`, `detect_uma()`,
  7 unit tests.
- `rust/src/vk/mod.rs` — module root with `#![allow(dead_code)]`.
- `rust/src/ep.rs` — `force_legacy_barriers: bool` added to `EpOptions`.
- `rust/tests/layering.rs` — 6 new barrier boundary tests; `BARRIER_RULES`, `SYNC2_FIELD_RULES`.
- `rust/src/ops/common/mod.rs` — created (Mouse had not created it; crate failed to compile).
- `rust/src/ops/shader_variants.txt` — generated via `MOUSE_BLESS_VARIANTS=1 cargo test`.
- `docs/ENGINE.md` §6.2 — rewritten to use `buffer_deps(&[BufferDep { ... }])` API.
- `docs/ENGINE.md` §6.3 — sync table updated to reference `DESIGN.md §7.5` and `barrier.rs`.
- `docs/ENGINE.md` §8 — fully rewritten to reflect frozen §7: Vulkan 1.1 baseline, dual-backend
  replaces 1.3-required sync2, subgroup_size_control MoltenVK quirk documented.
- `.squad/decisions/inbox/switch-barrier-abstraction.md` — 7 binding design decisions recorded.

**ash 0.38 API notes (for future reference):**
- All `ash::Instance` methods are `unsafe` in ash 0.38.
- `ash::khr::synchronization2::Device::new(instance, device)` is **safe** (not `unsafe`).
- `push_next` on structs is safe but `#[must_use]`; use `let _ = props2.push_next(...)`.
- Extension paths are `ash::khr::*` not `ash::extensions::khr::*` (old ash <0.37 API).
- `vk::DependencyInfo` has no `src_stage_mask`/`dst_stage_mask`; execution-only sync2 barriers
  use `vk::MemoryBarrier2` in the `p_memory_barriers` slot.

**Rust 2024 edition notes:**
- `#![deny(unsafe_op_in_unsafe_fn)]` — inside an `unsafe fn`, calls to other `unsafe fn`s still
  need explicit `unsafe {}` blocks with `// SAFETY:` comments.
- Raw pointer value assignments (not dereferences) are SAFE; no `unsafe {}` needed.
- `const { assert!(...) }` is the clippy-preferred form when the operand is a `const bool`.

**Layering lint notes:**
- `contains_token` uses whole-word matching: adjacent chars must be non-alphanumeric/underscore.
- `cmd_pipeline_barrier` does NOT match `cmd_pipeline_barrier2`; both must be listed explicitly.
- Clippy `large_enum_variant`: box both `Barriers` variants to avoid the warning.

**Mouse coordination:**
- Mouse modified `registry.rs` and `ops/mod.rs` but did not create `ops/common/mod.rs`.
- Mouse's `shape_plan.rs` had a scalar-input broadcast stride bug (stride loop started at 0
  instead of `off`); fixed in the same session.
- Mouse's `claim.rs` had a clippy error: `assert!(REQUIRE_STATIC_SHAPES)` → `const { assert! }`.

**Final state:** `cargo build`, `cargo clippy --all-targets -- -D warnings`, `cargo test` all
clean. 156 tests pass (141 lib + 15 integration).
