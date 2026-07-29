# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Lead / EP Architect — architecture, design docs, scope, review
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-07-28T17:59:54-07:00 — `docs/DESIGN.md` authored (architecture of record)

**The MLX reference is a pipeline, not a backend.** What transfers from `onnxruntime-mlx` is the
plugin-EP integration: `CreateEpFactories`/`ReleaseEpFactory`, the `#[repr(C)]` embed-ORT-struct-at-
offset-0 vtable pattern, `Box::into_raw`/`from_raw` ownership, panic guards at every `extern "C"`
entry, the `(domain, op_type, [min,max] opset) → {handler, claim}` registry, `NodeView`/`NodeDesc`,
convex clustering (union-find + reachability bitsets — non-convex fusion creates a cycle ORT
rejects), and the repo shape. What does **not** transfer is everything MLX supplied for free:
memory, scheduling, dtype genericity, and op semantics. Roughly: MLX gave the EP a backend; Vulkan
gives us a driver.

**The single structural divergence that drives all the others: unified vs. explicit memory.** The
MLX EP advertises no device allocator, returns null from `GetDefaultMemoryDevice`, and copies out
with one `memcpy`. Vulkan forces us to own `OrtAllocator`, `OrtDataTransferImpl`, staging, coherence,
barriers, and weight prepacking. Any future "just mirror MLX" instinct must stop at this line.

**ORT's allocator API is pointer-based; a `VkBuffer` is not a pointer.** This is the sharpest
concrete ABI problem in the project. Decided: opaque tagged-handle registry resolving to
`(VkBuffer, offset)`. Rejected `VK_KHR_buffer_device_address` — optional on every baseline, and
MoltenVK support is partial.

**Vulkan version floors: verify the premise before designing to it.** llama.cpp does **not** require
Vulkan 1.3 — its hard runtime floor is 1.2 (`if (api_version < VK_API_VERSION_1_2) throw`), it sets
`VkApplicationInfo::apiVersion` to whatever the instance reports, and `vulkan-shaders-gen.cpp`
compiles its **base shaders at `--target-env=vulkan1.2`**, reserving `vulkan1.3` for the NVIDIA
cooperative-matrix-2 variants; the `vulkan1.3` in its CMake is an extension-availability probe.
ExecuTorch hardcodes `VK_API_VERSION_1_1` with a 1.0 fallback path and initializes VMA at
`VK_API_VERSION_1_0`. Both widely-cited "requires 1.3" claims are wrong — independently confirmed
by Fact Checker (audit trail, claims 1–2, contradicted). Generalizable: when a design proposal
cites a project's requirement, read that project's source before building on it.

**The baseline decision generalizes: require a capability set, not a version number.** Switch found
that only two features materially simplify the engine (`synchronization2`, `subgroup_size_control`)
and both exist as standalone extensions on 1.1/1.2 drivers. So requiring the *features* gives the
single barrier code path and guaranteed subgroup sizing without a version floor's coverage cost.
Also learned from Link's data: **on Android the Vulkan 1.2 tier barely exists** — devices jumped
1.1 → 1.3 — so a 1.2 floor pays nearly the full Android cost of 1.3 while delivering less on desktop.

**Because CPU fallback is always correct, a plain output comparison is a vacuous test.** Every op
test must additionally assert the node ran on `VulkanExecutionProvider`. This is the highest-value
testing invariant in the project and the first thing to check in a review.

**Claim rate is a bad metric; fused-region compute volume is the good one.** One unclaimed node in
the middle of a graph splits it into two islands with a device round-trip between them. Op priority
is "does this merge two islands", not "is this op easy". Benchmarks must report island count and
largest fused region alongside wall time or the number is not interpretable.

**Prior-art split worth remembering:** llama.cpp re-records command buffers every eval (fine for a
few large matmuls, wrong for many small dispatches); ExecuTorch records once at init and replays,
with an explicit `prepack()` step for constants. For an ONNX EP the ExecuTorch model is right, and
it maps cleanly onto the MLX EP's `compiled.rs` (`mlx_compile`) role → our `recorded.rs`.

**Process:** Switch's `ENGINE.md` and Link's `PLATFORMS.md` already existed when I started, despite
the spawn prompt assuming they might not. Check the working tree before writing "pending X's
findings" — reading a sibling's actual output produced a materially better decision than reasoning
around its absence would have.

