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


---

## 2026-07-28T19:16:08-07:00 — Freezing DESIGN.md §7 (OQ-1 resolution)

**A "provisional" decision is only honest if you actually reverse it when the data arrives.** My
§7.2 of two hours earlier required `synchronization2` and `subgroup_size_control` and said so
"pending Link's findings". Link's findings said 31.43% of Android and 12.22% of Windows would be
excluded. I reversed it. The lesson is not "I was wrong" — it is that marking something provisional
creates an obligation, and the whole point of the capability-set framing was that the cost is
*measurable*, unlike a version-number floor. Design in units you can later measure.

**Never require a feature flag when you only need a property.** Link caught that MoltenVK reports
the `subgroup_size_control` *extension* (Vulkan 1.3 promotes it to core) while
`subgroupSizeControl` is `VK_FALSE`, because Metal cannot control SIMD-group width per pipeline.
Requiring the flag would have silently excluded all of macOS/iOS — and probably lavapipe and
SwiftShader, i.e. our own CI. Generalized rule: **a requirement that excludes the machines you test
on is a requirement you have not tested.** Always ask "extension string, property value, or feature
flag?" — they are three different requirements with three different coverage numbers.

**The right formulation of a hardware requirement is usually a correctness rule, not a gate.** For
subgroup width the answer was not "require the extension" but "a shader whose correctness depends
on an exact subgroup width may only be selected when the width is *known* exactly, otherwise use
the portable variant". That costs nobody coverage and preserves the actual guarantee. Look for this
shape whenever a capability requirement is proposed.

**The frozen principle worth carrying to any future EP: the device gate is minimal; capability
shortfalls degrade op coverage, not device availability.** A hard device requirement must be
justified by "no op we will *ever* ship can work without it". Everything else is a claim predicate.
This falls straight out of conservative-claiming-with-clean-CPU-fallback, and it makes the failure
mode "runs fewer ops" instead of "device does not exist".

**Verify the mechanism before you accept the mitigation.** Link proposed bundling the Khronos
`VK_LAYER_KHRONOS_synchronization2` layer, citing wgpu/Dawn/Godot as precedent. Two things
collapsed on inspection: (1) the AOSP Vulkan loader ignores `VK_LAYER_PATH`, uses no JSON
manifests, and searches only the *host application's* `nativeLibraryDir` — so a plugin `.so`
`dlopen`ed into someone else's process cannot enable a layer on retail Android at all, which was
100% of the motivation; (2) all three cited projects use legacy `vkCmdPipelineBarrier` exclusively
and none ships the layer. **A precedent you have not read in the source is not a precedent.** We
are a plugin, not an application — that distinction invalidates a whole class of otherwise-standard
Vulkan advice (layers, environment variables, instance ownership), and it should be the first thing
I check on any proposal of this shape.

**When you authorize a dual code path, ship the seam and the test lane in the same decision.** A
dual path becomes a bug farm exactly when it is `if caps.x { } else { }` at every call site. The
decision that makes it survivable is: one internal API, our own closed enums (so the legacy backend
is *total* by construction — no `VK_PIPELINE_STAGE_2_NONE` to translate), backend selected once at
device init, one mapping table, and a session option forcing the minority path so CI executes it
every run. Without the forced lane, the path we carry for 31% of Android would be run by no test we
own, because our Linux CI has sync2 99% of the time.

**Coverage-count is the metric most likely to be gamed by an ambitious plan.** When the op-coverage
ambition was raised, the constraint I had to write down explicitly was minimum viable subgraph
size — high op count that shreds a graph into transfer-dominated fragments is a regression wearing
a coverage badge. Attach a metric (island count, largest fused region) or the constraint is a
slogan.

**Don't resolve an open question on a symbol name.** ORT 1.28's
`CreateExternalResourceImporterForDeviceImpl` looks like a better answer to OQ-3 than my
opaque-handle registry, and it may well be — but inferring semantics from a name is the same
mistake as "llama.cpp requires 1.3", which we had already made once this week. Recorded it as a
live alternative with its cost (a 1.28 ABI floor, which is itself a compatibility regression) and
left OQ-3 open pending Fact Checker.
