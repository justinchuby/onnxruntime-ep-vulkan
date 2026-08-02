# Switch (Vulkan-Compute) — history.md

## Learnings

### [SUMMARY] Sessions 1–16+: ash engine, first execution, probe correctness, runtime extents (2026-07-28–2026-07-30)

**Sessions 1–11 (archived):** ENGINE.md produced; ash+gpu-allocator selected; buffer-only tensor storage; build-time GLSL→SPIR-V; one command buffer per subgraph; per-edge barriers. Dual-backend barrier abstraction (`legacy` + `sync2`). Capability probe implemented. Loader diagnostics added; `apiVersion` capped to loader-reported to avoid `ERROR_INCOMPATIBLE_DRIVER` on 1.0 loaders. R5 (subgroup BASIC) removed as gate criterion — now a probed capability governing shader variants, per §7.0.

**Session 12 (2026-07-29T08:13:58-07:00) — push_next chain bug root-caused:**
`let _ = props2.push_next(..)` discards the pNext chain in ash 0.38. Every chained capability (subgroup size, stages, float16, SSC) read zero before fix. Three-state probe added (`subgroup_probe_valid: bool`). Intel Iris Xe validated as spec-conformance oracle. `is_uma` predicate corrected: "every heap is DEVICE_LOCAL" (not "largest DEVICE_LOCAL heap is HOST_VISIBLE") — ReBAR on RTX 4060 was returning wrong `true`. `timestamp_period_ns` and `timestamp_valid_bits` added to `Capabilities`.

**Session 13 (2026-07-29) — teardown order:**
Vulkan struct teardown order enforced by field declaration order: `instance` must be LAST. Rust drops fields top-to-bottom; declaring `instance` first caused STATUS_ACCESS_VIOLATION in `cdylib_load`.

**Session 14 (2026-07-29T13:42:45-07:00) — §7.9 and R5 re-evaluation:**
§7.9 five rules for capability probe discipline. R5 rationale was false (lavapipe reading was probe bug); policy stands on §7.0 principle that shortfalls degrade op coverage, not device availability. `cargo ci` — green, 300 tests.

**Session 15 (2026-07-29T20:26:56-07:00) — SkipSimplifiedLayerNorm:**
`SkipSimplifiedLayerNormalization` kernel: single-pass, two outputs, 1 KiB shared memory, LOCAL_SIZE_X=256.

**Session 16 (2026-07-29T21:14:03-07:00) — descriptor-set lifetime:**
Fixed VUID-03047: descriptor sets released before command buffer submission. Descriptor pool now per-dispatch.

**Sessions 17–18 (2026-07-30) — runtime extents:**
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. `CompiledKernel` reads real shapes at Compute via `GetTensorTypeAndShape`. OQ-15 resolved for M0/M1: re-record per shape. Device 0 (Intel): 161 claimed, zero validation errors. Device 1 (RTX 4060): 161 claimed. Variable seqlen correct on both devices. 97 additional nodes unlocked without new kernels.
# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Vulkan Compute — device/memory/sync, SPIR-V shaders, pipelines
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-29T09:00:39-07:00 — full session details in decisions.md -->


<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- **[SUMMARY] Sessions 1–11: ENGINE.md through first real dispatch (2026-07-28–2026-07-29)** — **Stack chosen (session 1):** `ash` + `gpu-allocator`.
- **Cross-agent context appended (2026-07-29T09:00:39-07:00) — first-hardware round** — 📌 **Local GPU facts (2026-07-29):** Vulkan SDK at `C:\VulkanSDK\1.4.350.0` — NOT on default PATH; prefix it explicitly in every shell command that calls glslc or epctl.
- **Session 12 — Multi-device dispatch; Intel oracle; caps probe fix (2026-07-29T08:13:58-07:00)** — **Coordinator directive:** run dispatch on ALL capable devices with Intel as strictness oracle.
- **Session 12c — UMA predicate fix + timestamp seam (2026-07-29T09:47:45-07:00)** — **Coordinator directive:** fix failing `mem_class_download_maps_to_cpu_to_gpu` test, fix UMA predicate (Niobe's measurement), add timestamp seam for Niobe.
- **Session 14 — §7.9 probe-validity; R5 re-evaluation; cargo ci green (2026-07-29T13:42:45-07:00)** — **Coordinator task:** Re-check whether lavapipe's `supportedStages = 0` in session 10 was a real device fact or the push_next probe bug.
- **Session 15 — SkipSimplifiedLayerNormalization kernel; QGemv compile error fix (2026-07-29T20:26:56-07:00)** — **Coordinator task:** Implement `SkipSimplifiedLayerNormalization` — one of the three ops blocking a real model (Phi-3.5-mini) from running on GPU (64 of 366 nodes).
- **Session 16 — Descriptor-set lifetime fix (validation error VUID-03047) (2026-07-29T21:14:03-07:00)** — **Coordinator task:** Fix Vulkan validation error "A descriptor set is updated while bound to a recording command buffer, without UPDATE_AFTER_BIND" — caught by Trinity running the real Phi-3.5 model, identical on Intel Iris Xe and NVIDIA RTX 4060.
- **📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)** — ### Worktree layout and inbox portability constraint The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree.
- **Session 18 — Runtime extents: ENGINE_ACCEPTS_RUNTIME_EXTENTS flipped (2026-07-30T01:00:00-07:00)** — **Coordinator task:** Make tensor extents runtime parameters in the dispatch path, unblocking 97 nodes on Phi-3.5 (Mul 64, Sigmoid 32, Sub 1) that were declined [dynamic-shape] solely because the engine baked push constants at Compile time.
- **Session 20 — Counter scoping fix + profiling island fix (2026-07-30T01:32:15-07:00)** — **Coordinator task:** Reconcile the three contradictory numbers from the Phi-3.5 census run: `Claimed: 161, Islands: 0, counters {compile_calls:1, subgraphs_live:1, dispatches_executed:1}`.
- **Session 21 — Interior-pointer hazard + validation positive control (2026-07-30T02:30:00-07:00)** — **Coordinator tasks:** 1.
- **Session 22 — EP-side messenger, fence-leak plant, multi-run census (2026-07-30T03:52:28-07:00)** — **Coordinator tasks:** 1.
- **Session 23 — Dynamic-kernel binding mismatch: all-zero logits root cause and fix (2026-07-30T09:14:00-07:00)** — **Coordinator task:** Investigate all-zero logits in Phi-3.5 (161 dispatches, compute_failures=0, but argmax=0 on both devices).
- **Session 23 addendum — KV-cache "unwritten" explained (2026-07-30T08:16:02-07:00)** — **Tank's two-bug report (pre-fix):** Tank ran probe_run2.py BEFORE my fix was merged and found: - Outputs 1..64 (KV cache) differ bitwise between run 1 and runs 2/3 with identical feeds.
- **Session 24 — Mouse's independent fix merged; messenger positive control confirmed (2026-07-30T08:35:04-07:00)** — **Coordinator task:** Resolve merge conflict between `squad/switch` (HEAD, my binding fix from session 23) and `squad/mouse` (Mouse's independent fix for the same root cause).
- **Session 25 — VkQueryPool GPU timestamps + tracer wiring (2026-07-30T11:27:08-07:00)** — **Coordinator task:** Build VkQueryPool GPU timestamps behind `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1`; wire tracer into execution path; emit `PartitionStats`; confirm messenger positive control.
- **Session 26 — Multi-node island merge conflict resolution; clippy fixes (2026-07-30T15:41:27-07:00)** — **Coordinator task:** Merge `origin/main` (dc36166, Mouse's island-count fix: 321 → 33 islands, 3.7× Intel speedup) into `squad/switch`; resolve `session.rs` conflict; fix 29 clippy errors from Mouse's new partition/clustering code in `ep.rs`; verify with Mouse's tests and messenger positive control.
- **Session 27 — Tracer end-to-end, timestamp falsifiers, hypothesis verdict, island splitters, PartitionStats fix (2026-07-30T17:22:33-07:00)** — **Coordinator tasks:** (1) Verify tracer by running.
- **Session 29 (2026-07-30T17:19:29-07:00) — Sub-phase attribution + weight-tensor GPU buffer cache** — **Coordinator assignment recap:** (1) characterize what is expensive inside `vulkan.record`, (2) confirm device-1 phase split, (3) implement and measure the fix.
- **Session 30 (2026-07-30T20:34:34-07:00) — CB-caching prediction; batching feasibility; post-cache bottleneck shift** — ### Intel submit anomaly — explained Submit inf1 = 77.6ms/call, inf2 = 0.6ms/call, inf3 = 0.4ms/call.
- **Session 31 (2026-07-30T21:03:48-07:00) — Measurement contamination acknowledgement; quiet-machine request** — **Coordinator finding:** command-buffer recording inflates 9.5× under CPU contention from concurrent agents compiling Rust on the same machine.
- **Session 32 — Device-label fix; messenger positive control (valid); timestamp falsifiers; bench parsers (2026-07-30T19:47:00-07:00)** — **Context:** Resumed from summary.
- **Session 33 - Cache byte sweep confirms 2642x reduction; two-VkDevice flag (2026-07-30T22:06:01-07:00)** — **Context:** Merged origin/main (692e7d0).
- **Session 34 - Standing perf directive; MATCH + byte sweep reconfirmed (2026-07-30T22:23:35-07:00)** — Context: Coordinator issued standing directive - performance is first-class, continuous.
- **Session 35 - §6.5 VkDevice seam; byte sweep stands (2026-07-30T22:32:54-07:00)** — Context: Morpheus ruled §6.5 - exactly one VkDevice per (physical device, EP instance).
- **Session 35 (2026-07-31) — Defects 1/2/3 closed on real Phi-3.5; offer gated** — Fresh session; prior work merged.
- **Session 36 - §6.5 CLOSED (process-global VkDevice); containment RED is Niobe's gate; record residual (2026-07-31T20:28:45-07:00)** — Merged origin/main 77d5d2a into squad/switch (clean, 23 files).
- **Session 37 — 2026-08-01 — the GEMV column tile, and the Intel gap separated from the hardware** — Merged origin/main (efbf18c) into squad/switch — clean, 33 files, brought in Niobe's gpu_steady_tail, bench/exec_census.py, Mouse's ops/indexing.rs, the new gather and simplified_layer_norm shaders.
- **Session 38 — 2026-08-01 — packed 128-bit loads, and the Intel residual closes** — Merged origin/main (5eda83b — Fact Checker's docs/PERF.md and his hardware-clock decision record).
- **Session 39 — 2026-08-01 — the allocator adopts by identity; Tank was right that selector 0 was luck** — Merged origin/main (20cb57b).
- **Session 42 — 2026-08-01 — counts over clocks; the index space closes; the A/B does not** — Merged `origin/main` (`16f40ef`), which brought Niobe's `bench/device_state.py` certification gate.
- **Session 43 — 2026-08-01 — the two owed items: the leaked-device cost, and the frame reconciliation** — Both delivered. Neither needed the GPU, which was the point (three agents measuring concurrently).
📌 Team update (2026-08-01T20:39:12-07:00): Tank found `OwnedDevice::create(device_index)` ignores its `device_index` argument and resolves only through `select_device` — a fourth face of the index-space defect (after `epctl --probe-loader`, the `dispatches_executed`/`compute_calls` mismatch, and the two-selector §6.5 wrinkle). Consequence: two `HandleRegistry`s keyed by different device indices currently collapse onto the same physical device, so a two-*device* `MIXED`-frame mix is not reachable on Tank's box today, though a two-*frame* mix (same device, different declared frames) already is. Flagged as a dependency: if this index fix lands, `SHARED` becomes correct on selector 1 and the two-device mix becomes reachable — Tank's `MIXED` state should land before or with this fix, not after. — decided by Tank
