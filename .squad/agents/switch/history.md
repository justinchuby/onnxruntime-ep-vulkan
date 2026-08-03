# Switch (Vulkan-Compute) — history.md

<!-- SUMMARIZED by Scribe 2026-08-03T04-55-00-07-00 -- sessions 1-46m condensed below (was 92,884 bytes / 1296 lines, long-standing deferral since 88,927). Full text lives in git history and in decisions.md Rounds 4-9. -->

## Project Context

- **Owner:** Justin Chu. **Project:** onnxruntime-ep-vulkan — cross-platform Vulkan plugin EP for ONNX Runtime, Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` layout mirrored.
- **Stack:** Rust cdylib, Vulkan 1.1+ compute, SPIR-V/GLSL, ORT C API, Python bindings, GH Actions CI.
- **Cross-platform mandate:** Windows/Linux/Android/macOS(MoltenVK); NVIDIA/AMD/Intel/Adreno/Mali; lavapipe/SwiftShader for GPU-less CI.
- **My focus:** device/memory/sync, SPIR-V shaders, pipelines. Created 2026-07-28T17:52:04-07:00.
- Local GPU facts: Vulkan SDK `C:\VulkanSDK\1.4.350.0` not on PATH — prefix explicitly. Two devices: Intel Iris Xe (UMA, 32 KiB shared, oracle for spec-conformance) and NVIDIA RTX 4060 Laptop (discrete, 48 KiB shared). Physical/best-first index spaces are inverted on this desk (RTX = physical 1 / position 0).

## Sessions 1-43 (ash engine, first execution, runtime extents) — one-line-per-session

`ash`+`gpu-allocator` stack; dual-backend (sync2/legacy) barrier abstraction in `barrier.rs` (only file allowed to name barrier types); `push_next` must-use bug root-caused (silently dropped pNext chains); teardown order = field-declaration order (`instance` last); §7.9 probe-validity rules, R5 (subgroup BASIC) demoted from gate to probed capability; `SkipSimplifiedLayerNormalization` kernel; descriptor-set lifetime fix (VUID-03047); `ENGINE_ACCEPTS_RUNTIME_EXTENTS` flipped, runtime shapes read at Compute, unblocking 97 nodes; first real GPU dispatch (NVIDIA), 3 ash/sync2 bugs fixed; dynamic-kernel binding mismatch caused all-zero logits, fixed; VkQueryPool GPU timestamps + tracer; island-merge, clippy; sub-phase attribution, weight-tensor GPU buffer cache (2642x upload reduction); §6.5 ruled (exactly one VkDevice per physical device + EP instance); GEMV column tile; packed 128-bit loads close Intel gap; allocator adopts by identity, not selector luck; Tank found `OwnedDevice::create` ignored its `device_index` argument (4th face of the index-space defect) — held pending sequencing with Tank's `MIXED` state.

## Sessions 44-46m (2026-08-01 -- 2026-08-03) — performance push, KV correctness, KV round-trip

**Instrumentation / measurement discipline (44, 44d, 44e, 45):**
- `run_disturbance.py`: same-ordinal RSD vs whole-series RSD; threshold 0.20% sits in an empty gap between clean/dirty traces, but the gap narrowed to 11-20% (Intel: a single point) as more traces arrived — **an "empty gap" is a fact about the sample, not the population; do not quote it as fixed.** Corroborates existing refusals, adds zero new ones today.
- Retracted own claim that the guard was independent of `localise()`: they correlate at Spearman 0.919 — agreement is evidence *for* redundancy, not for two witnesses.
- `localise()` explains 84-87% of same-ordinal dispersion by dividing out per-inference level; splits disturbance into SUBMISSION_LEVEL (clock/power excursion) vs PER_DISPATCH (interleaved foreign work).
- **Structural limit, keep:** no statistic computed from inside one series can detect a bias that scales the whole series uniformly — dispersion is invariant under a constant multiplier. `baseline_certified` is cleanest by every dispersion measure and is 21.4x wrong; only `check_device_state.py` (evidence from outside the series) refuses it. No future dispersion guard replaces that external check.
- `ci/check_tautological_assertions.py`: 1,056 assertions scanned, 0 detections — a regression barrier, not proof of correctness (Link later classified it PLANTED/UNDEMONSTRATED). Its own build hit 3 instances of the exact defect it hunts (silent zero-coverage on Python files from a whitespace bug, string-blanking false positives, missing-polarity false positive on a NaN idiom). Does **not** detect two different expressions that happen to evaluate equal at runtime (mutation testing territory, not closed).
- Criterion 10 covered 1 output of 65: a planted all-zero KV write (64/65 outputs zero) passed every existing gate. Added `NOT_PERFORMED` as a third verdict (distinct from AGREE/DISAGREE) so degenerate-both-sides comparisons can't read as agreement; guard runs on both oracle and subject. Own tautological test (`tol == expected` compared to itself) caught by mutation after the static screen missed it.

**Roofline / KV bandwidth (46, 46b, 46f, 46g):**
- Recomputed weight stream from the graph: 1996.8 MiB int4+scales (scales = 11.1%), floor 8.18 ms at ctx0; measured 12.1847-12.1869 ms GPU-busy reproduces = 67.1% of roofline, 1.49x headroom (wall-clock speedup ratios separately withdrawn as CPU-fallback-contaminated — different instrument).
- Corrected own error: charging activation re-reads to DRAM gave an impossible 97% of spec peak — activation bytes are L1 hits (load issue), not DRAM; only weight bytes are DRAM traffic. `QB_MAX_COLS` 8->16 change removed 15.4% of activation load bytes; capped at 16 (not 32) because 32 columns would consume the entire §7.2 guaranteed 16 KiB shared-memory floor on one workgroup.
- InB load amplification measured exactly 1.000000 (loads x width = graph weight bytes) — blocking is not a defect; checked the check itself for tautology (two independent factors: 1 load/blob, 1 workgroup/blob).
- Fusion prize is only 0.47% (intermediates are vectors at batch 1, negligible even at 354 dispatch boundaries — "a large count of small things is not a large thing").
- **KV cache term is unbounded and was invisible at the low-context regime the 12.18 ms figure was measured in**: 0% of traffic at ctx0, growing to a corrected **82.2% at ctx 8192** (initial 60.5% figure was low — corrected after present-copy accounting below). Model has no grouping (32 KV heads = 32 query heads), so this EP's traffic is 4-8x a genuinely grouped model's.
- Refuted aliasing `present` onto `past` (present is always strictly larger under ORT's growing convention — infeasible everywhere, not just on the test set). Landed the real 1/3-unit reduction instead: fused the present-copy into the attention read (write-on-load) — predicted 1.377x speedup at ctx 8192, measured 1.377x exactly; scales with grouping ratio (1.5x/1.2x/1.11x at 1:1/4:1/8:1 — never a regression).
- **Any figure quoted against "the roofline" must state the context length** — the floor itself moves with past_len (8.18 ms at 0, 14.51 ms at 4096, 20.80 ms at 8192).

**GQA correctness (46c, 46d, 46e, 46f):**
- `session.rs` else-branch bug: symbolic (internally-consumed) island outputs were dropped from `computed_descs`, leaving 323/459 claimed nodes never executing (0 dispatches). Fixed by keying the consumer off "did an earlier kernel produce this token." Result: 33/73 islands executing (all of them). A near-null-result control (2/13 with and without the fix) was caught before reporting only because it ran against the *merged* tree, not an escape-hatch tree with too few islands to discriminate.
- GQA fixed 16.726 -> 0.00072939 worst_rel (MATCH). Three defects sharing one root: the name->token map was only built when `plan.nodes.len() > 1`, so with one node GQA's non-sequential input resolution order silently drifted the index space (4th instance of the two-index-space defect class this project). Sub-defects: `rotary_dim` wrongly defaulted to `head_dim` (must be derived from `cos_cache`'s actual second dim); `present` wrongly assumed `== past` (ORT has two conventions — shared-buffer and growing; both Phi-3.5 graphs are growing, present is one token longer, dropped writes silently); `unwrap_or(0)` on a missing desc collapsed to the same value as a genuine empty-past-cache, masking a caller-side bug as a plausible input.
- **General rule, all four defects share this shape:** a quantity with two definitions that coincide on the test set is invisible to that test set — never think harder about the quantity, run the case where the definitions separate. (decode-only test set hid: present==past, rotary_dim==head_dim, past_len_max==0, and the seqlens_k/total-seq formula below.)
- Sibling-key race in `gqa_f16.comp` (unordered read of `present` written by sibling invocations) fixed by recomputing K/V per-invocation from read-only `packed_qkv` instead of reading `present`.
- Prefill-race defect (found by the same fix's own falsifier, unpredicted, larger than the race itself): `past = seqlens_k[b] + 1 - seq_len`, which only equals the wrongly-assumed `seqlens_k[b]` when `seq_len == 1` (decode). Prefill indexed past_key out of extent. Fixed; DIVERGENT 774.8 -> MATCH 0.000724.

**KV round trip / device residency (46h, 46i, 46j, 46k, 46l, 46m):**
- Input cache was serving stale KV: keyed on `(cpu_ptr, byte_size)` with a 32 KiB floor and no content check, so growing past-KV crossing 32 KiB (past_len >= 6) was cached as if it were an immutable weight — every inference after the first replayed the first inference's KV. Root sub-finding: ORT does not mark fused-node inputs as constant even when the body node is (388/457 constant at the body, 0/457 at the fused boundary) — recovered via name-join in `subgraph_plan`. Fix made upload MEASURED at exactly 393,216 B/past-token (matches Niobe's independently-derived figure to the byte).
- ctx-512 intermittent `VK_ERROR_DEVICE_LOST`: driver/GPU fault (nvlddmkm events), not a TDR watchdog timeout (TDR signature Display/4101 reads zero in the same window). Faults cluster in one 79-minute window and do not reproduce on ~460 heavier inferences run afterward; both pre-fix and post-fix binaries fault inside the window under a stash-and-rebuild control — **mechanism named, reproduction not currently available, not claimed fixed.** `device_losses` counter added (emits at zero so probes can refuse on it); driver's real wording ("The logical device has been lost") matched neither of the first-try string patterns and fell through to the wrong bucket until fixed against the verbatim text.
- ORT *does* place fused-node outputs in this EP's device memory when armed (195/195 vs 0/195 disarmed) — refutes "ORT forbids device-resident KV" as originally framed. But bound outputs came back **entirely zero**: `transfer.rs`'s own invariant says host staging is authoritative and the device buffer is only a mirror — nothing wrote the host block. A first-cut scorer nearly certified the all-zero lane as a win (1.0 vs 1.9958, beaten by returning nothing) — fixed with a degeneracy guard and an `ep`-vs-`bound` (not `cpu`-vs-`bound`) primary criterion.
- Push-constant fix: every dispatch now writes the full declared 128-byte range unconditionally (6 distinct shortfalls found by Best-Practices validation) — padded rather than shrunk per-kernel because the pipeline cache key has no push-size component.
- **The fix:** `transfer.rs` no longer holds one global rule about which copy of a span is real — each span now carries `device_authoritative`, set only when a dispatch is about to write that span's device buffer directly. Result: `KV_CAN_STAY_DEVICE_RESIDENT`, bit-identical outputs, all nonzero. Two defects found only by running the previously-failing case: a partial host write was revoking authority over a *whole* span; `alloc_device_authoritative_spans` had two producers with different definitions summed into one counter (caught by a control lane reading 1 with zero binds) — split into `alloc_device_authority_grants`.
- **Not yet claimed:** the round trip was *moved* (host-lane pays 1792 B/step on readback), not removed; ctx > 4096/8192 is unmeasured on this hardware.
- On the real Phi-3.5 graph (64 KV outputs, 6-step decode chain): host lane slope 393,216 B/past-token (reproduces Niobe's figure to the byte); resident lane flat at 64,128 B, **slope 0**; bit-identical logits both lanes, both devices. Root cause of the remaining input-side round trip: `vk/session.rs` Step 1b looped `host_backing_for` over *all* inputs, downloading every device-authoritative KV span to return a staging address the dispatch never reads — one `continue` fixed it. Two of the round's own findings were probe bugs, not runtime defects: `binding.get_outputs()` (binding order) indexed against `sess.get_outputs()` (session order) fabricated a false "near-zero CPU-bound output" defect and a false 6,144 B/step residual — caught by step-0 disagreeing between two lanes computing the same inference, not by inspection. **Standing rule:** a bandwidth lane must carry a correctness control sharing its inputs exactly, read before the byte count.

## Owed / open at end of Round 9
- Certified NVIDIA packed-loads A/B still unobtained (3 attempts, never a quiet window).
- General GQA grouping case (Nq/Nkv != 1) never run — Phi-3.5's 1.00 ratio is degenerate.
- Growing-context KV round-trip measurement beyond ctx ~4096-8192 unmeasured; VRAM cost of arming device-resident KV on an 8 GB laptop GPU not yet measured (deferred jointly to Niobe).
- `DEVICE_MEMORY` default stays OFF pending the above.
- Localisation (`localise()`) built but not wired to any lane; whether foreign GPU work moves `gpu_steady_tail()` still open.

📌 Team update (2026-08-03T04-55-00-07-00): Link retired his own Session-13 method of quoting a rebuilt-DLL hash as evidence a binary changed: six builds of an unchanged tree produced six distinct Windows DLL hashes, so a hash witnesses nothing about content. Every DLL hash quoted in your sessions above (e.g. `A9A381602D8B4014`, `D408A901C4F6A454`) is a build identifier only, never evidence that the binary differs from a prior one — do not cite a hash change as proof of a code change going forward. — decided by Link
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
**Stack chosen (session 1):** `ash` + `gpu-allocator`. Buffer-only tensor storage, one command buffer per subgraph, per-edge barriers, GLSL→SPIR-V at build time.

**Barrier abstraction (session 2) — `rust/src/vk/barrier.rs`:** Dual-backend `Barriers` enum (`Sync2Backend` / `LegacyBackend`). Backend selected once at `Device::new`. `ep.force_legacy_barriers` session option. `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE` env writes "sync2"/"legacy" for Trinity parity test. `barrier.rs` is the ONLY file allowed to name barrier types (enforced by `tests/layering.rs`).

**ash 0.38 / Rust 2024 permanent reference:** `push_next` is `#[must_use]`; `ash::khr::*` path. `ash::Device` cannot be zeroed. `entry.try_enumerate_instance_version()` not the panicking form. `c"main"` for C strings. `bytes.len().div_ceil(4)` (clippy enforces). `#![deny(unsafe_op_in_unsafe_fn)]` requires explicit `unsafe {}` + SAFETY comment even inside unsafe fns.

**Engine seams (session 4):** `TileConfig`/`PackKey`/`PrepackRequest`; `bind_aliased_output` default method (KV-cache aliasing); `VariantRow`/`parse_shader_variants` in `build.rs`; `IndirectKernelRequest`/`dispatch_indirect` default. All new methods have defaults; no existing implementors broken. llama.cpp tiling strategy + subgroup reduction shape + dequant-in-register patterns transfer (D-S4-10); no code copying.

**Device enumeration (session 5):** `Instance::enumerate_capable_devices` applies R1–R6 gate (`passes_gate` pure function, 15 unit tests). `probe_devices` sorts discrete-first. `MemClass`: `DeviceLocal`, `Upload`, `Download`, `PackedWeights`. `PipelineCache`: lazy `(shader_stem, spec_constants) → (VkPipeline, layout, dset_layout)`. Test count reached 268.

**Shader-less guard (session 7+):** `shaders::has_any()` = `SHADER_MODULES.is_empty()`. `probe_devices` + `get_capability_impl` both early-exit. OQ-4 escape hatch: `ALLOW_MISSING_GLSLC=1` produces inert artifact.

**Session 9 — ICD diagnostics + apiVersion fix:** `Instance::create` caps requested `apiVersion` to loader version before calling `vkCreateInstance` (fixes latent `ERROR_INCOMPATIBLE_DRIVER` on 1.0 loaders; use `try_enumerate_instance_version` not the panicking form). Full loader diagnostic emitted on any create failure. `epctl --probe-loader` exits 1 when no capable device passes gate.

**Session 10 — R5 removed from gate (lavapipe `supportedStages=0`):** R5 (subgroup BASIC in compute) removed from `passes_gate`. Now stored in `Capabilities::subgroup_basic_in_compute`. `assess_gate` evaluates all criteria without early exit for verbose diagnostics. 283 tests.

**Session 11 — First real dispatch on NVIDIA RTX 4060 Laptop GPU:**
- `add_f32_dispatches_end_to_end`: 1024 f32 elements, exact arithmetic, zero validation layer errors. §9.1.2 "no shader has ever executed" is now false.
- **Bug D-S11-01:** NVIDIA 1.4 doesn't export `vkCmdPipelineBarrier2KHR` (extension alias); fixed: `Sync2Backend` is now `Core(Box<ash::Device>)` / `Khr(...)`.
- **Bug D-S11-02:** `Instance::create` was requesting Vulkan 1.1 hardcoded; `vkGetDeviceProcAddr` only returns pointers up to requested version. Fixed: request `loader_version.min(1.3)`.
- **Bug D-S11-03:** Missing feature chain in `VkDeviceCreateInfo`; fixed by adding `VkPhysicalDeviceSynchronization2Features` to `pNext` chain when enabling sync2.
- Test count at end: 258 lib + 6 dump-capabilities + 26 layering + 7 portability = 297.

**Outstanding (not yet done):** GPU timestamp query implementation (Niobe D-N4/D-N5 spec → Switch `vk/cmd.rs`). `dispatch_integration.rs` has 4 `undocumented_unsafe_blocks` + rustfmt. `bind_aliased_output` seam contract with Mouse. OQ-15 `vkCmdDispatchIndirect` evaluation.

---

## Cross-agent context appended (2026-07-29T09:00:39-07:00) — first-hardware round

📌 **Local GPU facts (2026-07-29):** Vulkan SDK at `C:\VulkanSDK\1.4.350.0` — NOT on default PATH; prefix it explicitly in every shell command that calls glslc or epctl. Two devices pass §7.2 gate: **Intel Iris Xe Graphics** (Vulkan 1.4.309, UMA, 32 KiB shared, `maxSubgroupSize=32`) and **NVIDIA GeForce RTX 4060 Laptop GPU** (Vulkan 1.4.325, discrete, 48 KiB shared, `maxSubgroupSize=32`). 168 shader variants compile cleanly with SDK on PATH.

📌 **Intel Iris Xe = spec-conformance oracle (2026-07-29, Morpheus D25 + Link):** Intel's implementation is stricter than NVIDIA's on undefined behaviour and extension interactions. When the two disagree, assume Intel is correct. Use Intel results when filing bug reports or proposing spec questions.

📌 **R5 changed — subgroup BASIC in compute is now a probed capability, not a gate criterion (2026-07-29, Switch D-S10-01):** Removed from `passes_gate`; stored in `Capabilities::subgroup_basic_in_compute`. Mesa llvmpipe reports `supportedStages = 0`. Shader variants that require subgroup ops must check this capability before claiming the node, not at device-enumeration time.

📌 **`rustfmt --edition 2021` silently no-ops on this edition-2024 crate (2026-07-29, Tank D-T12):** Always use `cargo fmt --all` — the xtask `cargo ci` command does this correctly. Never invoke rustfmt directly with an edition flag.

📌 **GPU timestamp query spec (2026-07-29, Niobe D-N4/D-N5):** `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` activates GPU spans. Niobe owns `trace.rs`; Switch must implement the `vkCmdWriteTimestamp` call sites, query pool creation/reset, `timestampPeriod` scaling, non-stalling `vkGetQueryPoolResults`, and `VK_EXT_calibrated_timestamps` path — full spec in `docs/PERF.md §3`. Return type is `GpuTimestampReport { calibration, queue_family, intervals }` with raw ticks; Niobe's `trace.rs` owns all arithmetic.

📌 **`dispatch_integration.rs` has 4 flagged `undocumented_unsafe_blocks` + rustfmt issue (2026-07-29, Niobe):** Switch must fix these before the next `cargo ci` pass.

---


<!-- SUMMARIZED by Scribe 2026-08-02T02-03-46-07-00 -- Sessions 12-43 full text removed as redundant: one-line-per-session bullets already exist above in the [SUMMARY] Compressed entries block (2026-08-01T20:39:12-07:00); full text lives in git history. Sessions 44 onward (most recent, highest-value work) kept verbatim below. -->

## Session 44 — 2026-08-01 — the load guard, built on same-ordinal RSD

Coordinator merged session 43 as `c6cc0f3` and reproduced `probe_frames.py` independently. New
task: build the long-pending "refuse benchmarks on a contended machine" guard, using the
same-ordinal RSD I stumbled onto while reconciling the frames.

Landed: `bench/run_disturbance.py`, `ci/check_run_disturbance.py`,
`bench/test_run_disturbance.py` (14 tests), `bench/results/run_disturbance_dev0.json`.

### The threshold, and where it does NOT separate

Bimodal across all 28 dev0 traces with an EMPTY gap: 19 traces 0.624%..10.507%, then nothing, then
9 traces 35.313%..137.352%. `DISTURBANCE_RSD_MAX = 0.20` sits in the gap (1.90x above the highest
clean, 1.77x below the lowest dirty). A test pins that both arms hold for any threshold in
0.11..0.35, so no verdict rests on the constant.

**But it does NOT separate the STEADY/refused populations** — publishing 0.624%..10.507%, refused
3.694%..137.352%, substantial overlap. It predicts *stationarity*, not whether the tail publishes.
Said so in the docstring and the record rather than letting the bimodality imply more than it does.

### It adds ZERO refusals today, and I measured that rather than assuming it

`--corroborate`: 9 flagged, 9 already refused by the tail's floors, **0 that the tail would
publish**. So it corroborates, it does not protect — today. Value is elsewhere: it refuses
independently of suffix selection; it is the check that survives if a MARGINAL_TAIL's withheld
median is ever published; and two statistics over two frames agreeing on nine runs is evidence.

The coordinator's catch, worse than he put it: `contended` is the dirtiest run in the census
(137.35%) and its **tail RSD is 0.1067%, third tightest of all 28, tighter than
`baseline_certified` at 0.1163%**. This statistic is 6.9x over its bar on that run. R9 amendment 5,
third independent appearance.

It also refuses **my own** `packed`/`packed2` A/B traces at 53.4%/53.2%. Correct outcome.

### The frame — the one I was asked for does not work, and that is pinned

Asked for an in-band measurement *of the run being certified*. Tried the tail's own suffix first
because a companion ought to cover the same inferences. **No discriminating power there:**
`contended` restricted to its suffix reads **3.07%**, inside the publishing range (0.62%..3.97%).
The tail's selection has already found a quiet window. So the guard measures the WHOLE run and its
claim is scoped: *a statement about the run, not the suffix cut from it.* That is exactly its value
— it can say the suffix was carved out of a violent run, which nothing computed inside the suffix
can say. Negative result pinned by `test_the_tail_suffix_frame_would_NOT_work_...`.

### The hole is real and is now a passing test, not a caveat

A uniformly slow run PASSES. `synthetic_uniform_slowdown()` builds it and
`test_a_uniformly_slow_run_PASSES_and_that_is_the_documented_hole` asserts the pass (plus that the
run really is 2x slower, so the pass is not empty-input). **PASS means "repetitions agreed", never
"the machine was quiet"** — the CLI prints that on every pass.

### Obligation 8: complements, does not subsume

Orthogonal halves, and the hole is why. Obligation 8 catches a wrong *level* (board pinned at
210 MHz, uniform inflation) — exactly what this is blind to. This catches a non-stationary *run* at
whatever clock. We have observed a run perfectly stationary at 21.4x wrong with better RSD than the
correct one. Coverage floor is a third question again (how representative the suffix is). Three
instruments, three failure modes; no two collapse.

### Drift vs jitter — the reason this is not just whole-series RSD

`baseline` (whole 10.36%, ord 10.51%) vs `ab_p0_r2` (whole 10.97%, ord 3.82%): same drift, 2.75x
different jitter. Per-inference spread conflates them; same-ordinal RSD isolates jitter.

### Next step for a fresh session

1. Guard is landed and self-contained; no GPU needed to re-run it.
2. Consider wiring `check_run_disturbance` into whatever lane first publishes a duration — it is
   built for it (`--scan`, exit 0/1/4) but is not wired to any lane, deliberately: no lane
   publishes a timing figure today. Same rationale as `check_device_state` living in `ci/`.
3. Still open and GPU-bound: the certified NVIDIA A/B for packed loads (3 attempts, never
   obtained — tenancy is necessary but not sufficient); `gpu_steady_tail` under foreign GPU work.
4. The packed-loads claim stands in counts (`7a1d12f`) and needs no clock.

---

## Session 44b — 2026-08-01 — clippy green; index fix HELD pending Tank's MIXED

### DO NOT LAND THE DEVICE-INDEX RESOLUTION FIX YET

Coordinator-enforced sequencing from Tank. Read this before touching
`vk/host_device_memory.rs::OwnedDevice::create`.

**The defect (fourth face of the index space, mine):** `create(device_index)` takes the factory
index and then *ignores* it — line ~274 resolves via `crate::vk::instance::select_device(&capables)`,
which reads the process-global `ONNXRUNTIME_EP_VULKAN_DEVICE` and therefore returns the SAME index
for every key. So `ensure_registered(0)` and `ensure_registered(1)` stand up the same physical
device, and the `PROVIDERS` HashMap key is inert. Verified by reading, and by Tank's
`bench/results/two_device_frame_probe.txt` (both indices -> SPLIT-DEVICE on the RTX 4060).

**Why the order matters, and it is not arbitrary:** today two providers collapse onto one physical
device, so a genuine two-*device* population cannot form and the single frame label is wrong in
principle but not in fact. The moment the index actually selects, providers become genuinely
distinct and every `alloc_device_*` aggregate in a two-provider run describes a population drawn
from two `VkDevice`s while carrying one frame label chosen by registration order. That is an R12
violation shipped *by a correctness fix*. Tank's `MIXED` label must land before or with it.

**My recommendation, given to the coordinator: land together, not serialised.** Reason is R10.
Until my fix lands, `MIXED` cannot be produced end-to-end on this box — it would land with a
positive arm that is unfalsifiable in fact (synthetic unit construction only). Landing together
lets me supply the real artifact: a two-provider run on two physical devices producing `MIXED`,
with the single-provider control producing `SHARED`/`SPLIT-DEVICE` and not `MIXED`.

**Finding for Tank, from reading the only production caller:** `ensure_registered` is called from
exactly one place, `allocator.rs:595 try_attach_device_buffer`, keyed on that allocator's own
`device_index`. So two providers only register if one process stands up allocators for two
different device indices. But my §6.5 fix made `devices_to_advertise()` advertise **only** the
pinned device when `ONNXRUNTIME_EP_VULKAN_DEVICE` is set — so:

  - **pinned**: one device advertised, one index reachable, `MIXED` is correctly UNOBSERVABLE.
  - **unpinned**: all capable devices advertised, ORT may create sessions on both -> `MIXED`
    reachable, but only after my fix makes the providers genuinely distinct.

So `MIXED`'s falsifier needs an **unpinned two-session run**. Worth Tank knowing before he writes
the control, because a pinned probe can never produce it and would read as a silent pass.

### clippy, red for the life of the project, now green (`9b2a916`)

`cargo clippy --release --all-targets -- -D warnings` failed on four items, all test-profile, all
mine. Invisible because `cargo build --release` is clean — a lint gate whose failure cannot be seen
from the step beside it.

- `transfer.rs:1103` unused `DeviceMemoryProvider` import (genuinely dead — inherent methods used).
- `quant.rs:888` manual range contains -> `(1..=GEMV_MAX_COLS).contains(&cols)`, and it was
  asserting bare, so it got the failure message it never had.
- `norm.rs:270`, `indexing.rs:118` function item cast to usize -> `std::ptr::fn_addr_eq` against
  the row's own `TranslateHandler` type, which is what the comparison always meant.

**Both fn_addr_eq tests gained the negative polarity they never had**: an address comparison that
can only return true is not a check, so each now also asserts the row is NOT
`templates::unimplemented`. Both arms pass, so the comparison discriminates.

440 lib tests pass at **default parallelism** — not `--test-threads=1`. Tank's reasoning applies to
me too: serialising would have hidden a shipping defect with the identical symptom and the opposite
fix. I had used `--test-threads=1` to characterise the counter flake in session 42; that was a
diagnostic, and it should not become how the suite is run.

---

## Session 44c — 2026-08-01 — the PROVIDERS key now selects a device (`1d2a663`)

Sequencing block lifted: Tank's `MIXED` landed on main at `ca283a9`. Merged (one conflict in
`transfer.rs` — he added a function-local `DeviceMemoryProvider as _` where I had removed the
file-level import; took his side, then clippy caught that one of his two local imports was itself
unused because that test calls inherent methods on the concrete `RecordingProvider`).

### The fix

`PROVIDERS` is a `HashMap` keyed by the factory's advertised device index; `OwnedDevice::create`
resolved through `select_device` and never read its argument. Every key produced the same device.

**The previous repair was not careless and it is worth understanding why.** Indexing the best-first
sorted list *with* a physical enumerate index had put the mirror on the wrong GPU (Tank, 07-30), and
ignoring the index fixed that. But *"don't index with it"* and *"don't read it"* are different
instructions, and only the first was needed. The two spaces meet at `position_of_physical` — the
same seam `VulkanSession::create` already uses for ORT's binding — so translate, don't ignore.

Split out `resolve_provider_position` with no Vulkan in it, returning
`Translated(pos)` / `Untranslatable(pos)` rather than a bare `usize`. **The two-variant return is
the point**: the fallback position is the same number the old code always produced, so collapsing
them leaves a caller unable to distinguish a working map from an inert one — which is the state we
were in for a week.

### Scope — state this precisely, it is narrower than it sounds

`create` is reached only on `ensure_registered`'s `NoOffer`/`Ambiguous` arms. When a session has
offered a device, the §6.5 `SoleDevice` identity rule adopts it and `create` never runs. **So this
does NOT change `alloc_device_frame` on the ordinary pinned production path**, which was already
`SHARED` by identity. What it changes is the fallback path — which is exactly where Tank's probe
landed (no session ⇒ `NoOffer` on both indices).

### Falsifiers — three of four fail on the old code, and I checked rather than assumed

Restored the old semantics temporarily and re-ran:
`distinct_provider_keys_resolve_to_distinct_devices`, `a_key_that_names_no_capable_device_...` and
`agreement_between_the_two_spaces_is_permitted_but_never_relied_on` **FAIL**.

`a_pinned_offer_translates_onto_the_selectors_own_position_on_both_selectors` **PASSES on the old
code** — because in the pinned case the right answer *is* `selected`, which is what the old code
always returned. It is kept (§6.5 depends on the invariant) but its doc comment now says in as many
words that it is **not evidence for this fix**. A test whose polarity I have not checked is a
printed opinion, and this is the second time this week I have caught one of mine.

### The artifact — `bench/results/provider_key_selects_probe.txt`

`probe_provider_key_resolution` enumerates capable devices and prints the table. It creates no
logical device and dispatches nothing, so it is admissible on a contended box — counts and
identities, not timings.

```
best-first position 0 -> physical enumerate index 1 -> 'NVIDIA GeForce RTX 4060'  <- select_device
best-first position 1 -> physical enumerate index 0 -> 'Intel(R) Iris(R) Xe Graphics'
key 0: was NVIDIA (old rule) -> now Intel   [Translated(1)]
key 1: was NVIDIA (old rule) -> now NVIDIA  [Translated(0)]
2 key(s) -> 2 distinct device(s).  Old rule: 2 key(s) -> 1 device.
```

**The two index spaces are inverted on this desk in fact, not hypothetically** — the RTX is physical
1 / position 0. That is why every constructed test above uses `[1, 0]`.

And Tank's detector fires. His probe read `SPLIT-DEVICE`/`SPLIT-DEVICE` with *"this box cannot
exhibit the mixed frame"*; on the same build it now reads:

```
device_index 0: frame now "SPLIT-DEVICE", 1 declared
device_index 1: frame now "MIXED", 2 declared
FINDING: production declared 2 frames in one process.
```

**The sequencing constraint was right and this is the evidence.** `MIXED` landed first; this change
made it observable in production code within the hour rather than by unit construction. Had the
order been reversed, a two-provider run would have described a two-`VkDevice` population under a
single frame label — an R12 violation shipped *by a correctness fix*.

503 passed / 0 failed. `cargo clippy --release --all-targets -- -D warnings` green.

### Also: `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` default re-justified (record filed)

Morpheus ruled the default intended but not re-decided. The recorded reason ("the handle→VkBuffer
seam is not filled") **has expired** — the seam is filled, 6 binds measured on both selectors.

The real blocker is one level up and the docs conflate them: `vk::session` still binds buffers it
allocated itself and reads through `host_backing_for`, so the device buffer stays a mirror.
`alloc_device_authoritative_spans` reads **0**, and that zero is *measured* — `allocator.rs:681`
evaluates every device-backed span at free against `buffer.is_some() && staging.is_none()`, and
counts evaluations separately from outcomes, so a measured zero is distinguishable from an unwired
one.

Recommendation: **keep it off, with the reason replaced.** New reason: turning it on pays full cost
(device allocation + a bus write per copy-in) for a benefit no counter can currently report.
Second, independent reason: the leaked production device makes the shutdown-validation gate
`UNOBSERVABLE`, and device-backed allocation is precisely the path that gate would police — enabling
it by default while its safety net cannot fire is the wrong order. The two reasons should not be
collapsed; if the residency work lands, the second still stands.

Filed to main's inbox: `switch-device-memory-default-rejustified.md`. Needs Tank's half (counter
surface readiness off this desk) before M2 entry.

### Next step for whoever resumes

1. **The named change that flips the default**: make `vk::session` bind `device_buffer_for`'s buffer
   instead of allocating and re-uploading its own, then run with the flag on and read
   `alloc_device_authoritative_spans`. **Non-zero is the entire argument.** Engine-side, mine.
2. Still GPU-bound and waiting on a quiet window: the certified NVIDIA packed-loads A/B (attempted
   3x, never obtained — `SOLE_TENANT` twice yet still `MARGINAL_TAIL`), and *does foreign GPU work
   move `gpu_steady_tail()`?*. Do not quote the uncertified 11.5673 ms archive.
3. The packed-loads claim stands **in counts** (`7a1d12f`: 465,297,408 -> 116,324,352 InB load
   instructions) and needs no clock, so it is quotable as-is.

---

## Session 44d — 2026-08-01 — the disturbance guard was not independent, and I was the one who said it was (`4ab6813`)

### The correction

`run_disturbance.py` claimed same-ordinal RSD and whole-series per-inference RSD were *"two
different statistics over two different frames agreeing on the same nine runs, **neither derived
from the other**."* **False.** Niobe computed the correlation over the census; I recomputed it at my
own commit rather than quote her (R13):

| | Spearman | log-log r | median ratio |
|---|---|---|---|
| mine, 29 dev0 traces | **0.919** | **0.970** | 1.170 |
| Niobe, 28 | 0.903 | 0.964 | 1.128 |

Confirmed. And **the false sentence was mine, not the coordinator's** — I had the agreement in front
of me (the same nine runs) and read it as independence. Agreement between two statistics is evidence
*for* redundancy. The coordinator offered to take the blame; it isn't his.

Mechanism (hers), now **measured** by `localise()`: dividing out per-inference level explains
**84–87%** of the same-ordinal dispersion on disturbed traces — `contended` 137.35% → 18.70%,
`ab_p0_r1` 109.23% → 17.39%, `notile` 55.84% → 7.25%. A disturbance that scales a submission moves
every dispatch inside it together.

### The rescope — localisation is what survives

Whole-series RSD says *that* the run moved. `localise()` says *what* moved. Two traces whole-series
ranks as neighbours are different conditions:

```
ab_p1_long   ordinal 35.31% -> norm  3.90%  ( 88.9% explained)  SUBMISSION_LEVEL
contended3   ordinal 41.11% -> norm 50.43%  (-22.7%)            PER_DISPATCH
```

Submission-level ⇒ clock/power excursion or queueing ahead of the submit. Per-dispatch ⇒ foreign
work interleaved *between* dispatches. Nothing else in the project separates those.

**And I checked the new quantity before claiming it** — the discipline I failed at last time. The
level-normalised statistic still correlates with whole-series at Spearman **0.710**. So it is a
**decomposition**, not a second opinion, and the docstring, the JSON and the CLI all say so.

Two by-products: my `notile` — the "70% spread" — is **87.0% SUBMISSION_LEVEL**, independent support
for session 43's `SAME_FRAME_ORDERED_SELECTION`; and my own `packed` A/B trace is the only badly
disturbed trace classified `MIXED` (44.2%), i.e. it carries real per-dispatch disagreement.

### The threshold's headline is falsified by its own sensitivity table

I claimed an **empty gap** 10.507% → 35.313%. **One further trace populated it**: `switch_resid`
reads **20.787%**, inside the gap, **1.04×** over the bar (was 1.77×). The flat band narrowed from
11%–35% to **11%–20%**. On **Intel it is 20%–20% — a single point**, so there the threshold does
*all* the deciding.

Left at 20% and published as a table (Niobe's discipline: she caught her own instinct to move a
threshold until the flagged count reached zero and called it *choosing the answer*). Moving it to
restore the old band would be exactly that. **An "empty gap" is a statement about the traces you
happen to have and it decays as you collect more** — `--sensitivity` recomputes it; do not quote
this paragraph.

### The result worth carrying beyond this module

**No statistic computed from inside a series can detect a bias that scales the whole series.**
Dispersion is invariant under multiplying every sample by a constant, so a uniformly wrong run is
indistinguishable from a uniformly right one.

`baseline_certified`: cleanest whole-series (0.118%), cleanest same-ordinal (0.624%), unchanged by
normalisation (0.632%), `STEADY` at n=46 / 100% coverage / zero discarded — **and 21.4× wrong**.
Every dispersion measure we own certifies it. Only `check_device_state.py` refuses it, **because its
evidence comes from outside the series**. A dispersion guard cannot replace obligation 8 and no
future one can either. That is structural, not a gap awaiting a better statistic.

### And one of my own new tests passed for free

`test_localise_inherits_the_level_blindness_hole` compared two normalised values that are **both
exactly 0.0** — an assertion that can only succeed. Added the ground-truth arm asserting the two
inputs really differ by 2×. Same failure mode as the `fn_addr_eq` tests I fixed this morning; twice
in one day, so it is a habit to watch rather than an incident.

### Worktree

Checked before staging, per the warning that worktrees are allocated per agent *name*: tree was
clean, `git worktree list` shows `ep-vulkan-switch` on `squad/switch` alone, nothing foreign swept
up by `git add -A`.

### Next

- Localisation is built but **not wired to any lane**, deliberately — no lane publishes a duration.
- Open question for the quiet window, unchanged: does foreign GPU work move `gpu_steady_tail()`?
  Note `contended3` is `PER_DISPATCH`, which is the signature foreign GPU work should produce, so
  that trace may already be a partial answer — worth checking against what was running.
- Still owed and GPU-bound: the certified NVIDIA packed-loads A/B.

---

## Session 44e — 2026-08-01 — a screen for assertions that pass without reading their subject (`49a4a47`)

### Standing items first: all three were already landed

The relay listed three open items. Verified rather than asserted, at `8505d43`:

1. **Index fix** — landed `1d2a663`. Artifact `bench/results/provider_key_selects_probe.txt`;
   Tank's detector goes `SPLIT-DEVICE`/`SPLIT-DEVICE` → `SPLIT-DEVICE`/**`MIXED`, 2 declared**.
2. **Clippy** — landed `9b2a916`. Re-ran `cargo clippy --release --all-targets -- -D warnings`
   at this commit: **green**.
3. **Device-memory default** — re-justified, record filed. Tank's half (counter-surface
   readiness) is still off this desk; joint by M2 entry.

`origin/main` is still `ca283a9` and already merged. Worktree checked: `squad/switch` alone,
nothing foreign staged.

### The mechanical screen — and it found nothing, which is the honest headline

Built `ci/check_tautological_assertions.py`. Census: **1,056 comparison assertions
(rs=614, py=442) across 139 files, 0 detections.**

**Neither of the two assertions that prompted it is detectable by it**, and that is the first
paragraph of the docstring rather than a footnote:

- `test_localise_inherits_the_level_blindness_hole` compared two *different expressions* that
  both evaluated to exactly `0.0`. A runtime property.
- The `fn_addr_eq` tests had no negative polarity. A property of the *set*, not of a line.

So the covered class is strictly and substantially smaller than "an assertion that cannot
fail", and every output path — `PASS`, `FAIL` and `ERROR` — prints what it does not cover.
Scoped to **regression**, not discovery. All of its evidence that it is a screen is
**planted** (10 tests in `ci/test_lane_checks.py`), because a check observed only to pass is
not known to be a check.

### Its own development produced three instances of the failure it hunts

This is the part worth keeping.

1. **It reported `PASS` over a language it had not read.** A leading-whitespace bug in
   `PY_ASSERT` meant 89 Python files yielded **zero** assertions; the total was non-zero only
   because Rust carried it. **A total another language paid for is not coverage.** Coverage is
   now asserted per language → `ERROR(instrument=language_scanned_nothing)`.
2. **Blanking string literals invented three false positives.** `frame["a"] == frame["b"]`
   blanks to a term compared to itself. Three of the first four detections were this shape and
   **all three were correct code.** Replaced with a length- and newline-preserving digest
   placeholder: distinct strings stay distinct, while an assertion written *inside* a string
   stays hidden (`layering.rs` has one as lint fixture text). The two requirements pull
   opposite ways; both have a test.
3. **Polarity was missing**, so `assert empty.median != empty.median` — the NaN idiom in
   `bench/test_harness.py` — was reported as a defect. Forced the definition to sharpen: **the
   hazard is *passing without reading the subject*, not sameness.** Identical operands are
   reported under equality only (under inequality they either always fail — safe, a
   permanently red assertion is fixed on its first run — or are a NaN probe). Two literals are
   reported at either polarity, since `assert_ne!(1, 2)` also passes touching nothing.

**Four confident detections, four wrong. An unscoped screen does not merely miss things — it
asserts things.** Same shape as D-T85, arrived at independently and within an hour of reading
it.

### Wiring

Added to the existing no-GPU `Lane-check self-test` job, after the two-polarity pytest step.
`PASS / FAIL(condition) / ERROR(instrument)`, exits 0/1/4, matching `check_device_state.py`.
51 lane-check tests pass (10 new). YAML validated.

### What this does *not* close

The class that actually bit me twice remains undetected and I do not currently see a cheap
mechanical form for it. A runtime screen would need to know that an assertion's reading cannot
move when its subject does — that is mutation testing, and it is not cheap. **Recorded as open
rather than closed by a screen that does not reach it.**

📌 Team update (2026-08-02T02:03:46-07:00): Link introduced a PLANTED/OBSERVED axis for CI check classification and applied it to your hostfree.tautological_assertions screen: it is registered PLANTED/UNDEMONSTRATED, not field-proven DEMONSTRATED — 1,056 assertions scanned with 0 detections is a regression barrier, not evidence the barrier works. Registering it correctly demoted the lane-checks lane from green to operational. Link applied the identical standard to his own same-day tick screen and classified it PLANTED too, rather than exempting himself. — decided by Scribe

📌 Team update (2026-08-02T02:03:46-07:00): Mouse checked rather than accepted the coordinator's claim that clippy's manual_contains was "the fourth union defect today," and found cargo clippy -D warnings produced five errors of which four were already present on origin/main verbatim, independent of any merge — only 
egistry.rs:2261 was a genuine union defect. This clears your "green at my commit" report: the clippy-red state at the time was mostly pre-existing, not introduced by your commit. The coordinator's own over-attribution is recorded in this session's log and ruled on by Morpheus as R13's second clause applied to a classifier (a newly named pattern attracting cases that don't belong to it), not a new rule. — decided by Scribe

---

## Session 45 — 2026-08-02 — criterion 10's oracle covered one output out of sixty-five (`8c43918`)

### The cheapest experiment came back negative, which is the load-bearing result

Morpheus asked for the plant *before* anything was built, and he was right to: a positive
result would have re-closed the row on existing evidence. I planted a **stable all-zero KV
write** — logits untouched, outputs 1..64 zero, byte-identical across three runs — and ran it
through every gate criterion 10 applied:

```
cross_run_identity          green
cpu_oracle_comparison       green
series_verdict              green     <- MATCH, with 64/65 outputs zero
phi35_guard1_logit_range    green
```

**Nothing goes red.** `bench/results/planted_kv_probe.json`; regenerate with
`bench/results/probe_planted_kv.py`.

Two decisions about that probe worth keeping:

- **It runs against the real gate functions, imported not copied.** A copy would let the
  gates drift away from their own falsifier.
- **The Phi-3.5 artifact is not on this machine** (`~/.foundry/cache/models/Microsoft/...`
  does not exist), so both `test_criterion10` and `test_phi35` skip before their first
  assertion. A GPU plant was not available at any price. But the question — *does any gate
  read KV* — is a question about the gate code, and the harness frame answers it for every
  device rather than for whichever one was free.
- Attribution is **held identical across both arms** so it cannot be the discriminator. Its
  trace is fabricated and the probe says so out loud rather than letting the artifact imply
  a real run.

### The arm

`m.compare_all_outputs_to_cpu` — three-way, because two-way is what broke:

| outcome | meaning |
|---|---|
| `AGREE` | all outputs compared, all within tolerance, all informative on **both** sides |
| `DISAGREE` | some output outside tolerance, or arity/shape/dtype mismatch |
| `NOT_PERFORMED` | some pair degenerate — **absence of evidence, not agreement** |

`NOT_PERFORMED` is the whole point. **64 pairs of zeros satisfy an all-output allclose
perfectly.** Morpheus called it *`0.0 == 0.0` in a fourth costume* and he is right — it is the
same shape I caught in my own tests twice last week. So absence of evidence gets its own
token instead of borrowing the passing one, the guard runs on **both** sides (a degenerate
*oracle* is just as vacuous), and it is written on **constancy rather than on zero** so a
buffer holding one repeated residue value is caught too.

**Tolerances are not chosen for this gate.** fp16 KV outputs get `MATMULNBITS_FP16` —
already derived from measured data in `_models.py`'s header — because MatMulNBits is the
arithmetic that produces them. Each tolerance carries its justification string into the
per-output record. Reusing a justified number is the justification; picking one that makes
this gate green would not be.

### Two counts, two names

`oracle_outputs_compared` and `cross_run_outputs_compared` are separate keys and the bare
`outputs_compared` is **asserted absent**. That key counted cross-run comparisons, sat among
the oracle facts, and was read as sixty-five oracle comparisons — with a `max_abs_diff` over
one tensor quoted beside it. `max_abs_diff` → `logits_max_abs_diff` for the same reason.
**Make the misreading impossible, not merely corrected.**

### And the mutation caught a tautology in my own test

13 falsifiers, deliberately needing **no device and no model** — a falsifier that requires an
absent artifact skips green on exactly the machines where nobody is watching.

I verified them by mutation rather than by watching them pass:

- neutering `_is_degenerate` → **4 red**
- widening `KV_CACHE_FP16` to 1e9 → the numeric arm red

The second mutation caught **my own assertion that could not fail**.
`test_every_tolerance_carries_its_justification` parametrised `expected` with
`m.KV_CACHE_FP16` and then asserted `tol == expected` — the constant compared to itself. It
now asserts against `MATMULNBITS_FP16`, the thing the tolerance *claims to derive from*, so
picking a number for this gate fails it.

**Third instance of that shape in three days**, and it is precisely the form
`ci/check_tautological_assertions.py` does not reach — the two sides are different text. The
screen I shipped yesterday passed this file cleanly. **The screen's stated hole is real and I
have now walked into it myself.** Mutation caught what the static screen structurally cannot;
that is the honest division of labour between them and it should be written down as such.

There is also a **ground-truth arm** — the old one-output comparison is asserted to `AGREE`
with the same plant. Without it the new refusal proves nothing about the gap.

### Not mine to claim

`(d)` attribution is untouched: still re-emitted per run from `end_profiling()` via
`ExecutionAttribution.from_profile`, which is private by R10 amendment 1. I did not re-argue
it and did not need to.

### Open

- **The arm has never run on the real artifact**, because the model is absent here. Its
  reading is pinned by construction and by mutation; its *value* on Phi-3.5 is unmeasured.
  Whoever has the model should run `test_criterion_10_three_consecutive_attributed_match` and
  read `oracle_outputs_degenerate` first — if the KV outputs come back degenerate on real
  data, that is the reopened defect still live, not a harness problem.
- No timing measurement taken. Nothing with a time term is certifiable on this box.

## Session 46 — the roofline, and the first kernel change of the performance push

**Relay:** priority change from the user — stop obsessing over errors, get performance up,
do it right the first time. Verify or destroy a roofline estimate first; then bytes required
vs bytes moved; then intermediate traffic; then name the gap owner. Explicitly: no more
instruments, screens or guards.

**The estimate survives, its arithmetic did not, and one of its premises was wrong in my favour.**

Derived the weight stream from the graph rather than from recollection: 1775.0 MiB int4 +
221.9 MiB fp16 scales = **1996.8 MiB**, confirming 1997.6 MiB independently. Scales are
**11.1% of the stream** — not previously stated separately, and worth knowing before anyone
proposes a finer block size.

Three corrections:
1. The floor is **8.18 ms, not 7.8** — 1997.6 MiB is 2.095 GB, not "2.0 GB".
2. **12.1847 ms is not withdrawn.** I checked the artifact instead of accepting the premise.
   `phi35-certified-dev0.json` says `certification.quotable = true`, STEADY at n=41, and it
   reproduces at 12.1869 ms in a second artifact. R13 withdrew the wall-clock *speedup ratios*,
   which were taken during CPU fallback. This is GPU-busy on the device counter. Different
   instrument, different quantity, own companion attached.
3. Therefore the ratio **is** establishable: 171.8 GB/s, **67.1% of roofline, 1.49× headroom**.
   The estimate's conclusion — micro-optimisation is not the work, architecture is — stands.

**The error I made and caught.** Charging activation re-reads to DRAM gives 248.2 GB/s, **97.0%
of spec peak**. No GDDR6 controller reaches that. The right reading of an impossibly good number
is that the model is wrong, not that the kernel is excellent — and `q_gemv.comp`'s own header
had said so all along ("the bytes hit L1 but the instructions still issue"). The activation row
is 6 KiB; its re-reads are cache hits. The probe now reports two counts because they are two
quantities: weight bytes are DRAM, activation bytes are load issue.

That distinction is what names the gap owner, which was the task. **DRAM sits at 67% while the
kernel runs — a third of the bandwidth is idle, so bandwidth is not the limit.** Not coalescing
either, on the same evidence. Of the candidates that are counts rather than guesses, activation
load issue is the largest.

**The change: `QB_MAX_COLS` 8 → 16, `QB_RED_WORDS` 1024 → 2048.** 887.5 → 443.7 MiB of
activation load bytes, 15.4% of all bytes named by loads, removed. Barriers per output halve too.

**Why not 32**, which the same model says removes another 221.8 MiB: `RED_WORDS` must cover
`local_size_x * QB_COLS`, so 32 columns at the 128-invocation workgroup `K=8192` takes needs
16 KiB of shared memory — *the whole of the §7.2 guaranteed floor*. One resident workgroup on a
device that only meets the floor trades away the latency hiding a bandwidth-bound kernel lives
on. Register pressure argues the same way, but that is an estimate about an unseen driver
allocator whereas the shared-memory floor is a number the specification promises. **The decision
rests on the promise, not the estimate** — worth keeping as a general rule for tuning constants.

**Prediction recorded before measurement**, per the standing rule: up to ~1.15× if load issue
owns the gap, nothing if it owns none. Not runnable today — the EP claims 0/363 nodes pending
the proof ledger, so there is no live inference to time.

**Baseline established rather than assumed.** `test_matmulnbits.py` is 23F/8P/1x with my change.
Rather than reason that those were the known proof-ledger reds, I stashed the change, rebuilt,
and re-ran: **identical, 23F/8P/1x**. They are `assert_vulkan_claims` failures, not numerics.
528 Rust tests pass; clippy `--all-targets -D warnings` green.

**What is not verified, said plainly:** the numeric result on the device, for the same reason.
The tile width does not change the summation order within a column, so the output should be
bit-identical — but that is an argument, and the evidence is `test_matmulnbits.py` the moment
the ledger lands.

Commit `d6628b3`.

## Session 46b — the multiplication, and the term nobody had costed

**Relay:** multiply the InB load count by the load width. If it lands near 2.09 GB each weight
byte is read once and the 67% is real; if it lands at 2× or 4×, re-reading is the defect and
blocking is the fix.

**It lands exactly.** 116,324,352 loads × 16 B = 1,861,189,632 B = 1775.0 MiB = the int4 weight
total from the graph. **Amplification 1.000000.** Blocking is not the defect. Branch one.

**I checked whether my own check could fail**, because it came within one factor of being a
tautology: `blobs × blob_bytes = weight_bytes` *is* an identity. Two factors are not, and both
were measured — **loads per blob = 1** (the def-use walk finds one `%v4uint` where the unpacked
path issues four `%uint`; a per-element loader would be 8× higher over the same blobs), and
**each blob is touched by exactly one workgroup** (`col0 = WorkGroupID.x * QB_COLS` partitions
columns; the tail-tile redirect would break it and is unreachable because all five Phi-3.5 `N`
values divide by 16). Remove either and the product stops matching.

**The fusion prize is 0.47%, and I had the cost model wrong in the same way twice.** Intermediates
across all 366 nodes at batch 1: **9.52 MiB against 1996.8 MiB of weights**. 354 dispatch
boundaries *sounds* like traffic, but at batch 1 every intermediate is a **vector** — 6 KiB at
hidden 3072, 16 KiB at FFN 8192 — so the boundary count multiplies something negligible. Same
error shape as charging activation re-reads to DRAM last session: **a large count of small things
is not a large thing, and I keep reaching for the count.**

**The term nobody had costed: the KV cache.** `past_key_values.N.key` is `[batch, 32, past_seq,
96]` and each of the 32 `GroupQueryAttention` nodes reads its whole history every token.

| past_len | weights | KV | inter | total | KV% | inter% | floor |
|---|---|---|---|---|---|---|---|
| 0 | 1996.8 | 0.0 | 9.52 | 2006.4 | 0.0% | 0.474% | 8.22 ms |
| 2048 | 1996.8 | 768.0 | 9.52 | 2774.4 | 27.7% | 0.343% | 11.36 ms |
| 8192 | 1996.8 | 3072.0 | 9.52 | 5078.4 | 60.5% | 0.187% | 20.80 ms |

It is **zero in the regime 12.1847 ms was measured in**, which is exactly why it has been
invisible. It passes the entire fusion prize at 32 tokens of context. Unlike weights it is not
irreducible; unlike intermediates it is not small; **it is unbounded.** And this model does no
grouping — 32 KV heads for 32 query heads despite the op's name — so the cache is 4–8× a
genuinely grouped model. We implement the op (`attention.rs`), so the traffic is ours.

**Consequence for the roofline itself, and it is a methodological one: the floor is not a
constant.** 8.22 ms at zero context, 14.51 ms at 4096. Any figure quoted against "the roofline"
must say which context length it was taken at, the way a timing must carry its device state.

Nothing here needed a clock, a device state, or a working EP. Commit `eaa9aef`.

## Session 46c — the branch that was called unusual, and the control that nearly lied

**Task:** fix the `else` in `vk/session.rs` that left 323 claimed nodes executing zero times.

**Result: 0/459 -> 33/73.** Every retained island executes. First Vulkan execution of
Phi-3.5 with the ledger in place. Commit on `squad/switch`.

**The fix.** The patch loop served two token ranges and left the middle one -- island
outputs that are *also* consumed internally -- symbolic. That is every residual stream in
the model. The desc existed all along: the producer loop recorded the island output's byte
size and shape and then dropped the desc itself. I inserted it into `computed_descs` there
and keyed the consumer off "did an earlier kernel produce this token" instead of off the
token range. Nothing infers a desc from a sibling; a token with no producer still returns
`None` and the handler still refuses, so the handler keeps its ability to detect loss.

**The thing worth keeping is the control, because it nearly went the other way.**
Before merging `main` I ran the probe in my unmerged tree behind the CLAIM_UNPROVEN escape
hatch: **2/13 with the fix, and 2/13 without it.** I had the fix in hand and a measurement
saying it did nothing. Had I reported then, I would have reported a null result at full
confidence. It was null because only two islands formed under the escape hatch and neither
took the branch -- **a control that cannot distinguish the two binaries is not evidence
that they are the same.** The discriminating control needed the merged tree, and there it
was unambiguous at two binaries differing only in this hunk.

**Second witness, free and independent:** total node executions moved 459 -> 73. Without
the fix no island survives, so no fusion happens and ORT reports the unfused graph. The
node-count collapse and the EP-execution count are separate readings of one event.

**Process note.** `origin/main` was stale at `c144210`; local `main` was at `a9e6e1b` with
Mouse's work already merged. My "0/363 claimed" reading last session came from fetching the
stale remote. **Merging `origin/main` is not merging `main` on this box.**

**Withdrawn:** my note that `test_phi35.py:_MODEL_DIR` was missing the
`cuda-int4-rtn-block-32` segment. It is present; the path resolves; the model exists.
Criterion-10 tests do not skip. That finding was stale and I should have re-checked it
before carrying it forward a second time.

**Still open:** GQA is DIVERGENT (`worst_rel` 16.726) and it produces the KV outputs
criterion 10 reopened over, so the KV bandwidth work -- 60.5% of the byte floor at
past_len 8192 -- stays blocked behind that correctness defect. Correctness before bandwidth.

## Session 46d — the flagged layer was the clean one, and the gate moves with the prompt

Criterion 10 flagged `present.31.key/value` OUTSIDE_TOLERANCE, layers 0..30 WITHIN. The
coordinator offered an island-shape hypothesis (layer 31 as an island output consumed by
nothing, a third case for my `session.rs` fix) and asked me to attack it. Four measurements:

1. **All 32 GQA nodes are DECLINED.** `present.*` is CPU-computed in *both* sessions.
   There is no Vulkan GQA path in the comparison, so this is neither a GQA kernel defect
   nor the same defect as GQA's standalone 16.726. The island hypothesis is refuted too:
   layer 31 takes no path, because it is not claimed.
2. **Layer 31 is the third-cleanest of 32.** max_abs_diff 0.015625 at layer 31 == max over
   0..30. At each tensor's own fp16 scale: layer 31 = 1.000 ULP, rank 30/32; worst is
   layer 3 at 2.000 ULP, reported WITHIN.
3. **Mechanism, predicted then measured.** `atol=0.001` was justified against max_abs
   3.6e-3 and applied to tensors of max_abs 25.25, where one fp16 ULP is 0.0156 — it asks
   for 1/16 ULP, unsatisfiable at that magnitude. Elementwise `atol+rtol*|b|` budgets by
   the *element*; a reduction's error is set by the *tensor*. Prediction: failures are the
   small elements. Measured median |b| 0.0396 failing vs 0.5732 overall.
4. **Feed sensitivity: 2/65 to 60/65** on one binary and one tolerance, varying only
   `input_ids`. Union 60, intersection 1.

**The part I want to remember is the one I did not do.** The obvious move was to propose a
scale-set tolerance and call the flag spurious — a conclusion that happens to clear my own
subsystem. So I wrote the plants first (all-zero, head zeroed, row zeroed, sign flip, 1%
and 0.1% scale error, gaussian noise, two true negatives) and a margin bar of 10x, then
scored incumbent against proposal. **My proposal lost**: it misses "1 ULP added everywhere",
and the margin came in at 9.3x. Rejected on its own suite; threshold unchanged.

*"The error is only rounding" sounds exactly the same whether it is true or false.* The
only thing that separates the two is whether the replacement was made to fail first.

**Blind spot recorded:** a systematic 0.1% scale error on a KV output is invisible to the
incumbent *and* to my proposal. Pre-existing; neither introduced it.

**Method note for the next tolerance:** a per-output tolerance justified against a tensor
four orders of magnitude smaller is R11 in the tolerance. And since the flagged set moves
with the prompt, a count of diverging outputs must carry its feed the way a timing carries
its device state.

---

## Session 46e — 2026-08-02 — GQA: 16.726 -> 0.00073, and the root was the token space

**Task (relay):** fix `com.microsoft::GroupQueryAttention`, the only DIVERGENT form on
Phi-3.5, at `worst_rel = 16.72642029784887`. Correctness before bandwidth: no KV
optimisation until GQA is MATCH.

**Result.** MATCH at `worst_rel = 0.00072939`, 1 claimed node / 1 dispatch (non-vacuous).
All 32 GQA nodes claimed on Phi-3.5. Islands per run 33 -> **1, covering 355 of 363
nodes** — the fragmentation Niobe was measuring was GQA being declined, and it is gone.
Ledger 73 -> 74 entries, digest `372bcd276a7aa35c`. DLL `BD08DEBC949C2E32` ->
`654630DD599C7209`. Commit `a29025e`.

**Three defects, one root.**

*Root — two index spaces, the fourth on this project.* `ep.rs` built the name->token map
only when `plan.nodes.len() > 1`. Without it the recorder numbers inputs by the order the
translate handler calls `resolve`, which matches ORT's order only if the handler resolves
every plan input exactly once in slot order. GQA resolves 0,3,4,5,7,8 — never `total_seq`
(slot 6, a real plan input) — and slots 1/2 are empty optionals. So `cos_cache` was bound
to `total_seq`, `sin_cache` to `cos_cache`, and the patch loop read `past_key`'s desc off
`seqlens_k`. Map now built unconditionally; `plan.inputs` IS the `KernelContext_GetInput`
order, so a name lookup into it cannot drift from it. Absent optionals get `NO_TOKEN`.

*A — `rotary_dim` guessed.* Defaulted to `head_dim` when the attribute is absent. Absence
is not full width: `cos_cache` is `[max_seq, rotary_dim/2]` and that is the only record of
the true width. Evidence case: head_dim 32, cos_cache [64,8] -> true 16, defaulted 32.
Now derived from the cache; refuses when neither source has it.

*B — `present` assumed to be `past`.* Two ORT conventions, distinguished only by the
declared present extent: shared-buffer (present == past) and growing (present == past+S).
Both graphs we target are growing. Present was bound one token short, the write at
`tok_pos` fell outside and was dropped, and present KV read back **all zero**. Shader now
carries two strides (present_len at offset 24, past_stride at 28, scale at 32) and copies
past into present when they are not the same buffer.

**What actually broke the case open.** I diagnosed A and B correctly from source, fixed
both, and 16.726 -> 7.5. It *improved*, which is the most misleading outcome a partial fix
can have: I had two real defects and a mechanism for each, and I nearly reported a partial
win. What stopped me was building `probe_gqa_present_copy.py`, which compares the copied
region of `present` against **the fed input tensor** rather than against the CPU EP —
a value known without reference to any other implementation. It read a mismatch on data
that is a straight copy, which neither A nor B could explain. Filed as
`switch-a-default-is-an-unlabelled-branch.md`: *a partial fix that moves the number is
harder to stop at than one that does not; a remaining gap is an unnamed mechanism, not
evidence your fix was too small.*

**The `unwrap_or(0)` that hid it.** `past_len_max` defaulted to 0 on a missing desc, and 0
is not neutral — it *is* the empty-past branch. A missing desc and an empty cache became
the same number and were thereafter indistinguishable, so a caller-side defect arrived
here as a plausible input and left as a wrong answer. Converting it to a refusal located
the root in one run: the error text named `past_key` and the shape it did not have.

**A control that arrived free.** criterion 10 on the real model still flags exactly
[0, 63, 64] with the same 0.0625 and the same 62/65 — but `present.31.*` is now produced
by the Vulkan GQA kernel instead of the CPU EP. **The producer changed completely and the
number did not move.** That is what a tolerance mis-scoping does and what a producer
defect cannot; it is a stronger form of the refutation I filed at `2453847` and it cost
nothing.

**Flagged, not fixed.** The attention loop reads `present_key` for `t` in
`[past_len, tok_pos)` — written by sibling invocations with no barrier. Unsound for
`seq_len > 1`. The past->present copy does not worsen it.

**Verification:** 473 lib + 59 integration green; clippy `--all-targets -D warnings`
green. Four new unit tests pin the contracts (rotary from cos_cache, refusal when
unrecoverable, growing extent, shared-buffer aliasing).

**Next:** KV bandwidth is now unblocked — 27.7% of the byte floor at past_len 2048, 60.5%
at 8192, and this model does no grouping at all (32 KV heads for 32 query heads).

## Session 46f — the prefill race, the defect underneath it, and the KV byte model

**Merged `main` (`db8affd`).** DLL `DE076B465404E6E1` either side; `rust/` untouched by the merge.

**Fixed the sibling-key race in `gqa_f16.comp` that I flagged last session** — the attention loop
read `present` for `t ∈ [past_len, tok_pos)`, positions written by sibling invocations of the same
dispatch with no ordering guarantee. Both branches now recompute K (with RoPE) and V from
`packed_qkv`, a read-only input. Bit-identical by construction, not approximate. Deleted
`f16_presk`/`f16_presv` so no reader of `present` remains.

**Then the falsifier found a defect I had not predicted, and it was the larger one.**
`probe_gqa_prefill_race.py` read `DIVERGENT` at `worst_rel = 774.8` *after* the race fix, and
deterministically so. The shader read `past_len = seqlens_k[b]`. ORT defines `seqlens_k[b]` as
`total_sequence_length - 1`, and `total = past + seq_len`, so `past = seqlens_k[b] + 1 - seq_len`.
**The two expressions are equal exactly when `seq_len == 1`.** Every case this kernel had ever run
was decode. Prefill placed the first query token at `past + seq_len - 1` and indexed past_key
beyond its extent.

`DIVERGENT 774.8 → MATCH 0.000724`, present KV bit-exact. Decode control unmoved to the digit
(`0.00072939`, `COPY EXACT`).

**Fourth decode-only assumption in this kernel** after `present == past`, `rotary_dim == head_dim`
and `past_len_max unwrap_or(0)`. They share one shape and it is worth naming: *a quantity with two
definitions that coincide on the test set is invisible to that test set.* The fix is never to think
harder about the quantity; it is to run the case where the definitions separate.

**KV byte model (`probe_kv_traffic.py`).** The read is **irreducible in elements, not in bytes** —
softmax has no zeros so every element is required, but bytes = elements × precision and precision
is a choice. And **we move 3× the irreducible term**: the present-copy round-trip (+2 units, forced
by the graph declaring `past` and `present` at different extents) and group amplification (`Nq/Nkv`,
exactly 1.00 on Phi-3.5 and 4× on a genuinely grouped model).

**This corrects my own published table upward** — KV at `past_len` 8192 is **82.2%** of traffic,
not 60.5%. Levers: dropping the present-copy is **2.21×** and is not a kernel change at all.

**My detector for the group term first read `False` on a fact that is plainly true** — it matched a
loop bound absent from the file, and False there *understates our own amplification*, the
comfortable direction. Fixed to read the actual gid decode, and the probe now refuses rather than
publishing a byte model whose shader facts it could not confirm.

Commits `b060f47`, `277f670`. Not pushed.

## Session 46g — the 2.21× was not there, and half of it was

**Merged `main` (`112d712`).** DLL `39CAE83A974D4BE7` → `2F5FC71ED0AB158E`.

**Refuted the lever I was asked to land, before building anything.** `bind_aliased_output` returns
the *input's* buffer for the output, so aliasing `present` onto `past` requires present to fit
inside past's allocation. The graph declares `past` at `past_sequence_length` and `present` at
`total_sequence_length` — present is strictly larger for every `seq_len ≥ 1` and every `past_len`
(3072.4 MiB into 3072.0 MiB at ctx 8192). **There is no regime where they coincide**, so unlike the
four earlier defects this is not a definition that hides on the test set; it is infeasible
everywhere in this convention. The runtime does not forbid it — the declared convention does, and
our `shares_past_buffer` branch fires the moment a graph declares otherwise.

**Landed the half that was real.** Of three units of KV traffic per token, exactly one was
removable: the copy *read*. The copy leader's own attention loop already reads every past element
of its `(b, kv_h)` one dispatch step later — the standalone step-3a loop was reading the same bytes
a second time to relocate them. Fused: the leader writes each element to `present` at the moment it
loads it for the attention sum.

**Predicted before building, then measured:** predicted 3→2 units, floor at ctx 8192
45.93 → 33.35 ms, 1.376×. Measured 33.34 ms, **1.377×**. At ctx 0 exactly zero, correctly — that is
the regime our one quotable figure lives in.

**Not tuned to Phi-3.5's degenerate grouping**: the fusion removes one unit of `(Nq/Nkv + 2)`, so
it is 1.500× here, 1.200× at 4:1, 1.111× at 8:1 — never a regression, worth most where grouping is
worst. The reverse trap stands: the `Nq/Nkv` attention term is 1.00 here and 4× on Llama-3.

Correctness from comparison against the CPU EP, not from the ledger agreeing: prefill `MATCH`
(present exactly 0, on a *genuine* 4:1 grouping), decode `COPY EXACT` and `0.00072939` unmoved,
criterion 10 identical to the digit at 62/65.

Commit `4b98b63`. Not pushed.

📌 Team update (2026-08-02T14-42-30-07-00): Niobe measured that past context length 2048 we are link-bound, not memory-bound (KV-cache readback exact at 393,216 B per past token, ratio 1.000000). This changes what the present-copy KV-cache fix is worth once contexts grow past that point, and connects to `offer_shared_device`, whose default is `OFF` and whose recorded reason for staying off Morpheus has separately ruled expired. Worth revisiting whether the default should change now that the bottleneck downstream of it has moved. — decided by Niobe

## Session 46h — 2026-08-02 — the input cache was serving stale KV, and the fix made Niobe's UNOBSERVABLE upload MEASURED

Chasing Niobe's `UNOBSERVABLE` upload axis to its mechanism found a live correctness defect.
The past-KV upload was in the **intercept, not the slope** — the whole past cache uploaded once
per session — because the EP's input cache was keyed on `(cpu_ptr, byte_size)` with a 32 KiB
floor and no content check. Phi-3.5's `past_key_values.N.key` is `past_len * 6144` B, so it
crosses 32 KiB at `past_len >= 6`. All 64 KV inputs were cached as weights; every inference
after the first read the first inference's KV.

`probe_kv_input_cache.py` (new falsifier): pre-fix `STALE_CACHE` — run B returned run A's
answer to the bit while the correct answer moved 0.0157. Post-fix `FOLLOWS_DATA`. Both guards
live (reference must move; Vulkan must have executed).

**The predicate was one struct away.** ORT answers it — but *not on the fused node*. Measured:
`ValueInfo_IsConstantInitializer` is false for all **457** fused-node inputs while the body
nodes report **388** constant. ORT surfaces initializers as fused inputs without re-marking
them at the boundary. Constancy recovered by name join in `subgraph_plan`; contradictions
resolve to `false` (a false negative costs an upload, a false positive costs a wrong answer).

**Two mistakes I made and caught with controls, not reasoning:**
1. First version keyed on the fused node alone → weight cache silently stopped firing,
   2.29 GB uploaded *per inference*. Caught only because I ran the byte control instead of
   trusting that "weights are initializers".
2. `probe_kv_bytes_earned.py` printed **negative bytes per inference**. The 512×25 worker had
   lost the device, exited 0, and written a complete counters file. Guards now refuse on
   `compute_failures`, short `compute_calls`, and zero `dispatches_executed`.

**Attribution I checked rather than assumed:** `VK_ERROR_DEVICE_LOST` at past 512 reproduces on
the **pre-fix binary on `main`** too (compute_calls 6, failures 1). Not mine. `past_len=512` is
outside the extent this box currently permits; added `--past-lens` so a narrowed extent is
stated, not assumed.

**The payoff for the project's byte model:** upload is now MEASURED at 393,216 B per past token,
ratio 1.000000 — matching Niobe's readback term to the byte. Her instrument was right; the
build was wrong. Link traffic at ctx 8192 is 6144 MiB/inference both directions vs 8140.8 MiB
DRAM. **Against my own interest: my 1.377× fused-copy result is a DRAM reduction and moves no
byte off the link axis, so at ctx >= 2048 it is admissible but not binding.** `probe_kv_traffic.py`
now prints the link axis and asserts **no** link rate — this box's rate is unmeasured.

Criterion 10 is identical to the digit (62/65, `[0,63,64]`, 0.0625/0.005859375/0.015625) and is
a **null result inadmissible on its own terms**: it runs at past 0 where the changed branch is
provably unreachable. My own rule, applied to my own change.

Commit `b602bbb`. Shipped DLL `8818A8C40729292F`. 473 lib + 59 integration green, clippy clean.
Not pushed.

Decision records filed: *an address identifies storage, not contents*; *a counters file is
evidence that counters were written, not that work was done*.

## Session 46i — 2026-08-02 — the ctx-512 device loss: mechanism named, reproduction not available

Merged `main` (4b5d46b) at 812d64d. Commit **a52024f** on `squad/switch`. DLL 4040C588638EDDED -> 87AB2650513EE0C8.

**The ledger declined my own shader, and it was right to.** Mouse's digest-and-reprove path
landed in main: the GQA entry was proven against `2c565607e9179ccb`, this build's `gqa_f16`
hashes to `c55e2e518a0559d4`, so GQA was declined on my branch and my first isolation arm ran
entirely on the CPU EP — `calls=0 dispatches=0`. It would have reported "GQA is innocent" from
a run GQA never entered. Re-proved: MATCH, worst_rel 7.294e-04, ledger 74 entries / 845a7fc97d1a87aa.
The gap I filed earlier is closed and it fired against me first.

**Instrument defect for Mouse:** `--reprove` without `--append` rewrote the ledger from 74
entries to 1 and printed `PASS`. A destructive write reported as a passing check.

**TDR hypothesis refuted mechanically.** nvlddmkm/13 (FECS exception) x19 and nvlddmkm/153 x21
in 14 h; **Display/4101, the TDR signature, zero in 24 h**. It is a driver/GPU fault, not a
watchdog timeout — so "the GQA loop is too long for the watchdog" is not the mechanism.

**And the events track the clock, not the build.** All 40 fall in 15:02-16:21. Afterwards I ran
~460 inferences at ctx >= 512 (gqa 512x200, gqa 2048x100, phi 512x25 five times, phi 1024x25,
phi 2048x15 — 8875 dispatches per phi inference) with zero losses and zero new events, on a
heavier workload than the fault window carried. Earlier today BOTH binaries faulted inside it,
including pre-fix `main` under a stash-and-rebuild control. Not a claim of a fix: mechanism
named, reproduction not currently available.

**The reporting defect is fixed and is independent of the mechanism.** `device_losses` counter
(cross-owner edit to Tank's counters.rs, emitted at zero so probes can refuse on it), status text
naming VK_ERROR_DEVICE_LOST, `failure_condition_token` -> "device-lost" checked before the
shape family, `disable_fallback()` in every harness, non-triviality guard exiting 2 on
`compute_calls == 0`, and `probe_kv_bytes_earned.py` refusing any point with device_losses != 0.

**The token test failed on the driver's own wording, first try.** The matcher had "device was
lost" and "device_lost"; the text that arrives is "The logical device has been lost", which
matched neither and fell through to `shape` because it also contains "bytes". *A classifier
tested only against strings we wrote ourselves is tested against the one input it cannot get
wrong.*

478 lib tests, 15 epctl, clippy clean.

## Session 46j — 2026-08-02 — the KV arena: allocatable, bindable, and still not removable

Merged `main` (2397f5a, docs-only fast-forward; DLL hash correctly unchanged at
87AB2650513EE0C8). Commit **898a2ba** on `squad/switch`. DLL -> F226AD136BE1842E.

**Which world are we in: a third one.** ORT *does* allocate fused-node outputs through this
EP's device provider — measured 195/195 armed vs 0/195 in the disarmed control, frame SHARED
vs OFF. So "ORT forbids it" is refuted, and the shipped path is worse than it looked: armed, we
download all 65 outputs to host staging and memcpy them into a pointer that is itself device
memory. A device->host->device round trip.

**But binding them returns zeros.** I built the output-side `bind_target_for` (Step 1c,
`ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS`, default OFF) and it produced ALL-ZERO logits and
present tensors. `transfer`'s own doc predicted it — the host staging block is authoritative
and the device buffer is a mirror. The kernel writes the device buffer, the caller reads the
host block, nothing writes the host block.

So: allocation permitted, binding permitted, **authority not available** — because the consumer
of `present.*` is the caller and the caller reads host memory. The 2.21x is not available in
the shape it was costed in. It needs IOBinding with device OrtValues (a caller-side change) or
an EP-owned cache ORT never sees.

**My scorer nearly certified the zeros and that is the lesson.** First verdict rule scored
`cpu~bound` against `cpu~ep` and returned BOUND_OUTPUT_IS_SOUND: the zero lane won at
**1.0 vs 1.9958**, because an all-zero tensor scores 1.0 on a relative metric and the
incumbent's logits score saturates near 2 on a 1e-3 denominator floor.

*A scoring rule that can be beaten by returning nothing is not a scoring rule.*

Fixed with a degeneracy guard and by making the primary criterion `ep` vs `bound` rather
than `cpu` vs `bound` — same kernel, same inputs, only the writeback path differs, so
anything but agreement to the digit is the writeback path changing the answer. Scoring against
the CPU EP let the EP's own tolerance hide it. Second time in two days a noisy incumbent nearly
admitted a worse replacement.

**Corollary:** the bound lane made `outputs_device_resident` read 0, because a bound output
never reaches the site that records residency — indistinguishable from the change not firing.
The probe refused it as ERROR(instrument). *A change that removes a code path also removes
whatever that path was counting.* Fixed with a separate `outputs_device_bound` counter
recorded at the bind.

Default path re-verified unchanged on the shipped binary. 478 lib tests, 15 epctl, clippy clean.

---

## Session 46k — 2026-08-02 — push constants, then the KV ruling

Merged `main` at `d375a4d` (`ee3648c`). Two commits, neither pushed.

### `abcb9af` — every dispatch writes every byte of the range it declares

Trinity's in-frame reading: `vkCmdDispatch(): Pipeline uses a push constant range with offset 0
and size 128, but 104 bytes were never set` — six lines, shortfalls `{4,20,36,72,88,104}`, which
is 128 minus the six distinct pack sizes across 355 dispatches. Not a VUID. Still a defect:
**unwritten push-constant bytes are undefined, not zero**, and nothing misbehaved only because
no shader reads past its declared block — a property of the shaders, not of the contract.

Padded rather than shrank the declared range: the pipeline cache is keyed on
`(shader, spec_constants)` with no push size in it, so a per-kernel range would have to enter
that key and a layout disagreeing with its dispatch is a hard error, not a warning. Push is now
unconditional (a kernel packing nothing would leave all 128 unwritten). Over-128 logs ERROR once
naming the shader.

**The zero is earned twice.** Liveness via Trinity's `BEST_PRACTICES_EXT`, and *sensitivity*:
the same reading against the **pre-fix binary**. A detector never seen in its positive state has
no demonstrated positive state — the probe reports `UNPROVEN_DETECTOR` without one, and refuses
a control whose DLL hash equals the subject's. 6 lines on `44D21A451D269F82`, 0 on
`A8BAB570AB8BE38D`, both devices, device read off the run.

Liveness count moved 14 → 8 — exactly the six lines removed. Trinity's assertions are `> 0` and
`!= clean`, never `== 14`, so a fix did not turn her control red. Footnoted her table.

### `ed48f5b` — the KV ruling: ORT does not forbid it, the obstacle is ours

| lane | caller-side bind | + EP-side Step 1c |
|---|---|---|
| `alloc_device_frame` | `SHARED` | `SHARED` |
| `outputs_device_bound` | 0 | 6 |
| nonzero returned | 256/320/320 | 0/0/0 |
| rel vs **unbound EP** | 0.0 | 1.0 |
| verdict | `KV_CAN_STAY_DEVICE_RESIDENT` | `DEVICE_BOUND_OUTPUTS_RETURN_NOTHING` |

A caller **can** allocate an `OrtValue` in our device memory and bind it as a graph output,
bit-identical to the unbound run. The obstacle is `transfer.rs`'s own invariant: **host staging
is authoritative, the device buffer is a mirror.** Nothing makes a directly-written device
buffer authoritative. That is EP-side work, in our hands.

Route: `device_type='gpu'` + our vendor id fails with *"Can't allocate memory on the CUDA
device"* — ORT 1.28's Python binding maps `gpu`→CUDA, so a plugin EP is unaddressable by the
documented spelling. `OrtEpDevice.memory_info(DEFAULT)` as `memory_info=` is the escape. The
binding labels the result `'cuda'`; recorded, never used as evidence.

**Ordering is load-bearing.** Asking for the allocator before the session exists builds a second
`VkDevice` — `SPLIT-DEVICE`, unbindable by any dispatch. The probe's first run read it and
refused rather than publishing plausible numbers about a device the kernels never ran on. Any
arena inherits this.

Ran on the GQA evidence case: seconds per lane instead of six minutes, which is the only reason
both lanes exist to be compared.

**Not claimed:** the round trip is not removed; `readback_bytes` is not quoted because it is not
yet expected to have moved.

### Lessons
- *A property of the shaders as they happen to be written is not a property of the contract.*
- *A detector never observed in its positive state has no demonstrated positive state* — hence
  the pre-fix sensitivity record, and the refusal when its hash equals the subject's.
- *A falsifier that asserts the exact value of a number it does not own goes red on a fix.*
- The `ep`-vs-`bound` criterion and the degeneracy guard, both minted last round, both fired
  this round: the epbind lane scores `1.0` and is caught by the nonzero count, not the score.

### State
478 lib + 15 epctl green, clippy clean, shipped DLL `A9A381602D8B4014`. Decision records filed:
`switch-ort-permits-device-resident-kv.md`,
`switch-declared-push-constant-range-must-be-fully-written.md`.

**Next:** make a directly-written device buffer authoritative in `transfer.rs` /
`host_device_memory.rs`. That is the whole remaining distance to the KV arena, and it is ours.

## Session 46m — 2026-08-03 — the round trip on the real graph

**Question:** does the real Phi-3.5-mini graph, with its 64 KV outputs, actually decline the round
trip across a multi-step decode chain? The GQA case fixed `past` at 4, so `ROUND_TRIP_REMOVED` was
a lower bound and never a number.

**Answer: yes, and the slope is flat.** `bench/results/probe_kv_chain_phi35.py`, real 355-node
island, 64 `present.*` outputs, 6-step chain with each `present` fed back as the next `past`:

- `host` (shipping): 2,030,208 -> 3,996,288 B link traffic, slope **393,216 B per past token** —
  Niobe's declared figure reproduced to the byte, which is what licenses reading the other lane.
- `resident`: **64,128 B flat**, slope **0**. Same 355 dispatches/step, 2130 total, both lanes.
- Bit-identical logits vs `host` on all 6 steps. Same token chain as the CPU EP.
- Both devices, names read off the run: RTX 4060 Laptop (0x10de) and Iris Xe (0x8086).

**The fix that made it fire.** Step 1b in `vk/session.rs` looped `host_backing_for` over *all*
inputs, and that refreshes a device-authoritative span — a download — before returning a staging
address nothing in the dispatch reads. So every KV input Step 1a had just bound on the device was
downloaded anyway: 64 downloads/step, the whole 393,216 B/past-token slope, sitting on the *input*
side after the output side stopped paying it. One `continue` and a long comment.

### What surprised me
**Both of my "findings" this round were my own probe.** `copy_outputs_to_cpu()` materialises every
bound output — 65 downloads and the entire round trip, charged by the instrument to the thing it
was measuring. And `binding.get_outputs()` is in **binding** order while `sess.get_outputs()` is in
**session** order, so indexing one with the other handed me `present.0.key` when I asked for
`logits`. That fabricated two credible defects: an unexplained residual of exactly 6,144 B/step
(`32*96*fp16` — a number I could derive from the model, which is precisely why I believed it), and
a correctness bug I had written up as "ORT's CPU-bound output path returns near-zeros under
`BIND_OUTPUTS=1`". Neither exists. After the fix the residual is zero and the lanes are bitwise
equal.

What caught it was not inspection. It was **step 0 disagreeing between two lanes that are the same
inference**. A byte count cannot tell you it measured the wrong tensor; a bitwise comparison
against an identical computation can. New standing rule: a bandwidth lane carries a correctness
control sharing its inputs exactly, and the correctness check is read *before* the byte count.

### Also
- The four separating cases now run in-probe and pass: session outliving an inference (and the
  OrtValues it wrote), two sessions on one device interleaved, a context outgrowing its first
  allocation over 6 growing spans, and a readback taken at the first instant the API permits with
  no caller sync. None of the four is exercised by the chain.
- `output_bind_requested()`'s INFO text was still claiming only `copy_outputs_to_cpu` pays.
  Replaced with the real-graph numbers.
- Degeneracy guard held: 100% nonzero, ~14,666 distinct values, so the agreement figures are
  admissible.

### State
492 lib green, clippy clean (`-D warnings`), DLL `D408A901C4F6A454`. Decision records filed:
`switch-round-trip-declined-on-real-graph.md`,
`switch-bound-input-must-not-be-refreshed-through-host.md`,
`switch-instrument-defects-that-looked-like-runtime-defects.md`.

**Not quoted, deliberately:** no end-to-end improvement. The round trip is declined on the axis it
was measured on; that is not the same claim as a faster decode.

**Untested and said here rather than in a comment:** `Nq/Nkv = 1.00` on this model — the degenerate
grouping. It is 4x on Llama-3 8B and the general grouping case has not been run.

**Next:** the general grouping case. Nothing in the fix is keyed on head counts, but nothing has
proved that either.

## Session 46n — 2026-08-03 — what `DEVICE_MEMORY` is still protecting against: four callers, none separates

Merged `main` (`607056a`) first. 513 lib + 15 epctl green, clippy clean (`-D warnings`).

**The hazard family a *memory* flag exposes, run for the first time.** New
`bench/results/probe_device_memory_hazards.py`: allocator-asked-for-before-the-session, two
sessions on one device interleaved, 65 device `OrtValue`s read after their session is gone, and
an allocation that fails partway through the run. All 65 outputs compared byte for byte against
the shipping path, twice per lane. `NO_HAZARD_LANE_SEPARATES` on both vendors — 130/130 per lane,
`alloc_failed_lookups = 0`, `alloc_frees_after_release = 0` in the two lanes written to provoke
them.

**`SPLIT-DEVICE` declines itself.** The ordering trap I filed as a reason to be careful is closed
by a guard that already existed: `bind_target_for` condition 2 refuses a frame that is not
`Shared`, so the early-allocator lane binds 0 of 130, takes the shipping route, and returns the
same bytes. The check was written before a caller existed who could reach it. One now does.

**An allocation failure is a first-class case and now has an instrument.**
`try_attach_device_buffer` had four exits and all four were the same silent missing increment.
Split into `alloc_device_attach_{attempts,failures,unavailable}`. Added
`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB` — a provider cap, uncapped by default, reported as
`alloc_device_memory_budget_bytes` so no artifact recorded under it can be quoted as if its
failures were discovered. At 8 MB: 605 of 648 attaches refused, 43 spans device-backed, 130 output
binds declined, **0 compute failures, outputs identical to the byte**, traffic back through the
staging doors. Uncapped control on the same lane: 648 attempts, 0 failures.

**One flag, two parsers, and they disagreed.** `factory` took `1|true|yes|on`; the allocator took
"anything non-empty that is not `0`". `DEVICE_MEMORY=off` therefore **armed the allocator's attach
while leaving the allocator un-advertised** — half-armed from a spelling that reads as "off". Same
shape as the `disable_cpu_ep_fallback` trap Trinity found, inside our own flag. One function now,
with a test that calls both entry points on twelve spellings and asserts agreement. Polarity is the
opposite of `BIND_OUTPUTS` on purpose: a typo must fail towards whichever path *ships*.

**ctx: the boundary is 6144 and 8192 is arithmetic, not a defect.** Predicted from the model's
shapes before running (`2 × 393,216 × C` + 2.29 GB weights): 6144 → 7.13 GB fits, 8192 → 8.73 GB
does not. Measured on the 8 GB discrete GPU: at **6144 the shipping lane fails at the first
Compute (0 dispatches) and the resident lane completes 1065 dispatches at 64,128 B/step flat**; at
8192 both fail. Largest context ever reached on this box: **6144, resident route only** — 75% of
the operating point the 82.2% figure is quoted at. Nothing extrapolated past it.

### What surprised me
**Every hazard I could name was already closed, and by guards written before any caller could
reach them.** I expected at least one of the four lanes to separate — the split-device one most of
all, since I had filed it myself as the ordering trap. It declines cleanly. The honest reading is
that the flag is not protecting against anything I can still name from the code; it is protecting
against the three things nobody has *looked* at (device loss at ctx 512, two devices in one
process, concurrent sessions) plus an operating point that does not fit in the card.

Also: **my own probe died once, and the cause was mine again** — the outlive lane derived the KV
extent from a loop index instead of from the tensor, was one token short, and ORT refused the
pre-allocated output. Fourth instrument defect in this family. It now reads the extent off
`past["past_key_values.0.key"].shape[2]`, which cannot be re-derived wrongly.

### Verdict
**`DEVICE_MEMORY` does not flip this round, and the reason is now a list of four rather than a
doubt:** Tank's intermittent device loss at ctx 512; the `MIXED` two-device frame
(`two_allocators_on_two_devices` still `#[ignore]`); concurrent sessions on threads; ctx 8192.
Items 2 and 3 are a day each and mine; item 1 is Tank's; item 4 is the KV arena.

Decision record: `.squad/decisions/inbox/switch-device-memory-hazard-family-and-the-four-gaps.md`.
Artifacts: `bench/results/device_memory_hazards-dev{0,1}.json`,
`bench/results/phi35_kv_chain-ctx6144-{resident,host}-dev0.json`.

---

## Session 47 (2026-08-03) — the KV arena: `present` aliases `past`, ctx 8192 reached at 5.51 GB

**Housekeeping first: the Scribe condensation was undone by the merge.** `a9d8693` condensed this
file to 13,914 bytes, but `822ac0c` was cut from a pre-condense parent and `.gitattributes` sets
`merge=union` on `.squad/agents/*/history.md`, so the merge re-appended the full body — 108,195
bytes again. **Nothing is lost.** The condensation is what was lost, and it is a Scribe item, not a
content one. Union merge and summarisation are incompatible on the same file.

### What shipped
`ONNXRUNTIME_EP_VULKAN_KV_ARENA=1` makes `present.*` and `past_key_values.*` **one allocation**.
Peak KV memory `2 × 393,216 × C` → `1 × 393,216 × C`.

- **ctx 8192 reached and measured for the first time in this project: 5,512,528,520 B**, 355
  dispatches/step, 0 compute failures, 0 device losses. The shipping lane dies with `alloc failed
  for output buffer`. Verdict `ARENA_RAN_WHERE_GROWING_COULD_NOT`, reproduced twice. **5.51 GB was
  written down before the run** and the run landed on it.
- ctx 2048: 3,900,736,136 → 3,096,609,416 B, a saving of **804,126,720 B = 2045 × 393,216 exactly**
  — the present copy dropping out to the token.
- **BIT_IDENTICAL on all 65 outputs** at A=64 and A=2048 against the *shipping Vulkan lane*, and on
  **Intel Iris Xe** as well as the RTX 4060. Correctness read before the byte count.

### Soundness came from the kernel, not from a residual
`gqa_f16.comp` step 3 writes `present[tok_pos]`, step 4 reads `past[t]`, `t < past_len`. Under one
common stride these are disjoint for all invocations because `tok_pos = past_len + s_local ≥
past_len`. The single read of `present` that used to exist was removed on 2026-08-02 and is
recomputed from read-only `packed_qkv` — **had it survived, the arena would have turned a benign
redundancy into silent corruption.** Also put to ORT on the CPU EP first (`--mode graph`, poisoned
arena tail): `ARENA_SHAPE_HONOURED_BITWISE`, `max_abs 0.0`. Unpredicted: ORT's own GQA returns
`present` at the *past extent* — ORT already uses the shared-buffer convention.

### The defect I shipped and then found
`translate_gqa` shortened `present` on the strength of the **flag alone**. A caller who binds
nothing got `max_abs 60.82`, all 64 KV tensors wrong, **exit 0, no counter moving**. That is the
two-parser failure one level up: a *declaration* treated as a *fact* about where ORT put a tensor.
Fix: a sweep after the whole output-binding block (`session.rs` ~1629) that **refuses the Compute**
when an aliased output is not bound to its input's `VkBuffer` — placed outside the block so
`BIND_OUTPUTS=0`, a failed authority mark and a declined span are all caught. No fallback exists:
once `present` is short, the staging route writes the same short tensor. After the fix:
`dispatches_executed 0`, `compute_failures 1`, `broken_commitments 1`, CPU fallback, answer correct.

### Separating cases, all run
growing-caller → `GROWING_CALLER_REFUSED_LOUDLY(ORT shape check)` (a fact about ORT 1.28, not about
this EP); unbound caller → refused loudly; **allocation failing partway** (budget 2250 MB, A=512) →
43 of 454 attaches fail, 43 declines, 0 dispatches, refused; boundary ≈ 2377 MB.

### The instrument that lied by staying green
My `gqa_f16.comp` edit moved the SPIR-V, the ledger subject changed, and the EP declined **all 32
GQA nodes** — while `gen_proof_ledger.py --check` said `PASS — 103 entries` the whole time. `--check`
checks the file against itself; the subject comparison happens at runtime against *this build's*
embedded digests. Only `subject_changed_declines` saw it. And the ledger is `include_str!`'d
(`registry.rs:1890`), so **`--reprove` has no effect until you rebuild** — that cost two full probe
runs. Re-proof gave `worst_rel = 0.0007293946024799417`, **identical to pre-edit**: the capacity
guard changed no arithmetic. Separate decision record filed; this is project-wide, not arena-local.

### Residuals I am naming rather than rounding away
1. **The arena capacity is a ceiling, and overrun is dropped, not refused** — the shader guard
   discards a step past the allocation and the EP cannot detect it, because the true past length
   lives in `seqlens_k` on device. The one place the arena can still be quietly wrong.
2. **`Nq/Nkv = 1.00` on Phi-3.5, 4× on Llama-3.** Nothing was tuned to it and the disjointness
   argument does not use it, but **no run this round exercised a non-unit grouping** — and that is
   exactly where an aliasing bug would hide.
3. 7 ledger forms still carry `entry-device=device0`; the GQA entry no longer does.

### For Mouse
**The arena introduces no specialisation constant** — `pipeline_variants` shows `"gqa_f16:"` with an
empty selector list. It changes a **push constant** (`present_len`) and the **binding topology**.
`kv_cache_convention` is the witness for that class, recorded in the dispatch loop off the effective
push constants — where his frame witness sits.

### Gates
526 lib passed / 0 failed / 4 ignored; clippy `-D warnings --all-targets` clean; `counters_abi.py
--check` PASS, layout unchanged `(8, 0xdf71f4e6a59271b3)`; `gen_proof_ledger.py --check` PASS, 103
entries, digest `94d994ba54821056`. **Nothing went red-once-green-after this round** — nothing for
Trinity. No clock anywhere. Device names read off the run. The DLL hash is quoted as evidence of
nothing.

Decision records: `.squad/decisions/inbox/switch-kv-arena-present-aliases-past.md`,
`.squad/decisions/inbox/switch-ledger-check-cannot-see-subject-changed.md`.
Artifacts: `bench/results/kv_arena_{graph_accepts,chain-A64,chain-A2048,chain-A8192,chain-intel,separating,unbound,budget}.json`.

## Session 48 (2026-08-03) — the int8 KV error budget, measured before a kernel: it saturates

**Merged `main` (`8ac1172`) first.** 532 lib passed / 0 failed / 4 ignored, clippy `-D warnings
--all-targets` clean, `counters_abi.py --check` PASS `(8, 0xdf71f4e6a59271b3)`,
`gen_proof_ledger.py --check` PASS 103 entries digest `94d994ba54821056`. No shader was touched
this round, so the `--check`-cannot-see-subject-changed defect had nothing to hide.

**The merge conflict in `counters.rs` resolved correctly.** Both sets of JSON fields are kept and
the format string's order matches the argument order exactly through the contested block:
`pipeline_variants` -> `gemv_packed_spec_constant` -> `shaders_dispatched_spec_digest` ->
`specialisation_delta_forms` -> `specialisation_unrecorded_forms` ->
`ledger_specialisation_unrecorded_entries` -> `kv_cache_convention`. Checked against both parents
and re-run: 532 green, and the three counters tests that assert on the emitted JSON text
(`pipeline_variants`, `gemv_packed_spec_constant`, `UNOBSERVABLE`) pass, which is what would catch
two same-typed arguments transposed.

### No kernel was written, deliberately

The task was the error budget, and the budget is what gates the kernel. `probe_kv_int8_budget.py`
quantises the cache at the **host** boundary — storage error modelled exactly, kernel write
rounding and accumulation order not modelled at all, so **every residual is a lower bound on a real
int8 kernel's, never an estimate of it**. If int8 is unaffordable at the lower bound, no kernel
makes it affordable. Trinity's `ulp_residual` used **unmodified**; I did not build a second one.

### What it cost, both devices, inputs shared exactly

8-step chain, fp16 CPU-EP oracle, one seed-KV hash and one fixed non-argmax token sequence asserted
equal across all nine lanes, correctness read before any byte count, liveness checked (2840
dispatches, 0 failures, 0 device losses on every Vulkan lane, RTX 4060 and Iris Xe):

| lane | KV worst-median ULP | logits median | top-1 | footprint ratio (MODEL) |
|---|---|---|---|---|
| fp16 Vulkan control | **3** | 1 | 8/8 | 1.000 |
| int8 `per_block32` | 18–22 | 2–5 | 8/8 | 1.377 |
| int8 `per_head` | 27–28 | 6–12 | 8/8 | 1.401 |
| int8 `per_tensor` | 63 | 14 | 7/8 | 1.412 |
| int4 `per_block32` | 343 | 60 | 5/8 | 1.724 |
| int4 `per_head` | 732 | 81 | 5/8 | 1.761 |

Granularity monotone as predicted. **Vulkan and CPU agree to within 2 ULP at every granularity, on
both devices — the EP does not amplify the quantisation error.** Verdict
`NO_ULP_BAND_ADMITS_INT8_AND_STILL_CATCHES_FP16`: the best granularity sits at **6-7x** the fp16
path's own residual, so any band admitting int8 stops policing fp16. **No tolerance was chosen.**
Three candidate ruling shapes filed with Morpheus; I rejected shape 1 (widen the ULP band) in
advance and named shape 2 (change the observable on the logits) as the one the data supports —
without ruling it, because that is his.

### What surprised me, and it reverses my own reading twice

**1. The residual saturates.** I predicted flat in `past_len`. The 8-step run said rising —
**1.60 ULP/step** for `per_head` — and I filed that as compounding. So I ran it to **past_len 259**
in lockstep (`probe_kv_int8_depth.py`, oracle and quantised lane in one process, comparison thrown
away each step because the saved tensors would have been 6.6 GB). It compounds and then **stops**:

| lane | storage-only (seed tokens) | saturated | reached by | power-law exponent | top-1 / 256 |
|---|---|---|---|---|---|
| int8 `per_block32` | **9** | **18–19** | past_len ~20 | 0.065 | **250/256** |
| int8 `per_head` | **11** | **29** | past_len ~28 | 0.113 | **249/256** |
| int4 `per_block32` | **165.5** | **~340** | past_len ~20 | 0.073 | **177/256** |

At past_len 259 the profile along the token axis is **flat** — position 4 and position 256 read the
same ULP. Old tokens are not worse than new ones. The multiplier from pure storage error to the
fixed point is **~2x at every bit width and granularity**, which is a fixed point, not an
accumulation: error feeds forward through attention, and attention's convex combination dilutes it
by `1/past_len` at the same time.

**Carried linearly to ctx 8192, that 1.60 ULP/step predicts ~13,000 ULP. The measured value is 29.
A slope over 8 steps would have been wrong by ~450x, in the direction that kills the lever.** This
project's refusal to extrapolate slopes now has a number on it, out of its own tree, and it cost one
CPU-EP lockstep run.

**2. A max-ULP criterion would rank the fp16 GPU path as worse than the quantised cache.** The fp16
Vulkan control's max ULP on the logits is **337,178**; every int8 CPU lane's is smaller (7,886 /
45,638 / 38,278). Trinity's R11 said max ULP cannot acquit; this is the same sentence pointed at a
criterion, with a number on it.

**3. My own cancellation counter explained nothing.** It counted `b == 0.0` exactly and returned
**0 everywhere** while `max_ulp` read 6.3e6 — the counter that exists to explain the max explained
nothing, and a reader would have concluded the max was real. The spacing floor is reached by any
reference below the smallest fp16 normal, not only by an exact zero. Both are recorded now: 18,765
subnormal references (0.45% of the worst tensor) against 0 exact zeros. **Fifth instrument defect
in this family, mine again, and again found by an observable disagreeing with another observable
rather than by inspection.**

**4. The ledger's ratios do not reproduce.** The lever ledger quotes **2.21x / 3.17x / 4.06x**. I
cannot reproduce any of the three from any artifact in this tree, on any baseline — footprint,
modelled stream, KV-only, with or without the present write. What the artifacts support is int8
**1.40x** and int4 **1.76x** on the footprint (**1.42x / 1.81x** on the modelled stream). **This
disagreement was written down before the first int8 run**, in
`bench/results/kv-int8-budget-prediction.md` §3, so that if int8 landed near 1.4x the ledger would
be what was wrong rather than the measurement being explained afterwards. At 1.40x int8 is the same
order as the 2.21x already banked, and it costs a correctness argument the arena did not — that
changes the ranking.

### The byte saving is NOT measured, and is not quoted as if it were

Every int8 byte figure above is class **MODEL** (`provenance` in the record). The arena's
5,512,528,520 B at ctx 8192 was measured; nothing int8 can be until a kernel exists. The
measurement that would settle it is the one the arena round established: a slope at equal work on
both devices with a correctness control sharing inputs exactly, read before the byte count. I will
not quote a modelled ratio as a measured one to close a task.

### Named rather than rounded away
1. **`Nq/Nkv = 1.00` on Phi-3.5, 4x on Llama-3.** `per_head` here is a per-head-group scale over a
   group of **one** query head. No run exercises a non-unit grouping; **no number here may be
   quoted for a 4x model.** This is the lever where tuning to this model would hurt most.
2. **ctx 8192 is not reached by the depth run** — measured to past_len 259, and a fitted exponent is
   not a licence to extrapolate, which is exactly the mistake this round documents one level up.
   What would make it knowable: the same lockstep run at 8192, one CPU-EP oracle chain.
3. The arena overrun is still **dropped by the shader guard, not refused**. Carried.

### For Mouse
**An int8 KV kernel will introduce a specialisation constant** — cache element width and scale
group size are pipeline-build-time facts, not push constants — so `gqa_f16` acquires a non-empty
selector list and `shaders_dispatched_spec_digest` must move. It is the first case in this project
where the specialisation is **load-bearing for correctness**: the wrong group size dequantises with
the wrong stride and returns plausible wrong numbers. Filed, not left to be discovered.

Decision records: `switch-int8-kv-error-budget.md`, `switch-int8-kv-residual-saturates.md`,
`switch-kv-ledger-ratios-not-reproducible.md`, `switch-int8-kv-spec-constant-for-mouse.md`.
Artifacts: `bench/results/kv_int8_budget-dev{0,1}.json`,
`bench/results/kv_int8_depth-i{8,8,4}-{per_head,per_block32,per_block32}-n256.json`,
`bench/results/kv-int8-budget-prediction.md` (written before the first run).
