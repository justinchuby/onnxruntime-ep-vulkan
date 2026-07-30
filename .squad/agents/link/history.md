# Link (Platform-Support) — history.md

## Learnings

### [SUMMARY] Sessions 1–6: extension availability, capability set, CI root causes, hardware matrix, LVP2 retraction (2026-07-28–2026-07-30)

**Sessions 1–2 (archived):** OQ-1 measured: `VK_KHR_synchronization2` at 68.57% on Android, `VK_EXT_subgroup_size_control` at 85.88%. Option B (Khronos layer shim) rejected; retained only as optional integrator deployment note. Wgpu/Dawn/Godot claim of Vulkan 1.2 requirement found false — these use extensions, not 1.2 core. OQ-12 hardware experiment defined.

**Session 3 (archived) — CI root causes:**
Windows CI: `VK_ICD_FILENAMES` env var ignored under elevation — must register ICD in `HKLM\SOFTWARE\Khronos\Vulkan\Drivers`. Linux failure was a compile error (not a Vulkan problem); `glslc` arrives via LunarG apt repo.

**Session 4 (2026-07-29T09:19:35-07:00) — hardware matrix:**
First execution-derived hardware data. Both CI lanes CI-VERIFIED (Linux llvmpipe, Windows lavapipe). Both local GPUs local-dev-verified. Intel Iris Xe is the spec-conformance oracle — do not special-case Intel failures. UMA is a first-class platform column (Intel Iris Xe and mobile are UMA; results on Iris Xe are a closer mobile proxy for memory model than RTX 4060). Memory model column added to platform matrix. LVP2 initially recorded (lavapipe `supportedStages=0`) — but see retraction below.

**Session 5 (2026-07-29T09:39:59-07:00) — cross-platform standing directive:**
Cross-platform generality is structural, not a review step. Derive from reported limits. Every `cfg` is a portability hazard (`tests/portability.rs`). Intel is the spec oracle; Intel failures predict MoltenVK failures. No required extensions per §7.2.

**Session 6 (2026-07-29T20:26:56-07:00) — LVP2 RETRACTED:**
LVP2 retracted. Mesa 23.2.1 lavapipe supports subgroup BASIC+ARITHMETIC+BALLOT+SHUFFLE+SHUFFLE_RELATIVE+QUAD in compute, `subgroup_size=8`. The original `supportedStages=0` was the discarded `push_next` chain — our bug, not a device fact. PLATFORMS.md LVP2 entry updated to retraction notice. Lavapipe is UMA (`is_uma=true`, single DEVICE_LOCAL heap), `maxComputeSharedMemorySize=32 KiB`. CI now exercises the mobile-warp path (lavapipe `subgroup_size=8` vs local 32).

**Current state:**
PLATFORMS.md current. Both CI lanes verified. Hardware matrix up to date. M0 criterion 9 met (LVP2 retracted). OQ-12 experiment design defined; hardware borrow needed after M0 for Adreno/Mali. Intel Iris Xe is the spec oracle; every Intel failure is a real portability signal.
---

## 📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)

### Worktree layout and inbox portability constraint
The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree. `.squad/decisions/inbox/` is **gitignored** — records written in a worktree do NOT travel with the branch. The inbox in `main` is authoritative.

### Vulkan SDK path
`C:\VulkanSDK\1.4.350.0` — installed but **not on the default PATH**. `glslc` discovery must search this path; `VULKAN_SDK` env var is the canonical pointer.

### Local hardware — both GPUs pass the §7.2 gate
- Intel Iris Xe: Vulkan 1.4.309, UMA, `subgroup_size=32`, 32 KiB shared. Spec-conformance oracle. Do not special-case Intel.
- RTX 4060 Laptop: Vulkan 1.4.325, discrete, `subgroup_size=32`, 48 KiB shared.
- Lavapipe (CI): `subgroup_size=8`, 32 KiB shared, `is_uma=true`. CI exercises the mobile-warp path. LVP2 retracted.

### ORT's planner hands back interior pointers from run 2 onward
Memory-pattern planner does not engage on run 1. From run 2 onward hands back interior pointers. 52 observed, `pointers_in_guard_band=0`. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON. `dispatches_executed > 0` is the only reliable indicator.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards pNext chain. Rebind, never discard.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE
`(claimed_op_coverage, island_count, largest_island_flops)` per producer at version. Portability floor = §7.2. `SUBGROUP_SIZE_IS_GUARANTEED=False`.

---

## Session 8 (2026-07-30T08:21-07:00) — Gate artifact design, is_uma verification, subgroup red instrument, lane classification

### Context received from coordinator
- DESIGN.md §8.9 ruling (b7c2305): `operational` vs `green` distinction. A lane that runs is operational; a lane with a gate artifact and MATCH verdict is green. UNMEASURED is the default — not PASS, not FAIL. 
- Tank's single-run blindness finding: ORT arena reuse starts from run 2; every `tests/ops/` helper runs exactly once → 196-pass count is blind to interior-pointer bugs.
- Rai 🔴 on the class: silently-wrong output at any layer, architecture-level concern.
- Fact Checker OQ-12 figure correction: ~67.33% have sync2 (gap ~32.67%) as of 2026-07-30 vs 68.57% (31.43%) on 2026-07-28. Figure is moving.

### Decisions made this session

**1. Gate artifact for criterion 10:** `gate_chain_fp32` — a 2-node `Add → Relu` fp32 chain on [256] tensors. Meets Morpheus's three criteria: claims non-zero nodes, has a 2-node island, exercises the fp32 proof keys for both template families (ew_binary, ew_unary). fp16 artifact deferred until `storageBuffer16BitAccess` confirmed on lavapipe. UNMEASURED by default — verdict file written before session open, overwritten with MATCH/DIVERGENT after comparison. Trinity implements mechanism; Link owns artifact spec.

**2. UNMEASURED-by-default enforcement:** verdict file created at run start. If process exits before comparison → UNMEASURED remains. CI gate check (`epctl --check-verdict` or equivalent) exits non-zero on UNMEASURED unless `--allow-unmeasured` passed explicitly. `--allow-unmeasured` must not appear in CI step definitions. Coordinate with Trinity on file format and vocab (same as model_output_equivalence: MATCH/DIVERGENT/UNMEASURED).

**3. Subgroup-32 red instrument — closed:** Executing on `subgroup_size = 8` IS sufficient by construction. The falsifier is the numerical correctness suite (test_elementwise.py + test_matmulnbits.py) running on lavapipe. A baked-32 shader produces wrong reduction outputs on subgroup_size=8, diverging from CPU reference, failing assert_matches_cpu. The risk item is NOT open — instrument exists. New rule: new shader templates must have a lavapipe numerical test before the op is moved to Ready.

**4. is_uma predicate — verified correct:** `is_uma_memory` in caps.rs uses "every heap is DEVICE_LOCAL" — the corrected predicate. Unit tests cover the ReBAR false-positive case (two heaps, one without DEVICE_LOCAL → false). lavapipe has one heap (DEVICE_LOCAL|HOST_VISIBLE); the corrected predicate returns true for the right reason. Not the old bug agreeing by coincidence.

**5. Single-run blindness documented:** Added to §7.4.2 "What GPU-less CI does NOT cover". The 196-pass count is not evidence about multi-run behaviour. The instrument for the multi-run failure class is probe_run2.py (local-dev only, not yet in CI).

**6. OQ-12 figure corrected:** Updated to ~32.67% (2026-07-30 Fact Checker revision) in §7.7.5 and §10.0.3 table. Figure now carries date and error direction in all primary references. The figure is a ceiling (some gap devices fail §7.2) and a floor (gpuinfo.org under-represents budget hardware). 

### PLATFORMS.md sections updated this session
- §7.4.2: single-run blindness note added
- §7.7.5: OQ-12 figure updated to ~32.67% with date and provenance reference
- §7.7.6 (new): `operational` vs `green` lane classification
- §7.8 (new): gate artifact design for criterion 10, verdict mechanism, coordination with Trinity
- §7.9 (new): is_uma predicate verification — corrected predicate confirmed, unit tests cited
- §7.10 (new): subgroup-32 red instrument — YES, instrument exists by construction, risk item closed, maintenance rule stated
- §10.0.3 table: Android figure updated to ~67.33%/~32.67% with date

### What was done
Full build and test run on WSL2 Ubuntu 24.04 (Mesa 25.2.8 / LLVM 20.1.2, lavapipe 1.4.318). First execution of a claimed node end-to-end on a Linux Vulkan stack with lavapipe.

### Build chain findings
- `glslc` is in Ubuntu 24.04 repos directly (package `glslc`, v2023.8). Ubuntu 22.04 CI lane must still use LunarG `shaderc` apt repo — the two paths diverge.
- Vendored ORT headers at `third_party/onnxruntime/include/` work without setting `ORT_INCLUDE_DIR`. Do not set it to a missing path.
- `CARGO_TARGET_DIR=/root/ep-build` required to avoid systemd private-tmp recycling between WSL invocations.
- `PATH` must be set explicitly in WSL root bash subshells (it is empty otherwise → "linker `cc` not found").
- `sudo` requires password for `justinchu` in this WSL install. Use `wsl -d Ubuntu -u root` for all privileged operations.
- Build: CLEAN. Artifacts: `libonnxruntime_vulkan_ep.so` (1.78 MB), `epctl` (904 KB).

### Gate and capability results (lavapipe, new)
All R1–R6 PASS. Key new data:
- `subgroup_size = 8` (confirmed — prediction was correct; this is a real portability surface)
- `is_uma = true`; `maxComputeSharedMemorySize = 32 KiB`; `timestamp_period_ns = 1.0`; `timestamp_valid_bits = 64`
- `apiVersion = 1.4.318`
- See §7.5 of PLATFORMS.md for full three-way diff (lavapipe vs Intel Iris Xe vs RTX 4060).

### Barrier path: sync2 (expected)
lavapipe Vulkan 1.4 → sync2 is core → `Barriers::select` → `Sync2Backend::Core`. Probe file confirms `"sync2"`. Forced-legacy path validated by parity suite.

### Test suite results (lavapipe)
- M0 canonical: `test_binary_elementwise[Add-fp32]` PASSED ✅
- `test_barrier_parity.py`: 58 passed / 0 failed / 28 skipped — **third independent implementation** (after Intel Iris Xe and RTX 4060) confirming barrier parity bit-exactly.
- Full suite: 196 passed / 34 failed (all staged ops, platform-independent) / 32 skipped.
- Provider assertion confirmed: no silent CPU fallbacks.

### Subgroup size audit
Zero of 168+ compiled shader variants use subgroup intrinsics. All use shared-memory tree reductions. `q_gemv.comp` lines 9–12 document this explicitly. The subgroup-size-8 difference has zero affected variants today. Future shaders must be authored to handle `subgroupSize ∈ [4, 128]`.

### OQ-12 constraint
lavapipe results DO NOT provide Android evidence. UMA topology matches, but ISA, driver bugs, and command-submission model are entirely different. The 31.43% Android usability figure remains fully unverified.

### PLATFORMS.md sections updated this session
- §5: WSL Ubuntu 24.04 row added
- §7.4.2: WSL local-dev lane row added; lanes table restructured with claimed-node execution column
- §7.5 (new): Three-way capability diff table
- §7.6 (new): Subgroup-size audit, zero variants affected
- §7.7 (new): Lavapipe execution record — build, gate, barrier path, test results, OQ-12 scope
- §9: lavapipe barrier parity result added
📌 Team update (2026-07-30T05:48:29-07:00): A green suite has been shown not to imply a correct model. Phi-3.5: 161 MatMulNBits dispatched, compute_failures:0, entire suite green — vk logits all-zero (argmax 0 vs CPU argmax 30751). R9 (Morpheus): for every claim, name the instrument that would go red if the claim were false; if none, the claim is UNMEASURED. model_output_equivalence verdict required alongside all counter summaries; default UNMEASURED. Any comparison must first assert EP_NAME in session.get_providers() before calling sess.run() — failure to do so compares CPU to CPU and reports agreement. Coordinator's own first comparison reported bit-identical on both devices due to this exact error. Trinity has landed xfail(strict=True) correctness gate. M0 criterion 10 added (NOT MET: DIVERGENT). Criteria 2, 4, 5 reopened. — decided by Morpheus, Trinity, Switch, Mouse; coordinator-verified.
