# Tank (Runtime-FFI) — history.md

## Learnings

### [SUMMARY] Sessions 1–13b: crate foundation, ORT bindings, logging crash, allocator, execution verification (2026-07-28–2026-07-30)

**Sessions 1–7 (archived):** Crate structure (`ort-ep-vulkan`, cdylib). ORT C API bindings via bindgen. Three-number version negotiation: EXPECTED 28 / MIN 24 / negotiated 28. `logging::forward_to_ort` null-pointer crash fixed (ORT annotates `file_path` as `_In_z_` and dereferences unconditionally). `tests/cdylib_load.rs` dlopens shipped cdylib and resolves exports. `tests/portability.rs` added after `ort::wchar_t` broke Linux lane. `cargo ci` command added with edition preflight.

**Session 9 (2026-07-29T10:50:02-07:00) — Compile/Compute seam:**
`Compile`→`OrtNodeComputeInfo`→`dispatch_ort` wire complete. Inputs/outputs from fused node, not subgraph body. `ep.rs` imports via `engine.rs` re-export (layering rule 4.3). `Compute` must return real `OrtStatus` on failure — null means success to ORT.

**Session 10 (2026-07-29T20:26:56-07:00) — CI and counters:**
`cargo ci` edition preflight: refuses rustfmt that doesn't know the crate's edition. Execution counters (`rust/src/counters.rs`): six relaxed atomics, always on, `ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` env var, written on first dispatch and at teardown. `epctl --check-counters`: exit 0/1/3 (1≠3 distinction is load-bearing for CI). `glslc` discovery now searches `C:\VulkanSDK\<version>\Bin\` as fallback.

**Session 11 (2026-07-29) — allocator and lavapipe crash:**
Real allocator: 64 GiB VA reservation per device (`MEM_RESERVE`/`PROT_NONE`), `BTreeMap<usize, Span>`. Lavapipe crash diagnosed: OOB storage buffer access = real host fault; `robustBufferAccess` not enabled. Lavapipe `subgroup_size=8`, `maxComputeSharedMemorySize=32 KiB`.

**Session 12 (2026-07-29) — device memory and probe failure:**
Device memory behind `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` (default off). `transfer.rs` (724 lines): `CanCopy`, `CopyTensors`, `Release`. `EpDevice_AddAllocatorInfo`: do NOT release `MemoryInfo` after success (bounded intentional leak; ORT retains the pointer). `probe_allocator.py` was a false-green machine — must check counters file for `dispatches_executed > 0`.

**Session 13 (2026-07-30) — interior pointer verification:**
ORT's planner does NOT engage on run 1 (records the pattern, hands back sub-ranges from run 2 onward). 52 interior pointers observed across 5 runs, identical on Intel and NVIDIA, all within span, `pointers_in_guard_band=0`. Every earlier "0 interior" probe was pointed at the wrong moment — instrument failure, not negative result. `allocator::ledger` classifies every pointer by `LookupError` taxonomy.

**Session 13b — positive controls and honest scope:**
Quarantine detector positive-control present (`the_quarantine_detector_fires_when_a_stale_handle_is_presented`). `pointers_use_after_free=0` in real sessions is worth nothing alone — detector proven able to fire, not exercised by ORT. `probe_planner.py` runs session in child process. Phi-3.5 still claims 0 nodes as of this session (blocked on Switch's runtime extents — now landed). CI contract: `pointers_in_guard_band > 0` is a hard assertion.

**Current state:**
- `cargo ci` — green, 300 tests.
- Allocator ready, interior-pointer observation complete.
- Device memory blocked on at least one claimed node (now unblocked — Switch's extents landed 161 claims).
- D-T51: quarantine detector not yet exercised by a real ORT allocation pattern.
- Standing: `ort::wchar_t` Windows-only bindgen type; `tests/portability.rs` guards Linux lane.
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
Memory-pattern planner does not engage on run 1. From run 2 onward hands back interior pointers. 52 observed, identical on both devices, all within span, `pointers_in_guard_band=0`. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON. `dispatches_executed > 0` is the only reliable indicator.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards pNext chain. Rebind, never discard. Root cause of LVP2, `subgroup_size=0`, ReBAR UMA misclassification.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE (Niobe — critical)
`(claimed_op_coverage, island_count, largest_island_flops)` per producer at version. Portability floor = §7.2. `SUBGROUP_SIZE_IS_GUARANTEED=False`.
