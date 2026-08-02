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


<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- **📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)** — ### Worktree layout and inbox portability constraint The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree.
- **Session 10 — 2026-07-29T20:26:56-07:00 — CI has never executed a claimed node** — **The task was "make CI prove it", and the first thing I found was that CI cannot currently prove anything: both lanes crash.** Run `30510593046`, eight consecutive failures.
- **Session 11 — 2026-07-29 — the allocator, and a crash that was mine** — **What I built.** `src/allocator.rs` — a real `OrtAllocator` over a per-device reserved virtual-address arena.
- **Session 12 — 2026-07-29 — device memory becomes real, and a probe that lied to me** — **Worktree.** Moved to `C:\Users\justinchu\dev\ep-vulkan-tank` on `squad/tank`.
- **Session 13 — 2026-07-30 — the verification I could not get was an instrument problem** — **Worktree** `C:\Users\justinchu\dev\ep-vulkan-tank`, branch `squad/tank`, rebased on `main`.
- **Session 13b — 2026-07-30 — every mechanism casts a shadow** — The coordinator ruled that decision records must be written into the integration tree's inbox, because `.squad/decisions/inbox/` is gitignored and a record written in a worktree is invisible to everyone.
- **2026-08-01 — Tank — the broken-commitment WARN, through ORT's own sink, with a control that bites** — Ruling 2 specified the mechanism; my job was to build it and then to make it *falsifiable*.
- **2026-08-01 addendum — the load was misattributed, and my evidence is unaffected by the correction** — The coordinator withdrew his attribution of the machine load: it is a second development project of Justin's running CPU **and GPU** tests, not squad orchestration.
- **STOP POINT 2026-08-01T11:39 — read this first if you are resuming as Tank with no memory** — **Everything is committed.** Worktree `C:\Users\justinchu\dev\ep-vulkan-tank`, branch `squad/tank`, commit `bce87cd`, on top of `main` at `17c2fab`.
