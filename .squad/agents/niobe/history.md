# Niobe (Performance) — history.md

## Learnings

### [SUMMARY] Sessions 1–4: tracing, harness, producer provenance, portability, two fabricated speedups caught (2026-07-28–2026-07-30)

**Sessions 1–3 (archived):** `onnx-runtime-tracer` integrated. Vulkan-specific span phases defined. GPU timestamp-query requirements routed to Switch (`DeviceInfo` must carry `timestamp_period_ns`, `timestamp_valid_bits`). `bench/` and `docs/PERF.md` built with no performance numbers (no kernel executed at that point). OQ-12 anchor: `matmulnbits_q4_b32_K4096_N4096`, ≥1.5× bar measured at that case only. `bench/transfer_calibration.py` sweeps doubling byte staircase, fits fixed+bandwidth model, prints paste-ready Rust literal. MVS constants replaced per device via review. `bench/environment.py` stamps OS/CPU/ORT/EP/Vulkan/env-vars at run start.

**Session 4 — portability envelope (2026-07-30):**
- **D-N24** — Portability floor is §7.2 (16 KiB shared, 256 invocations), not the smaller local GPU. Iris Xe is UMA proxy for mobile memory model, not for mobile shared-memory budget. A 32 KiB tile passing on Iris Xe is not portability evidence.
- **D-N25** — `bench/portability.py`: `evaluate(Configuration) → Verdict` in {`portable`, `needs-fallback`, `unknown`}; `quotable_as_ep_behaviour` true only for `portable`. Every row is `unknown` today (engine does not report tile shape or workgroup size). `fits_device(config, shared, invocations)` uses reported limits, not constants.
- **D-N26** — UMA and discrete transfer models may not be blended. `portability.transfer_model_merge_refusal()` closes the obvious path.
- **D-N27** — Routing to Switch: engine must report `tile_config`, workgroup size, shared-memory bytes, and memory path (UMA mapped write vs staging copy). Until then, portability column is honest but empty.
- **D-N28** — Two fabricated speedups caught: 1.70× (ORT 1.27 prints failure without raising; result was CPU vs CPU); 1.45× (EP loads under ORT 1.28, declines everything — all "vulkan" columns are CPU EP). Neither claimed. `bench/README.md` records both as "the 1.70× that wasn't" and "the 1.45× that wasn't."

**Current state:**
- `pytest bench/` — 50 passed (11 new portability tests).
- No real Vulkan bench row yet — no kernel has executed through the bench harness.
- All timing rows are CPU-only, clearly labelled.
- First quotable Vulkan row: after Switch reports tile_config + workgroup size in `DeviceInfo`, and `dispatches_executed > 0` in counters file.
- `SUBGROUP_SIZE_IS_GUARANTEED=False` constant present — both local GPUs happen to report 32, not a guarantee.
- Standing rule: metric of record is the triple `(claimed_coverage, island_count, largest_island_flops)`, never any component alone.
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

## Turn 5 — 2026-07-30 — the first honest measurement (Phi-3.5, both devices)

### Coverage figures go stale fast — read the counter, never the briefing
I was told 161 claimed nodes. The run reported **257 claimed of 363 probed**. Always re-read
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` in the same run that produces the number.

### Island count is `subgraphs_live`, not the claimed-node count
They happened to be equal (257 == 257). Equality is a *coincidence to be falsified*, not a
definition. The falsifier: `compute_calls == subgraphs_live x inferences` — 7967 == 257 x 31
exactly, on both devices. Integer equality, no tolerance, free.

### The Intel iGPU gets SLOWER with warmup — a "take the min" convention would lie 4x
Iris Xe per-inference: 724 -> 695 -> 903 -> 1447 -> 2080 -> 2669 -> flat ~2790 ms. Monotone ramp
into steady state, not out of it. Added `stats.drift()` (first/second-half median ratio +
monotone fraction) and raised phi35 warmup default to 10. Spread cannot tell "noisy but stable"
from "moving steadily"; they demand opposite responses.

### Within-run spread is not run-to-run spread — carry both
With warmup 10 the within-run rsd is 1.7% (Intel) / 2.6% (NVIDIA), yet two whole runs minutes
apart differed by 28%. Added `--repeats` (default 3) launching whole processes.

### The CPU baseline is not a constant
218 ms then 665 ms for the same CPU-only session, minutes apart — page-cache pressure after a
2.2 GB model. Hence: each device's vulkan-vs-cpu delta must be measured back-to-back in ONE
process; `baseline_disagreement()` fires above 2x between workers.

### The counters file in this path carries no `alloc_*` keys
Only `abi_version, compile_calls, subgraphs_live, subgraphs_stub, compute_calls,
compute_failures, dispatches_executed`. So the staging label derives from **configuration**
(`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` unset => staging-bound), with observation as a weaker
second ground. "unknown" would have invited the reader to assume the number was general.

### Timestamp verdict: inputs VERIFIED, arithmetic VERIFIED, end-to-end UNMEASURED
`bench/timestamp_audit.py` cross-checks `epctl --probe-loader` against `vulkaninfoSDK`. Agree on
both devices: Intel 52.0833 ns / 36 valid bits / UMA true (wrap 3579 s ~= 0.99 h);
NVIDIA 1.0 / 64 / UMA false. **No `VkQueryPool` exists yet**, so nothing end-to-end is measured.
Crucially: lavapipe and NVIDIA both report 1.0/64, so dropping BOTH the period scale and the
mask is green on the discrete GPU and green in CI while under-reporting every Intel duration by
52x. **The Iris Xe is the only instrument on this desk for that bug class, and CI has none.**

### The tracer is written and env-wired but NOT called
Verified empirically, not by reading: with `ONNXRUNTIME_EP_VULKAN_TRACE` and `TRACE_GPU=1` set,
a run that executed 257 islands over four inferences produced **no trace file**. Report adoption
as four facts (pinned / written / env-wired / invoked), never as one word.

### The number: we are 8-12x SLOWER, and that is the useful result
Intel 2790.7 vs 229.8 ms (12.1x); NVIDIA 1465.9 vs 185.9 ms (7.9x). Both `MATCH`, staging-bound.
Per island: >= 9.96 ms (Intel) / >= 4.98 ms (NVIDIA) — a lower bound, since the host delta nets
boundary cost against the GEMV saved.
**For Mouse:** fewer islands beats faster kernels by an order of magnitude right now.
**For Switch:** Intel costs ~2x per island *with no bus to cross* => argues for a fixed
per-submission cost (submit-and-wait per island), not per-boundary PCIe transfer. Hypothesis;
the §3 timestamps decide it.

### Tooling
- `.squad/decisions/inbox/` is **gitignored inside a worktree** — decision records must be
  written into the integration tree's inbox or they never reach the Scribe. `cargo ci` says so;
  git never will.
- Crate edition is 2024: `rustfmt --edition 2021 <file>` silently no-ops. Use `cargo fmt --all`.
- PowerShell `Select-String` piped after a native command can return nothing; redirect to a file.

---

📌 Team update (2026-07-30T19:05:03-07:00) — Scribe

Two findings apply to every agent on the team:

**(a) A mechanism that exists in a file but not in a call graph is indistinguishable from
one that does not exist.**  Verification by reading is insufficient.  Verify by running.
Five such mechanisms surfaced in this single batch: partition.rs, the GPU tracer,
model_output_equivalence, compute_failures, and should_claim_island.  In every
case the code was correct; the wiring was absent; the absence was invisible to review.

**(b) 85.9% of inference wall-time involves no GPU work** (recording 68.3%, fence-wait
idle 16.3%, submit 0.3%; GPU kernels 14.1%).  Optimising GPU kernels before the
command-buffer recording bottleneck is resolved is low-leverage.  Align work priorities
accordingly.

