# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Performance — benchmarks, profiling, regression tracking
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-29T09:00:39-07:00 — full session details in decisions.md -->

### [SUMMARY] Turns 1–2: tracing integration, benchmark harness, first hardware facts (2026-07-28–2026-07-29)

**Foundational decisions from team (rounds 1–2):**
- `largest_island_flops` is the metric of record (NOT `claimed_node_fraction`). Required per run: `island_count`, `largest_island_nodes`, `largest_island_flops`, `boundary_bytes_per_inference`, `boundary_time_fraction`, `declined_nodes` histogram.
- OQ-12 pass bar: ≥1.5× over the device's own ORT CPU EP on a GEMM-anchored subgraph, zero numerical failures.
- MVS constants are provisional (`SAFETY=3.0`, `node_count≥4`, `64 KiB` floor) — must be re-derived from M2 measurements via `TransferModel::fit`.
- Record-once / replay-many: benchmarks must distinguish first-inference latency (recording path) from steady-state (replay path).

**`onnx-runtime-tracer` adoption (D-N1):** Pin `0.1.0-dev.5, default-features = false`. Absolute UNIX-microsecond epoch — plugin cdylib spans overlay host timeline with no offset negotiation. Re-verify before bumping. `default-features = false` keeps prost/protobuf out.

**Seven Vulkan span phases (D-N2):** `compile`, `prepack`, `upload`, `record`, `submit`, `fence_wait`, `readback`. `Phase::Submit::observes_gpu_work() == false` (unit test asserts this). `fence_wait` is UPPER BOUND. `vkQueueSubmit` wall time measures almost nothing — driver bookkeeping only.

**GPU timing rules (D-N4/D-N5):**
- `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` required to activate GPU spans.
- `timestampValidBits == 0` = no timestamps on that queue family — skip entirely, record reason, never emit zeros.
- `timestampPeriod` is NOT 1.0 on real hardware (Intel Iris Xe = 52.08 ns/tick, AMD ~20–83). Assuming 1.0 = up to 80× under-reporting. Conversion owned by `trace.rs`; Switch hands over raw ticks via `GpuTimestampReport`.
- Never call `vkGetQueryPoolResults` with `WAIT_BIT` on possibly-in-flight submissions — stalls host. Read after fence signals; prefer `WITH_AVAILABILITY`.
- `VK_EXT_calibrated_timestamps`: returns `maxDeviation`, pass as `anchor_uncertainty_us`. Fallback: half-round-trip bracket.
- GPU spans on synthetic lane (`0x7600_0000 + queue_family`), never the submitting thread's.
- `trace.rs` names NO Vulkan type (enforced by layering lint). Switch owns the call sites.

**Benchmark harness (D-N7–D-N11):** Claim gate + noise gate (MAD/IQR/p05/p95, robust RSD). Environment stamped on every result. OQ-12 anchor: `matmulnbits_q4_b32_K4096_N4096`. `TransferModel` calibration script. `bench/test_harness.py` tests the gates. 27 tests at end of turn 1.

**Local GPU facts (turn 2, 2026-07-29):**

| | Intel Iris Xe | NVIDIA RTX 4060 Laptop |
|---|---|---|
| `timestampPeriod` | 52.0833 ns/tick | 1.0 ns/tick |
| `timestampValidBits` | 36 | 64 |
| shared memory | 32 KiB | 48 KiB |
| memory | 1 heap, all DEVICE_LOCAL (UMA) | discrete + host |
| API / driver | 1.4.309 / 101.6737 | 1.4.325 / 591.55 |

Both expose `VK_EXT_calibrated_timestamps` and `VK_EXT_host_query_reset`. Intel = spec-conformance oracle. SDK at `C:\VulkanSDK\1.4.350.0` — not on default PATH.

**The 1.70× that wasn't:** ORT 1.27 was installed; `register_execution_provider_library` printed a rejection but didn't raise; session ran on CPU EP; `matmulnbits_q4_b32_K4096_N4096` reported 1.70× on what appeared to be a discrete GPU — entirely fictional, rsd 38–63%. Lesson: check the *effect* (which EP actually ran), not the return of the registration call. `MIN_ORT = (1, 28)` now prevents the column from existing.

**UMA classifier bug caught:** "Does any type carry DEVICE_LOCAL|HOST_VISIBLE" reports a discrete resizable-BAR window as UMA. Correct: UMA iff no heap lacks DEVICE_LOCAL. Rule: be suspicious of a classifier whose answer lets you delete work.

**rustfmt note:** `rustfmt --edition 2021` silently no-ops on this edition-2024 crate. Use `cargo fmt --all`. The xtask `cargo ci` does this correctly.

**Status as of turn 2:** No performance numbers. One `add_f32_dispatches_end_to_end` dispatched on NVIDIA RTX 4060 — a smoke test, not a speedup. A result obtained only on this desk is not a result this project has.

---

## Cross-agent context appended (2026-07-29T09:00:39-07:00) — first-hardware round

📌 **Local GPU facts (2026-07-29):** **Intel Iris Xe Graphics** (Vulkan 1.4.309, UMA, 32 KiB shared) and **NVIDIA GeForce RTX 4060 Laptop GPU** (Vulkan 1.4.325, discrete, 48 KiB shared). Both pass §7.2 gate. Intel is the stricter implementation — treat as spec-conformance oracle. Vulkan SDK at `C:\VulkanSDK\1.4.350.0` — not on default PATH.

📌 **T3 sequencing: `ai.onnx::Attention` first, GQA second (2026-07-29, Morpheus D23):** Mouse's standard-domain rows (`ai.onnx::Attention`, `RMSNormalization`, `RotaryEmbedding`) are registered and constrain Niobe's benchmark anchor — `matmulnbits_q4_b32_K4096_N4096` is the OQ-12 anchor op, not `ai.onnx::Attention`. Niobe's OQ-12 pass bar: ≥1.5× over the device's own CPU EP measured at the same device.

📌 **`bind_aliased_output` seam (2026-07-29, Mouse + Switch outstanding):** GQA implementation requires an output aliasing mechanism. Niobe needs this confirmed before designing the GQA benchmark timing breakdown (the KV outputs are aliased to the KV inputs; separate upload/readback spans are not applicable).

📌 **`largest_island_flops` metric (2026-07-29, Morpheus D21):** The metric of record for performance progress is largest fused region compute volume (`largest_island_flops`), not claim rate or `claimed_node_fraction`. Every benchmark run must report this alongside wall time. Niobe's benchmark harness already tracks this; report its value in all perf-tracking artefacts.

📌 **No performance numbers exist yet (2026-07-29):** A shader dispatched on hardware this round (one `add_f32_dispatches_end_to_end` on NVIDIA RTX 4060). No MatMulNBits, no attention variant, no profiling-JSON timing artefact has been collected. The 1.70× figure referenced in coordinator discussions was a CPU-only measurement artefact — it is not a GPU speedup claim.
---

## 2026-07-29 — turn 3: producer provenance, and a bug report for a bug I had already fixed

### The clippy error routed to me was already fixed

`trace.rs:1412 assert!(GPU_LANE_BASE > 1 << 24)` — I hit that lint in turn 1 and it has been
`const { assert!(..) }` ever since. Clippy emits nothing for `trace.rs`. The red Mouse saw was
Switch's `vk/instance.rs` and `vk/dispatch_integration.rs`, mid-flight; it cleared as he landed.

Lesson: when the tree is red across several people's files at once, "clippy said X near your
file" is a weak signal. Check `--message-format=short` filtered to your own paths before touching
anything. I nearly edited a correct line.

Also worth knowing: `cargo ci` results are **not stable while other agents are mid-edit**. I got
green, then 2 lib-test failures, then green again within ten minutes, without touching a Rust
file. Confirm with a re-run before believing either colour.

### The finding that actually mattered

Mouse read Justin's `onnx-genai-models` (`mobius`) and found it emits a *different op set* from
the ORT GenAI builder for the same architecture: `ai.onnx::Attention`@23, `RMSNormalization`,
`RotaryEmbedding` versus the `com.microsoft` contrib equivalents. `MatMulNBits` is the only op
both agree on. His generalisation is `OP_COVERAGE.md` §4.18: **op coverage is relative to a
producer, not to a model architecture.**

The version of that I have to hold: **a benchmark artefact is relative to its producer too, and
it is easier to be fooled by**, because a timing has no shape to disagree about. A wrong-shaped
graph fails a correctness test loudly. A wrong-provenance graph produces a perfectly plausible
millisecond figure with a model's name on it.

Two consequences I had not thought about before writing it down:

1. The two graphs **partition differently**. "The EP claimed 40% of the graph" is a statement
   about the exporter as much as about us.
2. `largest_island_flops` — our metric of record — is computed on a specific graph. An island
   size without a producer is not reproducible.

### What I built

`bench/producers.py`: `Producer{name, kind, version, digest, opsets, model_family}` with
fingerprint `name@version#digest`; `digest` is a SHA-256 of the builder's own source, so a silent
edit to the builder shows up as a different producer rather than as a performance change. Same
instinct as putting the driver version in the device fingerprint.

Two refusals, both structural rather than conventional:

* **A case cannot be named after a model family its producer did not export.** Enforced in
  `Case.__post_init__`, so `qwen3_decoder_layer` built from hand-written ops cannot be
  *constructed*. Earning the label needs kind=`model` **and** a family **and** a version.
* **`compare.py` refuses across producers** — exit 2, no table, `--cross-producer-study` to
  relabel with no verdict. Unrecorded producer refuses too.

Verified both on real result files, not just in unit tests. 39 bench tests pass (was 27).

### The design rule I want to remember

Every gate I have added this week has the same shape: *the wrong answer and the right answer look
equally reasonable in a table, so make the wrong one unconstructible.* Phase::Submit not observing
GPU work. UMA being "no heap lacks DEVICE_LOCAL" rather than the obvious flag test. Cross-device
comparison as exit 2 rather than a banner. Now: a model-family name being a claim that must be
earned. When I next add a metric, the question to ask first is not "is this right?" but "what is
the plausible misreading, and can I make it impossible?"

### Green at end of turn

`cargo ci` — 300 tests, fmt + clippy clean. `pytest bench/` — 39 passed.
Still no performance number. No kernel has executed.

---

## 2026-07-29 — turn 4: the portability floor is not the smaller GPU on my desk

### The mistake I was about to make

Asked to make performance numbers portable, my first instinct was to treat **32 KiB** as the
shared-memory budget, because that is what the *smaller* of the two local GPUs has. Wrong, and
wrong in the flattering direction: `DESIGN.md` §7.2 R4 admits devices at **16 KiB**, and §7.0 says
shortfalls degrade op coverage, not device availability — so a 16 KiB device is one we *promised*
to run on. The floor is a decision this project already made, not a property of my desk.

The Iris Xe is our proxy for the mobile **memory model** (UMA, as Adreno and Mali are). It is not
a proxy for the mobile **shared-memory budget**. Those are different things and the same device
being both tempting answers is exactly why this needed writing down.

Related coincidence, now a named constant rather than an assumption: **both local GPUs report
`subgroupSize == 32`.** Vulkan 1.1 guarantees subgroup `BASIC` in compute and nothing about the
size. Two devices agreeing is the strongest possible invitation to bake in a 32 and pass every
local test.

### The general shape, now four for four

Every gate I have built this week is the same move: *the wrong answer and the right answer look
equally reasonable in a table, so make the wrong one unconstructible.*

1. `Phase::Submit` does not observe GPU work.
2. UMA is "no heap lacks DEVICE_LOCAL", not the obvious flag test (the BAR window).
3. Cross-device and cross-producer comparison exit 2 rather than warn.
4. A configuration above the admission floor is `needs-fallback`, not "fine, it ran here".

And the same escape hatch each time: **`unknown` is never `equal` and never `fine`.** Today every
portability verdict is `unknown`, and the table says so.

### The EP loaded, and lied to me within about ninety seconds

ORT 1.28 got installed mid-turn, so `MIN_ORT` passed and the plugin loaded for the first time
ever: registration succeeded, both devices enumerated, claim predicates ran on real graphs. Every
op declined honestly (`Add`: "its compute shader compiles but has never executed on a device, so
claiming it would be a bet").

Which means every "vulkan" column is still the CPU EP — and `add_fp32_4096x1024` promptly reported
0.858 ms "vulkan" vs 1.247 ms "cpu" = **1.45×**. Second fabricated speedup of the day, second one
caught by the claim gate. §5.1 was 1.70× via an unloadable EP; this one is via a loadable EP that
declines everything. **Two different routes to the same false number.** The lesson is that the
claim gate — not the version gate, not the device gate — is the one doing the real work, because
it checks the *effect* rather than any precondition.

Also: the producer digest changed mid-session (`120b3983c341` → `a8c67afee5f9`) because someone
edited `tests/ops/_models.py`. That is the feature working. If I had compared across that edit
without the digest, a graph change would have looked like a performance change.

### Green at end of turn

`cargo ci` — 300 tests, fmt + clippy clean. `pytest bench/` — 50 passed.
Still no performance number. No shader has executed.
