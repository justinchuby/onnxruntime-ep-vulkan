# Switch (Vulkan-Compute) — history.md

<!-- CONDENSED by Scribe 2026-08-03T10-35-00-07-00 -- sessions 1-47 condensed below (was 113,127 bytes / 1456 lines). This is the second condensation of this file: the first (2026-08-03T04-55-00-07-00, 92,884 -> 13,914 bytes) was silently reverted by the next `merge=union` merge of squad/switch, because plain union merge cannot represent a deletion -- see decisions.md, "the merge=union condensation defect". .gitattributes now routes this file through the `squad-history` driver instead, which preserves this condensation across future branch merges while still allowing concurrent agent appends. Full uncondensed text lives in git history (pre-2026-08-03T10-35-00-07-00 commits) and in decisions.md Rounds 4-9. -->
<!-- MARKER: do not delete this file's condensation by re-appending pre-condensation content from a stale branch. If squad-history merge driver is not registered in your worktree, run .squad/tools/setup-merge-drivers.ps1 first. -->

## Project Context

- **Owner:** Justin Chu. **Project:** onnxruntime-ep-vulkan — cross-platform Vulkan plugin EP for ONNX Runtime, Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` layout mirrored.
- **Stack:** Rust cdylib, Vulkan 1.1+ compute, SPIR-V/GLSL, ORT C API, Python bindings, GH Actions CI.
- **Cross-platform mandate:** Windows/Linux/Android/macOS(MoltenVK); NVIDIA/AMD/Intel/Adreno/Mali; lavapipe/SwiftShader for GPU-less CI.
- **My focus:** device/memory/sync, SPIR-V shaders, pipelines. Created 2026-07-28T17:52:04-07:00.
- Local GPU facts: Vulkan SDK `C:\VulkanSDK\1.4.350.0` not on PATH — prefix explicitly. Two devices: Intel Iris Xe (UMA, 32 KiB shared, oracle for spec-conformance) and NVIDIA RTX 4060 Laptop (discrete, 48 KiB shared). Physical/best-first index spaces are inverted on this desk (RTX = physical 1 / position 0).

## Sessions 1-43 (ash engine, first execution, runtime extents) — one-line-per-session

`ash`+`gpu-allocator` stack; dual-backend (sync2/legacy) barrier abstraction in `barrier.rs` (only file allowed to name barrier types); `push_next` must-use bug root-caused (silently dropped pNext chains); teardown order = field-declaration order (`instance` last); §7.9 probe-validity rules, R5 (subgroup BASIC) demoted from gate to probed capability; `SkipSimplifiedLayerNormalization` kernel; descriptor-set lifetime fix (VUID-03047); `ENGINE_ACCEPTS_RUNTIME_EXTENTS` flipped, runtime shapes read at Compute, unblocking 97 nodes; first real GPU dispatch (NVIDIA), 3 ash/sync2 bugs fixed; dynamic-kernel binding mismatch caused all-zero logits, fixed; VkQueryPool GPU timestamps + tracer; island-merge, clippy; sub-phase attribution, weight-tensor GPU buffer cache (2642x upload reduction); §6.5 ruled (exactly one VkDevice per physical device + EP instance); GEMV column tile; packed 128-bit loads close Intel gap; allocator adopts by identity, not selector luck; Tank found `OwnedDevice::create` ignored its `device_index` argument (4th face of the index-space defect) — held pending sequencing with Tank's `MIXED` state.

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

## Owed / open at end of Round 9
- Certified NVIDIA packed-loads A/B still unobtained (3 attempts, never a quiet window).
- General GQA grouping case (Nq/Nkv != 1) never run — Phi-3.5's 1.00 ratio is degenerate.
- Growing-context KV round-trip measurement beyond ctx ~4096-8192 unmeasured; VRAM cost of arming device-resident KV on an 8 GB laptop GPU not yet measured (deferred jointly to Niobe).
- `DEVICE_MEMORY` default stays OFF pending the above.
- Localisation (`localise()`) built but not wired to any lane; whether foreign GPU work moves `gpu_steady_tail()` still open.

📌 Team update (2026-08-03T04-55-00-07-00): Link retired his own Session-13 method of quoting a rebuilt-DLL hash as evidence a binary changed: six builds of an unchanged tree produced six distinct Windows DLL hashes, so a hash witnesses nothing about content. Every DLL hash quoted in your sessions above (e.g. `A9A381602D8B4014`, `D408A901C4F6A454`) is a build identifier only, never evidence that the binary differs from a prior one — do not cite a hash change as proof of a code change going forward. — decided by Link

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
