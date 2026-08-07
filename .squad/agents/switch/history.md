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

<!-- CONDENSED-AT: 816bd4adc5d9637b0f07d8477d36af91d832c5f4 -->

### [SUMMARY] Sessions 1-43, `abcb9af`/`ed48f5b`, 46m/46n, 47-49 (2026-07-28-2026-08-04): KV device residency established, arena built, DEVICE_MEMORY flip scored

- **Sessions 1-43 (one-line):** ash+gpu-allocator stack; barrier abstraction; push_next must-use bug; teardown order fixed; R5 demoted; SkipSimplifiedLayerNormalization kernel; descriptor-set lifetime fix; ENGINE_ACCEPTS_RUNTIME_EXTENTS unblocked 97 nodes; first real GPU dispatch; dynamic-kernel binding mismatch (all-zero logits) fixed; VkQueryPool tracer; weight-tensor GPU buffer cache (2642x upload reduction); one VkDevice per physical device ruled; GEMV column tile; packed 128-bit loads close Intel gap; allocator adopts by identity; Tank found `OwnedDevice::create` ignored `device_index` (4th face of the index-space defect).
- **`abcb9af`:** every dispatch writes every declared push-constant byte -- unwritten bytes are undefined, not zero; 6 shortfall lines removed by making push unconditional. Liveness 14->8.
- **`ed48f5b`, the KV ruling:** ORT does not forbid device-resident KV; the obstacle was `transfer.rs`'s own invariant (host staging authoritative, device buffer a mirror) -- EP-side work. `device_type='gpu'` fails (maps to CUDA in ORT's Python binding); `OrtEpDevice.memory_info(DEFAULT)` is the escape.
- **Session 46m, round trip on the real graph:** `host` (shipping) 393,216 B/past-token slope; `resident` flat, slope 0, bit-identical logits both devices. Root cause of the original round trip: Step 1b's `host_backing_for` loop downloaded every KV input even though the device binding was already authoritative. **Two of Switch's own "findings" this round were his own probe's bugs** (`copy_outputs_to_cpu` charging its own downloads to the thing measured; `binding.get_outputs()` vs `sess.get_outputs()` ordering mismatch fabricating a fake residual and a fake correctness bug) -- caught only because step 0 disagreed between two lanes computing the same inference. Standing rule: a bandwidth lane must carry a correctness control sharing its inputs exactly, read before the byte count.
- **Session 46n:** four device-memory hazard lanes (early allocator, two sessions interleaved, outlived OrtValues, partial-allocation-failure) all `NO_HAZARD_LANE_SEPARATES`, closed by pre-existing guards. One flag, two parsers disagreeing (`DEVICE_MEMORY=off` half-armed the allocator) -- unified. ctx boundary predicted then measured: 6144 fits (7.13 GB), 8192 does not (8.73 GB) on the 8 GB card; largest ctx reached that round: 6144, resident route only.
- **Session 47, the KV arena:** `KV_ARENA=1` makes `present`/`past_key_values` one allocation (`2x393,216xC -> 1x393,216xC`). ctx 8192 reached for the first time: 5,512,528,520 B, 0 failures/losses. Soundness came from the kernel's own disjointness (`tok_pos = past_len + s_local >= past_len`), not a residual. Shipped-then-found defect: `translate_gqa` shortened `present` on the flag alone without confirming binding -- fixed with a post-binding sweep that refuses Compute if an aliased output isn't bound. **`gen_proof_ledger.py --check` stayed green while the EP silently declined all 32 GQA nodes** after a shader edit, because the ledger is `include_str!`'d and `--reprove` has no effect until rebuild -- project-wide defect, not arena-local.
- **Session 48, int8 KV error budget (no kernel written):** host-boundary quantisation modelled as a lower bound on any real kernel's error. Residual **saturates** rather than growing linearly (predicted 1.60 ULP/step compounding to ~13,000 ULP at ctx 8192; measured saturation ~29 ULP by past_len ~28 -- a ~450x miss if extrapolated linearly). Best granularity (`per_block32`) sits at 6-7x fp16's own residual -- no ULP band admits int8 without also passing fp16 at the same criterion. Found and fixed a cancellation-counter bug reading 0 while max_ulp read 6.3e6 (subnormal references, not exact zeros, cause spacing-floor blowups). Ledger's quoted ratios (2.21x/3.17x/4.06x) do not reproduce from any artifact; measured MODEL-class ratios are int8 1.40x / int4 1.76x on footprint.
- **Session 49, DEVICE_MEMORY flip scored (3 closed, 1 blocker):** ctx-512 blocks nothing (lanes indistinguishable there); concurrent sessions is NOT a blocker but argues FOR the flip (shipping lane silently rebuilds on CPU under concurrency, 6/6 dirty, exit 0); MIXED two-device frame closed (R12 guard correctly refuses to publish `device_authoritative_observable()`); ctx 8192 closed (arena completes, 355 dispatches/step). Real blocker: an intermittent device loss Switch produced at ctx 4096 in the resident lane (1-in-8), call site `vkWaitForFences` (Tank's class). Root code fix: `vkQueueSubmit`'s queue was externally-synchronised with nothing enforcing it -- added a per-queue mutex (not across the fence wait). Self-corrections: predicted the wrong lane in writing (own probe had no unarmed control at first); missed a device-loss line in stderr he had already read (Link's screen caught it); own aggregator-by-direction tool marked its own citable claim as SPLIT/not-citable.
- Team update (2026-08-03T04-55, Link): a DLL hash is a build identifier only, never evidence a binary's content differs -- do not cite a hash change as proof of a code change.
- Team update (2026-08-03T19-55, Trinity): "nobody has run the non-unit-grouping case" was itself unchecked -- the suite already had G=4 all along; Trinity's GREEN bit-exact-at-G=4 result stands, and the disjointness argument is algebraically invariant in group size.
- Team update (2026-08-04T12-25, Trinity): `np.spacing` returns `inf` at fp16's largest finite value, so a 504-unit error can read back as 0.0 ULP -- check operand distance from the fp16 max-finite boundary before trusting a near-zero ULP reading.

## Session 50 — 2026-08-04 — the prefix alias: the growing KV cache stopped allocating `past`
**Defect:** `device_memory_ctx4096_shipping_lane_cannot_run`. **Closed on the counter, not the exit.**
**The arithmetic.** Phi-3.5 is 355 nodes in 1 island, so one `Compute` allocates everything. The
shipping lane held **two** full-size KV buffers: `past_key_values.*` at `past_len x 393,216 B` and
`present.*` at `(past_len+1) x 393,216 B`. The second copy is pure waste — under the growing GQA
convention the shader's own first act (`copy_leader`) is to copy `past` into `present`'s prefix. The
EP allocated a buffer, uploaded into it, had the shader copy it into another buffer the EP also
allocated, and threw the first away. Same shape as the `host_backing_for` defect: a loop over *all*
inputs doing expensive work for entries that do not need it.
**The fix — the prefix alias.** `bind_prefix_output` hands the `present` buffer back for the `past_*`
slots too; the engine stages `past`'s host bytes straight into `present`'s prefix with a
multi-region `vkCmdCopyBuffer` (32 regions on Phi-3.5). **No `past` device buffer at all.** No shader
change: setting `past_stride = present_len` makes past reads land where the copy put them *and*
switches `copy_leader` off, because the copy already happened. Ledger confirms: 129 = **129 identical**.
**What it bought** (RTX 4060 Laptop, device name off the run, one binary, one flag apart, no clock):
the shipping lane ran to **2560** before and dies at 3072; with the alias it runs 3072, 4096, 5120,
**6144**, all 355 dispatches. High-water delta is **exactly `past_len x 393,216`** at every point.
`closes_when`: `transient_input_reuses = 64` (= 32 layers x 2) and `transient_input_reused_bytes =
past_len x 393,216`; `transient_input_device_bytes` is now **flat in `past_len`**.
**Correctness before bandwidth.** Bit-identical `logits_sha256` at past 0, 1, 512, 2048 — the lengths
where **both** lanes executed on the EP, which is the condition that makes a pair comparable. 5-step
decode chains on one session, seeded at past 0 and 512: identical **every step**, not just at the
end. Zero leaks on every arm including the one where allocation fails partway. `check_device_loss.py`
PASS over 54 artifacts. `counters_abi.py --check` PASS with the lib set, ABI still
`(8, 0xdf71f4e6a59271b3)` — JSON-only. 585 lib tests (574 + 11), clippy clean.
1. **The boundary was 2560, not 4096.** The defect is named after the string I found it by
   (`alloc_device failed` at 4096). At 3072 the lane already fails, silently, printing nothing that
   names it. Naming a defect after its symptom string puts the boundary in the wrong place — and the
   6144/8192 figures this project quotes were never the shipping lane's.
2. **The fix reproduced the defect before it fixed it.** I appended prefix staging to `staging_ups`.
   Two loops that walk it are `zip`s, which stop short; a **third** is `enumerate()` + index and does
   not. Index-out-of-bounds inside `Compute` -> ORT rebuilt all 355 nodes on CPU -> **exit 0, zero
   dispatches at every length including 512**. The `dispatches_executed > 0` screen is the only
   reason "the fix broke something" and "the fix did nothing" were distinguishable.
3. **My own evidence-preservation fix failed on its first real use.** Last round I made stderr
   unconditional. This round the run I needed had 1,096,775 bytes of it and the last 6,000 characters
   were *entirely* allocator leak spam — the panic message was gone. Not discarded, **crowded out**.
   Worse than the original, because a populated `stderr_tail` looks like the fix working. Third
   instrument on this project to report a state it never observed.
4. **The ceiling I inferred was wrong, and the counter cannot see it.** I inferred ~4.2 GiB of usable
   device-local budget from two failing points; the fixed lane then reached **4,737,381,056 B** and
   ran. `session_device_high_water_bytes` counts only `DeviceLocal|PackedWeights` and misses the
   ~2.29 GiB of Upload staging held concurrently — so it is a **difference**-measuring instrument,
   not an absolute-budget one. The deltas are exact; the absolutes are not a budget.
**I did not flip `DEVICE_MEMORY`**, per the constraint. But the blocker now has the control lane it
lacked: the shipping lane runs at ctx 4096.
Decision records: `switch-kv-prefix-alias-removes-the-past-buffer.md`,
`switch-shipping-lane-boundary-is-2560-not-4096.md`,
`switch-parallel-vector-index-carries-meaning.md`,
`switch-stderr-tail-can-be-crowded-out.md`,
`switch-for-mouse-attention-touch-no-spec-constant.md`.
Artifacts: `bench/results/ctx4096_{BEFORE,BEFORE_samebinary,AFTER,identity,separating,chain}.json`,
`bench/results/_ctx4096_scratch/stderr_*.log` (full text, always).

📌 

📌 Team update (2026-08-04T20-25-00-07-00): Mouse's `claimed_nodes` != `dispatches_executed` -- BERT claims 481 of 1274 rows at `GetCapability` but the partitioner's net-benefit gate retains only 4; every coverage figure quoted against `claimed_nodes` alone (including prior island/counterfactual rankings) is affected. `dispatches_executed` is the honest metric going forward. -- decided by Mouse

## Session 51 — 2026-08-05 — issue #10: the queue-lock contention test was itself unguarded

**Defect:** udit_counter_test_lock.py --check reported `UNGUARDED vk/cmd.rs:428
queue_lock_excludes_and_counts_contention` once #1's CI fix let the auditor step run to
completion. Real finding, not a tool artefact: the test I added in session 49 asserts a
**delta** over `counters::queue_submit_contentions()` — reads `before`, calls
`record_queue_submit_contention()`, re-reads — and took no test lock while doing it. Any
concurrently scheduled `counters::reset()` or `record_*` lands between the two reads and turns
a real pass into an intermittent red. Ironic shape worth naming: the test that exists to make
serialization falsifiable was the one test in its family not serialized.
**Fix:** one line, the convention already in the tree —
`let _g = crate::allocator::ledger::test_lock();` at the top of the test, same lock, same module
as `counters.rs` and `vk/barrier.rs`. No allowlist, no auditor change: the lock contract did not
change, only this test's conformance to it. Concurrency semantics are untouched — the queue lock,
the `try_lock` branch and the counted contention are exactly as before.
**Evidence:** auditor `--selftest --check --pairs` 0 findings (was 1) with selftest 9/9;
negative control — the pre-fix blob still flags line 428 through `audit_text`, so the auditor is
not merely quiet; `contention_gate.py --selftest` 5/5 and the full gate GREEN, 0/20 red on all 5
pools including `counters` (32 tests) and `all-contended` (49); `cargo test --lib` 620 passed,
0 failed; `cargo fmt --check` and `cargo clippy --all-targets` clean.

## Session 52 — 2026-08-06 — issue #4: the driver was conformant and the test was right

**Defect:** `test_op_table.py::[Asin-fp32]`/`[Acos-fp32]` red on lavapipe at 1.56e-4 against a
1e-5 tolerance, green on NVIDIA. The tempting reading — "lavapipe is buggy, widen the tolerance"
— is wrong in a way worth remembering: Vulkan's precision table gives `asin`/`acos` **no bound of
their own**, defining each by inheritance from `atan2`, which is allowed **4096 ULP** in single
precision. Measured 3831/3903 ULP: lavapipe was inside its allowance with room to spare. NVIDIA's
4/5 ULP is goodwill, not contract. A first web search told me Vulkan gives these "no accuracy
requirement" at all; fetching the actual spec table corrected it. Fetch the table.

**Fix:** shared minimax core in `ew_unary.comp` — `ew_asin_core` (Cephes `asinf`, degree-4 in s²)
+ `ew_asin_abs` (range reduction at s=1/2), with `ew_asin`/`ew_acos` on top. Both ops share the
core so they cannot drift. Bitwise sign copy (`sign(-0.0)` is `+0.0`, which would break
`asin(-0) = -0`), bit-pattern quiet NaN (`0.0/0.0` is not dependable under fast-math), clamped
radicand (`sqrt` of a negative is undefined in GLSL). Bound is **derived** (16 ULP, from Vulkan's
own guarantees) not fitted — 5x wider than anything measured, deliberately.

**Evidence:** 2,000,010-point sweep vs the ORT CPU EP. Built-in: lavapipe 3831/3903 ULP, NVIDIA
4/5. Portable: **4/5 on both**. CPU EP is itself 4 ULP from float64, so this is at the oracle's
noise floor. 91/91 `test_op_table.py` on both devices; 23/23 new `test_inverse_trig.py`; full
`tests/ops` 933 passed / 5 failed NVIDIA, 931 / 7 lavapipe — every failure reproduced identically
on a rebuild of `94a4bd6`, so none is mine. `cargo ci` ALL CHECKS PASSED.

**Cost me an hour, twice:** (1) `evidence/proof_ledger.jsonl` is `include_str!`'d — `--reprove`
does nothing until you rebuild, and until you do, the EP correctly declines the ops as
`SUBJECT-CHANGED`. (2) I wrote a restore script that backed up the working shader to the same
filename it had backed up on the previous run, and the second run overwrote my only copy of the
implementation with the version I was benchmarking against. Recovered by rewriting from the
design notes; the rewrite reproduced the measurements to every digit, which is the one good thing
about having written the derivation down first. Back up to a distinct path, and verify the backup
contains what you think before destroying the original.

**Left open:** `Sin`/`Cos`/`Tan`/`Atan` carry the same exposure (`sin`/`cos` allowed an absolute
2^-11 = 4.9e-4, 49x our atol) and are on the RoPE path. Not fixed here — second body of math,
would make the change unreviewable. Recorded in `BUILTIN_SCREEN` and §8.9.28(6).

## Session 53 — 2026-08-06 — issue #7: the row tile is a register budget, and the A/B harness lied twice before the kernel did

**Shipped:** `QB_ROWS` (spec constant 6) tiles MatMulNBits prefill rows in `q_gemv.comp`. Weight
amplification over the real 161-node Phi-3.5 graph: M=2 2.0->1.0, M=4 4.0->2.0, M=5 5.0->3.0,
M=1 unchanged at 1.0. Wall clock on the A1000: M=8 **1.618x**, M=4 1.367x, M=2 1.162x, M=1 0.994x
(both arms are the same pipeline at M=1 — that number is the noise floor, and it is the control).
Rows reduce sequentially through one reuse of `red[]`, so shared memory is unchanged and no device
limit is consulted; the cap is the accumulator register budget. `QB_ROWS == 1` holds the verbatim
old body.

**The harness was wrong twice, and both times it was wrong in a way that looked like a result.**
(1) Fixed arm order — untiled then tiled, every repeat — produced a **0.905x "slowdown" at M=1**,
where the two arms build the *identical* pipeline. That is a systematic bias reported as a
measurement. Alternating arm order per repeat brought it to 0.994x. (2) The A/B bypassed
`bench.py::select_device` and ran on `device None`, i.e. silently possibly not the A1000. `bench.py`
refuses to do that; my script had reimplemented the easy half of it. **If you write a new timing
harness, call the existing device selector, and put a null arm in it — one where you know the
answer must be 1.0 — and do not believe any other number until that one lands.**

**fp16 `max_rel` over a GEMM output is a cancellation meter, not an accuracy meter.** The real-weight
differential reported `max_rel` 2.1e-1 at M=4 and I nearly filed it as accuracy loss. Cancellation
chances grow with M, so the metric grows with M for a kernel that is exactly as accurate. Reported
`max_abs` (exact fp16 ULPs) plus relative error restricted to elements >= 0.1x RMS: **flat at
~9.5e-04 from M=1 to M=8**, 72/72 match.

**Cost me time:** `src/ops/` forbids `unsafe` (`tests/layering.rs`), so env-var plumbing cannot be
tested in-module — split the pure logic (`clamp_max_rows`, `gemv_tile_with`) into `src/` and put the
env test in `rust/tests/`. The lib crate is `onnxruntime_vulkan_ep`, not `onnxruntime_ep_vulkan`.
`bench/results/_ledger_models/` is **version-controlled**; I deleted it as scratch and had to
restore it. `bench.py` needs `PYTHONIOENCODING=utf-8` or its own output crashes it on cp1252.
`VUID-vkCmdDispatch-groupCountY-00418` has been **renumbered to -00387** — assert on the stable
text `maxComputeWorkGroupCount[1]`, not the number. A validation-layer callback that returns
`VK_FALSE` forwards the invalid dispatch to the driver and you get 0xC0000005; return `VK_TRUE`
while armed.

**Left open:** the tiled arm uses 32-bit scalar B loads where decode uses 128-bit, which is why
time improves less than traffic; raising `GEMV_MAX_TILE` past 32 and widening those loads are the
next two levers. The modelrunner is still honestly UNSUPPORTED end-to-end (GroupQueryAttention
rejects generated inputs on the **CPU reference arm**), so the operator-level real-weights probe is
the strongest proof available and the PR says so rather than implying a model-level one.
