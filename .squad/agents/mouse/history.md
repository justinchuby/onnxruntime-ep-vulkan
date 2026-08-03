# Mouse (Op-Coverage) — history.md

## Learnings

### [SUMMARY] Turns 1–19: registry, producers, proofs, runtime extents, and early execution (2026-07-28–2026-07-30)

- Registry/claim discipline was built first: 174 standard-domain rows plus `com.microsoft` rows, per-decision JSONL claim logs, GQA fingerprint self-audit, and the rule that coverage is quoted as `(claimed_coverage, island_count, largest_island_flops)` / concentration, never percentage alone.
- Producer truth was corrected and pinned: the authoritative producer is `onnxruntime/mobius@87fd878`, not the mirror repo; builder source is intent, but the emitted model file is the fact. This forced standard-domain rows (`ai.onnx::Attention`, `RMSNormalization`, `RotaryEmbedding`), recorded `SimplifiedLayerNormalization` as `domain=""`, and corrected real-graph facts like `do_rotary=1`, packed QKV presence, and QMoE top-4.
- Opset windows became part of claim logic, not metadata: `ONNX_OPSET_LAST_RELEASED=26`, `ONNX_OPSET_REGISTERED=27`; windows key off schema version, not model opset; `Attention` had to close at opset 24 because optional input 6 (`nonpad_kv_seqlen`) changes semantics; `LinearAttention-27` and `CausalConvWithState-27` were added as standard ops.
- Evidence rules tightened: row status stayed `Live | Staged(reason)`; `EXERCISED` became the positive evidence list; `Add` went Live for f32 only; template similarity was ruled insufficient evidence for `Sub/Mul/Div/Pow`; a mechanism that exists in a file but not in a call graph counts as absent until run.
- Diagnostic plumbing itself was repaired: CLAIM_LOG stopped freezing its env-var path behind `OnceLock`; profiling JSON stayed only for `is_vulkan_claimed`; silent-pass guards were found because missing logs had been reading as ordinary negative results.
- One mechanism unblocked many rows: an unconditional four-float push-constant tail unlocked `Selu`, `Elu`, `HardSigmoid`, `Shrink`, `ThresholdedRelu`, `LeakyRelu`, and `CeluAlpha`. `Clip` still declines when bounds are omitted/dynamic because those are runtime inputs or dispatch-shape differences, not baked parameters.
- `com.microsoft::MatMulNBits` shipped Live for all `M`, fp32/fp16. Key facts were empirical: nibble order/layout came from a CPU oracle (`A = I`), all 161 Phi-3.5 nodes are fp16, and the prepack path is still a pass-through seam until `compile_hook_for` is wired.
- The census repeatedly disproved first-match stories: full-set Phi-3.5 is `dynamic-shape=356/363`; landing all staged kernels under static-shape gating unlocks **0** nodes; the shape split became `extents-symbolic` vs structural, then claimable symbolic extents / unknown-rank decline / data-dependent decline; symbolic broadcast checking was fixed so runtime-extents admission would not silently skip compatibility checks.
- Runtime extents then became measured rather than hypothetical: 227 Phi-3.5 nodes are predicate-clean under runtime extents, 161 were immediately claimable, and pinned-dims execution became the first real model run on the EP; the next blocker exposed by the same artifact was dtype (mostly fp16), not shape.
- Operational facts recorded for later work: `.squad/decisions/inbox/` is authoritative only in `main` because it is gitignored in worktrees; `VULKAN_SDK` is `C:\VulkanSDK\1.4.350.0`; both local GPUs satisfy §7.2; Lavapipe is the CI/mobile-warp proxy; ORT's planner starts returning interior pointers only from run 2 onward; `ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` with `dispatches_executed > 0` is the only reliable execution witness; `push_next` must rebind, never discard; ABI notes include borrowed `GetValueInfoTypeInfo`, nullable `CastTypeInfoToTensorInfo`/`Node_GetAttributeByName`, size-then-fill `ReadOpAttr`, and null `OrtValueInfo` for omitted optional inputs.
- fp16 elementwise widened the real model only after two hidden bugs were closed: `only_f32` was replaced by `only_proved_dtypes`, and all fp16 modules stopped depending on unsupported 16-bit storage by using packed-`uint` half I/O. Intel then exposed the odd-tail/subword bug the 4060 tolerated; the durable rule is to decline ORT-sized subword tensors unless 4-byte safety is proved.
- Capability and scratch rules were also turned into instruments: `GENERATED_CAPABILITIES` was split from live `ENGINE_ENABLED_CAPABILITIES` (`Shader` only), an `Int64` guard was deliberately fired to prove it rejects buildable-but-disabled modules, P6 scratchlessness was asserted structurally by `alloc_temp` count, and the harness-shape blind spot was named explicitly: one-inference-per-session evidence can never see run-2 planner failures.

### [SUMMARY] Sessions 20–26: zero-logit fix, multi-run discipline, island wiring, last-ten-nodes closeout, RAI-011, and gate-arm attribution (2026-07-30–2026-08-01)

- **R9 / containment:** a green suite was falsified when Phi-3.5 dispatched 161 `MatMulNBits` kernels with `compute_failures=0` yet produced all-zero logits. From then on every claim had to name the instrument that would go red if false; `model_output_equivalence` became mandatory beside counters; CPU-vs-CPU agreement had to be ruled out by asserting the EP is actually in `session.get_providers()`.
- **Session 20 root cause:** dynamic `MatMulNBits` built a 4-binding descriptor from node counts while the fp16 kernel dispatches 5 bindings (`[a,b,scales,zp,y]`, with `zp=scales` when zero-points are absent). The output slot was never bound on the dynamic path, so the shader wrote nowhere and fresh GPU memory read back as zero. Fix: `ShapeOnlyRecorder` now preserves `k.bindings`, and `dispatch_ort` uses those captured bindings. Regression test: `test_matmulnbits_fp16_dynamic_batch`. Post-fix Phi-3.5 matched CPU at top-1/top-10 on both devices; `accuracy_level=0` vs oracle `1` was re-checked and ruled irrelevant.
- **Session 21:** three-run session tests were added because run-1-only harnesses cannot distinguish clean unwritten buffers from real computation. Dynamic-batch `MatMulNBits` and full Phi-3.5 logits were proved non-zero and stable across repeated runs in one session, separating the fixed output-binding bug from Tank's distinct KV-cache/run-2 arena issue.
- **Session 23 / SkipNorm + temps:** an fp16 `SkipSimplifiedLayerNormalization` shader landed, then the first real `alloc_temp` use exposed infrastructure debt: temp tokens above ORT outputs were being routed into `gpu_outputs`. `pending_temp_sizes`, `temp_byte_sizes`, `gpu_temps`, temp offsets, and `free_all` were extended accordingly. The code worked, but the hypothesis failed: claiming 64 SkipNorm nodes moved Phi-3.5 **257 -> 321 islands**, proving node count is not island-removal evidence. Proof-ledger scaffolding (`ProofKey`, validation, wildcard rejection) was recorded but not yet activated.
- **Partition wiring / multi-node dispatch:** 33-island partitioning was already measured, but compute still panicked because intermediate outputs were tokenized positionally per kernel. Island-wide name-based tokens, `gpu_intermediates`, inter-kernel barriers, and pre-pass intermediate descriptor propagation fixed it. Result: `compute_calls = 1023 == 33 islands × 31 inferences`, `model_output_equivalence = MATCH`, and "island-count == claimed-count" became the red falsifier for unwired partitioning.
- **Cross-agent performance lesson:** 85.9% of inference wall-time was measured as non-GPU work (recording 68.3%, fence-wait idle 16.3%, submit 0.3%, kernels 14.1%), so GPU-kernel tuning was explicitly deprioritized behind command-buffer recording. This sat beside the broader team rule: verify by running, not by reading.
- **Session 24 (last 10 nodes):** graph-neighbourhood reading split the remaining gaps into a true data path (`Gather` -> `LayerNorm`), a tiny INT64 control plane, and `If` cache-control flow. Only `SimplifiedLayerNormalization` and embed `Gather` were claimed; the six-node control cluster, `Shape`, and `If` were declined permanently. Predictions were written first: islands stay 1; claimed `353 -> 355`; declines `10 -> 8`; first-inference upload `+187.9 MiB`; zero new cuts; host->device bytes drop `12,280 B` at `s=1`. All but the byte-drop magnitude confirmed; P3 missed high by 2× (actual upload drop `6,136 B`). Final measured state: **355 claimed / 1 island / 8 declines / 0 cuts**, 24 CPU nodes per run, byte-identical on both devices, with recalibrated boundary cost **399,376 B upload + 457,344 B readback = 856,720 B** and readback explicitly larger.
- **R13 / admissibility:** all wall-clock figures, including headline speedups, were withdrawn pending certified device-clock evidence; Intel device-clock figures were later ruled permanently uncertifiable on this hardware; only counts, bytes, and certified companion-clock figures remained quotable.
- **Session 25 / RAI-011:** Rai was right — the gate was unreachable on single-cluster Phi-3.5 because `GetCapability` short-circuited to `Verdict::Claim`. `partition::gate_islands` became the only entry point; evaluation always runs; single-island keep-alive is represented as `GateOutcome::SoleIslandOverride(RejectReason)`; `retain_viable` became a projection of the same function; counters split `viable_islands_retained`, `net_benefit_sole_island_overrides`, and `net_benefit_gate_bypasses`.
- **R10 / R11 artifacts from Sessions 25–26:** with shipping settings the gate evaluates once and keeps Phi-3.5; with anchor exemption off it overrides at every tested `fixed_ns`; removing the byte term restores a real flip at ~`3,836,739.6 ns`, proving `fixed_ns` is not the critical uncertainty. The real defect was Mouse's own estimator: `ep.rs` counted internal island edges as boundary and substituted `128` for every unknown dim, yielding **89,199,100,032 B** against the measured **856,720 B** — a **104,116×** disagreement. Session 26 then sharpened the question from "was the gate evaluated" to "which arm kept the island" and showed: shipping uses the gate's own claim verdict (`retained=1`, `overrides=0`), but disabling the exemption flips the same graph to `TRANSFER_DOMINATED`; the deciding term is the anchor exemption, and the economics arm is wrong, not merely untested. To stop override provenance dying at the counter boundary, `net_benefit_override_reason` was added with `UNOBSERVABLE` / `TOO_SMALL` / `TRANSFER_DOMINATED` / `MIXED` / `UNRECORDED`. Session 26 also recorded the worktree hazard explicitly: shared-worktree builds and diffs can attribute a sibling's uncommitted file state to you and manufacture false findings.
- **Remaining declines / late pre-ledger state:** the eight post-Session-24 declines were re-attributed as `DETACHED ×5` and `EDGE_ENTRY ×3`, with no `INTERIOR` declines; the post-merge R10 probe re-confirmed the gate artifact byte-identically; one unreproduced failing lib-test run was recorded as `ERROR(instrument)` rather than a detection; and `ledger_lookup` was left as the final named `UNWIRED` mechanism before the verbatim entries below close and then reopen criterion 11.

---

<!-- SUMMARIZED by Scribe 2026-08-02T02:34:23-07:00 -- older entries condensed below; full text lives in git history -->


<!-- SUMMARIZED by Scribe 2026-08-02T22:37:04-07:00 -- entries from 2026-08-01T21:15:16 through the roofline-split and re-proof-path/staged-op-sweep sessions condensed below; full text lives in git history -->

### [SUMMARY] 2026-08-01T21:15:16 through 2026-08-02 (pre-Switch's-round): criterion 11 closes then reopens, the ledger is populated 9->95, runtime extents, roofline split, and the reprove-path defects

- **Criterion 11, first close (2026-08-01T21:15:16):** proof ledger wired end-to-end (`ledger_lookup` ALL-PROVEN, 9 entries, digest baked in via `include_str!`). Two real defects self-caught: `CLAIM_UNPROVEN`'s hatch split proof keys on `,` (keys contain commas), shredding them — separator moved to `;`; and a truncated key fragment (`ai.onnx::Add/7+/f32`) passed `ProofKey::validate`, which only checked for a `/` — validate now requires `::`, five `/`, no empty component. `sqrt_f32` false-DIVERGENT (NaN≠NaN vs `max(0,nan)=0`) fixed with an `INPUT_DOMAIN` table and an ERROR-on-non-finite-reference rule. Estimator defect (89.2 GB vs measured 856,720 B, a 104,116× miss) split into a closed half (internal island edges wrongly counted as boundary, fixed) and an open, self-disclosing half (`slot_bytes` substitutes 128 for every unknown runtime-extent dim — a fabricated input, not an over-broad one, ~16,268× residual).
- **Criterion 11 reopened (2026-08-02):** Morpheus ruled MET was wrong — the ledger was scaffolding, indistinguishable in the artifact from one derived from the claim table itself (R11 on Mouse's own mechanism). Repaired: every entry now carries `claimed_nodes`/`dispatches_executed`/`worst_rel` (absent treated like zero, quoted-zero treated like absent); a header-vs-file digest refusal (`Ledger::faults` on any mismatch or unreadable file); and a three-token `LedgerLookup::{Hit,KeyAbsent,Faulted,NeverAttempted}` replacing a bool, with `LEDGER-FAULTED` outranking `KEY-ABSENT`. The economics bound was converted from prose ("concurs with") to an assertion (`the_claim_survives_an_adversarial_inflation_of_the_term_opposing_it`) — the claim survives even a 16,268× adversarial inflation of the opposing term, with a standing falsifier where the substituted 128 under-counts at long prefill and the bound evaporates.
- **Populating the ledger (9->73 entries):** op suite 154 failed -> 37 failed. GQA entered as `DIVERGENT worst_rel=16.726`, reproducible to the digit, corroborating the pre-existing `xfail`. A non-MATCH verdict cannot enter `proof_ledger.jsonl` (would fault the whole ledger) — added `evidence/proof_attempts.jsonl` as a no-grant append log. Self-caught an enumeration instrument that silently truncated the claim log when read whole-suite vs per-file (786 vs 3,140 records) — read per file from then on. 3 of 5 pre-written predictions falsified, all on second-order effects of a decline, none on the key algebra itself.
- **Phi-3.5 at runtime extent (0->323/363):** the "0/363" premise being escalated was already stale (four of five runtime-extent keys were already proven); GQA was the only real gap. Unblocking further claims exposed the actual remaining blocker: `vk/session.rs`'s island-output-consumed-internally branch (an `else` left as `None` under a comment calling the case "unusual") — Switch's, not the ledger's. `session.disable_cpu_ep_fallback` wired into `prove()` so ORT's own refusal (`CpuFallbackRefusal`) is distinguished from an instrument error, mutation-tested to differ from `UNATTRIBUTED`.
- **Roofline split (Session 27):** CPU share of bytes/FLOPs as a function of context (0.07%/0%→73.77%/30.2%, both devices identical) is a curve, not the flat "~3%"/"~30%" figures previously quoted at ctx=0. Claiming GQA (323->355 nodes, 33 islands->1) is what collapses the CPU share, not a separate lever. `ep.rs`'s FLOP estimator was found to report a constant 16.58% at every context — a node-count ratio wearing a FLOP's clothes. A first cut mis-flagged fabricated-extent share at exactly the CPU-byte share (an identity with no reason to hold) — fifth time this week the alarming number was the broken instrument.
- **The ledger had no re-proof path (Switch's find):** `--append` silently skipped already-claimed forms, so GQA's ledger entry outlived two same-day shader rewrites with nothing catching it. Fixed with per-entry `shader_digest` (FNV-1a/64 over dispatched SPIR-V) recomputed at parse, `--reprove`, and demotion tokens `STALE-SHADER`/`NO-SUBJECT-WITNESS` — frame stated explicitly: covers dispatched SPIR-V bytes only, not host-side numeric changes (named residual). No grandfathering: all 74 shipped entries re-proved from scratch, key set byte-identical. Criterion 10's first real reading: `DISAGREE`, 65/65 compared, 0 degenerate — a self-caught tolerance-ratio bug (dividing by the wrong denominator) had been reporting false 24.6× errors on outputs that actually passed; honest ratio shows a smooth mid-curve crossing (layer 30 passing at 0.85, layer 31 failing at 1.17), not a discontinuity.
- **Staged-op sweep (21 promoted, 3 refused):** census found 5 staging reasons, not 4; only 22 of 41 staged rows were dischargeable by proof (19 need code, the 13 `XL_KERNEL`/contrib rows cannot be advanced by proof runs at all). Found a harness hole before using it: 12 of 22 ops return bool and `compare()` had no non-degenerate-reference guard, so `Equal`/`IsNaN` could have reported false MATCH having tested nothing — fixed as `ERROR(instrument)`, mutation-tested both polarities. `Sum`/`Mean`/`Max`/`Min` 3-input case caught a real claim/lowering-arity mismatch (predicate allowed 1..8, lowering handled <=2) before it could have shipped as a claim-then-crash. `Swish` blocked by the first `EXERCISED` hand-written evidence list (not added by hand — would assert something unmeasured). Two ledger-mechanics defects found by breaking things on purpose: default (non-`--append`) write silently discarded 94 of 95 entries while `--check` reported PASS on the empty result ("the same defect Scribe's health report had, one level out" — entries now always carried forward, `--rebuild` required to discard); and `--reprove` never re-measured against a healthy ledger (`unproven_forms_enabled` only fires on a ledger *miss*, so an already-proven key offered through the hatch produced nothing) — yesterday's reported 74-entry re-proof only worked because the on-disk ledger had drifted from the DLL's baked copy, an accident of state rather than a real path; fixed with a distinct `reproof_forms_admitted` witness. Op-suite reds 43->18, all attributed to pre-existing or documented refusals, none to the ledger.
- 📌 Team update (2026-08-02T02:03:46-07:00, from Scribe): Morpheus's R12 fourth generalisation — for a test result, the frame is the binary that ran it — drawn partly from two of Mouse's self-caught near-misses this session (a shared-worktree build linking a sibling's in-flight file; `Copy-Item` preserving `LastWriteTime` letting cargo silently re-run a mutated binary after a restore).
- 📌 Team update (2026-08-02T14:42:30-07:00, from Switch/Mouse): the re-proof-path fix above (`shader_digest`, `--reprove`, `STALE-SHADER`/`NO-SUBJECT-WITNESS`) landed in response to Switch's finding that the ledger could not be invalidated by changing its own subject; all 74 entries re-proved under the new scheme.

---

## 2026-08-02 — Switch's round: a destructive success, a witness, a staleness the digest cannot see, and an ABI eight bytes off

Commit `4d47362` on `squad/mouse`. DLL `5F3977CB2260737E` before, `74FA8018ABDAFE36` after.

**1. `--reprove` shrank the ledger and printed `PASS`** — second time this file learned the same lesson, this time via the flag added to fix the first instance. `write_ledger()` now refuses a shrink, names the dropped keys, writes nothing, and skips `--check` on refusal (asking it would answer PASS about the stale file). `--rebuild` is the deliberate override.

**2. `NO-SUBJECT-WITNESS` was already reachable** via three existing guards (`disable_cpu_ep_fallback`, `prove()`'s `UNATTRIBUTED`, `entry_line()`'s refusal); Switch's hand-read of the source was a fourth, but R10 says a code reading is not a falsifier — both shapes are now planted in the lane with a healthy control.

**3. Stale input cache does not affect any of the 95 entries** — every proof arm is a fresh subprocess with exactly one `sess.run` (`compute_calls: 1` verified), and the defect needs a *second* `Compute()` in-session. Made durable as a field: `compute_calls` is recorded per entry and `entry_line()` refuses a run that computed more than once, so a future multi-inference form cannot inherit today's immunity by coincidence.

**4. The actual find: `device_losses` inserted mid-struct, three ctypes mirrors reading the old layout.** Census read `partitioner: UNWIRED (dispatches_executed delta = 0)` on a run that also reported `claimed=1`/`compile_calls=1`/`compute_calls=1` — two artifacts from one run disagreeing is a reader fault. `a52024f` inserted the field between `compute_failures` and `dispatches_executed` (against the struct's own doc comment) without bumping `COUNTERS_ABI_VERSION`; everything below shifted eight bytes, so `dispatches_executed` silently read `device_losses` (always 0 healthy) and `unproven_forms_claimed` silently read `ledger_entries` (95 — exactly what `--check-counters` fails on). Nothing went red because the wrong number was stable and plausible. ABI bumped to 4, all three mirrors repaired, reader now raises outside the `try` that returns `{}` (an empty dict reads as delta-0, which reads as `UNWIRED`).

**Judgement call on record:** the lane now asserts `struct_size` equality, not `>=`, forfeiting documented forward-compatibility, because an append and an insertion are indistinguishable from the reader's side and only one is safe. Cost: a red lane on every counter added, until a per-field offset manifest exists.

**Filed, not done:** three hand-maintained ctypes mirrors of one C ABI is the real defect (two in `test_phi35.py` repaired but unguarded; one shared reader is the honest fix, not mine to land unilaterally). **Every ctypes reading between `a52024f` and `4d47362` is suspect.**

Verified: 479 lib tests, clippy clean, ledger lane 14/14, both census lanes green on device 0, `--check PASS 95 entr(ies)`, shrink guard and ABI guard both mutation-tested in both polarities.

---


## 2026-08-02 — The second evidence list (§8.9.16). Op suite 11 red -> 6 red.

**The defect was mine and it was three weeks old.** `Add-i32`/`Mul-i32` declined `[dtype] ... has
never executed on a device`. True, and unfixable: `elementwise::EXERCISED` was a hand-written
`(op, dtype)` list consulted *inside the claim predicate*, which runs before a proof key is
computed. The form reported no key at all -- not `[unproven]`, which the generator can unlock --
so `gen_proof_ledger.py` could never reach it and the only exit was to type the pair in by hand.
A form was unproven because it was unproven. Criterion 11's own shape, arriving from inside
criterion 11's own module.

**The rule I want to keep:** a gate that runs before the evidence is computed can only ever be
satisfied by hand. Split it -- capability upstream (claiming an uncreatable module is a crash,
not a decline), evidence downstream where a run can clear it.

`EXERCISED` and `TEMPLATE_LIVE` deleted; `only_proved_dtypes` -> `only_loadable_variants`, backed
by `variants::variant_is_loadable`, which reads the SPIR-V and refuses every `_i64` stem because
`Int64` needs `shaderInt64` and `vk::device` passes no `pEnabledFeatures`. Derived, not
remembered. `no_live_claim_rests_on_an_unloadable_variant` used to scope itself by `proved_at` --
it could only see pairs somebody had written down, i.e. not the forms most at risk. It now walks
every dtype the caps accept and asserts `refused > 0` (R12).

**Stated before running, per R10:** add_i32 and mul_i32 each offer one unlockable key and clear
it; swish_f32 offers none, because the *row* is still Staged and that is a separate gate. Held
exactly. MATCH `worst_rel 0.0` both, `dispatches_executed 1`, shader named, `compute_calls 1`.
Ledger 95 -> 97. Green on device 1 as well as 0.

**My own guard, applied to me:** the ledger grew and `claimed_nodes` did *not* move -- 355 before
and after. Correct here, and I would rather say why than let it be noticed: the two forms are i32
at static extent, Phi-3.5 is f16 with no i32 elementwise node, so no key of theirs can be looked
up on it. The falsifier that moved is the op suite, which is their actual surface.

**Six left, and they are four different pieces of work, none reachable by a proof run.**
`clip_no_bounds` is *not* a claim-predicate defect -- `claim::ew_clip` already documents the
refusal and the repair (a variant substituting +/-infinity; an omitted bound is a different
dispatch shape, and widening the predicate would bind a buffer with no producer). `Cast` x3 needs
a template and a manifest column keyed on a dtype *pair* -- the only op in the table whose stem is
not a single dtype. `IsInf` is the selector case, four bodies not one uniform. `Flatten`/`Reshape`
I deliberately did not register: a lone shape op in a one-node island buys nothing, their only
value is not breaking an island, and no graph we have asks for that today. Registering them to
turn a test green would be widening the claim table for the suite's benefit rather than a
model's.

Verified: 478 lib tests, clippy clean, ledger `--check PASS 97 entr(ies)` digest `eb7c4e1f90cd7ec2`,
ledger + diagnostics lanes 22/22, census + elementwise 42 pass / 1 expected red, op table 85/6.
DLL `A61DC855FF85FCAD` (pre) -> `96C19E95C16E4295` (post).

---

## Session 28 — 2026-08-02 — kernel identity for GEMV, and `a52024f` again

**Ask:** `ONNXRUNTIME_EP_VULKAN_GEMV_PACKED` selects a different kernel and no artifact records
whether it was in force, so every kernel reading we hold is silent about its own subject.

**Merged `main` (`2c1e2c7`) -> merge `d3f79eb`. Rebuilt. DLL hashed either side:**
before `96C19E95...FBA1FF`, after `F2A1D728...36DD9C33`.

**Done.** `counters.rs` emits `pipeline_variants` and `gemv_packed_spec_constant`, recorded in
`vk/session.rs` from `eff_shader`/`eff_spec_constants` -- the effective pair, not the request.
**JSON-only, no ABI bump**, following the `model_output_equivalence` precedent, so the three-mirror
hazard is not enlarged. Five-state string token; `UNOBSERVABLE` when no GEMV pipeline was built,
which is the common case and would have been a lying `0`.

**Falsifier:** `probe_gemv_kernel_identity.py`, 5 arms predicted before running, **PASS on both
devices, 5/5**. Arm B is the one that matters -- env untouched, block 16, token moves `1 -> 0` by
shape alone. `shaders_dispatched` is byte-identical across the packed/unpacked arms; the old field
cannot name the kernel, the new one can.

**Found while validating:** `898a2ba` inserted three fields mid-struct without an ABI bump. Three
stale ctypes mirrors, seven fields of shift, `ledger_entries` stale-reads **0** against a true
**97**. One defect, three red tests -- and **not mine**: my diff adds no struct field. My own
equality guard is what caught it; the older `<` guard cannot see a grow.

**Built the real fix:** `counters_abi.py` derives the mirror by parsing `counters.rs`. Not wired
into Trinity's three call sites -- four agents live in this tree, routed to her instead.

**Own error, disclosed:** first `--compare` run crashed because I called the export without its
length argument. ERROR(instrument), mine, fixed. Then checked all three test call sites pass the
length correctly -- they do, so the drift is misattribution only, not memory unsafety.

**Fourth time this week the invisible bug was the plausible one.** `dispatches_executed` landing on
`outputs_device_resident` reads 0 on every healthy run. I keep learning the same lesson: the
dangerous reading is not the wrong-looking one.

**Scope:** does not generalise to the other eight env switches (not spec constants) -- they stay with
Link. Does generalise across all kernels.

📌 Team update (2026-08-02T22:37:04-07:00): Link's `link-ledger-toolchain-not-device` finding — it's your ledger, and Morpheus's ruling is pending. `registry::shader_digest_for` hashes SPIR-V bytes, so Ubuntu's `glslc` faults all 74 entries with no kernel change; meanwhile `"device": "device0"` is recorded on 74 of 75 entries and no predicate reads it — demonstrated by forcing Intel Iris Xe on Windows, where the EP claims a form "proven ... on device0" though nothing has been proven there. Morpheus has ruled a three-state remedy (`PROVEN`/`PROVEN-ELSEWHERE`/`UNPROVEN`, plus `SUBJECT-CHANGED` vs `TOOLCHAIN-CHANGED` demotion) — the predicate change (read `device`), the states, and the demotion split are named as still owed to you. Do not resolve by re-proving per platform: that turns the digest into a per-machine fingerprint and `--reprove` without `--append` was destructive at the time it was checked. — decided by Link, Morpheus

---

## 2026-08-02 — The mirrors are gone, the layout is compiler-checked, and `PROVEN-ELSEWHERE` runs

**What I got wrong yesterday, in one sentence.** I built `counters_abi.py`, called it "the real fix,
built but not yet installed", routed the three call sites to Trinity because four agents were live in
the tree, and the same defect bit again. The concurrency reasoning was fine. The safety reasoning was
not: **a generator that co-exists with the thing it replaces is a fourth mirror.** Filing the removal
of a hazard is not removing the hazard.

**Verified three mirrors, not assumed three** — `test_phi35.py` x2, `test_wiring_census.py` x1. The
JSON emission and `snapshot()` are name-keyed and compiler-exhaustive, so they are not mirrors.
All three deleted; `tests/ops/test_counters_abi_singleton.py` fails on any file outside
`counters.rs` and `counters_abi.py` that declares two or more counter names in a `_fields_` block,
and carries both a planted-mirror control and a consumer control.

**Version discipline is now computed.** One field list in `counters_abi_struct!`, offsets from
`offset_of!`, `COUNTERS_LAYOUT_HASH` const-evaluated, and a `const _` assertion that fails the build
unless `(version, hash)` is in `COUNTERS_LAYOUT_REGISTRY`. A compile-time assertion rather than a
test, because a test can be filtered out by the person inserting the field. The DLL now also exports
`OrtEpVulkanGetCountersLayout` — the per-field offset manifest I said last time would be strictly
better; it is, and it took an afternoon, and I should have built it then.

**Acceptance run, not reasoned about.** Applied the exact `898a2ba` insertion; the build died with
`error[E0080]` naming the registry and the repair, and the tool exited 1 printing the row to append.
Reverted.

**Three surprises.**

1. **The build error also showed `E0063 missing fields ... in initializer`.** Appends were *already*
   compiler-checked, in five places. What was never checked was the meaning of the version number.
   I had been treating the whole struct as unguarded when only the numbering was.
2. **The Phi-3.5 probe never saw the defect and never could.** It reads the name-keyed JSON counters
   file. The offset defect lived only in the ctypes readers. So "the probe reads the same" is not
   evidence the ABI repair is safe — it is evidence the probe was never in the blast radius. Same
   digest `eb7c4e1f90cd7ec2`, same 97 entries, same 355 claimed nodes, `ALL-PROVEN`: unchanged, and
   neither a fix nor a new defect. The fix shows up where the mirrors were: `ledger_entries` reads
   97 through the derived mirror where the stale one read 0.
3. **`cargo test --lib` was already racy on `main`** — 2 failures in 6 full runs before I touched
   anything. Several `counters::tests` call `reset()` on process-global statics without
   `allocator::ledger::test_lock()`. Added the missing guards; 8/8 clean runs after. Not my defect,
   but it was quietly eating the signal I needed to trust my own change.

**`PROVEN-ELSEWHERE` observed in both polarities** on real hardware: device 0 `ALL-PROVEN`,
`proven_elsewhere_claims=0`; device 1 `PROVEN-ELSEWHERE-PRESENT`, 355 claims across 8 named forms,
each disclosed with `proved-on=device0 running-on=device1`. A missing key stays `UNPROVEN` and
declines.

**Two honest gaps, recorded rather than papered over.** `docs/DESIGN.md` §8.9 contains no four
numbered discharge conditions — §8.9 ends at §8.9.7; the ruling is R12 in §10.0.1, and I implemented
four obligations taken verbatim from its text. And `PROVEN-ELSEWHERE` discloses at INFO rather than
WARN, because on a non-`device0` run every form is elsewhere-proven and a WARN per form would cost
the `UNMEASURED` WARN its audience.

**Standing weakness, unmitigated by design.** The device identity is a *selector index*, and Trinity
established a selector is a request and not an identity. `device_frame_matches` also accepts a
physical-name match. I did not add an env override for the frame: that is a fail-open lever on the
one predicate whose failure mode is fail-open.


---

## 2026-08-02 (later) — `PROVEN-ELSEWHERE` withdrawn; the device field made load-bearing instead

The coordinator stopped me mid-implementation: Fact Checker's audit returned ❌ on *"model-level ULP
evidence cannot promote unexercised per-form keys"*, which is the premise Morpheus's cost argument
rests on. So the entry above describing `PROVEN-ELSEWHERE` as implemented is **withdrawn**, and
`docs/OP_COVERAGE.md` §7.19(c) with it. The slot exists in `ProofState` and declines.

**What I built instead, and what it cost.** `registry::device_state` is now a four-state classifier
read on every claim: `PROVEN` / `DEVICE-UNATTRIBUTED` / `PROVEN-ON-ANOTHER-DEVICE` / `UNPROVEN`.
`DEVICE-UNATTRIBUTED` still claims — declining it would take the EP from 355 nodes to zero over a
bookkeeping question — but it is counted on every claim and named per form with `entry-device=` and
`running-device=` in both the session disclosure and the counters file. Being in every artifact of
every run is what keeps it from becoming the field nobody reads, which was Fact Checker's question.

**The surprise, and it is a good one.** I keyed the predicate on the device name read off the run
rather than the selector, on Morpheus's finding that a selector is a request. Then the very run that
validates the fix reproduced it: `ONNXRUNTIME_EP_VULKAN_DEVICE=0` opened
`1=NVIDIA GeForce RTX 4060 Laptop GPU`. Had I keyed on the ordinal, the predicate would have read
`device0 == device0` and reported a match against hardware it had never looked at — a predicate that
is always true, which is the exact shape I spent the morning removing from the counters ABI.

**`parse_ledger` faulted 103 proofs on one stale entry**, directly contradicting its own comment
three lines above. Split into `Ledger::faults` (whole-file) and `Ledger::entry_faults` (per-entry).
The header-count check had to move to `entries + entry_faults` or the demotion re-creates the global
fault through the back door — that one nearly slipped past me, and it is the same shape as the
defect: a second path to the state you thought you had closed.

**Condition 4 cannot be satisfied as written and I said so rather than reinterpreting it.** For
`PROVEN-ELSEWHERE` to be a guard something must be able to come out negative on the second device.
With no per-form evidence there, nothing can. Written up in §7.20(c), together with a counter-
proposal I have *not* verified: every entry names a per-form case model under `evidence/cases/`, so
second-device proof may be a replay of 103 tiny graphs rather than the fatal cost the ruling assumes
— in which case the answer is an entry per `(key, device)` and the status is unnecessary, not merely
unsound.

**Readings.** Device 0: `claimed_nodes` 355, `ledger_hits` 355, `unproven_declines` 3 — unchanged.
`claimed_form_evidence` `ALL-PROVEN` → `DEVICE-UNATTRIBUTED-PRESENT`: **a fix**. No node changed
hands; `ALL-PROVEN` had been asserting a device frame nothing checked. `ledger_entries` 97 → 103 is
the `main` merge, not this change. 507 lib tests pass, clippy clean, census lane 7 passed/1 xfailed.


## 2026-08-02 — §8.9.18 alignment: the ruling landed and the guard bit me

**Morpheus upheld the refutation and withdrew his own paragraph in place.** `PROVEN-ELSEWHERE`
**keeps disclosure, loses promotion**. His arithmetic is the part I will keep: `proven_key_lookups`
6 against `ledger_entries` 95 — one clean ULP curve would have promoted 89 keys nothing ever
touched. My §7.20(c) "condition 4 cannot be satisfied as written" now resolves the other way, and it
is worth being precise about how: it is not that I was wrong, it is that with promotion withdrawn
the predicate that must read the status is the *declining* one, and that one does come out negative
on a planted entry. The condition was unsatisfiable for a status that promotes; it is satisfiable
for a status that discloses and refuses.

**The layout guard fired on a pure rename, and I did not plan the demonstration.** Renaming
`device_mismatch_*` → `proven_elsewhere_*` for the ruling's vocabulary moved no offset and changed
no size; `cargo build --lib` failed with `E0080` regardless, because the hash covers
`name:offset:size`. v7 `0x16eacc53e6e18d97` ≠ v6 `0xf3fac68aa2c3a3ef` at an identical 152 bytes and
20 fields. This is the mechanism working in anger on work that was not about layout — and it is
right, because a name-keyed ctypes reader would have read `0` for the renamed field exactly as
`ledger_entries` read 0 under `898a2ba`.

**Fault scope: I had the split right and the boundary wrong.** I had put "unparseable line" and
"invalid key" in `entry_faults`. Morpheus's rule — *fault scope is set by the scope of what you
cannot locate, not by the severity of what you found* — puts them back on the artifact: you cannot
locate what a line you cannot read meant to say. Moved. Both attached obligations discharged: a
demotion count printed on every disclosure (INFO at zero, WARN otherwise), and a demotion test that
cannot read zero by construction.

**Per-key replay: recorded, not commissioned.** He was careful about that and I will be too. I do
judge it right, structurally: per-key by construction, so it cannot promote what it did not
exercise. Still unrun on a second device, so still a proposal.

**Readings after the ABI change: 355 / 355 / 3, unchanged.** 509 lib tests pass, clippy clean, 12
passed + 1 xfailed on the census + singleton lanes, `counters_abi.py --check` PASS at v7.

📌 Team update (2026-08-03T04-55-00-07-00): Link found the eleven Linux `cargo test --lib` failures are a representational difference in bindgen typing (MSVC `c_int` vs. GCC `c_uint`, no negative enumerator, no arithmetic), not a signedness bug — three carrier declarations needed, not eleven casts, and `as i32` was rejected on principle (now portability rule P3). More load-bearing for the ledger: Ubuntu shaderc 2023.8 vs. Windows SDK v2026.2 compile different SPIR-V bytes from identical GLSL, so `shader_digest_for` faults every one of the ledger's 74/75 entries on Linux — proved by perturbing one GLSL template *on Windows* and getting a superset of the same test names. This is the keying decision Morpheus's `PROVEN`/`PROVEN-ELSEWHERE`/`UNPROVEN` ruling still needs settled: per-toolchain digest vs. device-independent shader correctness. — decided by Link

📌 Team update (2026-08-03T04-55-00-07-00): Trinity measured that at the final RMSNorm, Vulkan is bit-exact against a float64 reference while ORT's CPU EP is the side carrying 1 ULP of error — the residual criterion 10 flags is not evidence of a Vulkan defect. Residual is flat ~2 ULP across all 32 blocks (not monotone), with a `3 → 6 → 12` jump only in the last two hops. Bears on the open tolerance ruling: an oracle-side rounding difference should not be scored against Vulkan as if Vulkan were the imprecise side. — decided by Trinity

---

## Round 10 (cont.) — §8.9.19: one key, two digests, and an entry that survives its frame

**The defect was one `continue`, and Morpheus named the line.** `parse_ledger` skipped past a
`shader_digest` mismatch, so the entry never entered `Ledger::entries` and `Ledger::get` returned
the **same `None`** it returns for a form nobody ever proved. A frame mismatch and a key absence
were one observation with two repairs, only one of them actionable. That is the entire Linux
symptom, and it is my own §8.9.18 part 3 edit one level up: I had moved *demotion* off the artifact
and left *deletion* in place.

**Two digests.** `source_digest` hashes the tree — variant row, resolved `#include` closure, argv
minus version **and minus absolute paths** — so it is the same number on both platforms. That
cross-platform property is the whole design; hashing the raw `-I`/`-o` paths would have turned it
into a machine fingerprint.

**Five rows, not four.** The ruling's table has no row for "SPIR-V differs and the entry records no
source digest", which is every entry written before today. Guessing `toolchain` there grants every
legacy entry a claim on a possibly-rewritten kernel, so it is `SUBJECT-CHANGED{source_comparable:
false}` and the decline names `--backfill-frame`.

### What surprised me

**A single-valued state can silence a row the ruling requires to be named.** I ran the row-4
acceptance instead of reasoning about it — comment-only edit to `ew_binary.comp` — and the entries
went `SOURCE-COSMETIC` on the subject axis while the disclosure printed `DEVICE-UNATTRIBUTED` and
nothing else, because the frame verdict outranks a cosmetic subject move and *every entry in the
shipped ledger is device-unattributed*. So the row §8.9.19 calls "the row that proves the pair does
work" would have been unobservable in the only ledger that exists, and `source_cosmetic` would have
been a counter whose only possible value is zero — the exact defect class I spent the week removing,
re-created by me, one axis over. Subject and frame are different axes and one token carries one;
the subject verdict is now counted and printed beside the state.

**`disclose_demotions_of` was reading the wrong two numbers the moment entry survival landed.** It
took `live = entries.len()` and `demoted = entry_faults.len()`. Entry survival moves the demoted
population *into* `entries`, so §8.9.18's obligation would have been satisfied on paper while
reporting every drifted entry as live. Fixed to `demotion_count()`. It was a stale test that found
it, which is the argument for tests that encode behaviour rather than shape.

**The include closure had a positive state I nearly missed.** `Select-String` on `*.comp` reported
no `#include` and I briefly believed it; the file has two at lines 11–12. Editing a comment inside
`shaders/include/indexing.glsl` moved `ew_binary_add_f32`'s source digest `c96284de → c7edca19`
with SPIR-V unchanged — the transitive closure demonstrated, not assumed.

**All 103 entries backfilled with zero skips.** The refusal condition is "recorded `shader_digest`
must equal this build's", and nothing tripped it. That is a fact worth stating: the shipped ledger
is subject-consistent with the shipped binary.

### Readings

Phi-3.5 claim probe, device 0, release build: **355 claimed / 355 hits / 3 unproven declines,
`DEVICE-UNATTRIBUTED-PRESENT`, 103 entries — every reading identical to `c1d2a63`.** Nothing moved,
and that is the correct outcome: on Windows the toolchain matches, so §8.9.19 changes nothing about
what claims *here*. It changes what happens on Linux. `cargo test --lib` 513 passed / 0 failed,
clippy `--all-targets -D warnings` clean, `counters_abi.py --check` PASS at v8
`(8, 0xdf71f4e6a59271b3)` including against the built DLL, `gen_proof_ledger.py --check` PASS.

**Residual I did not close and am not absorbing:** runtime-chosen specialisation values sit outside
both digests. The observing instrument already exists (`pipeline_variants`,
`gemv_packed_spec_constant` record the effective `(stem, spec_constants)` pair); the digest does not
consume it. Closing it is a dispatch-time frame witness, not a third build-time digest. Cost
comparable to `source_digest`, smaller because the collection exists. Nobody owns it, and it
interacts with Switch's selectors.

## Session: §8.9.20 — the dispatch-time frame witness (2026-08-03)

Closed the residual I named last round: runtime-chosen specialisation values sat outside both
digests. `spec_digest` is a digest over the sorted `(stem, spec_constants)` set the run actually
bound, computed from the `pipeline_variants` collection that already existed, and audited from the
**dispatch** path — never the claim path, because a claim is decided before any pipeline exists and
a witness read there could only ever say `SPEC-UNOBSERVED`. That is my own defect class from last
round, and I refused to rebuild it a third time. `SpecWitness` has five states; `PARTIAL` exists so
a part-set digest can never be compared against a recorded full-set one and invent a delta.

`SPEC-UNRECORDED` **claims**, deliberately unlike §8.9.19 row 5: a missing `source_digest` is
repairable from the tree, a missing `spec_digest` is a fact about a run that has ended. All 103
entries are `SPEC-UNRECORDED`; their meaning has narrowed to "this kernel's bytes, under a pipeline
nobody recorded", and that narrowing is disclosed on every session and by `--check`.

Also: whole-file ledger faults are re-emitted at session-disclosure time rather than once from a
`OnceLock` that fires before the ORT logger attaches (Link's finding — the mechanism half was
mine); and the decline text no longer claims "nothing has proven it correct" when the ledger merely
could not be read.

### The positive state, demonstrated

`probe_specialisation_witness.py`: unset and forced-off `GEMV_PACKED` share `shader_digest`
`4be613c24634ec9e` and `source_digest` `270e8086408f69a4` and differ only in `spec_digest`
(`776968369d964eb4` vs `776cce369d9931dd`). Forced-on matches unset — the instrument does not move
when nothing moved. Six checks, PASS.

### Readings

Phi-3.5 claim probe, device 0, release build: **355 claimed / 355 hits / 5 unproven declines**,
`DEVICE-UNATTRIBUTED-PRESENT`, 103 entries. **A reading moved** — declines 3 → 5. It is not mine: I
reproduced 5 on unmodified `main` (`d46327b`) in a detached worktree with its own release build.
The two additions are `ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32`, static and runtime-extent —
Tank's Cast kernel arriving ahead of its proof. **A fix behaving**, and Tank's to prove.

Establishing that cost a second worktree and a second release build, because `unproven_declines`
was a bare count with no key list while `subject_changed_forms` has carried its keys since
§8.9.19. Added `unproven_decline_forms`; it answered the question on its first run.

`cargo test --lib` 527 passed / 0 failed, clippy `--all-targets -D warnings` clean,
`counters_abi.py --check` PASS at v8 `(8, 0xdf71f4e6a59271b3)` — unchanged, all new fields are
JSON-only — `gen_proof_ledger.py --check` PASS at 103 entries, digest `493616a874425910`.

**Touched that others own:** one line in `vk/session.rs` (Switch) for the dispatch hook, and the
claim decline text (Link/Morpheus). If Switch's selector work wants a shared structure for selector
identity, that is the better answer and mine should fold into it.

## §8.9.21 — `--check` was verifying the ledger against itself

Switch found it: a shader edit moved the SPIR-V, the EP declined all 32 GQA nodes, and
`gen_proof_ledger.py --check` said `PASS: 103 entries` throughout. It checked the file's header
count, its fnv1a64 and each entry's shape — all true statements about the wrong subject. The
entry directly above this one in this file quotes that PASS as merge evidence. So did six other
places today. Not broken; it resolves anyway.

`--check` now asks the artifact. `registry::baked_ledger_identity()` + FFI
`OrtEpVulkanGetLedgerIdentity` join the existing `OrtEpVulkanGetShaderSubject`; Python re-derives
nothing, so there is still exactly one implementation of the hashing rule. The red/green line
mirrors `registry::subject_verdict` deliberately — moved SPIR-V FAILs and names the entries,
`SOURCE-COSMETIC` and toolchain-only are NOTE. When no build can be found the verdict is
`ERROR(instrument)`, not PASS. That is the whole repair in one sentence: a check that cannot see
its subject must say so rather than answer about something else.

`include_str!`: I chose **refuse**, not read-from-disk. Reading the on-disk copy at run time hands
back the exact property baking exists to deny. `--reprove` refuses before measuring when baked and
disk differ; a generation run passes `expect_rebuild=True` so the by-construction difference prints
as `NOTE: REBUILD REQUIRED` instead of failing every successful proof run — failing the happy path
is how a gate gets switched off.

`rust/tools/probe_ledger_subject_check.py` is the positive control, 5 arms, predictions written
first, 5/5. Arm 2 *is* Switch's case. `ci/check_verification_subjects.py` is the sweep:
**CHECKED 22, classified 22, FOUND 0 SELF** (13 ARTIFACT, 9 EXTERNAL).

**Three things surprised me.**

The first run of the new `--check` went red on its own — `_find_lib` preferred the release DLL by
name and the release artifact in this worktree is stale w.r.t. main's `gqa_f16`. A genuine positive
control I did not have to build, and it changed the selection rule to newest-by-mtime plus always
naming the artifact in the verdict.

`clear_session_devices` was never an uninvoked instrument. `audit_instruments.py` split production
from test *positionally* at the `#[cfg(test)] mod tests` marker, so a `#[cfg(test)]` item declared
beside its subject scored uninvoked forever — mis-scoring in the dead direction, the one thing that
file's own doctrine forbids. I fixed the screen, not the code it accused.

`proof_ledger_writer_refuses` unwound one refusal at a time: `source_digest`, then `toolchain`,
then `spec_digest`. The fixture was defective in three independent ways and had collapsed three
distinct refusal tests into one red.

**What my verification established:** `cargo test --lib` 532/0, clippy clean, `counters_abi.py
--check` PASS v8 against the DLL, `--check` PASS with all 103 entries' `shader_digest` agreeing
with the debug `onnxruntime_vulkan_ep.dll` and baked == on-disk. **What it did not:** anything
about the release artifact (stale here), anything about another device, and anything about
`spec_digest` — all 103 remain `SPEC-UNRECORDED`. No claim probe or Phi-3.5 session was run, so
there is no claimed/hits/declines reading this round. And one of my gates is the thing under
repair: the PASS I am quoting is produced by the code I just wrote. Its only independent
corroboration is the 5-arm falsifier.

**Touched that others own:** `ci/open_reds.json` (Link) — deleted only my three entries, by hand,
because `json.dumps` reformatted his whole file the first time. `audit_instruments.py` (Link) —
`survey()` now skips `#[cfg(test)]` items.
