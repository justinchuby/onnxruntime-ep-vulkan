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

## 2026-08-01T21:15:16-07:00 — criterion 11: the proof ledger, and the last `UNWIRED` is closed

**The census has no `UNWIRED` line left.** `ledger_lookup` now reads, byte-identically on device 0
and device 1:

```
ALL-PROVEN proven_key_lookups=6 ledger_hits=6 ledger_entries=9 unproven_declines=0
unproven_forms_enabled=[] (hits is typed: 'UNWIRED'/'UNOBSERVABLE'/int)
```

The artifact is `evidence/proof_ledger.jsonl` — 9 entries, digest `e4436e93c19c8744`, written by
`rust/tools/gen_proof_ledger.py`, never hand-edited, baked in with `include_str!` so a build cannot
claim a form whose proof is absent from the binary doing the claiming.

**What I predicted before running, and what happened.** Written to
`bench/results/proof_ledger_prediction.json` before any run, per R10:

- **P1** (the gate discriminates: proven arm `ledger_hits=1 ALL-PROVEN`, unproven arm
  `ledger_hits=0 ALL-DECLINED unproven_declines=1`) — **confirmed byte-exactly, both devices.**
- **P4** (Phi-3.5 355 → 0) — **confirmed exactly.** `claimed_nodes=0`,
  `proven_key_lookups=357`, `unproven_declines=357`, and the claim-log histogram shows 355 nodes
  declining with `('unproven',)` **and nothing else** — no second reason riding along.
- **P5** (two forms, two keys) — confirmed.
- **P8** (the `MatMulNBits` `zero_points` pair produces keys differing *only* in the optional-input
  component) — **confirmed on substance, and I got the variant token wrong.** I predicted
  `qgemv_f16`; it is `q_gemv_matmul_nbits_f16`. Recorded as a miss because a prediction that is
  only scored on the part I got right is not a prediction.

**Two real defects my own controls caught, and neither was the one I was looking for.**

The first attributed generator run returned `UNATTRIBUTED {'claimed_nodes': 0,
'dispatches_executed': 0}`. The earlier `MATCH` had been **CPU against CPU**. Root cause: proof keys
contain `,` (`f32,f32>f32`) and the `CLAIM_UNPROVEN` hatch split its list on `,`, shredding every
key into invalid fragments. The list was *correctly* discarded, the run claimed nothing, and the
comparison still said `MATCH`. **Only the attribution check caught it** — the mechanism that exists
because of 2026-07-30 caught the successor of 2026-07-30. Separator is now `;`.

Then the regression test I wrote for *that* found the second one: `ai.onnx::Add/7+/f32` — the first
comma-fragment of a real key — **passed `ProofKey::validate`**, which only required a `/`. A
truncated key matches nothing and reads exactly like a key that matches something. `validate()` now
demands `::`, five `/`, and no empty component.

**One ERROR(instrument), and R13 says it is never a detection.** `sqrt_f32` returned `DIVERGENT`
with `worst_rel: 0.0` — self-contradictory on its face, which is the tell. `standard_normal` inputs
meant `Sqrt` of a negative gave NaN on *both* sides; `np.allclose` calls NaN≠NaN a divergence while
`max(0.0, nan)` returns `0.0`. I fixed both halves rather than the loud one: an `ERROR` verdict when
the **reference** output is non-finite (EP-only non-finite stays a genuine `DIVERGENT`), plus an
`INPUT_DOMAIN` table so `Sqrt` samples positive and `Div` non-zero.

**Rai's planted control is in the lane, not behind `#[ignore]`.** `mul_f16_unproven` is deliberately
never proven and its sibling `mul_f32` is, so the pair is two-armed and the arms are *asserted* to
differ — Switch's `arms_must_differ` lesson applied directly, because a ledger probe whose two arms
return the same key is a perfectly stable, perfectly wrong answer. `tests/ops/test_proof_ledger.py`:
**10 passed, 0 skipped, 0 ignored.**

**A lane bug the gate exposed, worth keeping.** The first lane run was 7 passed / **3 skipped**: the
suite's `_probe_vulkan_device()` requires a *claim*, and MatMulNBits was now unproven, so **"no
device" and "no proof" read as the same silent SKIP.** I fixed it by proving the op that matters —
which is the right fix for today and not for the shape of the problem. The fragility remains: remove
that proof and the two states collapse again. Flagged for Trinity.

**The price, paid and not softened. Phi-3.5: 355 → 0.** Exactly what Morpheus accepted when he ruled
§8.9. **The fall is temporary and the work is bounded** — the 355 nodes reduce to **8 distinct proof
obligations**, mechanically discoverable because every claim-log audit line now carries `proof_key`;
`bench/results/_phi35_keys.txt` came out of one gated run, not out of my head. Populating them from
existing differential runs is a harness job, not a design one.

**The 104,116× estimator defect: it was two defects, and I can only close one honestly.**

- *Closed.* Internal island edges were counted as boundary. A whole-graph per-value consumer map in
  `ep.rs` fixed it — **89,199,100,032 → 13,936,509,056 B**, and `net_benefit_sole_island_overrides`
  went **1 → 0**: Phi-3.5's island is now claimed on the gate's own economics, not on the
  no-alternative override. That is the more interesting half of the result.
- *Open.* `slot_bytes` substitutes **128 for every unknown dim**, and every Phi-3.5 boundary tensor
  is `runtime-extent`. Residual ~16,268×. This is a **fabricated** input, not an over-broad one, and
  R9 amendment 5 applies: the number moves *with* the reader's confidence, so it cannot be repaired
  by tightening a threshold. Different fix — resolve the extents or decline to answer. Until then it
  is **self-disclosing**: `Island::symbolic_boundary_slots` travels with the number and the WARN
  reports the fabricated-slot count.

**Also landed:** `epctl --check-counters` now exits 1 on a non-empty `unproven_forms_enabled`, with
`--allow-unproven` that **downgrades and never erases** — it still names the forms, because a flag
that deletes the finding is how a run with unproven claims reads as clean three weeks later. A
scalar hatch (`1`, `true`, `*`) is pinned as *not a list* — C1's shape, named in §8.9.

**State:** `cargo test --release --lib` 446 passed / 0 failed / 2 ignored; `epctl` 15 passed;
census 3 passed on each device; ledger lane 10 passed. Counters ABI **2 → 3** (five appended fields,
size-versioned so old readers are safe). Decision record in
`.squad/decisions/inbox/mouse-proof-ledger.md`, including the declared crossings into Trinity's
`tests/` and Tank's `counters.rs`/`epctl.rs`.

**Disclosed, not hidden:** `counters::tests::a_pinned_authoritative_counter_reports_unobservable_and_never_zero`
failed once in four full lib runs and passes in isolation — it `set_var`s a process-global env var
shared with sibling tests. **ERROR(instrument), not a detection.** Pre-existing, Tank's file, and
recorded so nobody rediscovers it as new.


---

## 2026-08-02 — criterion 11 reopened over my own write-up; provenance, the digest refusal, and the three-token miss (Mouse)

Four items from the coordinator, in his priority order.

**1. BLOCKING — clippy at the union.** Ran `cargo clippy --release --all-targets -- -D warnings` at
the merged state and got **five errors, not the one quoted**: `registry.rs:2261` `manual-contains`
(the union defect), plus unused import in `transfer.rs:1130`, `manual RangeInclusive::contains` in
`ops/quant.rs:888`, and fn-item-to-integer casts in `ops/norm.rs:270` and `ops/indexing.rs:118`.
Fixed all five. **The other four pre-exist on `origin/main`** — verified with
`git show origin/main:<file>` — so only `registry.rs` was a union defect and *the quoted CI command
was already red on main independently of my merge*. Whatever was green at Switch's commit cannot
have been this command at this scope. Two further violations appeared from my own new tests
(`cloned_ref_to_slice_refs`, two `undocumented_unsafe_blocks`) and were fixed. **Green at the final
state, not mid-way.**

**2. The rename.** `Island::MEASURED_PHI35_DEV0` → `ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED`,
via `MEASURED_PHI35_DEV0(?!_)` so `..._REAL_BYTES` was untouched. Two test names that had become
ambiguous about *which* estimate went with it, and
`the_override_carries_the_verdict_it_overrode` gained a note explaining in one place why its
`TransferDominated` assertion is consistent with `overrides 1 → 0` shipping. Morpheus's reason is
the part I kept: **names outlive doc comments**, and keeping both constants was correct.

**3. The bound, held as an assertion rather than prose.** I had said the economics arm *concurs*
with the exemption; Morpheus refused it — *"agreement between two things fed the same fabricated
input is not a second opinion."* What survives is an inequality:
`the_claim_survives_an_adversarial_inflation_of_the_term_opposing_it` asserts monotonicity of
`transfer_ns` in bytes over six sizes incl. `u64::MAX/4`, that the gate **claims** at the inflated
13,936,509,056 B, that measured (856,720 B) is smaller, and therefore the truthful island claims
**a fortiori** — the claim survives a **16,268× adversarial inflation of the term opposing it**.
§10.0.4's third form: **prefer the bound you can sign.** The narrow half is a standing falsifier of
its own: `the_substituted_extent_under_counts_on_a_long_prefill_and_the_bound_evaporates` — 128
over-counts at decode extent 1 and **under-counts** at prefill extent 4096, where the inequality
reverses and the bound *evaporates rather than weakening*. Both mutation-tested in both polarities.

**4. Criterion 11 — reopened, and I agree with the call against me.** My row said MET; Morpheus's
said *not met — scaffolding only*; the coordinator took his. His argument is one I could not answer:
the cheapest satisfaction is a ledger derived from the same enumeration that produces the claims,
under which `ledger_hits == proven_key_lookups` forever and **`6/6` reads identically under both
stories**. Mine is not that shape — but *nothing in the artifact distinguished the two shapes*,
which is R11 on my own mechanism.

- **(a) provenance — done.** Every entry carries `claimed_nodes`, `dispatches_executed`, `worst_rel`.
  A dispatch count only exists after a session executed; an enumeration cannot forge one. The
  generator **raises** rather than writing an unattributed entry, `--check` fails on one, and
  `parse_ledger` **faults** it. **Absent is treated exactly like zero**, and a **quoted** count like
  absent. Four ledgers differing only in those fields, four outcomes.
- **(b)(iii) the digest refusal — done.** `ONNXRUNTIME_EP_VULKAN_LEDGER_FILE`; disagreement (or a
  named file that cannot be read) → `Ledger::faults` → **every form declines**. This is a *second*
  threat from the header-vs-body digest: that catches a hand-edit before the build, this catches the
  file changing after it — the case where the artifact a reviewer reads is not the one the binary
  claimed from. Three arms, including the identical-file arm, **which is what makes the other arm a
  detection rather than a check that fails on everything**.
- **(d) the three-token miss — done.** `LedgerLookup::{Hit,KeyAbsent,Faulted,NeverAttempted}`;
  `record_ledger_lookup` takes the outcome, not a `bool`; counters carry `"ledger_miss"`.
  `LEDGER-FAULTED` **outranks** `KEY-ABSENT` (R13: a run with no reading about any form must not
  spell an outage the way it spells a detection). `NEVER-ATTEMPTED` is derived, never counted —
  recording it would be a lookup, which is what it asserts did not happen.
- (b)(i) `mul_f16_unproven` and (b)(ii) the `MatMulNBits` `zero_points` pair were already in the
  lane and in the ledger. **(c) and lane membership are Trinity's.** I did not close the row.

**Ledger digest `e4436e93c19c8744` → `331003e0ff88df3f`** on regeneration with provenance; 9 entries,
all re-attributed `MATCH` at `claimed_nodes=1 dispatches_executed=1`.

**Predicted before running** (`bench/results/proof_ledger_prediction.json`, R10): the `mul_f16`
decline, both `MatMulNBits` keys hitting, `HIT`/`LEDGER-FAULTED`/`NEVER-ATTEMPTED` in their three
frames, and a provenance-stripped entry faulting. All confirmed.

**State:** `cargo test --release` **455 lib passed / 0 failed / 3 ignored**, epctl 15, all bins green;
clippy green at the merged state; `test_proof_ledger.py` 10 passed; census 3 passed on Intel, and
`ledger_lookup` still reads `ALL-PROVEN proven_key_lookups=6 ledger_hits=6 ledger_entries=9
unproven_declines=0` — unchanged by all of the above, which is the point.

**ERROR(instrument), recorded so nobody rediscovers it:** `Copy-Item` preserves the source file's
`LastWriteTime`, so cargo's fingerprint does not notice a restore-from-backup and silently re-runs
the **previously compiled mutated** test binary. This produced a persistent false failure I nearly
"fixed" by weakening a correct assertion. Touch the mtime after any restore. The assertion now
**quotes its numbers**, because I could not diagnose what it would not tell me.

**Declared crossings:** `rust/src/transfer.rs` (Switch — one unused import), `rust/src/counters.rs`
(Tank — `record_ledger_lookup` signature, two statics, `ledger_miss`), `docs/DESIGN.md` criterion 11
row. **Collision to sequence:** I edited `partition.rs` this session, which collides with sibling
instance `mouse`'s held `Verdict::Claim` return-site change.

📌 Team update (2026-08-02T02:03:46-07:00): Morpheus named R12's fourth generalisation — for a test result, the frame is the binary that ran it — from two of Mouse's self-caught near-misses this session: a build in a shared worktree that linked a sibling's in-flight file (nearly reported as a false ALL-DECLINED finding), and Copy-Item preserving LastWriteTime, which let cargo silently keep running a mutated binary after a restore-from-backup. Your own union-check work independently reproduced the same failure shape (a stale DLL in a shared worktree nearly read as an UNWIRED §8.9 ledger) — worth checking any mutation harness you build touches or hashes a restored file and asserts the rebuild happened before reading a result as a control. — decided by Scribe

---

## 2026-08-02 — Populating the ledger: 154 reds cleared with proofs, and GQA is DIVERGENT

**Suite: 154 failed / 276 passed -> 37 failed / 393 passed. Ledger 9 -> 73 entries, digest `331003e0ff88df3f` -> `e3ea94196b4fd84f`.** Census on both devices, byte-identical: `ledger_lookup: ALL-PROVEN proven_key_lookups=6 ledger_hits=6 ledger_entries=73 unproven_declines=0 unproven_forms_enabled=[]`. `cargo test --release --lib` 469/0/4-ignored; `cargo clippy --release --all-targets -- -D warnings` green.

No guard relaxed, no tolerance widened, no entry derived from the claim table.

### The enumeration instrument under-counts, and I believed it once

The claim log is **truncated by whichever process opens it**, and several tests spawn a child that loads the DLL. A whole-suite run gives **786** records; the same tests run **per file** give **3,140**. My first residual triage used the whole-suite log and concluded the residual was one form. The per-file triage found five. This is the same defect class as the `§` decode that swallowed child stderr: an instrument that fails quietly turns a FAIL into a wrong number rather than a mystery, which is worse. Enumerate per file.

### Predictions, written before the run (`bench/results/skipsln_static_prediction.json`)

- **S1 confirmed** — all four static SkipSLN cases MATCH (`worst_rel` 0.0 for f16, 1.86e-07 for f32).
- **S2 confirmed** — four entries, four distinct keys, no collision with the 69.
- **S3 falsified** — I predicted `test_skipnorm` would go 7 -> 1, keeping `test_skip_norm_f16_phi35_shape` red because a run-time `INVALID_ARGUMENT: input is expected to have 3 or 2 dimensions, got 1` is not something a ledger entry can fix. It went 7 -> **0**. The shape rejection was itself downstream of the node not being claimed. I read a symptom of the decline as an independent defect.
- **S4 falsified in method, confirmed in outcome** — I predicted no single-node case model in this generator could produce the GQA key honestly. It can, once `feed_plan` exists. GQA still did not enter the ledger, but for a completely different reason than I gave.
- **S5 falsified** — I predicted `test_op_table` would not move. It went 28 -> 26. I did not retain the earlier per-name list, so I cannot say which two, and I am not going to reconstruct a number I did not record.

Three of five falsified. The two confirmed were the ones about the key algebra; the three falsified were all about *what else the decline was causing downstream*. I am consistently over-confident about second-order effects of the gate.

### GQA: DIVERGENT, and that is the finding

`DIVERGENT {'reason': 'output o0 outside tolerance', 'worst_rel': 16.72642029784887}` — reproducible to the digit across two runs, on a case model whose discovered key matches the residual key exactly. It corroborates the pre-existing strict-`xfail` `_GQA_COMPUTE_BUG` in `tests/ops/test_gqa.py`, from a second independent instrument. Phi-3.5's five tests and criterion 10 stay red **for the correct reason**.

A non-MATCH verdict cannot go in `proof_ledger.jsonl`: `parse_ledger` pushes it to `Ledger::faults` and a faulted ledger refuses *every* claim. So attempts now append to **`evidence/proof_attempts.jsonl`** — grants nothing, not baked in. Generator-side counterpart to Tank's `Ledger::demoted`.

### An outage that was mine

The criterion-5 shader-less witness builds from a copy of `rust/` alone, and the crate `include_str!`s the ledger from **outside** `rust/`:

`error: couldn't read `src\..\..\evidence/proof_ledger.jsonl`: The system cannot find the path specified.`

It reported `ERROR(instrument)` and was right to. Fixed in `tests/ops/_shaderless.py` (cross-owner, Trinity): the scratch tree carries the ledger, and the ledger's mtime joins the staleness check so a regenerated ledger cannot be witnessed against the previous binary.

### Answered Tank

**Yes** to `net_benefit_single_island` as `UNOBSERVABLE`/`BYPASSED`/`EVALUATED`. `sole_island_overrides=1` tells a reader an override happened but not whether the single-island path reached the gate; those are R12's two states sharing one reading.

### Residual, by decline code and not by count

37 red: **26** `test_op_table` (`[staged]`/`[not-registered]`/`[dtype]`/`[attribute]`/`[opset]` — the ledger has no authority over any of them), **5** Phi-3.5 behind the GQA divergence, **3** `Min`/`Max`/`Clip`-no-bounds, **1** criterion 10 behind the same divergence, **0** instrument errors.

## 2026-08-02 — Phi-3.5 at runtime extent: the count moved, the ledger did not

**Request:** prove the five `runtime-extent` keys behind Phi-3.5's `0/363`, on the premise that
everything the ledger proves is `static`.

**The premise was stale.** It was taken against the pre-`e97b186` ledger. At the merged state four
of the five keys were already present at `runtime-extent`; only `GroupQueryAttention` was missing.
I did not mint keys to satisfy it — evidence whose effect is already achieved is indistinguishable,
in the artifact, from progress.

### R10 first, then the run

`bench/results/phi35_runtime_extent_prediction.json`, written **before** loading the real model:
five predictions, five falsifiers. All five CONFIRMED; P2 and P5 exactly.

- **0 → 323/363 nodes claimed**, `33/33` islands retained, `ledger_hits=323`,
  `unproven_forms_claimed=0`, `claimed_form_evidence=ALL-PROVEN`, `ledger_gate=MIXED`.
- **Ledger unchanged at 73 entries**, digest `e3ea94196b4fd84f`.

The guard I was handed was "a ledger that grows without the claimed count moving." What happened is
its **inverse**. That asymmetry is the whole result.

### GQA is a finding, not an obstacle

`DIVERGENT`, `worst_rel=16.72642029784887`, reproducible to the digit. Not written to the ledger — a
non-`MATCH` entry becomes a `Ledger::fault` and a faulted ledger refuses *every* claim. Recorded in
`evidence/proof_attempts.jsonl`. The handler claims the form and then disagrees by 16.7×;
**claiming-then-diverging is the defect**, and declining would not be.

### What I unblocked, and what it revealed

323 claimed nodes and ORT still fell back wholesale. Tank's broken-commitment WARN fired on fused
subgraph #15. A five-line R13 change in `vk/session.rs` (Switch's) — bind the translate `Result`,
log `err()` instead of "translate failed" — produced the text in one run:

`Unsupported("`SimplifiedLayerNormalization` input 0 has no element type at compile time")`

from `common_dtype(node, 0, 2)` in `ops/common/templates.rs::simplified_norm`, on the **dynamic**
re-run. Island #15's SLN input 0 is island-internal; the `patched_node` has a shape and no dtype.
The handler is right to refuse — the caller's construction is the defect, and `vk/` is Switch's.

**The consequence to not read past:** islands execute zero times, so
`own_provider_execution_count: 0` and `executed_by: {CPUExecutionProvider: 1377}`. The all-65 oracle
arm has **still** had no real reading; `oracle_outputs_degenerate: 0` was measured CPU-versus-CPU.

### Union defect: my deletion, Trinity's new controls

Her criterion-11(c) tests referenced `mul_f16_unproven.onnx`, deleted in `e97b186` because a proof
run had entered that very form. **Restoring it would have restored a control that passes for the
wrong reason.** Control 1 (dtype) is now `Abs` f32 proven vs `Abs` f16 unproven, the f16 arm built in
`tmp_path` so the generator cannot disarm it; if it is ever proven the readings converge and the test
goes **red**. Control 2 gained its own static `Mul` arm — two different ops in two arms is not a
shape-class control. And the `MatMulNBits` test asserted `len(nbits) == 2` over all entries; the
ledger grew to five, so it selects by form now. A control keyed to a total goes red when the artifact
it guards improves: R9 amendment 5, on my own test.

**Green at this state:** census + ledger lanes `16 passed / 0 failed / 0 ERROR(instrument)` on both
device 0 and device 1.

## 2026-08-02 (later) — ORT's refusal, and the one `else` that holds the flagship

**Escalated as "the EP claims 0/363, your ledger is the critical path for the project."** At the
build in my worktree the reading is **323/363**. This is the third routing of a `0/363` diagnostic
taken against an older binary, so I answered it with an artifact instead of a reply:
`rust/tools/probe_phi35_claim_reading.py` records claimed count, ledger digest and DLL mtime
together. The frame of a result is the binary that produced it.

I minted no keys. Four of the five escalated forms were already proven at `runtime-extent`, and
evidence whose effect is already achieved is indistinguishable in the artifact from progress.

### The one genuinely new thing in the request, taken and wired

`session.disable_cpu_ep_fallback = 1` on the EP arm of `prove()`. Our attribution counters are all
written by the thing being audited; this is ORT refusing from outside our code. Raised as
`CpuFallbackRefusal`, distinct from `InstrumentError`, and turned into `UNATTRIBUTED` with the text
quoted — a reading and an outage must not spell the same.

Not set on discovery: there the node is declined by design and a refusal is the expected state.
It also conflicts with naming the CPU EP explicitly (`Conflicting session configuration`), which is
`ERROR(instrument)` and says nothing about our EP; the strict arm offers this EP alone.

**Mutation-tested before trusted.** Two arms on the planted control, whose key the generator refuses
to write under any circumstance: real key → `MATCH`, ORT silent; nonexistent key → `UNATTRIBUTED`,
ORT refused. The probe asserts the arms *differ*. A guard that fires in both arms is worse than none.

### GQA: the best hypothesis for "it was vacuous" is now excluded

Re-run under the guard, ORT did not refuse — the EP took the node and executed — and the verdict is
`DIVERGENT`, `worst_rel = 16.72642029784887`, identical to the digit for the third time. It stays
out. Claiming-then-diverging is the defect; declining would not be.

### What actually holds Phi-3.5, measured rather than inferred

An instrumented build (reverted; the finding is now a comment) named the branch:

    node=/model/layers.0/input_layernorm/LayerNorm op=SimplifiedLayerNormalization
    slot=0 token=5 n_plan_inputs=5 n_plan_outputs=2 branch=island-output-consumed-internally

The patch loop in `vk/session.rs` handles external inputs and prior-kernel intermediates and leaves
the middle range — island outputs also consumed internally — as `None`, under a comment saying it is
"unusual" and the handler "will degrade gracefully". It is island #15's normal shape, and the handler
does not degrade: it refuses, correctly. **323 claimed nodes execute zero times because of one
`else`.** Switch's.

### The shortcut I did not take

`common_dtype` could infer the missing dtype from a sibling input and the number would have moved
today, under real schedule pressure. That is a check moving with the reader's confidence rather than
with its subject (R9 amendment 5) — the caller lost the information, and a handler that fabricates it
can no longer detect that the caller lost it. This was the same shape as the ledger-from-claim-table
shortcut, arriving at the same moment for the same reason.

**Green:** census + ledger lanes 16/16 on both devices; `cargo test --release --lib` 469/0; clippy
clean.

## Session 27 — 2026-08-02 — What fraction of the model's work runs on the CPU EP

**Asked:** per-EP FLOPs and boundary bytes as a fraction of a graph-derived whole, as a function of
context length, with the fabricated-`128` exposure disclosed. Clock-free.

**Built** `rust/tools/roofline_split.py`. Prediction written first to
`bench/results/roofline_split-prediction.md`; all six predictions held (one under-predicted).

**Answer.** CPU share of bytes: **0.07% at ctx=0, 4.26% at 128, 14.95% at 512, 41.21% at 2048,
73.77% at 8192.** FLOPs: 0.00% → 30.20%. Identical on both devices. Not 3%, not 30% — a curve, and
the regime we had been quoting (ctx=0) is the one that hides it.

**Counterfactual on the same instrument:** claiming the 32 GQA nodes (323 → 355 claimed, the island
we quote) drops CPU bytes to ≤0.29% everywhere. So Switch's GQA fix is worth 41.2 points at ctx=2048
and 73.5 at 8192, and it collapses **33 islands into 1** — the fragmentation is a consequence of the
GQA decline, not a separate lever.

**Retired two numbers.** `{CPU: 120, Vulkan: 99}` reproduces nothing in this build (366 = 363 + 3
folded Constants; 363 = 323 + 40; profile = 33 Vulkan partitions + 40 CPU nodes). And `ep.rs`'s FLOP
estimator reports **16.58% at every context** — which is `32/193`, the anchor ratio. Its FLOP number
is a node count wearing a FLOP's clothes; it is blind to the only axis this model's cost varies on.

**Fabricated extents: zero.** Shape inference resolves everything once ctx is stated. The one
unresolved extent (`cos_cache`/`sin_cache` `[None, 48]`) is *conditional*, not unknown — the `If`
predicate is `total_sequence_length > 4096` and the branches are `[4096,48]` / `[131072,48]`
Constants. Read it, don't invent it.

**R13 against myself.** First run said `UNOBSERVABLE(fabricated extents carry 73.69% of the bytes)`.
That was ERROR(instrument): fabrication flagged per node instead of per tensor, so one 48-wide
operand condemned a 50 MB node. The tell was that the fabricated fraction equalled the CPU fraction
to two decimals at every ctx — an identity with no reason to hold. **Fifth time this week that the
alarming number was the broken instrument.** Check the coincidence before you check the conclusion.

**Left standing:** Switch's 60.5% KV share at 8192 vs my 73.5% GQA share. Different quantities, but
not obviously 13 points different. Both recorded, neither adjusted.

**Touched:** `rust/tools/roofline_split.py` (new), `docs/OP_COVERAGE.md` §7.17, `bench/results/`.
**No gate, `partition.rs`, `registry.rs` or `ops/` change** — `mouse-1` is in the same worktree, so
everything was staged by explicit path.

---

## 2026-08-02 — the ledger had no re-proof path (Switch's find), and the first real reading

**The hole.** `gen_proof_ledger.py --append` printed `UNMEASURED … no unlockable keys` then `PASS`,
writing nothing: an already-claimed form is skipped as an optimisation and never re-measured. GQA's
entry therefore outlived *two* shader rewrites made the same day. Morpheus's shape from an unguarded
direction — not a ledger derived from the claim table, but **an entry that silently outlives its
subject**. `ledger_hits == proven_key_lookups` stays true forever while the kernel drifts out from
under it. **Nothing caught it.** Switch proved GQA correct himself; the ledger agreed, and would have
agreed identically had he broken it.

**Fix (four parts).** `shaders` + `shader_digest` (FNV-1a/64 over the SPIR-V the run *dispatched*)
in every entry, recomputed at parse against this binary; `--reprove`; no `PASS` over a run that
measured nothing; and two per-entry demotion tokens `STALE-SHADER` / `NO-SUBJECT-WITNESS`. Demotion
is per entry — a blanket refusal lets one shader edit disable every claim, and that is the blunt
shape that gets switched off in a hurry.

**The frame I was asked to state rather than assume.** Covers dispatched SPIR-V bytes only. Does
*not* cover: shaders the run did not dispatch (a whole-set digest costs 73 re-proofs per unrelated
edit, and a gate that expensive gets relaxed); **host-side code — named residual, exact falsifier:
a host-only numeric change leaves the entry green**; comment-only GLSL edits, which do not reach
SPIR-V. One relaxation, taken because the compiler verifies it rather than me asserting it.

**No grandfathering.** Every shipped entry lacked the witness and faulted. Admitting them "for
compatibility" would exempt exactly the entries with longest to drift. Re-proved the whole thing:
**74 entries, digest `d07643b0c4cd2e8f`, key set byte-identical — nothing lost, nothing gained.**
The claims did not change; they became falsifiable. That is the right outcome to be able to state
plainly, because I ran it hoping it would invalidate something.

**GQA's margin, recorded not smoothed.** `0.00072939` vs `rtol 0.001` = 1.37×. Next-tightest entry
in the ledger is **160× further from its bound**. Switch called it out himself rather than letting
a thin margin read as comfort; he was right to.

**Then the bill.** Phi-3.5: **355/363 claimed**, `ledger_hits 355`, `unproven_forms_claimed 0`,
`ALL-PROVEN`. Morpheus's own honest-cost number, reached from the other side. The guard held: the
ledger grew by one entry and the claimed count moved by 32 — not a ledger growing while `0/363`
stands still.

**Criterion 10's first real reading: DISAGREE.** 65/65 compared, **0 degenerate**, 0 CPU-only, 0
unobservable, bit-identical across three runs. So the reopened all-zero-KV defect is *not*
reproducing. Failures are `[0, 63, 64]` — logits, `present.31.key`, `present.31.value`.

**R11 on my own diagnostic.** My first per-output table quoted `max_rel/rtol` and reported `24.6x`
for outputs that *passed* — the facts divide by `atol+|b|`, the pass criterion by `atol+rtol·|b|`.
Fixed before quoting anything. **Sixth time the alarming number was my instrument.** With the honest
ratio the picture inverts: 0.044 at layer 0 rising monotonically to 0.81/0.85 at layer 30 and
1.66/1.17 at layer 31. **The pass/fail line falls mid-curve, not at a discontinuity** — layer 30
passing at 0.85 is the same phenomenon as layer 31 failing. So the 62 passes are not 62 clean
results, and relaxing the tolerance would only move the crossing point. Logits diverge at 46.9× but
`argmax 30751` and top-10 agree exactly. Changed no tolerance and relaxed no gate; handed it on.

**Attribution before blame.** 43 op-suite reds remain; read the text and they are `[staged]`, not
`[unproven]` — ops deliberately not enabled in the op table. Nothing to do with the ledger.

**Touched:** `registry.rs`, `gen_proof_ledger.py`, `probe_phi35_oracle_detail.py` (new),
`evidence/proof_ledger.jsonl`, `docs/OP_COVERAGE.md` §8.9.11–§8.9.12; declared cross-owner edits in
`counters.rs` (Tank) and one line in `vk/session.rs` (Switch). Rust 476/0, clippy clean, census
`ALL-PROVEN` on both devices. **Still the actual blocker on the last 8 nodes:** the
`island-output-consumed-internally` branch in `vk/session.rs`, which is Switch's.

📌 Team update (2026-08-02T14-42-30-07-00): Switch found the proof ledger's `--append` mode skips already-claimed forms — fixing the GQA prefill race required changing a shader the ledger had already marked `PASS` for a different bug, and re-running `--append` afterward printed `PASS` having measured nothing against the new shader. "A proof that cannot be invalidated by changing its subject is not a proof of that subject." You (Mouse) fixed it: `shader_digest`/`shaders` fields, a `--reprove` flag, two new tokens `STALE-SHADER` (digest changed since last proof) and `NO-SUBJECT-WITNESS` (nothing to compare against); all 74 ledger entries re-proved under the new scheme. — decided by Switch, Mouse


---

## 2026-08-02 — The staged-op sweep: 21 promoted, 3 refused, two holes in my own harness

**The census was sharper than the brief.** `epctl --dump-capabilities` gave 91 rows, 50 live, 41
staged, and **five** staging reasons, not four. Only **22 of the 41 were dischargeable by a proof
run**; the other 19 are missing code. The 13 `XL_KERNEL` rows are almost exactly the
`com.microsoft` contrib set — MoE, QMoE, MultiHeadAttention, RotaryEmbedding, CausalConvWithState,
LinearAttention, GatherBlockQuantized, Attention, Q/DQ. **The contrib-op commitment cannot be
advanced by proof runs at all.** Said so before promoting anything, not after. The JSON dump did
not carry the staging reason, so I added `staged_reason` first — otherwise the table would have
been a code reading (R10).

**Promotion grants nothing.** `Live` is deprecated in favour of `Ready` = "kernel exists,
claimability is derived from the ledger". So a 22-row flip is safe by construction; the proof run
is the gate.

**I found a hole in the harness before I used it.** 12 of the 22 return bool, and `compare()` had
no guard on the CPU reference. `Equal` on two normals is all-False; `IsNaN` on a finite tensor is
all-False; both would have reported `MATCH worst_rel 0.0` having tested nothing. **The cheapest way
to prove twelve ops was to prove none of them.** Guard added as `ERROR(instrument)` — the kernel
was not shown wrong, the case model was shown inadequate — and mutation-tested in both polarities
**in the lane**, not as a script.

**Two predictions written down before the runs turned out wrong, which is what the file is for.**
I predicted MATCH for `Sum`/`Mean`/`Max`/`Min` and for `Swish`.

* The variadics raised `EP_FAIL … 'Sum' with 3 inputs needs the chained-dispatch lowering, which
  is not written yet`. My deliberate **3-input** case caught a claim/translate invariant
  violation: the predicate allowed 1..=8, the lowering handled ≤2. A 2-input case would have
  proved the binary path and left the fold untested, and a real 3-input node would have been
  claimed and then failed at session creation. Fixed by construction —
  `MAX_VARIADIC_INPUTS_LOWERED`, read by both sides. Proved them at `n2` only; arity rides the
  key, so `n3` stays unclaimable and declines honestly. §8.9's key vindicated a second time.
* `Swish` offered no key at all: a **second, hand-written evidence list** (`EXERCISED`) vetoes it
  before the ledger is consulted. Reverted to `Staged` and reported rather than papered over. I
  did not add `("Swish","f32")` by hand — that would make the list assert something no run has
  shown.

**Then I broke the ledger and it taught me two things.** Probing the re-proof path, I ran the
generator without `--append` and it **rewrote 95 entries down to 1, then printed `PASS`** — because
`--check` was asked whether the file it had just written was self-consistent, which an empty file
is. Two halves of a report describing different things: Scribe's health report, one level out.
Entries are now always carried forward; `--rebuild` has to be asked for. I restored them by
**re-running all 21 proof runs**, not by reconstructing entries from the attempts log.

The second thing is worse. **`--reprove` never re-measured anything against a healthy ledger.**
`claim_audit` records `unproven_forms_enabled` only when the ledger *misses*, so an already-proven
key offered through the hatch produced an empty admission set and `UNATTRIBUTED`. **The 74-entry
re-proof I reported yesterday succeeded only because the on-disk ledger had drifted from the baked
copy and every lookup was `Faulted`** — an accident of state, not a path. That is §8.9.11's own
defect one level up: a re-proof that silently measures nothing. Fixed with a distinct witness,
`reproof_forms_admitted`, deliberately **not** folded into `unproven_forms_enabled` — that list is
the §8.9.4 disclosure of forms claimed *without* evidence, and a proven form named there would be
false and would fail `--check-counters`. The two arms now differ observably.

**The falsifier is the op-suite red count, not Phi-3.5's** — none of these 22 ops appears in
Phi-3.5, so quoting 355/363 would be dishonest. **43 → 18**, and I read all 18:
7 were `XPASS(stale)` on Switch's GQA `xfail(strict)` whose own removal condition was met (marker
removed, file now 7 green); 1 is `test_census_baseline_has_no_drift`, drift `- bench/phases.py::load`,
**pre-existing** — I stashed and re-ran rather than assuming; 8 are documented refusals.

**The refusals, which I would rather have than 8 more claims:** `Swish`/f32, `Add`/i32, `Mul`/i32
are all blocked by `EXERCISED`, the hand-written list, even though `ew_binary_add_i32.spv` and
`ew_binary_mul_i32.spv` exist and compile — **a named criterion-11 residual**, a flag its author
set standing beside an artifact-derived ledger, and because it runs inside the predicate no proof
run can reach those forms. `IsInf` needs a shader variant; `Cast` ×3 needs a template;
**`Flatten` and `Reshape` have no row in the op table at all.** The same 8, byte-for-byte, are the
only op-table reds on device 1, so the 21 promotions hold on the Intel conformance oracle too.

**A test of mine asserted a stand-in.** `families_..._are_still_staged` asserted `Staged` for the
15 families; after each got its own proof that was backwards — it would have passed for a row
flipped to `Ready` with nothing measuring it, and failed after a genuine proof. It now asserts that
none of the 15 rides `add_f32`'s evidence, because each names itself in a ledger key.

**Touched:** `epctl.rs`, `counters.rs` (cross-owner, Tank, declared), `registry.rs`,
`ops/elementwise.rs`, `ops/norm.rs`, `ops/common/claim.rs`, `ops/common/templates.rs`,
`gen_proof_ledger.py`, `ledger_case_models.py`, `tests/ops/test_proof_ledger.py`,
`tests/ops/test_gqa.py` (cross-owner, Switch, stale marker removal),
`evidence/proof_ledger.jsonl` (74 → 95), `docs/OP_COVERAGE.md` §8.9.13. Rust **477/0**, clippy
clean, ledger+census lanes green on **both** devices. **No timing figure quoted.**
