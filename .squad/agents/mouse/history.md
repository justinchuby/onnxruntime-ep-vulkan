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
